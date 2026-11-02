import json
from dataclasses import replace
from pathlib import Path

import pytest

from budget_agent.diagnostics import analyze_trajectory, first_workspace_change
from budget_agent.budget import TokenCounter
from budget_agent.cli import main
from budget_agent.models import ScriptedModel
from budget_agent.paired import analyze_paired
from budget_agent.report import generate_report, load_results
from budget_agent.runner import run_task
from budget_agent.schema import Event, RunConfig
from budget_agent.tasks import load_tasks


LIST = {"tool": "list_files"}
SHELL = {"tool": "run", "command": "make partial-fix"}


def test_runner_diagnose_report_and_paired_agree_on_the_billed_edit(tmp_path):
    task = load_tasks(Path(__file__).resolve().parents[1] / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    actions = [{"tool": "edit_file", "path": "slugify.py", "old": "missing\u2028anchor", "new": "x"},
               {"tool": "edit_file", "path": "slugify.py", "old": "[^a-z]+", "new": "[^a-z0-9]+"},
               {"tool": "finish"}]
    counter = TokenCounter()
    config = RunConfig(reflection_policy="failure", strategy="full")
    runs = tmp_path / "runs"
    results = [run_task(task, replace(config, strategy=strategy), runs,
                        model=ScriptedModel(actions, counter), counter=counter) for strategy in ("full", "recent")]
    result = results[0]
    change = result["diagnostics"]["first_workspace_change"]
    prefix = result["budget"]["requests"][:3]
    assert [row["purpose"] for row in prefix] == ["agent", "reflection", "agent"]
    assert (change["event_id"], change["tool"], change["usage"]["method"]) == ("e0002", "edit_file", "request_index")
    assert change["usage"]["requests"] == 3
    tokens = sum(row["input_tokens"] + row["output_tokens"] for row in prefix)
    assert change["usage"]["total_tokens"] == tokens
    assert result["resolved"] is True

    output = tmp_path / "diagnosis.json"
    assert main(["diagnose", "--input", result["result_path"], "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["runs"][0]["diagnostics"]["first_workspace_change"] == change
    rendered = generate_report(runs, tmp_path / "report")
    for kind in ("markdown", "html"):
        assert f"e0002 (edit_file), 3 requests, {tokens} tokens" in rendered[kind].read_text(encoding="utf-8")
    pair = analyze_paired(load_results(runs), "full", "recent")["pairs"][0]
    assert pair["baseline"]["first_workspace_change"] == change
    assert pair["candidate"]["first_workspace_change"] == results[1]["diagnostics"]["first_workspace_change"]


def event(step, action, revision, **fields):
    return Event(f"e{step:04d}", step, action, "", revision, **fields)


def call(purpose, content, inputs, outputs):
    return {"purpose": purpose, "messages": [], "requested_max_output_tokens": 100,
            "response": {"content": content, "input_tokens": inputs, "output_tokens": outputs,
                         "usage_source": "provider", "model": "test"}}


def ledger(calls):
    return [{"purpose": c["purpose"], "estimated_input_tokens": c["response"]["input_tokens"],
             "reserved_output_tokens": c["requested_max_output_tokens"], "overrun_reasons": [],
             **{k: c["response"][k] for k in ("input_tokens", "output_tokens", "model", "usage_source")}}
            for c in calls if "response" in c]


def test_prefix_includes_auxiliary_requests_and_triggering_call_but_not_later_unknown():
    calls = [call("agent", json.dumps(LIST), 11, 1), call("reflection", "review", 17, 3),
             call("summary", "summary", 23, 5), call("agent", json.dumps(SHELL), 31, 7),
             {"purpose": "agent", "status": "usage_unknown"}]
    events = [event(1, LIST, "base", revision_before="base", request_index=1, model_content=json.dumps(LIST)),
              event(2, SHELL, "changed", ok=False, revision_before="base", request_index=4,
                    model_content=json.dumps(SHELL))]
    diagnostics = analyze_trajectory(events, calls, initial_revision="base", requests=ledger(calls))
    metric = diagnostics["first_workspace_change"]
    assert metric["status"] == "observed"
    assert metric["step"] == 2 and metric["event_id"] == "e0002" and metric["tool"] == "run"
    usage = metric["usage"]
    assert usage["requests"] == 4 and usage["method"] == "request_index"
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (82, 16, 98)


def test_legacy_trace_matches_parsed_actions_and_retains_change_before_restore():
    # Older JSONL did not include request_index, revision_before or model_content.
    events = [event(1, {"tool": "invalid_action"}, "base", ok=False),
              event(2, SHELL, "changed"), event(3, {"tool": "finish"}, "base")]
    calls = [call("agent", '{"tool":', 5, 3), call("summary", "context", 7, 1),
             call("agent", "```json\n" + json.dumps(SHELL) + "\n```", 11, 3),
             call("agent", '{"tool":"finish"}', 19, 5)]
    metric = first_workspace_change(events, calls, "base", ledger(calls))
    assert metric["status"] == "observed" and metric["step"] == 2
    assert metric["usage"]["method"] == "response_order"
    assert metric["usage"]["requests"] == 3 and metric["usage"]["total_tokens"] == 30


@pytest.mark.parametrize("problem", ["unknown_prefix", "wrong_index", "wrong_actor", "raw_conflict", "wrong_ledger"])
def test_uncertain_request_attribution_keeps_observation_and_withholds_usage(problem):
    events = [event(1, LIST, "base"), event(2, SHELL, "changed")]
    calls = [call("agent", json.dumps(LIST), 5, 1), call("reflection", "review", 5, 1),
             call("agent", json.dumps(SHELL), 7, 3)]
    recorded = ledger(calls)
    if problem == "unknown_prefix":
        calls[1] = {"purpose": "reflection", "status": "usage_unknown"}
        del recorded[1]
    elif problem == "wrong_index":
        events[0].request_index, events[1].request_index = 1, 2
    elif problem == "wrong_actor":
        # Even a mismatch before the triggering event invalidates ordinal attribution.
        calls[0]["response"]["content"] = '{"tool":"diff"}'
    elif problem == "raw_conflict":
        events[1].request_index = 3
        events[1].model_content = '{"tool":"list_files"}'
    else:
        recorded[0]["input_tokens"] += 1
    metric = first_workspace_change(events, calls, "base", recorded)
    assert metric["status"] == "observed" and metric["usage"] is None


@pytest.mark.parametrize("revisions,initial,status", [
    ([], "base", "unavailable"),
    (["base", "base"], "base", "not_observed"),
    (["base", "changed"], None, "unavailable"),
])
def test_distinguishes_missing_state_from_no_observed_change(revisions, initial, status):
    events = [event(i, LIST, revision) for i, revision in enumerate(revisions, 1)]
    assert first_workspace_change(events, [], initial, [])["status"] == status
