import json
from dataclasses import replace
from pathlib import Path

import pytest

from budget_agent.budget import TokenCounter
from budget_agent.models import ScriptedModel
from budget_agent.runner import SYSTEM, run_task, workspace_feedback
from budget_agent.schema import Event, RunConfig
from budget_agent.tasks import load_tasks


FEEDBACK = "Workspace feedback for "
REPAIR = {"tool": "edit_file", "path": "slugify.py", "old": "[^a-z]+", "new": "[^a-z0-9]+"}
NO_OP = {"tool": "run",
         "command": '''python -c "from pathlib import Path; p=Path('slugify.py'); p.write_bytes(p.read_bytes())"'''}


@pytest.fixture
def slug_task():
    manifest = Path(__file__).resolve().parents[1] / "tasks/local.jsonl"
    return load_tasks(manifest, task_ids=["slugify-digits"])[0]


def rows(result, name):
    with Path(result["result_path"]).with_name(name).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


@pytest.mark.parametrize("tool,output,ok,before,expected", [
    ("edit_file", "Edited example.py", True, "rev", True),
    ("write_file", "Wrote example.py", True, "rev", True),
    ("run", " \t\n", True, "rev", True),
    ("run", "test passed", True, "rev", False),
    ("edit_file", "anchor missing", False, "rev", False),
    ("run", "", True, "older-rev", False),
    ("read_file", "", True, "rev", False),
    ("edit_file", "Edited example.py", True, None, False),
])
def test_feedback_requires_evidence_of_unchanged_workspace(tool, output, ok, before, expected):
    event = Event("e0002", 2, {"tool": tool}, output, "rev", ok=ok)
    if before is not None:
        event.revision_before = before
    feedback = workspace_feedback(event)
    if expected:
        assert feedback.startswith(f"{FEEDBACK}e0002 ({tool}):")
    else:
        assert feedback == ""


class FeedbackActor(ScriptedModel):
    """A fixed policy with one response to the observation, for testing the control path."""

    def __init__(self, counter):
        super().__init__([NO_OP, NO_OP, {"tool": "finish"}], counter)

    def complete(self, messages, max_output_tokens, *, purpose):
        if self.index == 1 and FEEDBACK in messages[0]["content"]:
            self.actions[self.index] = REPAIR
        return super().complete(messages, max_output_tokens, purpose=purpose)


def test_same_budget_actor_receives_billed_feedback_and_can_submit_a_repair(slug_task, tmp_path):
    config = RunConfig(strategy="full", max_steps=3, total_tokens=32000)
    counter = TokenCounter()
    direct = run_task(slug_task, config, tmp_path, model=FeedbackActor(counter), counter=counter)
    candidate = run_task(slug_task, replace(config, decision_policy="workspace-feedback"), tmp_path,
                         model=FeedbackActor(counter), counter=counter)
    events = rows(candidate, "trajectory.jsonl")
    direct_calls, calls = rows(direct, "calls.jsonl"), rows(candidate, "calls.jsonl")
    assert [e["action"]["tool"] for e in events] == ["run", "edit_file", "finish"]
    assert events[0]["ok"] and events[0]["output"] == ""
    assert events[0]["revision_before"] == events[0]["revision"] == events[1]["revision_before"]
    assert events[1]["revision_before"] != events[1]["revision"]
    assert calls[0]["messages"] == direct_calls[0]["messages"]
    assert calls[1]["messages"][1:] == direct_calls[1]["messages"][1:]
    assert f"{FEEDBACK}e0001 (run):" in calls[1]["messages"][0]["content"]
    assert calls[2]["messages"][0]["content"] == SYSTEM
    assert all(call["messages"][0]["content"] == SYSTEM for call in direct_calls)
    for call, charge in zip(calls, candidate["budget"]["requests"], strict=True):
        assert counter.messages(call["messages"]) == charge["estimated_input_tokens"] == charge["input_tokens"]
    assert calls[1]["response"]["input_tokens"] > direct_calls[1]["response"]["input_tokens"]
    assert direct["calls"] == candidate["calls"] == 3
    assert direct["resolved"] is False and candidate["resolved"] is True
    assert candidate["evidence_kind"] == "scripted_smoke"
    assert "[^a-z0-9]+" in Path(candidate["patch_path"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("constraint", ["total_tokens", "max_cost_usd"])
def test_feedback_omission_preserves_direct_requests_and_output_caps(slug_task, tmp_path, constraint):
    read = {"tool": "run", "command": '''python -c "from pathlib import Path; assert Path('slugify.py').is_file()"'''}
    actions = [read, {"tool": "finish"}]
    counter = TokenCounter()
    config = RunConfig(strategy="full")
    direct = run_task(slug_task, config, tmp_path, model=ScriptedModel(actions, counter), counter=counter)
    first, second = direct["budget"]["requests"]
    spent = first["input_tokens"] + first["output_tokens"]
    if constraint == "total_tokens":
        limit = spent + second["input_tokens"] + config.min_action_output_tokens
        limited = replace(config, total_tokens=limit)
    else:
        # Direct can reserve the full configured output. Adding feedback could
        # still pay for a small action, but would quietly narrow that allowance.
        cost_limit = (spent + second["input_tokens"] + config.max_output_tokens) / 1_000_000
        limited = replace(config, input_price_per_million=1, output_price_per_million=1,
                          max_cost_usd=cost_limit)
    control = run_task(slug_task, limited, tmp_path, model=ScriptedModel(actions, counter), counter=counter)
    result = run_task(slug_task, replace(limited, decision_policy="workspace-feedback"), tmp_path,
                      model=ScriptedModel(actions, counter), counter=counter)
    event = rows(result, "trajectory.jsonl")[0]
    assert event["action"] == read and event["ok"] and event["output"] == ""
    assert event["revision_before"] == event["revision"]
    assert result["status"] == control["status"] == "finished"
    assert result["calls"] == control["calls"] == 2
    assert result["budget"]["requests"] == control["budget"]["requests"]
    for fallback, reference in zip(rows(result, "calls.jsonl"), rows(control, "calls.jsonl"), strict=True):
        assert fallback["messages"] == reference["messages"]
        assert fallback["requested_max_output_tokens"] == reference["requested_max_output_tokens"]
    omission, = result["context_stats"]["workspace_feedback_omissions"]
    assert omission["event_id"] == event["id"] and omission["reason"]
