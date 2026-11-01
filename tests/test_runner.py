import json
from pathlib import Path

import pytest

from budget_agent.environment import LocalEnvironment
from budget_agent.models import ModelError
from budget_agent.runner import run_task
from budget_agent.schema import RunConfig
from budget_agent.tasks import load_tasks


ROOT = Path(__file__).resolve().parents[1]


def test_scripted_repair_has_trace_independent_grade_and_provenance(tmp_path):
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    original = (Path(task.repo) / "slugify.py").read_bytes()
    result = run_task(task, RunConfig(max_output_tokens=1536), tmp_path)
    assert result["resolved"] is True
    assert result["evidence_kind"] == "scripted_smoke"
    assert result["cost_usd"] is None
    assert result["usage_sources"] == ["scripted_estimate"]
    assert result["total_tokens"] > 0
    path = Path(result["patch_path"])
    assert "a-z0-9" in path.read_text()
    events = [json.loads(s) for s in (path.parent / "trajectory.jsonl").read_text().splitlines()]
    assert events[-1]["action"]["tool"] == "finish"
    assert result["diagnostics"]["counts"]["successful_write_file"] == 1
    assert result["diagnostics"]["usage"]["total_tokens"] == result["total_tokens"]
    assert (Path(task.repo) / "slugify.py").read_bytes() == original


def test_budget_stop_does_not_submit_unmade_repair(tmp_path):
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    result = run_task(task, RunConfig(total_tokens=100), tmp_path)
    assert result["status"] in {"context_exhausted", "budget_exhausted"}
    assert result["resolved"] is False
    assert result["calls"] == 0


def test_real_provider_local_execution_is_explicit(tmp_path):
    task = load_tasks(ROOT / "tasks/local.jsonl")[0]
    with pytest.raises(ValueError, match="allow-local-execution"):
        run_task(task, RunConfig(provider="openai-compatible"), tmp_path)


def test_summary_calls_share_the_action_budget(tmp_path):
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    config = RunConfig(strategy="summary", context_tokens=4096, max_output_tokens=1536)
    result = run_task(task, config, tmp_path)
    records = result["budget"]["requests"]
    assert result["resolved"] is True
    assert any(r["purpose"] == "summary" for r in records)
    assert result["calls"] > result["steps"]
    assert result["total_tokens"] == sum(r["input_tokens"] + r["output_tokens"] for r in records)
    assert result["total_tokens"] <= config.total_tokens


def test_provider_failure_is_unknown_usage(tmp_path, monkeypatch):
    class BrokenModel:
        name = "broken-provider"

        def __init__(self, config):
            pass

        def complete(self, *args, **kwargs):
            raise ModelError("Provider transport/JSON failure (UnicodeDecodeError); usage unknown; no retry")

    monkeypatch.setattr("budget_agent.runner.OpenAICompatibleModel", BrokenModel)
    task = load_tasks(ROOT / "tasks/local.jsonl")[0]
    result = run_task(task, RunConfig(provider="openai-compatible", allow_local_execution=True), tmp_path)
    assert result["status"] == "model_error"
    assert result["usage_unknown"] is True
    assert result["cost_usd"] is None
    assert result["budget"]["unreconciled_request"] is True
    assert result["steps"] == 0
    assert result["diagnostics"]["usage"]["unknown_usage_calls"] == 1


@pytest.mark.parametrize("failure_stage", ["model", "environment", "grading"])
def test_programming_errors_surface_after_environment_cleanup(tmp_path, monkeypatch, failure_stage):
    closed = []

    def broken(*args, **kwargs):
        raise TypeError("programmer error")

    if failure_stage == "model":
        monkeypatch.setattr("budget_agent.runner.ScriptedModel.complete", broken)
    elif failure_stage == "environment":
        monkeypatch.setattr(LocalEnvironment, "execute", broken)
    else:
        monkeypatch.setattr(LocalEnvironment, "grade", broken)
    monkeypatch.setattr(LocalEnvironment, "close", lambda env: closed.append(env.workspace))
    task = load_tasks(ROOT / "tasks/local.jsonl")[0]
    with pytest.raises(TypeError, match="programmer error"):
        run_task(task, RunConfig(max_output_tokens=1536), tmp_path)
    assert len(closed) == 1
    assert not list(tmp_path.rglob("result.json"))


def test_environment_io_failure_is_persisted(tmp_path, monkeypatch):
    def unavailable(*args):
        raise OSError("environment unavailable")

    monkeypatch.setattr("budget_agent.runner.create_environment", unavailable)
    task = load_tasks(ROOT / "tasks/local.jsonl")[0]
    result = run_task(task, RunConfig(), tmp_path)
    assert result["status"] == "environment_error"
    assert result["resolved"] is None
    assert Path(result["result_path"]).is_file()


def test_malformed_action_is_an_observation_and_agent_can_continue(tmp_path, monkeypatch):
    monkeypatch.setattr("budget_agent.runner.smoke_actions", lambda task: [
        {"tool": "read_file", "path": "slugify.py", "start_line": None},
        {"tool": "finish"},
    ])
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    result = run_task(task, RunConfig(), tmp_path)
    trajectory = Path(result["result_path"]).with_name("trajectory.jsonl")
    events = [json.loads(line) for line in trajectory.read_text().splitlines()]
    assert result["status"] == "finished"
    assert events[0]["action"]["tool"] == "invalid_action"
    assert events[0]["ok"] is False
    assert "start_line" in events[0]["output"]
    assert events[1]["action"]["tool"] == "finish"


def test_interrupted_request_is_persisted_with_unknown_usage(tmp_path, monkeypatch):
    class InterruptedModel:
        name = "interrupted-provider"

        def __init__(self, config):
            pass

        def complete(self, *args, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr("budget_agent.runner.OpenAICompatibleModel", InterruptedModel)
    task = load_tasks(ROOT / "tasks/local.jsonl")[0]
    result = run_task(task, RunConfig(provider="openai-compatible", allow_local_execution=True), tmp_path)
    assert result["status"] == "interrupted"
    assert result["usage_unknown"] is True
    assert Path(result["result_path"]).is_file()
    assert json.loads(Path(result["result_path"]).read_text())["budget"]["attempted_calls"] == 1
    assert result["diagnostics"]["usage"]["unknown_usage_calls"] == 1


@pytest.mark.parametrize("minimum, expected_calls", [(128, 1), (1, 2)])
def test_action_floor_stops_short_tail_without_model_call(tmp_path, monkeypatch, minimum, expected_calls):
    from budget_agent.budget import TokenCounter
    from budget_agent.models import ScriptedModel

    class FixedInputCounter(TokenCounter):
        def messages(self, messages):
            return 64

    counter = FixedInputCounter()
    actions = [{"tool": "list_files"}, {"tool": "finish"}]
    first_output = counter.count(json.dumps(actions[0], ensure_ascii=False))
    total = 64 + first_output + 64 + 127
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    result = run_task(task, RunConfig(total_tokens=total, min_action_output_tokens=minimum), tmp_path,
                      model=ScriptedModel(actions, counter), counter=counter)
    assert result["calls"] == expected_calls
    assert result["status"] == ("budget_exhausted" if minimum == 128 else "finished")
    assert result["budget"]["attempted_calls"] == expected_calls
    assert result["total_tokens"] <= total
    calls = [json.loads(line) for line in Path(result["result_path"]).with_name("calls.jsonl").read_text().splitlines()]
    assert all(call["requested_max_output_tokens"] >= minimum for call in calls)


def test_scripted_edit_recovery_keeps_failure_and_independent_grade(tmp_path):
    from budget_agent.budget import TokenCounter
    from budget_agent.models import ScriptedModel

    actions = [
        {"tool": "list_files", "path": ".", "limit": 1},
        {"tool": "search", "path": "slugify.py", "query": "return", "limit": 1},
        {"tool": "edit_file", "path": "slugify.py", "old": '"-"', "new": '"_"'},
        {"tool": "read_file", "path": "slugify.py", "start_line": 4, "end_line": 6},
        {"tool": "edit_file", "path": "slugify.py", "old": "[^a-z]+", "new": "[^a-z0-9]+"},
        {"tool": "run", "command": "python -m unittest discover -s tests -v"},
        {"tool": "diff"},
        {"tool": "finish"},
    ]
    counter = TokenCounter()
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    result = run_task(task, RunConfig(strategy="recent"), tmp_path,
                      model=ScriptedModel(actions, counter), counter=counter)
    events = [json.loads(line) for line in Path(result["result_path"]).with_name("trajectory.jsonl").read_text().splitlines()]
    assert not events[2]["ok"] and "found 2" in events[2]["output"]
    assert "line 6, column" in events[2]["output"]
    assert events[2]["revision"] == events[3]["revision"] != events[4]["revision"]
    assert result["diagnostics"]["counts"]["successful_edit_file"] == 1
    assert result["resolved"] is True and result["evidence_kind"] == "scripted_smoke"


def test_shared_transformers_model_resets_seed_and_keeps_attempts_independent(tmp_path, monkeypatch):
    from dataclasses import replace
    from budget_agent.budget import TokenCounter
    from budget_agent.schema import ModelResponse, Task

    class Environment:
        provenance = {"kind": "test"}
        def revision(self): return "baseline"
        def patch(self): return ""
        def grade(self): return {"resolved": False, "status": "unresolved"}
        def close(self): pass

    class Model:
        name = "model@" + "a" * 40
        metadata = {"test_double": True, "generation": {"top_p": 1.0}}

        def __init__(self):
            self.counter = TokenCounter()
            self.counter.name = "test-shared-counter"
            self.seeds = []
            self.prompts = []

        def reset_seed(self, seed):
            self.seeds.append(seed)

        def complete(self, messages, allowance, *, purpose):
            self.prompts.append(messages)
            text = '{"tool":"finish"}'
            return ModelResponse(text, self.counter.messages(messages), self.counter.count(text),
                                 "tokenizer_exact", self.name)

    monkeypatch.setattr("budget_agent.runner.create_environment", lambda *args: Environment())
    model = Model()
    task = Task("test", "Fix the issue.", "dev")
    config = RunConfig(provider="transformers", model="model", model_revision="a" * 40,
                       environment="docker", seed=9)
    first = run_task(task, config, tmp_path, model=model, counter=model.counter)
    second = run_task(task, replace(config, strategy="full"), tmp_path, model=model, counter=model.counter)
    assert model.seeds == [9, 9]
    assert model.prompts[0] == model.prompts[1]  # No first-attempt event leaks into the second.
    assert first["result_path"] != second["result_path"]
    assert first["model_metadata"] == model.metadata
    assert first["evidence_kind"] == second["evidence_kind"] == "scripted_smoke"
    assert first["config"]["tokenizer"] == "test-shared-counter"
    assert first["budget"]["usage_is_estimated"] is False
