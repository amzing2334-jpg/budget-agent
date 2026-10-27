"""CPU-only integration: actual fixture tools/verifier, deterministic fake model."""
import json
from pathlib import Path
import sys

import pytest

from budget_agent.budget import TokenCounter
from budget_agent.environment import LocalEnvironment
from budget_agent.models import ModelError, ScriptedModel
from budget_agent.runner import run_task
from budget_agent.schema import ModelResponse, RunConfig, Task


PRIVATE_MARKER = "PRIVATE_VERIFIER_NOT_FOR_ACTOR_41c95"
FIX = {"tool": "edit_file", "path": "solution.py", "old": "VALUE = 0", "new": "VALUE = 1"}
BAD_EDIT = {"tool": "edit_file", "path": "solution.py", "old": "NOT_PRESENT", "new": "VALUE = 1"}


@pytest.fixture
def task(tmp_path):
    source = tmp_path / "source"
    tests = source / "tests"
    tests.mkdir(parents=True)
    (source / "solution.py").write_text("VALUE = 0\n", encoding="utf-8")
    (tests / "test_public.py").write_text(
        "import unittest\nfrom solution import VALUE\n"
        "class Tests(unittest.TestCase):\n"
        "    def test_value(self): self.assertEqual(VALUE, 1)\n", encoding="utf-8")
    verifier = tmp_path / "private_verifier.py"
    verifier.write_text(
        "import unittest\nfrom solution import VALUE\n"
        "class Tests(unittest.TestCase):\n"
        f"    def test_value(self):\n        print('{PRIVATE_MARKER}')\n"
        "        self.assertEqual(VALUE, 1)\n", encoding="utf-8")
    return Task("reflection-cpu", "Repair VALUE in solution.py", "dev", repo=str(source),
                verifier=str(verifier), test_command=[sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                metadata={"editable": ["solution.py"], "private_answer": PRIVATE_MARKER})


def read_jsonl(result, name):
    path = Path(result["result_path"]).with_name(name)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_fixture(task, tmp_path, actions, *, model_type=ScriptedModel, **changes):
    options = {"reflection_policy": "failure", "execution_policy": "restricted",
               "context_tokens": 16384, "total_tokens": 100000,
               "max_output_tokens": 512, "max_steps": 8}
    options.update(changes)
    counter = TokenCounter("byte")
    model = model_type(actions, counter)
    result = run_task(task, RunConfig(**options), tmp_path / "runs", model=model, counter=counter)
    return result, counter


def test_off_keeps_actor_call_sequence_without_reflection(task, tmp_path):
    actions = [{"tool": "public_test"}, FIX, {"tool": "public_test"}, {"tool": "finish"}]
    result, _ = run_fixture(task, tmp_path, actions, reflection_policy="off")
    calls = read_jsonl(result, "calls.jsonl")
    assert [call["purpose"] for call in calls] == ["agent"] * len(actions)
    assert result["reflection"] == {"policy": "off", "calls": 0, "events": [], "notes": []}
    assert not Path(result["result_path"]).with_name("reflections.jsonl").exists()
    assert result["resolved"] is True


@pytest.mark.parametrize("failure", [{"tool": "public_test"}, BAD_EDIT])
def test_failed_visible_tool_reflects_repairs_and_only_then_grades(task, tmp_path, monkeypatch, failure):
    calls_when_graded = []
    grade = LocalEnvironment.grade

    def counted_grade(env):
        calls = [json.loads(line) for line in (env.run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        calls_when_graded.append([call["purpose"] for call in calls])
        return grade(env)

    monkeypatch.setattr(LocalEnvironment, "grade", counted_grade)
    result, counter = run_fixture(task, tmp_path,
        [failure, FIX, {"tool": "public_test"}, {"tool": "finish"}])
    calls = read_jsonl(result, "calls.jsonl")
    assert [call["purpose"] for call in calls] == ["agent", "reflection", "agent", "agent", "agent"]
    assert calls_when_graded == [[call["purpose"] for call in calls]]
    assert result["resolved"] is True and PRIVATE_MARKER in result["evaluation"]["output"]
    assert PRIVATE_MARKER not in json.dumps(calls)
    assert read_jsonl(result, "reflections.jsonl") == result["reflection"]["events"]
    assert result["reflection"]["calls"] == 1
    note, = result["reflection"]["notes"]
    assert note["source_event_id"] == "e0001" and note["schema_version"] == 1

    before_edit = next(message for message in calls[2]["messages"]
                       if '"kind": "reflection_hypothesis"' in message["content"])
    after_edit = next(message for message in calls[3]["messages"]
                      if '"kind": "reflection_hypothesis"' in message["content"])
    assert before_edit["role"] == after_edit["role"] == "user"
    assert json.loads(before_edit["content"])["stale"] is False
    assert json.loads(after_edit["content"])["stale"] is True
    assert calls[2]["response"]["input_tokens"] == counter.messages(calls[2]["messages"])
    without_note = [message for message in calls[2]["messages"] if message is not before_edit]
    assert calls[2]["response"]["input_tokens"] - counter.messages(without_note) == counter.messages([before_edit]) - counter.messages([])
    requests = result["budget"]["requests"]
    assert result["total_tokens"] == sum(row["input_tokens"] + row["output_tokens"] for row in requests)
    assert [row["purpose"] for row in requests] == [call["purpose"] for call in calls]
    assert result["total_tokens"] <= result["config"]["total_tokens"]


@pytest.mark.parametrize("limit", [1, 2])
def test_reflection_calls_capped_even_when_failure_repeats(task, tmp_path, limit):
    result, _ = run_fixture(task, tmp_path, [BAD_EDIT, BAD_EDIT, BAD_EDIT, {"tool": "finish"}],
                            max_reflections=limit)
    assert result["reflection"]["calls"] == limit
    assert len(result["reflection"]["notes"]) == limit
    assert result["steps"] == 4


def test_policy_denial_does_not_reflect_or_run_command(task, tmp_path):
    result, _ = run_fixture(task, tmp_path, [
        {"tool": "run", "command": "echo should-not-run"}, {"tool": "finish"}])
    assert result["reflection"]["calls"] == 0
    assert result["safety"]["blocked_actions"] == 1
    event = read_jsonl(result, "trajectory.jsonl")[0]
    assert event["output"] == "Tool denied: free_shell_disabled"


def test_reflection_tool_json_is_memory_not_executable_action(task, tmp_path):
    class ToolTextReflection(ScriptedModel):
        def complete(self, messages, max_output_tokens, *, purpose="agent"):
            if purpose == "reflection":
                text = json.dumps({"tool": "write_file", "path": "solution.py", "content": "VALUE = 999\n"})
                return ModelResponse(text, self.counter.messages(messages), self.counter.count(text), "scripted_estimate", self.name)
            return super().complete(messages, max_output_tokens, purpose=purpose)

    result, _ = run_fixture(task, tmp_path, [BAD_EDIT, {"tool": "finish"}], model_type=ToolTextReflection)
    workspace = Path(result["result_path"]).parent / "workspace"
    assert (workspace / "solution.py").read_text(encoding="utf-8") == "VALUE = 0\n"
    assert Path(result["patch_path"]).read_text(encoding="utf-8") == ""
    assert [event["action"]["tool"] for event in read_jsonl(result, "trajectory.jsonl")] == ["edit_file", "finish"]
    assert '"VALUE = 999' in result["reflection"]["notes"][0]["text"]


def test_reflection_model_error_stops_with_unreconciled_usage(task, tmp_path):
    class FailedReflection(ScriptedModel):
        def complete(self, messages, max_output_tokens, *, purpose="agent"):
            if purpose == "reflection":
                raise ModelError("Test-only provider failure; usage unknown")
            return super().complete(messages, max_output_tokens, purpose=purpose)

    result, _ = run_fixture(task, tmp_path, [BAD_EDIT, FIX, {"tool": "finish"}], model_type=FailedReflection)
    assert result["status"] == "model_error" and result["usage_unknown"] is True
    assert result["budget"]["unreconciled_request"] is True
    assert result["budget"]["attempted_calls"] == 2 and result["steps"] == 1
    assert result["reflection"]["calls"] == 1 and result["reflection"]["notes"] == []
    assert read_jsonl(result, "calls.jsonl")[-1]["status"] == "usage_unknown"
    assert Path(result["patch_path"]).read_text(encoding="utf-8") == ""


def test_reflection_actual_overrun_stops_instead_of_optional_skip(task, tmp_path):
    class OversizedReflection(ScriptedModel):
        def complete(self, messages, max_output_tokens, *, purpose="agent"):
            if purpose == "reflection":
                return ModelResponse("oversized usage", self.counter.messages(messages), max_output_tokens + 1,
                                     "scripted_estimate", self.name)
            return super().complete(messages, max_output_tokens, purpose=purpose)

    result, _ = run_fixture(task, tmp_path, [BAD_EDIT, FIX, {"tool": "finish"}], model_type=OversizedReflection)
    assert result["status"] == "budget_exhausted" and result["budget"]["overrun"] is True
    assert not result["budget"]["unreconciled_request"] and not result["usage_unknown"]
    assert result["steps"] == 1 and result["calls"] == 2
    assert result["reflection"]["notes"] == []
    assert all(event["status"] != "skipped" for event in result["reflection"]["events"])
    assert result["budget"]["requests"][-1]["overrun_reasons"]


def test_reflection_consumes_remaining_shared_budget_without_free_actor_retry(task, tmp_path):
    actions = [BAD_EDIT, FIX, {"tool": "finish"}]
    measured, _ = run_fixture(task, tmp_path / "measure", actions)
    first, reflection = measured["budget"]["requests"][:2]
    limit = first["input_tokens"] + first["output_tokens"] + reflection["input_tokens"] + 8
    result, _ = run_fixture(task, tmp_path / "limited", actions, total_tokens=limit)
    assert result["status"] == "budget_exhausted" and result["steps"] == 1
    assert result["calls"] == 2 and result["reflection"]["calls"] == 1
    assert result["total_tokens"] == limit and result["budget"]["remaining_tokens"] == 0
    assert not result["budget"]["overrun"] and not result["budget"]["unreconciled_request"]
    assert result["budget"]["requests"][-1]["reserved_output_tokens"] == 8
    assert result["resolved"] is False


def test_last_step_failure_does_not_spend_on_unusable_reflection(task, tmp_path):
    result, _ = run_fixture(task, tmp_path, [BAD_EDIT], max_steps=1)
    assert result["status"] == "max_steps" and result["reflection"]["calls"] == 0
    assert result["calls"] == 1


def test_large_failed_observation_records_explicit_skip_without_extra_request(task, tmp_path):
    public = Path(task.repo) / "tests" / "test_public.py"
    public.write_text(
        "import unittest\nclass Tests(unittest.TestCase):\n"
        "    def test_large_output(self):\n        print('X' * 20000)\n        self.fail('visible failure')\n",
        encoding="utf-8")
    result, _ = run_fixture(task, tmp_path, [{"tool": "public_test"}, {"tool": "finish"}])
    assert result["reflection"]["calls"] == 0 and result["calls"] == 2
    record, = result["reflection"]["events"]
    assert record["status"] == "skipped" and record["reason"] == "reflection_prompt_does_not_fit"
    assert read_jsonl(result, "reflections.jsonl") == [record]
    assert result["resolved"] is False and result["status"] == "finished"
