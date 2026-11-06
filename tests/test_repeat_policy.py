"""Repeated failed actions and edit hints: the loop seen with a 7B model on SWE-smith."""
from pathlib import Path

import pytest

from budget_agent.budget import TokenCounter
from budget_agent.environment import LocalEnvironment
from budget_agent.models import ScriptedModel
from budget_agent.runner import run_task
from budget_agent.schema import RunConfig
from budget_agent.tasks import load_tasks


ROOT = Path(__file__).resolve().parents[1]
BAD_EDIT = {"tool": "edit_file", "path": "slugify.py", "old": "def slugify(value):", "new": "def slugify(text):"}
READ = {"tool": "read_file", "path": "slugify.py"}


def attempt(tmp_path, policy, actions):
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    counter = TokenCounter()
    return run_task(task, RunConfig(repeat_policy=policy), tmp_path, model=ScriptedModel(actions, counter), counter=counter)


def outputs(result):
    import json
    path = Path(result["result_path"]).with_name("trajectory.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_warn_names_earlier_failures_without_stopping(tmp_path):
    result = attempt(tmp_path, "warn", [BAD_EDIT, READ, BAD_EDIT, READ, BAD_EDIT, {"tool": "finish"}])
    events = outputs(result)
    assert "[repeat" not in events[0]["output"]
    assert "[repeat 2: this exact action already failed at step(s) 1" in events[2]["output"]
    assert "[repeat 3: this exact action already failed at step(s) 1, 3" in events[4]["output"]
    assert result["status"] == "finished"


def test_stop_ends_the_attempt_at_the_third_identical_failure(tmp_path):
    result = attempt(tmp_path, "stop", [BAD_EDIT, READ, BAD_EDIT, READ, BAD_EDIT, READ, {"tool": "finish"}])
    assert result["status"] == "stuck" and result["steps"] == 5
    assert result["evaluation"]["status"] == "unresolved"  # still graded independently
    kinds = [e["kind"] for e in result["loop_control"]["events"]]
    assert kinds.count("repeated_failed_action") == 2


def test_off_keeps_observations_unchanged(tmp_path):
    events = outputs(attempt(tmp_path, "off", [BAD_EDIT, BAD_EDIT, {"tool": "finish"}]))
    assert all("[repeat" not in e["output"] for e in events)


def test_a_changed_workspace_resets_the_count(tmp_path):
    fix = {"tool": "edit_file", "path": "slugify.py", "old": "[^a-z]+", "new": "[^a-z0-9]+"}
    events = outputs(attempt(tmp_path, "stop", [BAD_EDIT, BAD_EDIT, fix, BAD_EDIT, {"tool": "finish"}]))
    assert "[repeat 2" in events[1]["output"] and "[repeat" not in events[3]["output"]


@pytest.mark.parametrize("old,expected", [
    ("def slugify(value):", "Closest lines in this file: line 5: def slugify(text):"),
    ("return re.sub(r'[^A-Z]+', '-', text)", "Closest lines in this file: line 6:"),
])
def test_missing_anchor_lists_the_closest_real_lines(tmp_path, old, expected):
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    env = LocalEnvironment(task, RunConfig(), tmp_path / "run")
    result = env.execute({"tool": "edit_file", "path": "slugify.py", "old": old, "new": "x"})
    assert not result["ok"] and expected in result["output"] and "use search" in result["output"]


def test_ambiguous_anchor_asks_for_a_distinguishing_line(tmp_path):
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    env = LocalEnvironment(task, RunConfig(), tmp_path / "run")
    (env.workspace / "slugify.py").write_text("def a():\n    return 1\n\ndef b():\n    return 1\n", encoding="utf-8")
    result = env.execute({"tool": "edit_file", "path": "slugify.py", "old": "    return 1", "new": "    return 2"})
    assert "found 2" in result["output"] and "neighbouring line that differs" in result["output"]


def test_file_errors_say_what_is_actually_wrong(tmp_path):
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]  # editable: ["slugify.py"]
    env = LocalEnvironment(task, RunConfig(), tmp_path / "run")
    missing = env.execute({"tool": "edit_file", "path": "schema.py", "old": "a", "new": "b"})
    assert "path does not exist: schema.py" in missing["output"] and "search" in missing["output"]
    assert missing["policy_decision"]["rule"] == "tool_allowed"  # a wrong path is not a policy denial
    created = env.execute({"tool": "write_file", "path": "reproduce.py", "content": "print(1)\n"})
    assert "cannot be created" in created["output"] and "run tool" in created["output"]
    assert created["policy_decision"]["rule"] == "repository_boundary"
    protected = env.execute({"tool": "write_file", "path": "tests/test_public.py", "content": "x"})
    assert "tests and verifier are protected" in protected["output"]
