"""Fixed revision labels keep the agent's history append-only across edits."""
import json
from pathlib import Path

import pytest

from budget_agent.budget import TokenCounter
from budget_agent.context import ContextManager
from budget_agent.models import ScriptedModel
from budget_agent.runner import run_task
from budget_agent.schema import Event, RunConfig
from budget_agent.tasks import load_tasks

ROOT = Path(__file__).resolve().parents[1]


def chat_event(number, revision, tool="read_file"):
    action = {"tool": tool, "path": "mod.py"}
    return Event(f"e{number}", number, action, f"output {number}", revision, True, model_content=json.dumps(action))


def two_builds(labels):
    """The request before an edit, and the one after it (the edit moved the workspace from r1 to r2)."""
    manager = ContextManager("recent", TokenCounter(), "chat", revision_labels=labels)
    before = manager.build("system", "issue", [chat_event(1, "r1")], "r1", 20_000)
    after = manager.build("system", "issue", [chat_event(1, "r1"), chat_event(2, "r2", tool="edit_file")], "r2", 20_000)
    return before, after


def test_fixed_labels_leave_earlier_messages_unchanged_after_an_edit():
    before, after = two_builds("fixed")
    assert after[:len(before)] == before
    assert not any("stale:" in m["content"] or "current revision" in m["content"] for m in after)
    assert "revision=r1" in after[3]["content"] and "revision=r2" in after[-1]["content"]


def test_relabelling_rewrites_earlier_headers_after_an_edit():
    before, after = two_builds("relabel")
    assert after[:len(before)] != before
    assert "stale: observed at an older revision" in after[len(before) - 1]["content"]


def test_unknown_revision_label_mode_is_rejected():
    with pytest.raises(ValueError, match="revision_labels"):
        ContextManager("recent", TokenCounter(), "chat", revision_labels="sometimes")
    with pytest.raises(ValueError, match="revision_labels"):
        RunConfig(revision_labels="sometimes").validate()


def test_fixed_labels_tell_the_agent_how_to_judge_staleness(tmp_path):
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    counter = TokenCounter()
    result = run_task(task, RunConfig(revision_labels="fixed"), tmp_path,
                      model=ScriptedModel([{"tool": "finish"}], counter), counter=counter)
    call = json.loads(Path(result["result_path"]).with_name("calls.jsonl").read_text().splitlines()[0])
    assert "Event headers are fixed when recorded" in call["messages"][0]["content"]
