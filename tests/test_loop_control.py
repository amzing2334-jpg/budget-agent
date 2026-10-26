"""CPU checks for visible evidence, recovery opportunities and paid feedback."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from budget_agent.budget import TokenCounter
from budget_agent.context import ContextManager
from budget_agent.loop_control import ReadLoopMonitor
from budget_agent.models import ScriptedModel
from budget_agent.runner import run_task
from budget_agent.schema import Event, RunConfig
from budget_agent.tasks import load_tasks


ROOT = Path(__file__).resolve().parents[1]
READ = {"tool": "read_file", "path": "slugify.py"}


def read_event(number, start=1, end=3, *, revision="r1", path="source.py",
               changed=None, clipped=False, ok=True):
    metadata = {"path": path, "total_lines": 20, "start_line": start,
                "end_line": end, "truncated_line_content": clipped}
    rows = [] if start is None else [f"{n}: source {n}" for n in range(start, end + 1)]
    if changed is not None:
        rows[0] = f"{start}: {changed}"
    output = "read_file " + json.dumps(metadata) + "\n" + "\n".join(rows)
    # Canonical paths and actual ranges come from output, not action arguments.
    action = {"tool": "read_file", "path": "./" + path, "start_line": 1, "end_line": 200}
    return Event(f"e{number:04d}", number, action, output, revision, ok,
                 model_content=json.dumps(action))


def messages(events, strategy="full", revision="r1"):
    return ContextManager(strategy, TokenCounter()).build(
        "Read the repository.", "Fix the issue.", events, revision, 100_000)


def rows(result, name):
    path = Path(result["result_path"]).with_name(name)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_script(tmp_path, actions, *, policy="recover", **settings):
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    counter = TokenCounter()
    config = RunConfig(strategy="mask", context_tokens=16_000,
                       loop_policy=policy, **settings)
    return run_task(task, config, tmp_path, model=ScriptedModel(actions, counter), counter=counter)


def test_identical_visible_read_warns_then_stops_only_after_feedback():
    monitor = ReadLoopMonitor()
    first, second, third = [read_event(n) for n in range(1, 4)]
    originals = [event.to_dict() for event in (first, second, third)]
    assert monitor.observe(first, messages([])) is None
    repeat = monitor.observe(second, messages([first]))
    assert repeat.source_event_ids == [first.id]
    assert (repeat.start_line, repeat.end_line) == (1, 3)
    assert repeat.consecutive_repeats == 1
    assert monitor.pending_feedback and not monitor.should_stop
    monitor.feedback_sent()
    assert monitor.pending_feedback is None and not monitor.should_stop
    repeat = monitor.observe(third, messages([first, second]))
    assert repeat.consecutive_repeats == 2 and monitor.should_stop
    assert originals == [event.to_dict() for event in (first, second, third)]


def test_unsent_warning_cannot_stop_even_after_multiple_repeats():
    monitor = ReadLoopMonitor()
    events = []
    for number in range(1, 5):
        current = read_event(number)
        monitor.observe(current, messages(events))
        events.append(current)
    assert monitor.consecutive_repeats == 3
    assert monitor.pending_feedback and not monitor.warning_sent
    assert not monitor.should_stop


def test_complete_visible_evidence_can_be_combined_from_multiple_reads():
    monitor = ReadLoopMonitor()
    first, second = read_event(1, 1, 2), read_event(2, 3, 4)
    assert monitor.observe(first, messages([])) is None
    assert monitor.observe(second, messages([first])) is None
    repeat = monitor.observe(read_event(3, 1, 4), messages([first, second]))
    assert repeat.source_event_ids == [first.id, second.id]


def test_masked_evidence_can_be_recovered_then_immediate_reread_warns():
    monitor = ReadLoopMonitor()
    archive = []
    for event in (read_event(1, 1, 3), read_event(2, 4, 6), read_event(3, 7, 9)):
        assert monitor.observe(event, messages(archive, "mask")) is None
        archive.append(event)
    restored = read_event(4, 1, 3)
    assert monitor.observe(restored, messages(archive, "mask")) is None
    archive.append(restored)
    repeat = monitor.observe(read_event(5, 1, 3), messages(archive, "mask"))
    assert repeat.source_event_ids == [restored.id]
    assert not monitor.should_stop


@pytest.mark.parametrize("variant", ["omitted", "assistant_only", "partial_line", "unrelated_event"])
def test_archived_content_requires_matching_complete_lines_in_actual_user_message(variant):
    monitor = ReadLoopMonitor()
    first = read_event(1)
    monitor.observe(first, messages([]))
    request = messages([first])
    if variant == "omitted":
        request = messages([])
    elif variant == "assistant_only":
        request[-1]["role"] = "assistant"
    elif variant == "partial_line":
        request[-1]["content"] = request[-1]["content"].replace("3: source 3", "3: source")
    else:
        request[-1]["content"] = request[-1]["content"].replace(first.id, "e9999")
    assert monitor.observe(read_event(2), request) is None
    assert monitor.pending_feedback is None


@pytest.mark.parametrize("current", [read_event(2, 2, 4), read_event(2, changed="new content"),
                                     read_event(2, path="other.py")])
def test_partial_new_content_changed_text_or_other_file_is_not_a_repeat(current):
    monitor = ReadLoopMonitor()
    first = read_event(1)
    monitor.observe(first, messages([]))
    assert monitor.observe(current, messages([first])) is None
    assert monitor.consecutive_repeats == 0


@pytest.mark.parametrize("revision", ["r2", "r1"])
def test_revision_change_and_reversion_reset_the_repeat_segment(revision):
    monitor = ReadLoopMonitor()
    first, repeated = read_event(1), read_event(2)
    monitor.observe(first, messages([]))
    monitor.observe(repeated, messages([first]))
    monitor.feedback_sent()
    monitor.begin_revision("r2")
    if revision == "r1":
        monitor.begin_revision("r1")
    current = read_event(3, revision=revision)
    assert monitor.observe(current, messages([first, repeated], revision=revision)) is None
    assert monitor.consecutive_repeats == 0 and not monitor.warning_sent


@pytest.mark.parametrize("current", [
    read_event(2, clipped=True), read_event(2, clipped=None), read_event(2, ok=False),
    read_event(2, None, None), replace(read_event(2), output="unsupported legacy output"),
    replace(read_event(2), output="read_file {bad}\n1: source 1"),
    replace(read_event(2), action={"tool": "recall", "event_id": "e0001"}),
])
def test_partial_unknown_failed_empty_or_recalled_observation_does_not_trigger(current):
    monitor = ReadLoopMonitor()
    first = read_event(1)
    monitor.observe(first, messages([]))
    assert monitor.observe(current, messages([first])) is None
    assert not monitor.should_stop and monitor.pending_feedback is None


def test_truncated_previous_read_cannot_establish_complete_prior_evidence():
    monitor = ReadLoopMonitor()
    first = read_event(1, clipped=True)
    assert monitor.observe(first, messages([])) is None
    assert monitor.observe(read_event(2), messages([first])) is None


def test_non_read_action_clears_warning_and_starts_a_new_segment():
    monitor = ReadLoopMonitor()
    first, repeated = read_event(1), read_event(2)
    monitor.observe(first, messages([]))
    monitor.observe(repeated, messages([first]))
    monitor.feedback_sent()
    command = Event("e0003", 3, {"tool": "run", "command": "python reproducer.py"},
                    "failing case reproduced", "r1", False,
                    model_content='{"tool":"run","command":"python reproducer.py"}')
    assert monitor.observe(command, messages([first, repeated])) is None
    assert not monitor.warning_sent
    monitor.observe(read_event(4), messages([first, repeated, command]))
    assert monitor.consecutive_repeats == 1 and not monitor.should_stop


def test_runner_stops_after_paid_feedback_preserving_raw_observations_and_grade(tmp_path):
    result = run_script(tmp_path, [READ, READ, READ, {"tool": "finish"}])
    calls, events = rows(result, "calls.jsonl"), rows(result, "trajectory.jsonl")
    assert result["status"] == "stalled" and result["steps"] == result["calls"] == 3
    assert result["resolved"] is False and result["evaluation"]["status"] != "not_evaluated"
    assert result["loop_control"]["warnings_sent"] == 1
    assert "Read-loop feedback" not in calls[1]["messages"][0]["content"]
    assert "Read-loop feedback" in calls[2]["messages"][0]["content"]
    assert events[0]["output"] == events[1]["output"] == events[2]["output"]
    assert all("Read-loop feedback" not in event["output"] for event in events)
    assert rows(result, "controls.jsonl") == result["loop_control"]["events"]
    assert result["evidence_kind"] == "scripted_smoke"
    assert result["total_tokens"] == sum(
        call["response"]["input_tokens"] + call["response"]["output_tokens"] for call in calls)
    counter = TokenCounter()
    assert all(call["response"]["input_tokens"] == counter.messages(call["messages"]) for call in calls)


def test_runner_allows_recovery_edit_and_independent_success_after_warning(tmp_path):
    result = run_script(tmp_path, [READ, READ,
        {"tool": "edit_file", "path": "slugify.py", "old": "[^a-z]+", "new": "[^a-z0-9]+"},
        {"tool": "run", "command": "python -m unittest discover -s tests -v"},
        {"tool": "diff"}, {"tool": "finish"}])
    assert result["status"] == "finished" and result["resolved"] is True
    assert result["loop_control"]["warnings_sent"] == 1
    assert result["loop_control"]["stopped"] is False
    assert "a-z0-9" in Path(result["patch_path"]).read_text(encoding="utf-8")
    events = rows(result, "trajectory.jsonl")
    assert events[1]["revision"] != events[2]["revision"]
    assert all("Read-loop feedback" not in event["output"] for event in events)


def test_disabled_policy_preserves_requests_and_does_not_interrupt_repeated_reads(tmp_path):
    actions = [READ, READ, READ, {"tool": "finish"}]
    off = run_script(tmp_path / "off", actions, policy="off")
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    counter = TokenCounter()
    default = run_task(task, RunConfig(strategy="mask", context_tokens=16_000), tmp_path / "default",
                       model=ScriptedModel(actions, counter), counter=counter)
    assert off["status"] == default["status"] == "finished"
    assert off["calls"] == 4 and off["loop_control"]["events"] == []
    assert not Path(off["result_path"]).with_name("controls.jsonl").exists()
    for left, right in zip(rows(off, "calls.jsonl"), rows(default, "calls.jsonl"), strict=True):
        assert left["messages"] == right["messages"]
        assert left["response"] == right["response"]
        assert "Read-loop feedback" not in left["messages"][0]["content"]


def test_budget_exhaustion_before_feedback_is_not_reported_as_warning_sent(tmp_path):
    actions = [READ, READ, {"tool": "finish"}]
    baseline = run_script(tmp_path / "baseline", actions, policy="off")
    first_two = rows(baseline, "calls.jsonl")[:2]
    spent = sum(call["response"]["input_tokens"] + call["response"]["output_tokens"] for call in first_two)
    result = run_script(tmp_path / "limited", actions, total_tokens=spent + 128)
    assert result["status"] == "budget_exhausted" and result["calls"] == 2
    assert result["loop_control"]["warnings_sent"] == 0
    assert result["loop_control"]["stopped"] is False
    assert [control["kind"] for control in result["loop_control"]["events"]] == ["repeated_visible_read"]
    assert result["total_tokens"] == spent
    assert result["budget"]["unreconciled_request"] is False
