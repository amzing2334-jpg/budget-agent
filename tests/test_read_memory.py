"""CPU acceptance checks for bounded source retention, not repair quality."""
from dataclasses import replace
import json
from pathlib import Path
import re

import pytest

from budget_agent.budget import TokenCounter
from budget_agent.context import ContextManager, ContextOverflow
from budget_agent.loop_control import ReadLoopMonitor
from budget_agent.models import ScriptedModel
from budget_agent.observations import parse_read_observation
from budget_agent.runner import run_task
from budget_agent.schema import Event, RunConfig
from budget_agent.tasks import load_tasks


ROOT = Path(__file__).resolve().parents[1]


def read_event(number, start=1, end=5, *, path="source.py", revision="r1",
               clipped=False, ok=True, text="source"):
    metadata = {"path": path, "total_lines": 100, "start_line": start,
                "end_line": end, "truncated_line_content": clipped}
    rows = [] if start is None else [f"{n}: {text} {n}" for n in range(start, end + 1)]
    action = {"tool": "read_file", "path": "./" + path, "start_line": 1, "end_line": 100}
    return Event(f"e{number}", number, action,
                 "read_file " + json.dumps(metadata) + "\n" + "\n".join(rows),
                 revision, ok, model_content="  " + json.dumps(action) + "\n")


def other_event(number, *, revision="r1"):
    action = {"tool": "search", "pattern": "source"}
    return Event(f"e{number}", number, action, f"search output {number}", revision,
                 model_content=json.dumps(action))


def build(events, *, strategy="mask-read", history_format="chat", memory=1024,
          limit=100_000, revision="r1", counter=None):
    manager = ContextManager(strategy, counter or TokenCounter(), history_format,
                             read_memory_tokens=memory)
    messages = manager.build("Keep provenance.", "Preserve 完整 requirements.",
                             events, revision, limit)
    return manager, messages


def observation(messages, event):
    message = next(item for item in messages if item["role"] == "user"
                   and item["content"].startswith(f"[event_id={event.id} "))
    return message["content"].partition("Observation:\n")[2]


def source_rows(text):
    return [line for line in text.splitlines() if re.match(r"^\d+: ", line)]


@pytest.mark.parametrize("history_format", ["chat", "transcript"])
@pytest.mark.parametrize("count", [0, 1, 2, 5])
def test_zero_allowance_is_exactly_mask(history_format, count):
    events = [read_event(n) for n in range(1, count + 1)]
    _, baseline = build(events, strategy="mask", history_format=history_format)
    manager, actual = build(events, memory=0, history_format=history_format)
    assert actual == baseline
    assert manager.stats["last_read_memory_tokens"] == 0
    assert manager.stats["last_read_memory_lines"] == 0
    assert manager.stats["last_read_memory_events"] == 0


@pytest.mark.parametrize("history_format", ["chat", "transcript"])
def test_restoration_preserves_task_actions_recent_evidence_and_original_events(history_format):
    events = [read_event(1), other_event(2), other_event(3)]
    originals = [event.to_dict() for event in events]
    _, baseline = build(events, strategy="mask", history_format=history_format)
    manager, actual = build(events, history_format=history_format)
    assert actual[:2] == baseline[:2]
    assert len(actual) == len(baseline)
    recent_messages = 4 if history_format == "chat" else 2
    assert actual[-recent_messages:] == baseline[-recent_messages:]
    for before, after in zip(baseline[2:], actual[2:], strict=True):
        assert before["role"] == after["role"]
        if before["role"] == "assistant":
            assert before == after
        else:
            assert before["content"].partition("Observation:\n")[0] == after["content"].partition("Observation:\n")[0]
    assert source_rows(observation(actual, events[0])) == source_rows(events[0].output)
    assert observation(actual, events[0]).startswith("read_file excerpt ")
    assert manager.stats["summary_calls"] == 0
    assert originals == [event.to_dict() for event in events]


@pytest.mark.parametrize("memory,spare", [(0, 2000), (1, 2000), (100, 2000),
                                           (300, 2000), (1024, 0), (1024, 300)])
def test_input_increment_includes_markers_metadata_and_obeys_both_caps(memory, spare):
    counter = TokenCounter()
    events = [read_event(1, 1, 40), read_event(2, 41, 80), other_event(3), other_event(4)]
    _, baseline = build(events, strategy="mask", counter=counter)
    baseline_tokens = counter.messages(baseline)
    manager, actual = build(events, memory=memory, limit=baseline_tokens + spare,
                            counter=counter)
    increment = counter.messages(actual) - baseline_tokens
    assert 0 <= increment <= min(memory, spare)
    assert manager.stats["last_read_memory_tokens"] == increment
    assert manager.stats["last_input_tokens"] == counter.messages(actual)
    if not memory or not spare:
        assert actual == baseline


def test_line_text_fitting_alone_does_not_bypass_excerpt_overhead():
    events = [read_event(1, 1, 1), other_event(2), other_event(3)]
    _, baseline = build(events, strategy="mask")
    manager, actual = build(events, memory=TokenCounter().count("1: source 1"))
    assert actual == baseline
    assert manager.stats["last_read_memory_lines"] == 0


def test_memory_cannot_silently_remove_mask_baseline_to_fit():
    events = [read_event(1), other_event(2), other_event(3)]
    _, baseline = build(events, strategy="mask")
    limit = TokenCounter().messages(baseline) - 1
    for strategy in ("mask", "mask-read"):
        with pytest.raises(ContextOverflow, match="Observation masking cannot fit"):
            build(events, strategy=strategy, limit=limit)


def test_complete_task_is_never_clipped_for_read_memory():
    with pytest.raises(ContextOverflow, match="complete task issue"):
        ContextManager("mask-read", TokenCounter()).build(
            "system", "must keep this requirement" * 100, [], "r1", 100)


def test_retains_whole_line_prefixes_without_partial_source():
    events = [read_event(1, 1, 20, text="完整源码"), other_event(2), other_event(3)]
    manager, actual = build(events, memory=350)
    retained = source_rows(observation(actual, events[0]))
    assert 0 < len(retained) < 20
    assert retained == source_rows(events[0].output)[:len(retained)]
    assert manager.stats["last_read_memory_lines"] == len(retained)


def test_dedup_uses_canonical_observed_paths_and_rows_not_requested_ranges():
    events = [read_event(1, 1, 5), read_event(2, 3, 4), other_event(3)]
    # Different aliases and requested ranges still refer to the same observed rows.
    events[1].action["path"] = "subdir/../source.py"
    _, actual = build(events)
    assert source_rows(observation(actual, events[0])) == ["1: source 1", "2: source 2", "5: source 5"]
    assert observation(actual, events[1]) == events[1].output
    excerpt = observation(actual, events[0])
    metadata = json.loads(excerpt.splitlines()[0].removeprefix("read_file excerpt "))
    assert metadata == {"path": "source.py", "source_event_id": "e1", "retained_lines": 3}
    # A gapped memory is explicitly an excerpt, not a complete tool observation.
    assert parse_read_observation(observation(actual, events[0])) is None


def test_newer_source_takes_precedence_and_other_files_do_not_collide():
    events = [read_event(1, 1, 2, text="old"), read_event(2, 1, 2, text="new"),
              read_event(3, 1, 2, path="other.py"), other_event(4), other_event(5)]
    _, actual = build(events, memory=2000)
    assert source_rows(observation(actual, events[0])) == []
    assert source_rows(observation(actual, events[1])) == ["1: new 1", "2: new 2"]
    assert source_rows(observation(actual, events[2])) == ["1: source 1", "2: source 2"]


def test_newest_read_gets_allowance_before_old_read():
    events = [read_event(1, 1, 10), read_event(2, 11, 20), other_event(3), other_event(4)]
    _, actual = build(events, memory=200)
    assert source_rows(observation(actual, events[0])) == []
    retained = source_rows(observation(actual, events[1]))
    assert retained and retained == source_rows(events[1].output)[:len(retained)]


@pytest.mark.parametrize("changed", [False, True])
def test_rejected_large_candidate_allows_only_matching_old_source(changed):
    older = read_event(1, 2, 2)
    newer = read_event(2, 1, 2)
    newer.output = newer.output.replace("1: source 1", "1: " + "long line " * 100)
    if changed:
        newer.output = newer.output.replace("2: source 2", "2: new source 2")
    _, actual = build([older, newer, other_event(3), other_event(4)], memory=200)
    assert source_rows(observation(actual, newer)) == []
    # Exclusion by budget is not evidence that a line is already visible. But a
    # conflicting later observation must not permit resurrection of old text.
    assert source_rows(observation(actual, older)) == ([] if changed else ["2: source 2"])


def test_current_revision_segment_excludes_old_revision_and_reverted_hash():
    events = [read_event(1, 1, 3, revision="r1"), other_event(2, revision="r2"),
              read_event(3, 4, 5, revision="r1"), other_event(4), other_event(5)]
    _, actual = build(events, revision="r1")
    assert source_rows(observation(actual, events[0])) == []
    assert source_rows(observation(actual, events[2])) == ["4: source 4", "5: source 5"]
    _, fresh_revision = build(events, revision="r3")
    assert source_rows(observation(fresh_revision, events[2])) == []


@pytest.mark.parametrize("old_read", [
    read_event(1, clipped=True), read_event(1, clipped=None), read_event(1, ok=False),
    read_event(1, None, None), replace(read_event(1), output="legacy source"),
    replace(read_event(1), output="read_file {broken}\n1: source 1"),
    replace(read_event(1), action={"tool": "recall", "event_id": "e0"}),
])
def test_unknown_incomplete_failed_empty_and_recalled_outputs_are_not_restored(old_read):
    events = [old_read, other_event(2), other_event(3)]
    _, baseline = build(events, strategy="mask")
    manager, actual = build(events)
    assert actual == baseline
    assert manager.stats["last_read_memory_lines"] == 0


@pytest.mark.parametrize("history_format", ["chat", "transcript"])
def test_loop_monitor_recognizes_only_complete_rows_really_restored(history_format):
    events = [read_event(1, 1, 5), other_event(2), other_event(3)]
    monitor = ReadLoopMonitor()
    monitor.observe(events[0], [])
    _, actual = build(events, history_format=history_format)
    repeat = monitor.observe(read_event(4, 1, 5), actual)
    assert repeat is not None and repeat.source_event_ids == [events[0].id]

    monitor = ReadLoopMonitor()
    monitor.observe(events[0], [])
    _, masked = build(events, memory=0, history_format=history_format)
    assert monitor.observe(read_event(4, 1, 5), masked) is None


@pytest.mark.parametrize("invalid", [-1, 1.5, True])
def test_run_config_rejects_invalid_allowance_at_external_boundary(invalid):
    with pytest.raises(ValueError, match="read_memory_tokens"):
        RunConfig(strategy="mask-read", read_memory_tokens=invalid).validate()


def test_scripted_runner_charges_retained_context_without_extra_calls(tmp_path):
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    actions = [{"tool": "read_file", "path": "slugify.py"},
               {"tool": "search", "pattern": "slugify"},
               {"tool": "search", "pattern": "import"}, {"tool": "finish"}]
    counter = TokenCounter()
    config = RunConfig(strategy="mask-read", context_tokens=16_000, read_memory_tokens=1024)
    result = run_task(task, config, tmp_path, model=ScriptedModel(actions, counter), counter=counter)
    calls_path = Path(result["result_path"]).with_name("calls.jsonl")
    calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
    assert result["evidence_kind"] == "scripted_smoke"
    assert result["calls"] == len(actions) == len(calls)
    assert result["context_stats"]["summary_calls"] == 0
    assert result["context_stats"]["last_read_memory_tokens"] > 0
    assert result["total_tokens"] == sum(call["response"]["input_tokens"] + call["response"]["output_tokens"] for call in calls)
    assert all(call["response"]["input_tokens"] == counter.messages(call["messages"]) for call in calls)
    assert result["loop_control"]["events"] == []
