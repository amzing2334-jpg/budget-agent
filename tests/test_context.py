from pathlib import Path
from types import SimpleNamespace

import pytest

from budget_agent.budget import TokenCounter
from budget_agent.context import ContextManager, ContextOverflow
from budget_agent.schema import Event
from budget_agent.cli import load_config


def event(number, *, revision="r1", ok=True, tool="run", output="output", **action):
    return Event(f"e{number}", number, {"tool": tool, **action}, output, revision, ok)


def test_full_keeps_history_and_refuses_overflow():
    counter = TokenCounter()
    manager = ContextManager("full", counter, history_format="transcript")
    events = [event(1, output="x" * 500)]
    messages = manager.build("system", "must preserve Unicode 中文", events, "r1", 1500)
    assert "中文" in messages[1]["content"]
    assert "x" * 500 in messages[-1]["content"]
    with pytest.raises(ContextOverflow, match="Full-history"):
        manager.build("system", "must preserve Unicode 中文", events, "r1", 400)


@pytest.mark.parametrize("strategy", ["full", "recent", "mask", "summary", "structured"])
def test_issue_never_truncated(strategy):
    manager = ContextManager(strategy, TokenCounter(), history_format="transcript")
    with pytest.raises(ContextOverflow, match="complete task issue"):
        manager.build("system", "mandatory" * 200, [], "r1", 100)


def test_structured_retains_provenance_stale_evidence_and_remembered_note():
    counter = TokenCounter()
    manager = ContextManager("structured", counter, history_format="transcript")
    events = [
        event(1, output="verbose" * 300),
        event(2, tool="remember", note="Keep backward compatibility", source="e1", output="saved"),
        event(3, ok=False, output="test_compatibility FAILED", revision="r1"),
        event(4, tool="write_file", path="a.py", content="x" * 1000, output="written", revision="r2"),
    ]
    messages = manager.build("system", "FIX EXACTLY THIS ISSUE", events, "r2", 2000)
    text = "\n".join(m["content"] for m in messages)
    assert counter.messages(messages) <= 2000
    assert "FIX EXACTLY THIS ISSUE" in text
    assert "event_id=e3" in text and "stale" in text
    assert "Keep backward compatibility" in text
    assert "event_id=e4" in text
    assert "x" * 1000 not in text
    assert manager.stats["compactions"] == 1


def test_structured_never_erases_canonical_events():
    event1 = event(1, output="long" * 1000)
    manager = ContextManager("structured", TokenCounter(), history_format="transcript")
    manager.build("system", "issue", [event1], "r1", 1000)
    assert event1.output == "long" * 1000
    assert manager.stats["excerpted_events"] > 0


def test_summary_is_accounted_through_callback_and_incremental():
    counter = TokenCounter()
    manager = ContextManager("summary", counter, history_format="transcript")
    calls = []
    def summarize(messages, max_output_tokens):
        assert counter.messages(messages) <= 1100
        calls.append((messages, max_output_tokens))
        return "Summary: prior tests failed; inspect the current code before concluding success."
    events = [event(1, output="x" * 1000), event(2, output="tests failed", ok=False)]
    messages = manager.build("system", "complete issue", events, "r1", 1100, summarize)
    assert len(calls) == 1
    assert "complete issue" in messages[1]["content"]
    assert counter.messages(messages) <= 1100
    assert manager.stats["summary_calls"] == 1
    manager.build("system", "complete issue", events, "r1", 1100, summarize)
    assert len(calls) == 1


def test_summary_without_callback_fails_explicitly():
    manager = ContextManager("summary", TokenCounter(), history_format="transcript")
    with pytest.raises(ContextOverflow, match="accounted model callback"):
        manager.build("system", "issue", [event(1, output="x" * 1000)], "r1", 500)


def test_summary_does_not_need_model_if_history_fits():
    manager = ContextManager("summary", TokenCounter(), history_format="transcript")
    messages = manager.build("system", "issue", [event(1)], "r1", 1500)
    assert manager.stats["summary_calls"] == 0
    assert "event_id=e1" in messages[-1]["content"]


@pytest.mark.parametrize("strategy", ["full", "recent", "mask", "summary", "structured"])
def test_new_note_preserves_staleness_of_old_source(strategy):
    manager = ContextManager(strategy, TokenCounter(), history_format="transcript")
    note = event(2, revision="r2", tool="remember", source="e1", source_revision="r1", note="test passed before the edit")
    messages = manager.build("system", "issue", [note], "r2", 1500)
    text = messages[-1]["content"]
    assert "revision=r2" in text
    assert "source_revision=r1" in text
    assert "stale: observed at an older revision" in text


def test_structured_failure_excerpt_preserves_traceback_tail():
    manager = ContextManager("structured", TokenCounter(), history_format="transcript")
    failure = event(1, ok=False, output="setup logs\n" * 500 + "AssertionError: compatibility must be preserved")
    messages = manager.build("system", "issue", [failure], "r1", 1000)
    text = "\n".join(m["content"] for m in messages)
    assert "AssertionError: compatibility must be preserved" in text
    assert "event_id=e1" in text
    assert "excerpt" in text


def test_recent_keeps_complete_history_when_it_fits():
    manager = ContextManager("recent", TokenCounter(), history_format="transcript")
    events = [event(1, tool="write_file", content="x" * 100, output="written"), event(2)]
    messages = manager.build("system", "complete issue", events, "r1", 2000)
    assert "x" * 100 in messages[2]["content"]
    assert "event_id=e2" in messages[3]["content"]
    assert manager.stats["compactions"] == 0


def test_recent_keeps_contiguous_complete_suffix_without_ranking():
    counter = TokenCounter()
    events = [event(1, tool="remember", note="old note", output="x" * 1000), event(2), event(3)]
    suffix = ContextManager("full", counter, history_format="transcript").build("system", "complete issue", events[-2:], "r1", 2000)
    limit = counter.messages(suffix)
    manager = ContextManager("recent", counter, history_format="transcript")
    messages = manager.build("system", "complete issue", events, "r1", limit)
    assert messages == suffix
    assert manager.stats["last_visible_events"] == 2
    assert manager.stats["last_omitted_events"] == 1
    assert manager.stats["excerpted_events"] == 0
    assert manager.stats["summary_calls"] == 0


def test_recent_excerpts_oversized_latest_failure_without_mutating_history():
    counter = TokenCounter()
    manager = ContextManager("recent", counter, history_format="transcript")
    output = "setup logs\n" * 500 + "AssertionError: preserve compatibility"
    events = [event(1), event(2, ok=False, output=output)]
    messages = manager.build("system", "preserve the entire issue 中文", events, "r2", 1000)
    text = messages[-1]["content"]
    assert counter.messages(messages) <= 1000
    assert messages[1]["content"].endswith("preserve the entire issue 中文")
    assert "event_id=e2" in text and "stale" in text
    assert "AssertionError: preserve compatibility" in text
    assert "[excerpt; full output available by recall]" in text
    assert events[1].output == output
    assert manager.stats["excerpted_events"] == 1
    assert manager.stats["last_omitted_events"] == 1


def test_recent_fails_when_even_an_excerpt_cannot_fit():
    counter = TokenCounter()
    base = ContextManager("full", counter, history_format="transcript").build("system", "issue", [], "r1", 1000)
    with pytest.raises(ContextOverflow, match="No recent event evidence"):
        ContextManager("recent", counter, history_format="transcript").build("system", "issue", [event(1)], "r1", counter.messages(base))


def test_recent_marks_excerpt_when_only_event_header_would_fit():
    counter = TokenCounter()
    events = [event(1, output="x" * 1000)]
    full = ContextManager("full", counter, history_format="transcript").build("system", "issue", events, "r1", 2000)
    prefix = full[-1]["content"].partition("Observation:\n")[0] + "Observation:\n"
    limit = counter.messages(full[:2] + [{"role": "user", "content": prefix}]) + 8
    messages = ContextManager("recent", counter, history_format="transcript").build("system", "issue", events, "r1", limit)
    assert "[excerpt; full output available by recall]" in messages[-1]["content"]
    assert counter.messages(messages) <= limit


@pytest.mark.parametrize("history_format", ["chat", "transcript"])
@pytest.mark.parametrize("size", [0, 1, 2])
def test_mask_before_third_event_is_identical_to_full(history_format, size):
    events = [event(i, output=f"observation {i}") for i in range(size)]
    for item in events:
        item.model_content = f'{{"tool":"run","command":"echo {item.step}"}}'
    counter = TokenCounter()
    full = ContextManager("full", counter, history_format).build("system", "issue", events, "r1", 4000)
    manager = ContextManager("mask", counter, history_format)
    assert manager.build("system", "issue", events, "r1", 4000) == full
    assert manager.stats["last_masked_events"] == manager.stats["compactions"] == 0


@pytest.mark.parametrize("history_format", ["chat", "transcript"])
def test_mask_changes_only_old_observation_bodies_even_when_full_fits(history_format):
    counter = TokenCounter()
    events = [event(i, output=f"unique-output-{i}\n" * 40, ok=i != 1) for i in range(1, 5)]
    # A delimiter inside the action must not be mistaken for observation framing.
    events[0].action["command"] = "echo 'Observation:\\nwithin an action'"
    for item in events:
        item.model_content = f'  raw assistant reply {item.step}: Observation:\n  '
    originals = [item.to_dict() for item in events]
    full = ContextManager("full", counter, history_format).build("system", "完整要求", events, "r2", 10000)
    manager = ContextManager("mask", counter, history_format)

    def forbidden(*args):
        pytest.fail("Mask must not call a summary model")

    messages = manager.build("system", "完整要求", events, "r2", 10000, forbidden)
    text = "\n".join(message["content"] for message in messages)
    assert messages[:2] == full[:2]
    assert "unique-output-1" not in text and "unique-output-2" not in text
    assert events[2].output in text and events[3].output in text
    assert text.count("[observation masked; full output available by recall]") == 2
    assert "event_id=e1" in text and "ok=False" in text
    assert "revision=r1" in text and "source_revision=r1" in text and "stale:" in text
    if history_format == "chat":
        assert [m for m in messages if m["role"] == "assistant"] == [m for m in full if m["role"] == "assistant"]
        assert messages[-4:] == full[-4:]
    else:
        assert "within an action" in text
        assert messages[-2:] == full[-2:]
    assert [item.to_dict() for item in events] == originals
    assert counter.messages(messages) < counter.messages(full)
    assert manager.stats["last_masked_events"] == 2
    assert manager.stats["last_visible_events"] == 4 and manager.stats["last_omitted_events"] == 0
    assert manager.stats["summary_calls"] == manager.stats["excerpted_events"] == 0
    assert manager.build("system", "完整要求", events, "r2", 10000, forbidden) == messages


def test_mask_does_not_drop_actions_or_excerpt_latest_observations_on_overflow():
    events = [event(i, output="x" * 700) for i in range(1, 5)]
    originals = [item.to_dict() for item in events]
    counter = TokenCounter()
    manager = ContextManager("mask", counter, "transcript")
    messages = manager.build("system", "issue", events, "r1", 4000)
    exact = counter.messages(messages)
    assert manager.build("system", "issue", events, "r1", exact) == messages
    with pytest.raises(ContextOverflow, match="Observation masking cannot fit"):
        manager.build("system", "issue", events, "r1", exact - 1)
    assert [item.to_dict() for item in events] == originals
    assert manager.stats["excerpted_events"] == manager.stats["summary_calls"] == 0


def test_mask_cloud_config_only_changes_policy_after_exact_tokenizer_normalization():
    root = Path(__file__).parents[1]
    baseline = load_config(SimpleNamespace(config=root / "configs/cloud-v5-qwen3-4b.toml"))
    masked = load_config(SimpleNamespace(config=root / "configs/cloud-v5-qwen3-mask.toml"))
    baseline.tokenizer = f"hf:{baseline.model}@{baseline.model_revision}"
    baseline.strategy = "mask"
    assert masked == baseline
    assert masked.total_tokens == 32000
