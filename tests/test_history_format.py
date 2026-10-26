"""Protocol regression tests use CPU counters and scripted responses only."""
import json
from pathlib import Path

import pytest

from budget_agent.budget import TokenCounter
from budget_agent.cli import build_parser, load_config
from budget_agent.context import ContextManager, ContextOverflow
from budget_agent.hf_model import HFTokenCounter
from budget_agent.runner import run_task
from budget_agent.schema import Event, ModelResponse, RunConfig
from budget_agent.tasks import load_tasks


STRATEGIES = ["full", "recent", "mask", "mask-read", "summary", "structured"]


class ChatTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(text.encode())

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        assert messages and tokenize and add_generation_prompt and not return_dict
        return [1] + [token for m in messages for token in [2, *m["role"].encode(), 3, *m["content"].encode(), 4]] + [5]


@pytest.fixture(params=["byte", "chat_template"])
def counter(request):
    return TokenCounter() if request.param == "byte" else HFTokenCounter(ChatTokenizer(), "test-model", "a" * 40)


def event(number, *, output="tool output", content=None, action=None, ok=True):
    action = action or {"tool": "read_file", "path": f"file{number}.py"}
    return Event(f"e{number}", number, action, output, "r1", ok,
                 model_content=json.dumps(action) if content is None else content)


def assert_complete_groups(messages, events):
    by_content = {e.model_content: e for e in events}
    visible = 0
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            source = by_content[message["content"]]
            observation = messages[index + 1]
            assert observation["role"] == "user"
            assert f"event_id={source.id} " in observation["content"]
            visible += 1
        if message["role"] == "user" and message["content"].startswith("[event_id="):
            assert messages[index - 1]["role"] == "assistant"
    return visible


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_chat_preserves_invalid_raw_reply_and_source_provenance(strategy, counter):
    invalid = "\nAction: {bad JSON}\nObservation: invented output\n"
    events = [event(1, action={"tool": "invalid_action"}, content=invalid, output="Invalid JSON", ok=False),
              event(2, action={"tool": "remember", "source": "e1", "source_revision": "r0", "note": "old claim"})]
    manager = ContextManager(strategy, counter)
    messages = manager.build("system", "完整 issue", events, "r1", 3000)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[2]["content"] == invalid
    assert messages[3]["content"].endswith("Observation:\nInvalid JSON")
    assert "source_revision=r0" in messages[-1]["content"] and "stale:" in messages[-1]["content"]
    assert manager.stats["last_visible_events"] == 2
    assert manager.stats["last_input_tokens"] == counter.messages(messages)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_legacy_events_require_explicit_transcript(strategy):
    old = Event("e1", 1, {"tool": "run", "command": "pytest"}, "failed", "r1")
    with pytest.raises(ValueError, match="original model_content"):
        ContextManager(strategy, TokenCounter()).build("system", "issue", [old], "r1", 1000)
    messages = ContextManager(strategy, TokenCounter(), "transcript").build("system", "issue", [old], "r1", 1000)
    assert messages[-1]["role"] == "user" and "Action:" in messages[-1]["content"]


def test_recent_selects_contiguous_complete_event_groups(counter):
    events = [event(1, output="small"), event(2, output="middle" * 2000), event(3), event(4)]
    suffix = ContextManager("full", counter).build("system", "issue", events[-2:], "r1", 5000)
    limit = counter.messages(suffix)
    manager = ContextManager("recent", counter)
    messages = manager.build("system", "issue", events, "r1", limit)
    assert messages == suffix
    assert assert_complete_groups(messages, events) == manager.stats["last_visible_events"] == 2
    assert manager.stats["last_omitted_events"] == 2


@pytest.mark.parametrize("strategy", ["recent", "structured"])
def test_excerpt_changes_only_observation_preserving_raw_write_and_failure_tail(strategy, counter):
    action = {"tool": "write_file", "path": "large.py", "content": "actual code\n" * 160}
    source = event(1, action=action, output="build logs\n" * 1000 + "AssertionError: preserve API", ok=False)
    original = source.to_dict()
    manager = ContextManager(strategy, counter)
    messages = manager.build("system", "issue", [source], "r2", 3000)
    assert messages[-2] == {"role": "assistant", "content": source.model_content}
    assert "source_revision=r1" in messages[-1]["content"] and "stale:" in messages[-1]["content"]
    assert "AssertionError: preserve API" in messages[-1]["content"] and "[excerpt;" in messages[-1]["content"]
    assert counter.messages(messages) <= 3000
    assert manager.stats["last_input_tokens"] == counter.messages(messages)
    assert manager.stats["last_visible_events"] == manager.stats["excerpted_events"] == 1
    assert source.to_dict() == original


def test_structured_selection_never_detaches_assistant_from_observation(counter):
    events = [event(i, output="verbose output\n" * 500, ok=i % 2 == 0) for i in range(1, 7)]
    manager = ContextManager("structured", counter)
    messages = manager.build("system", "issue", events, "r2", 1600)
    visible = assert_complete_groups(messages, events)
    assert 0 < visible < len(events)
    assert manager.stats["last_visible_events"] == visible
    assert manager.stats["last_omitted_events"] == len(events) - visible
    assert manager.stats["last_input_tokens"] == counter.messages(messages) <= 1600


def test_summary_input_retains_original_assistant_and_compacts_each_event_once(counter):
    invalid = "  Action: still invalid JSON\n"
    events = [event(1, content=invalid, action={"tool": "invalid_action"}, output="parser detail\n" * 500, ok=False), event(2)]
    calls = []

    def summarize(messages, allowance):
        assert counter.messages(messages) <= 1100
        assert messages[-2] == {"role": "assistant", "content": invalid}
        assert messages[-1]["role"] == "user" and "event_id=e1" in messages[-1]["content"]
        assert "[excerpt;" in messages[-1]["content"]
        calls.append((messages, allowance))
        return "e1 failed to parse at r1; no verified test result."

    manager = ContextManager("summary", counter)
    messages = manager.build("system", "issue", events, "r1", 1100, summarize)
    assert len(calls) == 1 and manager.stats["summary_calls"] == 1
    assert assert_complete_groups(messages, events) == manager.stats["last_visible_events"] == 1
    assert manager.stats["last_input_tokens"] == counter.messages(messages) <= 1100
    assert manager.build("system", "issue", events, "r1", 1100, summarize) == messages
    assert len(calls) == 1


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_oversized_assistant_is_never_truncated_or_summarized_from_truncated_text(strategy, counter):
    source = event(1, content="assistant payload" * 1000)

    def unexpected_summary(*args):
        pytest.fail("A complete assistant event cannot fit in the summarizer request")

    with pytest.raises(ContextOverflow):
        ContextManager(strategy, counter).build("system", "issue", [source], "r1", 1000, unexpected_summary)
    assert source.model_content == "assistant payload" * 1000


def test_first_prompt_identical_across_formats(counter):
    chat = ContextManager("recent", counter).build("system", "issue", [], "r1", 1000)
    transcript = ContextManager("recent", counter, "transcript").build("system", "issue", [], "r1", 1000)
    assert chat == transcript


def test_history_format_config_default_cli_and_validation(tmp_path):
    assert RunConfig().history_format == "chat"
    parser = build_parser()
    assert load_config(parser.parse_args(["run", "--provider", "scripted"])).history_format == "chat"
    config_path = tmp_path / "config.toml"
    config_path.write_text('[run]\nprovider="scripted"\nhistory_format="transcript"\n')
    assert load_config(parser.parse_args(["run", "--config", str(config_path)])).history_format == "transcript"
    args = parser.parse_args(["run", "--config", str(config_path), "--history-format", "chat"])
    assert load_config(args).history_format == "chat"
    with pytest.raises(ValueError, match="history_format"):
        RunConfig(history_format="unknown").validate()


def test_runner_persists_invalid_response_and_replays_it_verbatim(tmp_path):
    counter = TokenCounter()
    invalid = '  Action: {"tool":"read_file","path":"slugify.py"}\nObservation: fabricated\n'
    prompts = []
    replies = iter([invalid, '{"tool":"finish"}'])

    class Model:
        name = "scripted-protocol-check"

        def complete(self, messages, allowance, *, purpose):
            prompts.append(messages)
            content = next(replies)
            return ModelResponse(content, counter.messages(messages), counter.count(content), "scripted_estimate", self.name)

    root = Path(__file__).resolve().parents[1]
    task = load_tasks(root / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    result = run_task(task, RunConfig(), tmp_path, model=Model(), counter=counter)
    trajectory = Path(result["result_path"]).with_name("trajectory.jsonl")
    events = [json.loads(line) for line in trajectory.read_text().splitlines()]
    assert result["status"] == "finished" and result["steps"] == 2
    assert result["config"]["history_format"] == "chat"
    assert events[0]["action"] == {"tool": "invalid_action"}
    assert events[0]["model_content"] == invalid and events[0]["ok"] is False
    assert prompts[1][-2] == {"role": "assistant", "content": invalid}
    assert prompts[1][-1]["role"] == "user" and "event_id=e0001" in prompts[1][-1]["content"]
    assert events[1]["model_content"] == '{"tool":"finish"}'


def test_mask_runner_charges_actual_prompt_and_preserves_full_trajectory(tmp_path):
    counter = TokenCounter()
    prompts = []
    replies = iter(['{"tool":"list_files"}'] * 4 + ['{"tool":"finish"}'])

    class Model:
        name = "scripted-mask-check"

        def complete(self, messages, allowance, *, purpose):
            assert purpose == "agent"
            prompts.append(messages)
            content = next(replies)
            return ModelResponse(content, counter.messages(messages), counter.count(content),
                                 "scripted_estimate", self.name)

    root = Path(__file__).resolve().parents[1]
    task = load_tasks(root / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    result = run_task(task, RunConfig(strategy="mask"), tmp_path, model=Model(), counter=counter)
    run_dir = Path(result["result_path"]).parent
    events = [json.loads(line) for line in (run_dir / "trajectory.jsonl").read_text().splitlines()]
    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text().splitlines()]
    assert result["status"] == "finished" and result["steps"] == result["calls"] == 5
    assert result["input_tokens"] == sum(counter.messages(prompt) for prompt in prompts)
    assert result["budget"]["unreconciled_request"] is False
    assert all(call["purpose"] == "agent" for call in calls)
    marker = "[observation masked; full output available by recall]"
    assert prompts[3][3]["content"].endswith(marker)
    assert prompts[3][2]["content"] == events[0]["model_content"]
    assert prompts[3][-1]["content"].endswith(events[2]["output"])
    assert events[0]["output"] and marker not in events[0]["output"]
    assert calls[3]["messages"] == prompts[3]
    assert result["context_stats"]["summary_calls"] == 0
    assert result["context_stats"]["last_masked_events"] == 2
