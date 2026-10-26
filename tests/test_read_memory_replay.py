"""Offline coverage and accounting checks; no inference or training imports."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from budget_agent.budget import TokenCounter
from budget_agent.context import ContextManager
from budget_agent.schema import Event


SPEC = importlib.util.spec_from_file_location("replay_read_memory", Path(__file__).parents[1] / "scripts/replay_read_memory.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def read(step, path="a.py", text="same source", revision="a"):
    action = {"tool": "read_file", "path": path}
    metadata = {"path": path, "total_lines": 1, "start_line": 1, "end_line": 1, "truncated_line_content": False}
    return Event(f"e{step:04}", step, action, "read_file " + json.dumps(metadata) + "\n1: " + text,
                 revision, model_content=json.dumps(action))


def search(step, revision="a"):
    action = {"tool": "search", "query": "source"}
    return Event(f"e{step:04}", step, action, "search found nothing", revision, model_content=json.dumps(action))


def write_run(folder, events, *, context_tokens=20000, task_id="task-a", strategy="mask"):
    folder.mkdir(parents=True)
    counter = TokenCounter()
    result = {"task_id": task_id, "run_id": folder.name, "strategy": strategy, "initial_revision": "a",
              "config": {"strategy": strategy, "history_format": "chat", "context_tokens": context_tokens,
                         "max_output_tokens": 256, "tokenizer": "hf:Qwen/frozen-model"}}
    (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
    calls = []
    for index, event in enumerate(events):
        messages = ContextManager("mask", counter).build("system for call " + str(index), "repair issue", events[:index],
                                                       events[index - 1].revision if index else "a", 100000)
        calls.append({"purpose": "agent", "messages": messages, "response": {"content": event.model_content,
                      "input_tokens": 7, "output_tokens": 3, "usage_source": "tokenizer_exact"}})
    (folder / "calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in calls), encoding="utf-8")
    (folder / "trajectory.jsonl").write_text("".join(json.dumps(asdict(event)) + "\n" for event in events), encoding="utf-8")
    return folder


def test_same_file_numbered_lines_visible_with_memory_not_cross_file_string_search(tmp_path):
    folder = write_run(tmp_path / "run", [read(1), read(2, "b.py"), search(3), read(4)])
    result = replay.replay_run(folder, TokenCounter())
    last = result["prefixes"][-1]
    assert last["known_target_read_lines"] == 1
    assert last["policies"]["mask"]["visible_known_target_lines"] == 0
    assert last["policies"]["mask-read"]["visible_known_target_lines"] == 1
    assert 0 < last["extra_input_tokens_estimate"] <= 1024
    assert last["policies"]["mask-read"]["input_tokens_estimate"] <= last["input_cap"]


def test_known_target_requires_same_file_not_matching_text(tmp_path):
    folder = write_run(tmp_path / "run", [read(1), search(2), search(3), read(4, "b.py")])
    last = replay.replay_run(folder, TokenCounter())["prefixes"][-1]
    assert last["known_target_read_lines"] == 0
    assert last["policies"]["mask-read"]["visible_known_target_lines"] == 0


def test_inputs_do_not_depend_on_current_target_or_future_action(tmp_path):
    common = [read(1), search(2), search(3)]
    first = write_run(tmp_path / "first", common + [read(4), search(5)])
    second = write_run(tmp_path / "second", common + [read(4, "future.py", "unseen secret"), read(5, "next.py")])
    a = replay.replay_run(first, TokenCounter())["prefixes"]
    b = replay.replay_run(second, TokenCounter())["prefixes"]
    for left, right in zip(a[:4], b[:4]):
        for strategy in ("mask", "mask-read"):
            assert left["policies"][strategy]["messages_sha256"] == right["policies"][strategy]["messages_sha256"]
            assert left["policies"][strategy]["input_tokens_estimate"] == right["policies"][strategy]["input_tokens_estimate"]
    assert a[3]["known_target_read_lines"] == 1
    assert b[3]["known_target_read_lines"] == 0


def test_a_b_a_revision_return_does_not_reuse_earlier_a(tmp_path):
    folder = write_run(tmp_path / "run", [read(1), search(2, "b"), search(3), read(4)])
    last = replay.replay_run(folder, TokenCounter())["prefixes"][-1]
    assert last["known_target_read_lines"] == 0
    assert last["extra_input_tokens_estimate"] == 0


def test_each_call_uses_actual_system_and_full_task(tmp_path, monkeypatch):
    folder = write_run(tmp_path / "run", [read(1), search(2)])
    calls = replay._read_jsonl(folder / "calls.jsonl")
    calls[1]["messages"][1]["content"] = replay.TASK_PREFIX + "distinct full issue\nwith newline"
    (folder / "calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in calls), encoding="utf-8")
    captured = []
    build = ContextManager.build

    def capture(self, system, issue, events, revision, max_input_tokens):
        captured.append((system, issue, len(events), revision, max_input_tokens))
        return build(self, system, issue, events, revision, max_input_tokens)

    monkeypatch.setattr(ContextManager, "build", capture)
    replay.replay_run(folder, TokenCounter())
    assert captured == [("system for call 0", "repair issue", 0, "a", 19744)] * 2 + [
        ("system for call 1", "distinct full issue\nwith newline", 1, "a", 19744)] * 2


def test_overflows_stay_in_denominator_and_visibility_is_unknown(tmp_path):
    folder = write_run(tmp_path / "run", [read(1), search(2), search(3), read(4)], context_tokens=300)
    report = replay.replay_roots([folder], TokenCounter())
    assert report["summary"]["recorded_prefixes"] == report["summary"]["overflow_prefixes"] == 4
    assert report["summary"]["paired_constructed_prefixes"] == 0
    assert report["summary"]["mean_extra_input_tokens_per_constructed_prefix"] is None
    for prefix in report["runs"][0]["prefixes"]:
        assert prefix["extra_input_tokens_estimate"] is None
        for policy in prefix["policies"].values():
            assert policy["status"] == "overflow"
            assert policy["input_tokens_estimate"] is policy["visible_known_target_lines"] is None


def test_zero_allowance_same_inputs(tmp_path):
    folder = write_run(tmp_path / "run", [read(1), search(2), search(3), read(4)])
    for prefix in replay.replay_run(folder, TokenCounter(), read_memory_tokens=0)["prefixes"]:
        assert prefix["extra_input_tokens_estimate"] == 0
        assert prefix["policies"]["mask"]["messages_sha256"] == prefix["policies"]["mask-read"]["messages_sha256"]


def test_summary_and_unknown_usage_rows_excluded_without_shifting_alignment(tmp_path):
    folder = write_run(tmp_path / "run", [read(1), search(2)])
    path = folder / "calls.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    rows.insert(1, json.dumps({"purpose": "summary", "response": {"content": "summary"}}))
    rows.insert(2, json.dumps({"purpose": "agent", "status": "usage_unknown"}))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = replay.replay_run(folder, TokenCounter())
    assert result["excluded_call_rows"] == 2
    assert [prefix["call_jsonl_line"] for prefix in result["prefixes"]] == [1, 4]


@pytest.mark.parametrize("change", ["raw", "count", "missing_raw"])
def test_alignment_errors_reject_instead_of_guessing(tmp_path, change):
    folder = write_run(tmp_path / "run", [read(1), search(2)])
    path = folder / "trajectory.jsonl"
    rows = replay._read_jsonl(path)
    if change == "raw":
        rows[0]["model_content"] = " " + rows[0]["model_content"]
    elif change == "missing_raw":
        rows[0]["model_content"] = None
    else:
        rows.pop()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        replay.replay_run(folder, TokenCounter())


def test_multiple_histories_one_task_and_explicit_exclusions(tmp_path):
    write_run(tmp_path / "one", [read(1)])
    write_run(tmp_path / "two", [read(1)])
    write_run(tmp_path / "recent", [read(1)], strategy="recent")
    external = write_run(tmp_path / "external", [read(1)])
    result = json.loads((external / "result.json").read_text(encoding="utf-8"))
    result["harness"] = "external"
    (external / "result.json").write_text(json.dumps(result), encoding="utf-8")
    report = replay.replay_roots([tmp_path, tmp_path / "one"], TokenCounter())
    assert report["summary"]["replayed_runs"] == 2
    assert report["summary"]["excluded_runs"] == 2
    assert report["summary"]["distinct_task_ids"] == 1
    assert report["summary"]["histories_per_task_id"] == {"task-a": 2}
    assert "Excluded runs:" in replay._markdown(report)


def test_cli_labels_usage_hashes_inputs_and_never_overwrites(tmp_path):
    folder = write_run(tmp_path / "run", [read(1), search(2)])
    before = {path: path.read_bytes() for path in folder.iterdir()}
    output = tmp_path / "analysis"
    assert replay.main(["--runs", str(folder), "--output", str(output)]) == 0
    report = json.loads((output / "replay.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["evidence_kind"] == "offline_fixed_prefix_replay"
    assert report["analysis_counter"] == {"name": "byte", "description": TokenCounter().description,
                                          "usage_is_estimated": True, "exact_qwen_tokenizer": False, "provider_billing": False}
    assert len(report["source_fingerprint"]) == len(report["driver_sha256"]) == 64
    for path, original in before.items():
        assert path.read_bytes() == original
        assert report["runs"][0]["sources"][path.name]["sha256"] == hashlib.sha256(original).hexdigest()
    for prefix in report["runs"][0]["prefixes"]:
        assert prefix["historical_usage"]["input_tokens"] == 7
        assert prefix["historical_usage"]["usage_source"] == "tokenizer_exact"
        assert prefix["historical_usage"]["not_analysis_counter"] is True
        assert prefix["recorded_input_estimate_with_analysis_counter"] != 7
        assert set(prefix["source_sha256"]) == {path.name for path in before}
    with pytest.raises(FileExistsError):
        replay.main(["--runs", str(folder), "--output", str(output)])


def test_incomplete_target_output_is_unknown_not_zero(tmp_path):
    target = read(2)
    target.output = target.output.replace('"truncated_line_content": false', '"truncated_line_content": true')
    folder = write_run(tmp_path / "run", [read(1), target])
    prefix = replay.replay_run(folder, TokenCounter())["prefixes"][-1]
    assert prefix["target_read_status"] == "unknown_or_incomplete_read_output"
    assert prefix["known_target_read_lines"] is None
    assert prefix["policies"]["mask-read"]["visible_known_target_lines"] is None
