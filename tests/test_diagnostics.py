import json

import pytest

from budget_agent.diagnostics import analyze_trajectory, diagnose_run
from budget_agent.context import MASKED_OBSERVATION
from budget_agent.schema import Event


def event(step, tool="read_file", *, output="1: value = 1", ok=True, revision="r1", content=None, **arguments):
    return Event(f"e{step:04d}", step, {"tool": tool, **arguments}, output, revision, ok, 0.25, content)


def call(content, *, output_tokens=10, limit=100, purpose="agent", messages=None, input_tokens=90):
    return {"purpose": purpose, "messages": messages or [], "response": {
        "content": content, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "usage_source": "tokenizer_exact", "model": "fixture",
    }, "requested_max_output_tokens": limit, "elapsed_seconds": 0.5}


def test_multiple_actions_and_limit_pressure_can_coexist():
    content = '{"tool":"list_files"}\n{"tool":"search","query":"x"}\n{"'
    result = analyze_trajectory([event(1, "invalid_action", ok=False, content=content)],
                                [call(content, output_tokens=100)])
    row = result["invalid_actions"][0]
    assert row["multiple_json_values"] is True
    assert row["output_limit_reached"] is True
    assert row["unparsed_text_at_output_limit"] is True
    assert row["json_values"] == 2


@pytest.mark.parametrize("content, tokens, limit, syntax, multiple, reached", [
    ('{"', 1, 1, "unparsed_text_remaining", False, True),
    ('Action: {"tool":"list_files"}', 20, 100, "unparsed_text_remaining", False, False),
    ('{"tool":"unknown"}', 10, 10, "complete", False, True),
    ('{"tool":"list_files"}\n{"tool":"diff"}', 20, 100, "complete", True, False),
    ("", 0, 100, "empty", False, False),
])
def test_invalid_syntax_and_allowance_are_independent(content, tokens, limit, syntax, multiple, reached):
    result = analyze_trajectory([event(1, "invalid_action", ok=False, content=content)],
                                [call(content, output_tokens=tokens, limit=limit)])
    row = result["invalid_actions"][0]
    assert row["json_syntax"] == syntax
    assert row["multiple_json_values"] is multiple
    assert row["output_limit_reached"] is reached


def test_repeat_counts_require_revision_and_observation_equality():
    events = [
        event(1, "edit_file", ok=False, output="ambiguous", path="a.py", old="x", new="y"),
        event(2, "list_files", output="a.py"),
        event(3, "edit_file", ok=False, output="ambiguous again", path="a.py", old="x", new="y"),
        event(4, "list_files", output="a.py"),
        event(5, "list_files", output="a.py\nb.py"),
        event(6, "list_files", output="a.py", revision="r2"),
        event(7, "read_file", output="same", path="a.py", start_line=1),
        event(8, "read_file", output="same", path="a.py", start_line=2),
        event(9, "invalid_action", ok=False),
        event(10, "invalid_action", ok=False),
    ]
    result = analyze_trajectory(events, [])
    assert result["counts"]["repeated_failed_actions"] == 1
    assert result["counts"]["repeated_observations"] == 1
    assert result["repeated_observations"][0]["repeat_event_ids"] == ["e0004"]
    assert result["repeated_failed_actions"][0]["repeat_event_ids"] == ["e0003"]


def test_success_is_tool_success_not_a_repair_claim():
    events = [event(1, "remember"), event(2, "read_file", ok=False),
              event(3, "list_files"), event(4), event(5, "edit_file"),
              event(6, "write_file"), event(7, "run", ok=False),
              event(8, "run"), event(9, "diff"), event(10, "finish")]
    result = analyze_trajectory(events, [])
    assert result["first_successful_tool_action"] == {"event_id": "e0003", "step": 3, "tool": "list_files"}
    for tool in ("read_file", "edit_file", "write_file", "run", "diff"):
        assert result["counts"][f"successful_{tool}"] == 1
    assert result["tools"]["run"] == {"calls": 2, "successful": 1, "failed": 1,
                                      "duration_seconds": 0.5, "output_characters": 24}
    assert "resolved" not in result


def test_calls_align_without_summary_or_unknown_usage_records():
    events = [event(1, content="first"), event(2, "invalid_action", ok=False, content='{"')]
    calls = [call("summary", purpose="summary"), call("first"), call('{"', output_tokens=1, limit=1),
             {"purpose": "agent", "status": "usage_unknown", "max_output_tokens": 5}]
    result = analyze_trajectory(events, calls)
    assert result["invalid_actions"][0]["request_index"] == 3
    assert result["usage"]["unknown_usage_calls"] == 1
    assert result["usage"]["calls"] == 4
    assert result["usage"]["total_tokens"] == 291
    assert result["call_alignment"]["unpaired_agent_responses"] == 0


def test_mismatched_content_does_not_assign_wrong_allowance():
    result = analyze_trajectory([event(1, "invalid_action", ok=False, content="wrong")],
                                [call('{"', output_tokens=1, limit=1)])
    assert result["invalid_actions"][0]["output_limit_reached"] is None
    assert result["call_alignment"]["raw_content_mismatches"] == [{"event_id": "e0001", "request_index": 1}]


def test_observation_exposure_counts_characters_not_allocated_tokens():
    events = [event(1, "list_files", output="abc"), event(2, "search", output="defgh")]
    first = {"role": "user", "content": '[event_id=e0001 step=1]\nAction: {"tool":"list_files"}\nObservation:\nabc'}
    second = {"role": "user", "content": '[event_id=e0002 step=2]\nObservation:\ndefgh'}
    calls = [call("x", messages=[first, second], input_tokens=100),
             call("y", messages=[first], input_tokens=50)]
    result = analyze_trajectory(events, calls)
    exposure = result["observation_exposure"]
    assert exposure["visible_observation_characters"] == 11
    assert exposure["events"][0]["event_id"] == "e0001"
    assert exposure["events"][0]["visible_observation_characters"] == 6
    assert exposure["events"][0]["associated_request_input_tokens"] == 150
    assert exposure["events"][1]["associated_request_input_tokens"] == 100
    assert result["usage"]["input_tokens"] == 150


def test_offline_loader_supports_legacy_events_and_does_not_write(tmp_path):
    legacy = event(1, "invalid_action", ok=False).to_dict()
    del legacy["model_content"]
    (tmp_path / "trajectory.jsonl").write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    (tmp_path / "calls.jsonl").write_text(json.dumps(call('{"', output_tokens=1, limit=1)) + "\n", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result = diagnose_run(tmp_path)
    assert result["invalid_actions"][0]["unparsed_text_at_output_limit"] is True
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    json.dumps(result)


def test_empty_run_is_valid_but_corrupt_json_is_not(tmp_path):
    result = diagnose_run(tmp_path)
    assert result["counts"]["events"] == 0
    assert result["usage"]["input_fraction"] is None
    assert result["first_successful_tool_action"] is None
    (tmp_path / "calls.jsonl").write_text("{bad}\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        diagnose_run(tmp_path)


def read_output(start, end, *, total=20, path="a.py", clipped=False):
    metadata = {"path": path, "total_lines": total, "start_line": start,
                "end_line": end, "truncated_line_content": clipped}
    body = "\n".join(f"{number}: source {number}" for number in range(start, end + 1)) if start is not None else ""
    return "read_file " + json.dumps(metadata) + "\n" + body


def test_read_coverage_uses_actual_output_and_tracks_overlap_by_revision_and_path():
    events = [
        event(1, output=read_output(1, 3), path="./a.py", start_line=1, end_line=200),
        event(2, output=read_output(3, 5), path="a.py", start_line=3, end_line=5),
        event(3, output=read_output(8, 9), path="a.py"),
        event(4, output=read_output(1, 3), path="a.py", revision="r2"),
        event(5, output=read_output(1, 3, path="b.py"), path="b.py"),
    ]
    coverage = analyze_trajectory(events, [])["read_file_coverage"]
    assert coverage["status"] == "complete"
    assert coverage["measured_returned_lines"] == 14
    assert coverage["measured_new_lines"] == 13
    assert coverage["measured_repeated_lines"] == 1
    assert coverage["events"][1]["new_lines"] == 2
    assert coverage["events"][1]["repeated_lines"] == 1
    assert coverage["events"][2]["covered_intervals"] == [[1, 5], [8, 9]]
    assert coverage["events"][2]["covered_interval_count"] == 2
    assert coverage["paths"][0] == {"revision": "r1", "path": "a.py", "unique_lines": 7,
                                     "covered_intervals": [[1, 5], [8, 9]], "covered_interval_count": 2}


def test_read_coverage_caps_at_observed_eof_and_accepts_empty_output():
    events = [
        event(1, output=read_output(290, 317, total=317), path="a.py", start_line=290, end_line=400),
        event(2, output=read_output(None, None, total=317), path="a.py", start_line=400, end_line=500),
        event(3, output=read_output(300, 302, total=317, clipped=True), path="a.py"),
    ]
    coverage = analyze_trajectory(events, [])["read_file_coverage"]
    assert coverage["measured_new_lines"] == 28
    assert coverage["measured_repeated_lines"] == 3
    assert coverage["events"][1]["returned_lines"] == 0
    assert coverage["events"][2]["truncated_line_content"] is True


@pytest.mark.parametrize("output", [
    "1: legacy source without metadata", "read_file {bad}\n1: source", "read_file []\n1: source",
    read_output(1, 3).replace("2: source 2", "3: source 2"),
    read_output(1, 3).replace("3: source 3", ""),
    read_output(1, 3).replace('"total_lines": 20', '"total_lines": 2'),
    read_output(1, 3).replace("1: source 1", "unprefixed source"),
])
def test_unknown_read_formats_are_not_inferred_from_requested_ranges(output):
    coverage = analyze_trajectory([event(1, output=output, path="a.py", start_line=1, end_line=99)], [])["read_file_coverage"]
    assert coverage["status"] == "partial"
    assert coverage["unknown_events"] == 1
    assert coverage["measured_events"] == 0
    assert coverage["events"][0]["new_lines"] is None
    assert coverage["events"][0]["repeated_lines"] is None
    assert coverage["paths"] == []


def test_failed_reads_do_not_establish_coverage():
    coverage = analyze_trajectory([event(1, ok=False, output=read_output(1, 3))], [])["read_file_coverage"]
    assert coverage["failed_events"] == 1
    assert coverage["events"][0]["status"] == "failed"
    assert coverage["measured_returned_lines"] == 0


def test_mask_placeholders_are_not_counted_as_visible_observation_evidence():
    events = [event(1, output="actual source")]
    def message(observation):
        return {"role": "user", "content": "[event_id=e0001 step=1]\nObservation:\n" + observation}
    calls = [call("x", messages=[message("actual source")], input_tokens=100),
             call("y", messages=[message(MASKED_OBSERVATION + "\n")], input_tokens=200),
             call("z", messages=[message(MASKED_OBSERVATION)], input_tokens=300)]
    exposure = analyze_trajectory(events, calls)["observation_exposure"]
    assert exposure["visible_observation_characters"] == len("actual source")
    assert exposure["masked_messages"] == 2
    assert exposure["masked_marker_characters"] == 2 * len(MASKED_OBSERVATION) + 1
    row, = exposure["events"]
    assert row["request_indices"] == [1]
    assert row["masked_request_indices"] == [2, 3]
    assert row["associated_request_input_tokens"] == 100


def test_coverage_loader_remains_read_only(tmp_path):
    rows = [event(1, output=read_output(1, 3)).to_dict(), event(2, output=read_output(2, 4)).to_dict()]
    path = tmp_path / "trajectory.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    before = path.read_bytes()
    assert diagnose_run(tmp_path)["read_file_coverage"]["measured_repeated_lines"] == 2
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
