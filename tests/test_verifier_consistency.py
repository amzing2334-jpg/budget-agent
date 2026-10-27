"""Saved verifier evidence must agree before reports count a repair as graded."""
from copy import deepcopy
import json

import pytest

from budget_agent.cli import main
from budget_agent.evaluation import evaluate_run
from budget_agent.paired import analyze_paired
from budget_agent.report import aggregate, generate_report, verified_outcome
from test_evaluation import complete_result
from test_report import row


def result_record():
    record = row(**complete_result())
    record["config"].update(strategy="full", seed=0)
    record["task_fingerprint"] = "same-task"
    return record


@pytest.mark.parametrize("evaluation_change,verdict_change,resolved", [
    ({}, {"successful": False}, True),
    ({"status": "unresolved", "resolved": False}, {"successful": True}, False),
    ({"status": "unresolved"}, {}, True),
    ({"status": "resolved", "resolved": False}, {"successful": False}, False),
    ({"status": "invalid_patch"}, {}, True),
    ({}, {"failures": 1}, True),
    ({}, {"errors": 1}, True),
])
def test_contradictory_verifier_evidence_stays_unknown(evaluation_change, verdict_change, resolved):
    record = result_record()
    record["resolved"] = resolved
    record["evaluation"].update(evaluation_change)
    record["evaluation"]["test_results"].update(verdict_change)
    original = deepcopy(record)
    assert verified_outcome(record) is None
    quality = evaluate_run(record)["dimensions"]["quality"]
    assert quality["status"] == "unknown" and quality["score"] is None
    assert record == original


@pytest.mark.parametrize("field,value", [
    ("tests_run", True), ("tests_run", -1), ("tests_run", "1"),
    ("skipped", False), ("skipped", -1), ("skipped", 0.0),
    ("failures", True), ("failures", -1), ("failures", None),
    ("errors", False), ("errors", -1), ("errors", "0"),
    ("successful", 1), ("successful", "true"), ("successful", None),
])
def test_invalid_verifier_fields_are_unknown_without_coercion(field, value):
    record = result_record()
    record["evaluation"]["test_results"][field] = value
    assert verified_outcome(record) is None
    assert evaluate_run(record)["dimensions"]["quality"]["status"] == "unknown"


@pytest.mark.parametrize("field", ["tests_run", "skipped", "successful"])
def test_missing_required_verifier_field_does_not_crash_or_pass(field):
    record = result_record()
    del record["evaluation"]["test_results"][field]
    assert verified_outcome(record) is None
    assert evaluate_run(record)["acceptance_status"] == "unknown"


@pytest.mark.parametrize("verdict", [None, [], "not a verdict"])
def test_invalid_verdict_container_is_unknown(verdict):
    record = result_record()
    record["evaluation"]["test_results"] = verdict
    assert verified_outcome(record) is None
    assert evaluate_run(record)["acceptance_status"] == "unknown"


@pytest.mark.parametrize("evaluation,resolved", [
    # Older result summaries did not record failure/error counters or a verdict.
    ({"status": "resolved", "resolved": True,
      "test_results": {"tests_run": 1, "skipped": 0, "successful": True}}, True),
    ({"status": "unresolved", "resolved": False}, False),
    # unittest unexpectedSuccess can make wasSuccessful false with zero errors/failures.
    ({"status": "unresolved", "resolved": False,
      "test_results": {"tests_run": 1, "skipped": 0, "successful": False, "failures": 0, "errors": 0}}, False),
    # Multiple failing subtests belong to a single executed test method.
    ({"status": "unresolved", "resolved": False,
      "test_results": {"tests_run": 1, "skipped": 0, "successful": False, "failures": 3, "errors": 0}}, False),
    ({"status": "invalid_patch", "resolved": False}, False),
    ({"status": "official_swebench", "resolved": True}, True),
    ({"status": "official_swebench", "resolved": False}, False),
])
def test_consistent_failures_and_legacy_or_official_verdicts_remain_graded(evaluation, resolved):
    record = result_record()
    record.update(evaluation=evaluation, resolved=resolved)
    assert verified_outcome(record) is resolved
    assert aggregate([record])[0]["graded"] == 1
    assert evaluate_run(record)["dimensions"]["quality"]["status"] == ("pass" if resolved else "fail")


def test_contradiction_is_excluded_from_success_rate_paired_wins_and_html(tmp_path):
    candidate = result_record()
    candidate["evaluation"]["test_results"]["successful"] = False
    path = tmp_path / "run" / "result.json"
    path.parent.mkdir()
    candidate["_result_path"] = str(path)
    path.write_text(json.dumps(candidate), encoding="utf-8")
    original = path.read_bytes()
    baseline = result_record()
    baseline.update(strategy="recent", _result_path=str(tmp_path / "baseline" / "result.json"))
    baseline["config"]["strategy"] = "recent"
    group = aggregate([candidate])[0]
    assert group["attempted"] == group["pending_or_ungraded"] == 1
    assert group["graded"] == group["successes"] == 0
    assert group["success_rate"] is None and group["comparison_valid"] is False
    paired = analyze_paired([baseline, candidate], "recent", "full")
    assert paired["eligible_graded_pairs"] == paired["baseline_wins"] == paired["candidate_wins"] == 0
    assert paired["pairs"][0]["ineligible_reasons"] == ["candidate: ungraded"]
    rendered = generate_report(path.parent, tmp_path / "report")
    html = rendered["html"].read_text(encoding="utf-8")
    assert "<td>0 / 0 / 1</td>" in html
    assert "<td>success</td>" not in html
    assert path.read_bytes() == original


@pytest.mark.parametrize("engine", ["builtin", "openjudge"])
def test_mixed_quality_batch_keeps_unknown_and_valid_negative_results(tmp_path, engine):
    if engine == "openjudge":
        pytest.importorskip("openjudge", reason="Optional OpenJudge contract environment is not configured")
    inputs = tmp_path / "inputs"
    originals = {}
    expected = {"valid": "pass", "contradictory": "unknown", "missing_count": "unknown", "negative": "fail"}
    for name in expected:
        record = result_record()
        record.update(task_id=name, run_id=name)
        if name == "contradictory":
            record["evaluation"]["test_results"]["successful"] = False
        elif name == "missing_count":
            del record["evaluation"]["test_results"]["tests_run"]
        elif name == "negative":
            record["resolved"] = False
            record["evaluation"].update(status="unresolved", resolved=False)
            record["evaluation"]["test_results"].update(successful=False, failures=1, errors=0)
        path = inputs / name / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        originals[path] = path.read_bytes()
    output = tmp_path / "audit.json"
    assert main(["audit", "--engine", engine, "--input", str(inputs), "--output", str(output)]) == 1
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert {item["task_id"]: item["evaluation"]["acceptance_status"] for item in audit["runs"]} == expected
    assert all(path.read_bytes() == original for path, original in originals.items())
    if engine == "openjudge":
        for item, grade in zip(audit["runs"], audit["openjudge"]["quality"]):
            if expected[item["task_id"]] == "unknown":
                assert grade["error"] == "insufficient_evidence"
                assert "score" not in grade
            else:
                assert grade["score"] == int(expected[item["task_id"]] == "pass")
