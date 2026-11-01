import csv
import json
from pathlib import Path

import pytest

from budget_agent.paired import analyze_paired
from budget_agent.report import aggregate, generate_report


def saved_run(tmp_path, strategy):
    source = Path(__file__).resolve().parents[1] / "runs/iteration-20261027-06/cases/valid/result.json"
    run = json.loads(source.read_text(encoding="utf-8"))
    run["strategy"] = run["config"]["strategy"] = strategy
    run["_result_path"] = str(tmp_path / strategy / "result.json")
    return run


@pytest.mark.parametrize("missing_mirrors", [True, False])
def test_report_and_pair_read_the_audited_token_counts(tmp_path, missing_mirrors):
    baseline = saved_run(tmp_path, "recent")
    candidate = saved_run(tmp_path, "structured")
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        if missing_mirrors:
            del candidate[field]
        else:
            candidate[field] += 100000
    row, = aggregate([candidate])
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        assert row[field] == candidate["budget"][field]
    pair, = analyze_paired([baseline, candidate], "recent", "structured")["pairs"]
    assert pair["eligible_graded_pair"]
    assert pair["candidate"]["total_tokens"] == row["total_tokens"]
    assert pair["candidate_minus_baseline"]["total_tokens"] == 0


@pytest.mark.parametrize("field,bad_value", [
    ("input_tokens", -1), ("output_tokens", True), ("total_tokens", "unknown"),
    ("total_tokens", None),
])
def test_missing_or_invalid_count_stays_unknown_in_group_and_pair(tmp_path, field, bad_value):
    candidate = saved_run(tmp_path, "structured")
    if bad_value is None:
        del candidate["budget"][field]
    else:
        candidate["budget"][field] = bad_value
    second = saved_run(tmp_path, "structured")
    second["task_id"] = "another-task"
    row, = aggregate([candidate, second])
    assert row[field] is None
    assert row["invalid_budget_attempts"] == 1
    pair, = analyze_paired([saved_run(tmp_path, "recent"), candidate], "recent", "structured")["pairs"]
    assert not pair["eligible_graded_pair"]
    if field == "total_tokens":
        assert row["mean_tokens"] is None
        assert pair["candidate"]["total_tokens"] is None
        assert pair["candidate_minus_baseline"]["total_tokens"] is None
        source = tmp_path / "result.json"
        source.write_text(json.dumps(candidate), encoding="utf-8")
        paths = generate_report(source, tmp_path / "report")
        with paths["csv"].open(encoding="utf-8-sig", newline="") as handle:
            assert next(csv.DictReader(handle))["total_tokens"] == ""
        assert "| invalid budget | — |" in paths["markdown"].read_text(encoding="utf-8")
        assert "<td>invalid budget</td><td>—</td>" in paths["html"].read_text(encoding="utf-8")


def test_pre_ledger_results_keep_their_summary_and_paired_counts(tmp_path):
    baseline = saved_run(tmp_path, "recent")
    candidate = saved_run(tmp_path, "structured")
    for result in (baseline, candidate):
        del result["budget"]
    del candidate["total_tokens"]
    row, = aggregate([candidate])
    assert row["total_tokens"] == candidate["input_tokens"] + candidate["output_tokens"]
    pair, = analyze_paired([baseline, candidate], "recent", "structured")["pairs"]
    assert pair["eligible_graded_pair"]
    assert pair["baseline"]["total_tokens"] == baseline["total_tokens"]
    assert pair["candidate_minus_baseline"]["total_tokens"] is None
