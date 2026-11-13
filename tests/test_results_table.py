"""The README results tables and the forest plot are generated from the raw comparison outputs."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("results_table", ROOT / "scripts/results_table.py")
assert SPEC is not None and SPEC.loader is not None
results_table = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(results_table)


def test_the_readmes_and_the_figure_match_the_raw_outputs():
    assert results_table.main(["--check"]) == 0


def test_every_stated_verdict_follows_from_its_interval_under_its_rule():
    rows = results_table.load()
    assert len(rows) == 11
    assert [results_table.verdict(row) for row in rows] == [row["expect"] for row in rows]
    assert {row["rule"] for row in rows} == {"superiority", "non-inferiority"}


def test_a_verdict_the_interval_does_not_support_is_an_error(tmp_path):
    (tmp_path / "compare.json").write_text(json.dumps(
        {"baseline_pass@1": 0.10, "candidate_pass@1": 0.12, "delta": 0.02, "ci95": [-0.01, 0.05]}))
    (tmp_path / "comparisons.json").write_text(json.dumps([
        {"data": "compare.json", "record": "X.md", "rule": "superiority", "expect": "improved", "tag": "X",
         "zh": {"experiment": "x", "verdict": "提升", "link": "x"}, "en": {"experiment": "x", "verdict": "improved", "link": "x"}}]))
    rows = results_table.load(tmp_path / "comparisons.json")
    with pytest.raises(ValueError, match="the interval gives no_detectable_change"):
        results_table.check(rows)


@pytest.mark.parametrize(("low", "expected"), [(-2.0, "non_inferior"), (-2.004, "non_inferior"), (-2.01, "non_inferiority_not_shown")])
def test_the_non_inferiority_margin_applies_at_the_reported_precision(low, expected):
    assert results_table.verdict({"rule": "non-inferiority", "low": low, "high": 1.0, "data": "x"}) == expected


def test_numbers_are_written_the_way_the_records_write_them():
    assert [results_table.points(v) for v in (0.0, 0.001, 1.234, -2.0)] == ["0.00", "0.00", "+1.23", "−2.00"]
