"""Counting classified passing patches as failures for the sensitivity comparison."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mark_failures", ROOT / "scripts/mark_failures.py")
assert SPEC is not None and SPEC.loader is not None
mark_failures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mark_failures)

ROWS = [{"task_id": "t1", "sample": 0, "resolved": True}, {"task_id": "t1", "sample": 1, "resolved": True},
        {"task_id": "t2", "sample": 0, "resolved": False}]
WEAK = [{"pool": "grpo", "task_id": "t1", "sample": 0, "kind": "strong"},
        {"pool": "grpo", "task_id": "t1", "sample": 1, "kind": "mild"},
        {"pool": "sft", "task_id": "t1", "sample": 1, "kind": "strong"}]


def test_marks_only_the_chosen_kinds_of_the_named_pool():
    out, count = mark_failures.mark(ROWS, WEAK, "grpo", {"strong"})
    assert count == 1 and [r["resolved"] for r in out] == [False, True, False]
    out, count = mark_failures.mark(ROWS, WEAK, "grpo", {"strong", "mild"})
    assert count == 2 and [r["resolved"] for r in out] == [False, False, False]


def test_a_listed_patch_that_is_not_a_resolved_attempt_fails_loudly():
    with pytest.raises(ValueError, match="not resolved attempts"):
        mark_failures.mark(ROWS[2:], WEAK, "grpo", {"strong"})


def test_round3_classification_reproduces_the_reported_counts():
    weak = json.loads((ROOT / "docs/data/round3/original_check.json").read_text())["weak_patches"]
    pools = json.loads((ROOT / "docs/data/round3/original_check.json").read_text())["pools"]
    for pool, counts in pools.items():
        strong = sum(w["pool"] == pool and w["kind"] == "strong" for w in weak)
        mild = sum(w["pool"] == pool and w["kind"] == "mild" for w in weak)
        assert (strong, mild) == (counts["weak_strong"], counts["weak_mild"]), pool
        assert counts["restores_original_exactly"] + counts["equivalent_residual"] + mild + strong == counts["resolved"]
