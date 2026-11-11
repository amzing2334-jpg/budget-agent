"""The pass@k estimator and pool summary behind every reported number."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("rollout_pool", Path(__file__).parents[1] / "scripts/rollout_pool.py")
rollout_pool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rollout_pool)


@pytest.mark.parametrize("n,c,k,expected", [(4, 1, 1, 0.25), (4, 0, 4, 0.0), (4, 1, 4, 1.0), (4, 2, 2, 5 / 6),
                                            (8, 3, 4, 1 - 5 / 70)])
def test_pass_at_k_is_the_unbiased_estimator(n, c, k, expected):
    assert rollout_pool.pass_at_k(n, c, k) == pytest.approx(expected)


def test_summary_excludes_ungraded_attempts_from_pass_at_k_but_counts_them():
    def row(task, resolved, status="finished", split="eval"):
        return {"task_id": task, "split": split, "resolved": resolved, "status": status, "steps": 10}
    rows = [row("a", True), row("a", False), row("a", None, "model_error"), row("b", False), row("b", False),
            row("c", True, split="train")]
    summary = rollout_pool.summarize(rows, [1, 2])
    entry = summary["eval"]
    assert entry["tasks"] == 2 and entry["attempts"] == 5 and entry["ungraded_attempts"] == 1
    assert entry["pass@1"] == pytest.approx((0.5 + 0.0) / 2) and entry["pass@2"] == pytest.approx((1.0 + 0.0) / 2)
    assert entry["solved_tasks"] == 1 and entry["statuses"] == {"finished": 4, "model_error": 1}
    assert summary["train"]["pass@1"] == 1.0 and summary["train"]["pass@2"] is None
