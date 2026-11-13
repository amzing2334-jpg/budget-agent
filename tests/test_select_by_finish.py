"""Picking among samples by the agent's own finish: per-task expected success against a random pick."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("select_by_finish", ROOT / "scripts/select_by_finish.py")
assert SPEC is not None and SPEC.loader is not None
select_by_finish = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(select_by_finish)


def ledger(path, rows):
    path.write_text("".join(json.dumps({"task_id": t, "sample": i, "split": "eval", "status": s, "resolved": r}) + "\n"
                            for i, (t, s, r) in enumerate(rows)))
    return path


def test_the_selector_takes_finished_attempts_and_falls_back_to_all(tmp_path):
    rows = ([("a", "finished", True), ("a", "finished", True), ("a", "stuck", True)] + [("a", "max_steps", False)] * 5
            + [("b", "stuck", True)] + [("b", "max_steps", False)] * 7
            + [("c", "finished", False), ("c", "stuck", True), ("c", "stuck", None)])
    report = select_by_finish.evaluate("x", ledger(tmp_path / "x.jsonl", rows), resamples=200, seed=0)
    assert report["tasks"] == 3 and report["tasks_without_finish"] == 1
    assert abs(report["pass@1"] - (3 / 8 + 1 / 8 + 1 / 2) / 3) < 1e-12          # the ungraded attempt is left out
    assert abs(report["selector"] - (1 + 1 / 8 + 0) / 3) < 1e-12
    assert (report["tasks_better"], report["tasks_worse"], report["pass@k"]) == (1, 1, 1.0)
    assert report["finish_precision"] == 2 / 3


def test_pass_at_one_matches_compare_pools_on_a_published_pool():
    pool = ROOT / "docs/data/selection/round10_eval.jsonl"
    report = select_by_finish.evaluate("round10", pool, resamples=100, seed=0)
    published = json.loads((ROOT / "docs/data/round10/compare_eval.json").read_text())
    assert abs(report["pass@1"] - published["candidate_pass@1"]) < 1e-12
