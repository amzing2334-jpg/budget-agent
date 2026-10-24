import json
from pathlib import Path

import pytest

from budget_agent.paired import analyze_paired


def result(tmp_path, strategy, task="task1", **changes):
    folder = tmp_path / f"{task}-{strategy}"
    folder.mkdir(exist_ok=True)
    value = {"task_id": task, "task_fingerprint": f"fingerprint-{task}", "strategy": strategy,
             "provider": "transformers", "model": "model", "actual_models": ["model"],
             "usage_sources": ["tokenizer"], "split": "dev", "task_kind": "swebench",
             "evidence_kind": "model_run", "config": {"strategy": strategy, "seed": 42, "total_tokens": 64000},
             "status": "finished", "resolved": True, "budget": {}, "usage_unknown": False,
             "total_tokens": 1000, "elapsed_seconds": 10, "cost_usd": None,
             "_result_path": str(folder / "result.json")}
    value.update(changes)
    return value


def test_paired_keeps_failures_and_ungraded_in_ledger_without_false_wins(tmp_path):
    runs = [result(tmp_path, "recent", resolved=False), result(tmp_path, "structured", total_tokens=800),
            result(tmp_path, "recent", "task2", resolved=None),
            result(tmp_path, "structured", "task2", status="environment_error", resolved=None, total_tokens=0)]
    report = analyze_paired(runs, "recent", "structured")
    assert report["matched_pairs"] == 2
    assert report["eligible_graded_pairs"] == 1
    assert report["candidate_wins"] == 1
    assert report["outcome_counts"]["candidate"]["infrastructure_error"] == 1
    assert report["outcome_counts"]["baseline"]["ungraded"] == 1
    assert report["mean_candidate_minus_baseline"]["total_tokens"] == {"n": 1, "mean": -200}
    assert report["mean_candidate_minus_baseline"]["cost_usd"] == {"n": 0, "mean": None}


@pytest.mark.parametrize("change, message", [({"task_fingerprint": "different"}, "Unmatched"),
                                            ({"model": "different-model"}, "Mismatched"),
                                            ({"config": {"total_tokens": 128000, "seed": 42}}, "Mismatched"),
                                            ({"actual_models": ["other"]}, "actual_models"),
                                            ({"usage_sources": ["estimated"]}, "usage_sources")])
def test_paired_refuses_noncontrolled_comparisons(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        analyze_paired([result(tmp_path, "recent"), result(tmp_path, "structured", **change)], "recent", "structured")


def test_paired_rejects_duplicate_attempts_instead_of_selecting_best(tmp_path):
    baseline = result(tmp_path, "recent")
    with pytest.raises(ValueError, match="Duplicate attempt"):
        analyze_paired([baseline, baseline, result(tmp_path, "structured")], "recent", "structured")


def test_paired_does_not_pool_different_budgets_across_tasks(tmp_path):
    runs = [result(tmp_path, "recent"), result(tmp_path, "structured"),
            result(tmp_path, "recent", "task2", config={"total_tokens": 128000, "seed": 42}),
            result(tmp_path, "structured", "task2", config={"total_tokens": 128000, "seed": 42})]
    with pytest.raises(ValueError, match="Mismatched model/configuration"):
        analyze_paired(runs, "recent", "structured")


def test_budget_overrun_success_does_not_become_fixed_budget_win(tmp_path):
    report = analyze_paired([result(tmp_path, "recent", resolved=False),
                             result(tmp_path, "structured", budget={"overrun": True})], "recent", "structured")
    assert report["eligible_graded_pairs"] == 0
    assert report["candidate_wins"] == 0
    assert report["pairs"][0]["candidate"]["resolved"] is True


def test_grader_infrastructure_failure_is_not_a_model_loss(tmp_path):
    report = analyze_paired([result(tmp_path, "recent", resolved=None,
                                    evaluation={"status": "infrastructure_error"}),
                             result(tmp_path, "structured")], "recent", "structured")
    assert report["pairs"][0]["baseline"]["outcome"] == "infrastructure_error"
    assert report["eligible_graded_pairs"] == 0


def test_code_change_cannot_form_a_controlled_pair(tmp_path):
    with pytest.raises(ValueError, match="Mismatched"):
        analyze_paired([result(tmp_path, "recent", source_fingerprint="before"),
                        result(tmp_path, "structured", source_fingerprint="after")], "recent", "structured")


def test_repeated_reads_require_same_revision_and_range(tmp_path):
    left, right = result(tmp_path, "recent"), result(tmp_path, "structured")
    read = {"action": {"tool": "read_file", "path": "a.py", "start_line": 1, "end_line": 10}, "revision": "r1", "ok": True}
    events = [read, read, {**read, "revision": "r2"}, {"action": {"tool": "invalid_action"}, "revision": "r2", "ok": False}]
    Path(left["_result_path"]).with_name("trajectory.jsonl").write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    report = analyze_paired([left, right], "recent", "structured")
    assert report["pairs"][0]["baseline"]["repeat_reads_same_revision"] == 1
    assert report["pairs"][0]["baseline"]["invalid_actions"] == 1
    assert report["pairs"][0]["candidate"]["repeat_reads_same_revision"] is None
