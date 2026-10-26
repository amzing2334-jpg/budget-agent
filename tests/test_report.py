import json
from pathlib import Path

import pytest

from budget_agent.report import aggregate, generate_report


def row(**changes):
    item = {"task_id": "task-a", "split": "dev", "task_kind": "local", "provider": "scripted", "model": "fixture", "strategy": "full", "config": {"context_tokens": 4000, "total_tokens": 16000, "tokenizer": "byte", "seed": 0, "strategy": "full"}, "resolved": True, "status": "resolved", "input_tokens": 100, "output_tokens": 20, "cost_usd": None, "elapsed_seconds": 1, "evidence_kind": "scripted_smoke", "usage_sources": ["estimated"]}
    item.update(changes)
    return item


def test_unknown_cost_and_scripted_not_statistical():
    result = aggregate([row()])[0]
    assert result["total_cost_usd"] is None
    assert result["cost_per_success_usd"] is None
    assert result["wilson95"] is None


def test_pending_does_not_become_failure_or_inflate_rate():
    results = aggregate([row(), row(task_id="b", resolved=None, status="pending_official_evaluation")])[0]
    assert results["attempted"] == 2
    assert results["graded"] == 1
    assert results["successes"] == 1
    assert results["success_rate"] is None


def test_infra_and_empty_patch_outcomes_are_separate_ungraded_counts():
    result = aggregate([
        row(), row(task_id="failed", resolved=False),
        row(task_id="infra", resolved=False, evaluation={"status": "infrastructure_error", "infra_failure": True}),
        row(task_id="empty", resolved=None, evaluation={"status": "official_empty_patch", "tests_executed": False}),
    ])[0]
    assert result["attempted"] == 4
    assert result["graded"] == 2
    assert result["successes"] == 1
    assert result["pending_or_ungraded"] == 2
    assert result["infrastructure_errors"] == result["official_empty_patches"] == 1
    assert result["evaluation_status_counts"]["official_empty_patch"] == 1
    assert result["success_rate"] is None
    assert result["cost_per_success_usd"] is None
    assert result["wilson95"] is None
    assert result["comparison_valid"] is False


def test_official_infra_flag_overrides_legacy_false_result():
    result = aggregate([row(resolved=False, evaluation={"status": "official_swebench", "infra_failure": True})])[0]
    assert result["graded"] == 0
    assert result["infrastructure_errors"] == 1
    assert result["success_rate"] is None


def test_cost_per_success_includes_failed_attempt_spend():
    result = aggregate([row(cost_usd=2), row(task_id="b", cost_usd=3, resolved=False)])[0]
    assert result["cost_per_success_usd"] == 5


def test_different_budgets_not_compared():
    a = row()
    b = row(config={**a["config"], "total_tokens": 32000})
    rows = aggregate([a, b])
    assert len({r["group_id"] for r in rows}) == 2


def test_unmatched_task_sets_flagged():
    a = row()
    b = row(strategy="structured", task_id="b", config={**a["config"], "strategy": "structured"})
    rows = aggregate([a, b])
    assert len(rows) == 2
    assert all(not r["matched_tasks_and_repetition_labels"] for r in rows)


def test_actual_model_alias_drift_is_not_combined():
    rows = aggregate([row(actual_models=["model-version-a"]), row(actual_models=["model-version-b"])])
    assert len({r["group_id"] for r in rows}) == 2


def test_external_harness_is_not_grouped_with_internal_runs():
    rows = aggregate([row(), row(harness="mini-swe-agent")])
    assert len(rows) == 2
    assert len({result["group_id"] for result in rows}) == 2
    assert {result["condition"]["harness"] for result in rows} == {None, "mini-swe-agent"}
    assert all(result["attempted"] == 1 for result in rows)


def test_source_or_generation_changes_are_not_combined():
    rows = aggregate([row(source_fingerprint="a"), row(source_fingerprint="b")])
    assert len({r["group_id"] for r in rows}) == 2
    rows = aggregate([row(model_metadata={"generation": {"top_k": 0}}),
                      row(model_metadata={"generation": {"top_k": 50}})])
    assert len({r["group_id"] for r in rows}) == 2


def test_different_fingerprints_are_not_matched():
    a = row(task_fingerprint="before")
    b = row(strategy="structured", task_fingerprint="after", config={**a["config"], "strategy": "structured"})
    rows = aggregate([a, b])
    assert all(not r["matched_tasks_and_repetition_labels"] for r in rows)
    assert all(not r["comparison_valid"] for r in rows)


@pytest.mark.parametrize("flag", [{"budget": {"overrun": True}}, {"usage_unknown": True}])
def test_invalid_budget_never_becomes_a_fixed_budget_win(flag):
    result = aggregate([row(**flag)])[0]
    assert result["attempted"] == result["successes"] == 1
    assert result["invalid_budget_attempts"] == 1
    assert result["success_rate"] is None
    assert result["comparison_valid"] is False


def test_invalid_budget_warning_is_visible(tmp_path):
    run = tmp_path / "runs" / "a"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps(row(budget={"overrun": True})), encoding="utf-8")
    paths = generate_report(run.parent, tmp_path / "report")
    for kind in ["markdown", "html"]:
        assert "not valid fixed-budget wins" in paths[kind].read_text(encoding="utf-8")


def test_report_escapes_html_and_writes_three_formats(tmp_path):
    run = tmp_path / "runs" / "a"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps(row(model='<script>alert("x")</script>')), encoding="utf-8")
    paths = generate_report(run.parent, tmp_path / "report")
    assert set(paths) == {"csv", "markdown", "html"}
    assert all(path.exists() for path in paths.values())
    rendered = paths["html"].read_text(encoding="utf-8")
    assert '<script>alert' not in rendered
    assert '&lt;script&gt;' in rendered
    assert 'not LLM performance evidence' in rendered


def test_run_ledger_shows_official_empty_patch_and_infrastructure_reason(tmp_path):
    for name, evaluation in [
        ("infra", {"status": "infrastructure_error", "infra_failure": True, "infra_failure_reason": "dependency <unavailable>"}),
        ("empty", {"status": "official_empty_patch", "tests_executed": False}),
    ]:
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        (run / "result.json").write_text(json.dumps(row(task_id=name, resolved=None, evaluation=evaluation)), encoding="utf-8")
    paths = generate_report(tmp_path / "runs", tmp_path / "report")
    for kind in ["markdown", "html"]:
        text = paths[kind].read_text(encoding="utf-8")
        assert "official_empty_patch" in text
        assert "tests not executed" in text
        assert "infrastructure_error" in text
        assert "ungraded: infrastructure error" in text
        assert "failed verification" not in text
        assert "0.0%" not in text
    assert "dependency &lt;unavailable&gt;" in paths["html"].read_text(encoding="utf-8")
    assert "infrastructure_errors,official_empty_patches" in paths["csv"].read_text(encoding="utf-8-sig")


def test_empty_reports_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match="No result"):
        generate_report(tmp_path, tmp_path / "report")


def test_behavior_diagnostics_show_stored_counts_without_changing_summary_csv(tmp_path):
    run = tmp_path / "runs" / "recorded"
    run.mkdir(parents=True)
    result = row(run_id="recorded", diagnostics={"schema_version": 1, "counts": {
        "successful_read_file": 5, "successful_edit_file": 2, "successful_write_file": 1,
        "successful_run": 4, "successful_diff": 1, "invalid_actions": 2,
        "repeated_failed_actions": 3, "repeated_observations": 6,
    }})
    path = run / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    before = path.read_bytes()
    rendered = generate_report(run.parent, tmp_path / "report")
    markdown = rendered["markdown"].read_text(encoding="utf-8")
    assert "| recorded | task-a | full | 5 | 3 | 4 | 1 | 2 | 3 | 6 |" in markdown
    html = rendered["html"].read_text(encoding="utf-8")
    assert "<td>recorded</td><td>task-a</td><td>full</td><td>5</td><td>3</td><td>4</td><td>1</td><td>2</td><td>3</td><td>6</td>" in html
    assert "Tool success does not establish a correct repair" in markdown and "repetition does not necessarily indicate an error" in html
    assert "successful_read_file" not in rendered["csv"].read_text(encoding="utf-8-sig")
    assert path.read_bytes() == before


def test_legacy_behavior_counts_are_unavailable_without_reading_trajectories(tmp_path):
    run = tmp_path / "runs" / "legacy"
    run.mkdir(parents=True)
    path = run / "result.json"
    path.write_text(json.dumps(row()), encoding="utf-8")
    before = path.read_bytes()
    (run / "trajectory.jsonl").write_text("not valid JSON; this file must not be read", encoding="utf-8")
    rendered = generate_report(run.parent, tmp_path / "report")
    for kind in ("markdown", "html"):
        text = rendered[kind].read_text(encoding="utf-8")
        assert "Run behavior diagnostics" in text and "unavailable" in text
        assert "does not reanalyze old trajectories" in text
    assert "| legacy | task-a | full | " + " | ".join(["unavailable"] * 7) + " |" in rendered["markdown"].read_text(encoding="utf-8")
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", [
    {"evaluation": {"status": "official_empty_patch"}},
    {"evaluation": {"status": "pending_official_evaluation"}},
    {"evaluation": {"status": "not_evaluated"}},
    {"evaluation": {"status": "ungraded"}},
    {"evaluation": {"tests_executed": False}},
    {"evaluation": {"status": "official_swebench", "resolved": False}},
    {"evaluation": {"status": "official_swebench", "infra_failure": True}},
    {"evaluation": {"status": "grading_error"}},
    {"status": "environment_error"},
    {"status": "interrupted"},
    {"status": "pending_official_evaluation"},
])
def test_aggregate_html_and_paired_share_verifier_interpretation(tmp_path, change):
    from budget_agent.paired import analyze_paired

    run = tmp_path / "candidate"
    run.mkdir()
    candidate = row(task_fingerprint="task-fingerprint", **change)
    candidate["_result_path"] = str(run / "result.json")
    Path(candidate["_result_path"]).write_text(json.dumps(candidate), encoding="utf-8")
    baseline = row(strategy="recent", task_fingerprint="task-fingerprint", resolved=False,
                   config={**candidate["config"], "strategy": "recent"}, _result_path=str(tmp_path / "baseline" / "result.json"))
    grouped = aggregate([candidate])[0]
    paired = analyze_paired([baseline, candidate], "recent", "full")
    paths = generate_report(run, tmp_path / "report")
    assert grouped["successes"] == grouped["graded"] == 0
    assert grouped["success_rate"] is None
    assert paired["candidate_wins"] == paired["eligible_graded_pairs"] == 0
    assert "<td>0 / 0 / 1</td>" in paths["html"].read_text(encoding="utf-8")
    assert "<td>success</td>" not in paths["html"].read_text(encoding="utf-8")


def test_read_loop_report_uses_stored_signals_and_preserves_legacy_unknowns(tmp_path):
    signals = [
        {"kind": "repeated_visible_read", "event_id": "e0002", "source_event_ids": ["e0001"],
         "path": "<script>file|name.py", "start_line": 1, "end_line": 10},
        {"kind": "warning_sent", "after_event_id": "e0002"},
    ]
    cases = {
        "recover": row(run_id="recover", loop_control={"policy": "recover", "events": signals, "warnings_sent": 1, "stopped": True}),
        "off": row(run_id="off", loop_control={"policy": "off", "events": [], "warnings_sent": 0, "stopped": False}),
        "legacy": row(run_id="legacy"),
    }
    originals = {}
    for name, result in cases.items():
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        path = run / "result.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        originals[path] = path.read_bytes()
        (run / "trajectory.jsonl").write_text("must not read this malformed trajectory", encoding="utf-8")
    rendered = generate_report(tmp_path / "runs", tmp_path / "report")
    markdown = rendered["markdown"].read_text(encoding="utf-8")
    page = rendered["html"].read_text(encoding="utf-8")
    assert "| recover | recover | 1 | 1 | True |" in markdown
    assert "| off | off | not monitored | 0 | False |" in markdown
    assert "| legacy | " + " | ".join(["unavailable"] * 5) + " |" in markdown
    assert "e0002 &lt;- e0001: &lt;script&gt;file\\|name.py:1-10" in markdown
    assert "Read-loop control" in page
    assert "<script>file" not in page and "<script>file" not in markdown
    assert "&lt;script&gt;file|name.py:1-10" in page
    assert all(path.read_bytes() == content for path, content in originals.items())


@pytest.mark.parametrize("control", [None, {"policy": "off", "events": [], "warnings_sent": 0, "stopped": False}])
def test_read_loop_table_is_omitted_when_no_controller_was_active(tmp_path, control):
    run = tmp_path / "run"
    run.mkdir()
    value = row() if control is None else row(loop_control=control)
    (run / "result.json").write_text(json.dumps(value), encoding="utf-8")
    paths = generate_report(run, tmp_path / "report")
    assert "Read-loop control" not in paths["html"].read_text(encoding="utf-8")
