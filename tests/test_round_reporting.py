"""Round reporting: paired comparison with suspicious samples dropped, and the round summary."""
import importlib.util
import json
from pathlib import Path
import sys


def load(name, script):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_pools = load("compare_pools", "compare_pools.py")
summarize_round = load("summarize_round", "summarize_round.py")


def ledger(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def attempt(task, resolved, signals=(), **extra):
    return {"task_id": task, "split": "eval", "sample": 0, "status": "finished", "resolved": resolved,
            "patch_signals": list(signals), "ignored_changes": [], **extra}


def run_compare(monkeypatch, capsys, baseline, candidate, *flags):
    monkeypatch.setattr(sys, "argv", ["compare_pools", "--baseline", str(baseline), "--candidate", str(candidate),
                                      "--resamples", "200", *flags])
    assert compare_pools.main() == 0
    return json.loads(capsys.readouterr().out)


def test_drop_flagged_removes_only_resolved_attempts_with_signals(tmp_path, monkeypatch, capsys):
    baseline = ledger(tmp_path / "base.jsonl", [attempt("a", False), attempt("a", False),
                                                attempt("b", True), attempt("b", False)])
    candidate = ledger(tmp_path / "cand.jsonl", [attempt("a", True, ["process_exit"]),
                                                 attempt("a", False, ["test_file_modified"]),
                                                 attempt("b", True), attempt("b", True)])
    plain = run_compare(monkeypatch, capsys, baseline, candidate)
    assert plain["delta"] == 0.5 and plain["dropped_flagged"] is None
    dropped = run_compare(monkeypatch, capsys, baseline, candidate, "--drop-flagged")
    assert dropped["delta"] == 0.25 and dropped["candidate_pass@1"] == 0.5
    assert dropped["dropped_flagged"] == {"baseline": 0, "candidate": 1}


def test_round_summary_reports_pools_audit_training_and_cost(tmp_path, monkeypatch):
    pool = tmp_path / "rollouts" / "student-sft-eval"
    run = pool / "runs" / "s00" / "run-a"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({"patch_signals": [
        {"signal": "process_exit", "file": "mod.py", "line": "os._exit(0)"},
        {"signal": "test_file_modified", "file": "tests/test_mod.py", "line": ""}]}), encoding="utf-8")
    events = [("edit_file", False), ("edit_file", True), ("replace_lines", True), ("run", True)]
    ledger(run / "trajectory.jsonl", [{"action": {"tool": tool}, "ok": ok} for tool, ok in events])
    ledger(pool / "attempts.jsonl", [
        attempt("a", True, ["process_exit", "test_file_modified"], result_path=str(run / "result.json"),
                ignored_changes=["tests/test_mod.py"]),
        attempt("a", False), attempt("b", True), attempt("b", False, ["process_exit"])])
    (pool / "summary.json").write_text(json.dumps({
        "splits": {"eval": {"tasks": 2, "attempts": 4, "pass@1": 0.5, "pass@2": 0.75, "solved_tasks": 2,
                            "statuses": {"finished": 4}}},
        "attempts_per_hour": 100.0, "wall_seconds": 144.0, "tokens": {"input": 10, "output": 5}}), encoding="utf-8")
    (tmp_path / "rollouts" / "unfinished").mkdir()
    out = tmp_path / "outputs"
    ledger(out / "sft.jsonl", [{"task_id": "t1", "mode": "last"}, {"task_id": "t1", "mode": "last"},
                               {"task_id": "t2", "mode": "full"}])
    ledger(out / "sft" / "train_log.jsonl", [{"step": 1, "loss": 1.23456}, {"step": 2, "loss": 0.5}])
    (out / "sft" / "result.json").write_text(json.dumps({"examples": 3}), encoding="utf-8")
    for name, low in (("eval", 0.0), ("eval_drop_flagged", -0.25)):
        (out / f"compare_{name}.json").write_text(json.dumps({
            "baseline_pass@1": 0.25, "candidate_pass@1": 0.5, "delta": 0.25, "ci95": [low, 0.5],
            "verdict": "no detectable change"}), encoding="utf-8")
    ledger(out / "steps.jsonl", [{"step": "teacher-train", "seconds": 3600}, {"step": "train-sft", "seconds": 1800}])

    monkeypatch.setattr(sys, "argv", ["summarize_round", "--rollouts", str(tmp_path / "rollouts"),
                                      "--outputs", str(out)])
    assert summarize_round.main() == 0

    report = json.loads((out / "results.json").read_text())
    stats = report["pools"]["student-sft-eval"]
    assert list(report["pools"]) == ["student-sft-eval"]
    assert stats["edit_success"] == {"edit_file": "1/2", "replace_lines": "1/1"}
    assert stats["patch_signals"] == {"process_exit": 2, "test_file_modified": 1}
    assert [(s["signal"], s["graded"]) for s in stats["tamper_audit"][0]["signals"]] == [
        ("process_exit", True), ("test_file_modified", False)]
    assert len(stats["tamper_audit"]) == 1
    assert report["sft"] == {"examples": 3, "tasks": 2, "by_mode": {"last": 2, "full": 1}}
    assert report["loss_curve"] == [[1, 1.2346], [2, 0.5]]
    assert sorted(report["compare"]) == ["eval", "eval_drop_flagged"]
    assert (report["gpu_hours"], report["cost_cny"]) == (1.5, 4.92)
    markdown = (out / "results.md").read_text()
    assert "Paired eval: base 0.250 → candidate 0.500, delta +0.250, 95% CI [+0.000, +0.500]" in markdown
    assert "Tamper audit student-sft-eval: 1 resolved attempts with signals, 1 in graded files" in markdown
    assert "process_exit in mod.py (graded)" in markdown and "tests/test_mod.py (ignored)" in markdown
