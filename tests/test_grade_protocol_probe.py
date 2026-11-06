"""Exercise the real export/import path with synthetic official grader output."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from test_protocol_probe import artifacts, probe, write_json, write_rows  # noqa: F401 (pytest fixture)


@pytest.fixture
def grader(monkeypatch):
    monkeypatch.setitem(sys.modules, "probe_history_format", probe)
    spec = importlib.util.spec_from_file_location(
        "grade_protocol_probe", Path(__file__).parents[1] / "scripts/grade_protocol_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def completed_probe(artifacts):  # noqa: F811 (uses the imported fixture)
    task_id = "pallets__flask-1234"
    plan = json.loads((artifacts / "plan.json").read_text())
    plan["task_id"] = task_id
    write_json(artifacts / "plan.json", plan)
    for history_format in probe.FORMATS:
        folder = artifacts / history_format / "attempt"
        result = json.loads((folder / "result.json").read_text())
        result.update(task_id=task_id, model="test/model", actual_models=["test/model"],
                      evaluation={"status": "pending_official_evaluation", "resolved": None})
        write_json(folder / "result.json", result)
        calls = probe._rows(folder / "calls.jsonl")
        for call in calls:
            call["response"]["model"] = "test/model"
        write_rows(folder / "calls.jsonl", calls)
    return artifacts


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def arguments(runs, grading):
    return ["--runs", str(runs), "--grading-dir", str(grading),
            "--grader-python", "unused-grader-python", "--run-id", "test-protocol"]


def test_default_dry_plan_does_not_run_grader_or_write_files(grader, completed_probe, monkeypatch, capsys):
    grading = completed_probe.with_name(completed_probe.name + "-grading")
    before = snapshot(completed_probe)
    monkeypatch.setattr(grader.subprocess, "run", lambda *a, **kw: pytest.fail("dry run invoked grader"))
    grader.main(arguments(completed_probe, grading))
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["commands"]) == 2
    assert snapshot(completed_probe) == before
    assert not grading.exists()


def test_official_empty_patch_outputs_attach_to_both_real_exports(grader, completed_probe, monkeypatch):
    grading = completed_probe.with_name(completed_probe.name + "-grading")
    calls = []

    def official_empty_output(command, *, cwd, stdout, stderr, check):
        # Replace only the external grader. Export, provenance checking,
        # attachment, reporting and protocol summarization remain real.
        predictions = Path(command[command.index("--predictions_path") + 1])
        prediction = json.loads(predictions.read_text().strip())
        assert prediction["model_patch"] == ""
        assert predictions.with_suffix(".jsonl.manifest.json").is_file()
        run_id = command[command.index("--run_id") + 1]
        model_slug = prediction["model_name_or_path"].replace("/", "__")
        summary_path = cwd / f"{model_slug}.{run_id}.json"
        write_json(summary_path, {
            "schema_version": 2, "submitted_ids": [prediction["instance_id"]],
            "empty_patch_ids": [prediction["instance_id"]], "empty_patch_instances": 1,
            "completed_ids": [], "resolved_ids": [], "unresolved_ids": [],
        })
        stdout.write("No tests executed: empty patch\n")
        calls.append(predictions)

    monkeypatch.setattr(grader.subprocess, "run", official_empty_output)
    grader.main([*arguments(completed_probe, grading), "--execute"])
    assert len(calls) == 2
    after = json.loads((completed_probe / "protocol-after-grading.json").read_text())
    for history_format in probe.FORMATS:
        result = json.loads((completed_probe / history_format / "attempt/result.json").read_text())
        assert result["resolved"] is None
        assert result["evaluation"]["status"] == "official_empty_patch"
        assert result["evaluation"]["tests_executed"] is False
        assert after["formats"][history_format]["evaluation_status"] == "official_empty_patch"
        assert after["formats"][history_format]["official_resolved"] is None
    report = (completed_probe / "report/report.html").read_text(encoding="utf-8")
    assert "tests not executed" in report
    assert "failed verification" not in report
    assert "0.0%" not in report


def test_grading_inside_project_is_rejected_before_execution(grader, completed_probe, monkeypatch):
    project = Path(grader.__file__).resolve().parents[1]
    grading = project / "runs" / "forbidden-protocol-grading-test"
    existed = grading.exists()
    before = snapshot(completed_probe)
    monkeypatch.setattr(grader.subprocess, "run", lambda *a, **kw: pytest.fail("unsafe grader invocation"))
    with pytest.raises(SystemExit) as exc:
        grader.main([*arguments(completed_probe, grading), "--execute"])
    assert exc.value.code == 2
    assert grading.exists() == existed
    assert snapshot(completed_probe) == before


def test_reusing_grading_directory_preserves_existing_artifacts(grader, completed_probe, monkeypatch):
    grading = completed_probe.with_name(completed_probe.name + "-grading")
    grading.mkdir()
    (grading / "existing-summary.json").write_text("preserve this evidence", encoding="utf-8")
    before_grading, before_runs = snapshot(grading), snapshot(completed_probe)
    monkeypatch.setattr(grader.subprocess, "run", lambda *a, **kw: pytest.fail("reused grading directory"))
    with pytest.raises(FileExistsError):
        grader.main([*arguments(completed_probe, grading), "--execute"])
    assert snapshot(grading) == before_grading
    assert snapshot(completed_probe) == before_runs
