import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath
import shutil
import subprocess

import pytest

from budget_agent.swebench import attach_grades, evaluation_command, export_predictions, image_name, import_tasks


def source():
    return {"instance_id": "pallets__flask-1234", "problem_statement": "Fix broken route handling", "repo": "pallets/flask", "base_commit": "abcdef" * 6 + "abcd", "version": "1.0", "patch": "SECRET_GOLD_PATCH", "test_patch": "SECRET_TEST", "FAIL_TO_PASS": ["hidden_test"], "PASS_TO_PASS": ["regression"], "hints_text": "solution spoiler"}


def write_run(root, name="r1", **changes):
    path = root / name
    path.mkdir(parents=True)
    result = {"task_id": "pallets__flask-1234", "split": "dev", "task_kind": "swebench", "provider": "openai-compatible", "model": "test-model", "strategy": "full", "config": {"strategy": "full", "context_tokens": 4000, "seed": 0}, "status": "pending_official_evaluation", "resolved": None, "patch_path": "patch.diff", "evidence_kind": "model_run", "usage_sources": ["provider"]}
    result.update(changes)
    (path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (path / "patch.diff").write_text("diff --git a/a.py b/a.py\n+fixed\n", encoding="utf-8")
    return path


def test_import_whitelists_fields_and_never_leaks_labels(tmp_path):
    original = tmp_path / "source.json"
    original.write_text(json.dumps([source()]), encoding="utf-8")
    target = tmp_path / "tasks.jsonl"
    assert import_tasks(original, target, "test") == 1
    content = target.read_text(encoding="utf-8")
    assert all(secret not in content for secret in ["SECRET", "hidden_test", "regression", "spoiler", "FAIL_TO_PASS"])
    task = json.loads(content)
    assert task["split"] == "test"
    assert task["kind"] == "swebench"
    assert task["workdir"] == "/testbed"
    assert task["image"] == "swebench/sweb.eval.x86_64.pallets_1776_flask-1234:latest"
    assert "SECRET" in original.read_text(encoding="utf-8")


@pytest.mark.parametrize("commit", [None, "", "abcdef", "x" * 40, "HEAD; echo unsafe"])
def test_import_requires_exact_git_commit(tmp_path, commit):
    original = tmp_path / "source.json"
    original.write_text(json.dumps([{**source(), "base_commit": commit}]), encoding="utf-8")
    with pytest.raises(ValueError, match="base_commit"):
        import_tasks(original, tmp_path / "out")


def test_import_split_matches_task_loader(tmp_path):
    with pytest.raises(ValueError, match="dev or test"):
        import_tasks(tmp_path / "source.json", tmp_path / "out", split="train")


def test_import_rejects_duplicate_ids_and_overwrite(tmp_path):
    original = tmp_path / "source.json"
    original.write_text(json.dumps([source(), source()]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        import_tasks(original, tmp_path / "out")
    original.write_text(json.dumps([source()]), encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        import_tasks(original, original)


def test_invalid_instance_cannot_escape_path():
    with pytest.raises(ValueError):
        image_name("../../bad")


def test_export_is_official_format_and_skips_local(tmp_path):
    write_run(tmp_path / "runs")
    write_run(tmp_path / "runs", "local", task_kind="local")
    target = tmp_path / "predictions.jsonl"
    assert export_predictions(tmp_path / "runs", target) == 1
    item = json.loads(target.read_text(encoding="utf-8"))
    assert set(item) == {"instance_id", "model_name_or_path", "model_patch"}
    assert "+fixed" in item["model_patch"]
    assert target.with_suffix(".jsonl.manifest.json").is_file()


def test_moved_runner_result_uses_only_its_loaded_sibling(tmp_path):
    original = tmp_path / "original" / "r1"
    write_run(original.parent, result_path=str(original / "result.json"),
              patch_path=str(original / "patch.diff"))
    copy = shutil.copytree(original, tmp_path / "downloaded" / "r1")
    (copy / "patch.diff").write_bytes(b"copied attempt\n")
    predictions = tmp_path / "predictions.jsonl"
    export_predictions(copy, predictions)
    assert json.loads(predictions.read_bytes())["model_patch"] == "copied attempt\n"
    (copy / "patch.diff").unlink()
    with pytest.raises(FileNotFoundError):
        export_predictions(copy, tmp_path / "missing.jsonl")


@pytest.mark.parametrize("recorded", [PurePosixPath("/remote/runs/r1"), PureWindowsPath("Z:/remote/runs/r1")],
                         ids=["posix", "windows"])
def test_downloaded_runner_result_accepts_either_path_flavor(tmp_path, recorded):
    run = write_run(tmp_path / "downloads", result_path=str(recorded / "result.json"),
                    patch_path=str(recorded / "patch.diff"))
    (run / "patch.diff").write_bytes(b"downloaded\n")
    predictions = tmp_path / "predictions.jsonl"
    export_predictions(run, predictions)
    assert json.loads(predictions.read_bytes())["model_patch"] == "downloaded\n"


def test_explicit_relative_patch_stays_relative_to_loaded_result(tmp_path):
    run = write_run(tmp_path / "runs", patch_path="exports/custom.patch")
    (run / "exports").mkdir()
    (run / "exports" / "custom.patch").write_bytes(b"custom\n")
    predictions = tmp_path / "predictions.jsonl"
    export_predictions(run, predictions)
    assert json.loads(predictions.read_bytes())["model_patch"] == "custom\n"


def test_export_rejects_duplicate_attempts(tmp_path):
    write_run(tmp_path / "runs")
    write_run(tmp_path / "runs", "second")
    with pytest.raises(ValueError, match="Duplicate attempt"):
        export_predictions(tmp_path / "runs", tmp_path / "out")


def test_export_rejects_mixed_strategies(tmp_path):
    write_run(tmp_path / "runs")
    write_run(tmp_path / "runs", "second", strategy="summary", task_id="pallets__flask-1235")
    with pytest.raises(ValueError, match="one strategy"):
        export_predictions(tmp_path / "runs", tmp_path / "out")


def test_evaluation_ids_are_fresh_and_arguments_are_structured(tmp_path):
    a = evaluation_command(tmp_path / "with space.jsonl")
    b = evaluation_command(tmp_path / "with space.jsonl")
    assert a[-1] != b[-1]
    assert a[a.index("--predictions_path") + 1].endswith("with space.jsonl")
    assert "swebench.harness.run_evaluation" in a
    assert a[a.index("--dataset_name") + 1] == "SWE-bench/SWE-bench_Lite"


def test_evaluation_can_limit_instances_and_test_runtime(tmp_path):
    command = evaluation_command(tmp_path / "predictions.jsonl", run_id="cloud-first-task", max_workers=1,
                                 instance_ids=["astropy__astropy-12907"], timeout=600)
    assert command[command.index("--instance_ids") + 1] == "astropy__astropy-12907"
    assert command[command.index("--timeout") + 1] == "600"
    assert command[command.index("--max_workers") + 1] == "1"


@pytest.mark.parametrize("kwargs, message", [({"instance_ids": []}, "must not be empty"),
                                            ({"instance_ids": ["../../bad"]}, "Invalid SWE-bench"),
                                            ({"timeout": 0}, "timeout must be positive")])
def test_evaluation_rejects_invalid_external_limits(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        evaluation_command(tmp_path / "predictions.jsonl", **kwargs)


def test_official_grade_requires_matching_patch(tmp_path):
    run = write_run(tmp_path / "runs", status="budget_exhausted")
    report = tmp_path / "reports" / "model" / "pallets__flask-1234"
    report.mkdir(parents=True)
    (report / "report.json").write_text(json.dumps({"pallets__flask-1234": {"resolved": True}}), encoding="utf-8")
    (report / "patch.diff").write_text("wrong patch", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        attach_grades(tmp_path / "runs", tmp_path / "reports")
    assert json.loads((run / "result.json").read_text())["resolved"] is None
    (report / "patch.diff").write_text((run / "patch.diff").read_text(), encoding="utf-8")
    summary = attach_grades(tmp_path / "runs", tmp_path / "reports")
    assert summary["attached"] == 1
    result = json.loads((run / "result.json").read_text())
    assert result["resolved"] is True
    assert result["evaluation"]["status"] == "official_swebench"
    assert result["status"] == "budget_exhausted"


def write_official_report(root, run, **grade):
    instance = json.loads((run / "result.json").read_text())["task_id"]
    directory = root / "model" / instance
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(json.dumps({instance: grade}), encoding="utf-8")
    (directory / "patch.diff").write_bytes((run / "patch.diff").read_bytes())
    return directory


def test_matching_non_utf8_artifacts_cannot_receive_a_prediction_grade(tmp_path):
    run = write_run(tmp_path / "runs")
    (run / "patch.diff").write_bytes(b"\xff")
    reports = tmp_path / "reports"
    write_official_report(reports, run, resolved=True)
    original = (run / "result.json").read_bytes()
    with pytest.raises(UnicodeDecodeError):
        attach_grades(run, reports)
    assert (run / "result.json").read_bytes() == original


@pytest.fixture
def crlf_run(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "core.autocrlf=false"]
    subprocess.run([*git, "init", "-q"], check=True)
    target = repo / "source.txt"
    before = "old\r\ncontext\u2028data\r\n".encode()
    after = before.replace(b"old", b"new")
    target.write_bytes(before)
    subprocess.run([*git, "add", "--", target.name], check=True)
    target.write_bytes(after)
    patch = subprocess.check_output([*git, "diff", "--no-ext-diff", "--", target.name])
    target.write_bytes(before)
    run = write_run(tmp_path / "runs")
    (run / "patch.diff").write_bytes(patch)
    return run, git, target, after


def test_exported_prediction_replays_exact_git_patch(tmp_path, crlf_run):
    run, git, target, after = crlf_run
    predictions = tmp_path / "predictions.jsonl"
    assert export_predictions(run, predictions) == 1
    prediction = json.loads(predictions.read_bytes())
    exported = prediction["model_patch"].encode()
    patch = (run / "patch.diff").read_bytes()
    assert exported == patch
    manifest = json.loads(predictions.with_suffix(".jsonl.manifest.json").read_bytes())
    assert manifest[prediction["instance_id"]]["patch_sha256"] == hashlib.sha256(patch).hexdigest()
    subprocess.run([*git, "apply", "-"], input=exported, check=True, capture_output=True)
    assert target.read_bytes() == after


def test_exact_report_attaches_with_unicode_in_prediction_jsonl(tmp_path, crlf_run):
    run, _, _, _ = crlf_run
    predictions = tmp_path / "predictions.jsonl"
    export_predictions(run, predictions)
    report = write_official_report(tmp_path / "reports", run, resolved=True)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"schema_version": 2, "submitted_ids": [report.name],
                                   "empty_patch_ids": [], "empty_patch_instances": 0}), encoding="utf-8")
    attached = attach_grades(run, report, summary_path=summary, predictions_path=predictions)
    result = json.loads((run / "result.json").read_bytes())
    assert attached["attached"] == 1 and result["resolved"] is True
    assert result["evaluation"]["patch_sha256"] == hashlib.sha256((run / "patch.diff").read_bytes()).hexdigest()


def test_report_with_one_changed_hunk_line_ending_is_rejected(tmp_path, crlf_run):
    run, git, target, after = crlf_run
    report = write_official_report(tmp_path / "reports", run, resolved=True)
    patch_path = report / "patch.diff"
    patch = patch_path.read_bytes()
    assert patch.count(b"+new\r\n") == 1
    patch_path.write_bytes(patch.replace(b"+new\r\n", b"+new\n"))
    subprocess.run([*git, "apply", str(patch_path)], check=True, capture_output=True)
    assert target.read_bytes() == after.replace(b"new\r\n", b"new\n")
    result_path = run / "result.json"
    before = result_path.read_bytes()
    with pytest.raises(ValueError, match="does not match"):
        attach_grades(run, report)
    assert result_path.read_bytes() == before


def empty_export(tmp_path):
    run = write_run(tmp_path / "runs")
    (run / "patch.diff").write_text("", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    export_predictions(run, predictions)
    summary = tmp_path / "official-summary.json"
    summary.write_text(json.dumps({"schema_version": 2, "submitted_ids": ["pallets__flask-1234"],
                                   "empty_patch_ids": ["pallets__flask-1234"], "empty_patch_instances": 1}), encoding="utf-8")
    return run, predictions, summary


def test_infrastructure_failure_retains_official_metadata_without_becoming_model_failure(tmp_path):
    run = write_run(tmp_path / "runs", status="budget_exhausted")
    write_official_report(tmp_path / "reports", run, resolved=False, infra_failure=True,
                          infra_failure_reason="dependency_install_failed", patch_exists=True,
                          patch_successfully_applied=False, tests_status={"FAIL_TO_PASS": ["hidden_test"]})
    attached = attach_grades(run, tmp_path / "reports")
    result = json.loads((run / "result.json").read_text())
    assert attached["infrastructure_errors"] == 1
    assert result["resolved"] is None
    assert result["status"] == "budget_exhausted"
    assert result["evaluation"]["status"] == "infrastructure_error"
    assert result["evaluation"]["official_resolved"] is False
    assert result["evaluation"]["infra_failure"] is True
    assert result["evaluation"]["infra_failure_reason"] == "dependency_install_failed"
    assert result["evaluation"]["patch_successfully_applied"] is False
    assert "hidden_test" not in (run / "result.json").read_text()


def test_infrastructure_failure_still_requires_patch_provenance(tmp_path):
    run = write_run(tmp_path / "runs")
    directory = write_official_report(tmp_path / "reports", run, resolved=False, infra_failure=True)
    (directory / "patch.diff").write_text("different attempt", encoding="utf-8")
    before = (run / "result.json").read_text()
    with pytest.raises(ValueError, match="does not match"):
        attach_grades(run, tmp_path / "reports")
    assert (run / "result.json").read_text() == before


def test_empty_patch_uses_explicit_summary_and_export_manifest(tmp_path):
    run, predictions, summary = empty_export(tmp_path)
    # A downloaded export may have originated at a different absolute root.
    manifest_path = predictions.with_suffix(".jsonl.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["pallets__flask-1234"]["result_path"] = "/remote/runs/r1/result.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    attached = attach_grades(run, tmp_path / "reports", summary_path=summary, predictions_path=predictions)
    result = json.loads((run / "result.json").read_text())
    assert attached["attached"] == attached["official_empty_patches"] == 1
    assert attached["unmatched_runs"] == 0
    assert result["resolved"] is None
    assert result["evaluation"]["status"] == "official_empty_patch"
    assert result["evaluation"]["tests_executed"] is False
    assert result["evaluation"]["summary_path"] == str(summary.resolve())
    assert result["evaluation"]["manifest_path"] == str(manifest_path.resolve())


def test_aggregate_is_never_discovered_implicitly(tmp_path):
    run, predictions, summary = empty_export(tmp_path)
    attached = attach_grades(run, summary.parent)
    assert attached["attached"] == 0
    assert "evaluation" not in json.loads((run / "result.json").read_text())


@pytest.mark.parametrize("argument", ["summary_path", "predictions_path"])
def test_summary_and_predictions_are_required_together(tmp_path, argument):
    with pytest.raises(ValueError, match="provided together"):
        attach_grades(tmp_path, tmp_path, **{argument: tmp_path / "file.json"})


@pytest.mark.parametrize("mutation", ["manifest_hash", "manifest_model", "manifest_result", "manifest_id",
                                      "prediction_patch", "duplicate_prediction", "summary_id", "summary_count"])
def test_empty_patch_rejects_mismatched_provenance_before_writing(tmp_path, mutation):
    run, predictions, summary = empty_export(tmp_path)
    instance = "pallets__flask-1234"
    manifest_path = predictions.with_suffix(".jsonl.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if mutation == "manifest_hash":
        manifest[instance]["patch_sha256"] = "incorrect"
    elif mutation == "manifest_model":
        manifest[instance]["model_name_or_path"] = "another-model"
    elif mutation == "manifest_result":
        manifest[instance]["result_path"] = "/different/attempt/result.json"
    elif mutation == "manifest_id":
        manifest["pallets__flask-9999"] = manifest.pop(instance)
    elif mutation == "prediction_patch":
        prediction = json.loads(predictions.read_text())
        prediction["model_patch"] = "unexpected patch"
        predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    elif mutation == "duplicate_prediction":
        predictions.write_text(predictions.read_text() * 2, encoding="utf-8")
    elif mutation == "summary_id":
        payload = json.loads(summary.read_text())
        payload["submitted_ids"] = payload["empty_patch_ids"] = ["pallets__flask-9999"]
        summary.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "summary_count":
        payload = json.loads(summary.read_text())
        payload["empty_patch_instances"] = 0
        summary.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = (run / "result.json").read_text()
    with pytest.raises(ValueError):
        attach_grades(run, tmp_path / "reports", summary_path=summary, predictions_path=predictions)
    assert (run / "result.json").read_text() == before


def test_conflicting_empty_summary_and_instance_grade_are_rejected_atomically(tmp_path):
    run, predictions, summary = empty_export(tmp_path)
    write_official_report(tmp_path / "reports", run, resolved=False, infra_failure=False)
    before = (run / "result.json").read_text()
    with pytest.raises(ValueError, match="Conflicting"):
        attach_grades(run, tmp_path / "reports", summary_path=summary, predictions_path=predictions)
    assert (run / "result.json").read_text() == before


def test_empty_summary_requires_an_actual_empty_prediction(tmp_path):
    run, predictions, summary = empty_export(tmp_path)
    (run / "patch.diff").write_text("not empty", encoding="utf-8")
    export_predictions(run, predictions)
    with pytest.raises(ValueError, match="nonempty exported patch"):
        attach_grades(run, tmp_path / "reports", summary_path=summary, predictions_path=predictions)
