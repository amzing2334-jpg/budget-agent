"""Sanitized SWE-bench import and official-harness interchange.

Inference never sees patch/test_patch or grading labels. Official grading stays
outside the agent process; a report must match the actual exported patch.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import uuid

from .report import condition_key, load_results
from .schema import Task


def _rows(path: Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("Input dataset is empty")
    rows = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.split("\n") if line.strip()]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Expected a JSON array or JSONL objects")
    return rows


def image_name(instance_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+", instance_id):
        raise ValueError(f"Invalid SWE-bench instance ID: {instance_id}")
    return "swebench/sweb.eval.x86_64." + instance_id.lower().replace("__", "_1776_") + ":latest"


def import_tasks(input_path: Path, output_path: Path, split: str = "dev", limit: int | None = None) -> int:
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    source = _rows(input_path)
    source = source[:limit] if limit else source
    tasks = []
    seen = set()
    for row in source:
        instance_id = row.get("instance_id", "")
        if instance_id in seen:
            raise ValueError(f"Duplicate instance: {instance_id}")
        seen.add(instance_id)
        issue = row.get("problem_statement")
        if not isinstance(issue, str) or not issue.strip():
            raise ValueError(f"Missing problem_statement for {instance_id}")
        base_commit = row.get("base_commit")
        if not isinstance(base_commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", base_commit):
            raise ValueError(f"base_commit must be a full 40-character hexadecimal Git commit for {instance_id}")
        # Deliberately whitelist metadata; hints_text can contain solution clues.
        metadata = {key: row[key] for key in ("repo", "base_commit", "version") if key in row}
        metadata["benchmark"] = "SWE-bench"
        task = Task(id=instance_id, issue=issue, split=split, kind="swebench", image=image_name(instance_id), workdir="/testbed", metadata=metadata)
        tasks.append(asdict(task))
    output_path = Path(output_path)
    if output_path.resolve() == Path(input_path).resolve():
        raise ValueError("Sanitized manifest must not overwrite the source dataset")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), encoding="utf-8")
    return len(tasks)


def _read_patch(result: dict) -> str:
    directory = Path(result["_result_path"]).parent
    candidate = result.get("patch_path") or "patch.diff"
    recorded = PurePosixPath(candidate.replace("\\", "/"))
    if recorded.is_absolute() or PureWindowsPath(candidate).is_absolute():
        original = PurePosixPath((result.get("result_path") or "").replace("\\", "/"))
        # The runner saves this pair together. A copied run owns its sibling patch.
        if original.name == "result.json" and recorded == original.with_name("patch.diff"):
            candidate = "patch.diff"
    # model_patch is UTF-8 text. A text-mode read would translate hunk bytes.
    return (directory / candidate).read_bytes().decode("utf-8")


def export_predictions(run_root: Path, output_path: Path) -> int:
    results = [r for r in load_results(run_root) if r.get("task_kind") == "swebench"]
    if not results:
        raise ValueError("No SWE-bench runs found; local smoke tasks cannot be exported as benchmark predictions")
    conditions = {(condition_key(r), r.get("strategy")) for r in results}
    if len(conditions) != 1:
        raise ValueError("Export one strategy/model/configuration at a time")
    seen = set()
    predictions = []
    manifest = {}
    for result in results:
        instance = result["task_id"]
        image_name(instance)  # Validate external identifier before use in paths.
        if instance in seen:
            raise ValueError(f"Duplicate attempt for {instance}; select one run per instance explicitly")
        seen.add(instance)
        patch = _read_patch(result)
        model = f'{result.get("model") or result.get("provider")}-{result.get("strategy")}'
        predictions.append({"instance_id": instance, "model_name_or_path": model, "model_patch": patch})
        manifest[instance] = {"result_path": result["_result_path"], "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(), "model_name_or_path": model}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions), encoding="utf-8")
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(predictions)


def evaluation_command(predictions_path: Path, dataset_name: str = "SWE-bench/SWE-bench_Lite", run_id: str | None = None, max_workers: int = 2, python_executable: str | None = None, *, instance_ids: list[str] | None = None, timeout: int | None = None) -> list[str]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if run_id is None:
        run_id = "budget-agent-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id must contain only letters, digits, dot, underscore, or hyphen")
    command = [python_executable or sys.executable, "-m", "swebench.harness.run_evaluation", "--dataset_name", dataset_name, "--predictions_path", str(Path(predictions_path).resolve()), "--max_workers", str(max_workers), "--run_id", run_id]
    if instance_ids is not None:
        if not instance_ids:
            raise ValueError("instance_ids must not be empty; omit it to evaluate all predictions")
        for instance_id in instance_ids:
            image_name(instance_id)
        command += ["--instance_ids", *instance_ids]
    if timeout is not None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        command += ["--timeout", str(timeout)]
    return command


def _empty_patch_grades(by_id: dict[str, dict], summary_path: Path, predictions_path: Path) -> list[tuple[dict, dict]]:
    """Verify explicitly selected aggregate evidence against its export manifest."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("schema_version") != 2:
        raise ValueError("Expected an official SWE-bench aggregate report with schema_version 2")
    ids = {}
    for field in ("submitted_ids", "empty_patch_ids"):
        values = summary.get(field)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values) or len(values) != len(set(values)):
            raise ValueError(f"Invalid or duplicate {field} in official aggregate report")
        ids[field] = set(values)
    if not ids["empty_patch_ids"] <= ids["submitted_ids"]:
        raise ValueError("Official empty_patch_ids must be submitted predictions")
    if summary.get("empty_patch_instances") != len(ids["empty_patch_ids"]):
        raise ValueError("Official empty_patch_instances does not match empty_patch_ids")

    predictions = _rows(predictions_path)
    manifest_path = predictions_path.with_suffix(predictions_path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_ids = [prediction.get("instance_id") for prediction in predictions]
    if not all(isinstance(instance, str) for instance in prediction_ids) or len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Invalid or duplicate prediction instance IDs")
    if not isinstance(manifest, dict) or set(prediction_ids) != set(manifest) or set(prediction_ids) != set(by_id) or set(prediction_ids) != ids["submitted_ids"]:
        raise ValueError("Prediction, manifest, local run and official submitted IDs must match")

    candidates = []
    for prediction in predictions:
        instance = prediction["instance_id"]
        result = by_id[instance]
        record = manifest[instance]
        patch = prediction.get("model_patch")
        if not isinstance(record, dict) or not isinstance(patch, str):
            raise ValueError(f"Invalid exported prediction or manifest: {instance}")
        digest = hashlib.sha256(patch.encode("utf-8")).hexdigest()
        if record.get("patch_sha256") != digest or _read_patch(result) != patch:
            raise ValueError(f"Prediction manifest patch does not match agent patch: {instance}")
        model = f'{result.get("model") or result.get("provider")}-{result.get("strategy")}'
        if prediction.get("model_name_or_path") != model or record.get("model_name_or_path") != model:
            raise ValueError(f"Prediction manifest model does not match local run: {instance}")
        # Downloads can move a run from Linux to Windows; its unique directory
        # and result filename must still identify the original exported attempt.
        exported_path = PurePosixPath(str(record.get("result_path", "")).replace("\\", "/"))
        local_path = Path(result["_result_path"])
        if exported_path.parts[-2:] != (local_path.parent.name, local_path.name):
            raise ValueError(f"Prediction manifest result path does not match local run: {instance}")
        if instance in ids["empty_patch_ids"]:
            if patch != "":
                raise ValueError(f"Official empty patch has a nonempty exported patch: {instance}")
            candidates.append((result, {
                "status": "official_empty_patch", "resolved": None,
                "tests_executed": False, "patch_sha256": digest,
                "summary_path": str(summary_path.resolve()),
                "predictions_path": str(predictions_path.resolve()),
                "manifest_path": str(manifest_path.resolve()),
            }))
    return candidates


def _grading_inputs(report: Path, grade: dict) -> dict[str, str] | None:
    """Identify the archived evaluation script and primary test targets, independent of their outcomes."""
    status = grade.get("tests_status")
    script = report.with_name("eval.sh")
    if (grade.get("patch_is_None") is not False or grade.get("patch_exists") is not True
            or grade.get("patch_successfully_applied") is not True or grade.get("infra_failure") is not False
            or not isinstance(status, dict) or not script.is_file()):
        return None

    targets = {}
    for category in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        outcomes = status.get(category)
        if not isinstance(outcomes, dict) or set(outcomes) != {"success", "failure"}:
            return None
        success, failure = outcomes["success"], outcomes["failure"]
        if not isinstance(success, list) or not isinstance(failure, list):
            return None
        names = success + failure
        if not all(isinstance(name, str) and name.strip() for name in names):
            return None
        targets[category] = sorted(names)
    names = targets["FAIL_TO_PASS"] + targets["PASS_TO_PASS"]
    if not names or len(names) != len(set(names)):
        return None
    script_bytes = script.read_bytes()
    if not script_bytes.strip():
        return None
    canonical = json.dumps(targets, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {"kind": "swebench", "eval_script_sha256": hashlib.sha256(script_bytes).hexdigest(),
            "test_targets_sha256": hashlib.sha256(canonical).hexdigest()}


def check_controls(spec: dict, gold_report: Path, negative_report: Path) -> dict:
    """Check archived positive and negative controls against predeclared patch and target hashes."""
    return _check_control_reports(spec, {"gold": Path(gold_report), "negative": Path(negative_report)})


def check_gold_control(spec: dict, gold_report: Path) -> dict:
    """Check whether the gold result qualifies the next control for dispatch."""
    return _check_control_reports(spec, {"gold": Path(gold_report)})


def _check_control_reports(spec: dict, reports: dict[str, Path]) -> dict:
    if not isinstance(spec, dict) or not isinstance(spec.get("instance_id"), str):
        raise ValueError("Control spec must include an instance_id")
    instance = spec["instance_id"]
    image_name(instance)
    if spec.get("gold_patch_sha256") == spec.get("negative_patch_sha256"):
        raise ValueError("Gold and negative controls must specify different patch hashes")

    controls = {}
    expected_inputs = {"kind": "swebench", "eval_script_sha256": spec.get("eval_script_sha256"),
                       "test_targets_sha256": spec.get("test_targets_sha256")}
    for role, report in reports.items():
        if report.name != "report.json" or not report.is_file():
            raise ValueError(f"Missing {role} report.json")
        report_bytes = report.read_bytes()
        payload = json.loads(report_bytes)
        grade = payload.get(instance) if isinstance(payload, dict) else None
        if not isinstance(grade, dict):
            raise ValueError(f"{role} report has no grade for {instance}")
        patch = report.with_name("patch.diff")
        if not patch.is_file():
            raise ValueError(f"Missing {role} patch.diff")
        data = patch.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if not data.strip() or digest != spec.get(f"{role}_patch_sha256"):
            raise ValueError(f"{role} patch is empty or differs from the control spec")
        inputs = _grading_inputs(report, grade)
        if inputs is None:
            raise ValueError(f"Incomplete {role} grading inputs or unapplied patch")
        if inputs != expected_inputs:
            raise ValueError(f"{role} evaluation script or test targets differ from the control spec")
        if grade.get("resolved") is not (role == "gold"):
            raise ValueError(f"Unexpected {role} resolved verdict")
        counts = {group: {state: len(names) for state, names in grade["tests_status"][group].items()}
                  for group in ("FAIL_TO_PASS", "PASS_TO_PASS")}
        if counts["PASS_TO_PASS"]["failure"]:
            raise ValueError(f"{role} control has a PASS_TO_PASS regression")
        if bool(counts["FAIL_TO_PASS"]["failure"]) != (role == "negative"):
            raise ValueError(f"{role} control has unexpected FAIL_TO_PASS outcomes")
        controls[role] = {"report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                          "patch_sha256": digest, "test_counts": counts}
    return {"instance_id": instance, "grading_inputs": expected_inputs, **controls}


def attach_grades(runsdir: Path, reportsdir: Path, *, summary_path: Path | None = None,
                  predictions_path: Path | None = None) -> dict[str, int]:
    """Attach verified official reports; reject missing/mismatched patch provenance.

    Pass exactly one official evaluation run directory, not its global parent.
    Each instance report must include its sibling patch.diff. Reports without
    resolved booleans remain ungraded rather than becoming model failures.
    To import empty-patch outcomes, explicitly provide the aggregate summary and
    original predictions with their export manifest. No global summaries are
    discovered. Infrastructure failures retain the official value separately.
    """
    if (summary_path is None) != (predictions_path is None):
        raise ValueError("summary_path and predictions_path must be provided together")
    results = [r for r in load_results(runsdir) if r.get("task_kind") == "swebench"]
    by_id = {}
    for result in results:
        if result["task_id"] in by_id:
            raise ValueError("Duplicate attempts: attach grades to a directory with one attempt per task")
        by_id[result["task_id"]] = result
    candidates = []
    seen = set()
    skipped = 0
    for path in sorted(Path(reportsdir).rglob("report.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for instance, grade in payload.items():
            if instance not in by_id or not isinstance(grade, dict):
                continue
            if instance in seen:
                raise ValueError(f"Multiple official reports for {instance}; choose a single evaluation run")
            seen.add(instance)
            if type(grade.get("resolved")) is not bool:
                skipped += 1
                continue
            patch_path = path.parent / "patch.diff"
            if not patch_path.is_file():
                raise ValueError(f"Missing official patch.diff beside {path}; cannot verify evaluation provenance")
            result = by_id[instance]
            graded_patch = patch_path.read_bytes()
            if graded_patch != _read_patch(result).encode("utf-8"):
                raise ValueError(f"Official report patch does not match agent patch: {instance}")
            infra_failure = grade.get("infra_failure") is True
            evaluation = {
                "status": "infrastructure_error" if infra_failure else "official_swebench",
                "resolved": None if infra_failure else grade["resolved"],
                "official_resolved": grade["resolved"],
                "report_path": str(path.resolve()),
                "patch_sha256": hashlib.sha256(graded_patch).hexdigest(),
            }
            # Keep grading metadata without copying hidden test names into runs.
            for key in ("infra_failure", "infra_failure_reason", "patch_is_None", "patch_exists", "patch_successfully_applied"):
                if key in grade:
                    evaluation[key] = grade[key]
            if graded_patch:
                inputs = _grading_inputs(path, grade)
                if inputs is not None:
                    evaluation["grading_inputs"] = inputs
            candidates.append((result, evaluation))
    if summary_path is not None and predictions_path is not None:   # provided together (checked above)
        empty_grades = _empty_patch_grades(by_id, Path(summary_path), Path(predictions_path))
        for result, _ in empty_grades:
            if result["task_id"] in seen:
                raise ValueError(f"Conflicting instance report and official empty patch: {result['task_id']}")
        candidates.extend(empty_grades)
    # Validate the whole batch before changing any result.
    for result, evaluation in candidates:
        target = Path(result.pop("_result_path"))
        result["resolved"] = evaluation["resolved"]
        result["evaluation"] = evaluation
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"attached": len(candidates), "ungraded_reports": skipped, "unmatched_runs": len(results) - len(candidates),
            "infrastructure_errors": sum(grade["status"] == "infrastructure_error" for _, grade in candidates),
            "official_empty_patches": sum(grade["status"] == "official_empty_patch" for _, grade in candidates)}
