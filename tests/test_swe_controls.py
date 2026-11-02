import hashlib
import json
from pathlib import Path

import pytest

from budget_agent.cli import main
from budget_agent.swebench import check_controls
from budget_agent.tasks import load_tasks


TASK = "owner__repo-1234"
FIX, KEEP = "synthetic-fix-check", "synthetic-regression-check"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def test_14182_solver_manifest_only_changes_the_image_reference():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "runs/iteration-20261102-14182-controls/14182-spec.json").read_bytes())
    solver, image = spec["solver"], spec["image"]
    pilot, pinned = (root / solver[k] for k in ("manifest", "pinned_solver_manifest"))
    assert digest(pilot.read_bytes()) == solver["manifest_sha256"]
    assert digest(pinned.read_bytes()) == solver["pinned_solver_manifest_sha256"]
    original, = load_tasks(pilot, task_ids=[spec["instance_id"]])
    task, = load_tasks(pinned)
    assert task.image == original.image.removesuffix(":latest") + "@" + image["manifest_digest"]
    assert task.image == solver["pinned_image"]
    assert vars(task) == {**vars(original), "image": task.image}


def grade(gold):
    return {"resolved": gold, "patch_exists": True, "patch_is_None": False,
            "patch_successfully_applied": True, "infra_failure": False, "tests_status": {
                "FAIL_TO_PASS": {"success": [FIX] if gold else [], "failure": [] if gold else [FIX]},
                "PASS_TO_PASS": {"success": [KEEP], "failure": []}}}


@pytest.fixture
def controls(tmp_path):
    patches = (b"gold patch\n", b"no repair patch\n")
    targets = json.dumps({"FAIL_TO_PASS": [FIX], "PASS_TO_PASS": [KEEP]},
                         sort_keys=True, separators=(",", ":")).encode()
    spec = {"instance_id": TASK, "test_targets_sha256": digest(targets),
            "eval_script_sha256": digest(b"synthetic evaluator\n"),
            "gold_patch_sha256": digest(patches[0]), "negative_patch_sha256": digest(patches[1])}
    reports = []
    for label, patch in zip(("gold", "negative"), patches):
        directory = tmp_path / label / TASK
        directory.mkdir(parents=True)
        report = directory / "report.json"
        report.write_text(json.dumps({TASK: grade(label == "gold")}), encoding="utf-8")
        report.with_name("eval.sh").write_bytes(b"synthetic evaluator\n")
        report.with_name("patch.diff").write_bytes(patch)
        reports.append(report)
    return spec, *reports


@pytest.mark.parametrize("mutation", ["regression_only", "script_drift", "wrong_shared_script", "wrong_shared_targets",
                                      "wrong_patch", "empty_patch", "wrong_instance", "gold_false_positive"])
def test_control_pair_rejects_uncalibrated_environment(controls, mutation):
    spec, gold, negative = controls
    if mutation in {"regression_only", "gold_false_positive"}:
        path = negative if mutation == "regression_only" else gold
        row = grade(True)
        row["resolved"] = path == gold
        bucket = "PASS_TO_PASS" if path == negative else "FAIL_TO_PASS"
        case = row["tests_status"][bucket]
        case["failure"], case["success"] = case["success"], []
        path.write_text(json.dumps({TASK: row}), encoding="utf-8")
    elif mutation == "script_drift":
        negative.with_name("eval.sh").write_bytes(b"different evaluator\n")
    elif mutation == "wrong_shared_script":
        for path in (gold, negative):
            path.with_name("eval.sh").write_bytes(b"same unintended evaluator\n")
    elif mutation == "wrong_shared_targets":
        for path in (gold, negative):
            row = grade(path == gold)
            row["tests_status"]["PASS_TO_PASS"]["success"] = ["different-regression-check"]
            path.write_text(json.dumps({TASK: row}), encoding="utf-8")
    elif mutation == "wrong_instance":
        negative.write_text(json.dumps({"owner__repo-4567": grade(False)}), encoding="utf-8")
    elif mutation == "wrong_patch":
        gold.with_name("patch.diff").write_bytes(b"unrelated patch\n")
    elif mutation == "empty_patch":
        negative.with_name("patch.diff").write_bytes(b"")
        spec["negative_patch_sha256"] = digest(b"")
    with pytest.raises(ValueError):
        check_controls(spec, gold, negative)


def test_control_cli_writes_only_verified_evidence(controls, tmp_path):
    spec, gold, negative = controls
    expected = check_controls(spec, gold, negative)
    assert expected["gold"]["report_sha256"] == digest(gold.read_bytes())
    assert expected["negative"]["report_sha256"] == digest(negative.read_bytes())
    spec_path, output = tmp_path / "spec.json", tmp_path / "receipt.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    args = ["swe-check-controls", "--spec", str(spec_path), "--gold-report", str(gold),
            "--negative-report", str(negative), "--output", str(output)]
    assert main(args) == 0
    assert json.loads(output.read_bytes()) == {**expected, "control_spec_sha256": digest(spec_path.read_bytes())}
    assert all(name not in output.read_text(encoding="utf-8") for name in (FIX, KEEP))

    output.write_bytes(b"original evidence")
    assert main(args) != 0
    assert output.read_bytes() == b"original evidence"
    negative.with_name("eval.sh").unlink()
    args[-1] = str(tmp_path / "invalid.json")
    assert main(args) != 0
    assert not (tmp_path / "invalid.json").exists()


def test_gold_cli_uses_pair_criteria_before_negative_exists(controls, tmp_path, capsys):
    spec, gold, negative = controls
    paired = check_controls(spec, gold, negative)
    negative.unlink()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    args = ["swe-check-gold", "--spec", str(spec_path), "--gold-report", str(gold)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out) == {k: v for k, v in paired.items() if k != "negative"}

    # A resolved flag alone can hide a failed repair target.
    row = grade(True)
    row["tests_status"]["FAIL_TO_PASS"] = {"failure": [FIX], "success": []}
    gold.write_text(json.dumps({TASK: row}), encoding="utf-8")
    assert main(args) != 0
    assert "FAIL_TO_PASS" in capsys.readouterr().err
    gold.unlink()
    assert main(args) != 0
    assert "Missing gold report.json" in capsys.readouterr().err
