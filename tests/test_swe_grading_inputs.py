import hashlib
import json

import pytest

from budget_agent.swebench import attach_grades
from test_swebench import write_official_report, write_run


def targets(success=("private-A",), failure=("private-B",), retained=("private-C",)):
    return {"FAIL_TO_PASS": {"success": list(success), "failure": list(failure)},
            "PASS_TO_PASS": {"success": list(retained), "failure": []}}


def attach(root, status, script=b"run-tests\n", **changes):
    run = write_run(root / "runs")
    grade = {"resolved": False, "patch_is_None": False, "patch_exists": True,
             "patch_successfully_applied": True, "infra_failure": False, "tests_status": status, **changes}
    reports = root / "reports"
    report_dir = write_official_report(reports, run, **grade)
    if script is not None:
        (report_dir / "eval.sh").write_bytes(script)
    attach_grades(run, reports)
    result = (run / "result.json").read_text(encoding="utf-8")
    assert "private-" not in result and "run-tests" not in result
    return json.loads(result)["evaluation"]


def test_grading_inputs_identify_script_bytes_and_target_groups_independently_of_verdict(tmp_path):
    original = attach(tmp_path / "before", targets())["grading_inputs"]
    passed = attach(tmp_path / "after", targets(("private-B", "private-A"), ()), resolved=True)
    assert passed["resolved"] and passed["grading_inputs"] == original
    assert original["kind"] == "swebench"
    assert original["eval_script_sha256"] == hashlib.sha256(b"run-tests\n").hexdigest()
    switched = attach(tmp_path / "class", targets(("private-C",), ("private-B",), ("private-A",)))
    assert switched["grading_inputs"]["test_targets_sha256"] != original["test_targets_sha256"]
    changed = attach(tmp_path / "script", targets(), script=b"run-tests\r\n")
    assert changed["grading_inputs"]["eval_script_sha256"] != original["eval_script_sha256"]
    assert changed["grading_inputs"]["test_targets_sha256"] == original["test_targets_sha256"]


@pytest.mark.parametrize("status,script,details", [
    (targets(), None, {}),
    (targets(), b"", {}),
    (None, b"run-tests", {}),
    ({"FAIL_TO_PASS": targets()["FAIL_TO_PASS"]}, b"run-tests", {}),
    ({**targets(), "PASS_TO_PASS": {"success": []}}, b"run-tests", {}),
    (targets((), (), ()), b"run-tests", {}),
    (targets(("private-A",), ("private-A",)), b"run-tests", {}),
    (targets(("private-C",)), b"run-tests", {}),
    (targets((False,)), b"run-tests", {}),
    (targets(), b"run-tests", {"patch_successfully_applied": False}),
    (targets(), b"run-tests", {"infra_failure": True}),
])
def test_incomplete_grading_evidence_retains_old_verdict_without_an_identity(tmp_path, status, script, details):
    evaluation = attach(tmp_path, status, script=script, **details)
    assert "grading_inputs" not in evaluation
    assert evaluation["resolved"] is (None if details.get("infra_failure") else False)
