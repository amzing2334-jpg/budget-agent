"""Hidden behaviour checks re-grade passing patches; a check must pass on the pre-bug source."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("regrade_hidden", ROOT / "scripts/regrade_hidden.py")
assert SPEC is not None and SPEC.loader is not None
regrade = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(regrade)
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git applies the patches")

TASK = "owner__lib.abc123.func_basic__x1"
ORIGINAL = "def first(items):\n    return items[0] if items else None\n"
BUGGY = "def first(items):\n    return items[-1] if items else None\n"


def patch(new):
    return ("--- a/lib.py\n+++ b/lib.py\n@@ -1,2 +1,2 @@\n def first(items):\n"
            "-    return items[-1] if items else None\n" f"+    {new}\n")


def setup(tmp_path, patches, check="from lib import first\nassert first([1, 2]) == 1 and first([]) is None\n"):
    (tmp_path / "original" / "owner__lib.abc123").mkdir(parents=True)
    (tmp_path / "original" / "owner__lib.abc123" / "lib.py").write_text(ORIGINAL)
    repo = tmp_path / "swesmith" / "tasks" / TASK / "repo"
    repo.mkdir(parents=True)
    (repo / "lib.py").write_text(BUGGY)
    (tmp_path / "swesmith" / "eval.jsonl").write_text(json.dumps({"id": TASK, "repo": f"tasks/{TASK}/repo"}) + "\n")
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / f"{TASK}.py").write_text(check)
    pool = tmp_path / "pool"
    rows = []
    for sample, text in enumerate(patches):
        run = pool / "runs" / f"s{sample}"
        run.mkdir(parents=True)
        (run / "patch.diff").write_text(text)
        rows.append({"task_id": TASK, "sample": sample, "resolved": True, "result_path": str(run / "result.json")})
    rows.append({"task_id": TASK, "sample": 99, "resolved": False, "result_path": "unused"})
    (pool / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return pool


def regrade_report(tmp_path, pool, capsys, *extra):
    regrade.main([str(pool), "--manifest", str(tmp_path / "swesmith" / "eval.jsonl"),
                  "--original", str(tmp_path / "original"), "--checks", str(tmp_path / "checks"), *extra])
    return json.loads(capsys.readouterr().out)


def test_patches_are_regraded_against_a_check_that_passes_on_the_original(tmp_path, capsys):
    pool = setup(tmp_path, [
        patch("return items[0] if items else None"),           # restores the original
        patch("return items[0]"),                              # passes a [1, 2] test, fails on []
        patch("while True: pass"),                             # hangs
        patch("return x").replace("items[-1]", "items[-2]"),   # does not apply
    ])
    report = regrade_report(tmp_path, pool, capsys, "--timeout", "3")
    assert report["checks"][TASK] == {"original": "pass", "buggy": "fail"}
    assert [a["result"] for a in report["attempts"]] == ["pass", "fail", "timeout", "patch_failed"]
    assert report["summary"][str(pool)] == {"checked": 4, "pass": 1, "fail": 1, "timeout": 1, "patch_failed": 1}


def test_a_check_that_fails_on_the_original_grades_nothing(tmp_path, capsys):
    pool = setup(tmp_path, [patch("return items[0] if items else None")], check="raise SystemExit(1)\n")
    report = regrade_report(tmp_path, pool, capsys)
    assert report["checks"][TASK]["original"] == "fail"
    assert report["attempts"] == [] and report["summary"][str(pool)]["checked"] == 0


def test_every_shipped_check_is_a_python_script_named_after_a_swesmith_task():
    checks = sorted((ROOT / "tasks" / "hidden_checks").glob("*.py"))
    assert len(checks) >= 10
    for check in checks:
        assert check.stem.count(".") == 2 and "__" in check.stem, check.name
        compile(check.read_text(), str(check), "exec")
