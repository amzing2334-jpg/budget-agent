"""The patch review compares each passing patch with the source before the bug was injected."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

SPEC = importlib.util.spec_from_file_location("review_patches", Path(__file__).parents[1] / "scripts/review_patches.py")
assert SPEC is not None and SPEC.loader is not None
review_patches = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_patches)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git applies the patches")

TASK = "owner__pkg.abc123.func_basic__x1"
ORIGINAL = "def keep(value):\n    return value\n\n\ndef first(items):\n    return items[0]\n"
BUGGY = "def keep(value):\n    return value\n\n\ndef first(items):\n    return items[-1]\n"


def patch(old, new):
    return (f"--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -5,2 +5,2 @@\n def first(items):\n"
            f"-    return {old}\n+    return {new}\n")


def make_pool(tmp_path, patches):
    original = tmp_path / "original" / "owner__pkg.abc123" / "pkg"
    original.mkdir(parents=True)
    (original / "mod.py").write_text(ORIGINAL)
    buggy = tmp_path / "tasks" / TASK / "repo" / "pkg"
    buggy.mkdir(parents=True)
    (buggy / "mod.py").write_text(BUGGY)
    pool = tmp_path / "pool"
    rows = []
    for sample, text in enumerate(patches):
        run = pool / "runs" / f"s{sample:02d}" / "run"
        run.mkdir(parents=True)
        (run / "patch.diff").write_text(text)
        (run / "result.json").write_text("{}")
        rows.append({"task_id": TASK, "sample": sample, "resolved": True, "status": "finished", "steps": 3,
                     "ignored_changes": [], "patch_signals": [], "result_path": str(run / "result.json")})
    (pool / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return pool


def test_marks_exact_restores_and_prints_what_remains(tmp_path, capsys):
    pool = make_pool(tmp_path, [patch("items[-1]", "items[0]"), patch("items[-1]", "items[:1][0] if items else None")])
    review_patches.main([str(pool), "--original", str(tmp_path / "original"), "--tasks", str(tmp_path / "tasks")])
    out = capsys.readouterr().out
    assert out.count("[original check] restores original") == 1
    assert "differs from the original:\n" in out and "+    return items[:1][0] if items else None" in out
    assert out.rstrip().endswith("pool: 1 of 2 resolved patches restore the original source exactly")


def test_reports_a_patch_that_does_not_apply(tmp_path, capsys):
    pool = make_pool(tmp_path, [patch("items[1]", "items[0]")])
    review_patches.main([str(pool), "--original", str(tmp_path / "original"), "--tasks", str(tmp_path / "tasks")])
    assert "the patch does not apply" in capsys.readouterr().out


def test_original_and_tasks_go_together(tmp_path):
    with pytest.raises(SystemExit):
        review_patches.main([str(tmp_path), "--original", str(tmp_path)])
