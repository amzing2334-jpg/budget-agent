"""Task selection by earlier rollouts: exclusions, minimum successes, absolute paths."""
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("select_tasks", ROOT / "scripts/select_tasks.py")
assert SPEC is not None and SPEC.loader is not None
select_tasks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(select_tasks)


def ledger(path, rows):
    path.write_text("".join(json.dumps({"task_id": t, "sample": s, "resolved": r}) + "\n" for t, s, r in rows))
    return path


def test_candidates_drop_what_the_teacher_solved_and_get_absolute_paths(tmp_path, capsys):
    manifest = tmp_path / "swesmith" / "train.jsonl"
    manifest.parent.mkdir()
    manifest.write_text("".join(json.dumps({"id": t, "repo": f"repos/{t}", "verifier": f"verifiers/{t}.py",
                                            "test_timeout_seconds": 30}) + "\n" for t in ("a", "b", "c")))
    teacher = ledger(tmp_path / "teacher.jsonl", [("a", 0, False), ("a", 1, True), ("b", 0, False), ("b", 1, None)])
    assert select_tasks.main([str(manifest), "--not-solved-in", str(teacher)]) == 0
    out = capsys.readouterr()
    rows = [json.loads(line) for line in out.out.splitlines()]
    assert [r["id"] for r in rows] == ["b", "c"] and out.err.strip() == "2 tasks kept"
    assert rows[0]["repo"] == os.path.join(manifest.parent.resolve(), "repos/b") and rows[0]["test_timeout_seconds"] == 30


def test_selection_keeps_tasks_solved_often_enough_and_rewriting_absolute_paths_is_stable(tmp_path):
    manifest, absolute = tmp_path / "out" / "candidates.jsonl", tmp_path / "abs"
    manifest.parent.mkdir()
    manifest.write_text("".join(json.dumps({"id": t, "repo": str(absolute / "repos" / t), "verifier": str(absolute / f"{t}.py")})
                                + "\n" for t in ("b", "c", "d")))
    pool = ledger(tmp_path / "select.jsonl", [("b", 0, True), ("b", 1, False), ("c", 0, False), ("c", 1, None),
                                              ("d", 0, True), ("d", 1, True)])
    once = select_tasks.select(manifest, [], [pool], 1)
    assert [r["id"] for r in once] == ["b", "d"] and once[0]["repo"] == str(absolute / "repos" / "b")
    assert [r["id"] for r in select_tasks.select(manifest, [], [pool], 2)] == ["d"]
