"""Task manifests are a trust boundary: only known fields, safe unique IDs, valid splits, paths resolved."""
import json

import pytest

from budget_agent.schema import Task
from budget_agent.tasks import load_tasks, smoke_actions


def manifest(tmp_path, rows, name="tasks.jsonl", as_array=False):
    path = tmp_path / name
    path.write_text(json.dumps(rows) if as_array else "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def row(task_id="a", **extra):
    return {"id": task_id, "issue": "fix it", "split": "train", "kind": "local", **extra}


def test_jsonl_and_json_arrays_load_with_paths_resolved_against_the_manifest(tmp_path):
    (tmp_path / "sub").mkdir()
    rows = [row("a", repo="repos/a", verifier="verifiers/a.py"), row("b", split="eval")]
    for as_array in (False, True):
        path = manifest(tmp_path / "sub", rows, as_array=as_array)
        tasks = load_tasks(path, split="all")
        assert [t.id for t in tasks] == ["a", "b"]
        assert tasks[0].repo == str((tmp_path / "sub" / "repos/a").resolve())
        assert tasks[0].verifier == str((tmp_path / "sub" / "verifiers/a.py").resolve()) and tasks[1].repo == ""
    assert [t.id for t in load_tasks(path, split="eval")] == ["b"]
    assert [t.id for t in load_tasks(path, split="all", task_ids=["b"])] == ["b"]


@pytest.mark.parametrize(("rows", "message"), [
    ([row(grading_secret="x")], "unknown fields"),
    ([row("../escape")], "Unsafe or duplicate"),
    ([row("a/b")], "Unsafe or duplicate"),
    ([row("a"), row("a")], "Unsafe or duplicate"),
    ([row(split="validation")], "Invalid split/kind"),
    ([row(kind="docker")], "Invalid split/kind"),
    ([row(issue="   ")], "must not be empty"),
])
def test_unsafe_or_malformed_rows_are_rejected(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        load_tasks(manifest(tmp_path, rows), split="all")


def test_selections_that_match_nothing_are_errors_not_empty_runs(tmp_path):
    path = manifest(tmp_path, [row("a")])
    with pytest.raises(ValueError, match="No tasks selected"):
        load_tasks(path, split="eval")
    with pytest.raises(ValueError, match="absent or excluded"):
        load_tasks(path, split="all", task_ids=["a", "missing"])


def test_scripted_repairs_only_cover_the_bundled_fixtures_as_written(tmp_path):
    with pytest.raises(ValueError, match="only supports"):
        smoke_actions(Task(id="other", issue="x", split="dev", repo=str(tmp_path)))
    (tmp_path / "slugify.py").write_text("def slugify(s):\n    return s\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no longer matches"):
        smoke_actions(Task(id="slugify-digits", issue="x", split="dev", repo=str(tmp_path)))
