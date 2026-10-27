"""Exercise the real local and inline-container file tool code on CPU only."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from budget_agent.environment import DockerEnvironment, LocalEnvironment
from budget_agent.models import parse_action
from budget_agent.schema import RunConfig, Task


@pytest.fixture(params=["local", "docker"])
def edit_environment(tmp_path, request):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "source.py").write_text("HEADER = '中文'\nVALUE = 1\nREPEAT = 'xx xx'\nTAIL = 3\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_public.py").write_text("assert True\n", encoding="utf-8")
    task = Task(id="edit-test", issue="edit", split="dev", repo=str(root))
    if request.param == "local":
        env = LocalEnvironment(task, RunConfig(), tmp_path / "run")
        return env, env.workspace
    env = DockerEnvironment.__new__(DockerEnvironment)
    env.local = None
    env.task = task
    env.config = RunConfig()

    def execute_inline(command):
        assert command[:2] == ["python", "-c"]
        result = subprocess.run([sys.executable, *command[1:]], cwd=root, capture_output=True, text=True, encoding="utf-8")
        return {"ok": result.returncode == 0, "output": result.stdout + result.stderr,
                "returncode": result.returncode}

    env._exec = execute_inline
    return env, root


def test_edit_file_preserves_unmatched_content(edit_environment):
    env, root = edit_environment
    path = root / "source.py"
    before = path.read_text(encoding="utf-8")
    action = {"tool": "edit_file", "path": "source.py", "old": "VALUE = 1", "new": "VALUE = '中文'\nEXTRA = 2"}
    assert env.execute(parse_action(json.dumps(action)))["ok"]
    assert path.read_text(encoding="utf-8") == before.replace(action["old"], action["new"])


@pytest.mark.parametrize("old,error", [("", "non-empty"), ("missing", "found 0"), ("xx", "found 2")])
def test_edit_file_rejects_invalid_match_without_mutating(edit_environment, old, error):
    env, root = edit_environment
    path = root / "source.py"
    before = path.read_bytes()
    result = env.execute({"tool": "edit_file", "path": "source.py", "old": old, "new": "changed"})
    assert not result["ok"] and error in result["output"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("path", ["../outside.py", ".git/config", "tests/test_public.py"])
def test_edit_file_respects_path_and_test_boundary(edit_environment, path):
    env, root = edit_environment
    before = (root / "tests" / "test_public.py").read_bytes()
    result = env.execute({"tool": "edit_file", "path": path, "old": "True", "new": "False"})
    assert not result["ok"]
    assert (root / "tests" / "test_public.py").read_bytes() == before


@pytest.mark.parametrize("field", ["path", "old", "new"])
def test_edit_action_requires_strings(field):
    action = {"tool": "edit_file", "path": "source.py", "old": "before", "new": "after"}
    action[field] = None
    with pytest.raises(ValueError, match=field):
        parse_action(json.dumps(action))


def _page(result):
    assert result["ok"], result["output"]
    heading, _, body = result["output"].partition("\n")
    return json.loads(heading.partition(" ")[2]), body.splitlines()


def test_directory_pages_are_scoped_bounded_and_complete(edit_environment):
    env, root = edit_environment
    package = root / "package"
    package.mkdir()
    names = [f"{index:03d}_" + "a" * 80 + ".py" for index in range(190)]
    for name in names:
        (package / name).write_text("payload", encoding="utf-8")
    for directory in (".git", "__pycache__", ".pytest_cache"):
        (package / directory).mkdir()
        (package / directory / "private").write_text("excluded", encoding="utf-8")
    collected, offset = [], 0
    while True:
        result = env.execute({"tool": "list_files", "path": "package", "offset": offset, "limit": 200})
        metadata, rows = _page(result)
        assert len(result["output"]) < 17_000
        assert metadata["returned"] == len(rows)
        collected.extend(rows)
        if not metadata["has_more"]:
            assert metadata["next_offset"] is None
            break
        assert metadata["next_offset"] > offset
        offset = metadata["next_offset"]
    assert collected == ["package/" + name for name in names]
    _, rows = _page(env.execute({"tool": "list_files", "path": "source.py"}))
    assert rows == ["source.py"]


def test_search_pages_preserve_match_locations_and_long_line_snippets(edit_environment):
    env, root = edit_environment
    source = root / "matches.py"
    source.write_text("".join(f"needle {index}\n" for index in range(75)) + "x" * 90_000 + "needle last\n", encoding="utf-8")
    first, rows = _page(env.execute({"tool": "search", "path": "matches.py", "query": "needle"}))
    assert first["returned"] == 50 and first["next_offset"] == 50
    assert rows[0] == "matches.py:1: needle 0"
    final, rows = _page(env.execute({"tool": "search", "path": "matches.py", "query": "needle", "offset": first["next_offset"]}))
    assert final["returned"] == 26 and not final["has_more"]
    assert rows[0] == "matches.py:51: needle 50"
    assert rows[-1].startswith("matches.py:76: [line prefix omitted]")
    assert "needle last" in rows[-1]
    assert len(rows[-1]) < 600
    assert all(row.startswith("matches.py:") for row in rows)


def test_read_file_reports_total_and_next_range(edit_environment):
    env, root = edit_environment
    (root / "module.py").write_text("".join(f"value_{index} = {index}\n" for index in range(1, 318)), encoding="utf-8")
    first, rows = _page(env.execute({"tool": "read_file", "path": "module.py"}))
    assert first["total_lines"] == 317 and first["end_line"] == 100
    assert first["next_start_line"] == 101 and len(rows) == 100
    assert rows[0] == "1: value_1 = 1" and rows[-1] == "100: value_100 = 100"
    last, rows = _page(env.execute({"tool": "read_file", "path": "module.py", "start_line": 301}))
    assert last["total_lines"] == 317 and last["end_line"] == 317
    assert last["next_start_line"] is None and len(rows) == 17


def test_read_file_char_budget_keeps_contiguous_lines_and_marks_long_lines(edit_environment):
    env, root = edit_environment
    (root / "large.py").write_text(("x" * 20_000 + "\n") * 30, encoding="utf-8")
    result = env.execute({"tool": "read_file", "path": "large.py"})
    metadata, rows = _page(result)
    assert len(result["output"]) < 17_000
    assert metadata["total_lines"] == 30 and metadata["truncated_line_content"]
    assert 0 < len(rows) < 30
    assert [int(row.partition(":")[0]) for row in rows] == list(range(1, len(rows) + 1))
    assert metadata["next_start_line"] == len(rows) + 1
    assert all("[line truncated after 1000 characters]" in row for row in rows)
    next_page, following = _page(env.execute({"tool": "read_file", "path": "large.py", "start_line": metadata["next_start_line"]}))
    assert next_page["start_line"] == len(rows) + 1
    assert following[0].startswith(str(len(rows) + 1) + ":")


def test_edit_feedback_points_to_ambiguity_without_guessing(edit_environment):
    env, root = edit_environment
    before = (root / "source.py").read_bytes()
    result = env.execute({"tool": "edit_file", "path": "source.py", "old": "xx", "new": "fixed"})
    assert not result["ok"]
    assert "found 2" in result["output"] and "line 3, column 11" in result["output"]
    assert "line 3, column 14" in result["output"] and "surrounding text" in result["output"]
    stale = env.execute({"tool": "edit_file", "path": "source.py", "old": "VALUE = 1\nOUTDATED = 2", "new": "fixed"})
    assert not stale["ok"] and "found 0" in stale["output"]
    assert "First nonempty old line occurs: line 2, column 1" in stale["output"]
    assert (root / "source.py").read_bytes() == before


@pytest.mark.parametrize("path", [".GIT/config", ".git./config", "C:/secret.txt", "../secret"])
def test_browsing_paths_respect_same_boundary(edit_environment, path):
    env, _ = edit_environment
    result = env.execute({"tool": "list_files", "path": path})
    assert not result["ok"] and "path must remain inside the repository" in result["output"]
