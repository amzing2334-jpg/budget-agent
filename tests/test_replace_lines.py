"""Line-range edits: numbers from read_file instead of copied anchors."""
import pytest

from budget_agent.models import parse_action
from test_edit_file import edit_environment  # noqa: F401 (local and inline-container fixture)


def read(env, path="source.py"):
    return env.execute({"tool": "read_file", "path": path})["output"]


def test_replaces_an_inclusive_line_range_in_both_environments(edit_environment):  # noqa: F811
    env, root = edit_environment
    result = env.execute({"tool": "replace_lines", "path": "source.py", "start_line": 2, "end_line": 3,
                          "new": "VALUE = 2\nREPEAT = 'yy'\nEXTRA = True"})
    assert result["ok"] and "moved by +1" in result["output"]
    assert (root / "source.py").read_text(encoding="utf-8") == (
        "HEADER = '中文'\nVALUE = 2\nREPEAT = 'yy'\nEXTRA = True\nTAIL = 3\n")


def test_keeps_crlf_line_endings(edit_environment):  # noqa: F811
    env, root = edit_environment
    (root / "source.py").write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")
    assert env.execute({"tool": "replace_lines", "path": "source.py", "start_line": 2, "end_line": 2,
                        "new": "b = 20"})["ok"]
    assert (root / "source.py").read_bytes() == b"a = 1\r\nb = 20\r\nc = 3\r\n"


@pytest.mark.parametrize("start,end", [(0, 1), (3, 2), (2, 99)])
def test_rejects_out_of_range_lines_without_changing_the_file(edit_environment, start, end):  # noqa: F811
    env, root = edit_environment
    before = (root / "source.py").read_bytes()
    result = env.execute({"tool": "replace_lines", "path": "source.py", "start_line": start, "end_line": end, "new": "x"})
    assert not result["ok"] and "has 4 lines" in result["output"]
    assert (root / "source.py").read_bytes() == before


def test_respects_the_test_boundary_and_missing_files(edit_environment):  # noqa: F811
    env, _ = edit_environment
    protected = env.execute({"tool": "replace_lines", "path": "tests/test_public.py", "start_line": 1, "end_line": 1, "new": "x"})
    assert not protected["ok"] and "protected" in protected["output"]
    missing = env.execute({"tool": "replace_lines", "path": "nope.py", "start_line": 1, "end_line": 1, "new": "x"})
    assert not missing["ok"] and "does not exist" in missing["output"]


def test_parser_requires_integer_line_numbers():
    assert parse_action('{"tool":"replace_lines","path":"a.py","start_line":1,"end_line":2,"new":"x"}')["tool"] == "replace_lines"
    with pytest.raises(ValueError, match="integer 'end_line'"):
        parse_action('{"tool":"replace_lines","path":"a.py","start_line":1,"new":"x"}')


def test_failed_edit_suggests_a_concrete_line_range_call(edit_environment):  # noqa: F811
    env, _ = edit_environment
    missing = env.execute({"tool": "edit_file", "path": "source.py", "old": "VALUE = 7", "new": "VALUE = 8"})
    assert '"tool":"replace_lines","path":"source.py","start_line":2,"end_line":2' in missing["output"]
    ambiguous = env.execute({"tool": "edit_file", "path": "source.py", "old": "xx", "new": "zz"})
    assert '"start_line":3' in ambiguous["output"]
