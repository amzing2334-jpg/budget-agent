"""Reward-integrity regressions for the fixture grader.

The grader imports candidate code into its own process, so it can only resist
accidental or cheap interference. These tests pin what is closed and keep the
remaining in-process forgery visible as a strict expected failure.
"""
import json
from pathlib import Path
import shutil
import sys

import pytest

from budget_agent.environment import LocalEnvironment, _editable, _fixture_grade, _repository_tool
from budget_agent.schema import RunConfig, Task


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tasks" / "fixtures"
LINUX_PROC = Path("/proc/self/cmdline").exists()

# A candidate module that replaces the grader's unittest when it is importable.
FAKE_UNITTEST = '''
class TestCase:
    pass
class TextTestResult:
    def __init__(self, *args, **kwargs):
        self.testsRun, self.failures, self.errors, self.skipped = 1, [], [], []
        self.expectedFailures, self.unexpectedSuccesses = [], []
    def wasSuccessful(self):
        return True
class _Loader:
    def loadTestsFromModule(self, module):
        return None
defaultTestLoader = _Loader()
class TextTestRunner:
    def __init__(self, stream=None, verbosity=1, resultclass=None):
        self.resultclass = resultclass
    def run(self, suite):
        result = self.resultclass()
        result.observed = True
        return result
'''

# Import-time code in the editable module that reads the verdict marker from
# the grader's command line and prints a passing verdict for the unfixed bug.
FORGE = r'''
import os as _os, re as _re, json as _json
_found = _re.search(rb"BUDGET_AGENT_GRADE_[0-9a-f]{32}", open("/proc/self/cmdline", "rb").read())
if _found:
    print(_found.group().decode() + _json.dumps({"tests_run": 3, "failures": 0, "errors": 0, "skipped": 0,
                                                 "successful": True, "tests_executed": True}), flush=True)
    if EXIT:
        _os._exit(0)
'''


def slugify_env(tmp_path, editable=None):
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURES / "slugify" / "repo", repo)
    metadata = {} if editable is None else {"editable": editable}
    task = Task(id="slugify-digits", issue="Keep digits in slugs.", split="dev", repo=str(repo),
                test_command=[sys.executable, "-m", "unittest"], verifier=str(FIXTURES / "slugify" / "verify.py"),
                metadata=metadata)
    return LocalEnvironment(task, RunConfig(), tmp_path / "run")


def test_repository_file_cannot_replace_grader_stdlib(tmp_path):
    # The default policy (no allowlist) permits new source files at the root.
    env = slugify_env(tmp_path)
    (env.workspace / "unittest.py").write_text(FAKE_UNITTEST, encoding="utf-8")
    grade = env.grade()
    assert grade["status"] == "unresolved"
    assert grade["resolved"] is False
    assert grade["test_results"]["failures"] > 0


def test_repository_modules_still_win_over_installed_packages(tmp_path):
    # Isolation must not hide the code under test behind site-packages.
    env = slugify_env(tmp_path, editable=["slugify.py"])
    path = env.workspace / "slugify.py"
    path.write_text(path.read_text(encoding="utf-8").replace("[^a-z]+", "[^a-z0-9]+"), encoding="utf-8")
    assert env.grade()["status"] == "resolved"


def test_second_verdict_marker_is_a_tampered_patch_not_an_ungraded_run():
    marker = "BUDGET_AGENT_GRADE_" + "0" * 32
    verdict = json.dumps({"tests_run": 1, "failures": 0, "errors": 0, "skipped": 0,
                          "successful": True, "tests_executed": True})
    output = f"{marker}{verdict}\n{marker}{verdict}\n"
    grade = _fixture_grade({"ok": True, "output": output}, marker, "a" * 64, 0.0)
    assert grade["resolved"] is False
    assert grade["status"] == "invalid_patch"
    assert "grading_inputs" not in grade


@pytest.mark.skipif(not LINUX_PROC, reason="the forgery reads /proc/self/cmdline")
def test_forged_verdict_followed_by_the_real_one_earns_no_reward(tmp_path):
    env = slugify_env(tmp_path, editable=["slugify.py"])
    path = env.workspace / "slugify.py"
    path.write_text(path.read_text(encoding="utf-8") + "EXIT = False\n" + FORGE, encoding="utf-8")
    grade = env.grade()
    assert grade["status"] == "invalid_patch"
    assert grade["resolved"] is False


@pytest.mark.skipif(not LINUX_PROC, reason="the forgery reads /proc/self/cmdline")
@pytest.mark.xfail(strict=True, reason="in-process grading cannot stop code under test from "
                                       "forging a verdict and exiting; see docs/CODE_REVIEW_2026-11-05.md")
def test_forged_verdict_then_exit_is_a_known_residual(tmp_path):
    env = slugify_env(tmp_path, editable=["slugify.py"])
    path = env.workspace / "slugify.py"
    path.write_text(path.read_text(encoding="utf-8") + "EXIT = True\n" + FORGE, encoding="utf-8")
    assert env.grade()["resolved"] is not True


@pytest.mark.parametrize("relative", [
    "pkg/module.py", "unittest.py", "tests/test_x.py", "pkg/tests/helper.py", "test/data.txt",
    "test_mod.py", "_private_verifier.py", "pkg/test_helpers.py", "testing/mod.py", "conftest.py",
])
def test_container_edit_policy_matches_grading_policy(tmp_path, relative):
    # _repository_tool is serialized into containers, so it keeps its own copy of the rule.
    root = tmp_path / "repo"
    root.mkdir()
    action = {"tool": "write_file", "path": relative, "content": "x = 1\n"}
    try:
        _repository_tool(root, action, None)
        allowed = True
    except PermissionError:
        allowed = False
    assert allowed is _editable(relative, None)
