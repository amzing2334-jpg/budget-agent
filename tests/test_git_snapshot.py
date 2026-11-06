"""Run the in-container git snapshot code against a real repository on the host.

DockerEnvironment executes _GIT_SNAPSHOT with `docker exec`. Here `_exec` runs
the same argv locally, so git semantics are exercised without a Docker daemon.
"""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from budget_agent.environment import DockerEnvironment, _process


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def env(tmp_path):
    repo = tmp_path / "testbed"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@invalid", "commit", "-qm", "base")
    env = DockerEnvironment.__new__(DockerEnvironment)
    env.local = None
    env.base_commit = git(repo, "rev-parse", "HEAD").strip()
    env._exec = lambda command: _process([sys.executable, *command[1:]], repo, 30)
    env.repo = repo
    return env


def test_diff_tool_does_not_change_the_fingerprint_of_an_unchanged_tree(env):
    repo = env.repo
    (repo / "mod.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "reproduce_issue.py").write_text("print('repro')\n", encoding="utf-8")
    before = env.revision()
    patch = env.patch()
    assert env.revision() == before
    assert "+x = 2" in patch and "new file mode" in patch and "reproduce_issue.py" in patch
    # The agent's index is untouched: the helper script is still plainly untracked.
    assert git(repo, "diff", "--cached", "--name-only") == ""
    assert "reproduce_issue.py" in git(repo, "ls-files", "--others", "--exclude-standard")


def test_fingerprint_follows_content_not_git_bookkeeping(env):
    repo = env.repo
    initial = env.revision()
    (repo / "mod.py").write_text("x = 3\n", encoding="utf-8")
    edited = env.revision()
    assert edited != initial
    git(repo, "add", "mod.py")
    assert env.revision() == edited
    git(repo, "-c", "user.name=a", "-c", "user.email=a@invalid", "commit", "-qm", "agent commit")
    assert env.revision() == edited
    assert "+x = 3" in env.patch()  # still relative to the image's base commit
    (repo / "debug.log").write_text("ignored\n", encoding="utf-8")
    assert env.revision() == edited
    (repo / "mod.py").write_text("x = 1\n", encoding="utf-8")
    assert env.revision() == initial


def test_git_warnings_never_enter_the_exported_patch(env):
    repo = env.repo
    nested = repo / "vendored"
    nested.mkdir()
    git(nested, "init", "-q")
    (nested / "a.py").write_text("a = 1\n", encoding="utf-8")
    git(nested, "add", ".")
    git(nested, "-c", "user.name=t", "-c", "user.email=t@invalid", "commit", "-qm", "nested")
    patch = env.patch()  # `git add` warns about the embedded repository on stderr
    assert "warning" not in patch.lower()
    assert patch.startswith("diff --git a/vendored b/vendored")
    assert len(env.revision()) == 16
