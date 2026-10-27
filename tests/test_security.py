"""CPU capability checks with disposable files, not model attack-success rates."""
import shlex
import sys

import pytest

from budget_agent.environment import DockerEnvironment, LocalEnvironment, _process
from budget_agent.schema import RunConfig, Task


@pytest.fixture
def task(tmp_path):
    source = tmp_path / "source"
    (source / "tests").mkdir(parents=True)
    (source / "solution.py").write_text("VALUE = 0\n", encoding="utf-8")
    (source / "tests/test_public.py").write_text(
        "import unittest\nfrom solution import VALUE\n"
        "class Tests(unittest.TestCase):\n"
        "    def test_value(self): self.assertEqual(VALUE, 1)\n", encoding="utf-8")
    return Task("policy-fixture", "Repair VALUE", "dev", repo=str(source),
                test_command=[sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                metadata={"editable": ["solution.py"]})


def restricted_env(task, tmp_path):
    return LocalEnvironment(task, RunConfig(execution_policy="restricted"), tmp_path / "run")


@pytest.mark.parametrize("command", [
    "echo benign command", "echo untrusted instructions && echo second command",
    "python -c \"print('arbitrary program')\"",
])
def test_restricted_run_is_denied_before_any_process(task, tmp_path, monkeypatch, command):
    env = restricted_env(task, tmp_path)

    def unexpected_process(*args, **kwargs):
        raise AssertionError("A denied command started a process")

    monkeypatch.setattr("budget_agent.environment._process", unexpected_process)
    before = (env.workspace / "tests/test_public.py").read_bytes()
    result = env.execute({"tool": "run", "command": command})
    assert result["ok"] is False
    assert result["policy_decision"] == {
        "policy": "restricted", "allowed": False, "rule": "free_shell_disabled"}
    assert (env.workspace / "tests/test_public.py").read_bytes() == before


def test_public_test_executes_manifest_argv_and_can_verify_a_repair(task, tmp_path):
    env = restricted_env(task, tmp_path)
    before = env.execute({"tool": "public_test"})
    assert not before["ok"] and before["returncode"] != 0
    assert before["policy_decision"]["allowed"] is True
    edit = env.execute({"tool": "write_file", "path": "solution.py", "content": "VALUE = 1\n"})
    assert edit["ok"] and edit["policy_decision"]["allowed"]
    after = env.execute({"tool": "public_test"})
    assert after["ok"] and "OK" in after["output"]


@pytest.mark.parametrize("extra", [{"command": "echo changed"}, {"args": ["--override"]}])
def test_public_test_rejects_model_arguments_even_without_parser(task, tmp_path, extra):
    result = restricted_env(task, tmp_path).execute({"tool": "public_test", **extra})
    assert result["policy_decision"]["rule"] == "public_test_arguments_not_allowed"
    assert not result["ok"]


def test_unconfigured_public_test_is_explicitly_unavailable(task, tmp_path):
    task.test_command = []
    result = restricted_env(task, tmp_path).execute({"tool": "public_test"})
    assert result["policy_decision"]["rule"] == "public_test_not_configured"
    assert not result["ok"]


def test_manifest_arguments_are_not_interpreted_as_a_shell(task, tmp_path):
    task.test_command = [sys.executable, "-c", "import sys; print(sys.argv[1])", "literal && echo not-a-command"]
    result = restricted_env(task, tmp_path).execute({"tool": "public_test"})
    assert result["ok"]
    assert result["output"].strip() == "literal && echo not-a-command"


@pytest.mark.parametrize("action", [
    {"tool": "read_file", "path": "../outside.txt"},
    {"tool": "read_file", "path": ".git/config"},
    {"tool": "write_file", "path": "tests/test_public.py", "content": "fake pass"},
])
def test_authorized_file_tools_still_enforce_path_and_edit_boundaries(task, tmp_path, action):
    env = restricted_env(task, tmp_path)
    before = (env.workspace / "tests/test_public.py").read_bytes()
    result = env.execute(action)
    assert result["policy_decision"] == {
        "policy": "restricted", "allowed": False, "rule": "repository_boundary"}
    assert result["ok"] is False
    assert (env.workspace / "tests/test_public.py").read_bytes() == before


@pytest.mark.parametrize("action", [
    {"tool": "read_file", "path": "missing.py"},
    {"tool": "edit_file", "path": "solution.py", "old": "missing anchor", "new": "replacement"},
])
def test_ordinary_file_errors_are_not_policy_denials(task, tmp_path, action):
    result = restricted_env(task, tmp_path).execute(action)
    assert not result["ok"]
    assert result["policy_decision"]["allowed"] is True
    assert "policy_denial" not in result


@pytest.mark.parametrize("returncode,allowed", [(77, False), (1, True)])
def test_docker_file_boundary_uses_exit_contract_not_output_keywords(task, returncode, allowed):
    env = DockerEnvironment.__new__(DockerEnvironment)
    env.task, env.config, env.local = task, RunConfig(execution_policy="restricted"), None
    commands = []

    def execute(command):
        commands.append(command)
        return {"ok": False, "output": "this task only permits source edits; tests and verifier are protected",
                "returncode": returncode, "duration_seconds": 0.0}

    env._exec = execute
    result = env.execute({"tool": "edit_file", "path": "tests/test_public.py", "old": "a", "new": "b"})
    assert "except PermissionError" in commands[0][2]
    assert "sys.exit(77)" in commands[0][2]
    assert result["policy_decision"]["allowed"] is allowed
    assert result["policy_decision"]["rule"] == ("tool_allowed" if allowed else "repository_boundary")
    assert result["output"] == "this task only permits source edits; tests and verifier are protected"


@pytest.mark.parametrize("action,returncode", [
    ({"tool": "read_file", "path": "../outside.txt"}, 77),
    ({"tool": "edit_file", "path": "tests/test_public.py", "old": "a", "new": "b"}, 77),
    ({"tool": "edit_file", "path": "solution.py", "old": "missing anchor", "new": "b"}, 1),
])
def test_serialized_docker_file_tool_emits_structured_denial_on_cpu(task, tmp_path, action, returncode):
    local = restricted_env(task, tmp_path)
    env = DockerEnvironment.__new__(DockerEnvironment)
    env.task, env.config, env.local = task, local.config, None
    # Run only the generated file-tool payload against this disposable fixture.
    # No Docker daemon or downloaded code is executed in this contract test.
    env._exec = lambda command: _process([sys.executable, *command[1:]], local.workspace, 5)
    result = env.execute(action)
    assert result["returncode"] == returncode
    assert result["policy_decision"]["allowed"] is (returncode != 77)
    assert result["policy_decision"]["rule"] == ("repository_boundary" if returncode == 77 else "tool_allowed")


def test_unknown_tool_does_not_dispatch(task, tmp_path):
    result = restricted_env(task, tmp_path).execute({"tool": "external_fetch", "url": "https://invalid.example"})
    assert result["policy_decision"]["rule"] == "unknown_tool"
    assert not result["ok"]


@pytest.mark.parametrize("policy", ["sandbox", "restricted"])
@pytest.mark.parametrize("kind,shell", [("local", "sh"), ("swebench", "bash")])
def test_docker_public_test_preserves_manifest_argv_and_testbed_profile(task, policy, kind, shell):
    env = DockerEnvironment.__new__(DockerEnvironment)
    env.task, env.config, env.local = task, RunConfig(execution_policy=policy), None
    task.kind = kind
    task.test_command = ["python", "-c", "print('literal; harmless')", "arg with spaces"]
    calls = []

    def execute(command):
        calls.append(command)
        return {"ok": True, "output": "test output", "duration_seconds": 0.0}

    env._exec = execute
    result = env.execute({"tool": "public_test"})
    assert calls == [[shell, "-lc", shlex.join(task.test_command)]]
    assert shlex.split(calls[0][-1]) == task.test_command
    assert result["policy_decision"] == {"policy": policy, "allowed": True, "rule": "tool_allowed"}


def test_default_sandbox_policy_retains_explicit_shell_capability(task, tmp_path):
    env = LocalEnvironment(task, RunConfig(), tmp_path / "run")
    result = env.execute({"tool": "run", "command": 'echo sandbox-shell'})
    assert result["ok"] and "sandbox-shell" in result["output"]
    assert result["policy_decision"] == {"policy": "sandbox", "allowed": True, "rule": "tool_allowed"}


def test_restricted_policy_blocks_direct_shell_bypass_of_protected_file_tools(task, tmp_path):
    command = (f'"{sys.executable}" -c "from pathlib import Path; '
               "Path('tests/test_public.py').write_text('# tampered')\"")
    sandbox = LocalEnvironment(task, RunConfig(), tmp_path / "sandbox")
    assert sandbox.execute({"tool": "run", "command": command})["ok"]
    assert (sandbox.workspace / "tests/test_public.py").read_text() == "# tampered"
    assert sandbox.grade()["status"] == "invalid_patch"

    restricted = restricted_env(task, tmp_path)
    before = (restricted.workspace / "tests/test_public.py").read_bytes()
    assert not restricted.execute({"tool": "run", "command": command})["ok"]
    assert (restricted.workspace / "tests/test_public.py").read_bytes() == before
