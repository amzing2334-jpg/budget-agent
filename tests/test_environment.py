import hashlib
import json
import os
from pathlib import Path

import pytest

from budget_agent.environment import DockerEnvironment, EnvironmentError, create_environment, _GIT_SNAPSHOT, _verified
from budget_agent.schema import RunConfig, Task


ROOT = Path(__file__).resolve().parents[1]


def test_grade_marker_after_interleaved_docker_test_progress():
    # Actual v2 failure: Docker interleaved stderr progress before the stdout verdict.
    marker = "BUDGET_AGENT_GRADE_fde87b61f4af484aa17c0d00fc6f5094"
    verdict = {"tests_run": 5, "failures": 0, "errors": 0, "skipped": 0, "successful": True}
    output = "test_hidden_4 (private_verifier.Tests.test_hidden_4) ... " + marker + json.dumps(verdict) + "\nok\nOK\n"
    assert _verified({"ok": True, "output": output}, marker) == (True, verdict)


def load_task(name="slugify-digits"):
    for line in (ROOT / "tasks/local.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["id"] == name:
            row["repo"] = str(ROOT / "tasks" / row["repo"])
            row["verifier"] = str(ROOT / "tasks" / row["verifier"])
            return Task(**row)
    raise AssertionError(name)


def test_baseline_fails_and_repair_passes_without_touching_fixture(tmp_path):
    task = load_task()
    original = (Path(task.repo) / "slugify.py").read_text(encoding="utf-8")
    env = create_environment(task, RunConfig(), tmp_path / "run")
    assert env.grade()["resolved"] is False
    revision = env.revision()
    fixed = original.replace("[^a-z]+", "[^a-z0-9]+")
    assert env.execute({"tool": "write_file", "path": "slugify.py", "content": fixed})["ok"]
    assert env.revision() != revision
    assert env.grade()["resolved"] is True
    assert "+" in env.patch() and "[^a-z0-9]+" in env.patch()
    assert (Path(task.repo) / "slugify.py").read_text(encoding="utf-8") == original
    assert not (env.workspace / "_private_verifier.py").exists()


@pytest.mark.parametrize("original", [b"target\nnext\n", b"target\r\nnext\r\n", b"target\r\nnext\n"],
                         ids=["lf", "crlf", "mixed"])
def test_edit_replacing_text_with_itself_leaves_file_untouched(tmp_path, original):
    env = create_environment(load_task(), RunConfig(), tmp_path / "run")
    path = env.workspace / "slugify.py"
    path.write_bytes(original)
    os.utime(path, (946684800, 946684800))
    modified = path.stat().st_mtime_ns
    result = env.execute({"tool": "edit_file", "path": "slugify.py", "old": "target", "new": "target"})
    assert result["ok"]
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == modified
    assert "unchanged" in result["output"].lower()


@pytest.mark.parametrize("old,error", [("", "non-empty"), ("missing", "found 0"), ("target", "found 2")])
def test_identical_replacement_still_requires_a_unique_anchor(tmp_path, old, error):
    env = create_environment(load_task(), RunConfig(), tmp_path / "run")
    path = env.workspace / "slugify.py"
    original = b"target\r\ntarget\n"
    path.write_bytes(original)
    result = env.execute({"tool": "edit_file", "path": "slugify.py", "old": old, "new": old})
    assert not result["ok"] and error in result["output"]
    assert path.read_bytes() == original


@pytest.mark.parametrize("path", ["../secret.txt", "../../secret.txt", ".git/config", ".GIT/config", ".git./config", "C:/secret.txt", "/etc/passwd", "tests/../../secret"])
def test_paths_cannot_escape(tmp_path, path):
    env = create_environment(load_task(), RunConfig(), tmp_path / "run")
    result = env.execute({"tool": "read_file", "path": path})
    assert not result["ok"]


def test_protected_test_edits_are_rejected_even_when_shell_bypasses_file_tool(tmp_path):
    env = create_environment(load_task(), RunConfig(), tmp_path / "run")
    assert not env.execute({"tool": "write_file", "path": "tests/test_public.py", "content": ""})["ok"]
    # An arbitrary command can mutate tests in a local workspace; grading rejects
    # that delta and never trusts this modified test suite.
    (env.workspace / "tests/test_public.py").write_text("# fake success\n", encoding="utf-8")
    result = env.grade()
    assert result["resolved"] is False
    assert result["status"] == "invalid_patch"


def test_real_model_cannot_execute_locally_without_opt_in(tmp_path):
    with pytest.raises(EnvironmentError, match="not sandboxed"):
        create_environment(load_task(), RunConfig(provider="openai-compatible"), tmp_path / "run")


def test_shell_does_not_receive_api_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "should-never-be-present")
    env = create_environment(load_task(), RunConfig(), tmp_path / "run")
    command = 'python -c "import os; print(os.environ.get(\'LLM_API_KEY\', \'absent\'))"'
    result = env.execute({"tool": "run", "command": command})
    assert result["ok"]
    assert result["output"].strip() == "absent"


def test_command_timeout(tmp_path):
    env = create_environment(load_task(), RunConfig(timeout_seconds=1), tmp_path / "run")
    result = env.execute({"tool": "run", "command": 'python -c "import time; time.sleep(20)"'})
    assert not result["ok"]
    assert result["timed_out"]
    assert result["duration_seconds"] < 10


@pytest.mark.parametrize("task_id,source,old,new", [
    ("chunks-positive-size", "chunks.py", "    return [", "    if size <= 0:\n        raise ValueError('size must be positive')\n    return ["),
    ("merge-touching-intervals", "intervals.py", "start < merged", "start <= merged"),
])
def test_other_fixture_graders_detect_actual_repairs(tmp_path, task_id, source, old, new):
    env = create_environment(load_task(task_id), RunConfig(), tmp_path / "run")
    assert env.grade()["resolved"] is False
    text = (env.workspace / source).read_text(encoding="utf-8")
    assert env.execute({"tool": "write_file", "path": source, "content": text.replace(old, new)})["ok"]
    assert env.grade()["resolved"] is True


def test_patch_includes_new_files_for_unrestricted_custom_task(tmp_path):
    task = load_task()
    task.metadata.pop("editable")
    env = create_environment(task, RunConfig(), tmp_path / "run")
    assert env.execute({"tool": "write_file", "path": "helper.py", "content": "VALUE = 1\n"})["ok"]
    assert "+++ b/helper.py" in env.patch()


def test_swebench_requires_docker(tmp_path):
    with pytest.raises(EnvironmentError, match="require environment=docker"):
        create_environment(Task(id="swe", issue="fix", split="test", kind="swebench"), RunConfig(), tmp_path)


def test_fixture_patch_refuses_lossy_replacement_of_non_utf8_bytes(tmp_path):
    env = create_environment(load_task(), RunConfig(), tmp_path / "run")
    (env.workspace / "slugify.py").write_bytes(b"# \xff")
    with pytest.raises(EnvironmentError, match="non-UTF-8.*slugify.py"):
        env.patch()


def test_a_verifier_hung_by_the_candidate_is_a_failed_repair(tmp_path):
    from budget_agent.evaluation import verified_outcome

    env = create_environment(load_task(), RunConfig(timeout_seconds=2), tmp_path / "run")
    env.execute({"tool": "write_file", "path": "slugify.py", "content": "while True:\n    pass\n"})
    result = env.grade()
    assert (result["resolved"], result["status"], result["timed_out"]) == (False, "unresolved", True)
    assert "test_results" not in result and result["grading_inputs"]["kind"] == "fixture"
    assert verified_outcome({"resolved": False, "evaluation": result}) is False


def test_the_task_limit_bounds_the_agents_commands(tmp_path):
    task = load_task()
    task.metadata = {**task.metadata, "test_timeout_seconds": 2}
    task.test_command = ["python", "-c", "import time; time.sleep(20)"]
    env = create_environment(task, RunConfig(timeout_seconds=60), tmp_path / "run")
    for action in ({"tool": "public_test"}, {"tool": "run", "command": 'python -c "import time; time.sleep(20)"'}):
        result = env.execute(action)
        assert result["timed_out"] and result["duration_seconds"] < 15, action


def test_the_task_limit_overrides_the_command_limit_for_grading(tmp_path):
    task = load_task()
    task.metadata = {**task.metadata, "test_timeout_seconds": 2}
    env = create_environment(task, RunConfig(timeout_seconds=60), tmp_path / "run")
    env.execute({"tool": "write_file", "path": "slugify.py", "content": "while True:\n    pass\n"})
    result = env.grade()
    assert result["timed_out"] and result["duration_seconds"] < 30


@pytest.mark.parametrize("value", [0, -5, 2.5, "30"])
def test_an_invalid_task_verifier_limit_is_an_infrastructure_error(tmp_path, value):
    task = load_task()
    task.metadata = {**task.metadata, "test_timeout_seconds": value}
    result = create_environment(task, RunConfig(), tmp_path / "run").grade()
    assert result["status"] == "infrastructure_error" and result["resolved"] is None


@pytest.mark.skipif(os.name == "nt", reason="ulimit is POSIX-only")
def test_command_memory_limit_turns_runaway_allocation_into_a_failed_command(tmp_path):
    hog = 'python -c "x = bytearray(600 * 1024 * 1024)"'
    unlimited = create_environment(load_task(), RunConfig(), tmp_path / "a")
    assert unlimited.execute({"tool": "run", "command": hog})["ok"]
    limited = create_environment(load_task(), RunConfig(command_memory_mb=256), tmp_path / "b")
    result = limited.execute({"tool": "run", "command": hog})
    assert not result["ok"] and "MemoryError" in result["output"]
    assert limited.execute({"tool": "run", "command": 'python -c "print(1)"'})["ok"]


@pytest.mark.skipif(os.name == "nt", reason="ulimit is POSIX-only")
def test_memory_limited_grading_scores_a_memory_hog_as_a_failure(tmp_path):
    env = create_environment(load_task(), RunConfig(command_memory_mb=256), tmp_path / "run")
    env.execute({"tool": "write_file", "path": "slugify.py",
                 "content": "HOG = bytearray(600 * 1024 * 1024)\n"})
    grade = env.grade()
    assert (grade["resolved"], grade["status"], grade["memory_exceeded"]) == (False, "unresolved", True)


def test_command_memory_limit_must_be_positive():
    for value in (0, -1, 1.5, "4096"):
        with pytest.raises(ValueError, match="command_memory_mb"):
            RunConfig(command_memory_mb=value).validate()


def test_zero_exit_during_import_cannot_fake_test_success(tmp_path):
    env = create_environment(load_task(), RunConfig(), tmp_path / "run")
    env.execute({"tool": "write_file", "path": "slugify.py", "content": "import os\nos._exit(0)\n"})
    result = env.grade()
    assert result["resolved"] is None
    assert result["status"] == "grading_error"
    assert result["test_results"] is None


@pytest.mark.parametrize("verifier", [
    "import missing_budget_agent_verifier_dependency\n",
    "# No tests discovered\n",
    "import unittest\n@unittest.skip('dependency unavailable')\n"
    "class Tests(unittest.TestCase):\n    def test_skipped(self): pass\n",
])
def test_unscored_local_verifier_is_not_a_failed_repair(tmp_path, verifier):
    task = load_task()
    path = tmp_path / "verifier.py"
    path.write_text(verifier, encoding="utf-8")
    task.verifier = str(path)
    result = create_environment(task, RunConfig(), tmp_path / "run").grade()
    assert result["resolved"] is None
    assert result["status"] == "grading_error"


@pytest.mark.parametrize("output,ok", [
    ("missing result marker", True),
    ("GRADE_not-json", True),
    ("GRADE_{}", True),
    ("GRADE_" + json.dumps({"successful": True, "tests_run": 1, "skipped": 0}), False),
])
def test_incomplete_verification_cannot_produce_reward(output, ok):
    assert _verified({"ok": ok, "output": output}, "GRADE_") == (None, None)


@pytest.mark.parametrize("verdict,expected", [
    (None, None),
    ({"successful": True, "tests_run": 0, "skipped": 0}, None),
    ({"successful": False, "tests_run": 1, "skipped": 0}, False),
    ({"successful": True, "tests_run": 1, "skipped": 0}, True),
])
def test_docker_grading_distinguishes_unscored_from_failure(tmp_path, monkeypatch, verdict, expected):
    import budget_agent.environment as module

    env = DockerEnvironment.__new__(DockerEnvironment)
    env.task, env.config = load_task(), RunConfig(environment="docker")
    env.provenance = {"image_id": "sha256:" + "a" * 64}
    env.run_dir = tmp_path / "run"
    env.local = create_environment(env.task, RunConfig(), env.run_dir)

    def process(command, *args):
        # Exercise the real grading command and verdict parser without starting Docker.
        import re
        assert command[-6:-2] == [env.provenance["image_id"], "python", "-I", "-B"]
        marker = re.search(r"BUDGET_AGENT_GRADE_[a-f0-9]+", command[-1])[0]
        output = "ModuleNotFoundError" if verdict is None else marker + json.dumps(verdict)
        return {"ok": verdict is not None, "output": output,
                "returncode": 1 if verdict is None else 0, "timed_out": False}

    monkeypatch.setattr(module, "_process", process)
    result = env.grade()
    assert result["resolved"] is expected
    assert result["status"] == ("grading_error" if expected is None else
                                "resolved" if expected else "unresolved")
    assert result["test_results"] == verdict
    if expected is None:
        assert "grading_inputs" not in result
    else:
        digest = hashlib.sha256(Path(env.task.verifier).read_bytes()).hexdigest()
        assert result["grading_inputs"] == {"kind": "fixture", "verifier_sha256": digest}


def test_docker_patch_refuses_truncation_and_uses_original_commit():
    env = DockerEnvironment.__new__(DockerEnvironment)
    env.local = None
    env.base_commit = "a" * 40
    commands = []

    def fake_exec(command):
        commands.append(command)
        return {"ok": True, "output": "partial diff", "output_truncated": command[-2] == "diff"}

    env._exec = fake_exec
    with pytest.raises(EnvironmentError, match="truncated patch"):
        env.patch()
    # One read-only snapshot command; the agent's git index is never staged into.
    assert commands == [["python", "-c", _GIT_SNAPSHOT, "diff", "a" * 40]]


def test_docker_revision_names_the_tree_inside_the_container():
    env = DockerEnvironment.__new__(DockerEnvironment)
    env.local = None
    env.base_commit = "b" * 40
    calls = []

    def fake_exec(command):
        calls.append(command)
        return {"ok": True, "output": "c" * 40 + "\n"}

    env._exec = fake_exec
    assert env.revision() == "c" * 16
    assert calls == [["python", "-c", _GIT_SNAPSHOT, "tree", "b" * 40]]
    assert "write-tree" in calls[0][2]


def test_docker_start_failure_is_not_a_model_failure(tmp_path, monkeypatch):
    import budget_agent.environment as module

    env = DockerEnvironment.__new__(DockerEnvironment)
    env.task = load_task()
    env.config = RunConfig(environment="docker")
    env.provenance = {"image_id": "sha256:" + "a" * 64}
    env.run_dir = tmp_path / "run"
    env.local = create_environment(env.task, RunConfig(), env.run_dir)
    monkeypatch.setattr(module, "_process", lambda *args, **kwargs: {"ok": False, "output": "daemon unavailable", "returncode": 125, "timed_out": False})
    result = env.grade()
    assert result["resolved"] is None
    assert result["status"] == "infrastructure_error"
    assert "grading_inputs" not in result


@pytest.fixture
def docker_setup(monkeypatch):
    """An image whose setup commit differs from the issue's source baseline."""
    import budget_agent.environment as module

    calls = []
    state = {"head": "b" * 40, "running": False, "failure": None,
             "image_id": "sha256:" + "c" * 64}

    def process(command, cwd, timeout, **kwargs):
        calls.append(command)
        output, ok = "", True
        if command[:2] == ["docker", "run"]:
            state["running"] = True
        elif command[:2] == ["docker", "inspect"]:
            output = state["image_id"]
        elif command[:3] == ["docker", "rm", "-f"]:
            state["running"] = False
        elif command[:2] == ["docker", "exec"]:
            argv = command[5:]
            if argv == ["git", "rev-parse", "HEAD"]:
                ok = state["failure"] != "revision"
                output = "not-a-commit" if state["failure"] == "invalid_head" else state["head"] + "\n"
            elif argv[:3] == ["git", "merge-base", "--is-ancestor"]:
                ok = state["failure"] != "ancestry"
            else:
                output = "command output"
        else:
            raise AssertionError(f"Unexpected host command: {command}")
        return {"ok": ok, "output": output, "returncode": 0 if ok else 1,
                "timed_out": False}

    monkeypatch.setattr(module, "_process", process)
    task = Task(id="astropy__astropy-12907", issue="fix", split="dev", kind="swebench",
                image="swebench/task:latest", metadata={"base_commit": "a" * 40})
    return task, calls, state


def test_swebench_preserves_official_setup_checkout_and_patch_baseline(tmp_path, docker_setup):
    task, calls, state = docker_setup
    env = DockerEnvironment(task, RunConfig(environment="docker"), tmp_path)
    assert state["head"] == env.base_commit == "b" * 40
    assert env.base_commit != task.metadata["base_commit"]
    assert env.local is None
    assert not any("--mount" in call for call in calls)
    git_calls = [call[5:] for call in calls if call[:2] == ["docker", "exec"]]
    assert git_calls == [["git", "rev-parse", "HEAD"],
                         ["git", "merge-base", "--is-ancestor", task.metadata["base_commit"], env.base_commit]]
    assert not any("reset" in call for call in calls)
    assert env.provenance == {"kind": "docker", "requested_image": task.image,
                              "image_id": state["image_id"], "task_base_commit": "a" * 40,
                              "image_checkout": "b" * 40}
    env.patch()
    assert calls[-1][5:] == ["python", "-c", _GIT_SNAPSHOT, "diff", "b" * 40]
    assert not any(call[5:7] == ["git", "add"] for call in calls)


@pytest.mark.parametrize("failure", ["revision", "invalid_head", "ancestry"])
def test_swebench_failed_checkout_validation_removes_container(tmp_path, docker_setup, failure):
    task, calls, state = docker_setup
    state["failure"] = failure
    with pytest.raises(EnvironmentError, match="SWE-bench"):
        DockerEnvironment(task, RunConfig(environment="docker"), tmp_path)
    assert not state["running"]
    assert calls[-1][:3] == ["docker", "rm", "-f"]


@pytest.mark.parametrize("base", ["", "HEAD", "a" * 7, "a" * 41, "a" * 63, "a" * 40 + "; echo bad"])
def test_swebench_requires_full_manifest_sha_before_starting_container(tmp_path, docker_setup, base):
    task, calls, state = docker_setup
    task.metadata["base_commit"] = base
    with pytest.raises(EnvironmentError, match="full commit SHA"):
        DockerEnvironment(task, RunConfig(environment="docker"), tmp_path)
    assert calls == []


def test_missing_image_identity_removes_container(tmp_path, docker_setup):
    task, calls, state = docker_setup
    state["image_id"] = "swebench/task:latest"
    with pytest.raises(EnvironmentError, match="actual Docker image identity"):
        DockerEnvironment(task, RunConfig(environment="docker"), tmp_path)
    assert not state["running"]
    assert calls[-1][:3] == ["docker", "rm", "-f"]


@pytest.mark.parametrize("kind,shell", [("swebench", "bash"), ("local", "sh")])
def test_container_run_uses_profile_for_swebench_only(tmp_path, docker_setup, kind, shell):
    task, calls, state = docker_setup
    env = DockerEnvironment(task, RunConfig(environment="docker"), tmp_path)
    env.task.kind = kind
    assert env.execute({"tool": "run", "command": "python --version"})["ok"]
    assert calls[-1][5:] == [shell, "-lc", "python --version"]
