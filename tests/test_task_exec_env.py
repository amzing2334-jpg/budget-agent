"""Per-task interpreters and editable-only grading for real-repository tasks."""
import sys

from budget_agent.environment import LocalEnvironment
from budget_agent.schema import RunConfig, Task


def make_task(tmp_path, **metadata):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "mod.py").write_text("VALUE = 0\n", encoding="utf-8")
    (repo / "tests" / "test_mod.py").write_text("from mod import VALUE\n", encoding="utf-8")
    verifier = tmp_path / "verify.py"
    verifier.write_text("import unittest\nfrom mod import VALUE\n\n\nclass T(unittest.TestCase):\n"
                        "    def test_value(self):\n        self.assertEqual(VALUE, 1)\n", encoding="utf-8")
    return Task(id="exec-env", issue="Set VALUE to 1.", split="dev", repo=str(repo), verifier=str(verifier),
                test_command=["python", "-c", "import os; print(os.environ['TASK_MARK'])"], metadata=metadata)


def test_exec_env_prepends_path_and_substitutes_the_workspace(tmp_path):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    task = make_task(tmp_path, exec_env={"PATH": str(bin_dir), "TASK_MARK": "{workspace}/src"})
    task.test_command = [sys.executable, "-c", "import os; print(os.environ['TASK_MARK'])"]
    env = LocalEnvironment(task, RunConfig(), tmp_path / "run")
    # Portable across sh and cmd.exe: let Python print the environment it received.
    probe = f'"{sys.executable}" -c "import os; print(os.environ[\'PATH\'].split(os.pathsep)[0]); print(os.environ[\'TASK_MARK\'])"'
    shown = env.execute({"tool": "run", "command": probe})["output"].splitlines()
    assert shown[0] == str(bin_dir)
    assert shown[1] == f"{env.workspace}/src"
    assert env.execute({"tool": "public_test"})["output"].strip() == f"{env.workspace}/src"


def test_editable_only_scope_grades_source_and_records_ignored_files(tmp_path):
    task = make_task(tmp_path, editable=["mod.py"], grading_scope="editable_only")
    env = LocalEnvironment(task, RunConfig(), tmp_path / "run")
    (env.workspace / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (env.workspace / "reproduce.py").write_text("print('repro')\n", encoding="utf-8")
    (env.workspace / "tests" / "test_mod.py").write_text("raise SystemExit\n", encoding="utf-8")
    grade = env.grade()
    assert grade["status"] == "resolved"
    assert grade["ignored_changes"] == ["reproduce.py", "tests/test_mod.py"]


def test_ignored_files_never_reach_the_grading_tree(tmp_path):
    # A new module that would shadow the code under test is not applied.
    task = make_task(tmp_path, editable=["mod.py"], grading_scope="editable_only")
    env = LocalEnvironment(task, RunConfig(), tmp_path / "run")
    (env.workspace / "mod").mkdir()
    (env.workspace / "mod" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert env.grade()["status"] == "unresolved"


def test_default_scope_still_rejects_protected_changes(tmp_path):
    task = make_task(tmp_path, editable=["mod.py"])
    env = LocalEnvironment(task, RunConfig(), tmp_path / "run")
    (env.workspace / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (env.workspace / "reproduce.py").write_text("print('repro')\n", encoding="utf-8")
    assert env.grade()["status"] == "invalid_patch"


def test_unknown_grading_scope_is_an_infrastructure_error(tmp_path):
    task = make_task(tmp_path, grading_scope="everything")
    env = LocalEnvironment(task, RunConfig(), tmp_path / "run")
    grade = env.grade()
    assert grade["status"] == "infrastructure_error" and grade["resolved"] is None


def test_network_deny_policy_points_clients_at_a_dead_proxy(tmp_path):
    import json
    task = make_task(tmp_path, network_policy="deny")
    env = LocalEnvironment(task, RunConfig(), tmp_path / "run")
    probe = (f'"{sys.executable}" -c "import json, os; print(json.dumps({{k: os.environ.get(k) for k in '
             f'(\'https_proxy\', \'HTTP_PROXY\', \'NO_PROXY\', \'PIP_NO_INDEX\')}}))"')
    shown = json.loads(env.execute({"tool": "run", "command": probe})["output"].strip().splitlines()[-1])
    assert shown == {"https_proxy": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
                     "NO_PROXY": "", "PIP_NO_INDEX": "1"}
    fetch = (f'"{sys.executable}" -c "import urllib.request; '
             f'urllib.request.urlopen(\'http://example.com\', timeout=5)"')
    assert not env.execute({"tool": "run", "command": fetch})["ok"]


def test_unknown_network_policy_is_rejected(tmp_path):
    env = LocalEnvironment(make_task(tmp_path, network_policy="open-ish"), RunConfig(), tmp_path / "run")
    result = env.execute({"tool": "run", "command": "echo hi"})
    assert not result["ok"] and "network_policy" in result["output"]
