import hashlib
from pathlib import Path

import pytest

from budget_agent import environment
from budget_agent.schema import RunConfig
from budget_agent.tasks import load_tasks


@pytest.fixture
def fixture_env(tmp_path):
    manifest = Path(__file__).resolve().parents[1] / "tasks/local.jsonl"
    task = load_tasks(manifest, task_ids=["slugify-digits"])[0]
    task.verifier = str(tmp_path / "verifier.py")
    return environment.create_environment(task, RunConfig(), tmp_path / "run")


def test_fixture_grades_bind_the_staged_bytes_and_ignore_the_source_path(fixture_env, tmp_path, monkeypatch):
    source = Path(fixture_env.task.verifier)
    passing = (b"import unittest\r\nfrom slugify import slugify\r\n"
               b"class Check(unittest.TestCase):\r\n"
               b"    def test_slug(self):\r\n        self.assertEqual(slugify('test'), 'test')\r\n")
    failing = passing.replace(b", 'test')", b", 'wrong')")
    source.write_bytes(passing)
    process = environment._process

    def replace_after_staging(command, cwd, timeout):
        assert (cwd / "_private_verifier.py").read_bytes() == passing
        source.write_bytes(failing)
        return process(command, cwd, timeout)

    with monkeypatch.context() as patch:
        patch.setattr(environment, "_process", replace_after_staging)
        first = fixture_env.grade()
    assert first["resolved"] is True
    assert first["grading_inputs"] == {"kind": "fixture", "verifier_sha256": hashlib.sha256(passing).hexdigest()}

    second = fixture_env.grade()
    assert second["resolved"] is False
    assert second["grading_inputs"]["verifier_sha256"] == hashlib.sha256(failing).hexdigest()
    relocated = source.rename(tmp_path / "renamed.py")
    fixture_env.task.verifier = str(relocated)
    assert fixture_env.grade()["grading_inputs"] == second["grading_inputs"]


def test_result_without_verifier_execution_has_no_verifier_digest(fixture_env):
    missing = fixture_env.grade()
    assert missing["status"] == "infrastructure_error"
    assert "grading_inputs" not in missing

    (fixture_env.workspace / "tests/test_public.py").write_bytes(b"# modified test\n")
    rejected = fixture_env.grade()
    assert rejected["status"] == "invalid_patch"
    assert "grading_inputs" not in rejected
