"""Real unittest subprocesses distinguish executed outcomes from skip callbacks."""
import json
import subprocess
import sys
import textwrap

import pytest

from budget_agent.environment import EnvironmentError, _verification_command, _verified
from budget_agent.evaluation import evaluate_run
from budget_agent.learning.rollout import sample_episode
from budget_agent.learning.schema import TrainingConfig
from budget_agent.report import verified_outcome
from budget_agent.schema import Task
from test_evaluation import complete_result
from test_learning_rollout import TokenPolicy


def verifier(body):
    return "import unittest\nclass Tests(unittest.TestCase):\n" + textwrap.indent(textwrap.dedent(body), "    ")


MIXED_PASS = verifier('''\
    def test_value(self):
        with self.subTest(part='skip'):
            self.skipTest('not available')
        with self.subTest(part='pass'):
            self.assertEqual(1 + 1, 2)
''')
MIXED_FAIL = MIXED_PASS.replace("1 + 1, 2", "1 + 1, 3")
ALL_SKIP = verifier('''\
    def test_value(self):
        for part in range(2):
            with self.subTest(part=part):
                self.skipTest('not available')
''')


@pytest.mark.parametrize("source,expected", [
    pytest.param(MIXED_PASS, True, id="mixed-pass-skip"),
    pytest.param(MIXED_FAIL, False, id="mixed-failure-skip"),
    pytest.param(MIXED_PASS.replace("self.assertEqual(1 + 1, 2)", "raise RuntimeError('test error')"),
                 False, id="mixed-error-skip"),
    pytest.param(ALL_SKIP, None, id="all-subtests-skipped"),
    pytest.param(verifier("@unittest.skip('unavailable')\ndef test_value(self): pass\n"),
                 None, id="method-skipped"),
    pytest.param("import unittest\n", None, id="empty-suite"),
    pytest.param("import os\nos._exit(0)\n", None, id="import-exits-zero"),
    pytest.param(verifier("@unittest.expectedFailure\ndef test_value(self): pass\n"),
                 False, id="unexpected-success"),
    pytest.param(verifier("@unittest.expectedFailure\ndef test_value(self): self.fail('expected')\n"),
                 True, id="expected-failure"),
    pytest.param(verifier('''\
        def test_value(self):
            for part in range(2):
                with self.subTest(part=part):
                    self.skipTest('unavailable')
            with self.subTest(part='outer'):
                with self.subTest(part='inner'):
                    self.assertTrue(True)
    '''), True, id="nested-pass-multiple-skips"),
    pytest.param(verifier("@classmethod\ndef setUpClass(cls): raise RuntimeError('setup')\n"
                         "def test_value(self): pass\n"), None, id="class-setup-error"),
    pytest.param(verifier("def setUp(self): raise RuntimeError('setup')\n"
                         "def test_value(self): pass\n"), False, id="method-setup-error"),
])
def test_real_verifier_uses_observed_non_skip_outcomes(tmp_path, source, expected):
    (tmp_path / "_private_verifier.py").write_text(source, encoding="utf-8")
    command, marker = _verification_command(sys.executable)
    process = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=10)
    outcome, verdict = _verified({"ok": process.returncode == 0,
                                 "output": process.stdout + process.stderr}, marker)
    assert outcome is expected
    if verdict is not None:
        assert verdict["tests_executed"] is (expected is not None)
        record = complete_result()
        record.update(resolved=expected, evaluation={
            "status": "grading_error" if expected is None else "resolved" if expected else "unresolved",
            "resolved": expected, "test_results": verdict})
        assert verified_outcome(record) is expected
        assert evaluate_run(record)["dimensions"]["quality"]["status"] == (
            "unknown" if expected is None else "pass" if expected else "fail")


@pytest.mark.parametrize("source,expected", [(MIXED_PASS, True), (MIXED_FAIL, False), (ALL_SKIP, None)])
def test_real_local_grader_controls_episode_reward_without_a_model(tmp_path, source, expected):
    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "solution.py").write_text("VALUE = 0\n", encoding="utf-8")
    path = tmp_path / "verifier.py"
    path.write_text(source, encoding="utf-8")
    task = Task("subtest-reward", "Fix VALUE", "train", repo=str(repo), verifier=str(path))
    config = TrainingConfig(environment="local", allow_local_execution=True,
                            context_tokens=4096, total_tokens=16000)
    policy = TokenPolicy(['{"tool":"finish"}'])
    run = tmp_path / "run"
    if expected is None:
        with pytest.raises(EnvironmentError, match="No training reward: grader returned grading_error"):
            sample_episode(task, policy, config, run)
        assert not (run / "episode.json").exists()
    else:
        episode = sample_episode(task, policy, config, run)
        assert episode.resolved is expected
        assert episode.reward == float(expected)
        assert (run / "episode.json").is_file()
    grade = json.loads((run / "grade.json").read_text(encoding="utf-8"))
    assert grade["resolved"] is expected
    assert grade["test_results"]["tests_executed"] is (expected is not None)
    assert (run / "patch.diff").is_file() and (run / "trajectory.jsonl").is_file()
    assert (repo / "solution.py").read_text(encoding="utf-8") == "VALUE = 0\n"


@pytest.mark.parametrize("executed,tests_run", [(1, 1), ("true", 1), (True, 0), (False, 3)])
def test_explicit_execution_evidence_is_not_coerced_or_replaced(executed, tests_run):
    verdict = {"successful": True, "tests_run": tests_run, "skipped": 0,
               "tests_executed": executed}
    outcome, _ = _verified({"ok": True, "output": "GRADE_" + json.dumps(verdict)}, "GRADE_")
    assert outcome is None
    record = complete_result()
    record["evaluation"]["test_results"] = verdict
    assert verified_outcome(record) is None
    assert evaluate_run(record)["dimensions"]["quality"]["status"] == "unknown"


def test_legacy_counts_do_not_invent_missing_subtest_evidence():
    legacy = {"successful": True, "tests_run": 1, "skipped": 1}
    assert _verified({"ok": True, "output": "GRADE_" + json.dumps(legacy)}, "GRADE_") == (None, legacy)
    record = complete_result()
    record["evaluation"]["test_results"] = legacy
    assert verified_outcome(record) is None
