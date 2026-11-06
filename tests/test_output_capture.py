import sys

from budget_agent.environment import OUTPUT_CAPTURE_BYTES, _process


def test_long_output_keeps_the_failure_summary_at_the_end(tmp_path):
    code = ("import sys; print('SETUP-LINE'); sys.stdout.write('x' * 300000); "
            "print(); print('FAILED tests/test_mod.py::test_edge - AssertionError')")
    result = _process([sys.executable, "-c", code], tmp_path, 30)
    assert result["output_truncated"] is True
    assert result["output"].startswith("SETUP-LINE")
    assert result["output"].rstrip().endswith("FAILED tests/test_mod.py::test_edge - AssertionError")
    assert "bytes omitted from the middle" in result["output"]
    assert len(result["output"].encode("utf-8")) < OUTPUT_CAPTURE_BYTES + 200


def test_short_output_is_returned_whole(tmp_path):
    result = _process([sys.executable, "-c", "print('ok')"], tmp_path, 30)
    assert result["output_truncated"] is False
    assert result["output"].strip() == "ok"


def test_verbose_verifier_keeps_its_verdict_line(tmp_path):
    # The verdict is printed last. Head-only capture dropped it once the
    # verifier's own output passed the limit, turning a pass into grading_error.
    from budget_agent.environment import LocalEnvironment
    from budget_agent.schema import RunConfig, Task

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    verifier = tmp_path / "verify.py"
    verifier.write_text("import unittest\nfrom mod import VALUE\n\n\nclass T(unittest.TestCase):\n"
                        "    def test_value(self):\n        print('x' * 250000)\n"
                        "        self.assertEqual(VALUE, 1)\n", encoding="utf-8")
    task = Task(id="verbose", issue="noop", split="dev", repo=str(repo), verifier=str(verifier),
                metadata={"editable": ["mod.py"]})
    grade = LocalEnvironment(task, RunConfig(), tmp_path / "run").grade()
    assert grade["status"] == "resolved"
