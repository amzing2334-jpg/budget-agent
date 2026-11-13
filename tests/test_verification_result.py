"""The fixture verifier's result records whether tests really executed, not just whether none failed."""
import io
import unittest

from budget_agent.verification import VerificationResult


def run(*cases):
    suite = unittest.TestSuite(cases)
    result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=VerificationResult, verbosity=0).run(suite)
    return result


def cases():
    """Built inside a function so pytest does not collect (and run) these as its own tests."""
    class Cases(unittest.TestCase):
        def test_pass(self):
            self.assertTrue(True)

        def test_fail(self):
            self.assertTrue(False)

        @unittest.skip("not applicable")
        def test_skipped(self):
            pass

        def test_subtests(self):
            for value in (1, 2):
                with self.subTest(value=value):
                    self.assertGreater(value, 0)
    return Cases


def test_a_passing_test_counts_as_executed():
    result = run(cases()("test_pass"))
    assert result.observed and result.tests_executed()


def test_only_skipped_tests_do_not_count_as_executed():
    result = run(cases()("test_skipped"))
    assert result.testsRun == 1 and not result.observed and not result.tests_executed()


def test_failures_and_passing_subtests_are_evidence_of_execution():
    assert run(cases()("test_fail")).tests_executed()
    assert run(cases()("test_subtests")).observed


def test_an_empty_suite_is_not_executed():
    assert not run().tests_executed()
