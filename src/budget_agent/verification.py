"""Observed unittest outcomes and the shared fixture-verdict contract."""
from __future__ import annotations

import unittest


class VerificationResult(unittest.TextTestResult):
    """Record successful outcomes that the default result does not retain."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed = False

    def addSuccess(self, test):
        self.observed = True
        super().addSuccess(test)

    def addSubTest(self, test, subtest, err):
        self.observed = True
        super().addSubTest(test, subtest, err)

    def tests_executed(self) -> bool:
        return self.testsRun > 0 and bool(
            self.observed or self.failures or self.errors
            or self.expectedFailures or self.unexpectedSuccesses)


def fixture_outcome(verdict: object) -> bool | None:
    """Reject malformed evidence; a valid but unexecuted fixture has no outcome."""
    if not isinstance(verdict, dict):
        raise ValueError("Verifier result must be an object")
    counts = [verdict.get(key) for key in ("tests_run", "skipped")]
    counts.extend(verdict[key] for key in ("failures", "errors") if key in verdict)
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("Verifier counters must be nonnegative integers")
    successful = verdict.get("successful")
    if type(successful) is not bool:
        raise ValueError("Verifier successful must be a boolean")
    # Subtest failures can exceed tests_run; unexpectedSuccess can fail a suite
    # with zero failures/errors, so only a claimed success can contradict them.
    if successful and any(verdict.get(key, 0) > 0 for key in ("failures", "errors")):
        raise ValueError("Successful verifier result records failures or errors")
    if "tests_executed" in verdict:
        executed = verdict["tests_executed"]
        if type(executed) is not bool or (executed and verdict["tests_run"] == 0):
            raise ValueError("Verifier execution evidence is invalid")
    else:
        # Legacy records lack callback evidence; do not reconstruct lost facts.
        executed = verdict["tests_run"] > verdict["skipped"]
    return successful if executed else None
