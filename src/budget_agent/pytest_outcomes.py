"""Pytest plugin that writes one exact outcome per collected node ID.

Loaded with `-p budget_agent.pytest_outcomes` and BUDGET_AGENT_OUTCOMES=<path>.
It imports nothing from budget_agent, so test environments only need this file
on PYTHONPATH. Log parsing is avoided: node IDs come from pytest itself.
"""
import json
import os

_outcomes: dict[str, str] = {}


def pytest_runtest_logreport(report):
    # Any failing phase fails the test. A skip in setup or call is a skip.
    previous = _outcomes.get(report.nodeid)
    if report.failed:
        _outcomes[report.nodeid] = "failed" if report.when == "call" else "error"
    elif report.skipped and previous != "failed":
        _outcomes[report.nodeid] = "xfailed" if hasattr(report, "wasxfail") else "skipped"
    elif report.passed and report.when == "call" and previous is None:
        _outcomes[report.nodeid] = "xpassed" if hasattr(report, "wasxfail") else "passed"


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("BUDGET_AGENT_OUTCOMES")
    if path:
        with open(path, "w", encoding="utf-8") as stream:
            json.dump({"exitstatus": int(exitstatus), "outcomes": _outcomes}, stream)
