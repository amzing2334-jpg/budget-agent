"""Batch execution must not start new attempts after unsettled model usage."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from budget_agent import report, runner, tasks
from budget_agent.cli import main
from budget_agent.models import ModelError
from budget_agent.schema import ModelResponse, Task


ROOT = Path(__file__).resolve().parents[1]
MATRIX = [(task, strategy, repeat) for repeat in (0, 1)
          for task in ("a", "b") for strategy in ("mask", "mask-read")]


def result(status="finished", **changes):
    row = {"status": status, "resolved": False, "total_tokens": 1,
           "usage_unknown": False, "budget": {"overrun": False, "unreconciled_request": False},
           "evaluation": {"status": "unresolved"}, "result_path": "fixture/result.json"}
    row.update(changes)
    return row


def batch(tmp_path, monkeypatch, outcome):
    attempts, reports = [], []

    def run(task, config, output):
        attempts.append((task.id, config.strategy, config.seed))
        return deepcopy(outcome(len(attempts)))

    def generate(source, output):
        reports.append(list(attempts))
        output.mkdir(parents=True)
        path = output / "report.html"
        path.write_text("CPU report fixture", encoding="utf-8")
        return {"html": path}

    monkeypatch.setattr(runner, "run_task", run)
    monkeypatch.setattr(tasks, "load_tasks", lambda *args: [Task("a", "a", "dev"), Task("b", "b", "dev")])
    monkeypatch.setattr(report, "generate_report", generate)
    arguments = ["compare", "--provider", "scripted", "--strategies", "mask", "mask-read",
                 "--repeat", "2", "--output", str(tmp_path / "runs")]
    return arguments, attempts, reports


@pytest.mark.parametrize("failure", [
    result("model_error", usage_unknown=True),
    result("model_error", budget={"overrun": False, "unreconciled_request": True}),
    result("budget_exhausted", budget={"overrun": True, "unreconciled_request": False}),
], ids=["unknown_usage", "unreconciled_request", "actual_overrun"])
@pytest.mark.parametrize("stop_at", [1, 5], ids=["first_attempt", "later_repeat"])
def test_batch_stops_all_remaining_dimensions_and_reports_completed_prefix(tmp_path, monkeypatch, failure, stop_at):
    arguments, attempts, reports = batch(tmp_path, monkeypatch, lambda n: failure if n == stop_at else result())
    assert main(arguments) == 1
    assert attempts == MATRIX[:stop_at]
    assert reports == [MATRIX[:stop_at]]
    assert (tmp_path / "runs/report/report.html").is_file()


def test_run_with_multiple_tasks_also_stops_on_unknown_usage(tmp_path, monkeypatch):
    _, attempts, reports = batch(tmp_path, monkeypatch, lambda n: result("model_error", usage_unknown=True))
    assert main(["run", "--provider", "scripted", "--strategy", "mask",
                 "--output", str(tmp_path / "runs")]) == 1
    assert attempts == [("a", "mask", 0)]
    assert reports == [attempts]


@pytest.mark.parametrize("status", ["finished", "budget_exhausted"])
def test_known_unresolved_or_normal_budget_limit_runs_full_matrix(tmp_path, monkeypatch, status):
    arguments, attempts, reports = batch(tmp_path, monkeypatch, lambda n: result(status))
    assert main(arguments) == 0
    assert attempts == MATRIX
    assert reports == [MATRIX]


def test_interruption_retains_exit_130_even_with_unreconciled_usage(tmp_path, monkeypatch):
    interrupted = result("interrupted", usage_unknown=True,
                         budget={"overrun": False, "unreconciled_request": True})
    arguments, attempts, reports = batch(tmp_path, monkeypatch,
                                        lambda n: interrupted if n == 3 else result())
    assert main(arguments) == 130
    assert attempts == MATRIX[:3]
    assert reports == [MATRIX[:3]]


def test_real_runner_saves_prior_results_unknown_ledger_and_report_before_batch_stop(tmp_path, monkeypatch):
    requests = []

    class CPUResponses:
        name = "scripted-batch-stop-fixture"

        def __init__(self, actions, counter):
            self.counter = counter

        def complete(self, messages, max_output_tokens, *, purpose="agent"):
            requests.append(messages)
            if len(requests) == 2:
                raise ModelError("CPU fixture transport failure; usage unknown")
            content = '{"tool":"finish"}'
            return ModelResponse(content, self.counter.messages(messages), self.counter.count(content),
                                 "scripted_estimate", self.name)

    monkeypatch.setattr(runner, "ScriptedModel", CPUResponses)
    output = tmp_path / "runs"
    task = tasks.load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    original_source = Path(task.repo) / "slugify.py"
    original_bytes = original_source.read_bytes()
    code = main(["compare", "--provider", "scripted", "--environment", "local",
                 "--tasks", str(ROOT / "tasks/local.jsonl"), "--task", "slugify-digits",
                 "--strategies", "mask", "mask-read", "--repeat", "2", "--output", str(output)])
    assert code == 1
    assert len(requests) == 2
    results = [json.loads(path.read_text(encoding="utf-8")) for path in output.rglob("result.json")]
    assert len(results) == 2
    settled, = [row for row in results if row["status"] == "finished"]
    unknown, = [row for row in results if row["status"] == "model_error"]
    assert settled["calls"] == 1 and settled["resolved"] is False
    assert unknown["usage_unknown"] is True
    assert unknown["budget"]["unreconciled_request"] is True
    assert unknown["budget"]["attempted_calls"] == 1
    assert unknown["calls"] == 0
    calls_path = Path(unknown["result_path"]).with_name("calls.jsonl")
    calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 1 and calls[0]["status"] == "usage_unknown"
    assert (output / "report/report.html").is_file()
    assert original_source.read_bytes() == original_bytes
