"""Rebuilding recorded requests under another revision-label mode matches a run made in that mode."""
import json
from pathlib import Path

import pytest

from budget_agent.budget import TokenCounter
from budget_agent.learning.rerender import RerenderError, rerender_requests
from budget_agent.models import ScriptedModel
from budget_agent.runner import run_task
from budget_agent.schema import RunConfig
from budget_agent.tasks import load_tasks

ROOT = Path(__file__).resolve().parents[1]
# Read, edit (a new revision), read again and test: the second request relabels the first read as stale.
ACTIONS = [
    {"tool": "read_file", "path": "slugify.py"},
    {"tool": "edit_file", "path": "slugify.py", "old": "[^a-z]+", "new": "[^a-z0-9]+"},
    {"tool": "read_file", "path": "slugify.py"},
    {"tool": "public_test"},
    {"tool": "finish"},
]


def run(tmp_path, labels, **config):
    tmp_path = tmp_path / "_".join(f"{k}{v}" for k, v in sorted(config.items()))
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    counter = TokenCounter()
    result = run_task(task, RunConfig(revision_labels=labels, **config), tmp_path / labels,
                      model=ScriptedModel([dict(a) for a in ACTIONS], counter), counter=counter)
    run_dir = Path(result["result_path"]).parent
    recorded = [json.loads(line)["messages"] for line in (run_dir / "calls.jsonl").read_text().splitlines()]
    return run_dir, recorded


@pytest.mark.parametrize("config", [
    {},                                                                      # structured (the default)
    {"strategy": "recent"},                                                  # the SWE-smith runs
    {"strategy": "recent", "context_tokens": 4000, "max_output_tokens": 512},  # the window slides
])
def test_relabelled_run_rebuilt_with_fixed_labels_equals_a_fixed_run(tmp_path, config):
    relabel_dir, relabel_calls = run(tmp_path, "relabel", **config)
    fixed_dir, fixed_calls = run(tmp_path, "fixed", **config)
    assert relabel_calls != fixed_calls                      # the modes really render differently
    assert rerender_requests(relabel_dir, "relabel") == relabel_calls
    assert rerender_requests(relabel_dir, "fixed") == fixed_calls
    assert rerender_requests(fixed_dir, "relabel") == relabel_calls


def test_a_request_that_does_not_reproduce_is_refused(tmp_path):
    run_dir, _ = run(tmp_path, "relabel")
    calls = (run_dir / "calls.jsonl").read_text().splitlines()
    first = json.loads(calls[1])
    first["messages"][-1]["content"] += " (edited)"
    (run_dir / "calls.jsonl").write_text("\n".join([calls[0], json.dumps(first), *calls[2:]]) + "\n")
    with pytest.raises(RerenderError, match="does not reproduce"):
        rerender_requests(run_dir, "fixed")


def test_runs_with_a_dynamic_system_prompt_are_refused(tmp_path):
    run_dir, _ = run(tmp_path, "relabel", decision_policy="budget-aware")
    with pytest.raises(RerenderError, match="decision_policy"):
        rerender_requests(run_dir, "fixed")


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        rerender_requests(tmp_path, "sometimes")
