"""The optional decision policy reports observed state and pays for its prompt."""
import json
from pathlib import Path

import pytest

from budget_agent.budget import TokenCounter
from budget_agent.models import ScriptedModel
from budget_agent.runner import SYSTEM, decision_prompt, run_task
from budget_agent.schema import RunConfig
from budget_agent.tasks import load_tasks


def test_direct_policy_keeps_original_prompt():
    assert decision_prompt(RunConfig(), 1000, 3, [], "changed", "base") == SYSTEM


def test_budget_guidance_tracks_actual_execution_and_is_billed(tmp_path):
    task = load_tasks(Path(__file__).resolve().parents[1] / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    counter = TokenCounter()
    actions = [
        {"tool": "read_file", "path": "slugify.py"},
        {"tool": "edit_file", "path": "slugify.py", "old": "[^a-z]+", "new": "[^a-z0-9]+"},
        {"tool": "finish"},
    ]
    config = RunConfig(strategy="mask", decision_policy="budget-aware", context_tokens=16000)
    result = run_task(task, config, tmp_path, model=ScriptedModel(actions, counter), counter=counter)
    calls = [json.loads(line) for line in Path(result["result_path"]).with_name("calls.jsonl").read_text().splitlines()]
    remaining = config.total_tokens
    for index, call in enumerate(calls, 1):
        prompt = call["messages"][0]["content"]
        assert f"{remaining}/{config.total_tokens} tokens remain" in prompt
        assert f"step {index}/{config.max_steps}" in prompt
        assert call["response"]["input_tokens"] == counter.messages(call["messages"])
        remaining -= call["response"]["input_tokens"] + call["response"]["output_tokens"]
    assert "consecutive evidence-reading attempts: 1" in calls[1]["messages"][0]["content"]
    assert "initial revision: false" in calls[1]["messages"][0]["content"]
    assert "initial revision: true" in calls[2]["messages"][0]["content"]
    assert "consecutive evidence-reading attempts: 0" in calls[2]["messages"][0]["content"]
    assert result["config"]["decision_policy"] == "budget-aware"
    assert result["total_tokens"] == config.total_tokens - remaining


def test_unknown_decision_policy_is_rejected():
    with pytest.raises(ValueError, match="decision_policy"):
        RunConfig(decision_policy="unknown").validate()


def test_budget_guidance_refreshes_after_paid_summary(tmp_path, monkeypatch):
    def summarize_during_build(self, system, issue, events, revision, cap, summarize_callback):
        summarize_callback([{"role": "user", "content": "Summarize evidence."}], 256)
        return [{"role": "system", "content": system}, {"role": "user", "content": issue}]

    monkeypatch.setattr("budget_agent.runner.ContextManager.build", summarize_during_build)
    task = load_tasks(Path(__file__).resolve().parents[1] / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    counter = TokenCounter()
    config = RunConfig(strategy="summary", decision_policy="budget-aware", context_tokens=16000)
    result = run_task(task, config, tmp_path, model=ScriptedModel([{"tool": "finish"}], counter), counter=counter)
    summary, agent = [json.loads(line) for line in Path(result["result_path"]).with_name("calls.jsonl").read_text().splitlines()]
    spent = summary["response"]["input_tokens"] + summary["response"]["output_tokens"]
    assert spent > 0
    assert f"{config.total_tokens-spent}/{config.total_tokens} tokens remain" in agent["messages"][0]["content"]
    assert agent["response"]["input_tokens"] == counter.messages(agent["messages"])
