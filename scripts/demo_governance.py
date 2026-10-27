"""Offline, predetermined repairs demonstrating contracts, never model capability."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from budget_agent.budget import TokenCounter
from budget_agent.evaluation import evaluate_run
from budget_agent.models import ScriptedModel
from budget_agent.report import generate_report
from budget_agent.runner import run_task, source_fingerprint
from budget_agent.schema import RunConfig
from budget_agent.tasks import load_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    task = load_tasks(root / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    original = (Path(task.repo) / "slugify.py").read_text(encoding="utf-8")
    repaired = original.replace("[^a-z]+", "[^a-z0-9]+")
    config = RunConfig(strategy="recent", context_tokens=16000, max_output_tokens=1536,
                       execution_policy="restricted", reflection_policy="failure")
    scenarios = {
        "guarded-repair": [
            {"tool": "read_file", "path": "slugify.py"},
            {"tool": "edit_file", "path": "slugify.py", "old": "intentionally absent anchor", "new": "unused"},
            {"tool": "run", "command": "echo direct shell attempt"},
            {"tool": "write_file", "path": "slugify.py", "content": repaired},
            {"tool": "public_test"}, {"tool": "diff"}, {"tool": "finish"},
        ],
        "public-pass-without-repair": [{"tool": "public_test"}, {"tool": "finish"}],
    }
    results = []
    for seed, (name, actions) in enumerate(scenarios.items()):
        counter = TokenCounter()
        result = run_task(task, replace(config, seed=seed), args.output / name,
                          model=ScriptedModel(actions, counter), counter=counter)
        results.append({"scenario": name, "source_result": result["result_path"],
                        "contract": evaluate_run(result)})
    summary = {"kind": "scripted_software_contract_demo_not_model_evaluation",
               "source_fingerprint": source_fingerprint(), "model_requests": 0, "runs": results}
    (args.output / "acceptance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = generate_report(args.output, args.output / "report")
    print(json.dumps({"kind": summary["kind"], "report": str(report["html"]),
                      "outcomes": {row["scenario"]: row["contract"]["acceptance_status"] for row in results}}, indent=2))


if __name__ == "__main__":
    main()
