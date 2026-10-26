"""CPU-only acceptance scenarios with predetermined actions, not model results."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from budget_agent.budget import TokenCounter
from budget_agent.models import ScriptedModel
from budget_agent.paired import analyze_paired
from budget_agent.report import generate_report, load_results
from budget_agent.runner import run_task, source_fingerprint
from budget_agent.schema import RunConfig
from budget_agent.tasks import load_tasks


ROOT = Path(__file__).resolve().parents[1]
READ = {"tool": "read_file", "path": "slugify.py"}
SCENARIOS = {
    "persistent-read": [READ] * 6 + [{"tool": "finish"}],
    "edit-after-read": [READ, READ,
        {"tool": "edit_file", "path": "slugify.py", "old": "[^a-z]+", "new": "[^a-z0-9]+"},
        {"tool": "run", "command": "python -m unittest discover -s tests -v"},
        {"tool": "diff"}, {"tool": "finish"}],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    config = RunConfig(strategy="mask", context_tokens=16000, max_steps=12)
    cases = []
    for name, actions in SCENARIOS.items():
        # These are two software scenarios on ONE fixture, not independent issues.
        task = replace(fixture, id=fixture.id + "-" + name)
        for policy in ("off", "recover"):
            counter = TokenCounter()
            result = run_task(task, replace(config, loop_policy=policy), args.output / "runs",
                              model=ScriptedModel(actions, counter), counter=counter)
            expected = "stalled" if name == "persistent-read" and policy == "recover" else "finished"
            assert result["status"] == expected, result["error"]
            assert result["resolved"] is (name == "edit-after-read")
            assert not result["budget"]["overrun"] and not result["usage_unknown"]
            cases.append({"scenario": name, "policy": policy, "status": result["status"],
                          "resolved": result["resolved"], "calls": result["calls"],
                          "total_tokens": result["total_tokens"],
                          "warnings_sent": result["loop_control"]["warnings_sent"],
                          "result": str(Path(result["result_path"]).relative_to(args.output.resolve()))})
    paired = analyze_paired(load_results(args.output / "runs"), "off", "recover", factor="loop_policy")
    (args.output / "paired.json").write_text(json.dumps(paired, indent=2) + "\n", encoding="utf-8")
    generate_report(args.output / "runs", args.output / "report")
    receipt = {"evidence_kind": "scripted_smoke", "source_fingerprint": source_fingerprint(),
               "scenarios_on_one_fixture": len(SCENARIOS), "cases": cases,
               "limitations": "Predetermined actions: verifies controller, budget and independent grader only. "
                              "A scripted edit after feedback is not evidence that a model learned or recovered. "
                              "Estimated byte tokens and fewer calls do not establish model cost or repair gains."}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
