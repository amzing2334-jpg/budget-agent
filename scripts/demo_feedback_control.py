"""Three feedback policies with fixed actions: software accounting, not model gains."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    task = load_tasks(root / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    config = RunConfig(strategy="recent", execution_policy="restricted", context_tokens=16000,
                       total_tokens=64000, max_output_tokens=1536,
                       reflection_max_output_tokens=256, max_reflections=1)
    actions = [
        {"tool": "read_file", "path": "slugify.py"},
        {"tool": "edit_file", "path": "slugify.py", "old": "absent anchor", "new": "unused"},
        {"tool": "edit_file", "path": "slugify.py", "old": "[^a-z]+", "new": "[^a-z0-9]+"},
        {"tool": "public_test"}, {"tool": "diff"}, {"tool": "finish"},
    ]
    rows = []
    for policy in ("off", "evidence", "failure"):
        counter = TokenCounter()
        result = run_task(task, replace(config, reflection_policy=policy), args.output / policy,
                          model=ScriptedModel(actions, counter), counter=counter)
        calls = [json.loads(line) for line in Path(result["result_path"]).with_name("calls.jsonl").read_text(encoding="utf-8").splitlines()]
        memory_tokens = sum(counter.messages([message]) - counter.messages([])
                            for call in calls if call["purpose"] == "agent"
                            for message in call["messages"]
                            if message["content"].startswith(('{"kind": "failure_evidence"', '{"kind": "reflection_hypothesis"')))
        rows.append({"policy": policy, "resolved": result["resolved"], "calls": result["calls"],
                     "reflection_calls": result["reflection"]["calls"],
                     "memory_notes": len(result["reflection"]["notes"]),
                     "feedback_events": result["reflection"]["events"],
                     "total_tokens": result["total_tokens"], "memory_input_tokens": memory_tokens,
                     "source_result": result["result_path"]})
    results = load_results(args.output)
    comparisons = {}
    for baseline, candidate in (("off", "evidence"), ("evidence", "failure"), ("off", "failure")):
        comparison = analyze_paired(results, baseline, candidate, factor="reflection_policy")
        filename = f"paired-{baseline}-{candidate}.json"
        (args.output / filename).write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        comparisons[f"{baseline}/{candidate}"] = filename
    summary = {"kind": "scripted_feedback_control_not_model_evaluation", "source_fingerprint": source_fingerprint(),
               "real_model_requests": 0, "token_counter": "byte", "total_token_ceiling": config.total_tokens,
               "limitations": ["Actions are predetermined and do not respond to feedback.",
                               "Equal total ceiling is not equal actual spend or equal information.",
                               "No success-rate, causal reasoning benefit or provider billing claim."],
               "runs": rows, "comparisons": comparisons}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths = generate_report(args.output, args.output / "report")
    print(json.dumps({"report": str(paths["html"]), "rows": [
        {key: row[key] for key in ("policy", "resolved", "calls", "reflection_calls", "total_tokens", "memory_input_tokens")}
        for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
