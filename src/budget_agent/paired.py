"""Offline, matched-task context-policy comparisons; no model calls or training."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from .report import load_results


def _condition(result: dict) -> dict:
    fields = {key: result.get(key) for key in ("provider", "model", "split", "task_kind", "evidence_kind")}
    fields["config"] = {key: value for key, value in result["config"].items() if key not in {"strategy", "seed"}}
    fields["source_fingerprint"] = result.get("source_fingerprint")
    fields["model_metadata"] = result.get("model_metadata")
    return fields


def _outcome(result: dict) -> str:
    if result["status"] in {"environment_error", "model_error"} or result.get("evaluation", {}).get("status") in {"grading_error", "infrastructure_error"}:
        return "infrastructure_error"
    if result.get("resolved") is True:
        return "success"
    if result.get("resolved") is False:
        return "failure"
    return "ungraded"


def _budget_valid(result: dict) -> bool:
    budget = result.get("budget", {})
    return not (result.get("usage_unknown") or budget.get("overrun") or budget.get("unreconciled_request")
                or result["status"] in {"budget_overrun", "usage_unknown"})


def _behavior(result: dict) -> dict:
    path = Path(result["_result_path"]).parent / "trajectory.jsonl"
    if not path.is_file():
        return {"repeat_reads_same_revision": None, "invalid_actions": None, "failed_tool_actions": None}
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reads: set[tuple] = set()
    repeated = 0
    for event in events:
        action = event["action"]
        if action["tool"] == "read_file":
            key = (event["revision"], action["path"], action.get("start_line", 1), action.get("end_line"))
            repeated += key in reads
            reads.add(key)
    return {"repeat_reads_same_revision": repeated,
            "invalid_actions": sum(event["action"]["tool"] == "invalid_action" for event in events),
            "failed_tool_actions": sum(event["ok"] is False for event in events)}


def analyze_paired(results: list[dict], baseline: str, candidate: str) -> dict:
    if baseline == candidate:
        raise ValueError("Baseline and candidate must differ")
    indexed: dict[str, dict[tuple, dict]] = {baseline: {}, candidate: {}}
    for result in results:
        strategy = result["strategy"]
        if strategy not in indexed:
            continue
        if not result.get("task_fingerprint"):
            raise ValueError(f"Missing task fingerprint: {result['task_id']}")
        key = (result["task_id"], result["task_fingerprint"], result["config"].get("seed", 0))
        if key in indexed[strategy]:
            raise ValueError(f"Duplicate attempt for {strategy}: {key}")
        indexed[strategy][key] = result
    if not indexed[baseline] or not indexed[candidate]:
        raise ValueError("Both requested strategies need saved results")
    if indexed[baseline].keys() != indexed[candidate].keys():
        raise ValueError("Unmatched task/fingerprint/repetition sets; do not silently drop missing attempts")
    condition = _condition(next(iter(indexed[baseline].values())))
    pairs = []
    for key, left in sorted(indexed[baseline].items()):
        right = indexed[candidate][key]
        if _condition(left) != condition or _condition(right) != condition:
            raise ValueError(f"Mismatched model/configuration for {key[0]}")
        for field in ("actual_models", "usage_sources"):
            if left.get(field) and right.get(field) and set(left[field]) != set(right[field]):
                raise ValueError(f"Mismatched {field} for {key[0]}")
        outcomes = [_outcome(left), _outcome(right)]
        eligible = (all(outcome in {"success", "failure"} for outcome in outcomes)
                    and _budget_valid(left) and _budget_valid(right))
        records = []
        for result, outcome in zip((left, right), outcomes):
            records.append({"outcome": outcome, "resolved": result.get("resolved"),
                            "budget_valid": _budget_valid(result), "status": result["status"],
                            **{field: result.get(field) for field in ("total_tokens", "elapsed_seconds", "cost_usd")},
                            **_behavior(result)})
        deltas = {}
        for field in ("total_tokens", "elapsed_seconds", "cost_usd", "repeat_reads_same_revision", "invalid_actions", "failed_tool_actions"):
            values = [record[field] for record in records]
            deltas[field] = values[1] - values[0] if all(isinstance(value, (int, float)) for value in values) else None
        pairs.append({"task_id": key[0], "task_fingerprint": key[1], "seed": key[2],
                      "evidence_kind": left.get("evidence_kind"), "eligible_graded_pair": eligible,
                      "baseline": records[0], "candidate": records[1], "candidate_minus_baseline": deltas})
    eligible_pairs = [pair for pair in pairs if pair["eligible_graded_pair"]]
    means = {}
    for field in pairs[0]["candidate_minus_baseline"]:
        values = [pair["candidate_minus_baseline"][field] for pair in eligible_pairs
                  if pair["candidate_minus_baseline"][field] is not None]
        means[field] = {"n": len(values), "mean": mean(values) if values else None}
    return {"baseline": baseline, "candidate": candidate, "condition": condition, "matched_pairs": len(pairs),
            "eligible_graded_pairs": len(eligible_pairs),
            "outcome_counts": {side: dict(Counter(pair[side]["outcome"] for pair in pairs)) for side in ("baseline", "candidate")},
            "candidate_wins": sum(pair["baseline"]["resolved"] is False and pair["candidate"]["resolved"] is True for pair in eligible_pairs),
            "baseline_wins": sum(pair["baseline"]["resolved"] is True and pair["candidate"]["resolved"] is False for pair in eligible_pairs),
            "mean_candidate_minus_baseline": means, "pairs": pairs,
            "notes": ["Descriptive paired results only; no significance or generalization claim.",
                      "Means and wins exclude ungraded, infrastructure-error, and invalid-budget pairs; all attempts remain in the ledger.",
                      "Repeated reads and failed tool actions are behavior proxies, not proof of wasted work or incorrect reasoning.",
                      "GPU rental and setup cost are not measured by model cost_usd; unknown cost remains null.",
                      "Scripted smoke results are infrastructure evidence, not model capability evidence."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze_paired(load_results(args.runs), args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Matched pairs: {report['matched_pairs']}; eligible graded pairs: {report['eligible_graded_pairs']}; report: {args.output}")


if __name__ == "__main__":
    main()
