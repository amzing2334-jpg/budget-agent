"""Offline, matched-task single-factor comparisons; no model calls or training."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from .report import is_infrastructure_error, load_results, verified_outcome


FACTORS = ("strategy", "decision_policy", "loop_policy", "reflection_policy")


def _factor_value(result: dict, factor: str) -> str:
    config = result["config"]
    if factor == "strategy":
        value = config.get("strategy", result.get("strategy"))
        if "strategy" in result and result["strategy"] != value:
            raise ValueError(f"Mismatched strategy fields for {result['task_id']}")
    elif factor in {"loop_policy", "reflection_policy"}:
        # Legacy runs predate loop control. This default applies only when it
        # is the explicitly selected comparison factor, not to other settings.
        value = config.get(factor, "off")
    else:
        value = config.get(factor)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {factor} for {result['task_id']}")
    return value


def _condition(result: dict, factor: str) -> dict:
    fields = {key: result.get(key) for key in ("provider", "model", "split", "task_kind", "evidence_kind", "harness")}
    config = {**result["config"], "strategy": _factor_value(result, "strategy")}
    fields["config"] = {key: value for key, value in config.items() if key not in {factor, "seed"}}
    fields["source_fingerprint"] = result.get("source_fingerprint")
    fields["model_metadata"] = result.get("model_metadata")
    return fields


def _outcome(result: dict) -> str:
    if is_infrastructure_error(result):
        return "infrastructure_error"
    resolved = verified_outcome(result)
    if resolved is True:
        return "success"
    if resolved is False:
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


def analyze_paired(results: list[dict], baseline: str, candidate: str, *, factor: str = "strategy") -> dict:
    if factor not in FACTORS:
        raise ValueError("factor must be one of " + ", ".join(FACTORS))
    if baseline == candidate:
        raise ValueError("Baseline and candidate must differ")
    indexed: dict[str, dict[tuple, dict]] = {baseline: {}, candidate: {}}
    for result in results:
        level = _factor_value(result, factor)
        if level not in indexed:
            continue
        if not result.get("task_fingerprint"):
            raise ValueError(f"Missing task fingerprint: {result['task_id']}")
        key = (result["task_id"], result["task_fingerprint"], result["config"].get("seed", 0))
        if key in indexed[level]:
            raise ValueError(f"Duplicate attempt for {factor}={level}: {key}")
        indexed[level][key] = result
    if not indexed[baseline] or not indexed[candidate]:
        raise ValueError(f"Both requested {factor} levels need saved results")
    if indexed[baseline].keys() != indexed[candidate].keys():
        raise ValueError("Unmatched task/fingerprint/repetition sets; do not silently drop missing attempts")
    reference = next(iter(indexed[baseline].values()))
    condition = _condition(reference, factor)
    pairs = []
    known_metadata: dict[str, set[str]] = {}
    for key, left in sorted(indexed[baseline].items()):
        right = indexed[candidate][key]
        if _condition(left, factor) != condition or _condition(right, factor) != condition:
            raise ValueError(f"Mismatched model/configuration for {key[0]}")
        for field in ("actual_models", "usage_sources"):
            for result in (left, right):
                observed = set(result.get(field) or [])
                if observed and observed != known_metadata.setdefault(field, observed):
                    raise ValueError(f"Mismatched {field} for {key[0]}")
        outcomes = [_outcome(left), _outcome(right)]
        records, ineligible_reasons = [], []
        for side, result, outcome in zip(("baseline", "candidate"), (left, right), outcomes):
            budget_valid = _budget_valid(result)
            if outcome not in {"success", "failure"}:
                ineligible_reasons.append(f"{side}: {outcome}")
            if not budget_valid:
                ineligible_reasons.append(f"{side}: invalid_budget")
            for field in ("actual_models", "usage_sources"):
                if not result.get(field):
                    ineligible_reasons.append(f"{side}: missing {field}")
            records.append({"outcome": outcome, "resolved": result.get("resolved"),
                            "budget_valid": budget_valid, "status": result["status"],
                            "evaluation_status": result.get("evaluation", {}).get("status"),
                            **{field: result.get(field) for field in ("actual_models", "usage_sources", "total_tokens", "elapsed_seconds", "cost_usd")},
                            **_behavior(result)})
        deltas = {}
        for field in ("total_tokens", "elapsed_seconds", "cost_usd", "repeat_reads_same_revision", "invalid_actions", "failed_tool_actions"):
            values = [record[field] for record in records]
            deltas[field] = values[1] - values[0] if all(isinstance(value, (int, float)) for value in values) else None
        pairs.append({"task_id": key[0], "task_fingerprint": key[1], "seed": key[2],
                      "evidence_kind": left.get("evidence_kind"), "eligible_graded_pair": not ineligible_reasons,
                      "ineligible_reasons": ineligible_reasons,
                      "baseline": records[0], "candidate": records[1], "candidate_minus_baseline": deltas})
    eligible_pairs = [pair for pair in pairs if pair["eligible_graded_pair"]]
    means = {}
    for field in pairs[0]["candidate_minus_baseline"]:
        values = [pair["candidate_minus_baseline"][field] for pair in eligible_pairs
                  if pair["candidate_minus_baseline"][field] is not None]
        means[field] = {"n": len(values), "mean": mean(values) if values else None}
    return {"factor": factor, "baseline": baseline, "candidate": candidate, "condition": condition, "matched_pairs": len(pairs),
            "eligible_graded_pairs": len(eligible_pairs),
            "outcome_counts": {side: dict(Counter(pair[side]["outcome"] for pair in pairs)) for side in ("baseline", "candidate")},
            "candidate_wins": sum(pair["baseline"]["resolved"] is False and pair["candidate"]["resolved"] is True for pair in eligible_pairs),
            "baseline_wins": sum(pair["baseline"]["resolved"] is True and pair["candidate"]["resolved"] is False for pair in eligible_pairs),
            "mean_candidate_minus_baseline": means, "pairs": pairs,
            "notes": ["Descriptive paired results only; no significance or generalization claim.",
                      "Only the declared factor may differ; task, source, model, other configuration, and repetition labels must match.",
                      "Means and wins exclude ungraded, infrastructure-error, invalid-budget, and missing-metadata pairs; all attempts remain in the ledger.",
                      "Repeated reads and failed tool actions are behavior proxies, not proof of wasted work or incorrect reasoning.",
                      "GPU rental and setup cost are not measured by model cost_usd; unknown cost remains null.",
                      "Scripted smoke results are infrastructure evidence, not model capability evidence."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--factor", choices=FACTORS, default="strategy")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze_paired(load_results(args.runs), args.baseline, args.candidate, factor=args.factor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Matched pairs: {report['matched_pairs']}; eligible graded pairs: {report['eligible_graded_pairs']}; report: {args.output}")


if __name__ == "__main__":
    main()
