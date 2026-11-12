"""Summarize one RFT round into Markdown and JSON (no model calls).

Reads every rollout pool under --rollouts (summary.json, attempts.jsonl and
trajectories), the training outputs under --outputs (sft.jsonl,
sft/result.json, sft/train_log.jsonl, compare_*.json, steps.jsonl), and
writes results.json plus results.md with pass@k per pool, status counts,
edit-tool success, a tamper audit of every resolved attempt that tripped a
patch signal, SFT data size, the loss curve, paired comparisons, and wall
time and cost per stage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tamper_audit(attempts: list[dict]) -> list[dict]:
    """Resolved attempts with patch signals, each signal marked graded or ignored by the grader."""
    audit = []
    for row in attempts:
        if not (row["resolved"] and row["patch_signals"]):
            continue
        ignored = set(row["ignored_changes"])
        records = json.loads(Path(row["result_path"]).read_text(encoding="utf-8"))["patch_signals"]
        audit.append({"task_id": row["task_id"], "sample": row["sample"],
                      "signals": [{**record, "graded": record["file"] not in ignored} for record in records]})
    return audit


def grpo_summary(rows: list[dict]) -> dict:
    """One GRPO run from grpo/metrics.jsonl: volume, data-path checks, update health and time."""
    def span(key: str) -> list[float] | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return [round(min(values), 4), round(max(values), 4)] if values else None
    train, rollout = [r["train_seconds"] for r in rows], [r["rollout_seconds"] for r in rows]
    turns = sum(r["turns"] for r in rows)
    return {"steps": len(rows), "updates": sum(r["updates"] for r in rows),
            "attempts": sum(r["attempts"] for r in rows), "ungraded": sum(r["ungraded_attempts"] for r in rows),
            "turns": turns, "sequences": sum(r.get("sequences", r["turns"]) for r in rows),
            "prompt_mismatch": sum(r["prompt_mismatch"] for r in rows),
            "logprob_gap": span("sampler_trainer_abs_logprob_gap"), "tis_weight_mean": span("tis_weight_mean"),
            "tis_capped_fraction_max": (span("tis_capped_fraction") or [0, 0])[1],
            "clip_fraction_max": (span("clip_fraction") or [0, 0])[1],
            "entropy_first_last": [round(rows[0]["entropy"], 4), round(rows[-1]["entropy"], 4)],
            "train_seconds": {"total": round(sum(train)), "median": round(statistics.median(train))},
            "rollout_seconds": {"total": round(sum(rollout)), "median": round(statistics.median(rollout))},
            "train_share": round(sum(train) / max(1e-9, sum(train) + sum(rollout)), 3)}


def pool_stats(pool: Path) -> dict:
    summary = json.loads((pool / "summary.json").read_text(encoding="utf-8"))
    attempts = jsonl(pool / "attempts.jsonl")
    edits: Counter = Counter()
    edits_ok: Counter = Counter()
    # Only recorded attempts count: a run directory missing from attempts.jsonl is an attempt that
    # was interrupted (a crash, a full disk) and its trajectory may be truncated.
    for path in (Path(row["result_path"]).parent / "trajectory.jsonl" for row in attempts if row.get("result_path")):
        if not path.exists():
            continue
        for event in jsonl(path):
            tool = event["action"]["tool"]
            if tool in {"edit_file", "replace_lines", "write_file"}:
                edits[tool] += 1
                edits_ok[tool] += bool(event["ok"])
    signals = Counter(signal for row in attempts for signal in row["patch_signals"])
    return {"splits": summary["splits"], "attempts_per_hour": summary["attempts_per_hour"],
            "wall_seconds": summary["wall_seconds"], "tokens": summary["tokens"],
            "evaluation_statuses": dict(Counter(row["evaluation_status"] for row in attempts)),
            "verifier_timeouts": sum(bool(row.get("verifier_timed_out")) for row in attempts),
            "edit_success": {tool: f"{edits_ok[tool]}/{edits[tool]}" for tool in edits},
            "patch_signals": dict(signals), "tamper_audit": tamper_audit(attempts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--price-per-hour", type=float, default=3.28)
    args = parser.parse_args()
    pools = {pool.name: pool_stats(pool) for pool in sorted(args.rollouts.iterdir()) if (pool / "summary.json").exists()}
    out = args.outputs
    sft_rows = jsonl(out / "sft.jsonl") if (out / "sft.jsonl").exists() else []
    result = out / "sft" / "result.json"
    training = json.loads(result.read_text(encoding="utf-8")) if result.exists() else None
    curve = jsonl(out / "sft" / "train_log.jsonl") if (out / "sft" / "train_log.jsonl").exists() else []
    compare = {path.stem.removeprefix("compare_"): json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(out.glob("compare_*.json"))}
    steps = jsonl(out / "steps.jsonl") if (out / "steps.jsonl").exists() else []
    grpo_rows = jsonl(out / "grpo" / "metrics.jsonl") if (out / "grpo" / "metrics.jsonl").exists() else []
    hours = sum(step["seconds"] for step in steps) / 3600
    report = {"pools": pools, "sft": {"examples": len(sft_rows), "tasks": len({r["task_id"] for r in sft_rows}),
                                      "by_mode": dict(Counter(r["mode"] for r in sft_rows))},
              "training": training, "loss_curve": [(c["step"], round(c["loss"], 4)) for c in curve],
              "grpo": grpo_summary(grpo_rows) if grpo_rows else None,
              "compare": compare, "stages": steps, "gpu_hours": round(hours, 2),
              "cost_cny": round(hours * args.price_per_hour, 2)}
    (out / "results.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["| pool | split | tasks | attempts | pass@1 | pass@k | solved tasks | statuses | evaluation | edit success "
             "| attempts/h |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, stats in pools.items():
        for split, entry in stats["splits"].items():
            ks = {k: v for k, v in entry.items() if k.startswith("pass@") and k != "pass@1"}
            lines.append(f"| {name} | {split} | {entry['tasks']} | {entry['attempts']} | {entry.get('pass@1')} | "
                         f"{ks} | {entry['solved_tasks']} | {entry['statuses']} | {stats['evaluation_statuses']} "
                         f"(timeouts {stats['verifier_timeouts']}) | {stats['edit_success']} | {stats['attempts_per_hour']} |")
    lines.append("")
    if sft_rows or training:
        lines += [f"SFT data: {report['sft']}", f"Training: {training}",
                  f"Loss (first, last): {report['loss_curve'][:1]} … {report['loss_curve'][-1:]}", ""]
    if report["grpo"]:
        g = report["grpo"]
        lines += [f"GRPO: {g['steps']} steps, {g['updates']} updates, {g['attempts']} attempts ({g['ungraded']} ungraded), "
                  f"{g['turns']} turns in {g['sequences']} sequences, prompt mismatches {g['prompt_mismatch']}, "
                  f"logprob gap {g['logprob_gap']}, TIS mean {g['tis_weight_mean']} (capped at most "
                  f"{g['tis_capped_fraction_max']}), clip at most {g['clip_fraction_max']}, entropy "
                  f"{g['entropy_first_last'][0]} → {g['entropy_first_last'][1]}, train share {g['train_share']}", ""]
    for name, value in compare.items():
        lines.append(f"Paired {name}: base {value['baseline_pass@1']:.3f} → candidate {value['candidate_pass@1']:.3f}, "
                     f"delta {value['delta']:+.3f}, 95% CI [{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}], {value['verdict']}")
    lines.append("")
    for name, stats in pools.items():
        audit = stats["tamper_audit"]
        graded = sum(any(s["graded"] for s in entry["signals"]) for entry in audit)
        lines.append(f"Tamper audit {name}: {len(audit)} resolved attempts with signals, {graded} in graded files")
        for entry in audit:
            for s in entry["signals"]:
                lines.append(f"- {entry['task_id']} s{entry['sample']}: {s['signal']} in {s['file']} "
                             f"({'graded' if s['graded'] else 'ignored'}) {s['line']!r}")
    lines += ["", f"GPU stage time: {report['gpu_hours']} h, cost ≈ ¥{report['cost_cny']}",
              "Stages: " + ", ".join(f"{s['step']} {s['seconds'] // 60} min" for s in steps)]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
