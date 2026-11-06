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


def tamper_audit(attempts: list[dict]) -> list[dict]:
    """Resolved attempts with patch signals, each signal marked graded or ignored by the grader."""
    audit = []
    for row in attempts:
        if not (row["resolved"] and row["patch_signals"]):
            continue
        ignored = set(row["ignored_changes"])
        records = json.loads(Path(row["result_path"]).read_text())["patch_signals"]
        audit.append({"task_id": row["task_id"], "sample": row["sample"],
                      "signals": [{**record, "graded": record["file"] not in ignored} for record in records]})
    return audit


def pool_stats(pool: Path) -> dict:
    summary = json.loads((pool / "summary.json").read_text())
    attempts = [json.loads(line) for line in (pool / "attempts.jsonl").read_text().splitlines() if line.strip()]
    edits: Counter = Counter()
    edits_ok: Counter = Counter()
    for path in pool.glob("runs/*/*/trajectory.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            tool = event["action"]["tool"]
            if tool in {"edit_file", "replace_lines", "write_file"}:
                edits[tool] += 1
                edits_ok[tool] += bool(event["ok"])
    signals = Counter(signal for row in attempts for signal in row["patch_signals"])
    return {"splits": summary["splits"], "attempts_per_hour": summary["attempts_per_hour"],
            "wall_seconds": summary["wall_seconds"], "tokens": summary["tokens"],
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
    sft_rows = [json.loads(line) for line in (out / "sft.jsonl").read_text().splitlines() if line.strip()] if (out / "sft.jsonl").exists() else []
    training = json.loads((out / "sft" / "result.json").read_text()) if (out / "sft" / "result.json").exists() else None
    curve = ([json.loads(line) for line in (out / "sft" / "train_log.jsonl").read_text().splitlines() if line.strip()]
             if (out / "sft" / "train_log.jsonl").exists() else [])
    compare = {path.stem.removeprefix("compare_"): json.loads(path.read_text()) for path in sorted(out.glob("compare_*.json"))}
    steps = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines() if line.strip()] if (out / "steps.jsonl").exists() else []
    hours = sum(step["seconds"] for step in steps) / 3600
    report = {"pools": pools, "sft": {"examples": len(sft_rows), "tasks": len({r["task_id"] for r in sft_rows}),
                                      "by_mode": dict(Counter(r["mode"] for r in sft_rows))},
              "training": training, "loss_curve": [(c["step"], round(c["loss"], 4)) for c in curve],
              "compare": compare, "stages": steps, "gpu_hours": round(hours, 2),
              "cost_cny": round(hours * args.price_per_hour, 2)}
    (out / "results.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    lines = ["| pool | split | tasks | attempts | pass@1 | pass@k | solved tasks | statuses | edit success | attempts/h |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, stats in pools.items():
        for split, entry in stats["splits"].items():
            ks = {k: v for k, v in entry.items() if k.startswith("pass@") and k != "pass@1"}
            lines.append(f"| {name} | {split} | {entry['tasks']} | {entry['attempts']} | {entry.get('pass@1')} | "
                         f"{ks} | {entry['solved_tasks']} | {entry['statuses']} | {stats['edit_success']} | {stats['attempts_per_hour']} |")
    lines += ["", f"SFT data: {report['sft']}", f"Training: {training}",
              f"Loss (first, last): {report['loss_curve'][:1]} … {report['loss_curve'][-1:]}", ""]
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
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
