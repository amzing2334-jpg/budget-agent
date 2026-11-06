"""Paired comparison of two rollout pools on the same tasks.

Per task, pass@1 is the fraction of graded attempts that resolved it. The
difference (candidate minus baseline) is averaged over tasks present in
both pools, with a percentile bootstrap over tasks for a 95% interval.
Ungraded attempts are excluded and counted, never scored as failures.
With --drop-flagged, resolved attempts whose patch tripped a tamper signal
(the protocol's suspicious samples) are dropped from both pools and counted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def per_task(path: Path, split: str, drop_flagged: bool = False) -> tuple[dict[str, float], int, int]:
    attempts: dict[str, list[bool]] = {}
    ungraded = dropped = 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["split"] != split:
            continue
        if row["resolved"] is None:
            ungraded += 1
            continue
        if drop_flagged and row["resolved"] and row["patch_signals"]:
            dropped += 1
            continue
        attempts.setdefault(row["task_id"], []).append(bool(row["resolved"]))
    return {task: sum(v) / len(v) for task, v in attempts.items()}, ungraded, dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drop-flagged", action="store_true",
                        help="drop resolved attempts with patch tamper signals from both pools")
    args = parser.parse_args()
    base, base_ungraded, base_dropped = per_task(args.baseline, args.split, args.drop_flagged)
    cand, cand_ungraded, cand_dropped = per_task(args.candidate, args.split, args.drop_flagged)
    tasks = sorted(base.keys() & cand.keys())
    deltas = [cand[t] - base[t] for t in tasks]
    rng = random.Random(args.seed)
    boots = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(args.resamples))
    low, high = boots[int(0.025 * args.resamples)], boots[int(0.975 * args.resamples) - 1]
    verdict = "improved" if low > 0 else "worse" if high < 0 else "no detectable change"
    report = {"split": args.split, "paired_tasks": len(tasks),
              "baseline_pass@1": sum(base[t] for t in tasks) / len(tasks),
              "candidate_pass@1": sum(cand[t] for t in tasks) / len(tasks),
              "delta": sum(deltas) / len(deltas), "ci95": [low, high], "verdict": verdict,
              "tasks_improved": sum(d > 0 for d in deltas), "tasks_worse": sum(d < 0 for d in deltas),
              "ungraded_attempts": {"baseline": base_ungraded, "candidate": cand_ungraded},
              "dropped_flagged": {"baseline": base_dropped, "candidate": cand_dropped} if args.drop_flagged else None,
              "unpaired_tasks": sorted(base.keys() ^ cand.keys())}
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
