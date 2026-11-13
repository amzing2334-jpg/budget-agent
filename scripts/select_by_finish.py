"""Pick one attempt per task by the agent's own finish signal, against a random pick (no model calls).

    python scripts/select_by_finish.py NAME=POOL/attempts.jsonl [...] [--resamples 10000] [--seed 0]

docs/ANALYSIS_FINISH_SELECTOR.md fixes the rule. Per task, over graded attempts only: a random
pick succeeds with the task's pass rate (pass@1, as in compare_pools.py); the selector picks
uniformly among the attempts whose status is "finished", or among all of them when none finished,
so its expected success is the fraction resolved in that set. Per pool it reports both means,
their difference with a percentile bootstrap over tasks (as compare_pools.py does), the share of
tasks any attempt solved (pass@k), the tasks with no finished attempt, and the precision of
finished attempts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def per_task(path: Path, split: str = "eval") -> dict[str, list[tuple[bool, bool]]]:
    """task -> [(finished, resolved)] over graded attempts of the split."""
    tasks: dict[str, list[tuple[bool, bool]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["split"] != split or row["resolved"] is None:
            continue
        tasks.setdefault(row["task_id"], []).append((row["status"] == "finished", bool(row["resolved"])))
    return tasks


def bootstrap(deltas: list[float], resamples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(resamples))
    return boots[int(0.025 * resamples)], boots[int(0.975 * resamples) - 1]


def evaluate(name: str, path: Path, resamples: int = 10_000, seed: int = 0) -> dict:
    tasks = per_task(path)
    random_pick, selector, deltas = [], [], []
    solved = without_finish = finished_attempts = finished_resolved = 0
    for task in sorted(tasks):   # the order compare_pools.py resamples in, so seeded intervals agree
        rows = tasks[task]
        rate = sum(resolved for _, resolved in rows) / len(rows)
        finished = [resolved for done, resolved in rows if done]
        pick = sum(finished) / len(finished) if finished else rate
        random_pick.append(rate)
        selector.append(pick)
        deltas.append(pick - rate)
        solved += any(resolved for _, resolved in rows)
        without_finish += not finished
        finished_attempts += len(finished)
        finished_resolved += sum(finished)
    low, high = bootstrap(deltas, resamples, seed)
    n = len(tasks)
    return {"pool": name, "tasks": n, "pass@1": sum(random_pick) / n, "selector": sum(selector) / n,
            "delta": sum(deltas) / n, "ci95": [low, high],
            "verdict": "improved" if low > 0 else "worse" if high < 0 else "no detectable change",
            "tasks_better": sum(d > 0 for d in deltas), "tasks_worse": sum(d < 0 for d in deltas),
            "pass@k": solved / n, "tasks_without_finish": without_finish,
            "finish_precision": finished_resolved / finished_attempts if finished_attempts else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pools", nargs="+", help="NAME=path/to/attempts.jsonl")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if any("=" not in item for item in args.pools):
        parser.error("pools are NAME=path")
    report = [evaluate(name, Path(path), args.resamples, args.seed)
              for name, path in (item.split("=", 1) for item in args.pools)]
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
