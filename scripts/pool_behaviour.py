"""Describe how a policy behaved in rollout pools (no model calls).

    python scripts/pool_behaviour.py POOL [POOL ...] > behaviour.json

For each pool (a directory with attempts.jsonl, and run directories with trajectory.jsonl):
pass@1 over attempts, solved tasks, final statuses, how often a voluntary finish was a real
fix, passes per repository, per-attempt means of steps, input and output tokens, wall and
model seconds, and how often each tool was called. These are the descriptive behaviour
rows in the experiment records; only recorded attempts count (an interrupted attempt's run
directory is ignored).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def repository(task_id: str) -> str:
    """SWE-smith ids look like owner__repo.commit.bug; anything else is its own repository."""
    head = task_id.split(".")[0]
    return head.split("__", 1)[1] if "__" in head else head


def describe(pool: Path) -> dict:
    if (pool / "PACKED.json").exists() and not (pool / "runs").exists():   # scripts/pack_rollouts.py
        raise SystemExit(f"{pool} is packed; restore it first: tar -xzf {pool / 'runs.tar.gz'} -C {pool}")
    rows = jsonl(pool / "attempts.jsonl")
    n = max(1, len(rows))
    finished = [r for r in rows if r["status"] == "finished"]
    per_repo: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        per_repo[repository(row["task_id"])][0] += bool(row["resolved"])
        per_repo[repository(row["task_id"])][1] += 1
    tools: Counter = Counter()
    for row in rows:
        trajectory = Path(row["result_path"]).parent / "trajectory.jsonl" if row.get("result_path") else None
        if trajectory is not None and trajectory.exists():
            tools.update(event["action"].get("tool") for event in jsonl(trajectory))
    mean = lambda key: round(sum(r.get(key, 0) or 0 for r in rows) / n, 1)  # noqa: E731
    return {"pool": str(pool), "attempts": len(rows), "resolved": sum(bool(r["resolved"]) for r in rows),
            "ungraded": sum(r["resolved"] is None for r in rows),
            "pass@1": round(sum(bool(r["resolved"]) for r in rows) / n, 4),
            "solved_tasks": len({r["task_id"] for r in rows if r["resolved"]}),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "finished_and_resolved": [sum(bool(r["resolved"]) for r in finished), len(finished)],
            "per_repository": {repo: f"{a}/{b}" for repo, (a, b) in sorted(per_repo.items())},
            "per_attempt": {"steps": mean("steps"), "input_tokens": mean("input_tokens"),
                            "output_tokens": mean("output_tokens"), "elapsed_seconds": mean("elapsed_seconds"),
                            "model_seconds": mean("model_seconds")},
            "tool_calls": dict(tools.most_common())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pools", nargs="+", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps([describe(pool) for pool in args.pools], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
