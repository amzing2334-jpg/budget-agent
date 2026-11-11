"""Count the commands that the harness limits stopped in rollout pools (no model calls).

    python scripts/limit_events.py POOL [POOL ...]

A pool is a directory with runs/<task>/<run>/trajectory.jsonl and result.json, as
written by rollout_pool.py, or a GRPO step directory with the same layout. For each
pool this counts:

- agent `run` and `public_test` commands stopped at the per-task time limit
  (the observation ends with the harness's "[command timed out after" marker);
- agent commands whose output shows MemoryError (the command memory cap);
- verifier runs graded unresolved because they timed out or ran out of memory.

Prints one JSON object per pool.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

TIMEOUT_MARKER = "[command timed out after"
COMMAND_TOOLS = {"run", "public_test"}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pool_limits(pool: Path) -> dict:
    commands: Counter = Counter()
    timed_out: Counter = Counter()
    memory: Counter = Counter()
    attempts = verifier_timeouts = verifier_memory = 0
    for run in sorted(path.parent for path in pool.glob("runs/*/*/result.json")):
        attempts += 1
        trajectory = run / "trajectory.jsonl"
        for event in jsonl(trajectory) if trajectory.exists() else []:
            tool = event["action"].get("tool")
            if tool not in COMMAND_TOOLS:
                continue
            commands[tool] += 1
            output = event.get("output") or ""
            timed_out[tool] += TIMEOUT_MARKER in output
            memory[tool] += "MemoryError" in output
        evaluation = json.loads((run / "result.json").read_text(encoding="utf-8")).get("evaluation") or {}
        verifier_timeouts += bool(evaluation.get("timed_out"))
        verifier_memory += bool(evaluation.get("memory_exceeded"))
    return {"pool": str(pool), "attempts": attempts, "commands": dict(commands),
            "timed_out": dict(timed_out), "memory_error": dict(memory),
            "verifier_timed_out": verifier_timeouts, "verifier_memory_exceeded": verifier_memory}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pools", nargs="+", type=Path)
    args = parser.parse_args()
    for pool in args.pools:
        print(json.dumps(pool_limits(pool)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
