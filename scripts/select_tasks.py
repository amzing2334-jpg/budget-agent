"""Select tasks from a manifest by what earlier rollouts solved (no model calls).

    python scripts/select_tasks.py MANIFEST [--not-solved-in ATTEMPTS ...] [--solved-in ATTEMPTS ...]
        [--min-solved K] > tasks.jsonl

--not-solved-in drops every task resolved at least once in those attempt ledgers;
--solved-in keeps only tasks resolved at least K times there (summed over the ledgers).
The repo and verifier paths of each row are made absolute against the manifest's directory,
so the output can live anywhere. The number of tasks kept is printed to stderr.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys


def resolved_counts(ledgers: list[Path]) -> Counter:
    counts: Counter = Counter()
    for ledger in ledgers:
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                counts[row["task_id"]] += bool(row["resolved"])
    return counts


def select(manifest: Path, not_solved_in: list[Path], solved_in: list[Path], min_solved: int) -> list[dict]:
    base = manifest.resolve().parent
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    excluded = {task for task, n in resolved_counts(not_solved_in).items() if n > 0}
    solved = resolved_counts(solved_in)
    kept = [row for row in rows if row["id"] not in excluded and (not solved_in or solved[row["id"]] >= min_solved)]
    return [{**row, "repo": os.path.join(base, row["repo"]), "verifier": os.path.join(base, row["verifier"])}
            for row in kept]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--not-solved-in", type=Path, action="append", default=[])
    parser.add_argument("--solved-in", type=Path, action="append", default=[])
    parser.add_argument("--min-solved", type=int, default=1)
    args = parser.parse_args(argv)
    if args.min_solved < 1:
        parser.error("--min-solved must be at least 1")
    rows = select(args.manifest, args.not_solved_in, args.solved_in, args.min_solved)
    sys.stdout.writelines(json.dumps(row) + "\n" for row in rows)
    print(f"{len(rows)} tasks kept", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
