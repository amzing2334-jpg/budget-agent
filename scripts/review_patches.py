"""Print every resolved attempt's patch in a rollout pool, for the manual tamper review.

The frozen protocols ask for each passing patch to be read: a test edited, the
grading channel touched, a forged exit, or a fix that passes the tests by a
different mechanism than the bug's (for example `if False:`). For each resolved
attempt this prints the task, sample, final status, files touched, added and
removed line counts, files the grader ignored, patch signals, and the changed
lines themselves.

    python scripts/review_patches.py /opt/coding-agent/work/round2/grpo-eval
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> int:
    pool = Path(sys.argv[1])
    rows = [json.loads(line) for line in (pool / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    winners = sorted((r for r in rows if r["resolved"]), key=lambda r: (r["task_id"], r["sample"]))
    print(f"{pool.name}: {len(winners)} resolved of {len(rows)}")
    for row in winners:
        patch = (Path(row["result_path"]).parent / "patch.diff").read_text(encoding="utf-8", errors="replace")
        lines = patch.splitlines()
        files = [line[6:] for line in lines if line.startswith("+++ b/")]
        added = sum(line.startswith("+") and not line.startswith("+++") for line in lines)
        removed = sum(line.startswith("-") and not line.startswith("---") for line in lines)
        print(f"\n### {row['task_id']} s{row['sample']} status={row['status']} steps={row['steps']} files={files} "
              f"+{added}/-{removed} ignored={row['ignored_changes']} signals={row['patch_signals']}")
        print("\n".join(line for line in lines if line.startswith(("+", "-", "@@")) and not line.startswith(("+++", "---"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
