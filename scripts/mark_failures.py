"""Rewrite a pool's attempts with chosen passing patches counted as failures (no model calls).

The check against the pre-bug source (review_patches.py --original, judged by hand
into original_check.json) classifies passing patches that did not restore the
original behaviour as "mild" or "strong". This writes the pool's attempts.jsonl to
stdout with those attempts marked unresolved, for a sensitivity comparison:

    python scripts/mark_failures.py POOL/attempts.jsonl docs/data/round3/original_check.json \\
        --pool eval-grpo --kinds strong > grpo_strong.jsonl
    python scripts/compare_pools.py --baseline sft_strong.jsonl --candidate grpo_strong.jsonl --split eval

Every listed patch must match a resolved attempt in the file, so a wrong pool name
or a stale list fails loudly instead of changing nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

KINDS = ("mild", "strong")


def mark(rows: list[dict], weak: list[dict], pool: str, kinds: set[str]) -> tuple[list[dict], int]:
    chosen = {(w["task_id"], w["sample"]) for w in weak if w["pool"] == pool and w["kind"] in kinds}
    resolved = {(r["task_id"], r["sample"]) for r in rows if r["resolved"]}
    missing = chosen - resolved
    if missing:
        raise ValueError(f"{len(missing)} listed patches are not resolved attempts of this pool: {sorted(missing)[:3]}")
    out = [{**r, "resolved": False} if (r["task_id"], r["sample"]) in chosen else r for r in rows]
    return out, len(chosen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("attempts", type=Path)
    parser.add_argument("original_check", type=Path, help="JSON with a weak_patches list (pool, task_id, sample, kind)")
    parser.add_argument("--pool", required=True, help="pool name as written in the weak_patches list")
    parser.add_argument("--kinds", default="strong", help="comma-separated: strong, mild")
    args = parser.parse_args(argv)
    kinds = set(args.kinds.split(","))
    if not kinds <= set(KINDS):
        parser.error(f"--kinds takes {', '.join(KINDS)}")
    rows = [json.loads(line) for line in args.attempts.read_text(encoding="utf-8").splitlines() if line.strip()]
    weak = json.loads(args.original_check.read_text(encoding="utf-8"))["weak_patches"]
    out, count = mark(rows, weak, args.pool, kinds)
    sys.stdout.write("".join(json.dumps(row) + "\n" for row in out))
    print(f"marked {count} attempts unresolved", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
