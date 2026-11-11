"""Snapshot vLLM's Prometheus counters, or summarize the change between two snapshots.

    python scripts/vllm_metrics.py > before.json            # snapshot
    python scripts/vllm_metrics.py --diff before.json after.json

The summary gives prompt and generated tokens, prefix-cache queries and hits,
and the hit rate over the interval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import urllib.request


def metrics(server: str) -> dict[str, float]:
    text = urllib.request.urlopen(f"{server}/metrics", timeout=10).read().decode()
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = re.match(r"^(vllm:[a-z_]+?)(?:_total)?(?:\{[^}]*\})? ([0-9.eE+-]+)$", line)
        if match:
            totals[match[1]] = totals.get(match[1], 0.0) + float(match[2])
    return totals


def summarize(before: dict[str, float], after: dict[str, float]) -> dict:
    delta = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in after}
    queries = delta.get("vllm:prefix_cache_queries") or delta.get("vllm:gpu_prefix_cache_queries") or 0.0
    hits = delta.get("vllm:prefix_cache_hits") or delta.get("vllm:gpu_prefix_cache_hits") or 0.0
    return {"prompt_tokens": delta.get("vllm:prompt_tokens", 0.0),
            "generation_tokens": delta.get("vllm:generation_tokens", 0.0),
            "prefix_cache_queries": queries, "prefix_cache_hits": hits,
            "prefix_cache_hit_rate": round(hits / queries, 4) if queries else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http://127.0.0.1:18017")
    parser.add_argument("--diff", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()
    if args.diff:
        before, after = (json.loads(path.read_text(encoding="utf-8")) for path in args.diff)
        print(json.dumps(summarize(before, after), indent=1))
    else:
        print(json.dumps(metrics(args.server)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
