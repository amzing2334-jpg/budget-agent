"""Rollout throughput versus concurrency on a fixed task subset (GPU host).

For each worker count, runs one attempt per task through the rollout pool
(as the unprivileged user), samples GPU utilization every second, and diffs
vLLM's Prometheus counters around the run. Reports attempts per hour,
generation and prompt tokens per second, prefix-cache hit rate, mean GPU
utilization, attempt latency percentiles, and the share of attempt time spent
outside model calls (tools, tests, grading). Writes bench.json and bench.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import threading
import time

from vllm_metrics import metrics  # scripts/ is on sys.path when this file runs as a script


def sample_gpu(stop: threading.Event, samples: list[float]) -> None:
    while not stop.is_set():
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        if out:
            samples.append(float(out[0]))
        stop.wait(1.0)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--workers", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--output", type=Path, required=True, help="directory writable by the rollout user")
    parser.add_argument("--launcher", default="/opt/coding-agent/budget-agent/scripts/autodl/run_pool.sh")
    parser.add_argument("--server", default="http://127.0.0.1:18017")
    parser.add_argument("--model", default="teacher")
    args = parser.parse_args()
    rows = []
    for workers in args.workers:
        out = args.output / f"workers-{workers:03d}"
        before, samples, stop = metrics(args.server), [], threading.Event()
        sampler = threading.Thread(target=sample_gpu, args=(stop, samples), daemon=True)
        sampler.start()
        started = time.monotonic()
        subprocess.run([args.launcher, "--config", str(args.config), "--tasks", str(args.tasks), "--split", "all",
                        "--limit", str(args.limit), "--samples", "1", "--workers", str(workers), "--output", str(out),
                        "--set", f"model={args.model}"], check=True)
        wall = time.monotonic() - started
        stop.set()
        sampler.join()
        after = metrics(args.server)
        delta = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in after}
        attempts = [json.loads(line) for line in (out / "attempts.jsonl").read_text().splitlines() if line.strip()]
        latency = [a["elapsed_seconds"] for a in attempts]
        outside_model = sum(a["elapsed_seconds"] - a["model_seconds"] for a in attempts) / max(1e-9, sum(latency))
        queries = delta.get("vllm:prefix_cache_queries") or delta.get("vllm:gpu_prefix_cache_queries")
        hits = delta.get("vllm:prefix_cache_hits") or delta.get("vllm:gpu_prefix_cache_hits")
        rows.append({"workers": workers, "attempts": len(attempts), "wall_seconds": round(wall, 1),
                     "attempts_per_hour": round(len(attempts) / wall * 3600, 1),
                     "generation_tokens_per_s": round(delta.get("vllm:generation_tokens", 0.0) / wall, 1),
                     "prompt_tokens_per_s": round(delta.get("vllm:prompt_tokens", 0.0) / wall, 1),
                     "prefix_cache_hit_rate": round(hits / queries, 3) if queries else None,
                     "gpu_util_mean": round(sum(samples) / len(samples), 1) if samples else None,
                     "latency_p50_s": percentile(latency, 0.5), "latency_p95_s": percentile(latency, 0.95),
                     "share_outside_model": round(outside_model, 3),
                     "resolved": sum(bool(a["resolved"]) for a in attempts)})
        print(json.dumps(rows[-1]), flush=True)
    (args.output / "bench.json").write_text(json.dumps(rows, indent=1))
    header = "| workers | attempts/h | gen tok/s | prompt tok/s | prefix hit | GPU util % | p50 s | p95 s | time outside model |"
    lines = [header, "|" + "---|" * 9]
    for r in rows:
        lines.append(f"| {r['workers']} | {r['attempts_per_hour']} | {r['generation_tokens_per_s']} | {r['prompt_tokens_per_s']} | "
                     f"{r['prefix_cache_hit_rate']} | {r['gpu_util_mean']} | {r['latency_p50_s']} | {r['latency_p95_s']} | "
                     f"{r['share_outside_model']} |")
    (args.output / "bench.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
