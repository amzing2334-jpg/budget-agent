"""Run K harness attempts per task concurrently against an OpenAI-compatible server.

Each attempt is an ordinary `run_task` call (same tools, budget ledger and
independent grading as single runs), so evaluation rollouts and training data
come from one code path. Threads are enough: an attempt waits on HTTP or a
subprocess almost all the time, and the server batches concurrent requests.

Writes <output>/attempts.jsonl (one row per attempt, appended as attempts
finish, so an interrupted pool can be resumed with --resume) and summary.json
with pass@1, unbiased pass@k per split and throughput.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from math import comb
from pathlib import Path
import threading
import time
import tomllib

from budget_agent.runner import run_task
from budget_agent.schema import RunConfig
from budget_agent.tasks import load_tasks


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator from n samples with c successes (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def summarize(rows: list[dict], k_values: list[int]) -> dict:
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    summary = {}
    for split in sorted({row["split"] for row in rows}):
        tasks = {task: attempts for task, attempts in by_task.items() if attempts[0]["split"] == split}
        graded = {task: [a for a in attempts if a["resolved"] is not None] for task, attempts in tasks.items()}
        entry = {"tasks": len(tasks), "attempts": sum(len(a) for a in tasks.values()),
                 "ungraded_attempts": sum(len(tasks[t]) - len(graded[t]) for t in tasks),
                 "solved_tasks": sum(any(a["resolved"] for a in g) for g in graded.values())}
        for k in k_values:
            usable = [g for g in graded.values() if len(g) >= k]
            entry[f"pass@{k}"] = (sum(pass_at_k(len(g), sum(bool(a["resolved"]) for a in g), k) for g in usable)
                                  / len(usable)) if usable else None
        statuses: dict[str, int] = {}
        for attempts in tasks.values():
            for attempt in attempts:
                statuses[attempt["status"]] = statuses.get(attempt["status"], 0) + 1
        entry["statuses"] = statuses
        entry["mean_steps"] = sum(a["steps"] for a in sum(tasks.values(), [])) / max(1, entry["attempts"])
        summary[split] = entry
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="TOML with a [run] table")
    parser.add_argument("--tasks", type=Path, action="append", required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, help="first N tasks per manifest (smoke runs)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a [run] field, e.g. --set model=student --set temperature=0.7")
    args = parser.parse_args()
    values = tomllib.loads(args.config.read_text(encoding="utf-8"))["run"]
    for item in args.set:
        key, _, raw = item.partition("=")
        try:
            values[key] = json.loads(raw)  # numbers, booleans, null
        except json.JSONDecodeError:
            values[key] = raw
    config = RunConfig(**values)
    config.validate()
    tasks = []
    for manifest in args.tasks:
        selected = load_tasks(manifest, args.split)
        tasks.extend(selected[:args.limit] if args.limit else selected)
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = args.output / "attempts.jsonl"
    done = set()
    rows: list[dict] = []
    if args.resume and ledger.exists():
        rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        done = {(row["task_id"], row["sample"]) for row in rows}
    jobs = [(task, sample) for sample in range(args.samples) for task in tasks if (task.id, sample) not in done]
    lock = threading.Lock()
    started = time.monotonic()

    def attempt(task, sample):
        result = run_task(task, RunConfig(**{**values, "seed": sample}), args.output / "runs" / f"s{sample:02d}")
        row = {"task_id": task.id, "split": task.split, "sample": sample, "status": result["status"],
               "resolved": result["resolved"], "evaluation_status": result["evaluation"].get("status"),
               "steps": result["steps"], "input_tokens": result["input_tokens"],
               "output_tokens": result["output_tokens"], "elapsed_seconds": round(result["elapsed_seconds"], 2),
               "model_seconds": round(result["model_seconds"], 2), "result_path": result["result_path"],
               "ignored_changes": result["evaluation"].get("ignored_changes", []),
               "patch_signals": sorted({s["signal"] for s in result.get("patch_signals", [])})}
        with lock:
            rows.append(row)
            with ledger.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        return row

    finished = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(attempt, task, sample) for task, sample in jobs]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            finished += 1
            if finished % 10 == 0 or finished == len(jobs):
                rate = finished / (time.monotonic() - started) * 3600
                print(f"[{finished}/{len(jobs)}] {rate:.0f} attempts/h; last {row['task_id']} "
                      f"{row['status']} resolved={row['resolved']}", flush=True)
    elapsed = time.monotonic() - started
    summary = {"config": values, "samples": args.samples, "workers": args.workers,
               "wall_seconds": round(elapsed, 1), "new_attempts": len(jobs),
               "attempts_per_hour": round(len(jobs) / elapsed * 3600, 1) if jobs and elapsed else None,
               "tokens": {"input": sum(r["input_tokens"] for r in rows), "output": sum(r["output_tokens"] for r in rows)},
               "splits": summarize(rows, [k for k in (1, 2, 4, 8) if k <= args.samples])}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["splits"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
