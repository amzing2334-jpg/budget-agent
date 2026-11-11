"""Turn independently graded successful attempts into chat SFT samples.

Rejection-sampling fine-tuning data: only attempts whose final workspace
passed the private verifier are kept (at most --per-task per task, so easy
tasks do not dominate). Each attempt's calls.jsonl holds the exact model
inputs and replies. When every agent request extends the previous one (the
`full` strategy without compaction), the whole attempt becomes one sample
whose assistant turns are all trained; otherwise each request becomes a
sample that trains only its final reply. Summary and reflection calls are
never trained.

--revision-labels relabel|fixed rebuilds every request in that revision-label mode
instead of using the recorded inputs (budget_agent.learning.rerender): the run is
first rebuilt in its own mode and must reproduce the recorded requests exactly. The
replies stay the recorded ones, so this re-renders the same trajectories in another
context format without new rollouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def agent_calls(run_dir: Path, labels: str = "recorded") -> list[dict]:
    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if labels != "recorded":
        from budget_agent.learning.rerender import rerender_requests
        calls = [{**call, "messages": messages}
                 for call, messages in zip(calls, rerender_requests(run_dir, labels), strict=True)]
    return [call for call in calls if call.get("purpose") == "agent" and call.get("response") is not None]


def samples_for(run_dir: Path, labels: str = "recorded") -> tuple[str, list[list[dict]]]:
    calls = agent_calls(run_dir, labels)
    turns = [call["messages"] + [{"role": "assistant", "content": call["response"]["content"]}] for call in calls]
    final = turns[-1]
    if all(final[:len(turn)] == turn for turn in turns):
        return "full", [final]
    return "last", turns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attempts", type=Path, action="append", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--per-task", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-flagged", action="store_true",
                        help="keep winners whose patch tripped a tamper signal (excluded by default)")
    parser.add_argument("--revision-labels", choices=["recorded", "relabel", "fixed"], default="recorded",
                        help="use the recorded requests, or rebuild them in this revision-label mode")
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.attempts for line in path.read_text().splitlines() if line.strip()]
    winners: dict[str, list[dict]] = {}
    flagged = 0
    for row in rows:
        if row["resolved"] is True and row["split"] == args.split:
            if row.get("patch_signals") and not args.keep_flagged:
                flagged += 1  # never teach a policy to tamper with grading
                continue
            winners.setdefault(row["task_id"], []).append(row)
    rng = random.Random(args.seed)
    written = {"full": 0, "last": 0}
    examples = []
    for task_id in sorted(winners):
        chosen = sorted(winners[task_id], key=lambda row: row["sample"])
        rng.shuffle(chosen)
        for row in chosen[:args.per_task]:
            mode, conversations = samples_for(Path(row["result_path"]).parent, args.revision_labels)
            for messages in conversations:
                examples.append({"task_id": task_id, "sample": row["sample"], "mode": mode,
                                 "source": row["result_path"], "messages": messages})
                written[mode] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples), encoding="utf-8")
    print(json.dumps({"tasks_with_success": len(winners), "examples": len(examples), "by_mode": written,
                      "attempts_seen": len(rows), "flagged_winners_excluded": flagged,
                      "revision_labels": args.revision_labels}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
