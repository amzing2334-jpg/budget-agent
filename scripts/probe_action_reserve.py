"""One cloud-only development pair for action-output reservation, not training."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from budget_agent.cli import load_config
from budget_agent.runner import run_task, source_fingerprint
from budget_agent.tasks import load_tasks


ROOT = Path(__file__).resolve().parents[1]
RESERVES = (1, 128)
TERMINAL_STATUSES = {"finished", "max_steps", "budget_exhausted", "context_exhausted"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_sha(task) -> str:
    return hashlib.sha256(json.dumps(asdict(task), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require_cloud_cuda() -> None:
    if sys.platform != "linux":
        raise ValueError("Execution requires the authorized Linux CUDA server; no local inference")
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; the probe will not fall back to CPU")


def run_probe(config_path: Path, tasks_path: Path, task_id: str, output: Path, *, execute: bool = False):
    config = load_config(SimpleNamespace(config=config_path))
    if (config.provider, config.strategy, config.history_format, config.environment, config.device) != ("transformers", "recent", "chat", "docker", "cuda"):
        raise ValueError("Probe requires transformers, recent, chat, docker and cuda")
    if config.max_output_tokens < max(RESERVES):
        raise ValueError("max_output_tokens must accommodate the 128-token reserve")
    task, = load_tasks(tasks_path, split="dev", task_ids=[task_id])
    if task.kind != "swebench":
        raise ValueError("Probe requires a sanitized SWE-bench development task")
    config = replace(config, tokenizer=f"hf:{config.model}@{config.model_revision}")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    plan = {
        "kind": "development_action_reserve_diagnostic", "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": task.id, "split": task.split, "task_sha256": _task_sha(task),
        "tasks_path": str(tasks_path.resolve()), "tasks_sha256": _sha(tasks_path),
        "config_path": str(config_path.resolve()), "config_sha256": _sha(config_path),
        "source_fingerprint": source_fingerprint(), "driver_sha256": _sha(Path(__file__)),
        "common_config": {key: value for key, value in asdict(config).items() if key != "min_action_output_tokens"},
        "attempts": [{"min_action_output_tokens": reserve, "output": str(output / f"reserve-{reserve}")} for reserve in RESERVES],
        "attempts_per_reserve": 1,
        "notes": [
            "The reserve is an admission/context budget requirement, not forced generation or a guarantee of a complete action.",
            "One known development task, one attempt per condition; this is not a benchmark success rate.",
            "Only min_action_output_tokens differs. Upstream mini-SWE-agent is a separate external-harness run.",
            "Official export, evaluation and attachment run separately; a finished run or nonempty patch is not repair success.",
        ],
    }
    print(json.dumps({"execute": execute, "plan": plan}, ensure_ascii=False, indent=2), flush=True)
    if not execute:
        return plan
    _require_cloud_cuda()
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "plan.json", plan)
    # This run uses prepared assets only. Missing weights must fail, not download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from budget_agent.hf_model import TransformersModel
    model = TransformersModel(config)
    records, first_condition = [], None
    for reserve in RESERVES:
        current = {"source_fingerprint": source_fingerprint(), "config_sha256": _sha(config_path),
                   "tasks_sha256": _sha(tasks_path), "task_sha256": _task_sha(task), "driver_sha256": _sha(Path(__file__))}
        if any(value != plan[key] for key, value in current.items()):
            raise ValueError("Frozen source/config/task/driver changed; no further attempt will run")
        settings = replace(config, min_action_output_tokens=reserve)
        result = run_task(task, settings, output / f"reserve-{reserve}", model=model, counter=model.counter)
        records.append({"min_action_output_tokens": reserve, "status": result["status"], "result_path": result["result_path"]})
        _write(output / "attempts.json", records)
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
        if result["status"] == "interrupted":
            raise KeyboardInterrupt("Probe interrupted; remaining attempts were not started")
        if (result["status"] not in TERMINAL_STATUSES or result["usage_unknown"]
                or result["budget"]["overrun"] or result["budget"]["unreconciled_request"]
                or result["evaluation"]["status"] in {"infrastructure_error", "grading_error"}
                or result["evaluation"].get("cleanup_error")):
            raise ValueError("Attempt ended with an execution/accounting error; no further attempt will run")
        if result["source_fingerprint"] != plan["source_fingerprint"] or result["config"] != asdict(settings):
            raise ValueError("Run does not match the frozen source/config")
        condition = {key: result[key] for key in ("task_id", "task_fingerprint", "initial_revision", "environment_provenance", "model", "model_metadata")}
        if first_condition is not None and condition != first_condition:
            raise ValueError("Model/task/environment differs between attempts; do not compare this pair")
        first_condition = condition
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cloud-v5-qwen7b.toml")
    parser.add_argument("--tasks", type=Path, default=ROOT / "tasks/swebench-pilot.jsonl")
    parser.add_argument("--task", default="astropy__astropy-12907")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Run the two predeclared attempts on the authorized Linux CUDA server")
    args = parser.parse_args(argv)
    run_probe(args.config, args.tasks, args.task, args.output, execute=args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
