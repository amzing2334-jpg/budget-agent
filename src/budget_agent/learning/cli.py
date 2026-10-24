"""Optional learning CLI; importing the v1 evaluator does not load PyTorch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

from .schema import TrainingConfig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="budget-agent-learn")
    sub = parser.add_subparsers(dest="command", required=True)
    curriculum = sub.add_parser("curriculum", help="Generate disjoint synthetic train/eval tasks")
    curriculum.add_argument("--output", type=Path, default=Path("data/curriculum"))
    curriculum.add_argument("--train-count", type=int, default=12)
    curriculum.add_argument("--eval-count", type=int, default=6)
    for name in ("sft", "train", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/train-smoke.toml"))
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--adapter", type=Path)
        if name == "sft":
            command.add_argument("--data", type=Path, required=True)
            command.add_argument("--steps", type=int, default=60)
        else:
            command.add_argument("--tasks", type=Path, required=True)
            command.add_argument("--task", action="append", dest="task_ids")
    args = parser.parse_args(argv)
    if args.command == "curriculum":
        from .curriculum import generate_curriculum
        result = generate_curriculum(args.output, args.train_count, args.eval_count)
    else:
        from .policy import HFPolicy
        from .trainer import evaluate, seed_everything, train_grpo, train_sft
        from ..tasks import load_tasks
        with args.config.open("rb") as stream:
            config = TrainingConfig(**tomllib.load(stream)["train"])
        if args.adapter:
            config.adapter_path = str(args.adapter.resolve())
        config.validate()
        seed_everything(config.seed)
        policy = HFPolicy(config.model, revision=config.revision, device=config.device,
                          lora_r=config.lora_r, temperature=config.temperature,
                          max_context_tokens=config.context_tokens, adapter_path=config.adapter_path)
        if args.command == "sft":
            if args.steps < 1:
                parser.error("--steps must be positive")
            result = train_sft(policy, args.data, config, args.output, args.steps)
        else:
            tasks = load_tasks(args.tasks, "train" if args.command == "train" else "eval", args.task_ids)
            if any(task.kind != "local" for task in tasks):
                parser.error("Learning currently supports local fixture manifests; use v1 for SWE-bench")
            result = (train_grpo if args.command == "train" else evaluate)(policy, tasks, config, args.output)
    print(json.dumps(result, default=str, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
