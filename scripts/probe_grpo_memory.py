"""Peak GPU memory of one GRPO update on the longest allowed sequence (run on the GPU host).

    python scripts/probe_grpo_memory.py --model MODEL [--max-len 32768] [--completion 8192] [--rank 32]

Loads the model as scripts/grpo_harness.py does (bf16, sdpa, gradient checkpointing, a LoRA on
every linear layer), builds two synthetic sequences of --max-len tokens whose last --completion
positions are trained, and runs the harness's own update() on them with two minibatches, so the
frozen old-log-probability pass runs too. Prints, as JSON, the peak memory torch allocated and
reserved and what nvidia-smi reports for the whole GPU, including a vLLM server that may share
it. Run it with that server up to see whether training fits next to it. No rollouts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM

HARNESS = Path(__file__).resolve().parent / "grpo_harness.py"


def load_harness():
    spec = importlib.util.spec_from_file_location("grpo_harness", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gpu_used_mib() -> list[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(x) for x in out.split("\n")[0].split(",")] if out.strip() else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-len", type=int, default=32768)
    parser.add_argument("--completion", type=int, default=8192, help="trained positions at the end of each sequence")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if not 0 < args.completion < args.max_len:
        parser.error("--completion must be between 0 and --max-len")
    harness = load_harness()
    random.seed(0)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = harness.attach_lora(model, args.rank)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.0)
    vocab = model.config.vocab_size
    start = args.max_len - args.completion - 1
    batch = [harness.Sequence(inputs=[random.randrange(vocab) for _ in range(args.max_len)],
                              positions=list(range(start, args.max_len - 1)),
                              targets=[random.randrange(vocab) for _ in range(args.completion)],
                              behavior=[-1.0] * args.completion, advantage=1.0, turns=1) for _ in range(2)]
    settings = argparse.Namespace(minibatches=2, clip_low=0.2, clip_high=0.28, tis_cap=2.0, loss_agg="token")
    cuda = args.device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    harness.update(model, optimizer, batch, settings)
    report = {"model": args.model, "max_len": args.max_len, "completion": args.completion, "rank": args.rank,
              "seconds": round(time.monotonic() - started, 1),
              "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None,
              "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2) if cuda else None,
              "gpu_used_total_mib": gpu_used_mib()}
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
