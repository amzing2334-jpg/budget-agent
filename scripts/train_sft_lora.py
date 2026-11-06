"""LoRA SFT on multi-turn agent trajectories with assistant-only loss.

Runs on the GPU host only (AGENTS.md: no local training). Each sample is a
chat conversation from sft_data_from_runs.py. Loss covers assistant replies
(and their end-of-turn token); system prompts, task text and tool
observations are masked. Spans are found by rendering prefixes with the
model's own chat template, so the trained tokens are exactly what the
template produces at inference time. Logits are computed only at positions
that predict a trained token (about 1% of positions in round 1), so a
16k-token sample never materializes a float32 row over the 152k vocabulary
for every position.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def encode(tokenizer, messages: list[dict], mode: str, max_len: int) -> tuple[list[int], list[int]] | None:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    labels = [-100] * len(ids)
    targets = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if mode == "last":
        targets = targets[-1:]
    for index in targets:
        prefix = tokenizer.apply_chat_template(messages[:index], tokenize=True, add_generation_prompt=True,
                                               return_dict=False)
        upto = tokenizer.apply_chat_template(messages[:index + 1], tokenize=True, return_dict=False)
        if ids[:len(prefix)] != prefix or ids[:len(upto)] != upto:
            raise ValueError("chat template is not prefix-consistent; assistant spans would be wrong")
        labels[len(prefix):len(upto)] = ids[len(prefix):len(upto)]
    if len(ids) > max_len or all(label == -100 for label in labels):
        return None
    return ids, labels


def assistant_loss(model, ids: list[int], labels: list[int]) -> torch.Tensor:
    """Mean next-token loss over trained tokens, equal to the model's own loss with these labels.

    Only hidden states that predict a trained token go through the LM head
    (`logits_to_keep` with indices). The full-vocabulary loss on a 16k-token
    sample peaks at about 28 GiB of float32 logits and their gradients.
    """
    device = next(model.parameters()).device
    targets = torch.tensor(labels[1:], device=device)
    keep = (targets != -100).nonzero().squeeze(1)
    logits = model(input_ids=torch.tensor([ids], device=device), logits_to_keep=keep).logits[0]
    return torch.nn.functional.cross_entropy(logits.float(), targets[keep])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=16384)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--merge", action="store_true", help="also save a merged model for vLLM")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples, dropped = [], 0
    for row in rows:
        encoded = encode(tokenizer, row["messages"], row["mode"], args.max_len)
        if encoded is None:
            dropped += 1
        else:
            examples.append(encoded)
    trained_tokens = sum(sum(label != -100 for label in labels) for _, labels in examples)
    print(json.dumps({"examples": len(examples), "dropped_too_long_or_empty": dropped,
                      "tokens": sum(len(ids) for ids, _ in examples), "trained_tokens": trained_tokens}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(examples) * args.epochs / args.accumulate)
    warmup = max(1, int(0.05 * total_steps))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup)
                                                 * 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)))
    args.output.mkdir(parents=True, exist_ok=True)
    log = (args.output / "train_log.jsonl").open("w")
    order = []
    while len(order) < len(examples) * args.epochs:
        epoch = list(range(len(examples)))
        random.shuffle(epoch)
        order.extend(epoch)
    order = order[:int(len(examples) * args.epochs)]
    model.train()
    started, seen_tokens, running = time.monotonic(), 0, []
    for position, index in enumerate(order):
        ids, labels = examples[index]
        loss = assistant_loss(model, ids, labels)
        (loss / args.accumulate).backward()
        running.append(loss.item())
        seen_tokens += len(ids)
        if (position + 1) % args.accumulate == 0 or position + 1 == len(order):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step = (position + 1 + args.accumulate - 1) // args.accumulate
            record = {"step": step, "of": total_steps, "loss": sum(running) / len(running), "grad_norm": float(norm),
                      "lr": schedule.get_last_lr()[0], "tokens_per_s": seen_tokens / (time.monotonic() - started)}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 5 == 0 or step == total_steps:
                print(json.dumps(record), flush=True)
            running = []
    model.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "adapter")
    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(args.output / "merged", safe_serialization=True)
        tokenizer.save_pretrained(args.output / "merged")
    (args.output / "result.json").write_text(json.dumps({
        "model": args.model, "data": str(args.data), "examples": len(examples), "dropped": dropped,
        "trained_tokens": trained_tokens, "epochs": args.epochs, "lr": args.lr, "rank": args.rank,
        "optimizer_steps": total_steps, "seconds": round(time.monotonic() - started, 1)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
