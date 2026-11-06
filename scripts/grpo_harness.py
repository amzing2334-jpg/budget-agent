"""Single-GPU GRPO over harness rollouts, with LoRA hot-swapping into vLLM.

Runs on the GPU host only (AGENTS.md). Each step:
  1. rollout: B training tasks x G attempts through `run_task` (the same
     harness used for evaluation) as the unprivileged user, served by vLLM
     with the current adapter and api_logprobs=true (exact sampled tokens and
     behaviour log probabilities are recorded);
  2. groups: verifier rewards, ungraded attempts excluded, zero-variance
     groups dropped (dynamic-sampling filter), group-relative advantages;
  3. update: token-level clipped surrogate (clip-higher) with truncated
     importance weights against the sampler's log probabilities. With one
     optimizer step per batch the PPO ratio is exactly 1 and clipping is
     inert; --minibatches N takes N steps against log probabilities frozen
     before the first, which is where clipping acts;
  4. sync: save adapter v{n+1} and load it into the running vLLM server.

Token in, token out: the trainer re-renders each request with its own chat
template and keeps the turn only if the result is token-identical to the
prompt the server sampled from (the recorded prompt digest). Mismatches are
dropped and counted, never trained in a context the policy never saw.

Rollouts use temperature 1.0 so vLLM's returned log probabilities describe
the actual sampling distribution. vLLM must run with --enable-lora and
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True (scripts/autodl/serve_vllm.sh).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import time
import urllib.request

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from budget_agent.learning.grpo_core import aggregate, build_groups, extract_turns, subsample_turns, token_policy_loss
from budget_agent.models import token_digest
from budget_agent.tasks import load_tasks


def post(url: str, payload: dict) -> None:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        response.read()


def load_adapter(server: str, name: str, path: Path, previous: str | None) -> None:
    post(f"{server}/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
    if previous:
        post(f"{server}/v1/unload_lora_adapter", {"lora_name": previous})


def completion_logprobs(model, prompt_ids: list[int], completion_ids: list[int]):
    """Log probabilities of the sampled completion and its mean entropy; the last prompt token predicts the first."""
    inputs = torch.tensor([prompt_ids + completion_ids[:-1]], device="cuda")
    logits = model(input_ids=inputs, logits_to_keep=len(completion_ids)).logits[0].float()
    logp = logits.log_softmax(-1)
    targets = torch.tensor(completion_ids, device="cuda")
    with torch.no_grad():
        entropy = -(logp.exp() * logp).sum(-1).mean()
    return logp.gather(-1, targets[:, None]).squeeze(-1), entropy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tasks", type=Path, required=True, help="train manifest")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="root-owned training outputs")
    parser.add_argument("--rollouts", type=Path, required=True, help="directory writable by the rollout user")
    parser.add_argument("--pool-launcher", default="/opt/coding-agent/budget-agent/scripts/autodl/run_pool.sh")
    parser.add_argument("--server", default="http://127.0.0.1:18017")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--tasks-per-step", type=int, default=16)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--clip-low", type=float, default=0.2)
    parser.add_argument("--clip-high", type=float, default=0.28)
    parser.add_argument("--tis-cap", type=float, default=2.0)
    parser.add_argument("--no-std-scale", action="store_true", help="Dr. GRPO centering only")
    parser.add_argument("--loss-agg", choices=["token", "sequence"], default="token")
    parser.add_argument("--minibatches", type=int, default=1, help="optimizer steps per rollout batch")
    parser.add_argument("--max-minutes", type=float, help="start no new step after this much wall time")
    parser.add_argument("--turns-per-attempt", type=int, help="train at most this many uniformly sampled turns "
                        "of each attempt (default: all)")
    parser.add_argument("--max-len", type=int, default=24576)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tasks = load_tasks(args.tasks, "train")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    args.output.mkdir(parents=True, exist_ok=True)
    log = (args.output / "metrics.jsonl").open("a")
    model.save_pretrained(args.output / "adapters" / "v0")  # zero-initialised B: identical to the base model
    load_adapter(args.server, "policy-v0", args.output / "adapters" / "v0", None)
    order = list(range(len(tasks)))
    random.shuffle(order)
    run_started = time.monotonic()
    for step in range(args.steps):
        if args.max_minutes is not None and time.monotonic() - run_started > args.max_minutes * 60:
            print(json.dumps({"stopped_before_step": step, "reason": "max_minutes"}), flush=True)
            break
        started = time.monotonic()
        chosen = [tasks[order[(step * args.tasks_per_step + i) % len(order)]] for i in range(args.tasks_per_step)]
        manifest = args.rollouts / f"step-{step:04d}.jsonl"
        manifest.write_text("".join(json.dumps({**task.__dict__}) + "\n" for task in chosen))
        pool_dir = args.rollouts / f"step-{step:04d}"
        subprocess.run([args.pool_launcher, "--config", str(args.config), "--tasks", str(manifest), "--split", "train",
                        "--samples", str(args.group), "--workers", str(args.workers), "--output", str(pool_dir),
                        "--set", f"model=policy-v{step}", "--set", "api_logprobs=true", "--set", "temperature=1.0"],
                       check=True)
        rollout_seconds = time.monotonic() - started
        attempts = [json.loads(line) for line in (pool_dir / "attempts.jsonl").read_text().splitlines() if line.strip()]
        kept, groups = build_groups(attempts, scale_by_std=not args.no_std_scale)
        graded = [a for a in attempts if a["resolved"] is not None]
        record = {"step": step, "attempts": len(attempts), "graded": len(graded),
                  "pass_rate": sum(bool(a["resolved"]) for a in graded) / max(1, len(graded)), **groups,
                  "rollout_seconds": round(rollout_seconds, 1)}
        train_started = time.monotonic()
        batch, mismatched, available = [], 0, 0
        for attempt in kept:
            turns = extract_turns(Path(attempt["result_path"]).parent)
            available += len(turns)
            for turn in subsample_turns(turns, args.turns_per_attempt, random):
                prompt = tokenizer.apply_chat_template(turn.messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=False)
                if token_digest(prompt) != turn.prompt_digest:
                    mismatched += 1
                elif turn.completion_ids and len(prompt) + len(turn.completion_ids) <= args.max_len:
                    batch.append((prompt, turn, attempt["advantage"]))
        record.update(prompt_mismatch=mismatched, turns_available=available)
        if mismatched and not batch:
            raise RuntimeError(f"step {step}: no kept turn re-renders to the prompt the server sampled from; "
                               "trainer and server must load the same tokenizer and chat template")
        if batch:
            model.train()
            turn_order = list(range(len(batch)))  # not `order`: that is the task schedule across steps
            random.shuffle(turn_order)
            minibatches = [turn_order[i::args.minibatches] for i in range(min(args.minibatches, len(turn_order)))]
            olds = None  # one update: the ratio is 1 and the old log-probabilities are the current ones, detached
            if len(minibatches) > 1:
                with torch.no_grad():
                    olds = [completion_logprobs(model, prompt, turn.completion_ids)[0] for prompt, turn, _ in batch]
            stats, entropies, norms = [], [], []
            for indices in minibatches:
                total_tokens = sum(len(batch[i][1].completion_ids) for i in indices)
                for i in indices:
                    prompt, turn, advantage = batch[i]
                    logp, entropy = completion_logprobs(model, prompt, turn.completion_ids)
                    old = olds[i] if olds is not None else logp.detach()
                    behavior = (torch.tensor(turn.behavior_logprobs, device="cuda")
                                if turn.behavior_logprobs is not None else None)
                    losses, info = token_policy_loss(logp, old, advantage, clip_low=args.clip_low,
                                                     clip_high=args.clip_high, behavior_logprobs=behavior,
                                                     tis_cap=args.tis_cap)
                    # Token aggregation without holding every graph: scale each turn by the minibatch token count.
                    loss = (losses.sum() / total_tokens if args.loss_agg == "token"
                            else aggregate([losses], mode="sequence") / len(indices))
                    loss.backward()
                    stats.append(info)
                    entropies.append(entropy.item())
                norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            record.update(turns=len(batch), updates=len(minibatches),
                          completion_tokens=sum(len(turn.completion_ids) for _, turn, _ in batch),
                          grad_norm=sum(norms) / len(norms), entropy=sum(entropies) / len(entropies),
                          **{k: sum(s.get(k, 0.0) for s in stats) / len(stats) for k in stats[0]})
        record["train_seconds"] = round(time.monotonic() - train_started, 1)
        sync_started = time.monotonic()
        path = args.output / "adapters" / f"v{step + 1}"
        model.save_pretrained(path)
        load_adapter(args.server, f"policy-v{step + 1}", path, f"policy-v{step}")
        record["sync_seconds"] = round(time.monotonic() - sync_started, 1)
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
