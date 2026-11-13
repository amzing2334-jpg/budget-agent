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

Prefix packing (--pack-turns): consecutive turns of an attempt whose prompt
starts with the previous turn's prompt plus completion form a chain trained
as one sequence, with logits taken only at the completion positions. The
per-token log probabilities equal those of separate forwards, and the tokens
encoded drop to about the length of each chain's last turn.

Rollout health: every step records the share of agent steps whose output was
not a valid action and the mean output tokens per attempt (rollout_health).
The reward cannot see either: a policy that loops on invalid output fails
tasks it would fail anyway. With --halt-invalid-share X the run stops after
--halt-patience consecutive batches above X, and final.json names the last
policy whose batch stayed at or below it; otherwise final.json names the last
policy published.

Rollouts use temperature 1.0 so vLLM's returned log probabilities describe
the actual sampling distribution. vLLM must run with --enable-lora and
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True (scripts/autodl/serve_vllm.sh).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import NamedTuple
import urllib.request

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from budget_agent.learning.grpo_core import (HealthGuard, Turn, aggregate, build_groups, chain_layout, drive_steps,
                                             extract_turns, minibatch_indices, pack_turns, rollout_health, step_tasks,
                                             subsample_turns, token_policy_loss)
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


class Sequence(NamedTuple):
    """One training forward: a single turn, or a prefix chain of turns from one attempt (--pack-turns)."""
    inputs: list[int]
    positions: list[int]          # logits here predict the targets
    targets: list[int]            # sampled completion tokens, in order
    behavior: list[float] | None  # the sampler's log probabilities of the targets
    advantage: float
    turns: int


def sequence_logprobs(model, sequence: Sequence):
    """Log probabilities of the targets at their positions, and the mean entropy there."""
    device = next(model.parameters()).device
    inputs = torch.tensor([sequence.inputs], device=device)
    keep = torch.tensor(sequence.positions, device=device)
    logits = model(input_ids=inputs, logits_to_keep=keep).logits[0].float()
    logp = logits.log_softmax(-1)
    targets = torch.tensor(sequence.targets, device=device)
    with torch.no_grad():
        entropy = -(logp.exp() * logp).sum(-1).mean()
    return logp.gather(-1, targets[:, None]).squeeze(-1), entropy


def attach_lora(model, rank: int, init_adapter: Path | None = None):
    """A fresh LoRA (B = 0, so the base model's behaviour) or a trainable copy of a saved adapter."""
    if init_adapter is not None:
        return PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
    return get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                            target_modules="all-linear", task_type="CAUSAL_LM"))


def build_sequences(verified: list[tuple[list[int], Turn]], advantage: float, pack: bool) -> list[Sequence]:
    """Turn one attempt's verified (prompt ids, turn) pairs into training sequences."""
    prompts = [prompt for prompt, _ in verified]
    completions = [turn.completion_ids for _, turn in verified]
    chains = pack_turns(prompts, completions) if pack else [[index] for index in range(len(verified))]
    sequences = []
    for chain in chains:
        inputs, positions, targets = chain_layout(prompts, completions, chain)
        logprobs = [verified[index][1].behavior_logprobs for index in chain]
        known = [lp for lp in logprobs if lp is not None]
        behavior = [x for lp in known for x in lp] if len(known) == len(logprobs) else None
        sequences.append(Sequence(inputs, positions, targets, behavior, advantage, len(chain)))
    return sequences


def update(model, optimizer, batch: list[Sequence], args) -> dict:
    """Train on one step's sequences; returns the step record's training fields.

    One update: the ratio is 1 and the old log-probabilities are the current
    ones, detached. More updates freeze them before the first.
    """
    model.train()
    device = next(model.parameters()).device
    minibatches = minibatch_indices(len(batch), args.minibatches, random)
    olds = None
    if len(minibatches) > 1:
        with torch.no_grad():
            olds = [sequence_logprobs(model, sequence)[0] for sequence in batch]
    stats, entropies, norms = [], [], []
    for indices in minibatches:
        total_tokens = sum(len(batch[i].targets) for i in indices)
        for i in indices:
            sequence = batch[i]
            logp, entropy = sequence_logprobs(model, sequence)
            old = olds[i] if olds is not None else logp.detach()
            behavior = torch.tensor(sequence.behavior, device=device) if sequence.behavior is not None else None
            losses, info = token_policy_loss(logp, old, sequence.advantage, clip_low=args.clip_low,
                                             clip_high=args.clip_high, behavior_logprobs=behavior, tis_cap=args.tis_cap)
            # Token aggregation without holding every graph: scale each sequence by the minibatch token count.
            loss = (losses.sum() / total_tokens if args.loss_agg == "token"
                    else aggregate([losses], mode="sequence") / len(indices))
            loss.backward()
            stats.append(info)
            entropies.append(entropy.item())
        norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"turns": sum(sequence.turns for sequence in batch), "sequences": len(batch), "updates": len(minibatches),
            "input_tokens": sum(len(sequence.inputs) for sequence in batch),
            "completion_tokens": sum(len(sequence.targets) for sequence in batch),
            "grad_norm": sum(norms) / len(norms), "entropy": sum(entropies) / len(entropies),
            **{k: sum(s.get(k, 0.0) for s in stats) / len(stats) for k in stats[0]}}


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
    parser.add_argument("--pack-turns", action="store_true",
                        help="train each prefix chain of an attempt's turns as one sequence (same loss, far fewer tokens)")
    parser.add_argument("--turns-per-attempt", type=int, help="train at most this many uniformly sampled turns "
                        "of each attempt (default: all)")
    parser.add_argument("--max-len", type=int, default=24576)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", help="torch device for the trainer (cpu in the end-to-end test)")
    parser.add_argument("--pipeline", action="store_true",
                        help="launch each rollout with the newest policy before the previous step trains, so sampling "
                             "overlaps training; every rollout after the first is one update stale (TIS corrects it)")
    parser.add_argument("--init-adapter", type=Path,
                        help="continue from this saved LoRA (trainable; the optimizer starts fresh) instead of a new one")
    parser.add_argument("--rollout-set", action="append", default=[], metavar="KEY=VALUE",
                        help="extra run-config override for every rollout pool (repeatable), e.g. revision_labels=fixed")
    parser.add_argument("--halt-invalid-share", type=float,
                        help="stop once this many consecutive rollout batches (--halt-patience) have a larger share of "
                             "invalid actions, and keep the last policy whose batch did not (final.json)")
    parser.add_argument("--halt-patience", type=int, default=2)
    parser.add_argument("--keep-adapters", type=int,
                        help="keep on disk only v0, the last K adapters, those vLLM still holds and the one the guard "
                             "would keep (default: keep every adapter)")
    args = parser.parse_args()
    if any("=" not in item for item in args.rollout_set):
        parser.error("--rollout-set takes KEY=VALUE")
    if args.pack_turns and args.loss_agg != "token":
        parser.error("--pack-turns trains several turns per sequence; use --loss-agg token")
    if args.halt_invalid_share is not None and not 0 <= args.halt_invalid_share < 1:
        parser.error("--halt-invalid-share is a share in [0, 1)")
    if args.halt_patience < 1:
        parser.error("--halt-patience must be at least 1")
    if args.keep_adapters is not None and args.keep_adapters < 1:
        parser.error("--keep-adapters must be at least 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tasks = load_tasks(args.tasks, "train")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(args.device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = attach_lora(model, args.rank, args.init_adapter)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "args.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                                        for k, v in vars(args).items()}, indent=1))
    log = (args.output / "metrics.jsonl").open("a")
    model.save_pretrained(args.output / "adapters" / "v0")  # the base model (B = 0) or the initial adapter
    load_adapter(args.server, "policy-v0", args.output / "adapters" / "v0", None)
    schedule = list(range(len(tasks)))
    random.shuffle(schedule)
    run_started = time.monotonic()
    loaded = {0}   # adapter versions the server holds
    published = [0]   # adapter versions written, in order
    guard = HealthGuard(args.halt_invalid_share, args.halt_patience)

    def time_up() -> bool:
        return args.max_minutes is not None and time.monotonic() - run_started > args.max_minutes * 60

    def stop() -> bool:
        return time_up() or guard.tripped

    def launch(step: int, policy: int) -> dict:
        started = time.monotonic()
        chosen = [tasks[i] for i in step_tasks(schedule, step, args.tasks_per_step)]
        manifest = args.rollouts / f"step-{step:04d}.jsonl"
        manifest.write_text("".join(json.dumps({**task.__dict__}) + "\n" for task in chosen))
        pool_dir = args.rollouts / f"step-{step:04d}"
        command = [args.pool_launcher, "--config", str(args.config), "--tasks", str(manifest), "--split", "train",
                   "--samples", str(args.group), "--workers", str(args.workers), "--output", str(pool_dir),
                   "--set", f"model=policy-v{policy}", "--set", "api_logprobs=true", "--set", "temperature=1.0",
                   *(part for item in args.rollout_set for part in ("--set", item))]
        return {"process": subprocess.Popen(command), "command": command, "pool": pool_dir, "started": started,
                "policy": policy}

    def finish(handle: dict) -> dict:
        waiting = time.monotonic()
        code = handle["process"].wait()
        if code:
            raise subprocess.CalledProcessError(code, handle["command"])
        done = time.monotonic()
        attempts = [json.loads(line) for line in (handle["pool"] / "attempts.jsonl").read_text().splitlines()
                    if line.strip()]
        health = rollout_health(attempts)
        # Checked before the next rollout is launched, so a tripped guard starts nothing new.
        halted = guard.observe(handle["policy"], health["invalid_action_share"])
        return {"attempts": attempts, "rollout_seconds": round(done - handle["started"], 1),
                "wait_seconds": round(done - waiting, 1), "health": health, "halted": halted}

    def train(step: int, rollout: dict, behaviour: int, in_use: int | None) -> None:
        attempts = rollout["attempts"]
        kept, groups = build_groups(attempts, scale_by_std=not args.no_std_scale)
        graded = [a for a in attempts if a["resolved"] is not None]
        record = {"step": step, "behaviour_policy": behaviour, "attempts": len(attempts), "graded": len(graded),
                  "pass_rate": sum(bool(a["resolved"]) for a in graded) / max(1, len(graded)), **groups,
                  "rollout_seconds": rollout["rollout_seconds"], "rollout_wait_seconds": rollout["wait_seconds"],
                  **rollout["health"]}
        if rollout["halted"]:
            record["halted"] = {"invalid_action_share_above": args.halt_invalid_share, "patience": args.halt_patience,
                                "keep_policy": guard.healthy_policy}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            return
        train_started = time.monotonic()
        batch, mismatched, available = [], 0, 0
        for attempt in kept:
            turns = extract_turns(Path(attempt["result_path"]).parent)
            available += len(turns)
            verified = []
            for turn in subsample_turns(turns, args.turns_per_attempt, random):
                prompt = tokenizer.apply_chat_template(turn.messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=False)
                if token_digest(prompt) != turn.prompt_digest:
                    mismatched += 1
                elif turn.completion_ids and len(prompt) + len(turn.completion_ids) <= args.max_len:
                    verified.append((prompt, turn))
            batch.extend(build_sequences(verified, attempt["advantage"], args.pack_turns))
        record.update(prompt_mismatch=mismatched, turns_available=available)
        if mismatched and not batch:
            raise RuntimeError(f"step {step}: no kept turn re-renders to the prompt the server sampled from; "
                               "trainer and server must load the same tokenizer and chat template")
        if batch:
            record.update(update(model, optimizer, batch, args))
        record["train_seconds"] = round(time.monotonic() - train_started, 1)
        sync_started = time.monotonic()
        path = args.output / "adapters" / f"v{step + 1}"
        model.save_pretrained(path)
        load_adapter(args.server, f"policy-v{step + 1}", path, None)
        # Unload every older policy except the one a running rollout still samples from (pipelined mode).
        for version in sorted(loaded - {step + 1, in_use}):
            post(f"{args.server}/v1/unload_lora_adapter", {"lora_name": f"policy-v{version}"})
            loaded.discard(version)
        loaded.add(step + 1)
        published.append(step + 1)
        if args.keep_adapters is not None:
            keep = {0, guard.healthy_policy, *loaded, *range(step + 2 - args.keep_adapters, step + 2)}
            for version in published:
                if version not in keep and (args.output / "adapters" / f"v{version}").exists():
                    shutil.rmtree(args.output / "adapters" / f"v{version}")
        record["sync_seconds"] = round(time.monotonic() - sync_started, 1)
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    steps_run = drive_steps(args.steps, pipeline=args.pipeline, launch=launch, finish=finish, train=train,
                            time_up=stop)
    final = guard.healthy_policy if guard.tripped else published[-1]
    stopped = "halted" if guard.tripped else "max_minutes" if steps_run < args.steps else "steps"
    summary = {"policy": final, "adapter": str(args.output / "adapters" / f"v{final}"), "steps_run": steps_run,
               "stopped": stopped}
    (args.output / "final.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
