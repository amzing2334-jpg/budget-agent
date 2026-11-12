"""GRPO core on harness artifacts: groups, advantages, token-level loss.

A rollout is an ordinary `run_task` attempt. Its reward is the independent
verifier outcome; ungraded attempts are left out of their group, never
scored as failures. Every agent request of an attempt is one training turn
whose completion tokens share the attempt's advantage.

The torch functions import torch lazily so group selection and turn
extraction stay importable in the CPU test environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Turn:
    messages: list[dict]          # exact request sent to the server
    completion_ids: list[int]     # sampled tokens (incl. end-of-turn when generated)
    behavior_logprobs: list[float] | None
    prompt_digest: str | None     # models.token_digest of the prompt IDs the server rendered


def extract_turns(run_dir: Path, *, require_token_ids: bool = True) -> list[Turn]:
    """Agent requests of one attempt; summaries and reflections are never trained."""
    turns = []
    for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines():
        call = json.loads(line)
        if call.get("purpose") != "agent" or "response" not in call:
            continue
        response = call["response"]
        ids, digest = response.get("completion_token_ids"), response.get("prompt_token_digest")
        if (ids is None or digest is None) and require_token_ids:
            raise ValueError(f"{run_dir}: attempt has no token-level record; rerun with api_logprobs=true")
        turns.append(Turn(call["messages"], ids or [], response.get("completion_logprobs"), digest))
    return turns


def step_tasks(schedule: list[int], step: int, per_step: int) -> list[int]:
    """Task indices for one step: consecutive slices of a fixed shuffled schedule, wrapping around."""
    return [schedule[(step * per_step + i) % len(schedule)] for i in range(per_step)]


def minibatch_indices(size: int, minibatches: int, rng) -> list[list[int]]:
    """Shuffle `size` items into at most `minibatches` interleaved, non-empty index groups."""
    order = list(range(size))
    rng.shuffle(order)
    return [order[i::minibatches] for i in range(min(minibatches, size))]


def subsample_turns(turns: list[Turn], limit: int | None, rng) -> list[Turn]:
    """At most `limit` turns of one attempt, uniformly without replacement, in their original order.

    Every turn re-encodes its whole prompt, so training cost grows with the
    number of turns while rollout cost does not. With token-level
    aggregation the sampled turns give a consistent estimate of the full
    gradient, at higher variance.
    """
    if limit is None or len(turns) <= limit:
        return turns
    return [turns[i] for i in sorted(rng.sample(range(len(turns)), limit))]


def pack_turns(prompts: list[list[int]], completions: list[list[int]]) -> list[list[int]]:
    """Group consecutive turns into prefix chains: each turn's prompt starts with the previous member's prompt
    plus completion, token for token.

    The `recent` context policy keeps the full history while it fits, so an
    attempt's early turns re-send everything before them; training each turn
    separately re-encodes that shared prefix every time. A chain is trained as
    one sequence instead. The chain breaks where the history was cut or the
    chat template re-tokenized a reply differently from how it was sampled.
    """
    chains: list[list[int]] = []
    for index, prompt in enumerate(prompts):
        if chains:
            head = prompts[chains[-1][-1]] + completions[chains[-1][-1]]
            if prompt[:len(head)] == head:
                chains[-1].append(index)
                continue
        chains.append([index])
    return chains


def chain_layout(prompts: list[list[int]], completions: list[list[int]], chain: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Inputs of a chain's single sequence, the positions whose logits predict each completion token, and the targets.

    Under causal attention the logits at those positions see exactly the
    prompt each turn was sampled from, so the log probabilities equal those
    of separate per-turn forwards.
    """
    last = chain[-1]
    inputs = prompts[last] + completions[last][:-1]
    positions: list[int] = []
    targets: list[int] = []
    for index in chain:
        start = len(prompts[index]) - 1
        positions.extend(range(start, start + len(completions[index])))
        targets.extend(completions[index])
    return inputs, positions, targets


def drive_steps(steps: int, *, pipeline: bool, launch: Callable[[int, int], Any], finish: Callable[[Any], Any],
                train: Callable[[int, Any, int, int | None], None], time_up: Callable[[], bool]) -> int:
    """Run up to `steps` GRPO steps of rollout then update; return how many steps were trained.

    Synchronous: rollout k is sampled by policy v{k}, the one the trainer is about to update.
    Pipelined: rollout k+1 is launched with policy v{k} before step k trains, so sampling (mostly
    the environment's tests) overlaps training and every rollout after the first is exactly one
    update stale; the trainer's TIS weights correct for the behaviour policy.

    launch(step, policy) starts a rollout and returns a handle; finish(handle) waits for it and
    returns the rollout; train(step, rollout, behaviour, in_use) updates the model, publishes
    v{step + 1}, and may unload any policy except v{step + 1} and `in_use` (the policy a running
    rollout still samples from). No new rollout starts once time_up() is true.
    """
    pending: tuple[int, Any] | None = (0, launch(0, 0))
    for step in range(steps):
        assert pending is not None
        behaviour, handle = pending
        rollout = finish(handle)
        pending = None
        if pipeline and step + 1 < steps and not time_up():
            pending = (step, launch(step + 1, step))
        train(step, rollout, behaviour, pending[0] if pending else None)
        if not pipeline and step + 1 < steps and not time_up():
            pending = (step + 1, launch(step + 1, step + 1))
        if pending is None:
            return step + 1
    return steps


def rollout_health(attempts: list[dict]) -> dict:
    """What a rollout batch did that the reward does not see.

    invalid_action_share: agent steps whose output was not a valid action (tool
    "invalid_action" in each attempt's trajectory.jsonl) over all agent steps of the batch;
    output_tokens_per_attempt: mean completion tokens per attempt. In round 7 the invalid
    share rose from about 1% to 80% while the training pass rate stayed flat, moving about
    9 steps before the entropy did.
    """
    actions = invalid = 0
    for row in attempts:
        if not row.get("result_path"):
            continue
        trajectory = Path(row["result_path"]).parent / "trajectory.jsonl"
        if not trajectory.exists():
            continue
        for line in trajectory.read_text(encoding="utf-8").splitlines():
            if line.strip():
                actions += 1
                invalid += json.loads(line)["action"].get("tool") == "invalid_action"
    tokens = [row.get("output_tokens") or 0 for row in attempts]
    return {"invalid_action_share": round(invalid / actions, 4) if actions else 0.0,
            "output_tokens_per_attempt": round(sum(tokens) / max(1, len(tokens)), 1)}


@dataclass
class HealthGuard:
    """Stop a run whose rollouts degenerate while the reward does not show it.

    observe(policy, share) takes one rollout batch's invalid-action share and the policy that
    sampled it. The guard trips after `patience` consecutive batches above `threshold`;
    healthy_policy is then the last policy whose batch stayed at or below the threshold (v0 if
    none did), the one to keep. A threshold of None never trips.
    """
    threshold: float | None
    patience: int = 2
    healthy_policy: int = 0
    streak: int = 0
    tripped: bool = False

    def observe(self, policy: int, share: float) -> bool:
        if self.threshold is None or self.tripped:
            return self.tripped
        if share > self.threshold:
            self.streak += 1
            self.tripped = self.streak >= self.patience
        else:
            self.streak = 0
            self.healthy_policy = policy
        return self.tripped


def group_advantages(rewards: list[float], *, scale_by_std: bool = True, eps: float = 1e-6) -> list[float] | None:
    """Group-relative advantages; None when every reward is equal (no learning signal).

    scale_by_std=True is GRPO; False keeps only centering (Dr. GRPO), which
    avoids up-weighting groups whose rewards barely differ.
    """
    if len(rewards) < 2:
        return None
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    if std < eps:
        return None
    return [(r - mean) / (std if scale_by_std else 1.0) for r in rewards]


def build_groups(attempts: list[dict], *, scale_by_std: bool = True) -> tuple[list[dict], dict]:
    """Keep per-task groups with a learning signal (dynamic-sampling filter).

    Each attempt row needs task_id, resolved (bool or None) and result_path.
    Returns kept attempts with an `advantage` field, plus counts for logging.
    """
    by_task: dict[str, list[dict]] = {}
    ungraded = 0
    for row in attempts:
        if row["resolved"] is None:
            ungraded += 1
            continue
        by_task.setdefault(row["task_id"], []).append(row)
    kept: list[dict] = []
    stats = {"tasks": len(by_task), "ungraded_attempts": ungraded, "all_solved": 0, "all_failed": 0,
             "kept_groups": 0, "kept_attempts": 0}
    for rows in by_task.values():
        rewards = [float(bool(row["resolved"])) for row in rows]
        advantages = group_advantages(rewards, scale_by_std=scale_by_std)
        if advantages is None:
            stats["all_solved" if rewards[0] else "all_failed"] += 1
            continue
        stats["kept_groups"] += 1
        stats["kept_attempts"] += len(rows)
        kept.extend({**row, "advantage": advantage} for row, advantage in zip(rows, advantages, strict=True))
    return kept, stats


def token_policy_loss(logprobs, old_logprobs, advantage: float, *, clip_low: float = 0.2, clip_high: float = 0.28,
                      behavior_logprobs=None, tis_cap: float | None = 2.0):
    """Per-token clipped surrogate (PPO/GRPO) with asymmetric clip and optional TIS.

    logprobs carry gradients; old_logprobs come from the trainer before the
    update. behavior_logprobs are the sampler's (vLLM) values: the truncated
    importance weight min(pi_old / pi_behavior, cap) corrects the gap between
    inference and training numerics for the sampled tokens.
    """
    import torch

    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_low, 1.0 + clip_high) * advantage
    losses = -torch.minimum(unclipped, clipped)
    stats = {"clip_fraction": ((ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)).float().mean().item(),
             "ratio_max": ratio.max().item()}
    if behavior_logprobs is not None and tis_cap is not None:
        raw = torch.exp(old_logprobs - behavior_logprobs).detach()
        weight = raw.clamp(max=tis_cap)
        losses = losses * weight
        stats.update(tis_weight_mean=weight.mean().item(), tis_capped_fraction=(raw > tis_cap).float().mean().item(),
                     sampler_trainer_abs_logprob_gap=(old_logprobs - behavior_logprobs).abs().mean().item())
    return losses, stats


def aggregate(losses: list, *, mode: str = "token"):
    """token: mean over every completion token in the batch (DAPO), so long
    replies are not down-weighted; sequence: mean per turn, then over turns."""
    import torch

    if mode == "token":
        return torch.cat(losses).sum() / sum(loss.numel() for loss in losses)
    if mode == "sequence":
        return torch.stack([loss.mean() for loss in losses]).mean()
    raise ValueError("mode must be token or sequence")
