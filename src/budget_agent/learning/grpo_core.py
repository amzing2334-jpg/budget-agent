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
    kept, stats = [], {"tasks": len(by_task), "ungraded_attempts": ungraded, "all_solved": 0, "all_failed": 0,
                       "kept_groups": 0, "kept_attempts": 0}
    for rows in by_task.values():
        rewards = [float(bool(row["resolved"])) for row in rows]
        advantages = group_advantages(rewards, scale_by_std=scale_by_std)
        if advantages is None:
            stats["all_solved" if rewards[0] else "all_failed"] += 1
            continue
        stats["kept_groups"] += 1
        stats["kept_attempts"] += len(rows)
        kept.extend({**row, "advantage": advantage} for row, advantage in zip(rows, advantages))
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
