"""Stepwise GRPO with terminal verification rewards and completion-only gradients.

Each stored prompt is replayed exactly, so context may differ between agent steps.
One fresh rollout group precedes each optimizer step; beta=0 (no reference KL).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch

from .rollout import sample_episode
from .schema import Episode, TrainingConfig


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def parameter_hash(policy) -> str:
    digest = hashlib.sha256()
    for parameter in policy.parameters():
        digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def clipped_objective(log_probs, old_log_probs, advantage: float, epsilon: float):
    ratio = (log_probs - old_log_probs).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)


def grpo_update(policy, episodes: list[Episode], optimizer, config: TrainingConfig) -> dict:
    """Equal trajectory weighting, token-mean within each trajectory."""
    if any(e.policy_version != policy.version for e in episodes):
        raise ValueError("GRPO batch contains stale policy rollouts")
    if any(e.resolved is None for e in episodes):
        raise ValueError("Ungraded trajectories cannot supply RL rewards")
    rewards = torch.tensor([e.reward for e in episodes], dtype=torch.float32)
    centered = rewards - rewards.mean()
    metrics = {"rewards": rewards.tolist(), "reward_mean": rewards.mean().item(),
               "policy_version_before": policy.version}
    if rewards.std(correction=0).item() == 0:
        return {**metrics, "updated": False, "reason": "equal_rewards", "grad_norm": 0.0,
                "parameter_delta_l2": 0.0, "policy_version_after": policy.version}
    advantages = (centered / rewards.std(correction=0)).tolist()
    parameters = list(policy.parameters())
    before = [p.detach().clone() for p in parameters]
    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.0
    max_ratio_deviation = 0.0
    for episode, advantage in zip(episodes, advantages):
        token_count = sum(len(sample.completion_ids) for sample in episode.steps)
        for sample in episode.steps:
            log_probs = policy.log_probs(sample)
            old = torch.tensor(sample.old_log_probs, device=log_probs.device)
            max_ratio_deviation = max(max_ratio_deviation,
                                      (log_probs.detach() - old).exp().sub(1).abs().max().item())
            loss = clipped_objective(log_probs, old, advantage, config.clip_epsilon).sum()
            loss = loss / token_count / len(episodes)
            loss.backward()
            loss_value += loss.detach().item()
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    delta = sum((p.detach().float() - old.float()).square().sum().item()
                for p, old in zip(parameters, before)) ** 0.5
    policy.version += 1
    return {**metrics, "advantages": advantages, "loss": loss_value, "updated": True,
            "grad_norm": float(grad_norm), "parameter_delta_l2": delta,
            "max_ratio_deviation_before_update": max_ratio_deviation,
            "policy_version_after": policy.version}


def save_checkpoint(policy, optimizer, output: Path, config: TrainingConfig, stage: str) -> None:
    policy.save(output / "adapter")
    torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "policy_version": policy.version}, output / "optimizer.pt")
    write_json(output / "checkpoint.json", {"stage": stage, "policy_version": policy.version,
               "parameter_sha256": parameter_hash(policy), "config": asdict(config)})


def task_provenance(tasks) -> list[dict]:
    records = []
    for task in tasks:
        digest = hashlib.sha256(task.issue.encode())
        for path in sorted(Path(task.repo).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                digest.update(path.relative_to(task.repo).as_posix().encode())
                digest.update(path.read_bytes())
        digest.update(Path(task.verifier).read_bytes())
        records.append({"task_id": task.id, "split": task.split, "sha256": digest.hexdigest()})
    return records


def train_grpo(policy, tasks, config: TrainingConfig, output: Path) -> dict:
    if any(task.split != "train" for task in tasks):
        raise ValueError("Training accepts only the train split")
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(config.seed)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate, weight_decay=0.0)
    before = parameter_hash(policy)
    write_json(output / "config.json", asdict(config))
    write_json(output / "tasks.json", task_provenance(tasks))
    records = []
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for group in range(config.updates):
            task = tasks[group % len(tasks)]
            episodes = [sample_episode(task, policy, config, output / f"group-{group:03d}" / f"sample-{i:02d}")
                        for i in range(config.group_size)]
            record = {"group": group, "task_id": task.id,
                      "input_tokens": sum(e.input_tokens for e in episodes),
                      "output_tokens": sum(e.output_tokens for e in episodes),
                      **grpo_update(policy, episodes, optimizer, config)}
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(json.dumps(record), flush=True)
    save_checkpoint(policy, optimizer, output, config, "grpo")
    result = {"stage": "grpo", "attempted_groups": len(records),
              "optimizer_steps": sum(r["updated"] for r in records),
              "parameter_sha256_before": before, "parameter_sha256_after": parameter_hash(policy),
              "policy_version": policy.version,
              "note": "Synthetic curriculum verification; does not establish real-repository gains."}
    write_json(output / "result.json", result)
    return result


def sft_examples(policy, path: Path, context_tokens: int):
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["split"] != "train":
            raise ValueError("SFT demonstrations must come from train tasks")
        messages = row["messages"]
        for i, message in enumerate(messages):
            if message["role"] == "assistant":
                prompt = policy.encode(messages[:i])
                completion = policy.tokenizer.encode(message["content"], add_special_tokens=False)
                completion.append(policy.tokenizer.eos_token_id)
                if len(prompt) + len(completion) > context_tokens:
                    raise ValueError(f"SFT example exceeds context budget: {row['task_id']}")
                examples.append((prompt, completion))
    if not examples:
        raise ValueError("No assistant demonstrations found")
    return examples


def train_sft(policy, data: Path, config: TrainingConfig, output: Path, steps: int) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(config.seed)
    examples = sft_examples(policy, data, config.context_tokens)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate, weight_decay=0.0)
    before = parameter_hash(policy)
    write_json(output / "config.json", {**asdict(config), "sft_steps": steps,
               "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest()})
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(steps):
            prompt, completion = examples[random.randrange(len(examples))]
            optimizer.zero_grad(set_to_none=True)
            loss = policy.supervised_loss(prompt, completion)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(policy.parameters()), config.max_grad_norm,
                                                       error_if_nonfinite=True)
            optimizer.step()
            policy.version += 1
            record = {"step": step + 1, "loss": loss.detach().item(), "grad_norm": float(grad_norm)}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
                print(json.dumps(record), flush=True)
    save_checkpoint(policy, optimizer, output, config, "sft")
    result = {"stage": "sft", "optimizer_steps": steps, "examples": len(examples),
              "parameter_sha256_before": before, "parameter_sha256_after": parameter_hash(policy),
              "note": "Synthetic train-only teacher demonstrations; SFT is not RL."}
    write_json(output / "result.json", result)
    return result


def evaluate(policy, tasks, config: TrainingConfig, output: Path) -> dict:
    if any(task.split == "train" for task in tasks):
        raise ValueError("Evaluation requires a held-out split")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", asdict(config))
    write_json(output / "tasks.json", task_provenance(tasks))
    episodes = []
    for i, task in enumerate(tasks):
        seed_everything(config.seed + i)
        episode = sample_episode(task, policy, config, output / task.id)
        episodes.append(episode)
        print(json.dumps({"task": task.id, "resolved": episode.resolved,
                          "status": episode.status, "tokens": episode.input_tokens + episode.output_tokens}), flush=True)
    result = {"stage": "eval", "tasks": len(episodes), "resolved": sum(e.resolved for e in episodes),
              "resolve_rate": sum(e.resolved for e in episodes) / len(episodes),
              "total_tokens": sum(e.input_tokens + e.output_tokens for e in episodes),
              "policy_version": policy.version, "parameter_sha256": parameter_hash(policy),
              "note": "One seeded sample per task. Same-template held-out parameters, not repository generalization."}
    write_json(output / "result.json", result)
    return result
