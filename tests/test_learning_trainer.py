import json

import pytest

torch = pytest.importorskip("torch")

from budget_agent.learning.schema import Episode, PolicySample, TrainingConfig
from budget_agent.learning.trainer import clipped_objective, grpo_update, sft_examples


class TwoActionPolicy:
    def __init__(self):
        self.weights = torch.nn.Parameter(torch.zeros(2))
        self.version = 0

    def parameters(self):
        return iter([self.weights])

    def log_probs(self, sample):
        return self.weights.log_softmax(-1)[sample.completion_ids]


def episode(action, reward, version=0):
    sample = PolicySample([42], [action], [-0.69314718056], "action", version)
    return Episode("task", "train", version, [sample], reward, bool(reward),
                   "finished", 1, 1, "", [])


def test_grpo_increases_rewarded_action_and_records_update():
    policy = TwoActionPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    result = grpo_update(policy, [episode(0, 1), episode(1, 0)], optimizer, TrainingConfig())
    assert policy.weights.softmax(-1)[0].item() > 0.5
    assert result["updated"] and result["grad_norm"] > 0
    assert result["parameter_delta_l2"] > 0
    assert policy.version == 1


def test_equal_rewards_skip_parameter_update():
    policy = TwoActionPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    result = grpo_update(policy, [episode(0, 1), episode(1, 1)], optimizer, TrainingConfig())
    assert result["reason"] == "equal_rewards"
    assert policy.version == 0
    assert policy.weights.tolist() == [0, 0]


def test_clip_prevents_increasing_already_overweighted_good_action():
    logp = torch.tensor([0.5], requires_grad=True)
    loss = clipped_objective(logp, torch.zeros(1), 1.0, 0.2).sum()
    loss.backward()
    assert loss.item() == pytest.approx(-1.2)
    assert logp.grad.item() == 0


def test_negative_advantage_clips_underweighted_bad_action():
    logp = torch.tensor([-0.5], requires_grad=True)
    loss = clipped_objective(logp, torch.zeros(1), -1.0, 0.2).sum()
    loss.backward()
    assert loss.item() == pytest.approx(0.8)
    assert logp.grad.item() == 0


def test_stale_rollout_is_not_optimized():
    policy = TwoActionPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="stale"):
        grpo_update(policy, [episode(0, 1, version=1), episode(1, 0)], optimizer, TrainingConfig())


def test_ungraded_rollout_is_not_a_negative_reward():
    policy = TwoActionPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    missing = episode(1, 0)
    missing.resolved = None
    with pytest.raises(ValueError, match="Ungraded"):
        grpo_update(policy, [episode(0, 1), missing], optimizer, TrainingConfig())


def test_sft_rejects_eval_answer_leakage(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"split": "eval", "messages": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="train tasks"):
        sft_examples(None, data, 1024)
