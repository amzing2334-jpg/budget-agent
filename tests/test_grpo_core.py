import json
import math

import pytest

from budget_agent.learning.grpo_core import (Turn, aggregate, build_groups, extract_turns, group_advantages,
                                             subsample_turns, token_policy_loss)


def write_calls(run_dir, calls):
    run_dir.mkdir(parents=True)
    (run_dir / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))


def test_extract_turns_keeps_agent_requests_with_their_token_records(tmp_path):
    agent = {"purpose": "agent", "messages": [{"role": "user", "content": "x"}],
             "response": {"content": "a", "completion_token_ids": [1, 2], "completion_logprobs": [-0.1, -0.2],
                          "prompt_token_digest": "d" * 32}}
    write_calls(tmp_path / "run", [agent, {"purpose": "summary", "messages": [], "response": {"content": "s"}},
                                   {"purpose": "agent", "status": "usage_unknown"}])
    turns = extract_turns(tmp_path / "run")
    assert len(turns) == 1 and turns[0].completion_ids == [1, 2] and turns[0].behavior_logprobs == [-0.1, -0.2]
    assert turns[0].prompt_digest == "d" * 32


def test_extract_turns_needs_the_prompt_digest_for_rl(tmp_path):
    response = {"content": "a", "completion_token_ids": [1], "completion_logprobs": [-0.1]}
    write_calls(tmp_path / "run", [{"purpose": "agent", "messages": [], "response": response}])
    with pytest.raises(ValueError, match="api_logprobs"):
        extract_turns(tmp_path / "run")


def test_extract_turns_refuses_text_only_attempts_for_rl(tmp_path):
    write_calls(tmp_path / "run", [{"purpose": "agent", "messages": [], "response": {"content": "a"}}])
    with pytest.raises(ValueError, match="api_logprobs"):
        extract_turns(tmp_path / "run")
    assert extract_turns(tmp_path / "run", require_token_ids=False)[0].completion_ids == []


def test_group_advantages_center_and_scale():
    assert group_advantages([1, 0, 0, 1]) == [1.0, -1.0, -1.0, 1.0]
    assert group_advantages([1, 0, 0, 1], scale_by_std=False) == [0.5, -0.5, -0.5, 0.5]
    assert group_advantages([1, 1, 1]) is None and group_advantages([0]) is None


def test_build_groups_filters_zero_signal_and_ungraded_attempts():
    rows = ([{"task_id": "mixed", "resolved": r, "result_path": f"m{i}"} for i, r in enumerate([True, False, None])]
            + [{"task_id": "easy", "resolved": True, "result_path": f"e{i}"} for i in range(2)]
            + [{"task_id": "hard", "resolved": False, "result_path": f"h{i}"} for i in range(2)])
    kept, stats = build_groups(rows)
    assert [(r["result_path"], r["advantage"]) for r in kept] == [("m0", 1.0), ("m1", -1.0)]
    assert stats == {"tasks": 3, "ungraded_attempts": 1, "all_solved": 1, "all_failed": 1,
                     "kept_groups": 1, "kept_attempts": 2}


def test_token_loss_clips_asymmetrically_and_truncates_importance_weights():
    torch = pytest.importorskip("torch")
    old = torch.zeros(3)
    logp = torch.tensor([0.0, math.log(1.5), math.log(0.5)], requires_grad=True)
    losses, stats = token_policy_loss(logp, old, advantage=1.0, behavior_logprobs=None)
    assert losses.tolist() == pytest.approx([-1.0, -1.28, -0.5])  # upper clip 1+0.28; lower side unclipped for A>0
    losses.sum().backward()
    assert logp.grad.tolist() == pytest.approx([-1.0, 0.0, -0.5])  # no gradient beyond the clip
    behavior = torch.tensor([0.0, 0.0, -math.log(3)])
    weighted, stats = token_policy_loss(torch.zeros(3), old, advantage=-1.0, behavior_logprobs=behavior, tis_cap=2.0)
    assert weighted.tolist() == pytest.approx([1.0, 1.0, 2.0])  # pi_old/pi_behavior = 3 capped at 2
    assert stats["tis_capped_fraction"] == pytest.approx(1 / 3)


def test_token_aggregation_weights_every_token_equally():
    torch = pytest.importorskip("torch")
    short, long = torch.tensor([1.0]), torch.tensor([0.0, 0.0, 0.0])
    assert aggregate([short, long], mode="token").item() == pytest.approx(0.25)
    assert aggregate([short, long], mode="sequence").item() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        aggregate([short], mode="batch")


def test_subsample_turns_keeps_order_is_seeded_and_leaves_short_attempts_alone():
    import random

    turns = [Turn([{"role": "user", "content": str(i)}], [i], None, None) for i in range(10)]
    picked = subsample_turns(turns, 4, random.Random(7))
    assert len(picked) == 4 and picked == sorted(picked, key=lambda turn: turn.completion_ids[0])
    assert picked == subsample_turns(turns, 4, random.Random(7))
    assert subsample_turns(turns, None, random.Random(7)) is turns
    assert subsample_turns(turns[:3], 4, random.Random(7)) == turns[:3]
