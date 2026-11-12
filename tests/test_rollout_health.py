"""Rollout health: what a batch did that the reward does not see, and the guard that stops a degenerating run."""
import json
from pathlib import Path

from budget_agent.learning.grpo_core import HealthGuard, rollout_health

ROOT = Path(__file__).resolve().parents[1]


def attempt(tmp_path, name, tools, output_tokens=100):
    run = tmp_path / name
    run.mkdir()
    (run / "trajectory.jsonl").write_text("".join(json.dumps({"step": i + 1, "action": {"tool": tool}}) + "\n"
                                                  for i, tool in enumerate(tools)))
    return {"task_id": name, "resolved": False, "output_tokens": output_tokens, "result_path": str(run / "result.json")}


def test_health_counts_invalid_actions_over_all_agent_steps_and_mean_output(tmp_path):
    rows = [attempt(tmp_path, "a", ["search", "edit_file", "finish"], 90),
            attempt(tmp_path, "b", ["read_file", "invalid_action", "invalid_action", "invalid_action", "invalid_action"],
                    2010),
            {"task_id": "c", "resolved": None, "output_tokens": None, "result_path": str(tmp_path / "gone/result.json")}]
    assert rollout_health(rows) == {"invalid_action_share": 0.5, "output_tokens_per_attempt": 700.0}
    assert rollout_health([]) == {"invalid_action_share": 0.0, "output_tokens_per_attempt": 0.0}


def test_the_guard_needs_consecutive_unhealthy_batches_and_keeps_the_last_healthy_policy():
    guard = HealthGuard(0.05, patience=2)
    observed = [guard.observe(policy, share) for policy, share in enumerate([0.01, 0.02, 0.09, 0.03, 0.06, 0.08, 0.01])]
    assert observed == [False, False, False, False, False, True, True]
    assert (guard.tripped, guard.healthy_policy) == (True, 3)   # stays tripped; v3 was the last healthy batch


def test_pipelined_batches_report_the_policy_that_sampled_them():
    guard = HealthGuard(0.05, patience=1)
    for policy, share in [(0, 0.0), (0, 0.01), (1, 0.02), (2, 0.4)]:   # step k samples with v{k-1}
        guard.observe(policy, share)
    assert (guard.tripped, guard.healthy_policy) == (True, 1)


def test_a_guard_without_threshold_never_trips_and_an_unhealthy_start_keeps_v0():
    off = HealthGuard(None)
    assert not any(off.observe(step, 1.0) for step in range(10))
    early = HealthGuard(0.05, patience=1)
    assert early.observe(0, 0.5) and early.healthy_policy == 0


def test_the_guard_would_have_stopped_round_7_at_step_28_and_kept_v26():
    # Round 7 ran synchronously, so step k's rollouts came from v{k}; the per-step shares are the
    # published behaviour rows (scripts/pool_behaviour.py over each step's pool).
    pools = json.loads((ROOT / "docs/data/round7/behaviour.json").read_text(encoding="utf-8"))
    steps = [pool for pool in pools if "/grpo/step-" in pool["pool"]]
    assert len(steps) == 46
    guard = HealthGuard(0.04, patience=2)
    stopped = next(step for step, pool in enumerate(steps)
                   if guard.observe(step, pool["tool_calls"].get("invalid_action", 0) / sum(pool["tool_calls"].values())))
    assert (stopped, guard.healthy_policy) == (28, 26)
