"""The GRPO step driver: synchronous and pipelined (one-step off-policy) rollout schedules."""
from budget_agent.learning.grpo_core import drive_steps


def run(steps, pipeline, stop_after_launches=None):
    events, launches = [], []

    def launch(step, policy):
        launches.append(step)
        events.append(("launch", step, policy))
        return (step, policy)

    def finish(handle):
        events.append(("finish", handle[0]))
        return {"rollout": handle[0], "policy": handle[1]}

    def train(step, rollout, behaviour, in_use):
        assert rollout["rollout"] == step and rollout["policy"] == behaviour
        events.append(("train", step, behaviour, in_use))

    def time_up():
        return stop_after_launches is not None and len(launches) >= stop_after_launches

    trained = drive_steps(steps, pipeline=pipeline, launch=launch, finish=finish, train=train, time_up=time_up)
    return trained, events


def test_synchronous_rollouts_sample_the_policy_being_trained():
    trained, events = run(3, pipeline=False)
    assert trained == 3
    assert events == [("launch", 0, 0), ("finish", 0), ("train", 0, 0, None),
                      ("launch", 1, 1), ("finish", 1), ("train", 1, 1, None),
                      ("launch", 2, 2), ("finish", 2), ("train", 2, 2, None)]


def test_pipelined_rollouts_overlap_training_and_are_one_update_stale():
    trained, events = run(3, pipeline=True)
    assert trained == 3
    assert events == [("launch", 0, 0), ("finish", 0), ("launch", 1, 0), ("train", 0, 0, 0),
                      ("finish", 1), ("launch", 2, 1), ("train", 1, 0, 1),
                      ("finish", 2), ("train", 2, 1, None)]
    # every update after the first trains on data sampled by the previous policy
    assert [event[1] - event[2] for event in events if event[0] == "train"] == [0, 1, 1]


def test_no_rollout_starts_after_the_time_limit_and_launched_ones_are_trained():
    for pipeline in (False, True):
        trained, events = run(10, pipeline=pipeline, stop_after_launches=2)
        assert trained == 2 and [e[0] for e in events].count("launch") == 2
        assert events[-1][0] == "train"


def test_a_single_step_trains_once():
    for pipeline in (False, True):
        assert run(1, pipeline=pipeline)[0] == 1
