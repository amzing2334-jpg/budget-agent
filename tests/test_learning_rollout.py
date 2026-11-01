import json

import pytest

from budget_agent.environment import EnvironmentError
from budget_agent.learning.rollout import initial_messages, sample_episode
from budget_agent.learning.schema import PolicySample, TrainingConfig
from budget_agent.schema import Task


class TokenPolicy:
    """Deterministic character-token policy for trajectory/budget unit tests only."""

    version = 7

    def __init__(self, actions):
        self.responses = iter(actions)
        self.calls = []

    def encode(self, messages):
        return [ord(char) for char in json.dumps(messages, ensure_ascii=False)]

    def sample(self, messages, max_new_tokens):
        self.calls.append((list(messages), max_new_tokens))
        text = next(self.responses)[:max_new_tokens]
        return PolicySample(self.encode(messages), [ord(char) for char in text],
                            [-1.0] * len(text), text, self.version)


class FakeEnvironment:
    def __init__(self, grade=None):
        self.verdict = grade or {"resolved": True, "status": "resolved"}
        self.actions = []
        self.closed = False
        self.graded = False

    def execute(self, action):
        self.actions.append(action)
        return {"ok": True, "output": "visible tool result"}

    def patch(self):
        return "--- a/solution.py\n+++ b/solution.py\n"

    def grade(self):
        self.graded = True
        return self.verdict

    def close(self):
        self.closed = True


@pytest.fixture
def task():
    return Task("train-threshold-1", "Repair the boundary in solution.py", "train",
                test_command=["python", "-m", "unittest"],
                metadata={"private_answer": "never send this"}, verifier="secret-verifier.py")


def install_environment(monkeypatch, env):
    monkeypatch.setattr("budget_agent.learning.rollout.create_environment", lambda *args: env)


def test_multiturn_episode_preserves_each_prompt_and_reward(tmp_path, monkeypatch, task):
    env = FakeEnvironment()
    install_environment(monkeypatch, env)
    policy = TokenPolicy([json.dumps(action) for action in [
        {"tool": "read_file", "path": "solution.py"},
        {"tool": "write_file", "path": "solution.py", "content": "fixed"},
        {"tool": "run", "command": "python -m unittest"}, {"tool": "finish"},
    ]])
    episode = sample_episode(task, policy, TrainingConfig(context_tokens=4096, total_tokens=16000), tmp_path)
    assert episode.resolved is True and episode.reward == 1.0
    assert (tmp_path / "patch.diff").read_bytes() == episode.patch.encode("utf-8")
    assert episode.policy_version == 7 and len(episode.steps) == 4
    assert len(env.actions) == 3 and env.closed and env.graded
    assert episode.input_tokens == sum(len(step.prompt_ids) for step in episode.steps)
    assert episode.output_tokens == sum(len(step.completion_ids) for step in episode.steps)
    assert episode.input_tokens + episode.output_tokens <= 16000
    assert episode.steps[1].prompt_ids == policy.encode(policy.calls[1][0])
    assert policy.calls[1][0][-1]["content"].find("visible tool result") >= 0
    prompt_text = json.dumps(policy.calls)
    assert "never send this" not in prompt_text and "secret-verifier" not in prompt_text
    saved = json.loads((tmp_path / "episode.json").read_text())
    assert saved["steps"][0]["old_log_probs"] == episode.steps[0].old_log_probs
    assert (tmp_path / "patch.diff").read_text() == episode.patch


def test_remaining_budget_caps_generation_and_counts_invalid_output(tmp_path, monkeypatch, task):
    env = FakeEnvironment({"resolved": False, "status": "unresolved"})
    install_environment(monkeypatch, env)
    policy = TokenPolicy(['{"tool":"finish"}'])
    budget = len(policy.encode(initial_messages(task))) + 5
    episode = sample_episode(task, policy, TrainingConfig(total_tokens=budget, context_tokens=4096), tmp_path)
    assert policy.calls[0][1] == 5
    assert episode.output_tokens == 5
    assert episode.input_tokens + episode.output_tokens == budget
    assert episode.status == "budget_exhausted" and episode.reward == 0
    assert len(policy.calls) == 1 and env.graded


def test_full_context_stops_before_sampling_but_still_grades(tmp_path, monkeypatch, task):
    env = FakeEnvironment({"resolved": False, "status": "unresolved"})
    install_environment(monkeypatch, env)
    policy = TokenPolicy([])
    limit = len(policy.encode(initial_messages(task)))
    episode = sample_episode(task, policy, TrainingConfig(context_tokens=limit), tmp_path)
    assert episode.status == "context_exhausted"
    assert episode.steps == [] and episode.input_tokens == 0 and env.graded


@pytest.mark.parametrize("bad_action", ["not JSON", '{"tool":"remember","note":"free memory"}'])
def test_bad_action_is_observation_and_never_positive_reward_by_itself(tmp_path, monkeypatch, task, bad_action):
    env = FakeEnvironment({"resolved": False, "status": "unresolved"})
    install_environment(monkeypatch, env)
    policy = TokenPolicy([bad_action, '{"tool":"finish"}'])
    episode = sample_episode(task, policy, TrainingConfig(context_tokens=4096), tmp_path)
    assert not episode.trace[0]["result"]["ok"]
    assert json.loads(policy.calls[1][0][-1]["content"])["ok"] is False
    assert episode.reward == 0 and episode.resolved is False and not env.actions


def test_missing_grader_reward_is_an_error_with_diagnostics(tmp_path, monkeypatch, task):
    env = FakeEnvironment({"resolved": None, "status": "infrastructure_error", "output": "Docker unavailable"})
    install_environment(monkeypatch, env)
    policy = TokenPolicy(['{"tool":"finish"}'])
    with pytest.raises(EnvironmentError, match="No training reward"):
        sample_episode(task, policy, TrainingConfig(), tmp_path)
    assert env.closed
    assert json.loads((tmp_path / "grade.json").read_text())["resolved"] is None
    assert not (tmp_path / "episode.json").exists()


def test_environment_is_closed_when_generation_fails(tmp_path, monkeypatch, task):
    env = FakeEnvironment()
    install_environment(monkeypatch, env)
    with pytest.raises(StopIteration):
        sample_episode(task, TokenPolicy([]), TrainingConfig(), tmp_path)
    assert env.closed


def test_local_model_execution_requires_explicit_opt_in(tmp_path, task):
    with pytest.raises(EnvironmentError, match="not sandboxed"):
        sample_episode(task, TokenPolicy([]), TrainingConfig(environment="local"), tmp_path)


@pytest.mark.parametrize("mode", ["missing_dependency", "early_exit", "failure", "success"])
def test_real_grader_only_completed_tests_create_training_episode(tmp_path, mode):
    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "solution.py").write_text("VALUE = 0\n", encoding="utf-8")
    verifier = tmp_path / "verifier.py"
    verifier.write_text(
        "import missing_budget_agent_verifier_dependency\n" if mode == "missing_dependency" else
        "import unittest\nfrom solution import VALUE\n"
        "class Tests(unittest.TestCase):\n"
        "    def test_value(self): self.assertEqual(VALUE, 1)\n", encoding="utf-8")
    task = Task("reward-contract", "Fix VALUE", "train", repo=str(repo), verifier=str(verifier))
    actions = []
    if mode in {"early_exit", "success"}:
        actions.append(json.dumps({"tool": "write_file", "path": "solution.py",
                                   "content": "import os\nos._exit(0)\n" if mode == "early_exit" else "VALUE = 1\n"}))
    actions.append('{"tool":"finish"}')
    config = TrainingConfig(environment="local", allow_local_execution=True,
                            context_tokens=4096, total_tokens=16000)
    run = tmp_path / "run"
    if mode in {"missing_dependency", "early_exit"}:
        with pytest.raises(EnvironmentError, match="No training reward: grader returned grading_error"):
            sample_episode(task, TokenPolicy(actions), config, run)
        assert not (run / "episode.json").exists()
        grade = json.loads((run / "grade.json").read_text())
        assert grade["resolved"] is None and grade["test_results"] is None
        if mode == "missing_dependency":
            assert "ModuleNotFoundError" in grade["output"]
    else:
        episode = sample_episode(task, TokenPolicy(actions), config, run)
        assert episode.resolved is (mode == "success")
        assert episode.reward == float(mode == "success")
        assert (run / "episode.json").is_file()
    assert (run / "patch.diff").is_file()
    assert (run / "trajectory.jsonl").is_file()
