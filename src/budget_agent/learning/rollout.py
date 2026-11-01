"""Multi-turn repair trajectories with exact, shared inference budgets."""
from __future__ import annotations

import json
from pathlib import Path

from ..environment import EnvironmentError, create_environment
from ..models import parse_action
from ..schema import RunConfig, Task
from .schema import Episode, TrainingConfig


SYSTEM_PROMPT = """Repair the Python code. Reply with one JSON tool action, no prose.
Tools:
{"tool":"read_file","path":"solution.py"}
{"tool":"write_file","path":"solution.py","content":"complete file text"}
{"tool":"run","command":"python -m unittest discover -s tests -v"}
{"tool":"list_files"}
{"tool":"search","query":"text"}
{"tool":"diff"}
{"tool":"finish"}
Read the source, fix it, run public tests, then finish. Edit source only."""


def initial_messages(task: Task) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.issue + "\nPublic tests: " + " ".join(task.test_command)},
    ]


def observation(result: dict) -> dict[str, str]:
    return {"role": "user", "content": json.dumps(
        {"ok": result["ok"], "output": result["output"]}, ensure_ascii=False)}


def sample_episode(task: Task, policy, config: TrainingConfig, run_dir: Path) -> Episode:
    """Sample each turn from the current policy; only the final verifier rewards it."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    execution = RunConfig(
        provider="openai-compatible", environment=config.environment,
        docker_image=config.docker_image, timeout_seconds=config.tool_timeout,
        allow_local_execution=config.allow_local_execution,
    )
    env = create_environment(task, execution, run_dir)
    messages = initial_messages(task)
    steps, trace = [], []
    input_tokens = output_tokens = 0
    status = "max_steps"
    try:
        for step in range(config.max_steps):
            prompt_ids = policy.encode(messages)
            context_left = config.context_tokens - len(prompt_ids)
            budget_left = config.total_tokens - input_tokens - output_tokens - len(prompt_ids)
            allowance = min(config.max_new_tokens, context_left, budget_left)
            if allowance <= 0:
                status = "context_exhausted" if context_left <= 0 else "budget_exhausted"
                break

            prompt_messages = list(messages)
            sample = policy.sample(messages, max_new_tokens=allowance)
            steps.append(sample)
            input_tokens += len(sample.prompt_ids)
            output_tokens += len(sample.completion_ids)
            messages.append({"role": "assistant", "content": sample.text})
            action = None
            try:
                action = parse_action(sample.text)
                if action["tool"] in {"remember", "recall"}:
                    raise ValueError("This agent uses full history; remember/recall are not available")
            except ValueError as exc:
                result = {"ok": False, "output": str(exc)}
            else:
                result = ({"ok": True, "output": "Finished."} if action["tool"] == "finish"
                          else env.execute(action))
            trace.append({"step": step, "messages": prompt_messages,
                          "completion": sample.text, "action": action, "result": result})
            if action and action["tool"] == "finish":
                status = "finished"
                break
            messages.append(observation(result))

        patch = env.patch()
        (run_dir / "patch.diff").write_bytes(patch.encode("utf-8"))
        (run_dir / "trajectory.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trace), encoding="utf-8")
        grade = env.grade()
        (run_dir / "grade.json").write_text(json.dumps(grade, indent=2), encoding="utf-8")
        if grade["resolved"] is None:
            raise EnvironmentError(f"No training reward: grader returned {grade['status']}; see {run_dir / 'grade.json'}")
        episode = Episode(
            task_id=task.id, split=task.split, policy_version=policy.version,
            steps=steps, reward=float(grade["resolved"]), resolved=grade["resolved"],
            status=grade["status"] if status == "finished" else status,
            input_tokens=input_tokens, output_tokens=output_tokens, patch=patch, trace=trace,
        )
        (run_dir / "episode.json").write_text(
            json.dumps(episode.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return episode
    finally:
        env.close()
