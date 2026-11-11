"""Rebuild a recorded attempt's agent requests under another revision-label mode.

At each step the runner sends the context manager's rendering of the system prompt,
the task issue and the events so far, within the token budget that remains. For a
run whose system prompt is static (direct decisions, no loop feedback, no
reflection notes, sandbox execution) and whose context strategy makes no model
calls, every request can be rebuilt from result.json, calls.jsonl and
trajectory.jsonl.

`rerender_requests` first rebuilds the run in its own mode and requires the result
to equal the recorded requests exactly; only then does it rebuild in the requested
mode. The rebuilt requests are what the runner would have sent for the same
actions and observations, which is how SFT data for another context format is made
without new rollouts (the actions themselves stay those of the original policy).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..budget import TokenCounter
from ..context import ContextManager
from ..runner import FIXED_LABELS_PROMPT, SYSTEM
from ..schema import Event

TASK_PREFIX = "Task requirements (preserve in full):\n"
STATIC_PROMPT = {"decision_policy": "direct", "loop_policy": "off", "reflection_policy": "off",
                 "execution_policy": "sandbox"}
MODEL_FREE_STRATEGIES = {"full", "recent", "mask", "mask-read", "structured"}   # only "summary" calls the model
LABEL_MODES = ("relabel", "fixed")


class RerenderError(ValueError):
    """The run cannot be rebuilt faithfully; nothing is produced for it."""


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def recorded_mode(config: dict[str, Any]) -> str:
    return config.get("revision_labels") or "relabel"   # runs from before the option relabelled


def load_run(run_dir: Path) -> tuple[dict, list[dict], list[dict], list[Event]]:
    """result.json, every call, the budget ledger rows and the events, checked for alignment."""
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    config = result["config"]
    for key, expected in STATIC_PROMPT.items():
        if config.get(key, expected) != expected:
            raise RerenderError(f"{run_dir.name}: {key}={config.get(key)} makes the system prompt dynamic")
    if config.get("strategy") not in MODEL_FREE_STRATEGIES:
        raise RerenderError(f"{run_dir.name}: strategy {config.get('strategy')} needs model calls to rebuild")
    calls = _jsonl(run_dir / "calls.jsonl")
    ledger = result["budget"]["requests"]
    if len(ledger) != len(calls) or any(call.get("purpose") != "agent" for call in calls):
        raise RerenderError(f"{run_dir.name}: calls and budget ledger differ, or a non-agent call was made")
    events = [Event(**row) for row in _jsonl(run_dir / "trajectory.jsonl")]
    answered = [call for call in calls if call.get("response") is not None]
    if len(answered) != len(events) or any(event.model_content != call["response"]["content"]
                                           for event, call in zip(events, answered, strict=True)):
        raise RerenderError(f"{run_dir.name}: events do not match the recorded responses one to one")
    return result, calls, ledger, events


def _rebuild(result: dict, calls: list[dict], ledger: list[dict], events: list[Event], mode: str,
             counter: TokenCounter) -> list[list[dict[str, str]]]:
    config = result["config"]
    issue = calls[0]["messages"][1]["content"]
    if not issue.startswith(TASK_PREFIX):
        raise RerenderError("the recorded task message has an unexpected prefix")
    system = SYSTEM + (FIXED_LABELS_PROMPT if mode == "fixed" else "")
    manager = ContextManager(config["strategy"], counter, config.get("history_format", "chat"),
                             read_memory_tokens=config.get("read_memory_tokens", 1024), revision_labels=mode)
    context_cap = config["context_tokens"] - config["max_output_tokens"]
    requests, used, answered = [], 0, 0
    for call, row in zip(calls, ledger, strict=True):
        remaining = config["total_tokens"] - used
        cap = min(context_cap, remaining - config["min_action_output_tokens"])
        prefix = events[:answered]
        revision = prefix[-1].revision if prefix else result["initial_revision"]
        requests.append(manager.build(system, issue.removeprefix(TASK_PREFIX), prefix, revision, cap))
        used += row["input_tokens"] + row["output_tokens"]
        answered += call.get("response") is not None
    return requests


def rerender_requests(run_dir: Path, mode: str) -> list[list[dict[str, str]]]:
    """Every agent request of the run, rebuilt with revision labels `mode` ("relabel" or "fixed")."""
    if mode not in LABEL_MODES:
        raise ValueError(f"mode must be one of {LABEL_MODES}")
    result, calls, ledger, events = load_run(run_dir)
    counter = TokenCounter(result["config"].get("tokenizer") or "byte")
    own = recorded_mode(result["config"])
    replay = _rebuild(result, calls, ledger, events, own, counter)
    for index, (rebuilt, call) in enumerate(zip(replay, calls, strict=True)):
        if rebuilt != call["messages"]:
            raise RerenderError(f"{run_dir.name}: request {index} does not reproduce in its own mode ({own})")
    return replay if mode == own else _rebuild(result, calls, ledger, events, mode, counter)
