"""Tool capability decisions; execution isolation remains the sandbox's job."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .schema import RunConfig, Task


TOOLS = {"read_file", "list_files", "search", "write_file", "edit_file", "run", "public_test", "diff"}


def tool_decision(action: dict[str, Any], config: RunConfig, task: Task) -> dict[str, Any]:
    tool = action.get("tool")
    rule = "tool_allowed"
    if tool not in TOOLS:
        rule = "unknown_tool"
    elif tool == "run" and config.execution_policy == "restricted":
        rule = "free_shell_disabled"
    elif tool == "public_test":
        if set(action) != {"tool"}:
            rule = "public_test_arguments_not_allowed"
        elif not task.test_command:
            rule = "public_test_not_configured"
    return {"policy": config.execution_policy, "allowed": rule == "tool_allowed", "rule": rule}


def execute_authorized(action: dict[str, Any], config: RunConfig, task: Task,
                       execute: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    decision = tool_decision(action, config, task)
    if not decision["allowed"]:
        return {"ok": False, "output": f"Tool denied: {decision['rule']}",
                "duration_seconds": 0.0, "policy_decision": decision}
    result = execute(action)
    if result.get("policy_denial") == "repository_boundary":
        decision.update(allowed=False, rule="repository_boundary")
    result["policy_decision"] = decision
    return result
