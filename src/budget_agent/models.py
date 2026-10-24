"""Model boundary. Scripted responses are plumbing fixtures, never model results."""
from __future__ import annotations

import json
import http.client
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .schema import ModelResponse, RunConfig


class ModelError(RuntimeError):
    pass


class OpenAICompatibleModel:
    def __init__(self, config: RunConfig):
        self.config = config
        self.name = config.model
        if not self.name:
            raise ValueError("Set --model to the exact model/version served by your endpoint")
        parsed = urllib.parse.urlsplit(config.base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("--base-url must be an HTTP(S) API root, typically ending in /v1")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Do not put credentials, query parameters, or fragments in --base-url")
        self.key = os.environ.get(config.api_key_env, "")
        if not self.key and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError(f"Missing API key environment variable: {config.api_key_env}")
        self.url = config.base_url.rstrip("/") + "/chat/completions"

    def complete(self, messages: list[dict[str, str]], max_output_tokens: int, *, purpose: str = "agent") -> ModelResponse:
        body: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            self.config.api_max_tokens_field: max_output_tokens,
            "temperature": self.config.temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            # Do not automatically retry billed requests: a timeout has unknown usage.
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise ModelError("Provider response exceeds the 8 MB response limit; usage unknown")
            data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            # Provider error bodies can echo keys or full prompts. Never persist them.
            raise ModelError(f"Provider HTTP {exc.code}; request was not retried; usage may be unknown") from None
        except (OSError, json.JSONDecodeError, UnicodeError, http.client.HTTPException) as exc:
            raise ModelError(f"Provider transport/JSON failure ({type(exc).__name__}); usage unknown; no retry") from None
        try:
            content = data["choices"][0]["message"]["content"]
            usage = data["usage"]
            input_tokens, output_tokens = usage["prompt_tokens"], usage["completion_tokens"]
            actual_model = data.get("model", self.name)
            if not isinstance(content, str):
                raise ValueError("non-text completion")
            if not isinstance(actual_model, str) or not actual_model:
                raise ValueError("invalid model name")
            if any(type(x) is not int or x < 0 for x in (input_tokens, output_tokens)):
                raise ValueError("invalid token usage")
        except (KeyError, IndexError, TypeError, ValueError):
            raise ModelError("Endpoint must return text content, a string model name when present, and non-negative integer prompt_tokens/completion_tokens; usage unknown") from None
        return ModelResponse(content, input_tokens, output_tokens, "provider", actual_model)


class ScriptedModel:
    """Known actions used only to exercise file editing, budgets and graders offline.

    This class receives an explicit action script. It does not solve tasks, and its
    measurements must never be reported as LLM quality or SWE-bench performance.
    """

    name = "scripted-smoke-v1"

    def __init__(self, actions: list[dict[str, Any]], counter):
        self.actions = list(actions)
        self.counter = counter
        self.index = 0

    def complete(self, messages: list[dict[str, str]], max_output_tokens: int, *, purpose: str = "agent") -> ModelResponse:
        if purpose == "summary":
            text = "Offline fixture summary: inspect the source, preserve the issue requirements, verify the current revision before finishing."
        else:
            action = self.actions[self.index] if self.index < len(self.actions) else {"tool": "finish"}
            self.index += 1
            text = json.dumps(action, ensure_ascii=False)
        if self.counter.count(text) > max_output_tokens:
            # Simulate a genuinely truncated completion, not a free oversized action.
            while text and self.counter.count(text) > max_output_tokens:
                text = text[:-1]
        return ModelResponse(text, self.counter.messages(messages), self.counter.count(text), "scripted_estimate", self.name)


def parse_action(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        action = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("Return exactly one JSON object with a 'tool' field; no prose") from exc
    if not isinstance(action, dict) or not isinstance(action.get("tool"), str):
        raise ValueError("Action must be a JSON object with a string 'tool' field")
    allowed = {"list_files", "read_file", "search", "write_file", "edit_file", "run", "diff", "remember", "recall", "finish"}
    if action["tool"] not in allowed:
        raise ValueError("Unknown tool; allowed: " + ", ".join(sorted(allowed)))
    string_fields = {
        "read_file": ("path",), "search": ("query",),
        "write_file": ("path", "content"), "run": ("command",),
        "edit_file": ("path", "old", "new"),
        "remember": ("note", "source"), "recall": ("event_id",),
    }
    for field in string_fields.get(action["tool"], ()):
        if not isinstance(action.get(field), str):
            raise ValueError(f"{action['tool']} requires a string '{field}'")
    if action["tool"] in {"list_files", "search"} and "path" in action and not isinstance(action["path"], str):
        raise ValueError(f"{action['tool']} field 'path' must be a string")
    integer_fields = {
        "read_file": ("start_line", "end_line"), "recall": ("offset", "limit"),
        "list_files": ("offset", "limit"), "search": ("offset", "limit"),
    }
    for field in integer_fields.get(action["tool"], ()):
        if field in action and type(action[field]) is not int:
            raise ValueError(f"{action['tool']} field '{field}' must be an integer")
    return action
