"""Small shared contracts; no vendor SDK is required."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any


@dataclass
class Event:
    id: str
    step: int
    action: dict[str, Any]
    output: str
    revision: str
    ok: bool = True
    duration_seconds: float = 0.0
    model_content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelResponse:
    content: str
    input_tokens: int
    output_tokens: int
    usage_source: str = "provider"
    model: str = ""


@dataclass
class Task:
    id: str
    issue: str
    split: str
    kind: str = "local"
    repo: str = ""
    test_command: list[str] = field(default_factory=list)
    verifier: str = ""
    image: str = ""
    workdir: str = "/workspace"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    strategy: str = "structured"
    history_format: str = "chat"
    decision_policy: str = "direct"
    loop_policy: str = "off"
    provider: str = "scripted"
    model: str = ""
    base_url: str = ""
    api_key_env: str = "LLM_API_KEY"
    context_tokens: int = 8192
    total_tokens: int = 64000
    max_output_tokens: int = 1024
    min_action_output_tokens: int = 128
    read_memory_tokens: int = 1024
    max_steps: int = 30
    timeout_seconds: int = 30
    tokenizer: str = "byte"
    temperature: float = 0.0
    seed: int = 0
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    max_cost_usd: float | None = None
    allow_local_execution: bool = False
    environment: str = "local"
    docker_image: str = "python:3.11-slim"
    api_max_tokens_field: str = "max_tokens"
    api_enable_thinking: bool | None = None
    model_revision: str = ""
    device: str = "cuda"
    model_dtype: str = "bfloat16"

    def validate(self) -> None:
        if self.strategy not in {"full", "recent", "mask", "mask-read", "summary", "structured"}:
            raise ValueError("strategy must be full, recent, mask, mask-read, summary, or structured")
        if self.history_format not in {"transcript", "chat"}:
            raise ValueError("history_format must be transcript or chat")
        if self.decision_policy not in {"direct", "budget-aware"}:
            raise ValueError("decision_policy must be direct or budget-aware")
        if self.loop_policy not in {"off", "recover"}:
            raise ValueError("loop_policy must be off or recover")
        if self.provider not in {"scripted", "openai-compatible", "transformers"}:
            raise ValueError("unknown provider")
        if self.provider == "transformers":
            if not self.model or not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
                raise ValueError("transformers requires a model and a full 40-character model_revision commit")
            if self.model_dtype not in {"bfloat16", "float16", "float32"}:
                raise ValueError("model_dtype must be bfloat16, float16, or float32")
        for name in ("context_tokens", "total_tokens", "max_output_tokens", "min_action_output_tokens", "max_steps", "timeout_seconds"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.context_tokens <= self.max_output_tokens:
            raise ValueError("context_tokens must exceed max_output_tokens")
        if self.min_action_output_tokens > self.max_output_tokens:
            raise ValueError("min_action_output_tokens must not exceed max_output_tokens")
        if type(self.read_memory_tokens) is not int or self.read_memory_tokens < 0:
            raise ValueError("read_memory_tokens must be a non-negative integer")
        if self.environment not in {"local", "docker"}:
            raise ValueError("environment must be local or docker")
        if self.api_max_tokens_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("unsupported API output limit field")
        if self.api_enable_thinking is not None and type(self.api_enable_thinking) is not bool:
            raise ValueError("api_enable_thinking must be a boolean when specified")
        for name in ("input_price_per_million", "output_price_per_million", "max_cost_usd"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite non-negative number")
        if not isinstance(self.temperature, (int, float)) or not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if self.max_cost_usd is not None and (self.input_price_per_million is None or self.output_price_per_million is None):
            raise ValueError("a cost ceiling requires both input and output prices")
