"""Auditable token accounting shared by action and compaction calls.

Preflight token counts are estimates, even with a known tokenizer: the server's
chat template is not necessarily public. Actual usage is reconciled immediately.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable

from .schema import ModelResponse, RunConfig


class BudgetExceeded(RuntimeError):
    """No request may be made, or a completed request exceeded its reservation."""


class TokenCounter:
    """A reproducible preflight counter, never advertised as provider billing.

    ``byte`` counts UTF-8 bytes plus conservative message framing. It usually
    overestimates text token usage and is useful without a tokenizer dependency,
    but it does not bound real tokens: dense code runs near 2.5 bytes per token,
    so a byte budget can still overflow the server's context window.
    ``hf-file:<path>`` counts with the served model's own ``tokenizer.json``
    (the ``tokenizers`` package), which makes a token budget hold. A tiktoken
    encoding can be selected explicitly (e.g. ``cl100k_base``). Chat framing
    is estimated in every case.
    """

    # Context policies binary-search excerpt lengths and re-count the same
    # messages at every probe. Memoize tokenizer counts by content digest:
    # a few dozen bytes per entry, however long the text is.
    _CACHE_ENTRIES = 65_536

    def __init__(self, name: str = "byte"):
        self.name = name
        self._tokens: Callable[[str], int] | None = None
        self._counts: dict[bytes, int] = {}
        if name.startswith("hf-file:"):
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise ValueError("Install budget-agent[tokens] to count with a tokenizer.json file") from exc
            try:
                tokenizer = Tokenizer.from_file(name.removeprefix("hf-file:"))
            except Exception as exc:  # the Rust binding raises a bare Exception for unreadable files
                raise ValueError(f"Cannot load tokenizer file: {name.removeprefix('hf-file:')}") from exc
            self._tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
        elif name != "byte":
            try:
                import tiktoken
            except ImportError as exc:
                raise ValueError("Install budget-agent[tokens] to use a tiktoken encoding") from exc
            try:
                encoding = tiktoken.get_encoding(name.removeprefix("tiktoken:"))
            except ValueError as exc:
                raise ValueError(f"Unknown explicit tiktoken encoding: {name}") from exc
            self._tokens = lambda text: len(encoding.encode(text, disallowed_special=()))

    def count(self, text: str) -> int:
        if self._tokens is None:
            return len(text.encode("utf-8"))
        key = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).digest()
        count = self._counts.get(key)
        if count is None:
            if len(self._counts) >= self._CACHE_ENTRIES:
                self._counts.clear()
            count = self._counts[key] = self._tokens(text)
        return count

    def messages(self, messages: list[dict[str, str]]) -> int:
        # Plain role/content messages only; tool schemas and images are not used.
        return 8 + sum(12 + self.count(m["role"]) + self.count(m["content"]) for m in messages)

    @property
    def description(self) -> str:
        return "UTF-8 byte upper estimate + framing estimate" if self.name == "byte" else f"{self.name} tokens + framing estimate"


@dataclass
class _Reservation:
    input_tokens: int
    output_tokens: int


class Budget:
    """One in-flight request at a time, using an already validated run config."""

    def __init__(self, counter: TokenCounter, config: RunConfig):
        self.counter = counter
        self.config = config
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.records: list[dict[str, Any]] = []
        self.usage_sources: set[str] = set()
        self.overrun = False
        self._pending: _Reservation | None = None

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.config.total_tokens - self.input_tokens - self.output_tokens)

    @property
    def cost_usd(self) -> float | None:
        if self.config.input_price_per_million is None or self.config.output_price_per_million is None:
            return None
        return (self.input_tokens * self.config.input_price_per_million + self.output_tokens * self.config.output_price_per_million) / 1_000_000

    def reserve(self, messages: list[dict[str, str]], max_output_tokens: int, *, min_output_tokens: int = 1) -> int:
        """Return an allowed output cap, or fail before any billable request."""
        if self.overrun:
            raise BudgetExceeded("Previous provider usage exceeded its reservation; execution halted")
        if self._pending is not None:
            raise RuntimeError("A request is already reserved; reconcile it before reserving another")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if type(min_output_tokens) is not int or not 1 <= min_output_tokens <= max_output_tokens:
            raise ValueError("min_output_tokens must be between 1 and max_output_tokens")
        inputs = self.counter.messages(messages)
        allowed = min(max_output_tokens, self.remaining_tokens - inputs, self.config.context_tokens - inputs)
        if self.config.max_cost_usd is not None:
            remaining_cost = self.config.max_cost_usd - self.cost_usd
            after_input = remaining_cost - inputs * self.config.input_price_per_million / 1_000_000
            if after_input < 0:
                raise BudgetExceeded("Estimated input cost exceeds remaining cost budget")
            if self.config.output_price_per_million:
                allowed = min(allowed, math.floor(after_input * 1_000_000 / self.config.output_price_per_million + 1e-9))
        if allowed < min_output_tokens:
            raise BudgetExceeded(f"Remaining total/context/cost budget permits {max(0, allowed)} output tokens; this call requires at least {min_output_tokens}")
        self._pending = _Reservation(inputs, allowed)
        return allowed

    def record(self, response: ModelResponse, purpose: str = "agent") -> None:
        """Reconcile once, keeping any real overrun visible in the run artifact."""
        pending = self._pending
        if pending is None:
            raise RuntimeError("record() requires a preceding reserve()")
        if any(type(n) is not int or n < 0 for n in (response.input_tokens, response.output_tokens)):
            raise ValueError("Usage must contain non-negative integer token counts")
        if not response.usage_source:
            raise ValueError("Usage source is required; unknown usage must halt the run")
        self._pending = None
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.calls += 1
        self.usage_sources.add(response.usage_source)
        reasons = []
        if response.output_tokens > pending.output_tokens:
            reasons.append("provider output exceeded reserved output cap")
        if response.input_tokens + response.output_tokens > pending.input_tokens + pending.output_tokens:
            reasons.append("provider usage exceeded estimated reservation")
        if response.input_tokens + response.output_tokens > self.config.context_tokens:
            reasons.append("provider usage exceeded per-call context limit")
        if self.input_tokens + self.output_tokens > self.config.total_tokens:
            reasons.append("actual cumulative usage exceeded total token limit")
        if self.config.max_cost_usd is not None and self.cost_usd > self.config.max_cost_usd + 1e-12:
            reasons.append("actual cumulative usage exceeded cost limit")
        self.records.append({
            "purpose": purpose,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "estimated_input_tokens": pending.input_tokens,
            "reserved_output_tokens": pending.output_tokens,
            "usage_source": response.usage_source,
            "model": response.model,
            "overrun_reasons": reasons,
        })
        if reasons:
            self.overrun = True
            raise BudgetExceeded("; ".join(reasons))

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "remaining_tokens": self.remaining_tokens,
            "calls": self.calls,
            "cost_usd": self.cost_usd,
            "cost_kind": "configured_price_estimate" if self.cost_usd is not None else "unknown_prices",
            "usage_sources": sorted(self.usage_sources),
            "preflight_counter": self.counter.name,
            "preflight_description": self.counter.description,
            "usage_is_estimated": any(source not in {"provider", "tokenizer_exact"} for source in self.usage_sources),
            "overrun": self.overrun,
            "unreconciled_request": self._pending is not None,
            "attempted_calls": self.calls + int(self._pending is not None),
            "pending_reservation": ({"estimated_input_tokens": self._pending.input_tokens, "reserved_output_tokens": self._pending.output_tokens} if self._pending else None),
            "requests": list(self.records),
        }
