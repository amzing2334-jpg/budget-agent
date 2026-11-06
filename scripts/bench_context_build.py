"""Time context construction on a synthetic trajectory; CPU only, no model calls.

Example:
    python scripts/bench_context_build.py --counter tiktoken:cl100k_base --counter byte

The trajectory has 30 steps: two thirds are 120-line read_file pages of one
file and the rest are 150-line command outputs, every fifth step failing.
Contexts are built after 5, 10, ..., 30 steps under one input limit. Use
--check-uncached to confirm that the memoized counter builds identical inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time

from budget_agent.budget import TokenCounter
from budget_agent.context import ContextManager, ContextOverflow
from budget_agent.schema import Event


def trajectory(seed: int = 0, steps: int = 30) -> list[Event]:
    rng = random.Random(seed)
    words = [f"tok{i}" for i in range(5000)]

    def lines(count: int) -> list[str]:
        return [" ".join(rng.choices(words, k=12)) for _ in range(count)]

    events = []
    for step in range(1, steps + 1):
        if step % 3:
            action = {"tool": "read_file", "path": "pkg/mod.py"}
            header = {"path": "pkg/mod.py", "total_lines": 400, "start_line": 1, "end_line": 120,
                      "next_start_line": 121, "truncated_line_content": False}
            output = "read_file " + json.dumps(header) + "\n" + "\n".join(
                f"{number}: {text}" for number, text in enumerate(lines(120), 1))
        else:
            action = {"tool": "run", "command": "pytest -q"}
            output = "\n".join(lines(150))
        events.append(Event(f"e{step:04d}", step, action, output, "rev0", ok=step % 5 != 0,
                            model_content=json.dumps(action), revision_before="rev0"))
    return events


class UncachedCounter(TokenCounter):
    def count(self, text: str) -> int:
        if self._tokens is None:
            return super().count(text)
        return self._tokens(text)


def build_all(counter: TokenCounter, strategy: str, events: list[Event], limit: int) -> list:
    manager = ContextManager(strategy, counter, read_memory_tokens=4096)
    built = []
    for count in range(5, len(events) + 1, 5):
        try:
            built.append(manager.build("system", "issue text", events[:count], "rev0", limit))
        except ContextOverflow as exc:
            built.append(f"ContextOverflow: {exc}")
    return built


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counter", action="append", default=None, help="byte or tiktoken:<encoding>")
    parser.add_argument("--strategy", action="append", default=None)
    parser.add_argument("--limit", type=int, default=60_000, help="max input tokens per request")
    parser.add_argument("--check-uncached", action="store_true")
    args = parser.parse_args()
    events = trajectory()
    for name in args.counter or ["byte"]:
        for strategy in args.strategy or ["recent", "structured", "mask-read"]:
            counter = TokenCounter(name)
            started = time.perf_counter()
            built = build_all(counter, strategy, events, args.limit)
            elapsed = time.perf_counter() - started
            line = f"{name:22s} {strategy:10s} {len(built)} builds: {elapsed * 1000:9.1f} ms"
            if args.check_uncached:
                reference = build_all(UncachedCounter(name), strategy, events, args.limit)
                digest = lambda value: hashlib.sha256(json.dumps(value).encode()).hexdigest()
                line += "  identical" if digest(built) == digest(reference) else "  DIFFERENT"
            print(line)


if __name__ == "__main__":
    main()
