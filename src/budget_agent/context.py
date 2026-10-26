"""Context policies with explicit provenance and inspectable compaction.

The complete Event list stays with the runner. These policies only construct the
next model input; they never erase the canonical trajectory or the task issue.
"""
from __future__ import annotations

import json
from typing import Callable

from .budget import TokenCounter
from .observations import parse_read_observation
from .schema import Event


class ContextOverflow(RuntimeError):
    pass


Summarizer = Callable[[list[dict[str, str]], int], str]
MASK_KEEP_RECENT = 2
MASKED_OBSERVATION = "[observation masked; full output available by recall]"


def _clip(text: str, limit: int, counter: TokenCounter, *, tail: bool = False) -> str:
    if counter.count(text) <= limit:
        return text
    marker = "\n[excerpt; full output available by recall]\n"
    if counter.count(marker) >= limit:
        return ""
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        part = text[-mid:] if tail and mid else text[:mid]
        value = marker + part if tail else part + marker
        if counter.count(value) <= limit:
            low = mid
        else:
            high = mid - 1
    part = text[-low:] if tail and low else text[:low]
    return marker + part if tail else part + marker


def _event_header(event: Event, revision: str) -> str:
    # Recording an old observation as a fresh note does not make it current.
    # The runner sets source_revision from the referenced evidence, not the model.
    source_revision = event.action.get("source_revision", event.revision) if event.action.get("tool") == "remember" else event.revision
    stale = source_revision != revision
    status = "stale: observed at an older revision; revalidate before relying on it" if stale else "current revision"
    return f"[event_id={event.id} step={event.step} revision={event.revision} source_revision={source_revision} {status} ok={event.ok}]\n"


def _event_text(event: Event, revision: str, *, compact: bool = False, observation: str | None = None) -> str:
    action = dict(event.action)
    # Full write contents are already on disk; repeated inclusion crowds out tests.
    if compact and action.get("tool") == "write_file" and "content" in action:
        action["content"] = "[written content omitted; read_file or diff to inspect]"
    return (
        _event_header(event, revision)
        + f"Action: {json.dumps(action, ensure_ascii=False, sort_keys=True)}\n"
        f"Observation:\n{event.output if observation is None else observation}"
    )


class ContextManager:
    def __init__(self, strategy: str, counter: TokenCounter, history_format: str = "chat", *, read_memory_tokens: int = 1024):
        if strategy not in {"full", "recent", "mask", "mask-read", "summary", "structured"}:
            raise ValueError("Unknown context strategy")
        self.strategy = strategy
        if history_format not in {"transcript", "chat"}:
            raise ValueError("history_format must be transcript or chat")
        self.history_format = history_format
        self.counter = counter
        self.read_memory_tokens = read_memory_tokens
        self._summary = ""
        self._summarized_ids: set[str] = set()
        self.stats = {"builds": 0, "compactions": 0, "summary_calls": 0, "excerpted_events": 0, "last_input_tokens": 0, "last_visible_events": 0, "last_omitted_events": 0, "last_masked_events": 0,
                      "last_read_memory_tokens": 0, "last_read_memory_lines": 0, "last_read_memory_events": 0}

    def _event_messages(self, event: Event, revision: str, *, observation: str | None = None) -> list[dict[str, str]]:
        if self.history_format == "transcript":
            return [{"role": "user", "content": _event_text(event, revision, observation=observation)}]
        if event.model_content is None:
            raise ValueError("Chat history requires original model_content; use transcript for legacy events or restore it from calls.jsonl")
        return [
            {"role": "assistant", "content": event.model_content},
            {"role": "user", "content": _event_header(event, revision) + "Observation:\n" + (event.output if observation is None else observation)},
        ]

    def _history(self, events: list[Event], revision: str) -> list[dict[str, str]]:
        return [message for event in events for message in self._event_messages(event, revision)]

    def _clip_chat_observation(self, group: list[dict[str, str]], base: list[dict[str, str]], limit: int, *, tail: bool = False) -> list[dict[str, str]]:
        """Retain the exact assistant turn and provenance; excerpt only tool output."""
        if self.counter.messages(base + group) <= limit:
            return group
        prefix, _, observation = group[-1]["content"].partition("Observation:\n")
        prefix += "Observation:\n"
        marker = "\n[excerpt; full output available by recall]\n"

        def candidate(length: int) -> list[dict[str, str]]:
            part = observation[-length:] if tail and length else observation[:length]
            text = marker + part if tail else part + marker
            return group[:-1] + [{"role": "user", "content": prefix + text}]

        if self.counter.messages(base + candidate(0)) > limit:
            return []
        low, high = 0, len(observation)
        while low < high:
            middle = (low + high + 1) // 2
            if self.counter.messages(base + candidate(middle)) <= limit:
                low = middle
            else:
                high = middle - 1
        return candidate(low)

    def build(self, system: str, issue: str, events: list[Event], revision: str,
              max_input_tokens: int, summarize_callback: Summarizer | None = None) -> list[dict[str, str]]:
        self.stats["builds"] += 1
        base = [{"role": "system", "content": system}, {"role": "user", "content": "Task requirements (preserve in full):\n" + issue}]
        if self.counter.messages(base) > max_input_tokens:
            raise ContextOverflow("System instructions and complete task issue do not fit; increase context budget")
        if self.strategy == "mask":
            return self._build_mask(base, events, revision, max_input_tokens)
        if self.strategy == "mask-read":
            return self._build_mask_reads(base, events, revision, max_input_tokens)
        full = base + self._history(events, revision)
        if self.strategy == "full":
            if self.counter.messages(full) > max_input_tokens:
                raise ContextOverflow("Full-history baseline reached its context budget; no hidden truncation")
            return self._finish(full, len(events), len(events))
        if self.strategy == "summary":
            return self._build_summary(base, events, revision, max_input_tokens, summarize_callback)
        if self.counter.messages(full) <= max_input_tokens:
            return self._finish(full, len(events), len(events))
        if self.strategy == "recent":
            return self._build_recent(base, events, revision, max_input_tokens)
        return self._build_structured(base, events, revision, max_input_tokens)

    def _finish(self, messages: list[dict[str, str]], visible: int, total: int) -> list[dict[str, str]]:
        self.stats["last_input_tokens"] = self.counter.messages(messages)
        self.stats["last_visible_events"] = visible
        self.stats["last_omitted_events"] = max(0, total - visible)
        return messages

    def _build_mask(self, base: list[dict[str, str]], events: list[Event], revision: str, limit: int) -> list[dict[str, str]]:
        """Keep every action/header, but only the latest two observation bodies.

        Mask even when full history fits: saving repeated inputs is the policy.
        No ranking, summary calls, action truncation, or sliding-window fallback.
        """
        masked = max(0, len(events) - MASK_KEEP_RECENT)
        messages = base + [message for index, event in enumerate(events)
                           for message in self._event_messages(
                               event, revision, observation=MASKED_OBSERVATION if index < masked else None)]
        self.stats["last_masked_events"] = masked
        self.stats["compactions"] += int(masked > 0)
        if self.counter.messages(messages) > limit:
            raise ContextOverflow("Observation masking cannot fit complete actions, provenance and latest two observations")
        return self._finish(messages, len(events), len(events))

    def _build_mask_reads(self, base: list[dict[str, str]], events: list[Event], revision: str, limit: int) -> list[dict[str, str]]:
        """Spend a bounded extra input allowance on older, unique source lines.

        Keep the mask baseline's action history and two recent observations intact.
        Select using the past prefix only: newest reads in the current continuous
        revision segment first, then complete line prefixes. No model summaries.
        """
        messages = self._build_mask(base, events, revision, limit)
        baseline_tokens = self.counter.messages(messages)
        capacity = min(limit, baseline_tokens + self.read_memory_tokens)
        self.stats.update(last_read_memory_tokens=0, last_read_memory_lines=0, last_read_memory_events=0)
        if self.read_memory_tokens == 0:
            return messages

        seen: set[tuple[str, int]] = set()
        newest_text: dict[tuple[str, int], str] = {}
        masked = max(0, len(events) - MASK_KEEP_RECENT)
        stride = 2 if self.history_format == "chat" else 1
        for index in range(len(events) - 1, -1, -1):
            event = events[index]
            # A -> B -> A starts a new segment, even if the hash matches old A.
            if event.revision != revision:
                break
            if not event.ok or event.action["tool"] != "read_file":
                continue
            observed = parse_read_observation(event.output)
            if observed is None or observed.truncated_line_content is not False:
                continue
            for number, text in observed.lines.items():
                newest_text.setdefault((observed.path, number), text)
            if index >= masked:
                seen.update((observed.path, number) for number in observed.lines)
                continue
            rows = [(number, text) for number, text in observed.lines.items()
                    if (observed.path, number) not in seen and newest_text[(observed.path, number)] == text]
            if not rows:
                continue
            position = len(base) + (index + 1) * stride - 1

            def candidate(count: int) -> list[dict[str, str]]:
                prefix = "read_file excerpt " + json.dumps({"path": observed.path, "source_event_id": event.id,
                                                           "retained_lines": count}, ensure_ascii=False)
                observation = prefix + "\n[omitted lines available by recall]\n" + "\n".join(text for _, text in rows[:count])
                group = self._event_messages(event, revision, observation=observation)
                trial = list(messages)
                trial[position] = group[-1]
                return trial

            low, high = 0, len(rows)
            while low < high:
                middle = (low + high + 1) // 2
                if self.counter.messages(candidate(middle)) <= capacity:
                    low = middle
                else:
                    high = middle - 1
            if low:
                messages = candidate(low)
                seen.update((observed.path, number) for number, _ in rows[:low])
                self.stats["last_read_memory_lines"] += low
                self.stats["last_read_memory_events"] += 1
                self.stats["excerpted_events"] += 1
        self.stats["last_read_memory_tokens"] = self.counter.messages(messages) - baseline_tokens
        self.stats["last_masked_events"] -= self.stats["last_read_memory_events"]
        return self._finish(messages, len(events), len(events))

    def _build_recent(self, base: list[dict[str, str]], events: list[Event], revision: str, limit: int) -> list[dict[str, str]]:
        """Keep a contiguous suffix of whole events; excerpt an oversized latest event.

        This is an explicit sliding-window baseline, not the unmodified full-history
        mini-SWE-agent. It neither ranks evidence nor calls a summarizing model.
        """
        self.stats["compactions"] += 1
        selected: list[dict[str, str]] = []
        visible = 0
        for event in reversed(events):
            group = self._event_messages(event, revision)
            if self.counter.messages(base + group + selected) <= limit:
                selected[:0] = group
                visible += 1
                continue
            if self.history_format == "chat":
                if not selected:
                    clipped_group = self._clip_chat_observation(group, base, limit, tail=not event.ok)
                    if clipped_group:
                        selected.extend(clipped_group)
                        visible += 1
                        self.stats["excerpted_events"] += 1
                break
            if not selected:
                text = group[0]["content"]
                framing = self.counter.messages(base + [{"role": "user", "content": ""}])
                available = limit - framing
                prefix, _, observation = text.partition("Observation:\n")
                prefix += "Observation:\n"
                if self.counter.count(prefix) < available:
                    excerpt = _clip(observation, available - self.counter.count(prefix), self.counter, tail=not event.ok)
                    clipped = prefix + excerpt if excerpt else _clip(text, available, self.counter)
                else:
                    clipped = _clip(text, available, self.counter)
                candidate = {"role": "user", "content": clipped}
                if clipped and self.counter.messages(base + [candidate]) <= limit:
                    selected.append(candidate)
                    visible += 1
                    self.stats["excerpted_events"] += 1
            break
        if events and not selected:
            raise ContextOverflow("No recent event evidence fits after preserving the task; increase context budget")
        return self._finish(base + selected, visible, len(events))

    def _build_structured(self, base: list[dict[str, str]], events: list[Event], revision: str, limit: int) -> list[dict[str, str]]:
        self.stats["compactions"] += 1
        notice = {"role": "user", "content": (
            "Working memory: excerpts selected from the execution trace. Facts from older revisions are unverified now. "
            'Use {"tool":"recall","event_id":"..."} for original evidence. '
            "Do not treat a remembered claim as a passed test."
        )}
        messages = list(base)
        if self.counter.messages(messages + [notice]) + 250 < limit:
            messages.append(notice)
        priorities = []
        for e in reversed(events[-2:]):
            priorities.append(e)
        priorities += [e for e in reversed(events) if e.action.get("tool") == "remember"]
        priorities += [e for e in reversed(events) if not e.ok]
        priorities += [e for e in reversed(events) if e.action.get("tool") == "write_file"]
        priorities += list(reversed(events))
        selected: dict[str, list[dict[str, str]]] = {}
        # Divide the free budget to avoid one verbose event consuming all memory.
        free = max(0, limit - self.counter.messages(messages))
        slots = min(max(len(events), 1), 4)
        framing = self.counter.messages(messages + [{"role": "user", "content": ""}]) - self.counter.messages(messages)
        per_event = max(160, min(1600, (free - framing * slots) // slots))
        for event in priorities:
            if event.id in selected:
                continue
            if self.history_format == "chat":
                group = self._event_messages(event, revision)
                # Allocate excerpts without charging the full assistant turn to
                # the observation allowance; the complete input still must fit.
                prefix = group[-1]["content"].partition("Observation:\n")[0] + "Observation:\n"
                fixed_group = group[:-1] + [{"role": "user", "content": prefix}]
                excerpt_limit = min(limit, self.counter.messages(messages + fixed_group) + per_event)
                clipped_group = self._clip_chat_observation(group, messages, excerpt_limit, tail=not event.ok)
                if clipped_group:
                    selected[event.id] = clipped_group
                    messages.extend(clipped_group)
                    self.stats["excerpted_events"] += int(clipped_group != group)
                continue
            text = _event_text(event, revision, compact=True)
            available = min(per_event, limit - self.counter.messages(messages) - framing)
            prefix, _, observation = text.partition("Observation:\n")
            # Keep provenance at the front and the failure traceback's tail.
            if self.counter.count(prefix + "Observation:\n") < available:
                prefix += "Observation:\n"
                clipped = prefix + _clip(observation, available - self.counter.count(prefix), self.counter, tail=not event.ok)
            else:
                clipped = _clip(text, available, self.counter)
            if clipped != text:
                self.stats["excerpted_events"] += 1
            if not clipped:
                continue
            candidate = {"role": "user", "content": clipped}
            if self.counter.messages(messages + [candidate]) <= limit:
                selected[event.id] = [candidate]
                messages.append(candidate)
        if events and not selected:
            raise ContextOverflow("No event evidence fits after preserving the task; increase context budget")
        # Chronological order is easier to audit, regardless of selection priority.
        base_count = len(base) + int(len(messages) > len(base) and messages[len(base)] == notice)
        messages = messages[:base_count] + [message for e in events if e.id in selected for message in selected[e.id]]
        return self._finish(messages, len(selected), len(events))

    def _build_summary(self, base: list[dict[str, str]], events: list[Event], revision: str, limit: int, callback: Summarizer | None) -> list[dict[str, str]]:
        pending = [e for e in events if e.id not in self._summarized_ids]
        def materialize() -> list[dict[str, str]]:
            summary = [{"role": "user", "content": "Rolling summary (may contain stale claims; verify against current files/tests):\n" + self._summary}] if self._summary else []
            return base + summary + self._history(pending, revision)
        messages = materialize()
        if self.counter.messages(messages) <= limit:
            return self._finish(messages, len(pending), len(events))
        if callback is None:
            raise ContextOverflow("Summary policy needs an accounted model callback when compaction is required")
        self.stats["compactions"] += 1
        # Each iteration consumes at least one previously unsummarized event.
        while self.counter.messages(messages) > limit and pending:
            event = pending.pop(0)
            summary_system = "Summarize execution evidence for a coding agent. Preserve task constraints, file paths, failed attempts, test outcomes, event IDs, and revision IDs. Mark claims as uncertain; never invent passed tests. Return concise plain text."
            prefix = "Prior summary:\n" + self._summary + "\nNext event:\n"
            if self.history_format == "chat":
                summary_base = [{"role": "system", "content": summary_system}, {"role": "user", "content": prefix}]
                group = self._event_messages(event, revision)
                clipped_group = self._clip_chat_observation(group, summary_base, limit, tail=not event.ok)
                if not clipped_group:
                    raise ContextOverflow("Complete assistant event and provenance do not fit the compaction prompt")
                summary_input = summary_base + clipped_group
                self.stats["excerpted_events"] += int(clipped_group != group)
            else:
                summary_input = [{"role": "system", "content": summary_system}, {"role": "user", "content": prefix + _event_text(event, revision)}]
                # Summary invocation must itself fit the same input ceiling.
                overhead = self.counter.messages([{"role": "system", "content": summary_system}, {"role": "user", "content": ""}])
                available = limit - overhead
                if available < 80:
                    raise ContextOverflow("Context is too small for the compaction prompt")
                summary_input[1]["content"] = _clip(prefix + _event_text(event, revision), available, self.counter)
            target = max(32, min(512, (limit - self.counter.messages(base)) // 3))
            self._summary = callback(summary_input, target)
            if not isinstance(self._summary, str) or not self._summary.strip():
                raise ContextOverflow("Summarizer returned no usable text")
            self.stats["summary_calls"] += 1
            self._summarized_ids.add(event.id)
            messages = materialize()
        if self.counter.messages(messages) > limit:
            # Never silently remove the issue to accommodate a summary.
            raise ContextOverflow("Even the compacted summary does not fit alongside the complete task issue")
        return self._finish(messages, len(pending), len(events))
