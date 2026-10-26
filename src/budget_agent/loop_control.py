"""A narrow recovery policy for rereading evidence already in the model input.

Coverage in the archive alone is insufficient: masked evidence may legitimately
need reacquiring. Shell commands, recalls, truncated lines and inferred summaries
are deliberately outside this controller's evidence contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from .observations import ReadObservation, parse_read_observation
from .schema import Event


_EVENT_HEADER = re.compile(r"^\[event_id=([^\s\]]+)(?:\s|\])")


@dataclass(frozen=True)
class ReadRepeat:
    event_id: str
    revision: str
    path: str
    start_line: int
    end_line: int
    source_event_ids: list[str]
    consecutive_repeats: int


class ReadLoopMonitor:
    """Warn once per repeat segment; allow a model response before stopping.

    A repeat requires every returned, untruncated line to match an earlier read
    explicitly visible in the request that produced this action. Any other action
    or revision transition ends the segment. The default runner policy is off.
    """

    def __init__(self):
        self.revision = ""
        self.reads: dict[str, ReadObservation] = {}
        self.consecutive_repeats = 0
        self.pending_feedback: str | None = None
        self.warning_sent = False

    def reset_segment(self) -> None:
        self.consecutive_repeats = 0
        self.pending_feedback = None
        self.warning_sent = False

    def begin_revision(self, revision: str) -> None:
        if revision != self.revision:
            self.revision = revision
            self.reads.clear()
            self.reset_segment()

    def feedback_sent(self) -> None:
        self.pending_feedback = None
        self.warning_sent = True

    @property
    def should_stop(self) -> bool:
        return self.consecutive_repeats >= 2 and self.warning_sent

    def observe(self, event: Event, messages: list[dict[str, str]]) -> ReadRepeat | None:
        self.begin_revision(event.revision)
        current = parse_read_observation(event.output) if event.ok and event.action["tool"] == "read_file" else None
        if current is None or not current.lines or current.truncated_line_content is not False:
            self.reset_segment()
            return None

        covered: set[int] = set()
        sources: list[str] = []
        for message in messages:
            if message["role"] != "user":
                continue
            match = _EVENT_HEADER.match(message["content"])
            prior = self.reads.get(match[1]) if match else None
            if prior is None or prior.path != current.path:
                continue
            visible = set(message["content"].partition("Observation:\n")[2].splitlines())
            matched = {number for number, text in current.lines.items()
                       if prior.lines.get(number) == text and text in visible}
            if matched - covered:
                sources.append(match[1])
                covered.update(matched)

        self.reads[event.id] = current
        if covered != set(current.lines):
            self.reset_segment()
            return None
        self.consecutive_repeats += 1
        signal = ReadRepeat(event.id, event.revision, current.path, min(current.lines),
                            max(current.lines), sources, self.consecutive_repeats)
        if self.consecutive_repeats == 1:
            evidence = json.dumps({"event_id": signal.event_id, "path": signal.path,
                                   "lines": [signal.start_line, signal.end_line],
                                   "source_event_ids": signal.source_event_ids}, ensure_ascii=False)
            self.pending_feedback = (
                "Read-loop feedback (observed tool behavior, not a correctness judgment): " + evidence
                + ". Every returned line was already present in the request that selected this read. "
                "Use that evidence to resolve the issue, or inspect a specific missing fact. "
                "All tools remain available; do not invent an edit to satisfy this warning. "
                "Another consecutive read consisting entirely of already-visible lines will stop this attempt; "
                "the saved patch will still receive independent evaluation."
            )
        return signal
