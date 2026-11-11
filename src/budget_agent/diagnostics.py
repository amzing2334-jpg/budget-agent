"""Offline descriptions of recorded behavior, not model intent or root causes."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any

from .context import MASKED_OBSERVATION
from .evaluation import ledger_token_count
from .loop_control import ReadLoopMonitor
from .models import parse_action
from .observations import parse_read_observation
from .schema import Event


_BROWSE_TOOLS = {"list_files", "search", "read_file"}
_INTERNAL_TOOLS = {"invalid_action", "remember", "recall", "finish"}
_EVENT_HEADER = re.compile(r"^\[event_id=([^\s\]]+)(?:\s|\])")


def _response_shape(content: str | None) -> dict[str, Any]:
    """Describe JSON syntax independently of the recorded action validation."""
    if content is None:
        return {"json_values": None, "json_syntax": "unavailable"}
    candidate = content.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = "\n".join(candidate.splitlines()[1:-1]).strip()
    if not candidate:
        return {"json_values": 0, "json_syntax": "empty"}
    decoder = json.JSONDecoder()
    values = 0
    while candidate:
        try:
            _, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            return {"json_values": values, "json_syntax": "unparsed_text_remaining"}
        values += 1
        candidate = candidate[end:].lstrip()
    return {"json_values": values, "json_syntax": "complete"}


def _repetitions(events: Sequence[Event], *, observations: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Event]] = defaultdict(list)
    for event in events:
        tool = event.action["tool"]
        eligible = event.ok and tool in _BROWSE_TOOLS if observations else not event.ok and tool != "invalid_action"
        if not eligible:
            continue
        key: tuple[str, ...] = (event.revision, json.dumps(event.action, sort_keys=True, ensure_ascii=False))
        if observations:
            key += (event.output,)
        groups[key].append(event)
    return [
        {"tool": group[0].action["tool"], "action": group[0].action,
         "revision": group[0].revision, "first_event_id": group[0].id,
         "repeat_event_ids": [event.id for event in group[1:]], "repeat_count": len(group) - 1}
        for group in groups.values() if len(group) > 1
    ]


def _observation_exposure(events: Sequence[Event], calls: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sources = {event.id: event for event in events}
    exposure: dict[str, dict[str, Any]] = {}
    for call_index, call in enumerate(calls, 1):
        seen: set[str] = set()
        for message in call.get("messages", []):
            if message["role"] != "user":
                continue
            content = message["content"]
            # The transcript summarizer prefixes the same event serialization.
            if content.startswith("Prior summary:"):
                content = content.partition("\nNext event:\n")[2]
            match = _EVENT_HEADER.match(content)
            if match is None or match[1] not in sources:
                continue
            event_id = match[1]
            row = exposure.setdefault(event_id, {
                "event_id": event_id, "tool": sources[event_id].action["tool"],
                "original_output_characters": len(sources[event_id].output),
                "visible_observation_characters": 0, "request_indices": [],
                "masked_messages": 0, "masked_marker_characters": 0, "masked_request_indices": [],
                "associated_request_input_tokens": 0,
            })
            observation = content.partition("Observation:\n")[2]
            if observation.strip() == MASKED_OBSERVATION:
                row["masked_messages"] += 1
                row["masked_marker_characters"] += len(observation)
                if call_index not in row["masked_request_indices"]:
                    row["masked_request_indices"].append(call_index)
                continue
            row["visible_observation_characters"] += len(observation)
            if event_id not in seen:
                row["request_indices"].append(call_index)
                if "response" in call:
                    row["associated_request_input_tokens"] += call["response"]["input_tokens"]
                seen.add(event_id)
    return {
        "events": sorted(exposure.values(), key=lambda row: row["visible_observation_characters"], reverse=True),
        "visible_observation_characters": sum(row["visible_observation_characters"] for row in exposure.values()),
        "masked_messages": sum(row["masked_messages"] for row in exposure.values()),
        "masked_marker_characters": sum(row["masked_marker_characters"] for row in exposure.values()),
        "measurement": "Characters in explicit event observation messages, including excerpt markers but excluding masked placeholders. "
                       "Associated request tokens include all input, are not event token costs, and overlap across events. "
                       "Free-text summaries and recalled payloads are not attributed back to their source events.",
    }


def _intervals(lines: set[int]) -> list[list[int]]:
    intervals: list[list[int]] = []
    for number in sorted(lines):
        if intervals and number == intervals[-1][1] + 1:
            intervals[-1][1] = number
        else:
            intervals.append([number, number])
    return intervals


def _read_file_coverage(events: Sequence[Event]) -> dict[str, Any]:
    """Coverage of numbered tool output, never the requested read range."""
    seen: dict[tuple[str, str], set[int]] = defaultdict(set)
    rows = []
    for event in events:
        if event.action["tool"] != "read_file":
            continue
        row: dict[str, Any] = {"event_id": event.id, "revision": event.revision, "path": None,
               "status": "unknown", "returned_lines": None, "new_lines": None,
               "repeated_lines": None, "covered_intervals": None, "covered_interval_count": None}
        rows.append(row)
        if not event.ok:
            row["status"] = "failed"
            continue
        observation = parse_read_observation(event.output)
        if observation is None or not event.revision:
            continue
        path, numbers = observation.path, observation.lines
        previous = seen[(event.revision, path)]
        current = set(numbers)
        new, repeated = len(current - previous), len(current & previous)
        previous.update(current)
        intervals = _intervals(previous)
        row.update(path=path, status="measured", returned_lines=len(numbers), new_lines=new,
                   repeated_lines=repeated, covered_intervals=intervals,
                   covered_interval_count=len(intervals),
                   truncated_line_content=observation.truncated_line_content)
    measured = [row for row in rows if row["status"] == "measured"]
    return {
        "status": "partial" if any(row["status"] == "unknown" for row in rows) else "complete",
        "events": rows, "measured_events": len(measured),
        "unknown_events": sum(row["status"] == "unknown" for row in rows),
        "failed_events": sum(row["status"] == "failed" for row in rows),
        "measured_returned_lines": sum(row["returned_lines"] for row in measured),
        "measured_new_lines": sum(row["new_lines"] for row in measured),
        "measured_repeated_lines": sum(row["repeated_lines"] for row in measured),
        "paths": [{"revision": revision, "path": path, "unique_lines": len(lines),
                   "covered_intervals": _intervals(lines), "covered_interval_count": len(_intervals(lines))}
                  for (revision, path), lines in sorted(seen.items())],
        "measurement": "Actual numbered lines validated against read_file output header; grouped by revision and canonical output path. "
                       "New/repeated is relative to supported prior outputs only; unknown legacy formats are excluded, not guessed. "
                       "Line coverage does not establish complete line content, comprehension, or unnecessary rereading.",
    }


def _matches_response(event: Event, content: str) -> bool:
    if event.model_content is not None:
        return content == event.model_content
    # Older trajectories retained the parsed action but omitted its raw text.
    try:
        action = parse_action(content)
    except ValueError:
        return event.action == {"tool": "invalid_action"}
    if action["tool"] == "remember" and "source_revision" in event.action:
        action["source_revision"] = event.action["source_revision"]
    return action == event.action


def _usage_through_event(events: Sequence[Event], calls: Sequence[dict],
                         requests: list[dict] | None) -> tuple[dict | None, str | None]:
    """Match the actor turns and charge every model call through the last event."""
    inputs = outputs = event_index = 0
    for index, call in enumerate(calls, 1):
        response = call.get("response")
        purpose = call.get("purpose")
        if not isinstance(response, dict) or call.get("status") == "usage_unknown":
            return None, f"Missing model usage at request {index}"
        prompt = ledger_token_count(response, "input_tokens")
        completion = ledger_token_count(response, "output_tokens")
        if prompt is None or completion is None:
            return None, f"Invalid token counts at request {index}"
        if requests is not None:
            row = requests[index - 1] if isinstance(requests, list) and index <= len(requests) else None
            if (not isinstance(row, dict) or row.get("purpose") != purpose
                    or ledger_token_count(row, "input_tokens") != prompt
                    or ledger_token_count(row, "output_tokens") != completion):
                return None, f"Request {index} does not match the budget ledger"
        inputs += prompt
        outputs += completion
        if purpose in {"summary", "reflection"}:
            continue
        if purpose != "agent":
            return None, f"Unknown request purpose at request {index}"
        event = events[event_index]
        if event.request_index is not None and (type(event.request_index) is not int or event.request_index != index):
            return None, f"Request index mismatch for {event.id}"
        content = response.get("content")
        if not isinstance(content, str) or not _matches_response(event, content):
            return None, f"Agent response mismatch for {event.id} at request {index}"
        event_index += 1
        if event_index == len(events):
            return {"method": "request_index" if event.request_index is not None else "response_order",
                    "requests": index, "input_tokens": inputs, "output_tokens": outputs,
                    "total_tokens": inputs + outputs}, None
    return None, "Missing agent request through the changed snapshot"


def first_workspace_change(events: Sequence[Event], calls: Sequence[dict[str, Any]],
                           initial_revision: str | None, requests: list[dict] | None = None) -> dict[str, Any]:
    """First recorded post-step snapshot differing from the initial workspace."""
    if not initial_revision or not events:
        return {"status": "unavailable", "usage": None, "reason": "Missing initial revision or trajectory"}
    for step, event in enumerate(events, 1):
        if event.step != step or not event.revision:
            return {"status": "unavailable", "usage": None, "reason": f"Missing snapshot in step {step}"}
        if event.revision != initial_revision:
            usage, reason = _usage_through_event(events[:step], calls, requests)
            result = {"status": "observed", "event_id": event.id, "step": step, "tool": event.action["tool"],
                      "revision": event.revision, "usage": usage}
            if reason:
                result["reason"] = reason
            return result
    return {"status": "not_observed", "usage": None}


def analyze_trajectory(events: Sequence[Event], calls: Sequence[dict[str, Any]], *,
                       initial_revision: str | None = None, requests: list[dict] | None = None) -> dict[str, Any]:
    """Summarize Event records and existing calls.jsonl rows without inference.

    Calls are paired with events in successful agent-response order. Summary and
    unknown-usage calls do not produce events. Raw content mismatches are reported
    and prevent an output allowance from being attributed to the wrong event.
    """
    agent_calls = [(index, call) for index, call in enumerate(calls, 1)
                   if call["purpose"] == "agent" and "response" in call]
    tools: dict[str, dict[str, Any]] = {}
    invalid = []
    alignment_mismatches = []
    read_loop = ReadLoopMonitor()
    visible_read_repetitions = []
    first_successful = None
    for index, event in enumerate(events):
        tool = event.action["tool"]
        stats = tools.setdefault(tool, {
            "calls": 0, "successful": 0, "failed": 0,
            "duration_seconds": 0.0, "output_characters": 0,
        })
        stats["calls"] += 1
        stats["successful" if event.ok else "failed"] += 1
        stats["duration_seconds"] += event.duration_seconds
        stats["output_characters"] += len(event.output)
        if first_successful is None and event.ok and tool not in _INTERNAL_TOOLS:
            first_successful = {"event_id": event.id, "step": event.step, "tool": tool}
        paired = agent_calls[index] if index < len(agent_calls) else None
        if paired is not None and event.model_content is not None and event.model_content != paired[1]["response"]["content"]:
            alignment_mismatches.append({"event_id": event.id, "request_index": paired[0]})
            paired = None
        repeat = read_loop.observe(event, paired[1].get("messages", []) if paired else [])
        if repeat is not None:
            visible_read_repetitions.append(asdict(repeat))
        if tool != "invalid_action":
            continue
        content = event.model_content if event.model_content is not None else paired[1]["response"]["content"] if paired else None
        shape = _response_shape(content)
        allowance = paired[1].get("requested_max_output_tokens") if paired else None
        output_tokens = paired[1]["response"]["output_tokens"] if paired else None
        limit_reached = output_tokens >= allowance if allowance is not None else None
        invalid.append({
            "event_id": event.id, "step": event.step, "request_index": paired[0] if paired else None,
            "validation_error": event.output, **shape,
            "multiple_json_values": shape["json_values"] >= 2 if shape["json_values"] is not None else None,
            "output_tokens": output_tokens, "requested_max_output_tokens": allowance,
            "output_limit_reached": limit_reached,
            "unparsed_text_at_output_limit": limit_reached is True and shape["json_syntax"] == "unparsed_text_remaining",
        })
    responses = [call["response"] for call in calls if "response" in call]
    input_tokens = sum(response["input_tokens"] for response in responses)
    output_tokens = sum(response["output_tokens"] for response in responses)
    total_tokens = input_tokens + output_tokens
    failed_repeats = _repetitions(events, observations=False)
    observation_repeats = _repetitions(events, observations=True)
    return {
        "schema_version": 1,
        "counts": {
            "events": len(events), "invalid_actions": len(invalid),
            "invalid_actions_at_output_limit": sum(row["output_limit_reached"] is True for row in invalid),
            "repeated_failed_actions": sum(row["repeat_count"] for row in failed_repeats),
            "repeated_observations": sum(row["repeat_count"] for row in observation_repeats),
            **{f"successful_{tool}": sum(event.ok and event.action["tool"] == tool for event in events)
               for tool in ("read_file", "edit_file", "write_file", "run", "diff")},
        },
        "first_successful_tool_action": first_successful,
        "first_workspace_change": first_workspace_change(events, calls, initial_revision, requests),
        "invalid_actions": invalid,
        "repeated_failed_actions": failed_repeats,
        "repeated_observations": observation_repeats,
        "tools": tools,
        "usage": {
            "calls": len(calls), "calls_with_response": len(responses),
            "unknown_usage_calls": sum(call.get("status") == "usage_unknown" for call in calls),
            "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens,
            "input_fraction": input_tokens / total_tokens if total_tokens else None,
            "usage_sources": sorted({response["usage_source"] for response in responses}),
            "recorded_model_seconds": sum(call.get("elapsed_seconds", 0.0) for call in calls),
        },
        "call_alignment": {
            "method": "ordered_agent_responses", "agent_responses": len(agent_calls),
            "events": len(events), "raw_content_mismatches": alignment_mismatches,
            "unpaired_events": max(0, len(events) - len(agent_calls)),
            "unpaired_agent_responses": max(0, len(agent_calls) - len(events)),
        },
        "observation_exposure": _observation_exposure(events, calls),
        "read_file_coverage": _read_file_coverage(events),
        "visible_read_repetitions": visible_read_repetitions,
        "notes": [
            "Tool success and the first successful action do not establish a correct repair or a passed test.",
            "First workspace change compares recorded post-step snapshots with the initial fingerprint. "
            "Usage includes all requests through the action that precedes that snapshot.",
            "Repeated observations mean identical action, output, and recorded revision; they do not prove that repeating the check was unnecessary.",
            "Visible read repetitions compare complete returned line text with explicit evidence in the actual preceding request. Offline detection does not demonstrate recovery or token savings under a new policy.",
            "Reaching an output allowance is evidence of budget pressure, not proof that truncation caused the invalid action; both flags can coexist.",
            "Repeated failures count identical actions at the same recorded revision, excluding invalid_action; different failure messages may be present.",
        ],
    }


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def diagnose_run(run_dir: Path, *, result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read a run without writing artifacts; zero-call environment failures are valid."""
    if result is None:
        path = run_dir / "result.json"
        result = json.loads(path.read_bytes()) if path.exists() else {}
    return analyze_trajectory([Event(**row) for row in read_jsonl(run_dir / "trajectory.jsonl")],
                              read_jsonl(run_dir / "calls.jsonl"),
                              initial_revision=result.get("initial_revision"),
                              requests=result.get("budget", {}).get("requests"))
