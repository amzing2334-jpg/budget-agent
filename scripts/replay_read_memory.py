"""CPU-only fixed-prefix analysis of mask versus mask-read; never calls a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import mean

from budget_agent.budget import TokenCounter
from budget_agent.context import ContextManager, ContextOverflow
from budget_agent.observations import parse_read_observation
from budget_agent.runner import source_fingerprint
from budget_agent.schema import Event


TASK_PREFIX = "Task requirements (preserve in full):\n"
ROOT = Path(__file__).resolve().parents[1]
EVENT_HEADER = re.compile(r"^\[event_id=(\S+) step=\d+ revision=(\S+) source_revision=(\S+) current revision ok=True\]\n")
LIMITATIONS = [
    "Fixed recorded prefixes only: no new actions, model calls, repair successes, training or billed usage.",
    "Analysis token estimates use the selected encoding and estimated chat framing; they are not exact Qwen tokens or provider billing.",
    "Each prefix uses config.context_tokens - config.max_output_tokens, not the historical remaining total-budget reservation.",
    "Extra input inclusion costs do not establish whole-rollout savings or how a model would act after seeing different context.",
    "Overflow prefixes remain in the report with unknown visibility; they are not silently dropped or counted as zero.",
    "Repeated histories from the same task are not independent tasks; no statistical success-rate claim is made.",
    "Coverage means exact complete numbered source lines observed earlier in the current contiguous revision segment, not semantic usefulness.",
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_hash(value: object) -> str:
    return _sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_path(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else str(resolved)


def _read(event: Event):
    if not event.ok or event.action.get("tool") != "read_file":
        return None
    observed = parse_read_observation(event.output)
    return observed if observed is not None and observed.truncated_line_content is False else None


def _current_reads(events: list[Event], revision: str) -> dict:
    sources = {}
    for event in reversed(events):
        if event.revision != revision:
            break
        observed = _read(event)
        if observed is not None:
            sources[event.id] = (event, observed)
    return sources


def _known_target_lines(target: Event, sources: dict) -> tuple[dict | None, str]:
    if target.action.get("tool") != "read_file":
        return None, "not_read_file"
    observed = _read(target)
    if observed is None:
        return None, "unknown_or_incomplete_read_output"
    # Sources are newest first. Conflicting older content cannot override it.
    latest = {}
    for _, source in sources.values():
        for number, text in source.lines.items():
            latest.setdefault((source.path, number), text)
    known = {(observed.path, number): text for number, text in observed.lines.items()
             if latest.get((observed.path, number)) == text}
    return known, "complete_read_output"


def _visible_known_lines(messages: list[dict], known: dict, sources: dict, revision: str) -> int:
    visible = set()
    for message in messages:
        if message["role"] != "user":
            continue
        header = EVENT_HEADER.match(message["content"])
        if header is None or header[1] not in sources or header[2] != revision or header[3] != revision:
            continue
        event, source = sources[header[1]]
        _, separator, observation = message["content"].partition("Observation:\n")
        if not separator:
            continue
        first, _, body = observation.partition("\n")
        if first.startswith("read_file excerpt "):
            metadata = json.loads(first.removeprefix("read_file excerpt "))
            if metadata["path"] != source.path or metadata["source_event_id"] != event.id:
                continue
            rows = body.splitlines()[1:]  # explicit omitted-lines marker
        else:
            parsed = parse_read_observation(observation)
            if parsed is None or parsed.path != source.path or parsed.truncated_line_content is not False:
                continue
            rows = parsed.lines.values()
        # Match only numbered rows of the source event. Identical text from
        # another path, assistant action or task issue is not read evidence.
        row_numbers = {text: number for number, text in source.lines.items()}
        for text in rows:
            number = row_numbers.get(text)
            key = (source.path, number)
            if number is not None and known.get(key) == text:
                visible.add(key)
    return len(visible)


def _call_task(call: dict) -> tuple[str, str]:
    messages = call["messages"]
    if len(messages) < 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user" or not messages[1]["content"].startswith(TASK_PREFIX):
        raise ValueError("Expected actual system and complete task messages at the start of the recorded call")
    return messages[0]["content"], messages[1]["content"].removeprefix(TASK_PREFIX)


def replay_run(run_dir: Path, counter: TokenCounter, read_memory_tokens: int = 1024) -> dict:
    """Rebuild each input before consulting its next action for diagnostics."""
    paths = {name: run_dir / name for name in ("result.json", "calls.jsonl", "trajectory.jsonl")}
    result = json.loads(paths["result.json"].read_text(encoding="utf-8"))
    record = {"run_directory": _source_path(run_dir), "task_id": result["task_id"],
              "task_fingerprint": result.get("task_fingerprint"), "run_id": result.get("run_id")}
    config = result["config"]
    harness = result.get("harness") or config.get("harness")
    if harness:
        return {**record, "status": "excluded", "reason": f"External harness {harness}; not an internal mask policy run"}
    if result.get("strategy") != "mask" or config.get("strategy") != "mask":
        return {**record, "status": "excluded", "reason": f"Original strategy result={result.get('strategy')}, config={config.get('strategy')}; only mask is replayed"}
    sources = {name: {"path": _source_path(path), "sha256": _sha256(path.read_bytes())} for name, path in paths.items()}
    raw_calls = _read_jsonl(paths["calls.jsonl"])
    calls = [(index, call) for index, call in enumerate(raw_calls, 1)
             if call.get("purpose") == "agent" and call.get("status") != "usage_unknown"
             and call.get("response") is not None and call["response"].get("usage_source") not in {None, "unknown", "usage_unknown"}]
    raw_events = _read_jsonl(paths["trajectory.jsonl"])
    events = [Event(**row) for row in raw_events]
    if len(calls) != len(events):
        raise ValueError(f"{run_dir}: agent response/event count mismatch ({len(calls)} != {len(events)})")
    if len({event.id for event in events}) != len(events):
        raise ValueError(f"{run_dir}: duplicate event IDs prevent provenance matching")
    for (_, call), event in zip(calls, events):
        if event.model_content is None or event.model_content != call["response"]["content"]:
            raise ValueError(f"{run_dir}: raw response mismatch for {event.id}; refusing positional repair")
    input_cap = config["context_tokens"] - config["max_output_tokens"]
    prefixes = []
    for index, ((call_line, call), target) in enumerate(zip(calls, events)):
        system, issue = _call_task(call)
        prefix = events[:index]
        revision = prefix[-1].revision if prefix else result["initial_revision"]
        constructed, policies = {}, {}
        for strategy in ("mask", "mask-read"):
            manager = ContextManager(strategy, counter, config.get("history_format", "chat"), read_memory_tokens=read_memory_tokens)
            try:
                messages = manager.build(system, issue, prefix, revision, input_cap)
            except ContextOverflow as exc:
                policies[strategy] = {"status": "overflow", "reason": str(exc), "input_tokens_estimate": None,
                                      "visible_known_target_lines": None}
            else:
                constructed[strategy] = messages
                policies[strategy] = {"status": "constructed", "input_tokens_estimate": counter.messages(messages),
                                      "messages_sha256": _json_hash(messages),
                                      "read_memory_lines": manager.stats["last_read_memory_lines"],
                                      "read_memory_events": manager.stats["last_read_memory_events"]}
        # Target action/output is used only now, after both policies are built.
        reads = _current_reads(prefix, revision)
        known, read_status = _known_target_lines(target, reads)
        if target.action.get("tool") == "read_file" and target.revision != revision:
            known, read_status = None, "target_revision_differs_from_prefix"
        for strategy, messages in constructed.items():
            policies[strategy]["visible_known_target_lines"] = None if known is None else _visible_known_lines(messages, known, reads, revision)
        extra = (policies["mask-read"]["input_tokens_estimate"] - policies["mask"]["input_tokens_estimate"]
                 if len(constructed) == 2 else None)
        response = call["response"]
        prefixes.append({"call_jsonl_line": call_line, "event_id": target.id, "prefix_event_count": index,
                         "revision_before_call": revision, "input_cap": input_cap,
                         "source_sha256": {name: item["sha256"] for name, item in sources.items()},
                         "prefix_events_sha256": _json_hash(raw_events[:index]),
                         "actual_system_sha256": _sha256(system.encode("utf-8")),
                         "actual_issue_sha256": _sha256(issue.encode("utf-8")),
                         "recorded_input_estimate_with_analysis_counter": counter.messages(call["messages"]),
                         "historical_usage": {"input_tokens": response.get("input_tokens"), "output_tokens": response.get("output_tokens"),
                                              "usage_source": response["usage_source"], "not_analysis_counter": True},
                         "target_read_status": read_status, "known_target_read_lines": None if known is None else len(known),
                         "extra_input_tokens_estimate": extra, "policies": policies})
    return {**record, "status": "replayed", "sources": sources, "excluded_call_rows": len(raw_calls) - len(calls),
            "input_cap": input_cap, "original_tokenizer": config.get("tokenizer"), "prefixes": prefixes}


def replay_roots(roots: list[Path], counter: TokenCounter, read_memory_tokens: int = 1024) -> dict:
    if read_memory_tokens < 0:
        raise ValueError("read_memory_tokens must be non-negative")
    result_paths = set()
    for root in roots:
        found = [root] if root.is_file() and root.name == "result.json" else list(root.rglob("result.json"))
        if not found:
            raise ValueError(f"No result.json found at {root}")
        result_paths.update(path.resolve() for path in found)
    runs = [replay_run(path.parent, counter, read_memory_tokens) for path in sorted(result_paths)]
    prefixes = [prefix for run in runs if run["status"] == "replayed" for prefix in run["prefixes"]]
    paired = [prefix for prefix in prefixes if prefix["extra_input_tokens_estimate"] is not None]
    diagnostics = [prefix for prefix in paired if prefix["known_target_read_lines"] is not None]
    task_counts = {}
    for run in runs:
        if run["status"] == "replayed":
            task_counts[run["task_id"]] = task_counts.get(run["task_id"], 0) + 1
    summary = {"replayed_runs": sum(run["status"] == "replayed" for run in runs), "excluded_runs": sum(run["status"] == "excluded" for run in runs),
               "distinct_task_ids": len(task_counts), "histories_per_task_id": task_counts, "recorded_prefixes": len(prefixes),
               "paired_constructed_prefixes": len(paired), "overflow_prefixes": len(prefixes) - len(paired),
               "mean_extra_input_tokens_per_constructed_prefix": mean(prefix["extra_input_tokens_estimate"] for prefix in paired) if paired else None,
               "known_target_line_occurrences_all_read_prefixes": sum(prefix["known_target_read_lines"] for prefix in prefixes if prefix["known_target_read_lines"] is not None),
               "known_target_line_occurrences_on_paired_read_prefixes": sum(prefix["known_target_read_lines"] for prefix in diagnostics),
               "visible_known_line_occurrences": {strategy: sum(prefix["policies"][strategy]["visible_known_target_lines"] for prefix in diagnostics)
                                                  for strategy in ("mask", "mask-read")}}
    return {"schema_version": 1, "evidence_kind": "offline_fixed_prefix_replay", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_path_base": "project root for relative paths; absolute paths otherwise",
            "analysis_counter": {"name": counter.name, "description": counter.description, "usage_is_estimated": True,
                                 "exact_qwen_tokenizer": False, "provider_billing": False},
            "read_memory_tokens": read_memory_tokens, "source_fingerprint": source_fingerprint(),
            "driver_sha256": _sha256(Path(__file__).read_bytes()), "limitations": LIMITATIONS,
            "summary": summary, "runs": runs}


def _markdown(report: dict) -> str:
    summary = report["summary"]
    lines = ["# Read-memory fixed-prefix replay", "", f"Counter: `{report['analysis_counter']['name']}` — analysis estimate, not exact Qwen tokens or billing.", "",
             f"{summary['replayed_runs']} histories, {summary['distinct_task_ids']} distinct task IDs; "
             f"{summary['recorded_prefixes']} prefixes, {summary['paired_constructed_prefixes']} paired constructed, {summary['overflow_prefixes']} overflow.", "",
             f"Mean additional input inclusion cost per constructed prefix: {summary['mean_extra_input_tokens_per_constructed_prefix']} estimated tokens.", "",
             f"Known target line occurrences on paired read prefixes: {summary['known_target_line_occurrences_on_paired_read_prefixes']}; "
             f"visible with mask: {summary['visible_known_line_occurrences']['mask']}; "
             f"visible with mask-read: {summary['visible_known_line_occurrences']['mask-read']}.", "",
             "These are repeated line occurrences across fixed prefixes, not independent source lines or tasks.", "",
             "| Task ID | Recorded histories |", "| --- | ---: |"]
    lines += [f"| {task} | {count} |" for task, count in summary["histories_per_task_id"].items()]
    exclusions = [run for run in report["runs"] if run["status"] == "excluded"]
    if exclusions:
        lines += ["", "Excluded runs:", ""] + [f"- `{run['run_directory']}`: {run['reason']}" for run in exclusions]
    lines += ["", "Limitations:", ""] + [f"- {note}" for note in report["limitations"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="Run directories, trees, or result.json paths")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing artifacts are never overwritten")
    parser.add_argument("--tokenizer", default="byte", help="Analysis encoding, e.g. byte or tiktoken:cl100k_base; not model billing")
    parser.add_argument("--read-memory-tokens", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = replay_roots(args.runs, TokenCounter(args.tokenizer), args.read_memory_tokens)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "replay.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
