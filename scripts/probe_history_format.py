"""One predeclared transcript/chat development diagnostic; no training or downloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from budget_agent.cli import load_config
from budget_agent.report import load_results
from budget_agent.runner import run_task, source_fingerprint
from budget_agent.tasks import load_tasks


FORMATS = ("transcript", "chat")


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_probe(config_path: Path, tasks_path: Path, task_id: str, output: Path) -> None:
    config = load_config(SimpleNamespace(config=config_path))
    if (config.provider, config.strategy, config.environment) != ("transformers", "recent", "docker"):
        raise ValueError("Probe requires transformers, recent, and docker in the config")
    task, = load_tasks(tasks_path, split="dev", task_ids=[task_id])
    config = replace(config, tokenizer=f"hf:{config.model}@{config.model_revision}")
    common_config = asdict(config)
    del common_config["history_format"]
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "plan.json", {
        "kind": "development_protocol_diagnostic", "created_at": datetime.now(timezone.utc).isoformat(),
        "formats": list(FORMATS), "attempts_per_format": 1,
        "task_id": task.id, "split": task.split, "config": common_config,
        "tasks_path": str(tasks_path.resolve()),
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "source_fingerprint": source_fingerprint(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    # Set before importing the inference stack; missing cached files must fail.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from budget_agent.hf_model import TransformersModel

    model = TransformersModel(config)
    for history_format in FORMATS:
        result = run_task(task, replace(config, history_format=history_format),
                          output / history_format, model=model, counter=model.counter)
        print(json.dumps({"history_format": history_format, "status": result["status"],
                          "result_path": result["result_path"]}, ensure_ascii=False))
        if result["status"] == "interrupted":
            raise KeyboardInterrupt


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _condition(result: dict) -> dict:
    fields = ("provider", "model", "model_metadata", "actual_models", "usage_sources",
              "source_fingerprint", "task_id", "task_fingerprint", "initial_revision",
              "environment_provenance", "split", "task_kind", "evidence_kind")
    for field in fields:
        if not result.get(field):
            raise ValueError(f"Missing comparison evidence: {field}")
    condition = {field: result[field] for field in fields}
    condition["config"] = {key: value for key, value in result["config"].items() if key != "history_format"}
    return condition


def _metrics(result: dict, calls: list[dict], events: list[dict], folder: Path) -> dict:
    responses = [call["response"] for call in calls if "response" in call]
    if any(call["purpose"] != "agent" for call in calls):
        raise ValueError("Recent-only probe cannot contain non-agent model calls")
    inputs = sum(response["input_tokens"] for response in responses)
    outputs = sum(response["output_tokens"] for response in responses)
    budget, config = result["budget"], result["config"]
    for value in (result, budget):
        if (value["input_tokens"], value["output_tokens"], value["total_tokens"], value["calls"]) != (
                inputs, outputs, inputs + outputs, len(responses)):
            raise ValueError("Saved calls do not match result/budget token accounting")
    if sorted({response["model"] for response in responses}) != result["actual_models"]:
        raise ValueError("Call models do not match actual_models")
    if sorted({response["usage_source"] for response in responses}) != result["usage_sources"]:
        raise ValueError("Call usage sources do not match usage_sources")
    if result["steps"] != len(events) or len(events) > len(responses):
        raise ValueError("Trajectory length does not match the saved run")
    maximum_actual = max(response["input_tokens"] + response["output_tokens"] for response in responses)
    maximum_reserved = max(call["response"]["input_tokens"] + call["requested_max_output_tokens"]
                           for call in calls if "response" in call)
    maximum_output_cap = max(call["requested_max_output_tokens"] for call in calls if "response" in call)
    budget_valid = not (result["usage_unknown"] or budget["overrun"] or budget["unreconciled_request"]
                        or len(responses) != len(calls) or inputs + outputs > config["total_tokens"]
                        or maximum_actual > config["context_tokens"]
                        or maximum_reserved > config["context_tokens"]
                        or maximum_output_cap > config["max_output_tokens"] or len(events) > config["max_steps"]
                        or any(call["response"]["output_tokens"] > call["requested_max_output_tokens"]
                               for call in calls if "response" in call))
    seen_failures: set[tuple[str, str, str]] = set()
    repeated_failures = 0
    for event in events:
        if event["ok"] is False and event["action"]["tool"] != "invalid_action":
            key = (event["revision"], json.dumps(event["action"], sort_keys=True), event["output"])
            repeated_failures += key in seen_failures
            seen_failures.add(key)
    patch = (folder / "patch.diff").read_bytes()
    evaluation = result["evaluation"]
    return {
        "result_path": str(folder / "result.json"), "status": result["status"],
        "budget": {"valid": budget_valid, "context_limit": config["context_tokens"],
                   "total_limit": config["total_tokens"], "input_tokens": inputs,
                   "output_tokens": outputs, "total_tokens": inputs + outputs,
                   "max_call_actual_tokens": maximum_actual, "max_call_reserved_tokens": maximum_reserved,
                   "output_limit": config["max_output_tokens"], "max_requested_output": maximum_output_cap,
                   "step_limit": config["max_steps"], "usage_unknown": result["usage_unknown"],
                   "overrun": budget["overrun"], "unreconciled_request": budget["unreconciled_request"]},
        "model_calls": len(calls), "steps": len(events),
        "invalid_actions": sum(event["action"]["tool"] == "invalid_action" for event in events),
        "repeated_identical_failed_actions": repeated_failures,
        "successful_edit_file": sum(event["ok"] is True and event["action"]["tool"] == "edit_file" for event in events),
        "successful_write_file": sum(event["ok"] is True and event["action"]["tool"] == "write_file" for event in events),
        "diff_actions": sum(event["action"]["tool"] == "diff" for event in events),
        "run_actions": sum(event["action"]["tool"] == "run" for event in events),
        "patch_bytes": len(patch), "empty_patch": len(patch) == 0,
        "resolved": result["resolved"], "evaluation_status": evaluation["status"],
        "official_resolved": evaluation.get("official_resolved", evaluation["resolved"]
                                   if evaluation["status"] == "official_swebench" else None),
        "tests_executed": evaluation.get("tests_executed"),
        "evaluation": evaluation,
    }


def summarize(root: Path) -> dict:
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if (plan["kind"] != "development_protocol_diagnostic" or plan["formats"] != list(FORMATS)
            or plan["attempts_per_format"] != 1 or plan["split"] != "dev"):
        raise ValueError("Expected a predeclared single-attempt dev transcript/chat plan")
    results = load_results(root)
    if len(results) != 2:
        raise ValueError("Expected exactly one result per format; missing/extra attempts cannot be compared")
    indexed = {}
    for result in results:
        history_format = result["config"]["history_format"]
        if history_format not in FORMATS or history_format in indexed:
            raise ValueError("Expected exactly one transcript and one chat attempt")
        if (result["strategy"] != "recent" or result["config"]["strategy"] != "recent"
                or result["provider"] != "transformers" or result["split"] != "dev"):
            raise ValueError("Probe results must use recent, transformers, and dev")
        indexed[history_format] = result
    condition = _condition(indexed["transcript"])
    if _condition(indexed["chat"]) != condition:
        raise ValueError("Mismatched model/metadata/source/task/config; only history_format may differ")
    for field in ("task_id", "split", "source_fingerprint", "config"):
        if condition[field] != plan[field]:
            raise ValueError(f"Result does not match predeclared plan: {field}")
    first_messages = None
    records = {}
    for history_format in FORMATS:
        result = indexed[history_format]
        folder = Path(result["_result_path"]).parent
        if folder.relative_to(root.resolve()).parts[0] != history_format:
            raise ValueError("Result is stored outside its declared format directory")
        calls, events = _rows(folder / "calls.jsonl"), _rows(folder / "trajectory.jsonl")
        agent_calls = [call for call in calls if call["purpose"] == "agent"]
        if not agent_calls or not agent_calls[0].get("messages"):
            raise ValueError("Missing first agent messages; cannot compare protocols")
        messages = agent_calls[0]["messages"]
        if first_messages is not None and messages != first_messages:
            raise ValueError("First agent messages differ; not a controlled history-format probe")
        first_messages = messages
        records[history_format] = _metrics(result, calls, events, folder)
    return {
        "kind": "development_protocol_diagnostic", "condition": condition,
        "attempts_per_format": 1, "first_agent_messages_identical": True, "formats": records,
        "notes": [
            "Single development task, one attempt per format: no win rate, significance, or generalization claim.",
            "Repeated failure counts occurrences after the first with identical action JSON, repository revision, and error output; invalid_action is excluded and counted separately.",
            "Successful edit/write means the tool reported ok, not that the code repairs the issue; run_actions includes all shell commands, not only tests.",
            "Finished status and nonempty patches are not repair success. Unattached official evaluation and skipped empty patches remain ungraded.",
            "A false budget.valid prevents a fixed-budget efficacy claim; all diagnostic records remain visible.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    run = commands.add_parser("run", help="Explicitly run two cloud inference attempts using cached weights")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--tasks", required=True, type=Path)
    run.add_argument("--task", required=True)
    run.add_argument("--output", required=True, type=Path)
    summary = commands.add_parser("summarize", help="Read saved artifacts only; refuse incomparable attempts")
    summary.add_argument("--runs", required=True, type=Path)
    summary.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "run":
            run_probe(args.config, args.tasks, args.task, args.output)
        else:
            report = summarize(args.runs)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(f"Development diagnostic saved: {args.output}")
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, f"Probe refused: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
