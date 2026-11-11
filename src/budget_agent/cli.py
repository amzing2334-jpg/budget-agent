"""Command-line interface; runs are explicit, finite, and recorded locally."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from .schema import RunConfig


ROOT = Path(__file__).resolve().parents[2]


def load_config(args) -> RunConfig:
    values = {}
    if getattr(args, "config", None):
        with Path(args.config).open("rb") as stream:
            data = tomllib.load(stream)
        values = data.get("run", data)
        if not isinstance(values, dict):
            raise ValueError("Config must contain a [run] table")
    names = {f.name for f in fields(RunConfig)}
    if set(values) - names:
        raise ValueError("Unknown config fields: " + ", ".join(sorted(set(values) - names)))
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    values.setdefault("provider", "openai-compatible")
    if values["provider"] != "scripted":
        values.setdefault("environment", "docker")
    if values["provider"] == "openai-compatible":
        values.setdefault("model", os.environ.get("LLM_MODEL", ""))
        values.setdefault("base_url", os.environ.get("LLM_BASE_URL", ""))
    config = RunConfig(**values)
    config.validate()
    return config


def add_run_arguments(parser):
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tasks", type=Path, default=ROOT / "tasks/local.jsonl")
    parser.add_argument("--task", action="append", dest="task_ids")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--output", type=Path, default=Path("runs/pilot"))
    for name in ("strategy", "history_format", "decision_policy", "loop_policy", "repeat_policy", "revision_labels", "reflection_policy", "execution_policy", "provider", "model", "base_url", "api_key_env", "tokenizer", "environment", "docker_image", "api_max_tokens_field", "model_revision", "device", "model_dtype"):
        parser.add_argument("--" + name.replace("_", "-"))
    for name in ("context_tokens", "total_tokens", "max_output_tokens", "min_action_output_tokens", "read_memory_tokens", "max_reflections", "reflection_max_output_tokens", "max_steps", "timeout_seconds", "seed"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("temperature", "input_price_per_million", "output_price_per_million", "max_cost_usd"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    parser.add_argument("--allow-local-execution", action="store_true", default=None)
    parser.add_argument("--api-enable-thinking", action=argparse.BooleanOptionalAction, default=None)


def build_parser():
    parser = argparse.ArgumentParser(prog="budget-agent", description="Budget-controlled coding repair experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Offline scripted smoke test, NOT an LLM benchmark")
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    demo.add_argument("--context-tokens", type=int, default=8192)
    demo.add_argument("--total-tokens", type=int, default=64000)
    run = sub.add_parser("run", help="Run one strategy on selected tasks")
    add_run_arguments(run)
    compare = sub.add_parser("compare", help="Run paired strategies under identical settings")
    add_run_arguments(compare)
    compare.add_argument("--strategies", nargs="+", choices=["full", "recent", "mask", "mask-read", "summary", "structured"], default=["full", "recent", "summary", "structured"])
    compare.add_argument("--repeat", type=int, default=1)
    report = sub.add_parser("report", help="Generate CSV, Markdown and offline HTML")
    report.add_argument("--input", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    diagnose = sub.add_parser("diagnose", help="Analyze saved trajectories without model calls or rewriting runs")
    diagnose.add_argument("--input", required=True, type=Path)
    diagnose.add_argument("--output", required=True, type=Path)
    audit = sub.add_parser("audit", help="Offline independent quality, budget and direct-tool policy checks; no model calls")
    audit.add_argument("--input", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    audit.add_argument("--engine", choices=["builtin", "openjudge"], default="builtin")
    doctor = sub.add_parser("doctor", help="Inspect prerequisites without exposing secrets")
    doctor.add_argument("--config", type=Path)
    imp = sub.add_parser("swe-import", help="Sanitize downloaded SWE-bench JSON/JSONL into a task manifest")
    imp.add_argument("--input", required=True, type=Path)
    imp.add_argument("--output", required=True, type=Path)
    imp.add_argument("--split", choices=["dev", "test"], default="dev")
    imp.add_argument("--limit", type=int)
    exp = sub.add_parser("swe-export", help="Export patches from one comparable experiment group")
    exp.add_argument("--input", required=True, type=Path)
    exp.add_argument("--output", required=True, type=Path)
    grade = sub.add_parser("swe-evaluate", help="Print or execute the official SWE-bench evaluation command")
    grade.add_argument("--predictions", required=True, type=Path)
    grade.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    grade.add_argument("--max-workers", type=int, default=2)
    grade.add_argument("--run-id")
    grade.add_argument("--instance-ids", nargs="+")
    grade.add_argument("--timeout", type=int)
    grade.add_argument("--execute", action="store_true")
    controls = sub.add_parser("swe-check-controls", help="Verify archived official gold and no-repair controls for one frozen task")
    controls.add_argument("--spec", type=Path, required=True, help="Frozen task and expected patch/target hashes")
    controls.add_argument("--gold-report", type=Path, required=True, help="Gold control's report.json")
    controls.add_argument("--negative-report", type=Path, required=True, help="No-repair control's report.json")
    controls.add_argument("--output", type=Path, required=True, help="New verification receipt")
    gold = sub.add_parser("swe-check-gold", help="Check a frozen task's gold report before running its negative control")
    gold.add_argument("--spec", type=Path, required=True)
    gold.add_argument("--gold-report", type=Path, required=True)
    cache = sub.add_parser("swe-cache-files", help="List verified OCI archive members for a pinned single-platform cache")
    cache.add_argument("--layout", type=Path, required=True)
    cache.add_argument("--spec", type=Path, required=True)
    cache.add_argument("--manifest", type=Path, required=True, help="Frozen platform manifest from the publisher")
    attach = sub.add_parser("swe-attach", help="Attach official grades after checking prediction provenance")
    attach.add_argument("--runs", required=True, type=Path)
    attach.add_argument("--reports", required=True, type=Path)
    attach.add_argument("--summary-path", type=Path, help="Explicit official aggregate report, including skipped empty patches")
    attach.add_argument("--predictions-path", type=Path, help="Original predictions and sibling export manifest; required with --summary-path")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            config = load_config(args)
            docker = shutil.which("docker")
            docker_ready = False
            if docker:
                try:
                    docker_ready = subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"], capture_output=True, timeout=10).returncode == 0
                except (OSError, subprocess.TimeoutExpired):
                    pass
            print(json.dumps({"python": platform.python_version(), "git": bool(shutil.which("git")), "docker_installed": bool(docker), "docker_ready": docker_ready, "api_key_variable": config.api_key_env, "api_key_configured": bool(os.environ.get(config.api_key_env)), "model_configured": bool(config.model), "base_url_configured": bool(config.base_url), "note": "Doctor does not invoke a model or install anything."}, indent=2))
            return 0
        if args.command == "report":
            from .report import generate_report
            paths = generate_report(args.input, args.output)
            print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
            return 0
        if args.command == "audit":
            from .evaluation import evaluate_run
            from .report import load_results
            results = load_results(args.input)
            report: dict[str, Any] = {"kind": "single_run_contract_audit", "runs": [
                {"task_id": result["task_id"], "run_id": result["run_id"],
                 "source_result": result["_result_path"], "evaluation": evaluate_run(result)}
                for result in results]}
            upstream_ok = True
            if args.engine == "openjudge":
                import asyncio
                try:
                    from .openjudge_adapter import evaluate_openjudge
                except ModuleNotFoundError as exc:
                    if exc.name != "openjudge":
                        raise
                    raise ValueError("OpenJudge is optional; configure the pinned contract environment described in docs/UPSTREAM_EVALUATION_REVIEW.md") from exc
                report["openjudge"] = asyncio.run(evaluate_openjudge(results))
                upstream_ok = all(
                    len(report["openjudge"].get(name, [])) == len(results)
                    and all(grade.get("score") == 1 and "error" not in grade
                            for grade in report["openjudge"][name])
                    for name in ("quality", "budget", "safety", "acceptance"))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(f"Audited {len(results)} saved runs: {args.output}")
            return 0 if upstream_ok and all(row["evaluation"]["acceptance_status"] == "pass" for row in report["runs"]) else 1
        if args.command == "diagnose":
            from .diagnostics import diagnose_run
            from .report import load_results
            results = load_results(args.input)
            report = {"kind": "offline_behavior_diagnostics", "runs": [
                {"task_id": result["task_id"], "run_id": result["run_id"],
                 "source_result": result["_result_path"],
                 "diagnostics": diagnose_run(Path(result["_result_path"]).parent, result=result)}
                for result in results
            ]}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(f"Diagnosed {len(results)} saved runs: {args.output}")
            return 0
        if args.command.startswith("swe-"):
            from .swebench import (
                attach_grades, check_controls, check_gold_control, evaluation_command, export_predictions, import_tasks,
            )
            if args.command == "swe-import":
                print(f"Imported {import_tasks(args.input, args.output, split=args.split, limit=args.limit)} tasks")
            elif args.command == "swe-export":
                print(f"Exported {export_predictions(args.input, args.output)} predictions")
            elif args.command == "swe-check-gold":
                checked = check_gold_control(json.loads(args.spec.read_bytes()), args.gold_report)
                print(json.dumps(checked, indent=2))
            elif args.command == "swe-cache-files":
                from .oci_cache import cache_files
                members = cache_files(args.layout, json.loads(args.spec.read_bytes()), args.manifest.read_bytes())
                # tar -T consumes POSIX member names, including when invoked from Git Bash.
                sys.stdout.buffer.write(("\n".join(members) + "\n").encode("ascii"))
            elif args.command == "swe-check-controls":
                spec = args.spec.read_bytes()
                receipt = check_controls(json.loads(spec), args.gold_report, args.negative_report)
                receipt["control_spec_sha256"] = hashlib.sha256(spec).hexdigest()
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as stream:
                    json.dump(receipt, stream, indent=2)
                    stream.write("\n")
                print(f"Verified grading controls for {receipt['instance_id']}: {args.output}")
            elif args.command == "swe-evaluate":
                command = evaluation_command(args.predictions, dataset_name=args.dataset, run_id=args.run_id, max_workers=args.max_workers, instance_ids=args.instance_ids, timeout=args.timeout)
                print(json.dumps(command, ensure_ascii=False))
                if args.execute:
                    return subprocess.run(command, shell=False).returncode
            else:
                print(json.dumps(attach_grades(args.runs, args.reports, summary_path=args.summary_path,
                                              predictions_path=args.predictions_path), default=str, ensure_ascii=False))
            return 0
        from .report import generate_report
        from .runner import run_task
        from .tasks import load_tasks
        if args.command == "demo":
            config = RunConfig(provider="scripted", model="scripted-smoke-v1", context_tokens=args.context_tokens, total_tokens=args.total_tokens, max_output_tokens=1536, max_steps=12)
            config.validate()
            tasks = load_tasks(ROOT / "tasks/local.jsonl", split="all")
            strategies, repeats = ["full", "summary", "structured"], 1
            print("SCRIPTED SMOKE: predetermined fixture repairs. These results are not LLM performance.")
        else:
            config = load_config(args)
            tasks = load_tasks(args.tasks, args.split, args.task_ids)
            strategies = args.strategies if args.command == "compare" else [config.strategy]
            repeats = args.repeat if args.command == "compare" else 1
            if repeats < 1:
                raise ValueError("--repeat must be positive")
        shared_model = None
        if config.provider == "transformers":
            from .hf_model import TransformersModel
            shared_model = TransformersModel(config)
        failed = False
        for repeat in range(repeats):
            for task in tasks:
                for strategy in strategies:
                    settings = replace(config, strategy=strategy, seed=config.seed + repeat)
                    result = (run_task(task, settings, args.output, model=shared_model, counter=shared_model.counter)
                              if shared_model is not None else run_task(task, settings, args.output))
                    print(f"{task.id:28} {strategy:10} {result['status']:20} resolved={result['resolved']} tokens={result['total_tokens']} evaluation={result['evaluation'].get('status')}")
                    if result["status"] == "interrupted":
                        generate_report(args.output, args.output / "report")
                        print("Interrupted attempt saved: " + result["result_path"], file=sys.stderr)
                        return 130
                    budget_state = result.get("budget", {})
                    if (result.get("usage_unknown") or budget_state.get("unreconciled_request")
                            or budget_state.get("overrun")):
                        paths = generate_report(args.output, args.output / "report")
                        print("Stopped remaining attempts: model usage is unknown, unreconciled, or exceeded its budget. "
                              "Review the saved ledger before starting a new batch. Report: " + str(paths["html"]), file=sys.stderr)
                        return 1
                    failed |= (
                        result["status"] in {"model_error", "environment_error"}
                        or result["evaluation"].get("status") in {"grading_error", "infrastructure_error"}
                        or bool(result["evaluation"].get("cleanup_error"))
                    )
        paths = generate_report(args.output, args.output / "report")
        print("Report: " + str(paths["html"]))
        return 1 if failed else 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
