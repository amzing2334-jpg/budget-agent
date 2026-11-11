"""One pinned mini-SWE-agent reference attempt; plan-only unless --execute is set.

The upstream DefaultAgent owns the loop. This adapter supplies the existing
exact-token HF backend, finite budget, and isolated Docker task environment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from budget_agent.budget import Budget, BudgetExceeded
from budget_agent.cli import load_config
from budget_agent.environment import EnvironmentError, create_environment
from budget_agent.evaluation import evidence_kind
from budget_agent.models import ModelError
from budget_agent.runner import source_fingerprint
from budget_agent.swebench import export_predictions
from budget_agent.tasks import load_tasks

UPSTREAM_VERSION = "2.4.6"
UPSTREAM_SHA = "a83fcae82d2a08f0ee0c688f9d137b3566c097f8"
HARNESS = "mini-swe-agent-textbased-budget-adapter"
STRATEGY = "mini-swe-textbased"


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _append(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def verify_upstream(path: Path) -> Path:
    """Check the source used at execution time, without importing it."""
    root = path.resolve(strict=True)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    if head != UPSTREAM_SHA:
        raise ValueError(f"mini-SWE-agent HEAD must be {UPSTREAM_SHA}; found {head}")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--", "src/minisweagent"],
        check=True, capture_output=True, text=True).stdout
    if dirty.strip():
        raise ValueError("mini-SWE-agent source must be clean, including untracked package files")
    return root


def load_upstream(path: Path):
    """Import only after the explicit execution gate and pinned-tree check."""
    root = verify_upstream(path)
    sys.path.insert(0, str(root / "src"))
    package = importlib.import_module("minisweagent")
    if package.__file__ is None or Path(package.__file__).resolve() != root / "src/minisweagent/__init__.py":
        raise ValueError("Imported mini-SWE-agent is not the verified checkout")
    if package.__version__ != UPSTREAM_VERSION:
        raise ValueError("mini-SWE-agent package version differs from the pinned release")
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.exceptions import FormatError, LimitsExceeded
    from minisweagent.models.litellm_textbased_model import LitellmTextbasedModelConfig
    from minisweagent.models.utils.actions_text import format_observation_messages, parse_regex_actions
    import yaml

    config_path = root / "src/minisweagent/config/benchmarks/swebench_backticks.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return SimpleNamespace(
        Agent=DefaultAgent, FormatError=FormatError, LimitsExceeded=LimitsExceeded,
        parse_actions=parse_regex_actions, format_observations=format_observation_messages,
        check_submission=lambda output: DockerEnvironment._check_finished(None, output),
        action_regex=LitellmTextbasedModelConfig(model_name="adapter").action_regex,
        config=config, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
    )


class BudgetedTextModel:
    """Upstream Model protocol; every generation is metered before parsing."""

    def __init__(self, backend, config, upstream, run_dir: Path):
        self.backend, self.run_config, self.upstream = backend, config, upstream
        self.config = SimpleNamespace(model_name=backend.name)
        self.budget = Budget(backend.counter, config)
        self.run_dir = run_dir
        self.usage_unknown = False
        self.usage_sources: set[str] = set()
        self.actual_models: set[str] = set()
        self.model_seconds = 0.0

    def format_message(self, **kwargs) -> dict:
        return kwargs

    def get_template_vars(self, **kwargs) -> dict:
        return {"model_name": self.config.model_name, **kwargs}

    def _stop(self, reason: str):
        return self.upstream.LimitsExceeded({
            "role": "exit", "content": reason,
            "extra": {"exit_status": "budget_exhausted", "submission": "", "budget_reason": reason},
        })

    def query(self, messages: list[dict], **kwargs) -> dict:
        # Upstream stores raw responses in extra; only role/content enter the LM.
        prompt = [{"role": m["role"], "content": m["content"]} for m in messages]
        try:
            cap = self.budget.reserve(prompt, self.run_config.max_output_tokens,
                                      min_output_tokens=self.run_config.min_action_output_tokens)
        except BudgetExceeded as exc:
            raise self._stop(str(exc)) from exc
        before_cost = self.budget.cost_usd
        started = time.monotonic()
        try:
            response = self.backend.complete(prompt, cap, purpose="agent")
        except (ModelError, KeyboardInterrupt):
            self.usage_unknown = True
            _append(self.run_dir / "calls.jsonl", {"purpose": "agent", "status": "usage_unknown",
                                                   "requested_max_output_tokens": cap})
            raise
        finally:
            self.model_seconds += time.monotonic() - started
        self.usage_sources.add(response.usage_source)
        self.actual_models.add(response.model)
        _append(self.run_dir / "calls.jsonl", {
            "purpose": "agent", "messages": prompt, "response": asdict(response),
            "requested_max_output_tokens": cap, "elapsed_seconds": time.monotonic() - started,
        })
        try:
            self.budget.record(response, purpose="agent")
        except BudgetExceeded as exc:
            raise self._stop(str(exc)) from exc
        # Zero is only the disabled upstream cost counter; financial results
        # retain null when no prices are configured in the existing Budget.
        after_cost = self.budget.cost_usd
        cost = after_cost - before_cost if after_cost is not None and before_cost is not None else 0.0
        extra = {"cost": cost, "response": asdict(response)}
        try:
            actions = self.upstream.parse_actions(
                response.content, action_regex=self.upstream.action_regex,
                format_error_template=self.upstream.config["model"]["format_error_template"],
            )
        except self.upstream.FormatError as exc:
            exc.messages[0].setdefault("extra", {}).update(extra)
            raise
        return {"role": "assistant", "content": response.content, "extra": {"actions": actions, **extra}}

    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars=None) -> list[dict]:
        return self.upstream.format_observations(
            outputs, observation_template=self.upstream.config["model"]["observation_template"],
            template_vars=template_vars,
        )

    def serialize(self) -> dict:
        return {"info": {"config": {"model": {"model_name": self.config.model_name,
                          "adapter": HARNESS}}, "budget": self.budget.to_dict(),
                         "usage_unknown": self.usage_unknown}}


class IsolatedBashEnvironment:
    """Use upstream Bash submission semantics inside our existing Docker boundary."""

    def __init__(self, environment, upstream):
        self.environment, self.upstream = environment, upstream
        self.config = environment.config
        self.steps = 0

    def execute(self, action: dict, cwd: str = "") -> dict:
        self.steps += 1
        output = self.environment.execute({"tool": "run", "command": action["command"]})
        # Truncation must not turn a long submitted patch into a different patch.
        if output.get("output_truncated") and output["output"].lstrip().startswith("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"):
            raise EnvironmentError("Submitted patch exceeded the environment capture limit")
        result = {"output": output["output"], "returncode": output["returncode"],
                  "exception_info": "Command timed out" if output.get("timed_out") else ""}
        self.upstream.check_submission(result)
        return result

    def get_template_vars(self, **kwargs) -> dict:
        return {"cwd": self.environment.workdir, "timeout": self.environment.config.timeout_seconds, **kwargs}

    def serialize(self) -> dict:
        return {"info": {"environment_provenance": self.environment.provenance,
                         "environment_adapter": "budget_agent.environment.DockerEnvironment",
                         "shell": "bash -lc", "network": "none"}}


def run_attempt(task, config, output: Path, backend, upstream, *, environment_factory=create_environment) -> dict:
    """The doubles in CPU tests replace only explicit external boundaries."""
    config = replace(config, tokenizer=backend.counter.name)
    run_dir = output / f"{task.id}__{STRATEGY}"
    run_dir.mkdir(parents=True, exist_ok=False)
    backend.reset_seed(config.seed)
    model = BudgetedTextModel(backend, config, upstream, run_dir)
    started = time.monotonic()
    environment = None
    wrapper = None
    provenance, initial_revision = None, ""
    info = {"exit_status": "not_started", "submission": ""}
    error = None
    cleanup_error = None
    try:
        environment = environment_factory(task, config, run_dir)
        provenance, initial_revision = environment.provenance, environment.revision()
        wrapper = IsolatedBashEnvironment(environment, upstream)
        agent_config = dict(upstream.config["agent"])
        agent_config.update(step_limit=config.max_steps, cost_limit=0.0,
                            output_path=run_dir / "mini.traj.json")
        agent = upstream.Agent(model, wrapper, **agent_config)
        info = agent.run(task.issue)
    except (ModelError, EnvironmentError, OSError, subprocess.SubprocessError) as exc:
        info = {"exit_status": type(exc).__name__, "submission": ""}
        error = str(exc)
    except KeyboardInterrupt:
        info = {"exit_status": "interrupted", "submission": ""}
        error = "Interrupted; partial upstream trajectory retained"
    finally:
        if environment is not None:
            try:
                environment.close()
            except (EnvironmentError, OSError, subprocess.SubprocessError) as exc:
                cleanup_error = str(exc)
    patch = info["submission"]
    (run_dir / "patch.diff").write_bytes(patch.encode("utf-8"))
    budget = model.budget.to_dict()
    unknown = model.usage_unknown or budget["unreconciled_request"]
    recorded_config = {**asdict(config), "strategy": STRATEGY, "history_format": "upstream_textbased",
                       "harness": HARNESS, "upstream_sha": UPSTREAM_SHA,
                       "context_policy": "upstream_full_history", "submission_policy": "explicit_upstream_submission"}
    fingerprint = hashlib.sha256(source_fingerprint().encode() + Path(__file__).read_bytes() + UPSTREAM_SHA.encode()).hexdigest()
    evaluation = {"resolved": None, "status": "pending_official_evaluation"}
    if cleanup_error:
        evaluation["cleanup_error"] = cleanup_error
    result = {
        "run_id": run_dir.name, "task_id": task.id, "split": task.split, "task_kind": task.kind,
        "provider": config.provider, "model": backend.name, "model_metadata": backend.metadata,
        "harness": HARNESS, "upstream_version": UPSTREAM_VERSION, "upstream_sha": UPSTREAM_SHA,
        "upstream_config_sha256": upstream.config_sha256,
        "strategy": STRATEGY, "config": recorded_config, "status": info["exit_status"],
        "upstream_exit": info, "error": error, "resolved": None, "evaluation": evaluation,
        "input_tokens": budget["input_tokens"],
        "output_tokens": budget["output_tokens"], "total_tokens": budget["total_tokens"],
        "calls": budget["calls"], "cost_usd": None if unknown else model.budget.cost_usd,
        "known_cost_usd": model.budget.cost_usd, "usage_unknown": unknown,
        "usage_sources": sorted(model.usage_sources), "actual_models": sorted(model.actual_models),
        "budget": budget, "elapsed_seconds": time.monotonic() - started,
        "model_seconds": model.model_seconds, "steps": wrapper.steps if wrapper else 0,
        "patch_path": str((run_dir / "patch.diff").resolve()), "initial_revision": initial_revision,
        "environment_provenance": provenance, "source_fingerprint": fingerprint,
        "task_fingerprint": hashlib.sha256(json.dumps({"id": task.id, "issue": task.issue,
            "revision": initial_revision, "image": task.image, "environment": provenance,
            "metadata": task.metadata}, sort_keys=True).encode()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_path": str((run_dir / "result.json").resolve()),
    }
    result["evidence_kind"] = evidence_kind(result, default="model_run")
    _json(run_dir / "result.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task", required=True, help="One sanitized SWE-bench task ID")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Run on the cloud Linux CUDA host; default only prints a plan")
    args = parser.parse_args(argv)
    config = load_config(args)
    task = load_tasks(args.tasks, args.split, [args.task])[0]
    if task.kind != "swebench" or config.environment != "docker":
        raise ValueError("The mini-SWE reference requires a SWE-bench task and Docker")
    if config.provider != "transformers" or config.device != "cuda":
        raise ValueError("This reference supports only the exact-token Transformers CUDA backend")
    output = args.output.resolve()
    plan = {"execute": args.execute, "harness": HARNESS, "upstream_version": UPSTREAM_VERSION,
            "required_upstream_sha": UPSTREAM_SHA, "upstream_repo": str(args.upstream_repo.resolve()),
            "task_id": task.id, "split": task.split, "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
            "output": str(output), "model": config.model, "model_revision": config.model_revision,
            "config": asdict(config), "context_policy": "upstream_full_history",
            "comparison": "external harness reference; not an internal context ablation"}
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if platform.system() != "Linux":
        raise ValueError("--execute is restricted to the cloud Linux host; local inference is not enabled here")
    if output.exists():
        raise ValueError("Use a fresh output directory; existing attempts are never overwritten")
    upstream = load_upstream(args.upstream_repo)
    from budget_agent.hf_model import TransformersModel
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "plan.json", plan)
    backend = TransformersModel(config)
    result = run_attempt(task, config, output, backend, upstream)
    predictions = output / "predictions.jsonl"
    export_predictions(output, predictions)
    print(json.dumps({"result": result["result_path"], "predictions": str(predictions),
                      "status": result["status"], "resolved": None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
