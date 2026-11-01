"""CPU adapter checks, plus opt-in pinned-upstream integration.

Default checks use explicit boundary doubles. Setting MINI_SWE_UPSTREAM_REPO
also exercises the real upstream loop, parser and persistence with a fake model
and environment. Neither mode performs inference or launches Docker.
"""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from budget_agent.budget import TokenCounter
from budget_agent.models import ModelError
from budget_agent.report import condition_key
from budget_agent.schema import ModelResponse, RunConfig, Task
from budget_agent.swebench import export_predictions

SPEC = importlib.util.spec_from_file_location(
    "mini_baseline_test_target", Path(__file__).parents[1] / "scripts/run_mini_swe_baseline.py")
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


class FlowDouble(Exception):
    def __init__(self, *messages):
        self.messages = messages


class FormatErrorDouble(FlowDouble):
    pass


class LimitsExceededDouble(FlowDouble):
    pass


class BackendDouble:
    name = "test/model@revision"
    metadata = {"test_double": True}
    counter = TokenCounter("byte")

    def __init__(self):
        self.calls = []

    def reset_seed(self, seed):
        self.seed = seed

    def complete(self, messages, max_output_tokens, *, purpose):
        self.calls.append((messages, max_output_tokens, purpose))
        return ModelResponse("test response", self.counter.messages(messages), 10,
                             usage_source="tokenizer_exact", model=self.name)


@pytest.fixture
def config():
    return RunConfig(provider="transformers", model="test/model", model_revision="a" * 40,
                     environment="docker", device="cuda", context_tokens=8192,
                     total_tokens=32000, max_output_tokens=768, min_action_output_tokens=128)


@pytest.fixture
def upstream():
    return SimpleNamespace(
        FormatError=FormatErrorDouble, LimitsExceeded=LimitsExceededDouble,
        parse_actions=lambda content, **kwargs: [{"command": "echo test"}],
        format_observations=lambda outputs, **kwargs: [{"role": "user", "content": outputs[0]["output"]}],
        check_submission=lambda output: None, action_regex="test-double-regex",
        config={"agent": {"system_template": "system", "instance_template": "{{task}}"},
                "model": {"format_error_template": "bad format", "observation_template": "output"}},
        config_sha256="test-double-config-sha",
    )


def test_plan_does_not_load_upstream_model_or_create_files(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Plan imported upstream or started execution")
    monkeypatch.setattr(baseline, "load_upstream", forbidden)
    monkeypatch.setattr(baseline, "run_attempt", forbidden)
    output = tmp_path / "not-created"
    root = Path(__file__).parents[1]
    assert baseline.main([
        "--tasks", str(root / "tasks/swebench-pilot.jsonl"), "--task", "astropy__astropy-12907",
        "--config", str(root / "configs/cloud-qwen7b.toml"),
        "--upstream-repo", str(tmp_path / "not-installed"), "--output", str(output),
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["execute"] is False
    assert plan["required_upstream_sha"] == baseline.UPSTREAM_SHA
    assert not output.exists()


@pytest.mark.parametrize("head,dirty,error", [
    ("b" * 40, "", "HEAD must"),
    (baseline.UPSTREAM_SHA, " M src/minisweagent/agents/default.py", "must be clean"),
    (baseline.UPSTREAM_SHA, "?? src/minisweagent/untracked.py", "must be clean"),
])
def test_pinned_checkout_rejects_wrong_or_modified_source(tmp_path, monkeypatch, head, dirty, error):
    def git(command, **kwargs):
        return SimpleNamespace(stdout=head if "rev-parse" in command else dirty)
    monkeypatch.setattr(baseline.subprocess, "run", git)
    with pytest.raises(ValueError, match=error):
        baseline.verify_upstream(tmp_path)


def test_import_must_come_from_verified_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "verify_upstream", lambda path: tmp_path)
    monkeypatch.setattr(baseline.importlib, "import_module",
                        lambda name: SimpleNamespace(__file__=str(tmp_path / "wrong/__init__.py")))
    with pytest.raises(ValueError, match="not the verified checkout"):
        baseline.load_upstream(tmp_path)


def test_query_charges_format_failures_before_upstream_parser(config, upstream, tmp_path):
    def invalid(content, **kwargs):
        raise FormatErrorDouble({"role": "user", "content": "invalid", "extra": {}})
    upstream.parse_actions = invalid
    model = baseline.BudgetedTextModel(BackendDouble(), config, upstream, tmp_path)
    with pytest.raises(FormatErrorDouble) as exc:
        model.query([{"role": "user", "content": "issue", "extra": {"hidden": "not sent"}}])
    assert model.budget.calls == 1
    assert model.budget.output_tokens == 10
    assert exc.value.messages[0]["extra"]["response"]["content"] == "test response"
    assert model.backend.calls[0][0] == [{"role": "user", "content": "issue"}]
    assert model.budget.to_dict()["unreconciled_request"] is False


def test_budget_floor_stops_before_generation(config, upstream, tmp_path):
    config = replace(config, total_tokens=120)
    backend = BackendDouble()
    model = baseline.BudgetedTextModel(backend, config, upstream, tmp_path)
    with pytest.raises(LimitsExceededDouble) as exc:
        model.query([{"role": "user", "content": "issue"}])
    assert backend.calls == []
    assert exc.value.messages[0]["extra"]["submission"] == ""
    assert model.budget.to_dict()["unreconciled_request"] is False


@pytest.mark.parametrize("failure", [ModelError("connection lost"), KeyboardInterrupt()])
def test_unknown_model_usage_halts_and_stays_unreconciled(config, upstream, tmp_path, failure):
    backend = BackendDouble()
    def failed(*args, **kwargs):
        raise failure
    backend.complete = failed
    model = baseline.BudgetedTextModel(backend, config, upstream, tmp_path)
    with pytest.raises(type(failure)):
        model.query([{"role": "user", "content": "issue"}])
    assert model.usage_unknown
    assert model.budget.to_dict()["unreconciled_request"]
    call = json.loads((tmp_path / "calls.jsonl").read_text())
    assert call["status"] == "usage_unknown"
    assert call["requested_max_output_tokens"] == config.max_output_tokens
    assert model.budget.to_dict()["attempted_calls"] == 1
    assert model.budget.calls == 0


class EnvironmentDouble:
    provenance = {"kind": "docker", "image_id": "sha256:" + "1" * 64}
    workdir = "/testbed"

    def __init__(self, config):
        self.config = config
        self.closed = False
        self.actions = []

    def revision(self):
        return "test-revision"

    def execute(self, action):
        self.actions.append(action)
        return {"output": "result", "returncode": 0, "ok": True}

    def patch(self):
        pytest.fail("External reference must not replace upstream submission with workspace diff")

    def close(self):
        self.closed = True


def test_bash_adapter_delegates_submission_and_uses_existing_environment(config, upstream):
    observed = []
    upstream.check_submission = observed.append
    environment = EnvironmentDouble(config)
    wrapper = baseline.IsolatedBashEnvironment(environment, upstream)
    wrapper.execute({"command": "echo test"})
    assert environment.actions == [{"tool": "run", "command": "echo test"}]
    assert observed == [{"output": "result", "returncode": 0, "exception_info": ""}]


def test_truncated_submission_is_rejected(config, upstream):
    environment = EnvironmentDouble(config)
    environment.execute = lambda action: {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npartial patch",
                                          "returncode": 0, "output_truncated": True}
    with pytest.raises(baseline.EnvironmentError, match="capture limit"):
        baseline.IsolatedBashEnvironment(environment, upstream).execute({"command": "submit"})


@pytest.mark.parametrize("submission,status", [
    ("--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\r\n+new\r\n", "Submitted"),
    ("", "LimitsExceeded"),
])
def test_real_result_export_contract_preserves_upstream_submission(config, upstream, tmp_path, submission, status):
    calls = []
    class UpstreamAgentDouble:
        """A boundary spy, not an implementation or simulation of the upstream loop."""
        def __init__(self, model, env, **kwargs):
            self.model = model
            calls.append(kwargs)
        def run(self, issue):
            calls.append(issue)
            self.model.query([{"role": "user", "content": issue}])
            return {"exit_status": status, "submission": submission}
    upstream.Agent = UpstreamAgentDouble
    environment = EnvironmentDouble(config)
    task = Task(id="pallets__flask-1234", issue="repair this", kind="swebench", split="dev",
                image="test/image", metadata={"base_commit": "a" * 40})
    result = baseline.run_attempt(task, config, tmp_path, BackendDouble(), upstream,
                                  environment_factory=lambda *args: environment)
    predictions = tmp_path / "predictions.jsonl"
    export_predictions(tmp_path, predictions)
    prediction = json.loads(predictions.read_text().strip())
    assert prediction["model_patch"] == submission
    assert Path(result["patch_path"]).read_bytes() == submission.encode("utf-8")
    assert predictions.with_suffix(".jsonl.manifest.json").is_file()
    assert environment.closed
    assert calls[0]["step_limit"] == config.max_steps
    assert calls[0]["cost_limit"] == 0.0
    assert calls[1] == task.issue
    assert result["harness"] == baseline.HARNESS
    assert result["resolved"] is None
    internal = dict(result, source_fingerprint="internal-source", config=asdict_config(config))
    assert condition_key(internal) != condition_key(result)


def asdict_config(config):
    from dataclasses import asdict
    return asdict(config)


@pytest.fixture
def real_upstream(tmp_path, monkeypatch):
    checkout = os.environ.get("MINI_SWE_UPSTREAM_REPO")
    if not checkout:
        pytest.skip("Set MINI_SWE_UPSTREAM_REPO to run the pinned upstream CPU integration")
    # Isolate upstream's global dotenv/config and avoid its optional price-map fetch.
    monkeypatch.setenv("MSWEA_GLOBAL_CONFIG_DIR", str(tmp_path / "mini-global"))
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    loaded = baseline.load_upstream(Path(checkout))
    assert loaded.Agent.__module__ == "minisweagent.agents.default"
    assert loaded.parse_actions.__module__ == "minisweagent.models.utils.actions_text"
    yield loaded
    assert not {"torch", "transformers", "peft", "accelerate", "budget_agent.hf_model"}.intersection(sys.modules)


class ScriptedCPUBackend(BackendDouble):
    """Synthetic text/usage only; no tokenizer, model, endpoint or weights."""
    name = "cpu-integration-double"
    metadata = {"test_double": True, "inference": False}

    def __init__(self, responses):
        super().__init__()
        self.responses = iter(responses)

    def complete(self, messages, max_output_tokens, *, purpose):
        self.calls.append((messages, max_output_tokens, purpose))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return ModelResponse(response, self.counter.messages(messages), self.counter.count(response),
                             usage_source="cpu_test_counter", model=self.name)


class ScriptedCPUEnvironment(EnvironmentDouble):
    """Record Bash actions and return supplied outputs without executing them."""
    provenance = {"kind": "cpu_test_double", "commands_executed": False}

    def __init__(self, config, outputs):
        super().__init__(config)
        self.outputs = iter(outputs)

    def execute(self, action):
        self.actions.append(action)
        return {"output": next(self.outputs), "returncode": 0, "ok": True}


def real_loop_attempt(tmp_path, config, upstream, responses, outputs, *, total_tokens=128000):
    # Byte counts are synthetic test accounting, not the production HF tokenizer.
    config = replace(config, context_tokens=32768, total_tokens=total_tokens)
    backend = ScriptedCPUBackend(responses)
    environment = ScriptedCPUEnvironment(config, outputs)
    task = Task(id="cpu__integration-1", issue="Synthetic protocol check; no repository is present.",
                kind="swebench", split="dev", image="unused-test-image", metadata={"base_commit": "a" * 40})
    result = baseline.run_attempt(task, config, tmp_path, backend, upstream,
                                  environment_factory=lambda *args: environment)
    folder = Path(result["result_path"]).parent
    trajectory = json.loads((folder / "mini.traj.json").read_text(encoding="utf-8"))
    assert trajectory["info"]["config"]["agent_type"] == "minisweagent.agents.default.DefaultAgent"
    assert trajectory["info"]["mini_version"] == baseline.UPSTREAM_VERSION
    assert environment.closed
    return result, trajectory, backend, environment


def test_real_upstream_format_recovery_submission_and_export(config, real_upstream, tmp_path):
    patch = "diff --git a/example.py b/example.py\n--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old\n+new\n"
    responses = ["missing action", "```mswea_bash_command\ninspect\n```",
                 "```mswea_bash_command\nsubmit\n```"]
    result, trajectory, backend, environment = real_loop_attempt(
        tmp_path, config, real_upstream, responses,
        ["synthetic repository output", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n" + patch],
    )
    assert result["status"] == "Submitted"
    assert result["calls"] == 3
    assert len(backend.calls) == 3
    assert len(environment.actions) == 2
    assert trajectory["info"]["submission"] == patch
    assert any(m.get("extra", {}).get("interrupt_type") == "FormatError" for m in trajectory["messages"])
    assert any(m.get("extra", {}).get("raw_output") == "synthetic repository output" for m in trajectory["messages"])
    assert all(set(message) == {"role", "content"} for call in backend.calls for message in call[0])
    prediction_path = tmp_path / "predictions.jsonl"
    export_predictions(tmp_path, prediction_path)
    assert json.loads(prediction_path.read_text())["model_patch"] == patch
    assert prediction_path.with_suffix(".jsonl.manifest.json").is_file()


def test_real_upstream_repeated_format_error_limit(config, real_upstream, tmp_path):
    result, trajectory, backend, environment = real_loop_attempt(
        tmp_path, config, real_upstream, ["invalid"] * 3, [],
    )
    assert result["status"] == "RepeatedFormatError"
    assert result["calls"] == len(backend.calls) == 3
    assert environment.actions == []
    assert trajectory["info"]["submission"] == ""
    assert len(result["budget"]["requests"]) == 3


def test_real_upstream_step_limit(config, real_upstream, tmp_path):
    result, trajectory, backend, environment = real_loop_attempt(
        tmp_path, replace(config, max_steps=1), real_upstream,
        ["```mswea_bash_command\ninspect\n```"], ["synthetic observation"],
    )
    assert result["status"] == "LimitsExceeded"
    assert len(backend.calls) == len(environment.actions) == 1
    assert trajectory["info"]["submission"] == ""


def test_real_upstream_budget_exception_and_save(config, real_upstream, tmp_path):
    result, trajectory, backend, environment = real_loop_attempt(
        tmp_path, config, real_upstream, [], [], total_tokens=100,
    )
    assert result["status"] == "budget_exhausted"
    assert backend.calls == environment.actions == []
    assert result["budget"]["unreconciled_request"] is False
    assert trajectory["info"]["exit_status"] == "budget_exhausted"
    # Upstream counts entry into query(), even if our gate sends no request.
    assert trajectory["info"]["model_stats"]["api_calls"] == 1
    assert result["calls"] == result["budget"]["attempted_calls"] == 0


@pytest.mark.parametrize("failure,status", [(ModelError("synthetic failure"), "ModelError"),
                                            (KeyboardInterrupt(), "interrupted")])
def test_real_upstream_failed_model_call_keeps_ledger(config, real_upstream, tmp_path, failure, status):
    result, trajectory, backend, environment = real_loop_attempt(
        tmp_path, config, real_upstream, [failure], [],
    )
    assert result["status"] == status
    assert result["usage_unknown"] is True
    assert result["budget"]["unreconciled_request"] is True
    assert result["budget"]["attempted_calls"] == len(backend.calls) == 1
    assert environment.actions == []
    calls = (Path(result["result_path"]).parent / "calls.jsonl").read_text().splitlines()
    assert len(calls) == 1
    assert json.loads(calls[0])["status"] == "usage_unknown"
    assert trajectory["info"]["budget"]["unreconciled_request"] is True
    assert trajectory["info"]["exit_status"] == ("" if status == "interrupted" else status)
