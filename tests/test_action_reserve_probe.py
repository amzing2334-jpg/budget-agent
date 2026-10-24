"""CPU-only driver checks with model and run_task doubles; no inference."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location("action_reserve_probe", Path(__file__).parents[1] / "scripts/probe_action_reserve.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('''[run]
provider = "transformers"
model = "test/model"
model_revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
strategy = "recent"
history_format = "chat"
environment = "docker"
device = "cuda"
max_output_tokens = 768
min_action_output_tokens = 128
''', encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"id": "astropy__astropy-12907", "issue": "Repair the real issue", "split": "dev", "kind": "swebench", "image": "task:fixed"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(probe, "source_fingerprint", lambda: "frozen-source")
    return config, tasks, "astropy__astropy-12907", tmp_path / "output"


@pytest.fixture
def execution(inputs, monkeypatch):
    import budget_agent.hf_model
    config, tasks, task_id, output = inputs
    created, calls = [], []
    state = {"changes": {}, "raise": None, "mutate_config": False}

    def factory(settings):
        plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        assert plan["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
        assert plan["tasks_sha256"] == hashlib.sha256(tasks.read_bytes()).hexdigest()
        assert len(plan["task_sha256"]) == len(plan["driver_sha256"]) == 64
        assert probe.os.environ["HF_HUB_OFFLINE"] == probe.os.environ["TRANSFORMERS_OFFLINE"] == "1"
        model = SimpleNamespace(counter=object())
        created.append(model)
        return model

    def run(task, settings, folder, *, model, counter):
        calls.append((settings, folder, model, counter))
        if state["raise"]:
            raise state["raise"]
        result = {
            "status": "finished", "result_path": str(folder / "attempt/result.json"),
            "usage_unknown": False, "budget": {"overrun": False, "unreconciled_request": False},
            "evaluation": {"status": "pending_official_evaluation"},
            "source_fingerprint": "frozen-source", "config": asdict(settings),
            "task_id": task.id, "task_fingerprint": "task-fingerprint", "initial_revision": "revision",
            "environment_provenance": {"image_id": "image"}, "model": "test/model", "model_metadata": {"revision": "a" * 40},
        }
        result.update(state["changes"])
        if state["mutate_config"]:
            config.write_text(config.read_text(encoding="utf-8") + "\n# modified\n", encoding="utf-8")
        return result

    monkeypatch.setattr(probe, "_require_cloud_cuda", lambda: None)
    monkeypatch.setattr(budget_agent.hf_model, "TransformersModel", factory)
    monkeypatch.setattr(probe, "run_task", run)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    return created, calls, state


def test_default_cli_only_prints_plan_and_does_not_create_output(inputs, monkeypatch, capsys):
    config, tasks, task, output = inputs
    monkeypatch.setattr(probe, "_require_cloud_cuda", lambda: pytest.fail("preview tried CUDA"))
    monkeypatch.setattr(probe, "run_task", lambda *a, **k: pytest.fail("preview tried inference"))
    assert probe.main(["--config", str(config), "--tasks", str(tasks), "--task", task, "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["execute"] is False and printed["plan"]["attempts_per_reserve"] == 1
    assert [row["min_action_output_tokens"] for row in printed["plan"]["attempts"]] == [1, 128]
    assert "not forced generation" in printed["plan"]["notes"][0]
    assert not output.exists()


def test_execution_freezes_plan_and_reuses_one_model(inputs, execution):
    created, calls, _ = execution
    records = probe.run_probe(*inputs, execute=True)
    assert len(created) == 1 and len(calls) == 2
    assert [call[0].min_action_output_tokens for call in calls] == [1, 128]
    assert [call[1].name for call in calls] == ["reserve-1", "reserve-128"]
    assert all(call[2] is created[0] and call[3] is created[0].counter for call in calls)
    first, second = [asdict(call[0]) for call in calls]
    assert {key for key in first if first[key] != second[key]} == {"min_action_output_tokens"}
    assert json.loads((inputs[-1] / "attempts.json").read_text(encoding="utf-8")) == records
    with pytest.raises(FileExistsError, match="new output"):
        probe.run_probe(*inputs, execute=True)
    assert len(created) == 1 and len(calls) == 2


@pytest.mark.parametrize("changes", [
    {"status": "model_error"}, {"status": "environment_error"},
    {"usage_unknown": True}, {"budget": {"overrun": True, "unreconciled_request": False}},
    {"budget": {"overrun": False, "unreconciled_request": True}},
    {"evaluation": {"status": "grading_error"}},
    {"evaluation": {"status": "pending_official_evaluation", "cleanup_error": "container failed to close"}},
])
def test_failed_first_attempt_stops_remaining_attempt(inputs, execution, changes):
    _, calls, state = execution
    state["changes"] = changes
    with pytest.raises(ValueError, match="no further attempt"):
        probe.run_probe(*inputs, execute=True)
    assert len(calls) == 1
    assert len(json.loads((inputs[-1] / "attempts.json").read_text(encoding="utf-8"))) == 1


def test_interrupted_attempt_is_retained_and_stops_pair(inputs, execution):
    _, calls, state = execution
    state["changes"] = {"status": "interrupted"}
    with pytest.raises(KeyboardInterrupt):
        probe.run_probe(*inputs, execute=True)
    assert len(calls) == 1 and (inputs[-1] / "attempts.json").is_file()


def test_unhandled_run_exception_does_not_start_second_attempt(inputs, execution):
    _, calls, state = execution
    state["raise"] = RuntimeError("unexpected execution failure")
    with pytest.raises(RuntimeError, match="unexpected execution failure"):
        probe.run_probe(*inputs, execute=True)
    assert len(calls) == 1 and (inputs[-1] / "plan.json").is_file()


def test_changed_config_between_attempts_is_not_used(inputs, execution):
    _, calls, state = execution
    state["mutate_config"] = True
    with pytest.raises(ValueError, match="Frozen source/config/task/driver changed"):
        probe.run_probe(*inputs, execute=True)
    assert len(calls) == 1


def test_different_container_identity_cannot_be_reported_as_matched_pair(inputs, execution, monkeypatch):
    _, calls, _ = execution
    original = probe.run_task

    def run(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(calls) == 2:
            result["environment_provenance"] = {"image_id": "different-image"}
        return result

    monkeypatch.setattr(probe, "run_task", run)
    with pytest.raises(ValueError, match="do not compare this pair"):
        probe.run_probe(*inputs, execute=True)
    assert len(calls) == 2


def test_non_linux_execution_refused_before_optional_torch_import(monkeypatch):
    monkeypatch.setattr(probe.sys, "platform", "win32")
    with pytest.raises(ValueError, match="Linux CUDA"):
        probe._require_cloud_cuda()


def test_missing_cuda_does_not_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr(probe.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(ValueError, match="not fall back to CPU"):
        probe._require_cloud_cuda()
