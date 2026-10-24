"""Synthetic artifacts and model doubles only: no torch, downloads, or inference."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from budget_agent.schema import RunConfig


SPEC = importlib.util.spec_from_file_location("protocol_probe", Path(__file__).parents[1] / "scripts/probe_history_format.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def artifacts(tmp_path):
    config = asdict(RunConfig(provider="transformers", strategy="recent", model="test/model",
                              model_revision="a" * 40, environment="docker"))
    common = {key: value for key, value in config.items() if key != "history_format"}
    write_json(tmp_path / "plan.json", {"kind": "development_protocol_diagnostic", "formats": list(probe.FORMATS),
               "attempts_per_format": 1, "task_id": "task", "split": "dev", "source_fingerprint": "code", "config": common})
    failed = {"action": {"tool": "edit_file", "path": "a.py", "old": "x", "new": "y"},
              "revision": "r1", "ok": False, "output": "found 2 matches"}
    events = [failed, failed, {**failed, "revision": "r2"},
              {**failed, "output": "found 0 matches"},
              {**failed, "action": {"tool": "invalid_action"}},
              {**failed, "ok": True},
              {**failed, "ok": True, "action": {"tool": "write_file"}},
              {**failed, "ok": True, "action": {"tool": "diff"}},
              {**failed, "ok": True, "action": {"tool": "run", "command": "ls"}}]
    for history_format in probe.FORMATS:
        folder = tmp_path / history_format / "attempt"
        folder.mkdir(parents=True)
        calls = [{"purpose": "agent", "messages": [{"role": "user", "content": "repair"}],
                  "response": {"content": "{}", "input_tokens": 100, "output_tokens": 10,
                               "model": "model", "usage_source": "tokenizer_exact"},
                  "requested_max_output_tokens": 20} for _ in events]
        usage = {"input_tokens": 900, "output_tokens": 90, "total_tokens": 990, "calls": 9}
        result = {"task_id": "task", "task_fingerprint": "task-fp", "split": "dev", "task_kind": "swebench",
                  "source_fingerprint": "code", "initial_revision": "r1", "environment_provenance": {"image": "pinned"},
                  "provider": "transformers", "model": "model", "model_metadata": {"revision": "a" * 40},
                  "actual_models": ["model"], "usage_sources": ["tokenizer_exact"], "evidence_kind": "model_run",
                  "strategy": "recent", "config": {**config, "history_format": history_format},
                  "status": "finished", "steps": len(events), "usage_unknown": False, **usage,
                  "budget": {**usage, "overrun": False, "unreconciled_request": False},
                  "resolved": None, "evaluation": {"status": "official_empty_patch", "resolved": None, "tests_executed": False}}
        write_json(folder / "result.json", result)
        write_rows(folder / "calls.jsonl", calls)
        write_rows(folder / "trajectory.jsonl", events)
        (folder / "patch.diff").write_text("", encoding="utf-8")
    return tmp_path


def mutate(root, file, change):
    path = root / "chat/attempt" / file
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    write_json(path, value)


def test_summary_preserves_ungraded_and_defines_behavior(artifacts):
    report = probe.summarize(artifacts)
    row = report["formats"]["chat"]
    assert report["first_agent_messages_identical"] is True
    assert row["invalid_actions"] == row["repeated_identical_failed_actions"] == 1
    assert row["successful_edit_file"] == row["successful_write_file"] == 1
    assert row["diff_actions"] == row["run_actions"] == 1
    assert row["empty_patch"] is True and row["resolved"] is None and row["official_resolved"] is None
    assert row["tests_executed"] is False and row["budget"]["valid"] is True


@pytest.mark.parametrize("field,value", [
    ("model", "other"), ("model_metadata", {"revision": "b" * 40}),
    ("source_fingerprint", "other"), ("task_fingerprint", "other"),
    ("environment_provenance", {"image": "other"}), ("actual_models", ["other"]),
    ("initial_revision", "r2"), ("usage_sources", ["estimated"]),
])
def test_refuses_changed_conditions(artifacts, field, value):
    mutate(artifacts, "result.json", lambda result: result.update({field: value}))
    with pytest.raises(ValueError, match="Mismatched"):
        probe.summarize(artifacts)


@pytest.mark.parametrize("field,value", [("seed", 8), ("total_tokens", 100000), ("max_output_tokens", 32)])
def test_only_history_format_may_differ(artifacts, field, value):
    mutate(artifacts, "result.json", lambda result: result["config"].update({field: value}))
    with pytest.raises(ValueError, match="Mismatched"):
        probe.summarize(artifacts)


def test_first_prompt_must_match(artifacts):
    path = artifacts / "chat/attempt/calls.jsonl"
    calls = probe._rows(path)
    calls[0]["messages"][0]["content"] = "different task framing"
    write_rows(path, calls)
    with pytest.raises(ValueError, match="First agent messages differ"):
        probe.summarize(artifacts)


@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_missing_or_extra_attempt_refused(artifacts, mode):
    path = artifacts / "chat/attempt/result.json"
    if mode == "missing":
        path.unlink()
    else:
        duplicate = artifacts / "chat/second"
        duplicate.mkdir()
        (duplicate / "result.json").write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="exactly one result"):
        probe.summarize(artifacts)


def test_rejects_two_attempts_of_one_format(artifacts):
    mutate(artifacts, "result.json", lambda result: result["config"].update(history_format="transcript"))
    with pytest.raises(ValueError, match="one transcript and one chat"):
        probe.summarize(artifacts)


def test_missing_fingerprint_rejected(artifacts):
    mutate(artifacts, "result.json", lambda result: result.pop("task_fingerprint"))
    with pytest.raises(ValueError, match="Missing comparison evidence"):
        probe.summarize(artifacts)


def test_both_runs_must_match_predeclared_plan(artifacts):
    path = artifacts / "plan.json"
    plan = json.loads(path.read_text())
    plan["source_fingerprint"] = "before"
    write_json(path, plan)
    with pytest.raises(ValueError, match="predeclared plan"):
        probe.summarize(artifacts)


def test_budget_overrun_and_nonempty_patch_do_not_become_success(artifacts):
    mutate(artifacts, "result.json", lambda result: result["budget"].update(overrun=True))
    mutate(artifacts, "result.json", lambda result: result.update(
        evaluation={"status": "pending_official_evaluation", "resolved": None}))
    (artifacts / "chat/attempt/patch.diff").write_text("ungraded patch", encoding="utf-8")
    row = probe.summarize(artifacts)["formats"]["chat"]
    assert row["budget"]["valid"] is False and row["empty_patch"] is False
    assert row["resolved"] is None and row["official_resolved"] is None


def test_infrastructure_failure_keeps_raw_official_value(artifacts):
    mutate(artifacts, "result.json", lambda result: result.update(
        evaluation={"status": "infrastructure_error", "resolved": None, "official_resolved": False}))
    row = probe.summarize(artifacts)["formats"]["chat"]
    assert row["resolved"] is None and row["official_resolved"] is False
    assert row["evaluation_status"] == "infrastructure_error"


def test_tampered_usage_refused(artifacts):
    mutate(artifacts, "result.json", lambda result: result.update(total_tokens=1))
    with pytest.raises(ValueError, match="token accounting"):
        probe.summarize(artifacts)


def test_default_does_not_start_model(monkeypatch, capsys):
    monkeypatch.setattr(probe, "run_probe", lambda *args: pytest.fail("implicit model run"))
    assert probe.main([]) == 0
    assert "summarize" in capsys.readouterr().out


def test_run_saves_plan_first_and_reuses_one_offline_model(tmp_path, monkeypatch):
    import budget_agent.hf_model

    config = RunConfig(provider="transformers", strategy="recent", environment="docker",
                       model="test/model", model_revision="a" * 40)
    task = SimpleNamespace(id="task", split="dev")
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text("{}", encoding="utf-8")
    output = tmp_path / "probe"
    constructed, runs = [], []

    def model_factory(settings):
        assert (output / "plan.json").is_file()
        assert probe.os.environ["HF_HUB_OFFLINE"] == probe.os.environ["TRANSFORMERS_OFFLINE"] == "1"
        model = SimpleNamespace(counter=object())
        constructed.append(model)
        return model

    def run(task_arg, settings, folder, *, model, counter):
        runs.append((task_arg, settings, folder, model, counter))
        return {"status": "finished", "result_path": str(folder / "result.json")}

    monkeypatch.setattr(probe, "load_config", lambda args: config)
    monkeypatch.setattr(probe, "load_tasks", lambda *args, **kwargs: [task])
    monkeypatch.setattr(budget_agent.hf_model, "TransformersModel", model_factory)
    monkeypatch.setattr(probe, "run_task", run)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    probe.run_probe(tmp_path / "config", manifest, "task", output)
    assert len(constructed) == 1
    assert [run[1].history_format for run in runs] == list(probe.FORMATS)
    assert all(run[3] is constructed[0] and run[4] is constructed[0].counter for run in runs)
    assert runs[0][1].seed == runs[1][1].seed
    with pytest.raises(FileExistsError):
        probe.run_probe(tmp_path / "config", manifest, "task", output)
    assert len(runs) == 2
