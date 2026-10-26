from dataclasses import asdict
import json
from pathlib import Path

from budget_agent.cli import build_parser, load_config, main
from budget_agent.tasks import load_tasks


def test_grading_infrastructure_failure_returns_nonzero(tmp_path):
    root = Path(__file__).resolve().parents[1]
    task = load_tasks(root / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]
    task.verifier = str(tmp_path / "missing_verifier.py")
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(json.dumps(asdict(task)) + "\n", encoding="utf-8")
    code = main(["run", "--config", str(root / "configs/smoke.toml"), "--tasks", str(manifest), "--output", str(tmp_path / "runs")])
    assert code == 1
    result = json.loads(next((tmp_path / "runs").rglob("result.json")).read_text())
    assert result["resolved"] is None
    assert result["evaluation"]["status"] == "infrastructure_error"


def test_transformers_compare_loads_weights_once(tmp_path, monkeypatch):
    from budget_agent import hf_model, runner, report, tasks
    from budget_agent.schema import Task

    loaded, attempts = [], []

    class Model:
        def __init__(self, config):
            loaded.append(config)
            self.counter = object()

    def run(task, config, output, *, model, counter):
        attempts.append((task.id, config.strategy, config.seed, model, counter))
        return {"status": "finished", "resolved": False, "total_tokens": 1,
                "evaluation": {"status": "unresolved"}}

    monkeypatch.setattr(hf_model, "TransformersModel", Model)
    monkeypatch.setattr(runner, "run_task", run)
    monkeypatch.setattr(tasks, "load_tasks", lambda *args: [Task("a", "a", "dev"), Task("b", "b", "dev")])
    monkeypatch.setattr(report, "generate_report", lambda *args: {"html": tmp_path / "report.html"})
    assert main(["compare", "--provider", "transformers", "--model", "test/model",
                 "--model-revision", "a" * 40, "--strategies", "recent", "structured",
                 "--repeat", "2", "--seed", "7", "--output", str(tmp_path)]) == 0
    assert len(loaded) == 1 and len(attempts) == 8
    assert {id(attempt[3]) for attempt in attempts} == {id(attempts[0][3])}
    assert {attempt[2] for attempt in attempts} == {7, 8}
    assert all(attempt[4] is attempt[3].counter for attempt in attempts)


def test_transformers_config_does_not_inherit_api_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    args = build_parser().parse_args(["run", "--provider", "transformers", "--model", "test/model",
                                      "--model-revision", "a" * 40])
    config = load_config(args)
    assert config.base_url == "" and config.environment == "docker"


def test_diagnose_reads_zero_call_run_without_changing_source(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    result = source / "result.json"
    result.write_text(json.dumps({"task_id": "a", "run_id": "r1"}), encoding="utf-8")
    original = result.read_bytes()
    output = tmp_path / "analysis.json"
    assert main(["diagnose", "--input", str(source), "--output", str(output)]) == 0
    analysis = json.loads(output.read_text())
    assert analysis["runs"][0]["diagnostics"]["counts"]["events"] == 0
    assert result.read_bytes() == original
    assert list(source.iterdir()) == [result]
    assert main(["diagnose", "--input", str(source), "--output", str(output)]) == 2


def test_action_floor_cli_and_config_validation():
    import pytest
    args = build_parser().parse_args(["run", "--min-action-output-tokens", "256"])
    assert load_config(args).min_action_output_tokens == 256
    args = build_parser().parse_args(["run", "--min-action-output-tokens", "2048"])
    with pytest.raises(ValueError, match="min_action_output_tokens"):
        load_config(args)


def test_explicit_thinking_cli_overrides_config_without_default_injection(tmp_path):
    config = tmp_path / "api.toml"
    config.write_text('[run]\napi_enable_thinking = false\n', encoding="utf-8")
    assert load_config(build_parser().parse_args(["run"])).api_enable_thinking is None
    assert load_config(build_parser().parse_args(["run", "--config", str(config)])).api_enable_thinking is False
    assert load_config(build_parser().parse_args(["run", "--config", str(config), "--api-enable-thinking"])).api_enable_thinking is True
    assert load_config(build_parser().parse_args(["run", "--no-api-enable-thinking"])).api_enable_thinking is False


def test_reference_pilot_config_is_pinned_and_non_thinking():
    root = Path(__file__).resolve().parents[1]
    config = load_config(build_parser().parse_args(["run", "--config", str(root / "configs/reference-qwen38-max.toml")]))
    assert config.provider == "openai-compatible" and config.environment == "docker"
    assert config.model == "qwen3.8-max-0902"
    assert config.api_enable_thinking is False
    assert config.api_max_tokens_field == "max_completion_tokens"
    assert config.loop_policy == "off" and config.strategy == "recent"
    assert config.total_tokens == 128000 and config.max_output_tokens == 2048
    assert config.input_price_per_million is None and config.output_price_per_million is None
