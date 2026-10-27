"""CPU end-to-end policy, provenance and independent-acceptance contracts."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from budget_agent.budget import TokenCounter
from budget_agent.cli import build_parser, load_config, main
from budget_agent.models import ScriptedModel, parse_action
from budget_agent.runner import run_task
from budget_agent.schema import RunConfig
from budget_agent.tasks import load_tasks


ROOT = Path(__file__).resolve().parents[1]


def task():
    return load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]


def test_restricted_run_records_denial_and_audits_verified_repair(tmp_path):
    target = task()
    source = (Path(target.repo) / "slugify.py").read_text(encoding="utf-8")
    counter = TokenCounter()
    actions = [
        {"tool": "run", "command": "echo arbitrary shell must not run"},
        {"tool": "write_file", "path": "slugify.py", "content": source.replace("[^a-z]+", "[^a-z0-9]+")},
        {"tool": "public_test"}, {"tool": "finish"},
    ]
    config = RunConfig(execution_policy="restricted", reflection_policy="failure",
                       context_tokens=16000, max_output_tokens=1536)
    result = run_task(target, config, tmp_path / "runs", model=ScriptedModel(actions, counter))
    assert result["resolved"] is True
    assert result["reflection"]["calls"] == 0  # A policy denial is not a reflection opportunity.
    assert result["safety"] == {"policy": "restricted", "scope": "direct_tool_authorization",
                                "complete": True, "evaluated_actions": 3,
                                "blocked_actions": 1, "violations": []}
    source_result = Path(result["result_path"])
    original = source_result.read_bytes()
    decisions = [json.loads(line) for line in source_result.with_name("security.jsonl").read_text().splitlines()]
    assert decisions[0]["decision"]["rule"] == "free_shell_disabled"
    assert decisions[0]["event_id"] == "e0001"
    output = tmp_path / "audit.json"
    assert main(["audit", "--input", str(source_result), "--output", str(output)]) == 0
    audit = json.loads(output.read_text())["runs"][0]["evaluation"]
    assert audit["acceptance_status"] == "pass"
    assert audit["evidence_kind"] == "scripted_smoke"
    assert source_result.read_bytes() == original
    assert main(["audit", "--input", str(source_result), "--output", str(output)]) == 2
    from budget_agent.report import generate_report
    report = generate_report(source_result.parent, tmp_path / "report")
    for kind in ("html", "markdown"):
        text = report[kind].read_text(encoding="utf-8")
        assert "Run acceptance contracts" in text
        assert "not production certification" in text


def test_original_sandbox_result_does_not_claim_restricted_safety(tmp_path):
    result = run_task(task(), RunConfig(max_output_tokens=1536), tmp_path / "runs")
    output = tmp_path / "audit.json"
    assert main(["audit", "--input", result["result_path"], "--output", str(output)]) == 1
    audit = json.loads(output.read_text())["runs"][0]["evaluation"]
    assert audit["dimensions"]["quality"]["status"] == "pass"
    assert audit["dimensions"]["safety"]["status"] == "unknown"
    assert audit["acceptance_status"] == "unknown"


def test_public_command_is_part_of_task_fingerprint(tmp_path):
    target = task()
    config = RunConfig(execution_policy="restricted")
    counter = TokenCounter()
    first = run_task(target, config, tmp_path, model=ScriptedModel([{"tool": "finish"}], counter))
    changed = replace(target, test_command=["python", "-m", "unittest"])
    second = run_task(changed, config, tmp_path, model=ScriptedModel([{"tool": "finish"}], counter))
    assert first["task_fingerprint"] != second["task_fingerprint"]
    assert first["public_test_command"] == target.test_command


def test_protected_file_denial_is_audited_and_never_reflected(tmp_path):
    counter = TokenCounter()
    actions = [{"tool": "edit_file", "path": "tests/test_public.py", "old": "assert", "new": "pass"},
               {"tool": "finish"}]
    result = run_task(task(), RunConfig(execution_policy="restricted", reflection_policy="failure"),
                      tmp_path, model=ScriptedModel(actions, counter))
    assert result["reflection"]["calls"] == 0
    assert result["safety"]["blocked_actions"] == 1
    assert result["safety"]["complete"] is True
    rows = [json.loads(line) for line in Path(result["result_path"]).with_name("security.jsonl").read_text().splitlines()]
    assert rows[0]["decision"]["rule"] == "repository_boundary"


@pytest.mark.parametrize("changes", [
    {"reflection_policy": "automatic"}, {"max_reflections": 0},
    {"reflection_max_output_tokens": 0}, {"execution_policy": "unrestricted"},
    {"execution_policy": "restricted", "provider": "openai-compatible", "allow_local_execution": True},
])
def test_invalid_governance_configuration_fails_before_execution(changes):
    with pytest.raises(ValueError):
        RunConfig(**changes).validate()


@pytest.mark.parametrize("policy", ["failure", "evidence"])
def test_governance_flags_are_explicit_and_default_off(policy):
    defaults = load_config(build_parser().parse_args(["run"]))
    assert defaults.reflection_policy == "off" and defaults.execution_policy == "sandbox"
    configured = load_config(build_parser().parse_args([
        "run", "--reflection-policy", policy, "--max-reflections", "2",
        "--reflection-max-output-tokens", "512", "--execution-policy", "restricted"]))
    assert configured.reflection_policy == policy and configured.max_reflections == 2
    assert configured.reflection_max_output_tokens == 512 and configured.environment == "docker"
    assert parse_action('{"tool":"public_test"}') == {"tool": "public_test"}
    with pytest.raises(ValueError, match="no parameters"):
        parse_action('{"tool":"public_test","command":"echo replaced"}')


def test_openjudge_audit_cli_uses_real_optional_runner(tmp_path):
    pytest.importorskip("openjudge", reason="Optional OpenJudge contract environment is not configured")
    result = run_task(task(), RunConfig(execution_policy="restricted", max_output_tokens=1536), tmp_path / "runs")
    # Existing scripted smoke uses run; it is blocked, but independent grading remains decisive.
    output = tmp_path / "openjudge-audit.json"
    assert main(["audit", "--engine", "openjudge", "--input", result["result_path"], "--output", str(output)]) == 0
    actual = json.loads(output.read_text())
    assert actual["openjudge"]["acceptance"][0]["score"] == 1
    assert actual["openjudge"]["acceptance"][0]["metadata"]["scope"] == "single_run_contract"


def test_openjudge_grader_error_cannot_fall_back_to_builtin_success(tmp_path, monkeypatch):
    pytest.importorskip("openjudge", reason="Optional OpenJudge contract environment is not configured")
    from budget_agent.openjudge_adapter import RunContractGrader
    original = RunContractGrader._aevaluate

    async def failed_quality(self, result):
        if self.dimension == "quality":
            raise RuntimeError("injected grader failure")
        return await original(self, result)

    monkeypatch.setattr(RunContractGrader, "_aevaluate", failed_quality)
    result = run_task(task(), RunConfig(execution_policy="restricted", max_output_tokens=1536), tmp_path / "runs")
    output = tmp_path / "failed-judge.json"
    assert main(["audit", "--engine", "openjudge", "--input", result["result_path"], "--output", str(output)]) == 1
    audit = json.loads(output.read_text())
    assert audit["runs"][0]["evaluation"]["acceptance_status"] == "pass"
    assert "error" in audit["openjudge"]["quality"][0]
