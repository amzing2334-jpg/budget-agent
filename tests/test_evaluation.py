from copy import deepcopy

import pytest

from budget_agent.budget import Budget, TokenCounter
from budget_agent.evaluation import evaluate_run
from budget_agent.schema import ModelResponse, RunConfig


def complete_result():
    config = RunConfig(total_tokens=1000)
    budget = Budget(TokenCounter(), config)
    budget.reserve([{"role": "user", "content": "fix"}], 10)
    budget.record(ModelResponse("done", 10, 2, usage_source="provider", model="fixture"))
    return {"task_id": "task", "run_id": "run", "evidence_kind": "scripted_smoke",
            "status": "finished", "resolved": True, "usage_unknown": False,
            "evaluation": {"status": "resolved", "resolved": True,
                           "test_results": {"tests_run": 1, "skipped": 0, "successful": True}},
            "budget": budget.to_dict(), "config": {"total_tokens": config.total_tokens},
            "safety": {"policy": "restricted", "scope": "direct_tool_authorization", "complete": True,
                       "evaluated_actions": 2, "blocked_actions": 0, "violations": []}}


def test_complete_contract_preserves_scope_and_does_not_mutate_input():
    result = complete_result()
    original = deepcopy(result)
    report = evaluate_run(result)
    assert report["acceptance_status"] == "pass"
    assert report["scope"] == "single_run_contract" and report["evidence_kind"] == "scripted_smoke"
    assert report["blocking_dimensions"] == []
    assert all(value["score"] == 1 for value in report["dimensions"].values())
    assert result == original


@pytest.mark.parametrize("field,axis", [("evaluation", "quality"), ("budget", "budget"), ("safety", "safety")])
def test_missing_evidence_never_becomes_pass(field, axis):
    result = complete_result()
    del result[field]
    report = evaluate_run(result)
    assert report["acceptance_status"] == "unknown"
    assert report["dimensions"][axis]["score"] is None
    assert report["blocking_dimensions"] == [axis]


@pytest.mark.parametrize("evaluation", [
    {"status": "unresolved", "resolved": False, "test_results": None},
    {"status": "official_empty_patch", "resolved": False},
    {"status": "pending_official_evaluation", "resolved": None},
    {"status": "grading_error", "resolved": None},
    {"status": "unresolved", "resolved": False, "test_results": {"tests_run": 1, "skipped": 1}},
])
def test_ungraded_final_outcome_is_unknown(evaluation):
    result = complete_result()
    result.update(evaluation=evaluation, resolved=evaluation["resolved"])
    assert evaluate_run(result)["dimensions"]["quality"]["status"] == "unknown"


def test_known_failure_is_not_hidden_by_high_scores_or_unknown_axis():
    result = complete_result()
    result.update(resolved=False, evaluation={"resolved": False, "status": "official_swebench"})
    del result["safety"]
    report = evaluate_run(result)
    assert report["acceptance_status"] == "fail"
    assert report["blocking_dimensions"] == ["quality", "safety"]


@pytest.mark.parametrize("change,status", [
    ({"overrun": True}, "fail"), ({"unreconciled_request": True}, "unknown"),
    ({"requests": []}, "unknown"), ({"input_tokens": 9}, "unknown"),
    ({"total_tokens": -1}, "unknown"), ({"calls": 2}, "unknown"),
    ({"overrun": None}, "unknown"), ({"unreconciled_request": None}, "unknown"),
])
def test_budget_uses_complete_reconciled_ledger(change, status):
    result = complete_result()
    result["budget"].update(change)
    assert evaluate_run(result)["dimensions"]["budget"]["status"] == status


def test_token_limit_overrun_does_not_depend_on_stored_overrun_flag():
    result = complete_result()
    result["config"]["total_tokens"] = 11
    assert evaluate_run(result)["dimensions"]["budget"]["status"] == "fail"


def test_unknown_usage_is_not_known_free_usage():
    result = complete_result()
    result["usage_unknown"] = True
    assert evaluate_run(result)["dimensions"]["budget"]["status"] == "unknown"


@pytest.mark.parametrize("change,status", [
    ({"policy": "sandbox"}, "unknown"), ({"complete": False}, "unknown"),
    ({"scope": "claimed_full_security"}, "unknown"), ({"blocked_actions": 2}, "pass"),
    ({"evaluated_actions": None}, "unknown"), ({"blocked_actions": 3}, "unknown"),
    ({"violations": [{"rule": "executed_unapproved_action"}]}, "fail"),
])
def test_safety_distinguishes_blocked_attempts_violations_and_missing_audit(change, status):
    result = complete_result()
    result["safety"].update(change)
    report = evaluate_run(result)
    assert report["dimensions"]["safety"]["status"] == status
    assert report["acceptance_status"] == status


@pytest.mark.parametrize("requests", [None, {}, "unavailable", [None], ["row"], [1]])
def test_request_ledger_invalid_container_or_row_is_unknown(requests):
    result = complete_result()
    result["budget"]["requests"] = requests
    report = evaluate_run(result)
    assert report["dimensions"]["budget"]["status"] == "unknown"
    assert report["acceptance_status"] == "unknown"


@pytest.mark.parametrize("field", [
    "input_tokens", "output_tokens", "estimated_input_tokens",
    "reserved_output_tokens", "overrun_reasons",
])
def test_request_ledger_missing_required_field_is_unknown(field):
    result = complete_result()
    del result["budget"]["requests"][0][field]
    assert evaluate_run(result)["dimensions"]["budget"]["status"] == "unknown"


@pytest.mark.parametrize("field,value", [
    ("input_tokens", -1), ("input_tokens", True), ("input_tokens", 10.0),
    ("output_tokens", -1), ("output_tokens", False), ("output_tokens", 2.0),
    ("estimated_input_tokens", -1), ("estimated_input_tokens", True),
    ("estimated_input_tokens", 10.0), ("reserved_output_tokens", -1),
    ("reserved_output_tokens", False), ("reserved_output_tokens", 10.0),
    ("input_tokens", "10"), ("output_tokens", None),
    ("overrun_reasons", None), ("overrun_reasons", "exceeded"),
    ("overrun_reasons", [1]), ("overrun_reasons", [None]),
])
def test_request_ledger_invalid_field_is_unknown(field, value):
    result = complete_result()
    result["budget"]["requests"][0][field] = value
    report = evaluate_run(result)
    assert report["dimensions"]["budget"]["status"] == "unknown"
    assert report["acceptance_status"] == "unknown"


def test_request_negative_usage_cannot_cancel_across_rows_into_a_pass():
    result = complete_result()
    first = result["budget"]["requests"][0]
    second = deepcopy(first)
    first["input_tokens"], first["output_tokens"] = -1, 0
    second["input_tokens"], second["output_tokens"] = 11, 2
    result["budget"]["requests"].append(second)
    result["budget"]["calls"] = 2
    # Both aggregate totals still reconcile, but a negative request is invalid.
    assert sum(row["input_tokens"] for row in result["budget"]["requests"]) == result["budget"]["input_tokens"]
    assert sum(row["output_tokens"] for row in result["budget"]["requests"]) == result["budget"]["output_tokens"]
    assert evaluate_run(result)["dimensions"]["budget"]["status"] == "unknown"


@pytest.mark.parametrize("change", [
    {"overrun_reasons": ["provider output exceeded reserved output cap"]},
    {"reserved_output_tokens": 1},
    {"estimated_input_tokens": 0, "reserved_output_tokens": 10},
])
def test_request_overrun_is_fail_even_when_summary_flag_is_false(change):
    result = complete_result()
    result["budget"]["requests"][0].update(change)
    assert result["budget"]["overrun"] is False
    report = evaluate_run(result)
    assert report["dimensions"]["budget"]["status"] == "fail"
    assert report["acceptance_status"] == "fail"


@pytest.mark.parametrize("bad_first", [False, True])
@pytest.mark.parametrize("change", [
    {"overrun_reasons": ["known request overrun"]},
    {"reserved_output_tokens": 1},
    {"estimated_input_tokens": 0, "reserved_output_tokens": 10},
])
def test_known_request_overrun_is_not_hidden_by_another_invalid_row(change, bad_first):
    result = complete_result()
    known = result["budget"]["requests"][0]
    known.update(change)
    result["budget"]["requests"] = [None, known] if bad_first else [known, None]
    result["budget"]["calls"] = 2
    report = evaluate_run(result)
    assert report["dimensions"]["budget"]["status"] == "fail"
    assert report["acceptance_status"] == "fail"
