"""Independent outcome, token accounting, and tool-policy acceptance contracts."""
from __future__ import annotations

import re
from typing import Any

from .verification import fixture_outcome


def grading_input_match(records: list[dict]) -> str:
    """Compare the recorded grading materials for a single task."""
    identities = set()
    for record in records:
        material = record.get("evaluation", {}).get("grading_inputs")
        match record.get("task_kind"), material:
            case "local", {"kind": "fixture", "verifier_sha256": digest}:
                hashes = (digest,)
            case "swebench", {"kind": "swebench", "eval_script_sha256": script, "test_targets_sha256": targets}:
                hashes = (script, targets)
            case _:
                hashes = ()
        valid = hashes and all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes)
        identities.add(hashes if valid else None)
    if len(identities - {None}) > 1:
        return "mismatched"
    return "matched" if identities and None not in identities else "unavailable"


def evidence_kind(result: dict[str, Any], default: str = "unknown") -> str:
    """Older mini-SWE CPU runs declared the configured provider despite using a test double."""
    metadata = result.get("model_metadata")
    if isinstance(metadata, dict) and metadata.get("test_double") is True:
        return "scripted_smoke"
    return result.get("evidence_kind", default)


def _dimension(status: str, reason: str, evidence: dict) -> dict:
    return {"status": status, "score": None if status == "unknown" else int(status == "pass"),
            "reason": reason, "evidence": evidence}


def is_infrastructure_error(result: dict[str, Any]) -> bool:
    evaluation = result.get("evaluation", {})
    return (result.get("status") in {"environment_error", "model_error"}
            or evaluation.get("infra_failure") is True
            or evaluation.get("status") in {"infrastructure_error", "grading_error"})


def verified_outcome(result: dict[str, Any]) -> bool | None:
    """One verifier interpretation for aggregate reports and paired comparisons."""
    evaluation = result.get("evaluation", {})
    value = result.get("resolved")
    status = evaluation.get("status")
    if status in {"resolved", "unresolved", "invalid_patch"} and value is not (status == "resolved"):
        return None
    # A stored fixture verdict must agree with its summary before it is counted.
    if status in {"resolved", "unresolved"} and "test_results" in evaluation:
        try:
            outcome = fixture_outcome(evaluation["test_results"])
        except ValueError:
            return None
        if outcome is None or outcome is not value:
            return None
    ungraded = {"official_empty_patch", "pending_official_evaluation", "not_evaluated", "pending", "ungraded"}
    if (is_infrastructure_error(result) or result.get("status") == "interrupted"
            or result.get("status") in ungraded or evaluation.get("status") in ungraded
            or evaluation.get("tests_executed") is False
            or ("resolved" in evaluation and evaluation["resolved"] is not result.get("resolved"))):
        return None
    return value if type(value) is bool else None


def _quality(result: dict) -> dict:
    evaluation = result.get("evaluation", {})
    evidence = {"evaluation_status": evaluation.get("status"),
                "resolved": evaluation.get("resolved")}
    if (evaluation.get("status") not in {"resolved", "unresolved", "invalid_patch", "official_swebench"}
            or "resolved" not in evaluation):
        return _dimension("unknown", "No completed independent verifier outcome", evidence)
    outcome = verified_outcome(result)
    if outcome is None:
        return _dimension("unknown", "Verifier evidence is incomplete or inconsistent", evidence)
    return _dimension("pass" if outcome else "fail", "Independent final verifier outcome", evidence)


def _budget(result: dict) -> dict:
    budget = result.get("budget", {})
    evidence = {"total_tokens": budget.get("total_tokens"),
                "limit_tokens": result.get("config", {}).get("total_tokens"),
                "usage_is_estimated": budget.get("usage_is_estimated"),
                "usage_sources": budget.get("usage_sources")}
    if budget.get("overrun") is True or result.get("status") == "budget_overrun":
        return _dimension("fail", "Recorded inference budget overrun", evidence)
    if result.get("usage_unknown") or result.get("status") == "usage_unknown" or budget.get("unreconciled_request"):
        return _dimension("unknown", "Request usage has not been reconciled", evidence)
    required = {"input_tokens", "output_tokens", "total_tokens", "calls", "requests",
                "overrun", "unreconciled_request"}
    if (not required.issubset(budget) or result.get("usage_unknown") is not False
            or budget["overrun"] is not False or budget["unreconciled_request"] is not False):
        return _dimension("unknown", "Token ledger is missing required accounting fields", evidence)
    counts = [budget[key] for key in ("input_tokens", "output_tokens", "total_tokens", "calls")]
    limit = evidence["limit_tokens"]
    if any(type(value) is not int or value < 0 for value in counts) or type(limit) is not int or limit <= 0:
        return _dimension("unknown", "Token ledger or configured limit is invalid", evidence)
    requests = budget["requests"]
    if not isinstance(requests, list):
        return _dimension("unknown", "Request ledger must be a list of request records", evidence)
    invalid_requests, overrun_requests = [], []
    for index, row in enumerate(requests):
        if not isinstance(row, dict):
            invalid_requests.append(index)
            continue
        values = [row.get(key) for key in
                  ("input_tokens", "output_tokens", "estimated_input_tokens", "reserved_output_tokens")]
        valid_counts = all(type(value) is int and value >= 0 for value in values)
        reasons = row.get("overrun_reasons")
        valid_reasons = isinstance(reasons, list) and all(isinstance(reason, str) for reason in reasons)
        if not valid_counts or not valid_reasons:
            invalid_requests.append(index)
        exceeded = valid_reasons and bool(reasons)
        if valid_counts:
            inputs, outputs, estimated_inputs, reserved_outputs = values
            exceeded = exceeded or outputs > reserved_outputs or inputs + outputs > estimated_inputs + reserved_outputs
        if exceeded:
            overrun_requests.append(index)
    evidence.update(invalid_request_indices=invalid_requests, overrun_request_indices=overrun_requests)
    if overrun_requests:
        return _dimension("fail", "Request evidence records usage beyond its reservation", evidence)
    if invalid_requests:
        return _dimension("unknown", "Request ledger contains missing or invalid accounting fields", evidence)
    if (len(requests) != budget["calls"]
            or budget["input_tokens"] + budget["output_tokens"] != budget["total_tokens"]
            or sum(row["input_tokens"] for row in requests) != budget["input_tokens"]
            or sum(row["output_tokens"] for row in requests) != budget["output_tokens"]):
        return _dimension("unknown", "Request ledger does not reconcile with recorded totals", evidence)
    if budget["total_tokens"] > limit:
        return _dimension("fail", "Recorded token total exceeds the configured limit", evidence)
    return _dimension("pass", "Recorded usage reconciles within the token limit; not a billing audit", evidence)


def request_ledger(result: dict) -> dict | None:
    """Recognize a saved request ledger even when one marker is missing."""
    budget = result.get("budget", {})
    return budget if "requests" in budget or "calls" in budget else None


def ledger_token_count(ledger: dict, field: str) -> int | None:
    count = ledger.get(field)
    return count if type(count) is int and count >= 0 else None


def comparison_budget_valid(result: dict) -> bool:
    """Use the audit verdict for request ledgers and retain legacy flag semantics."""
    if request_ledger(result) is not None:
        return _budget(result)["status"] == "pass"
    budget = result.get("budget", {})
    return not (result.get("usage_unknown") or budget.get("overrun") or budget.get("unreconciled_request")
                or result.get("status") in {"budget_overrun", "usage_unknown"})


def _safety(result: dict) -> dict:
    audit = result.get("safety", {})
    evidence = {key: audit.get(key) for key in
                ("policy", "scope", "complete", "evaluated_actions", "blocked_actions", "violations")}
    if audit.get("violations"):
        return _dimension("fail", "Execution-policy violations were recorded", evidence)
    if (audit.get("policy") != "restricted" or audit.get("scope") != "direct_tool_authorization"
            or audit.get("complete") is not True or audit.get("violations") != []):
        return _dimension("unknown", "Complete restricted tool-policy evidence is unavailable", evidence)
    counts = [audit.get("evaluated_actions"), audit.get("blocked_actions")]
    if any(type(value) is not int or value < 0 for value in counts) or counts[1] > counts[0]:
        return _dimension("unknown", "Tool-policy audit counters are incomplete or inconsistent", evidence)
    return _dimension("pass", "Direct tool authorizations satisfied the restricted policy; not a full security audit", evidence)


def evaluate_run(result: dict[str, Any]) -> dict[str, Any]:
    """Keep hard failures and missing evidence visible instead of averaging them away."""
    dimensions = {"quality": _quality(result), "budget": _budget(result), "safety": _safety(result)}
    states = {dimension["status"] for dimension in dimensions.values()}
    acceptance = "fail" if "fail" in states else "unknown" if "unknown" in states else "pass"
    return {"schema_version": 1, "scope": "single_run_contract", "task_id": result.get("task_id"),
            "run_id": result.get("run_id"), "evidence_kind": evidence_kind(result),
            "dimensions": dimensions, "acceptance_status": acceptance,
            "blocking_dimensions": [name for name, value in dimensions.items() if value["status"] != "pass"],
            "limitations": ["Acceptance covers this run's recorded contracts, not production readiness.",
                            "Scripted runs validate software behavior, not model capability.",
                            "Token accounting does not verify external billing or complete sandbox isolation."]}
