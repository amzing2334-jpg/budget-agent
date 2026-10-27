"""Optional tests run against real OpenJudge, never a module mock."""
import asyncio

import pytest

pytest.importorskip("openjudge", reason="Optional OpenJudge contract environment is not configured")

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderError, GraderScore
from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator

from budget_agent.openjudge_adapter import RunContractGrader, evaluate_openjudge
from test_evaluation import complete_result


def test_real_openjudge_async_runner_preserves_scores_and_unknowns():
    passed = complete_result()
    unknown = complete_result()
    del unknown["safety"]
    failed = complete_result()
    failed["safety"]["violations"] = [{"rule": "forbidden_execution"}]
    reports = asyncio.run(evaluate_openjudge([passed, unknown, failed]))
    assert len(reports) == 4
    acceptance = reports["acceptance"]
    assert acceptance[0]["score"] == 1
    assert acceptance[1]["error"] == "insufficient_evidence" and "score" not in acceptance[1]
    assert acceptance[2]["score"] == 0
    assert all(row["metadata"]["evidence_kind"] == "scripted_smoke" for row in acceptance)


def test_adapter_uses_upstream_types_without_converting_unknown_to_zero():
    grader = RunContractGrader("quality")
    assert isinstance(grader, BaseGrader)
    score = asyncio.run(grader.aevaluate(result=complete_result()))
    assert isinstance(score, GraderScore) and score.score == 1
    result = complete_result()
    del result["evaluation"]
    error = asyncio.run(grader.aevaluate(result=result))
    assert isinstance(error, GraderError) and error.error == "insufficient_evidence"


def test_upstream_weighted_average_is_not_an_acceptance_gate():
    # Demonstrates why our gate must not use the upstream weighted aggregator.
    upstream = WeightedSumAggregator("average")
    averaged = upstream({"quality": GraderScore(name="quality", reason="ok", score=1),
                         "safety": GraderError(name="safety", error="missing", reason="unknown")})
    assert averaged.score == 1
    result = complete_result()
    del result["safety"]
    actual = asyncio.run(RunContractGrader().aevaluate(result=result))
    assert isinstance(actual, GraderError)
