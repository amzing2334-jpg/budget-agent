"""Optional OpenJudge integration; deterministic graders make no model requests."""
from __future__ import annotations

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderError, GraderScore
from openjudge.runner.grading_runner import GradingRunner

from .evaluation import evaluate_run


class RunContractGrader(BaseGrader):
    """Expose one contract through the actual OpenJudge score/error protocol."""

    def __init__(self, dimension: str = "acceptance"):
        if dimension not in {"quality", "budget", "safety", "acceptance"}:
            raise ValueError(f"Unknown run-contract dimension: {dimension}")
        super().__init__(name=f"budget_agent_{dimension}", description="Recorded coding-agent contract")
        self.dimension = dimension

    async def _aevaluate(self, result: dict) -> GraderScore | GraderError:
        report = evaluate_run(result)
        if self.dimension == "acceptance":
            status = report["acceptance_status"]
            reason = "All three contracts must pass; no weighted compensation"
            evidence = report
        else:
            dimension = report["dimensions"][self.dimension]
            status, reason, evidence = dimension["status"], dimension["reason"], dimension["evidence"]
        metadata = {"status": status, "scope": report["scope"], "evidence_kind": report["evidence_kind"],
                    "evidence": evidence}
        if status == "unknown":
            return GraderError(name=self.name, reason=reason, error="insufficient_evidence", metadata=metadata)
        return GraderScore(name=self.name, reason=reason, score=float(status == "pass"), metadata=metadata)


async def evaluate_openjudge(results: list[dict]) -> dict[str, list[dict]]:
    """Use OpenJudge's bounded async runner without its weighted aggregator."""
    runner = GradingRunner({name: RunContractGrader(name) for name in
                            ("quality", "budget", "safety", "acceptance")},
                           max_concurrency=4, show_progress=False)
    grades = await runner.arun([{"result": result} for result in results])
    return {name: [grade.model_dump() for grade in values] for name, values in grades.items()}
