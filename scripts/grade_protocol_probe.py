"""Grade a completed protocol probe in a separate Linux evaluator directory."""
import argparse
import json
from pathlib import Path
import subprocess

from budget_agent.report import generate_report
from budget_agent.swebench import attach_grades, evaluation_command, export_predictions
from probe_history_format import summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--grading-dir", required=True, type=Path)
    parser.add_argument("--grader-python", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true", help="Otherwise print the plan without running the grader")
    args = parser.parse_args(argv)
    runs, grading = args.runs.resolve(), args.grading_dir.resolve()
    project = Path(__file__).resolve().parents[1]
    if grading.is_relative_to(project):
        parser.error("Keep gold and hidden evaluator outputs outside the Agent project")
    diagnostic = summarize(runs)
    task_id = diagnostic["condition"]["task_id"]
    plan = []
    for history_format, record in diagnostic["formats"].items():
        run = Path(record["result_path"]).parent
        run_id = f"{args.run_id}-{history_format}"
        predictions = grading / f"{run_id}.jsonl"
        command = evaluation_command(predictions, run_id=run_id, max_workers=1,
                                     python_executable=args.grader_python,
                                     instance_ids=[task_id], timeout=600)
        plan.append((run, run_id, predictions, command))
    print(json.dumps({"task_id": task_id, "grading_dir": str(grading),
                      "commands": [item[-1] for item in plan]}, indent=2))
    if not args.execute:
        return
    grading.mkdir(parents=True, exist_ok=False)
    for run, run_id, predictions, command in plan:
        # A fresh evaluator directory preserves earlier reports and predictions.
        export_predictions(run, predictions)
        with (grading / f"{run_id}.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, cwd=grading, stdout=log, stderr=subprocess.STDOUT, check=True)
        prediction = json.loads(predictions.read_bytes())
        model_slug = prediction["model_name_or_path"].replace("/", "__")
        attached = attach_grades(run, grading / "logs/run_evaluation" / run_id,
                                 summary_path=grading / f"{model_slug}.{run_id}.json",
                                 predictions_path=predictions)
        print(json.dumps({"run_id": run_id, **attached}), flush=True)
    generate_report(runs, runs / "report")
    with (runs / "protocol-after-grading.json").open("x", encoding="utf-8") as out:
        json.dump(summarize(runs), out, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
