"""Offline reports that keep smoke tests and incompatible experiments separate."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any


def load_results(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    paths = [root] if root.is_file() else sorted(root.rglob("result.json"))
    results = []
    for path in paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(item, dict) or "task_id" not in item:
            raise ValueError(f"Not a run result: {path}")
        item = dict(item)
        item["_result_path"] = str(path.resolve())
        results.append(item)
    if not results:
        raise ValueError(f"No result.json files found in {root}")
    return results


def condition_key(result: dict[str, Any]) -> str:
    """Only strategy and the local repetition label may vary in a comparison."""
    config = {k: v for k, v in result.get("config", {}).items() if k not in {"strategy", "seed"}}
    fields = {k: result.get(k) for k in ("provider", "model", "split", "task_kind", "evidence_kind", "harness")}
    fields["config"] = config
    fields["usage_sources"] = sorted(result.get("usage_sources", []))
    fields["actual_models"] = sorted(set(result.get("actual_models", [])))
    fields["source_fingerprint"] = result.get("source_fingerprint")
    fields["model_metadata"] = result.get("model_metadata")
    return json.dumps(fields, ensure_ascii=False, sort_keys=True)


def _invalid_budget(result: dict[str, Any]) -> bool:
    budget = result.get("budget", {})
    return bool(result.get("usage_unknown") or budget.get("overrun") or
                budget.get("unreconciled_request") or
                result.get("status") in {"budget_overrun", "usage_unknown"})


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
        verdict = evaluation["test_results"]
        if not isinstance(verdict, dict):
            return None
        counts = [verdict.get(key) for key in ("tests_run", "skipped")]
        counts.extend(verdict[key] for key in ("failures", "errors") if key in verdict)
        if (any(type(count) is not int or count < 0 for count in counts)
                or verdict["tests_run"] <= verdict["skipped"]
                or type(verdict.get("successful")) is not bool
                or verdict["successful"] is not value):
            return None
        # Subtest failures may exceed tests_run; zero counts do not rule out an
        # unexpectedSuccess, which also makes unittest.wasSuccessful() false.
        if value and any(verdict.get(key, 0) > 0 for key in ("failures", "errors")):
            return None
    ungraded = {"official_empty_patch", "pending_official_evaluation", "not_evaluated", "pending", "ungraded"}
    if (is_infrastructure_error(result) or result.get("status") == "interrupted"
            or result.get("status") in ungraded or evaluation.get("status") in ungraded
            or evaluation.get("tests_executed") is False
            or ("resolved" in evaluation and evaluation["resolved"] is not result.get("resolved"))):
        return None
    return value if type(value) is bool else None


def _wilson(successes: int, n: int) -> tuple[float, float] | None:
    if n == 0:
        return None
    z = 1.96
    p = successes / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, center - half), min(1, center + half)


def aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[condition_key(result)].append(result)
    rows = []
    for condition, group in sorted(grouped.items()):
        group_id = hashlib.sha256(condition.encode()).hexdigest()[:10]
        policies: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in group:
            policies[result.get("strategy", "unknown")].append(result)
        samples = [Counter((r["task_id"], r.get("task_fingerprint"), r.get("config", {}).get("seed", 0)) for r in records) for records in policies.values()]
        matched = all(sample == samples[0] for sample in samples)
        group_invalid_budget = any(_invalid_budget(r) for r in group)
        group_ungraded = any(verified_outcome(r) is None for r in group)
        for strategy, records in sorted(policies.items()):
            n = len(records)
            graded = sum(verified_outcome(r) is not None for r in records)
            successes = sum(verified_outcome(r) is True for r in records)
            infrastructure_errors = sum(is_infrastructure_error(r) for r in records)
            empty_patches = sum(r.get("evaluation", {}).get("status") == "official_empty_patch" for r in records)
            invalid_budget_attempts = sum(_invalid_budget(r) for r in records)
            costs = [r.get("cost_usd") for r in records]
            cost = sum(costs) if all(isinstance(c, (int, float)) for c in costs) else None
            input_tokens = sum(r.get("input_tokens", 0) for r in records)
            output_tokens = sum(r.get("output_tokens", 0) for r in records)
            kind = records[0].get("evidence_kind", "unknown")
            # Repeats on the same issue are dependent: do not manufacture confidence.
            unique_tasks = len({r["task_id"] for r in records})
            interval = _wilson(successes, n) if kind == "model_run" and graded == n and unique_tasks == n and not invalid_budget_attempts else None
            rows.append({
                "group_id": group_id, "condition": json.loads(condition),
                "strategy": strategy, "provider": records[0].get("provider", ""),
                "model": records[0].get("model", ""), "split": records[0].get("split", ""),
                "evidence_kind": kind, "attempted": n, "graded": graded,
                "pending_or_ungraded": n - graded, "successes": successes,
                "infrastructure_errors": infrastructure_errors,
                "official_empty_patches": empty_patches,
                "success_rate": successes / n if graded == n and not invalid_budget_attempts else None,
                "wilson95": interval, "matched_tasks_and_repetition_labels": matched,
                "invalid_budget_attempts": invalid_budget_attempts,
                "comparison_valid": matched and not group_invalid_budget and not group_ungraded,
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "mean_tokens": (input_tokens + output_tokens) / n,
                "total_cost_usd": cost,
                "cost_per_success_usd": cost / successes if cost is not None and successes and graded == n and not invalid_budget_attempts else None,
                "mean_elapsed_seconds": sum(r.get("elapsed_seconds", 0) for r in records) / n,
                "status_counts": dict(Counter(r.get("status", "unknown") for r in records)),
                "evaluation_status_counts": dict(Counter(r.get("evaluation", {}).get("status", "unrecorded") for r in records)),
            })
    return rows


def _display(value: Any, places: int = 2) -> str:
    return "—" if value is None else f"{value:,.{places}f}"


def _safe_cell(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def generate_report(input_dir: Path, output_dir: Path) -> dict[str, Path]:
    results = load_results(Path(input_dir))
    rows = aggregate(results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"csv": output_dir / "summary.csv", "markdown": output_dir / "report.md", "html": output_dir / "report.html"}
    columns = ["group_id", "strategy", "provider", "model", "split", "evidence_kind", "attempted", "graded", "pending_or_ungraded", "infrastructure_errors", "official_empty_patches", "successes", "success_rate", "matched_tasks_and_repetition_labels", "invalid_budget_attempts", "comparison_valid", "input_tokens", "output_tokens", "total_tokens", "mean_tokens", "total_cost_usd", "cost_per_success_usd", "mean_elapsed_seconds"]
    with paths["csv"].open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _safe_cell(row[k]) for k in columns})
    notice = ("Scripted smoke runs verify plumbing only; they are not LLM performance evidence. "
              "Unknown cost is not zero. Pending/ungraded tasks are not counted as failures or successes. "
              "Official empty patches have no executed tests; infrastructure failures are ungraded and counted separately. "
              "A success rate is shown only when every attempt has a verifier outcome and valid budget accounting. "
              "Budget overruns and unknown usage invalidate fixed-budget comparisons. "
              "Harnesses, configurations, models, splits and token-accounting modes are grouped separately; "
              "compare strategies only with matched task fingerprints and repetition labels. "
              "API seeds are repetition labels; the Transformers backend resets its RNG per attempt. Neither guarantees cross-hardware reproducibility.")
    md = ["# Budget Agent · Evaluation report", "", notice, ""]
    body = [f'<aside>{html.escape(notice)}</aside>']
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    for group_id, group_rows in by_group.items():
        condition = group_rows[0]["condition"]
        config = condition["config"]
        label = f'{condition.get("evidence_kind")} · {condition.get("model") or condition.get("provider")} · {condition.get("split")} · {group_id}'
        budget_label = f'Context: {config.get("context_tokens", "?")} · total: {config.get("total_tokens", "?")} · tokenizer: {config.get("tokenizer", "?")}'
        md.extend([f"## {label}", "", budget_label, ""])
        body.append(f'<section><div class="eyebrow">EXPERIMENT {html.escape(group_id)}</div><h2>{html.escape(label)}</h2><p class="muted">{html.escape(budget_label)}</p>')
        if not group_rows[0]["matched_tasks_and_repetition_labels"]:
            warning = "Unmatched task/fingerprint/repetition-label sets: these rows are descriptive, not a controlled comparison."
            md.extend([f"<!-- {warning} -->"])
            body.append(f'<p class="warning">{warning}</p>')
        if any(row["invalid_budget_attempts"] for row in group_rows):
            warning = "Invalid budget accounting: at least one attempt exceeded its reservation or has unknown usage. Verifier successes below are not valid fixed-budget wins."
            md.extend(["", f"**{warning}**", ""])
            body.append(f'<p class="warning">{warning}</p>')
        md.extend(["| Strategy | Success / graded / attempted | Infra errors | Empty patches | Rate | Tokens | Total USD | USD / success | Seconds / task |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        body.append('<div class="table-wrap"><table><thead><tr><th>Strategy</th><th>Success / graded / attempted</th><th>Infra errors</th><th>Empty patches</th><th>Rate</th><th>Tokens</th><th>Total USD</th><th>USD / success</th><th>Seconds / task</th></tr></thead><tbody>')
        for row in group_rows:
            rate = "invalid budget" if row["invalid_budget_attempts"] else "ungraded" if row["success_rate"] is None else f'{100 * row["success_rate"]:.1f}%'
            counts = f'{row["successes"]} / {row["graded"]} / {row["attempted"]}'
            cells = [row["strategy"], counts, row["infrastructure_errors"], row["official_empty_patches"], rate, f'{row["total_tokens"]:,}', _display(row["total_cost_usd"], 4), _display(row["cost_per_success_usd"], 4), _display(row["mean_elapsed_seconds"])]
            md.append("| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |")
            body.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>")
        body.append("</tbody></table></div>")
        body.append('<details><summary>Exact comparison conditions</summary><pre>' + html.escape(json.dumps(condition, ensure_ascii=False, indent=2)) + '</pre></details></section>')
        md.extend(["", "```json", json.dumps(condition, ensure_ascii=False, indent=2), "```", ""])
        if not group_rows[0]["matched_tasks_and_repetition_labels"]:
            md.extend(["**Unmatched tasks/fingerprints/repetition labels: do not interpret this group as a controlled comparison.**", ""])
    md.extend(["## Run ledger", "", "| Task | Strategy | Agent status | Evaluation status | Verified outcome |", "|---|---|---|---|---|"])
    body.append('<section><h2>Run ledger</h2><div class="table-wrap"><table><thead><tr><th>Task</th><th>Strategy</th><th>Agent status</th><th>Evaluation status</th><th>Verified outcome</th></tr></thead><tbody>')
    for result in results:
        evaluation = result.get("evaluation", {})
        evaluation_status = evaluation.get("status", "unrecorded")
        resolved = verified_outcome(result)
        outcome = "success" if resolved is True else "failed verification" if resolved is False else "ungraded"
        if is_infrastructure_error(result):
            outcome = "ungraded: infrastructure error"
            if evaluation.get("infra_failure_reason"):
                outcome += f' ({evaluation["infra_failure_reason"]})'
        elif evaluation_status == "official_empty_patch":
            outcome = "ungraded: official empty patch; tests not executed"
        if _invalid_budget(result):
            outcome += " (invalid budget accounting)"
        values = [result["task_id"], result.get("strategy", ""), result.get("status", ""), evaluation_status, outcome]
        md.append("| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in values) + " |")
        body.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>")
    body.append('</tbody></table></div></section>')
    diagnostic_notice = ("Tool success does not establish a correct repair or a passed test. "
                         "Repeated observations have identical action, output and recorded revision; repetition does not necessarily indicate an error. "
                         "unavailable means the result has no stored count. This report does not reanalyze old trajectories.")
    diagnostic_headers = ["Run", "Task", "Strategy", "Successful read", "Successful edit + write", "Successful run", "Successful diff", "Invalid actions", "Failed action repeats", "Identical observation repeats"]
    md.extend(["", "## Run behavior diagnostics", "", diagnostic_notice, "",
               "| " + " | ".join(diagnostic_headers) + " |", "|" + "---|" * len(diagnostic_headers)])
    body.append('<section><h2>Run behavior diagnostics</h2><p class="muted">' + html.escape(diagnostic_notice) + '</p><div class="table-wrap"><table><thead><tr>' + ''.join('<th>' + label + '</th>' for label in diagnostic_headers) + '</tr></thead><tbody>')
    for result in results:
        counts = (result.get("diagnostics") or {}).get("counts", {})
        edit, write = counts.get("successful_edit_file"), counts.get("successful_write_file")
        edits = edit + write if edit is not None and write is not None else None
        metrics = [counts.get("successful_read_file"), edits, counts.get("successful_run"), counts.get("successful_diff"),
                   counts.get("invalid_actions"), counts.get("repeated_failed_actions"), counts.get("repeated_observations")]
        values = [result.get("run_id", Path(result["_result_path"]).parent.name), result["task_id"], result.get("strategy", ""),
                  *("unavailable" if value is None else value for value in metrics)]
        md.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |")
        body.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>")
    body.append('</tbody></table></div></section>')
    if any((result.get("loop_control") or {}).get("policy") == "recover"
           or (result.get("loop_control") or {}).get("events") for result in results):
        loop_notice = ("Stored controller events only; no trajectory replay. A repeated visible read is not proof of faulty reasoning. "
                       "off means not monitored; unavailable means the run predates controller logging.")
        headers = ["Run", "Policy", "Observed repeats", "Warnings sent", "Stopped", "Read evidence"]
        md.extend(["", "## Read-loop control", "", loop_notice, "",
                   "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)])
        body.append('<section><h2>Read-loop control</h2><p class="muted">' + html.escape(loop_notice)
                    + '</p><div class="table-wrap"><table><thead><tr>'
                    + ''.join('<th>' + label + '</th>' for label in headers) + '</tr></thead><tbody>')
        for result in results:
            control = result.get("loop_control")
            run_id = result.get("run_id", Path(result["_result_path"]).parent.name)
            if control is None:
                values = [run_id, *(["unavailable"] * 5)]
            else:
                repeats = [event for event in control["events"] if event["kind"] == "repeated_visible_read"]
                evidence = "; ".join(f'{event["event_id"]} <- {", ".join(event["source_event_ids"])}: '
                                     f'{event["path"]}:{event["start_line"]}-{event["end_line"]}' for event in repeats)
                values = [run_id, control["policy"], len(repeats) if control["policy"] != "off" else "not monitored",
                          control["warnings_sent"], control["stopped"], evidence or "—"]
            cells = [html.escape(str(value)) for value in values]
            md.append("| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in cells) + " |")
            body.append("<tr>" + "".join(f"<td>{value}</td>" for value in cells) + "</tr>")
        body.append('</tbody></table></div></section>')
    if any("safety" in result or "reflection" in result for result in results):
        from .evaluation import evaluate_run
        notice = ("Per-run recorded contracts only: not production certification. Safety covers direct tool authorization, "
                  "not arbitrary behavior inside tests. Unknown evidence cannot pass; reflection text is a hypothesis. "
                  "Scripted runs remain software checks, not model capability evidence.")
        headers = ["Run", "Quality", "Token ledger", "Tool policy", "Acceptance",
                   "Feedback policy", "Memory notes", "Reflection calls", "Blocked tools"]
        md.extend(["", "## Run acceptance contracts", "", notice, "",
                   "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)])
        body.append('<section><h2>Run acceptance contracts</h2><p class="muted">' + html.escape(notice)
                    + '</p><div class="table-wrap"><table><thead><tr>'
                    + ''.join('<th>' + label + '</th>' for label in headers) + '</tr></thead><tbody>')
        for result in results:
            contract = evaluate_run(result)
            values = [result.get("run_id", "unavailable"),
                      *[contract["dimensions"][key]["status"] for key in ("quality", "budget", "safety")],
                      contract["acceptance_status"], result.get("reflection", {}).get("policy", "unavailable"),
                      len(result["reflection"]["notes"]) if "reflection" in result else "unavailable",
                      result.get("reflection", {}).get("calls", "unavailable"),
                      result.get("safety", {}).get("blocked_actions", "unavailable")]
            cells = [html.escape(str(value)) for value in values]
            md.append("| " + " | ".join(value.replace("|", "\\|") for value in cells) + " |")
            body.append("<tr>" + "".join(f"<td>{value}</td>" for value in cells) + "</tr>")
        body.append('</tbody></table></div></section>')
    paths["markdown"].write_text("\n".join(md) + "\n", encoding="utf-8")
    template = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Budget Agent · Evaluation</title><style>
:root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#20302c;background:#f3f6f3}body{margin:0}main{max-width:1200px;margin:0 auto;padding:48px 24px 64px}header{margin-bottom:28px}h1{font-size:clamp(30px,5vw,46px);letter-spacing:-1.5px;margin:8px 0}h2{font-size:19px;overflow-wrap:anywhere}.eyebrow{color:#21735a;letter-spacing:2px;font-size:11px;font-weight:700}.muted{color:#64756e}aside{background:#e5eee7;border-left:4px solid #3a876b;padding:18px 22px;line-height:1.7;border-radius:4px}section{margin-top:24px;background:#fff;border:1px solid #dfe7e0;border-radius:12px;padding:24px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th{color:#66786e;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.4px}th,td{padding:15px 12px;border-bottom:1px solid #edf0ed;white-space:nowrap}td:first-child{font-weight:600}details{margin-top:18px;color:#64756e;font-size:12px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}.warning{background:#fff5db;color:#805600;padding:12px;border-radius:6px}footer{color:#64756e;font-size:12px;margin-top:24px}</style></head><body><main><header><div class="eyebrow">BUDGET AGENT / REPRODUCIBLE EXPERIMENTS</div><h1>Code repair, measured.</h1><p class="muted">Context policies · verified outcomes · complete inference accounting</p></header>__BODY__<footer>Generated locally. No external assets, telemetry, or model calls are used by this report.</footer></main></body></html>"""
    paths["html"].write_text(template.replace("__BODY__", "\n".join(body)), encoding="utf-8")
    return paths
