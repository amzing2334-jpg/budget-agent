"""Read returned cloud artifacts; emit descriptive JSON/Markdown without changing runs.

Usage: .venv/Scripts/python.exe runs/v5-followup-20261022-01/analyze_cloud.py
Missing fields remain null. This is a development-run inventory, not a benchmark
score or a causal comparison between the internal and external harnesses.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


ROOT = Path(__file__).resolve().parents[2]
PLANNED = {
    ("reserve-probe", "reserve-1"): "reserve-1",
    ("reserve-probe", "reserve-128"): "reserve-128",
    ("mini-baseline",): "mini-baseline",
}
DIAGNOSTIC_COUNTS = (
    "events", "invalid_actions", "invalid_actions_at_output_limit",
    "repeated_failed_actions", "repeated_observations", "successful_edit_file",
    "successful_write_file", "successful_run", "successful_diff",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def select(record: dict, names) -> dict:
    return {name: record.get(name) for name in names}


def evidence(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def arm_for(relative: Path) -> tuple[str, str]:
    # Primary layout is exactly ARM/RUN_ID/result.json. Deeper retry directories
    # and other branches are additional attempts, never folded into the pair.
    for prefix, arm in PLANNED.items():
        if relative.parts[:len(prefix)] == prefix and len(relative.parts) == len(prefix) + 2:
            return arm, "planned_layout"
    return relative.parts[0], "additional_not_preregistered"


def ledger_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    responses = [row["response"] for row in rows if "response" in row]
    return {
        "rows": len(rows), "response_rows": len(responses),
        "unknown_usage_rows": sum(row.get("status") == "usage_unknown" for row in rows),
        "known_input_tokens": sum(row["input_tokens"] for row in responses),
        "known_output_tokens": sum(row["output_tokens"] for row in responses),
        "usage_sources": sorted({row["usage_source"] for row in responses}),
    }


def mini_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    trajectory = read_json(path)
    messages, info = trajectory["messages"], trajectory["info"]
    observations = [row["extra"] for row in messages if "returncode" in row.get("extra", {})]
    return {
        "trajectory_format": trajectory.get("trajectory_format"),
        "mini_version": info.get("mini_version"), "exit_status": info.get("exit_status"),
        "native_model_stats": info.get("model_stats"),
        "roles": dict(Counter(row["role"] for row in messages)),
        "parsed_actions": sum(len(row.get("extra", {}).get("actions", [])) for row in messages if row["role"] == "assistant"),
        "format_error_messages": sum(row.get("extra", {}).get("interrupt_type") == "FormatError" for row in messages),
        "returncode_observations": len(observations),
        "nonzero_returncode_observations": sum(row["returncode"] != 0 for row in observations),
        "metric_scope": "upstream messages and shell actions; not equivalent to internal JSON tool counts",
        "native_call_caveat": "Upstream api_calls increments before the adapter budget gate; use budget/calls ledger for metering.",
    }


def official_counts(evaluation: dict, task_id: str, grading_root: Path) -> dict | None:
    """Match the recorded run/model/task suffix, then verify its patch digest.

    Only counts leave the evaluator directory. FAILURE can include unobserved
    expected tests, so nonzero classified counts alone do not prove execution.
    """
    recorded = evaluation.get("report_path")
    if not recorded or not grading_root.is_dir():
        return None
    parts = PurePosixPath(recorded.replace("\\", "/")).parts
    marker = next((index for index in range(len(parts) - 1) if parts[index:index + 2] == ("logs", "run_evaluation")), None)
    if marker is None:
        raise ValueError(f"Recorded evaluator path has no exact run-id suffix: {recorded}")
    suffix = parts[marker:]
    matches = [path for path in grading_root.rglob("report.json") if path.parts[-len(suffix):] == suffix]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Ambiguous evaluator copy for {recorded}")
    path, = matches
    grade = read_json(path)[task_id]
    digest = hashlib.sha256((path.parent / "patch.diff").read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    if digest != evaluation.get("patch_sha256"):
        raise ValueError(f"Evaluator patch hash does not match attached evaluation: {path}")
    counts = {group: {state: len(names) for state, names in values.items()} for group, values in grade.get("tests_status", {}).items()}
    observed_success = sum(values.get("success", 0) for values in counts.values())
    return {"report": evidence(path), "patch_sha256": digest, "test_counts": counts,
            "tests_executed": True if observed_success > 0 and grade.get("patch_successfully_applied") is True else None,
            "execution_basis": "At least one classified success after patch application" if observed_success > 0 and grade.get("patch_successfully_applied") is True else "No independent execution assertion; failure lists can include missing expected tests"}


def summarize(path: Path, source: Path, grading_root: Path) -> dict:
    result = read_json(path)
    config, budget, evaluation = result.get("config", {}), result.get("budget", {}), result.get("evaluation", {})
    directory = path.parent
    arm, registration = arm_for(path.relative_to(source))
    is_mini = result.get("harness") == "mini-swe-agent-textbased-budget-adapter"
    diagnostic = result.get("diagnostics")
    diagnostic_source = "result.diagnostics" if diagnostic is not None else None
    if not is_mini and diagnostic is None and (directory / "trajectory.jsonl").is_file():
        sys.path.insert(0, str(ROOT / "src"))
        from budget_agent.diagnostics import diagnose_run
        diagnostic = diagnose_run(directory)
        diagnostic_source = "recomputed from local trajectory/calls with existing diagnose_run"
    internal = None
    if not is_mini and diagnostic is not None:
        internal = {
            "source": diagnostic_source,
            "counts": select(diagnostic["counts"], DIAGNOSTIC_COUNTS),
            "tools": {name: diagnostic.get("tools", {}).get(name) for name in ("edit_file", "write_file", "run", "diff")},
            "call_alignment": diagnostic.get("call_alignment"),
        }
    paths = {name: evidence(directory / name) for name in ("result.json", "patch.diff", "calls.jsonl", "trajectory.jsonl", "mini.traj.json")}
    official = official_counts(evaluation, result["task_id"], grading_root)
    grade_summary = select(evaluation, ("status", "resolved", "tests_executed", "test_counts", "official_resolved", "infra_failure", "infra_failure_reason", "patch_successfully_applied", "patch_sha256", "report_path", "summary_path"))
    if official is not None:
        grade_summary["test_counts"] = official["test_counts"]
        if grade_summary["tests_executed"] is None:
            grade_summary["tests_executed"] = official["tests_executed"]
    return {
        "arm": arm, "registration": registration,
        "registration_scope": "directory matches predeclared driver; not independent verification of creation time",
        **select(result, ("task_id", "split", "run_id", "provider", "model", "actual_models", "strategy", "status", "resolved", "evidence_kind", "source_fingerprint", "task_fingerprint", "initial_revision", "created_at", "model_metadata")),
        "model_revision": config.get("model_revision"),
        "harness": result.get("harness", config.get("harness")),
        "harness_group": "external_mini_swe" if is_mini else "internal_budget_agent",
        "harness_note": None if is_mini else "Internal result schema does not currently record a harness field.",
        "upstream": select(result, ("upstream_sha", "upstream_version", "upstream_config_sha256")) if is_mini else None,
        "configuration": select(config, ("context_tokens", "total_tokens", "max_output_tokens", "min_action_output_tokens", "max_steps", "history_format", "strategy", "seed", "temperature", "model_dtype", "context_policy", "submission_policy")),
        "usage": select(result, ("input_tokens", "output_tokens", "total_tokens", "calls", "usage_unknown", "usage_sources", "elapsed_seconds", "model_seconds")),
        "budget": select(budget, ("input_tokens", "output_tokens", "total_tokens", "calls", "attempted_calls", "remaining_tokens", "overrun", "unreconciled_request", "usage_is_estimated", "preflight_counter")),
        "evaluation": grade_summary, "independent_evaluator_counts": official,
        "patch_bytes": paths["patch.diff"]["bytes"] if paths["patch.diff"] else None,
        "internal_diagnostics": internal,
        "mini_native_metrics": mini_summary(directory / "mini.traj.json") if is_mini else None,
        "call_ledger": ledger_summary(directory / "calls.jsonl"),
        "evidence": paths,
    }


def cell(value) -> str:
    if value is None:
        return "未知"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown(payload: dict) -> str:
    lines = ["# 本轮云端产物对照", "", f"读取目录：`{payload['input']}`", "",
             "这是已返回产物的描述性清单。缺失字段保留为未知；单任务开发尝试不构成基准成功率。", "",
             "| 组 | 注册标记 | Task | 模型 / revision | Harness |", "| --- | --- | --- | --- | --- |"]
    for row in payload["runs"]:
        lines.append("| " + " | ".join(map(cell, (row["arm"], row["registration"], row["task_id"], f"{row['model']} / {row['model_revision']}", row["harness"] or row["harness_group"]))) + " |")
    lines += ["", "| 组 | 输入 / 输出 / 合计 tokens | calls | 运行状态 | 评分状态 | eval.resolved | 测试执行 | Patch bytes |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in payload["runs"]:
        usage, grade = row["usage"], row["evaluation"]
        tokens = " / ".join(cell(usage[key]) for key in ("input_tokens", "output_tokens", "total_tokens"))
        lines.append("| " + " | ".join(map(cell, (row["arm"], tokens, usage["calls"], row["status"], grade["status"], grade["resolved"], grade["tests_executed"], row["patch_bytes"]))) + " |")
    lines += ["", "内部诊断仅统计记录到的 JSON 工具动作；重复失败是在相同 revision 下重复同一失败动作。", "",
              "| 组 | Invalid | 到输出上限的 invalid | 重复失败 | edit / write 成功 | run 成功 | diff 成功 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in payload["runs"]:
        if row["internal_diagnostics"] is None:
            continue
        counts = row["internal_diagnostics"]["counts"]
        lines.append("| " + " | ".join(map(cell, (row["arm"], counts["invalid_actions"], counts["invalid_actions_at_output_limit"], counts["repeated_failed_actions"], f"{cell(counts['successful_edit_file'])} / {cell(counts['successful_write_file'])}", counts["successful_run"], counts["successful_diff"]))) + " |")
    lines += ["", "mini-SWE 单独读取官方 trajectory 与调用账本。它的 Bash 命令数不能与内部 edit/run/diff 次数直接比较；原生 api_calls 也不能替代预算账本。", ""]
    for row in payload["runs"]:
        native = row["mini_native_metrics"]
        if native is not None:
            lines.append(f"- {row['arm']}：原生退出 {cell(native['exit_status'])}；解析动作 {native['parsed_actions']}；格式错误消息 {native['format_error_messages']}；有返回码观察 {native['returncode_observations']}；非零返回码 {native['nonzero_returncode_observations']}。")
    lines += ["", "mini-SWE 的补丁来自显式 submission；内部 agent 导出工作区 diff。空补丁与未执行测试不等于已执行测试失败。additional_not_preregistered 表示额外/适应性尝试，不能并入原配对；同一预设组存在多份结果时全部标为 ambiguous_multiple_attempts。", "", "证据：", ""]
    for row in payload["runs"]:
        links = [f"[{name}](<{record['path']}>)" for name, record in row["evidence"].items() if record]
        lines.append(f"- {row['arm']} / {row['run_id']}：" + " · ".join(links))
        if row["independent_evaluator_counts"] is not None:
            official = row["independent_evaluator_counts"]
            lines.append(f"  独立评分计数：`{json.dumps(official['test_counts'], ensure_ascii=False)}`；[官方报告](<{official['report']['path']}>)。")
    if payload["missing_planned_arms"]:
        lines += ["", "尚无匹配产物的预设组：" + ", ".join(payload["missing_planned_arms"]) + "。"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "runs/cloud-v5-a4000-20261022-01")
    parser.add_argument("--grading-root", type=Path, default=ROOT.parent / "budget-agent-evaluation/cloud-v5-a4000-20261022-01")
    parser.add_argument("--output-prefix", type=Path, default=Path(__file__).with_name("cloud-comparison"))
    args = parser.parse_args(argv)
    source = args.input.resolve()
    if not source.is_dir():
        parser.error(f"Cloud results have not been returned: {source}")
    paths = sorted(source.rglob("result.json"))
    if not paths:
        parser.error(f"No result.json artifacts found under {source}; no report written")
    grading_root = args.grading_root.resolve()
    runs = [summarize(path, source, grading_root) for path in paths]
    counts = Counter(row["arm"] for row in runs if row["registration"] == "planned_layout")
    for row in runs:
        if row["registration"] == "planned_layout" and counts[row["arm"]] > 1:
            row["registration"] = "ambiguous_multiple_attempts"
    payload = {
        "kind": "descriptive_cloud_artifact_inventory", "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(source), "grading_root": str(grading_root), "analysis_source": evidence(Path(__file__)),
        "missing_planned_arms": [arm for arm in PLANNED.values() if arm not in counts],
        "runs": runs,
        "notes": ["No effectiveness conclusion or benchmark score is calculated.",
                  "Unknown tests_executed remains null; resolved does not independently establish test execution. Matched official success classifications can establish at least one observed test.",
                  "External harness and internal reserve pair are distinct comparisons.",
                  "Source hashes identify these local bytes; they do not independently validate upstream grading."],
    }
    prefix = args.output_prefix.resolve()
    targets = [prefix.with_suffix(".json"), prefix.with_suffix(".md")]
    if any(target.is_relative_to(source) for target in targets):
        parser.error("Write derived output outside the input run directory")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    targets[1].write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"runs": len(runs), "json": str(targets[0]), "markdown": str(targets[1])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
