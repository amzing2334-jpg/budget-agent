"""Re-grade passing patches with hidden behaviour checks (no model calls).

The SWE-smith verifier runs the repository's own tests. They cover the injected bug's
main symptom but not always the whole original behaviour (docs/TECH_REPORT.md §3). For
the tasks where passing patches left part of the bug in place,
tasks/hidden_checks/<task id>.py pins the original behaviour with plain assertions.

Each check first runs on the pre-bug source, where it must pass, and on the buggy source.
Then every resolved attempt of a checked task is re-graded: its patch is applied to the
task's buggy files and the check runs with the task's own interpreter and environment
under a timeout. With --as-nobody the checks (which execute model-written code) run as
the unprivileged user, as rollouts do.

    python scripts/regrade_hidden.py --manifest /opt/coding-agent/swesmith/eval.jsonl \\
        --original /opt/coding-agent/swesmith-src --as-nobody POOL [POOL ...] > regrade.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

CHECKS = Path(__file__).resolve().parents[1] / "tasks" / "hidden_checks"
NOBODY = ["setpriv", "--reuid=65534", "--regid=65534", "--clear-groups"]


def load_manifest(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["id"]: {**row, "repo": str(path.parent / row["repo"])} for row in rows}


def run_check(task: dict, workspace: Path, check: Path, *, as_nobody: bool, timeout: int) -> str:
    """'pass', 'fail' or 'timeout' for one check on one source tree."""
    env_spec = task.get("metadata", {}).get("exec_env", {})
    python = str(Path(env_spec["PATH"]) / "python") if "PATH" in env_spec else sys.executable
    env = {"PATH": f"{Path(python).parent}:/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
           "HOME": str(workspace), "PYTHONPATH": f"{workspace}/src:{workspace}"}
    command = [python, str(check)]
    if as_nobody:
        command = [*NOBODY, "env", "-i", *(f"{k}={v}" for k, v in env.items()), *command]
    try:
        done = subprocess.run(command, cwd=workspace, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout"
    return "pass" if done.returncode == 0 else "fail"


def prepared(source: Path, patch: Path | None, root: Path) -> Path | None:
    """A copy of `source` with `patch` applied, readable by every user; None if the patch fails."""
    workspace = Path(tempfile.mkdtemp(dir=root))
    shutil.copytree(source, workspace, dirs_exist_ok=True, symlinks=True)
    if patch is not None:
        applied = subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch)], cwd=workspace,
                                 capture_output=True, text=True)
        if applied.returncode:
            return None
    for path in [workspace, *workspace.rglob("*")]:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o555 if path.is_dir() else path.stat().st_mode | 0o444)
    return workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pools", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path, required=True, help="task manifest (repo paths and exec_env)")
    parser.add_argument("--original", type=Path, required=True, help="pre-bug source of each repository, as <repo>/")
    parser.add_argument("--checks", type=Path, default=CHECKS)
    parser.add_argument("--as-nobody", action="store_true", help="run the checks as uid 65534 (GPU host)")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)
    tasks = load_manifest(args.manifest)
    checks = {path.stem: path for path in sorted(args.checks.glob("*.py")) if path.stem in tasks}
    report: dict[str, Any] = {"checks": {}, "attempts": [], "summary": {}}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        root.chmod(0o755)
        usable = {}
        for task_id, check in checks.items():
            task = tasks[task_id]
            original = prepared(args.original / task_id.rsplit(".", 1)[0], None, root)
            buggy = prepared(Path(task["repo"]), None, root)
            assert original is not None and buggy is not None
            outcome = {name: run_check(task, tree, check, as_nobody=args.as_nobody, timeout=args.timeout)
                       for name, tree in (("original", original), ("buggy", buggy))}
            report["checks"][task_id] = outcome
            if outcome["original"] == "pass":
                usable[task_id] = check
        for pool in args.pools:
            counts = {"checked": 0, "pass": 0, "fail": 0, "timeout": 0, "patch_failed": 0}
            rows = [json.loads(line) for line in (pool / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            for row in sorted(rows, key=lambda r: (r["task_id"], r["sample"])):
                if not row["resolved"] or row["task_id"] not in usable:
                    continue
                task = tasks[row["task_id"]]
                patch = (Path(row["result_path"]).parent / "patch.diff").resolve()
                workspace = prepared(Path(task["repo"]), patch, root)
                result = ("patch_failed" if workspace is None else
                          run_check(task, workspace, usable[row["task_id"]], as_nobody=args.as_nobody,
                                    timeout=args.timeout))
                counts["checked"] += 1
                counts[result] += 1
                report["attempts"].append({"pool": str(pool), "task_id": row["task_id"], "sample": row["sample"],
                                           "result": result})
            report["summary"][str(pool)] = counts
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
