"""Print every resolved attempt's patch in a rollout pool, for the manual tamper review.

The frozen protocols ask for each passing patch to be read: a test edited, the
grading channel touched, a forged exit, or a fix that passes the tests by a
different mechanism than the bug's (for example `if False:`). For each resolved
attempt this prints the task, sample, final status, files touched, added and
removed line counts, files the grader ignored, patch signals, and the changed
lines themselves.

    python scripts/review_patches.py /opt/coding-agent/work/round2/grpo-eval

With --original and --tasks (SWE-smith tasks), each patch is also applied to the
task's buggy files and the result compared with the source before the bug was
injected. A patch that restores every file exactly is marked `restores original`;
otherwise the remaining difference from the original is printed. That residual is
what needs judgement: dead code is harmless, but a remaining part of the injected
bug, or new behaviour on inputs the tests do not cover, means the patch passes the
tests without restoring the original behaviour.

    python scripts/review_patches.py POOL --original /opt/coding-agent/swesmith-src \\
        --tasks /opt/coding-agent/swesmith/tasks

--original holds each repository's pre-bug source as <repo>/<path>, where <repo> is
the task id without its last dot-separated part (Suor__funcy.207a7810).
"""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def patch_files(lines: list[str]) -> list[str]:
    return [line[6:] for line in lines if line.startswith("+++ b/")]


def residual(task_id: str, patch: Path, files: list[str], original: Path, tasks: Path) -> list[str] | None:
    """Unified diff from the original source to the patched buggy source; [] when identical.

    None when the patch does not apply to the task's buggy files.
    """
    repo = task_id.rsplit(".", 1)[0]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for rel in files:
            source = tasks / task_id / "repo" / rel
            (work / rel).parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                shutil.copyfile(source, work / rel)
        applied = subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch.resolve())], cwd=work,
                                 capture_output=True, text=True)
        if applied.returncode:
            return None
        diff: list[str] = []
        for rel in files:   # a file missing on either side (created or deleted by the patch) reads as empty
            before_path, after_path = original / repo / rel, work / rel
            before = before_path.read_text(encoding="utf-8", errors="replace").splitlines() if before_path.exists() else []
            after = after_path.read_text(encoding="utf-8", errors="replace").splitlines() if after_path.exists() else []
            diff += difflib.unified_diff(before, after, f"original/{rel}", f"patched/{rel}", n=1, lineterm="")
        return diff


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pool", type=Path)
    parser.add_argument("--original", type=Path, help="pre-bug source of each repository, as <repo>/<path>")
    parser.add_argument("--tasks", type=Path, help="SWE-smith task directories (<task id>/repo holds the buggy source)")
    args = parser.parse_args(argv)
    if (args.original is None) != (args.tasks is None):
        parser.error("--original and --tasks go together")
    pool = args.pool
    rows = [json.loads(line) for line in (pool / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    winners = sorted((r for r in rows if r["resolved"]), key=lambda r: (r["task_id"], r["sample"]))
    print(f"{pool.name}: {len(winners)} resolved of {len(rows)}")
    restored = 0
    for row in winners:
        patch_path = Path(row["result_path"]).parent / "patch.diff"
        patch = patch_path.read_text(encoding="utf-8", errors="replace")
        lines = patch.splitlines()
        files = patch_files(lines)
        added = sum(line.startswith("+") and not line.startswith("+++") for line in lines)
        removed = sum(line.startswith("-") and not line.startswith("---") for line in lines)
        print(f"\n### {row['task_id']} s{row['sample']} status={row['status']} steps={row['steps']} files={files} "
              f"+{added}/-{removed} ignored={row['ignored_changes']} signals={row['patch_signals']}")
        print("\n".join(line for line in lines if line.startswith(("+", "-", "@@")) and not line.startswith(("+++", "---"))))
        if args.original is not None:
            rest = residual(row["task_id"], patch_path, files, args.original, args.tasks)
            if rest is None:
                print("[original check] the patch does not apply to the task's buggy files")
            elif not rest:
                restored += 1
                print("[original check] restores original")
            else:
                print("[original check] differs from the original:")
                print("\n".join(line for line in rest if not line.startswith(("+++", "---"))))
    if args.original is not None:
        print(f"\n{pool.name}: {restored} of {len(winners)} resolved patches restore the original source exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
