"""Load explicit task manifests; never pass private grading metadata to a model."""
from __future__ import annotations

import json
import re
from dataclasses import fields
from pathlib import Path

from .schema import Task


def load_tasks(path: Path, split: str = "dev", task_ids: list[str] | None = None) -> list[Task]:
    path = path.resolve()
    text = path.read_text(encoding="utf-8-sig")
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    result, seen = [], set()
    valid_fields = {f.name for f in fields(Task)}
    for row in rows:
        if not isinstance(row, dict) or set(row) - valid_fields:
            raise ValueError("Task manifest has unknown fields; use import-swebench for raw benchmark rows")
        task = Task(**row)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,160}", task.id) or task.id in seen:
            raise ValueError(f"Unsafe or duplicate task ID: {task.id!r}")
        seen.add(task.id)
        if task.split not in {"train", "eval", "dev", "test"} or task.kind not in {"local", "swebench"}:
            raise ValueError(f"Invalid split/kind for {task.id}")
        if not task.issue.strip():
            raise ValueError("Task issue must not be empty")
        for name in ("repo", "verifier"):
            if getattr(task, name):
                setattr(task, name, str((path.parent / getattr(task, name)).resolve()))
        if split != "all" and task.split != split:
            continue
        if task_ids and task.id not in task_ids:
            continue
        result.append(task)
    if task_ids and set(task_ids) - {t.id for t in result}:
        raise ValueError("Requested task is absent or excluded by --split")
    if not result:
        raise ValueError("No tasks selected")
    return result


def smoke_actions(task: Task) -> list[dict]:
    """Predetermined fixture repairs. These are not a solver or evaluation baseline."""
    fixes = {
        "slugify-digits": ("slugify.py", "[^a-z]+", "[^a-z0-9]+"),
        "chunks-positive-size": ("chunks.py", "    return [", "    if size <= 0:\n        raise ValueError('size must be positive')\n    return ["),
        "merge-touching-intervals": ("intervals.py", "start < merged[-1][1]", "start <= merged[-1][1]"),
    }
    if task.id not in fixes or task.kind != "local":
        raise ValueError("The scripted provider only supports the three bundled smoke fixtures")
    file, old, new = fixes[task.id]
    original = (Path(task.repo) / file).read_text(encoding="utf-8")
    if old not in original:
        raise ValueError("Fixture source changed; scripted repair no longer matches")
    return [
        {"tool": "list_files"},
        {"tool": "read_file", "path": file},
        {"tool": "run", "command": "python -m unittest discover -s tests -v"},
        {"tool": "remember", "note": "The public tests do not necessarily cover the reported edge case. Verify the issue requirements after editing.", "source": "e0002"},
        {"tool": "write_file", "path": file, "content": original.replace(old, new)},
        {"tool": "run", "command": "python -m unittest discover -s tests -v"},
        {"tool": "diff"},
        {"tool": "finish"},
    ]
