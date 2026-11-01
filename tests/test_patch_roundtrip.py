"""Saved fixture patches must reproduce the independently graded workspace."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from budget_agent.budget import TokenCounter
from budget_agent.models import ScriptedModel
from budget_agent.runner import run_task
from budget_agent.schema import RunConfig, Task


@pytest.mark.parametrize(("before", "after"), [
    pytest.param(b"old", b"new", id="unterminated-replacement"),
    pytest.param(b"line\n", b"line", id="remove-final-newline"),
    pytest.param(b"line", b"line\n", id="add-final-newline"),
    pytest.param(b"old\nunchanged", b"new\nunchanged", id="unterminated-context"),
    pytest.param(b"old\r\nunchanged\r\n", b"new\r\nunchanged\r\n", id="crlf"),
    pytest.param(b"old\nunchanged\n", b"new\nunchanged\n", id="lf"),
    pytest.param("old\u2028data\rkept\n".encode(), "new\u2028data\rkept\n".encode(),
                 id="non-lf-data"),
    pytest.param(None, b"new\n", id="new-text-file"),
    pytest.param(b"old", None, id="delete-unterminated-file"),
])
def test_saved_patch_replays_exact_repair_bytes(tmp_path, before, after):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.txt").write_bytes(b"Fixture baseline\n")
    if before is not None:
        (source / "payload.txt").write_bytes(before)
    verifier = tmp_path / "verifier.py"
    verifier.write_text(
        "from pathlib import Path\nimport unittest\n"
        "class RepairTest(unittest.TestCase):\n"
        " def test_repair(self):\n"
        "  path = Path('payload.txt')\n"
        f"  self.assertEqual(path.read_bytes() if path.exists() else None, {after!r})\n",
        encoding="utf-8",
    )
    task = Task(id="patch-roundtrip", issue="Repair payload.txt", split="dev",
                repo=str(source), verifier=str(verifier))
    operation = "unlink()" if after is None else f"write_bytes(bytes.fromhex('{after.hex()}'))"
    command = f'"{sys.executable}" -c "from pathlib import Path; Path(\'payload.txt\').{operation}"'
    counter = TokenCounter()
    model = ScriptedModel([{"tool": "run", "command": command}, {"tool": "finish"}], counter)

    result = run_task(task, RunConfig(), tmp_path / "runs", model=model, counter=counter)

    assert result["resolved"] is True
    patch = Path(result["patch_path"])
    workspace = patch.parent / "workspace" / "payload.txt"
    assert (workspace.read_bytes() if workspace.exists() else None) == after
    baseline = tmp_path / "replay"
    shutil.copytree(source, baseline)
    git = ["git", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false"]
    subprocess.run([*git, "init", "-q"], cwd=baseline, check=True, capture_output=True)
    for options in (["--check"], []):
        applied = subprocess.run([*git, "apply", *options, "--"], input=patch.read_bytes(),
                                 cwd=baseline, capture_output=True)
        assert applied.returncode == 0, applied.stderr.decode("utf-8", errors="replace")
    restored = baseline / "payload.txt"
    assert (restored.read_bytes() if restored.exists() else None) == after
    assert (baseline / "README.txt").read_bytes() == b"Fixture baseline\n"
    original = source / "payload.txt"
    assert (original.read_bytes() if original.exists() else None) == before
