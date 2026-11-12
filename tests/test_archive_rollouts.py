"""Archiving finished rollouts keeps every recorded path valid and never loses the source."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/autodl/archive_rollouts.sh"
pytestmark = pytest.mark.skipif(sys.platform != "linux" or shutil.which("bash") is None,
                                reason="GPU-host shell script (GNU coreutils)")


def archive(tmp_path, *dirs):
    env = {**os.environ, "WORK": str(tmp_path / "work"), "ARCHIVE": str(tmp_path / "archive")}
    return subprocess.run(["bash", str(SCRIPT), *map(str, dirs)], env=env, capture_output=True, text=True,
                          timeout=120)


def make_round(tmp_path):
    run = tmp_path / "work" / "round9" / "eval" / "runs" / "s00" / "task__r1"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({"resolved": True}))
    attempts = tmp_path / "work" / "round9" / "eval" / "attempts.jsonl"
    attempts.write_text(json.dumps({"result_path": str(run / "result.json")}) + "\n")
    return tmp_path / "work" / "round9", attempts


def test_a_round_moves_to_the_archive_and_its_recorded_paths_still_resolve(tmp_path):
    round_dir, attempts = make_round(tmp_path)
    result = archive(tmp_path, round_dir)
    assert result.returncode == 0, result.stderr
    assert round_dir.is_symlink() and (tmp_path / "archive" / "round9").is_dir()
    recorded = Path(json.loads(attempts.read_text())["result_path"])
    assert json.loads(recorded.read_text()) == {"resolved": True}
    again = archive(tmp_path, round_dir)
    assert again.returncode == 0 and "already archived" in again.stdout


def test_nested_directories_get_a_flat_archive_name(tmp_path):
    round_dir, _ = make_round(tmp_path)
    assert archive(tmp_path, round_dir / "eval").returncode == 0
    assert (tmp_path / "archive" / "round9-eval").is_dir() and (round_dir / "eval").is_symlink()


def test_paths_outside_the_work_tree_and_existing_targets_are_refused(tmp_path):
    round_dir, _ = make_round(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    assert archive(tmp_path, outside).returncode != 0 and outside.is_dir() and not outside.is_symlink()
    (tmp_path / "archive" / "round9").mkdir(parents=True)
    refused = archive(tmp_path, round_dir)
    assert refused.returncode != 0 and round_dir.is_dir() and not round_dir.is_symlink()


def test_a_copy_that_differs_is_removed_and_the_source_kept(tmp_path):
    round_dir, _ = make_round(tmp_path)
    shim = tmp_path / "bin"
    shim.mkdir()
    # A cp that loses a file: the check must catch it, delete the partial copy and keep the source.
    (shim / "cp").write_text('#!/bin/sh\n/bin/cp "$@" && find "$3" -name result.json -delete\n')
    (shim / "cp").chmod(0o755)
    env = {**os.environ, "WORK": str(tmp_path / "work"), "ARCHIVE": str(tmp_path / "archive"),
           "PATH": f"{shim}:{os.environ['PATH']}"}
    result = subprocess.run(["bash", str(SCRIPT), str(round_dir)], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0 and "copy removed, source kept" in result.stderr
    assert round_dir.is_dir() and not round_dir.is_symlink() and not (tmp_path / "archive" / "round9").exists()
