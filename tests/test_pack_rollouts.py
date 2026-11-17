"""Packing finished rollout pools: nothing is deleted until the archive is read back and matches."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pack_rollouts, pool_behaviour, limit_events = load("pack_rollouts"), load("pool_behaviour"), load("limit_events")
POSIX = sys.platform != "win32"


def make_pool(path, tasks=("t1", "t2")):
    rows = []
    for i, task in enumerate(tasks):
        run = path / "runs" / f"s0{i}" / f"{task}__run"
        (run / "workspace" / "pkg").mkdir(parents=True)
        (run / "trajectory.jsonl").write_text(json.dumps({"step": 1, "action": {"tool": "finish"}}) + "\n")
        (run / "result.json").write_text(json.dumps({"evaluation": {"timed_out": False}}))
        (run / "patch.diff").write_text("-a\n+b\n" * 50)
        (run / "workspace" / "pkg" / "mod.py").write_bytes(os.urandom(3000))
        if POSIX:
            (run / "latest").symlink_to("trajectory.jsonl")
        rows.append({"task_id": task, "sample": 0, "split": "eval", "resolved": i == 0, "status": "finished",
                     "steps": 1, "result_path": str(run / "result.json")})
    (path / "runs" / "s09" / "empty").mkdir(parents=True)
    (path / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "summary.json").write_text("{}")
    return path


def snapshot(root):
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            item = Path(dirpath, name)
            key = item.relative_to(root).as_posix()
            found[key] = ("link", os.readlink(item)) if item.is_symlink() else \
                ("dir",) if item.is_dir() else ("file", item.read_bytes())
    return found


def test_a_pool_is_replaced_by_a_verified_archive_that_restores_it_exactly(tmp_path, capsys):
    pool = make_pool(tmp_path / "eval-grpo")
    before = snapshot(pool / "runs")
    ledger = (pool / "attempts.jsonl").read_text()
    assert pack_rollouts.main([str(pool)]) == 0
    assert not (pool / "runs").exists() and (pool / "attempts.jsonl").read_text() == ledger
    record = json.loads((pool / "PACKED.json").read_text())
    assert (record["files"], record["symlinks"] > 0 or not POSIX) == (8, True)
    assert record["directories"] == 1 + 3 + 2 * 3 + 1          # runs, s00/s01/s09, run+workspace+pkg each, empty
    printed = json.loads(capsys.readouterr().out)
    assert printed["pool"] == str(pool.resolve()) and printed["archive_bytes"] == (pool / "runs.tar.gz").stat().st_size
    with tarfile.open(pool / "runs.tar.gz") as archive:
        archive.extractall(pool, filter="tar")
    assert snapshot(pool / "runs") == before


def test_a_parent_directory_packs_every_pool_below_it_once(tmp_path, capsys):
    make_pool(tmp_path / "round7" / "eval-grpo")
    make_pool(tmp_path / "round7" / "grpo" / "step-0000")
    (tmp_path / "round7" / "notes").mkdir()
    assert pack_rollouts.main([str(tmp_path / "round7"), "--dry-run"]) == 0
    assert capsys.readouterr().out.split() == [str(tmp_path / "round7" / "eval-grpo"),
                                               str(tmp_path / "round7" / "grpo" / "step-0000")]
    assert (tmp_path / "round7" / "eval-grpo" / "runs").exists()          # a dry run changes nothing
    assert pack_rollouts.main([str(tmp_path / "round7"), "--jobs", "2"]) == 0
    assert all((tmp_path / "round7" / p / "PACKED.json").exists() for p in ("eval-grpo", "grpo/step-0000"))
    capsys.readouterr()
    assert pack_rollouts.main([str(tmp_path / "round7")]) == 0
    assert capsys.readouterr().out.strip() == "nothing to pack"


def test_an_archive_that_does_not_match_deletes_nothing(tmp_path, monkeypatch):
    pool = make_pool(tmp_path / "pool")
    before = snapshot(pool)
    real = pack_rollouts.read_archive
    monkeypatch.setattr(pack_rollouts, "read_archive",
                        lambda path: {**real(path), "runs/s00/t1__run/patch.diff": ("file", 1, "0" * 64)})
    with pytest.raises(ValueError, match="archive differs"):
        pack_rollouts.pack(pool)
    assert snapshot(pool) == before                                        # no archive, no partial, runs/ intact


def test_it_refuses_an_unreadable_archive_in_place_and_files_it_cannot_represent(tmp_path):
    pool = make_pool(tmp_path / "pool")
    (pool / "runs.tar.gz").write_bytes(b"old")
    with pytest.raises(tarfile.ReadError):
        pack_rollouts.pack(pool)
    assert (pool / "runs").exists() and not (pool / "PACKED.json").exists()
    if POSIX:
        other = make_pool(tmp_path / "other")
        os.mkfifo(other / "runs" / "s00" / "pipe")
        with pytest.raises(ValueError, match="not a regular file"):
            pack_rollouts.pack(other)
        assert (other / "runs").exists() and not (other / "runs.tar.gz.partial").exists()


def test_analysis_tools_refuse_a_packed_pool_instead_of_counting_nothing(tmp_path):
    pool = make_pool(tmp_path / "pool")
    assert pool_behaviour.describe(pool)["tool_calls"] == {"finish": 2}
    pack_rollouts.pack(pool)
    for describe in (pool_behaviour.describe, limit_events.pool_limits):
        with pytest.raises(SystemExit, match="is packed"):
            describe(pool)


def test_a_run_stopped_while_removing_runs_is_finished(tmp_path, monkeypatch):
    pool = make_pool(tmp_path / "pool")
    real = pack_rollouts.shutil.rmtree

    def stopped(path, *args, **kwargs):
        raise KeyboardInterrupt   # the machine went down after PACKED.json, before runs/ was gone
    monkeypatch.setattr(pack_rollouts.shutil, "rmtree", stopped)
    with pytest.raises(KeyboardInterrupt):
        pack_rollouts.pack(pool)
    assert (pool / "PACKED.json").exists() and (pool / "runs").exists()
    monkeypatch.setattr(pack_rollouts.shutil, "rmtree", real)
    record = pack_rollouts.pack(pool)
    assert record["resumed"] and not (pool / "runs").exists() and record["files"] == 8


def test_a_run_stopped_before_recording_rechecks_the_archive_against_runs(tmp_path, monkeypatch):
    pool = make_pool(tmp_path / "pool")
    before = snapshot(pool / "runs")
    real = pack_rollouts.finish
    monkeypatch.setattr(pack_rollouts, "finish", lambda pool, written: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        pack_rollouts.pack(pool)
    assert (pool / "runs.tar.gz").exists() and not (pool / "PACKED.json").exists()
    monkeypatch.setattr(pack_rollouts, "finish", real)
    other = make_pool(tmp_path / "other")
    (other / "runs.tar.gz").write_bytes((pool / "runs.tar.gz").read_bytes())
    (other / "runs" / "s00" / "t1__run" / "patch.diff").write_text("changed after packing\n")
    with pytest.raises(ValueError, match="already in place"):
        pack_rollouts.pack(other)                                       # an archive that no longer matches
    assert (other / "runs").exists() and not (other / "PACKED.json").exists()
    record = pack_rollouts.pack(pool)
    assert record["resumed"] and not (pool / "runs").exists()
    with tarfile.open(pool / "runs.tar.gz") as archive:
        archive.extractall(pool, filter="tar")
    assert snapshot(pool / "runs") == before


def test_tarfile_does_not_keep_every_header_while_packing(tmp_path, monkeypatch):
    """A TarFile remembers each header it writes or reads; an evaluation pool has half a million."""
    pool = make_pool(tmp_path / "eval-grpo", tasks=tuple(f"t{i}" for i in range(30)))
    sizes = []
    addfile, next_member = tarfile.TarFile.addfile, tarfile.TarFile.next

    def counting_addfile(self, *args, **kwargs):
        sizes.append(len(self.members))
        return addfile(self, *args, **kwargs)

    def counting_next(self):
        sizes.append(len(self.members))
        return next_member(self)

    monkeypatch.setattr(tarfile.TarFile, "addfile", counting_addfile)
    monkeypatch.setattr(tarfile.TarFile, "next", counting_next)
    before = snapshot(pool / "runs")
    assert pack_rollouts.main([str(pool)]) == 0
    assert len(sizes) > 400 and max(sizes) <= 8
    with tarfile.open(pool / "runs.tar.gz") as archive:
        archive.extractall(tmp_path / "restored", filter="tar")
    assert snapshot(tmp_path / "restored" / "runs") == before
