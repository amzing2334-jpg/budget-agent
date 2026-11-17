"""Pack finished rollout pools into one compressed archive each (no model calls).

    python scripts/pack_rollouts.py POOL_OR_PARENT [...] [--jobs N] [--dry-run]

A pool is a directory with attempts.jsonl and a runs/ tree: one directory per attempt with
its trajectory, requests, patch and a copy of the workspace. A finished pool is hundreds of
thousands of small files, and on the data disk each costs at least one block. For each pool
this writes runs/ into runs.tar.gz, hashing every file as it is read; reads the archive back
and checks that it holds exactly the same directories, symlinks and regular files with the
same SHA-256; and only then deletes runs/ and writes PACKED.json. The rest of the pool
(attempts.jsonl, summary.json) stays in place, so the ledgers that comparisons read remain
usable. A run stopped at any point can be repeated: a pool whose verified archive was already
moved into place is finished (its archive is checked against runs/ again unless PACKED.json
says it was). Restore a pool with

    tar -xzf POOL/runs.tar.gz -C POOL

Given a parent directory, it packs every pool below it. Never run it on a pool that a running
experiment still writes to or reads trajectories from. Memory grows with a pool's size only
through the per-file records compared above: tarfile's own list of every header it wrote or read
is dropped as it goes (on a synthetic pool of 98,000 files in 28,009 directories, peak memory fell
from 192 MiB to 95 MiB), which matters in a 2 GiB container packing pools of half a million files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

ARCHIVE = "runs.tar.gz"
MARKER = "PACKED.json"
Entry = tuple  # ("dir",) | ("link", target) | ("file", size, sha256)


class HashingReader:
    """File object that hashes what tarfile reads from it."""

    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def find_pools(root: Path) -> list[Path]:
    """Pools at or below root; nothing below a pool's own directory is searched."""
    pools = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "attempts.jsonl" in filenames and "runs" in dirnames and not Path(dirpath, "runs").is_symlink():
            pools.append(Path(dirpath))
            dirnames.clear()
        else:
            dirnames.sort()
    return pools


def write_archive(pool: Path, target: Path, compresslevel: int) -> dict[str, Entry]:
    """Write pool/runs into target; return what was written, keyed by archive name."""
    written: dict[str, Entry] = {}
    with tarfile.open(target, "w:gz", compresslevel=compresslevel) as archive:
        for dirpath, dirnames, filenames in os.walk(pool / "runs"):
            dirnames.sort()
            base = Path(dirpath)
            name = base.relative_to(pool).as_posix()
            archive.addfile(archive.gettarinfo(str(base), name))
            written[name] = ("dir",)
            for child in [base / d for d in dirnames if (base / d).is_symlink()] + [base / f for f in sorted(filenames)]:
                arcname = child.relative_to(pool).as_posix()
                info = archive.gettarinfo(str(child), arcname)
                if info.issym():
                    archive.addfile(info)
                    written[arcname] = ("link", info.linkname)
                elif info.isfile():
                    with child.open("rb") as stream:
                        reader = HashingReader(stream)
                        archive.addfile(info, reader)
                    written[arcname] = ("file", info.size, reader.digest.hexdigest())
                else:
                    raise ValueError(f"{child}: not a regular file, directory or symlink")
            archive.members.clear()   # a TarFile keeps every header it wrote; `written` already records them
    return written


def scan_tree(pool: Path) -> dict[str, Entry]:
    """What write_archive records for pool/runs, without writing anything."""
    entries: dict[str, Entry] = {}
    for dirpath, dirnames, filenames in os.walk(pool / "runs"):
        dirnames.sort()
        base = Path(dirpath)
        entries[base.relative_to(pool).as_posix()] = ("dir",)
        for child in [base / d for d in dirnames if (base / d).is_symlink()] + [base / f for f in sorted(filenames)]:
            name = child.relative_to(pool).as_posix()
            if child.is_symlink():
                entries[name] = ("link", os.readlink(child))
            elif child.is_file():
                digest = hashlib.sha256()
                with child.open("rb") as stream:
                    for block in iter(lambda: stream.read(1 << 20), b""):  # noqa: B023 (consumed in this iteration)
                        digest.update(block)
                entries[name] = ("file", child.stat().st_size, digest.hexdigest())
            else:
                raise ValueError(f"{child}: not a regular file, directory or symlink")
    return entries


def differences(pool: Path, expected: dict[str, Entry], found: dict[str, Entry]) -> str | None:
    if found == expected:
        return None
    missing, extra = sorted(expected.keys() - found.keys()), sorted(found.keys() - expected.keys())
    changed = sorted(k for k in expected.keys() & found.keys() if expected[k] != found[k])
    return f"{pool}: archive differs from runs/ (missing {missing[:3]}, extra {extra[:3]}, changed {changed[:3]})"


def read_archive(path: Path) -> dict[str, Entry]:
    found: dict[str, Entry] = {}
    with tarfile.open(path, "r|gz") as archive:
        while (member := archive.next()) is not None:
            if member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1 << 20), b""):  # noqa: B023 (consumed in this iteration)
                    digest.update(chunk)
                found[member.name] = ("file", member.size, digest.hexdigest())
            elif member.issym():
                found[member.name] = ("link", member.linkname)
            elif member.isdir():
                found[member.name] = ("dir",)
            else:
                found[member.name] = ("other", member.type)
            archive.members.clear()   # likewise for every header it read
    return found


def pack(pool: Path, compresslevel: int = 6, min_free_gb: float = 2.0) -> dict:
    """Pack one pool; return what PACKED.json records. Raises without deleting anything on any doubt."""
    target, partial = pool / ARCHIVE, pool / (ARCHIVE + ".partial")
    if target.exists():
        return resume(pool)
    free = shutil.disk_usage(pool).free / 1e9
    if free < min_free_gb:
        raise OSError(f"{pool}: {free:.1f} GB free, below {min_free_gb} GB")
    try:
        written = write_archive(pool, partial, compresslevel)
        problem = differences(pool, written, read_archive(partial))
        if problem:
            raise ValueError(problem)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, target)   # only a verified archive ever gets the final name
    return finish(pool, written)


def finish(pool: Path, written: dict[str, Entry]) -> dict:
    """Record a verified archive in PACKED.json, then remove runs/ (in that order)."""
    target = pool / ARCHIVE
    files = [entry for entry in written.values() if entry[0] == "file"]
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    record = {"archive": ARCHIVE, "files": len(files), "bytes": sum(entry[1] for entry in files),
              "symlinks": sum(entry[0] == "link" for entry in written.values()),
              "directories": sum(entry[0] == "dir" for entry in written.values()),
              "archive_bytes": target.stat().st_size, "archive_sha256": digest.hexdigest(),
              "packed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "restore": f"tar -xzf {ARCHIVE} -C <this pool>"}
    (pool / MARKER).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(pool / "runs")
    return record


def resume(pool: Path) -> dict:
    """Finish a pool a stopped run left with its archive in place next to runs/.

    The archive got its final name only after it was verified, and runs/ is removed only after
    PACKED.json is written. With PACKED.json present the run stopped while removing runs/: finish
    the removal. Without it runs/ is still whole: check the archive against it again first.
    """
    if (pool / MARKER).exists():
        shutil.rmtree(pool / "runs")
        return {**json.loads((pool / MARKER).read_text(encoding="utf-8")), "resumed": True}
    written = scan_tree(pool)
    problem = differences(pool, written, read_archive(pool / ARCHIVE))
    if problem:
        raise ValueError(f"{problem}; {ARCHIVE} was already in place, check both")
    return {**finish(pool, written), "resumed": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="pools, or directories to search for pools")
    parser.add_argument("--jobs", type=int, default=4, help="pools packed in parallel")
    parser.add_argument("--compresslevel", type=int, default=6)
    parser.add_argument("--min-free-gb", type=float, default=2.0, help="refuse to start a pool below this much space")
    parser.add_argument("--dry-run", action="store_true", help="list the pools that would be packed")
    args = parser.parse_args(argv)
    pools = [pool for root in args.roots for pool in find_pools(root.resolve())]
    if not pools:
        print("nothing to pack")
        return 0
    if args.dry_run:
        print("\n".join(str(pool) for pool in pools))
        return 0
    failed = 0
    # Threads: hashing and zlib compression release the GIL, and the rest is file I/O.
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {pool: executor.submit(pack, pool, args.compresslevel, args.min_free_gb) for pool in pools}
        for pool, future in futures.items():
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 (report every pool, keep the others)
                failed += 1
                print(json.dumps({"pool": str(pool), "error": str(error)}), flush=True)
            else:
                print(json.dumps({"pool": str(pool), **record}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
