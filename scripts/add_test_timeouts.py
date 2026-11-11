"""Give existing SWE-smith manifests the per-task test-suite limit from the admission ledger.

Manifests built before build_swesmith_local.py wrote `test_timeout_seconds`
get the same value it writes now (10x the slowest admission run, at least
30 s), without re-admitting every task. It bounds grading and the agent's
public_test and run commands. Each manifest is rewritten in place
through a temporary file; running it again changes nothing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("build_swesmith_local", Path(__file__).with_name("build_swesmith_local.py"))
if SPEC is None or SPEC.loader is None:
    raise ImportError("build_swesmith_local.py must sit next to this script")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--admission", type=Path, required=True, help="admission.jsonl from build_swesmith_local.py")
    parser.add_argument("manifests", type=Path, nargs="+")
    args = parser.parse_args()
    seconds = {row["instance_id"]: row["seconds"] for row in map(json.loads, args.admission.read_text().splitlines())
               if row.get("admitted")}
    for manifest in args.manifests:
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            row["metadata"]["test_timeout_seconds"] = build.test_timeout_seconds(seconds[row["id"]])
        temporary = manifest.with_suffix(manifest.suffix + ".tmp")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        os.replace(temporary, manifest)
        limits = sorted(row["metadata"]["test_timeout_seconds"] for row in rows)
        print(f"{manifest}: {len(rows)} tasks, limit min {limits[0]} s, median {limits[len(limits) // 2]} s, "
              f"max {limits[-1]} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
