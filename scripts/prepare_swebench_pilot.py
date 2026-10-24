"""Download a deterministic, small development slice through the official HF viewer.

Only sanitized inference fields are saved. No gold patch or hidden test labels
are kept in the project. The resulting manifest is the reproducibility snapshot.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request

from budget_agent.swebench import import_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--output", type=Path, default=Path("tasks/swebench-pilot.jsonl"))
    args = parser.parse_args()
    if not 1 <= args.count <= 100 or args.offset < 0:
        parser.error("count must be 1..100 and offset non-negative")
    if args.output.exists():
        parser.error("output already exists; choose a new path to preserve the frozen task set")
    query = urllib.parse.urlencode({"dataset": "princeton-nlp/SWE-bench_Lite", "config": "default", "split": "test", "offset": args.offset, "length": args.count})
    url = "https://datasets-server.huggingface.co/rows?" + query
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)
    rows = [item["row"] for item in data["rows"]]
    if len(rows) != args.count:
        raise ValueError("The requested slice is incomplete; manifest was not written")
    # Raw rows may contain grading labels. Keep them only in a temporary file,
    # removed immediately after the whitelist-based conversion.
    with tempfile.TemporaryDirectory(prefix="budget-agent-import-") as directory:
        source = Path(directory) / "rows.json"
        source.write_text(json.dumps(rows), encoding="utf-8")
        count = import_tasks(source, args.output, split=args.split)
    provenance = {"source_url": url, "source_dataset_split": "test", "project_split": args.split,
                  "selection": "contiguous dataset order, before any model run; development convenience sample",
                  "offset": args.offset, "count": count, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                  "manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  "note": "Project split dev is our development partition, not an official SWE-bench split. No generalization claim."}
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Saved {count} sanitized tasks to {args.output}; gold patches/tests were discarded.")


if __name__ == "__main__":
    main()
