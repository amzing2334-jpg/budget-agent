"""Prepare private 14182 control files from a local parquet. Requires pyarrow."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq


SPEC = Path(__file__).with_name("14182-spec.json")
ROOT = SPEC.resolve().parents[2]
NEGATIVE_PATCH = (
    "diff --git a/.budget_agent_negative_control.txt b/.budget_agent_negative_control.txt\n"
    "new file mode 100644\n--- /dev/null\n+++ b/.budget_agent_negative_control.txt\n"
    "@@ -0,0 +1 @@\n+No source changes; negative control.\n"
)


def checked(data, expected, label):
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {label}")
    return data


def encode(value):
    return (json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii")


def prepare(parquet, output):
    output = output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("The private grading directory must be outside the solver repository")
    if output.exists():
        raise FileExistsError(f"Grading directory already exists: {output}")
    spec = json.loads(SPEC.read_bytes())
    frozen = spec["dataset"]
    data = checked(parquet.read_bytes(), frozen["parquet_sha256"], "input parquet")
    row, = pq.read_table(io.BytesIO(data), filters=[("instance_id", "=", spec["instance_id"])]).to_pylist()
    pinned = dict(row, image=spec["solver"]["pinned_image"])
    negative = {"instance_id": spec["instance_id"], "model_name_or_path": "no_repair_control",
                "model_patch": NEGATIVE_PATCH}
    files = {
        "dataset.json": checked(encode([row]), frozen["frozen_local_dataset_json_sha256"], "dataset.json"),
        "pinned-dataset.json": checked(encode([pinned]), frozen["pinned_grader_dataset_sha256"], "pinned-dataset.json"),
        "negative.jsonl": checked(encode(negative), frozen["frozen_negative_jsonl_sha256"], "negative.jsonl"),
    }
    with TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / "controls"
        staged.mkdir(mode=0o700)
        for name, contents in files.items():
            (staged / name).write_bytes(contents)
        staged.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True, help="Exact local SWE-bench Lite test parquet")
    parser.add_argument("--output", type=Path, required=True, help="New private directory under an existing parent")
    args = parser.parse_args()
    prepare(args.parquet, args.output)
    print("Prepared verified dataset.json, pinned-dataset.json, and negative.jsonl.")
