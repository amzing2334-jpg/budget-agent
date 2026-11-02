#!/usr/bin/env bash
# On a prepared Linux grading host: run the two official controls only.
set -euo pipefail
umask 077
grader=$(realpath "${1:?absolute path to the Python executable in the grading venv}")
inputs=$(realpath "${2:?private grading-input directory}")
work=$(realpath -m "${3:?new private result directory}")
here=$(cd "$(dirname "$0")" && pwd -P)
root=$(realpath "$here/../..")
task=astropy__astropy-14182
for dir in "$inputs" "$work"; do
    case "$dir/" in "$root/"*) echo "Use a private directory outside the solver repository" >&2; exit 2;; esac
done
export PYTHONPATH="$root/src"
mkdir -- "$work"
cd "$work"
cp -- "$here/14182-spec.json" control-spec.json
spec="$work/control-spec.json"

"$grader" - "$spec" "$inputs" <<'PY'
import hashlib, json, subprocess, sys
from importlib.metadata import version
from pathlib import Path

def sha(data):
    return hashlib.sha256(data).hexdigest()

spec_bytes = Path(sys.argv[1]).read_bytes()
spec = json.loads(spec_bytes)
private = Path(sys.argv[2])
data = (private / "pinned-dataset.json").read_bytes()
frozen = spec["dataset"]
if (sha(data) != frozen["pinned_grader_dataset_sha256"] or
        sha((private / "negative.jsonl").read_bytes()) != frozen["frozen_negative_jsonl_sha256"]):
    raise SystemExit("Grading input files differ from the frozen spec")
if version("swebench") != "5.0.2":
    raise SystemExit("Expected swebench==5.0.2")
from swebench.harness.utils import make_test_spec
row, = json.loads(data)
case = make_test_spec(row)
ref = spec["solver"]["pinned_image"]
if case.image != ref or case.image_assets or sha(case.eval_script.encode("utf-8")) != spec["eval_script_sha256"]:
    raise SystemExit("Image, assets or generated test script differ from the frozen spec")
image, = json.loads(subprocess.check_output(["docker", "image", "inspect", ref]))
platform = f"{image['Os']}/{image['Architecture']}"
if platform != spec["image"]["platform"]:
    raise SystemExit(f"Unexpected local image platform: {platform}")
Path("preflight.json").write_text(json.dumps({
    "swebench": version("swebench"), "spec_sha256": sha(spec_bytes),
    "image_reference": ref, "host_image_id": image["Id"], "platform": platform
}, indent=2) + "\n", encoding="utf-8")
PY

label="controls-14182-$("$grader" -c 'import uuid; print(uuid.uuid4().hex)')"
harness=("$grader" -m swebench.harness.run_evaluation --dataset_name "$inputs/pinned-dataset.json"
         --split test --instance_ids "$task" --max_workers 1 --timeout 600)
"${harness[@]}" --predictions_path gold --run_id "$label-gold"
"${harness[@]}" --predictions_path "$inputs/negative.jsonl" --run_id "$label-negative"
"$grader" -m budget_agent swe-check-controls --spec "$spec" \
    --gold-report "logs/run_evaluation/$label-gold/gold/$task/report.json" \
    --negative-report "logs/run_evaluation/$label-negative/no_repair_control/$task/report.json" \
    --output controls-verified.json
