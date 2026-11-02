"""Read historical diagnostics; pass --all only in the workspace with all 59 frozen runs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from budget_agent.diagnostics import diagnose_run

MANIFEST = ROOT / "runs/iteration-20261027-06/historical-audit.json"
TRACKED = ("runs/enterprise-contract-v1/demo-final/",
           "runs/cloud-v5-a4000-20261022-01/adaptive-guidance/direct/")
SUMMARY = "runs/cloud-v3-swe-dev-02/astropy__astropy-12907__summary__20261021T170954Z-30ae6ec4/result.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checksums(result_path):
    folder = result_path.parent
    trace = "mini.traj.json" if (folder / "mini.traj.json").is_file() else "trajectory.jsonl"
    return {name: sha(folder / name) for name in ("result.json", "calls.jsonl", trace)}


def report(entry):
    result = ROOT / entry["path"]
    before = checksums(result)
    assert before["result.json"] == entry["sha256"]
    record = json.loads(result.read_bytes())
    audit = diagnose_run(result.parent)
    change = audit["first_workspace_change"]
    charges = Counter()
    if change.get("usage"):
        calls = [json.loads(line) for line in (result.parent / "calls.jsonl").read_bytes().split(b"\n") if line.strip()]
        for call in calls[:change["usage"]["requests"]]:
            response = call["response"]
            charges[call["purpose"]] += response["input_tokens"] + response["output_tokens"]
        assert sum(charges.values()) == change["usage"]["total_tokens"]
    assert checksums(result) == before
    return {"path": entry["path"], "input_sha256": before,
            "evidence_kind": record["evidence_kind"], "resolved": record["resolved"],
            "first_successful_tool_action": audit["first_successful_tool_action"],
            "first_workspace_change": change, "prefix_tokens_by_purpose": dict(charges)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="require local archives from the entire frozen manifest")
    options = parser.parse_args()
    frozen = json.loads(MANIFEST.read_bytes())["runs"]
    tracked = [entry for entry in frozen if entry["path"].startswith(TRACKED)]
    assert len(tracked) == 3
    receipt = {"manifest_sha256": sha(MANIFEST), "tracked_examples": [report(entry) for entry in tracked]}
    if options.all:
        assert len(frozen) == 59
        totals, methods = Counter(), Counter()
        files = {}
        for entry in frozen:
            row = report(entry)
            change = row["first_workspace_change"]
            totals[row["evidence_kind"], change["status"]] += 1
            if change["status"] == "observed":
                methods[change["usage"]["method"] if change.get("usage") else "unavailable"] += 1
            folder = Path(entry["path"]).parent
            files.update({(folder / name).as_posix(): digest for name, digest in row["input_sha256"].items()})
            if entry["path"] == SUMMARY:
                receipt["local_summary_example"] = row
        receipt["local_frozen_set"] = {
            "runs": len(frozen), "unchanged_files": len(files), "observed_usage_methods": dict(methods),
            "sorted_path_tab_hash_lf_sha256": hashlib.sha256(
                "".join(f"{name}\t{digest}\n" for name, digest in sorted(files.items())).encode()).hexdigest(),
            "counts": [{"evidence_kind": kind, "status": status, "runs": number}
                       for (kind, status), number in sorted(totals.items())]}
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
