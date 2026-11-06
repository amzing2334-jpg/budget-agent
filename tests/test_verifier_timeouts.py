"""Per-task verifier limits for SWE-smith manifests, from the admission ledger."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("add_verifier_timeouts", ROOT / "scripts/add_verifier_timeouts.py")
migrate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migrate)


def test_limit_is_ten_times_the_slowest_admission_run_with_a_floor():
    assert migrate.build.verifier_timeout_seconds({"gold": 0.47, "bug": 0.52}) == 30
    assert migrate.build.verifier_timeout_seconds({"gold": 4.01, "bug": 1.0}) == 41


def test_manifests_gain_the_limit_in_place_and_reruns_change_nothing(tmp_path, monkeypatch):
    admission = tmp_path / "admission.jsonl"
    admission.write_text("".join(json.dumps(row) + "\n" for row in [
        {"instance_id": "a", "admitted": True, "seconds": {"gold": 1.47, "bug": 1.42}},
        {"instance_id": "b", "admitted": True, "seconds": {"gold": 6.2, "bug": 5.0}},
        {"instance_id": "c", "admitted": False, "reason": "flaky on clean tree"}]), encoding="utf-8")
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("".join(json.dumps({"id": i, "metadata": {"suite": "s"}}) + "\n" for i in ("a", "b")),
                        encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["add_verifier_timeouts", "--admission", str(admission), str(manifest)])
    assert migrate.main() == 0
    first = manifest.read_text(encoding="utf-8")
    assert [json.loads(line)["metadata"] for line in first.splitlines()] == [
        {"suite": "s", "verifier_timeout_seconds": 30}, {"suite": "s", "verifier_timeout_seconds": 62}]
    assert migrate.main() == 0 and manifest.read_text(encoding="utf-8") == first
    assert not manifest.with_suffix(".jsonl.tmp").exists()
