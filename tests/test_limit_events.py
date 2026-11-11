"""Counting the commands that the harness limits stopped."""
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("limit_events", Path(__file__).parents[1] / "scripts/limit_events.py")
assert SPEC is not None and SPEC.loader is not None
limit_events = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(limit_events)


def write_run(pool, name, events, evaluation):
    run = pool / "runs" / name / "r1"
    run.mkdir(parents=True)
    (run / "trajectory.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    (run / "result.json").write_text(json.dumps({"evaluation": evaluation}))


def event(tool, output):
    return {"action": {"tool": tool}, "output": output, "ok": False}


def test_counts_timeouts_memory_errors_and_verifier_limits(tmp_path):
    write_run(tmp_path, "a", [
        event("run", "...\n[command timed out after 41s; process tree terminated]"),
        event("public_test", "E   MemoryError"),
        event("public_test", "1 passed"),
        event("read_file", "MemoryError is mentioned in a file the agent read"),
    ], {"resolved": False, "timed_out": True})
    write_run(tmp_path, "b", [event("run", "ok")], {"resolved": False, "memory_exceeded": True})
    write_run(tmp_path, "c", [], {"resolved": True})
    assert limit_events.pool_limits(tmp_path) == {
        "pool": str(tmp_path), "attempts": 3, "commands": {"run": 2, "public_test": 2},
        "timed_out": {"run": 1, "public_test": 0}, "memory_error": {"run": 0, "public_test": 1},
        "verifier_timed_out": 1, "verifier_memory_exceeded": 1}


def test_empty_pool_counts_nothing(tmp_path):
    assert limit_events.pool_limits(tmp_path)["attempts"] == 0
