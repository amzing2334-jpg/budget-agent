"""Behaviour rows of the experiment records, computed from a rollout pool."""
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("pool_behaviour", Path(__file__).parents[1] / "scripts/pool_behaviour.py")
assert SPEC is not None and SPEC.loader is not None
pool_behaviour = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pool_behaviour)


def attempt(pool, task, sample, status, resolved, tools, **extra):
    run = pool / "runs" / f"s{sample}" / f"{task}__r"
    run.mkdir(parents=True)
    (run / "trajectory.jsonl").write_text("".join(json.dumps({"action": {"tool": t}}) + "\n" for t in tools))
    return {"task_id": task, "sample": sample, "status": status, "resolved": resolved,
            "result_path": str(run / "result.json"), "steps": len(tools), "input_tokens": 100, "output_tokens": 10,
            "elapsed_seconds": 30, "model_seconds": 3, **extra}


def test_describes_statuses_finishing_precision_repositories_and_tools(tmp_path):
    pool = tmp_path / "pool"
    rows = [attempt(pool, "Suor__funcy.abc.bug1", 0, "finished", True, ["read_file", "edit_file", "finish"]),
            attempt(pool, "Suor__funcy.abc.bug1", 1, "finished", False, ["finish"]),
            attempt(pool, "msiemens__tinydb.def.bug2", 0, "max_steps", False, ["run", "run"]),
            attempt(pool, "msiemens__tinydb.def.bug2", 1, "stuck", None, ["edit_file"])]
    orphan = pool / "runs" / "s9" / "interrupted"       # not recorded: ignored
    orphan.mkdir(parents=True)
    (orphan / "trajectory.jsonl").write_text('{"action": {"tool": "run"}}\n')
    (pool / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = pool_behaviour.describe(pool)
    assert (report["attempts"], report["resolved"], report["ungraded"], report["pass@1"]) == (4, 1, 1, 0.25)
    assert report["solved_tasks"] == 1 and report["finished_and_resolved"] == [1, 2]
    assert report["statuses"] == {"finished": 2, "max_steps": 1, "stuck": 1}
    assert report["per_repository"] == {"funcy": "1/2", "tinydb": "0/2"}
    assert report["tool_calls"] == {"finish": 2, "edit_file": 2, "run": 2, "read_file": 1}
    assert report["per_attempt"]["steps"] == 1.8 and report["per_attempt"]["model_seconds"] == 3.0
