"""Rejection-sampled SFT data: which winners are kept and how their requests become samples."""
import importlib.util
import json
from pathlib import Path
import sys

SPEC = importlib.util.spec_from_file_location("sft_data_from_runs", Path(__file__).parents[1] / "scripts/sft_data_from_runs.py")
sft_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sft_data)


def write_run(run_dir, requests):
    """requests: list of (messages, reply) for agent calls; a summary call is mixed in and must be ignored."""
    run_dir.mkdir(parents=True)
    calls = [{"purpose": "agent", "messages": m, "response": {"content": r}} for m, r in requests]
    calls.insert(1, {"purpose": "summary", "messages": [{"role": "user", "content": "s"}], "response": {"content": "x"}})
    calls.append({"purpose": "agent", "status": "usage_unknown"})
    (run_dir / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls), encoding="utf-8")
    (run_dir / "result.json").write_text("{}", encoding="utf-8")
    return str(run_dir / "result.json")


SYSTEM = {"role": "system", "content": "sys"}
TASK = {"role": "user", "content": "task"}


def test_prefix_consistent_requests_become_one_full_sample(tmp_path):
    obs = {"role": "user", "content": "obs"}
    write_run(tmp_path / "run", [([SYSTEM, TASK], "a1"),
                                 ([SYSTEM, TASK, {"role": "assistant", "content": "a1"}, obs], "a2")])
    mode, samples = sft_data.samples_for(tmp_path / "run")
    assert mode == "full" and len(samples) == 1
    assert [m["content"] for m in samples[0]] == ["sys", "task", "a1", "obs", "a2"]


def test_rewritten_history_trains_each_request_on_its_last_reply(tmp_path):
    write_run(tmp_path / "run", [([SYSTEM, TASK], "a1"), ([SYSTEM, TASK, {"role": "user", "content": "window"}], "a2")])
    mode, samples = sft_data.samples_for(tmp_path / "run")
    assert mode == "last" and [s[-1]["content"] for s in samples] == ["a1", "a2"]


def test_only_capped_unflagged_winners_of_the_split_are_kept(tmp_path, monkeypatch, capsys):
    rows = []
    for index, (task, split, resolved, signals) in enumerate([
            ("t1", "train", True, []), ("t1", "train", True, []), ("t1", "train", True, []),
            ("t2", "train", True, ["process_exit"]), ("t3", "train", False, []), ("t4", "eval", True, [])]):
        result = write_run(tmp_path / f"r{index}", [([SYSTEM, TASK], f"reply {index}")])
        rows.append({"task_id": task, "split": split, "sample": index, "resolved": resolved,
                     "patch_signals": signals, "result_path": result})
    attempts = tmp_path / "attempts.jsonl"
    attempts.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    output = tmp_path / "sft.jsonl"
    monkeypatch.setattr(sys, "argv", ["sft", "--attempts", str(attempts), "--per-task", "2", "--output", str(output)])
    assert sft_data.main() == 0
    examples = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    report = json.loads(capsys.readouterr().out)
    assert [e["task_id"] for e in examples] == ["t1", "t1"] and report["flagged_winners_excluded"] == 1
    assert {e["mode"] for e in examples} == {"full"} and report["tasks_with_success"] == 1
    monkeypatch.setattr(sys, "argv", ["sft", "--attempts", str(attempts), "--per-task", "2", "--output", str(output),
                                      "--keep-flagged"])
    assert sft_data.main() == 0
    assert sorted({json.loads(line)["task_id"] for line in output.read_text(encoding="utf-8").splitlines()}) == ["t1", "t2"]
