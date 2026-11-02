"""Replay four reviewed no-op edits with the internal runner. Requires Git and GNU sed."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from budget_agent.budget import TokenCounter
from budget_agent.models import ScriptedModel
from budget_agent.runner import run_task
from budget_agent.schema import RunConfig, Task

ARCHIVE = ROOT / "runs/cloud-v5-a4000-20261022-01/mini-baseline/astropy__astropy-12907__mini-swe-textbased"
TARGET = "astropy/modeling/separable.py"
PATTERNS = ["def _cstack(left, right):"] * 2 + ["    if isinstance(left, Model):"] * 2
VERIFIER = """from pathlib import Path
import unittest

class MarkerCheck(unittest.TestCase):
    def test_requested_marker(self):
        self.assertTrue(Path('astropy/modeling/separable.py').read_bytes().endswith(b'# replay complete\\n'))
"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_bytes().split(b"\n") if line.strip()]


def replay(policy, task, output):
    # Only construct these fixed commands. The archived command strings are never executed.
    actions = [{"tool": "run", "command": f'sed -i "s/{text}/{text}/g" {TARGET}'} for text in PATTERNS]
    actions.append({"tool": "finish"})
    counter = TokenCounter("byte")
    config = RunConfig(strategy="full", decision_policy=policy, provider="scripted", environment="local",
                       context_tokens=8192, total_tokens=32000, max_output_tokens=768, max_steps=5)
    result = run_task(task, config, output, model=ScriptedModel(actions, counter), counter=counter)
    folder = Path(result["result_path"]).parent
    events = read_jsonl(folder / "trajectory.jsonl")
    calls = read_jsonl(folder / "calls.jsonl")
    assert [e["action"] for e in events] == actions and len(calls) == len(actions)
    assert result["evidence_kind"] == "scripted_smoke" and result["resolved"] is False
    assert result["evaluation"]["test_results"]["tests_run"] == 1
    assert (folder / "patch.diff").read_bytes() == b""
    assert (folder / "workspace" / TARGET).read_bytes() == (Path(task.repo) / TARGET).read_bytes()
    steps = []
    base = calls[0]["messages"][0]["content"]
    for index, event in enumerate(events[:-1]):
        before, after = event["revision_before"], event["revision"]
        assert event["ok"] and event["output"] == "" and before == after == result["initial_revision"]
        next_system = calls[index + 1]["messages"][0]["content"]
        assert next_system.startswith(base)
        suffix = next_system[len(base):]
        if policy == "workspace-feedback":
            assert f"Workspace feedback for {event['id']} (run)" in suffix
        else:
            assert not suffix
        steps.append({"event": event["id"], "before": before, "after": after, "ok": event["ok"],
                      "stdout_and_stderr": event["output"], "next_system_suffix": suffix.strip()})
    ledger = result["budget"]
    assert not ledger["overrun"] and not ledger["unreconciled_request"]
    # Bind both arms to the same actor inputs after removing just the checked suffixes.
    core_inputs = [[{"role": "system", "content": base}, *c["messages"][1:]] for c in calls]
    return {"steps": steps, "calls": result["calls"], "input_units": ledger["input_tokens"],
            "output_units": ledger["output_tokens"], "total_units": ledger["total_tokens"],
            "fixture_resolved": result["resolved"], "patch_sha256": digest((folder / "patch.diff").read_bytes()),
            "inputs_without_feedback_sha256": digest(json.dumps(core_inputs, sort_keys=True).encode())}


def main():
    if os.name == "nt" and not shutil.which("sed"):
        git_root = Path(shutil.which("git")).resolve().parents[1]
        os.environ["PATH"] += os.pathsep + str(git_root / "usr/bin")
    sed = subprocess.check_output(["sed", "--version"], text=True).splitlines()[0]
    assert "GNU sed" in sed, "This replay uses GNU sed -i."
    trajectory = (ARCHIVE / "mini.traj.json").read_bytes()
    messages = json.loads(trajectory)["messages"]
    shell = [a["command"] for m in messages if m["role"] == "assistant" for a in m["extra"]["actions"]]
    assert shell[1] == "cat /testbed/" + TARGET
    assert shell[2:] == [f"sed -i 's/{text}/{text}/g' /testbed/{TARGET}" for text in PATTERNS]
    observed = [m["extra"] for m in messages if "raw_output" in m.get("extra", {})]
    assert len(observed) == 6 and all(o["returncode"] == 0 and o["raw_output"] == "" for o in observed[2:])
    source = observed[1]["raw_output"].encode("utf-8")
    assert digest(source) == "cac38fe9744203c720d50d6c88dcdcb0d93a7b2326fe89911e83640ad5676b0f"
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        repo = temp / "repo"
        target = repo / TARGET
        target.parent.mkdir(parents=True)
        target.write_bytes(source)
        verifier = temp / "verify.py"
        verifier.write_text(VERIFIER, encoding="utf-8")
        task = Task("four-identity-edits", "Append the comment '# replay complete' as the final line.",
                    "dev", repo=str(repo), verifier=str(verifier), metadata={"editable": [TARGET]})
        arms = {policy: replay(policy, task, temp / policy) for policy in ("direct", "workspace-feedback")}
    baseline, candidate = arms["direct"], arms["workspace-feedback"]
    assert baseline["inputs_without_feedback_sha256"] == candidate["inputs_without_feedback_sha256"]
    assert candidate["input_units"] - baseline["input_units"] == sum(
        len(("\n" + e["next_system_suffix"]).encode()) for e in candidate["steps"])
    receipt = {"scope": "Fixed actions 3-6 of one external trace mapped into the internal runner. "
                        "Synthetic marker verifier; no model or official Astropy tests.",
               "archive": ARCHIVE.relative_to(ROOT).as_posix(), "trajectory_sha256": digest(trajectory),
               "source_sha256": digest(source), "sed": sed, "counter": "UTF-8 bytes with message framing",
               "context_limit": 8192, "total_limit": 32000, "arms": arms}
    # Explicit output keeps the checked-in receipt untouched on a fresh-clone reproduction.
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
