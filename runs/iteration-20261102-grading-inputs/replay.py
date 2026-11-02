"""Replay two fixture comparisons in temporary directories with Git and a CPU Python."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
BEFORE = "0406dbb2e14ad48a290385e2279a09a62a547e96"


def worker(source, temp, version):
    sys.path.insert(0, str(source / "src"))
    from budget_agent.budget import TokenCounter
    from budget_agent.models import ScriptedModel
    from budget_agent.paired import analyze_paired
    from budget_agent.report import load_results
    from budget_agent.runner import run_task
    from budget_agent.schema import RunConfig, Task

    folder = temp / version
    repo = folder / "fixture"
    repo.mkdir(parents=True)
    (repo / "counter.py").write_bytes(b"value = 0\n")
    actions = [{"tool": "write_file", "path": "counter.py", "content": "value = 1"}, {"tool": "finish"}]
    grades = {}
    for case, policy, filename, expected in (
        ("fail", "direct", "verify.py", 2),
        ("pass", "workspace-feedback", "verify.py", 1),
        ("copy", "direct", "verify_copy.py", 1),
    ):
        verifier = folder / filename
        verifier.write_bytes(("import unittest\nfrom counter import value\n"
                              "class SyntheticVerifier(unittest.TestCase):\n"
                              f"    def test_value(self): self.assertEqual(value, {expected})\n").encode())
        task = Task("grading-input-demo", "Set value to one.", "dev", repo=str(repo), verifier=str(verifier))
        counter = TokenCounter("byte")
        result = run_task(task, RunConfig(decision_policy=policy), folder / case,
                          model=ScriptedModel(actions, counter), counter=counter)
        verifier_sha = hashlib.sha256(verifier.read_bytes()).hexdigest()
        if version == "candidate":
            inputs = result["evaluation"]["grading_inputs"]
            assert inputs["kind"] == "fixture" and inputs["verifier_sha256"] == verifier_sha
        grades[case] = {"resolved": result["resolved"], "fixture_verifier_sha256": verifier_sha,
                        "tests_run": result["evaluation"]["test_results"]["tests_run"],
                        "patch_sha256": hashlib.sha256(Path(result["patch_path"]).read_bytes()).hexdigest()}
    assert [g["resolved"] for g in grades.values()] == [False, True, True]
    assert {g["tests_run"] for g in grades.values()} == {1}
    assert len({g["patch_sha256"] for g in grades.values()}) == 1

    def compare(root, left):
        results = load_results(root / left) + load_results(root / "pass")
        assert results[0]["initial_revision"] == results[1]["initial_revision"]
        assert results[0]["task_fingerprint"] == results[1]["task_fingerprint"]
        report = analyze_paired(results, "direct", "workspace-feedback", factor="decision_policy")
        pair, = report["pairs"]
        return {"raw_outcomes": [pair[k]["resolved"] for k in ("baseline", "candidate")],
                "eligible": pair["eligible_graded_pair"], "candidate_wins": report["candidate_wins"],
                "grading_input_match": pair.get("grading_input_match")}

    replay = {"grades": grades, "changed_contents_same_path": compare(folder, "fail"),
              "same_contents_moved": compare(folder, "copy")}
    if version == "candidate":
        replay["read_legacy_results"] = compare(temp / "before", "fail")
    return replay


def main():
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        prior = temp / "prior"
        archive = subprocess.check_output(["git", "archive", "--format=zip", BEFORE, "--", "src"], cwd=ROOT)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            bundle.extractall(prior)
        reports = {}
        for version, source in (("before", prior), ("candidate", ROOT)):
            reports[version] = json.loads(subprocess.check_output(
                [sys.executable, "-I", "-B", str(Path(__file__).resolve()), str(source), str(temp), version]))

    before, after = reports["before"], reports["candidate"]
    assert before["grades"] == after["grades"]
    assert before["changed_contents_same_path"]["candidate_wins"] == 1
    for case, status, eligible in (("changed_contents_same_path", "mismatched", False),
                                   ("same_contents_moved", "matched", True),
                                   ("read_legacy_results", "unavailable", False)):
        pair = after[case]
        assert (pair["grading_input_match"], pair["eligible"], pair["candidate_wins"]) == (status, eligible, 0)
    return {"before_commit": BEFORE, "scope": "Synthetic CPU fixture; same scripted patch in all runs.",
            "versions": reports}


if __name__ == "__main__":
    report = worker(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]) if len(sys.argv) > 1 else main()
    print(json.dumps(report, indent=2))
