import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "runs/iteration-20261102-14182-controls/run_controls.sh"
# Last host clock sample before entering run_controls.sh and the verified stop,
# both 2026-11-02 UTC. Actual dispatch after preflight had at most 106 seconds.
ENTRY_EPOCH = 1793638629  # 16:57:09
STOP_EPOCH = 1793638735   # 16:58:55
HARNESS = ["-m", "swebench.harness.run_evaluation"]
GOLD_VERIFIER = ["-m", "budget_agent", "swe-check-gold"]
VERIFIER = ["-m", "budget_agent", "swe-check-controls"]
GOLD_REPORT = "logs/run_evaluation/controls-14182-replay-gold/gold/astropy__astropy-14182/report.json"


def run_control_replay(directory, *, script=SCRIPT, initial_epoch=ENTRY_EPOCH,
                       deadline=STOP_EPOCH, after_preflight=None, after_gold=None, gold_result="pass"):
    """Run a shell entry point with CPU stubs. Pass deadline=None for its old three-argument form."""
    root = Path(directory)
    (root / "inputs").mkdir()
    (root / "clock").write_text(str(initial_epoch))
    date = root / "date"
    date.write_text('#!/bin/sh\n[ "$#" = 1 ] && [ "$1" = +%s ] || exit 2\ncat "$REPLAY_DIR/clock"\n')
    date.chmod(0o755)

    grader = root / "fake grader"
    grader.write_text(f"#!{sys.executable}\n" + """\
import json
import os
from pathlib import Path
import sys

root = Path(os.environ["REPLAY_DIR"])
args = sys.argv[1:]
with (root / "grader-argv.jsonl").open("a") as log:
    print(json.dumps(args), file=log)

if args[0] == "-":
    sys.stdin.read()
    (root / "clock").write_text(os.environ["PREFLIGHT_END"])
elif args[0] == "-c":
    print("replay")
elif args[:2] == ["-m", "swebench.harness.run_evaluation"]:
    if args[args.index("--predictions_path") + 1] == "gold":
        outcome = os.environ["GOLD_RESULT"]
        if outcome != "missing":
            run_id = args[args.index("--run_id") + 1]
            task = args[args.index("--instance_ids") + 1]
            report = Path("logs/run_evaluation") / run_id / "gold" / task / "report.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"resolved": outcome == "pass"}))
        (root / "clock").write_text(os.environ["GOLD_END"])
elif args[:2] == ["-m", "budget_agent"] and args[2] in ("swe-check-gold", "swe-check-controls"):
    # Stand in for the CLI's pass/fail result; its report rules are tested in test_swe_controls.
    report = Path(args[args.index("--gold-report") + 1])
    if not report.is_file() or not json.loads(report.read_bytes())["resolved"]:
        sys.exit(2)
    if args[2] == "swe-check-controls":
        Path(args[args.index("--output") + 1]).write_text("simulated verifier receipt")
else:
    sys.exit("unexpected grader invocation")
""")
    grader.chmod(0o755)
    preflight_end = initial_epoch if after_preflight is None else after_preflight
    arguments = [str(script), str(grader), "inputs", "work"]
    if deadline is not None:
        arguments.append(str(deadline))
    # The marker represents a caller that gates downstream work on shell success.
    return subprocess.run(
        ["bash", "-c", 'bash "$@" && touch model-started', "replay", *arguments],
        cwd=root, capture_output=True, text=True, timeout=120,   # a hang guard, not a speed limit
        env={**os.environ, "PATH": f"{root}:{os.environ['PATH']}", "REPLAY_DIR": str(root),
             "PREFLIGHT_END": str(preflight_end),
             "GOLD_RESULT": gold_result,
             "GOLD_END": str(preflight_end if after_gold is None else after_gold)},
    )


@unittest.skipUnless(sys.platform == "linux", "controls run on a Linux grading host")
class ControlWindowTest(unittest.TestCase):
    def replay(self, **options):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        return run_control_replay(self.root, **options)

    def assert_outcome(self, result, *, admissions, predictions, returncode):
        self.assertEqual(result.returncode, returncode, result.stderr)
        work = self.root / "work"
        events = [json.loads(line) for line in (work / "control-admission.jsonl").read_text().splitlines()]
        keys = ("stage", "decision", "checked_epoch", "deadline_epoch", "remaining_seconds", "required_seconds")
        expected = [(stage, decision, STOP_EPOCH - left, STOP_EPOCH, left, required)
                    for stage, decision, left, required in admissions]
        self.assertEqual([tuple(event[key] for key in keys) for event in events], expected)

        calls = [json.loads(line) for line in (self.root / "grader-argv.jsonl").read_text().splitlines()]
        self.assertEqual(calls[0], ["-", str(work / "control-spec.json"), str(self.root / "inputs")])
        harnesses = [call for call in calls if call[:2] == HARNESS]
        self.assertEqual([call[call.index("--predictions_path") + 1] for call in harnesses], predictions)
        for call in harnesses:
            self.assertEqual(call[call.index("--timeout") + 1], "600")
        gold_checks = [call for call in calls if call[:3] == GOLD_VERIFIER]
        if harnesses:
            self.assertEqual(gold_checks, [GOLD_VERIFIER + ["--spec", str(work / "control-spec.json"),
                                                          "--gold-report", GOLD_REPORT]])
            self.assertEqual(calls.index(gold_checks[0]), calls.index(harnesses[0]) + 1)
        else:
            self.assertEqual(gold_checks, [])
        verified = returncode == 0
        self.assertEqual(sum(call[:3] == VERIFIER for call in calls), int(verified))
        self.assertEqual((work / "controls-verified.json").exists(), verified)
        self.assertEqual((self.root / "model-started").exists(), verified)

    def test_archived_window_defers_both_controls(self):
        result = self.replay()
        self.assert_outcome(result, admissions=[("gold", "defer", 106, 1320)], predictions=[], returncode=75)

    def test_preflight_consumes_last_second_before_gold(self):
        result = self.replay(initial_epoch=STOP_EPOCH - 1321, after_preflight=STOP_EPOCH - 1320)
        self.assert_outcome(result, admissions=[("gold", "defer", 1320, 1320)], predictions=[], returncode=75)

    def test_gold_finishes_with_insufficient_time_for_negative(self):
        for left in (719, 720):
            with self.subTest(remaining=left):
                result = self.replay(initial_epoch=STOP_EPOCH - 1321, after_gold=STOP_EPOCH - left)
                self.assert_outcome(result, admissions=[("gold", "admit", 1321, 1320),
                                                        ("negative", "defer", left, 720)],
                                    predictions=["gold"], returncode=75)
                self.assertEqual((self.root / "work" / GOLD_REPORT).read_bytes(), b'{"resolved": true}')

    def test_both_controls_dispatch_one_second_above_each_boundary(self):
        result = self.replay(initial_epoch=STOP_EPOCH - 1321, after_gold=STOP_EPOCH - 721)
        self.assert_outcome(result, admissions=[("gold", "admit", 1321, 1320), ("negative", "admit", 721, 720)],
                            predictions=["gold", str(self.root / "inputs/negative.jsonl")], returncode=0)

    def test_zero_exit_gold_without_passing_report_stops_before_negative(self):
        for outcome in ("missing", "unresolved"):
            with self.subTest(gold_result=outcome):
                result = self.replay(initial_epoch=STOP_EPOCH - 3600, gold_result=outcome)
                self.assert_outcome(result, admissions=[("gold", "admit", 3600, 1320)],
                                    predictions=["gold"], returncode=2)
                report = self.root / "work" / GOLD_REPORT
                if outcome == "unresolved":
                    self.assertEqual(json.loads(report.read_bytes()), {"resolved": False})
                else:
                    self.assertFalse(report.exists())

    def test_absent_or_invalid_deadline_stops_before_grader(self):
        for invalid in (None, "", "0", "-1", "1793638735.5", "1793638735ms"):
            with self.subTest(deadline=invalid):
                result = self.replay(deadline=invalid)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / "grader-argv.jsonl").exists())
                self.assertFalse((self.root / "model-started").exists())
