"""Chaining rounds: the next one starts only after a clean finish, a verified deploy and an idle GPU."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/autodl/chain_round.sh"
pytestmark = pytest.mark.skipif(sys.platform != "linux" or not all(shutil.which(c) for c in ("bash", "pgrep", "setsid")),
                                reason="GPU-host shell script")


def tree_hash(root):
    pipeline = "find src scripts configs -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16"
    return subprocess.run(["bash", "-c", pipeline], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def host(tmp_path):
    agent, ml, stubs = tmp_path / "agent", tmp_path / "ml", tmp_path / "bin"
    current = agent / "budget-agent" / "scripts" / "autodl"
    current.mkdir(parents=True)
    (current / "archive_rollouts.sh").write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/archived\n')
    (ml / "logs").mkdir(parents=True)
    (ml / "logs" / "round7.log").write_text("training...\n== round 7 done 2026-11-12T10:00:00Z\n")
    new = tmp_path / "new"
    for directory in ("src", "configs", "scripts/autodl"):
        (new / directory).mkdir(parents=True)
    (new / "src" / "lib.py").write_text("x = 1\n")
    (new / "configs" / "c.toml").write_text("[run]\n")
    (new / "scripts" / "autodl" / "round8_grpo.sh").write_text(
        f'echo "SEED=$SEED STEPS=$STEPS" > {tmp_path}/started\nsleep 3\n')
    tarball = tmp_path / "budget-agent-new.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        for path in sorted(new.rglob("*")):
            archive.add(path, arcname=str(path.relative_to(new)), recursive=False)
    stubs.mkdir()
    (stubs / "nvidia-smi").write_text("#!/bin/sh\ncat \"$GPU_PROCESSES\" 2>/dev/null\n")
    (stubs / "nvidia-smi").chmod(0o755)
    return {"tmp": tmp_path, "agent": agent, "ml": ml, "tarball": tarball, "tree": tree_hash(new),
            "env": {**os.environ, "AGENT": str(agent), "ML": str(ml), "POLL": "1", "START_CHECK": "1",
                    "PATH": f"{stubs}:{os.environ['PATH']}", "GPU_PROCESSES": str(tmp_path / "gpu")}}


def chain(host, tree=None, marker="== round 7 done"):
    return subprocess.run(
        ["bash", str(SCRIPT), "--after", "round7_grpo", "--marker", marker, "--archive", "/opt/x/round7",
         "--deploy", str(host["tarball"]), "--tree", tree or host["tree"], "--start", "round8_grpo",
         "--env", "SEED=1", "--env", "STEPS=29"], env=host["env"], capture_output=True, text=True, timeout=60)


def test_a_clean_finish_archives_deploys_and_starts_the_next_round(host):
    result = chain(host)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (host["tmp"] / "archived").read_text().split() == ["/opt/x/round7"]
    assert (host["agent"] / "budget-agent" / "src" / "lib.py").exists()
    assert (host["agent"] / "budget-agent.round7_grpo" / "scripts" / "autodl" / "archive_rollouts.sh").exists()
    for _ in range(50):
        if (host["tmp"] / "started").exists():
            break
        time.sleep(0.1)
    assert (host["tmp"] / "started").read_text().split() == ["SEED=1", "STEPS=29"]
    assert "== round8_grpo started" in result.stdout


@pytest.mark.parametrize("problem", ["no marker", "wrong tree", "gpu busy"])
def test_any_surprise_stops_before_the_next_round(host, problem):
    if problem == "gpu busy":
        (host["tmp"] / "gpu").write_text("12345\n")
    result = chain(host, tree="0" * 16 if problem == "wrong tree" else None,
                   marker="== round 7 finished" if problem == "no marker" else "== round 7 done")
    assert result.returncode != 0
    assert not (host["tmp"] / "started").exists()
    assert not (host["agent"] / "budget-agent" / "src").exists()       # the old tree is still in place
    assert not (host["tmp"] / "archived").exists()                      # nothing archived either
