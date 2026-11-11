"""The CPU training-math job, the pyproject extras and the GPU host's lock must agree."""
import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
LOCKS = ("requirements-gpu-host.lock.txt", "requirements-rollout.lock.txt")


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def pins(name):
    rows = {}
    for line in (ROOT / name).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            package, _, version = line.partition("==")
            rows[normalize(package)] = version
    return rows


def test_lock_files_pin_every_package_exactly():
    for name in LOCKS:
        for line in (ROOT / name).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                assert re.fullmatch(r"[A-Za-z0-9._-]+==[A-Za-z0-9.+!]+", line), (name, line)


def test_ci_training_math_job_uses_the_host_versions():
    host = pins("requirements-gpu-host.lock.txt")
    workflow = (ROOT / ".github/workflows/tests.yml").read_text()
    for package in ("torch", "transformers", "peft"):
        assert f"{package}=={host[package]}" in workflow, package


def test_ml_extras_admit_both_tested_stacks():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    stacks = {"gpu host": pins("requirements-gpu-host.lock.txt"),
              "windows": pins("requirements-train.verified.txt")}
    for extra in ("train", "inference"):
        for spec in extras[extra]:
            requirement = Requirement(spec)
            for stack, versions in stacks.items():
                version = versions[normalize(requirement.name)].split("+")[0]   # drop local tags such as +cu128
                assert requirement.specifier.contains(version), (extra, spec, stack, version)


def test_rollout_interpreter_lock_carries_the_token_counter():
    assert "tokenizers" in pins("requirements-rollout.lock.txt")
