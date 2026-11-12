"""Every GPU experiment driver refuses to start while another one runs; the guards must agree."""
import re
from pathlib import Path

AUTODL = Path(__file__).resolve().parents[1] / "scripts" / "autodl"
GUARD = re.compile(r'pgrep -f "\^bash \$CODE/scripts/autodl/\(([^)]*)\)\\\.sh"')


def guards():
    found = {}
    for script in sorted(AUTODL.glob("*.sh")):
        match = GUARD.search(script.read_text(encoding="utf-8"))
        if match:
            found[script.stem] = match[1].split("|")
    return found


def test_every_guard_lists_the_same_drivers():
    found = guards()
    assert len(found) >= 6
    assert len({tuple(names) for names in found.values()}) == 1, found


def test_guards_cover_every_guarded_driver_and_only_real_scripts():
    found = guards()
    names = set(next(iter(found.values())))
    assert set(found) <= names                      # each guarded driver also blocks the others
    assert all((AUTODL / f"{name}.sh").exists() for name in names)


def test_rollout_pools_only_read_manifests_the_rollout_user_can_open():
    # run_pool.sh runs the harness as nobody, who cannot enter /root (round 10's first start
    # failed on a manifest under $OUT). The grpo harness itself runs as root and writes its
    # step manifests into the rollout directory, so only direct run_pool.sh calls matter.
    for script in sorted(AUTODL.glob("*.sh")):
        text = script.read_text(encoding="utf-8").replace("\\\n", " ")
        for command in re.findall(r"run_pool\.sh[^\n]*", text):
            for value in re.findall(r'--tasks "([^"]+)"', command):
                assert not value.startswith(("$OUT", "$ML", "/root")), f"{script.name}: {value}"
