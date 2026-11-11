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
