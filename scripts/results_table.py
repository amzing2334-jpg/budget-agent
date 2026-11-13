"""Build the README results tables and the forest plot from the frozen comparisons' raw outputs.

    python scripts/results_table.py --check     # fail if a README or the figure differs from the data
    python scripts/results_table.py --write     # rewrite them from the data

docs/data/comparisons.json lists every frozen comparison: its raw compare_pools.py output, the
decision rule the protocol fixed (superiority, or non-inferiority with a margin of 2 points) and
the verdict the record states. Every number in the tables and the figure is read from the raw
output, and a stated verdict that the interval does not support under its rule is an error.
The tables live between <!-- results:start --> and <!-- results:end --> in README.md (Chinese)
and README.en.md (English); the figure is docs/figures/frozen_comparisons.svg.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/data/comparisons.json"
READMES = {"zh": ROOT / "README.md", "en": ROOT / "README.en.md"}
FIGURE = ROOT / "docs/figures/frozen_comparisons.svg"
START, END = "<!-- results:start -->", "<!-- results:end -->"
MARGIN = 2.0   # non-inferiority margin, percentage points
HEADERS = {"zh": ("实验", "留出集 pass@1", "差值 [95% CI]", "结论", "记录"),
           "en": ("Experiment", "Held-out pass@1", "Delta [95% CI]", "Verdict", "Record")}


def load(registry: Path = REGISTRY) -> list[dict]:
    rows = json.loads(registry.read_text(encoding="utf-8"))
    for row in rows:
        data = json.loads((registry.parent / row["data"]).read_text(encoding="utf-8"))
        row.update(baseline=100 * data["baseline_pass@1"], candidate=100 * data["candidate_pass@1"],
                   delta=100 * data["delta"], low=100 * data["ci95"][0], high=100 * data["ci95"][1])
    return rows


def verdict(row: dict) -> str:
    """What the protocol's rule concludes from the interval (rounded to the reported precision)."""
    low, high = round(row["low"], 2), round(row["high"], 2)
    if row["rule"] == "non-inferiority":
        return "non_inferior" if low >= -MARGIN else "non_inferiority_not_shown"
    if row["rule"] != "superiority":
        raise ValueError(f"{row['data']}: unknown rule {row['rule']!r}")
    return "improved" if low > 0 else "worse" if high < 0 else "no_detectable_change"


def check(rows: list[dict]) -> None:
    for row in rows:
        if verdict(row) != row["expect"]:
            raise ValueError(f"{row['data']}: the record says {row['expect']}, the interval gives {verdict(row)}")


def points(value: float) -> str:
    value = round(value, 2)
    return "0.00" if value == 0 else f"+{value:.2f}" if value > 0 else f"−{-value:.2f}"


def table(rows: list[dict], lang: str) -> str:
    lines = ["| " + " | ".join(HEADERS[lang]) + " |", "|---|---|---|---|---|"]
    for row in rows:
        text, bold = row[lang], row.get("bold")
        candidate = f"{row['candidate']:.2f}%"
        delta = f"{points(row['delta'])} [{points(row['low'])}, {points(row['high'])}]"
        if bold in ("candidate", "both"):
            candidate = f"**{candidate}**"
        if bold == "both":
            delta = f"**{delta}**"
        link = f"docs/{row['record']}"
        lines.append(f"| {text['experiment']} | {row['baseline']:.2f}% → {candidate} | {delta} | {text['verdict']} | "
                     f"[{text['link']}]({link}) |")
    return "\n".join(lines)


def figure(rows: list[dict]) -> str:
    """A forest plot: one row per comparison, the interval and the point estimate, 0 and −2 marked."""
    lo, hi, left, width, row_h, top = -5.0, 10.0, 190, 560, 34, 36
    x = lambda v: round(left + (min(max(v, lo), hi) - lo) / (hi - lo) * width, 1)  # noqa: E731
    height = top + row_h * len(rows) + 46
    blue, orange, ink, grey = "#2A62C9", "#C9600F", "#14213D", "#5F6979"
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}" '
           'font-family="Helvetica, Arial, sans-serif" font-size="14">',
           f'<rect width="900" height="{height}" fill="#FFFFFF"/>',
           f'<text x="{left}" y="20" fill="{ink}" font-weight="bold">Held-out pass@1 difference, percentage points (95% CI)</text>']
    bottom = top + row_h * len(rows)
    for tick in (-4, -2, 0, 2, 4, 6, 8, 10):
        out.append(f'<line x1="{x(tick)}" y1="{top - 6}" x2="{x(tick)}" y2="{bottom}" stroke="#E2E0D8"/>')
        out.append(f'<text x="{x(tick)}" y="{bottom + 18}" fill="{grey}" text-anchor="middle">{points(tick).replace(".00", "")}</text>')
    out.append(f'<line x1="{x(0)}" y1="{top - 6}" x2="{x(0)}" y2="{bottom}" stroke="{grey}" stroke-width="2"/>')
    out.append(f'<line x1="{x(-MARGIN)}" y1="{top - 6}" x2="{x(-MARGIN)}" y2="{bottom}" stroke="{orange}" '
               'stroke-width="2" stroke-dasharray="6 5"/>')
    for i, row in enumerate(rows):
        y = top + row_h * i + row_h // 2
        color = orange if row["rule"] == "non-inferiority" else blue
        out.append(f'<text x="{left - 12}" y="{y + 5}" fill="{ink}" text-anchor="end">{row["tag"]}</text>')
        out.append(f'<line x1="{x(row["low"])}" y1="{y}" x2="{x(row["high"])}" y2="{y}" stroke="{color}" '
                   'stroke-width="4" stroke-linecap="round"/>')
        fill = "#FFFFFF" if row.get("kind") == "replication" else color
        out.append(f'<circle cx="{x(row["delta"])}" cy="{y}" r="6" fill="{fill}" stroke="{color}" stroke-width="2.5"/>')
        out.append(f'<text x="{left + width + 14}" y="{y + 5}" fill="{color}">{points(row["delta"])} '
                   f'[{points(row["low"])}, {points(row["high"])}]</text>')
    out.append(f'<text x="{left}" y="{height - 6}" fill="{grey}">blue: superiority; orange: non-inferiority '
               '(dashed line: margin −2); hollow: re-evaluation</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def splice(text: str, block: str, path: Path) -> str:
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError(f"{path}: needs exactly one {START} and one {END}")
    head, rest = text.split(START)
    _, tail = rest.split(END)
    return f"{head}{START}\n{block}\n{END}{tail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    rows = load()
    check(rows)
    wanted = {path: splice(path.read_text(encoding="utf-8"), table(rows, lang), path) for lang, path in READMES.items()}
    wanted[FIGURE] = figure(rows)
    stale = [path for path, text in wanted.items() if not path.exists() or path.read_text(encoding="utf-8") != text]
    if args.write:
        for path in stale:
            path.write_text(wanted[path], encoding="utf-8")
        print(f"wrote {len(stale)} file(s)")
        return 0
    for path in stale:
        print(f"{path.relative_to(ROOT)} differs from docs/data; run scripts/results_table.py --write", file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
