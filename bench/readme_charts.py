"""Draws the README charts from desktop/bench/benchmarks.json (written by
runtimes_report.py): docs/images/errors.svg and docs/images/speed.svg.

usage (from the repository root): python desktop/bench/readme_charts.py

Each chart has one row per speech model with two bars: Trit as it runs by
default, and the best other runtime (fewest errors, or the fastest time per
clip, at any of its settings).
"""

import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "images"
TRIT, OTHER, INK, MUTED, GRID = "#5b6cf0", "#a3a8b8", "#1d2030", "#5d6275", "#e6e8ef"
WIDTH, LABEL, BAR, GAP, ROW = 760, 190, 18, 4, 64


def short(runtime: str) -> str:
    return runtime.replace(" (PyTorch)", "").replace(" (CTranslate2)", "").replace(" (Kestrel, PyTorch)", "")


def chart(title: str, note: str, rows: list[tuple[str, float, str, float]], unit: str) -> str:
    """rows: (model, Trit value, other runtime, its value). Shorter is better."""
    top = 70
    height = top + ROW * len(rows) + 30
    scale = (WIDTH - LABEL - 150) / max(max(t, o) for _, t, _, o in rows)
    out = [(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
            f'viewBox="0 0 {WIDTH} {height}" font-family="Segoe UI, Helvetica, Arial, sans-serif">'),
           f'<rect width="{WIDTH}" height="{height}" rx="12" fill="#ffffff" stroke="{GRID}"/>',
           f'<text x="24" y="34" font-size="18" font-weight="600" fill="{INK}">{escape(title)}</text>',
           f'<text x="24" y="54" font-size="12" fill="{MUTED}">{escape(note)}</text>']
    for i, (model, trit, runtime, other) in enumerate(rows):
        y = top + i * ROW
        out.append(f'<text x="24" y="{y + BAR + 4}" font-size="13" font-weight="600" fill="{INK}">{escape(model)}</text>')
        for j, (name, value, color) in enumerate((("Trit", trit, TRIT), (short(runtime), other, OTHER))):
            by = y + j * (BAR + GAP)
            w = max(2, value * scale)
            out.append(f'<rect x="{LABEL}" y="{by}" width="{w:.1f}" height="{BAR}" rx="3" fill="{color}"/>')
            weight = "600" if j == 0 else "400"
            out.append(f'<text x="{LABEL + w + 8:.1f}" y="{by + 13}" font-size="12" font-weight="{weight}" '
                       f'fill="{INK if j == 0 else MUTED}">{value:g}{unit}  {escape(name)}</text>')
    out.append(f'<text x="24" y="{height - 12}" font-size="11" fill="{MUTED}">Shorter is better.</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    data = json.loads((ROOT / "desktop" / "bench" / "benchmarks.json").read_text(encoding="utf-8"))
    errors, speed = [], []
    for model in data["models"]:
        runs = model["runs"]
        trit = next(r for r in runs if r["default"])
        others = [r for r in runs if not r["trit"]]
        name = model["name"].split(" (")[0].replace(", 1.58-bit", "").replace(" 0.6B", "")
        best = min(others, key=lambda r: r["errors"])
        fastest = min(others, key=lambda r: r["ms"])
        errors.append((name, trit["errors"], best["runtime"], best["errors"]))
        speed.append((name, round(trit["ms"]), fastest["runtime"], round(fastest["ms"])))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "errors.svg").write_text(chart(
        "Word errors on 456 clips",
        f"Trit's default against the most accurate other runtime, CPU, {data['date']}",
        errors, ""), encoding="utf-8")
    (OUT / "speed.svg").write_text(chart(
        "Median time per clip (LibriSpeech test-clean)",
        f"Trit's default against the fastest other runtime at any setting, CPU, {data['date']}",
        speed, " ms"), encoding="utf-8")
    print(f"wrote {OUT / 'errors.svg'} and {OUT / 'speed.svg'}")


if __name__ == "__main__":
    main()
