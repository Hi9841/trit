"""Errors per model and runtime from target/runtimes/results.jsonl, as text.

usage: python bench/runtimes_table.py [MODEL ...]

The last result of each engine counts. For each model: errors on each set,
the total over all clips, WER of the total, and median ms on test-clean.
A `*` marks the lowest count in a column.
"""

import json
import sys
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[1]
SETS = ["clean", "other", "tech", "tts16"]


def main():
    latest = {}
    for line in (DESKTOP / "target" / "runtimes" / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if all(any(s["set"] == name for s in r["sets"]) for name in SETS):
                latest[r["engine"]] = r
    models = sys.argv[1:] or list(dict.fromkeys(r["model"] for r in latest.values()))
    for model in models:
        rows = [r for r in latest.values() if r["model"] == model]
        if not rows:
            continue
        def counts(r):
            by = {s["set"]: s for s in r["sets"]}
            errors = [by[n]["errors"] for n in SETS]
            words = sum(by[n]["words"] for n in SETS)
            return errors, sum(errors), words, by["clean"]["median_ms"]
        table = [(r, *counts(r)) for r in rows]
        best = [min(t[1][i] for t in table) for i in range(len(SETS))]
        best_total = min(t[2] for t in table)
        print(f"\n{model}")
        print(f"  {'engine':28} {'clean':>7} {'other':>7} {'tech':>6} {'tts16':>6} {'total':>7} {'WER':>7} {'ms':>6}")
        for r, errors, total, words, ms in sorted(table, key=lambda t: t[2]):
            cells = [f"{e}{'*' if e == b else ' '}" for e, b in zip(errors, best)]
            mark = "*" if total == best_total else " "
            print(f"  {r['engine']:28} {cells[0]:>7} {cells[1]:>7} {cells[2]:>6} {cells[3]:>6} "
                  f"{str(total) + mark:>7} {100 * total / words:6.2f}% {ms:6.0f}")


if __name__ == "__main__":
    main()
