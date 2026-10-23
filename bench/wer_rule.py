"""Checks the rule "Trit's default makes no more errors than any other
runtime" per model: errors over all 456 clips (the four sets together),
from target/runtimes/results.jsonl (the last result of each engine, CPU
only, full runs).

usage: python bench/wer_rule.py

Trit's default is the engine named for the model (`trit` for Parakeet
Redux): auto precision, the model's own decoding. Prints each model with
Trit's total and the best other runtime's, then the sets where another
runtime has fewer errors (for information: no runtime has the fewest on
every set). Exits 1 when Trit's total is higher on a model.
"""

import json
import sys
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESKTOP / "bench"))
from runtimes import ENGINES

SETS = ["clean", "other", "tech", "tts16"]
MODELS = {
    "parakeet-redux": "trit",
    "parakeet-v3": "trit-parakeet-v3",
    "whisper-base.en": "trit-whisper-base.en",
    "whisper-tiny": "trit-whisper-tiny",
    "moonshine-tiny": "trit-moonshine-tiny",
    "moonshine-base": "trit-moonshine-base",
}


def main() -> int:
    latest = {}
    for line in (DESKTOP / "target" / "runtimes" / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            name = r["engine"]
            if name in ENGINES and ENGINES[name][3] == "cpu" and {s["set"] for s in r["sets"]} >= set(SETS):
                latest[name] = r
    errors = lambda r: {s["set"]: s["errors"] for s in r["sets"]}
    broken = 0
    for model, default in MODELS.items():
        others = [r for name, r in latest.items() if ENGINES[name][0] == model and ENGINES[name][1] != "Trit"]
        trit = latest[default]
        total = sum(errors(trit).values())
        best = min(others, key=lambda r: sum(errors(r).values()))
        best_total = sum(errors(best).values())
        verdict = "ok" if total <= best_total else "HIGHER"
        broken += verdict != "ok"
        print(f"{model:16} Trit {total:4}  best other {best_total:4} ({best['engine']})  {verdict}")
        for s in SETS:
            lower = min(others, key=lambda r: errors(r)[s])
            if errors(lower)[s] < errors(trit)[s]:
                print(f"    {s}: Trit {errors(trit)[s]}, {lower['engine']} {errors(lower)[s]}")
    print(f"{broken} models where Trit's total is higher")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
