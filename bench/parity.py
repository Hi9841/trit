"""Trit against transcribe.cpp on one GGUF file: does Trit run the model right?

Both engines (bench/gguf) transcribe the same clips from the same file. A set
with reference transcripts (`<clip>.txt`) gets each engine's word errors. Every
set gets, per clip, the character difference between the two engines' texts
(CER between engines, after lowercasing and dropping punctuation and spaces)
and the count of identical texts. Close texts and equal errors mean Trit runs
the model as transcribe.cpp does; transcribe.cpp's files are checked against
the upstream model by their publisher (handy-computer).

usage:
  uv run --no-project --python 3.12 --with whisper-normalizer python bench/parity.py <model folder> <set>... [--lang <code>|auto] [--precision <p>]

<set> is a folder under bench/clips: parity-en (66 English clips) or
native-<lang> (FLEURS clips from transcribe.cpp's samples/, no transcripts).
Build bench/gguf first. Run it through a low-load wrapper on a computer that
crashes under full load.
"""

import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "bench" / "clips"
OUT = ROOT / "target" / "parity"
BIN = ROOT / "bench" / "gguf" / "target" / "release"


def edits(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref):
        cur = [i + 1] + [0] * len(hyp)
        for j, h in enumerate(hyp):
            cur[j + 1] = min(prev[j] + (r != h), prev[j + 1] + 1, cur[j] + 1)
        prev = cur
    return prev[-1]


def chars(text: str) -> list[str]:
    """Letters and digits only, lowercased: the text without its formatting."""
    text = unicodedata.normalize("NFKC", text).lower()
    return [c for c in text if c.isalnum()]


def run(engine: str, model: Path, sets: list[str], lang: str, precision: str) -> dict:
    out = OUT / f"{model.name}-{engine}-{precision}.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TRIT_LANGUAGE": lang, "HEXENG_LANGUAGE": lang, "HEXENG_THREADS": "8"}
    if engine == "trit":
        env["TRIT_PRECISION"] = precision
    exe = BIN / (f"{engine}.exe" if os.name == "nt" else engine)
    subprocess.run([str(exe), str(model), str(CLIPS), str(out), *sets], env=env, check=True)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return {(r["set"], r["clip"]): r["hyp"] for r in rows if "clip" in r}


def main() -> None:
    # Transcripts in any script print on a Windows console too.
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if len(args) < 2 or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(2)
    option = lambda name, default: args[args.index(name) + 1] if name in args else default
    lang, precision = option("--lang", "en"), option("--precision", "auto")
    plain = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    model, sets = Path(plain[0]), plain[1:]
    trit = run("trit", model, sets, lang, precision)
    them = run("transcribe_cpp", model, sets, lang, precision)
    from whisper_normalizer.english import EnglishTextNormalizer

    normalize = EnglishTextNormalizer()
    for s in sets:
        keys = sorted(k for k in trit if k[0] == s)
        same = sum(trit[k] == them[k] for k in keys)
        diff = sum(edits(chars(them[k]), chars(trit[k])) for k in keys)
        total = sum(len(chars(them[k])) for k in keys) or 1
        line = f"{s}: {len(keys)} clips, same text {same}, CER between engines {100 * diff / total:.2f} %"
        refs = {k: CLIPS / s / f"{k[1]}.txt" for k in keys}
        if all(p.exists() for p in refs.values()):
            ref = {k: normalize(p.read_text(encoding="utf-8-sig")).split() for k, p in refs.items()}
            words = sum(len(r) for r in ref.values())
            et = sum(edits(ref[k], normalize(trit[k]).split()) for k in keys)
            eh = sum(edits(ref[k], normalize(them[k]).split()) for k in keys)
            line += f"; errors Trit {et}, transcribe.cpp {eh} of {words} words"
        print(line)
        shown = 0
        for k in keys:
            if trit[k] != them[k] and shown < 3:
                print(f"  {k[1]}\n    trit:           {trit[k][:160]}\n    transcribe.cpp: {them[k][:160]}")
                shown += 1


if __name__ == "__main__":
    main()
