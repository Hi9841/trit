"""Trit and transcribe.cpp (the engine of Hex) on the same GGUF files.

Both engines read the very same file: the same model, the same quantization
(Q8_0, F16, or F32 as stored), on the same 456 clips, with 8 threads each.
transcribe.cpp is built as Hex 3.2.0 builds its Windows release
(bench/gguf/.cargo/config.toml) and called as Hex calls it. Trit runs as
TDT runs it, with the file's stored types (`Precision::Auto`).

usage:
  uv run --no-project --python 3.12 --with whisper-normalizer python bench/gguf.py run <models dir> [--rounds 3] [--only unified-q8_0,...]
  uv run --no-project --python 3.12 --with whisper-normalizer python bench/gguf.py report

<models dir> holds one folder per file, `gguf-<model>-<quant>`, with the one
`.gguf` file of that name from Hugging Face (handy-computer/parakeet-*-gguf).
Round 1 runs all 456 clips (errors, speed, memory); each later round runs
the 200 clean clips again for the time only. Results go to target/gguf/;
`report` writes bench/gguf-results.json for the site and prints the table:
the median clean clip is the median of the rounds, with the lowest and
highest round. Build first: `cargo build --release` in bench/gguf.
Run with nothing else on the computer.
"""

import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "bench" / "clips"
OUT = ROOT / "target" / "gguf"
BIN = ROOT / "bench" / "gguf" / "target" / "release"
SETS = ["clean", "other", "tech", "tts16"]
THREADS = "8"

MODELS = {
    "unified": ("Parakeet Unified English 0.6B (RNNT)", "handy-computer/parakeet-unified-en-0.6b-gguf"),
    "v2": ("Parakeet TDT 0.6B v2", "handy-computer/parakeet-tdt-0.6b-v2-gguf"),
    "v3": ("Parakeet TDT 0.6B v3", "handy-computer/parakeet-tdt-0.6b-v3-gguf"),
    "ultra": ("Parakeet Ultra 0.6B", "handy-computer/parakeet-ultra-gguf"),
}
QUANTS = ["q8_0", "f16", "f32"]

# name: (binary, runtime, instruction set, extra environment)
ENGINES = {
    "trit": ("trit", "Trit", "AVX-512 VNNI", {}),
    "trit-avx2": ("trit", "Trit", "AVX2", {"TRIT_ISA": "avx2"}),
    "transcribe.cpp": ("transcribe_cpp", "transcribe.cpp (Hex 3.2.0 build)", "AVX2", {"HEXENG_THREADS": THREADS}),
}


def run(models_dir: Path, only: set[str] | None, rounds: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for round_ in range(1, rounds + 1):
        run_round(models_dir, only, round_)


def run_round(models_dir: Path, only: set[str] | None, round_: int) -> None:
    sets = SETS if round_ == 1 else ["clean"]
    for model in MODELS:
        for quant in QUANTS:
            key = f"{model}-{quant}"
            folder = models_dir / f"gguf-{key}"
            if (only and key not in only) or not any(folder.glob("*.gguf")):
                continue
            # Engines alternate per file, so a slow minute of the computer
            # does not land on one engine only.
            for engine, (binary, _, _, env) in ENGINES.items():
                out = OUT / f"{key}-{engine}-r{round_}.jsonl"
                if out.exists() and '"run_ws_mb"' in out.read_text(encoding="utf-8"):
                    print(f"have {out.name}")
                    continue
                print(f"run  {key} {engine} round {round_}", flush=True)
                subprocess.run(
                    [str(BIN / f"{binary}.exe" if os.name == "nt" else BIN / binary), str(folder), str(CLIPS), str(out), *sets],
                    env={**os.environ, **env},
                    check=True,
                )


def edits(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref):
        cur = [i + 1] + [0] * len(hyp)
        for j, h in enumerate(hyp):
            cur[j + 1] = min(prev[j] + (r != h), prev[j + 1] + 1, cur[j] + 1)
        prev = cur
    return prev[-1]


def report() -> None:
    from whisper_normalizer.english import EnglishTextNormalizer

    normalize = EnglishTextNormalizer()
    words = lambda text: normalize(text).split()
    refs = {}
    for s in SETS:
        for txt in (CLIPS / s).glob("*.txt"):
            refs[(s, txt.stem)] = words(txt.read_text(encoding="utf-8-sig"))
    rows, texts = [], {}
    for model, (name, repo) in MODELS.items():
        for quant in QUANTS:
            for engine, (_, runtime, isa, _) in ENGINES.items():
                path = OUT / f"{model}-{quant}-{engine}-r1.jsonl"
                if not path.exists():
                    continue
                round_ms = []
                for later in sorted(OUT.glob(f"{model}-{quant}-{engine}-r*.jsonl")):
                    timed = [json.loads(l) for l in later.read_text(encoding="utf-8").splitlines()]
                    if "run_ws_mb" in timed[-1]:
                        round_ms.append(statistics.median(t["ms"] for t in timed if t.get("set") == "clean"))
                lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
                memory = lines[-1] if "run_ws_mb" in lines[-1] else None
                clips = [l for l in lines if "clip" in l]
                if memory is None or len(clips) != len(refs):
                    print(f"skip {path.name}: incomplete")
                    continue
                per = {s: [0, 0] for s in SETS}
                for c in clips:
                    r = refs[(c["set"], c["clip"])]
                    per[c["set"]][0] += edits(r, words(c["hyp"]))
                    per[c["set"]][1] += len(r)
                texts[(model, quant, engine)] = {(c["set"], c["clip"]): c["hyp"] for c in clips}
                total_ms = sum(c["ms"] for c in clips)
                audio = sum(c["audio_s"] for c in clips)
                errors = sum(e for e, _ in per.values())
                total_words = sum(w for _, w in per.values())
                rows.append({
                    "model": model, "name": name, "repo": repo, "quant": quant,
                    "engine": engine, "runtime": runtime, "isa": isa,
                    "errors": errors, "wer": round(100 * errors / total_words, 2),
                    **{f"{s}_wer": round(100 * per[s][0] / per[s][1], 2) for s in SETS},
                    "clean_ms": round(statistics.median(round_ms), 1),
                    "clean_ms_lo": round(min(round_ms), 1),
                    "clean_ms_hi": round(max(round_ms), 1),
                    "rounds": len(round_ms),
                    "total_s": round(total_ms / 1000, 1),
                    "speed": round(audio * 1000 / total_ms),
                    "load_ms": memory["load_ms"],
                    "run_ws_mb": memory["run_ws_mb"], "run_private_mb": memory["run_private_mb"],
                })
    # How often the two engines write the same text from the same file.
    for row in rows:
        mine = texts[(row["model"], row["quant"], row["engine"])]
        other = texts.get((row["model"], row["quant"], "transcribe.cpp"))
        if other and row["engine"] != "transcribe.cpp":
            row["same_text_as_transcribe_cpp"] = sum(mine[k] == other[k] for k in mine)
    result = {
        "date": __import__("datetime").date.today().isoformat(),
        "cpu": platform.processor() or "unknown",
        "clips": len(refs),
        "threads": int(THREADS),
        "rows": rows,
    }
    (ROOT / "bench" / "gguf-results.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    table = [
        "| Model | Quant | Engine | ISA | Errors | Clean WER | Other WER | Median clean clip | Speed | Working set | Same text |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        same = r.get("same_text_as_transcribe_cpp", "")
        table.append(
            f"| {r['name']} | {r['quant'].upper()} | {r['runtime']} | {r['isa']} | {r['errors']} | {r['clean_wer']:.2f} % | "
            f"{r['other_wer']:.2f} % | {r['clean_ms']:.0f} ms ({r['clean_ms_lo']:.0f}-{r['clean_ms_hi']:.0f}) | "
            f"{r['speed']}x | {r['run_ws_mb']} MB | {same} |"
        )
    print("\n".join(table))
    # The table of section 7 of the report, between its markers.
    doc = ROOT / "docs" / "engine-comparison.md"
    text = doc.read_text(encoding="utf-8")
    start, end = "<!-- gguf:start -->", "<!-- gguf:end -->"
    if start in text and end in text:
        head, rest = text.split(start, 1)
        doc.write_text(head + start + "\n" + "\n".join(table) + "\n" + end + rest.split(end, 1)[1], encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        only = None
        if "--only" in sys.argv:
            only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
        rounds = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 1
        run(Path(sys.argv[2]), only, rounds)
    elif len(sys.argv) == 2 and sys.argv[1] == "report":
        report()
    else:
        print(__doc__)
        sys.exit(2)
