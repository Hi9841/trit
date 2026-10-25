"""Trit's voice against the reference runtime: the same text through Trit
(`trit speak`) and through Transformers `VitsModel`, then both through the
same speech model (`trit transcribe`, parakeet-redux), so any error that
Trit's engine adds shows up as extra word errors.

usage: uv run --no-project --python 3.12 --with torch --with transformers --with numpy --with scipy --with whisper-normalizer --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match python bench/tts.py TRIT_EXE VOICE_DIR ASR_DIR [--seeds 3]

Run from desktop/. The sentences are bench/corpus.txt and bench/tech-corpus.txt.
Both runtimes get the same parts (Trit's sentence cuts), the same pause, and
the same ids: a blank (id 0) between all characters, as MMS was trained.
(Transformers' tokenizer takes the blank "k" for a special token and drops
the blanks around a "k" at the very start or end of a text.)
Noise off: the two waveforms must be the same (SNR per clip is printed).
Noise on: the two runtimes draw different noise, so each runs --seeds seeds
and the error totals are compared. Clips go to target/tts/.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from transformers import VitsModel, VitsTokenizer
from whisper_normalizer.english import EnglishTextNormalizer

DESKTOP = Path(__file__).resolve().parents[1]
OUT = DESKTOP / "target" / "tts"
CORPORA = ["corpus.txt", "tech-corpus.txt"]
# Trit's speech.rs: the pause between sentences, and where a sentence ends.
PAUSE_SECONDS = 0.3
SENTENCE_END = re.compile(r"(?<=[.!?;:])(?=\s|$)")
NORMALIZE = EnglishTextNormalizer()


def sentences() -> list[tuple[str, str]]:
    rows = []
    for name in CORPORA:
        for line in (DESKTOP / "bench" / name).read_text(encoding="utf-8").splitlines():
            if "|" in line:
                clip, text = line.split("|", 1)
                rows.append((clip, text.strip()))
    return rows


def parts(text: str) -> list[str]:
    return [p.strip() for line in text.splitlines() for p in SENTENCE_END.split(line) if p.strip()]


def edits(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref):
        cur = [i + 1] + [0] * len(hyp)
        for j, h in enumerate(hyp):
            cur[j + 1] = min(prev[j] + (r != h), prev[j + 1] + 1, cur[j] + 1)
        prev = cur
    return prev[-1]


def reference(model, tokenizer, text: str, noise: float, seed: int) -> np.ndarray:
    model.noise_scale = noise * 0.667
    model.noise_scale_duration = noise * 0.8
    torch.manual_seed(seed)
    pause = np.zeros(int(PAUSE_SECONDS * 16000), dtype=np.float32)
    pieces = []
    for part in parts(text):
        chars = tokenizer.prepare_for_tokenization(part)[0]
        if not chars:
            continue
        ids = [0] * (2 * len(chars) + 1)
        ids[1::2] = [tokenizer.encoder[c] for c in chars]
        ids = torch.tensor([ids])
        with torch.no_grad():
            wave = model(ids).waveform[0].numpy()
        if pieces:
            pieces.append(pause)
        pieces.append(wave)
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def write(path: Path, wave: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, 16000, (np.clip(wave, -1, 1) * 32767).round().astype(np.int16))


def transcribe(trit: str, asr: str, folder: Path) -> dict[str, str]:
    out = subprocess.run(
        [trit, "transcribe", str(folder), "--model", asr, "--language", "en", "--format", "json"],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return {Path(r["file"]).stem: r["text"] for r in json.loads(out.stdout)["results"]}


def errors(rows, texts: dict[str, str]) -> tuple[int, int]:
    total = words = 0
    for clip, text in rows:
        ref = NORMALIZE(text).split()
        total += edits(ref, NORMALIZE(texts.get(clip, "")).split())
        words += len(ref)
    return total, words


def main() -> None:
    args = sys.argv[1:]
    seeds = 3
    if "--seeds" in args:
        at = args.index("--seeds")
        seeds = int(args[at + 1])
        del args[at : at + 2]
    if len(args) != 3:
        sys.exit(__doc__)
    trit, voice, asr = args
    rows = sentences()
    tokenizer = VitsTokenizer.from_pretrained(voice)
    model = VitsModel.from_pretrained(voice).eval()
    runs = [("off", 0.0, 0)] + [("on", 1.0, seed) for seed in range(seeds)]
    snrs = []
    print(f"clips: {len(rows)}")
    print("runs{noise,seed,runtime,errors,words,wer}:")
    for noise_name, noise, seed in runs:
        folders = {}
        for runtime in ("reference", "trit"):
            folder = OUT / f"{runtime}-noise-{noise_name}-seed-{seed}"
            folders[runtime] = folder
            for clip, text in rows:
                path = folder / f"{clip}.wav"
                if runtime == "reference":
                    write(path, reference(model, tokenizer, text, noise, seed))
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        [trit, "speak", text, "--model", voice, "--out", str(path), "--seed", str(seed),
                         "--noise", str(noise * 0.667), "--noise-duration", str(noise * 0.8)],
                        capture_output=True, check=True,
                    )
        for runtime, folder in folders.items():
            total, words = errors(rows, transcribe(trit, asr, folder))
            print(f"  {noise_name},{seed},{runtime},{total},{words},{100 * total / words:.2f}")
        if noise_name == "off":
            for clip, _ in rows:
                ref = wavfile.read(folders["reference"] / f"{clip}.wav")[1].astype(np.float64)
                ours = wavfile.read(folders["trit"] / f"{clip}.wav")[1].astype(np.float64)
                if len(ref) != len(ours):
                    snrs.append((clip, float("-inf")))
                    continue
                noise_power = max(((ref - ours) ** 2).sum(), 1e-9)
                snrs.append((clip, 10 * np.log10((ref**2).sum() / noise_power)))
    worst = min(snrs, key=lambda s: s[1])
    lengths = sum(1 for _, s in snrs if s == float("-inf"))
    print(f"noise_off_snr_db: min {worst[1]:.1f} ({worst[0]}), median {np.median([s for _, s in snrs]):.1f}")
    print(f"noise_off_length_mismatches: {lengths}")


if __name__ == "__main__":
    main()
