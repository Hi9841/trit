"""Trit's voices against their reference runtimes. The same text goes
through Trit (`trit speak`) and through the reference, then both through the
same speech model (`trit transcribe`, parakeet-redux), so an error that
Trit adds shows up as extra word errors.

usage:
  python bench/tts.py vits TRIT_EXE MMS_DIR ASR_DIR [--seeds 3]
  python bench/tts.py kokoro TRIT_EXE KOKORO_DIR ASR_DIR [--seeds 3] [--voice af_heart]

Run from the repository root, in an environment with the references. VITS
needs torch and transformers; Kokoro needs the kokoro package (misaki,
spaCy en_core_web_sm). On Windows, spaCy 3.8 fails to load its model; this
works:

  uv run --no-project --python 3.11 --with torch --with kokoro --with scipy --with whisper-normalizer \\
    --with "spacy==3.7.5" --with "preshed==3.0.9" --with "cymem==2.0.8" --with "murmurhash==1.0.10" \\
    --with "thinc==8.2.5" --with "numpy<2" \\
    --with https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl \\
    --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match python bench/tts.py ...

The sentences are bench/corpus.txt and bench/tech-corpus.txt. Clips go to target/tts/.

VITS: both runtimes get the same parts (Trit's sentence cuts), the same
pause, and the same ids: a blank (id 0) between all characters, as MMS was
trained (Transformers' tokenizer drops the blanks around a "k" at the very
start or end of a text). Noise off: the waveforms must be the same. Noise
on: the runtimes draw different noise, so each runs --seeds seeds and the
error totals are compared.

Kokoro: (1) the engine: the reference KModel speaks Trit's phonemes with
Trit's noise (SplitMix64 and Box-Muller, reproduced here), so the two
waveforms must match up to float rounding (PyTorch's own float32 against
float64 is about 39 dB). With the noise off, unvoiced frames feed the STFT
a constant source, and the phases of its empty bins are float residue that
depends on the FFT library; that is not a check. (2) the phonemes: Trit's
phonemizer against misaki with spaCy and espeak, word by word. (3) the round
trip: each full pipeline, text to speech, then transcribed.
"""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from whisper_normalizer.english import EnglishTextNormalizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "target" / "tts"
CORPORA = ["corpus.txt", "tech-corpus.txt"]
# Trit's tts/mod.rs: the pause between VITS sentences, and where a sentence ends.
PAUSE_SECONDS = 0.3
SENTENCE_END = re.compile(r"(?<=[.!?;:])(?=\s|$)")
NORMALIZE = EnglishTextNormalizer()


def sentences() -> list[tuple[str, str]]:
    rows = []
    for name in CORPORA:
        for line in (ROOT / "bench" / name).read_text(encoding="utf-8").splitlines():
            if "|" in line:
                clip, text = line.split("|", 1)
                rows.append((clip, text.strip()))
    return rows


def edits(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref):
        cur = [i + 1] + [0] * len(hyp)
        for j, h in enumerate(hyp):
            cur[j + 1] = min(prev[j] + (r != h), prev[j + 1] + 1, cur[j] + 1)
        prev = cur
    return prev[-1]


def write(path: Path, wave: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, rate, (np.clip(wave, -1, 1) * 32767).round().astype(np.int16))


def read(path: Path) -> np.ndarray:
    return wavfile.read(path)[1].astype(np.float64) / 32767


def to_16k(wave: np.ndarray, rate: int) -> np.ndarray:
    return wave if rate == 16000 else resample_poly(wave, 16000 // math.gcd(16000, rate), rate // math.gcd(16000, rate))


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


def snr(ref: np.ndarray, ours: np.ndarray) -> float:
    if len(ref) != len(ours):
        return float("-inf")
    return 10 * np.log10((ref**2).sum() / max(((ref - ours) ** 2).sum(), 1e-12))


def report_snr(name: str, values: list[tuple[str, float]]) -> None:
    worst = min(values, key=lambda s: s[1])
    finite = [s for _, s in values if s != float("-inf")]
    lengths = sum(1 for _, s in values if s == float("-inf"))
    print(f"{name}_snr_db: min {worst[1]:.1f} ({worst[0]}), median {np.median(finite):.1f}")
    print(f"{name}_length_mismatches: {lengths}")


# ---------- VITS (Meta MMS-TTS) ----------


def vits(trit: str, voice: str, asr: str, seeds: int) -> None:
    from transformers import VitsModel, VitsTokenizer

    rows = sentences()
    tokenizer = VitsTokenizer.from_pretrained(voice)
    model = VitsModel.from_pretrained(voice).eval()

    def parts(text: str) -> list[str]:
        return [p.strip() for line in text.splitlines() for p in SENTENCE_END.split(line) if p.strip()]

    def reference(text: str, noise: float, seed: int) -> np.ndarray:
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
            with torch.no_grad():
                wave = model(torch.tensor([ids])).waveform[0].numpy()
            if pieces:
                pieces.append(pause)
            pieces.append(wave)
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    runs = [("off", 0.0, 0)] + [("on", 1.0, seed) for seed in range(seeds)]
    snrs = []
    print(f"clips: {len(rows)}")
    print("runs{noise,seed,runtime,errors,words,wer}:")
    for noise_name, noise, seed in runs:
        folders = {}
        for runtime in ("reference", "trit"):
            folder = OUT / "vits" / f"{runtime}-noise-{noise_name}-seed-{seed}"
            folders[runtime] = folder
            for clip, text in rows:
                path = folder / f"{clip}.wav"
                if runtime == "reference":
                    write(path, reference(text, noise, seed), 16000)
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
                snrs.append((clip, snr(read(folders["reference"] / f"{clip}.wav"), read(folders["trit"] / f"{clip}.wav"))))
    report_snr("noise_off", snrs)


# ---------- Kokoro ----------

MASK = (1 << 64) - 1


def trit_normals(seed: int, count: int) -> np.ndarray:
    """Trit's noise (tts/vits.rs `Noise`): SplitMix64, Box-Muller, f32 values."""
    state = seed & MASK
    out = np.empty(count, dtype=np.float32)
    spare = None

    def next_u64() -> int:
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    for i in range(count):
        if spare is not None:
            out[i], spare = spare, None
            continue
        radius = math.sqrt(-2.0 * math.log(((next_u64() >> 11) + 1) / float(1 << 53)))
        angle = 2.0 * math.pi * (((next_u64() >> 11) + 1) / float(1 << 53))
        spare = np.float32(radius * math.sin(angle))
        out[i] = np.float32(radius * math.cos(angle))
    return out


def kokoro(trit: str, folder: str, asr: str, seeds: int, voice: str) -> None:
    from kokoro import KModel, KPipeline

    rows = sentences()
    root = Path(folder)
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(root / "config.json"), model=str(next(root.glob("*.pth")))).eval()
    pack_path = str(root / "voices" / f"{voice}.pt")
    pack = torch.load(pack_path, weights_only=True)
    pipeline = KPipeline(lang_code="a" if voice.startswith("a") else "b", repo_id="hexgrad/Kokoro-82M", model=model)
    british = voice.startswith("b")
    print(f"clips: {len(rows)}, voice {voice}")

    # (2) Phonemes: Trit's against the pipeline's (spaCy, espeak for unknown words).
    lines = OUT / "kokoro" / "sentences.txt"
    lines.parent.mkdir(parents=True, exist_ok=True)
    lines.write_text("\n".join(text for _, text in rows), encoding="utf-8")
    command = [trit, "phonemes", "--file", str(lines)] + (["--british"] if british else [])
    ours = subprocess.run(command, capture_output=True, encoding="utf-8", check=True, cwd=Path(folder).resolve().parent.parent).stdout.splitlines()
    differ = total = 0
    for (clip, text), trit_ps in zip(rows, ours):
        ref_ps = " ".join(r.phonemes for r in pipeline(text, voice=pack_path) if r.phonemes)
        a, b = ref_ps.split(), trit_ps.split()
        differ += edits(a, b)
        total += len(a)
    print(f"phoneme_words_differing: {differ} of {total} ({100 * differ / total:.2f} %)")

    # (1) The engine: the same phonemes and the same noise.
    real_randn_like = torch.randn_like
    snrs = []
    for (clip, _), phonemes in zip(rows, ours):
        phonemes = phonemes.strip()[: model.context_length - 2]

        def randn_like(t, **kwargs):
            if t.shape[-1] == 9:  # SineGen: the noise of the nine harmonics, as Trit draws it
                return torch.from_numpy(trit_normals(0, t.numel())).reshape(t.shape).to(t.dtype)
            return torch.zeros_like(t)

        torch.randn_like = randn_like
        with torch.no_grad():
            ref = model(phonemes, pack[len(phonemes) - 1], 1.0).numpy()
        torch.randn_like = real_randn_like
        path = OUT / "kokoro" / "engine" / f"{clip}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([trit, "speak", phonemes, "--phonemes", "--model", folder, "--voice", voice, "--seed", "0", "--out", str(path)],
                       capture_output=True, check=True)
        snrs.append((clip, snr(np.clip(ref.astype(np.float64), -1, 1), read(path))))
    report_snr("engine_same_noise", snrs)

    # (3) The round trip: each full pipeline, text to speech, transcribed.
    print("runs{seed,runtime,errors,words,wer}:")
    for seed in range(seeds):
        folders = {}
        for runtime in ("reference", "trit"):
            out = OUT / "kokoro" / f"{runtime}-seed-{seed}"
            folders[runtime] = out
            for clip, text in rows:
                path = out / f"{clip}.wav"
                if runtime == "reference":
                    torch.manual_seed(seed)
                    audio = [r.audio.numpy() for r in pipeline(text, voice=pack_path) if r.audio is not None]
                    wave = np.concatenate(audio) if audio else np.zeros(0)
                else:
                    raw = OUT / "kokoro" / "raw" / f"{clip}.wav"
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run([trit, "speak", text, "--model", folder, "--voice", voice, "--seed", str(seed), "--out", str(raw)],
                                   capture_output=True, check=True, cwd=Path(folder).resolve().parent.parent)
                    wave = read(raw)
                write(path, to_16k(np.asarray(wave, dtype=np.float64), 24000), 16000)
        for runtime, out in folders.items():
            errs, words = errors(rows, transcribe(trit, asr, out))
            print(f"  {seed},{runtime},{errs},{words},{100 * errs / words:.2f}")


def main() -> None:
    args = sys.argv[1:]
    options = {"--seeds": "3", "--voice": "af_heart"}
    for flag in list(options):
        if flag in args:
            at = args.index(flag)
            options[flag] = args[at + 1]
            del args[at : at + 2]
    if len(args) != 4 or args[0] not in ("vits", "kokoro"):
        sys.exit(__doc__)
    mode, trit, voice, asr = args
    if mode == "vits":
        vits(trit, voice, asr, int(options["--seeds"]))
    else:
        kokoro(trit, voice, asr, int(options["--seeds"]), options["--voice"])


if __name__ == "__main__":
    main()
