"""Audio at another rate: the text that Trit gives after its own resample.

usage (from the repository root):
  uv run --no-project --python 3.12 --with numpy --with scipy --with whisper-normalizer==0.1.12 \\
      python bench/rates.py <model> [<trit_ffi library>] [--clips 200] [--rates 48000,44100,8000]

Each clip of `clean` goes to the rate with SciPy (`resample_poly`). Then two ways back:
Trit (`transcribe(samples, sample_rate=rate)`, and a WAV file at that rate for 20 clips), and
SciPy (`resample_poly` to 16 kHz, then `transcribe`). Both ways change the samples a little,
and a model changes a few texts for that, so the measure is the error total.
Pass: for a rate of 16 kHz or more, Trit's errors are at most 3 above SciPy's and at most 5
above the errors of the 16 kHz clips. A rate below 16 kHz (8 kHz holds no audio above 4 kHz)
only prints its numbers. Exit 0 or 1. The filter itself has its own check with tones:
`cargo run --release --example resample_check`.
"""

import os
import sys
import tempfile
import time
import wave
from math import gcd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bindings" / "python"))


def main() -> None:
    args = sys.argv[1:]
    option = lambda name, default: args[args.index(name) + 1] if name in args else default
    positional = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    if len(positional) > 1:
        os.environ["TRIT_LIB"] = str(Path(positional[1]).resolve())
    import numpy as np
    import runtimes
    import trit
    from scipy.signal import resample_poly
    from whisper_normalizer.english import EnglishTextNormalizer

    runtimes.np = np
    from runtimes import edits, read_clip

    normalize = EnglishTextNormalizer()
    engine = trit.Trit(positional[0])
    clips = sorted((ROOT / "bench" / "clips" / "clean").glob("*.wav"))[: int(option("--clips", "200"))]
    rates = [int(r) for r in option("--rates", "48000,44100,8000").split(",")]
    refs = [normalize(p.with_suffix(".txt").read_text(encoding="utf-8-sig")).split() for p in clips]
    audio = [read_clip(p) for p in clips]
    base = [engine.transcribe(a, language="en") for a in audio]
    base_errors = sum(edits(r, normalize(t).split()) for r, t in zip(refs, base))
    seconds = sum(len(a) for a in audio) / 16000
    print(f"16000 Hz: {len(clips)} clips, {seconds:.0f} s, errors {base_errors}")
    failed = False
    with tempfile.TemporaryDirectory() as folder:
        for rate in rates:
            common = gcd(rate, 16000)
            errors = scipy_errors = differ = wav_differ = 0
            busy = at_rate = 0.0
            for i, (ref, samples, text16) in enumerate(zip(refs, audio, base)):
                other = resample_poly(samples, rate // common, 16000 // common).astype(np.float32)
                start = time.perf_counter()
                text = engine.transcribe(other, language="en", sample_rate=rate)
                busy += time.perf_counter() - start
                at_rate += len(other) / rate
                errors += edits(ref, normalize(text).split())
                differ += text != text16
                back = resample_poly(other, 16000 // common, rate // common).astype(np.float32)
                scipy_errors += edits(ref, normalize(engine.transcribe(back, language="en")).split())
                if i < 20:  # the WAV reader takes the same path: a few files are enough
                    path = str(Path(folder) / "clip.wav")
                    with wave.open(path, "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(rate)
                        w.writeframes((np.clip(other, -1, 1) * 32767).astype(np.int16).tobytes())
                    wav_differ += engine.transcribe_wav(path, language="en") != text
            ok = rate < 16000 or (errors <= scipy_errors + 3 and errors <= base_errors + 5)
            failed |= not ok
            print(f"{rate} Hz: errors {errors} (SciPy's way back: {scipy_errors}), texts that differ from 16000 Hz "
                  f"{differ} of {len(clips)}, WAV file against samples {wav_differ} of {min(20, len(clips))} differ, "
                  f"{busy / at_rate * 1000:.2f} ms for each second with the resample: {'ok' if ok else 'FAIL'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
