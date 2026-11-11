"""What it costs to attach Trit to a program: the dynamicality journey (J13).

usage (from the repository root):
  uv run --no-project --python 3.12 --with numpy --with scipy \\
      python oploop/attach.py <model> <trit.exe> <trit_ffi.dll> [--runs 7]

Prints:
  cold      the time of one `trit transcribe <clip> --format json` (a program with no FFI
            pays it for each call), the engine time inside it, and `trit doctor`;
            then a fresh Python process: import (the library), load, first text.
  rates     what a caller gets for audio that is not 16 kHz mono f32: a 48 kHz and a 44.1 kHz
            WAV file, 48 kHz samples, stereo, and int16-scale samples. A wrong text with no
            error is the worst result: the caller cannot see it.
  size      the bytes of the two binaries and the count of C functions.
"""

import json
import os
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATE = 16000


def first_text(model: str, clip: str) -> None:
    sys.path.insert(0, str(ROOT / "bindings" / "python"))
    t0 = time.perf_counter()
    import trit

    t1 = time.perf_counter()
    engine = trit.Trit(model)
    t2 = time.perf_counter()
    text = engine.transcribe_wav(clip, language="en")
    t3 = time.perf_counter()
    engine.transcribe_wav(clip, language="en")
    t4 = time.perf_counter()
    print(json.dumps({"import": (t1 - t0) * 1000, "load": (t2 - t1) * 1000, "first": (t3 - t2) * 1000,
                      "second": (t4 - t3) * 1000, "text": text}))


def rates(model: str, clip: str, folder: str) -> None:
    sys.path.insert(0, str(ROOT / "bindings" / "python"))
    import numpy as np
    import trit
    from scipy.signal import resample_poly

    with wave.open(clip) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    samples = pcm.astype(np.float32) / 32768.0
    engine = trit.Trit(model)
    right = engine.transcribe(samples, language="en")

    def write(name: str, data, rate: int, channels: int = 1) -> str:
        path = str(Path(folder) / name)
        with wave.open(path, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes())
        return path

    at48 = resample_poly(samples, 3, 1).astype(np.float32)
    at441 = resample_poly(samples, 441, 160).astype(np.float32)
    stereo = np.stack([samples, samples], axis=1).reshape(-1)
    cases = {
        "48 kHz WAV file": lambda: engine.transcribe_wav(write("48k.wav", at48, 48000), language="en"),
        "44.1 kHz WAV file": lambda: engine.transcribe_wav(write("441k.wav", at441, 44100), language="en"),
        "16 kHz stereo WAV file": lambda: engine.transcribe_wav(write("stereo.wav", stereo, RATE, 2), language="en"),
        "48 kHz samples, sample_rate": lambda: engine.transcribe(at48, language="en", sample_rate=48000),
        "44.1 kHz samples, sample_rate": lambda: engine.transcribe(at441, language="en", sample_rate=44100),
        "48 kHz samples, no rate": lambda: engine.transcribe(at48, language="en"),
        "44.1 kHz samples, no rate": lambda: engine.transcribe(at441, language="en"),
        "int16-scale samples": lambda: engine.transcribe(samples * 32768.0, language="en"),
        "int16 numpy array": lambda: engine.transcribe(pcm, language="en"),
    }
    out = {}
    for name, case in cases.items():
        try:
            text = case()
            out[name] = "right text" if text == right else f"WRONG TEXT, no error: {text[:60]!r}"
        except trit.TritError as error:
            out[name] = f"error: {str(error)[:110]}"
    print(json.dumps(out))


def main() -> None:
    args = sys.argv[1:]
    if args[0] == "--first-text":
        first_text(args[1], args[2])
        return
    if args[0] == "--rates":
        rates(args[1], args[2], args[3])
        return
    model, exe, lib = args[0], str(Path(args[1]).resolve()), str(Path(args[2]).resolve())
    runs = int(args[args.index("--runs") + 1]) if "--runs" in args else 7
    clips = sorted((ROOT / "bench" / "clips" / "clean").glob("*.wav"), key=lambda p: p.stat().st_size)
    clip = str(clips[len(clips) // 2])
    with wave.open(clip) as w:
        seconds = w.getnframes() / w.getframerate()
    env = dict(os.environ, TRIT_LIB=lib)

    one_shot, engine_ms, doctor, python = [], [], [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        done = subprocess.run([exe, "transcribe", clip, "--model", model, "--language", "en", "--format", "json"],
                              capture_output=True, text=True, encoding="utf-8")
        one_shot.append((time.perf_counter() - t0) * 1000)
        engine_ms.append(json.loads(done.stdout)["results"][0]["ms"])
        t0 = time.perf_counter()
        subprocess.run([exe, "doctor"], capture_output=True)
        doctor.append((time.perf_counter() - t0) * 1000)
        done = subprocess.run([sys.executable, __file__, "--first-text", model, clip], env=env, capture_output=True,
                              text=True, encoding="utf-8")
        if done.returncode:
            sys.exit(done.stderr[-1500:])
        python.append(json.loads(done.stdout.strip().splitlines()[-1]))
    med = statistics.median
    print(f"clip: {seconds:.1f} s; {runs} runs each, medians")
    print(f"cold: `trit transcribe` one call {med(one_shot):.0f} ms (engine time inside {med(engine_ms):.0f} ms); "
          f"`trit doctor` {med(doctor):.0f} ms")
    print("cold: Python process: import " + f"{med(p['import'] for p in python):.0f} ms, load "
          f"{med(p['load'] for p in python):.0f} ms, first text {med(p['first'] for p in python):.0f} ms, "
          f"second {med(p['second'] for p in python):.0f} ms")

    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        done = subprocess.run([sys.executable, __file__, "--rates", model, clip, folder], env=env, capture_output=True,
                              text=True, encoding="utf-8")
    if done.returncode:
        sys.exit(done.stderr[-1500:])
    print("rates:")
    for name, result in json.loads(done.stdout.strip().splitlines()[-1]).items():
        print(f"  {name:30} {result}")

    header = (ROOT / "include" / "trit.h").read_text(encoding="utf-8")
    import re

    functions = len(set(re.findall(r"\b(trit_\w+)\s*\(", re.sub(r"/\*.*?\*/|//[^\n]*", "", header, flags=re.S))))
    print(f"size: {Path(exe).name} {Path(exe).stat().st_size / 2**20:.1f} MB, {Path(lib).name} "
          f"{Path(lib).stat().st_size / 2**20:.1f} MB, {functions} C functions in include/trit.h")


if __name__ == "__main__":
    main()
