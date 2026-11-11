"""How steady the time and the text of one model are: the stability journeys.

usage (from the repository root):
  uv run --no-project --python 3.12 --with numpy --with scipy --with whisper-normalizer==0.1.12 \\
      python oploop/steady.py <model> <A trit_ffi.dll> [<B trit_ffi.dll>] [--tests repeat,gap,length,delivery,load]
      [--procs 3] [--loads 0,8,16] [--save FILE]

Each process of a side loads the model (TRIT_LIB picks the library), runs one warm-up
clip, then the tests. With two libraries the processes run in turns (A, B, A, B).

  repeat    one clip 40 times with no pause, for a short, a middle, and a long clip:
            median, p95, and max, and the spread p95 / median.
  gap       the middle clip after a pause of 0.2, 1, and 3 s, and 3 s after release_memory
            (what a dictation app does): the cost of the first call after an idle time.
  length    speech cut to 1 .. 90 s: ms and ms for each second of audio (steps show there).
  delivery  20 clips, each said "another way" with the same words: quieter, louder, with a
            pause before and after, with noise, moved by 5 ms, 10 % faster and slower.
            Errors against the reference, texts that differ from the plain clip, and time.
  load      the short and the middle clip while N other threads are busy (--loads), with
            0.5 s between the calls. Do not use more than 16: the computer of the ledger
            is not stable with every processor busy.
"""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
RATE = 16000


def clips_by_length(np, read_clip, count: int = 400):
    paths = sorted((ROOT / "bench" / "clips" / "clean").glob("*.wav"))[:count]
    clips = [(p, read_clip(p)) for p in paths]
    return sorted(clips, key=lambda c: len(c[1]))


def variants(np, samples):
    """The same words, said another way. Each value is 16 kHz mono f32."""
    from scipy.signal import resample_poly

    rng = np.random.default_rng(7)
    rms = float(np.sqrt(np.mean(samples**2))) or 1e-4
    hiss = lambda n, level: (rng.standard_normal(n) * level).astype(np.float32)
    zeros = lambda seconds: np.zeros(int(seconds * RATE), np.float32)
    return {
        "plain": samples,
        "quiet (-20 dB)": samples * 0.1,
        "loud (+10 dB, clipped)": np.clip(samples * 3.16, -1, 1),
        "pause 1 s before": np.concatenate([zeros(1), samples]),
        "pause 2 s after": np.concatenate([samples, zeros(2)]),
        "room noise 1 s before, 2 s after": np.concatenate([hiss(RATE, 0.002), samples, hiss(2 * RATE, 0.002)]),
        "noise at 20 dB SNR": samples + hiss(len(samples), rms / 10),
        "moved by 5 ms": samples[80:],
        "10 % faster": resample_poly(samples, 10, 11).astype(np.float32),
        "10 % slower": resample_poly(samples, 11, 10).astype(np.float32),
    }


def side(model: str, tests: list[str]) -> None:
    sys.path.insert(0, str(ROOT / "bindings" / "python"))
    import numpy as np
    import runtimes

    runtimes.np = np
    from runtimes import edits, read_clip
    from whisper_normalizer.english import EnglishTextNormalizer

    t0 = time.perf_counter()
    import trit

    engine = trit.Trit(model)
    load_ms = (time.perf_counter() - t0) * 1000
    normalize = EnglishTextNormalizer()
    ordered = clips_by_length(np, read_clip)
    pick = lambda q: ordered[int(q * (len(ordered) - 1))]
    short, middle, long = pick(0.05), pick(0.5), pick(0.97)

    def call(samples) -> tuple[float, str]:
        start = time.perf_counter()
        text = engine.transcribe(samples, language="en")
        return (time.perf_counter() - start) * 1000, text

    first_ms, _ = call(middle[1])
    second_ms, _ = call(middle[1])
    out = {"load_ms": load_ms, "first_ms": first_ms, "second_ms": second_ms,
           "seconds": {"short": len(short[1]) / RATE, "middle": len(middle[1]) / RATE, "long": len(long[1]) / RATE}}

    if "repeat" in tests:
        out["repeat"] = {}
        for name, (_, samples) in (("short", short), ("middle", middle), ("long", long)):
            rows = [call(samples) for _ in range(40)]
            out["repeat"][name] = {"ms": [r[0] for r in rows], "texts": len({r[1] for r in rows})}

    if "gap" in tests:
        out["gap"] = {}
        for name, pause, release in (("0.2 s", 0.2, False), ("1 s", 1, False), ("3 s", 3, False), ("3 s, released", 3, True)):
            times = []
            for _ in range(8):
                if release:
                    trit.release_memory()
                time.sleep(pause)
                times.append(call(middle[1])[0])
            out["gap"][name] = times

    if "length" in tests:
        speech = np.concatenate([c[1] for c in ordered[-12:]])
        out["length"] = {}
        for seconds in (1, 2, 3, 5, 8, 12, 16, 20, 25, 29, 31, 40, 55, 59, 61, 75, 90):
            part = speech[: seconds * RATE]
            call(part)
            out["length"][seconds] = min(call(part)[0] for _ in range(3))

    if "delivery" in tests:
        chosen = ordered[40:400:18][:20]
        rows = {}
        for path, samples in chosen:
            ref = normalize(path.with_suffix(".txt").read_text(encoding="utf-8-sig")).split()
            plain = None
            for name, audio in variants(np, samples).items():
                call(audio)
                ms, text = call(audio)
                words = normalize(text).split()
                plain = words if name == "plain" else plain
                row = rows.setdefault(name, {"errors": 0, "differ": 0, "ms": [], "seconds": 0.0})
                row["errors"] += edits(ref, words)
                row["differ"] += int(words != plain)
                row["ms"].append(ms)
                row["seconds"] += len(audio) / RATE
        out["delivery"] = rows

    if "load" in tests:
        out["load"] = {}
        for name, (_, samples) in (("short", short), ("middle", middle)):
            times = []
            for _ in range(12):
                time.sleep(0.5)
                times.append(call(samples)[0])
            out["load"][name] = times

    print(json.dumps(out))


def run(lib: str, model: str, tests: list[str], busy: int) -> dict:
    spinners = [subprocess.Popen([sys.executable, "-I", "-c", "while True: pass"]) for _ in range(busy)]
    try:
        if busy:
            time.sleep(1.0)
        env = dict(os.environ, TRIT_LIB=str(Path(lib).resolve()))
        done = subprocess.run([sys.executable, __file__, "--side", model, ",".join(tests)], cwd=ROOT, env=env,
                              capture_output=True, text=True, encoding="utf-8")
    finally:
        for spinner in spinners:
            spinner.kill()
    if done.returncode:
        sys.exit(f"side failed ({lib}):\n{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def spread(values: list[float]) -> str:
    ordered = sorted(values)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return f"{p50:7.1f} {p95:7.1f} {ordered[-1]:7.1f}  {p95 / p50:5.2f} {ordered[-1] / p50:5.2f}"


def main() -> None:
    args = sys.argv[1:]
    if args[0] == "--side":
        side(args[1], args[2].split(","))
        return
    option = lambda name, default: args[args.index(name) + 1] if name in args else default
    positional = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    model, libs = positional[0], positional[1:3]
    tests = option("--tests", "repeat,gap,length,delivery").split(",")
    procs = int(option("--procs", "3"))
    loads = [int(n) for n in option("--loads", "0,8,16").split(",")]
    names = ["A", "B"][: len(libs)]
    quiet = [t for t in tests if t != "load"]
    runs = {name: [] for name in names}
    if quiet:
        for _ in range(procs):
            for name, lib in zip(names, libs):
                runs[name].append(run(lib, model, quiet, 0))
    pooled = lambda name, get: [x for r in runs[name] for x in get(r)]
    head = "                                        p50     p95     max   p95/p50 max/p50"
    for name in names if quiet else []:
        first = runs[name]
        print(f"== {name}: {libs[names.index(name)]}  ({procs} processes)")
        print(f"load {statistics.median(r['load_ms'] for r in first):.0f} ms; first call {statistics.median(r['first_ms'] for r in first):.1f} ms, "
              f"second {statistics.median(r['second_ms'] for r in first):.1f} ms (middle clip, {first[0]['seconds']['middle']:.1f} s)")
        if "repeat" in tests:
            print("repeat, 40 calls in sequence" + head[28:])
            for clip in ("short", "middle", "long"):
                label = f"  {clip} ({first[0]['seconds'][clip]:.1f} s)"
                texts = max(r["repeat"][clip]["texts"] for r in first)
                print(f"{label:38}{spread(pooled(name, lambda r: r['repeat'][clip]['ms']))}  texts {texts}")
        if "gap" in tests:
            print("gap before each call (middle clip)" + head[34:])
            for gap in first[0]["gap"]:
                print(f"  {gap:36}{spread(pooled(name, lambda r: r['gap'][gap]))}")
        if "length" in tests:
            print("length: seconds of audio, ms (best of 3), ms for each second")
            for seconds in first[0]["length"]:
                ms = statistics.median(r["length"][seconds] for r in first)
                print(f"  {int(seconds):3d} s {ms:8.1f} ms {ms / int(seconds):6.2f}")
        if "delivery" in tests:
            print("delivery (20 clips): errors, texts that differ from plain, median ms, ms for each second")
            for variant in first[0]["delivery"]:
                row = first[0]["delivery"][variant]
                ms = pooled(name, lambda r: r["delivery"][variant]["ms"])
                per = sum(ms) / (row["seconds"] * len(first))
                print(f"  {variant:34} {row['errors']:4d} {row['differ']:3d} {statistics.median(ms):8.1f} {per:6.2f}")
    if "load" in tests:
        results = {name: {} for name in names}
        for busy in loads:
            for _ in range(procs):
                for name, lib in zip(names, libs):
                    results[name].setdefault(busy, []).append(run(lib, model, ["load"], busy))
        for name in names:
            print(f"== {name}: load, 12 calls for each process with 0.5 s between them" + head[62:])
            for busy, rows in results[name].items():
                for clip in ("short", "middle"):
                    values = [x for r in rows for x in r["load"][clip]]
                    print(f"  {busy:2d} busy threads, {clip:7}             {spread(values)}")
        runs = {name: {"quiet": runs[name], "load": results[name]} for name in names}
    if "--save" in args:
        Path(option("--save", "")).write_text(json.dumps(runs), encoding="utf-8")


if __name__ == "__main__":
    main()
