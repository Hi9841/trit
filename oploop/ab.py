"""Interleaved A/B of two Trit libraries on one model: time per clip and errors.

usage (from desktop/):
  uv run --no-project --python 3.12 --with numpy --with whisper-normalizer --with scipy \\
      python ../oploop/ab.py <model> <A trit_ffi.dll> <B trit_ffi.dll> [--rounds 3] [--sets clean]
      [--cleanup]

Each round runs side A, then side B, each in a fresh process (TRIT_LIB picks the
library), so warm-up and drift favor neither. A side loads the model, runs one
warm-up clip, then transcribes every clip of the sets, one call each, timed.
Prints per round: the median ms, the total errors, and the texts. Then the
verdict: median of the per-round medians, their range, the median of the
per-clip B/A ratios, a rank-sum test on the per-clip times of all rounds,
the error totals, and how many texts differ between A and B.

--cleanup: <model> is ignored; times S1-mini on the cleanup harness's cases.
"""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[1] / "desktop"
sys.path.insert(0, str(DESKTOP / "bench"))


def side(model: str, sets: list[str], cleanup: bool) -> None:
    """One side, in its own process: prints one JSON line."""
    sys.path.insert(0, str(DESKTOP.parent / "trit" / "bindings" / "python"))
    import trit

    if cleanup:
        from cleanup_runtimes import cases

        cleaner = trit.Cleaner(str(DESKTOP.parent / "models" / "s1-mini"))
        items = cases()
        cleaner.clean(items[0][0])
        out = []
        for raw, target in items:
            t0 = time.perf_counter()
            text = cleaner.clean(raw)
            out.append({"clip": raw[:40], "ms": (time.perf_counter() - t0) * 1000, "hyp": text,
                        "errors": int(text.strip() != target.strip())})
        print(json.dumps(out))
        return

    import numpy
    import runtimes

    runtimes.np = numpy  # runtimes.py imports numpy only in its own child processes
    from runtimes import edits, read_clip
    from whisper_normalizer.english import EnglishTextNormalizer

    normalize = EnglishTextNormalizer()
    engine = trit.Trit(model)
    clips = [p for s in sets for p in sorted((DESKTOP / "target" / "accuracy" / s).glob("*.wav"))]
    engine.transcribe(read_clip(clips[0]), language="en")
    out = []
    for path in clips:
        samples = read_clip(path)
        t0 = time.perf_counter()
        text = engine.transcribe(samples, language="en")
        ms = (time.perf_counter() - t0) * 1000
        ref = path.with_suffix(".txt").read_text(encoding="utf-8-sig")
        out.append({"clip": f"{path.parent.name}/{path.stem}", "ms": ms, "hyp": text,
                    "errors": edits(normalize(ref).split(), normalize(text).split())})
    print(json.dumps(out))


def run(lib: str, model: str, sets: list[str], cleanup: bool) -> list[dict]:
    env = dict(os.environ, TRIT_LIB=lib)
    command = [sys.executable, __file__, "--side", model, ",".join(sets)] + (["--cleanup"] if cleanup else [])
    result = subprocess.run(command, cwd=DESKTOP, env=env, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        sys.exit(f"side failed ({lib}):\n{result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    args = sys.argv[1:]
    if args[0] == "--side":
        side(args[1], args[2].split(","), "--cleanup" in args)
        return
    model, a, b = args[:3]
    rounds = int(args[args.index("--rounds") + 1]) if "--rounds" in args else 3
    sets = args[args.index("--sets") + 1].split(",") if "--sets" in args else ["clean"]
    cleanup = "--cleanup" in args
    runs = {"A": [], "B": []}
    for r in range(rounds):
        for name, lib in (("A", a), ("B", b)):
            rows = run(lib, model, sets, cleanup)
            runs[name].append(rows)
            print(f"round {r + 1} {name}: median {statistics.median(x['ms'] for x in rows):7.1f} ms, "
                  f"errors {sum(x['errors'] for x in rows)}", flush=True)
    med = {k: [statistics.median(x["ms"] for x in rows) for rows in v] for k, v in runs.items()}
    # Per clip: the median over rounds, then B/A.
    clips = len(runs["A"][0])
    per_clip = {k: [statistics.median(rows[i]["ms"] for rows in v) for i in range(clips)] for k, v in runs.items()}
    ratio = statistics.median(b / a for a, b in zip(per_clip["A"], per_clip["B"]))
    from scipy.stats import mannwhitneyu

    pooled = {k: [x["ms"] for rows in v for x in rows] for k, v in runs.items()}
    p = mannwhitneyu(pooled["A"], pooled["B"]).pvalue
    differ = sum(x["hyp"].strip() != y["hyp"].strip() for x, y in zip(runs["A"][0], runs["B"][0]))
    err = {k: sum(x["errors"] for x in v[0]) for k, v in runs.items()}
    print(f"A: median {statistics.median(med['A']):.1f} ms (rounds {min(med['A']):.1f}-{max(med['A']):.1f})")
    print(f"B: median {statistics.median(med['B']):.1f} ms (rounds {min(med['B']):.1f}-{max(med['B']):.1f})")
    print(f"per-clip B/A median ratio {ratio:.3f}; rank-sum p = {p:.2g}; "
          f"{'significant' if p < 0.01 else 'not significant'}")
    print(f"errors A {err['A']}, B {err['B']}; texts differing: {differ} of {clips}")


if __name__ == "__main__":
    main()
