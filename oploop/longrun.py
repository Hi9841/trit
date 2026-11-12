"""Long recordings: the time, the CPU time, and the memory of the `trit` CLI of two builds.

usage (from the repository root):
  uv run --no-project --python 3.12 --with numpy --with whisper-normalizer==0.1.12 \\
      python oploop/longrun.py <model> <A trit.exe> <B trit.exe> [--runs 3] [--sets long-clean,long-other,very-long]

`--make` first writes bench/clips/very-long: recordings of 5, 15, and 30 minutes, joined from
bench/clips/long-clean and long-other (bench/long_clips.py makes those).
Each run is one process: `trit transcribe <folders> --format json`. A and B run in turns.
For each side: the engine time and the audio it read (from the JSON), the wall time, the CPU
time (user and kernel, all threads), the peak working set and the peak private bytes of the
process (from Windows, after the process ends), the word errors, and the texts that differ.
"""

import ctypes
import json
import statistics
import subprocess
import sys
import time
import wave
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
CLIPS = ROOT / "bench" / "clips"


def make() -> None:
    sources = sorted((CLIPS / "long-clean").glob("*.wav")) + sorted((CLIPS / "long-other").glob("*.wav"))
    out = CLIPS / "very-long"
    out.mkdir(exist_ok=True)
    at = 0
    for minutes in (5, 15, 30):
        frames, texts = [], []
        while sum(len(f) for f in frames) < minutes * 60 * 16000 * 2 and at < len(sources):
            with wave.open(str(sources[at])) as w:
                frames.append(w.readframes(w.getnframes()))
            texts.append(sources[at].with_suffix(".txt").read_text(encoding="utf-8"))
            at += 1
        with wave.open(str(out / f"{minutes:02d}-minutes.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"".join(frames))
        (out / f"{minutes:02d}-minutes.txt").write_text(" ".join(texts), encoding="utf-8")
        print(f"very-long/{minutes:02d}-minutes.wav: {sum(len(f) for f in frames) / 32000 / 60:.1f} minutes")


class Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "peak_working_set", "working_set", "peak_paged", "paged", "peak_nonpaged", "nonpaged",
            "pagefile", "peak_pagefile", "private")]


def run(exe: str, model: str, folders: list[str]) -> dict:
    command = [str(Path(exe).resolve()), "transcribe", *folders, "--model", model, "--language", "en", "--format", "json"]
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=ROOT)
    output = process.communicate()[0]
    wall = time.perf_counter() - started
    handle = wintypes.HANDLE(int(process._handle))
    times = [wintypes.FILETIME() for _ in range(4)]
    ctypes.windll.kernel32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times])
    cpu = sum((t.dwHighDateTime << 32 | t.dwLowDateTime) for t in times[2:]) / 1e7
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    result = json.loads(output.decode("utf-8"))
    return {"wall": wall, "cpu": cpu, "peak_ws": counters.peak_working_set / 2**20,
            "peak_private": counters.peak_pagefile / 2**20, "audio": result["audio_seconds"],
            "busy": result["busy_seconds"], "rows": result["results"]}


def main() -> None:
    args = sys.argv[1:]
    if "--make" in args:
        make()
        return
    option = lambda name, default: args[args.index(name) + 1] if name in args else default
    model, a, b = args[:3]
    runs = int(option("--runs", "3"))
    sets = option("--sets", "long-clean,long-other,very-long").split(",")
    folders = [str(CLIPS / s) for s in sets]
    from runtimes import edits
    from whisper_normalizer.english import EnglishTextNormalizer

    normalize = EnglishTextNormalizer()
    sides = {"A": [], "B": []}
    for r in range(runs):
        for name, exe in (("A", a), ("B", b)):
            sides[name].append(run(exe, model, folders))
            last = sides[name][-1]
            print(f"run {r + 1} {name}: engine {last['busy']:.1f} s, CPU {last['cpu']:.1f} s, peak working set {last['peak_ws']:.0f} MB", flush=True)
    med = lambda name, key: statistics.median(x[key] for x in sides[name])
    audio = sides["A"][0]["audio"]
    files = len(sides["A"][0]["rows"])
    print(f"\n{Path(model).name}: {files} recordings, {audio / 60:.1f} minutes of audio, {runs} runs for each side, medians")
    print(f"{'':36}{'A':>12}{'B':>12}{'B / A':>9}")
    rows = [
        ("engine time, s", "busy", 1), ("audio seconds for each engine second", None, 0),
        ("wall time of the process, s", "wall", 1), ("CPU time, s (all threads)", "cpu", 1),
        ("CPU seconds for each hour of audio", None, 0), ("cores busy on average (CPU / wall)", None, 0),
        ("peak working set, MB", "peak_ws", 0), ("peak private memory, MB", "peak_private", 0),
    ]
    for label, key, _ in rows:
        if key:
            x, y = med("A", key), med("B", key)
        elif label.startswith("audio"):
            x, y = audio / med("A", "busy"), audio / med("B", "busy")
        elif label.startswith("CPU seconds"):
            x, y = med("A", "cpu") / audio * 3600, med("B", "cpu") / audio * 3600
        else:
            x, y = med("A", "cpu") / med("A", "wall"), med("B", "cpu") / med("B", "wall")
        print(f"{label:36}{x:12.1f}{y:12.1f}{y / x:9.3f}")
    errors = {}
    for name in sides:
        total = words = 0
        for row in sides[name][0]["rows"]:
            ref = normalize(Path(row["file"]).with_suffix(".txt").read_text(encoding="utf-8-sig")).split()
            total += edits(ref, normalize(row["text"]).split())
            words += len(ref)
        errors[name] = (total, words)
    differ = sum(x["text"] != y["text"] for x, y in zip(sides["A"][0]["rows"], sides["B"][0]["rows"]))
    print(f"word errors: A {errors['A'][0]}, B {errors['B'][0]} of {errors['A'][1]} words; recordings with another text: {differ} of {files}")
    by_length = sorted(zip(sides["A"][0]["rows"], sides["B"][0]["rows"]), key=lambda p: p[0]["seconds"])
    for lo, hi in ((0, 60), (60, 100), (100, 200), (200, 4000)):
        group = [(x, y) for x, y in by_length if lo <= x["seconds"] < hi]
        if group:
            ma = [statistics.median(s["rows"][i]["ms"] for s in sides["A"]) for i in range(files)]
            mb = [statistics.median(s["rows"][i]["ms"] for s in sides["B"]) for i in range(files)]
            index = [i for i, row in enumerate(sides["A"][0]["rows"]) if lo <= row["seconds"] < hi]
            sa = sum(ma[i] for i in index) / sum(sides["A"][0]["rows"][i]["seconds"] for i in index)
            sb = sum(mb[i] for i in index) / sum(sides["A"][0]["rows"][i]["seconds"] for i in index)
            print(f"  recordings of {lo} to {hi} s ({len(index)}): ms for each second of audio A {sa:.2f}, B {sb:.2f}, B / A {sb / sa:.3f}")
    if "--save" in args:
        Path(option("--save", "")).write_text(json.dumps({k: [{kk: vv for kk, vv in r.items() if kk != "rows"} | {"ms": [x["ms"] for x in r["rows"]], "seconds": [x["seconds"] for x in r["rows"]]} for r in v] for k, v in sides.items()}), encoding="utf-8")


if __name__ == "__main__":
    main()
