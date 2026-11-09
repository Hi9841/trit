"""The same text on every kernel path: the rule that all paths of a CPU give one text.

usage: python bench/paths.py <trit executable> <model folder or name> <clip.wav>...

Transcribes the clips once with the CPU's own paths, then once for each value
of TRIT_ISA that this kind of CPU takes, and once with one thread. Prints one
line per run. Exit 1 when a run fails, when two runs differ in the text of a
clip, or when no clip gives text (the model or the clips are not there).
Python's standard library only, so CI needs no environment for it.
"""

import json
import os
import platform
import subprocess
import sys

X86 = ("x86_64", "amd64")
RUNS = [{}] + [{"TRIT_ISA": name} for name in (
    ("scalar", "avx2", "avxvnni") if platform.machine().lower() in X86 else ("scalar", "neon", "dotprod")
)] + [{"TRIT_THREADS": "1"}]


def texts(trit: str, model: str, clips: list[str], extra: dict[str, str]) -> list[str]:
    done = subprocess.run(
        [trit, "transcribe", *clips, "--model", model, "--language", "en", "--format", "json"],
        capture_output=True, text=True, encoding="utf-8", env={**os.environ, **extra},
    )
    try:
        result = json.loads(done.stdout)
    except json.JSONDecodeError:
        sys.exit(f"{extra or 'default'}: no JSON (exit {done.returncode}): {done.stdout[:400]}{done.stderr[:400]}")
    if result["errors"] or len(result["results"]) != len(clips):
        sys.exit(f"{extra or 'default'}: {result['errors']}")
    return [row["text"] for row in result["results"]]


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    trit, model, clips = sys.argv[1], sys.argv[2], sys.argv[3:]
    first = texts(trit, model, clips, RUNS[0])
    print(f"default: {first}")
    if not any(first):
        sys.exit("no clip gives text")
    differ = 0
    for extra in RUNS[1:]:
        name = " ".join(f"{key}={value}" for key, value in extra.items())
        other = texts(trit, model, clips, extra)
        changed = [clip for clip, a, b in zip(clips, first, other) if a != b]
        print(f"{name}: {'same text' if not changed else f'DIFFERENT for {changed}: {other}'}")
        differ += len(changed)
    if differ:
        sys.exit(f"{differ} texts differ between kernel paths")


if __name__ == "__main__":
    main()
