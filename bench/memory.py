"""Memory and speed of one Trit model: RSS after load and after N clips,
and the median time per clip. Run it with nothing else on the computer.

usage: uv run --no-project --python 3.12 --with numpy --with psutil python bench/memory.py <model> <folder of wav> <N> [precision]

  <model>      a `trit models` name or folder
  <precision>  auto (default), int8, f16, f32
"""

import gc
import glob
import sys
import time
import wave
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "trit" / "bindings" / "python"))
import trit  # noqa: E402


def main() -> None:
    model, folder, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    precision = sys.argv[4] if len(sys.argv) > 4 else None
    process = psutil.Process()

    def mb() -> int:
        gc.collect()
        return round(process.memory_info().rss / 2**20)

    engine = trit.Trit(model, precision=precision)
    load = mb()
    files = sorted(glob.glob(folder + "/*.wav"))[:count]
    clips = [np.frombuffer(wave.open(f).readframes(10**8), dtype=np.int16).astype(np.float32) / 32768 for f in files]
    engine.transcribe(clips[0], language="en")
    times = []
    for samples in clips:
        started = time.perf_counter()
        engine.transcribe(samples, language="en")
        times.append((time.perf_counter() - started) * 1000)
    after = mb()
    trit.release_memory()
    idle = mb()
    times.sort()
    print(f"{Path(model).name} {precision or 'auto'}: load {load} MB, after {len(clips)} clips {after} MB, "
          f"after release_memory {idle} MB, median {times[len(times) // 2]:.1f} ms")


if __name__ == "__main__":
    main()
