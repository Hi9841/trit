# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow", "soundfile", "numpy"]
# ///
"""Long recordings from LibriSpeech: utterances of one chapter that follow each other, joined.

usage (from the repository root): uv run bench/long_clips.py [COUNT]
Reads bench/clips-src/{clean,other}-test.parquet (bench/librispeech.py gets them) and writes
bench/clips/long-clean and bench/clips/long-other: COUNT recordings each (default 40), of
45, 60, 75, 90, and 120 s in turn, as 16 kHz WAV plus .txt (the joined references)."""
import io, sys, wave
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

BENCH = Path(__file__).resolve().parent
count = int(sys.argv[1]) if len(sys.argv) > 1 else 40
targets = [45, 60, 75, 90, 120]
for subset in ("clean", "other"):
    table = pq.read_table(BENCH / "clips-src" / f"{subset}-test.parquet", columns=["audio", "text", "id"])
    chapters = defaultdict(list)
    for i in range(table.num_rows):
        key = table.column("id")[i].as_py()
        chapters[key.rsplit("-", 1)[0]].append((key, i))
    out = BENCH / "clips" / f"long-{subset}"
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    # One recording for each chapter first, so the set spreads over the speakers.
    for chapter in sorted(chapters):
        if written >= count:
            break
        target = targets[written % len(targets)]
        audio, texts, first = [], [], None
        for key, i in sorted(chapters[chapter]):
            data, rate = sf.read(io.BytesIO(table.column("audio")[i].as_py()["bytes"]), dtype="int16")
            assert rate == 16000 and data.ndim == 1
            audio.append(data); texts.append(table.column("text")[i].as_py()); first = first or key
            if sum(len(a) for a in audio) >= target * 16000:
                break
        joined = np.concatenate(audio)
        if len(joined) < 40 * 16000:
            continue
        with wave.open(str(out / f"{first}-{len(audio)}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(joined.tobytes())
        (out / f"{first}-{len(audio)}.txt").write_text(" ".join(texts), encoding="utf-8")
        written += 1
    seconds = sorted(wave.open(str(p)).getnframes() / 16000 for p in out.glob("*.wav"))
    print(f"long-{subset}: {written} recordings, {seconds[0]:.0f} to {seconds[-1]:.0f} s, {sum(seconds):.0f} s in total")
