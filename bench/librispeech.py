# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow", "soundfile", "numpy"]
# ///
"""Writes 200 LibriSpeech test utterances per subset as 16 kHz WAV plus .txt.

usage: uv run bench/librispeech.py [COUNT]

Run from desktop/. Downloads the test split of openslr/librispeech_asr from
Hugging Face into target/accuracy-src, then takes every n-th utterance, so the
clips spread over all speakers. Output: target/accuracy/clean and target/accuracy/other.
"""

import io
import sys
import urllib.request
import wave
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

URL = "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/{subset}/test/0000.parquet"
TARGET = Path(__file__).resolve().parents[1] / "target"


def extract(src: Path, out: Path, count: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(src, columns=["audio", "text", "id"])
    step = max(1, table.num_rows // count)
    written = 0
    for i in range(0, table.num_rows, step):
        if written >= count:
            break
        row = {k: table.column(k)[i].as_py() for k in ("audio", "text", "id")}
        audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
        assert rate == 16000
        with wave.open(str(out / f"{row['id']}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio.tobytes())
        (out / f"{row['id']}.txt").write_text(row["text"], encoding="utf-8")
        written += 1
    print(f"{written} of {table.num_rows} utterances -> {out}")


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    for subset in ("clean", "other"):
        src = TARGET / "accuracy-src" / f"{subset}-test.parquet"
        if not src.exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL.format(subset=subset), src)
        extract(src, TARGET / "accuracy" / subset, count)


if __name__ == "__main__":
    main()
