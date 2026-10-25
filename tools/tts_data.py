"""Writes the data of Trit's English phonemizer to src/tts/english/data/:
misaki's pronunciation lexicons and NLTK's averaged perceptron tagger,
each deflated (zlib). Run it from the repository root: python tools/tts_data.py

Sources (pinned; the hashes are checked):
- misaki (Apache-2.0), https://github.com/hexgrad/misaki at MISAKI_COMMIT:
  misaki/data/{us,gb}_{gold,silver}.json
- NLTK data (MIT, from textblob-aptagger): averaged_perceptron_tagger_eng.zip
"""

import hashlib
import io
import json
import pathlib
import urllib.request
import zipfile
import zlib

MISAKI_COMMIT = "fba1236595f2d2bf21d414ba6e57d25256afada3"
MISAKI = f"https://raw.githubusercontent.com/hexgrad/misaki/{MISAKI_COMMIT}/misaki/data/"
TAGGER = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip"
TAGGER_SHA256 = "6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b"
OUT = pathlib.Path(__file__).resolve().parents[1] / "src" / "tts" / "english" / "data"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def write(name: str, data: bytes) -> str:
    # Parse and dump compactly, so the stored bytes do not depend on the source's spacing.
    compact = json.dumps(json.loads(data), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    packed = zlib.compress(compact, 9)
    (OUT / f"{name}.z").write_bytes(packed)
    return f"| `{name}.z` | {len(compact)} | {len(packed)} | {hashlib.sha256(compact).hexdigest()} |"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ["us_gold.json", "us_silver.json", "gb_gold.json", "gb_silver.json"]:
        rows.append(write(name, fetch(MISAKI + name)))
    archive = fetch(TAGGER)
    if hashlib.sha256(archive).hexdigest() != TAGGER_SHA256:
        raise SystemExit("the tagger archive does not match its sha256")
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for member in z.namelist():
            if member.endswith(".json"):
                short = member.rsplit(".", 2)[-2] + ".json"  # weights, tagdict, classes
                rows.append(write(f"tagger_{short}", z.read(member)))
    readme = OUT / "README.md"
    readme.write_text(
        "# English phonemizer data\n\n"
        "Written by `tools/tts_data.py`; do not edit. Each file is compact JSON, deflated (zlib).\n\n"
        f"- `us_*`, `gb_*`: misaki's lexicons (Apache-2.0), https://github.com/hexgrad/misaki at {MISAKI_COMMIT}.\n"
        "- `tagger_*`: NLTK's averaged perceptron tagger for English (MIT, from textblob-aptagger).\n\n"
        "| File | JSON bytes | Stored bytes | SHA-256 of the JSON |\n|---|---:|---:|---|\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    print("\n".join(rows))


if __name__ == "__main__":
    main()
