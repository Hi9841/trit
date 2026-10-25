# English phonemizer data

Written by `tools/tts_data.py`; do not edit. Each file is compact JSON, deflated (zlib).

- `us_*`, `gb_*`: misaki's lexicons (Apache-2.0), https://github.com/hexgrad/misaki at fba1236595f2d2bf21d414ba6e57d25256afada3.
- `tagger_*`: NLTK's averaged perceptron tagger for English (MIT, from textblob-aptagger).

| File | JSON bytes | Stored bytes | SHA-256 of the JSON |
|---|---:|---:|---|
| `us_gold.json.z` | 2627718 | 719313 | acce8d48062d5d9ddbb95abf1e9a5fe3a61716c70f9cbcb514cd48d406a5388d |
| `us_silver.json.z` | 2726072 | 682489 | 7f90a125388084b9081108f0ca0dc64c8f448b849a8397b31ed752753663bbf8 |
| `gb_gold.json.z` | 2477227 | 696508 | 186be65d8b3f1fa8847e66358447484fbe9a03681cb5163a3246d536b0ec7c7e |
| `gb_silver.json.z` | 3224833 | 799636 | dff712feb32c7e287951d8bedebd966fe27a20233fa40f557feff6c0add6466f |
| `tagger_weights.json.z` | 5059875 | 1255148 | 4109f26d3dd254279a9edc18a553f70f66b18d9f0eae5fc285de65d4d1e9c0c6 |
| `tagger_classes.json.z` | 241 | 139 | f8d242a1d739d954dbd0799e176f8e9cf5bddcf1abc4850a41524212967efc10 |
| `tagger_tagdict.json.z` | 22691 | 6446 | 2e127c624b1f57a8feca9dcc94b560f334d742707809d2c51f0e455059a9b98f |
