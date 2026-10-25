# Third-party notices

Trit is under the PolyForm Noncommercial License 1.0.0 (`LICENSE`). Parts of
it come from other projects under their own licenses:

## misaki (Apache-2.0)

- Source: https://github.com/hexgrad/misaki, commit
  fba1236595f2d2bf21d414ba6e57d25256afada3.
- Used: the pronunciation lexicons `misaki/data/{us,gb}_{gold,silver}.json`,
  stored in `src/tts/english/data/` as compact JSON, deflated (zlib); the
  entries are unchanged. `src/tts/english/` (lexicon, numbers, grouping,
  stress) is a Rust port of `misaki/en.py` 0.9.4; the tokens and tags come
  from Trit's own tokenizer and tagger instead of spaCy, and the fallback
  for unknown words reads only words.
- License: Apache License 2.0, in `LICENSES/Apache-2.0.txt`.

## textblob-aptagger, through NLTK data (MIT)

- Source: the averaged perceptron tagger of NLTK data
  (`averaged_perceptron_tagger_eng`), from
  https://github.com/sloria/textblob-aptagger.
- Used: its weights, tag dictionary, and classes, stored in
  `src/tts/english/data/tagger_*.json.z` (compact JSON, deflated); the
  values are unchanged. `src/tts/english/tagger.rs` follows NLTK's
  `PerceptronTagger.tag`.

```text
Copyright 2013 Matthew Honnibal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Models

Trit ships no model weights. The models it runs have their own licenses
(README, "License"); follow them when you ship a model.
