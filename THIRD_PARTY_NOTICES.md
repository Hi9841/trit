# Third-party notices

Trit is under the GNU Affero General Public License 3.0 (`LICENSE`), or a
commercial licence (`LICENSING.md`). Parts of it come from other projects
under their own licenses:

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

Trit ships no model weights. `trit models` downloads them from Hugging
Face. Each model has its own licence; follow it when you use or ship the
model. The licences below are those of the repository that Trit downloads,
read on 2026-11-12. Check the model card before you ship: a licence can
change.

Models that **forbid commercial use**: `mms-tts-eng`, `canary-1b-q8_0`,
`granite-speech-5.0-470m-turboctc-nc-q8_0`. Models with **limited
commercial use**: the Moonshine language models (`-ar`, `-ja`, `-ko`,
`-uk`, `-vi`, `-zh`) and `medasr-q8_0`.

| Model (`trit models`) | Hugging Face repository | Licence | Commercial use |
|---|---|---|---|
| `parakeet-redux` | [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux) | CC-BY-4.0 | Yes, with attribution |
| `parakeet-tdt-0.6b-v3` | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | CC-BY-4.0 | Yes, with attribution |
| `parakeet-ultra` | [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) | CC-BY-4.0 | Yes, with attribution |
| `parakeet-rnnt-0.6b` | [nvidia/parakeet-rnnt-0.6b](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) | CC-BY-4.0 | Yes, with attribution |
| `parakeet-unified-en-0.6b` | [nvidia/parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) | NVIDIA Open Model License | Yes, with its conditions |
| `parakeet-unified-en-0.6b-q8_0` | [handy-computer/parakeet-unified-en-0.6b-gguf](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf) | NVIDIA Open Model License | Yes, with its conditions |
| `parakeet-tdt-0.6b-v3-q8_0` | [handy-computer/parakeet-tdt-0.6b-v3-gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf) | CC-BY-4.0 | Yes, with attribution |
| `parakeet-tdt-0.6b-v2-q8_0` | [handy-computer/parakeet-tdt-0.6b-v2-gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf) | CC-BY-4.0 | Yes, with attribution |
| `parakeet-ultra-q8_0` | [handy-computer/parakeet-ultra-gguf](https://huggingface.co/handy-computer/parakeet-ultra-gguf) | CC-BY-4.0 | Yes, with attribution |
| `parakeet-ctc-0.6b` | [nvidia/parakeet-ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b) | CC-BY-4.0 | Yes, with attribution |
| `canary-1b-v2` | [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) | CC-BY-4.0 | Yes, with attribution |
| `whisper-tiny` | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | Apache-2.0 | Yes |
| `whisper-base.en` | [openai/whisper-base.en](https://huggingface.co/openai/whisper-base.en) | Apache-2.0 | Yes |
| `whisper-small.en` | [openai/whisper-small.en](https://huggingface.co/openai/whisper-small.en) | Apache-2.0 | Yes |
| `whisper-large-v3-turbo` | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | MIT | Yes |
| `whisper-large-v3` | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | Apache-2.0 | Yes |
| `distil-large-v3.5` | [distil-whisper/distil-large-v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5) | MIT | Yes |
| `moonshine-tiny` | [UsefulSensors/moonshine-tiny](https://huggingface.co/UsefulSensors/moonshine-tiny) | MIT | Yes |
| `moonshine-base` | [UsefulSensors/moonshine-base](https://huggingface.co/UsefulSensors/moonshine-base) | MIT | Yes |
| `qwen3-asr-0.6b` | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Apache-2.0 | Yes |
| `wav2vec2-base-960h` | [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) | Apache-2.0 | Yes |
| `breeze-asr-25-q8_0` | [handy-computer/Breeze-ASR-25-gguf](https://huggingface.co/handy-computer/Breeze-ASR-25-gguf) | Apache-2.0 | Yes |
| `canary-180m-flash-q8_0` | [handy-computer/canary-180m-flash-gguf](https://huggingface.co/handy-computer/canary-180m-flash-gguf) | CC-BY-4.0 | Yes, with attribution |
| `canary-1b-flash-q8_0` | [handy-computer/canary-1b-flash-gguf](https://huggingface.co/handy-computer/canary-1b-flash-gguf) | CC-BY-4.0 | Yes, with attribution |
| `canary-1b-v2-q8_0` | [handy-computer/canary-1b-v2-gguf](https://huggingface.co/handy-computer/canary-1b-v2-gguf) | CC-BY-4.0 | Yes, with attribution |
| `canary-1b-q8_0` | [handy-computer/canary-1b-gguf](https://huggingface.co/handy-computer/canary-1b-gguf) | CC-BY-NC-4.0 | **NO: noncommercial only** |
| `canary-qwen-2.5b-q8_0` | [handy-computer/canary-qwen-2.5b-gguf](https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf) | CC-BY-4.0 | Yes, with attribution |
| `cohere-transcribe-03-2026-q8_0` | [handy-computer/cohere-transcribe-03-2026-gguf](https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf) | Apache-2.0 | Yes |
| `cohere-transcribe-arabic-07-2026-q8_0` | [handy-computer/cohere-transcribe-arabic-07-2026-gguf](https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf) | Apache-2.0 | Yes |
| `fun-asr-mlt-nano-2512-q8_0` | [handy-computer/Fun-ASR-MLT-Nano-2512-gguf](https://huggingface.co/handy-computer/Fun-ASR-MLT-Nano-2512-gguf) | FunASR Model License 1.1 | Yes, with attribution (revocable) |
| `fun-asr-nano-2512-q8_0` | [handy-computer/Fun-ASR-Nano-2512-gguf](https://huggingface.co/handy-computer/Fun-ASR-Nano-2512-gguf) | FunASR Model License 1.1 | Yes, with attribution (revocable) |
| `gigaam-v3-ctc-q8_0` | [handy-computer/gigaam-v3-ctc-gguf](https://huggingface.co/handy-computer/gigaam-v3-ctc-gguf) | MIT | Yes |
| `gigaam-v3-e2e-ctc-q8_0` | [handy-computer/gigaam-v3-e2e-ctc-gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf) | MIT | Yes |
| `gigaam-v3-e2e-rnnt-q8_0` | [handy-computer/gigaam-v3-e2e-rnnt-gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-rnnt-gguf) | MIT | Yes |
| `gigaam-v3-rnnt-q8_0` | [handy-computer/gigaam-v3-rnnt-gguf](https://huggingface.co/handy-computer/gigaam-v3-rnnt-gguf) | MIT | Yes |
| `granite-4.0-1b-speech-q8_0` | [handy-computer/granite-4.0-1b-speech-gguf](https://huggingface.co/handy-computer/granite-4.0-1b-speech-gguf) | Apache-2.0 | Yes |
| `granite-speech-4.1-2b-nar-q8_0` | [handy-computer/granite-speech-4.1-2b-nar-gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf) | Apache-2.0 | Yes |
| `granite-speech-4.1-2b-plus-q8_0` | [handy-computer/granite-speech-4.1-2b-plus-gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-plus-gguf) | Apache-2.0 | Yes |
| `granite-speech-4.1-2b-q8_0` | [handy-computer/granite-speech-4.1-2b-gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-gguf) | Apache-2.0 | Yes |
| `granite-speech-5.0-470m-turboctc-nc-q8_0` | [handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf) | CC-BY-NC-SA-4.0 | **NO: noncommercial only** |
| `granite-speech-5.0-470m-turboctc-q8_0` | [handy-computer/granite-speech-5.0-470m-turboctc-gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf) | Apache-2.0 | Yes |
| `medasr-q8_0` | [handy-computer/medasr-gguf](https://huggingface.co/handy-computer/medasr-gguf) | Health AI Developer Foundations Terms | **Conditional:** use restrictions |
| `moonshine-base-ar-q8_0` | [handy-computer/moonshine-base-ar-gguf](https://huggingface.co/handy-computer/moonshine-base-ar-gguf) | Moonshine AI Community License | **Conditional:** registration; paid licence above US$1M yearly revenue |
| `moonshine-base-ja-q8_0` | [handy-computer/moonshine-base-ja-gguf](https://huggingface.co/handy-computer/moonshine-base-ja-gguf) | Moonshine AI Community License | **Conditional:** registration; paid licence above US$1M yearly revenue |
| `moonshine-base-ko-q8_0` | [handy-computer/moonshine-base-ko-gguf](https://huggingface.co/handy-computer/moonshine-base-ko-gguf) | Moonshine AI Community License | **Conditional:** registration; paid licence above US$1M yearly revenue |
| `moonshine-base-uk-q8_0` | [handy-computer/moonshine-base-uk-gguf](https://huggingface.co/handy-computer/moonshine-base-uk-gguf) | Moonshine AI Community License | **Conditional:** registration; paid licence above US$1M yearly revenue |
| `moonshine-base-vi-q8_0` | [handy-computer/moonshine-base-vi-gguf](https://huggingface.co/handy-computer/moonshine-base-vi-gguf) | Moonshine AI Community License | **Conditional:** registration; paid licence above US$1M yearly revenue |
| `moonshine-base-zh-q8_0` | [handy-computer/moonshine-base-zh-gguf](https://huggingface.co/handy-computer/moonshine-base-zh-gguf) | Moonshine AI Community License | **Conditional:** registration; paid licence above US$1M yearly revenue |
| `moonshine-base-q8_0` | [handy-computer/moonshine-base-gguf](https://huggingface.co/handy-computer/moonshine-base-gguf) | MIT | Yes |
| `moonshine-streaming-medium-q8_0` | [handy-computer/moonshine-streaming-medium-gguf](https://huggingface.co/handy-computer/moonshine-streaming-medium-gguf) | MIT | Yes |
| `moonshine-streaming-small-q8_0` | [handy-computer/moonshine-streaming-small-gguf](https://huggingface.co/handy-computer/moonshine-streaming-small-gguf) | MIT | Yes |
| `moonshine-streaming-tiny-q8_0` | [handy-computer/moonshine-streaming-tiny-gguf](https://huggingface.co/handy-computer/moonshine-streaming-tiny-gguf) | MIT | Yes |
| `moonshine-tiny-ar-q8_0` | [handy-computer/moonshine-tiny-ar-gguf](https://huggingface.co/handy-computer/moonshine-tiny-ar-gguf) | Moonshine AI Community License | **Conditional:** registration; paid licence above US$1M yearly revenue |
| `moonshine-tiny-ja-q8_0` | [handy-computer/moonshine-tiny-ja-gguf](https://huggingface.co/handy-computer/moonshine-tiny-ja-gguf) | Moonshine AI Community License | **Conditional:** registration; paid licence above US$1M yearly revenue |
| `moonshine-tiny-ko-q8_0` | [handy-computer/moonshine-tiny-ko-gguf](https://huggingface.co/handy-computer/moonshine-tiny-ko-gguf) | Moonshine AI Community License | **Conditional:** registration; paid licence above US$1M yearly revenue |
| `moonshine-tiny-uk-q8_0` | [handy-computer/moonshine-tiny-uk-gguf](https://huggingface.co/handy-computer/moonshine-tiny-uk-gguf) | Moonshine AI Community License | **Conditional:** registration; paid licence above US$1M yearly revenue |
| `moonshine-tiny-vi-q8_0` | [handy-computer/moonshine-tiny-vi-gguf](https://huggingface.co/handy-computer/moonshine-tiny-vi-gguf) | Moonshine AI Community License | **Conditional:** registration; paid licence above US$1M yearly revenue |
| `moonshine-tiny-zh-q8_0` | [handy-computer/moonshine-tiny-zh-gguf](https://huggingface.co/handy-computer/moonshine-tiny-zh-gguf) | Moonshine AI Community License | **Conditional:** registration; paid licence above US$1M yearly revenue |
| `moonshine-tiny-q8_0` | [handy-computer/moonshine-tiny-gguf](https://huggingface.co/handy-computer/moonshine-tiny-gguf) | MIT | Yes |
| `moss-transcribe-diarize-q8_0` | [handy-computer/MOSS-Transcribe-Diarize-gguf](https://huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf) | Apache-2.0 | Yes |
| `multitalker-parakeet-streaming-0.6b-v1-q8_0` | [handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf) | NVIDIA Open Model License | Yes, with its conditions |
| `nemotron-3.5-asr-streaming-0.6b-q8_0` | [handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf) | OpenMDW-1.1 | Yes |
| `nemotron-speech-streaming-en-0.6b-q8_0` | [handy-computer/nemotron-speech-streaming-en-0.6b-gguf](https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf) | NVIDIA Open Model License | Yes, with its conditions |
| `orukeet-q8_0` | [handy-computer/orukeet-gguf](https://huggingface.co/handy-computer/orukeet-gguf) | CC-BY-SA-4.0 | Yes, with attribution and share-alike |
| `parakeet-ctc-0.6b-q8_0` | [handy-computer/parakeet-ctc-0.6b-gguf](https://huggingface.co/handy-computer/parakeet-ctc-0.6b-gguf) | CC-BY-4.0 | Yes, with attribution |
| `parakeet-ctc-1.1b-q8_0` | [handy-computer/parakeet-ctc-1.1b-gguf](https://huggingface.co/handy-computer/parakeet-ctc-1.1b-gguf) | CC-BY-4.0 | Yes, with attribution |
| `parakeet-primeline-q8_0` | [handy-computer/parakeet-primeline-gguf](https://huggingface.co/handy-computer/parakeet-primeline-gguf) | CC-BY-4.0 | Yes, with attribution |
| `parakeet-rnnt-0.6b-q8_0` | [handy-computer/parakeet-rnnt-0.6b-gguf](https://huggingface.co/handy-computer/parakeet-rnnt-0.6b-gguf) | CC-BY-4.0 | Yes, with attribution |
| `parakeet-rnnt-1.1b-q8_0` | [handy-computer/parakeet-rnnt-1.1b-gguf](https://huggingface.co/handy-computer/parakeet-rnnt-1.1b-gguf) | CC-BY-4.0 | Yes, with attribution |
| `parakeet-tdt-1.1b-q8_0` | [handy-computer/parakeet-tdt-1.1b-gguf](https://huggingface.co/handy-computer/parakeet-tdt-1.1b-gguf) | CC-BY-4.0 | Yes, with attribution |
| `parakeet-tdt_ctc-1.1b-q8_0` | [handy-computer/parakeet-tdt_ctc-1.1b-gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-1.1b-gguf) | CC-BY-4.0 | Yes, with attribution |
| `parakeet-tdt_ctc-110m-q8_0` | [handy-computer/parakeet-tdt_ctc-110m-gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf) | CC-BY-4.0 | Yes, with attribution |
| `qwen3-asr-0.6b-q8_0` | [handy-computer/Qwen3-ASR-0.6B-gguf](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf) | Apache-2.0 | Yes |
| `qwen3-asr-1.7b-q8_0` | [handy-computer/Qwen3-ASR-1.7B-gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf) | Apache-2.0 | Yes |
| `sensevoice-small-q8_0` | [handy-computer/SenseVoiceSmall-gguf](https://huggingface.co/handy-computer/SenseVoiceSmall-gguf) | FunASR Model License 1.1 | Yes, with attribution (revocable) |
| `voxtral-mini-3b-2507-q8_0` | [handy-computer/Voxtral-Mini-3B-2507-gguf](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-gguf) | Apache-2.0 | Yes |
| `voxtral-mini-4b-realtime-2602-q8_0` | [handy-computer/Voxtral-Mini-4B-Realtime-2602-gguf](https://huggingface.co/handy-computer/Voxtral-Mini-4B-Realtime-2602-gguf) | Apache-2.0 | Yes |
| `voxtral-small-24b-2507-q8_0` | [handy-computer/Voxtral-Small-24B-2507-gguf](https://huggingface.co/handy-computer/Voxtral-Small-24B-2507-gguf) | Apache-2.0 | Yes |
| `whisper-base.en-q8_0` | [handy-computer/whisper-base.en-gguf](https://huggingface.co/handy-computer/whisper-base.en-gguf) | Apache-2.0 | Yes |
| `whisper-base-q8_0` | [handy-computer/whisper-base-gguf](https://huggingface.co/handy-computer/whisper-base-gguf) | Apache-2.0 | Yes |
| `whisper-large-v2-q8_0` | [handy-computer/whisper-large-v2-gguf](https://huggingface.co/handy-computer/whisper-large-v2-gguf) | Apache-2.0 | Yes |
| `whisper-large-v3-turbo-q8_0` | [handy-computer/whisper-large-v3-turbo-gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf) | MIT | Yes |
| `whisper-large-v3-q8_0` | [handy-computer/whisper-large-v3-gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf) | Apache-2.0 | Yes |
| `whisper-large-q8_0` | [handy-computer/whisper-large-gguf](https://huggingface.co/handy-computer/whisper-large-gguf) | Apache-2.0 | Yes |
| `whisper-medium.en-q8_0` | [handy-computer/whisper-medium.en-gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf) | Apache-2.0 | Yes |
| `whisper-medium-q8_0` | [handy-computer/whisper-medium-gguf](https://huggingface.co/handy-computer/whisper-medium-gguf) | Apache-2.0 | Yes |
| `whisper-small.en-q8_0` | [handy-computer/whisper-small.en-gguf](https://huggingface.co/handy-computer/whisper-small.en-gguf) | Apache-2.0 | Yes |
| `whisper-small-q8_0` | [handy-computer/whisper-small-gguf](https://huggingface.co/handy-computer/whisper-small-gguf) | Apache-2.0 | Yes |
| `whisper-tiny.en-q8_0` | [handy-computer/whisper-tiny.en-gguf](https://huggingface.co/handy-computer/whisper-tiny.en-gguf) | Apache-2.0 | Yes |
| `whisper-tiny-q8_0` | [handy-computer/whisper-tiny-gguf](https://huggingface.co/handy-computer/whisper-tiny-gguf) | Apache-2.0 | Yes |
| `s1-mini` | [superwhisper/s1-mini](https://huggingface.co/superwhisper/s1-mini) | Apache-2.0 with a naming term | Yes; keep the name "S1-mini" by "Superwhisper" |
| `kokoro-82m` | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Apache-2.0 | Yes |
| `graphemes_to_phonemes_en_us` | [PeterReid/graphemes_to_phonemes_en_us](https://huggingface.co/PeterReid/graphemes_to_phonemes_en_us) | Apache-2.0 | Yes |
| `graphemes_to_phonemes_en_gb` | [PeterReid/graphemes_to_phonemes_en_gb](https://huggingface.co/PeterReid/graphemes_to_phonemes_en_gb) | Apache-2.0 | Yes |
| `mms-tts-eng` | [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng) | CC-BY-NC-4.0 | **NO: noncommercial only** |

Notes:

- `UsefulSensors/moonshine-tiny` and `-base` now redirect to
  `moonshine-ai/moonshine-tiny` and `-base` (MIT).
- `fun-asr-nano-2512-q8_0` and `fun-asr-mlt-nano-2512-q8_0`: the GGUF
  repository gives the FunASR Model License 1.1; the upstream
  `FunAudioLLM` repository gives Apache-2.0.
- NVIDIA Open Model License: commercial use is permitted; keep the
  notice, and do not bypass the model's safety guardrails.
- Kokoro-82M is a derivative of StyleTTS 2 (MIT).

## Rust dependencies

All crates that `trit` and `trit-ffi` can link (all features, all
targets), from `cargo tree`. Each licence is permissive and compatible
with the AGPL-3.0 and with closed-source use. None forbids commercial use.

| Crate | Version | Licence |
|---|---|---|
| `adler2` | 2.0.1 | 0BSD OR MIT OR Apache-2.0 |
| `allocator-api2` | 0.2.21 | MIT OR Apache-2.0 |
| `android_system_properties` | 0.1.6 | MIT OR Apache-2.0 |
| `arrayvec` | 0.7.8 | MIT OR Apache-2.0 |
| `ash` | 0.38.0+1.3.281 | MIT OR Apache-2.0 |
| `autocfg` | 1.5.1 | Apache-2.0 OR MIT |
| `bit-set` | 0.10.0 | Apache-2.0 OR MIT |
| `bit-vec` | 0.9.1 | Apache-2.0 OR MIT |
| `bitflags` | 2.13.2 | MIT OR Apache-2.0 |
| `block2` | 0.6.2 | MIT |
| `bytemuck` | 1.25.2 | Zlib OR Apache-2.0 OR MIT |
| `bytemuck_derive` | 1.12.1 | Zlib OR Apache-2.0 OR MIT |
| `cc` | 1.9.0 | MIT OR Apache-2.0 |
| `cfg-if` | 1.0.5 | MIT OR Apache-2.0 |
| `cfg_aliases` | 0.2.2 | MIT |
| `codespan-reporting` | 0.13.1 | Apache-2.0 |
| `crunchy` | 0.2.4 | MIT |
| `cty` | 0.2.2 | MIT OR Apache-2.0 |
| `document-features` | 0.2.12 | MIT OR Apache-2.0 |
| `equivalent` | 1.0.3 | Apache-2.0 OR MIT |
| `find-msvc-tools` | 0.1.15 | MIT OR Apache-2.0 |
| `foldhash` | 0.2.0 | Zlib |
| `gpu-allocator` | 0.28.0 | MIT OR Apache-2.0 |
| `half` | 2.7.1 | MIT OR Apache-2.0 |
| `hashbrown` | 0.16.1 | MIT OR Apache-2.0 |
| `hashbrown` | 0.17.1 | MIT OR Apache-2.0 |
| `indexmap` | 2.14.2 | Apache-2.0 OR MIT |
| `itoa` | 1.0.18 | MIT OR Apache-2.0 |
| `libc` | 0.2.191 | MIT OR Apache-2.0 |
| `libloading` | 0.8.9 | ISC |
| `libm` | 0.2.16 | MIT |
| `libmimalloc-sys` | 0.1.49 | MIT |
| `litrs` | 1.0.1 | MIT OR Apache-2.0 |
| `lock_api` | 0.4.14 | MIT OR Apache-2.0 |
| `log` | 0.4.35 | MIT OR Apache-2.0 |
| `memchr` | 2.8.3 | Unlicense OR MIT |
| `mimalloc` | 0.1.52 | MIT |
| `miniz_oxide` | 0.8.9 | MIT OR Zlib OR Apache-2.0 |
| `naga` | 30.0.1 | MIT OR Apache-2.0 |
| `naga-types` | 30.0.1 | MIT OR Apache-2.0 |
| `num-traits` | 0.2.19 | MIT OR Apache-2.0 |
| `objc2` | 0.6.5 | MIT |
| `objc2-core-foundation` | 0.3.2 | Zlib OR Apache-2.0 OR MIT |
| `objc2-core-graphics` | 0.3.2 | Zlib OR Apache-2.0 OR MIT |
| `objc2-encode` | 4.1.0 | MIT |
| `objc2-foundation` | 0.3.2 | MIT |
| `objc2-metal` | 0.3.2 | Zlib OR Apache-2.0 OR MIT |
| `objc2-quartz-core` | 0.3.2 | Zlib OR Apache-2.0 OR MIT |
| `once_cell` | 1.21.4 | MIT OR Apache-2.0 |
| `ordered-float` | 5.5.1 | MIT |
| `parking_lot` | 0.12.5 | MIT OR Apache-2.0 |
| `parking_lot_core` | 0.9.12 | MIT OR Apache-2.0 |
| `pollster` | 0.4.0 | Apache-2.0/MIT |
| `portable-atomic` | 1.15.0 | Apache-2.0 OR MIT |
| `portable-atomic-util` | 0.2.8 | Apache-2.0 OR MIT |
| `presser` | 0.3.1 | MIT OR Apache-2.0 |
| `proc-macro2` | 1.0.107 | MIT OR Apache-2.0 |
| `profiling` | 1.0.18 | MIT OR Apache-2.0 |
| `quote` | 1.0.47 | MIT OR Apache-2.0 |
| `range-alloc` | 0.1.5 | MIT OR Apache-2.0 |
| `raw-window-handle` | 0.6.2 | MIT OR Apache-2.0 OR Zlib |
| `raw-window-metal` | 1.1.0 | MIT OR Apache-2.0 |
| `redox_syscall` | 0.5.18 | MIT |
| `renderdoc-sys` | 1.1.0 | MIT OR Apache-2.0 |
| `rustc-hash` | 1.1.0 | Apache-2.0/MIT |
| `scopeguard` | 1.2.0 | MIT OR Apache-2.0 |
| `serde` | 1.0.229 | MIT OR Apache-2.0 |
| `serde_core` | 1.0.229 | MIT OR Apache-2.0 |
| `serde_derive` | 1.0.229 | MIT OR Apache-2.0 |
| `serde_json` | 1.0.154 | MIT OR Apache-2.0 |
| `shlex` | 2.0.1 | MIT OR Apache-2.0 |
| `smallvec` | 1.16.3 | MIT OR Apache-2.0 |
| `spirv` | 0.4.0+sdk-1.4.341.0 | Apache-2.0 |
| `static_assertions` | 1.1.0 | MIT OR Apache-2.0 |
| `syn` | 2.0.119 | MIT OR Apache-2.0 |
| `syn` | 3.0.9 | MIT OR Apache-2.0 |
| `thiserror` | 2.0.21 | MIT OR Apache-2.0 |
| `thiserror-impl` | 2.0.21 | MIT OR Apache-2.0 |
| `unicode-ident` | 1.0.27 | (MIT OR Apache-2.0) AND Unicode-3.0 |
| `unicode-width` | 0.2.3 | MIT OR Apache-2.0 |
| `wgpu` | 30.0.1 | MIT OR Apache-2.0 |
| `wgpu-core` | 30.0.1 | MIT OR Apache-2.0 |
| `wgpu-core-deps-apple` | 30.0.1 | MIT OR Apache-2.0 |
| `wgpu-core-deps-windows-linux-android` | 30.0.1 | MIT OR Apache-2.0 |
| `wgpu-hal` | 30.0.1 | MIT OR Apache-2.0 |
| `wgpu-naga-bridge` | 30.0.1 | MIT OR Apache-2.0 |
| `wgpu-types` | 30.0.1 | MIT OR Apache-2.0 |
| `windows` | 0.62.0 | MIT OR Apache-2.0 |
| `windows-collections` | 0.3.2 | MIT OR Apache-2.0 |
| `windows-core` | 0.62.2 | MIT OR Apache-2.0 |
| `windows-future` | 0.3.2 | MIT OR Apache-2.0 |
| `windows-implement` | 0.60.2 | MIT OR Apache-2.0 |
| `windows-interface` | 0.59.3 | MIT OR Apache-2.0 |
| `windows-link` | 0.2.1 | MIT OR Apache-2.0 |
| `windows-numerics` | 0.3.1 | MIT OR Apache-2.0 |
| `windows-result` | 0.4.1 | MIT OR Apache-2.0 |
| `windows-strings` | 0.5.1 | MIT OR Apache-2.0 |
| `windows-threading` | 0.2.1 | MIT OR Apache-2.0 |
| `zerocopy` | 0.8.63 | BSD-2-Clause OR Apache-2.0 OR MIT |
| `zerocopy-derive` | 0.8.63 | BSD-2-Clause OR Apache-2.0 OR MIT |
| `zmij` | 1.0.23 | MIT |

## Bindings and tools

| Part | Dependency | Licence | Ships in Trit |
|---|---|---|---|
| `bindings/node` | `koffi` | MIT | No (installed by npm) |
| `bindings/python` | Python standard library (`ctypes`) | PSF | No |
| `bindings/csharp` | .NET runtime | MIT | No |
| `bench/gguf` (benchmark only) | `transcribe.cpp`, `hound` | MIT; Apache-2.0 | No |
| `site/` (the website) | Astro, marked, shiki, Geist and Patrick Hand fonts | MIT; MIT; MIT; OFL-1.1 | No |
