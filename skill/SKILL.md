---
name: trit
description: Transcribe speech from WAV files to text on the CPU, offline, with the Trit engine and a Parakeet, Canary, Whisper, or Moonshine model, in any language the model speaks; clean a raw transcript with S1-mini; speak text into a WAV file with Kokoro (28 English voices) or MMS-TTS. Use when an agent must turn a recording into text or text into speech, or add speech to text or text to speech to an app in Rust, C, C++, Python, C#, Node, or Bun.
---

# Trit

Speech to text on the CPU, offline: Parakeet, Canary, Whisper, and Moonshine models from Hugging Face, in any language they speak; 16 kHz WAV in, text out. Text to speech with Kokoro-82M (28 English voices) and MMS-TTS. `--device gpu` or `--device auto` runs the matrix products on the GPU. `auto` uses the CPU when the computer has no adapter. Build with `--features gpu` for that path.

## Commands

- `trit transcribe <file.wav>` to get the text of a recording
- `trit models` to list the speech models (Parakeet, Canary, Whisper, Moonshine) and get one
- `trit transcribe <folder> --model <name> --format json` for every WAV file in a folder
- `trit clean "<raw transcript>"` to rewrite text as clean written English (S1-mini)
- `trit speak "<text>" --out speech.wav` to speak text (Kokoro: 28 English voices; MMS-TTS)
- `trit phonemes "<text>"` to see the phonemes Kokoro speaks for text
- `trit doctor` to check the models, the CPU, and the GPU
- `trit skill --check <file>` to check a copy of this skill

Output is TOON on stdout. Errors are on stdout too, as `error:` and `help:` lines. Exit codes: 0 success, 1 error, 2 usage error.

## transcribe

```text
usage: trit transcribe <path>... [--model <dir>] [--language <code>] [--dictionary <file>] [--task <t>] [--timestamps] [--cleanup] [--precision <p>] [--device <d>] [--beam <n>] [--format toon|json|text] [--full]
  <path>          a 16 kHz WAV file, or a folder: every .wav file in it and its subfolders
  --model <dir>   speech model folder or `trit models` name: Parakeet, Canary, Whisper, Moonshine (default: TRIT_MODEL, then the places `trit doctor` lists)
  --language <code>  spoken language, ISO 639-1 (en, de, ...); default: the model detects it
  --dictionary <file>  terms to favor, one per line (names, product words)
  --task <t>      transcribe (default), or translate into English (multilingual Whisper, Canary)
  --timestamps    timed segments too: sentences for Parakeet, windows of up to 30 s (40 s Canary) for the others
  --cleanup       clean each transcript with S1-mini (see `trit clean --help`); ms then includes it
  --cleanup-model <dir>, --styling, --structure, --context  as for `trit clean`
  --precision     auto (default: int8 for layers 384 wide and more, f16 below; a GGUF file keeps its stored types), int8 (least memory), f16, f32 (as the reference runtimes), q8_0 (ggml's Q8_0, as llama.cpp and transcribe.cpp run it)
  --device        cpu (default), gpu, auto. auto uses the GPU, or the CPU when the computer has no adapter. TRIT_DEVICE overrides
  --beam <n>      beams for Whisper, Moonshine, and Canary (default: 5 for Whisper and Moonshine, 1 (greedy) for Canary; at most 16)
  --format        toon (default), json (full text, for programs), text (one transcript per line)
  --full          do not cut transcripts longer than 500 characters in a list of several files
examples:
  trit transcribe meeting.wav
  trit transcribe clips/ --format json
  trit transcribe take.wav --dictionary terms.txt --cleanup
  trit transcribe interview.wav --model models/whisper-large-v3-turbo --language de
  trit transcribe meeting.wav --device gpu
```

## clean

```text
usage: trit clean <text>... [--model <dir>] [--styling <s>] [--structure <s>] [--context <c>] [--precision <p>] [--device <d>] [--format toon|json|text]
  Rewrites a raw transcript as written text with Superwhisper S1-mini (English): removes
  fillers, resolves self-corrections, adds punctuation and casing, writes numbers and addresses.
  <text>          the raw transcript; several arguments join with spaces
  --model <dir>   S1-mini folder (default: TRIT_CLEANUP_MODEL, models/s1-mini, the Hugging Face cache)
  --styling       casual, semi-casual, semi-formal (default), formal
  --structure     prose (default), lists
  --context       general (default), email
  --precision     auto (default), int8, f16, f32, q8_0; as for `trit transcribe`
  --device        cpu (default), gpu, auto; as for `trit transcribe`. TRIT_DEVICE overrides
  --format        toon (default), json, text
examples:
  trit clean "so um send the report by friday no wait thursday"
  trit clean "hey sarah can you send the numbers thanks john" --context email
```

## speak

```text
usage: trit speak <text>... [--out <file.wav>] [--model <dir>] [--voice <name>] [--phonemes] [--speed <x>] [--noise <x>] [--noise-duration <x>] [--seed <n>] [--format toon|json]
  Speaks text into a mono 16-bit WAV file: Kokoro-82M (24 kHz, 28 English voices, numbers and
  symbols read; the default when found) or a VITS voice (Meta MMS-TTS, 16 kHz, reads letters only).
  <text>          the text; several arguments join with spaces
  --out <file>    the WAV file to write (default: speech.wav)
  --model <dir>   voice folder or `trit models` name (default: TRIT_VOICE, then kokoro-82m, then mms-tts-eng)
  --voice <name>  Kokoro voice: af_heart (default), bm_george, ...; see `trit voices`; a,b blends two
  --phonemes      Kokoro: the text is phonemes (as `trit phonemes` writes them), one pass per line
  --speed <x>     speaking rate: 1 is the voice's pace (default), 1.25 faster, 0.8 slower
  --noise <x>     how much the voice varies: VITS noise_scale (default 0.667); Kokoro source noise (default 1)
  --noise-duration <x>  VITS: how much the timing varies (default 0.8)
  --seed <n>      seed of the noise (default 0); one text and one seed give the same audio
  --format        toon (default), json
examples:
  trit speak "The build passed on March 3rd, 2026." --out done.wav
  trit speak "Good morning from London." --voice bf_emma
  trit speak "Ship it." --model mms-tts-eng --speed 0.9
```

## phonemes

```text
usage: trit phonemes <text>... [--british] [--no-fallback] [--file <path>]
  The phonemes Kokoro speaks for English text: misaki's lexicon, numbers, and stress, one line
  per line. Put a word's own phonemes in the text as [word](/phonemes/).
  <text>          the text; several arguments join with spaces
  --british       British English (the b… voices); default American (a… voices)
  --no-fallback   no phonemes for words that no dictionary holds (default: the BART fallback
                  PeterReid/graphemes_to_phonemes_en_us or _gb, when it is found)
  --file <path>   read the text from a file (UTF-8)
examples:
  trit phonemes "The build passed on March 3rd."
  trit phonemes "[Kokoro](/kˈOkəɹO/) speaks." --british
```

## voices

```text
usage: trit voices [--model <dir>]
  Lists the voices of a Kokoro folder (the default voice folder when none is named), with
  their accent: af/am American female/male, bf/bm British female/male.
examples:
  trit voices
  trit speak "Hello." --voice bm_george
```

## Get a model

`trit models` lists the known models, where each is, and its download command. Pass a name or a folder to `--model`.

| Name | Family | Languages | MB |
|---|---|---|---:|
| parakeet-redux | parakeet_tdt | 25 European | 178 |
| parakeet-tdt-0.6b-v3 | parakeet_tdt | 25 European | 2508 |
| parakeet-ultra | parakeet_tdt | 25 European | 1255 |
| parakeet-rnnt-0.6b | parakeet_rnnt | en | 2467 |
| parakeet-unified-en-0.6b | parakeet_rnnt | en | 2474 |
| parakeet-unified-en-0.6b-q8_0 | parakeet_rnnt | en | 731 |
| parakeet-tdt-0.6b-v3-q8_0 | parakeet_tdt | 25 European | 740 |
| parakeet-tdt-0.6b-v2-q8_0 | parakeet_tdt | en | 730 |
| parakeet-ultra-q8_0 | parakeet_tdt | 25 European | 740 |
| parakeet-ctc-0.6b | parakeet_ctc | en | 2436 |
| canary-1b-v2 | canary | 25 European | 3916 |
| whisper-tiny | whisper | 99 | 151 |
| whisper-base.en | whisper | en | 290 |
| whisper-small.en | whisper | en | 967 |
| whisper-large-v3-turbo | whisper | 99 | 1618 |
| whisper-large-v3 | whisper | 99 | 3087 |
| distil-large-v3.5 | whisper | en | 3026 |
| moonshine-tiny | moonshine | en | 108 |
| moonshine-base | moonshine | en | 246 |
| qwen3-asr-0.6b | qwen3_asr | 30 | 1876 |
| wav2vec2-base-960h | wav2vec2 | en | 378 |
| breeze-asr-25-q8_0 | whisper | zh,en | 1668 |
| canary-180m-flash-q8_0 | canary | en,de,es,fr | 218 |
| canary-1b-flash-q8_0 | canary | en,de,es,fr | 1048 |
| canary-1b-v2-q8_0 | canary | 25 European | 1144 |
| canary-1b-q8_0 | canary | en,de,es,fr | 1163 |
| canary-qwen-2.5b-q8_0 | canary_qwen | en | 2798 |
| cohere-transcribe-03-2026-q8_0 | canary | 14 | 2411 |
| cohere-transcribe-arabic-07-2026-q8_0 | canary | en,ar | 2411 |
| fun-asr-mlt-nano-2512-q8_0 | fun_asr_nano | 31 | 891 |
| fun-asr-nano-2512-q8_0 | fun_asr_nano | zh,en,ja | 891 |
| gigaam-v3-ctc-q8_0 | gigaam_ctc | ru | 272 |
| gigaam-v3-e2e-ctc-q8_0 | gigaam_ctc | ru | 272 |
| gigaam-v3-e2e-rnnt-q8_0 | gigaam_rnnt | ru | 274 |
| gigaam-v3-rnnt-q8_0 | gigaam_rnnt | ru | 273 |
| granite-4.0-1b-speech-q8_0 | granite_speech | 6 | 2560 |
| granite-speech-4.1-2b-nar-q8_0 | granite_speech_nar | 5 | 2498 |
| granite-speech-4.1-2b-plus-q8_0 | granite_speech | 5 | 2346 |
| granite-speech-4.1-2b-q8_0 | granite_speech | 6 | 2560 |
| granite-speech-5.0-470m-turboctc-nc-q8_0 | granite_ctc | en | 506 |
| granite-speech-5.0-470m-turboctc-q8_0 | granite_ctc | en | 506 |
| medasr-q8_0 | medasr | en | 128 |
| moonshine-base-ar-q8_0 | moonshine | ar | 77 |
| moonshine-base-ja-q8_0 | moonshine | ja | 77 |
| moonshine-base-ko-q8_0 | moonshine | ko | 77 |
| moonshine-base-uk-q8_0 | moonshine | uk | 77 |
| moonshine-base-vi-q8_0 | moonshine | vi | 77 |
| moonshine-base-zh-q8_0 | moonshine | zh | 77 |
| moonshine-base-q8_0 | moonshine | en | 77 |
| moonshine-streaming-medium-q8_0 | moonshine_streaming | en | 296 |
| moonshine-streaming-small-q8_0 | moonshine_streaming | en | 199 |
| moonshine-streaming-tiny-q8_0 | moonshine_streaming | en | 50 |
| moonshine-tiny-ar-q8_0 | moonshine | ar | 35 |
| moonshine-tiny-ja-q8_0 | moonshine | ja | 35 |
| moonshine-tiny-ko-q8_0 | moonshine | ko | 35 |
| moonshine-tiny-uk-q8_0 | moonshine | uk | 35 |
| moonshine-tiny-vi-q8_0 | moonshine | vi | 35 |
| moonshine-tiny-zh-q8_0 | moonshine | zh | 35 |
| moonshine-tiny-q8_0 | moonshine | en | 35 |
| moss-transcribe-diarize-q8_0 | moss | en,zh | 987 |
| multitalker-parakeet-streaming-0.6b-v1-q8_0 | parakeet_rnnt | en | 873 |
| nemotron-3.5-asr-streaming-0.6b-q8_0 | parakeet_rnnt | 32 | 751 |
| nemotron-speech-streaming-en-0.6b-q8_0 | parakeet_rnnt | en | 730 |
| orukeet-q8_0 | parakeet_tdt | 25 European | 740 |
| parakeet-ctc-0.6b-q8_0 | parakeet_ctc | en | 722 |
| parakeet-ctc-1.1b-q8_0 | parakeet_ctc | en | 1260 |
| parakeet-primeline-q8_0 | parakeet_tdt | 25 European | 740 |
| parakeet-rnnt-0.6b-q8_0 | parakeet_rnnt | en | 730 |
| parakeet-rnnt-1.1b-q8_0 | parakeet_rnnt | en | 1267 |
| parakeet-tdt-1.1b-q8_0 | parakeet_tdt | en | 1267 |
| parakeet-tdt_ctc-1.1b-q8_0 | parakeet_tdt | en | 1267 |
| parakeet-tdt_ctc-110m-q8_0 | parakeet_tdt | en | 135 |
| qwen3-asr-0.6b-q8_0 | qwen3_asr | 30 | 850 |
| qwen3-asr-1.7b-q8_0 | qwen3_asr | 30 | 2185 |
| sensevoice-small-q8_0 | sensevoice | 5 | 253 |
| voxtral-mini-3b-2507-q8_0 | voxtral | 8 | 5000 |
| voxtral-mini-4b-realtime-2602-q8_0 | voxtral_realtime | 13 | 4732 |
| voxtral-small-24b-2507-q8_0 | voxtral | 8 | 25810 |
| whisper-base.en-q8_0 | whisper | en | 85 |
| whisper-base-q8_0 | whisper | 99 | 85 |
| whisper-large-v2-q8_0 | whisper | 99 | 1668 |
| whisper-large-v3-turbo-q8_0 | whisper | 100 | 886 |
| whisper-large-v3-q8_0 | whisper | 100 | 1669 |
| whisper-large-q8_0 | whisper | 99 | 1668 |
| whisper-medium.en-q8_0 | whisper | en | 831 |
| whisper-medium-q8_0 | whisper | 99 | 832 |
| whisper-small.en-q8_0 | whisper | en | 270 |
| whisper-small-q8_0 | whisper | 99 | 270 |
| whisper-tiny.en-q8_0 | whisper | en | 46 |
| whisper-tiny-q8_0 | whisper | 99 | 46 |
| s1-mini | cleanup | en | 1503 |
| kokoro-82m | kokoro | en (US, GB) | 327 |
| graphemes_to_phonemes_en_us | g2p | en (US) | 3 |
| graphemes_to_phonemes_en_gb | g2p | en (GB) | 3 |
| mms-tts-eng | vits | en | 145 |

Example: `hf download moondream/parakeet-redux config.json tokenizer.json model.safetensors --local-dir models/parakeet-redux`. The default model is parakeet-redux; `TRIT_MODEL=<dir>` sets another.

## Use it from code

Read the README (https://github.com/Hi9841/trit): the Rust API, the C ABI (`include/trit.h`), and bindings for Python, C#, and Node. Any other language with a C FFI uses `include/trit.h` directly.
