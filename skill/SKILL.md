---
name: trit
description: Transcribe speech from WAV files to text on the CPU, offline, with the Trit engine and a Parakeet, Canary, Whisper, or Moonshine model, in any language the model speaks; clean a raw transcript with S1-mini; speak text into a WAV file with Kokoro (28 English voices) or MMS-TTS. Use when an agent must turn a recording into text or text into speech, or add speech to text or text to speech to an app in Rust, C, C++, Python, C#, Node, or Bun.
---

# Trit

Speech to text on the CPU, offline: Parakeet, Canary, Whisper, and Moonshine models from Hugging Face, in any language they speak; 16 kHz WAV in, text out. Text to speech with Kokoro-82M (28 English voices) and MMS-TTS.

## Commands

- `trit transcribe <file.wav>` to get the text of a recording
- `trit models` to list the speech models (Parakeet, Canary, Whisper, Moonshine) and get one
- `trit transcribe <folder> --model <name> --format json` for every WAV file in a folder
- `trit clean "<raw transcript>"` to rewrite text as clean written English (S1-mini)
- `trit speak "<text>" --out speech.wav` to speak text (Kokoro: 28 English voices; MMS-TTS)
- `trit phonemes "<text>"` to see the phonemes Kokoro speaks for text
- `trit doctor` to check the models and the CPU
- `trit skill --check <file>` to check a copy of this skill

Output is TOON on stdout. Errors are on stdout too, as `error:` and `help:` lines. Exit codes: 0 success, 1 error, 2 usage error.

## transcribe

```text
usage: trit transcribe <path>... [--model <dir>] [--language <code>] [--dictionary <file>] [--task <t>] [--timestamps] [--cleanup] [--precision <p>] [--beam <n>] [--format toon|json|text] [--full]
  <path>          a 16 kHz WAV file, or a folder: every .wav file in it and its subfolders
  --model <dir>   speech model folder or `trit models` name: Parakeet, Canary, Whisper, Moonshine (default: TRIT_MODEL, then the places `trit doctor` lists)
  --language <code>  spoken language, ISO 639-1 (en, de, ...); default: the model detects it
  --dictionary <file>  terms to favor, one per line (names, product words)
  --task <t>      transcribe (default), or translate into English (multilingual Whisper, Canary)
  --timestamps    timed segments too: sentences for Parakeet, windows of up to 30 s (40 s Canary) for the others
  --cleanup       clean each transcript with S1-mini (see `trit clean --help`); ms then includes it
  --cleanup-model <dir>, --styling, --structure, --context  as for `trit clean`
  --precision     auto (default: int8 for layers 384 wide and more, f16 below), int8 (least memory), f16, f32 (as the reference runtimes)
  --beam <n>      beams for Whisper, Moonshine, and Canary (default: 5 for Whisper and Moonshine, 1 (greedy) for Canary; at most 16)
  --format        toon (default), json (full text, for programs), text (one transcript per line)
  --full          do not cut transcripts longer than 500 characters in a list of several files
examples:
  trit transcribe meeting.wav
  trit transcribe clips/ --format json
  trit transcribe take.wav --dictionary terms.txt --cleanup
  trit transcribe interview.wav --model models/whisper-large-v3-turbo --language de
```

## clean

```text
usage: trit clean <text>... [--model <dir>] [--styling <s>] [--structure <s>] [--context <c>] [--precision <p>] [--format toon|json|text]
  Rewrites a raw transcript as written text with Superwhisper S1-mini (English): removes
  fillers, resolves self-corrections, adds punctuation and casing, writes numbers and addresses.
  <text>          the raw transcript; several arguments join with spaces
  --model <dir>   S1-mini folder (default: TRIT_CLEANUP_MODEL, models/s1-mini, the Hugging Face cache)
  --styling       casual, semi-casual, semi-formal (default), formal
  --structure     prose (default), lists
  --context       general (default), email
  --precision     auto (default), int8, f16, f32; as for `trit transcribe`
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
| parakeet-rnnt-0.6b | parakeet_rnnt | en | 2467 |
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
| s1-mini | cleanup | en | 1503 |
| kokoro-82m | kokoro | en (US, GB) | 327 |
| graphemes_to_phonemes_en_us | g2p | en (US) | 3 |
| graphemes_to_phonemes_en_gb | g2p | en (GB) | 3 |
| mms-tts-eng | vits | en | 145 |

Example: `hf download moondream/parakeet-redux config.json tokenizer.json model.safetensors --local-dir models/parakeet-redux`. The default model is parakeet-redux; `TRIT_MODEL=<dir>` sets another.

## Use it from code

Read the README (https://github.com/Hi9841/trit): the Rust API, the C ABI (`include/trit.h`), and bindings for Python, C#, and Node. Any other language with a C FFI uses `include/trit.h` directly.
