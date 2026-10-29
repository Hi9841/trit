# Trit

Trit is a speech-to-text and text-to-speech engine for the CPU. It runs
Parakeet, Canary, Whisper, and Moonshine checkpoints as published on Hugging
Face, with its own kernels: no Python, no PyTorch, no ONNX Runtime, no GPU.
16 kHz audio in, text out. It can also clean a raw transcript with
Superwhisper S1-mini, and speak text with Kokoro-82M (28 English voices)
or a VITS voice (Meta MMS-TTS).

[TDT](https://github.com/Hi9841/tdt), an offline dictation app for Windows,
uses Trit for all of its speech.
Trit is free for noncommercial use only (see [License](#license)).
The website, with benchmarks and a comparison: <https://hi9841.github.io/trit/>.

**Agents:** read [AGENTS.md](AGENTS.md) first. It gives the steps to put Trit
into a program, and the rules to change Trit.

## Start here

| You want to | Go to |
|---|---|
| Transcribe files from a shell or a script | [Quick start](#quick-start), then [Command line](#command-line) |
| Put Trit into a Rust program | [Rust](#rust) |
| Put Trit into a C, C++, Go, Swift, or Zig program | [C ABI](#c-abi) |
| Put Trit into a Python, C#, or Node program | [Python](#python), [C# and .NET](#c-and-net), [Node and Bun](#node-and-bun) |
| Select a model | [Select a model](#select-a-model) |
| Make speech from text | [Text to speech](#text-to-speech) |
| Know what every integration must do | [The contract](#the-contract) |

## Quick start

```powershell
git clone https://github.com/Hi9841/trit
cd trit
cargo build --release                       # the CLI and the Rust library
hf download moondream/parakeet-redux config.json tokenizer.json model.safetensors --local-dir models/parakeet-redux
./target/release/trit transcribe take.wav
```

```text
model: ~/tdt/trit/models/parakeet-redux
files: 1 transcribed, 0 failed, 6.0 s of audio in 0.04 s (172x realtime)
results[1]{file,seconds,ms,text}:
  take.wav,6.0,35,"Restart the Kubernetes pod, then check the PostgreSQL logs."
```

To get only the command line, without the source folder:
`cargo install --git https://github.com/Hi9841/trit` puts `trit` on your PATH.

`trit doctor` tells you what is missing and how to fix it. `trit models`
lists the models, where they are on this computer, and the download command
of each.

## The contract

Every integration, in every language, follows these rules.

| Topic | Rule |
|---|---|
| Audio | 16 kHz, mono, `f32` samples in -1..1. Trit does not resample. Convert other audio first: `ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav`. `read_wav` and the CLI read 16 kHz WAV (16-bit PCM or 32-bit float) and mix channels to mono. |
| Load once | Loading takes 0.2 s (Redux) to 5 s (the large models). Load one engine when the program starts and keep it. |
| Threads | `Trit` and `Cleaner` are `Send + Sync`. Share one engine between threads (`Arc` in Rust). Calls from several threads are safe, but they share one worker team, so together they are not faster than one after another. |
| Model | A model is a Hugging Face folder with `config.json` and `model.safetensors`. Give a folder, a name from the [catalog](#models), or nothing (the default is `parakeet-redux`). |
| Output | Plain text. An empty string means no speech. `segments` gives timed parts. |
| Errors | Rust returns `Err(String)` with the cause and the fix. The C ABI returns NULL or -1; read `trit_last_error()`. The bindings raise `TritError` (Python, Node) or `TritException` (C#). |
| Memory | Trit keeps a buffer pool (up to 16 MB) and scratch per thread for the next call. Call `release_memory()` when the program goes idle. |
| Language | `en` for English apps. Without a language, multilingual Whisper detects it. See [Languages](#languages). |

## Select a model

Measured on 456 English clips (LibriSpeech test-clean and test-other,
developer terms, everyday dictation), AMD Ryzen 9 7950X3D, Trit's default
settings. Full results: [docs/engine-comparison.md](docs/engine-comparison.md).

| Model | Size | Word error rate | Time per clip | Memory | Use it for |
|---|---:|---:|---:|---:|---|
| `parakeet-redux` | 178 MB | 3.37 % | 36 ms | 627 MB | The default: small, fast, accurate. TDT uses it. |
| `parakeet-tdt-0.6b-v3` | 2.5 GB | **2.38 %** | 39 ms | 1187 MB | The fewest errors. 25 European languages. |
| `moonshine-tiny` | 108 MB | 6.41 % | 37 ms | 239 MB | The smallest English model. |
| `moonshine-base` | 246 MB | 4.78 % | 62 ms | 271 MB | Small English model, fewer errors than tiny. |
| `whisper-tiny` | 151 MB | 8.75 % | 67 ms | 229 MB | 99 languages in a small model. |
| `whisper-base.en` | 290 MB | 5.39 % | 117 ms | 293 MB | Whisper for English. |

Time per clip is the median on LibriSpeech test-clean (clips of 2 to 35 s).
Memory is the working set after all clips.

## Command line

`trit` is built for agents ([AXI](https://axi.md)): TOON output on stdout,
errors on stdout as `error:` and `help:` lines, exit 0 (success), 1 (error),
2 (usage error), and no prompts. `trit` with no arguments shows the live state.
The agent skill for the CLI is [skill/SKILL.md](skill/SKILL.md).

| Command | What it does |
|---|---|
| `trit` | Home view: the binary, the models, the CPU path, next commands |
| `trit transcribe <path>...` | Text of WAV files, or of every WAV file in a folder |
| `trit clean "<text>"` | A raw transcript as clean written English (S1-mini) |
| `trit speak "<text>"` | Speech of the text as a WAV file (Kokoro or MMS-TTS); see [Text to speech](#text-to-speech) |
| `trit phonemes "<text>"` | The phonemes Kokoro speaks for English text |
| `trit voices` | The voices of a Kokoro folder |
| `trit models` | The models Trit runs, where they are, how to get them |
| `trit doctor` | Where Trit looks for models, and the fix for each problem |
| `trit skill [--check <file>]` | The agent skill, made from the help text |
| `trit --version` | The version |

Flags of `transcribe`:

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir or name>` | parakeet-redux | The speech model |
| `--language <code>` | detect | ISO 639-1 code (`en`, `de`, `fr`) |
| `--dictionary <file>` | none | Terms to favor, one per line (not CTC models: they skip every term) |
| `--task transcribe\|translate` | `transcribe` | `translate` writes English (multilingual Whisper, Canary) |
| `--timestamps` | off | Timed segments too: sentences for Parakeet, windows for the others |
| `--precision auto\|int8\|f16\|f32` | `auto` | How the weights are held; see [Precision](#precision) |
| `--beam <n>` | 5 for Whisper and Moonshine, 1 for Canary | Beams; 1 is greedy, at most 16 |
| `--cleanup` | off | Clean each transcript with S1-mini; `--cleanup-model`, `--styling`, `--structure`, `--context` as for `clean` |
| `--format toon\|json\|text` | `toon` | `json` for programs; `text` gives one transcript per line |
| `--full` | off | Do not cut transcripts over 500 characters in a list of several files |

```powershell
trit transcribe clips/ --model whisper-large-v3-turbo --format json
trit transcribe interview.wav --model parakeet-tdt-0.6b-v3 --language de
trit transcribe take.wav --dictionary terms.txt --cleanup --format text
trit transcribe meeting.wav --timestamps --precision f32
trit clean "so um send the report by friday no wait thursday"
trit speak "The build passed. Ship it." --out done.wav
```

A program that cannot load a library can run the CLI as a subprocess with
`--format json` and read stdout.

## Rust

Trit is not on crates.io. Add it from git (or by path in a checkout):

```toml
[dependencies]
trit = { git = "https://github.com/Hi9841/trit" }
```

```rust
use std::path::Path;

let dir = trit::find_model(None).ok_or_else(|| trit::missing_model(None))?;  // parakeet-redux
let engine = trit::Trit::load(&dir)?;                    // load once, keep it
let samples = trit::read_wav(Path::new("take.wav"))?;   // 16 kHz mono f32
let text = engine.transcribe_with(&samples, &trit::Options::language("en"));

for segment in engine.segments(&samples, &trit::Options::language("en")) {
    println!("{:.2}-{:.2} {}", segment.start, segment.end, segment.text);
}
let skipped = engine.set_dictionary(&["Kubernetes".into()], trit::BiasWeights::default());
trit::release_memory();                                  // when the app goes idle
```

| Item | Meaning |
|---|---|
| `Trit::load(dir) -> Result<Trit, String>` | Loads a model folder; the family comes from `config.json` |
| `Trit::load_with(dir, &LoadOptions { precision })` | The same with a `Precision`: `Auto` (default), `Int8`, `F16`, `F32` |
| `Trit::transcribe(&self, &[f32]) -> String` | Text; the model detects the language |
| `Trit::transcribe_with(&self, &[f32], &Options) -> String` | Text with `Options { language, task, beam, fallback }` |
| `Trit::segments(&self, &[f32], &Options) -> Vec<Segment>` | Timed parts: `start`, `end` in seconds, `text` |
| `Trit::kind(&self) -> &str` | `parakeet_tdt`, `parakeet_ctc`, `canary`, `whisper`, or `moonshine` |
| `Trit::set_dictionary(&self, &[String], BiasWeights) -> Vec<String>` | Terms to favor; returns the terms the model cannot spell |
| `release_memory()` | Gives memory that earlier calls freed back to the system |
| `Cleaner::load(dir)`, `Cleaner::load_with(dir, &LoadOptions)`, `Cleaner::clean(&self, &str, &Style) -> String` | S1-mini transcript cleanup |
| `find_model(Option<&Path>)`, `find_cleanup_model(Option<&Path>)` | The model search in [Models](#models) |
| `missing_model(Option<&Path>) -> String` | Why `find_model` found nothing, with the fix |
| `read_wav(&Path) -> Result<Vec<f32>, String>` | A 16 kHz WAV file as mono samples |
| `Voice::load(dir) -> Result<Voice, String>` | Loads a voice: Kokoro-82M or VITS (MMS-TTS) |
| `Voice::speak(&self, &str) -> Vec<f32>` | Speech of the text, mono at `Voice::sample_rate()` (24000 Kokoro, 16000 VITS) |
| `Voice::speak_with(&self, &str, &Voicing) -> Result<Vec<f32>, String>` | The same with `Voicing { voice, phonemes, speed, noise, noise_duration, seed }` |
| `Voice::speak_stream(&self, &str, &Voicing, &mut dyn FnMut(&[f32]) -> bool)` | The same, part by part; return false to stop |
| `Voice::phonemes(&self, &str, &Voicing) -> Result<String, String>` | The phonemes Kokoro speaks, one line per pass |
| `Voice::voices(&self) -> Vec<String>`, `Voice::kind(&self) -> &str` | The voices of a Kokoro folder; `kokoro` or `vits` |
| `Voice::reads(&self, &str) -> String` | What the voice reads of the text |
| `phonemize(&str, british, fallback) -> Result<String, String>` | English phonemes without a voice |
| `find_voice(Option<&Path>)`, `missing_voice(Option<&Path>)`, `is_voice(&Path)` | The voice search, and why it found nothing |
| `write_wav(&Path, &[f32], rate) -> Result<(), String>` | Mono samples as a 16-bit WAV file |
| `CATALOG`, `known(name)`, `candidates(repo)` | The model catalog and the places of a repository |
| `VOICE_REPO`, `VOICE_REPOS` | The default voice (`hexgrad/Kokoro-82M`), and the search order (Kokoro, MMS-TTS) |
| `cpu_path() -> &str`, `threads(Spread) -> usize` | The kernel paths in use; the worker count |
| `SAMPLE_RATE`, `MODEL_TYPES`, `VERSION` | Constants |

## C ABI

Build the library: `cargo build --release -p trit-ffi`. It gives
`trit_ffi.dll` (Windows), `libtrit_ffi.so` (Linux), `libtrit_ffi.dylib`
(macOS), and a static library in `target/release`. The header is
[include/trit.h](include/trit.h). A complete program is
[examples/transcribe.c](examples/transcribe.c).

```c
#include "trit.h"

Trit *engine = trit_load("parakeet-redux");                 /* folder, name, or NULL */
if (!engine) { printf("error: %s\n", trit_last_error()); return 1; }
char *text = trit_transcribe_wav(engine, "take.wav", "en");
if (text) { puts(text); trit_free_text(text); }
trit_free(engine);
```

| Function | Returns |
|---|---|
| `trit_load(const char *model)` | An engine, or NULL. A folder, a name, or NULL for the default |
| `trit_load_with(const char *model, const char *precision)` | The same with `"auto"` (or NULL), `"int8"`, `"f16"`, `"f32"` |
| `trit_kind(engine)` | The family, static |
| `trit_transcribe(engine, const float *samples, size_t count, const char *language)` | Text, or NULL |
| `trit_transcribe_with(engine, samples, count, language, int beam)` | The same with beams (0: the model's default, 1: greedy, at most 16) |
| `trit_transcribe_wav(engine, const char *path, const char *language)` | Text, or NULL |
| `trit_transcribe_json(engine, samples, count, const char *options)`, `trit_transcribe_wav_json(engine, path, options)` | `{"text", "segments": [{"start", "end", "text"}]}` as JSON, or NULL. `options` is NULL or a JSON object with any of `language`, `task` (`"transcribe"`, `"translate"`), `beam`, `fallback`, `timestamps` |
| `trit_set_dictionary(engine, const char *terms)` | Terms skipped, or -1. One term per line; a term over 100 characters is skipped |
| `trit_cleaner_load(const char *model)`, `trit_cleaner_load_with(model, precision)` | A cleaner, or NULL |
| `trit_clean(cleaner, text, styling, structure, context)` | Clean text, or NULL. NULL settings take the default |
| `trit_voice_load(const char *model)` | A voice, or NULL. A folder, a name, or NULL for `kokoro-82m`, then `mms-tts-eng` |
| `trit_voice_sample_rate(voice)`, `trit_voice_kind(voice)` | The rate of the samples (24000 or 16000), or -1; `"kokoro"` or `"vits"` |
| `trit_voice_list(voice)`, `trit_phonemes(voice, text, options)` | The voice names, one per line; the phonemes of the text. Free with `trit_free_text` |
| `trit_speak(voice, const char *text, const char *options, size_t *count)` | Mono floats and their count, or NULL. `options` is NULL or a JSON object with any of `voice`, `phonemes`, `speed`, `noise`, `noise_duration`, `seed` |
| `trit_speak_stream(voice, text, options, TritSamplesFn callback, void *user_data)` | 0, or -1. The callback gets each part; it returns 1 to go on, 0 to stop |
| `trit_speak_wav(voice, text, const char *path, options)` | 0, or -1. A 16-bit mono WAV file |
| `trit_release_memory(void)` | Gives memory that earlier calls freed back to the system |
| `trit_last_error(void)` | The reason for the last failure on this thread |
| `trit_free_text(char *)`, `trit_free(Trit *)`, `trit_cleaner_free(TritCleaner *)`, `trit_voice_free(TritVoice *)`, `trit_free_samples(float *, size_t count)` | Free. NULL is a no-op |
| `trit_version(void)` | `"0.3.2"` |

Rules: strings are UTF-8. `language` is an ISO 639-1 code or NULL. Free
every returned text with `trit_free_text`, and the samples of `trit_speak`
with `trit_free_samples`. An unknown option fails with the
list of valid ones. No panic crosses the ABI; it
becomes NULL or -1 and an error text.

Other languages with a C FFI use the header as it is: Go (cgo), Swift,
Kotlin and Java (Panama or JNA), Zig, Ruby (FFI gem), Dart (dart:ffi), Lua
(LuaJIT FFI).

## Python

[bindings/python/trit.py](bindings/python/trit.py) is one file with no
dependencies (ctypes). Copy it next to your code. It finds the library at
`TRIT_LIB`, next to `trit.py`, or in `trit/target/release`.

```python
import trit

engine = trit.Trit("parakeet-redux")              # folder, name, or None; precision="f16" and so on
print(engine.kind)                                # "parakeet_tdt"
print(engine.transcribe_wav("take.wav", language="en"))
print(engine.transcribe(samples, language="en"))  # list, array("f"), or numpy float32, 16 kHz mono
print(engine.transcribe(samples, beam=1))         # greedy
print(engine.transcribe_wav("take.wav", task="translate"))  # into English (Whisper, Canary)
for s in engine.segments_wav("take.wav", language="en"):    # also segments(samples)
    print(s.start, s.end, s.text)                 # seconds from the start
engine.set_dictionary(["Kubernetes", "PostgreSQL"])
trit.release_memory()                             # when the app goes idle

cleaner = trit.Cleaner()                          # S1-mini
print(cleaner.clean("so um send it friday no wait thursday", context="email"))

voice = trit.Voice()                              # kokoro-82m, else mms-tts-eng
voice.speak_wav("The build passed on March 3rd.", "done.wav", voice="bf_emma")
samples = voice.speak("Ship it.", speed=1.1, seed=3)  # array("f") at voice.sample_rate
for part in voice.stream(long_text):              # each pass as soon as it is made
    play(part)
print(voice.voices, voice.phonemes("I read it."))
```

Errors raise `trit.TritError` with the cause and the fix. `Trit`,
`Cleaner`, and `Voice` are context managers.

## C# and .NET

Add [bindings/csharp/Trit.cs](bindings/csharp/Trit.cs) to the project, set
`<AllowUnsafeBlocks>true</AllowUnsafeBlocks>`, and copy `trit_ffi.dll` next
to the program. .NET 7 or later.

```csharp
using TritSpeech;

using var engine = new Trit("parakeet-redux");           // second argument: precision, "f16" and so on
Console.WriteLine(engine.TranscribeWav("take.wav", "en"));
Console.WriteLine(engine.Transcribe(samples, "en"));     // float[] or Span<float>, 16 kHz mono
Console.WriteLine(engine.TranscribeWav("take.wav", task: "translate"));  // into English
foreach (var s in engine.SegmentsWav("take.wav", "en"))  // also Segments(samples)
    Console.WriteLine($"{s.Start} {s.End} {s.Text}");    // seconds from the start
engine.SetDictionary("Kubernetes", "PostgreSQL");

using var cleaner = new TritCleaner();
Console.WriteLine(cleaner.Clean("so um send it friday no wait thursday"));

using var voice = new TritVoice();                       // kokoro-82m, else mms-tts-eng
voice.SpeakWav("The build passed on March 3rd.", "done.wav", voice: "bf_emma");
float[] samples = voice.Speak("Ship it.", speed: 1.1f);  // at voice.SampleRate
voice.SpeakStream(longText, part => { Play(part); return true; });
```

Errors throw `TritException`. The options of each call are `language`,
`beam`, `task`, and `fallback`, as in the C ABI.

## Node and Bun

[bindings/node/trit.mjs](bindings/node/trit.mjs) uses
[koffi](https://koffi.dev): `npm install koffi` or `bun add koffi`.

```js
import { Trit, Cleaner, Voice, releaseMemory } from "./trit.mjs";

const engine = new Trit("parakeet-redux");             // second argument: { precision: "f16" }
console.log(engine.transcribeWav("take.wav", "en"));
console.log(engine.transcribe(samples, "en"));         // Float32Array, 16 kHz mono
console.log(engine.transcribeWav("take.wav", null, { task: "translate" }));  // into English
for (const s of engine.segmentsWav("take.wav", "en"))  // also segments(samples)
  console.log(s.start, s.end, s.text);                 // seconds from the start
engine.setDictionary(["Kubernetes", "PostgreSQL"]);
engine.close();
releaseMemory();

const cleaner = new Cleaner();
console.log(cleaner.clean("so um send it friday no wait thursday", { styling: "formal" }));
cleaner.close();

const voice = new Voice();                             // kokoro-82m, else mms-tts-eng
voice.speakWav("The build passed on March 3rd.", "done.wav", { voice: "bf_emma" });
const samples = voice.speak("Ship it.", { speed: 1.1 });  // Float32Array at voice.sampleRate
voice.speakStream(longText, (part) => { play(part); return true; });
voice.close();
```

Errors throw `TritError`.

## Models

Trit reads `config.json` and `model.safetensors` of a Hugging Face folder and
selects the network from `model_type`. These checkpoints have been run; other
checkpoints of the same families load the same way.

| Name | Repository | Family | Languages | MB |
|---|---|---|---|---:|
| parakeet-redux | moondream/parakeet-redux | Parakeet TDT, 1.58-bit | 25 European | 178 |
| parakeet-tdt-0.6b-v3 | nvidia/parakeet-tdt-0.6b-v3 | Parakeet TDT | 25 European | 2508 |
| parakeet-ctc-0.6b | nvidia/parakeet-ctc-0.6b | Parakeet CTC | English | 2436 |
| parakeet-rnnt-0.6b | nvidia/parakeet-rnnt-0.6b | Parakeet RNNT | English | 2467 |
| canary-1b-v2 | nvidia/canary-1b-v2 | Canary (FastConformer + transformer decoder) | 25 European | 3916 |
| whisper-tiny | openai/whisper-tiny | Whisper | 99 | 151 |
| whisper-base.en | openai/whisper-base.en | Whisper | English | 290 |
| whisper-small.en | openai/whisper-small.en | Whisper | English | 967 |
| whisper-large-v3-turbo | openai/whisper-large-v3-turbo | Whisper | 99 | 1618 |
| whisper-large-v3 | openai/whisper-large-v3 | Whisper | 99 | 3087 |
| distil-large-v3.5 | distil-whisper/distil-large-v3.5 | Whisper | English | 3026 |
| moonshine-tiny | UsefulSensors/moonshine-tiny | Moonshine | English | 108 |
| moonshine-base | UsefulSensors/moonshine-base | Moonshine | English | 246 |
| wav2vec2-base-960h | facebook/wav2vec2-base-960h | wav2vec 2.0 (CTC) | English | 378 |
| qwen3-asr-0.6b | Qwen/Qwen3-ASR-0.6B | Qwen3-ASR | 30 | 1876 |
| s1-mini | superwhisper/s1-mini | Qwen3 text model (cleanup) | English | 1503 |
| kokoro-82m | hexgrad/Kokoro-82M | Kokoro voice (text to speech), 28 English voices | English (US, GB) | 327 |
| graphemes_to_phonemes_en_us, _en_gb | PeterReid/graphemes_to_phonemes_en_us, _gb | The phonemizer's fallback for unknown words | English | 3 |
| mms-tts-eng | facebook/mms-tts-eng | VITS voice (text to speech) | English | 145 |

Download only the files Trit reads:

```powershell
hf download openai/whisper-large-v3-turbo config.json generation_config.json tokenizer.json model.safetensors --local-dir models/whisper-large-v3-turbo
```

Trit looks for a model in this order:

1. The folder you give, or a name from the table.
2. `TRIT_MODEL` (when no model is named).
3. `models/<name>` and `../models/<name>` from the current folder.
4. `models/<name>` next to the executable, and two and three folders up.
5. `%LOCALAPPDATA%/TDT/models/<name>`.
6. The Hugging Face cache: `~/.cache/huggingface/hub/models--<owner>--<name>/snapshots/*`.

`trit doctor` prints each place and its status.

## Languages

| Family | Languages | `language` option |
|---|---|---|
| Whisper (multilingual) | 99 | Forces that language; without it, the model detects the language from the first 30 s |
| Whisper `.en`, Distil large-v3.5 | English | Ignored |
| Parakeet TDT v3, Redux | 25 European | Cannot be forced; `en` keeps the output to English tokens |
| Parakeet CTC, RNNT | English | `en` as above |
| wav2vec 2.0 | English | Ignored |
| Qwen3-ASR | 30 | Forces that language; without it, the model names the language it hears |
| Canary 1B v2 | 25 European | The spoken language; English when none is given (Canary does not detect it) |
| Moonshine | English | Ignored |

## Transcript cleanup

[Superwhisper S1-mini](https://huggingface.co/superwhisper/s1-mini) is a
0.6B text model (Qwen3) that turns the raw text of any speech model into
written English: fillers removed, self-corrections resolved ("friday no
wait thursday" becomes "Thursday"), punctuation, casing, numbers, dates,
times, currency, and email addresses. Trit sends the exact system prompt
and control line the model was trained on. About 200 ms for one sentence.

| Setting | Values |
|---|---|
| styling | `casual`, `semi-casual`, `semi-formal` (default), `formal` |
| structure | `prose` (default), `lists` |
| context | `general` (default), `email` |

Input that is only fillers gives an empty string. English only.

## Text to speech

Trit speaks with two kinds of voice:

| Voice | Model | Rate | Reads | Voices |
|---|---|---|---|---|
| `kokoro-82m` (the default) | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M): StyleTTS 2 with an iSTFTNet vocoder, 82 M parameters | 24 kHz | English text: words, numbers, money, times, dates, symbols | 28: `af_`/`am_` American, `bf_`/`bm_` British |
| `mms-tts-eng` | [Meta MMS-TTS](https://huggingface.co/facebook/mms-tts-eng): VITS, 36 M parameters | 16 kHz | Letters only | 1 (the other MMS languages load too) |

```powershell
hf download hexgrad/Kokoro-82M config.json kokoro-v1_0.pth voices/af_heart.pt voices/bf_emma.pt --local-dir models/kokoro-82m
hf download PeterReid/graphemes_to_phonemes_en_us config.json generation_config.json model.safetensors --local-dir models/graphemes_to_phonemes_en_us
./target/release/trit speak "The build passed on March 3rd, and it cost $3.50." --out done.wav
./target/release/trit speak "Good morning from London." --voice bf_emma
./target/release/trit voices
```

`trit models` prints the command that gets all 28 English voices.

### How Kokoro reads text

Kokoro speaks phonemes. Trit's phonemizer (`src/tts/english/`) is a port of
[misaki](https://github.com/hexgrad/misaki) 0.9.4, the phonemizer that the
`kokoro` package uses, function for function:

- **Lexicons:** misaki's gold and silver dictionaries, American and British
  (about 180,000 words each), with misaki's rules for -s, -ed, and -ing,
  stress, and capitals. They ship in the binary, deflated (4 MB).
- **Numbers and symbols:** cardinals, ordinals, years, decimals, money in
  dollars, pounds, and euros, times, `%`, `&`, `+`, `@`, and addresses
  ("take dot wav"), as misaki reads them with num2words.
- **Tags:** misaki asks spaCy which words are verbs, nouns, and so on
  ("read" and "read", "the object" and "to object"). Trit tags with NLTK's
  averaged perceptron, on Penn Treebank tokens (`ca n't`), with a few
  corrections for orders and addresses.
- **Unknown words:** a small BART (`PeterReid/graphemes_to_phonemes_en_us`
  and `_gb`, the fallback of misaki's main branch) gives phonemes to a word
  that no dictionary holds. Without it, such words are not spoken.
- **Your own pronunciation:** misaki's markup works: `[Kokoro](/kˈOkəɹO/)`
  gives a word its phonemes, `[record](-1)` lowers its stress.
- **Passes:** Kokoro reads at most 510 phonemes in one pass. Trit cuts the
  text at line breaks, then at sentence ends, as the `kokoro` package does.

`trit phonemes "<text>"` and `Voice::phonemes` show the phonemes; `--phonemes`
(`Voicing::phonemes`) speaks phonemes that you give.

### Settings

| Setting | Default | Meaning |
|---|---|---|
| `voice` | `af_heart` | Kokoro voice; `"af_bella,af_sky"` is the mean of two |
| `speed` | 1 | Speaking rate: 1.25 is a quarter faster |
| `noise` | Kokoro 1, VITS 0.667 | Kokoro: the noise of the vocoder's source. VITS: `noise_scale`, how much the voice varies; 0 is the flattest |
| `noise_duration` | 0.8 | VITS only: how much the timing varies |
| `seed` | 0 | The seed of the noise: one text and one seed give the same audio |
| `phonemes` | false | Kokoro: the text is phonemes, one pass per line |

`Voice::speak_stream` (`trit_speak_stream`, `Voice.stream` in Python,
`SpeakStream` in C#, `speakStream` in Node) gives each part as soon as it is
made: a pass for Kokoro, a sentence for VITS. Play can start before the
rest is spoken.

### Same as the reference

`bench/tts.py` checks each voice against its reference runtime, on the 56
sentences of `bench/corpus.txt` and `bench/tech-corpus.txt` (859 words).
Both runtimes speak; Parakeet Redux transcribes both.

- **Kokoro, the engine:** the `kokoro` package's `KModel` and Trit speak the
  same phonemes with the same noise (Trit's noise, reproduced in the bench).
  Every clip has the same length; the median SNR is 38 dB. That is float
  rounding, which the model amplifies: against exact float64 math, Trit is
  as close as PyTorch's own float32 (25.7 against 25.4 dB on the longest
  sentence, 37.0 against 36.6 dB on a short one).
- **Kokoro, the round trip:** text to speech with each full pipeline
  (`kokoro` with misaki, spaCy, and espeak; Trit), then transcribed: 70, 70,
  and 70 word errors with Trit over three seeds, 71, 71, and 70 with the
  reference.
- **The phonemizer:** given the same tokens, Trit's phonemes are misaki's,
  character for character (451 test lines with numbers, money, and markup;
  American and British). With its own tagger and the BART for unknown words,
  3.0 % of the phoneme words differ from misaki with spaCy and espeak.
- **VITS:** with the noise off, the audio is the same to 76 dB or more
  (near the limit of 16-bit WAV) on the AVX-512, AVX2, and scalar paths. The
  round trip gives 167 word errors with Trit and 171 with Transformers
  (noise off), and 536 against 553 (noise on, three seeds). Both runtimes
  get the same ids: Transformers' tokenizer drops the blanks around a "k"
  at the very start or end of a text, which MMS was not trained with.

### Limits

- Kokoro's other languages (Spanish, French, Hindi, Italian, Japanese,
  Portuguese, Chinese) need their own phonemizers. Their voices load; give
  them phonemes with `phonemes`.
- The tagger is not spaCy's. A word whose sound changes with its role
  ("read", "record", "that") can get the other sound. Mark it:
  `[read](/ɹˈɛd/)`.
- Trit does not apply Unicode NFKC to the text, as misaki does; ASCII and
  common text are the same under it.
- A VITS voice reads letters only (MMS English: a to z, digits 0 to 6,
  apostrophe, hyphen); write numbers as words. Sentences get 0.3 s of
  silence between them. One-speaker MMS checkpoints that read letters
  (`phonemize: false`) load; a language with `is_uroman: true` needs
  romanized text.
- Text to speech is the one part of Trit with lists of words: the
  lexicons, the number words, and the tagger's weights. A voice needs them
  to say text. Speech to text has none.

## Precision

`Precision` (`--precision`, `trit_load_with`, `precision=`) sets how a model
holds its dense layers and token tables. Parakeet Redux keeps its ternary
weights in every mode.

| Precision | Dense layers | Token tables | Use |
|---|---|---|---|
| `auto` (default) | int8 for layers 384 wide or more, f16 below | int8 storage with f32 math when 1024 wide or more (S1-mini, Canary), else f16 | The fastest; on the test clips as accurate as f32 or more |
| `int8` | int8 everywhere (int8 storage with f32 math where the int8 kernels cannot take the shape) | int8 storage, f32 math | The least memory; narrow models lose some accuracy |
| `f16` | f16 weights, f32 math | f16 | Half the memory of f32 |
| `f32` | f32, as the reference runtimes hold them | f32 | Checks against the reference |

## Behavior and limits

- **Long audio:** Trit cuts long audio at the quietest 100 ms near the end of
  each window: 30 s for Whisper and Moonshine, 40 s for Canary, 60 s for
  Parakeet.
- **Decoding:** Whisper and Moonshine use beam search with 5 beams, by the
  rules of Transformers' `generate(num_beams=5)`. All beams go through each
  decoder step as one batch, so beam 5 costs about 1.3 to 2.2 times the
  greedy time. Parakeet and Canary decode greedy. `beam = 1` makes every
  family greedy. Whisper decodes a window again at temperatures 0.2 to 1.0
  when its text repeats itself or is unlikely, as OpenAI Whisper does
  (`Options::fallback`).
- **Dictionary:** Parakeet TDT decodes greedy, then finds runs of words
  spelled close to a term ("Perakeet", "trip run time", "wire shark") and
  scores the words and the term over the same frames with the joint
  network. The term replaces the words when it loses by less than 4 + 2
  log-probability units per letter of the term, less for a far spelling:
  the decoder finds long names unlikely. Canary and Moonshine raise the
  logits of the terms' tokens while they decode; with terms set they
  decode greedy. Whisper puts the terms in a previous-text prompt, as
  OpenAI's `initial_prompt`.
- **English:** with `language: "en"`, Parakeet never writes pieces that
  only other languages use.
- **Timestamps:** `segments` gives sentences or pauses of 1 s for Parakeet,
  and one segment per window for the others.
- **Silence:** Parakeet returns an empty string under 80 ms of audio, and
  Redux when its speech head finds no 0.1 s of speech. Whisper and Moonshine
  skip a window under -60 dBFS, and Whisper drops a window that the model
  rates as no speech.
- **Threads:** Parakeet and S1-mini use one pinned worker per physical core
  of the largest L3 domain; Whisper and the voice use every physical core. Workers spin
  for 3 ms after a step, then sleep, so an idle engine uses no CPU. `TRIT_THREADS` sets the
  count.
- **CPU:** x86-64 with AVX-512 or AVX2 (with F16C) is fast; int8 also uses
  AVX-VNNI. Other CPUs, ARM included, use the scalar path: the same text,
  much slower. Only x86-64 Windows has been measured.
- **Whisper** always encodes a full 30 s window, so a short clip costs as
  much as a long one.

## Environment variables

| Variable | Effect |
|---|---|
| `TRIT_MODEL` | The speech model folder, when the caller names none |
| `TRIT_CLEANUP_MODEL` | The S1-mini folder, when the caller names none |
| `TRIT_VOICE` | The voice folder, when the caller names none (else Kokoro, then MMS-TTS) |
| `TRIT_THREADS` | The worker count; turns pinning off |
| `TRIT_ISA` | `scalar`, `avx2`, or `avxvnni` forces another kernel path, for tests |
| `TRIT_LIB` | The library path, for the Python and Node bindings |

## Layout of the code

| Folder or file | Part |
|---|---|
| `src/lib.rs` | Public API, model search |
| `src/main.rs` | The `trit` CLI |
| `src/catalog.rs` | The model catalog |
| `src/cleanup.rs` | `Cleaner`: S1-mini prompt and styles |
| `src/tts/` | Text to speech: `Voice` and `Voicing` (`mod.rs`), Kokoro (`kokoro/`), VITS (`vits.rs`), the English phonemizer (`english/`: tokenizer, tagger, lexicon, numbers, BART fallback, data) |
| `src/cpu/` | Kernels and threads: f32/f16 GEMM (`fgemm.rs`), dense layers and `Precision` (`gemm.rs`), int8 (`int8.rs`), ternary (`kernels.rs`), f16 matvec (`half.rs`), worker teams (`team.rs`), safetensors reader (`tensors.rs`), PyTorch checkpoint reader (`torch_file.rs`) |
| `src/audio/` | WAV reader and writer, STFT and its inverse (`stft.rs`), NeMo log-mel (`mel.rs`), Whisper log-mel (`whisper_mel.rs`), windows for long audio |
| `src/text/` | SentencePiece tokenizer (Parakeet, Moonshine), byte-level BPE (Whisper, Qwen) |
| `src/models/mod.rs` | `FAMILIES`: one row per `model_type` (its loader and thread spread), and the `Family` trait |
| `src/models/config.rs` | The `config.json` reader that every family uses |
| `src/models/layers.rs` | Shared layers: norms, GELU, rotary positions, attention, conv columns |
| `src/models/parts/` | Building blocks that the config picks: the FastConformer encoder (`fastconformer.rs`), the TDT and RNNT head (`transducer.rs`), CTC (`ctc.rs`), beam search (`beam.rs`), the Qwen3 language model (`qwen3.rs`), the sound-checked dictionary (`dictionary.rs`), the logit dictionary (`logit_bias.rs`), and 1-D convolutions: dilated, depthwise, transposed (`conv.rs`) |
| `src/models/families/` | One file per checkpoint family. It reads the family's files and wires the parts: Parakeet, Canary, Whisper, Moonshine, wav2vec 2.0, Qwen3-ASR, and the VITS voice (`vits.rs`) |
| `ffi/` | `trit-ffi`: the C ABI |
| `bindings/` | Python, C#, Node |

## License

Trit is under the [PolyForm Noncommercial License 1.0.0](LICENSE). You can
use, change, and share it for any noncommercial purpose. You cannot sell it
or use it to make money.

Each model has its own license: Parakeet and Canary CC-BY-4.0, Whisper and
Moonshine MIT, S1-mini Apache-2.0 with a naming clause, MMS-TTS
CC-BY-NC-4.0 (noncommercial), Kokoro-82M and the G2P fallback Apache-2.0.
Follow it when you ship the model. The phonemizer's lexicons (misaki, Apache-2.0) and
tagger (MIT) ship in Trit: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
