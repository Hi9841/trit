# Trit

Trit is a speech-to-text and text-to-speech engine for the CPU. It runs
Parakeet, Canary, Whisper, and Moonshine checkpoints as published on Hugging
Face, with its own kernels: no Python, no PyTorch, no ONNX Runtime, no GPU.
16 kHz audio in, text out. It can also clean a raw transcript with
Superwhisper S1-mini, and speak text with a VITS voice (Meta MMS-TTS).

TDT, the dictation app in this repository, uses Trit for all of its speech.
Trit is free for noncommercial use only (see [License](#license)).
The website, with benchmarks and a comparison: <https://hi9841.github.io/tdt/>.

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
git clone https://github.com/Hi9841/tdt
cd tdt/trit
cargo build --release                       # the CLI and the Rust library
hf download moondream/parakeet-redux --include "*.json" model.safetensors --local-dir models/parakeet-redux
./target/release/trit transcribe take.wav
```

```text
model: ~/tdt/trit/models/parakeet-redux
files: 1 transcribed, 0 failed, 6.0 s of audio in 0.04 s (172x realtime)
results[1]{file,seconds,ms,text}:
  take.wav,6.0,35,"Restart the Kubernetes pod, then check the PostgreSQL logs."
```

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
settings. Full results: [docs/engine-comparison.md](../docs/engine-comparison.md).

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
| `trit speak "<text>"` | Speech of the text as a WAV file (MMS-TTS); see [Text to speech](#text-to-speech) |
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
| `--beam <n>` | 5 for Whisper and Moonshine, 1 for Canary | Beams; 1 is greedy |
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
trit = { git = "https://github.com/Hi9841/tdt" }
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
| `Voice::load(dir) -> Result<Voice, String>` | Loads a VITS voice (MMS-TTS) |
| `Voice::speak(&self, &str) -> Vec<f32>` | Speech of the text, mono at `Voice::sample_rate()` |
| `Voice::speak_with(&self, &str, &Voicing) -> Result<Vec<f32>, String>` | The same with `Voicing { speed, noise, noise_duration, seed }`; fails on a speed of 0 or less |
| `Voice::reads(&self, &str) -> String` | The text as the voice reads it (what is not there is not spoken) |
| `find_voice(Option<&Path>)`, `missing_voice(Option<&Path>)` | The voice search, and why it found nothing |
| `write_wav(&Path, &[f32], rate) -> Result<(), String>` | Mono samples as a 16-bit WAV file |
| `CATALOG`, `known(name)`, `candidates(repo)` | The model catalog and the places of a repository |
| `VOICE_REPO` | The default voice, `facebook/mms-tts-eng` |
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
| `trit_transcribe_with(engine, samples, count, language, int beam)` | The same with beams (0: the model's default, 1: greedy) |
| `trit_transcribe_wav(engine, const char *path, const char *language)` | Text, or NULL |
| `trit_transcribe_json(engine, samples, count, const char *options)`, `trit_transcribe_wav_json(engine, path, options)` | `{"text", "segments": [{"start", "end", "text"}]}` as JSON, or NULL. `options` is NULL or a JSON object with any of `language`, `task` (`"transcribe"`, `"translate"`), `beam`, `fallback`, `timestamps` |
| `trit_set_dictionary(engine, const char *terms)` | Terms skipped, or -1. One term per line |
| `trit_cleaner_load(const char *model)`, `trit_cleaner_load_with(model, precision)` | A cleaner, or NULL |
| `trit_clean(cleaner, text, styling, structure, context)` | Clean text, or NULL. NULL settings take the default |
| `trit_voice_load(const char *model)` | A voice, or NULL. A folder, a name, or NULL for `mms-tts-eng` |
| `trit_voice_sample_rate(voice)` | The rate of the samples (16000), or -1 |
| `trit_speak(voice, const char *text, const char *options, size_t *count)` | Mono floats and their count, or NULL. `options` is NULL or a JSON object with any of `speed`, `noise`, `noise_duration`, `seed` |
| `trit_speak_wav(voice, text, const char *path, options)` | 0, or -1. A 16-bit mono WAV file |
| `trit_release_memory(void)` | Gives memory that earlier calls freed back to the system |
| `trit_last_error(void)` | The reason for the last failure on this thread |
| `trit_free_text(char *)`, `trit_free(Trit *)`, `trit_cleaner_free(TritCleaner *)`, `trit_voice_free(TritVoice *)`, `trit_free_samples(float *, size_t count)` | Free. NULL is a no-op |
| `trit_version(void)` | `"0.2.0"` |

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

voice = trit.Voice()                              # mms-tts-eng
voice.speak_wav("The build passed.", "done.wav")
samples = voice.speak("Ship it.", speed=1.1, seed=3)  # array("f") at voice.sample_rate
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

using var voice = new TritVoice();                       // mms-tts-eng
voice.SpeakWav("The build passed.", "done.wav");
float[] samples = voice.Speak("Ship it.", speed: 1.1f);  // at voice.SampleRate
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

const voice = new Voice();                             // mms-tts-eng
voice.speakWav("The build passed.", "done.wav");
const samples = voice.speak("Ship it.", { speed: 1.1 });  // Float32Array at voice.sampleRate
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
| mms-tts-eng | facebook/mms-tts-eng | VITS voice (text to speech) | English | 145 |

Download only the files Trit reads:

```powershell
hf download openai/whisper-large-v3-turbo --include "*.json" model.safetensors --local-dir models/whisper-large-v3-turbo
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

[Meta MMS-TTS](https://huggingface.co/facebook/mms-tts-eng) is a VITS voice:
a transformer reads the letters, a stochastic duration predictor gives each
letter its length, a flow turns a noisy prior into a spectrogram latent, and
HiFi-GAN writes 16 kHz audio. 36 M parameters, f32. Trit runs it as
Transformers' `VitsModel` does, step for step.

```powershell
hf download facebook/mms-tts-eng config.json tokenizer_config.json vocab.json model.safetensors --local-dir models/mms-tts-eng
./target/release/trit speak "Restart the pod, then check the logs." --out pod.wav
```

| Setting | Default | Meaning |
|---|---|---|
| `speed` | 1 (the voice's `speaking_rate`) | Speaking rate: 1.25 is a quarter faster |
| `noise` | 0.667 (the voice's `noise_scale`) | How much the voice varies; 0 is the flattest |
| `noise_duration` | 0.8 (the voice's `noise_scale_duration`) | How much the timing varies |
| `seed` | 0 | The seed of the noise: one text and one seed give the same audio |

- **Letters only:** the voice reads the characters of its `vocab.json`
  (English: a to z, digits 0 to 6, space, apostrophe, hyphen). Trit has no
  phonemizer and no word list. It drops other characters, so write numbers
  and symbols as words. `Voice::reads` and the `reads:` line of
  `trit speak` show what the voice reads.
- **Sentences:** Trit cuts the text after `.`, `!`, `?`, `;`, `:` and at
  line breaks, speaks each part, and puts 0.3 s of silence between parts.
  A part longer than 400 characters is cut at its last comma or space.
- **Blanks:** the id of the blank goes between all characters, as MMS was
  trained. Transformers' tokenizer drops the blanks around a "k" at the
  very start or end of a text; there the two runtimes differ.
- **Same as the reference:** with the noise off, the audio of Trit and of
  Transformers is the same to 76 dB or more (near the limit of 16-bit WAV) on the
  56 sentences of `desktop/bench/corpus.txt` and `tech-corpus.txt`, on the
  AVX-512, AVX2, and scalar paths. Spoken by each runtime and transcribed by
  Parakeet Redux, the sentences give no more word errors with Trit (noise
  off: 167 Trit, 171 Transformers; noise on, three seeds: 536 Trit, 553
  Transformers). `desktop/bench/tts.py` runs this check.
- **Other voices:** any one-speaker MMS-TTS checkpoint that reads letters
  (`phonemize: false`) loads. For a language with `is_uroman: true`, give
  romanized text. Voices with several speakers and VITS models that need a
  phonemizer do not load.

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
| `TRIT_VOICE` | The voice folder, when the caller names none |
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
| `src/speech.rs` | `Voice`: text to speech, sentence cuts, `Voicing` |
| `src/cpu/` | Kernels and threads: f32/f16 GEMM (`fgemm.rs`), dense layers and `Precision` (`gemm.rs`), int8 (`int8.rs`), ternary (`kernels.rs`), f16 matvec (`half.rs`), worker teams (`team.rs`), safetensors reader (`tensors.rs`) |
| `src/audio/` | WAV reader and writer, NeMo log-mel (`mel.rs`), Whisper log-mel (`whisper_mel.rs`), windows for long audio |
| `src/text/` | SentencePiece tokenizer (Parakeet, Moonshine), byte-level BPE (Whisper, Qwen), characters (VITS) |
| `src/models/mod.rs` | `FAMILIES`: one row per `model_type` (its loader and thread spread), and the `Family` trait |
| `src/models/config.rs` | The `config.json` reader that every family uses |
| `src/models/layers.rs` | Shared layers: norms, GELU, rotary positions, attention, conv columns |
| `src/models/parts/` | Building blocks that the config picks: the FastConformer encoder (`fastconformer.rs`), the TDT and RNNT head (`transducer.rs`), CTC (`ctc.rs`), beam search (`beam.rs`), the Qwen3 language model (`qwen3.rs`), the sound-checked dictionary (`dictionary.rs`), the logit dictionary (`logit_bias.rs`), and 1-D convolutions: dilated, depthwise, transposed (`conv.rs`) |
| `src/models/families/` | One file per checkpoint family. It reads the family's files and wires the parts: Parakeet, Canary, Whisper, Moonshine, wav2vec 2.0, Qwen3-ASR, and the VITS voice (`vits.rs`) |
| `ffi/` | `trit-ffi`: the C ABI |
| `bindings/` | Python, C#, Node |

## License

Trit is under the [PolyForm Noncommercial License 1.0.0](../LICENSE). You can
use, change, and share it for any noncommercial purpose. You cannot sell it
or use it to make money.

Each model has its own license: Parakeet and Canary CC-BY-4.0, Whisper and
Moonshine MIT, S1-mini Apache-2.0 with a naming clause, MMS-TTS
CC-BY-NC-4.0 (noncommercial). Follow it when you ship the model.
