# Trit

Trit is a speech to text engine for the CPU. It runs Parakeet, Canary,
Whisper, and Moonshine checkpoints as published on Hugging Face, with its own kernels: no
Python, no PyTorch, no ONNX Runtime, no GPU. 16 kHz WAV in, text out, in any
language the model speaks. It can also clean a raw transcript with
Superwhisper S1-mini.

TDT, the dictation app in this repository, uses Trit for all of its speech.

```powershell
cd trit
cargo build --release
./target/release/trit models                                  # what to download
hf download moondream/parakeet-redux --include "*.json" model.safetensors --local-dir models/parakeet-redux
./target/release/trit transcribe take.wav
```

```text
model: ~/work/TDT/trit/models/parakeet-redux
files: 1 transcribed, 0 failed, 6.0 s of audio in 0.04 s (172x realtime)
results[1]{file,seconds,ms,text}:
  take.wav,6.0,35,"Restart the Kubernetes pod, then check the PostgreSQL logs."
```

## Contents

1. [Models](#models)
2. [Command line](#command-line)
3. [Rust](#rust)
4. [C and C++](#c-and-c)
5. [Python](#python)
6. [C# and .NET](#c-and-net)
7. [Node and Bun](#node-and-bun)
8. [Other languages](#other-languages)
9. [Languages](#languages)
10. [Transcript cleanup](#transcript-cleanup)
11. [Behavior and limits](#behavior-and-limits)
12. [Precision](#precision)
13. [Environment variables](#environment-variables)
14. [Performance](#performance)
15. [Layout of the code](#layout-of-the-code)

## Models

Trit reads `config.json` and `model.safetensors` of a Hugging Face folder and
picks the network from `model_type`. These checkpoints have been run; any
other checkpoint of the same family loads the same way.

| Name | Repository | Family | Languages | MB |
|---|---|---|---|---:|
| parakeet-redux | moondream/parakeet-redux | Parakeet TDT, 1.58-bit | 25 European | 178 |
| parakeet-tdt-0.6b-v3 | nvidia/parakeet-tdt-0.6b-v3 | Parakeet TDT | 25 European | 2508 |
| parakeet-ctc-0.6b | nvidia/parakeet-ctc-0.6b | Parakeet CTC | English | 2436 |
| canary-1b-v2 | nvidia/canary-1b-v2 | Canary (FastConformer + transformer decoder) | 25 European | 3916 |
| whisper-tiny | openai/whisper-tiny | Whisper | 99 | 151 |
| whisper-base.en | openai/whisper-base.en | Whisper | English | 290 |
| whisper-small.en | openai/whisper-small.en | Whisper | English | 967 |
| whisper-large-v3-turbo | openai/whisper-large-v3-turbo | Whisper | 99 | 1618 |
| whisper-large-v3 | openai/whisper-large-v3 | Whisper | 99 | 3087 |
| distil-large-v3.5 | distil-whisper/distil-large-v3.5 | Whisper | English | 3026 |
| moonshine-tiny | UsefulSensors/moonshine-tiny | Moonshine | English | 108 |
| moonshine-base | UsefulSensors/moonshine-base | Moonshine | English | 246 |
| s1-mini | superwhisper/s1-mini | Qwen3 text model (cleanup) | English | 1503 |

Get one (only the files Trit reads):

```powershell
hf download openai/whisper-large-v3-turbo --include "*.json" model.safetensors --local-dir models/whisper-large-v3-turbo
```

`trit models` prints this table with the folder of each model it finds on
the computer and the download command. Pass the name or a folder to
`--model`. Trit looks for a named model in this order:

1. The folder you give, or a name from the table.
2. `TRIT_MODEL` (when no model is named).
3. `models/<name>` and `../models/<name>` from the current folder.
4. `models/<name>` next to the executable, and two and three folders up.
5. `%LOCALAPPDATA%/TDT/models/<name>`.
6. The Hugging Face cache: `~/.cache/huggingface/hub/models--<owner>--<name>/snapshots/*`.

The default model, when none is named, is parakeet-redux. `trit doctor`
prints each place and its status.

## Command line

`trit` is built for agents ([AXI](https://axi.md)): TOON output on stdout,
errors on stdout as `error:` and `help:` lines, exit 0 (success), 1 (error),
2 (usage error), and no prompts. `trit` with no arguments shows the live state.

| Command | What it does |
|---|---|
| `trit` | Home view: the binary, the models, the CPU path, next commands |
| `trit transcribe <path>...` | Text of WAV files, or of every WAV file in a folder |
| `trit clean "<text>"` | A raw transcript as clean written English (S1-mini) |
| `trit models` | The models Trit runs, where they are, how to get them |
| `trit doctor` | Where Trit looks for models, and the fix for each problem |
| `trit skill [--check <file>]` | The agent skill (`skill/SKILL.md`), made from the help text |
| `trit --version` | The version |

Flags of `transcribe`:

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir or name>` | parakeet-redux | The speech model |
| `--language <code>` | detect | ISO 639-1 code (`en`, `de`, `fr`) |
| `--dictionary <file>` | none | Terms to favor, one per line (every family) |
| `--task transcribe\|translate` | `transcribe` | `translate` writes English (multilingual Whisper, Canary) |
| `--timestamps` | off | Timed segments too: sentences for Parakeet, windows for the others |
| `--precision auto\|int8\|f16\|f32` | `auto` | How the weights are held; see [Precision](#precision) |
| `--beam <n>` | 5 for Whisper and Moonshine, 1 for Canary | Beams; 1 is greedy, as the reference runtimes decode by default |
| `--cleanup` | off | Clean each transcript with S1-mini; `--cleanup-model`, `--styling`, `--structure`, `--context` as for `clean` |
| `--format toon\|json\|text` | `toon` | `json` gives the full text for programs; `text` gives one transcript per line |
| `--full` | off | Do not cut transcripts over 500 characters in a list of several files |

```powershell
trit transcribe clips/ --model whisper-large-v3-turbo --format json
trit transcribe interview.wav --model parakeet-tdt-0.6b-v3 --language de
trit transcribe take.wav --dictionary terms.txt --cleanup --format text
trit transcribe meeting.wav --timestamps --precision f32
trit clean "so um send the report by friday no wait thursday"
```

## Rust

```toml
[dependencies]
trit = { path = "../trit" }
```

```rust
let dir = trit::find_model(Some(Path::new("whisper-large-v3-turbo")))
    .ok_or_else(|| trit::missing_model(None))?;
let engine = trit::Trit::load(&dir)?;            // keep it; loading takes 0.2 to 5 s
let samples = trit::read_wav(Path::new("take.wav"))?;
let text = engine.transcribe(&samples);           // the model detects the language
let german = engine.transcribe_with(&samples, &trit::Options::language("de"));
let english = engine.transcribe_with(&samples, &trit::Options { task: trit::Task::Translate, ..Default::default() });
for segment in engine.segments(&samples, &trit::Options::language("en")) {
    println!("{:.2}-{:.2} {}", segment.start, segment.end, segment.text);
}
let exact = trit::Trit::load_with(&dir, &trit::LoadOptions { precision: trit::Precision::F32 })?;
trit::release_memory();                           // when the app goes idle

let cleaner = trit::Cleaner::load(&trit::find_cleanup_model(None).unwrap())?;
let clean = cleaner.clean(&text, &trit::Style::default());
```

| Item | Meaning |
|---|---|
| `Trit::load(dir) -> Result<Trit, String>` | Loads a model folder; the family comes from `config.json` |
| `Trit::load_with(dir, &LoadOptions { precision })` | The same with a `Precision`: `Auto` (default), `Int8`, `F16`, `F32` |
| `Trit::transcribe(&self, &[f32]) -> String` | 16 kHz mono samples in -1..1 to text; the model detects the language |
| `Trit::transcribe_with(&self, &[f32], &Options) -> String` | The same with `Options { language, task, beam, fallback }` |
| `Trit::segments(&self, &[f32], &Options) -> Vec<Segment>` | The text in timed parts (`start`, `end` in seconds, `text`) |
| `Trit::kind(&self) -> &str` | `parakeet_tdt`, `parakeet_ctc`, `canary`, `whisper`, or `moonshine` |
| `Trit::set_dictionary(&self, &[String], BiasWeights) -> Vec<String>` | Terms to favor; returns the terms the model cannot spell |
| `release_memory()` | Returns memory freed by earlier calls to the system; call it when idle |
| `Cleaner::load(dir)`, `Cleaner::load_with(dir, &LoadOptions)`, `Cleaner::clean(&self, &str, &Style) -> String` | S1-mini transcript cleanup |
| `find_model(Option<&Path>)`, `find_cleanup_model(Option<&Path>)` | The search in [Models](#models) |
| `missing_model(Option<&Path>) -> String` | Why `find_model` found nothing, with the fix |
| `CATALOG`, `known(name)`, `candidates(repo)` | The table above and the places of a repository |
| `read_wav(&Path) -> Result<Vec<f32>, String>` | 16 kHz WAV (16-bit PCM or 32-bit float) to mono samples |
| `cpu_path() -> &str`, `threads(Spread) -> usize` | The ternary, int8, and float kernel paths; the worker count |
| `SAMPLE_RATE`, `MODEL_TYPES`, `VERSION` | Constants |

`Trit` and `Cleaner` are `Send + Sync`. Share them with `Arc`.

## C and C++

Build the library: `cargo build --release -p trit-ffi`. It gives
`trit_ffi.dll` (Windows), `libtrit_ffi.so` (Linux), `libtrit_ffi.dylib`
(macOS), and a static library. The header is
[include/trit.h](include/trit.h).

```c
#include "trit.h"

Trit *engine = trit_load("whisper-large-v3-turbo");     /* folder, name, or NULL */
if (!engine) { printf("error: %s\n", trit_last_error()); return 1; }
char *text = trit_transcribe_wav(engine, "take.wav", NULL);  /* NULL: detect language */
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
| `trit_set_dictionary(engine, const char *terms)` | Terms skipped, or -1. One term per line |
| `trit_cleaner_load(const char *model)`, `trit_cleaner_load_with(model, precision)` | A cleaner, or NULL |
| `trit_release_memory(void)` | Returns memory freed by earlier calls to the system; call it when idle |
| `trit_clean(cleaner, text, styling, structure, context)` | Clean text, or NULL. NULL settings take the default |
| `trit_last_error(void)` | The reason for the last failure on this thread |
| `trit_free_text(char *)`, `trit_free(Trit *)`, `trit_cleaner_free(TritCleaner *)` | Free. NULL is a no-op |
| `trit_version(void)` | `"0.1.0"` |

Rules: strings are UTF-8. `language` is an ISO 639-1 code or NULL. Text
that a function returns is yours to free with `trit_free_text`. No panic
crosses the ABI; it becomes NULL or -1 and an error text. A complete
program is in [examples/transcribe.c](examples/transcribe.c).

## Python

[bindings/python/trit.py](bindings/python/trit.py) is one file with no
dependencies (ctypes). Copy it next to your code, or add its folder to the path.

```python
import trit

engine = trit.Trit("whisper-large-v3-turbo")      # folder, name, or None; precision="f16" and so on
print(engine.kind)                                # "whisper"
print(engine.transcribe_wav("take.wav"))
print(engine.transcribe(samples, language="de"))  # list, array("f"), or numpy float32, 16 kHz mono
print(engine.transcribe(samples, beam=1))         # greedy (Whisper and Moonshine use 5 beams by default)
trit.release_memory()                             # when the app goes idle

cleaner = trit.Cleaner()                          # S1-mini
print(cleaner.clean("so um send it friday no wait thursday", context="email"))
```

Errors raise `trit.TritError` with the cause and the fix. The library is
found at `TRIT_LIB`, next to `trit.py`, or in `trit/target/release`.

## C# and .NET

Add [bindings/csharp/Trit.cs](bindings/csharp/Trit.cs) to the project, set
`<AllowUnsafeBlocks>true</AllowUnsafeBlocks>`, and copy `trit_ffi.dll` next
to the program. .NET 7 or later.

```csharp
using TritSpeech;

using var engine = new Trit("whisper-large-v3-turbo");   // second argument: precision, "f16" and so on
Console.WriteLine(engine.TranscribeWav("take.wav"));
Console.WriteLine(engine.Transcribe(samples, "de"));     // float[] or Span<float>, 16 kHz mono

using var cleaner = new TritCleaner();
Console.WriteLine(cleaner.Clean("so um send it friday no wait thursday"));
```

Errors throw `TritException`.

## Node and Bun

[bindings/node/trit.mjs](bindings/node/trit.mjs) uses
[koffi](https://koffi.dev): `npm install koffi` or `bun add koffi`.

```js
import { Trit, Cleaner } from "./trit.mjs";

const engine = new Trit("whisper-large-v3-turbo");     // second argument: { precision: "f16" }
console.log(engine.transcribeWav("take.wav"));
console.log(engine.transcribe(samples, "de"));         // Float32Array, 16 kHz mono
engine.close();

const cleaner = new Cleaner();
console.log(cleaner.clean("so um send it friday no wait thursday", { styling: "formal" }));
```

Errors throw `TritError`.

## Other languages

Any language with a C FFI uses [include/trit.h](include/trit.h) as it is:
Go (cgo), Swift, Kotlin/Java (Panama or JNA), Zig, Ruby (FFI gem), Dart
(dart:ffi), Lua (LuaJIT FFI). Bind the functions of the header and keep two
rules: free returned text with `trit_free_text`, and read `trit_last_error`
after a NULL or -1.

A program that cannot load a library can run the CLI:
`trit transcribe <file> --model <name> --format json`.

## Languages

| Family | Languages | `language` option |
|---|---|---|
| Whisper (multilingual) | 99 | Forces that language; without it, the model detects the language from the first 30 s |
| Whisper `.en`, Distil large-v3.5 | English | Ignored |
| Parakeet TDT v3, Redux | 25 European | Cannot be forced; `en` keeps the output to English tokens |
| Parakeet CTC | English | `en` as above |
| Canary 1B v2 | 25 European | The spoken language; English when none is given (Canary does not detect it) |
| Moonshine | English | Ignored |

TDT itself passes `language: "en"`, so the app stays English only.

## Transcript cleanup

[Superwhisper S1-mini](https://huggingface.co/superwhisper/s1-mini) is a
0.6B text model (Qwen3) that turns the raw text of any speech model into
written English: fillers removed, self-corrections resolved ("friday no
wait thursday" becomes "Thursday"), punctuation, casing, numbers, dates,
times, currency, and email addresses. Trit sends the exact system prompt
and control line the model was trained on.

| Setting | Values |
|---|---|
| styling | `casual`, `semi-casual`, `semi-formal` (default), `formal` |
| structure | `prose` (default), `lists` |
| context | `general` (default), `email` |

Input that is only fillers gives an empty string. English only. The model
is under Apache 2.0 with a naming clause; read its LICENSE before you ship it.

## Behavior and limits

- **Input:** 16 kHz mono float samples in -1..1. `read_wav` and the CLI read
  16 kHz WAV files and mix channels to mono. Other rates fail with an `ffmpeg`
  command that converts them; Trit does not resample.
- **Long audio:** windows cut at the quietest 100 ms near the end of each
  window: 30 s for Whisper and Moonshine, 40 s for Canary, 60 s for
  Parakeet (attention memory grows with the square of the length).
- **Decoding:** Whisper and Moonshine use beam search with 5 beams by
  default, by the rules of Transformers' `generate(num_beams=5)` (OpenAI's
  rules are `beam::Rules::OpenAi`). On the 456 English test clips it makes
  768 errors against 894 greedy (Whisper tiny), 479 against 519 (base.en),
  557 against 653 (Moonshine tiny), and 415 against 477 (Moonshine base);
  Transformers with 5 beams makes 772, 479, 557, and 416. The beams go
  through each decoder step as one batch, so it costs about 1.2 to 2 times
  the greedy time. Parakeet and Canary decode greedy; `Options::beam = 1`
  makes every family greedy, as the reference runtimes are. Whisper decodes a
  window again at temperatures 0.2 to 1.0 (fixed seed) when its text repeats
  itself (zlib ratio over 2.4) or is unlikely (mean log probability under
  -1), as OpenAI Whisper, faster-whisper, and whisper.cpp do
  (`Options::fallback`), and stops a loop (one span repeated four times).
- **Timestamps:** `segments` gives sentences or pauses of 1 s for Parakeet,
  from the frame (80 ms) where its decoder emits each token, and one segment
  per window for the others.
- **Silence:** Parakeet returns an empty string under 80 ms; Moonshine under
  56 ms. Parakeet Redux checks its own speech head (`vad_head.*`) first and
  returns an empty string when no 0.1 s of speech is found.
- **Dictionary:** Parakeet, Canary, and Moonshine raise the logits of the
  terms' tokens while greedy decoding (see `parakeet/bias.rs`), so with terms
  set they decode greedy; Whisper puts
  the terms in a previous-text prompt (`<|startofprev|>`), as OpenAI's
  `initial_prompt`.
- **Canary:** give it the spoken language, because it does not detect one
  (English when none is given).
- **Threads:** Parakeet and S1-mini use one worker per physical core of the
  largest L3 domain (8 on a Ryzen 7950X3D), pinned; their products are short,
  and exchanges across L3 domains cost more than extra cores give. Whisper
  uses every physical core: its encoder always has 1500 rows. Each spread
  has its own team, so a process with Whisper and Parakeet runs each on its
  own. Workers spin for 3 ms after a step, then sleep, so an idle engine
  uses no CPU. Linux reads the same topology from `/sys`; macOS uses the
  performance cores, unpinned.
- **Memory:** the engines keep their temporaries (a buffer pool of up to
  64 MB, per-thread scratch) for the next call. `release_memory` returns
  them; TDT calls it 5 s after a dictation.
- **Silence:** Whisper and Moonshine skip a window under -60 dBFS RMS, and
  Whisper drops a window when the model rates it as no speech (probability
  over 0.6) and its text as unlikely (mean log probability under -1), as
  OpenAI Whisper does. Without this, Whisper writes "you" for silence.
- **CPU:** x86-64 with AVX-512 or AVX2 (with F16C) is fast; int8 also uses
  AVX-VNNI (Intel client CPUs without AVX-512). Other CPUs, ARM included,
  use the scalar path, which gives the same text but is much slower. Only
  x86-64 Windows has been measured.

## Precision

`Precision` (`--precision`, `trit_load_with`, `precision=`) sets how a
model holds its dense layers and token tables. Parakeet Redux keeps its
ternary weights in every mode.

| Precision | Dense layers | Token tables | Use |
|---|---|---|---|
| `auto` (default) | int8 for layers 384 wide and more, f16 below | int8 storage with f32 math when 1024 wide and more (S1-mini, Canary), else f16 | The fastest; on the test clips as accurate as f32 or more |
| `int8` | int8 everywhere (int8 storage with f32 math where the int8 kernels cannot take the shape) | int8 storage, f32 math | The least memory; narrow models lose some accuracy |
| `f16` | f16 weights, f32 math | f16 | Half the memory of f32 |
| `f32` | f32, as the reference runtimes hold them | f32 | Checks against the reference |

Int8 multiplies int8 weights (one scale per row and 128 inputs) by int8
activations (quantized the same way per call). On the 456 English test clips
with beam search it does not cost accuracy from 384 wide: Whisper tiny 760
errors with `auto` (f16: 768), base.en 468 (f16: 479). Moonshine tiny (288
wide) does lose (568 against 557), so `auto` keeps it f16. The S1-mini
output projection (151936 x 1024) is read for every token; as int8 it takes
77 ms instead of 137 ms of a 24-token cleanup, with 42 errors and 37 exact
outputs of 62 against 43 and 36 in f16.

## Environment variables

| Variable | Effect |
|---|---|
| `TRIT_MODEL` | The speech model folder, when the caller names none |
| `TRIT_CLEANUP_MODEL` | The S1-mini folder, when the caller names none |
| `TRIT_THREADS` | The worker count; turns pinning off |
| `TRIT_ISA` | `scalar`, `avx2`, or `avxvnni` (the 256-bit int8 path on an AVX-512 CPU) forces another kernel path, for tests |
| `TRIT_LIB` | The library path, for the Python and Node bindings |

## Performance

Measured on an AMD Ryzen 9 7950X3D, Windows 11, 2026-10-21, one clip at a
time after a warm-up. Same model on other runtimes:
[docs/engine-comparison.md](../docs/engine-comparison.md).

| Model | Clips | Time per clip | Speed |
|---|---|---:|---:|
| parakeet-redux | LibriSpeech test-clean, 200 (median) | 35 ms | 159x realtime |
| parakeet-tdt-0.6b-v3 | LibriSpeech test-clean, 200 | | 148x realtime |
| parakeet-ctc-0.6b | everyday dictation, 16 | | 167x realtime |
| canary-1b-v2 | 5 s to 21 s clips | 85 to 400 ms | 54x realtime |
| moonshine-base | everyday dictation, 16 | | 146x realtime |
| whisper-base.en | LibriSpeech test-clean, 200 | | 57x realtime |
| whisper-large-v3-turbo | 5 s to 21 s clips | 0.92 s | 6x realtime on a 6 s clip |
| s1-mini cleanup | one sentence | 210 ms | |

Accuracy on the 200 test-clean clips: parakeet-redux 2.02 %, parakeet-tdt-0.6b-v3
1.60 %, whisper-base.en 3.42 % WER. Whisper large-v3-turbo takes 0.92 s per clip
here, faster-whisper (int8) 3.1 s on the same computer.

Whisper always encodes a full 30 s window, so a short clip costs as much as
a long one.

## Layout of the code

| Folder or file | Part |
|---|---|
| `src/lib.rs` | Public API, model search |
| `src/main.rs` | The `trit` CLI |
| `src/catalog.rs` | The models of the table above |
| `src/cleanup.rs` | `Cleaner`: S1-mini prompt and styles |
| `src/cpu/` | Kernels and threads: f32/f16 GEMM (`fgemm.rs`), dense layers and `Precision` (`gemm.rs`), int8 (`int8.rs`), ternary (`kernels.rs`), f16 matvec (`half.rs`), vector math, the worker teams, safetensors reader |
| `src/audio/` | WAV reader, NeMo log-mel (`mel.rs`), Whisper log-mel (`whisper_mel.rs`), windows for long audio |
| `src/text/` | SentencePiece tokenizer (Parakeet, Moonshine), byte-level BPE (Whisper, Qwen) |
| `src/models/parakeet/` | FastConformer encoder, TDT and CTC heads, dictionary biasing |
| `src/models/canary.rs` | Canary: the Parakeet encoder and a transformer decoder |
| `src/models/whisper/` | Whisper encoder and decoder, greedy decoding, language detection |
| `src/models/moonshine.rs` | Moonshine |
| `src/models/qwen3.rs` | Qwen3 decoder (S1-mini) |
| `src/models/transformer.rs` | Shared pieces: norms, GELU, rotary positions, attention, conv columns |
| `src/models/beam.rs` | Beam search for the decoder families |
| `ffi/` | `trit-ffi`: the C ABI |
| `bindings/` | Python, C#, Node |

## License

[PolyForm Noncommercial 1.0.0](../LICENSE): use, change, and share Trit for
any noncommercial purpose; do not sell it or use it to make money. Each model has its own license (Parakeet, Canary: CC-BY-4.0, Whisper and
Moonshine: MIT, S1-mini: Apache-2.0 with a naming clause); follow it when
you ship the model.
