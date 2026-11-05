# Trit: agent guide

Trit is a CPU speech-to-text and text-to-speech engine in Rust with no
inference library. It runs Parakeet, Canary, Whisper, and Moonshine
checkpoints from Hugging Face, the S1-mini transcript cleaner, and the
Kokoro-82M and VITS (Meta MMS-TTS) voices. [docs/reference.md](docs/reference.md) is the reference:
the API of each language, the models, and the options. This file gives the
steps.

Select your branch:

- **Integrate:** put Trit into a program or a script. Go to [Integrate Trit](#integrate-trit).
- **Change:** change the code of Trit itself. Go to [Change Trit](#change-trit).

## Integrate Trit

Do the steps in order. Each step ends on a check.

1. **Select the way in** from the program's language:

   | Program | Way in |
   |---|---|
   | Rust | `trit = { git = "https://github.com/Hi9841/trit" }` in `Cargo.toml` |
   | Python | Copy `bindings/python/trit.py`; it needs the `trit-ffi` library |
   | C# | Add `bindings/csharp/Trit.cs`; it needs the `trit-ffi` library |
   | Node or Bun | Copy `bindings/node/trit.mjs` and add `koffi`; it needs the `trit-ffi` library |
   | C, C++, Go, Swift, Zig, other FFI | `include/trit.h` and the `trit-ffi` library |
   | Shell, or no FFI | The `trit` CLI with `--format json` |

   Check: you can name the file to copy or the dependency line.

2. **Build** in the repository root: `cargo build --release` for the CLI, and
   `cargo build --release -p trit-ffi` for the library.
   Check: `target/release/trit` and `trit_ffi.dll` (`libtrit_ffi.so`,
   `libtrit_ffi.dylib`) exist.

3. **Get the model.** Use `parakeet-redux` unless the task asks for another
   language or a smaller model (docs/reference.md, "Select a model"). `trit models`
   prints the download command.
   Check: `trit doctor` reports the model as found.

4. **Record the expected text:**
   `trit transcribe <clip.wav> --model <name> --language en --format json`.
   Check: you have the text of one clip, from the CLI.

5. **Write the integration** by the rules of docs/reference.md, "The contract". The
   rules that integrations get wrong most:
   - Convert the audio to 16 kHz mono `f32` in -1..1 before Trit. Trit does
     not resample, and microphones usually give 44.1 or 48 kHz.
   - Load the engine once at start and keep it. Do not load it per call.
   - Pass `language: "en"` for an English program.
   - C ABI: free each returned text with `trit_free_text`; after NULL or -1,
     read `trit_last_error()`.
   - Call `release_memory()` when the program goes idle.

6. **Verify:** run your program on the clip of step 4.
   Check: the text is the same as the CLI text, character for character.

7. **License:** Trit is under PolyForm Noncommercial 1.0.0. If the program
   is commercial, tell the user before you ship it. The model has its own
   license (docs/reference.md, "License").

## Change Trit

### Commands (run in the repository root)

| Task | Command |
|---|---|
| Build CLI and library | `cargo build --release --workspace` |
| Lint | `cargo clippy --workspace --all-targets --release -- -D warnings` |
| Build with the GPU | `cargo build --release --workspace --features trit/gpu`; lint it with the same `--features trit/gpu` |
| Format | `cargo fmt --all` |
| Smoke test | `./target/release/trit doctor`, then `./target/release/trit transcribe <file.wav> --model <name>` |
| Skill current? | `./target/release/trit skill --check skill/SKILL.md` |
| Voices against their references | `python bench/tts.py kokoro <trit.exe> <kokoro dir> <parakeet-redux dir>` and `python bench/tts.py vits <trit.exe> <mms dir> <parakeet-redux dir>`, in the environment that `bench/tts.py` names |
| Phonemes of a text | `./target/release/trit phonemes "<text>"` (`--british`, `--no-fallback`) |
| Same model on other runtimes | `uv run --no-project --python 3.12 python bench/runtimes.py <engine>` (`--rounds 3` for published numbers: the report gives the median and the spread) |
| Same GGUF file as transcribe.cpp (Hex's engine) | `uv run --no-project --python 3.12 --with whisper-normalizer python bench/gguf.py run <models dir> --rounds 3`, then `... bench/gguf.py report`; build `bench/gguf` first (CMake) |
| Speed of one change | `uv run --no-project --python 3.12 --with numpy --with whisper-normalizer --with scipy python oploop/ab.py <model> <A.dll> <B.dll>` (interleaved A/B of two `trit_ffi` builds) |

There are no unit tests. The checks are:

- **TDT's model:** Parakeet Redux must keep clean 81 errors, other 172,
  tech 34, all without a dictionary (`bench/runtimes.py trit --sets
  clean,other,tech`). A change must not raise one of them. TDT checks the
  same numbers with its own ratchet (`desktop/bench/engine-budget.json` in
  the TDT repository).
- **Other families:** the error totals over the 456 clips. After a change,
  rerun the affected engines with `runtimes.py`, then
  `python bench/wer_rule.py`. It must exit 0: Trit's default
  has the fewest errors, or a tie, for each model. The last full run is
  `bench/runtimes-results.jsonl`.
- **Voices:** `bench/tts.py`. Kokoro: with the same phonemes and noise,
  every clip has the reference's length, and the SNR stays where float
  rounding puts it (last run: median 38.2 dB, lowest 31.0 dB; PyTorch's own
  float32 against float64 is 25 to 37 dB); the round-trip word errors are
  not above the reference's (last run: 70, 70, 70 against 71, 71, 70).
  VITS: with the noise off, the same length and 75 dB or more; errors 167
  against 171, and 536 against 553.
- **Phonemizer:** given the same tokens, Trit's phonemes equal misaki's
  (`misaki` 0.9.4; run misaki with `g2p.nlp` returning Trit's tokens); last
  check: 451 lines, 0 differences, American and British. The tagger's gap
  to spaCy is separate: 3.0 % of phoneme words on the bench corpus.
- **Speed:** a change is a gain only when `ab.py` shows p < 0.01 and the
  same text on every clip. Record it in `oploop/LEDGER.md`.

### Architecture

- `src/lib.rs`: the public API (`Trit`, `Options`, `Cleaner`, `Voice`, `find_model`, `read_wav`, `write_wav`). One way to do each thing.
- `src/tts/`: text to speech. `mod.rs` is `Voice`: it is not a `Family` (audio in, text out); it picks Kokoro (`kokoro/`: ALBERT, prosody, decoder, iSTFTNet generator) or VITS (`vits.rs`, `characters.rs`) from the folder's files, cuts text into passes, and seeds the noise (`Voicing::seed`), so one text always gives the same audio. `english/` is Kokoro's phonemizer, a port of misaki: `tokens.rs` (tokenizer and tag corrections), `tagger.rs` (NLTK's perceptron), `lexicon.rs` and `numbers.rs` (misaki's lexicon and num2words), `fallback.rs` (the BART for unknown words), `data/` (written by `tools/tts_data.py`). Voices keep f32 weights (`Precision` does not apply): the vocoder writes every sample from them. Kokoro's `.pth` is read by `cpu/torch_file.rs`; `parts/lstm.rs`, `parts/conv.rs`, and `audio/stft.rs` are its layers.
- `src/models/mod.rs`: `FAMILIES` has one row per `model_type`: its loader and its worker spread (`Spread::OneCache` for short products, `AllCores` for Whisper). `MODEL_TYPES` comes from it. Every family implements `Family`. Every load and call enters the team of its spread; each spread has its own team.
- `src/models/parts/`: building blocks that the config picks, shared across families. `fastconformer.rs` is the encoder of Parakeet and Canary; `transducer.rs` the TDT and RNNT head; `ctc.rs`, `beam.rs`, `qwen3.rs` (the language model of Qwen3-ASR and S1-mini). `dictionary.rs` is the TDT dictionary: terms replace close spellings when the joint network's score allows. `logit_bias.rs` is the logit dictionary of Canary and Moonshine.
- `src/models/gguf.rs`: GGUF files of Parakeet, as transcribe.cpp writes them. The metadata gives the config and tokenizer; the tensors keep their stored types (F32, F16, Q8_0) under `Precision::Auto` (`Tensors::precision_of`). `models::Package` is the one way in for a single-file model (GGUF or `.nemo`).
- `src/models/nemo.rs`: NVIDIA NeMo archives (`.nemo`). It reads the tar in place and gives the config, tensor names, and tokenizer that Transformers' `convert_nemo_to_hf.py` writes; the Parakeet loader then runs as for a Hugging Face folder.
- `src/models/families/`: one file per checkpoint family; it reads the files and wires the parts. `layers.rs` holds the shared layers (norms, GELU, rotary, attention with grouped heads, conv columns); `config.rs` reads `config.json`.
- Weights: store each large model buffer as `Store::weights(vec)` (`cpu/store.rs`), never as a bare `Vec` in a layer. During a load with `LoadOptions::cache` it becomes a view of the mapped cache file when the bytes are equal. A buffer that grows after the load (`RelCache`) is `Store::owned`. Every load goes through `cpu::store::load` or `in_turn`, one at a time, so the buffers of one load never reach the file of another.
- Precision: `Tensors::open_with(path, precision)` carries the `Precision` of `LoadOptions`; `Dense::load` and `Dense::table` (token tables) read it through `Tensors::precision_of`. Load every dense layer through them, so all five precisions work. `q8_0` is ggml's Q8_0 (`Blocks::quantize` makes ggml's bytes; `Int8` runs it with 32-value groups and f16 scales).
- `src/cpu/`: kernels. `fgemm.rs` is the one f32 GEMM; its weights are f32 or f16 (`Weight`). `kernels.rs` is ternary (Redux), `int8.rs` int8 (VNNI 512, VNNI 256, AVX2; on arm64 `smmla`, `sdot`, NEON). Each has AVX-512, AVX2, NEON, and scalar paths (docs/reference.md, "CPU kernels"); `TRIT_ISA=scalar|avx2|avxvnni` (x86-64) or `scalar|neon|dotprod` (arm64) forces a path. Check arm64 text under QEMU (`-cpu max` has `sdot` and `i8mm`). All paths must give the same text. Worker threads allocate nothing per task (thread-local scratch, `Buf` pool on the caller): allocator heaps of workers keep what they free.
- `src/gpu/`: matrix products and the Parakeet encoder on the GPU through wgpu (DX12, Vulkan, Metal). The `gpu` feature builds them; `trit-ffi` enables it. `auto` uses the CPU when the computer has no adapter, and a GPU error falls back to the CPU for that call (after 3 errors in sequence, for all later calls; a readback waits at most 10 s). The shaders are `.wgsl` files read with `include_str!` (CI allows this in `src/gpu` only). Text must stay the same on `--device cpu` and in a build without `gpu`.
- `src/audio/`, `src/text/`: frontends and tokenizers. `bpe.rs` encodes with hand-written pre-tokenizers (GPT-2 and Qwen); it matches Hugging Face `tokenizers` ids.
- `src/main.rs`: the `trit` CLI. AXI rules: TOON on stdout, errors on stdout with a `help:` fix, exit 0/1/2, unknown flags fail with the valid list.
- `ffi/`: `trit-ffi`, the C ABI. `include/trit.h` must match it by hand. Change both together, then the three bindings in `bindings/`, then the tables in docs/reference.md.

### Add a model family

1. Read the reference implementation (Transformers `modeling_<family>.py`) and the tensor names (`model.safetensors` header). Note every norm epsilon, activation form, rotary layout, and scale.
2. Reuse the parts in `src/models/parts/` that the config names (a FastConformer checkpoint needs only a new head or none). Put what is new in `parts/` when another family can share it, else in `src/models/families/<family>.rs`, from `layers.rs`, `Dense::load` layers, and `Dense::table` token tables.
3. Add one row to `FAMILIES` in `src/models/mod.rs` and implement `Family` for the new type.
4. Add the checkpoints you ran to `src/catalog.rs`, then run `trit skill > skill/SKILL.md` and update the tables in docs/reference.md (and in README.md when the model is in its short table).
5. Check: the same text as the reference runtime on a few clips of `target/accuracy/tts16`.

### Rules the code does not show

- **No word lists in speech to text.** Speech to text decodes with the
  model alone: no list of words, pieces, or terms ships in it, and no bench
  runs with one, so Trit is scored the same way as every other runtime.
  Rules that read the model's own vocabulary are allowed ("pieces with
  accented letters"); a file of chosen words is not. Failure seen: a 580 KB
  English word list, a 1,132-piece foreign list, and 130 built-in developer
  terms shipped and gave Trit help no other runtime had. CI fails on
  `include_str!` in `src/` and `ffi/`, and on text files in `src/`, outside
  `src/tts/`.
- **Text to speech may hold lists** (the owner's decision, 2026-10-25): a
  voice needs text normalization and a pronunciation lexicon to say text.
  Keep them in `src/tts/`, and their sources and licenses in
  `THIRD_PARTY_NOTICES.md` and `src/tts/english/data/README.md`.
- **The phonemizer follows misaki.** Port a misaki change function for
  function, then rerun the phonemizer check. Where Trit differs on purpose,
  say so in a comment: the tokens and tags (not spaCy), and the BART reads
  only word-like groups.

- `skill/SKILL.md` comes from `trit skill`. Change the text in `src/main.rs` (`HOME_HELP`, `TRANSCRIBE_HELP`, `CLEAN_HELP`, `SPEAK_HELP`, `PHONEMES_HELP`, `VOICES_HELP`, `skill_text`) and `src/catalog.rs`, then run `trit skill > skill/SKILL.md`. CI fails when it is stale.
- TDT (https://github.com/Hi9841/tdt) depends on a tag of this repository and passes `Options::language("en")`. A public API change must build there too. In a TDT checkout: `cargo build --release --manifest-path desktop/Cargo.toml --config "patch.'https://github.com/Hi9841/trit'.trit.path='<this repository>'"`.
- Read the model files as published. Converting at load (F32 or BF16 to f16) is the only change allowed; the files on disk stay as Hugging Face ships them. The weight cache (`<model>/.trit/`) is a separate file of converted buffers, written only with `LoadOptions::cache`.
- S1-mini's system prompt and control line are its trained input format (`src/cleanup.rs`). Keep them word for word.
- On this Windows computer, a debug build of `libmimalloc-sys` can fail with MSVC error D8050. Use `--release` for clippy.
