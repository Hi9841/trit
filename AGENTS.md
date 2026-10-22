# Trit: agent guide

Trit is a CPU speech-to-text engine in Rust with no inference library. It
runs Parakeet, Canary, Whisper, and Moonshine checkpoints from Hugging Face,
and the S1-mini transcript cleaner. [README.md](README.md) is the reference:
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
   | Rust | `trit = { git = "https://github.com/Hi9841/tdt" }` in `Cargo.toml` |
   | Python | Copy `bindings/python/trit.py`; it needs the `trit-ffi` library |
   | C# | Add `bindings/csharp/Trit.cs`; it needs the `trit-ffi` library |
   | Node or Bun | Copy `bindings/node/trit.mjs` and add `koffi`; it needs the `trit-ffi` library |
   | C, C++, Go, Swift, Zig, other FFI | `include/trit.h` and the `trit-ffi` library |
   | Shell, or no FFI | The `trit` CLI with `--format json` |

   Check: you can name the file to copy or the dependency line.

2. **Build** in `trit/`: `cargo build --release` for the CLI, and
   `cargo build --release -p trit-ffi` for the library.
   Check: `target/release/trit` and `trit_ffi.dll` (`libtrit_ffi.so`,
   `libtrit_ffi.dylib`) exist.

3. **Get the model.** Use `parakeet-redux` unless the task asks for another
   language or a smaller model (README, "Select a model"). `trit models`
   prints the download command.
   Check: `trit doctor` reports the model as found.

4. **Record the expected text:**
   `trit transcribe <clip.wav> --model <name> --language en --format json`.
   Check: you have the text of one clip, from the CLI.

5. **Write the integration** by the rules of README, "The contract". The
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
   license (README, "License").

## Change Trit

### Commands (run in `trit/`)

| Task | Command |
|---|---|
| Build CLI and library | `cargo build --release --workspace` |
| Lint | `cargo clippy --workspace --all-targets --release -- -D warnings` |
| Format | `cargo fmt --all` |
| Smoke test | `./target/release/trit doctor`, then `./target/release/trit transcribe <file.wav> --model <name>` |
| Skill current? | `./target/release/trit skill --check skill/SKILL.md` |
| Accuracy of the TDT model | `uv run bench/accuracy.py <abs path of bench-asr.exe> clean other tech --engine native` from `desktop/` |
| Same model on other runtimes | `uv run --no-project --python 3.12 python bench/runtimes.py <engine>` from `desktop/` |
| Speed of one change | `uv run --no-project --python 3.12 --with numpy --with whisper-normalizer --with scipy python ../oploop/ab.py <model> <A.dll> <B.dll>` from `desktop/` (interleaved A/B of two `trit_ffi` builds) |

There are no unit tests. The checks are:

- **TDT's model:** the ratchet in `desktop/bench/engine-budget.json`: clean
  80 errors, other 171, tech with developer terms 18. A change must not
  raise one of them.
- **Other families:** the error totals over the 456 clips. After a change,
  rerun the affected engines with `runtimes.py`, then
  `python bench/wer_rule.py` from `desktop/`. It must exit 0: Trit's default
  has the fewest errors, or a tie, for each model. The last full run is
  `desktop/bench/runtimes-results.jsonl`.
- **Speed:** a change is a gain only when `ab.py` shows p < 0.01 and the
  same text on every clip. Record it in `oploop/LEDGER.md`.

### Architecture

- `src/lib.rs`: the public API (`Trit`, `Options`, `Cleaner`, `find_model`, `read_wav`). One way to do each thing.
- `src/models/mod.rs`: `Model` selects the family from `model_type` in `config.json`. Every load and call enters the worker team of its spread (`cpu::enter`: `Spread::OneCache` for short products, `AllCores` for Whisper). Each spread has its own team.
- `src/models/<family>`: one network each. `transformer.rs` holds the shared pieces (norms, GELU, rotary, attention with grouped heads, conv columns, layer loading). `beam.rs` is beam search for the decoder families. `parakeet/dictionary.rs` is the Parakeet TDT dictionary: terms replace close spellings when the joint network's score allows. `parakeet/bias.rs` is the logit dictionary of Canary and Moonshine. `parakeet/english.rs` is the English word prior for drifted windows.
- Precision: `Tensors::open_with(path, precision)` carries the `Precision` of `LoadOptions`; `Dense::load` and `Dense::table` (token tables) read it. Load every dense layer through them, so all four precisions work.
- `src/cpu/`: kernels. `fgemm.rs` is the one f32 GEMM; its weights are f32 or f16 (`Weight`). `kernels.rs` is ternary (Redux), `int8.rs` int8 (VNNI 512, VNNI 256, AVX2). Each has AVX-512, AVX2, and scalar paths; `TRIT_ISA=scalar|avx2|avxvnni` forces a path. All paths must give the same text. Worker threads allocate nothing per task (thread-local scratch, `Buf` pool on the caller): allocator heaps of workers keep what they free.
- `src/audio/`, `src/text/`: frontends and tokenizers. `bpe.rs` encodes with hand-written pre-tokenizers (GPT-2 and Qwen); it matches Hugging Face `tokenizers` ids.
- `src/main.rs`: the `trit` CLI. AXI rules: TOON on stdout, errors on stdout with a `help:` fix, exit 0/1/2, unknown flags fail with the valid list.
- `ffi/`: `trit-ffi`, the C ABI. `include/trit.h` must match it by hand. Change both together, then the three bindings in `bindings/`, then the tables in README.md.

### Add a model family

1. Read the reference implementation (Transformers `modeling_<family>.py`) and the tensor names (`model.safetensors` header). Note every norm epsilon, activation form, rotary layout, and scale.
2. Write `src/models/<family>.rs` with `load(dir, config, precision)` and `pieces(samples, options)` (text per window), from `transformer.rs` pieces, `Dense::load` layers, and `Dense::table` token tables.
3. Register it in `src/models/mod.rs` (`MODEL_TYPES`, `Model`, `spread`).
4. Add the checkpoints you ran to `src/catalog.rs`, then run `trit skill > skill/SKILL.md` and update the README tables.
5. Check: the same text as the reference runtime on a few clips of `desktop/target/accuracy/tts16`.

### Rules the code does not show

- `skill/SKILL.md` comes from `trit skill`. Change the text in `src/main.rs` (`HOME_HELP`, `TRANSCRIBE_HELP`, `CLEAN_HELP`, `skill_text`) and `src/catalog.rs`, then run `trit skill > skill/SKILL.md`. CI fails when it is stale.
- The desktop app depends on this crate by path (`desktop/Cargo.toml`) and passes `Options::language("en")`. A public API change must build there too: `cargo build --manifest-path ../desktop/Cargo.toml`.
- Read the model files as published. Converting at load (F32 or BF16 to f16) is the only change allowed; the files on disk stay as Hugging Face ships them.
- `src/text/english-words.txt` and `foreign-pieces.txt` come from `scripts/english_lists.py` (SCOWL size 70 and wordfreq). Do not edit them by hand.
- S1-mini's system prompt and control line are its trained input format (`src/cleanup.rs`). Keep them word for word.
- On this Windows computer, a debug build of `libmimalloc-sys` can fail with MSVC error D8050. Use `--release` for clippy.
