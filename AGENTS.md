# Trit: agent notes

Trit is the CPU speech engine of TDT. It runs Parakeet, Whisper, and
Moonshine checkpoints from Hugging Face, and the S1-mini transcript cleaner,
in Rust with no inference library. The guide for users of the engine is
[README.md](README.md). This file is for agents that change it.

## Commands (run in `trit/`)

| Task | Command |
|---|---|
| Build CLI and library | `cargo build --release --workspace` |
| Lint | `cargo clippy --workspace --all-targets --release -- -D warnings` |
| Format | `cargo fmt --all` |
| Smoke test | `./target/release/trit doctor`, then `./target/release/trit transcribe <file.wav> --model <name>` |
| Models on this computer | `./target/release/trit models` |
| Skill current? | `./target/release/trit skill --check skill/SKILL.md` |
| Accuracy of the TDT model | `uv run bench/accuracy.py <abs path of bench-asr.exe> clean other --engine native` from `desktop/` |
| Same model on other runtimes | `uv run --no-project --python 3.12 python bench/runtimes.py <model>` from `desktop/` |

There are no unit tests. The accuracy ratchet in
`desktop/bench/engine-budget.json` is the check for TDT's model: clean 81
errors, other 172, tech with developer terms 22. A kernel change must not
raise one of them. For another family, transcribe a few clips of
`desktop/target/accuracy/tts16` and compare with the reference runtime's
transcripts in `desktop/target/runtimes/<engine>.transcripts.jsonl`.

## Architecture

- `src/lib.rs`: the public API (`Trit`, `Options`, `Cleaner`, `find_model`, `read_wav`). One way to do each thing.
- `src/models/mod.rs`: `Model` picks the family from `model_type` in `config.json`; every load and call enters the worker team of its spread (`cpu::enter`: `Spread::OneCache` for short products, `AllCores` for Whisper). Each spread has its own team.
- `src/models/<family>`: one network each. `transformer.rs` holds the shared pieces (norms, GELU, rotary, attention with grouped heads, conv columns, layer loading). `beam.rs` is beam search for the decoder families; `parakeet/bias.rs` is the dictionary for the SentencePiece families.
- Precision: `Tensors::open_with(path, precision)` carries the `Precision` of `LoadOptions`; `Dense::load` and `Dense::table` (token tables) read it. Load every dense layer through them, so all four precisions work.
- `src/cpu/`: kernels. `fgemm.rs` is the one f32 GEMM; its weights are f32 or f16 (`Weight`), and blocks of more than 4 rows widen f16 panels once per block. `kernels.rs` is ternary (Redux), `int8.rs` int8 (VNNI 512, VNNI 256, AVX2). Each has AVX-512, AVX2, and scalar paths; `TRIT_ISA=scalar|avx2|avxvnni` forces a path. All paths must give the same text. Worker threads allocate nothing per task (thread-local scratch, `Buf` pool on the caller): allocator heaps of workers keep what they free.
- `src/audio/`, `src/text/`: frontends and tokenizers. `bpe.rs` encodes with hand-written pre-tokenizers (GPT-2 and Qwen); it matches Hugging Face `tokenizers` ids.
- `src/main.rs`: the `trit` CLI. AXI rules: TOON on stdout, errors on stdout with a `help:` fix, exit 0/1/2, unknown flags fail with the valid list.
- `ffi/`: `trit-ffi`, the C ABI. `include/trit.h` must match it by hand; change both together, then the three bindings in `bindings/`.

## Add a model family

1. Read the reference implementation (Transformers `modeling_<family>.py`) and the tensor names (`model.safetensors` header). Note every norm epsilon, activation form, rotary layout, and scale.
2. Write `src/models/<family>.rs` with `load(dir, config, precision)` and `pieces(samples, options)` (text per window), from `transformer.rs` pieces, `Dense::load` layers, and `Dense::table` token tables.
3. Register it in `src/models/mod.rs` (`MODEL_TYPES`, `Model`, `spread`).
4. Add the checkpoints you ran to `src/catalog.rs`, then run `trit skill > skill/SKILL.md` and update the README tables.
5. Check: the same text as the reference runtime on a few clips.

## Rules the code does not show

- The skill in `skill/SKILL.md` comes from `trit skill`. Change the text in `src/main.rs` (`HOME_HELP`, `TRANSCRIBE_HELP`, `CLEAN_HELP`, `skill_text`) and `src/catalog.rs`, then run `trit skill > skill/SKILL.md`.
- The desktop app depends on this crate by path (`desktop/Cargo.toml`) and passes `Options::language("en")`. A public API change must build there too: `cargo build --manifest-path ../desktop/Cargo.toml`.
- The model files are read as published. Do not convert or repack them on disk; converting at load (F32/BF16 to f16) is fine.
- S1-mini's system prompt and control line are its trained input format (`src/cleanup.rs`). Do not reword them.
