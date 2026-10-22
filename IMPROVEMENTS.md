# Trit: improvements to implement

Review of the Trit runtime on 2026-10-21, after the English multi-runtime
benchmark. Nothing in this file is implemented yet, except where the
"Current state" section says so. Each item gives the evidence, the files,
the approach, and the check that proves it.

Measurement rules for every item:

- Accuracy check: `uv run --no-project --python 3.12 python bench/runtimes.py <engine>` from `desktop/` (all 456 English clips), then `python bench/runtimes_table.py <model>`.
- TDT ratchet (must stay 81 / 172 / 34 / 6): `uv run bench/accuracy.py <abs path>/target/release/bench-asr.exe clean other tech tts16 --engine native` from `desktop/`.
- Speed: run nothing else on the computer. One run gives about 10 % spread; compare two runs.
- Memory: see item M4 first; the current number is not stable.

## Current state (read first)

Uncommitted work in the tree, from the last session:

| Change | Files | Status |
|---|---|---|
| TDT crash fix ("RefCell already borrowed"): ShellExecute on its own thread; retry-microphone opens the recorder between updates | `desktop/src/ui/window_util.rs` (`open_file`), `desktop/src/main.rs` (retry block) | Done, builds, clippy clean. Not reproduced in the UI. |
| Tensor reads stream in 1 MB chunks; f16 layers read as f16 directly | `trit/src/cpu/tensors.rs`, `trit/src/cpu/gemm.rs` (`Dense::load`) | Done. Load memory: Whisper tiny 254 -> 123 MB, base.en 382 -> 191 MB, Moonshine tiny 181 -> 103 MB. |
| `mi_collect(true)` after a model loads | `trit/ffi/src/lib.rs`, `trit/src/main.rs`, `desktop/src/stt/engine.rs` | Done. |
| Buffer pool byte cap 64 MB | `trit/src/cpu/buf.rs` | Done. Little effect (Redux 739 -> 666 MB after 100 clips). Harmless. |
| Report page shows only "Each model on each runtime" | `desktop/bench/runtimes_report.html` | Done. |
| **Q8: narrow layers (< 1024 wide) stored as int8, f32 math, 32 values per scale** | `trit/src/cpu/gemm.rs` (`Q8`, `Q8_GROUP`, `Dense::load`), `trit/src/cpu/fgemm.rs` (`matmul_widened`) | **Experiment, still active. Decide first (item A1).** |

## A. Accuracy

### A1. Decide on Q8 for narrow layers (do this first)

Evidence, total errors over 456 clips, f16 against Q8 (32 values per scale):

| Model | f16 | Q8 | Best other runtime |
|---|---:|---:|---:|
| Whisper tiny | 892 | 912 | Transformers 896 |
| Whisper base.en | 517 | 518 | Transformers 519 |
| Moonshine tiny | 653 | 655 | Transformers 653 |
| Moonshine base | 477 | 476 | Transformers 479 |

Q8 with 128 values per scale was worse on the small sets too. Q8 makes
Whisper tiny worse than Transformers, which breaks the rule "Trit WER the
same or less". Approach, in order:

1. Restore f16 for narrow layers under `Precision::Int8` (one line in `Dense::load`: return `Self::half` when `!int8`), keep `Q8` and `matmul_widened` only if A1 step 2 uses them.
2. Try Q8 only for the decoder token embedding / output projection (the largest single tensor: 51864 x 384 in tiny, 51864 x 512 in base.en). Most of the memory gain, none of the encoder risk. Check that totals stay at or below the f16 row above.

### A2. Exact GELU (erf) for Whisper and Moonshine

`transformer::gelu` (`src/models/transformer.rs:88`) uses the tanh form. The
reference models (Transformers `ACT2FN["gelu"]`, OpenAI Whisper) use the
exact erf form. Whisper.cpp uses tanh. Approach: add `activation::erf`
(vectorized, Abramowitz-Stegun 7.1.26 or a minimax polynomial, max error
< 1e-6) and `gelu_exact`; use it for `hidden_act == "gelu"`. Keep tanh only
where a config says `gelu_new` / `gelu_pytorch_tanh`. Check A1 table models.
Expected: small, but it removes a known source of difference from the
reference.

### A3. Whisper temperature fallback (as OpenAI, faster-whisper, whisper.cpp)

Trit decodes greedy at temperature 0 only. The reference runtimes retry a
window at temperatures 0.2, 0.4, ... when the text compresses too well
(gzip ratio > 2.4) or the mean log probability is < -1. This fixes loops and
bad windows. Approach in `src/models/whisper/mod.rs` (`generate`, `transcribe`):
compute compression ratio (a small LZ-style or zlib-free byte-pair count is
enough; or use `miniz_oxide` if a dependency is acceptable) and mean log
probability; on failure, re-decode with seeded sampling (fixed seed so a
result repeats). Make it an `Options` field (`fallback: bool`, default on
to match the references). Check Whisper base.en test-clean (139 now,
whisper.cpp 134) and tiny/base developer terms (45 / 31 now, faster-whisper
40 / 30).

### A4. Whisper prompt as one batch, and `suppress_blank`

`transcribe` (`src/models/whisper/mod.rs:82`) runs the prompt tokens
(sot, language, task, notimestamps) as four decoder steps. Prefill them as
one batch (rows = 4, causal), as `qwen3.rs::forward` does. Speed only.
Also check that Trit applies `suppress_blank` exactly as Transformers (the
`begin_suppress_tokens` list covers it for HF configs; verify for older
checkpoints without `generation_config.json`).

### A5. Moonshine tiny test-clean and Moonshine base test-other

Trit gives exactly the Transformers output (653 = 653), so the gap to
sherpa-onnx (635) is not a Trit bug: sherpa-onnx runs its own export of the
original Useful Sensors model and writes differently on 108 of 456 clips.
Options, in order of value:

1. A2 (exact GELU) first; the HF port uses erf GELU in the conv stem and
   encoder MLP.
2. Token limit: Trit uses `6.5 tokens/s + 4` (`src/models/moonshine.rs:201`); the harness for Transformers used `+ 10`. Check whether any Trit output ends at the limit (count outputs with `ids.len() == limit`). If yes, raise to the reference rule.
3. Optional beam search (A7) for Moonshine only if the user accepts a
   decoding change beyond the reference runtimes.

### A6. Parakeet attention summation order

`matmul_nn` in Parakeet attention was 9 % faster on Redux but moved
test-other from 172 to 175 / 176 errors (float order). It is noise, but the
ratchet forbids it. To use it: enlarge the ratchet set (for example 1000
LibriSpeech clips per set via `bench/librispeech.py`) so 1-3 error swings
stop deciding; then re-test. Files: `src/models/parakeet/model.rs`
(`Block::attention`, around the three `fgemm::matmul` calls).

### A7. Beam search option (all decoder families)

`Options { beam: usize }` (default 1 = greedy, the reference). Whisper,
Canary, Moonshine, Qwen3. Keep beams in the KV cache layout
(`[beam][layer][tokens x d]`). Report results separately from the greedy
comparison, because the other runtimes ran greedy.

## B. Precision as a first-class option (user request)

Today precision is a process-wide environment variable read once
(`precision()`, `src/cpu/gemm.rs:239`). The user wants f32, f16, and int8 as
real choices.

1. `pub enum Precision { Auto, Int8, F16, F32 }` in `lib.rs` (`Auto` = the
   current rule: int8 for layers >= 1024 wide, f16 below).
2. `Trit::load_with(dir, &LoadOptions { precision })`; `Trit::load(dir)` =
   `Auto`. Same for `Cleaner`.
3. Pass the precision down without globals: put it in `Tensors`
   (`Tensors::open_with(path, precision)`), so `Dense::load(t, ...)` reads
   `t.precision()`. Every family loader already passes `&Tensors`.
4. CLI: `--precision auto|int8|f16|f32` on `transcribe` and `clean`.
5. C ABI: `Trit *trit_load_with(const char *model, const char *precision)`
   (NULL = auto); `trit_cleaner_load_with`. Header, three bindings
   (`precision=` / `Precision` / `{ precision }`).
6. Keep `TRIT_PRECISION` only as the default for `Auto`, or remove it (one
   way to do each thing).
7. Report: one row per precision for each model in `runtimes.py`
   (`trit-<model>-f32`, `-f16`, `-int8`).

## C. Speed

### C1. Int8 kernel for Intel client CPUs (AVX-VNNI) and ARM

`Isa::detect` (`src/cpu/kernels.rs:51`) knows AVX-512 VNNI, AVX2, scalar.
Intel Alder Lake and later have AVX-VNNI (VEX `vpdpbusd`, 256-bit) without
AVX-512: today they fall to the slower AVX2 `maddubs` path. Add an
`Isa::AvxVnni` path in `kernels.rs` and `int8.rs`. ARM (Apple Silicon,
Snapdragon) runs scalar everywhere: add NEON `sdot` / `i8mm` int8 and NEON
f32 FMA tiles. Check: same text on all paths (`TRIT_ISA=` forcing).

### C2. Linux and macOS thread topology

`topology()` returns `None` off Windows (`src/cpu/mod.rs:123`), so the team
neither pins nor picks an L3 domain there. Read
`/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list` and
`topology/core_id` on Linux, `sysctl hw.perflevel0.physicalcpu` on macOS;
pin with `sched_setaffinity` (Linux). Check Redux speed on a Linux box.

### C3. Team spread per call, not per process

The first model a process loads sets the team (`start_team`, `src/cpu/mod.rs:36`).
A process that loads Whisper and S1-mini, or Parakeet and Whisper, runs one
of them on the wrong spread. Make the team hold all physical cores and let
each `for_each` use either the pinned L3 subset or all workers (a per-call
`Spread` argument, or a team handle per model).

### C4. Whisper decoder step overhead

`Whisper::step` (`src/models/whisper/model.rs:214`) runs `attention` with
`tq = 1`, which dispatches `team::map` over heads and 128-row tasks for one
query. For one query, run the heads on the calling thread (or one task per
head with no row split) and fuse the K/V gather. Also keep the self K/V
cache transposed per head so no gather is needed per step.

### C5. Whisper encoder attention fused

Encoder attention builds a 1500 x 1500 score block per head and query block
in memory, then softmax, then P x V. A fused (flash-style) loop keeps a
running max and sum per row and never stores the full row: less memory
traffic and less allocator churn (see M2). Large models (turbo, large-v3)
spend about half the encoder time in attention.

### C6. Int8 VNNI tile tuning

`Int8::vnni_tile` (`src/cpu/int8.rs`) loads the activation broadcast from
memory for every frame and every k4 step. Pre-pack 4 rows of activations per
k4 step so one load feeds several broadcasts; try 8 frames x 2 blocks. The
turbo encoder runs at about 40 % of VNNI peak; 60 % would cut 0.92 s to
about 0.65 s per clip.

### C7. S1-mini prompt prefix cache

`Cleaner::clean` (`src/cleanup.rs:140`) prefills the fixed system prompt
(about 70 tokens) on every call. Run it once at load, keep its K/V, and
prefill only the control line and the transcript. About half the prefill
time per call.

### C8. Canary and Moonshine allocations per token

`Canary::generate` starts with `Vec::new()` caches and grows them per token
(`src/models/canary.rs`, `let mut keys = vec![Vec::new(); ...]`). Reserve
`prompt + max_tokens` rows, as Whisper does. Moonshine allocates `x`,
`gated`, and the gate copy per token; reuse buffers.

## M. Memory (user request: Whisper tiny and base.en)

### M1. Where the memory goes (measured)

Python harness alone: 36 MB. After load (with the streaming fix): Whisper
tiny 123 MB, base.en 191 MB. After 40 clips: tiny about 300 MB, base.en
about 390 MB. faster-whisper: 172 / 193 MB. The growth is allocator
retention: mimalloc keeps freed pages per worker thread, and worker threads
allocate temporaries (attention gathers, panels) that a later purge never
touches because the workers stop allocating. Measured options:

| Option | Memory after clips | Speed |
|---|---|---|
| `mi_option_purge_delay = 0` | Redux 739 -> 539, tiny 299 -> 178 | 30-45 % slower |
| `mi_collect(true)` after every call | Redux 565, tiny 222 | 6-24 ms per call slower |
| Pool byte cap 64 MB | Redux 666, Whisper unchanged | same |

So the fix is fewer allocations, not more collecting.

### M2. Allocate temporaries once, on the calling thread

In `transformer::attention` (`src/models/transformer.rs:176`) every task
allocates `qh`, `kt`, `vh`, and the probability rows; `fgemm::matmul_nn`
allocates its output `Buf` on the worker. Give each worker a thread-local
scratch arena (like `PANEL` in `fgemm.rs` and `QUANTIZED` in `kernels.rs`)
sized to the largest need, and write head outputs straight into the final
`out` buffer through `SendPtr` instead of returning `Vec<Buf>`. Same for the
Whisper mel (`frames`, `power` vectors in `src/audio/whisper_mel.rs:47`).
Check memory after 40 clips with `mem2.py` (see M4) and speed unchanged.

### M3. Smaller weights where they are big

- The token embedding and output projection are the largest tensors in
  Whisper tiny/base (51864 rows). Q8 for those two only (A1 step 2).
- `decoder_positions` and `encoder_positions` are f32; store f16.
- Mel basis (402 x 400 f32) and filters are small; leave them.

### M4. A stable memory measurement

The "memory after the run" column moves by 30-50 MB between runs of the
same code, because it depends on allocator state. Add to `runtimes.py`:
memory after load, after one warm clip, and after all clips, each read after
`gc.collect()`. Report "after load" and "after all clips" as two columns.
Keep the script used in this review as `desktop/bench/memory.py` (load a
model, transcribe N clips, print RSS after load and after the clips, and
the median time).

### M5. Idle trim API

`trit_release_memory()` in the C ABI (calls `mi_collect(true)`), and
`Trit::release_memory` / bindings. An app calls it when it goes idle (TDT:
some seconds after a dictation ends). No cost during work.

## D. Features and API

1. Long audio for Parakeet and Canary: today one pass over the whole
   recording (attention memory grows with the square of its length). Cut at
   quiet points with `audio::windows` (as Whisper does), with 1-2 s overlap
   and a merge of the overlapping words.
2. Use Redux's VAD head (`vad_head.*` tensors, 80 ms frames, listed in
   `ternary.json`) as a speech gate and a cut finder for every family,
   instead of the fixed -60 dBFS RMS gate (`audio::is_silent`).
3. Whisper dictionary: encode the terms with `Bpe::encode` and put them in
   the `<|startofprev|>` prompt (OpenAI `initial_prompt`). Canary and
   Moonshine: logit bias as in Parakeet's `bias.rs`.
4. `Options::task`: `transcribe` or `translate` (Whisper: `<|translate|>`;
   Canary: target language token different from the source).
5. Language detection for Canary (it needs the spoken language today):
   run Whisper-style detection only when the caller passes none, or document
   that English is the default.
6. Timestamps (segment level) for Whisper and Parakeet TDT (durations give
   frame times for free).
7. `Cleaner` input longer than about 1000 tokens: cut at sentence ends and
   clean in parts (the model card asks for it).
8. `trit models --download <name>`: needs an HTTP dependency; today the
   CLI prints the `hf download` command, which is enough for agents.

## E. TDT app

1. Confirm the crash fix: start TDT, open the dictionary (Settings), use
   "retry microphone", toggle the tray menu, dictate. Any remaining main
   thread COM or Shell call inside a GPUI `update` or `render` can panic the
   same way. Search: `grep -rn "ShellExecute\|CoCreateInstance\|AudioRecorder::new\|cpal" desktop/src`
   and check that each runs outside `this.update(...)` / render, or on its
   own thread.
2. Put a backtrace in the crash log: in the panic hook
   (`desktop/src/main.rs:114`) append
   `std::backtrace::Backtrace::force_capture()` to `tdt.log`. The crash
   dialog alone did not show where the second update came from.
3. After a crash the process waits behind the error dialog; seven such
   processes were running. Exit the process after the dialog
   (`std::process::exit(1)`), so a new start can claim the instance.
4. The app can use Parakeet v3 (148x realtime, fewer errors than Redux) as
   an option in Settings, now that int8 makes it fast.

## F. Code health

1. `src/models/parakeet/model.rs` has its own `Norm`, attention, and loader
   helpers that duplicate `models/transformer.rs`. Move the shared parts,
   keeping Parakeet's numbers bit-identical (ratchet).
2. `cpu_path()` reports the ternary/int8 ISA only; the float path has its
   own level (`fgemm::level`). `trit doctor` should print both.
3. `Dense::load` reads the env-based `precision()` on every call; with item
   B it reads it from `Tensors`.
4. README performance and precision sections must follow A1 and B.
5. `desktop/bench/runtimes.py` keeps experiment engines
   (`trit-parakeet-v3-f16/-f32`, `trit-whisper-tiny-f16/-f32`) that use the
   env variable; replace them with item B's precision option.
