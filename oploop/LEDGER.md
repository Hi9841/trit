# OpLoop ledger: Trit speed at beam-5 int8 accuracy

## Workspace

- Baseline: `oploop/baseline/trit_ffi.dll` and `trit-src.tgz`, the tree of 2026-10-22 03:12
  (Whisper/Moonshine 5 beams, Transformers rules; int8 from 384 wide; pool 16 MB; S1 int8 head).
- Work: the working tree (`trit/`). No commits: the owner asked for none; each kept change
  is a patch in `results/`.
- Machine: AMD Ryzen 9 7950X3D (16 cores, 2 CCDs), 32 GB, Windows 11.
- Runtime: Rust 1.98.1 release (fat LTO), Python 3.12 harness through the ctypes binding.

## Correctness floor

| Check | Command | Baseline result |
|---|---|---|
| WER totals | `ab.py ... --sets clean,other,tech,tts16` | tiny 760, base.en 468, moon-tiny 557, moon-base 415, Redux 293, v3 207 |
| TDT ratchet | `uv run bench/accuracy.py <bench-asr.exe> clean other tech tts16 --engine native` | 81/172/34/6 |
| Lint | `cargo clippy --workspace --all-targets --release -- -D warnings`, `cargo fmt --check` | clean |
| Skill | `trit skill --check skill/SKILL.md` | current |

## Noise

A/A Whisper tiny, 3 rounds x 200 clips: medians 64.8-66.9 ms, per-clip ratio 1.002,
p = 0.83, 0 texts differ.

## Journeys

| ID | Journey | Baseline (median, range, n) | Current | Change |
|---|---|---|---|---|
| J1 | Whisper tiny | 65.4 ms (64.8-66.9, 3x200) | | |

## Iterations

### Profile P-1 (Whisper, 200 test-clean clips, CLI, timers; one run, not a reported number)

| Stage | tiny ms | base.en ms |
|---|---:|---:|
| encode | 5581 (36 %) | 10863 (41 %) |
| decoder layers in beam steps (`whisper/model.rs` step_beams) | 3585 (23 %) | 6945 (27 %) |
| beam work: log-softmax, top-k, sort (`beam.rs` search) | 2651 (17 %) | 2571 (10 %) |
| output projection (`embed.forward`, f16 table 51864 rows) | 2060 (13 %) | 2952 (11 %) |
| cross K/V layout (`session`) | 597 (4 %) | 1229 (5 %) |
| prompt | 382 | 692 |
| mel | 132 | 121 |
| total | 15543 | 26198 |

### I-001 beam log-softmax without vocabulary-sized vectors

- Journey: J1 (also J2, J3)
- Profile: `beam.rs` `log_softmax` + `top` per beam per step: 2651 ms of 15543 (17 %) on tiny.
- Hypothesis: `log_softmax` allocates and fills two 51864-float vectors per beam per
  step (about 0.8 MB written per beam step), and `top` scans a third. Computing the
  log-sum-exp in a reused per-thread buffer and taking the top-k straight from the
  logits (minus the same normalizer) should cut the beam work by half or more:
  about 6 ms of 66 ms per tiny clip.
- Test: the same text on all 200 clips (A/B `texts differing: 0`) and the same errors:
  the arithmetic is the same, in the same order.
- Result: J1 tiny A 64.3 ms (62.9-65.5) vs B 61.7 ms (60.8-62.6), per-clip ratio 0.940,
  p = 0.0088, 3 rounds x 200 clips; errors 231 = 231, 0 of 200 texts differ.
- Other journeys: J2 base.en 110.1 vs 110.3 ms, ratio 0.997, p = 0.68 (no change), 0 texts differ.
- Decision: KEEP (`results/I-001.patch`). Base.en did not move: its beam work is a smaller
  share and the saving is inside its noise.

### Profile P-2 (Whisper encoder and decoder steps, 200 test-clean clips, timers)

| Part | tiny ms | base.en ms |
|---|---:|---:|
| encoder attention (f32, 1500 x 1500) | 2696 | 5224 |
| encoder dense + norms + GELU (int8 from 384 wide) | 2342 | 4936 |
| encoder conv stem | 395 | 517 |
| decoder self-attention (beams, per step) | 1148 | 1974 |
| decoder cross-attention | 1084 | 2330 |
| decoder dense | 915 | 1944 |

Encoder attention: ~6.9 G FMA per tiny clip in 13.5 ms = ~44 % of f32 FMA peak.
Encoder int8 dense: ~10.6 G int8 MAC in 11.7 ms = ~20 % of VNNI peak.

### I-002 int8 product parallel over frames too

- Journey: J1, J2 (also J3, J4 v3, J5)
- Profile: encoder int8 dense 2342 ms of 15543 (15 %) on tiny.
- Hypothesis: `Int8::forward` (`trit/src/cpu/int8.rs`) makes one task per 32 outputs and
  each task runs all frames. A 384-wide output (tiny attention out, fc2) gives 12 tasks for
  16 threads: 4 idle. Tasks over (frame chunk x output block) keep all threads busy;
  each output keeps the same sum in the same order. Expected: encoder dense about 25 %
  faster, about 3 ms of 62 ms per tiny clip.
- Test: 0 texts differ on 200 clips (same arithmetic per output).
- Result: J1 tiny A 62.3 ms (62.0-63.0) vs B 60.8 ms (58.9-62.6), ratio 0.975, p = 0.18;
  J2 base.en 113.0 vs 114.1 ms, ratio 0.995, p = 0.51; 0 texts differ.
- Decision: REVERT (no significant gain). Idle threads in narrow int8 layers are not the
  limiter of the int8 time; find it inside the layer before the next change.

### Profile P-3 (inside int8 layers, Whisper encoder, 200 clips)

| | tiny | base.en |
|---|---:|---:|
| activation quantization (`Quantized::new`) | 820 ms | 1522 ms |
| VNNI kernel | 1180 ms (2.2 T MAC/s) | 2788 ms (2.5 T MAC/s) |

### I-003 no zero fill of the reused quantized buffer

- Journey: J1, J2 (also J4 Redux and v3: the ternary and int8 products share `Quantized`)
- Profile: `Quantized::new` (`trit/src/cpu/kernels.rs:428`) 820 ms of 2000 ms int8 time on tiny.
- Hypothesis: `values.clear(); values.resize(m * k, 0)` zero-fills m x k bytes on the
  calling thread, single-threaded, before the parallel quantizer overwrites every byte:
  about 4.6 MB per tiny encoder layer, ~1.8 ms per clip. Growing the buffer only when it
  is too short removes the fill in steady state: about 1.5 ms of 62 ms per tiny clip.
- Test: 0 texts differ (the quantizer writes every value it reads later); Redux ratchet.
- Result: J2 base.en A 110.0 ms (109.4-117.4) vs B 107.3 ms (105.8-110.2), ratio 0.959,
  p = 0.0074; J1 tiny 61.7 vs 59.9 ms, ratio 0.952, p = 0.03; J4 Redux 34.8 vs 34.7 ms,
  p = 0.66. 0 texts differ on all three (231, 124, 81 errors on both sides).
- Decision: KEEP (`results/I-003.patch`).
