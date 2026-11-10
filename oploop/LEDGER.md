# OpLoop ledger: Trit speed at beam-5 int8 accuracy

## Workspace

- Baseline: `oploop/baseline/trit_ffi.dll` and `trit-src.tgz`, the tree of 2026-10-22 03:12
  (Whisper/Moonshine 5 beams, Transformers rules; int8 from 384 wide; pool 16 MB; S1 int8 head).
- Work: the working tree (`trit/`). The kept changes are in commit `cdec553`.
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
- Decision: KEEP (commit `cdec553`). Base.en did not move: its beam work is a smaller
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
- Hypothesis: `Int8::forward` (`src/cpu/int8.rs`) makes one task per 32 outputs and
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
- Profile: `Quantized::new` (`src/cpu/kernels.rs:428`) 820 ms of 2000 ms int8 time on tiny.
- Hypothesis: `values.clear(); values.resize(m * k, 0)` zero-fills m x k bytes on the
  calling thread, single-threaded, before the parallel quantizer overwrites every byte:
  about 4.6 MB per tiny encoder layer, ~1.8 ms per clip. Growing the buffer only when it
  is too short removes the fill in steady state: about 1.5 ms of 62 ms per tiny clip.
- Test: 0 texts differ (the quantizer writes every value it reads later); Redux ratchet.
- Result: J2 base.en A 110.0 ms (109.4-117.4) vs B 107.3 ms (105.8-110.2), ratio 0.959,
  p = 0.0074; J1 tiny 61.7 vs 59.9 ms, ratio 0.952, p = 0.03; J4 Redux 34.8 vs 34.7 ms,
  p = 0.66. 0 texts differ on all three (231, 124, 81 errors on both sides).
- Decision: KEEP (commit `cdec553`).

### I-004 TDT beam search (Parakeet Redux and v3), 2026-10-23

- Hypothesis: a frame-synchronous beam over TDT paths (top tokens x every
  duration, paths with the same tokens merged) lowers word errors of
  Parakeet Redux, TDT's model.
- Result (trit CLI, whole clips, errors clean/other/tech/tts16, median ms):
  Redux greedy 78/180/35/8 = 301 (27-44 ms); beam 2: 300 (108-133 ms);
  beam 4: 301 (276-350 ms); beam 8: 301 (435-539 ms). Parakeet v3 greedy
  207 (31-40 ms), beam 4: 202 (262-333 ms).
- Decision: REVERT. No gain on Redux; 2.4 % on v3 at 8x the time is not
  worth about 190 lines. The patch is kept outside the repo.

### I-005 app path: parts of at least 10 s between pause cuts, 2026-10-23

- Hypothesis: the app cuts a take at the first 500 ms pause after 3 s and
  transcribes each part alone; the cuts inside sentences cost words.
- Result (app path, word errors clean/other/tech/tts16): Redux 3 s
  81/185/34/8 = 308, 10 s 82/173/34/6 = 295, whole clips 293. v3 215 -> 208.
  Wait after release, 20 longest test-other clips at real speed (median):
  Redux 34 -> 80 ms, v3 39 -> 91 ms; 15 s parts: 90 / 97 ms, no fewer errors.
- Decision: REVERT. The owner judged 46 ms more wait not worth 4 % fewer errors;
  next: keep 3 s parts and give each part the audio before its cut as context.

### I-006 app path: the audio before a cut as context for the next part, 2026-10-23

- Hypothesis: keep 3 s parts, but decode each part with 1 to 4 s of the
  audio before its cut (started at the quietest 10 ms) and write only the
  words after the cut (new Options::context).
- Result (app path, errors all sets): Redux 308 -> 303; v3 215 -> 222.
  A context that starts inside a word made v3 write nothing for a whole
  part; the quiet start fixed that, but v3 stayed worse.
- Decision: REVERT. Too small and mixed for a new public option. Patch kept
  outside the repo.

## Sessions of 2026-10-27 (run in the TDT checkout before Trit moved here)

Their iteration IDs start with S, apart from the I-004 to I-006 above. Paths in
`oploop/results/` and `trit/src/...` are those of the TDT checkout; the code is
now in `src/cpu/team.rs`, `src/cpu/half.rs`, `src/models/parts/fastconformer.rs`,
`src/models/parts/transducer.rs`, and `src/models/families/parakeet.rs`.

### Session 2026-10-27: latency stability (J6)

The user saw 145 ms for "We need different icons." (about 2 s of speech); the same
engine takes about 12 ms for a phrase of that length. The app history
(`%APPDATA%\tdt\TDT\config\stats.json`) shows the app 2 to 10 times slower than the
isolated bench on the same takes, for example 144 ms in the app against 64 ms in the
bench for take-1793044070928 (10 s).

| ID | Journey | Command (from `desktop/`) | Metric |
|---|---|---|---|
| J6 | One Redux take by engine state and CPU load | `uv run --no-project --python 3.12 python ../oploop/idle.py --lib <dll> --states <s> --load <N>` | median and max ms per clip, n rounds |

Clips: three of the user's own takes (`target/accuracy/takes`, not in git): 1.6 s,
2.9 s, 10 s.

### Profile P-4 (J6 by state, load 0, n = 6)

| State | 1.6 s | 2.9 s | 10 s | faults (1.6 s) |
|---|---:|---:|---:|---:|
| warm | 11.5 | 18.4 | 63.9 | 40 |
| released (the app's state 5 s after a take) | 12.8 | 19.6 | 60.2 | 1934 |
| trimmed (EmptyWorkingSet) | 24.1 | 30.8 | 75.3 | 55386 |
| idle 15 s | 12.1 | | | 103 |
| released + idle 15 s | 12.8 | | | 2026 |

The idle release costs about 1 ms; idle time costs nothing; a trimmed working set
costs about 12 ms. None of them gives 145 ms.

### Profile P-5 (J6 warm, by CPU load, n = 15)

| Busy processes (32 logical CPUs) | 1.6 s | 2.9 s | 10 s |
|---|---:|---:|---:|
| 0 | 11.4 (max 12.0) | 18.7 (24.7) | 62.3 (66.4) |
| 16 | 15.8 (19.9) | 26.6 (29.2) | 93.2 (101.8) |
| 32 | 519.2 (793.0) | 516.0 (660.5) | 935.9 (1029.9) |

### SI-004 team threads above normal priority

- Journey: J6 under load (also J4: no change expected at load 0)
- Profile: P-5. At full load a 1.6 s take is 45 times slower.
- Hypothesis: `trit/src/cpu/team.rs` runs about 1,200 parallel steps per take, and the
  caller spin-waits each step until every worker is done (`busy == 0`). The workers
  and the caller run at normal priority, so with all CPUs busy each step waits for
  a descheduled worker to get a new quantum. Running the workers, and the caller while
  it is inside the engine (`enter`), at THREAD_PRIORITY_ABOVE_NORMAL lets them preempt
  normal-priority work. Expected: J6 at load 32 within 2 times of load 0 (under 30 ms
  for the 1.6 s clip), load 0 unchanged, same text.
- Test: same text on the J6 clips and the Redux ratchet (81/172/17).
- Result, back to back (no gap), load 32, n = 8 x 2 processes per side: A 3096 / 3856 ms
  median (1.6 s clip), B 39 / 34 ms; load 0: A 11.6 / 11.3 ms, B 11.1 / 11.6 ms. Same
  text on every clip. Gate: ratchet 81/172/17 (equal), clippy and fmt clean.
- Result, 1 s gap before each take (closer to real dictation), load 32, n = 10 x 3
  processes per side, 1.6 s clip: A 676 / 717 / 717 ms; B 2001 / 78 / 1968 ms. B is
  bimodal per process and its slow mode is worse than A. Load 24 (no
  oversubscription), n = 6 x 3: A 19-25 ms, B 20-27 ms, no difference.
- Decision: REVERT. Not better beyond noise; worse in some processes.
- Trace (temporary build, not kept; `oploop/results/T_trace_trit_ffi.dll`): in a slow
  process, 1065 stalled steps in 16 takes. On each, the caller's own share of the step
  took about 9.3 ms (normal: under 1 ms), the same for 8 or 125 tasks: a fixed wait,
  not slow arithmetic. The workers then spin out (3 ms) and park mid-take
  (`parked-at-start` 4-6), so each step also pays a wake-up.
- A team at THREAD_PRIORITY_TIME_CRITICAL (test build `X_tc_trit_ffi.dll`, not a
  candidate) is bimodal the same way (2086 / 36 / 2071 / 79 ms). So the starved
  threads' priority boost is not the cause. Next: an ETW context-switch trace (WPR,
  needs an elevated shell) of one slow process to name the wait.

### Journey J7: latency after release through the app path, under native load

`python ../oploop/realtime.py <bench-asr.exe> [<B bench-asr.exe>] --loads 0,24,32 --runs 3`
from `desktop/`: `bench-asr --realtime` feeds 8 of the user's takes (1.6 to 10 s,
`target/accuracy/takes-short`) at speaking pace through
`SttEngine::transcribe_live_reporting`; `oploop/results/spin.exe N` (native) is the load.

Baseline A (`oploop/results/realtime-A.txt`), Redux, n = 24 takes per load:

| Busy threads | median | p90 | max |
|---|---:|---:|---:|
| 0 | 26 | 57 | 87 |
| 24 | 43 | 86 | 136 |
| 32 | 6358 | 7907 | 10277 |

At load 32 the three processes gave medians of 698, 7006, 6433 ms: bimodal in the app
path too, so the Python harness did not cause it.

### SI-005 yield the CPU while a team thread waits

- Journey: J7 at load 32 (also J7 at load 0, J4: must not slow down)
- Hypothesis: a step ends only when all 8 team threads (pinned to one CCD) have run.
  With more ready threads than CPUs, the team threads that run spin and use their own
  time slice, while a teammate waits for a CPU, often one that a spinning teammate
  holds. Calling `std::thread::yield_now` (SwitchToThread) in the wait loops, after a
  short pure spin, gives that CPU to a ready thread, often the teammate. With no load,
  SwitchToThread returns at once when nothing else is ready. Expected: J7 at load 32
  under 200 ms median; load 0 within noise of 26 ms; same text.
- Result (`oploop/results/realtime-I005.txt`, stopped after the first load-32 pair):
  load 0 A 25/25/26 ms vs B 25/24/24 ms; load 32 A 686 ms vs B 17152 ms median.
- Decision: REVERT. A thread that yields goes to the back of the queue, and the busy
  threads, which never yield, take its CPU for a full time slice.

### Profile P-6 (J7, app path, normal priority; temporary trace build `T5_trace_bench-asr.exe`)

Per take: the caller's wait after its own tasks, split into the time while tasks still
run and the time after every task is done (waiting only for workers to check in).

| Load | latency | wait while tasks run | wait after all tasks done |
|---|---:|---:|---:|
| 0 | 16-84 ms | 0.6-17 ms | 0.1-0.2 ms |
| 32, slow process (2 of 3) | 5781-9030 ms | 335-1778 ms | 3015-5050 ms |
| 32, fast process (1 of 3) | 456-846 ms | 138-514 ms | 111-221 ms |

### SI-006 a step ends when its tasks are done, not when every worker checked in

- Journey: J7 at load 32 (J7 at load 0 and J4 must not slow down)
- Hypothesis: `Team::for_each` (`trit/src/cpu/team.rs`) waits for `busy == 0`, so every
  worker must get a CPU once per step, even one that took no task. P-6: that wait is
  3-5 s of a 6-9 s take at load 32. Claiming tasks with one atomic word that holds
  (job number, task count, next task), and waiting for a count of finished tasks,
  lets the threads that run finish the step: a worker that wakes late finds no task
  and no job to read. Expected: J7 at load 32 below the fast-process level (under
  500 ms median), with no slow mode; load 0 within noise; same text and ratchet.
- Correctness: same text as A on 486 clips (takes 46, clean 200, other 200, tech 40;
  `bench-asr --native`, times removed, `cmp`). Ratchet 81/172/17 (equal). Clippy and
  fmt clean on trit and desktop.
- Result (`oploop/results/realtime-I006.txt`), A = `A5_bench-asr.exe`, B =
  `B6_bench-asr.exe`, interleaved, 3 processes x 8 takes per side and load:

  | Load | A median / p90 / max | B median / p90 / max | rank-sum p (B faster) |
  |---|---|---|---:|
  | 0 | 32 / 76 / 119 | 32 / 72 / 110 | 0.55 (no change) |
  | 24 | 286 / 486 / 632 | 164 / 326 / 359 | 0.0026 |
  | 32 | 5385 / 6349 / 8992 | 944 / 2105 / 2628 | 6.5e-06 |

  The prediction (under 500 ms at load 32) was not met: the gain is 5.7 times, and the
  6 s mode is gone (B process medians 1042, 1112, 308 ms).
- Decision: KEEP (`trit/src/cpu/team.rs`, copy in `oploop/results/team.rs.I006`).
  Not committed: the file also holds earlier uncommitted work (the panic handling).
- Remaining at load 32: about 1 s. By P-6 it is the wait while claimed tasks run (a
  worker preempted inside its task) and the caller's own preemption.

### Profile P-7 (J7, SI-006 kept, load 32; temporary trace build `T6_trace_bench-asr.exe`)

Per take (whole take, speech included), 3 processes x 8 takes:

| | load 0 | load 32 |
|---|---:|---:|
| caller's own tasks | 13-100 ms | 90-683 ms |
| caller's wait for tasks other threads claimed | 1-10 ms | 114-613 ms |
| caller's share of tasks | 8-13 % | 13-50 % |
| latency after release | 19-85 ms | 300-1304 ms |

A Redux step already has 32 or more tasks for 8 threads (`kernels.rs:194`), so smaller
tasks would not help: a worker paused inside a task resumes only at its next time
slice, whatever the task size.

### SI-007 team threads above normal priority, on top of SI-006

- Journey: J7 at loads 24 and 32 (J7 at load 0 must not slow down)
- Hypothesis: SI-004 failed while each step waited for all 8 threads. With SI-006, only a
  thread that holds a task can delay a step, and P-7 shows that delay (114-613 ms) plus
  the caller's own paused time (90-683 ms) are now most of the take. Workers and the
  caller inside `enter` at THREAD_PRIORITY_ABOVE_NORMAL are not paused by
  normal-priority work. Expected: J7 at load 32 under 200 ms median; load 0 unchanged.
- Correctness: same text as B6 on 486 clips; ratchet 81/172/17; lint clean.
- Result (`oploop/results/realtime-I007.txt`), A = B6 (SI-006), B = B7:

  | Load | A median / p90 / max | B median / p90 / max | p (B faster) |
  |---|---|---|---:|
  | 0 | 26 / 56 / 84 | 25 / 58 / 83 | 0.6 |
  | 24 | 47 / 123 / 205 | 38 / 88 / 137 | 0.13 |
  | 32 | 566 / 639 / 688 | 136 / 898 / 1901 | 0.0096 |

  B process medians at load 32: 128, 838, 125 ms: bimodal again.
- Decision: NOT KEPT ALONE. The median is better, but the tail (p90, max) is worse,
  and the tail is the target of this session. Kept in the tree only as the base of
  SI-008; if SI-008 fails, both go out.

### SI-008 one ideal processor per team thread, on top of SI-007

- Hypothesis: Windows assigns ideal processors round-robin per process. A ready
  thread above normal priority preempts a lower-priority thread only on its ideal
  or last processor; if two team threads share an ideal processor, one waits a time
  slice behind its teammate while other CPUs of its set run normal-priority work.
  That is a per-process chance, which fits the per-process bimodality of SI-004 and
  SI-007. Giving thread i of the team (caller 0, workers 1..) the i-th physical core of
  its set as ideal processor removes the shared case. Expected: J7 at load 32 with
  every process near 130 ms median and max under 300 ms; load 0 unchanged.
- Correctness: same text as B6 on 486 clips; ratchet 81/172/17; clippy and fmt clean.
- Result, load 32, B6 / B7 / B8 interleaved, 6 processes x 8 takes each
  (`oploop/results/realtime-I008-load32.txt`):

  | Build | median | p90 | max | process medians |
  |---|---:|---:|---:|---|
  | B6 (SI-006) | 662 | 1372 | 1597 | 514-1164 |
  | B7 (+ priority) | 316 | 1416 | 1833 | 2 of 6 near 900 |
  | B8 (+ ideal processors) | 128 | 164 | 197 | 117-145 |

  B8 vs B6 p = 1.6e-17; B8 vs B7 p = 8.5e-05.
- Loads 0 and 24, B6 vs B8, 3 processes (`realtime-I008-load0-24.txt`): load 24 median
  67 -> 36 ms, max 266 -> 133 ms (p = 0.035). Load 0: 25 vs 26 ms; a paired test per
  take gave +0.9 ms (p = 0.0021).
- Load 0 again, 6 processes, B6 / B8 / B8b (odd CPUs, to avoid CPU 0)
  (`realtime-I008b-load0.txt`), paired per take, n = 48: B8 +0.48 ms (p = 0.031),
  B8b +0.33 ms (p = 0.098). Under the project's p < 0.01 rule this is not a
  significant cost; at most about 1 ms of a 25 ms take. B8 and B8b are within noise
  of each other, so B8 stays as built.
- Decision: KEEP SI-007 and SI-008 together (`oploop/results/team.rs.I008`). The
  slow mode is gone in all 6 processes. Not committed (the file holds earlier
  uncommitted work).

#### Session result: original (A5) vs current tree (B8), one interleaved run

Files: `oploop/results/realtime-session-A5-vs-B8.txt` (load 0, stopped after 4 A and
3 B processes) and `realtime-session-A5-vs-B8-loaded.txt` (3 processes x 8 takes).

| Load | A5 median / p90 / max | B8 median / p90 / max | p (B8 faster) |
|---|---|---|---:|
| 0 | 24-27 per process, max 85 | 25-26 per process, max 91 | (see SI-008 paired test) |
| 24 | 37 / 87 / 135 | 34 / 84 / 131 | 0.33 |
| 32 | 1038 / 4258 / 4624 | 60 / 132 / 139 | 1.5e-09 |

Load 24 varies between runs on this machine (A medians 37 to 286 ms in earlier runs);
in this run it shows no difference. Load 32: A process medians 816, 4217, 844 ms;
B8 60, 64, 64 ms.

### Session 2026-10-27 (2): engine review, Parakeet attention

### Profile P-8 (J8: 44 of the 46 takes, CLI, one process, load 0; temporary timer build, `oploop/results/prof_patch.py`)

Median take 61 ms (sums over the 24 blocks):

| Part | Share | Median per take |
|---|---:|---:|
| Feed-forward ternary products (up + down, both modules) | 38 % | 25.5 ms |
| Attention heads (scores, softmax, values; `team::map` over 8 heads) | 19 % | 8.7 ms |
| qkv, o, pw1, pw2 ternary products | 18 % | 11.2 ms |
| Subsampling | 9 % | 5.7 ms |
| Decoder (greedy TDT) | 8 % | 5.5 ms |
| Mel, speech head, depthwise conv | 3 % | 1.8 ms |

The attention ran one task per head (8 tasks for 8 threads), allocated its buffers on
the workers, computed 2t-1 position columns per query where t are used, and copied
every head's output on the caller after the step.

### SI-009 attention per head and block of queries, broadcast-form products

- Hypothesis: tasks per head and block of 32 queries (`transformer::attention_with`
  layout), `matmul_nn` (no horizontal sums), a band of t+m-1 position columns per
  block, thread-local scratch, output written in the task. Expected: heads part about
  half, take 5-10 % faster.
- Result (`oploop/results/ab-I009b.txt`), A9 (tree) vs B9, 3 rounds x 486 clips:
  31.2 -> 30.1 ms median, per-clip ratio 0.957, paired signed-rank p = 2.3e-69.
  Errors: takes 105/105, clean 81/81, other 172 -> 175, tech 34/34; 35 texts differ.
- Decision: REVERTED. `other` breaks the ratchet. The text change comes from the
  order of the sums: activations are rounded to 8 bits before each ternary product, so
  a last-bit difference moves some roundings and grows. A change to any sum's order
  must therefore pass the ratchet per set; identical text is not reachable for it.
- Harness: `ab.py` now prints a paired signed-rank test on the per-clip medians (the
  rank-sum test pools clips of very different lengths and gave p = 0.045 for the same
  data), the errors per set, and writes all rows with `--save FILE`.

### SI-009c the SI-009 structure with dot-form products (the old sums)

- Hypothesis: keep the task structure, scratch, band, and direct output of SI-009, but
  use the dot-form kernel (`fgemm::matmul_into`, new: `matmul` on the calling thread).
  A value of that kernel does not depend on the tile around it, so every value equals
  the old one: same text. K as rows and V transposed per head are laid out once per
  layer on the caller (pool); the position table is cached per head
  (`[head][position][hd]`), so a block's band of positions is consecutive rows.
- Result (`oploop/results/ab-I009c.txt`), A9 vs B9c, 3 rounds x 486 clips: 31.1 ->
  30.2 ms median, per-clip ratio 0.973, paired signed-rank p = 6.4e-65. Errors equal in
  every set (105/81/172/34); 0 texts differ.
- Decision: KEEP (`trit/src/models/parakeet/model.rs`, `trit/src/cpu/fgemm.rs`).

### SI-010 a per-thread buffer for the conv0 rows of the subsampling strips

- Finding: `conv_strip` (`parakeet/model.rs`) allocated and zero-filled up to 1 MB per
  task on the workers (`vec![0.0; rows * w1 * ch]`); `trit/AGENTS.md` forbids allocation
  per task on worker threads. `conv0_row` writes every value it reads, so no fill is
  needed.
- Change: a thread-local `CONV0` buffer, taken and put back around the loop.
- Result (`oploop/results/ab-I010.txt`), B9c vs B10: 30.3 -> 30.0 ms median, per-clip
  ratio 0.998, paired p = 5.2e-05. 0 texts differ, errors equal per set.
- Decision: KEEP (small gain; removes a breach of the worker rule).

### Dictionary lock (review fix, no speed claim)

`Parakeet` (`parakeet/mod.rs`) kept the dictionary and the English ban without its
pieces in two locks, written one after the other. A `set_dictionary` call during a
transcription on another thread could pair the new dictionary with the old ban. Both
are now one `RwLock<Terms>`. The app sets the dictionary on the transcribing thread,
so the app path did not hit it; the library promises `Send + Sync`.

### SI-011 four rows at once in the f16 matrix-vector product (decoder)

- Hypothesis: `HalfDense::matvec` (joint 8198 x 640 per frame, LSTM 2 x 2560 x 1280 per
  token) ran one row as one chain of dependent FMAs, bound by their latency. Four
  rows side by side, each with its own accumulator, overlap the chains and share the
  loads of `x`; each row's sum keeps its order, so the values do not change.
- Result (`oploop/results/ab-I011.txt`), B10 vs B11: 30.6 -> 29.9 ms median, per-clip
  ratio 0.987, paired p = 2.1e-41. 0 texts differ, errors equal per set.
- Decision: KEEP (`trit/src/cpu/half.rs`).

### Profile P-9 (J8 again, SI-009c, SI-010, SI-011 kept)

Median take 61.0 -> 54.9 ms (same 44 takes, same temporary timer build method). Heads
8.0 -> 5.9 ms, decoder 5.2 -> 4.4 ms. Ternary products are now 61 % of a take, and at
about 2.1 T multiply-adds per second they run near the 8-core VNNI peak (one 512-bit
`vpdpbusd` per cycle, about 2.6 T). The subsampling's f32 products (pw1 1.8 ms,
linear 1.6 ms) run at about half the f32 peak; a faster form changes the order of
their sums and so the text (see SI-009).

### Gates of this session's tree

- Clippy `-D warnings --release` and fmt: clean on trit and desktop.
- Ratchet (`accuracy.py`, `B11_bench-asr.exe`, native): clean 81, other 172, tech 17
  with `TDT_DICTIONARY=assets/dictionary/developer-terms.txt` (A5: 17). No unit tests in
  the repo (no `#[test]`, `#[cfg(test)]`, test folders, or `cargo test` in CI).
- Text: each kept change gave 0 differing texts of 486 against its parent.

### Port to this repository (Trit 0.3.1)

SI-006 to SI-011, the dictionary lock, the precision fix (Parakeet head, subsampling,
LSTM, and embedding load in the requested precision; `Auto` unchanged), and the CTC
ban guard, ported onto v0.3.0. Interleaved A/B, v0.3.0 vs the port, 3 rounds x 486
clips (`ab.py`): 30.9 -> 29.7 ms median, per-clip ratio 0.961, paired p = 1.9e-69;
0 texts differ; errors equal per set (takes 105, clean 81, other 172, tech 34).
Control: `--precision int8` changes 24 of 200 `other` texts (0 before the fix).
Clippy, fmt, and `trit skill --check` clean.

## Session 2026-10-28 (stability)

Goal: the speech-to-text path (Parakeet Redux and v3 first) must not crash, hang,
give wrong or empty text, or grow memory. Method: read the code, then prove each
defect with a real run through `bindings/python/trit.py` (one process per input,
probes kept outside git). Each kept change is a fix, not a speed claim.

### Probe P-S1 (inputs, Redux, before the fixes)

- No defect: empty audio, 1 sample, 0.05 to 0.13 s, 5 s of zeros, DC 0.5, DC under
  speech, speech clipped at full scale, a square wave, uniform noise, int16-scale
  samples (x 32768), 1e-30 and 1e-40 scale (empty, as digital silence), 120 s
  (0.8 s) and 600 s (4.2 s) of speech, `segments` on 120 s.
- 1000 calls on one engine (86 takes and tech clips, cycled): 0 texts change
  between repeats; private bytes 1481 MB after 100 calls, 1417 to 1519 MB up to
  1000, 1343 MB after `release_memory`. `release_memory` after every call (300):
  flat at 1343 MB. One take in three made three times longer (300): 1468 MB, 1344
  MB after release. No growth.
- Two threads on one engine and on two engines, 3 x 20 clips each, with
  `release_memory` from a third thread every 50 ms: 0 texts differ, no error.
- Dictionary: empty list, empty and blank terms, non-ASCII, punctuation, control
  characters, 5000 terms, 100 copies of one term: no error, same text.

### SI-S1 sanitize the samples (`src/audio/mod.rs`, `src/lib.rs`)

- Defect: one NaN, +Inf, -Inf, or 3.4e38 sample in a 10.6 s clip gave empty text
  (`transcribe` and `segments`): the log-mel of its window became NaN. Speech at
  1e20 scale gave empty text too (the power spectrum overflows f32).
- Change: `audio::sanitized` maps non-finite samples to 0 and clamps finite ones
  to +-1e10, before every family. Samples in range are borrowed, not copied.
- Result: the four one-sample cases give the clean text; 1e20 scale gives the
  text with 2 words changed; all-NaN gives "". Commit `7601d9e`.

### SI-S2 safetensors offsets near 2^64 (`src/cpu/tensors.rs`, coldstart's file)

- Defect: an end offset of 2^64 - 1 wrapped past the length check; the load
  panicked with "capacity overflow" (the C ABI gave "internal error (panic)").
- Change: `checked_add` on the offsets. Result: "tensor ... points outside the
  file". Commit `216d288`.

### SI-S3 config and tokenizer that do not fit (`fastconformer.rs`, `parakeet.rs`)

- Defects: `blank_token_id` 99999 loaded, and then every call panicked in the
  embedding lookup (`gemm.rs:234`). A `tokenizer.json` with 2 pieces loaded, and
  every call gave "".
- Change: `blank < vocab` in `Config::parse`; Parakeet load fails when the
  tokenizer has fewer pieces than `vocab - 1`. Commit `567e2fc`.

### SI-S4 long dictionary terms (`dictionary.rs`)

- Defect: `Tokenizer::encode` merges in time quadratic in a word's length. One
  term of 10,000 letters took 2.2 s in `set_dictionary`; 100,000 would take
  minutes.
- Change: terms over 100 characters are skipped (returned as skipped).
  Result: 0.11 s. Commit `d9b61eb`.

### SI-S5 at most 16 beams (`src/lib.rs`, PUBLIC API behavior)

- Defect: `trit_transcribe_with(beam = 10^8)` on Whisper tiny held 14 GB within
  a minute and still grew (each beam clones its decoder caches); the probe was
  stopped by hand.
- Change: `Options::beams` caps at 16; the signature does not change.
  Result: the call returns, with the same text as beam 16. Commit `dbaf577`.

### Not fixed

- One word (1 s) after 40 s or more of silence gives "" (35 s: "At most."; 40,
  50, 58 s of zeros, and 50, 58 s of noise at -70 dBFS: ""). The per-bin
  normalization of `audio/mel.rs` runs over the whole window, so the long
  silence sets the mean and spread. A fix changes the features of every take.
- `RelCache` (`fastconformer.rs`) grows from 375 to 750 positions after the
  first take over 30 s and keeps them: about 74 MB (24 layers x 751 rows x
  1024 x 4 B) that `release_memory` does not return. Bounded by the 60 s window.
- `durations` in `config.json` are not checked: a duration of 2^62 loads and
  gives "".

### Gates of the branch head (`dbaf577`)

- `ab.py parakeet-redux`, baseline v0.3.1 vs head, 1 round, 486 clips: 0 texts
  differ; errors takes 105, clean 81, other 172, tech 34 on both sides.
- `ab.py parakeet-tdt-0.6b-v3`, the same: 0 texts differ; errors takes 56, clean 58,
  other 124, tech 20 on both sides.
- Clippy `-D warnings --release` and `cargo fmt --all --check`: clean.

## Session 2026-11-01 (ultra)

Branch `feat/parakeet-ultra` from `origin/main` (0.4.0). Machine: Ryzen 9 7950X3D,
RTX 4090, Windows 11. Probe (not in the repo): `%TEMP%\ultra-scripts\floor`,
`floor.exe <model> <clip> cpu|auto|switch` (memory after load, after 10 takes, after
`release_memory`; `switch` = load on the GPU, one take, drop, load on the CPU).

### Parakeet Ultra (moondream/parakeet-ultra)

Parakeet TDT 0.6B v3 trained further, f16 matrices, f32 norms, plus `vad_head.*`.
It loads with no engine change. CLI, CPU, 456 clips, whole clips: 260 errors (v3 262,
Redux 356), 36.8 ms median on clean. On the GPU it is slower than on the CPU (101
against 59 ms a take): its int8 CPU weights become f16 on the GPU.

### G-001 the GPU path keeps memory

- Hypothesis: `release_memory` does not clear the GPU pool, and the device allocates
  large blocks, so a GPU model costs RAM and keeps it after the drop.
- Change (`src/gpu/mod.rs`, `release_memory`): `MemoryHints::MemoryUsage`; weight uploads
  flush the queued writes every 32 MB (wgpu kept one staging copy of the whole model
  until the next submit); `gpu::trim` frees the pooled activation and staging buffers
  and is called by `release_memory`. The device stays.
- Test: 200 clean clips, Redux and Ultra, `--device gpu` and `--device cpu`, before and
  after: 0 texts differ in the first round. Redux on the GPU is not stable run to run
  (one punctuation difference per round, in both builds).
- Result (probe, idle, working set / private MB): Redux GPU 736 / 1277 to 438 / 929;
  Redux GPU dropped and released 358 / 731 to 100 / 390; Ultra GPU load peak working set
  2143 to 1071 MB. CPU numbers and take times: same.
- Decision: KEEP.
- Left: the wgpu device and the driver keep about 75 MB working set and 300 MB private
  after the drop. The CPU copy of GPU weights stays (short takes under 48 frames, the
  decoder rows under 48, `warm_positions`, and the GPU fallback read it).

### Open: Ultra on the website (owner: investigate first)

- `bench/runtimes.py` (3 rounds): Trit int8 (the default) 200 errors, 36 ms a clip,
  1050 MB; Trit f16 and f32 198 (127 ms); Transformers fp32 197 (248 ms, 3719 MB).
  `wer_rule.py` would fail on Ultra, so Ultra is not in `TEST_MODELS` or
  `bench/benchmarks.json` yet. The results are in `target/runtimes/results.jsonl`.
- The gap by set: other 119 (int8) / 117 (f16) against 116; tech 20 / 19 against 19;
  clean 56 / 57 against 57.
- Next: find the layers whose int8 products cost the words (Ultra's source weights are
  f16, so int8 is a second rounding), and the one clip where f16 loses a word.
- Sensitivity: on Redux, a gain of 1.00003 (`/ 32767` against `/ 32768`) changes 15 of 46
  take texts and 5 errors on `other`. Error counts that differ by a few words are within
  this noise.

## Session 2026-11-01 (memory)

### M-001 the weight cache (`cpu/store.rs`, PUBLIC API: `LoadOptions::cache`, `Trit::load_reporting`)

- Measured: Parakeet Ultra, private memory 828 MB after load and `release_memory`,
  live heap 734 MB (int8 encoder 592 MB, `RelCache` 70 MB, f32 joint and
  subsampler 39 MB, LSTM and embedding 23 MB). mimalloc held about 120 MB
  over the live heap (Redux too: 425 MB against 300 MB live); the
  `MIMALLOC_ARENA_EAGER_COMMIT`, `MIMALLOC_PURGE_*` options did not change it.
- Rejected: the system allocator as the global allocator. It removes the
  gap (Redux 315 MB, Ultra 746 MB), but `ab.py` gives Redux 4.5 % and Ultra
  4.0 % slower (p < 1e-38), 0 texts differ.
- Change: large weight buffers are `Store`s. With `cache`, the first load
  writes them to `<model>/.trit/`, then loads again from the mapped file; a
  buffer is mapped only when its bytes equal the fresh conversion.
- Result (probe, after `release_memory`): Redux private 425 -> 24 MB, working
  set 324 -> 313 MB; Ultra private 828 -> 21 MB, working set 800 -> 747 MB.
  Load: Redux 110 -> 146 ms, Ultra 354 -> 458 ms; first load 0.75 s and 2.0 s.
- Gate: `ab.py` v0.4.1 against the cache build, 3 rounds, clean, other, tech,
  takes: Redux 0 of 486 texts differ, errors 392 = 392, medians 30.2 and 30.1 ms
  (per-clip ratio 1.006); Ultra 0 of 486, errors 255 = 255, 34.1 and 34.1 ms
  (ratio 0.994). The two ratios go opposite ways: noise.

## Session 2026-11-04 (stable under load)

- Load on the team's own L3 domain, 7950X3D, 16 workers on both hardware threads of
  CCD0, Parakeet Unified, a 20 s take, p50: quiet 120 ms, loaded 222 ms (1.9 times).
  `THREAD_PRIORITY_ABOVE_NORMAL` does not help: the load is on the second hardware
  thread of each team core. The same take with the process on CCD1: 159 ms.
- Quiet, the model runs at the same speed on both CCDs: 120 ms on CCD0 and on CCD1.
  The V-cache gives Parakeet nothing here, so `pick_domain` stays as the first choice
  only, and the team may leave it.
- Change: `Team::steer` reads the per-processor times at each `enter`. With a window of
  200 ms to 30 s and a gap of 25 points or more, the team moves to the quieter domain of
  the same size. It moves through the thread handles, so no barrier waits for a CPU.
- Result under the same load, with TDT's windows: 20 s take 138 to 103 ms, 8 s take
  92 to 70 ms, 46 s take 68 to 43 ms (4 rounds each, steering off to on).
- `keep_full_speed` takes team threads out of Windows power throttling, as the QoS class
  does on macOS. A background app can get lower clocks or, on a hybrid CPU, the slow cores.
  Not measured: this PC has neither, so only the call itself ran.
- Not covered: a load on all cores of both CCDs (this PC crashes under it), and Linux
  (`cpu_times` has no Linux source yet, so a team stays on its first domain there).

### Open: a recording can decode to nothing (seen 2026-11-05)

- Parakeet Unified, Trit 0.6.0 and 0.8.0, greedy: 29 s of clear speech that starts inside a
  sentence gave 0 tokens, and 2 frames less audio gave 98 words. The encoder is healthy (no
  NaN). At frame 4 the first word lost a close call (blank 0.04, word -0.28). The decoder
  then stays in its first state, where the blank leads by more than 10, for the whole clip.
- Rejected: run again from the closest call when a pass has no token. It rescued the clip,
  but the margins overlap: real misses 0.20 to 0.58, no-speech clips 0.18 to 0.6 and up. Room
  noise turned into "I don't want to do it."
- Seen only on audio that starts inside speech: 0 of 400 long takes and 0 of 143 owner takes
  came back empty. TDT's windows treat an empty window as a failed join. Not fixed in Trit.

## Session 2026-11-05 (review fixes, Trit 0.9.0)

A review of the engine for stability gave these changes. The gate compared the `trit` CLI of
`main` (2352bce) with the fixed build, clip by clip, through `safe.cmd` (8 processors).

Text gate, texts that differ between the two builds:

- Parakeet Redux: clean 0 of 200, other 0 of 200, tech 0 of 40, the owner's 143 takes 0, the
  long sets 0 of 200 and 0 of 200 (135 of these clips are over 60 s). With the owner's
  dictionary: takes 0 of 143, tech 0 of 40. With a dictionary of 22 names: long sets 0 of 400.
- Parakeet Unified English (GGUF, `--precision int8`): clean 0, other 0, tech 0, takes 0.
- Whisper tiny (tech 0 of 40, clean 0 of 200), Moonshine tiny, Parakeet TDT-CTC 110M Q8_0,
  Nemotron streaming Q8_0, Parakeet CTC 0.6B, wav2vec2 base: tech 0 of 40 each.
- GPU (`--device gpu`, RTX 4090, Vulkan), `main` against fixed: Redux tech 0 of 40 and clean 0
  of 200, Whisper tiny tech 0 of 40.

Speed, Redux on the 200 clean clips, total of the per-clip times, 4 rounds in turns, fixed
against `main`: 8,157 against 8,149 ms under the 8-processor mask (ratio 1.001), and 8,107
against 8,071 ms with the full mask (1.004). No speed claim for Redux. Whisper tiny under the
8-processor mask: 6,011 to 2,130 ms on tech and 34,789 to 13,367 ms on clean (one run each),
because the team no longer has more threads than the process has cores.

Changes:

- Team size: `engine_threads` counts only the cores inside the process mask and keeps the
  team mask inside it. Linux also stops at `available_parallelism` (a CPU quota is in no mask).
- Hybrid CPUs: with `TRIT_FAST_CORES=1` on Windows, with more than one `EfficiencyClass` and
  4 or more cores of the highest class, the team uses those cores, pinned. Off by default: it
  cannot be measured here (this PC reports class 0 on all 16 cores), and with few fast cores
  all the cores can be the faster team.
- Threads are `trit-speech-{slot}-{index}`. `on_every_thread` gives up after 1 s and runs on
  the caller (a forked child has no workers), so `release_memory` cannot wait without end.
- `trit_last_error` holds the message of a panic on the calling thread.
- `Level::Avx512` needs `avx512bw` and `avx512vl` too: `multiversion!` compiles with them.
- Damaged files: the GGUF header has bounds on the tensor count, the alignment, the sizes, and
  the array depth. `Tokenizer::parse` rejects an id far past its entries. The NeMo and PyTorch
  readers add sizes with a check, and an inflated zip member has a limit of 256 MB. The cache
  index rejects an offset that is not aligned.
- Cache: each cached load deletes `weights-*.tmp` in `.trit/` that are older than one hour.
- Position tables: the table of the load stays. A take over 30 s builds a table apart (about
  148 MB for a model 1,024 wide), and `release_memory` drops it. The long sets cover this path.
- `trit::Cancel` and `Options::cancel` stop a Parakeet call between encoder blocks, decoder
  steps, and windows. No C ABI change.
- Dictionary: a sound check starts at most `SPAN_REACH` frames before its first word, so a long
  pause before a candidate does not add joint rows for each frame. A token id past the
  dictionary tables counts as a piece with no letters.
- GPU: a readback waits at most 10 s (`Device::poll` with a time limit), and after 3 failures
  in sequence all later calls use the CPU.

Rejected, with the measurement:

- Steering without the team's own load (`GetProcessTimes` taken off the busy time of the
  current domain). The defect is real: on a quiet PC `main` moved the team 28 times in one
  7 s run over the 143 takes, about one move for each 250 ms, and the change gave 0 moves
  (1 move, at 313 ms, with five busy cores on the first domain). But in a batch the moves
  cost nothing here: with the change the 200 clean clips took 0.9 % more (8,144 against
  8,069 ms, 4 rounds, full mask), and under the 8-processor mask, where no build moves, the
  two were level (0.999). Not measured: isolated takes, where a move puts the read that the
  user waits for on cores that slept.
- The panic payload of a worker kept in the task loop of `run_tasks` (a lock in the panic
  branch): 0.8 % to 1.1 % more time on the 200 clean clips in five runs of 3 or 4 rounds.
  The layout of `Shared` was not the cause: three layouts, with each word on its own cache
  line too, gave the same loss, and the code of `main` in `run_tasks` removed it. The panic
  hook of the host still prints the message of the worker.
- A yield in the wait of `Team::for_each` for its last tasks. `SwitchToThread` can give the
  CPU to a thread of lower priority for a time slice. With the team size inside the process
  mask it has no case left.
- A window cut at a pause that the decoder shows, for recordings over 60 s. Against the cut at
  the quietest 100 ms: long-clean 726 to 729 errors, long-other 1,213 to 1,207 (55,414 words),
  and 25 % more time on both sets (64,183 to 80,466 ms and 60,911 to 75,631 ms), because each
  window is read at the full 60 s and read again when it has no pause.
- A join of two TDT windows at a pause that both show (with word times from Trit). When the
  first window drops speech near its end and the second drops the same speech near its start,
  both show a pause there, the join passes, and the speech is lost. The equal-word join fails
  in that case and the whole take runs, which loses nothing.

Not done:

- A cache that a load trusts with no conversion. Each layer converts its weights before it
  asks for the cached buffer, so this needs a change of every loader.
- One joint product for the frames between two tokens. It needs an A/B session.
- The new Linux and macOS lines did not compile on this PC (no C compiler for those targets).
- `trit ... | head` ends with a panic on the closed pipe (seen with `trit doctor`).

## Session 2026-11-09 (review fixes, then speed and CPU load of the decoder families)

Work: branch `oploop` (from `main` 4760865). A = `oploop/baseline/trit_ffi-base.dll` (main).
The clips of `clean` and `other` came back with `bench/librispeech.py` (now in `bench/clips`);
`tech`, `tts16`, and the owner's takes are not on this computer, so this session has no
gate on them.

### Floor F0 (commits 636e38f, fa2b66b, 26ebcba: the review fixes)

- Redux, `ab.py`, main against F0, 3 rounds, clean and other: 0 of 400 texts differ, errors
  81 and 172 on both sides, per-clip ratio 0.995 (paired p = 0.074).
- Whisper tiny, base.en, Moonshine tiny, 3 rounds x 200 clean: 0 texts differ each, errors
  231, 124, 133 on both sides.
- Clippy `-D warnings --release` (workspace, and `-p trit` with the GPU off), fmt, and
  `trit skill --check`: clean.
- Live: 25 edge-case WAV files with no crash; the same text for 3 clips on
  `TRIT_ISA=scalar|avx2|avxvnni`, 1 and 3 threads, and 4 models (before the fixes; `bench/paths.py`
  now checks it).
- Put back: the panic payload of a worker in `run_tasks`. I wrote it again in the review
  fixes; this ledger rejected it on 2026-11-05 (0.8 % to 1.1 % slower). `team.rs` has the
  task loop of main again.

### Noise (A against A, Redux, F0 twice, 3 rounds x 200 clean)

Medians 32.2 and 31.0 ms, per-clip ratio 0.991, paired signed-rank p = 3e-05. The paired
test calls a 0.9 % difference of one build with itself significant: the side that runs
second in a round is a little faster. A gain counts in this session only above 1.5 %,
and when it repeats. CPU time per clip: 250 and 250 ms (ratio 1.000).

### Harness: CPU time and memory (`ab.py`, commit after F0)

`ab.py` now charges each call its CPU time (`time.process_time`, all threads, with the
3 ms that the workers spin after a call), and reports the private memory and the peak
working set of each side. Redux: 250 ms of CPU for a 31 ms clip, so all 8 threads are busy
for the whole call. The private memory of a side (1.49 GB) is mostly numpy and scipy:
use it only to compare A with B.

### SI-012 beam search: a sequence that goes on in one beam keeps its state

- Written after the change (it was part of the review fixes), so this is a record, not a test
  of a forecast. The review estimated the copies at 20 MB per step for large-v3-turbo.
- Change: `beam.rs` cloned every kept beam's state (all self-attention caches) at each
  step. A parent with one child now gives its state; only a parent that splits is cloned.
- Result, main against F0, 3 rounds x 200 clean, 0 texts differ: Whisper tiny ratio 0.929
  (p = 1.6e-16; one noisy A round, 62.2 to 85.6 ms), base.en 0.971 (p = 2.9e-08), Moonshine
  tiny 0.956 (p = 2.5e-10). F0 holds the other review fixes too, but none of them is in the
  decode path of these models.
- Decision: KEEP (commit 26ebcba).

### Profile P-10 (the thread team by job: `oploop/baseline/trace.exe`, a temporary build)

`Team::for_each` counted each parallel job, its duration, and the serial time between two
jobs (a gap). CLI, 200 clean clips, one process, one run, not a reported number.

| | Redux | Whisper tiny | Moonshine tiny |
|---|---:|---:|---:|
| Calls, total | 9,139 ms | 17,604 ms | 13,227 ms |
| Inside parallel jobs | 8,804 ms (96 %) | 12,511 ms (71 %) | 7,374 ms (56 %) |
| Jobs under 5 us (count, time) | 44,940, 133 ms | 346,824, 902 ms | 528,575, 796 ms |
| Serial gaps of 100 us to 3 ms (count, time) | 528, 269 ms | 9,481, 4,353 ms | 7,400, 3,344 ms |
| Serial gaps of 3 ms or more | 9, 62 ms | 22, 158 ms | 98, 1,775 ms |

- Redux is parallel for 96 % of a call. Its jobs under 5 us are 1.5 % of the time.
- Whisper tiny: about 24 gaps of 0.65 ms and 23 gaps of 0.25 ms per clip, 22 ms of 88 ms.
  The 0.65 ms gap is the beam step on the caller: for each of 5 beams, `log_sum_exp` over
  51,864 logits, a scalar `max` fold, `mask`, and `top`. All other threads spin meanwhile.
- Whisper tiny runs 1,734 jobs under 5 us per clip (mean 2.6 us): the decoder's norms, GELU,
  per-beam attention, and dense products for 5 rows. The work of one is far under 1 us.
- Moonshine tiny: 45 % of a call is serial. 98 gaps of 18 ms (not found yet), and 32 gaps
  of 0.36 ms per clip (the beam step, vocabulary 32,768).

### SI-013 beam step: the beams' scores in parallel

- Journey: J1, J2, J3.
- Hypothesis: the beam step on the caller (`beam.rs` `search`: `log_sum_exp`, `mask`, `top`
  for each live beam, one after the other) costs about 16 ms of 88 ms per Whisper tiny clip
  (18 %), and about 11 ms of 66 ms per Moonshine tiny clip. One task per beam on the team
  should cut that stage about 4 times (5 beams), so tiny about 12 % faster and Moonshine
  tiny about 12 % faster. CPU time per clip falls too: 15 threads spin less.
- Test: each beam's candidates are computed alone and joined in beam order, so the sort
  sees the same list: 0 texts differ.
- Result (`ab.py`, F0 against B13, 3 rounds x 200 clean, 0 texts differ, errors equal):

  | Journey | A | B | per-clip ratio | paired p | CPU time of a round |
  |---|---:|---:|---:|---:|---:|
  | J1 Whisper tiny | 62.3 ms (58.0-62.5) | 54.6 ms (52.8-55.9) | 0.899 | 7e-33 | not summed yet |
  | J2 Whisper base.en | 105.8 ms (99.8-106.0) | 95.8 ms (95.0-98.1) | 0.940 | 9.4e-26 | 369.9 to 343.5 s (0.928) |
  | J3 Moonshine tiny | 34.8 ms (33.9-35.4) | 29.8 ms (29.3-30.2) | 0.847 | 9.1e-25 | not summed yet |

- Limiter after the change: the beam step is one job of 5 tasks, bound by the three passes
  of one beam over the vocabulary.
- Decision: KEEP. The forecast (12 % for both) held for tiny (10 %) and was low for
  Moonshine tiny (15 %).
- Harness: the CPU clock of Windows has steps of 15.6 ms, so `ab.py` now sums the CPU time
  of a round and gives the ratio of the round medians.

### SI-014 small steps on the calling thread

- Journey: J1, J2, J3 (J4 as a guard: its jobs under 5 us are 1.5 % of a call).
- Hypothesis: Whisper tiny runs 1,734 jobs under 5 us per clip (902 ms of 17,604 ms, 5 %),
  mean 2.6 us, and Moonshine tiny 2,643 (796 ms of 13,227 ms, 6 %). They are the layer norms
  of 5 rows, and the self-attention of one query per beam (`HeadMajor::new` and
  `attention_with`, 6 tasks each). The work of one is under 1 us; the rest is the hand-out
  to 16 threads. A scope that keeps the steps of a short input on the calling thread
  (`team::here`) should save about 3 ms per tiny clip (5 %) and about 2 ms per Moonshine
  clip, and some CPU time.
- Test: the same functions run per row and per head, in the same order: 0 texts differ.
- Result (`ab.py`, B13 against B14, 3 rounds x 200 clean, 0 texts differ, errors equal):
  Whisper tiny 50.4 and 50.0 ms, ratio 0.998 (p = 0.42). Moonshine tiny 29.7 to 31.8 ms,
  ratio 1.055 (p = 8.8e-07): slower. Redux ratio 0.991 (the A against A level).
- Decision: REVERT. The jobs under 5 us are not only hand-out cost: their work runs faster
  on the team than on one thread (the mean of 2.6 us holds work of several us in total).
  The forecast of the cost of one thread was too low. `SMALL` and `here_if` are gone.

### Profile P-11 (after SI-013; `oploop/baseline/trace2.exe` names the code around each gap of 100 us or more)

`oploop/results/s1109/trace_patch.py` applies the temporary patch. 200 clean clips, CLI, one run.

| | Whisper tiny | Moonshine tiny |
|---|---:|---:|
| Calls, total | 11,436 ms | 8,226 ms |
| Inside parallel jobs | 9,665 ms (85 %) | 6,469 ms (79 %) |
| Serial gaps under 100 us (count, time) | 496,173, 953 ms | 725,062, 839 ms |
| Serial gaps of 100 us or more | 3,993, 1,000 ms | 1,859, 1,016 ms |

Gaps of 100 us or more, by the jobs before and after:

- Whisper tiny: a dense product, then `HeadMajor::new` (`layers.rs:327`): 364 ms, 1,536 times.
  A norm, then the quantizer of an int8 layer (`kernels.rs:650`): 158 ms, 849 times.
  `HeadMajor::new`, then `attention_with`: 95 ms, 611 times. In each of them the caller
  takes a `Buf::scratch` of 2 to 9 MB that the pool does not hold (16 MB limit), and
  `Vec::resize` fills it with zeros on one thread: 617 ms, 5.4 % of the time.
- Moonshine tiny: a product, then `HeadMajor::new`: 610 ms, 1,218 times, 0.5 ms each. The code
  between them is `Rotary::apply` on the queries and the keys (`layers.rs:254`): it calls
  `sin_cos` for every row, every head, and every pair, and the result does not depend on
  the head. The decoder steps do the same for one row per beam and layer (in the small gaps).

### SI-015 rotary positions: the turn of a pair once per row, not once per head

- Journey: J3 (Moonshine). Also the language-model families and S1-mini (`causal_lm.rs` calls
  `Rotary::apply` twice per layer for each token; J5 cannot run here: no S1-mini folder).
- Hypothesis: `Rotary::apply` (`src/layers.rs:254`) costs at least 610 ms of 8,226 ms (7 %) of
  Moonshine tiny in the large gaps, and more in the decoder steps (5 beams x 6 layers x 2
  calls x 8 heads x 18 pairs of `sin_cos` per step). The turn of a pair depends on the row
  and the pair only. One `sin_cos` per row and pair, kept in a buffer of the thread and
  used for every head, removes 7 of 8 calls: about 10 % of a Moonshine tiny clip.
- Test: the same `sin_cos` values rotate the same numbers: 0 texts differ.
- Result (`ab.py`, B13 against B15, 3 rounds x 200 clean): Moonshine tiny 29.3 ms (28.5-29.9)
  to 26.9 ms (26.8-27.5), per-clip ratio 0.936, paired p = 1.1e-08; time of a round 7.71 to
  7.12 s (0.923); CPU time of a round 65.7 to 60.5 s (0.921). 0 of 200 texts differ, errors
  133 on both sides.
- Decision: KEEP. Not measured: the language-model families and S1-mini, which call
  `Rotary::apply` for each token and layer (no model folder here).

### SI-016 the pool keeps the buffers of a call that runs

- Journey: J1, J2 (J4 as a guard, and the peak working set of each side).
- Hypothesis: in a Whisper tiny clip the caller fills 2 to 9 MB buffers with zeros about
  15 times (`Buf::scratch`, `src/cpu/buf.rs:50`: the pool holds 16 MB at most, so a pass of
  1,500 frames finds no buffer and `Vec::resize` writes a new one on one thread): 617 ms
  of 11,436 ms (5.4 %). The 16 MB limit is there for the memory of an idle program. If the
  pool keeps every buffer while a call runs, and goes back under 16 MB when the last call
  ends, a pass finds its buffers again: about 4 % of a Whisper clip, with the same memory
  after a call.
- Test: a buffer from the pool holds old values where a new one holds zeros; every user of
  `scratch` writes all values before it reads them (the rule of `Buf::scratch` today, and
  the pool already gives old buffers). 0 texts differ on J1, J2, J4.
- Result (`ab.py`, B15 against B16, 3 rounds x 200 clean, two runs, 0 texts differ, errors
  equal). Another session ran a benchmark on 8 cores in the first run, and the rounds of the
  second run still spread by 40 % for Whisper (tiny 52.7 to 78.3 ms for one build):
  - Whisper tiny: per-clip ratio 0.968 (p = 0.29), then 0.995 (p = 0.79); time of a round
    1.004, then 1.000.
  - Whisper base.en: 0.746 in the first run (the A rounds were the disturbed ones), then 0.950
    with a round time ratio of 1.005. The two numbers of the second run do not agree: noise.
  - Redux: 1.021, then 0.984 (round time 0.982). The A against A level is 0.991.
  - Peak working set: equal (tiny 273 and 256 MB, base.en 343 and 343, Redux 655 and 636).
- Decision: REVERT. No gain above the noise in two runs, and the forecast (4 %) did not show
  in the round times of Whisper tiny. So the zero fill is not proven as the cause of the
  gaps of P-11. The patch is `oploop/results/s1109/SI-016-pool.patch` for a run on a quiet
  computer, after a trace that times `Buf::scratch` itself.
- Harness: `oploop/results/s1109/quiet.ps1` prints the cores that other processes use.
  Whisper runs on all 16 cores and is the first to show another load; Redux (8 cores of
  one L3 domain, with steering) is not.

### SI-017 one sinusoid table for all layers of a position table

- Journey: the load of a FastConformer model, and the first take over 30 s after a load or
  after `release_memory` (J4 as a guard).
- Hypothesis: `RelCache::rows` (`src/models/parts/fastconformer.rs`) builds the sinusoid
  table of its reach for each layer (`relative_positions`: 751 x 1,024 values at the load,
  1,501 x 1,024 for a take over 30 s, with one `sin` and one `cos` per pair), and every
  one of the 24 layers gets the same table. One table for all layers should save about
  23 of 24 builds: some tens of ms of the 140 ms load of Redux, and about 100 ms of the
  first take over 30 s.
- Test: the same table goes into the same product of each layer: 0 texts differ.
- Harness: `oploop/results/s1109/load.py` (a fresh process per run, A and B in turns: the load
  of Redux, a 5 s take, the first 31.6 s take, the same take again). A against A, 6 runs:
  load 139.1 and 139.1 ms, first long take 233.3 and 234.5 ms (ratios 1.000 and 1.005).
- Result (B15 against B17, 14 runs per side): load 141.7 to 137.8 ms (0.972); the first take
  over 30 s 239.6 to 225.0 ms (0.939; a run of 8 gave 0.930); the same take again 187.7 and
  189.2 ms (1.008). The text is the same in all runs.
- Gate (`ab.py`, Redux, 3 rounds, clean and other): 0 of 400 texts differ, errors 81 and 172,
  per-clip ratio 0.996.
- Decision: KEEP. The forecast was far too high (4 ms of the load, not tens; 15 ms of the
  first long take, not 100): `sin` and `cos` run in parallel and are fast. The first long take
  still costs 37 ms more than the next one: 24 products of 1,501 rows.

### SI-018 ternary product: tiles of 10 frames (AVX-512 VNNI)

- Journey: J4 (Redux; ternary products are 61 % of a take, profile P-9).
- Hypothesis: `Ternary::vnni_tile` (`src/cpu/kernels.rs`) unpacks the 2-bit codes of each 16
  inputs with 7 vector operations per block (3 shifts, 4 masks), and a tile of 6 frames then
  runs 24 `vpdpbusd` with them: 0.58 unpack operation for each product. A tile of 10 frames
  runs 40 products for the same unpack (0.35), and its 20 integer sums, 8 weight vectors, and
  2 constants still fit the 32 registers. If the products are bound by the vector ports
  (2.1 of 2.6 T multiply-adds per second today), the ternary products run about 8 % faster
  and a take about 4 %.
- Test: the group sums are integers and the float steps of each output and frame keep their
  order: every value is the same, so 0 texts differ.
- Result (`ab.py`, B17 against B18, 5 rounds x 200 clean, two runs; another session built
  with `rustc` in both, 1 to 5 cores, and the medians were 34 to 38 ms, not 31): per-clip
  ratio 1.017 (p = 0.0054), then 0.999 (p = 0.84); time of a round 1.039, then 1.028. 0 of
  200 texts differ and 81 errors in both runs, so the values are the same, as forecast.
- Decision: REVERT. No gain in either run, and the round times lean to a loss. The unpack
  is not what limits the ternary product; the limiter is not found. Together with P-9
  (2.1 of 2.6 T multiply-adds per second) this says: the ternary kernel has no cheap gain
  left on this CPU, and a change of the order of its float sums changes the text (SI-009).
  A new attempt needs a quiet computer and a counter of the vector operations, not a guess.

### Gates of the branch head (after SI-017, with SI-014, SI-016, SI-018 reverted)

- Clippy `-D warnings --release` (workspace, and `-p trit` with the GPU off), fmt, and
  `trit skill --check`: clean. The commit of SI-017 went in before clippy ran and broke it
  (`type_complexity`); the next commit has the fix. Run clippy before each commit.
- Text, main against the head (`ab.py`): Redux 0 of 400 texts differ (clean 81, other 172
  errors on both sides); Whisper tiny 0 of 200 (231), base.en 0 of 200 (124), Moonshine tiny
  0 of 200 (133).
- `bench/paths.py` (3 clips; `TRIT_ISA=scalar|avx2|avxvnni`, 1 thread): the same text for
  Moonshine tiny, Redux, and Whisper tiny. The script now makes the path of the executable
  absolute: Windows did not start `target/release/trit.exe` as written, so the new CI
  step failed before this fix.
- 25 edge-case WAV files: no crash; the six sample formats of one clip give its text.
  6 threads x 8 calls with `release_memory`: 0 wrong texts (Redux, Whisper tiny).
- Not run: `tech`, `tts16`, the owner's takes (no clips here), the voices, S1-mini, the
  language-model families, and a build of TDT against this branch (another session worked
  in that checkout).

### Journeys, main against the head

Another session used 6 cores during this last run, so the medians are high (Redux 49 ms, not
31). Both sides ran in turns. The ratios agree with the products of the steps measured
on a quieter computer (tiny 0.929 x 0.899, base.en 0.971 x 0.940, Moonshine 0.956 x 0.847 x 0.936).

| Journey | Per-clip ratio | Time of a round | CPU time of a round | Rounds x clips |
|---|---:|---:|---:|---:|
| J1 Whisper tiny | 0.875 | 0.848 | 0.853 | 2 x 200 |
| J2 Whisper base.en | 0.918 | 0.893 | 0.885 | 2 x 200 |
| J3 Moonshine tiny | 0.778 | 0.772 | 0.797 | 2 x 200 |
| J4 Redux | 0.988 (p = 0.096) | 1.059 (disturbed) | 0.984 | 3 x 400 |
| Redux load / first take over 30 s | 0.972 / 0.939 | | | 14 runs |

Kept: SI-012, SI-013, SI-015, SI-017. Reverted: SI-014, SI-016, SI-018 (and the panic
payload of the review fixes).

### Next

- Whisper, the gaps of P-11 again with a timer on `Buf::scratch` itself, on a quiet
  computer; then SI-016 again if the fill is the cause.
- Moonshine: the serial code between the conv layers of the encoder (0.77 ms per clip), and
  one table of rotary turns for the queries and the keys of all layers of a call.
- The decoder steps: self-attention builds the head-major copy of all keys and values
  again for each token (`layers.rs` `attention`, called from `causal_lm.rs` and the Whisper
  and Moonshine steps). Keep the caches in that layout. It needs S1-mini for J5.
- Memory: the first take over 30 s adds 148 MB of position tables (the reach doubles to the
  60 s of a full window). A reach in steps of 10 s gives 98 MB for a 31 s take.

## Session 2026-11-10 (the open items of the review; no speed change)

No hypothesis ran. This session changed the file readers, the C ABI, the bindings, and CI,
not a hot path. The only new work of a call is in `trit-ffi`: `Running::start` and `finish`
take one lock and keep one list entry for each call.

### Changes

- Readers: the pickle reader of `cpu/torch_file.rs` refuses values nested more than 32 deep
  (`shallow`); a Moonshine `partial_rotary_factor` must be above 0 and at most 1.
- C ABI: `trit_load_json` (`precision`, `device`, `cache`), `trit_cancel`, `trit_set_log`.
  A panic of a worker thread now gives its own message in `trit_last_error`: a panic hook
  in `trit-ffi` keeps the message of a `trit-speech-*` thread. `run_tasks` did not change
  (the payload there cost 0.8 % to 1.1 % on 2026-11-05).
- Bindings: `cache`, `cancel`, and the log function in Python and C#; `cache`, `cancel`,
  `transcribeAsync`, `transcribeWavAsync`, and a finalizer in Node.
- Weight cache: `Cleaner::load_with` takes `cache`; on Windows a load from a network,
  removable, or optical disk uses no cache (`local_disk`).
- Voices: `noise_duration` is from 0 to 2. MMS-TTS, one sentence: 3.9 s at 0.8, 6.0 s at
  1.5, 14 s at 2, 47 s at 2.5, and an error above.
- New check, `bench/hostile.py`: hostile pickles in a NeMo archive, a deep YAML, damaged WAV
  files, and random damage of `config.json`, `tokenizer.json`, and the safetensors header.
- CI: the damaged-files step (Windows), the text check and the Python binding on Linux
  (espeak-ng and sox make the clips), and `audit.yml` (`cargo audit`, each week).

### What `bench/hostile.py` found

- `main` stops with a stack overflow on `pickle-deep` (tuples 200,000 deep) and on
  `yaml-deep` (5,000 deep). The branch gives an error for both.
- Random damage, 150 copies of Moonshine tiny: 1 crash. `"partial_rotary_factor":
  18446744073709551615.9` asked for a position table of that size (`capacity overflow`).
  After the fix: 450 copies, 0 fail (seed 7).

### A finding of the review that was wrong

The review said that a Parakeet segment ends early (80 ms after the start of its last
token). Measured on three clips: the old end is 0.34 to 0.40 s after the real end of the
speech, because the last token starts late. A change that added the token's duration made
the ends later. Reverted; `docs/engine-review.html` now says so. Measure a timestamp claim
against the audio before the change.

### Gates of the result (build R3)

- Windows: fmt, clippy `-D warnings --release` (workspace, and `-p trit` with the GPU
  off), `cargo check -p trit-ffi --no-default-features`, `trit skill --check`, and the
  header check of CI (34 exports): clean.
- Text, main against R3 (`ab.py`, 1 round; another session ran `ab.py` on 15 cores, so no
  time is given): Redux 0 of 400 texts differ (clean 81, other 172); Whisper tiny 0 of 200
  (231); base.en 0 of 200 (124); Moonshine tiny 0 of 200 (133); S1-mini 0 of 62 (25 and 25).
- `bench/paths.py`, 3 clips: the same text on each `TRIT_ISA` value and on 1 thread, for
  Moonshine tiny, Redux, and Whisper tiny.
- Voices: Kokoro and MMS-TTS write the same bytes as main for one text.
- `bench/hostile.py --mutate <Moonshine tiny> --runs 150`: 164 of 164 pass. 25 edge-case
  WAV files: no crash. 6 threads x 8 calls with `release_memory`: 0 wrong texts.
- Cancel through the C ABI: Redux stops at 0.31 s of 4.15 s, Whisper tiny at 0.36 s of
  5.19 s; the next call gives the correct text. Node and Bun: 165 ms of 1,670 ms. C#:
  165 ms of 1,651 ms, 0 build warnings, 89 log lines through `SetLog` (GPU profile on).
- Cache through `trit_load_json` (Moonshine tiny): first load 482 ms, second 170 ms, the
  same text.
- Linux (WSL, Ubuntu 24.04, the same CPU): the workspace builds; both clippy runs and the
  `trit-ffi` check are clean; `paths.py` gives the same text on every path, and the text of
  Redux and Moonshine tiny equals the Windows text on 3 clips; `hostile.py` 104 of 104;
  cancel through the Python binding works. This is the first text check on Linux in this
  ledger.
- `cargo audit` on `Cargo.lock`: 105 crates, 0 advisories.
- Not run: `tech`, `tts16`, and the owner's takes (no clips here), the language-model
  families other than S1-mini, `bench/tts.py`, macOS, arm64, and a build of TDT against
  this branch.

### Not done, and why

- A speech check before Whisper and Moonshine: it needs a model or a threshold from data.
- The head-major cache of the decoder steps: a large change to families with no model here.
- A shutdown function for the team, and the hybrid-CPU default: the team loop is sensitive
  to 1 % (2026-11-05), and this computer cannot measure a hybrid CPU.
- Whisper segments from its timestamp tokens; dictionary weights and load progress in the
  C ABI; a log function in the Node binding (a callback from another thread).
- GPU device loss: CPU first. Fuzz targets, Miri, unit tests: `bench/hostile.py` is the
  check in their place. Actions pinned by SHA, more release targets: the owner's decision.

## Session 2026-11-10 (the kernels written again, branch `kernels-rewrite`)

A separate job, not an OpLoop iteration: write the product kernels again from nothing, then
compare them with the first set. Work: a worktree of branch `kernels-rewrite` (from 8494e19).
The second set is `src/cpu/v2/`, next to the first. `TRIT_KERNELS=v2` (or `v2-int`,
`v2-float`) selects it for a process; the default is the first set. It runs every CPU of the
first set (K-006). In Trit 0.11.0.

### Harness: `examples/kernel_bench.rs` (`src/cpu/v2/bench.rs`)

- `kernel_bench peak one|all`: the clock of the team (a chain of dependent shifts, 6 cycles
  for each step), then `vpdpbusd` alone and `vfmadd231ps` alone, 12 sums in registers.
- `kernel_bench ternary2|int82|q8_02|f322|f162 one|all <n> <k> <m>...`: the first set against
  the second on the same input, in turns, with the count of the values that have other bits.
- Quiet computer: 5.25 to 5.30 GHz with one thread, and one 512-bit `vpdpbusd` and one 512-bit
  `vfmadd231ps` for each cycle and core. 8 threads: 2.43 T int8 multiply-adds per second.
  16 threads: 4.86 T int8 and 1.29 T f32.

### Finding: the limit of the ternary product (the limiter that SI-018 did not find)

- Zen 4 (uops.info, measured): `vpdpbusd zmm` uses the pipes FP0 and FP1, one instruction for
  each cycle. `vmulps` and `vfmadd` use the same two pipes. Shifts use FP2 and FP3, `vpand`
  any pipe, `vcvtdq2ps` FP2 and FP3.
- One group (128 inputs) of 16 rows and one frame costs 32 `vpdpbusd`, 1 `vmulps`, and
  1 `vfmadd` on FP0 and FP1. So the product can reach 32 / 34 = 94 % of the `vpdpbusd` peak.
  The unpack of the codes and the tile shape do not change that.
- Measured, first kernel: 0.300 to 0.308 T/s of 0.323 with one thread (93 to 95 %), and 2.22
  to 2.36 T/s of 2.43 with 8 threads. The "2.1 of 2.6 T" of P-9 used a clock of 5.07 GHz and
  no float steps.
- So the ternary product has no gain left on this CPU without a change of its arithmetic,
  and that changes the text (SI-009).

### K-001 integer product: codes unpacked once into a byte panel, 14 frames side by side

- Hypothesis: the unpack of the 2-bit codes (7 operations for each 16 inputs and block) and the
  correction step take time from the products. A task that unpacks its rows once into a
  thread's byte panel, with the frames of a tile stored side by side (one stream), can run
  tiles of 14 frames that hold only `vpdpbusd`.
- Result (`kernel_bench ternary2`, one thread, 0 values with other bits): 5 to 7 % slower at
  90 to 375 frames, 16 to 26 % slower at 14 to 28 frames, 3 times slower at 1 frame. The inner
  loop was as planned (28 `vpdpbusd`, 14 broadcasts, 2 loads), so the loss is the panel: 4 times
  the bytes of the codes, written and read again for each call.
- Decision: REVERT. The unpack in registers uses pipes that the products do not use.

### K-002 integer product: one tile for ternary and int8 (`v2/product.rs`)

- Change: the tile shape of the first kernels (two 16-row blocks by 6 frames, codes unpacked
  in registers), written once for ternary and int8, for AVX-512 VNNI, 256-bit VNNI, AVX2, and
  scalar. A dot product starts at `-bias * sum(x)`, so there is no correction step. The scalar
  path uses the float steps of the SIMD paths (`mul_add`). The last unit of 16 rows can be
  partial. A layer with few rows splits its frames too.
- Result (`kernel_bench`, 0 values with other bits on the SIMD paths): ternary 0.97 to 1.01
  with one thread and 0.99 to 1.00 with 8 threads; int8 0.97 to 1.02 with one thread; Q8_0
  0.94 to 1.00. With 16 threads and 1,500 frames: 384 x 1536 0.83, 384 x 384 0.90,
  512 x 2048 0.89 (the frame split), and 1536 x 384 and 2048 x 512 0.98 to 1.00.
- A full transcription does not show the frame split (Whisper tiny and base.en), as I-002
  found.
- 256-bit VNNI for ternary layers (`TRIT_ISA=avxvnni`, EVEX form on this CPU): 0.189 to
  0.204 T/s, against 0.138 to 0.141 for the AVX2 path that the first kernel uses on a CPU
  with AVX-VNNI and no AVX-512. Not measured on such a CPU.
- The scalar path is 1.4 to 1.7 times slower than the first scalar path (`mul_add` without
  FMA is a library call). It gives the bits of the SIMD paths.

### K-003 float product: f16 widens in the tile, always (`v2/float.rs`)

- Finding: `fgemm::matmul` writes an f32 copy of a 64-row weight panel for each block of 5
  rows or more (`W::WIDEN && rows.len() > 4`). On Zen 4 `vcvtph2ps` does not use FP0 and FP1:
  f16 with 4 rows (no copy) runs as fast as f32. With 5 rows (a decoder step with 5 beams) the
  f16 product ran at 44 % of the `vfmadd` peak.
- Result (`kernel_bench f162`, one thread, 0 values with other bits): 5 rows 0.61 to 0.65
  (288 x 288, 1152 x 288, 288 x 1152, 32768 x 288), 8 rows 0.71, 16 rows 0.86, 64 to 300 rows
  0.92 to 0.98. With 16 threads and 5 rows: 51864 x 384 0.79, 51864 x 512 0.85, 32768 x 288
  0.75. f32 products: 0.97 to 1.00.
- This change is one line in the first kernel.

### K-005 float product: the tail of `k` as one masked load

- Finding: when `k` is not a multiple of 16, the AVX-512 tile of `fgemm.rs` writes the last
  values of each weight row into an array, one value at a time, and loads the array as one
  vector. The load waits for the small stores, and an f16 value converts in scalar code. The
  comment there says that `k` is a multiple of 16 in every model. It is not: the first layer
  of Moonshine has `k` = 127 (`moonshine.rs`, `conv1`, one row for each 64 samples), and the
  mel filters of Whisper have `k` = 201 (`whisper_mel.rs`).
- Change (`v2/float.rs`): `vmovups {k}{z}` for f32 and `vmovdqu16 {k}{z}` with `vcvtph2ps` for
  f16. The values are the same.
- Result (`kernel_bench`, one thread): 300 x 200 f16 0.50, 300 x 203 f32 0.57.

### K-004 float product: wider tiles for 1, 2, 3, and 5 rows

- Hypothesis: a strip of 1 to 3 rows has 3 to 9 sums in flight, so it waits on the 4 cycles of
  a fused multiply-add. Tiles 1 x 6, 2 x 6, 3 x 4 keep 12 in flight.
- Result (one thread): f32 1 x 6 the same as 1 x 3; 2 x 6 1.13 to 1.30; 3 x 4 1.23 to 1.31.
  f16: 0.89 to 1.00. One row runs at 45 % of the peak with any tile: it waits on the weights
  (one 64-byte line for each cycle from L2), not on the multiply-add.
- Decision: REVERT to 3 columns. Kept: 5 rows as one strip (5 x 3); it is level with 4 x 3
  and then 1 x 3 for f32.

### Finding: the first scalar path gives other text for Whisper tiny

`bench/paths.py` with the first three `clean` clips: Whisper tiny with `TRIT_ISA=scalar` writes
"arms" where every other path writes "alms" (clip 1089-134686-0009; the build of the `oploop`
head too). The first scalar int8 path computes `dot * weight scale * frame scale` without the
fused step of the SIMD paths. With the second set the scalar path gives the same text.

### Transcription, first set against second set

`ab.py <model> K4.dll K4.dll --rounds 3 --env-b TRIT_KERNELS=v2` (one library, the kernel set
from the environment of side B; `--env-a` and `--env-b` are new in `ab.py`). The computer was
quiet before and after each run (0.02 to 0.09 cores in use).

| Model | Clips | First | Second | Per-clip ratio | Time of a round | Texts that differ | Errors |
|---|---:|---:|---:|---:|---:|---:|---|
| Parakeet Redux | 400 | 29.9 ms | 29.1 ms | 0.992 | 0.989 | 0 | 81 and 172, both sides |
| Parakeet Ultra | 200 | 34.5 ms | 33.7 ms | 0.989 | 0.989 | 0 | 56 |
| Whisper tiny | 200 | 50.1 ms | 49.1 ms | 0.962, then 0.972 | 0.961, then 0.976 | 0 | 231 |
| Whisper base.en | 200 | 88.6 ms | 87.8 ms | 0.980, then 0.972 | 0.987, then 0.962 | 0 | 124 |
| Moonshine tiny | 200 | 26.3 ms | 20.9 ms | 0.824, then 0.823 | 0.849, then 0.837 | 0 | 133 |
| Moonshine base | 200 | 50.0 ms | 41.5 ms | 0.889 | 0.912 | 0 | 105 |

- `TRIT_KERNELS=v2-float` alone: Moonshine tiny 0.822, Whisper tiny 0.967. So the gains come
  from the float product (K-003 and K-005). The integer product gives the same time
  (`v2-int`, Whisper base.en: 0.995).
- With the build before K-005: Moonshine tiny 0.933 and 0.952, Moonshine base 0.960, Whisper
  tiny 1.001, Whisper base.en 0.992. So K-003 gives 4 to 7 % for Moonshine, and K-005 gives
  the rest for Moonshine and all of the gain for Whisper.
- Redux and Ultra are under the 1.5 % level of this ledger: the same.
- `bench/paths.py` with `TRIT_KERNELS=v2` (3 clips; `TRIT_ISA=scalar|avx2|avxvnni`, 1 thread):
  the same text for Redux, Whisper tiny, and Moonshine tiny.
- Clippy `-D warnings --release` (workspace): clean but for `type_complexity` in
  `fastconformer.rs:1450`, which is in the base commit (fixed on `oploop` in bd8b18e). fmt and
  `trit skill --check`: clean.
- Not run: `tech`, `tts16`, the owner's takes (no clips here), a CPU with AVX-VNNI and no
  AVX-512, and an Intel CPU (`vcvtph2ps` and the shifts use other ports there). arm64: see
  K-006.

### Finding: the first set gives other text on AVX2 than on AVX-512

`bench/paths.py`, Parakeet Ultra, the first three `clean` clips: `TRIT_ISA=avx2` writes "land."
where AVX-512 writes "land?" (clip 1089-134686-0022). The first float product adds in 16 lanes
on AVX-512, 8 on AVX2, and 4 on NEON, with the k tail value by value; the f16 matrix-vector
product (`half.rs`) does the same; `matmul_nn_into` adds the edge columns of AVX2 and NEON
without fusing; the scalar activations do not fuse. So a CPU without AVX-512 (Intel desktop
CPUs, Zen 2 and 3, every arm64 CPU) can give other text. `ab.py` on the AVX2 path, first set:
Whisper tiny 228 errors and Ultra 55, where AVX-512 gives 231 and 56.

### K-006 every CPU: arm64 paths, and one set of bits on every path

- Hypothesis: the second set can run every CPU that the first runs, and give the same bits on
  all of them, so that the text does not change with the CPU.
- Change:
  - Integer product, arm64 (`v2/product.rs`, `arm`): one unit (16 rows) by 4 frames with
    `sdot`, `smull` (Armv8.0), or `smmla` (i8mm, int8 layers; their bytes are `[k / 8][16][8]`).
    16 sums in flight; the first arm64 tiles keep 8. NEON quantizer (`v2/quantize.rs`).
  - Float products: one sum order on every path. 16 lanes, the k tail padded with zeros (what
    the AVX-512 masked load gives), then the tree of `_mm512_reduce_add_ps` (lane i + i+8,
    i+4, i+2, then the last two; checked on 200000 random vectors). AVX2 holds the lanes in 2
    registers (tiles 2 x 3), NEON in 4 (2 x 3, 28 registers), scalar in an array with `mul_add`.
    The k tails are padded once per block of columns and once per strip of rows.
  - `half.rs` matrix-vector product: through `v2::float::matmul` with one row.
  - `v2/nn.rs`: `matmul_nn_into` with fused steps in every column and on every path.
  - `v2/activation.rs`: the scalar activations with the fused steps of the SIMD paths.
  - `v2::enabled()` on every target (it was x86-64 only). The arm64 build did not compile
    before (x86 macros called on every target); fixed.
- Gate:
  - `kernel_bench` adds a hash of the second set's output. 40 shapes (ternary, int8, Q8_0,
    f32, f16; n of 48, 67, 80, 100, 384; k of 96 to 1536, with k = 127 and 201; m of 1 to 400):
    the same hash on all 9 configurations: x86 default (AVX-512 VNNI), `avxvnni` (256-bit EVEX),
    `avx2`, `scalar`; QEMU 8.2 user mode `-cpu max` (i8mm), `max` with `dotprod`, `neon`,
    `scalar`, and `cortex-a53`. First against second, arm64 SIMD paths: 0 values with other
    bits on 240 of 240 products (the first scalar path differs, as on x86).
  - Text: `bench/paths.py` with `TRIT_KERNELS=v2`, x86: Redux, Ultra, Whisper tiny, Moonshine
    tiny: the same text on every path (Ultra on AVX2 and Whisper tiny on scalar too). arm64
    under QEMU (`max` default, `dotprod`, `neon`, `scalar`, `cortex-a72`): Redux, Ultra, Ultra
    `--precision q8_0`, Whisper tiny, Moonshine tiny, Moonshine tiny `--precision f32`: one text
    for each, and that text is word for word the x86 text (AVX-512 and AVX2).
  - Clippy `-D warnings --release`: x86-64 and `aarch64-unknown-linux-gnu`, clean but for the
    finding of the base commit.
- Measure (`ab.py`, `K5.dll` both sides, 3 rounds, 200 `clean` clips; other sessions used 1 to 3
  cores during the runs, so the AVX2 numbers have more noise):

  | Model | Path | Per-clip ratio | Time of a round | Errors first / second | Texts that differ |
  |---|---|---:|---:|---|---:|
  | Moonshine tiny | AVX-512 | 0.854 | 0.857 | 133 / 133 | 0 |
  | Whisper tiny | AVX-512 | 0.961 | 0.952 | 231 / 231 | 0 |
  | Parakeet Redux | AVX-512 | 0.967 | 1.005 | 81 / 81 | 0 |
  | Moonshine tiny | AVX2 | 0.908 | 0.942 | 133 / 133 | 0 |
  | Whisper tiny | AVX2 (5 rounds) | 1.018 | 1.020 | 228 / 231 | 12 |
  | Parakeet Ultra | AVX2 | 1.051 | 1.042 | 55 / 56 | 8 |

  On AVX2 the second set gives the AVX-512 text, so its errors are the AVX-512 errors.
  `kernel_bench`, AVX2, 1 thread: f32 with many rows 1.04 to 1.21 (2 x 3 tiles where the first
  fits 4 x 3 with 8 lanes), f16 0.73 to 1.05, k = 127 0.70 to 0.80.
- Not run: arm64 speed (QEMU gives no times), a real arm64 CPU, AVX-VNNI in VEX form (Zen 4
  does not have it, and QEMU 8.2 does not emulate it: the path differs from the EVEX path in
  the encoding of one instruction), RISC-V and other targets (scalar path; not built).
- Decision: KEEP in the second set.

### Next

- Port K-005 and K-003 to `fgemm.rs` as two OpLoop iterations (a masked tail load; no f32
  panel for f16). Both keep every value.
- The scalar int8 path of the first set: use the float steps of the SIMD paths, so that
  `TRIT_ISA=scalar` gives the text of the other paths.
