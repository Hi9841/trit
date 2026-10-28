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
