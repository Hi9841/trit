# Journeys

What a user waits for: one dictation or one file, transcribed with Trit's default
(auto precision; Whisper and Moonshine with 5 beams). Measured with `ab.py`
(from the repository root): fresh process per side, one warm-up clip, then every clip of
the set, one call each.

| ID | Journey | Command (A, B = trit_ffi.dll paths) | Metric | WER floor (total errors, 456 clips) |
|---|---|---|---|---|
| J1 | Whisper tiny, one clip | `ab.py whisper-tiny A B --rounds 3` | median ms, test-clean (200) | 760 |
| J2 | Whisper base.en, one clip | `ab.py whisper-base.en A B --rounds 3` | median ms, test-clean | 468 |
| J3 | Moonshine tiny / base, one clip | `ab.py moonshine-tiny A B`, `ab.py moonshine-base A B` | median ms, test-clean | 557 / 415 |
| J4 | Parakeet Redux (TDT's model) | `ab.py parakeet-redux A B`; ratchet `bench/accuracy.py` | median ms; ratchet 81/172/34/6 | 293 |
| J5 | S1-mini cleanup | `ab.py x A B --cleanup` | median ms per case | 42 errors, 37 exact |
| J6 | Redux: the load, and the first take over 30 s | `python -I oploop/results/<session>/load.py A B 14` | median ms, fresh process per run | the same text |

WER check for a candidate: `ab.py <model> A B --rounds 1 --sets clean,other,tech,tts16`
(totals) and `bench/wer_rule.py` after a full rerun.

`ab.py` also gives the CPU time of a round (all threads) and the memory of each side. The clips are
in `bench/clips` (`uv run bench/librispeech.py` gets `clean` and `other`). Before a run, look at the
other load on the computer: Whisper uses all cores and shows it first.

## Stability and dynamicality (from 2026-11-11)

| ID | Journey | Command | Metric |
|---|---|---|---|
| J9 | One clip, 40 calls with no pause | `steady.py <model> A [B] --tests repeat` | p50, p95, max; p95 / p50 (Redux: 1.2) |
| J10 | One clip after an idle time of 0.2, 1, 3 s, and after `release_memory` | `steady.py <model> A [B] --tests gap` | p50 by idle time |
| J11 | The time for each second of audio, 1 to 90 s | `steady.py <model> A [B] --tests length` | ms for each second (Redux: 6 to 7 up to 40 s, 8.7 at 59 s) |
| J12 | The same words said another way (level, pause, noise, 5 ms move, pace) | `steady.py <model> A [B] --tests delivery` | texts that differ from the plain clip (Redux: 0 of 20), errors, ms |
| J9L | One clip while other threads are busy | `steady.py <model> A [B] --tests load --loads 0,8,16` | p50, p95, max by load. At most 16 busy threads on this computer |
| J13 | The cost to attach | `attach.py <model> <trit.exe> <trit_ffi.dll>` | ms of one cold `trit transcribe`, ms to the first text from Python, results for audio that is not 16 kHz |
| J14 | A long recording | `ab.py <model> A B --sets long-clean,long-other` (`bench/long_clips.py` makes the sets) | median ms, errors (Redux: 187 and 479), peak working set |

The CPU load is the "CPU time of a round" of `ab.py`. `oploop/results/<session>/idle_cpu.py` gives
the CPU time in the 20 ms after a call.
