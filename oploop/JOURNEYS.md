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
