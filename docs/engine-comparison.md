# Same model, different runtime: Trit and the main speech runtimes

This report runs each model on every CPU runtime that supports it, in
English, so the runtime is the only difference. Trit, TDT's own engine, runs
every model. The other runtimes: Photon (Moondream's Kestrel engine) and
Transformers for Parakeet Redux; ONNX Runtime, sherpa-onnx, and Transformers
for Parakeet TDT v3; faster-whisper, whisper.cpp, Transformers, and OpenAI
Whisper for Whisper; Transformers and sherpa-onnx for Moonshine;
Transformers and llama.cpp for S1-mini.

The same page with charts: [engine-comparison.html](engine-comparison.html).
Measured on 2026-10-21 on an AMD Ryzen 9 7950X3D (16 cores), 32 GB RAM, and
Windows 11, CPU only. Raw results:
[desktop/bench/runtimes-results.jsonl](../desktop/bench/runtimes-results.jsonl)
and [desktop/bench/cleanup-results.jsonl](../desktop/bench/cleanup-results.jsonl).

## 1. Summary

Errors over all 456 English clips (LibriSpeech test-clean and test-other,
developer terms, everyday dictation), Trit's default against each other
runtime's best setting. Lower is better; the lowest is bold.

<!-- summary:start -->
| Model | Trit (default) | Other runtimes (best setting each) | Trit time per clip | Fastest other |
|---|---:|---|---:|---:|
| Parakeet Redux 0.6B, 1.58-bit (TDT's model) | **293** | Transformers, weights expanded 299, Photon (Kestrel, PyTorch) 300 | 36 ms | 162 ms (Photon (Kestrel, PyTorch)) |
| Parakeet TDT 0.6B v3 | **207** | ONNX Runtime 208, Transformers 211, sherpa-onnx 293 | 39 ms | 130 ms (ONNX Runtime) |
| Whisper base.en | **468** | Transformers 479, whisper.cpp 505, faster-whisper (CTranslate2) 531, OpenAI Whisper 555 | 117 ms | 199 ms (whisper.cpp) |
| Whisper tiny | **760** | Transformers 772, faster-whisper (CTranslate2) 791, OpenAI Whisper 800, whisper.cpp 822 | 67 ms | 109 ms (whisper.cpp) |
| Moonshine tiny | **557** | Transformers **557**, sherpa-onnx 635 | 37 ms | 97 ms (sherpa-onnx) |
| Moonshine base | **415** | Transformers 416, sherpa-onnx 479 | 62 ms | 134 ms (sherpa-onnx) |
| S1-mini (formatted words, 62 cases) | **42** | Transformers fp32 43, llama.cpp f16 GGUF 43, llama.cpp q4_k_m GGUF 44 | 203 ms | 148 ms (llama.cpp q4_k_m GGUF) |
<!-- summary:end -->

1. Trit's default has the fewest errors over all clips, or ties, for every
   model. The check is `desktop/bench/wer_rule.py`.
2. Whisper and Moonshine decode with 5 beams by default, by the rules of
   Transformers. Every runtime with beam search ran it with 5 beams too;
   greedy rows of every runtime stay in the tables.
3. Single sets differ by a few errors between runtimes in both directions,
   so no runtime has the fewest errors on every set; the rule is the total.
4. sherpa-onnx runs Moonshine from its own model export and decodes greedy
   only.

## Cleanup model (S1-mini)

Text in, text out: the 56 sentences of the developer-term and dictation sets
as raw lowercase text without punctuation, plus the six examples of the
model card. Every runtime gets the model's exact prompt and greedy
decoding. Formatted WER counts casing and punctuation. All runtimes ran one
after another in one session.

<!-- cleanup:start -->
| Runtime | Precision | Formatted WER | Exact sentences | Median per case |
|---|---|---:|---:|---:|
| llama.cpp | q4_k_m GGUF | 4.92 % (44 / 894) | 38 / 62 | 148 ms |
| **Trit** | int8 | 4.70 % (42 / 894) | 37 / 62 | 198 ms |
| **Trit** | auto (int8) | 4.70 % (42 / 894) | 37 / 62 | 203 ms |
| llama.cpp | f16 GGUF | 4.81 % (43 / 894) | 36 / 62 | 362 ms |
| **Trit** | f16 | 4.81 % (43 / 894) | 36 / 62 | 402 ms |
| **Trit** | f32 | 4.81 % (43 / 894) | 36 / 62 | 735 ms |
| Transformers (PyTorch) | fp32 | 4.81 % (43 / 894) | 36 / 62 | 1591 ms |
<!-- cleanup:end -->

## 2. Results by model

Each table lists the runtimes of one model from fastest to slowest. Time and
speed are for LibriSpeech test-clean (200 clips of about 7 s). WER is the word
error rate; lower is better. Memory is the CPU working set of the process
after all 456 clips, including the same Python harness in every process.
Trit is called through its Python binding with `language="en"`.

<!-- results:start -->
### Parakeet Redux 0.6B, 1.58-bit (TDT's model)

| Runtime | Precision | Median per clip | Speed | Clean WER | Other WER | Dev terms WER | Memory after load | Memory after all clips |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Trit** | f32, CPU | 35 ms | 160x | 2.02 % | 4.51 % | 7.19 % | 357 MB | 737 MB |
| **Trit** | f16, CPU | 35 ms | 160x | 2.02 % | 4.51 % | 7.19 % | 358 MB | 726 MB |
| **Trit** | int8, CPU | 35 ms | 158x | 2.02 % | 4.51 % | 7.19 % | 358 MB | 735 MB |
| **Trit** | auto (ternary + f16), CPU | 36 ms | 158x | 2.02 % | 4.51 % | 7.19 % | 357 MB | 627 MB |
| Photon (Kestrel, PyTorch) | dense fallback, CPU | 162 ms | 37x | 2.07 % | 4.64 % | 7.19 % | 2,357 MB | 2,960 MB |
| Transformers (PyTorch), weights expanded | fp32, CPU | 331 ms | 19x | 2.05 % | 4.64 % | 7.19 % | 5,390 MB | 5,796 MB |

### Parakeet TDT 0.6B v3

| Runtime | Precision | Median per clip | Speed | Clean WER | Other WER | Dev terms WER | Memory after load | Memory after all clips |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Trit** | auto (int8), CPU | 39 ms | 146x | 1.45 % | 3.25 % | 4.23 % | 822 MB | 1,187 MB |
| **Trit** | int8, CPU | 39 ms | 146x | 1.45 % | 3.25 % | 4.23 % | 825 MB | 1,222 MB |
| ONNX Runtime | int8, CPU | 130 ms | 47x | 1.75 % | 4.59 % | 5.71 % | 774 MB | 1,151 MB |
| **Trit** | f16, CPU | 131 ms | 44x | 1.50 % | 3.33 % | 4.23 % | 1,372 MB | 1,711 MB |
| **Trit** | f32, CPU | 144 ms | 40x | 1.50 % | 3.33 % | 4.23 % | 2,502 MB | 2,822 MB |
| ONNX Runtime | fp32, CPU | 163 ms | 36x | 1.50 % | 3.22 % | 4.23 % | 2,265 MB | 2,820 MB |
| sherpa-onnx | int8, CPU | 194 ms | 31x | 1.67 % | 4.80 % | 6.98 % | 741 MB | 1,110 MB |
| Transformers (PyTorch) | fp32, CPU | 300 ms | 20x | 1.50 % | 3.30 % | 4.23 % | 3,150 MB | 3,750 MB |

### Whisper base.en

| Runtime | Precision | Median per clip | Speed | Clean WER | Other WER | Dev terms WER | Memory after load | Memory after all clips |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Trit** | auto (f16), greedy, CPU | 88 ms | 78x | 3.47 % | 8.99 % | 6.55 % | 175 MB | 268 MB |
| **Trit** | int8, beam 5, CPU | 108 ms | 60x | 3.17 % | 8.07 % | 7.19 % | 153 MB | 271 MB |
| **Trit** | auto (f16), beam 5, CPU | 117 ms | 56x | 3.09 % | 7.89 % | 7.19 % | 169 MB | 293 MB |
| **Trit** | f16, beam 5, CPU | 162 ms | 40x | 3.17 % | 8.18 % | 6.55 % | 201 MB | 321 MB |
| **Trit** | f32, beam 5, CPU | 193 ms | 34x | 3.17 % | 8.18 % | 6.55 % | 338 MB | 449 MB |
| whisper.cpp | f16, greedy, CPU | 199 ms | 35x | 3.34 % | 9.12 % | 6.55 % | 265 MB | 364 MB |
| whisper.cpp | f16, beam 5, CPU | 265 ms | 25x | 3.32 % | 8.75 % | 6.13 % | 266 MB | 395 MB |
| faster-whisper (CTranslate2) | int8, greedy, CPU | 306 ms | 23x | 3.69 % | 9.86 % | 6.34 % | 172 MB | 194 MB |
| faster-whisper (CTranslate2) | int8, beam 5, CPU | 313 ms | 22x | 3.62 % | 9.07 % | 6.55 % | 173 MB | 195 MB |
| OpenAI Whisper (PyTorch) | fp32, greedy, CPU | 375 ms | 17x | 3.82 % | 9.52 % | 6.34 % | 946 MB | 1,019 MB |
| Transformers (PyTorch) | fp32, greedy, CPU | 466 ms | 14x | 3.47 % | 8.91 % | 6.55 % | 919 MB | 1,007 MB |
| OpenAI Whisper (PyTorch) | fp32, beam 5, CPU | 548 ms | 11x | 5.21 % | 9.04 % | 7.19 % | 946 MB | 1,118 MB |
| Transformers (PyTorch) | fp32, beam 5, CPU | 900 ms | 7x | 3.17 % | 8.18 % | 6.55 % | 919 MB | 1,227 MB |

### Whisper tiny

| Runtime | Precision | Median per clip | Speed | Clean WER | Other WER | Dev terms WER | Memory after load | Memory after all clips |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Trit** | auto (f16), greedy, CPU | 48 ms | 144x | 6.76 % | 14.89 % | 10.15 % | 123 MB | 213 MB |
| **Trit** | int8, beam 5, CPU | 66 ms | 98x | 5.84 % | 12.69 % | 7.61 % | 103 MB | 208 MB |
| **Trit** | auto (f16), beam 5, CPU | 67 ms | 96x | 5.76 % | 12.77 % | 7.61 % | 122 MB | 229 MB |
| **Trit** | f16, beam 5, CPU | 82 ms | 78x | 5.94 % | 12.79 % | 7.61 % | 132 MB | 228 MB |
| **Trit** | f32, beam 5, CPU | 99 ms | 65x | 5.94 % | 12.79 % | 7.61 % | 204 MB | 298 MB |
| whisper.cpp | f16, greedy, CPU | 109 ms | 64x | 7.03 % | 15.07 % | 9.51 % | 185 MB | 256 MB |
| faster-whisper (CTranslate2) | int8, greedy, CPU | 162 ms | 43x | 6.98 % | 15.44 % | 8.46 % | 137 MB | 156 MB |
| whisper.cpp | f16, beam 5, CPU | 169 ms | 39x | 6.29 % | 13.63 % | 8.67 % | 186 MB | 282 MB |
| faster-whisper (CTranslate2) | int8, beam 5, CPU | 204 ms | 34x | 5.99 % | 13.24 % | 8.67 % | 137 MB | 157 MB |
| OpenAI Whisper (PyTorch) | fp32, greedy, CPU | 227 ms | 28x | 7.18 % | 15.12 % | 8.46 % | 747 MB | 850 MB |
| OpenAI Whisper (PyTorch) | fp32, beam 5, CPU | 308 ms | 20x | 6.29 % | 13.21 % | 7.82 % | 749 MB | 921 MB |
| Transformers (PyTorch) | fp32, greedy, CPU | 314 ms | 19x | 6.86 % | 14.89 % | 9.51 % | 780 MB | 870 MB |
| Transformers (PyTorch) | fp32, beam 5, CPU | 632 ms | 10x | 6.04 % | 12.79 % | 7.61 % | 781 MB | 1,031 MB |

### Moonshine tiny

| Runtime | Precision | Median per clip | Speed | Clean WER | Other WER | Dev terms WER | Memory after load | Memory after all clips |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Trit** | auto (f16), greedy, CPU | 17 ms | 322x | 4.04 % | 11.17 % | 11.84 % | 113 MB | 219 MB |
| **Trit** | int8, beam 5, CPU | 34 ms | 164x | 3.37 % | 10.09 % | 8.03 % | 99 MB | 210 MB |
| **Trit** | f16, beam 5, CPU | 36 ms | 154x | 3.32 % | 9.80 % | 8.46 % | 113 MB | 239 MB |
| **Trit** | auto (f16), beam 5, CPU | 37 ms | 152x | 3.32 % | 9.80 % | 8.46 % | 113 MB | 239 MB |
| **Trit** | f32, beam 5, CPU | 37 ms | 145x | 3.32 % | 9.80 % | 8.46 % | 167 MB | 289 MB |
| sherpa-onnx | int8, greedy (no beam search), CPU | 97 ms | 63x | 3.67 % | 11.30 % | 9.94 % | 197 MB | 564 MB |
| Transformers (PyTorch) | fp32, greedy, CPU | 174 ms | 34x | 4.04 % | 11.17 % | 11.84 % | 714 MB | 994 MB |
| Transformers (PyTorch) | fp32, beam 5, CPU | 216 ms | 25x | 3.32 % | 9.80 % | 8.46 % | 714 MB | 1,085 MB |

### Moonshine base

| Runtime | Precision | Median per clip | Speed | Clean WER | Other WER | Dev terms WER | Memory after load | Memory after all clips |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Trit** | auto (f16), greedy, CPU | 32 ms | 171x | 2.82 % | 8.39 % | 6.98 % | 173 MB | 286 MB |
| **Trit** | int8, beam 5, CPU | 58 ms | 94x | 2.57 % | 7.13 % | 6.77 % | 137 MB | 278 MB |
| **Trit** | auto (f16), beam 5, CPU | 62 ms | 88x | 2.62 % | 7.18 % | 6.77 % | 173 MB | 271 MB |
| **Trit** | f16, beam 5, CPU | 75 ms | 71x | 2.62 % | 7.18 % | 6.77 % | 181 MB | 280 MB |
| **Trit** | f32, beam 5, CPU | 89 ms | 60x | 2.62 % | 7.18 % | 6.77 % | 296 MB | 379 MB |
| sherpa-onnx | int8, greedy (no beam search), CPU | 134 ms | 44x | 3.14 % | 8.20 % | 6.77 % | 316 MB | 755 MB |
| Transformers (PyTorch) | fp32, greedy, CPU | 287 ms | 20x | 2.84 % | 8.47 % | 7.40 % | 849 MB | 1,144 MB |
| Transformers (PyTorch) | fp32, beam 5, CPU | 426 ms | 11x | 2.64 % | 7.18 % | 6.77 % | 847 MB | 1,363 MB |
<!-- results:end -->

## 3. How the numbers were measured

- **Clips:** 200 LibriSpeech test-clean and 200 test-other (every 13th and
  14th utterance), 40 sentences with developer terms, and 16 everyday
  dictation sentences spoken by the Windows voices. The method of the sets is
  in [desktop/bench/README.md](../desktop/bench/README.md).
- **Same input:** every runtime gets the same 16 kHz float arrays. The quiet
  after the last word is cut to 150 ms, as the TDT app does.
- **Same timer:** the model loads first and one warm-up clip runs. Then each
  clip is one call, timed from the call to the returned text. No batching.
- **Same scorer:** the Whisper English normalizer, as the Open ASR Leaderboard
  uses, then word edit distance.
- **Decoding:** English, no timestamps, no VAD, no prompt from earlier
  clips. Greedy, and for Whisper and Moonshine also 5 beams on every runtime
  that has beam search (sherpa-onnx has none for Moonshine).
- **Precision:** each runtime runs the precision it ships or recommends for
  the device: int8 for CTranslate2 on CPU, q5_0 for whisper.cpp turbo, fp16 on
  the GPU. Kestrel has no ternary CPU kernel for Windows, so Photon on the CPU
  runs dense weights, as TDT's old Photon worker did.
- **Threads:** the other CPU runtimes get 16 threads (all physical cores).
  Trit uses its default: 8 threads on the CPU complex with the large L3 cache
  for Parakeet and S1-mini, 16 for Whisper.
- **Memory:** the working set after loading, and after all 456 clips, both
  read after a garbage collection. The second varies by up to about 35 MB
  between runs of the same code.
- **Isolation:** each runtime runs in its own process and its own Python
  environment, one after another. One run each; expect about 10 % spread in
  speed. Other programs on the computer were idle but not stopped.
- **Trit from Python:** Trit is timed through its Python binding, the same
  path as the other runtimes. The Rust CLI is about 5 % faster.

## 4. What Trit is

| Part | Choice | Where |
|---|---|---|
| Model | `moondream/parakeet-redux`, a 1.58-bit ternary Parakeet TDT 0.6B | Hugging Face |
| Weights | 603,979,776 ternary weights plus 23,290,911 dense weights | `ternary.json` of the model |
| Frontend | log-mel: 512-point FFT, 25 ms window, 10 ms hop, 128 bins | `trit/src/features.rs` |
| Encoder | FastConformer, 24 layers, d_model 1024, 8 heads, subsampling 8 | `trit/src/model.rs` |
| Decoder | TDT, 2 layers, d_model 640, 8,193 tokens, durations 0 to 4 | `trit/src/model.rs` |
| Ternary kernel | i8 activations, `vpdpbusd` on AVX-512 VNNI, `vpmaddubsw` on AVX2, scalar | `trit/src/kernels.rs` |
| Float kernels | f32 GEMM with FMA register tiles; f16 decoder weights with F16C | `trit/src/fgemm.rs`, `trit/src/half.rs` |
| Threads | own team of spinning workers, one per physical core of one L3 domain | `trit/src/team.rs` |
| Dictionary | trie of token paths, logit bonus during greedy decoding | `trit/src/bias.rs` |
| Dependencies | `std`, `parking_lot`, `serde_json`, `mimalloc`. No inference library | `trit/Cargo.toml` |
| Ways in | `trit` CLI, Rust crate, C ABI, Python, C#, Node and Bun | [trit/README.md](../trit/README.md) |

## 5. Where Trit is behind

1. **One model.** Trit runs Parakeet Redux only. The other runtimes run many
   models; a better model on a slower runtime can still be the right choice
   (see Parakeet TDT v2 above).
2. **English only, by choice.** The model knows 25 languages; Trit blocks
   every token that is not English.
3. **No timestamps, no beam search, no streaming API.**
4. **CPU only.** Trit has no GPU path. On a server with a GPU and batches of
   files, a GPU runtime gives more throughput.
5. **x86 speed only.** The fast kernels need AVX2 or AVX-512. Other CPUs run
   the scalar path.

## 6. Reproduce

From `desktop/`, after `cargo build --release -p trit-ffi` in `trit/`:

```powershell
uv run --no-project --python 3.12 python bench/runtimes.py all
python bench/runtimes_report.py
```

`bench/runtimes.py --list` shows every engine; a model name (for example
`whisper-turbo`) runs all runtimes of that model. Details are in
[desktop/bench/README.md](../desktop/bench/README.md).
