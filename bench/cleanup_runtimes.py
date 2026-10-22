"""S1-mini (transcript cleanup) on different runtimes: the same prompts, scored the same way.

usage: uv run --no-project --python 3.12 python bench/cleanup_runtimes.py ENGINE|all [...] [--list]

Run from desktop/. Inputs: the 56 sentences of bench/corpus.txt and
bench/tech-corpus.txt as a raw transcript (lowercase, no punctuation but
apostrophes), plus the six examples of the S1-mini model card. The target
is the written sentence. Every runtime gets the exact prompt the model was
trained on, greedy decoding, and one warm-up call. Formatted WER counts
casing and punctuation (words split on spaces); `exact` counts outputs equal
to the target. Results append to target/runtimes/cleanup.jsonl.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[1]
MODELS = DESKTOP.parent / "models"
OUT = DESKTOP / "target" / "runtimes"
THREADS = os.cpu_count() // 2

SYSTEM = ("You are a text normalizer for speech-to-text transcripts. The input begins with a control "
          "line specifying the styling, structure, and context settings; clean the transcript to match "
          "those settings and output only the cleaned text.")
CONTROL = "[Styling: semi-formal] [Structure: prose] [Context: general]"

# The examples of the model card, with its outputs.
CARD = [
    ("so um i need to like send the the report by uh friday no wait make that thursday",
     "I need to send the report by Thursday."),
    ("i think the answer is forty two no sorry forty three", "I think the answer is 43."),
    ("let's meet at half past two tomorrow uh actually make it three fifteen p m",
     "Let's meet at 3:15pm tomorrow."),
    ("the invoice came to twenty three thousand four hundred and fifty dollars and it's due on march third twenty twenty six",
     "The invoice came to $23,450, and it's due on March 3, 2026."),
    ("send it to support at superwhisper dot com", "Send it to support@superwhisper.com."),
    ("um", ""),
]

TORCH = ["torch==2.9.*"]
LLAMA = ["llama-cpp-python"]
ENV = {
    "torch": ["--index", "https://download.pytorch.org/whl/cu128", "--index-strategy", "unsafe-best-match"],
    "llama": ["--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu",
              "--index-strategy", "unsafe-best-match"],
}

# name: (runtime, precision, packages, loader)
ENGINES = {
    "trit": ("Trit", "auto (int8)", [], lambda: load_trit("auto")),
    "trit-int8": ("Trit", "int8", [], lambda: load_trit("int8")),
    "trit-f16": ("Trit", "f16", [], lambda: load_trit("f16")),
    "trit-f32": ("Trit", "f32", [], lambda: load_trit("f32")),
    "hf": ("Transformers (PyTorch)", "fp32", ["transformers", *TORCH], lambda: load_hf()),
    "llamacpp-f16": ("llama.cpp", "f16 GGUF", LLAMA, lambda: load_llama("s1-mini-f16.gguf")),
    "llamacpp-q4": ("llama.cpp", "q4_k_m GGUF", LLAMA, lambda: load_llama("s1-mini-q4_k_m.gguf")),
}


def prompt(raw: str) -> str:
    """The Qwen3 chat template with thinking off, as S1-mini was trained."""
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{CONTROL}\n{raw}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def budget(raw: str) -> int:
    return min(len(raw) // 2 + 64, 1024)


def load_trit(precision):
    sys.path.insert(0, str(DESKTOP.parent / "trit" / "bindings" / "python"))
    import trit

    cleaner = trit.Cleaner(str(MODELS / "s1-mini"), precision=precision)
    return lambda raw: cleaner.clean(raw)


def load_hf():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(THREADS)
    tok = AutoTokenizer.from_pretrained(MODELS / "s1-mini")
    model = AutoModelForCausalLM.from_pretrained(MODELS / "s1-mini", torch_dtype=torch.float32).eval()
    stop = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]

    @torch.inference_mode()
    def run(raw):
        ids = tok(prompt(raw), return_tensors="pt", add_special_tokens=False).input_ids
        out = model.generate(ids, max_new_tokens=budget(raw), do_sample=False, eos_token_id=stop)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    return run


def load_llama(file):
    from llama_cpp import Llama

    llm = Llama(model_path=str(MODELS / "s1-mini-gguf" / file), n_ctx=2048, n_threads=THREADS,
                n_threads_batch=THREADS, verbose=False)

    def run(raw):
        out = llm(prompt(raw), max_tokens=budget(raw), temperature=0.0, top_k=1,
                  stop=["<|im_end|>", "<|endoftext|>"])
        return out["choices"][0]["text"].strip()

    return run


def cases():
    rows = []
    for name in ("corpus.txt", "tech-corpus.txt"):
        for line in (DESKTOP / "bench" / name).read_text(encoding="utf-8").splitlines():
            if "|" in line:
                target = line.split("|", 1)[1].strip()
                raw = re.sub(r"[^\w\s']", " ", target.lower())
                rows.append((" ".join(raw.split()), target))
    return rows + CARD


def edits(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref):
        cur = [i + 1] + [0] * len(hyp)
        for j, b in enumerate(hyp):
            cur[j + 1] = min(prev[j] + (a != b), prev[j + 1] + 1, cur[j] + 1)
        prev = cur
    return prev[-1]


def child(name):
    runtime, precision, _, loader = ENGINES[name]
    clean = loader()
    clean("hello there")  # warm-up
    errors = words = exact = 0
    times = []
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"cleanup-{name}.jsonl", "w", encoding="utf-8") as log:
        for raw, target in cases():
            t0 = time.perf_counter()
            text = clean(raw)
            ms = (time.perf_counter() - t0) * 1000
            ref, hyp = target.split(), text.split()
            e = edits(ref, hyp)
            errors, words, exact = errors + e, words + max(len(ref), 1), exact + (text == target)
            times.append(ms)
            log.write(json.dumps({"raw": raw, "target": target, "text": text, "errors": e, "ms": round(ms, 1)}) + "\n")
    times.sort()
    result = {"engine": name, "runtime": runtime, "precision": precision, "cases": len(times),
              "wer": round(100 * errors / words, 2), "errors": errors, "words": words, "exact": exact,
              "median_ms": round(times[len(times) // 2], 1), "date": time.strftime("%Y-%m-%d")}
    with open(OUT / "cleanup.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(f"  formatted WER {result['wer']:.2f} % ({errors} / {words}), exact {exact}/{len(times)}, "
          f"median {result['median_ms']:.0f} ms", flush=True)


def main():
    args = sys.argv[1:]
    if "--child" in args:
        args.remove("--child")
        child(args[0])
        return
    if "--list" in args:
        for name, (runtime, precision, _, _) in ENGINES.items():
            print(f"{name}: {runtime}, {precision}")
        return
    names = list(ENGINES) if args == ["all"] else args
    for name in names:
        if name not in ENGINES:
            sys.exit(f"unknown engine {name}; run with --list")
    for name in names:
        print(f"== {name}", flush=True)
        packages = ENGINES[name][2]
        command = ["uv", "run", "--no-project", "--python", "3.12", "--quiet"]
        for package in packages:
            command += ["--with", package]
        if packages == LLAMA:
            command += ENV["llama"]
        elif any(p.startswith("torch") for p in packages):
            command += ENV["torch"]
        command += ["python", __file__, name, "--child"]
        code = subprocess.run(command, cwd=DESKTOP).returncode
        if code:
            print(f"  failed with exit code {code}", flush=True)


if __name__ == "__main__":
    main()
