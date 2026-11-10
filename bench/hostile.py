"""Damaged and hostile files: `trit` must give an error, not crash, hang, or fill the memory.

usage: python bench/hostile.py <trit executable> [--mutate <model folder>] [--runs N] [--old]

Makes each file in a temporary folder and runs `trit transcribe` on it:
- NVIDIA NeMo archives whose PyTorch checkpoint holds a hostile pickle (tuples 200,000
  deep, a memo value that doubles 40 times, a tensor with a negative offset and one with a
  shape near 2^64), and one whose YAML config is nested 5,000 deep;
- WAV files that are empty, cut, or not WAV;
- with --mutate: N copies (default 60) of the folder's config.json, tokenizer.json, and the
  header of model.safetensors with random damage (bytes changed, a cut, a number made 0 or
  very large). Use a small model (Moonshine tiny).
A run passes when the exit code is 0, 1, or 2 within the time limit. Exit 1 when a run fails.
--old leaves out the memo case: a build without its limit fills the memory.
Python's standard library only.
"""

import io
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import zipfile

LIMIT_S = 120
YAML = """encoder:
  _target_: nemo.collections.asr.modules.ConformerEncoder
  subsampling: dw_striding
  self_attention_model: rel_pos
  feat_in: 80
  d_model: 64
  ff_expansion_factor: 4
  n_heads: 4
  n_layers: 1
  conv_kernel_size: 9
  subsampling_factor: 8
  subsampling_conv_channels: 64
preprocessor:
  sample_rate: 16000
  n_fft: 512
  window_size: 0.025
  window_stride: 0.01
  window: hann
  normalize: per_feature
  features: 80
decoder:
  _target_: nemo.collections.asr.modules.RNNTDecoder
  prednet:
    pred_hidden: 64
    pred_rnn_layers: 1
joint:
  _target_: nemo.collections.asr.modules.RNNTJoint
labels:
- a
- b
tokenizer:
  model_path: nemo:tok.model
"""


def text(s: str) -> bytes:
    """BINUNICODE."""
    raw = s.encode()
    return b"X" + struct.pack("<I", len(raw)) + raw


def tensor(offset: bytes, shape: bytes) -> bytes:
    """A state dict with one tensor `w`, as torch.save writes it; `offset` and `shape` are pickle values."""
    storage = b"(" + text("storage") + b"ctorch\nFloatStorage\n" + text("0") + text("cpu") + b"K\x04tQ"
    rebuild = b"ctorch._utils\n_rebuild_tensor_v2\n(" + storage + offset + shape + b"(K\x01tt" + b"R"
    return b"\x80\x02}(" + text("w") + rebuild + b"u."


def long(value: int) -> bytes:
    """LONG1 with 8 bytes."""
    return b"\x8a\x08" + struct.pack("<q", value)


PICKLES = {
    "pickle-control": tensor(b"K\x00", b"(K\x02t"),
    "pickle-deep": b"\x80\x02N" + b"\x85" * 200_000 + b".",
    "pickle-memo": b"\x80\x02]q\x00" + b"(h\x00h\x00tq\x00" * 40 + b".",
    "pickle-negative-offset": tensor(b"J\xff\xff\xff\xff", b"(K\x02t"),
    "pickle-huge-shape": tensor(b"K\x00", b"(" + long(2**62) + long(2**62) + b"t"),
}


def nemo(folder: str, pickle: bytes, yaml: str = YAML) -> None:
    checkpoint = io.BytesIO()
    with zipfile.ZipFile(checkpoint, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", pickle)
        archive.writestr("archive/data/0", bytes(16))
    with tarfile.open(os.path.join(folder, "model.nemo"), "w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in (("model_config.yaml", yaml.encode()), ("model_weights.ckpt", checkpoint.getvalue())):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def wav(data: bytes, rate: int = 16000, bits: int = 16, size: int | None = None) -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * bits // 8, bits // 8, bits)
    body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data) if size is None else size) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def damage(raw: bytes, rng: random.Random) -> bytes:
    """One random change of a JSON text: bytes changed, a cut, or a number made 0 or very large."""
    kind = rng.randrange(4)
    if kind == 0 and raw:
        out = bytearray(raw)
        for _ in range(rng.randrange(1, 6)):
            out[rng.randrange(len(out))] = rng.randrange(256)
        return bytes(out)
    if kind == 1:
        return raw[: rng.randrange(len(raw) + 1)]
    digits = [i for i, b in enumerate(raw) if 48 <= b <= 57 and (i == 0 or not 48 <= raw[i - 1] <= 57)]
    if not digits:
        return raw[::-1]
    at = rng.choice(digits)
    end = at
    while end < len(raw) and 48 <= raw[end] <= 57:
        end += 1
    return raw[:at] + rng.choice([b"0", b"1", b"18446744073709551615", b"4294967296", b"-1", b"999999"]) + raw[end:]


def run(trit: str, model: str, clip: str) -> tuple[str, str]:
    """`ok` or the failure, and the first line that trit wrote."""
    try:
        done = subprocess.run([trit, "transcribe", clip, "--model", model, "--format", "text"],
                              capture_output=True, timeout=LIMIT_S)
    except subprocess.TimeoutExpired:
        return f"no end in {LIMIT_S} s", ""
    line = (done.stdout or done.stderr).decode(errors="replace").strip().splitlines()[:1]
    return ("ok" if done.returncode in (0, 1, 2) else f"exit code {done.returncode:#x}"), (line[0][:110] if line else "")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    args = sys.argv[2:]
    trit = os.path.abspath(sys.argv[1])
    mutate = args[args.index("--mutate") + 1] if "--mutate" in args else None
    runs = int(args[args.index("--runs") + 1]) if "--runs" in args else 60
    failed = 0
    with tempfile.TemporaryDirectory(prefix="trit-hostile-") as root:
        clip = os.path.join(root, "tone.wav")
        open(clip, "wb").write(wav(bytes(32000)))

        def check(name: str, model: str, clip: str = clip) -> None:
            nonlocal failed
            result, line = run(trit, model, clip)
            failed += result != "ok"
            print(f"{'ok  ' if result == 'ok' else 'FAIL'} {name}: {result if result != 'ok' else ''} {line}")

        for name, pickle in PICKLES.items():
            if name == "pickle-memo" and "--old" in args:
                continue
            folder = os.path.join(root, name)
            os.mkdir(folder)
            nemo(folder, pickle)
            check(name, folder)
        folder = os.path.join(root, "yaml-deep")
        os.mkdir(folder)
        nemo(folder, PICKLES["pickle-control"], "encoder:\n" + "  " + "- " * 5000 + "1\n")
        check("yaml-deep", folder)

        files = {
            "wav-empty": b"", "wav-riff-only": b"RIFF\x04\x00\x00\x00WAVE", "wav-random": random.Random(1).randbytes(4000),
            "wav-no-data": wav(b"")[:36], "wav-cut-header": wav(bytes(3200))[:30], "wav-8-bit": wav(bytes(800), bits=8),
            "wav-size-too-large": wav(bytes(3200), size=0xFFFFFFFF), "wav-rate-0": wav(bytes(3200), rate=0),
        }
        # The WAV reader runs only when the model loads: use the folder of --mutate when given.
        default = mutate or "parakeet-redux"
        for name, data in files.items():
            path = os.path.join(root, name + ".wav")
            open(path, "wb").write(data)
            check(name, default, path)

        if mutate:
            rng = random.Random(7)
            folder = os.path.join(root, "mutate")
            shutil.copytree(mutate, folder, ignore=shutil.ignore_patterns(".trit"))
            weights = os.path.join(folder, "model.safetensors")
            originals = {name: open(os.path.join(folder, name), "rb").read() for name in ("config.json", "tokenizer.json")}
            header_len = struct.unpack("<Q", open(weights, "rb").read(8))[0]
            header = open(weights, "rb").read(8 + header_len)
            for i in range(runs):
                target = ("config.json", "tokenizer.json", "header")[i % 3]
                if target == "header":
                    changed = damage(header[8:], rng)[:header_len].ljust(header_len, b" ")
                    with open(weights, "r+b") as f:
                        f.write(header[:8] + changed)
                else:
                    open(os.path.join(folder, target), "wb").write(damage(originals[target], rng))
                check(f"mutate-{i:03}-{target}", folder)
                for name, data in originals.items():
                    open(os.path.join(folder, name), "wb").write(data)
                with open(weights, "r+b") as f:
                    f.write(header)
    print(f"{failed} failed" if failed else "all runs ended with an error or a text")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
