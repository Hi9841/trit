"""The same speech models on different runtimes: word error rate, speed, memory.

usage: uv run --no-project --python 3.12 --with numpy bench/runtimes.py ENGINE|MODEL|all [...] [--sets clean,other,tech,tts16] [--list]

Run from desktop/. ENGINE is one name from --list; MODEL runs every engine of
that model (parakeet-redux, parakeet-v2, whisper-turbo, ...); `all` runs all.
Each engine runs in its own process and its own uv environment, so packages
never clash and peak memory is its own. Every engine gets the same 16 kHz
arrays (quiet after the last word cut to 150 ms, as bench-asr does), one
warm-up clip, and the same timer around one call per clip.
Results append to target/runtimes/results.jsonl; transcripts go next to it.
Build Trit first: cargo build --release -p trit-ffi (in ../trit).
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import warnings
import wave
from pathlib import Path

warnings.filterwarnings("ignore")  # library deprecation noise; results are on stdout

DESKTOP = Path(__file__).resolve().parents[1]
OUT = DESKTOP / "target" / "runtimes"
CPU_THREADS = os.cpu_count() // 2  # physical cores; SMT siblings only add contention

BASE = ["numpy", "psutil", "whisper-normalizer", "huggingface_hub"]
CUDA = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*"]
TORCH = ["torch==2.9.*"]  # from the CUDA 12.8 index, see ENV_ARGS
ENV_ARGS = {"torch": ["--index", "https://download.pytorch.org/whl/cu128", "--index-strategy", "unsafe-best-match"]}

MODELS = {
    "parakeet-redux": "Parakeet Redux 0.6B, 1.58-bit (TDT's model)",
    "parakeet-v2": "Parakeet TDT 0.6B v2",
    "parakeet-v3": "Parakeet TDT 0.6B v3",
    "whisper-large-v3": "Whisper large-v3",
    "whisper-turbo": "Whisper large-v3-turbo",
    "distil-large-v3.5": "Distil-Whisper large-v3.5",
    "whisper-small.en": "Whisper small.en",
    "whisper-base.en": "Whisper base.en",
    "whisper-tiny.en": "Whisper tiny.en",
    "whisper-tiny": "Whisper tiny",
    "moonshine-tiny": "Moonshine tiny",
    "moonshine-base": "Moonshine base",
}

# name: (model, runtime, precision, device, packages, loader)
ENGINES = {
    "hf-parakeet-redux": ("parakeet-redux", "Transformers (PyTorch), weights expanded", "fp32", "cpu", ["transformers", "safetensors", "librosa", *TORCH], lambda: load_hf_parakeet(MODELS_DIR / "parakeet-redux")),
    "photon-redux": ("parakeet-redux", "Photon (Kestrel, PyTorch)", "dense fallback", "cpu", ["moondream>=2.4.0"], lambda: load_photon('cpu')),
    "photon-redux-gpu": ("parakeet-redux", "Photon (Kestrel, PyTorch)", "ternary", "gpu", ["moondream>=2.4.0", *TORCH], lambda: load_photon('cuda')),

    "onnx-parakeet-v2": ("parakeet-v2", "ONNX Runtime", "fp32", "cpu", ["onnx-asr[cpu,hub]"], lambda: load_onnx('nemo-parakeet-tdt-0.6b-v2', None, 'cpu')),
    "onnx-parakeet-v2-int8": ("parakeet-v2", "ONNX Runtime", "int8", "cpu", ["onnx-asr[cpu,hub]"], lambda: load_onnx('nemo-parakeet-tdt-0.6b-v2', 'int8', 'cpu')),
    "onnx-parakeet-v2-gpu": ("parakeet-v2", "ONNX Runtime", "fp32", "gpu", ["onnx-asr[hub]", "onnxruntime-gpu==1.22.*", *TORCH], lambda: load_onnx('nemo-parakeet-tdt-0.6b-v2', None, 'cuda')),
    "sherpa-parakeet-v2-int8": ("parakeet-v2", "sherpa-onnx", "int8", "cpu", ["sherpa-onnx"], lambda: load_sherpa_tdt('sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8')),
    "onnx-parakeet-v3": ("parakeet-v3", "ONNX Runtime", "fp32", "cpu", ["onnx-asr[cpu,hub]"], lambda: load_onnx('nemo-parakeet-tdt-0.6b-v3', None, 'cpu')),
    "hf-parakeet-v3": ("parakeet-v3", "Transformers (PyTorch)", "fp32", "cpu", ["transformers", "safetensors", "librosa", *TORCH], lambda: load_hf_parakeet(MODELS_DIR / "parakeet-tdt-0.6b-v3")),
    "onnx-parakeet-v3-int8": ("parakeet-v3", "ONNX Runtime", "int8", "cpu", ["onnx-asr[cpu,hub]"], lambda: load_onnx('nemo-parakeet-tdt-0.6b-v3', 'int8', 'cpu')),
    "sherpa-parakeet-v3-int8": ("parakeet-v3", "sherpa-onnx", "int8", "cpu", ["sherpa-onnx"], lambda: load_sherpa_tdt('sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8')),

    "fw-large-v3": ("whisper-large-v3", "faster-whisper (CTranslate2)", "int8", "cpu", ["faster-whisper"], lambda: load_fw('Systran/faster-whisper-large-v3', 'cpu')),
    "fw-large-v3-gpu": ("whisper-large-v3", "faster-whisper (CTranslate2)", "fp16", "gpu", ["faster-whisper", *CUDA], lambda: load_fw('Systran/faster-whisper-large-v3', 'cuda')),
    "hf-large-v3-gpu": ("whisper-large-v3", "Transformers (PyTorch)", "fp16", "gpu", ["transformers", *TORCH], lambda: load_hf_whisper('openai/whisper-large-v3', 'cuda')),
    "oai-large-v3-gpu": ("whisper-large-v3", "OpenAI Whisper (PyTorch)", "fp16", "gpu", ["openai-whisper", *TORCH], lambda: load_oai('large-v3', 'cuda')),

    "fw-large-v3-turbo": ("whisper-turbo", "faster-whisper (CTranslate2)", "int8", "cpu", ["faster-whisper"], lambda: load_fw('mobiuslabsgmbh/faster-whisper-large-v3-turbo', 'cpu')),
    "fw-large-v3-turbo-gpu": ("whisper-turbo", "faster-whisper (CTranslate2)", "fp16", "gpu", ["faster-whisper", *CUDA], lambda: load_fw('mobiuslabsgmbh/faster-whisper-large-v3-turbo', 'cuda')),
    "cpp-large-v3-turbo-q5": ("whisper-turbo", "whisper.cpp", "q5_0", "cpu", ["pywhispercpp"], lambda: load_cpp('ggml-large-v3-turbo-q5_0.bin')),
    "hf-large-v3-turbo": ("whisper-turbo", "Transformers (PyTorch)", "fp32", "cpu", ["transformers", *TORCH], lambda: load_hf_whisper('openai/whisper-large-v3-turbo', 'cpu')),
    "hf-large-v3-turbo-gpu": ("whisper-turbo", "Transformers (PyTorch)", "fp16", "gpu", ["transformers", *TORCH], lambda: load_hf_whisper('openai/whisper-large-v3-turbo', 'cuda')),
    "oai-large-v3-turbo-gpu": ("whisper-turbo", "OpenAI Whisper (PyTorch)", "fp16", "gpu", ["openai-whisper", *TORCH], lambda: load_oai('turbo', 'cuda')),

    "fw-distil-large-v3.5": ("distil-large-v3.5", "faster-whisper (CTranslate2)", "int8", "cpu", ["faster-whisper"], lambda: load_fw('distil-whisper/distil-large-v3.5-ct2', 'cpu')),
    "hf-distil-large-v3.5-gpu": ("distil-large-v3.5", "Transformers (PyTorch)", "fp16", "gpu", ["transformers", *TORCH], lambda: load_hf_whisper('distil-whisper/distil-large-v3.5', 'cuda')),

    "fw-small.en": ("whisper-small.en", "faster-whisper (CTranslate2)", "int8", "cpu", ["faster-whisper"], lambda: load_fw('Systran/faster-whisper-small.en', 'cpu')),
    "cpp-small.en": ("whisper-small.en", "whisper.cpp", "f16", "cpu", ["pywhispercpp"], lambda: load_cpp('ggml-small.en.bin')),
    "hf-small.en": ("whisper-small.en", "Transformers (PyTorch)", "fp32", "cpu", ["transformers", *TORCH], lambda: load_hf_whisper('openai/whisper-small.en', 'cpu')),
    "oai-small.en": ("whisper-small.en", "OpenAI Whisper (PyTorch)", "fp32", "cpu", ["openai-whisper", *TORCH], lambda: load_oai('small.en', 'cpu')),

    "fw-base.en": ("whisper-base.en", "faster-whisper (CTranslate2)", "int8", "cpu", ["faster-whisper"], lambda: load_fw('Systran/faster-whisper-base.en', 'cpu')),
    "cpp-base.en": ("whisper-base.en", "whisper.cpp", "f16", "cpu", ["pywhispercpp"], lambda: load_cpp('ggml-base.en.bin')),
    "hf-base.en": ("whisper-base.en", "Transformers (PyTorch)", "fp32", "cpu", ["transformers", *TORCH], lambda: load_hf_whisper('openai/whisper-base.en', 'cpu')),
    "oai-base.en": ("whisper-base.en", "OpenAI Whisper (PyTorch)", "fp32", "cpu", ["openai-whisper", *TORCH], lambda: load_oai('base.en', 'cpu')),

    "fw-tiny.en": ("whisper-tiny.en", "faster-whisper (CTranslate2)", "int8", "cpu", ["faster-whisper"], lambda: load_fw('Systran/faster-whisper-tiny.en', 'cpu')),

    "fw-tiny": ("whisper-tiny", "faster-whisper (CTranslate2)", "int8", "cpu", ["faster-whisper"], lambda: load_fw('Systran/faster-whisper-tiny', 'cpu')),
    "cpp-tiny": ("whisper-tiny", "whisper.cpp", "f16", "cpu", ["pywhispercpp"], lambda: load_cpp('ggml-tiny.bin')),
    "hf-tiny": ("whisper-tiny", "Transformers (PyTorch)", "fp32", "cpu", ["transformers", *TORCH], lambda: load_hf_whisper('openai/whisper-tiny', 'cpu')),
    "oai-tiny": ("whisper-tiny", "OpenAI Whisper (PyTorch)", "fp32", "cpu", ["openai-whisper", *TORCH], lambda: load_oai('tiny', 'cpu')),

    "hf-moonshine-tiny": ("moonshine-tiny", "Transformers (PyTorch)", "fp32", "cpu", ["transformers", *TORCH], lambda: load_hf_moonshine('UsefulSensors/moonshine-tiny')),
    "sherpa-moonshine-tiny": ("moonshine-tiny", "sherpa-onnx", "int8", "cpu", ["sherpa-onnx"], lambda: load_sherpa_moonshine('sherpa-onnx-moonshine-tiny-en-int8')),


    "hf-moonshine-base": ("moonshine-base", "Transformers (PyTorch)", "fp32", "cpu", ["transformers", *TORCH], lambda: load_hf_moonshine('UsefulSensors/moonshine-base')),
    "sherpa-moonshine-base": ("moonshine-base", "sherpa-onnx", "int8", "cpu", ["sherpa-onnx"], lambda: load_sherpa_moonshine('sherpa-onnx-moonshine-base-en-int8')),
}


# Trit at every precision. "auto" (the engine name without a suffix) is the
# default: int8 for layers 1024 wide and more, f16 below.
TRIT = {
    # model: (`trit models` name, what auto gives)
    "parakeet-redux": (None, "ternary + f16"),
    "parakeet-v3": ("parakeet-tdt-0.6b-v3", "int8"),
    "whisper-base.en": ("whisper-base.en", "f16"),
    "whisper-tiny": ("whisper-tiny", "f16"),
    "moonshine-tiny": ("moonshine-tiny", "f16"),
    "moonshine-base": ("moonshine-base", "f16"),
}
# The models with a text decoder: Trit decodes them with 5 beams by default.
DECODERS = ("whisper-base.en", "whisper-tiny", "moonshine-tiny", "moonshine-base")
for _model, (_name, _auto) in TRIT.items():
    _engine = "trit" if _name is None else f"trit-{_model}"
    _decoding = ", beam 5" if _model in DECODERS else ""
    for _precision in ("auto", "int8", "f16", "f32"):
        ENGINES[_engine if _precision == "auto" else f"{_engine}-{_precision}"] = (
            _model, "Trit", (f"auto ({_auto})" if _precision == "auto" else _precision) + _decoding, "cpu", [],
            lambda n=_name, p=_precision: load_trit(n, p))
# Greedy, as the other runtimes decode by default: for the like-for-like rows.
for _model in DECODERS:
    ENGINES[f"trit-{_model}-greedy"] = (_model, "Trit", f"auto ({TRIT[_model][1]}), greedy", "cpu", [],
                                        lambda n=TRIT[_model][0]: load_trit(n, "auto", beam=1))
# The other runtimes with 5 beams too, so the beam rows compare like for like.
# sherpa-onnx decodes Moonshine greedy only.
for _greedy in [name for name, e in ENGINES.items()
                if e[0] in DECODERS and e[1] != "Trit" and e[3] == "cpu" and not name.startswith("sherpa")]:
    _model, _runtime, _precision, _device, _packages, _loader = ENGINES[_greedy]
    ENGINES[f"{_greedy}-beam5"] = (_model, _runtime, f"{_precision}, beam 5", _device, _packages,
                                   lambda name=_greedy: BEAM_LOADERS[name]())
    ENGINES[_greedy] = (_model, _runtime, f"{_precision}, greedy", _device, _packages, _loader)
for _sherpa in ("sherpa-moonshine-tiny", "sherpa-moonshine-base"):
    _model, _runtime, _precision, _device, _packages, _loader = ENGINES[_sherpa]
    ENGINES[_sherpa] = (_model, _runtime, f"{_precision}, greedy (no beam search)", _device, _packages, _loader)


# The beam-5 loader of each greedy engine of another runtime.
BEAM_LOADERS = {
    "fw-base.en": lambda: load_fw('Systran/faster-whisper-base.en', 'cpu', beam=5),
    "cpp-base.en": lambda: load_cpp('ggml-base.en.bin', beam=5),
    "hf-base.en": lambda: load_hf_whisper('openai/whisper-base.en', 'cpu', beam=5),
    "oai-base.en": lambda: load_oai('base.en', 'cpu', beam=5),
    "fw-tiny": lambda: load_fw('Systran/faster-whisper-tiny', 'cpu', beam=5),
    "cpp-tiny": lambda: load_cpp('ggml-tiny.bin', beam=5),
    "hf-tiny": lambda: load_hf_whisper('openai/whisper-tiny', 'cpu', beam=5),
    "oai-tiny": lambda: load_oai('tiny', 'cpu', beam=5),
    "hf-moonshine-tiny": lambda: load_hf_moonshine('UsefulSensors/moonshine-tiny', beam=5),
    "hf-moonshine-base": lambda: load_hf_moonshine('UsefulSensors/moonshine-base', beam=5),
}


# Loaders. Each imports its own libraries and returns transcribe(samples) -> str.

MODELS_DIR = DESKTOP.parent / "models"


def load_trit(model, precision="auto", beam=0):
    """Trit through its Python binding; `model` is a `trit models` name or
    None for the default. The language is English, as the test is."""
    sys.path.insert(0, str(DESKTOP.parent / "trit" / "bindings" / "python"))
    import trit

    engine = trit.Trit(model, precision=precision)
    return lambda samples: engine.transcribe(samples, language="en", beam=beam)


def load_hf_parakeet(folder):
    """ParakeetForTDT from a Hugging Face folder. Ternary layers (Redux)
    expand to `scale * (code - 1)` in memory, the exact weights the codes
    stand for, so PyTorch runs the same model."""
    torch, _ = torch_device("cpu")
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor, GenerationConfig, ParakeetForTDT

    model = ParakeetForTDT(AutoConfig.from_pretrained(folder)).eval()
    # Redux ships no generation_config.json; it is the same TDT decoder as v3.
    model.generation_config = GenerationConfig.from_pretrained(MODELS_DIR / "parakeet-tdt-0.6b-v3")
    state = {k: v.float() for k, v in load_file(str(folder / "model.safetensors")).items()
             if not k.startswith("vad_head.") and v.is_floating_point() or k.endswith(".qweight")}
    ternary = folder / "ternary.json"
    if ternary.is_file():
        powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32)
        for module in json.loads(ternary.read_text())["quantized_modules"]:
            name, out, inp, group = module["name"], module["out_features"], module["in_features"], module["group_size"]
            packed = state.pop(f"{name}.qweight").to(torch.int32)
            codes = (packed.unsqueeze(-1) // powers % 3).reshape(out, -1)[:, :inp].float()
            scales = state.pop(f"{name}.scales").repeat_interleave(group, dim=1)
            weight = scales * (codes - 1)
            target = model.state_dict()[f"{name}.weight"]
            state[f"{name}.weight"] = weight.reshape(target.shape)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "num_batches_tracked" not in k]
    if missing or unexpected:
        raise RuntimeError(f"weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
    processor = AutoProcessor.from_pretrained(MODELS_DIR / "parakeet-tdt-0.6b-v3")

    @torch.inference_mode()
    def run(samples):
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
        out = model.generate(**inputs)
        return processor.batch_decode(out.sequences, skip_special_tokens=True)[0]

    return run


def load_photon(device):
    """The removed Photon worker, as TDT ran it: offline, telemetry off."""
    os.environ.update(MOONDREAM_API_KEY="", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
    import kestrel.model_download as kestrel_download
    import kestrel_kernels.ternary as kernels
    import moondream as md
    from kestrel import photon as kestrel_photon

    # Kestrel ships no ternary CPU kernel for Windows. TDT's Photon worker
    # fell back to dense weights here; do the same.
    resident_form = kernels.resident_form

    def dense_fallback(device):
        try:
            return resident_form(device)
        except NotImplementedError:
            return "dense"

    kernels.resident_form = dense_fallback

    async def no_report(self, *, rotate: bool = True):
        if rotate:
            self._rotate_window()
        self._pending_reports.clear()

    kestrel_photon.PhotonReporter._flush_window = no_report
    kestrel_download.probe_supported_model_configs = lambda *_, **__: None
    from huggingface_hub import snapshot_download

    client = md.photon("moondream/parakeet-redux", device=device,
                       model_path=snapshot_download("moondream/parakeet-redux"))

    def run(samples):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
        result = client.transcribe(audio=buffer.getvalue())
        return result.get("text", "") if isinstance(result, dict) else str(result)

    return run


def load_onnx(name, quantization, device):
    import onnx_asr
    import onnxruntime

    if device == "cuda":
        import torch  # noqa: F401  loads the CUDA and cuDNN DLLs that ship in its wheel
        onnxruntime.preload_dlls()
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    model = onnx_asr.load_model(name, quantization=quantization, providers=providers)
    return lambda samples: model.recognize(samples, sample_rate=16000)


def sherpa_model(name):
    """A sherpa-onnx model folder, downloaded once from its GitHub release."""
    root = OUT / "models"
    folder = root / name
    if not folder.is_dir():
        # Download and unpack under a name of this process, then rename: a
        # stopped run or a second run never leaves a half folder behind.
        part = root / f".{name}.{os.getpid()}"
        part.mkdir(parents=True)
        url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{name}.tar.bz2"
        archive = part / "model.tar.bz2"
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(part, filter="data")
        (part / name).rename(folder)
        shutil.rmtree(part)
    return folder


def load_sherpa_tdt(name):
    import sherpa_onnx

    d = sherpa_model(name)
    pick = lambda stem: str(next(d.glob(f"{stem}*.onnx")))
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=pick("encoder"), decoder=pick("decoder"), joiner=pick("joiner"), tokens=str(d / "tokens.txt"),
        model_type="nemo_transducer", num_threads=CPU_THREADS, decoding_method="greedy_search")
    return lambda samples: sherpa_decode(recognizer, samples)


def load_sherpa_moonshine(name):
    import sherpa_onnx

    d = sherpa_model(name)
    pick = lambda stem: str(next(d.glob(f"{stem}*.onnx")))
    recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
        preprocessor=pick("preprocess"), encoder=pick("encode"), uncached_decoder=pick("uncached_decode"),
        cached_decoder=pick("cached_decode"), tokens=str(d / "tokens.txt"), num_threads=CPU_THREADS)
    return lambda samples: sherpa_decode(recognizer, samples)


def sherpa_decode(recognizer, samples):
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, samples)
    recognizer.decode_stream(stream)
    return stream.result.text


def cuda_dlls():
    """CTranslate2 on Windows finds cuBLAS and cuDNN only on the DLL path."""
    import nvidia

    for root in nvidia.__path__:
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


def load_fw(repo, device, beam=1):
    if device == "cuda":
        cuda_dlls()
    from faster_whisper import WhisperModel

    model = WhisperModel(repo, device=device, compute_type="float16" if device == "cuda" else "int8",
                         cpu_threads=CPU_THREADS)

    def run(samples):
        # No timestamps, no VAD, no prompt from earlier clips: one dictation take.
        segments, _ = model.transcribe(samples, language="en", beam_size=beam, without_timestamps=True,
                                       condition_on_previous_text=False, vad_filter=False)
        return " ".join(segment.text.strip() for segment in segments)

    return run


def load_cpp(file, beam=1):
    from huggingface_hub import hf_hub_download
    from pywhispercpp.model import Model

    # Strategy 1 is whisper.cpp's beam search.
    search = {"params_sampling_strategy": 1, "beam_search": {"beam_size": beam, "patience": -1.0}} if beam > 1 else {}
    model = Model(hf_hub_download("ggerganov/whisper.cpp", file), n_threads=CPU_THREADS,
                  print_progress=False, print_realtime=False, language="en", no_context=True, **search)
    return lambda samples: " ".join(segment.text.strip() for segment in model.transcribe(samples))


def torch_device(device):
    import torch

    torch.set_num_threads(CPU_THREADS)
    return torch, (torch.float16 if device == "cuda" else torch.float32)


def load_hf_whisper(repo, device, beam=1):
    torch, dtype = torch_device(device)
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(repo)
    model = WhisperForConditionalGeneration.from_pretrained(repo, torch_dtype=dtype).to(device).eval()
    english_only = repo.endswith(".en")
    options = {"num_beams": beam, "do_sample": False}
    if not english_only:
        options.update(language="en", task="transcribe")

    @torch.inference_mode()
    def run(samples):
        long_form = len(samples) > 30 * 16000
        if long_form:  # sequential 30 s windows; Transformers needs timestamps to place them
            inputs = processor(samples, sampling_rate=16000, return_tensors="pt", truncation=False,
                               padding="longest", return_attention_mask=True)
        else:
            inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device, dtype if v.is_floating_point() else v.dtype) for k, v in inputs.items()}
        tokens = model.generate(**inputs, **options, return_timestamps=long_form)
        return processor.batch_decode(tokens, skip_special_tokens=True)[0]

    return run


def load_oai(name, device, beam=1):
    torch_device(device)
    import whisper

    model = whisper.load_model(name, device=device)
    fp16 = device == "cuda"

    def run(samples):
        result = model.transcribe(samples, language="en", temperature=0.0, beam_size=beam if beam > 1 else None, fp16=fp16,
                                  condition_on_previous_text=False, without_timestamps=True)
        return result["text"]

    return run


def load_hf_moonshine(repo, beam=1):
    torch, _ = torch_device("cpu")
    from transformers import AutoProcessor, MoonshineForConditionalGeneration

    processor = AutoProcessor.from_pretrained(repo)
    model = MoonshineForConditionalGeneration.from_pretrained(repo).eval()

    @torch.inference_mode()
    def run(samples):
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
        # Moonshine's own limit: about 6.5 tokens per second of audio.
        limit = int(len(samples) / 16000 * 6.5) + 10
        tokens = model.generate(**inputs, max_length=limit, num_beams=beam)
        return processor.decode(tokens[0], skip_special_tokens=True)

    return run


# The bench.

def read_clip(path: Path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2, path
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    samples = pcm.mean(axis=1).astype(np.float32) / 32767.0
    # The same cut as bench-asr: quiet after the last word down to 150 ms.
    frame, end = 160, len(samples)
    while end >= frame:
        power = float(np.mean(samples[end - frame:end] ** 2))
        if 10 * np.log10(power + 1e-12) >= -45.0:
            break
        end -= frame
    return samples[: min(end + 150 * 16, len(samples))]


def edits(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref):
        cur = [i + 1] + [0] * len(hyp)
        for j, h in enumerate(hyp):
            cur[j + 1] = min(prev[j] + (r != h), prev[j + 1] + 1, cur[j] + 1)
        prev = cur
    return prev[-1]


def child(name: str, sets: list[str]) -> None:
    import gc

    import psutil
    from whisper_normalizer.english import EnglishTextNormalizer

    normalize = EnglishTextNormalizer()
    words = lambda text: normalize(text).split()
    model, runtime, precision, device, _, loader = ENGINES[name]
    started = time.perf_counter()
    transcribe = loader()
    load_s = time.perf_counter() - started
    gc.collect()
    load_mb = psutil.Process().memory_info().rss / 2**20
    clips = {s: sorted((DESKTOP / "target" / "accuracy" / s).glob("*.wav")) for s in sets}
    transcribe(read_clip(clips[sets[0]][0]))  # warm-up: first-call allocations, caches, CUDA kernels
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(OUT / f"{name}.transcripts.jsonl", "w", encoding="utf-8") as log:
        for s in sets:
            errors = total = 0
            audio = busy = 0.0
            times = []
            for path in clips[s]:
                samples = read_clip(path)
                t0 = time.perf_counter()
                text = transcribe(samples)
                ms = (time.perf_counter() - t0) * 1000
                ref = path.with_suffix(".txt").read_text(encoding="utf-8-sig")
                r = words(ref)
                e = edits(r, words(text))
                errors, total = errors + e, total + len(r)
                audio, busy = audio + len(samples) / 16000, busy + ms / 1000
                times.append(ms)
                log.write(json.dumps({"set": s, "clip": path.stem, "errors": e, "ms": round(ms, 1),
                                      "ref": ref.strip(), "hyp": text}) + "\n")
            times.sort()
            rows.append({"set": s, "clips": len(times), "errors": errors, "words": total,
                         "wer": round(100 * errors / total, 2), "median_ms": round(times[len(times) // 2], 1),
                         "audio_s": round(audio, 2), "busy_s": round(busy, 3), "rtfx": round(audio / busy, 1)})
            print(f"  {s}: WER {rows[-1]['wer']:.2f} % ({errors} errors / {total} words, {len(times)} clips, "
                  f"median {rows[-1]['median_ms']:.0f} ms, {rows[-1]['rtfx']:.0f}x realtime)", flush=True)
    gc.collect()
    memory = psutil.Process().memory_info()
    peak_mb = getattr(memory, "peak_wset", memory.rss) / 2**20  # Linux: no peak, so the current RSS
    rss_mb = memory.rss / 2**20
    result = {"engine": name, "model": model, "model_name": MODELS[model], "runtime": runtime,
              "precision": precision, "device": device, "threads": CPU_THREADS, "load_s": round(load_s, 2),
              "load_mb": round(load_mb), "peak_mb": round(peak_mb), "rss_mb": round(rss_mb), "sets": rows, "date": time.strftime("%Y-%m-%d")}
    with open(OUT / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(f"  load {load_s:.2f} s, memory {load_mb:.0f} MB after load, {rss_mb:.0f} MB after the run, "
          f"{peak_mb:.0f} MB peak", flush=True)


def main() -> None:
    args = sys.argv[1:]
    sets = ["clean", "other", "tech", "tts16"]
    if "--sets" in args:
        i = args.index("--sets")
        sets = args[i + 1].split(",")
        del args[i: i + 2]
    if "--child" in args:
        args.remove("--child")
        child(args[0], sets)
        return
    if "--list" in args:
        for name, (model, runtime, precision, device, _, _) in ENGINES.items():
            print(f"{name}: {MODELS[model]}, {runtime}, {precision}, {device}")
        return
    names = []
    for arg in args:
        if arg == "all":
            names += list(ENGINES)
        elif arg in MODELS:
            names += [n for n, e in ENGINES.items() if e[0] == arg]
        elif arg in ENGINES:
            names.append(arg)
        else:
            sys.exit(f"unknown engine or model: {arg}\nmodels: {', '.join(MODELS)}\nengines: run with --list")
    if not names:
        sys.exit(__doc__)
    for name in names:
        print(f"== {name}", flush=True)
        packages = ENGINES[name][4]
        command = ["uv", "run", "--no-project", "--python", "3.12", "--quiet"]
        for package in BASE + packages:
            command += ["--with", package]
        if any(p.startswith("torch") for p in packages):
            command += ENV_ARGS["torch"]
        command += ["python", __file__, name, "--child", "--sets", ",".join(sets)]
        code = subprocess.run(command, cwd=DESKTOP, check=False).returncode
        if code:
            print(f"  failed with exit code {code}", flush=True)


if __name__ == "__main__":
    if "--child" in sys.argv:
        import numpy as np
    main()
