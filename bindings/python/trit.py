"""Trit for Python: speech to text and text to speech on the CPU, through the C ABI of trit-ffi.

    import trit
    engine = trit.Trit("whisper-large-v3-turbo")   # a folder, a `trit models` name, or None
    print(engine.kind)                              # "whisper"
    print(engine.transcribe_wav("take.wav"))        # the model detects the language
    print(engine.transcribe(samples, language="de"))  # 16 kHz mono floats: list, array, or numpy
    print(engine.transcribe_wav("take.wav", task="translate"))  # into English
    for s in engine.segments_wav("take.wav"):       # timed sentences
        print(f"{s.start:.2f}-{s.end:.2f} {s.text}")

    cleaner = trit.Cleaner()                        # Superwhisper S1-mini
    print(cleaner.clean("so um send it friday no wait thursday"))

    voice = trit.Voice()                            # Meta MMS-TTS English
    voice.speak_wav("The build passed.", "speech.wav")
    samples = voice.speak("Ship it.", speed=1.1)    # array('f') at voice.sample_rate

The library is found at TRIT_LIB, next to this file, or in ../../target/release.
Build it with `cargo build --release -p trit-ffi` in the trit folder.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self

__all__ = ["Cleaner", "Segment", "Trit", "TritError", "Voice", "release_memory", "version"]

_NAMES = {"win32": "trit_ffi.dll", "darwin": "libtrit_ffi.dylib"}


class TritError(RuntimeError):
    """A failed Trit call. The message names the cause and the fix."""


def _library() -> ctypes.CDLL:
    name = _NAMES.get(sys.platform, "libtrit_ffi.so")
    here = Path(__file__).resolve().parent
    candidates = [os.environ.get("TRIT_LIB"), here / name, here.parents[1] / "target" / "release" / name]
    for path in candidates:
        if path and Path(path).is_file():
            lib = ctypes.CDLL(str(path))
            break
    else:
        raise TritError(f"{name} not found; build it: cargo build --release -p trit-ffi (or set TRIT_LIB=<path>)")
    text, ptr, size = ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t
    floats = ctypes.POINTER(ctypes.c_float)
    signatures = {
        "trit_version": ([], text),
        "trit_last_error": ([], text),
        "trit_load_with": ([text, text], ptr),
        "trit_release_memory": ([], None),
        "trit_kind": ([ptr], text),
        "trit_free": ([ptr], None),
        "trit_transcribe_json": ([ptr, floats, size, text], ptr),
        "trit_transcribe_wav_json": ([ptr, text, text], ptr),
        "trit_free_text": ([ptr], None),
        "trit_set_dictionary": ([ptr, text], ctypes.c_int),
        "trit_cleaner_load_with": ([text, text], ptr),
        "trit_cleaner_free": ([ptr], None),
        "trit_clean": ([ptr, text, text, text, text], ptr),
        "trit_voice_load": ([text], ptr),
        "trit_voice_free": ([ptr], None),
        "trit_voice_sample_rate": ([ptr], ctypes.c_int),
        "trit_speak": ([ptr, text, text, ctypes.POINTER(size)], floats),
        "trit_speak_wav": ([ptr, text, text, text], ctypes.c_int),
        "trit_free_samples": ([floats, size], None),
    }
    for fn, (args, result) in signatures.items():
        getattr(lib, fn).argtypes = args
        getattr(lib, fn).restype = result
    return lib


_lib = _library()


def version() -> str:
    return _lib.trit_version().decode()


def release_memory() -> None:
    """Return memory freed by earlier calls to the system; call it when the
    app goes idle, not between calls."""
    _lib.trit_release_memory()


def _error() -> TritError:
    return TritError(_lib.trit_last_error().decode())


def _text(pointer: int | None) -> str:
    if not pointer:
        raise _error()
    try:
        return ctypes.string_at(pointer).decode()
    finally:
        _lib.trit_free_text(pointer)


def _arg(value: str | os.PathLike | None) -> bytes | None:
    return None if value is None else os.fsencode(value)


@dataclass(frozen=True)
class Segment:
    """A stretch of the recording and its text; times in seconds from the start."""

    start: float
    end: float
    text: str


def _options(language, beam, task, fallback, timestamps) -> bytes:
    options = {"language": language, "beam": beam, "timestamps": timestamps}
    if task is not None:
        options["task"] = task
    if fallback is not None:
        options["fallback"] = fallback
    return json.dumps(options).encode()


def _samples(samples):
    """A float pointer and a count for `samples`, and the buffer that holds them."""
    if hasattr(samples, "__array_interface__"):  # numpy, without importing it
        import numpy as np

        buffer = np.ascontiguousarray(samples, dtype=np.float32).ravel()
        return buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), buffer.size, buffer
    buffer = samples if isinstance(samples, array) and samples.typecode == "f" else array("f", samples)
    address, count = buffer.buffer_info()
    return ctypes.cast(address, ctypes.POINTER(ctypes.c_float)), count, buffer


class Trit:
    """One loaded speech model. Load once and keep it."""

    def __init__(self, model: str | os.PathLike | None = None, precision: str | None = None):
        """`model`: a folder, a `trit models` name, or None for the default.
        `precision`: "auto" (None; int8 for layers 384 wide and more, f16
        below), "int8" (least memory), "f16", or "f32" (as the reference)."""
        self._handle = None
        handle = _lib.trit_load_with(_arg(model), _arg(precision))
        if not handle:
            raise _error()
        self._handle = handle

    @property
    def kind(self) -> str:
        """"parakeet_tdt", "parakeet_ctc", "canary", "whisper", or "moonshine"."""
        return _lib.trit_kind(self._live()).decode()

    def transcribe(self, samples, language: str | None = None, beam: int = 0,
                   task: str | None = None, fallback: bool | None = None) -> str:
        """16 kHz mono samples in -1..1: list, array('f'), or numpy array.
        `language`: ISO 639-1 code, or None to let the model detect it.
        `beam`: beams for Whisper, Moonshine, and Canary (0: the model's
        default, 5 for Whisper and Moonshine; 1: greedy).
        `task`: "transcribe" (None), or "translate" into English
        (multilingual Whisper, Canary).
        `fallback`: Whisper decodes a window again at higher temperatures
        when its text repeats itself (None: on)."""
        return self._samples(samples, _options(language, beam, task, fallback, False))["text"]

    def transcribe_wav(self, path: str | os.PathLike, language: str | None = None, beam: int = 0,
                       task: str | None = None, fallback: bool | None = None) -> str:
        """A 16 kHz WAV file (16-bit PCM or 32-bit float); options as for `transcribe`."""
        return self._wav(path, _options(language, beam, task, fallback, False))["text"]

    def segments(self, samples, language: str | None = None, beam: int = 0,
                 task: str | None = None, fallback: bool | None = None) -> list[Segment]:
        """The text in timed segments: sentences or pauses for Parakeet,
        windows of up to 30 s (40 s Canary) for the others. Options as for
        `transcribe`."""
        result = self._samples(samples, _options(language, beam, task, fallback, True))
        return [Segment(**s) for s in result["segments"]]

    def segments_wav(self, path: str | os.PathLike, language: str | None = None, beam: int = 0,
                     task: str | None = None, fallback: bool | None = None) -> list[Segment]:
        """`segments` for a 16 kHz WAV file."""
        result = self._wav(path, _options(language, beam, task, fallback, True))
        return [Segment(**s) for s in result["segments"]]

    def _samples(self, samples, options: bytes) -> dict:
        pointer, count, _buffer = _samples(samples)
        return json.loads(_text(_lib.trit_transcribe_json(self._live(), pointer, count, options)))

    def _wav(self, path: str | os.PathLike, options: bytes) -> dict:
        return json.loads(_text(_lib.trit_transcribe_wav_json(self._live(), os.fsencode(path), options)))

    def _live(self):
        if not self._handle:
            raise TritError("engine is closed")
        return self._handle

    def set_dictionary(self, terms: list[str]) -> int:
        """Terms to favor while decoding (Parakeet). Returns how many it cannot spell.
        An empty list clears them."""
        skipped = _lib.trit_set_dictionary(self._live(), "\n".join(terms).encode())
        if skipped < 0:
            raise _error()
        return skipped

    def close(self) -> None:
        if self._handle:
            _lib.trit_free(self._handle)
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class Cleaner:
    """Superwhisper S1-mini: a raw transcript to clean written English."""

    def __init__(self, model: str | os.PathLike | None = None, precision: str | None = None):
        """`precision` as for `Trit`."""
        self._handle = None
        handle = _lib.trit_cleaner_load_with(_arg(model), _arg(precision))
        if not handle:
            raise _error()
        self._handle = handle

    def clean(self, text: str, styling: str | None = None, structure: str | None = None,
              context: str | None = None) -> str:
        """styling: casual, semi-casual, semi-formal (default), formal;
        structure: prose (default), lists; context: general (default), email."""
        if not self._handle:
            raise TritError("cleaner is closed")
        return _text(_lib.trit_clean(self._handle, text.encode(), _arg(styling), _arg(structure), _arg(context)))

    def close(self) -> None:
        if self._handle:
            _lib.trit_cleaner_free(self._handle)
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class Voice:
    """A VITS voice (Meta MMS-TTS): text to speech. It reads letters; write
    numbers and symbols as words."""

    def __init__(self, model: str | os.PathLike | None = None):
        """`model`: a folder, a `trit models` name ("mms-tts-eng"), or None for the default."""
        self._handle = None
        handle = _lib.trit_voice_load(_arg(model))
        if not handle:
            raise _error()
        self._handle = handle

    @property
    def sample_rate(self) -> int:
        """The rate of the samples of `speak` (16000 for MMS-TTS)."""
        return _lib.trit_voice_sample_rate(self._live())

    def speak(self, text: str, speed: float | None = None, noise: float | None = None,
              noise_duration: float | None = None, seed: int = 0) -> array:
        """Mono samples in -1..1 at `sample_rate`, as array('f').
        `speed`: 1 is the voice's pace, 1.25 faster (None: the voice's own).
        `noise`, `noise_duration`: how much the voice and the timing vary
        (None: the voice's own). One text and one `seed` give the same audio."""
        count = ctypes.c_size_t()
        options = _voicing(speed, noise, noise_duration, seed)
        pointer = _lib.trit_speak(self._live(), text.encode(), options, ctypes.byref(count))
        if not pointer:
            raise _error()
        try:
            samples = array("f")
            samples.frombytes(ctypes.string_at(pointer, count.value * 4))
            return samples
        finally:
            _lib.trit_free_samples(pointer, count)

    def speak_wav(self, text: str, path: str | os.PathLike, speed: float | None = None,
                  noise: float | None = None, noise_duration: float | None = None, seed: int = 0) -> None:
        """`speak` into a 16-bit mono WAV file."""
        options = _voicing(speed, noise, noise_duration, seed)
        if _lib.trit_speak_wav(self._live(), text.encode(), os.fsencode(path), options) < 0:
            raise _error()

    def _live(self):
        if not self._handle:
            raise TritError("voice is closed")
        return self._handle

    def close(self) -> None:
        if self._handle:
            _lib.trit_voice_free(self._handle)
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _voicing(speed, noise, noise_duration, seed) -> bytes:
    options = {"seed": seed}
    for key, value in (("speed", speed), ("noise", noise), ("noise_duration", noise_duration)):
        if value is not None:
            options[key] = value
    return json.dumps(options).encode()
