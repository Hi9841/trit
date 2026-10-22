"""Trit for Python: speech to text on the CPU, through the C ABI of trit-ffi.

    import trit
    engine = trit.Trit("whisper-large-v3-turbo")   # a folder, a `trit models` name, or None
    print(engine.kind)                              # "whisper"
    print(engine.transcribe_wav("take.wav"))        # the model detects the language
    print(engine.transcribe(samples, language="de"))  # 16 kHz mono floats: list, array, or numpy

    cleaner = trit.Cleaner()                        # Superwhisper S1-mini
    print(cleaner.clean("so um send it friday no wait thursday"))

The library is found at TRIT_LIB, next to this file, or in ../../target/release.
Build it with `cargo build --release -p trit-ffi` in the trit folder.
"""

from __future__ import annotations

import ctypes
import os
import sys
from array import array
from pathlib import Path

__all__ = ["Trit", "Cleaner", "TritError", "version"]

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
        "trit_transcribe_with": ([ptr, floats, size, text, ctypes.c_int], ptr),
        "trit_transcribe_wav": ([ptr, text, text], ptr),
        "trit_free_text": ([ptr], None),
        "trit_set_dictionary": ([ptr, text], ctypes.c_int),
        "trit_cleaner_load_with": ([text, text], ptr),
        "trit_cleaner_free": ([ptr], None),
        "trit_clean": ([ptr, text, text, text, text], ptr),
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
        return _lib.trit_kind(self._handle).decode()

    def transcribe(self, samples, language: str | None = None, beam: int = 0) -> str:
        """16 kHz mono samples in -1..1: list, array('f'), or numpy array.
        `language`: ISO 639-1 code, or None to let the model detect it.
        `beam`: beams for Whisper, Moonshine, and Canary (0: the model's
        default, 5 for Whisper and Moonshine; 1: greedy)."""
        lang = _arg(language)
        if hasattr(samples, "__array_interface__"):  # numpy, without importing it
            import numpy as np

            buffer = np.ascontiguousarray(samples, dtype=np.float32)
            pointer = buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            return _text(_lib.trit_transcribe_with(self._handle, pointer, buffer.size, lang, beam))
        buffer = samples if isinstance(samples, array) and samples.typecode == "f" else array("f", samples)
        address, count = buffer.buffer_info()
        pointer = ctypes.cast(address, ctypes.POINTER(ctypes.c_float))
        return _text(_lib.trit_transcribe_with(self._handle, pointer, count, lang, beam))

    def transcribe_wav(self, path: str | os.PathLike, language: str | None = None) -> str:
        """A 16 kHz WAV file (16-bit PCM or 32-bit float)."""
        return _text(_lib.trit_transcribe_wav(self._handle, os.fsencode(path), _arg(language)))

    def set_dictionary(self, terms: list[str]) -> int:
        """Terms to favor while decoding (Parakeet). Returns how many it cannot spell."""
        skipped = _lib.trit_set_dictionary(self._handle, "\n".join(terms).encode())
        if skipped < 0:
            raise _error()
        return skipped

    def close(self) -> None:
        if self._handle:
            _lib.trit_free(self._handle)
            self._handle = None

    def __enter__(self) -> "Trit":
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
        return _text(_lib.trit_clean(self._handle, text.encode(), _arg(styling), _arg(structure), _arg(context)))

    def close(self) -> None:
        if self._handle:
            _lib.trit_cleaner_free(self._handle)
            self._handle = None

    def __enter__(self) -> "Cleaner":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
