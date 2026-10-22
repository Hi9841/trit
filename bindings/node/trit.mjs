// Trit for Node and Bun: speech to text on the CPU, through the C ABI of trit-ffi.
//
//   import { Trit, Cleaner } from "./trit.mjs";
//   const engine = new Trit("whisper-large-v3-turbo");  // folder, `trit models` name, or null
//   console.log(engine.kind);                            // "whisper"
//   console.log(engine.transcribeWav("take.wav"));       // the model detects the language
//   console.log(engine.transcribe(samples, "de"));       // Float32Array at 16 kHz mono
//   engine.close();
//
//   const cleaner = new Cleaner();                       // Superwhisper S1-mini
//   console.log(cleaner.clean("so um send it friday no wait thursday"));
//
// The library is found at TRIT_LIB, next to this file, or in ../../target/release.
// Build it with `cargo build --release -p trit-ffi` in the trit folder.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import koffi from "koffi";

const NAMES = { win32: "trit_ffi.dll", darwin: "libtrit_ffi.dylib" };
const name = NAMES[process.platform] ?? "libtrit_ffi.so";
const here = path.dirname(fileURLToPath(import.meta.url));
const found = [process.env.TRIT_LIB, path.join(here, name), path.join(here, "../../target/release", name)]
  .find((candidate) => candidate && fs.existsSync(candidate));
if (!found) {
  throw new Error(`${name} not found; build it: cargo build --release -p trit-ffi (or set TRIT_LIB=<path>)`);
}

const lib = koffi.load(found);
const c = {
  version: lib.func("const char *trit_version(void)"),
  lastError: lib.func("const char *trit_last_error(void)"),
  load: lib.func("void *trit_load_with(const char *model_dir, const char *precision)"),
  releaseMemory: lib.func("void trit_release_memory(void)"),
  kind: lib.func("const char *trit_kind(void *engine)"),
  free: lib.func("void trit_free(void *engine)"),
  transcribe: lib.func("void *trit_transcribe_with(void *engine, const float *samples, size_t count, const char *language, int beam)"),
  transcribeWav: lib.func("void *trit_transcribe_wav(void *engine, const char *path, const char *language)"),
  freeText: lib.func("void trit_free_text(void *text)"),
  setDictionary: lib.func("int trit_set_dictionary(void *engine, const char *terms)"),
  cleanerLoad: lib.func("void *trit_cleaner_load_with(const char *model_dir, const char *precision)"),
  cleanerFree: lib.func("void trit_cleaner_free(void *cleaner)"),
  clean: lib.func("void *trit_clean(void *cleaner, const char *text, const char *styling, const char *structure, const char *context)"),
};

export class TritError extends Error {}

function text(pointer) {
  if (!pointer) throw new TritError(c.lastError());
  try {
    return koffi.decode(pointer, "char", -1);
  } finally {
    c.freeText(pointer);
  }
}

export const version = () => c.version();

/** Returns memory freed by earlier calls to the system; call it when the app goes idle. */
export const releaseMemory = () => c.releaseMemory();

/** One loaded speech model. Load once and keep it. */
export class Trit {
  #handle;

  /**
   * @param {string | null} model a folder, a `trit models` name, or null for the default.
   * @param {{precision?: string}} options precision: "auto" (default; int8 for layers 384 wide and
   *   more, f16 below), "int8" (least memory), "f16", or "f32" (as the reference).
   */
  constructor(model = null, { precision = null } = {}) {
    this.#handle = c.load(model, precision);
    if (!this.#handle) throw new TritError(c.lastError());
  }

  /** @returns {string} "parakeet_tdt", "parakeet_ctc", "canary", "whisper", or "moonshine". */
  get kind() {
    return c.kind(this.#live());
  }

  /**
   * @param {Float32Array} samples 16 kHz mono, in the range -1 to 1.
   * @param {string | null} language ISO 639-1 code, or null to let the model detect it.
   * @param {{beam?: number}} options beam: beams for Whisper, Moonshine, and Canary (0: the model's
   *   default, 5 for Whisper and Moonshine; 1: greedy).
   */
  transcribe(samples, language = null, { beam = 0 } = {}) {
    const floats = samples instanceof Float32Array ? samples : Float32Array.from(samples);
    return text(c.transcribe(this.#live(), floats, floats.length, language, beam));
  }

  /** @param {string} file a 16 kHz WAV file (16-bit PCM or 32-bit float). */
  transcribeWav(file, language = null) {
    return text(c.transcribeWav(this.#live(), file, language));
  }

  /** @param {string[]} terms @returns {number} terms the model cannot spell (Parakeet uses them). */
  setDictionary(terms) {
    const skipped = c.setDictionary(this.#live(), terms.join("\n"));
    if (skipped < 0) throw new TritError(c.lastError());
    return skipped;
  }

  close() {
    if (this.#handle) c.free(this.#handle);
    this.#handle = null;
  }

  #live() {
    if (!this.#handle) throw new TritError("engine is closed");
    return this.#handle;
  }
}

/** Superwhisper S1-mini: a raw transcript to clean written English. */
export class Cleaner {
  #handle;

  /** @param {{precision?: string}} options as for `Trit`. */
  constructor(model = null, { precision = null } = {}) {
    this.#handle = c.cleanerLoad(model, precision);
    if (!this.#handle) throw new TritError(c.lastError());
  }

  /**
   * @param {string} raw the raw transcript.
   * @param {{styling?: string, structure?: string, context?: string}} style
   *   styling: casual, semi-casual, semi-formal (default), formal; structure: prose (default),
   *   lists; context: general (default), email.
   */
  clean(raw, { styling = null, structure = null, context = null } = {}) {
    if (!this.#handle) throw new TritError("cleaner is closed");
    return text(c.clean(this.#handle, raw, styling, structure, context));
  }

  close() {
    if (this.#handle) c.cleanerFree(this.#handle);
    this.#handle = null;
  }
}
