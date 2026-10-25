// Trit for Node and Bun: speech to text and text to speech on the CPU, through the C ABI of trit-ffi.
//
//   import { Trit, Cleaner, Voice } from "./trit.mjs";
//   const engine = new Trit("whisper-large-v3-turbo");  // folder, `trit models` name, or null
//   console.log(engine.kind);                            // "whisper"
//   console.log(engine.transcribeWav("take.wav"));       // the model detects the language
//   console.log(engine.transcribe(samples, "de"));       // Float32Array at 16 kHz mono
//   console.log(engine.transcribeWav("take.wav", null, { task: "translate" }));  // into English
//   for (const s of engine.segmentsWav("take.wav")) console.log(s.start, s.end, s.text);
//   engine.close();
//
//   const cleaner = new Cleaner();                       // Superwhisper S1-mini
//   console.log(cleaner.clean("so um send it friday no wait thursday"));
//
//   const voice = new Voice();                           // Kokoro-82M, else MMS-TTS English
//   voice.speakWav("The build passed on March 3rd.", "speech.wav", { voice: "bf_emma" });
//   const samples = voice.speak("Ship it.", { speed: 1.1 });  // Float32Array at voice.sampleRate
//   voice.speakStream(longText, (part) => { play(part); return true; });  // each part as it is made
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
const SamplesFn = koffi.proto("int TritSamplesFn(const float *samples, size_t count, void *user_data)");

const c = {
  version: lib.func("const char *trit_version(void)"),
  lastError: lib.func("const char *trit_last_error(void)"),
  load: lib.func("void *trit_load_with(const char *model_dir, const char *precision)"),
  releaseMemory: lib.func("void trit_release_memory(void)"),
  kind: lib.func("const char *trit_kind(void *engine)"),
  free: lib.func("void trit_free(void *engine)"),
  transcribe: lib.func("void *trit_transcribe_json(void *engine, const float *samples, size_t count, const char *options)"),
  transcribeWav: lib.func("void *trit_transcribe_wav_json(void *engine, const char *path, const char *options)"),
  freeText: lib.func("void trit_free_text(void *text)"),
  setDictionary: lib.func("int trit_set_dictionary(void *engine, const char *terms)"),
  cleanerLoad: lib.func("void *trit_cleaner_load_with(const char *model_dir, const char *precision)"),
  cleanerFree: lib.func("void trit_cleaner_free(void *cleaner)"),
  clean: lib.func("void *trit_clean(void *cleaner, const char *text, const char *styling, const char *structure, const char *context)"),
  voiceLoad: lib.func("void *trit_voice_load(const char *model_dir)"),
  voiceFree: lib.func("void trit_voice_free(void *voice)"),
  voiceSampleRate: lib.func("int trit_voice_sample_rate(void *voice)"),
  speak: lib.func("void *trit_speak(void *voice, const char *text, const char *options, _Out_ size_t *count)"),
  speakWav: lib.func("int trit_speak_wav(void *voice, const char *text, const char *path, const char *options)"),
  freeSamples: lib.func("void trit_free_samples(void *samples, size_t count)"),
  speakStream: lib.func("int trit_speak_stream(void *voice, const char *text, const char *options, TritSamplesFn *callback, void *user_data)"),
  voiceList: lib.func("void *trit_voice_list(void *voice)"),
  voiceKind: lib.func("const char *trit_voice_kind(void *voice)"),
  phonemes: lib.func("void *trit_phonemes(void *voice, const char *text, const char *options)"),
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

/** The options JSON of trit_transcribe_json. */
function options(language, { beam = 0, task = null, fallback = null } = {}, timestamps = false) {
  const json = { language, beam, timestamps };

  if (task !== null) json.task = task;

  if (fallback !== null) json.fallback = fallback;

  return JSON.stringify(json);
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
   * @param {{beam?: number, task?: string, fallback?: boolean}} settings beam: beams for Whisper,
   *   Moonshine, and Canary (0: the model's default, 5 for Whisper and Moonshine; 1: greedy).
   *   task: "transcribe" (default), or "translate" into English (multilingual Whisper, Canary).
   *   fallback: Whisper decodes a window again at higher temperatures when its text repeats
   *   itself (default true).
   * @returns {string}
   */
  transcribe(samples, language = null, settings = {}) {
    return this.#samples(samples, options(language, settings)).text;
  }

  /** @param {string} file a 16 kHz WAV file (16-bit PCM or 32-bit float); the rest as for `transcribe`. */
  transcribeWav(file, language = null, settings = {}) {
    return this.#wav(file, options(language, settings)).text;
  }

  /**
   * The text in timed segments: sentences or pauses for Parakeet, windows of up to 30 s
   * (40 s Canary) for the others. Arguments as for `transcribe`.
   * @returns {{start: number, end: number, text: string}[]} times in seconds from the start.
   */
  segments(samples, language = null, settings = {}) {
    return this.#samples(samples, options(language, settings, true)).segments;
  }

  /** `segments` for a 16 kHz WAV file. */
  segmentsWav(file, language = null, settings = {}) {
    return this.#wav(file, options(language, settings, true)).segments;
  }

  #samples(samples, json) {
    const floats = samples instanceof Float32Array ? samples : Float32Array.from(samples);

    return JSON.parse(text(c.transcribe(this.#live(), floats, floats.length, json)));
  }

  #wav(file, json) {
    return JSON.parse(text(c.transcribeWav(this.#live(), file, json)));
  }

  /**
   * Terms to favor while decoding (Parakeet). An empty list clears them.
   * @param {string[]} terms @returns {number} terms the model cannot spell.
   */
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

/** The options JSON of trit_speak. */
function voicing({ voice = null, speed = null, noise = null, noiseDuration = null, seed = 0, phonemes = false } = {}) {
  const json = { seed, phonemes };

  if (voice !== null) json.voice = voice;

  if (speed !== null) json.speed = speed;

  if (noise !== null) json.noise = noise;

  if (noiseDuration !== null) json.noise_duration = noiseDuration;

  return JSON.stringify(json);
}

/**
 * A voice: Kokoro-82M (24 kHz, 28 English voices; reads numbers and symbols) or a VITS voice
 * (Meta MMS-TTS, 16 kHz; reads letters only).
 */
export class Voice {
  #handle;

  /** @param {string | null} model a folder, a `trit models` name ("kokoro-82m", "mms-tts-eng"), or null for the default. */
  constructor(model = null) {
    this.#handle = c.voiceLoad(model);

    if (!this.#handle) throw new TritError(c.lastError());
  }

  /** @returns {number} the rate of the samples of `speak` (24000 Kokoro, 16000 MMS-TTS). */
  get sampleRate() {
    return c.voiceSampleRate(this.#live());
  }

  /** @returns {string} "kokoro" or "vits". */
  get kind() {
    return c.voiceKind(this.#live());
  }

  /** @returns {string[]} the voices of a Kokoro folder: af_/am_ American, bf_/bm_ British ([] for VITS). */
  get voices() {
    const list = text(c.voiceList(this.#live()));

    return list ? list.split("\n") : [];
  }

  /** @returns {string} the phonemes Kokoro speaks for `text`, one line per pass, in the accent of `voice`. */
  phonemes(input, voice = null) {
    return text(c.phonemes(this.#live(), input, voicing({ voice })));
  }

  /**
   * @param {string} input
   * @param {{voice?: string, speed?: number, noise?: number, noiseDuration?: number, seed?: number, phonemes?: boolean}} settings
   *   voice: Kokoro voice (default af_heart; "af_bella,af_sky" blends two). speed: 1 is the voice's pace,
   *   1.25 faster. noise: how much the voice varies; noiseDuration: VITS timing. One text and one seed
   *   give the same audio. phonemes: Kokoro, the text is phonemes.
   * @returns {Float32Array} mono samples in -1..1 at `sampleRate`.
   */
  speak(input, settings = {}) {
    const count = [0];
    const pointer = c.speak(this.#live(), input, voicing(settings), count);

    if (!pointer) throw new TritError(c.lastError());

    try {
      return Float32Array.from(koffi.decode(pointer, "float", Number(count[0])));
    } finally {
      c.freeSamples(pointer, count[0]);
    }
  }

  /** `speak` into a 16-bit mono WAV file. */
  speakWav(input, file, settings = {}) {
    if (c.speakWav(this.#live(), input, file, voicing(settings)) < 0) throw new TritError(c.lastError());
  }

  /**
   * `speak`, part by part: `onPart(samples)` gets each part (a sentence for VITS, a line or a
   * 510-phoneme pass for Kokoro) as soon as it is made; return false to stop.
   */
  speakStream(input, onPart, settings = {}) {
    const callback = koffi.register(
      (samples, count) => (onPart(Float32Array.from(koffi.decode(samples, "float", Number(count)))) === false ? 0 : 1),
      koffi.pointer(SamplesFn),
    );

    try {
      if (c.speakStream(this.#live(), input, voicing(settings), callback, null) < 0) throw new TritError(c.lastError());
    } finally {
      koffi.unregister(callback);
    }
  }

  close() {
    if (this.#handle) c.voiceFree(this.#handle);
    this.#handle = null;
  }

  #live() {
    if (!this.#handle) throw new TritError("voice is closed");

    return this.#handle;
  }
}
