/* Trit: speech to text and text to speech on the CPU. Runs Parakeet (TDT,
 * CTC), Canary, Whisper (tiny to large-v3, turbo, Distil-Whisper), and
 * Moonshine checkpoints from Hugging Face, the S1-mini transcript cleaner,
 * and VITS voices (Meta MMS-TTS).
 * C ABI of the trit-ffi library (trit_ffi.dll, libtrit_ffi.so, libtrit_ffi.dylib).
 *
 * Rules for every function:
 * - Strings are UTF-8 and NUL-terminated.
 * - A call that fails returns NULL or -1. trit_last_error() then gives the
 *   reason and the fix, for the calling thread.
 * - Text that a function returns is yours: free it with trit_free_text().
 *   Samples from trit_speak() are yours: free them with trit_free_samples().
 * - One engine is safe to use from several threads; calls share one worker
 *   team, so they do not finish sooner in parallel.
 *
 * Example:
 *   Trit *engine = trit_load("whisper-large-v3-turbo");   // folder, name, or NULL
 *   if (!engine) { puts(trit_last_error()); return 1; }
 *   char *text = trit_transcribe_wav(engine, "take.wav", NULL);  // NULL: detect language
 *   if (text) { puts(text); trit_free_text(text); }
 *   trit_free(engine);
 */
#ifndef TRIT_H
#define TRIT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Trit Trit;
typedef struct TritCleaner TritCleaner;
typedef struct TritVoice TritVoice;

/* The engine version, for example "0.3.0". Static; do not free. */
const char *trit_version(void);

/* The error of the last failed call on this thread, or "" after a success.
 * Valid until the next trit call on this thread; do not free. */
const char *trit_last_error(void);

/* Loads a speech model: a folder with config.json and model.safetensors,
 * or a name from `trit models` ("whisper-large-v3-turbo"). The family comes
 * from config.json. NULL reads TRIT_MODEL, then searches ./models and the
 * Hugging Face cache for Parakeet Redux. NULL on failure. */
Trit *trit_load(const char *model_dir);

/* trit_load with a precision for the weights:
 *   "auto" or NULL: int8 for layers 384 wide and more, f16 below (fastest
 *                   at the reference accuracy)
 *   "int8": int8 everywhere (least memory; narrow models lose some accuracy)
 *   "f16":  f16 weights, f32 math
 *   "f32":  f32 weights, as the reference runtimes hold them (slowest)
 * NULL on failure, also for an unknown precision. */
Trit *trit_load_with(const char *model_dir, const char *precision);

/* Returns memory freed by earlier calls to the system. Call it when the app
 * goes idle (some seconds after the last transcript), not between calls. */
void trit_release_memory(void);

/* "parakeet_tdt", "parakeet_ctc", "canary", "whisper", or "moonshine". Static. */
const char *trit_kind(const Trit *engine);

/* Frees an engine. NULL is a no-op. */
void trit_free(Trit *engine);

/* Transcribes `count` mono samples at 16 kHz in the range -1 to 1.
 * `language`: ISO 639-1 code ("en", "de"), or NULL to let the model detect
 * it. Parakeet cannot be told a language; "en" keeps it to English.
 * Canary needs it (English when NULL).
 * Returns the text, or NULL on failure. Free with trit_free_text(). */
char *trit_transcribe(const Trit *engine, const float *samples, size_t count,
                      const char *language);

/* trit_transcribe with `beam` beams for the decoder families (Whisper,
 * Moonshine, Canary). 0 is the model's default, as trit_transcribe uses:
 * 5 for Whisper and Moonshine, greedy for Canary. 1 is greedy. */
char *trit_transcribe_with(const Trit *engine, const float *samples, size_t count,
                           const char *language, int beam);

/* Transcribes a 16 kHz WAV file (16-bit PCM or 32-bit float, any channels).
 * `language` as for trit_transcribe. Returns the text, or NULL on failure.
 * Free with trit_free_text(). */
char *trit_transcribe_wav(const Trit *engine, const char *path, const char *language);

/* trit_transcribe with options, and the result as JSON:
 *   {"text": "...", "segments": [{"start": 0.0, "end": 1.8, "text": "..."}]}
 * `options`: NULL or a JSON object; every key is optional:
 *   "language":   ISO 639-1 code, as for trit_transcribe
 *   "task":       "transcribe" (default), or "translate" into English
 *                 (multilingual Whisper, Canary)
 *   "beam":       as for trit_transcribe_with
 *   "fallback":   Whisper decodes a window again at higher temperatures
 *                 when its text repeats itself (default true)
 *   "timestamps": fill "segments" (default false): sentences or pauses for
 *                 Parakeet, windows of up to 30 s (40 s Canary) for others
 * An unknown key or value fails. Example: "{\"language\": \"en\", \"timestamps\": true}".
 * Returns the JSON, or NULL on failure. Free with trit_free_text(). */
char *trit_transcribe_json(const Trit *engine, const float *samples, size_t count,
                           const char *options);

/* trit_transcribe_json for a 16 kHz WAV file, as for trit_transcribe_wav. */
char *trit_transcribe_wav_json(const Trit *engine, const char *path, const char *options);

/* Frees text from the trit_transcribe and trit_clean functions.
 * NULL is a no-op. */
void trit_free_text(char *text);

/* Terms to favor while decoding, one per line ("Kubernetes\nPostgreSQL").
 * "" clears them. Returns how many terms the model cannot spell (skipped),
 * or -1 on failure. Parakeet uses them; other families skip every term. */
int trit_set_dictionary(const Trit *engine, const char *terms);

/* Loads the transcript cleaner (Superwhisper S1-mini) from a folder; NULL
 * reads TRIT_CLEANUP_MODEL, then searches ./models/s1-mini and the Hugging
 * Face cache. NULL on failure. */
TritCleaner *trit_cleaner_load(const char *model_dir);

/* trit_cleaner_load with a precision, as for trit_load_with. */
TritCleaner *trit_cleaner_load_with(const char *model_dir, const char *precision);

/* Frees a cleaner. NULL is a no-op. */
void trit_cleaner_free(TritCleaner *cleaner);

/* Rewrites a raw transcript as clean written English: fillers removed,
 * self-corrections resolved, punctuation, casing, numbers. NULL picks the
 * default of each setting:
 *   styling:   "casual", "semi-casual", "semi-formal" (default), "formal"
 *   structure: "prose" (default), "lists"
 *   context:   "general" (default), "email"
 * Returns the text ("" for only fillers), or NULL on failure. Free with
 * trit_free_text(). */
char *trit_clean(const TritCleaner *cleaner, const char *text, const char *styling,
                 const char *structure, const char *context);

/* Loads a voice (text to speech): Kokoro-82M (24 kHz, 28 English voices)
 * or a VITS folder (Meta MMS-TTS, 16 kHz), or a name from `trit models`
 * ("kokoro-82m", "mms-tts-eng"). NULL reads TRIT_VOICE, then searches
 * ./models and the Hugging Face cache for kokoro-82m, then mms-tts-eng.
 * NULL on failure. */
TritVoice *trit_voice_load(const char *model_dir);

/* Frees a voice. NULL is a no-op. */
void trit_voice_free(TritVoice *voice);

/* The rate of the samples of trit_speak (24000 Kokoro, 16000 MMS-TTS); -1 for NULL. */
int trit_voice_sample_rate(const TritVoice *voice);

/* "kokoro" or "vits"; NULL for NULL. Static; do not free. */
const char *trit_voice_kind(const TritVoice *voice);

/* The voices of a Kokoro folder, one name per line ("" for VITS): af_ and
 * am_ American, bf_ and bm_ British. NULL on failure. Free with trit_free_text(). */
char *trit_voice_list(const TritVoice *voice);

/* The phonemes a Kokoro voice speaks for `text`, one line per pass, with
 * the accent of the options' voice; `options` as for trit_speak. NULL on
 * failure. Free with trit_free_text(). */
char *trit_phonemes(const TritVoice *voice, const char *text, const char *options);

/* Speaks `text`: mono floats in -1 to 1 at trit_voice_sample_rate(), their
 * number in *count. Kokoro reads English text (numbers and symbols too);
 * a VITS voice reads letters only.
 * `options`: NULL or a JSON object; every key is optional:
 *   "voice":          Kokoro: a name of trit_voice_list (default "af_heart");
 *                     "af_bella,af_sky" blends two
 *   "phonemes":       Kokoro: true when the text is phonemes, one pass per line
 *   "speed":          speaking rate; 1 is the voice's pace, 1.25 faster
 *   "noise":          how much the voice varies (VITS 0.667, Kokoro 1)
 *   "noise_duration": VITS: how much the timing varies (0.8)
 *   "seed":           0 (default) or more; one text and one seed give the
 *                     same audio
 * Returns the samples, or NULL on failure. Free with trit_free_samples(). */
float *trit_speak(const TritVoice *voice, const char *text, const char *options,
                  size_t *count);

/* The callback of trit_speak_stream: the samples of one part (valid during
 * the call only) and your pointer. Return 1 to go on, 0 to stop. */
typedef int (*TritSamplesFn)(const float *samples, size_t count, void *user_data);

/* trit_speak part by part: `callback` gets each part (a sentence for VITS,
 * a line or a 510-phoneme pass for Kokoro) as soon as it is made, so play
 * can start before the rest is spoken. Returns 0, or -1 on failure. */
int trit_speak_stream(const TritVoice *voice, const char *text, const char *options,
                      TritSamplesFn callback, void *user_data);

/* trit_speak into a 16-bit mono WAV file at `path`. Returns 0, or -1. */
int trit_speak_wav(const TritVoice *voice, const char *text, const char *path,
                   const char *options);

/* Frees the samples of trit_speak; `count` is the count it gave. NULL is a no-op. */
void trit_free_samples(float *samples, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* TRIT_H */
