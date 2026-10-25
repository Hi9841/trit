//! The C ABI of `include/trit.h`, for every language with a C FFI.
//! A function that fails returns NULL or -1 and sets the error text of the
//! calling thread, which `trit_last_error` returns. No panic crosses the ABI.

use std::cell::RefCell;
use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::OnceLock;
use trit::{BiasWeights, Cleaner, LoadOptions, Options, Precision, Task, Trit, Voice, Voicing};

// The library's own allocations: the same allocator as the trit CLI and
// TDT. A host's allocator only serves the host; text crosses the ABI as
// CString and comes back through trit_free_text, so it never mixes heaps.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(message: String) {
    let text = CString::new(message.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|slot| *slot.borrow_mut() = text);
}

/// Runs `body`, turns an `Err` or a panic into the thread's error, and
/// returns `failed` in that case.
fn guard<T>(failed: T, body: impl FnOnce() -> Result<T, String>) -> T {
    set_error(String::new());
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(value)) => value,
        Ok(Err(message)) => {
            set_error(message);
            failed
        }
        Err(_) => {
            set_error("internal error in trit (panic)".into());
            failed
        }
    }
}

/// # Safety
/// `text` is NULL or a NUL-terminated string.
unsafe fn text_arg<'a>(text: *const c_char, name: &str) -> Result<&'a str, String> {
    if text.is_null() {
        return Err(format!("{name} is NULL"));
    }
    CStr::from_ptr(text)
        .to_str()
        .map_err(|_| format!("{name} is not UTF-8"))
}

/// # Safety
/// `engine` is NULL or a pointer from `trit_load` that was not freed.
unsafe fn engine_arg<'a>(engine: *const Trit) -> Result<&'a Trit, String> {
    engine
        .as_ref()
        .ok_or_else(|| "engine is NULL; call trit_load first".to_string())
}

/// # Safety
/// `samples` points to `count` floats, or is NULL when `count` is 0.
unsafe fn samples_arg<'a>(samples: *const f32, count: usize) -> Result<&'a [f32], String> {
    if count == 0 {
        return Ok(&[]);
    }
    if samples.is_null() {
        return Err("samples is NULL".into());
    }
    Ok(std::slice::from_raw_parts(samples, count))
}

fn owned_text(text: String) -> *mut c_char {
    CString::new(text.replace('\0', " "))
        .unwrap_or_default()
        .into_raw()
}

/// The engine version, for example "0.2.0". Static; do not free.
#[no_mangle]
pub extern "C" fn trit_version() -> *const c_char {
    static VERSION: OnceLock<CString> = OnceLock::new();
    VERSION
        .get_or_init(|| CString::new(trit::VERSION).unwrap_or_default())
        .as_ptr()
}

/// The error of the last failed call on this thread, or "" after a success.
/// Valid until the next trit call on this thread; do not free.
#[no_mangle]
pub extern "C" fn trit_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// Loads a model folder, or a model by its `trit models` name. The family
/// (Parakeet, Canary, Whisper, Moonshine) comes from its `config.json`. NULL
/// `model_dir` reads `TRIT_MODEL`, then searches the default places.
/// Returns NULL on failure. Free with `trit_free`.
///
/// # Safety
/// `model_dir` is NULL or a NUL-terminated UTF-8 path.
#[no_mangle]
pub unsafe extern "C" fn trit_load(model_dir: *const c_char) -> *mut Trit {
    trit_load_with(model_dir, std::ptr::null())
}

/// `trit_load` with a precision: "auto" (NULL), "int8", "f16", or "f32".
///
/// # Safety
/// `model_dir` and `precision` are NULL or NUL-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn trit_load_with(
    model_dir: *const c_char,
    precision: *const c_char,
) -> *mut Trit {
    guard(std::ptr::null_mut(), || {
        let explicit = if model_dir.is_null() {
            None
        } else {
            Some(Path::new(text_arg(model_dir, "model_dir")?))
        };
        let options = load_options(precision)?;
        let dir = trit::find_model(explicit).ok_or_else(|| trit::missing_model(explicit))?;
        let engine = Trit::load_with(&dir, &options)?;
        trit::release_memory();
        Ok(Box::into_raw(Box::new(engine)))
    })
}

/// # Safety
/// `precision` is NULL or a NUL-terminated UTF-8 string.
unsafe fn load_options(precision: *const c_char) -> Result<LoadOptions, String> {
    let precision = if precision.is_null() {
        Precision::Auto
    } else {
        Precision::parse(text_arg(precision, "precision")?)?
    };
    Ok(LoadOptions { precision })
}

/// Returns the memory freed by earlier calls to the system. Call it when
/// the app goes idle (some seconds after the last transcript), not between
/// calls.
#[no_mangle]
pub extern "C" fn trit_release_memory() {
    trit::release_memory();
}

/// Frees an engine from `trit_load`. NULL is a no-op.
///
/// # Safety
/// `engine` is NULL or a pointer from `trit_load` that was not freed.
#[no_mangle]
pub unsafe extern "C" fn trit_free(engine: *mut Trit) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}

/// The model family: "parakeet_tdt", "parakeet_ctc", "canary", "whisper",
/// or "moonshine"; NULL for a NULL engine. Static; do not free.
///
/// # Safety
/// `engine` is NULL or a pointer from `trit_load` that was not freed.
#[no_mangle]
pub unsafe extern "C" fn trit_kind(engine: *const Trit) -> *const c_char {
    let Some(engine) = engine.as_ref() else {
        return std::ptr::null();
    };
    static KINDS: OnceLock<Vec<(&'static str, CString)>> = OnceLock::new();
    let kinds = KINDS.get_or_init(|| {
        trit::MODEL_TYPES
            .iter()
            .map(|k| (*k, CString::new(*k).unwrap_or_default()))
            .collect()
    });
    kinds
        .iter()
        .find(|(k, _)| *k == engine.kind())
        .map_or(std::ptr::null(), |(_, c)| c.as_ptr())
}

/// Options from a language code; NULL or "" lets the model detect it.
unsafe fn options(language: *const c_char) -> Result<Options, String> {
    if language.is_null() {
        return Ok(Options::default());
    }
    let code = text_arg(language, "language")?.trim().to_lowercase();
    Ok(Options {
        language: (!code.is_empty()).then_some(code),
        ..Options::default()
    })
}

/// Transcribes `count` mono samples at 16 kHz, in the range -1 to 1.
/// `language` is an ISO 639-1 code ("en", "de"), or NULL to let the model
/// detect it. Returns UTF-8 text (NULL on failure). Free with `trit_free_text`.
///
/// # Safety
/// `engine` is from `trit_load`; `samples` points to `count` floats;
/// `language` is NULL or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn trit_transcribe(
    engine: *const Trit,
    samples: *const f32,
    count: usize,
    language: *const c_char,
) -> *mut c_char {
    trit_transcribe_with(engine, samples, count, language, 0)
}

/// `trit_transcribe` with `beam` beams for the decoder families (Whisper,
/// Moonshine, Canary): 0 is the model's default (5 for Whisper and
/// Moonshine, greedy for Canary), 1 is greedy.
///
/// # Safety
/// As for `trit_transcribe`.
#[no_mangle]
pub unsafe extern "C" fn trit_transcribe_with(
    engine: *const Trit,
    samples: *const f32,
    count: usize,
    language: *const c_char,
    beam: c_int,
) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let engine = engine_arg(engine)?;
        let mut options = options(language)?;
        options.beam = usize::try_from(beam).map_err(|_| format!("beam {beam} is below 0"))?;
        let samples = samples_arg(samples, count)?;
        Ok(owned_text(engine.transcribe_with(samples, &options)))
    })
}

/// Transcribes a 16 kHz WAV file; `language` as for `trit_transcribe`.
/// Returns UTF-8 text (NULL on failure). Free with `trit_free_text`.
///
/// # Safety
/// `engine` is from `trit_load`; `path` is a NUL-terminated UTF-8 path;
/// `language` is NULL or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn trit_transcribe_wav(
    engine: *const Trit,
    path: *const c_char,
    language: *const c_char,
) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let engine = engine_arg(engine)?;
        let options = options(language)?;
        let samples = trit::read_wav(Path::new(text_arg(path, "path")?))?;
        Ok(owned_text(engine.transcribe_with(&samples, &options)))
    })
}

const OPTION_KEYS: &str = "language, task, beam, fallback, timestamps";

/// Options from a JSON object, and whether to time segments. NULL or "" is
/// the default; every key is optional.
///
/// # Safety
/// `options` is NULL or a NUL-terminated string.
unsafe fn json_options(options: *const c_char) -> Result<(Options, bool), String> {
    let mut parsed = Options::default();
    let mut timestamps = false;
    if options.is_null() {
        return Ok((parsed, timestamps));
    }
    let text = text_arg(options, "options")?.trim();
    if text.is_empty() {
        return Ok((parsed, timestamps));
    }
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("options is not JSON: {e}"))?;
    let object = value
        .as_object()
        .ok_or("options is not a JSON object; example: {\"language\": \"en\"}")?;
    for (key, value) in object {
        let wrong = || format!("options.{key} cannot be {value}");
        match key.as_str() {
            "language" => {
                parsed.language = match value {
                    serde_json::Value::Null => None,
                    serde_json::Value::String(code) => {
                        Some(code.trim().to_lowercase()).filter(|code| !code.is_empty())
                    }
                    _ => return Err(wrong()),
                }
            }
            "task" => {
                parsed.task = value.as_str().and_then(Task::parse).ok_or_else(|| {
                    format!("{}; valid tasks: {}", wrong(), Task::NAMES.join(", "))
                })?
            }
            "beam" => {
                parsed.beam = value
                    .as_u64()
                    .and_then(|beam| usize::try_from(beam).ok())
                    .ok_or_else(wrong)?
            }
            "fallback" => parsed.fallback = Some(value.as_bool().ok_or_else(wrong)?),
            "timestamps" => timestamps = value.as_bool().ok_or_else(wrong)?,
            _ => {
                return Err(format!(
                    "unknown option {key}; valid options: {OPTION_KEYS}"
                ))
            }
        }
    }
    Ok((parsed, timestamps))
}

/// `{"text": ..., "segments": [{"start", "end", "text"}]}`, the shape of one
/// result of `trit transcribe --format json`. With timestamps, the text is
/// the segments joined.
fn json_result(engine: &Trit, samples: &[f32], options: &Options, timestamps: bool) -> String {
    let (text, segments) = if timestamps {
        let segments = engine.segments(samples, options);
        let texts: Vec<&str> = segments.iter().map(|s| s.text.as_str()).collect();
        (texts.join(" "), segments)
    } else {
        (engine.transcribe_with(samples, options), Vec::new())
    };
    serde_json::json!({
        "text": text,
        "segments": segments.iter().map(|s| serde_json::json!({
            "start": hundredths(s.start),
            "end": hundredths(s.end),
            "text": s.text,
        })).collect::<Vec<_>>(),
    })
    .to_string()
}

/// Transcribes `count` mono samples at 16 kHz with options, and returns
/// JSON: `{"text": "...", "segments": [{"start": 0.0, "end": 1.8, "text": "..."}]}`.
/// `options` is NULL or a JSON object with any of: `language` (ISO 639-1
/// code), `task` (`"transcribe"` or `"translate"`), `beam` (as for
/// `trit_transcribe_with`), `fallback` (Whisper temperature fallback,
/// default true), `timestamps` (fill `segments`, default false). An unknown
/// key or value fails. Free the result with `trit_free_text`.
///
/// # Safety
/// As for `trit_transcribe`; `options` is NULL or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn trit_transcribe_json(
    engine: *const Trit,
    samples: *const f32,
    count: usize,
    options: *const c_char,
) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let engine = engine_arg(engine)?;
        let (options, timestamps) = json_options(options)?;
        let samples = samples_arg(samples, count)?;
        Ok(owned_text(json_result(
            engine, samples, &options, timestamps,
        )))
    })
}

/// `trit_transcribe_json` for a 16 kHz WAV file.
///
/// # Safety
/// `engine` is from `trit_load`; `path` is a NUL-terminated UTF-8 path;
/// `options` is NULL or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn trit_transcribe_wav_json(
    engine: *const Trit,
    path: *const c_char,
    options: *const c_char,
) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let engine = engine_arg(engine)?;
        let (options, timestamps) = json_options(options)?;
        let samples = trit::read_wav(Path::new(text_arg(path, "path")?))?;
        Ok(owned_text(json_result(
            engine, &samples, &options, timestamps,
        )))
    })
}

/// Frees text from `trit_transcribe*`, `trit_transcribe_wav*`, or `trit_clean`.
/// NULL is a no-op.
///
/// # Safety
/// `text` is NULL or a pointer from a trit function that returns owned text.
#[no_mangle]
pub unsafe extern "C" fn trit_free_text(text: *mut c_char) {
    if !text.is_null() {
        drop(CString::from_raw(text));
    }
}

/// Sets the terms to favor while decoding, one per line ("Kubernetes\nPostgreSQL").
/// An empty string clears them. Returns the number of terms the model cannot
/// spell (they are skipped), or -1 on failure.
///
/// # Safety
/// `engine` is from `trit_load`; `terms` is a NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn trit_set_dictionary(engine: *const Trit, terms: *const c_char) -> c_int {
    guard(-1, || {
        let engine = engine_arg(engine)?;
        let terms: Vec<String> = text_arg(terms, "terms")?
            .lines()
            .map(str::to_string)
            .collect();
        let skipped = engine.set_dictionary(&terms, BiasWeights::default());
        Ok(skipped.len() as c_int)
    })
}

/// Loads the cleanup model (Superwhisper S1-mini) from a folder; NULL reads
/// `TRIT_CLEANUP_MODEL`, then searches the default places. Returns NULL on
/// failure. Free with `trit_cleaner_free`.
///
/// # Safety
/// `model_dir` is NULL or a NUL-terminated UTF-8 path.
#[no_mangle]
pub unsafe extern "C" fn trit_cleaner_load(model_dir: *const c_char) -> *mut Cleaner {
    trit_cleaner_load_with(model_dir, std::ptr::null())
}

/// `trit_cleaner_load` with a precision, as for `trit_load_with`.
///
/// # Safety
/// `model_dir` and `precision` are NULL or NUL-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn trit_cleaner_load_with(
    model_dir: *const c_char,
    precision: *const c_char,
) -> *mut Cleaner {
    guard(std::ptr::null_mut(), || {
        let options = load_options(precision)?;
        let explicit = if model_dir.is_null() {
            None
        } else {
            Some(Path::new(text_arg(model_dir, "model_dir")?))
        };
        let dir = trit::find_cleanup_model(explicit).ok_or_else(|| {
            format!(
                "no cleanup model found; get it: hf download {} --local-dir models/s1-mini",
                trit::CLEANUP_REPO
            )
        })?;
        let cleaner = Cleaner::load_with(&dir, &options)?;
        trit::release_memory();
        Ok(Box::into_raw(Box::new(cleaner)))
    })
}

/// Frees a cleaner from `trit_cleaner_load`. NULL is a no-op.
///
/// # Safety
/// `cleaner` is NULL or a pointer from `trit_cleaner_load` that was not freed.
#[no_mangle]
pub unsafe extern "C" fn trit_cleaner_free(cleaner: *mut Cleaner) {
    if !cleaner.is_null() {
        drop(Box::from_raw(cleaner));
    }
}

/// Rewrites a raw transcript as clean written English. `styling` is
/// "casual", "semi-casual", "semi-formal", or "formal"; `structure` is
/// "prose" or "lists"; `context` is "general" or "email". NULL picks the
/// default (semi-formal, prose, general). Returns UTF-8 text, "" when the
/// input was only fillers (NULL on failure). Free with `trit_free_text`.
///
/// # Safety
/// `cleaner` is from `trit_cleaner_load`; the strings are NULL (except
/// `text`) or NUL-terminated UTF-8.
#[no_mangle]
pub unsafe extern "C" fn trit_clean(
    cleaner: *const Cleaner,
    text: *const c_char,
    styling: *const c_char,
    structure: *const c_char,
    context: *const c_char,
) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let cleaner = cleaner
            .as_ref()
            .ok_or("cleaner is NULL; call trit_cleaner_load first")?;
        let text = text_arg(text, "text")?;
        let pick =
            |value: *const c_char, name: &str, valid: &[&str]| -> Result<Option<String>, String> {
                if value.is_null() {
                    return Ok(None);
                }
                let value = text_arg(value, name)?;
                if valid.contains(&value) {
                    Ok(Some(value.to_string()))
                } else {
                    Err(format!(
                        "unknown {name} `{value}`; valid: {}",
                        valid.join(", ")
                    ))
                }
            };
        let mut style = trit::Style::default();
        if let Some(v) = pick(styling, "styling", &trit::Styling::NAMES)? {
            style.styling = trit::Styling::parse(&v).unwrap_or_default();
        }
        if let Some(v) = pick(structure, "structure", &trit::Structure::NAMES)? {
            style.structure = trit::Structure::parse(&v).unwrap_or_default();
        }
        if let Some(v) = pick(context, "context", &trit::Context::NAMES)? {
            style.context = trit::Context::parse(&v).unwrap_or_default();
        }
        Ok(owned_text(cleaner.clean(text, &style)))
    })
}

/// Loads a voice (text to speech: a VITS folder such as Meta MMS-TTS), or a
/// voice by its `trit models` name; NULL reads `TRIT_VOICE`, then searches
/// the default places for `mms-tts-eng`. Returns NULL on failure. Free with
/// `trit_voice_free`.
///
/// # Safety
/// `model_dir` is NULL or a NUL-terminated UTF-8 path.
#[no_mangle]
pub unsafe extern "C" fn trit_voice_load(model_dir: *const c_char) -> *mut Voice {
    guard(std::ptr::null_mut(), || {
        let explicit = if model_dir.is_null() {
            None
        } else {
            Some(Path::new(text_arg(model_dir, "model_dir")?))
        };
        let dir = trit::find_voice(explicit).ok_or_else(|| trit::missing_voice(explicit))?;
        let voice = Voice::load(&dir)?;
        trit::release_memory();
        Ok(Box::into_raw(Box::new(voice)))
    })
}

/// Frees a voice from `trit_voice_load`. NULL is a no-op.
///
/// # Safety
/// `voice` is NULL or a pointer from `trit_voice_load` that was not freed.
#[no_mangle]
pub unsafe extern "C" fn trit_voice_free(voice: *mut Voice) {
    if !voice.is_null() {
        drop(Box::from_raw(voice));
    }
}

/// # Safety
/// `voice` is NULL or a pointer from `trit_voice_load` that was not freed.
unsafe fn voice_arg<'a>(voice: *const Voice) -> Result<&'a Voice, String> {
    voice
        .as_ref()
        .ok_or_else(|| "voice is NULL; call trit_voice_load first".to_string())
}

/// The rate of the samples that `trit_speak` returns (16000 for MMS-TTS),
/// or -1 for a NULL voice.
///
/// # Safety
/// `voice` is NULL or a pointer from `trit_voice_load` that was not freed.
#[no_mangle]
pub unsafe extern "C" fn trit_voice_sample_rate(voice: *const Voice) -> c_int {
    guard(-1, || Ok(voice_arg(voice)?.sample_rate() as c_int))
}

const VOICE_KEYS: &str = "speed, noise, noise_duration, seed";

/// Voicing from a JSON object; NULL or "" is the voice's own settings.
///
/// # Safety
/// `options` is NULL or a NUL-terminated string.
unsafe fn json_voicing(options: *const c_char) -> Result<Voicing, String> {
    let mut voicing = Voicing::default();
    if options.is_null() {
        return Ok(voicing);
    }
    let text = text_arg(options, "options")?.trim();
    if text.is_empty() {
        return Ok(voicing);
    }
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("options is not JSON: {e}"))?;
    let object = value
        .as_object()
        .ok_or("options is not a JSON object; example: {\"speed\": 1.1}")?;
    for (key, value) in object {
        let wrong = || format!("options.{key} cannot be {value}");
        let number = || value.as_f64().map(|v| v as f32).ok_or_else(wrong);
        match key.as_str() {
            "speed" => voicing.speed = Some(number()?),
            "noise" => voicing.noise = Some(number()?),
            "noise_duration" => voicing.noise_duration = Some(number()?),
            "seed" => voicing.seed = value.as_u64().ok_or_else(wrong)?,
            _ => return Err(format!("unknown option {key}; valid options: {VOICE_KEYS}")),
        }
    }
    Ok(voicing)
}

/// Speaks `text`: mono samples at `trit_voice_sample_rate`, in -1 to 1;
/// `*count` gets their number. `options` is NULL or a JSON object with any
/// of: `speed` (1 is the voice's pace), `noise` (how much the voice
/// varies), `noise_duration` (how much the timing varies), `seed` (one text
/// and one seed give the same audio; default 0). Returns NULL on failure.
/// Free the samples with `trit_free_samples`.
///
/// # Safety
/// `voice` is from `trit_voice_load`; `text` is a NUL-terminated UTF-8
/// string; `options` is NULL or a NUL-terminated string; `count` points to
/// a `size_t`.
#[no_mangle]
pub unsafe extern "C" fn trit_speak(
    voice: *const Voice,
    text: *const c_char,
    options: *const c_char,
    count: *mut usize,
) -> *mut f32 {
    guard(std::ptr::null_mut(), || {
        let voice = voice_arg(voice)?;
        let count = count.as_mut().ok_or("count is NULL")?;
        *count = 0;
        let samples = voice
            .speak_with(text_arg(text, "text")?, &json_voicing(options)?)?
            .into_boxed_slice();
        *count = samples.len();
        Ok(Box::into_raw(samples).cast::<f32>())
    })
}

/// Speaks `text` into a 16-bit mono WAV file at `path`; `options` as for
/// `trit_speak`. Returns 0, or -1 on failure.
///
/// # Safety
/// `voice` is from `trit_voice_load`; `text` and `path` are NUL-terminated
/// UTF-8 strings; `options` is NULL or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn trit_speak_wav(
    voice: *const Voice,
    text: *const c_char,
    path: *const c_char,
    options: *const c_char,
) -> c_int {
    guard(-1, || {
        let voice = voice_arg(voice)?;
        let samples = voice.speak_with(text_arg(text, "text")?, &json_voicing(options)?)?;
        trit::write_wav(
            Path::new(text_arg(path, "path")?),
            &samples,
            voice.sample_rate(),
        )?;
        Ok(0)
    })
}

/// Frees `count` samples from `trit_speak`. NULL is a no-op.
///
/// # Safety
/// `samples` is NULL or a pointer from `trit_speak` that was not freed, and
/// `count` is the count that call gave.
#[no_mangle]
pub unsafe extern "C" fn trit_free_samples(samples: *mut f32, count: usize) {
    if !samples.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            samples, count,
        )));
    }
}

/// Seconds to 0.01 s, so JSON shows 3.28 and not the f32 as f64 (3.2799999713897705).
fn hundredths(seconds: f32) -> f64 {
    (f64::from(seconds) * 100.0).round() / 100.0
}
