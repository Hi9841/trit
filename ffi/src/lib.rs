//! The C ABI of `include/trit.h`, for every language with a C FFI.
//! A function that fails returns NULL or -1 and sets the error text of the
//! calling thread, which `trit_last_error` returns. No panic crosses the ABI.

use std::cell::RefCell;
use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::OnceLock;
use trit::{BiasWeights, Cleaner, LoadOptions, Options, Precision, Trit};

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

fn owned_text(text: String) -> *mut c_char {
    CString::new(text.replace('\0', " "))
        .unwrap_or_default()
        .into_raw()
}

/// The engine version, for example "0.1.0". Static; do not free.
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
        if samples.is_null() && count > 0 {
            return Err("samples is NULL".into());
        }
        let samples = if count == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(samples, count)
        };
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

/// Frees text from `trit_transcribe`, `trit_transcribe_wav`, or `trit_clean`.
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
