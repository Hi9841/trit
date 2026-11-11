//! The C ABI of `include/trit.h`, for every language with a C FFI.
//! A function that fails returns NULL or -1 and sets the error text of the
//! calling thread, which `trit_last_error` returns. No panic crosses the ABI.

use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::{Arc, Mutex, Once, OnceLock};
use trit::{
    BiasWeights, Cleaner, Device, LoadOptions, Options, Precision, Task, Trit, Voice, Voicing,
};

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
    keep_worker_panics();
    set_error(String::new());
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(value)) => value,
        Ok(Err(message)) => {
            set_error(message);
            failed
        }
        Err(panic) => {
            set_error(panic_text(panic.as_ref()));
            failed
        }
    }
}

/// The message and the place of the last panic on a worker thread of the
/// engine. The panic that reaches the caller only says that a worker failed.
static WORKER_PANIC: Mutex<Option<String>> = Mutex::new(None);

/// Keeps the message of a worker's panic for `panic_text`. The engine does
/// not carry it to the caller (that costs time in its task loop), and a
/// program with no console never sees what the panic hook prints. This
/// library has its own copy of `std`, so the hook sees only its own panics;
/// it calls the hook that was there before.
fn keep_worker_panics() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let before = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let worker = std::thread::current()
                .name()
                .is_some_and(|name| name.starts_with("trit-speech-"));
            if worker {
                let mut kept = WORKER_PANIC.lock().unwrap_or_else(|p| p.into_inner());
                *kept = Some(info.to_string());
            }
            before(info);
        }));
    });
}

/// The error text of a panic, with its message when it has one, and with
/// the message of the worker when the panic came from a worker thread.
fn panic_text(panic: &(dyn std::any::Any + Send)) -> String {
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str));
    let worker = WORKER_PANIC
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .take()
        .filter(|_| message.is_some_and(|m| m.contains("worker thread panicked")));
    match (message, worker) {
        (Some(message), Some(worker)) => {
            format!("internal error in trit (panic): {message}: {worker}")
        }
        (Some(message), None) => format!("internal error in trit (panic): {message}"),
        (None, _) => "internal error in trit (panic)".into(),
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

/// # Safety
/// `path` is NULL or a NUL-terminated string.
unsafe fn path_arg<'a>(path: *const c_char, name: &str) -> Result<Option<&'a Path>, String> {
    if path.is_null() {
        return Ok(None);
    }
    text_arg(path, name).map(|path| Some(Path::new(path)))
}

/// # Safety
/// `options` is NULL or a NUL-terminated string.
unsafe fn json_object(
    options: *const c_char,
    example: &str,
) -> Result<Option<serde_json::Map<String, serde_json::Value>>, String> {
    if options.is_null() {
        return Ok(None);
    }
    let text = text_arg(options, "options")?.trim();
    if text.is_empty() {
        return Ok(None);
    }
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("options is not JSON: {e}"))?;
    match value {
        serde_json::Value::Object(object) => Ok(Some(object)),
        _ => Err(format!("options is not a JSON object; example: {example}")),
    }
}

/// `text` as a C string that lives as long as the library.
fn static_text(text: &'static str) -> *const c_char {
    static TEXTS: Mutex<Vec<(&str, CString)>> = Mutex::new(Vec::new());
    let mut texts = TEXTS.lock().unwrap_or_else(|poison| poison.into_inner());
    if let Some((_, known)) = texts.iter().find(|(known, _)| *known == text) {
        return known.as_ptr();
    }
    let owned = CString::new(text).unwrap_or_default();
    let pointer = owned.as_ptr();
    texts.push((text, owned));
    pointer
}

fn owned_text(text: String) -> *mut c_char {
    CString::new(text.replace('\0', " "))
        .unwrap_or_default()
        .into_raw()
}

/// The engine version, for example "0.3.0". Static; do not free.
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

/// `trit_load` with a precision: "auto" (NULL), "int8", "f16", "f32", or "q8_0".
/// The device stays "cpu".
///
/// # Safety
/// `model_dir` and `precision` are NULL or NUL-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn trit_load_with(
    model_dir: *const c_char,
    precision: *const c_char,
) -> *mut Trit {
    trit_load_ex(model_dir, precision, std::ptr::null())
}

/// `trit_load_with` with a device: "cpu" (NULL), "gpu", or "auto".
/// `auto` uses the GPU, or the CPU when the computer has no adapter.
/// `TRIT_DEVICE` overrides the device. Returns NULL on failure.
///
/// # Safety
/// `model_dir`, `precision`, and `device` are NULL or NUL-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn trit_load_ex(
    model_dir: *const c_char,
    precision: *const c_char,
    device: *const c_char,
) -> *mut Trit {
    guard(std::ptr::null_mut(), || {
        let explicit = path_arg(model_dir, "model_dir")?;
        let options = load_options(precision, device)?;
        let dir = trit::find_model(explicit).ok_or_else(|| trit::missing_model(explicit))?;
        let engine = Trit::load_with(&dir, &options)?;
        trit::release_memory();
        Ok(Box::into_raw(Box::new(engine)))
    })
}

const LOAD_KEYS: &str = "precision, device, cache";

/// `trit_load` with options as a JSON object; NULL or "" is the default.
/// Every key is optional: `precision` and `device` (as for `trit_load_ex`),
/// and `cache` (true keeps the converted weights in `<model>/.trit/` and
/// maps that file, so they are file memory and not private memory; the
/// folder must be on a local disk). An unknown key or value fails.
/// Returns NULL on failure. Free with `trit_free`.
///
/// # Safety
/// `model_dir` and `options` are NULL or NUL-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn trit_load_json(
    model_dir: *const c_char,
    options: *const c_char,
) -> *mut Trit {
    guard(std::ptr::null_mut(), || {
        let explicit = path_arg(model_dir, "model_dir")?;
        let mut load = LoadOptions::default();
        if let Some(object) = json_object(options, "{\"cache\": true}")? {
            for (key, value) in &object {
                let wrong = || format!("options.{key} cannot be {value}");
                match key.as_str() {
                    "precision" => {
                        load.precision = Precision::parse(value.as_str().ok_or_else(wrong)?)?
                    }
                    "device" => load.device = Device::parse(value.as_str().ok_or_else(wrong)?)?,
                    "cache" => load.cache = value.as_bool().ok_or_else(wrong)?,
                    _ => return Err(format!("unknown option {key}; valid options: {LOAD_KEYS}")),
                }
            }
        }
        let dir = trit::find_model(explicit).ok_or_else(|| trit::missing_model(explicit))?;
        let engine = Trit::load_with(&dir, &load)?;
        trit::release_memory();
        Ok(Box::into_raw(Box::new(engine)))
    })
}

/// The function of `trit_set_log`: one message (valid during the call
/// only) and the caller's pointer.
pub type TritLogFn = extern "C" fn(message: *const c_char, user_data: *mut c_void);

/// Sends the messages of the engine (a GPU failure and the change to the
/// CPU) to `callback`; NULL sends them to stderr again. The callback can
/// run on any thread, and must not call Trit.
#[no_mangle]
pub extern "C" fn trit_set_log(callback: Option<TritLogFn>, user_data: *mut c_void) {
    // The pointer goes to other threads as a number: the caller says that
    // its function may use it from any thread.
    let data = user_data as usize;
    trit::set_log(callback.map(|callback| -> Arc<dyn Fn(&str) + Send + Sync> {
        Arc::new(move |message: &str| {
            if let Ok(text) = CString::new(message.replace('\0', " ")) {
                callback(text.as_ptr(), data as *mut c_void);
            }
        })
    }));
}

/// The calls that run now, by engine: `trit_cancel` sets their flags.
static RUNNING: Mutex<Vec<(usize, trit::Cancel)>> = Mutex::new(Vec::new());

/// One call on an engine that `trit_cancel` can stop, until the value drops.
struct Running(usize, trit::Cancel);

impl Running {
    /// Gives `options` a new cancel flag and lists the call under `engine`.
    fn start(engine: &Trit, options: &mut Options) -> Self {
        let cancel = trit::Cancel::new();
        options.cancel = Some(cancel.clone());
        let key = std::ptr::from_ref(engine) as usize;
        let mut running = RUNNING.lock().unwrap_or_else(|p| p.into_inner());
        running.push((key, cancel.clone()));
        Self(key, cancel)
    }

    /// `value`, or the error of a call that `trit_cancel` stopped: its text
    /// is empty, and must not look like silence.
    fn finish<T>(self, value: T) -> Result<T, String> {
        match self.1.is_cancelled() {
            true => Err("the call was cancelled (trit_cancel)".into()),
            false => Ok(value),
        }
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        let mut running = RUNNING.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(at) = running
            .iter()
            .position(|(key, cancel)| *key == self.0 && *cancel == self.1)
        {
            running.swap_remove(at);
        }
    }
}

/// Stops the calls that run on `engine` now, on other threads. Each of them
/// returns NULL, and its `trit_last_error` says that it was cancelled.
/// Parakeet stops in a part of a second; the other families at the end of
/// their window (30 s to 2 minutes of audio). A call that starts later is
/// not stopped. Returns the number of calls it stopped, or -1 for NULL.
///
/// # Safety
/// `engine` is NULL or a pointer from `trit_load` that was not freed.
#[no_mangle]
pub unsafe extern "C" fn trit_cancel(engine: *const Trit) -> c_int {
    guard(-1, || {
        let key = std::ptr::from_ref(engine_arg(engine)?) as usize;
        let running = RUNNING.lock().unwrap_or_else(|p| p.into_inner());
        let mine = running.iter().filter(|(owner, _)| *owner == key);
        Ok(mine.map(|(_, cancel)| cancel.cancel()).count() as c_int)
    })
}

/// The GPU adapter, for example "NVIDIA GeForce RTX 4090, Dx12, f16".
/// NULL when there is no adapter. `trit_last_error` then gives the reason.
/// Free the text with `trit_free_text`.
#[no_mangle]
pub extern "C" fn trit_gpu_info() -> *mut c_char {
    guard(std::ptr::null_mut(), || trit::gpu_adapter().map(owned_text))
}

/// # Safety
/// `precision` and `device` are NULL or NUL-terminated UTF-8 strings.
unsafe fn load_options(
    precision: *const c_char,
    device: *const c_char,
) -> Result<LoadOptions, String> {
    let mut options = LoadOptions::default();
    if !precision.is_null() {
        options.precision = Precision::parse(text_arg(precision, "precision")?)?;
    }
    if !device.is_null() {
        options.device = Device::parse(text_arg(device, "device")?)?;
    }
    Ok(options)
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

/// The `model_type` of the model, one of `trit models`' families; NULL for
/// a NULL engine. Static; do not free.
///
/// # Safety
/// `engine` is NULL or a pointer from `trit_load` that was not freed.
#[no_mangle]
pub unsafe extern "C" fn trit_kind(engine: *const Trit) -> *const c_char {
    engine
        .as_ref()
        .map_or(std::ptr::null(), |engine| static_text(engine.kind()))
}

/// Options from a language code; NULL or "" lets the model detect it.
unsafe fn options(language: *const c_char) -> Result<Options, String> {
    if language.is_null() {
        return Ok(Options::default());
    }
    Ok(Options {
        language: trit::language_code(text_arg(language, "language")?),
        ..Options::default()
    })
}

/// Transcribes `count` mono samples at 16 kHz, in the range -1 to 1. For
/// samples at another rate, use `trit_transcribe_json` with `sample_rate`.
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
        engine.check(&options)?;
        let samples = samples_arg(samples, count)?;
        let running = Running::start(engine, &mut options);
        let text = engine.transcribe_with(samples, &options);
        Ok(owned_text(running.finish(text)?))
    })
}

/// Transcribes a WAV file (a file that is not at 16 kHz is resampled);
/// `language` as for `trit_transcribe`.
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
        let mut options = options(language)?;
        engine.check(&options)?;
        let samples = trit::read_wav(Path::new(text_arg(path, "path")?))?;
        let running = Running::start(engine, &mut options);
        let text = engine.transcribe_with(&samples, &options);
        Ok(owned_text(running.finish(text)?))
    })
}

const OPTION_KEYS: &str = "language, task, beam, fallback, timestamps, sample_rate";

/// What the JSON options of one call ask.
struct Request {
    options: Options,
    /// Fill `segments`.
    timestamps: bool,
    /// The rate of the caller's samples, when the caller gave it.
    sample_rate: Option<u32>,
}

impl Request {
    /// `samples` as the 16 kHz samples that the engine takes: resampled
    /// when the caller gave another rate.
    fn at_16_khz<'a>(&self, samples: &'a [f32]) -> Result<std::borrow::Cow<'a, [f32]>, String> {
        match self.sample_rate {
            Some(rate) if rate != trit::SAMPLE_RATE => Ok(trit::resample(samples, rate)?.into()),
            _ => Ok(samples.into()),
        }
    }
}

/// The request of a JSON options object. NULL or "" is the default; every
/// key is optional.
///
/// # Safety
/// `options` is NULL or a NUL-terminated string.
unsafe fn json_options(options: *const c_char) -> Result<Request, String> {
    let mut parsed = Options::default();
    let mut timestamps = false;
    let mut sample_rate = None;
    let Some(object) = json_object(options, "{\"language\": \"en\"}")? else {
        return Ok(Request {
            options: parsed,
            timestamps,
            sample_rate,
        });
    };
    for (key, value) in &object {
        let wrong = || format!("options.{key} cannot be {value}");
        match key.as_str() {
            "language" => {
                parsed.language = match value {
                    serde_json::Value::Null => None,
                    serde_json::Value::String(code) => trit::language_code(code),
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
            "sample_rate" => {
                let rate = value.as_u64().and_then(|rate| u32::try_from(rate).ok());
                sample_rate = Some(rate.ok_or_else(wrong)?);
            }
            _ => {
                return Err(format!(
                    "unknown option {key}; valid options: {OPTION_KEYS}"
                ))
            }
        }
    }
    Ok(Request {
        options: parsed,
        timestamps,
        sample_rate,
    })
}

/// `{"text": ..., "segments": [{"start", "end", "text"}]}`, the shape of one
/// result of `trit transcribe --format json`. With timestamps, the text is
/// the segments joined.
fn json_result(engine: &Trit, samples: &[f32], options: &Options, timestamps: bool) -> String {
    let (text, segments) = engine.transcript(samples, options, timestamps);
    serde_json::json!({
        "text": text,
        "segments": segments.iter().map(trit::Segment::json).collect::<Vec<_>>(),
    })
    .to_string()
}

/// Transcribes `count` mono samples with options, and returns
/// JSON: `{"text": "...", "segments": [{"start": 0.0, "end": 1.8, "text": "..."}]}`.
/// `options` is NULL or a JSON object with any of: `language` (ISO 639-1
/// code), `task` (`"transcribe"` or `"translate"`), `beam` (as for
/// `trit_transcribe_with`), `fallback` (Whisper temperature fallback,
/// default true), `timestamps` (fill `segments`, default false),
/// `sample_rate` (the rate of `samples` in Hz, default 16000; Trit
/// resamples another rate, such as the 44100 or 48000 of a microphone).
/// An unknown key or value fails. Free the result with `trit_free_text`.
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
        let mut request = json_options(options)?;
        engine.check(&request.options)?;
        let samples = request.at_16_khz(samples_arg(samples, count)?)?;
        let running = Running::start(engine, &mut request.options);
        let json = json_result(engine, &samples, &request.options, request.timestamps);
        Ok(owned_text(running.finish(json)?))
    })
}

/// `trit_transcribe_json` for a WAV file. The file has its own rate, so
/// `sample_rate` in `options` fails.
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
        let mut request = json_options(options)?;
        if request.sample_rate.is_some() {
            return Err("options.sample_rate is for samples; a WAV file has its own rate".into());
        }
        engine.check(&request.options)?;
        let samples = trit::read_wav(Path::new(text_arg(path, "path")?))?;
        let running = Running::start(engine, &mut request.options);
        let json = json_result(engine, &samples, &request.options, request.timestamps);
        Ok(owned_text(running.finish(json)?))
    })
}

/// Frees text from `trit_transcribe*`, `trit_transcribe_wav*`, `trit_clean`, or `trit_gpu_info`.
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
/// The device stays "cpu".
///
/// # Safety
/// `model_dir` and `precision` are NULL or NUL-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn trit_cleaner_load_with(
    model_dir: *const c_char,
    precision: *const c_char,
) -> *mut Cleaner {
    trit_cleaner_load_ex(model_dir, precision, std::ptr::null())
}

/// `trit_cleaner_load_with` with a device, as for `trit_load_ex`.
///
/// # Safety
/// `model_dir`, `precision`, and `device` are NULL or NUL-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn trit_cleaner_load_ex(
    model_dir: *const c_char,
    precision: *const c_char,
    device: *const c_char,
) -> *mut Cleaner {
    guard(std::ptr::null_mut(), || {
        let options = load_options(precision, device)?;
        let explicit = path_arg(model_dir, "model_dir")?;
        let dir = trit::find_cleanup_model(explicit)
            .ok_or_else(|| trit::missing_cleanup_model(explicit))?;
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
        let name = |value: *const c_char, setting: &str| -> Result<Option<&str>, String> {
            match value.is_null() {
                true => Ok(None),
                false => text_arg(value, setting).map(Some),
            }
        };
        let style = trit::Style::named(
            name(styling, "styling")?,
            name(structure, "structure")?,
            name(context, "context")?,
        )
        .map_err(|unknown| {
            format!(
                "unknown {} `{}`; valid: {}",
                unknown.setting,
                unknown.value,
                unknown.valid.join(", ")
            )
        })?;
        Ok(owned_text(cleaner.clean(text, &style)))
    })
}

/// Loads a voice (text to speech: Kokoro-82M or a VITS folder such as Meta
/// MMS-TTS), or a voice by its `trit models` name; NULL reads `TRIT_VOICE`,
/// then searches the default places for `kokoro-82m`, then `mms-tts-eng`.
/// Returns NULL on failure. Free with `trit_voice_free`.
///
/// # Safety
/// `model_dir` is NULL or a NUL-terminated UTF-8 path.
#[no_mangle]
pub unsafe extern "C" fn trit_voice_load(model_dir: *const c_char) -> *mut Voice {
    guard(std::ptr::null_mut(), || {
        let explicit = path_arg(model_dir, "model_dir")?;
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

/// The rate of the samples that `trit_speak` returns (24000 for Kokoro,
/// 16000 for MMS-TTS), or -1 for a NULL voice.
///
/// # Safety
/// `voice` is NULL or a pointer from `trit_voice_load` that was not freed.
#[no_mangle]
pub unsafe extern "C" fn trit_voice_sample_rate(voice: *const Voice) -> c_int {
    guard(-1, || Ok(voice_arg(voice)?.sample_rate() as c_int))
}

const VOICE_KEYS: &str = "speed, noise, noise_duration, seed, voice, phonemes";

/// Voicing from a JSON object; NULL or "" is the voice's own settings.
///
/// # Safety
/// `options` is NULL or a NUL-terminated string.
unsafe fn json_voicing(options: *const c_char) -> Result<Voicing, String> {
    let mut voicing = Voicing::default();
    let Some(object) = json_object(options, "{\"speed\": 1.1}")? else {
        return Ok(voicing);
    };
    for (key, value) in &object {
        let wrong = || format!("options.{key} cannot be {value}");
        let number = || value.as_f64().map(|v| v as f32).ok_or_else(wrong);
        match key.as_str() {
            "speed" => voicing.speed = Some(number()?),
            "noise" => voicing.noise = Some(number()?),
            "noise_duration" => voicing.noise_duration = Some(number()?),
            "seed" => voicing.seed = value.as_u64().ok_or_else(wrong)?,
            "voice" => {
                voicing.voice = match value {
                    serde_json::Value::Null => None,
                    serde_json::Value::String(name) => Some(name.clone()),
                    _ => return Err(wrong()),
                }
            }
            "phonemes" => voicing.phonemes = value.as_bool().ok_or_else(wrong)?,
            _ => return Err(format!("unknown option {key}; valid options: {VOICE_KEYS}")),
        }
    }
    Ok(voicing)
}

/// Speaks `text`: mono samples at `trit_voice_sample_rate`, in -1 to 1;
/// `*count` gets their number. `options` is NULL or a JSON object with any
/// of: `voice` (Kokoro: a name of `trit_voice_list`, default af_heart;
/// "a,b" blends), `phonemes` (Kokoro: the text is phonemes), `speed` (1 is
/// the voice's pace), `noise` (how much the voice varies), `noise_duration`
/// (VITS: how much the timing varies), `seed` (one text and one seed give
/// the same audio; default 0). Returns NULL on failure. Free the samples
/// with `trit_free_samples`.
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

/// The callback of `trit_speak_stream`: the samples of one part, and the
/// caller's pointer. Return 1 to go on, 0 to stop.
pub type TritSamplesFn =
    extern "C" fn(samples: *const f32, count: usize, user_data: *mut std::ffi::c_void) -> c_int;

/// `trit_speak` part by part: `callback` gets the samples of each part (a
/// sentence for VITS, a line or a 510-phoneme pass for Kokoro) as soon as
/// it is made. The samples are valid during the call only. Returns 0, or -1
/// on failure (not when the callback stops).
///
/// # Safety
/// As for `trit_speak`; `callback` is a valid function; `user_data` is
/// passed as it is.
#[no_mangle]
pub unsafe extern "C" fn trit_speak_stream(
    voice: *const Voice,
    text: *const c_char,
    options: *const c_char,
    callback: Option<TritSamplesFn>,
    user_data: *mut std::ffi::c_void,
) -> c_int {
    guard(-1, || {
        let voice = voice_arg(voice)?;
        let callback = callback.ok_or("callback is NULL")?;
        let text = text_arg(text, "text")?;
        let voicing = json_voicing(options)?;
        voice.speak_stream(text, &voicing, &mut |part| {
            callback(part.as_ptr(), part.len(), user_data) != 0
        })?;
        Ok(0)
    })
}

/// The voices of a Kokoro folder, one name per line ("" for a VITS voice).
/// Returns NULL on failure. Free with `trit_free_text`.
///
/// # Safety
/// `voice` is NULL or a pointer from `trit_voice_load` that was not freed.
#[no_mangle]
pub unsafe extern "C" fn trit_voice_list(voice: *const Voice) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        Ok(owned_text(voice_arg(voice)?.voices().join("\n")))
    })
}

/// "kokoro" or "vits"; NULL for a NULL voice. Static; do not free.
///
/// # Safety
/// `voice` is NULL or a pointer from `trit_voice_load` that was not freed.
#[no_mangle]
pub unsafe extern "C" fn trit_voice_kind(voice: *const Voice) -> *const c_char {
    voice
        .as_ref()
        .map_or(std::ptr::null(), |voice| static_text(voice.kind()))
}

/// The phonemes a Kokoro voice speaks for `text`, one line per pass, with
/// the accent of `options`' voice (a VITS voice gives the letters it
/// reads). Returns NULL on failure. Free with `trit_free_text`.
///
/// # Safety
/// `voice` is from `trit_voice_load`; `text` is a NUL-terminated UTF-8
/// string; `options` is NULL or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn trit_phonemes(
    voice: *const Voice,
    text: *const c_char,
    options: *const c_char,
) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let voice = voice_arg(voice)?;
        Ok(owned_text(voice.phonemes(
            text_arg(text, "text")?,
            &json_voicing(options)?,
        )?))
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
