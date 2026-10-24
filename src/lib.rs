//! Trit: speech to text on the CPU, with its own kernels and no inference
//! library, no Python, and no GPU. It runs the models in `models/`; the
//! files are read as published on Hugging Face.
//!
//! ```no_run
//! let model = trit::find_model(None).expect("model not found");
//! let engine = trit::Trit::load(&model).unwrap();
//! let samples = trit::read_wav(std::path::Path::new("take.wav")).unwrap();
//! println!("{}", engine.transcribe(&samples));
//! ```

// First, so its `multiversion!` macro is in scope for the modules below.
#[macro_use]
mod cpu;
mod audio;
mod catalog;
mod cleanup;
mod models;
mod text;

use std::path::{Path, PathBuf};

pub use audio::wav::read_wav;
pub use catalog::{known, Known, CATALOG};
pub use cleanup::{Cleaner, Context, Structure, Style, Styling, CLEANUP_REPO};
pub use cpu::gemm::Precision;
pub use cpu::{cpu_path, threads, Spread};
pub use models::parts::logit_bias::BiasWeights;
pub use models::MODEL_TYPES;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// `transcribe` takes mono samples at this rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// The Hugging Face repository of the default model.
pub const MODEL_REPO: &str = "moondream/parakeet-redux";
/// The files every model folder holds; a family may need more. A large
/// model may hold `model.safetensors.index.json` and its shards instead of
/// `model.safetensors`.
pub const MODEL_FILES: [&str; 2] = ["config.json", "model.safetensors"];

/// How to transcribe. `Options::default()` lets the model choose.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// The spoken language as an ISO 639-1 code (`"en"`, `"de"`). `None`
    /// lets a multilingual model detect it. Parakeet cannot be told a
    /// language; `"en"` keeps it to English tokens.
    pub language: Option<String>,
    /// Whisper: decode a 30 s window again at temperatures 0.2 to 1.0 when
    /// its text repeats itself or the model is unsure, as OpenAI Whisper,
    /// faster-whisper, and whisper.cpp do. `None` is on.
    pub fallback: Option<bool>,
    /// Beams for the decoder families. 0 (the default) is the model's own
    /// choice: 5 for Whisper and Moonshine (fewer errors than greedy on
    /// every test set, at 2 to 3 times the decoder time), greedy for Canary.
    /// 1 is greedy, as the reference runtimes decode by default.
    pub beam: usize,
    /// Write the speech as spoken (the default), or translate it into
    /// English (Whisper's multilingual models, Canary).
    pub task: Task,
}

/// What to write.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Task {
    /// The words as spoken, in the spoken language.
    #[default]
    Transcribe,
    /// The speech translated into English.
    Translate,
}

impl Task {
    pub const NAMES: [&str; 2] = ["transcribe", "translate"];

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "transcribe" => Some(Self::Transcribe),
            "translate" => Some(Self::Translate),
            _ => None,
        }
    }
}

impl Options {
    pub(crate) fn fallback(&self) -> bool {
        self.fallback.unwrap_or(true)
    }

    /// `beam`, or `default` when it is 0.
    pub(crate) fn beams(&self, default: usize) -> usize {
        if self.beam == 0 {
            default
        } else {
            self.beam
        }
    }

    /// Options for one language, for example `Options::language("en")`.
    pub fn language(code: &str) -> Self {
        Self {
            language: Some(code.trim().to_ascii_lowercase()),
            ..Self::default()
        }
    }
}

/// A stretch of the recording and its text. Times are seconds from the start.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

/// How to load a model. `LoadOptions::default()` is `Precision::Auto`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadOptions {
    pub precision: Precision,
}

/// One loaded model. Load once and keep it. `Trit` is `Send + Sync`; calls
/// from several threads share one worker team, so they take turns per step
/// and do not finish sooner than calls made one after another.
pub struct Trit {
    model: models::Model,
}

impl std::fmt::Debug for Trit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Trit")
            .field("kind", &self.kind())
            .finish_non_exhaustive()
    }
}

impl Trit {
    /// Loads a model folder. The family comes from `model_type` in its
    /// `config.json`; see `MODEL_TYPES`.
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        Self::load_with(model_dir, &LoadOptions::default())
    }

    /// `load`, with a choice of precision.
    pub fn load_with(model_dir: &Path, options: &LoadOptions) -> Result<Self, String> {
        Ok(Self {
            model: models::Model::load(model_dir, options.precision)?,
        })
    }

    /// The model family, for example `parakeet_tdt`.
    pub fn kind(&self) -> &'static str {
        self.model.kind()
    }

    /// 16 kHz mono samples to text, with `Options::default()`.
    pub fn transcribe(&self, samples: &[f32]) -> String {
        self.model.transcribe(samples, &Options::default())
    }

    /// 16 kHz mono samples to text.
    pub fn transcribe_with(&self, samples: &[f32], options: &Options) -> String {
        self.model.transcribe(samples, options)
    }

    /// The text in timed segments: sentences or pauses for Parakeet (from
    /// the times its decoder emits tokens), one per window of up to 30 s for
    /// Whisper and Moonshine, and 40 s for Canary.
    pub fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment> {
        self.model.segments(samples, options)
    }

    /// Terms to favor while decoding, as written ("Kubernetes", "PostgreSQL").
    /// Returns the terms the model cannot spell; they are skipped. An empty
    /// list clears the dictionary.
    pub fn set_dictionary(&self, terms: &[String], weights: BiasWeights) -> Vec<String> {
        self.model.set_dictionary(terms, weights)
    }
}

/// Returns the memory that loading and transcribing freed to the system.
/// The engines take their temporaries from mimalloc (when the host's global
/// allocator is mimalloc, as in the CLI, the C library, and TDT), which keeps
/// freed pages for reuse. Call it after a load and when the app goes idle,
/// not between calls: the next call would allocate the pages again.
pub fn release_memory() {
    cpu::buf::clear_pool();
    // Each thread's heap keeps the pages its own frees left (layers load
    // on the workers), and mi_collect reaches only the calling thread's.
    // SAFETY: mi_collect only walks mimalloc's own free lists.
    cpu::team::on_every_thread(|| unsafe { libmimalloc_sys::mi_collect(true) });
}

/// The model folder: `explicit` if given (a folder, or a `CATALOG` name
/// found in the places of `candidates`), else `TRIT_MODEL`, else the first
/// complete `parakeet-redux` folder under `./models`, `../models`, the folder
/// of the executable, the TDT app data, or the Hugging Face cache.
pub fn find_model(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = explicit {
        if is_model(dir) {
            return Some(dir.to_path_buf());
        }
        // A catalog name (`whisper-large-v3-turbo`) or repository: its usual places.
        let entry = known(&dir.to_string_lossy())?;
        return candidates(entry.repo).into_iter().find(|d| is_model(d));
    }
    if let Some(dir) = std::env::var_os("TRIT_MODEL") {
        let dir = PathBuf::from(dir);
        return is_model(&dir).then_some(dir);
    }
    model_candidates().into_iter().find(|dir| is_model(dir))
}

/// Why `find_model(explicit)` found nothing, with the command that fixes it.
pub fn missing_model(explicit: Option<&Path>) -> String {
    match explicit.map(|dir| (dir, known(&dir.to_string_lossy()))) {
        Some((_, Some(entry))) => format!(
            "{} is not on this computer; get it: {}",
            entry.name,
            entry.download_command()
        ),
        Some((dir, None)) => format!(
            "{} is not a model folder ({}) or a name from `trit models`",
            dir.display(),
            MODEL_FILES.join(", ")
        ),
        None => format!(
            "no model found; get it: {}",
            known(MODEL_REPO).map_or_else(String::new, Known::download_command)
        ),
    }
}

/// The folders `find_model` searches, in order, when no folder is named.
pub fn model_candidates() -> Vec<PathBuf> {
    candidates(MODEL_REPO)
}

/// The places a model of Hugging Face repository `repo` (`owner/name`) may
/// be: `models/<name>` under the current folder, its parent, the folder of
/// the executable and two and three folders up, the TDT app data, then the
/// snapshots of the Hugging Face cache.
pub fn candidates(repo: &str) -> Vec<PathBuf> {
    let name = repo.rsplit('/').next().unwrap_or(repo);
    let mut roots = vec![PathBuf::from("models"), PathBuf::from("../models")];
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        roots.push(exe_dir.join("models"));
        roots.push(exe_dir.join("../../models"));
        roots.push(exe_dir.join("../../../models"));
    }
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        roots.push(PathBuf::from(local).join("TDT").join("models"));
    }
    let mut dirs: Vec<PathBuf> = roots.into_iter().map(|root| root.join(name)).collect();
    let hub = std::env::var_os("HF_HOME")
        .map(|home| PathBuf::from(home).join("hub"))
        .or_else(|| {
            std::env::var_os("USERPROFILE")
                .or_else(|| std::env::var_os("HOME"))
                .map(|home| PathBuf::from(home).join(".cache/huggingface/hub"))
        });
    if let Some(hub) = hub {
        let snapshots = hub
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots");
        if let Ok(entries) = std::fs::read_dir(snapshots) {
            dirs.extend(entries.flatten().map(|entry| entry.path()));
        }
    }
    dirs
}

/// The cleanup model folder: `explicit` if given, else `TRIT_CLEANUP_MODEL`,
/// else the first complete `s1-mini` folder of `candidates(CLEANUP_REPO)`.
pub fn find_cleanup_model(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = explicit {
        return is_model(dir).then(|| dir.to_path_buf());
    }
    if let Some(dir) = std::env::var_os("TRIT_CLEANUP_MODEL") {
        let dir = PathBuf::from(dir);
        return is_model(&dir).then_some(dir);
    }
    candidates(CLEANUP_REPO)
        .into_iter()
        .find(|dir| is_model(dir))
}

/// True when `dir` holds every file of `MODEL_FILES` (the weights as one
/// file or as shards).
pub fn is_model(dir: &Path) -> bool {
    let weights =
        dir.join(cpu::tensors::WEIGHTS).is_file() || dir.join(cpu::tensors::SHARD_INDEX).is_file();
    dir.join("config.json").is_file() && weights
}
