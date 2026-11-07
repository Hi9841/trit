//! Trit: speech to text and text to speech with its own kernels and no
//! inference library, and no Python. It runs the models in `models/`; the
//! files are read as published on Hugging Face. The `gpu` feature runs the
//! Parakeet encoder, and the dense and ternary products of the speech
//! models, on a GPU. Without that feature every product runs on the CPU.
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
mod config;
mod device;
#[cfg(feature = "gpu")]
mod gpu;
mod layers;
mod locate;
mod models;
mod options;
mod text;
mod tts;

use std::path::Path;

pub use audio::wav::{read_wav, write_wav};
pub use catalog::{known, Known, CATALOG};
pub use cleanup::{Cleaner, Context, Structure, Style, Styling, UnknownName, CLEANUP_REPO};
pub use cpu::gemm::Precision;
pub use cpu::{cpu_path, threads, Spread};
pub(crate) use device::active_device;
pub use device::{gpu_adapter, gpu_path, gpu_weight_bytes, Device};
pub use locate::{
    candidates, find_cleanup_model, find_model, find_voice, is_model, is_voice,
    missing_cleanup_model, missing_model, missing_voice, model_candidates, MODEL_FILES, MODEL_REPO,
};
pub use models::parts::logit_bias::BiasWeights;
pub use models::MODEL_TYPES;
pub use options::{hundredths, joined, language_code, Cancel, Options, Segment, Task};
pub use tts::{
    phonemize, voice_accent, Voice, Voicing, PHONEMIZER_FALLBACKS, VOICE_REPO, VOICE_REPOS,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// `transcribe` takes mono samples at this rate.
pub const SAMPLE_RATE: u32 = 16_000;

/// How to load a model. `LoadOptions::default()` is `Precision::Auto` on the
/// CPU, with no cache. `TRIT_DEVICE` overrides `device` at load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadOptions {
    pub precision: Precision,
    pub device: Device,
    /// Keep the converted weights in `<model>/.trit/` and map that file.
    /// The weights then count as file memory, not private memory (about
    /// 590 MB of Parakeet Ultra's 750 MB); the system can drop those pages
    /// under memory pressure and read them back from the file. The file
    /// takes the same space on disk. The first load writes it and loads
    /// again. Every load still converts the weights and maps a buffer only
    /// when the file holds the same bytes, so the text never changes.
    pub cache: bool,
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

    /// `load`, with a choice of precision, device, and cache.
    pub fn load_with(model_dir: &Path, options: &LoadOptions) -> Result<Self, String> {
        Self::load_reporting(model_dir, options, None)
    }

    /// `load_with`; `progress` gets the fraction of the load done, from 0
    /// to 1, from the thread that loads.
    pub fn load_reporting(
        model_dir: &Path,
        options: &LoadOptions,
        progress: Option<std::sync::Arc<dyn Fn(f32) + Send + Sync>>,
    ) -> Result<Self, String> {
        let cache = options
            .cache
            .then(|| cpu::store::file_name(options.precision.name()));
        let model = cpu::store::load(model_dir, cache, progress, release_memory, || {
            models::Model::load(model_dir, options.precision, options.device)
        })?;
        Ok(Self { model })
    }

    /// The model family, for example `parakeet_tdt`.
    pub fn kind(&self) -> &'static str {
        self.model.kind()
    }

    /// 16 kHz mono samples to text, with `Options::default()`.
    pub fn transcribe(&self, samples: &[f32]) -> String {
        self.transcribe_with(samples, &Options::default())
    }

    /// 16 kHz mono samples to text.
    pub fn transcribe_with(&self, samples: &[f32], options: &Options) -> String {
        self.model.transcribe(&audio::sanitized(samples), options)
    }

    /// The text and, with `timestamps`, its segments. With timestamps the
    /// text is the segments joined.
    pub fn transcript(
        &self,
        samples: &[f32],
        options: &Options,
        timestamps: bool,
    ) -> (String, Vec<Segment>) {
        if timestamps {
            let segments = self.segments(samples, options);
            (joined(&segments), segments)
        } else {
            (self.transcribe_with(samples, options), Vec::new())
        }
    }

    /// The text in timed segments: sentences or pauses for Parakeet (from
    /// the times its decoder emits tokens), one per window of up to 30 s for
    /// Whisper and Moonshine, and 40 s for Canary.
    pub fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment> {
        self.model.segments(&audio::sanitized(samples), options)
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
/// not between calls: the next call would allocate the pages again. It also
/// drops the position tables that recordings longer than 30 s built. With the
/// `gpu` feature it also frees the pooled GPU buffers and the staging buffers
/// of dropped GPU models.
pub fn release_memory() {
    models::release_caches();
    cpu::buf::clear_pool();
    #[cfg(feature = "gpu")]
    gpu::trim();
    // Each thread's heap keeps the pages its own frees left (layers load
    // on the workers), and mi_collect reaches only the calling thread's.
    // SAFETY: mi_collect only walks mimalloc's own free lists.
    cpu::team::on_every_thread(|| unsafe { libmimalloc_sys::mi_collect(true) });
}
