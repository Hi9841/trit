//! Speech models. `Model::load` reads the `model_type` of a folder's
//! `config.json` and loads that family from `FAMILIES`; each family wires
//! the shared building blocks (`parts/`) its config names. A new family is
//! one row in `FAMILIES` and one `Family` impl.

pub mod families;
pub mod fermion;
pub mod files;
pub mod gguf;
pub mod nemo;
pub mod package;
pub mod parts;

use families::{
    canary, canary_qwen, fun_asr, granite_ctc, granite_nar, granite_speech, moonshine, moss,
    parakeet, qwen3_asr, sensevoice, voxtral, voxtral_realtime, wav2vec2, whisper,
};

use crate::{BiasWeights, Device, Options, Precision, Segment, Spread};
use std::path::Path;

pub trait Family: Send + Sync {
    fn load(
        files: &files::Files,
        config: &crate::config::Json,
        precision: Precision,
        device: Device,
    ) -> Result<Self, String>
    where
        Self: Sized;

    fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment>;

    fn transcribe(&self, samples: &[f32], options: &Options) -> String {
        crate::joined(&self.segments(samples, options))
    }

    /// Sets the terms to favor; returns the terms it cannot use. A family
    /// with no way to steer its decoder skips every term.
    fn set_dictionary(&self, terms: &[String], _weights: BiasWeights) -> Vec<String> {
        terms.to_vec()
    }

    /// The language codes the model takes, when it has a list of them
    /// (Whisper's language tokens, Canary's prompt languages). `None`: the
    /// model takes no language, or any code.
    fn languages(&self) -> Option<Vec<String>> {
        None
    }
}

type Load =
    fn(&files::Files, &crate::config::Json, Precision, Device) -> Result<Box<dyn Family>, String>;

/// One `model_type` Trit runs: the worker spread its products need, and
/// how to load it.
pub struct Row {
    pub model_type: &'static str,
    spread: Spread,
    /// `Device::Auto` selects the GPU: a measurement shows that the family
    /// runs faster there (see `active_device`).
    gpu_gain: bool,
    load: Load,
}

const fn row<F: Family + 'static>(model_type: &'static str, spread: Spread) -> Row {
    Row {
        model_type,
        spread,
        gpu_gain: false,
        load: |files, config, precision, device| {
            Ok(Box::new(F::load(files, config, precision, device)?))
        },
    }
}

/// `row` for a family whose encoder stays on the GPU for a whole take
/// (the FastConformer of Parakeet).
const fn gpu_row<F: Family + 'static>(model_type: &'static str, spread: Spread) -> Row {
    Row {
        gpu_gain: true,
        ..row::<F>(model_type, spread)
    }
}

/// Every family, by `model_type`. Tall products (Whisper) use every core.
pub const FAMILIES: [Row; 21] = [
    gpu_row::<parakeet::Parakeet>("parakeet_tdt", Spread::OneCache),
    gpu_row::<parakeet::Parakeet>("parakeet_rnnt", Spread::OneCache),
    gpu_row::<parakeet::Parakeet>("parakeet_ctc", Spread::OneCache),
    row::<parakeet::Parakeet>("gigaam_rnnt", Spread::OneCache),
    row::<parakeet::Parakeet>("gigaam_ctc", Spread::OneCache),
    row::<parakeet::Parakeet>("medasr", Spread::OneCache),
    row::<whisper::Whisper>("whisper", Spread::AllCores),
    row::<moonshine::Moonshine>("moonshine", Spread::OneCache),
    row::<moonshine::Moonshine>("moonshine_streaming", Spread::OneCache),
    row::<canary::Canary>("canary", Spread::OneCache),
    row::<wav2vec2::Wav2Vec2>("wav2vec2", Spread::OneCache),
    row::<qwen3_asr::Qwen3Asr>("qwen3_asr", Spread::OneCache),
    row::<canary_qwen::CanaryQwen>("canary_qwen", Spread::OneCache),
    row::<fun_asr::FunAsr>("fun_asr_nano", Spread::OneCache),
    row::<granite_speech::GraniteSpeech>("granite_speech", Spread::OneCache),
    row::<voxtral::Voxtral>("voxtral", Spread::OneCache),
    row::<voxtral_realtime::VoxtralRealtime>("voxtral_realtime", Spread::OneCache),
    row::<moss::Moss>("moss", Spread::OneCache),
    row::<granite_nar::GraniteNar>("granite_speech_nar", Spread::OneCache),
    row::<granite_ctc::GraniteCtc>("granite_ctc", Spread::OneCache),
    row::<sensevoice::SenseVoice>("sensevoice", Spread::OneCache),
];

/// The `model_type` values Trit runs, for messages and `trit doctor`.
pub const MODEL_TYPES: [&str; FAMILIES.len()] = {
    let mut names = [""; FAMILIES.len()];
    let mut i = 0;
    while i < FAMILIES.len() {
        names[i] = FAMILIES[i].model_type;
        i += 1;
    }
    names
};

pub struct Model {
    row: &'static Row,
    family: Box<dyn Family>,
}

impl Model {
    pub fn load(dir: &Path, precision: Precision, device: Device) -> Result<Self, String> {
        let files = files::Files::open(dir)?;
        let config = crate::config::Json::parse(&files.read("config.json")?)?;
        let kind = config
            .str("model_type")
            .ok_or("config.json has no `model_type`")?;
        let row = FAMILIES
            .iter()
            .find(|row| row.model_type == kind)
            .ok_or_else(|| {
                if kind == "vits" {
                    return format!(
                        "{} holds a `vits` voice (text to speech); load it with `Voice::load` or `trit speak`",
                        dir.display()
                    );
                }
                format!(
                    "{} holds a `{kind}` model; Trit runs {}",
                    dir.display(),
                    MODEL_TYPES.join(", ")
                )
            })?;
        let device = crate::active_device(device, row.gpu_gain)?;
        let _team = crate::cpu::enter(row.spread);
        let family = (row.load)(&files, &config, precision, device)?;
        Ok(Self { row, family })
    }

    pub fn kind(&self) -> &'static str {
        self.row.model_type
    }

    pub fn transcribe(&self, samples: &[f32], options: &Options) -> String {
        let _team = crate::cpu::enter(self.row.spread);
        self.family.transcribe(samples, options)
    }

    pub fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment> {
        let _team = crate::cpu::enter(self.row.spread);
        self.family.segments(samples, options)
    }

    pub fn set_dictionary(&self, terms: &[String], weights: BiasWeights) -> Vec<String> {
        self.family.set_dictionary(terms, weights)
    }

    /// Whether the model can obey `options`: the language is one of the
    /// model's list, or, for a model with no list, has the form of an
    /// ISO 639-1 code (`en`, `en-US`). A call with a code that fails here
    /// runs as if no language was given.
    pub fn check(&self, options: &Options) -> Result<(), String> {
        let Some(code) = options.language.as_deref() else {
            return Ok(());
        };
        match self.family.languages() {
            Some(known) if known.iter().any(|k| k.eq_ignore_ascii_case(code)) => Ok(()),
            Some(known) => Err(format!(
                "language `{code}` is not one that this {} model takes; it takes: {}",
                self.kind(),
                known.join(", ")
            )),
            None if iso_code(code) => Ok(()),
            None => Err(format!(
                "language `{code}` is not an ISO 639-1 code: two letters (`en`), or with a region (`en-US`)"
            )),
        }
    }
}

/// Two letters, or two letters, `-` or `_`, and a region of letters or digits.
fn iso_code(code: &str) -> bool {
    let (language, region) = match code.split_once(['-', '_']) {
        Some((language, region)) => (language, Some(region)),
        None => (code, None),
    };
    language.len() == 2
        && language.bytes().all(|b| b.is_ascii_alphabetic())
        && region.is_none_or(|r| {
            (2..=4).contains(&r.len()) && r.bytes().all(|b| b.is_ascii_alphanumeric())
        })
}

/// Drops what recordings longer than the usual built.
pub fn release_caches() {
    parts::fastconformer::release_positions();
}

/// One segment per window of at most `max` samples that gives text. A
/// cancelled call stops before the next window and gives no segments, as
/// Parakeet does.
fn windowed(
    samples: &[f32],
    max: usize,
    options: &Options,
    mut text_of: impl FnMut(&[f32]) -> String,
) -> Vec<Segment> {
    let seconds = |samples: usize| samples as f32 / crate::SAMPLE_RATE as f32;
    let mut segments = Vec::new();
    for window in crate::audio::windows(samples, max) {
        if options.cancelled() {
            break;
        }
        let text = text_of(window);
        let start = crate::audio::offset(samples, window);
        if !text.is_empty() {
            segments.push(Segment {
                start: seconds(start),
                end: seconds(start + window.len()),
                text,
            });
        }
    }
    if options.cancelled() {
        segments.clear();
    }
    segments
}

/// Shorter than a spoken word: 100 ms. A decoder writes a guess for a
/// window this short (Whisper tiny wrote "you" for one sample, after all
/// six temperatures: 1.4 s).
const MIN_SPEECH_SAMPLES: usize = crate::SAMPLE_RATE as usize / 10;

/// `windowed`, without the windows that hold no speech.
fn speech_windows(
    samples: &[f32],
    max: usize,
    options: &Options,
    mut text_of: impl FnMut(&[f32]) -> String,
) -> Vec<Segment> {
    windowed(samples, max, options, |window| {
        if window.len() < MIN_SPEECH_SAMPLES || crate::audio::is_silent(window) {
            String::new()
        } else {
            text_of(window)
        }
    })
}
