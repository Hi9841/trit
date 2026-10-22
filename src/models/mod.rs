//! Model families. Each folder holds one network; `Model::load` picks the
//! family from the `model_type` of the folder's `config.json`.

pub mod beam;
pub mod canary;
pub mod moonshine;
pub mod parakeet;
pub mod qwen3;
pub mod transformer;
pub mod whisper;

use crate::{BiasWeights, Options, Precision, Segment};
use std::path::Path;

/// The `model_type` values Trit runs, for messages and `trit doctor`.
pub const MODEL_TYPES: [&str; 5] = [
    "parakeet_tdt",
    "parakeet_ctc",
    "whisper",
    "moonshine",
    "canary",
];

/// One loaded network. Boxed: the families differ much in size, and a
/// model is built once.
pub enum Model {
    Parakeet(Box<parakeet::Parakeet>),
    Whisper(Box<whisper::Whisper>),
    Moonshine(Box<moonshine::Moonshine>),
    Canary(Box<canary::Canary>),
}

impl Model {
    pub fn load(dir: &Path, precision: Precision) -> Result<Self, String> {
        let config = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("cannot read {}: {e}", dir.join("config.json").display()))?;
        let kind = model_type(&config)?;
        let _team = crate::cpu::enter(spread(&kind));
        Ok(match kind.as_str() {
            "parakeet_tdt" | "parakeet_ctc" => {
                Self::Parakeet(Box::new(parakeet::Parakeet::load(dir, &config, precision)?))
            }
            "whisper" => Self::Whisper(Box::new(whisper::Whisper::load(dir, &config, precision)?)),
            "moonshine" => Self::Moonshine(Box::new(moonshine::Moonshine::load(
                dir, &config, precision,
            )?)),
            "canary" => Self::Canary(Box::new(canary::Canary::load(dir, &config, precision)?)),
            other => {
                return Err(format!(
                    "{} holds a `{other}` model; Trit runs {}",
                    dir.display(),
                    MODEL_TYPES.join(", ")
                ))
            }
        })
    }

    /// The family name, for example `parakeet_tdt`.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Parakeet(model) => model.kind(),
            Self::Whisper(_) => "whisper",
            Self::Moonshine(_) => "moonshine",
            Self::Canary(_) => "canary",
        }
    }

    pub fn transcribe(&self, samples: &[f32], options: &Options) -> String {
        let _team = crate::cpu::enter(spread(self.kind()));
        match self {
            Self::Parakeet(model) => model.transcribe(samples, options),
            Self::Whisper(model) => joined(model.pieces(samples, options)),
            Self::Moonshine(model) => joined(model.pieces(samples, options)),
            Self::Canary(model) => joined(model.pieces(samples, options)),
        }
    }

    pub fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment> {
        let _team = crate::cpu::enter(spread(self.kind()));
        match self {
            Self::Parakeet(model) => model.segments(samples, options),
            Self::Whisper(model) => windowed(model.pieces(samples, options)),
            Self::Moonshine(model) => windowed(model.pieces(samples, options)),
            Self::Canary(model) => windowed(model.pieces(samples, options)),
        }
    }

    pub fn set_dictionary(&self, terms: &[String], weights: BiasWeights) -> Vec<String> {
        match self {
            Self::Parakeet(model) => model.set_dictionary(terms, weights),
            Self::Moonshine(model) => model.set_dictionary(terms, weights),
            Self::Canary(model) => model.set_dictionary(terms, weights),
            Self::Whisper(model) => model.set_dictionary(terms),
        }
    }
}

/// One segment per window: `(first sample, samples, text)`.
fn windowed(pieces: Vec<(usize, usize, String)>) -> Vec<Segment> {
    let seconds = |samples: usize| samples as f32 / crate::SAMPLE_RATE as f32;
    pieces
        .into_iter()
        .map(|(start, len, text)| Segment {
            start: seconds(start),
            end: seconds(start + len),
            text,
        })
        .collect()
}

/// Pieces joined with one space, as the families write a transcript.
fn joined(pieces: Vec<(usize, usize, String)>) -> String {
    let texts: Vec<String> = pieces.into_iter().map(|(_, _, text)| text).collect();
    texts.join(" ")
}

/// The team spread for a family: tall products use every core.
pub fn spread(kind: &str) -> crate::Spread {
    match kind {
        "whisper" => crate::Spread::AllCores,
        _ => crate::Spread::OneCache,
    }
}

/// `model_type` of a Hugging Face `config.json`.
fn model_type(config: &str) -> Result<String, String> {
    let value: serde_json::Value =
        serde_json::from_str(config).map_err(|e| format!("config.json is not JSON: {e}"))?;
    value["model_type"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| "config.json has no `model_type`".to_string())
}
