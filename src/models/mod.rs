//! Speech models. `Model::load` reads the `model_type` of a folder's
//! `config.json` and loads that family from `FAMILIES`; each family wires
//! the shared building blocks (`parts/`) its config names. A new family is
//! one row in `FAMILIES` and one `Family` impl.

pub mod config;
pub mod families;
pub mod layers;
pub mod parts;

use families::{canary, moonshine, parakeet, qwen3_asr, wav2vec2, whisper};

use crate::{BiasWeights, Options, Precision, Segment, Spread};
use std::path::Path;

/// What every family does with audio.
pub trait Family: Send + Sync {
    /// The family name, for example `parakeet_tdt`.
    fn kind(&self) -> &'static str;
    fn transcribe(&self, samples: &[f32], options: &Options) -> String;
    fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment>;
    /// Sets the terms to favor; returns the terms it cannot use.
    fn set_dictionary(&self, terms: &[String], weights: BiasWeights) -> Vec<String>;
}

type Load = fn(&Path, &config::Json, Precision) -> Result<Box<dyn Family>, String>;

/// One `model_type` Trit runs: the worker spread its products need, and
/// how to load it.
pub struct Row {
    pub model_type: &'static str,
    spread: Spread,
    load: Load,
}

/// Every family, by `model_type`. Tall products (Whisper) use every core.
pub const FAMILIES: [Row; 8] = [
    Row {
        model_type: "parakeet_tdt",
        spread: Spread::OneCache,
        load: |dir, c, p| Ok(Box::new(parakeet::Parakeet::load(dir, c, p)?)),
    },
    Row {
        model_type: "parakeet_rnnt",
        spread: Spread::OneCache,
        load: |dir, c, p| Ok(Box::new(parakeet::Parakeet::load(dir, c, p)?)),
    },
    Row {
        model_type: "parakeet_ctc",
        spread: Spread::OneCache,
        load: |dir, c, p| Ok(Box::new(parakeet::Parakeet::load(dir, c, p)?)),
    },
    Row {
        model_type: "whisper",
        spread: Spread::AllCores,
        load: |dir, c, p| Ok(Box::new(whisper::Whisper::load(dir, c, p)?)),
    },
    Row {
        model_type: "moonshine",
        spread: Spread::OneCache,
        load: |dir, c, p| Ok(Box::new(moonshine::Moonshine::load(dir, c, p)?)),
    },
    Row {
        model_type: "canary",
        spread: Spread::OneCache,
        load: |dir, c, p| Ok(Box::new(canary::Canary::load(dir, c, p)?)),
    },
    Row {
        model_type: "wav2vec2",
        spread: Spread::OneCache,
        load: |dir, c, p| Ok(Box::new(wav2vec2::Wav2Vec2::load(dir, c, p)?)),
    },
    Row {
        model_type: "qwen3_asr",
        spread: Spread::OneCache,
        load: |dir, c, p| Ok(Box::new(qwen3_asr::Qwen3Asr::load(dir, c, p)?)),
    },
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

/// One loaded network.
pub struct Model(Box<dyn Family>);

impl Model {
    pub fn load(dir: &Path, precision: Precision) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("cannot read {}: {e}", dir.join("config.json").display()))?;
        let config = config::Json::parse(&text)?;
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
        let _team = crate::cpu::enter(row.spread);
        Ok(Self((row.load)(dir, &config, precision)?))
    }

    pub fn kind(&self) -> &'static str {
        self.0.kind()
    }

    pub fn transcribe(&self, samples: &[f32], options: &Options) -> String {
        let _team = crate::cpu::enter(spread(self.kind()));
        self.0.transcribe(samples, options)
    }

    pub fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment> {
        let _team = crate::cpu::enter(spread(self.kind()));
        self.0.segments(samples, options)
    }

    pub fn set_dictionary(&self, terms: &[String], weights: BiasWeights) -> Vec<String> {
        self.0.set_dictionary(terms, weights)
    }
}

/// The worker spread of a family; `OneCache` for an unknown name.
pub fn spread(kind: &str) -> Spread {
    FAMILIES
        .iter()
        .find(|row| row.model_type == kind)
        .map_or(Spread::OneCache, |row| row.spread)
}

/// Text per window: `(first sample, samples, text)`.
type Pieces = Vec<(usize, usize, String)>;

/// One segment per window.
fn windowed(pieces: Pieces) -> Vec<Segment> {
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
fn joined(pieces: Pieces) -> String {
    let texts: Vec<String> = pieces.into_iter().map(|(_, _, text)| text).collect();
    texts.join(" ")
}

impl Family for parakeet::Parakeet {
    fn kind(&self) -> &'static str {
        parakeet::Parakeet::kind(self)
    }
    fn transcribe(&self, samples: &[f32], options: &Options) -> String {
        parakeet::Parakeet::transcribe(self, samples, options)
    }
    fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment> {
        parakeet::Parakeet::segments(self, samples, options)
    }
    fn set_dictionary(&self, terms: &[String], weights: BiasWeights) -> Vec<String> {
        parakeet::Parakeet::set_dictionary(self, terms, weights)
    }
}

impl Family for whisper::Whisper {
    fn kind(&self) -> &'static str {
        "whisper"
    }
    fn transcribe(&self, samples: &[f32], options: &Options) -> String {
        joined(self.pieces(samples, options))
    }
    fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment> {
        windowed(self.pieces(samples, options))
    }
    fn set_dictionary(&self, terms: &[String], _: BiasWeights) -> Vec<String> {
        whisper::Whisper::set_dictionary(self, terms)
    }
}

impl Family for moonshine::Moonshine {
    fn kind(&self) -> &'static str {
        "moonshine"
    }
    fn transcribe(&self, samples: &[f32], options: &Options) -> String {
        joined(self.pieces(samples, options))
    }
    fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment> {
        windowed(self.pieces(samples, options))
    }
    fn set_dictionary(&self, terms: &[String], weights: BiasWeights) -> Vec<String> {
        moonshine::Moonshine::set_dictionary(self, terms, weights)
    }
}

impl Family for canary::Canary {
    fn kind(&self) -> &'static str {
        "canary"
    }
    fn transcribe(&self, samples: &[f32], options: &Options) -> String {
        joined(self.pieces(samples, options))
    }
    fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment> {
        windowed(self.pieces(samples, options))
    }
    fn set_dictionary(&self, terms: &[String], weights: BiasWeights) -> Vec<String> {
        canary::Canary::set_dictionary(self, terms, weights)
    }
}

impl Family for wav2vec2::Wav2Vec2 {
    fn kind(&self) -> &'static str {
        "wav2vec2"
    }
    fn transcribe(&self, samples: &[f32], _: &Options) -> String {
        joined(self.pieces(samples))
    }
    fn segments(&self, samples: &[f32], _: &Options) -> Vec<Segment> {
        windowed(self.pieces(samples))
    }
    /// Characters with no decoder to steer: no term can be favored.
    fn set_dictionary(&self, terms: &[String], _: BiasWeights) -> Vec<String> {
        terms.to_vec()
    }
}

impl Family for qwen3_asr::Qwen3Asr {
    fn kind(&self) -> &'static str {
        "qwen3_asr"
    }
    fn transcribe(&self, samples: &[f32], options: &Options) -> String {
        joined(self.pieces(samples, options))
    }
    fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment> {
        windowed(self.pieces(samples, options))
    }
    fn set_dictionary(&self, terms: &[String], _: BiasWeights) -> Vec<String> {
        qwen3_asr::Qwen3Asr::set_dictionary(self, terms)
    }
}
