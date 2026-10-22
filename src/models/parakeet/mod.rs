//! Parakeet: FastConformer encoder with a TDT decoder (NVIDIA Parakeet TDT,
//! Moondream Parakeet Redux). The network follows the Apache-2.0 Hugging
//! Face Parakeet implementation; the files are read as published.

pub mod bias;
mod dictionary;
mod english;
pub(crate) mod model;

use crate::audio::mel::MelFrontend;
use crate::cpu::tensors::Tensors;
use crate::text::sentencepiece::Tokenizer;
use crate::{Options, Precision, Segment};
use parking_lot::RwLock;
use std::path::Path;

pub use bias::BiasWeights;

pub struct Parakeet {
    model: model::Parakeet,
    /// The user's dictionary; none means plain greedy decoding.
    dictionary: RwLock<Option<dictionary::Dictionary>>,
    frontend: MelFrontend,
    tokenizer: Tokenizer,
    /// Tokens of other languages; blocked when the caller asks for English.
    non_english: Vec<usize>,
    /// The English word prior, for callers that ask for English.
    english: english::English,
}

/// One minute: 750 encoder frames, a few MB of attention per head.
const WINDOW_SAMPLES: usize = 60 * 16_000;
/// Seconds per encoder frame: 10 ms mel frames, subsampled 8 times.
const FRAME_SECONDS: f32 = 0.08;
/// A pause this long ends a segment.
const PAUSE_SECONDS: f32 = 1.0;

impl Parakeet {
    pub fn load(dir: &Path, config: &str, precision: Precision) -> Result<Self, String> {
        let config = model::Config::parse(config)?;
        let tensors = Tensors::open_with(&dir.join("model.safetensors"), precision)?;
        let frontend = MelFrontend::new(config.mels);
        let model = model::Parakeet::load(&tensors, config)?;
        // The weight file is no longer needed; free it before the position
        // tables grow, so load does not hold both at its peak.
        drop(tensors);
        model.warm_positions();
        let tokenizer = Tokenizer::open(&dir.join("tokenizer.json"))?;
        Ok(Self {
            dictionary: RwLock::new(None),
            model,
            frontend,
            english: english::English::new(&tokenizer),
            non_english: tokenizer.non_english(),
            tokenizer,
        })
    }

    /// `parakeet_tdt` or `parakeet_ctc`.
    pub fn kind(&self) -> &'static str {
        match self.model.config.head {
            model::HeadKind::Ctc => "parakeet_ctc",
            _ => "parakeet_tdt",
        }
    }

    pub fn set_dictionary(&self, terms: &[String], weights: BiasWeights) -> Vec<String> {
        self.english.set_terms(terms);
        let (terms, skipped) = dictionary::Dictionary::build(&self.tokenizer, terms, weights);
        *self.dictionary.write() = terms;
        skipped
    }

    /// Recordings longer than `WINDOW_SAMPLES` go through in windows cut at
    /// quiet points: attention memory grows with the square of the length.
    pub fn transcribe(&self, samples: &[f32], options: &Options) -> String {
        let windows = crate::audio::windows(samples, WINDOW_SAMPLES);
        if let [one] = windows[..] {
            return self.tokenizer.decode(&self.decode_window(one, options).0);
        }
        let pieces: Vec<String> = windows
            .into_iter()
            .map(|window| {
                self.tokenizer
                    .decode(&self.decode_window(window, options).0)
                    .trim()
                    .to_string()
            })
            .filter(|piece| !piece.is_empty())
            .collect();
        pieces.join(" ")
    }

    /// The text in sentences: a segment ends at `.`, `?`, or `!`, or where
    /// the speaker pauses for `PAUSE_SECONDS`. A token's time is the encoder
    /// frame the decoder emitted it at.
    pub fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment> {
        let mut segments = Vec::new();
        for window in crate::audio::windows(samples, WINDOW_SAMPLES) {
            let offset = crate::audio::offset(samples, window) as f32 / crate::SAMPLE_RATE as f32;
            let (tokens, frames) = self.decode_window(window, options);
            let time = |frame: usize| offset + frame as f32 * FRAME_SECONDS;
            let mut start = 0;
            for i in 0..tokens.len() {
                let text = self.tokenizer.decode(&tokens[i..=i]);
                let sentence_end = text.trim_end().ends_with(['.', '?', '!']);
                let pause = frames
                    .get(i + 1)
                    .is_some_and(|next| (next - frames[i]) as f32 * FRAME_SECONDS >= PAUSE_SECONDS);
                if sentence_end || pause || i + 1 == tokens.len() {
                    let text = self.tokenizer.decode(&tokens[start..=i]).trim().to_string();
                    if !text.is_empty() {
                        segments.push(Segment {
                            start: time(frames[start]),
                            end: time(frames[i] + 1),
                            text,
                        });
                    }
                    start = i + 1;
                }
            }
        }
        segments
    }

    /// Tokens of one window and the encoder frame of each.
    fn decode_window(&self, samples: &[f32], options: &Options) -> (Vec<usize>, Vec<usize>) {
        let (features, frames) = self.frontend.compute(samples);
        // The subsampling needs at least one output frame per stage.
        if frames < 8 {
            return (Vec::new(), Vec::new());
        }
        // Redux's speech head: no speech, no text.
        let Some((encoded, t)) = self.model.encode_speech(&features, frames) else {
            return (Vec::new(), Vec::new());
        };
        // Parakeet cannot be told a language; it can only be kept to English.
        let (blocked, english): (&[usize], _) = match options.language.as_deref() {
            Some("en") => (&self.non_english, Some(&self.english)),
            _ => (&[], None),
        };
        let dictionary = self.dictionary.read();
        self.model
            .decode(&encoded, t, dictionary.as_ref(), english, blocked)
    }
}
