//! Text to speech with a VITS voice (Meta MMS-TTS): text in, mono samples
//! out. The voice reads characters: no phonemizer and no word list, so it
//! says what the text spells. Numbers and symbols that the model's
//! vocabulary does not hold are not spoken; write them as words.

use crate::models::families::vits::{Noise, Settings, Vits};
use crate::text::characters::Characters;
use std::path::Path;

/// The Hugging Face repository of the default voice.
pub const VOICE_REPO: &str = "facebook/mms-tts-eng";

/// Silence between two sentences, in seconds. The model speaks one
/// sentence per pass; a pass ends close to its last sound.
const SENTENCE_PAUSE: f32 = 0.3;
/// Longest part of one pass, in characters. A longer sentence is cut at its
/// last comma, else its last space, before this length: the text attention
/// grows with the square of the characters, and the vocoder's buffers with
/// the length of the audio.
const MAX_PART_CHARS: usize = 400;

/// Every physical core: the vocoder's products are tall (16000 rows per
/// second of audio, as Whisper's 1500). On 16 cores in two L3 domains, 8.9 s
/// of speech took 410 ms, against 510 ms on the 8 cores of one domain.
const SPREAD: crate::Spread = crate::Spread::AllCores;

/// How to speak. `Voicing::default()` is the model's own settings.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Voicing {
    /// Speaking rate: 1.0 is the model's pace, 1.25 a quarter faster, 0.8
    /// slower. `None` is the model's (`speaking_rate` in `config.json`).
    pub speed: Option<f32>,
    /// How much the voice varies (VITS `noise_scale`); 0 is the flattest.
    /// `None` is the model's (0.667 for MMS-TTS).
    pub noise: Option<f32>,
    /// How much the timing varies (`noise_scale_duration`); 0 gives the
    /// same length to each letter in each context. `None` is the model's (0.8).
    pub noise_duration: Option<f32>,
    /// The seed of the noise: one text with one seed always gives the same audio.
    pub seed: u64,
}

/// A loaded voice. Load once and keep it. `Voice` is `Send + Sync`.
pub struct Voice {
    model: Vits,
    characters: Characters,
}

impl std::fmt::Debug for Voice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Voice")
            .field("sample_rate", &self.sample_rate())
            .finish_non_exhaustive()
    }
}

impl Voice {
    /// Loads a VITS folder (`config.json`, `model.safetensors`, `vocab.json`,
    /// `tokenizer_config.json`). The weights stay f32.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let _team = crate::cpu::enter(SPREAD);
        Ok(Self {
            model: Vits::load(dir)?,
            characters: Characters::open(dir)?,
        })
    }

    /// The rate of the samples that `speak` returns (16000 for MMS-TTS).
    pub fn sample_rate(&self) -> u32 {
        self.model.sample_rate
    }

    /// The text as the voice reads it: lowercase, only the characters of
    /// its vocabulary. What is missing here is not spoken.
    pub fn reads(&self, text: &str) -> String {
        self.characters.reads(text)
    }

    /// The speech of `text` with `Voicing::default()`, mono at `sample_rate()`.
    pub fn speak(&self, text: &str) -> Vec<f32> {
        self.speak_with(text, &Voicing::default())
            .unwrap_or_default()
    }

    /// The speech of `text`, mono at `sample_rate()`, in -1..1. Sentences are
    /// spoken one by one, with a short pause between them. Fails on a speed
    /// that is not above 0, or a noise below 0.
    pub fn speak_with(&self, text: &str, voicing: &Voicing) -> Result<Vec<f32>, String> {
        let defaults = self.model.defaults;
        let settings = Settings {
            speed: voicing.speed.unwrap_or(defaults.speed),
            noise: voicing.noise.unwrap_or(defaults.noise),
            noise_duration: voicing.noise_duration.unwrap_or(defaults.noise_duration),
        };
        if !(settings.speed.is_finite() && settings.speed > 0.0) {
            return Err(format!("speed {} must be above 0", settings.speed));
        }
        for (name, value) in [
            ("noise", settings.noise),
            ("noise_duration", settings.noise_duration),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(format!("{name} {value} must be 0 or more"));
            }
        }
        let _team = crate::cpu::enter(SPREAD);
        let mut noise = Noise::new(voicing.seed);
        let pause = vec![0.0f32; (SENTENCE_PAUSE * self.sample_rate() as f32) as usize];
        let mut out: Vec<f32> = Vec::new();
        for part in parts(text) {
            let ids = self.characters.encode(part);
            if ids.len() <= 1 {
                continue;
            }
            if !out.is_empty() {
                out.extend_from_slice(&pause);
            }
            out.extend(self.model.speak(&ids, &settings, &mut noise));
        }
        Ok(out)
    }
}

/// `text` cut after each sentence end (`.`, `!`, `?`, `;`, `:` before a
/// space or the end) and at line breaks, with each part at most
/// `MAX_PART_CHARS` long.
fn parts(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    for line in text.lines() {
        let mut start = 0;
        for (i, c) in line.char_indices() {
            let end = i + c.len_utf8();
            let next = line[end..].chars().next();
            if matches!(c, '.' | '!' | '?' | ';' | ':') && next.is_none_or(char::is_whitespace) {
                sentences.push(&line[start..end]);
                start = end;
            }
        }
        sentences.push(&line[start..]);
    }
    let mut out = Vec::new();
    for mut sentence in sentences.into_iter().map(str::trim) {
        while sentence.chars().count() > MAX_PART_CHARS {
            let limit = sentence
                .char_indices()
                .nth(MAX_PART_CHARS)
                .map_or(sentence.len(), |(i, _)| i);
            let head = &sentence[..limit];
            let cut = head
                .rfind(',')
                .map(|i| i + 1)
                .or_else(|| head.rfind(' '))
                .filter(|i| *i > 0)
                .unwrap_or(limit);
            out.push(sentence[..cut].trim());
            sentence = sentence[cut..].trim_start();
        }
        if !sentence.is_empty() {
            out.push(sentence);
        }
    }
    out
}
