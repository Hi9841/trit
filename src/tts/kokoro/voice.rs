//! A Kokoro voice: the model and its two phonemizers.

use super::Kokoro;
use crate::tts::noise::Noise;
use crate::tts::{british, english, fallback_dir, in_range, Voicing, NOISE_RANGE, SPEED_RANGE};
use std::path::Path;

/// The Kokoro voice when none is named (the `kokoro` package's example).
const DEFAULT_KOKORO_VOICE: &str = "af_heart";

/// Kokoro and its two phonemizers, each loaded on first use.
pub struct KokoroVoice {
    model: Kokoro,
    dir: std::path::PathBuf,
    american: std::sync::OnceLock<Result<english::English, String>>,
    british: std::sync::OnceLock<Result<english::English, String>>,
}

impl KokoroVoice {
    /// The phonemizer of a voice: British for `b…` voices, American for `a…`.
    fn english(&self, voice: &str) -> Result<&english::English, String> {
        let british = british(voice.trim()).ok_or_else(|| {
            format!(
                "voice {voice} does not speak English; the phonemizer is English (a… and b… voices): pass phonemes with `phonemes`"
            )
        })?;
        let cell = if british {
            &self.british
        } else {
            &self.american
        };
        cell.get_or_init(|| {
            english::English::load(british, fallback_dir(self.dir.parent(), british).as_deref())
        })
        .as_ref()
        .map_err(Clone::clone)
    }

    pub fn load(dir: &Path) -> Result<Self, String> {
        Ok(Self {
            model: Kokoro::load(dir)?,
            dir: dir.to_path_buf(),
            american: std::sync::OnceLock::new(),
            british: std::sync::OnceLock::new(),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.model.sample_rate
    }

    pub fn voices(&self) -> Vec<String> {
        self.model.voices()
    }

    pub fn reads_nothing_help(&self) -> &'static str {
        "Run `trit phonemes \"<text>\"` to see what Kokoro reads"
    }

    /// The phonemes with the default voice.
    pub fn reads(&self, text: &str) -> String {
        self.phonemes(text, &Voicing::default()).unwrap_or_default()
    }

    /// The phonemes of `text`, one line per pass, with the accent of
    /// `voicing.voice`; phoneme input as it is.
    pub fn phonemes(&self, text: &str, voicing: &Voicing) -> Result<String, String> {
        if voicing.phonemes {
            return Ok(text.trim().to_string());
        }
        let name = self.voice_name(voicing)?;
        let chunks = self.english(&name)?.chunks(text, self.model.max_phonemes)?;
        Ok(chunks
            .into_iter()
            .map(|c| c.phonemes)
            .collect::<Vec<_>>()
            .join("\n"))
    }

    /// One pass per line of phonemes, or per chunk of the phonemizer.
    pub fn speak_stream(
        &self,
        text: &str,
        voicing: &Voicing,
        noise: &mut Noise,
        on_part: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<(), String> {
        let model = &self.model;
        let speed = voicing.speed.unwrap_or(1.0);
        let scale = voicing.noise.unwrap_or(1.0);
        in_range("speed", speed, SPEED_RANGE)?;
        in_range("noise", scale, NOISE_RANGE)?;
        if voicing.noise_duration.is_some() {
            return Err(
                "noise_duration is a VITS setting; the timing of Kokoro has no noise".into(),
            );
        }
        let name = self.voice_name(voicing)?;
        let pack = model.pack(&name)?;
        let passes: Vec<String> = if voicing.phonemes {
            let lines: Vec<String> = text
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(String::from)
                .collect();
            if let Some(long) = lines
                .iter()
                .find(|l| l.chars().count() > model.max_phonemes)
            {
                return Err(format!(
                    "a line of {} phonemes is longer than the {} of one pass",
                    long.chars().count(),
                    model.max_phonemes
                ));
            }
            lines
        } else {
            self.english(&name)?
                .chunks(text, model.max_phonemes)?
                .into_iter()
                .map(|c| c.phonemes)
                .collect()
        };
        for phonemes in passes {
            let audio = model.speak(&phonemes, &pack, speed, scale, noise)?;
            if !on_part(&audio) {
                break;
            }
        }
        Ok(())
    }

    /// The voice of `voicing`, else `af_heart`, else the first voice of the folder.
    fn voice_name(&self, voicing: &Voicing) -> Result<String, String> {
        if let Some(name) = &voicing.voice {
            return Ok(name.clone());
        }
        let voices = self.model.voices();
        if voices.iter().any(|v| v == DEFAULT_KOKORO_VOICE) {
            return Ok(DEFAULT_KOKORO_VOICE.to_string());
        }
        voices
            .into_iter()
            .next()
            .ok_or_else(|| "the voice folder has no voices/*.pt".to_string())
    }
}
