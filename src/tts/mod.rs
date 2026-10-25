//! Text to speech: text in, mono samples out. Two kinds of voice:
//!
//! - Kokoro-82M (StyleTTS 2 + iSTFTNet, 24 kHz): English text goes through
//!   a phonemizer (`english`), then the model. A folder has many voices.
//! - VITS (Meta MMS-TTS, 16 kHz): the model reads letters, so it says what
//!   the text spells, in any of the MMS languages.
//!
//! This is the one part of Trit that may hold lists of words (text
//! normalization, the pronunciation lexicon): speech to text decodes with
//! the model alone.

mod characters;
mod english;
mod kokoro;
mod vits;

use characters::Characters;
pub use kokoro::is_kokoro;
use kokoro::Kokoro;
use std::path::Path;
use vits::{Noise, Settings, Vits};

/// The Hugging Face repository of the default voice (Kokoro-82M).
pub const VOICE_REPO: &str = "hexgrad/Kokoro-82M";
/// The voices `find_voice` looks for, in order: Kokoro, then MMS-TTS English.
pub const VOICE_REPOS: [&str; 2] = [VOICE_REPO, "facebook/mms-tts-eng"];

/// Silence between two sentences of a VITS voice, in seconds. The model
/// speaks one sentence per pass; a pass ends close to its last sound.
const SENTENCE_PAUSE: f32 = 0.3;
/// Longest part of one VITS pass, in characters. A longer sentence is cut
/// at its last comma, else its last space, before this length: the text
/// attention grows with the square of the characters, and the vocoder's
/// buffers with the length of the audio.
const MAX_PART_CHARS: usize = 400;
/// The Kokoro voice when none is named (the `kokoro` package's example).
const DEFAULT_KOKORO_VOICE: &str = "af_heart";

/// Every physical core: the vocoder's products are tall (16000 rows per
/// second of audio, as Whisper's 1500). On 16 cores in two L3 domains, 8.9 s
/// of speech took 410 ms, against 510 ms on the 8 cores of one domain.
const SPREAD: crate::Spread = crate::Spread::AllCores;

/// How to speak. `Voicing::default()` is the model's own settings.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Voicing {
    /// Speaking rate: 1.0 is the model's pace, 1.25 a quarter faster, 0.8
    /// slower. `None` is the model's (`speaking_rate` of a VITS config, 1 for Kokoro).
    pub speed: Option<f32>,
    /// How much the voice varies. VITS: `noise_scale` (`None`: 0.667 for
    /// MMS-TTS), 0 is the flattest. Kokoro: a scale of the noise of the
    /// vocoder's source (`None`: 1, the model's), 0 has none.
    pub noise: Option<f32>,
    /// VITS only: how much the timing varies (`noise_scale_duration`;
    /// `None`: 0.8). 0 gives each letter the same length in each context.
    pub noise_duration: Option<f32>,
    /// The seed of the noise: one text with one seed always gives the same audio.
    pub seed: u64,
    /// Kokoro: the voice, a name of `voices()` (`af_heart`), or several
    /// joined by commas, which are averaged. `None` is `af_heart`.
    pub voice: Option<String>,
    /// Kokoro: the text is phonemes (Kokoro's alphabet, as misaki writes
    /// it), not words. One line is one pass of at most 510 phonemes.
    pub phonemes: bool,
}

/// The Hugging Face repositories of the fallback for unknown words.
const FALLBACK_US: &str = "PeterReid/graphemes_to_phonemes_en_us";
const FALLBACK_GB: &str = "PeterReid/graphemes_to_phonemes_en_gb";

enum Model {
    Vits {
        model: Box<Vits>,
        characters: Characters,
    },
    Kokoro(Box<KokoroVoice>),
}

/// Kokoro and its two phonemizers, each loaded on first use.
struct KokoroVoice {
    model: Kokoro,
    dir: std::path::PathBuf,
    american: std::sync::OnceLock<Result<english::English, String>>,
    british: std::sync::OnceLock<Result<english::English, String>>,
}

impl KokoroVoice {
    /// The phonemizer of a voice: British for `b…` voices, American for `a…`.
    fn english(&self, voice: &str) -> Result<&english::English, String> {
        let british = match voice.trim().chars().next() {
            Some('a') => false,
            Some('b') => true,
            _ => {
                return Err(format!(
                    "voice {voice} does not speak English; the phonemizer is English (a… and b… voices): pass phonemes with `phonemes`"
                ))
            }
        };
        let (cell, repo) = if british {
            (&self.british, FALLBACK_GB)
        } else {
            (&self.american, FALLBACK_US)
        };
        cell.get_or_init(|| {
            english::English::load(british, fallback_dir(&self.dir, repo).as_deref())
        })
        .as_ref()
        .map_err(Clone::clone)
    }
}

/// The folder of a fallback model: next to the voice folder, else in the usual places.
fn fallback_dir(voice: &Path, repo: &str) -> Option<std::path::PathBuf> {
    let name = repo.rsplit('/').next().unwrap_or(repo);
    let sibling = voice.parent().map(|p| p.join(name));
    sibling
        .into_iter()
        .chain(crate::candidates(repo))
        .find(|dir| crate::is_model(dir))
}

/// A loaded voice. Load once and keep it. `Voice` is `Send + Sync`.
pub struct Voice {
    model: Model,
}

impl std::fmt::Debug for Voice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Voice")
            .field("kind", &self.kind())
            .field("sample_rate", &self.sample_rate())
            .finish_non_exhaustive()
    }
}

impl Voice {
    /// Loads a voice folder: Kokoro (`config.json` with `istftnet`, a
    /// `.pth` checkpoint, `voices/*.pt`) or VITS (`config.json`,
    /// `model.safetensors`, `vocab.json`, `tokenizer_config.json`). The
    /// weights stay f32.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let _team = crate::cpu::enter(SPREAD);
        let model = if kokoro::is_kokoro(dir) {
            Model::Kokoro(Box::new(KokoroVoice {
                model: Kokoro::load(dir)?,
                dir: dir.to_path_buf(),
                american: std::sync::OnceLock::new(),
                british: std::sync::OnceLock::new(),
            }))
        } else {
            Model::Vits {
                model: Box::new(Vits::load(dir)?),
                characters: Characters::open(dir)?,
            }
        };
        Ok(Self { model })
    }

    /// `kokoro` or `vits`.
    pub fn kind(&self) -> &'static str {
        match self.model {
            Model::Vits { .. } => "vits",
            Model::Kokoro(_) => "kokoro",
        }
    }

    /// The rate of the samples that `speak` returns (16000 for MMS-TTS,
    /// 24000 for Kokoro).
    pub fn sample_rate(&self) -> u32 {
        match &self.model {
            Model::Vits { model, .. } => model.sample_rate,
            Model::Kokoro(voice) => voice.model.sample_rate,
        }
    }

    /// The voices a Kokoro folder holds (`af_heart`, `bm_george`, ...);
    /// empty for a VITS voice, which has one.
    pub fn voices(&self) -> Vec<String> {
        match &self.model {
            Model::Vits { .. } => Vec::new(),
            Model::Kokoro(voice) => voice.model.voices(),
        }
    }

    /// What the voice reads of `text`. VITS: the text lowercase, only the
    /// characters of its vocabulary; what is missing is not spoken. Kokoro:
    /// the phonemes with the default voice (see `phonemes`).
    pub fn reads(&self, text: &str) -> String {
        match &self.model {
            Model::Vits { characters, .. } => characters.reads(text),
            Model::Kokoro(_) => self.phonemes(text, &Voicing::default()).unwrap_or_default(),
        }
    }

    /// The phonemes that a Kokoro voice speaks for `text`, one line per
    /// pass, with the accent of `voicing.voice`; a VITS voice gives `reads(text)`.
    pub fn phonemes(&self, text: &str, voicing: &Voicing) -> Result<String, String> {
        match &self.model {
            Model::Vits { characters, .. } => Ok(characters.reads(text)),
            Model::Kokoro(voice) => {
                if voicing.phonemes {
                    return Ok(text.trim().to_string());
                }
                let name = kokoro_voice(&voice.model, voicing)?;
                let chunks = voice.english(&name)?.chunks(text)?;
                Ok(chunks
                    .into_iter()
                    .map(|c| c.phonemes)
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
        }
    }

    /// The speech of `text` with `Voicing::default()`, mono at `sample_rate()`.
    pub fn speak(&self, text: &str) -> Vec<f32> {
        self.speak_with(text, &Voicing::default())
            .unwrap_or_default()
    }

    /// The speech of `text`, mono at `sample_rate()`, in -1..1. Fails on a
    /// speed that is not above 0, a noise below 0, or an unknown voice.
    pub fn speak_with(&self, text: &str, voicing: &Voicing) -> Result<Vec<f32>, String> {
        let mut out = Vec::new();
        self.speak_stream(text, voicing, &mut |part| {
            out.extend_from_slice(part);
            true
        })?;
        Ok(out)
    }

    /// `speak_with`, part by part: `on_part` gets the samples of each part
    /// (a sentence for VITS, a line or a 510-phoneme chunk for Kokoro) as
    /// soon as it is made, the pause before it included. Return false from
    /// it to stop.
    pub fn speak_stream(
        &self,
        text: &str,
        voicing: &Voicing,
        on_part: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<(), String> {
        let check = |name: &str, value: f32, above_zero: bool| -> Result<(), String> {
            let ok = value.is_finite()
                && if above_zero {
                    value > 0.0
                } else {
                    value >= 0.0
                };
            match (ok, above_zero) {
                (true, _) => Ok(()),
                (false, true) => Err(format!("{name} {value} must be above 0")),
                (false, false) => Err(format!("{name} {value} must be 0 or more")),
            }
        };
        let _team = crate::cpu::enter(SPREAD);
        let mut noise = Noise::new(voicing.seed);
        match &self.model {
            Model::Vits { model, characters } => {
                let defaults = model.defaults;
                let settings = Settings {
                    speed: voicing.speed.unwrap_or(defaults.speed),
                    noise: voicing.noise.unwrap_or(defaults.noise),
                    noise_duration: voicing.noise_duration.unwrap_or(defaults.noise_duration),
                };
                check("speed", settings.speed, true)?;
                check("noise", settings.noise, false)?;
                check("noise_duration", settings.noise_duration, false)?;
                if voicing.phonemes || voicing.voice.is_some() {
                    return Err(
                        "a VITS voice reads letters: it has no phoneme input and one voice".into(),
                    );
                }
                let pause = vec![0.0f32; (SENTENCE_PAUSE * model.sample_rate as f32) as usize];
                let mut first = true;
                for part in parts(text) {
                    let ids = characters.encode(part);
                    if ids.len() <= 1 {
                        continue;
                    }
                    let mut audio = if first { Vec::new() } else { pause.clone() };
                    audio.extend(model.speak(&ids, &settings, &mut noise));
                    first = false;
                    if !on_part(&audio) {
                        break;
                    }
                }
                Ok(())
            }
            Model::Kokoro(voice) => {
                let model = &voice.model;
                let speed = voicing.speed.unwrap_or(1.0);
                let scale = voicing.noise.unwrap_or(1.0);
                check("speed", speed, true)?;
                check("noise", scale, false)?;
                if voicing.noise_duration.is_some() {
                    return Err(
                        "noise_duration is a VITS setting; the timing of Kokoro has no noise"
                            .into(),
                    );
                }
                let name = kokoro_voice(model, voicing)?;
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
                    voice
                        .english(&name)?
                        .chunks(text)?
                        .into_iter()
                        .map(|c| c.phonemes)
                        .collect()
                };
                for phonemes in passes {
                    let audio = model.speak(&phonemes, &pack, speed, scale, &mut noise)?;
                    if !on_part(&audio) {
                        break;
                    }
                }
                Ok(())
            }
        }
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

/// The voice of `voicing`, else `af_heart`, else the first voice of the folder.
fn kokoro_voice(model: &Kokoro, voicing: &Voicing) -> Result<String, String> {
    if let Some(name) = &voicing.voice {
        return Ok(name.clone());
    }
    let voices = model.voices();
    if voices.iter().any(|v| v == DEFAULT_KOKORO_VOICE) {
        return Ok(DEFAULT_KOKORO_VOICE.to_string());
    }
    voices
        .into_iter()
        .next()
        .ok_or_else(|| "the voice folder has no voices/*.pt".to_string())
}

/// The phonemes of English `text` as Kokoro speaks them (misaki's result),
/// one line per line of `text`: American, or British with `british`. With
/// `fallback`, words that no dictionary holds get phonemes from the BART
/// (`PeterReid/graphemes_to_phonemes_en_us` or `_gb`, when found); else
/// they get none.
pub fn phonemize(text: &str, british: bool, fallback: bool) -> Result<String, String> {
    let repo = if british { FALLBACK_GB } else { FALLBACK_US };
    let dir = if fallback {
        crate::candidates(repo)
            .into_iter()
            .find(|d| crate::is_model(d))
    } else {
        None
    };
    let english = english::English::load(british, dir.as_deref())?;
    let lines: Result<Vec<String>, String> =
        text.lines().map(|line| english.phonemize(line)).collect();
    Ok(lines?.join("\n"))
}
