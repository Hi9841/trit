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
mod conv;
mod english;
mod kokoro;
mod lstm;
mod noise;
mod vits;

pub use kokoro::is_kokoro;
use kokoro::KokoroVoice;
use noise::Noise;
use std::path::Path;
use vits::VitsVoice;

/// The Hugging Face repository of the default voice (Kokoro-82M).
pub const VOICE_REPO: &str = crate::catalog::DEFAULT_VOICES[0].repo;
/// The voices `find_voice` looks for, in order: Kokoro, then MMS-TTS English.
pub const VOICE_REPOS: [&str; 2] = [VOICE_REPO, crate::catalog::DEFAULT_VOICES[1].repo];

/// The family name of the text-to-speech kinds of `CATALOG`.
pub(crate) const FAMILY_NAME: &str = "Text to speech";

/// The files of each text-to-speech kind of `CATALOG`: the two voices and
/// the phonemizer fallback.
pub(crate) const KIND_FILES: [(&str, &[&str]); 3] = [
    ("kokoro", kokoro::FILES),
    (
        "g2p",
        &["config.json", "generation_config.json", "model.safetensors"],
    ),
    (
        "vits",
        &[
            "config.json",
            "tokenizer_config.json",
            "vocab.json",
            "model.safetensors",
        ],
    ),
];

/// The files of the text-to-speech kind `kind`.
pub(crate) fn files_of(kind: &str) -> Option<&'static [&'static str]> {
    KIND_FILES
        .iter()
        .find(|(known, _)| *known == kind)
        .map(|(_, files)| *files)
}

/// The speeds a voice takes. The frames of a pass grow with `1 / speed`, and
/// so do the buffers of the vocoder: a speed of 0.000001 asked for more
/// memory than a computer has, and a request that fails ends the process.
pub(super) const SPEED_RANGE: (f32, f32) = (0.25, 4.0);
/// The noise scale a voice takes (`noise`).
pub(super) const NOISE_RANGE: (f32, f32) = (0.0, 4.0);

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
    /// it), not words. One line is one pass, at most `Kokoro::max_phonemes`.
    pub phonemes: bool,
}

/// The Hugging Face repositories of the fallback for unknown words.
const FALLBACK_US: &str = crate::catalog::PHONEMIZER_FALLBACK[0].repo;
const FALLBACK_GB: &str = crate::catalog::PHONEMIZER_FALLBACK[1].repo;
/// Each accent and the repository of its fallback.
pub const PHONEMIZER_FALLBACKS: [(&str, &str); 2] = [("us", FALLBACK_US), ("gb", FALLBACK_GB)];

/// Whether a Kokoro voice is British (`b…`) or American (`a…`), from the
/// first letter of its name; `None` for a voice that does not speak English.
pub(super) fn british(name: &str) -> Option<bool> {
    match name.chars().next() {
        Some('a') => Some(false),
        Some('b') => Some(true),
        _ => None,
    }
}

/// The accent of a Kokoro voice, from the first letters of its name.
pub fn voice_accent(name: &str) -> &'static str {
    match (british(name), name.chars().nth(1)) {
        (Some(false), Some('f')) => "American female",
        (Some(false), Some('m')) => "American male",
        (Some(true), Some('f')) => "British female",
        (Some(true), Some('m')) => "British male",
        _ => "not English (give phonemes)",
    }
}

/// The folder of the phonemizer fallback of `accent` (`us` or `gb`, as
/// `PHONEMIZER_FALLBACKS` names them) in the usual places, as `phonemize`
/// and a Kokoro voice find it; `None` for another accent.
pub fn find_phonemizer_fallback(accent: &str) -> Option<std::path::PathBuf> {
    let british = match accent {
        "us" => false,
        "gb" => true,
        _ => return None,
    };
    fallback_dir(None, british)
}

/// The folder of the fallback for unknown words of an accent: in `near`
/// first (the folder that holds the voice), else in the usual places.
pub(super) fn fallback_dir(near: Option<&Path>, british: bool) -> Option<std::path::PathBuf> {
    let repo = if british { FALLBACK_GB } else { FALLBACK_US };
    let name = repo.rsplit('/').next().unwrap_or(repo);
    near.map(|folder| folder.join(name))
        .into_iter()
        .chain(crate::candidates(repo))
        .find(|dir| crate::is_model(dir))
}

enum Model {
    Vits(Box<VitsVoice>),
    Kokoro(Box<KokoroVoice>),
}

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
        let voice = crate::cpu::store::in_turn(|| Self::load_now(dir))?;
        crate::release_memory();
        Ok(voice)
    }

    fn load_now(dir: &Path) -> Result<Self, String> {
        let _team = crate::cpu::enter(SPREAD);
        let model = if kokoro::is_kokoro(dir) {
            Model::Kokoro(Box::new(KokoroVoice::load(dir)?))
        } else {
            Model::Vits(Box::new(VitsVoice::load(dir)?))
        };
        Ok(Self { model })
    }

    pub fn kind(&self) -> &'static str {
        match self.model {
            Model::Vits(_) => "vits",
            Model::Kokoro(_) => "kokoro",
        }
    }

    /// What to tell a caller when this voice reads none of a text.
    pub fn reads_nothing_help(&self) -> &'static str {
        match &self.model {
            Model::Vits(voice) => voice.reads_nothing_help(),
            Model::Kokoro(voice) => voice.reads_nothing_help(),
        }
    }

    /// The rate of the samples that `speak` returns (16000 for MMS-TTS,
    /// 24000 for Kokoro).
    pub fn sample_rate(&self) -> u32 {
        match &self.model {
            Model::Vits(voice) => voice.sample_rate(),
            Model::Kokoro(voice) => voice.sample_rate(),
        }
    }

    /// The voices a Kokoro folder holds (`af_heart`, `bm_george`, ...);
    /// empty for a VITS voice, which has one.
    pub fn voices(&self) -> Vec<String> {
        match &self.model {
            Model::Vits(_) => Vec::new(),
            Model::Kokoro(voice) => voice.voices(),
        }
    }

    /// What the voice reads of `text`. VITS: the text lowercase, only the
    /// characters of its vocabulary; what is missing is not spoken. Kokoro:
    /// the phonemes with the default voice (see `phonemes`).
    pub fn reads(&self, text: &str) -> String {
        match &self.model {
            Model::Vits(voice) => voice.reads(text),
            Model::Kokoro(voice) => voice.reads(text),
        }
    }

    /// The phonemes that a Kokoro voice speaks for `text`, one line per
    /// pass, with the accent of `voicing.voice`; a VITS voice gives `reads(text)`.
    pub fn phonemes(&self, text: &str, voicing: &Voicing) -> Result<String, String> {
        match &self.model {
            Model::Vits(voice) => Ok(voice.reads(text)),
            Model::Kokoro(voice) => voice.phonemes(text, voicing),
        }
    }

    /// The speech of `text` with `Voicing::default()`, mono at `sample_rate()`.
    /// An error gives no samples; `speak_with` gives the error.
    pub fn speak(&self, text: &str) -> Vec<f32> {
        self.speak_with(text, &Voicing::default())
            .unwrap_or_default()
    }

    /// The speech of `text`, mono at `sample_rate()`, in -1..1. Fails on a
    /// speed outside 0.25 to 4, a noise outside 0 to 4, a `noise_duration`
    /// outside 0 to 2, or an unknown voice.
    pub fn speak_with(&self, text: &str, voicing: &Voicing) -> Result<Vec<f32>, String> {
        let mut out = Vec::new();
        self.speak_stream(text, voicing, &mut |part| {
            out.extend_from_slice(part);
            true
        })?;
        Ok(out)
    }

    /// `speak_with`, part by part: `on_part` gets the samples of each part
    /// (a sentence for VITS, a line or one phoneme-limit chunk for Kokoro) as
    /// soon as it is made, the pause before it included. Return false from
    /// it to stop.
    pub fn speak_stream(
        &self,
        text: &str,
        voicing: &Voicing,
        on_part: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<(), String> {
        let _team = crate::cpu::enter(SPREAD);
        let mut noise = Noise::new(voicing.seed);
        match &self.model {
            Model::Vits(voice) => voice.speak_stream(text, voicing, &mut noise, on_part),
            Model::Kokoro(voice) => voice.speak_stream(text, voicing, &mut noise, on_part),
        }
    }
}

/// `value` within `(low, high)`, or the error that names `name`.
pub(super) fn in_range(name: &str, value: f32, (low, high): (f32, f32)) -> Result<(), String> {
    // NaN fails both comparisons.
    if value >= low && value <= high {
        Ok(())
    } else {
        Err(format!("{name} {value} must be from {low} to {high}"))
    }
}

/// The phonemes of English `text` as Kokoro speaks them (misaki's result),
/// one line per line of `text`: American, or British with `british`. With
/// `fallback`, words that no dictionary holds get phonemes from the BART
/// (`PeterReid/graphemes_to_phonemes_en_us` or `_gb`, when found); else
/// they get none.
pub fn phonemize(text: &str, british: bool, fallback: bool) -> Result<String, String> {
    let dir = fallback.then(|| fallback_dir(None, british)).flatten();
    let english = english::English::load(british, dir.as_deref())?;
    let lines: Result<Vec<String>, String> =
        text.lines().map(|line| english.phonemize(line)).collect();
    Ok(lines?.join("\n"))
}
