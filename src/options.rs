//! What a caller asks of one transcription, and what it gets back.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// The most beams `Options::beam` gives. Each beam keeps its own decoder
/// caches: Whisper tiny with 10^8 beams held 14 GB within a minute, and
/// still grew.
const MAX_BEAMS: usize = 16;

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
    /// 1 is greedy, as the reference runtimes decode by default. More than
    /// 16 decode with 16.
    pub beam: usize,
    /// Write the speech as spoken (the default), or translate it into
    /// English (Whisper's multilingual models, Canary).
    pub task: Task,
    /// Stops the call early (see `Cancel`). `None` runs to the end.
    pub cancel: Option<Cancel>,
}

/// Stops a call that runs on another thread. Parakeet looks at it between
/// encoder blocks, between decoder steps, and between windows; the other
/// families between windows (30 s to 2 minutes of audio each). The call
/// then returns no text. The caller holds this value, so it knows that the
/// empty text is not silence. Clones share one flag; two values are equal
/// when they share it.
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stops the calls that hold this flag. A flag stays set: use a new one
    /// for the next call.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

impl PartialEq for Cancel {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Cancel {}

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

    pub(crate) fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(Cancel::is_cancelled)
    }

    /// `beam`, or `default` when it is 0; at most `MAX_BEAMS`.
    pub(crate) fn beams(&self, default: usize) -> usize {
        if self.beam == 0 {
            default
        } else {
            self.beam.min(MAX_BEAMS)
        }
    }

    /// Options for one language, for example `Options::language("en")`.
    pub fn language(code: &str) -> Self {
        Self {
            language: language_code(code),
            ..Self::default()
        }
    }
}

/// `code` as `Options::language` keeps it: trimmed, in lowercase, and `None`
/// when nothing is left.
pub fn language_code(code: &str) -> Option<String> {
    let code = code.trim().to_ascii_lowercase();
    (!code.is_empty()).then_some(code)
}

/// A stretch of the recording and its text. Times are seconds from the start.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

impl Segment {
    /// The segment as `trit transcribe --format json` and the C ABI write it.
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "start": hundredths(self.start),
            "end": hundredths(self.end),
            "text": self.text,
        })
    }
}

/// The text of `segments`, as the families write a transcript.
pub fn joined(segments: &[Segment]) -> String {
    let texts: Vec<&str> = segments.iter().map(|s| s.text.as_str()).collect();
    texts.join(" ")
}

/// Seconds to 0.01 s, so JSON shows 3.28 and not the f32 as f64 (3.2799999713897705).
pub fn hundredths(seconds: f32) -> f64 {
    (f64::from(seconds) * 100.0).round() / 100.0
}
