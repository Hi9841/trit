//! Transcript cleanup with Superwhisper S1-mini: a small text model that
//! removes fillers, resolves self-corrections, and writes punctuation,
//! casing, numbers, dates, and addresses. It works on the text of any
//! speech model. The prompt format is the one the model was trained on
//! (see its model card): a fixed system prompt, a control line, then the
//! raw transcript.

use crate::models::qwen3::{Prefix, Qwen3};
use crate::text::bpe::Bpe;
use std::path::Path;

/// The Hugging Face repository of the cleanup model.
pub const CLEANUP_REPO: &str = "superwhisper/s1-mini";

const SYSTEM: &str = "You are a text normalizer for speech-to-text transcripts. The input begins \
with a control line specifying the styling, structure, and context settings; clean the \
transcript to match those settings and output only the cleaned text.";

/// The register of the written text.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Styling {
    /// All lowercase, no apostrophes, colloquialisms kept.
    Casual,
    /// The speaker's phrasing; `I` capitalized; sentence starts lowercase.
    SemiCasual,
    /// Standard written English; contractions kept. The default.
    #[default]
    SemiFormal,
    /// Like semi-formal, with contractions expanded.
    Formal,
}

/// Whether enumerations may become a bulleted list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Structure {
    #[default]
    Prose,
    /// Three or more clear items become Markdown bullets.
    Lists,
}

/// Where the text goes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Context {
    #[default]
    General,
    /// Greeting line, body, and sign-off block.
    Email,
}

/// The control line of one cleanup. `Style::default()` is semi-formal prose
/// for general use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Style {
    pub styling: Styling,
    pub structure: Structure,
    pub context: Context,
}

impl Styling {
    pub const NAMES: [&str; 4] = ["casual", "semi-casual", "semi-formal", "formal"];

    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "casual" => Self::Casual,
            "semi-casual" => Self::SemiCasual,
            "semi-formal" => Self::SemiFormal,
            "formal" => Self::Formal,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        Self::NAMES[self as usize]
    }
}

impl Structure {
    pub const NAMES: [&str; 2] = ["prose", "lists"];

    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "prose" => Self::Prose,
            "lists" => Self::Lists,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        Self::NAMES[self as usize]
    }
}

impl Context {
    pub const NAMES: [&str; 2] = ["general", "email"];

    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "general" => Self::General,
            "email" => Self::Email,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        Self::NAMES[self as usize]
    }
}

/// Transcript tokens per call. The model card asks for inputs of up to
/// about 1000 tokens; a part this size leaves room for the output budget.
const PART_TOKENS: usize = 800;

/// The start of every prompt: the system turn and the opening of the user turn.
fn system_prefix() -> String {
    format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n")
}

/// A loaded cleanup model. Load once and keep it.
pub struct Cleaner {
    model: Qwen3,
    /// The system prompt, run once at load: every call starts with it.
    system: Prefix,
    bpe: Bpe,
    stop: Vec<usize>,
}

impl Cleaner {
    /// Loads an S1-mini folder (`config.json`, `model.safetensors`, `tokenizer.json`).
    pub fn load(dir: &Path) -> Result<Self, String> {
        Self::load_with(dir, &crate::LoadOptions::default())
    }

    /// `load`, with a choice of precision.
    pub fn load_with(dir: &Path, options: &crate::LoadOptions) -> Result<Self, String> {
        let _team = crate::cpu::enter(crate::Spread::OneCache);
        let bpe = Bpe::open(&dir.join("tokenizer.json"))?;
        let stop = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|token| bpe.id(token))
            .collect::<Vec<_>>();
        if stop.is_empty() {
            return Err(format!(
                "{} has no <|im_end|> token; it is not a chat model",
                dir.display()
            ));
        }
        let model = Qwen3::load(dir, options.precision)?;
        let system = model.prefix(&bpe.encode(&system_prefix()));
        Ok(Self {
            model,
            system,
            bpe,
            stop,
        })
    }

    /// The cleaned text of one raw transcript. Fillers alone give "". A
    /// transcript longer than `PART_TOKENS` is cleaned in parts cut at
    /// sentence ends (at word ends when it has none), as the model card asks.
    pub fn clean(&self, transcript: &str, style: &Style) -> String {
        let transcript = transcript.trim();
        if transcript.is_empty() {
            return String::new();
        }
        if self.bpe.encode(transcript).len() <= PART_TOKENS {
            return self.clean_part(transcript, style);
        }
        let parts: Vec<String> = self
            .parts(transcript)
            .iter()
            .map(|part| self.clean_part(part, style))
            .filter(|text| !text.is_empty())
            .collect();
        parts.join(" ")
    }

    /// `text` in pieces of at most `PART_TOKENS`: whole sentences, or whole
    /// words for a sentence that is longer alone.
    fn parts(&self, text: &str) -> Vec<String> {
        let mut units: Vec<&str> = Vec::new();
        let mut start = 0;
        for (i, c) in text.char_indices() {
            let next = text[i + c.len_utf8()..].chars().next();
            if matches!(c, '.' | '?' | '!') && next.is_none_or(char::is_whitespace) {
                units.push(text[start..i + c.len_utf8()].trim());
                start = i + c.len_utf8();
            }
        }
        units.push(text[start..].trim());
        let mut parts: Vec<String> = Vec::new();
        let mut current = String::new();
        let tokens = |s: &str| self.bpe.encode(s).len();
        for unit in units.into_iter().filter(|u| !u.is_empty()) {
            let words: Vec<&str> = if tokens(unit) > PART_TOKENS {
                unit.split_whitespace().collect()
            } else {
                vec![unit]
            };
            for piece in words {
                let joined = if current.is_empty() {
                    piece.to_string()
                } else {
                    format!("{current} {piece}")
                };
                if tokens(&joined) > PART_TOKENS && !current.is_empty() {
                    parts.push(std::mem::replace(&mut current, piece.to_string()));
                } else {
                    current = joined;
                }
            }
        }
        if !current.is_empty() {
            parts.push(current);
        }
        parts
    }

    fn clean_part(&self, transcript: &str, style: &Style) -> String {
        let _team = crate::cpu::enter(crate::Spread::OneCache);
        let control = format!(
            "[Styling: {}] [Structure: {}] [Context: {}]",
            style.styling.name(),
            style.structure.name(),
            style.context.name()
        );
        // The Qwen3 chat template with thinking off, as the model was trained.
        let prompt = format!(
            "{}{control}\n{transcript}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            system_prefix()
        );
        let ids = self.bpe.encode(&prompt);
        // Clean text is about as long as the transcript; lists and email
        // layout add a little.
        let budget = (transcript.len() / 2 + 64).min(1024);
        let out = self
            .model
            .generate(&ids, Some(&self.system), budget, &self.stop);
        self.bpe.decode(&out).trim().to_string()
    }
}
