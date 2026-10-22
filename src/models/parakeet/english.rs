//! An English word prior for greedy decoding. Parakeet v3 and Redux know 25
//! languages; on short, unclear, or accented speech they drift into other
//! languages or into words of none ("Trois, trente un", "Bunjora dan mai").
//! When the caller asks for English and `DRIFT_SHARE` of the words of a pass
//! are in no English word list, the window is decoded again: a token that
//! makes the current word a spelling no English word starts with then loses
//! `PENALTY` against the others. Clear speech keeps rare names; unclear
//! speech gets English words.

use crate::text::sentencepiece::{Tokenizer, WORD_START};
use parking_lot::RwLock;
use std::sync::OnceLock;

/// The lowercase words of SCOWL size 70, front-coded (`scripts/english_lists.py`).
const WORDS: &str = include_str!("../../text/english-words.txt");
/// A pass is decoded again when this share of its words is not English.
const DRIFT_SHARE: f32 = 0.4;
/// What a token that leaves English loses, in logits.
const PENALTY: f32 = 10.0;

pub struct English {
    /// Per token id: lowercase letters and digits of the piece.
    keys: Vec<String>,
    /// Per token id: the piece starts a word.
    starts: Vec<bool>,
    /// Per token id: the piece has a digit.
    digits: Vec<bool>,
    /// Per token id: the piece has a lowercase letter.
    lower: Vec<bool>,
    /// Per token id: the piece is an apostrophe inside a word ("I've").
    apostrophe: Vec<bool>,
    /// The words of the dictionary terms, sorted: English for this caller.
    terms: RwLock<Vec<String>>,
}

/// The word being decoded.
#[derive(Clone, Default)]
pub struct Partial {
    letters: String,
    digits: bool,
    lower: bool,
    /// No English word starts with the letters so far.
    left: bool,
}

impl English {
    pub fn new(tokenizer: &Tokenizer) -> Self {
        let pieces: Vec<&str> = (0..tokenizer.len()).map(|id| tokenizer.piece(id)).collect();
        Self {
            keys: pieces.iter().map(|p| key(p)).collect(),
            starts: pieces.iter().map(|p| p.starts_with(WORD_START)).collect(),
            digits: pieces
                .iter()
                .map(|p| p.chars().any(|c| c.is_ascii_digit()))
                .collect(),
            lower: pieces
                .iter()
                .map(|p| p.chars().any(char::is_lowercase))
                .collect(),
            apostrophe: pieces
                .iter()
                .map(|p| !p.is_empty() && p.chars().all(|c| matches!(c, '\'' | '\u{2019}')))
                .collect(),
            terms: RwLock::new(Vec::new()),
        }
    }

    /// Counts the words of `terms` as English.
    pub fn set_terms(&self, terms: &[String]) {
        let mut words: Vec<String> = terms
            .iter()
            .flat_map(|term| term.split_whitespace())
            .map(key)
            .collect();
        words.sort_unstable();
        words.dedup();
        *self.terms.write() = words;
    }

    /// Whether `DRIFT_SHARE` of the words of `tokens` are in no English
    /// word list: the model drifted into another language or into none.
    pub fn drifted(&self, tokens: &[usize]) -> bool {
        let (mut words, mut foreign) = (0usize, 0usize);
        let mut close = |word: &Partial| {
            if !word.letters.is_empty() && !word.free() {
                words += 1;
                if !self.is_word(&word.letters) {
                    foreign += 1;
                }
            }
        };
        let mut word = Partial::default();
        for &token in tokens {
            if (self.keys[token].is_empty() && !self.apostrophe[token]) || self.starts[token] {
                close(&word);
            }
            word = self.step(&word, token).0;
        }
        close(&word);
        foreign > 0 && foreign as f32 >= DRIFT_SHARE * words as f32
    }

    /// The token to emit instead of `best`: tokens within `PENALTY` of it
    /// compete, and each that leaves English loses `PENALTY`.
    pub fn pick(&self, word: &Partial, logits: &[f32], best: usize, blank: usize) -> usize {
        if best == blank || self.step(word, best).1 {
            return best;
        }
        let mut choice = (best, logits[best] - PENALTY);
        for (token, &logit) in logits.iter().enumerate() {
            if logit > choice.1 && token != best && (token == blank || self.step(word, token).1) {
                choice = (token, logit);
            }
        }
        choice.0
    }

    /// The word after `token`, and whether the token may follow without
    /// a penalty: the word stays a start of an English word, or it had
    /// already left English before this token (cutting it short with a
    /// blank would not make it English).
    pub fn step(&self, word: &Partial, token: usize) -> (Partial, bool) {
        let key = &self.keys[token];
        if key.is_empty() {
            // An apostrophe inside a word keeps it ("I've" is "ive");
            // other punctuation ends it.
            if self.apostrophe[token] {
                return (word.clone(), true);
            }
            return (Partial::default(), true);
        }
        let mut next = if self.starts[token] {
            Partial::default()
        } else {
            word.clone()
        };
        next.letters.push_str(key);
        next.digits |= self.digits[token];
        next.lower |= self.lower[token];
        let english = next.free() || self.starts_word(&next.letters);
        let gone = word.left && !self.starts[token];
        next.left = !english;
        (next, english || gone)
    }

    fn is_word(&self, letters: &str) -> bool {
        is_word(letters)
            || self
                .terms
                .read()
                .binary_search_by(|t| t.as_str().cmp(letters))
                .is_ok()
    }

    fn starts_word(&self, prefix: &str) -> bool {
        if lexicon()
            .first_from(prefix)
            .is_some_and(|w| w.starts_with(prefix))
        {
            return true;
        }
        let terms = self.terms.read();
        let at = terms.partition_point(|t| t.as_str() < prefix);
        terms.get(at).is_some_and(|t| t.starts_with(prefix))
    }
}

/// Whether `letters` (lowercase) is a word of the English word list.
pub fn is_word(letters: &str) -> bool {
    lexicon().first_from(letters) == Some(letters)
}

impl Partial {
    /// Numbers and acronyms ("GPUI", "CI") are not words to check.
    fn free(&self) -> bool {
        self.digits || (!self.lower && self.letters.len() >= 2)
    }
}

/// Lowercase letters and digits of `text`.
pub fn key(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// The English words, decoded once on first use: one string and the start
/// of each word.
struct Lexicon {
    text: String,
    starts: Vec<u32>,
}

impl Lexicon {
    fn word(&self, index: usize) -> &str {
        let end = self
            .starts
            .get(index + 1)
            .map_or(self.text.len() - 1, |&next| next as usize - 1);
        &self.text[self.starts[index] as usize..end]
    }

    /// The first word not before `key` in sort order.
    fn first_from(&self, key: &str) -> Option<&str> {
        let (mut low, mut high) = (0, self.starts.len());
        while low < high {
            let middle = (low + high) / 2;
            if self.word(middle) < key {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        (low < self.starts.len()).then(|| self.word(low))
    }
}

fn lexicon() -> &'static Lexicon {
    static LEXICON: OnceLock<Lexicon> = OnceLock::new();
    LEXICON.get_or_init(|| {
        let mut text = String::with_capacity(2 * WORDS.len());
        let mut starts = Vec::new();
        let mut previous = String::new();
        for line in WORDS.lines() {
            let mut chars = line.chars();
            let shared = chars.next().and_then(|c| c.to_digit(36)).unwrap_or(0) as usize;
            previous.truncate(shared);
            previous.push_str(chars.as_str());
            starts.push(text.len() as u32);
            text.push_str(&previous);
            text.push('\n');
        }
        Lexicon { text, starts }
    })
}
