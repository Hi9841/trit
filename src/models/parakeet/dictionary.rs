//! Dictionary terms for the TDT decoder, checked against the sound.
//!
//! Greedy decoding runs without help. Then each run of words spelled close
//! to a term (a few letters apart, or the same letters split or cased
//! another way) is a candidate. The joint network scores the words it heard
//! and the term over the same frames (`Tdt::score`), and the term
//! replaces the words when it loses by less than `prior`. Speech that does
//! not sound like the term keeps its own words, however close the spelling.

use crate::models::parakeet::bias::BiasWeights;
use crate::text::sentencepiece::{Tokenizer, WORD_START};
use std::collections::HashMap;
use std::ops::Range;

/// The longest run of heard words that one term may replace.
const MAX_SPAN_WORDS: usize = 4;
/// Terms shorter than this replace only the same letters: "Go" must not
/// replace "so", whatever the sound.
const MIN_FUZZY_LETTERS: usize = 4;
/// The words heard have at least this share of the letters of the term,
/// and the term of theirs.
const MIN_LENGTH_RATIO: f32 = 0.7;
/// Terms tried on one run of words, the closest first.
const TERMS_PER_SPAN: usize = 2;
/// Sound checks per window, the closest spellings first.
const MAX_CANDIDATES: usize = 16;
/// The sound cannot tell "rust" from "Rust". A term that adds only a first
/// capital is written when the model gives the capital nearly the same
/// log-probability: "figma" becomes "Figma", but "go" stays "go".
pub const CAPITAL_PRIOR: f32 = 3.0;

pub struct Dictionary {
    terms: Vec<Term>,
    /// Per token id: the piece without its word-start mark.
    text: Vec<String>,
    /// Per token id: lowercase letters and digits of the piece.
    keys: Vec<Vec<char>>,
    starts: Vec<bool>,
    /// Log-probability a term may lose by to the words heard and still replace them.
    pub prior: f32,
    /// How far the spelling heard may be from a term (0 the same, 1 nothing
    /// alike) for a sound check.
    distance: f32,
}

struct Term {
    /// The term as the user wrote it.
    tokens: Vec<usize>,
    /// Spellings to score the sound with: as written, and lowercase, which
    /// the model writes more often ("trit runtime" for "Trit Runtime").
    spellings: Vec<Vec<usize>>,
    key: Vec<char>,
    /// The words as the user wrote them.
    words: Vec<String>,
}

/// Words heard that a term may replace.
pub struct Candidate {
    /// Token indices of the words heard.
    pub span: Range<usize>,
    term: usize,
    pub check: Check,
    distance: f32,
}

/// What decides whether a term replaces the words heard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Check {
    /// The same letters, cased or split the way only the term spells them
    /// ("web socket" for "WebSocket", "PNPM" for "pnpm"): always written.
    None,
    /// Only first capitals differ: `CAPITAL_PRIOR` against the term as written.
    Capital,
    /// Other spellings: `prior` against the best spelling of the term.
    Sound,
}

/// One word of the transcript: its tokens without trailing punctuation.
struct Word {
    tokens: Range<usize>,
    /// The next word starts right after these tokens: no punctuation between.
    joined: bool,
}

impl Dictionary {
    /// The dictionary of `terms`, with the terms the tokenizer cannot spell;
    /// `None` when no term is left.
    pub fn build(
        tokenizer: &Tokenizer,
        terms: &[String],
        weights: BiasWeights,
    ) -> (Option<Self>, Vec<String>) {
        let text: Vec<String> = (0..tokenizer.len())
            .map(|id| tokenizer.piece(id).replace(WORD_START, ""))
            .collect();
        let keys = text.iter().map(|piece| key(piece)).collect();
        // A later term with the same letters replaces an earlier one: the
        // user's "pnpm" wins over the built-in "PNPM".
        let mut seen: HashMap<Vec<char>, usize> = HashMap::new();
        let mut skipped = Vec::new();
        let mut entries: Vec<Term> = Vec::new();
        for term in terms.iter().map(|t| t.trim()).filter(|t| !t.is_empty()) {
            match tokenizer.encode(term) {
                Some(tokens) if !tokens.is_empty() => {
                    let mut spellings = vec![tokens.clone()];
                    if let Some(lower) = tokenizer.encode(&term.to_lowercase()) {
                        if lower != tokens {
                            spellings.push(lower);
                        }
                    }
                    let entry = Term {
                        spellings,
                        tokens,
                        key: key(term),
                        words: term.split_whitespace().map(str::to_string).collect(),
                    };
                    match seen.get(&entry.key) {
                        Some(&at) => entries[at] = entry,
                        None => {
                            seen.insert(entry.key.clone(), entries.len());
                            entries.push(entry);
                        }
                    }
                }
                _ => skipped.push(term.to_string()),
            }
        }
        let dictionary = Self {
            terms: entries,
            text,
            keys,
            starts: tokenizer.word_starts(),
            prior: weights.prior,
            distance: weights.distance,
        };
        (
            (!dictionary.terms.is_empty()).then_some(dictionary),
            skipped,
        )
    }

    /// Runs of words in `tokens` that a term may replace, closest first.
    pub fn candidates(&self, tokens: &[usize]) -> Vec<Candidate> {
        let words = self.words(tokens);
        let mut out = Vec::new();
        for first in 0..words.len() {
            let mut heard: Vec<char> = Vec::new();
            for last in first..words.len().min(first + MAX_SPAN_WORDS) {
                if last > first && !words[last - 1].joined {
                    break;
                }
                heard.extend(
                    words[last]
                        .tokens
                        .clone()
                        .flat_map(|i| self.keys[tokens[i]].clone()),
                );
                let span = words[first].tokens.start..words[last].tokens.end;
                let count = last - first + 1;
                // Words that spell one term are not another term: "rust"
                // is "Rust", not "REST".
                let spelled = self.terms.iter().any(|term| heard == term.key);
                for (index, term) in self.terms.iter().enumerate() {
                    // "trip run time" may be "Trit Runtime"; "data dog" "Datadog".
                    if count + 1 < term.words.len() || count > term.words.len() + 2 {
                        continue;
                    }
                    if tokens[span.clone()] == term.tokens[..] {
                        continue;
                    }
                    let candidate = if heard == term.key {
                        match self.same_letters(tokens, &words[first..=last], term) {
                            Some(check) => Candidate {
                                span: span.clone(),
                                term: index,
                                check,
                                distance: 0.0,
                            },
                            None => continue,
                        }
                    } else {
                        // A word heard wrong still starts with the sound of
                        // the term: "Dino" for "Deno", "Cotlin" for "Kotlin".
                        if spelled
                            || term.key.len() < MIN_FUZZY_LETTERS
                            || swap(heard[0], term.key[0]) >= 1.0
                        {
                            continue;
                        }
                        // The term with an ending is another word: "notions"
                        // is not "Notion".
                        if inflected(&heard, &term.key) {
                            continue;
                        }
                        // About as long as the term: "to" is never "Tokio".
                        let longest = heard.len().max(term.key.len()) as f32;
                        if (heard.len().min(term.key.len()) as f32) < MIN_LENGTH_RATIO * longest {
                            continue;
                        }
                        let distance = spelling_distance(&heard, &term.key) / longest;
                        if distance > self.distance {
                            continue;
                        }
                        Candidate {
                            span: span.clone(),
                            term: index,
                            check: Check::Sound,
                            distance,
                        }
                    };
                    out.push(candidate);
                }
            }
        }
        out.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        // Only the closest run of words for each term ("promises", not also
        // "promises the"), and the closest terms for each run.
        let mut kept: Vec<Candidate> = Vec::new();
        for candidate in out {
            let overlaps = |other: &&Candidate| {
                other.span.start < candidate.span.end && candidate.span.start < other.span.end
            };
            let same_term = kept
                .iter()
                .filter(overlaps)
                .any(|k| k.term == candidate.term);
            let same_span = kept.iter().filter(|k| k.span == candidate.span).count();
            if !same_term && same_span < TERMS_PER_SPAN && kept.len() < MAX_CANDIDATES {
                kept.push(candidate);
            }
        }
        kept
    }

    /// Every piece the terms are spelled with.
    pub fn pieces(&self) -> std::collections::HashSet<usize> {
        self.terms
            .iter()
            .flat_map(|term| term.spellings.iter().flatten().copied())
            .collect()
    }

    /// The tokens of a candidate's term as the user wrote it.
    pub fn written(&self, candidate: &Candidate) -> &[usize] {
        &self.terms[candidate.term].tokens
    }

    /// The spellings of a candidate's term to score the sound with.
    pub fn spellings(&self, candidate: &Candidate) -> &[Vec<usize>] {
        &self.terms[candidate.term].spellings
    }

    /// How to check heard words with the same letters as `term`; `None` to
    /// keep the words.
    fn same_letters(&self, tokens: &[usize], words: &[Word], term: &Term) -> Option<Check> {
        let heard: Vec<String> = words
            .iter()
            .map(|w| {
                w.tokens
                    .clone()
                    .map(|i| self.text[tokens[i]].as_str())
                    .collect()
            })
            .collect();
        let inner_capitals = |word: &str| word.chars().skip(1).any(char::is_uppercase);
        if heard.len() != term.words.len()
            || heard.iter().zip(&term.words).any(|(h, w)| key(h) != key(w))
        {
            // Split or joined another way. "web socket" is "WebSocket", but
            // "ray cast" may be words and not "Raycast": the sound decides.
            let camel = term.words.iter().any(|w| inner_capitals(w));
            return Some(if camel { Check::None } else { Check::Sound });
        }
        let mut check = None;
        for (heard, written) in heard.iter().zip(&term.words) {
            if heard == written {
                continue;
            }
            let (h, w) = (heard.chars().skip(1), written.chars().skip(1));
            if !h.eq(w) || inner_capitals(written) {
                // "PNPM" for "pnpm", "Graphql" for "GraphQL".
                check = Some(Check::None);
                continue;
            }
            // Only the first letter differs. Never lowercase it: it may
            // start a sentence.
            if !written.chars().next().is_some_and(char::is_uppercase) {
                return None;
            }
            check = check.or(Some(Check::Capital));
        }
        check
    }

    /// The words of `tokens`, each without its trailing punctuation.
    fn words(&self, tokens: &[usize]) -> Vec<Word> {
        let mut words: Vec<Word> = Vec::new();
        // The last word may still grow: no piece that starts a word came
        // after it. A quote that starts a word ("▁'") ends it too.
        let mut open = false;
        for (i, &token) in tokens.iter().enumerate() {
            let letters = !self.keys[token].is_empty();
            if self.starts.get(token).copied().unwrap_or(false) {
                open = false;
            }
            match words.last_mut() {
                // Letters after punctuation without a space ("Node.js") join the word.
                Some(word) if open && letters => {
                    word.tokens.end = i + 1;
                    word.joined = true;
                }
                Some(word) if open => word.joined = false,
                _ if letters => {
                    words.push(Word {
                        tokens: i..i + 1,
                        joined: true,
                    });
                    open = true;
                }
                _ => {
                    if let Some(word) = words.last_mut() {
                        word.joined = false;
                    }
                }
            }
        }
        words
    }
}

/// Lowercase letters and digits of `text`.
fn key(text: &str) -> Vec<char> {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// `heard` is `term` with an English ending: "notions" for "Notion". A
/// doubled last "s" is a misspelling, not a plural: "postgress".
fn inflected(heard: &[char], term: &[char]) -> bool {
    match heard.strip_prefix(term) {
        Some(['s']) => term.last() != Some(&'s'),
        Some(['e', 's'] | ['e', 'd'] | ['i', 'n', 'g'] | ['e', 'r'] | ['e', 'r', 's']) => true,
        _ => false,
    }
}

/// Edit distance where letters that sound alike cost half: vowels for each
/// other, pairs such as c and k, and a vowel or h in or out.
fn spelling_distance(a: &[char], b: &[char]) -> f32 {
    let gap = |c: char| {
        if matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y' | 'h' | 'w') {
            0.5
        } else {
            1.0
        }
    };
    let mut previous: Vec<f32> = std::iter::once(0.0)
        .chain(b.iter().scan(0.0, |total, &c| {
            *total += gap(c);
            Some(*total)
        }))
        .collect();
    for &x in a {
        let mut current = vec![previous[0] + gap(x)];
        for (j, &y) in b.iter().enumerate() {
            let best = (previous[j] + swap(x, y))
                .min(previous[j + 1] + gap(x))
                .min(current[j] + gap(y));
            current.push(best);
        }
        previous = current;
    }
    previous[b.len()]
}

/// The cost of `x` where `y` is spelled: 0 the same, 0.5 alike, 1 apart.
fn swap(x: char, y: char) -> f32 {
    if x == y {
        return 0.0;
    }
    let vowel = |c: char| matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y');
    const ALIKE: [(char, char); 11] = [
        ('c', 'k'),
        ('c', 's'),
        ('k', 'q'),
        ('s', 'z'),
        ('d', 't'),
        ('b', 'p'),
        ('f', 'v'),
        ('m', 'n'),
        ('g', 'j'),
        ('g', 'k'),
        ('x', 's'),
    ];
    if (vowel(x) && vowel(y)) || ALIKE.contains(&(x, y)) || ALIKE.contains(&(y, x)) {
        0.5
    } else {
        1.0
    }
}
