//! Dictionary terms, boosted during greedy decoding (Canary, Moonshine).
//!
//! Each term is a path of token ids in a trie. At a decoding step, a token
//! that continues the term matched so far, or starts a new term, gets a
//! bonus on its logit, but only when the model already scores it within
//! `margin` of its best choice. Speech that does not contain the term keeps
//! its own words, because a gap larger than the margin is never closed.
//!
//! Short pieces such as "▁P" or "▁Car" begin thousands of ordinary words.
//! A term therefore starts only on a first piece with `min_start` letters,
//! only where the model itself starts a word, and a continuation is boosted
//! only once `min_prefix` letters of the term are matched.

use crate::text::sentencepiece::Tokenizer;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiasWeights {
    /// Bonus for the first token of a term.
    pub start: f32,
    /// Bonus for a token that continues a matched prefix.
    pub cont: f32,
    /// A token more than this far below the best logit gets no bonus.
    pub margin: f32,
    /// Letters a first piece needs before it may get the start bonus.
    pub min_start: usize,
    /// Letters of a term that must be matched before continuations get a bonus.
    pub min_prefix: usize,
    /// Parakeet TDT: how far the spelling heard may be from a term (0 the
    /// same, 1 nothing alike) for the sound check to run.
    pub distance: f32,
}

impl Default for BiasWeights {
    fn default() -> Self {
        Self {
            start: 4.0,
            cont: 8.0,
            margin: 10.0,
            min_start: 4,
            min_prefix: 3,
            distance: 0.45,
        }
    }
}

#[derive(Default)]
struct Node {
    children: HashMap<usize, usize>,
    /// Letters from the root to this node.
    letters: usize,
    /// Tokens from the root to this node.
    depth: usize,
    /// A term ends here; the index into `replacements`.
    ends: Option<usize>,
}

/// One path to boost, and the tokens to write instead when it completes.
pub struct Term {
    pub path: Vec<usize>,
    /// The term as the user wrote it, when this path spells it differently.
    pub written: Option<Vec<usize>>,
    /// Write `written` even when the model chose this path without help. True
    /// for splits and inner capitals ("web socket" -> "WebSocket"); false for a
    /// lowercase first letter, which may be an ordinary word ("vault").
    pub always: bool,
}

/// A term finished at the last emitted token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Completion {
    /// Emitted tokens that belong to the term.
    pub len: usize,
    replacement: usize,
}

pub struct Bias {
    nodes: Vec<Node>,
    replacements: Vec<(Option<Vec<usize>>, bool)>,
    startable: HashSet<usize>,
    /// Tokens that begin a word.
    word_start: Vec<bool>,
    pub weights: BiasWeights,
}

/// Position in the trie between decoding steps; 0 is the root.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BiasState(usize);

impl Bias {
    /// `terms` are token-id paths (empty ones are ignored); `letters[id]` and
    /// `word_start[id]` describe each token's piece.
    pub fn new(
        terms: impl IntoIterator<Item = Term>,
        letters: &[usize],
        word_start: Vec<bool>,
        weights: BiasWeights,
    ) -> Self {
        let letters_of = |token: usize| letters.get(token).copied().unwrap_or(0);
        let mut nodes = vec![Node::default()];
        let mut replacements = Vec::new();
        let mut startable = HashSet::new();
        for Term {
            path: term,
            written,
            always,
        } in terms
        {
            if term.is_empty() {
                continue;
            }
            if let Some(&first) = term.first() {
                if letters_of(first) >= weights.min_start {
                    startable.insert(first);
                }
            }
            let mut at = 0;
            for token in term {
                let next = nodes.len();
                let child = *nodes[at].children.entry(token).or_insert(next);
                if child == next {
                    let prefix = nodes[at].letters + letters_of(token);
                    let depth = nodes[at].depth + 1;
                    nodes.push(Node {
                        letters: prefix,
                        depth,
                        ..Node::default()
                    });
                }
                at = child;
            }
            if nodes[at].ends.is_none() {
                nodes[at].ends = Some(replacements.len());
                replacements.push((written, always));
            }
        }
        Self {
            nodes,
            replacements,
            startable,
            word_start,
            weights,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes[0].children.is_empty()
    }

    /// Adds the bonuses to `logits` (token part only) for the next step.
    pub fn apply(&self, state: BiasState, logits: &mut [f32]) {
        let (best_token, best) =
            logits
                .iter()
                .copied()
                .enumerate()
                .fold(
                    (0, f32::NEG_INFINITY),
                    |a, (i, v)| if v > a.1 { (i, v) } else { a },
                );
        let floor = best - self.weights.margin;
        let mut boost = |token: usize, bonus: f32| {
            if let Some(logit) = logits.get_mut(token) {
                if *logit >= floor {
                    *logit += bonus;
                }
            }
        };
        let node = &self.nodes[state.0];
        if state.0 != 0 && node.letters >= self.weights.min_prefix {
            for &token in node.children.keys() {
                boost(token, self.weights.cont);
            }
        }
        // A term starts only where the model itself starts a word, so a boost
        // never splits the word it is spelling.
        if !self.word_start.get(best_token).copied().unwrap_or(false) {
            return;
        }
        for &token in &self.startable {
            // Children of a matched prefix already got the continuation bonus.
            if state.0 == 0 || !node.children.contains_key(&token) {
                boost(token, self.weights.start);
            }
        }
    }

    /// The state after `token` was emitted, and the term it completes, if any.
    pub fn advance(&self, state: BiasState, token: usize) -> (BiasState, Option<Completion>) {
        let next = self.nodes[state.0]
            .children
            .get(&token)
            .or_else(|| self.nodes[0].children.get(&token));
        let Some(&next) = next else {
            return (BiasState(0), None);
        };
        let node = &self.nodes[next];
        let completion = node.ends.map(|replacement| Completion {
            len: node.depth,
            replacement,
        });
        let state = if node.children.is_empty() { 0 } else { next };
        (BiasState(state), completion)
    }

    /// The tokens to write for a completed term, when they differ from the
    /// path. `decisive` is whether a bonus changed a choice inside the term.
    pub fn replacement(&self, completion: Completion, decisive: bool) -> Option<&[usize]> {
        let (written, always) = &self.replacements[completion.replacement];
        written.as_deref().filter(|_| *always || decisive)
    }

    pub fn starts_word(&self, token: usize) -> bool {
        self.word_start.get(token).copied().unwrap_or(false)
    }
}

/// Spellings the model may produce for a term, each with whether to write
/// the term as given when that spelling is decoded.
///
/// - The term itself.
/// - All lowercase ("postgres" for "Postgres"). Rewritten only when the term
///   has capitals after its first letter ("PostgreSQL", "GitHub", "AWS"): a
///   difference in the first letter alone may be an ordinary word ("vault").
/// - CamelCase split into words ("web socket" for "WebSocket", "graph QL"
///   for "GraphQL"), lowercase except all-capital parts. Always rewritten.
pub fn variants(term: &str) -> Vec<(String, bool)> {
    let inner_capitals = term
        .split_whitespace()
        .any(|word| word.chars().skip(1).any(char::is_uppercase));
    let mut out = vec![(term.to_string(), false)];
    let lower = term.to_lowercase();
    if lower != term {
        out.push((lower, inner_capitals));
    }
    let split: Vec<String> = term
        .split_whitespace()
        .flat_map(|word| {
            let chars: Vec<char> = word.chars().collect();
            let mut parts = vec![String::new()];
            for (i, c) in chars.iter().enumerate() {
                let hump = i > 0 && c.is_uppercase() && chars[i - 1].is_lowercase();
                if hump {
                    parts.push(String::new());
                }
                if let Some(part) = parts.last_mut() {
                    part.push(*c);
                }
            }
            parts
        })
        .map(|part| {
            if part.chars().all(|c| !c.is_lowercase()) {
                part
            } else {
                part.to_lowercase()
            }
        })
        .collect();
    let split = split.join(" ");
    if split.contains(' ') && split.to_lowercase() != term.to_lowercase() {
        out.push((split, true));
    }
    out
}

/// The dictionary of `terms` for a SentencePiece model, with the terms it
/// cannot spell; `None` when no term is left.
pub fn build(
    tokenizer: &Tokenizer,
    terms: &[String],
    weights: BiasWeights,
) -> (Option<Bias>, Vec<String>) {
    let mut paths = Vec::new();
    let mut skipped = Vec::new();
    let mut seen = HashSet::new();
    for term in terms.iter().map(|t| t.trim()).filter(|t| !t.is_empty()) {
        let Some(written) = tokenizer.encode(term) else {
            skipped.push(term.to_string());
            continue;
        };
        for (variant, rewrite) in variants(term) {
            let Some(path) = tokenizer.encode(&variant) else {
                continue;
            };
            if seen.insert(path.clone()) {
                paths.push(Term {
                    written: (path != written).then(|| written.clone()),
                    always: rewrite,
                    path,
                });
            }
        }
    }
    let letters: Vec<usize> = (0..tokenizer.len())
        .map(|id| tokenizer.letters(id))
        .collect();
    let bias = Bias::new(paths, &letters, tokenizer.word_starts(), weights);
    ((!bias.is_empty()).then_some(bias), skipped)
}

/// A dictionary over one greedy decode of a text decoder (Canary,
/// Moonshine): `pick` boosts the logits and chooses the token, `push`
/// records it, `finish` gives the tokens with completed terms written as
/// the user wrote them. The same rules as Parakeet's decoder.
pub struct Run<'a> {
    bias: &'a Bias,
    matched: BiasState,
    /// Whether a bonus changed a choice in the current match.
    decisive: bool,
    /// A completed term waiting for the next word to start before it is rewritten.
    pending: Option<(std::ops::Range<usize>, Vec<usize>)>,
    tokens: Vec<usize>,
    own: usize,
}

impl<'a> Run<'a> {
    pub fn new(bias: &'a Bias) -> Self {
        Self {
            bias,
            matched: BiasState::default(),
            decisive: false,
            pending: None,
            tokens: Vec::new(),
            own: 0,
        }
    }

    /// The next token after the bonuses.
    pub fn pick(&mut self, logits: &mut [f32]) -> usize {
        self.own = argmax(logits);
        self.bias.apply(self.matched, logits);
        argmax(logits)
    }

    /// Records the token `pick` chose (end of text excluded).
    pub fn push(&mut self, token: usize) {
        // "rest" must not become "REST" inside "restart": a term is
        // rewritten only when the next word starts.
        if let Some((span, written)) = self.pending.take() {
            if self.bias.starts_word(token) {
                self.tokens.splice(span, written);
            }
        }
        let fresh = self.matched == BiasState::default();
        self.decisive = (self.decisive && !fresh) || token != self.own;
        self.tokens.push(token);
        let (next, completion) = self.bias.advance(self.matched, token);
        self.matched = next;
        if let Some(done) = completion {
            if let Some(written) = self.bias.replacement(done, self.decisive) {
                let end = self.tokens.len();
                self.pending = Some((end - done.len..end, written.to_vec()));
            }
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn finish(mut self) -> Vec<usize> {
        if let Some((span, written)) = self.pending.take() {
            self.tokens.splice(span, written);
        }
        self.tokens
    }
}

/// Index of the largest value; the first one wins a tie.
fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}
