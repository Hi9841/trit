//! Turns token ids back into text with the vocabulary in `tokenizer.json`,
//! and text into token ids for dictionary terms. The model uses Metaspace
//! pieces: `▁` marks the start of a word.

use std::collections::HashMap;
use std::path::Path;

pub const WORD_START: char = '\u{2581}';
/// Latin pieces of the Parakeet v3 vocabulary that common words of 23 other
/// languages use and common English words do not (`scripts/foreign_pieces.py`).
const FOREIGN_PIECES: &str = include_str!("foreign-pieces.txt");

pub struct Tokenizer {
    pieces: Vec<String>,
    special: Vec<bool>,
    ids: HashMap<String, usize>,
    /// Merge rank of each adjacent pair of pieces; lower merges first.
    merges: HashMap<(String, String), usize>,
}

impl Tokenizer {
    pub fn open(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("Could not read {}: {e}", path.display()))?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let json: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("tokenizer.json is not JSON: {e}"))?;
        let vocab = json["model"]["vocab"]
            .as_object()
            .ok_or("tokenizer.json has no model.vocab")?;
        let size = vocab
            .values()
            .filter_map(serde_json::Value::as_u64)
            .max()
            .map_or(0, |max| max as usize + 1);
        let mut pieces = vec![String::new(); size];
        let mut ids = HashMap::new();
        for (piece, id) in vocab {
            if let Some(id) = id.as_u64() {
                pieces[id as usize] = piece.clone();
                ids.insert(piece.clone(), id as usize);
            }
        }
        let mut special = vec![false; size];
        for added in json["added_tokens"].as_array().into_iter().flatten() {
            let id = added["id"].as_u64().unwrap_or(u64::MAX) as usize;
            if id < size && added["special"].as_bool() == Some(true) {
                special[id] = true;
            }
        }
        let mut merges = HashMap::new();
        for (rank, merge) in json["model"]["merges"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
        {
            // Either ["a", "b"] or "a b".
            let pair = match merge {
                serde_json::Value::Array(parts) => parts
                    .first()
                    .and_then(|a| a.as_str())
                    .zip(parts.get(1).and_then(|b| b.as_str())),
                serde_json::Value::String(joined) => joined.split_once(' '),
                _ => None,
            };
            if let Some((a, b)) = pair {
                merges.entry((a.to_string(), b.to_string())).or_insert(rank);
            }
        }
        Ok(Self {
            pieces,
            special,
            ids,
            merges,
        })
    }

    /// The id of a piece as written in the vocabulary, for example `<|en|>`.
    pub fn id(&self, piece: &str) -> Option<usize> {
        self.ids.get(piece).copied()
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// For each id, whether its piece begins a word. Special ids never do.
    pub fn word_starts(&self) -> Vec<bool> {
        self.pieces
            .iter()
            .zip(&self.special)
            .map(|(piece, special)| !special && piece.starts_with(WORD_START))
            .collect()
    }

    /// Ids of pieces that English does not use: Cyrillic, Greek, and
    /// accented Latin from the model's other languages, and the Latin pieces
    /// that only other languages use ("▁beaucoup", "▁bardzo"). TDT is English
    /// only, so the decoder never picks them.
    pub fn non_english(&self) -> Vec<usize> {
        let english = |c: char| c.is_ascii() || matches!(c, WORD_START | '£' | '€');
        let foreign: std::collections::HashSet<&str> = FOREIGN_PIECES.lines().collect();
        self.pieces
            .iter()
            .zip(&self.special)
            .enumerate()
            .filter(|(_, (piece, special))| {
                !**special && (!piece.chars().all(english) || foreign.contains(piece.as_str()))
            })
            .map(|(id, _)| id)
            .collect()
    }

    /// The piece of `id` as written in the vocabulary, with its word-start
    /// mark; empty for special and unknown ids.
    pub fn piece(&self, id: usize) -> &str {
        match self.special.get(id) {
            Some(false) => &self.pieces[id],
            _ => "",
        }
    }

    /// Characters in a piece, not counting the word-start mark.
    pub fn letters(&self, id: usize) -> usize {
        self.pieces.get(id).map_or(0, |piece| {
            piece.chars().filter(|c| *c != WORD_START).count()
        })
    }

    /// Unknown and special ids are skipped, as the reference tokenizer does.
    /// A byte-fallback piece (`<0x0A>`) adds its byte; the bytes join as UTF-8.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if self.special.get(id).copied().unwrap_or(true) {
                continue;
            }
            let piece = &self.pieces[id];
            let byte = piece
                .strip_prefix("<0x")
                .and_then(|hex| hex.strip_suffix('>'))
                .filter(|hex| hex.len() == 2)
                .and_then(|hex| u8::from_str_radix(hex, 16).ok());
            match byte {
                Some(byte) => bytes.push(byte),
                None => bytes.extend_from_slice(piece.replace(WORD_START, " ").as_bytes()),
            }
        }
        let text = String::from_utf8_lossy(&bytes);
        text.trim_start_matches(' ').to_string()
    }

    /// Token ids of `text` with byte-pair merges, or `None` when a piece is
    /// not in the vocabulary. Each word starts with `▁`.
    pub fn encode(&self, text: &str) -> Option<Vec<usize>> {
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            let mut symbols: Vec<String> = std::iter::once(WORD_START)
                .chain(word.chars())
                .map(String::from)
                .collect();
            loop {
                let best = symbols
                    .windows(2)
                    .enumerate()
                    .filter_map(|(i, pair)| {
                        self.merges
                            .get(&(pair[0].clone(), pair[1].clone()))
                            .map(|rank| (*rank, i))
                    })
                    .min();
                let Some((_, at)) = best else {
                    break;
                };
                let right = symbols.remove(at + 1);
                symbols[at].push_str(&right);
            }
            for symbol in &symbols {
                ids.push(*self.ids.get(symbol)?);
            }
        }
        Some(ids)
    }
}
