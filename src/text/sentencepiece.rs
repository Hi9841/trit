//! Turns token ids back into text with the vocabulary in `tokenizer.json`,
//! and text into token ids for dictionary terms. The model uses Metaspace
//! pieces: `▁` marks the start of a word.

use std::collections::HashMap;

pub const WORD_START: char = '\u{2581}';

pub struct Tokenizer {
    pieces: Vec<String>,
    special: Vec<bool>,
    ids: HashMap<String, usize>,
    /// Merge rank of each adjacent pair of pieces; lower merges first.
    merges: HashMap<(String, String), usize>,
}

impl Tokenizer {
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

    /// Ids of pieces with letters English does not use: Cyrillic, Greek, and
    /// accented Latin from the model's other languages. The rule reads only
    /// the characters of each piece; there is no list of words or pieces.
    pub fn non_english(&self) -> Vec<usize> {
        let english = |c: char| c.is_ascii() || matches!(c, WORD_START | '£' | '€');
        self.pieces
            .iter()
            .zip(&self.special)
            .enumerate()
            .filter(|(_, (piece, special))| !**special && !piece.chars().all(english))
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

/// A SentencePiece model (`tokenizer.model`, a protobuf, as a `.nemo`
/// archive holds it) as the `tokenizer.json` that `Tokenizer::parse` reads.
pub fn sentencepiece_json(bytes: &[u8]) -> Result<String, String> {
    let mut pieces = Vec::new();
    let mut scores = Vec::new();
    let mut kinds = Vec::new();
    for (field, value) in protobuf_fields(bytes)? {
        // ModelProto.pieces = 1.
        let Protobuf::Bytes(entry) = value else {
            continue;
        };
        if field != 1 {
            continue;
        }
        let (mut piece, mut score, mut kind) = (String::new(), 0.0f32, 1u64);
        for (field, value) in protobuf_fields(entry)? {
            match (field, value) {
                (1, Protobuf::Bytes(text)) => {
                    piece = String::from_utf8(text.to_vec())
                        .map_err(|_| "a piece is not UTF-8".to_string())?;
                }
                (2, Protobuf::Fixed32(bits)) => score = f32::from_bits(bits),
                (3, Protobuf::Varint(value)) => kind = value,
                _ => {}
            }
        }
        pieces.push(piece);
        scores.push(score);
        kinds.push(kind);
    }
    tokenizer_json(&pieces, &scores, &kinds)
}

/// A SentencePiece BPE vocabulary as a `tokenizer.json`: each piece, its
/// score, and its type (1 normal, 2 unknown, 3 control, 4 user defined,
/// 5 unused, 6 byte), as a SentencePiece model and GGUF's
/// `tokenizer.ggml.*` lists give them. Unknown and control pieces are
/// special. The merges are the ones Transformers derives (`generate_merges`
/// of `convert_slow_tokenizer.py`): every split of a piece into two pieces
/// of the vocabulary, the highest piece score first.
pub fn tokenizer_json(pieces: &[String], scores: &[f32], kinds: &[u64]) -> Result<String, String> {
    if pieces.is_empty() || scores.len() != pieces.len() || kinds.len() != pieces.len() {
        return Err("the tokenizer lists are empty or differ in length".into());
    }
    let special: Vec<bool> = kinds.iter().map(|kind| matches!(kind, 2 | 3)).collect();
    let ids: HashMap<&str, usize> = pieces
        .iter()
        .enumerate()
        .map(|(id, piece)| (piece.as_str(), id))
        .collect();
    let mut pairs: Vec<(&str, &str, f32)> = Vec::new();
    for ((piece, score), _) in pieces
        .iter()
        .zip(scores)
        .zip(&special)
        .filter(|(_, special)| !**special)
    {
        let mut local: Vec<(&str, &str, f32)> = piece
            .char_indices()
            .skip(1)
            .map(|(at, _)| (&piece[..at], &piece[at..], *score))
            .filter(|(left, right, _)| ids.contains_key(left) && ids.contains_key(right))
            .collect();
        local.sort_by_key(|(left, right, _)| (ids[left], ids[right]));
        pairs.extend(local);
    }
    // Python's stable sort with reverse=True on (score, len(left), len(right)).
    pairs.sort_by(|a, b| {
        let key = |p: &(&str, &str, f32)| (p.2, p.0.chars().count(), p.1.chars().count());
        let (ka, kb) = (key(a), key(b));
        kb.0.total_cmp(&ka.0)
            .then(kb.1.cmp(&ka.1))
            .then(kb.2.cmp(&ka.2))
    });
    let vocab: serde_json::Map<String, serde_json::Value> = pieces
        .iter()
        .enumerate()
        .map(|(id, piece)| (piece.clone(), serde_json::json!(id)))
        .collect();
    let merges: Vec<serde_json::Value> = pairs
        .iter()
        .map(|(left, right, _)| serde_json::json!([left, right]))
        .collect();
    let added: Vec<serde_json::Value> = pieces
        .iter()
        .enumerate()
        .filter(|(id, _)| special[*id])
        .map(|(id, piece)| serde_json::json!({ "id": id, "content": piece, "special": true }))
        .collect();
    Ok(serde_json::json!({
        "model": { "type": "BPE", "vocab": vocab, "merges": merges },
        "added_tokens": added,
    })
    .to_string())
}

/// One field of a protobuf message.
enum Protobuf<'a> {
    Varint(u64),
    Fixed32(u32),
    Bytes(&'a [u8]),
}

/// The fields of a protobuf message, in order. Unknown 64-bit fields are skipped.
fn protobuf_fields(mut bytes: &[u8]) -> Result<Vec<(u64, Protobuf<'_>)>, String> {
    fn varint(bytes: &mut &[u8]) -> Result<u64, String> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let (&byte, rest) = bytes.split_first().ok_or("the protobuf ends early")?;
            *bytes = rest;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("a protobuf varint is too long".into())
    }
    let mut fields = Vec::new();
    while !bytes.is_empty() {
        let key = varint(&mut bytes)?;
        let value = match key & 7 {
            0 => Protobuf::Varint(varint(&mut bytes)?),
            1 => {
                bytes = bytes.get(8..).ok_or("the protobuf ends early")?;
                continue;
            }
            2 => {
                let len = varint(&mut bytes)? as usize;
                let (data, rest) = bytes
                    .split_at_checked(len)
                    .ok_or("the protobuf ends early")?;
                bytes = rest;
                Protobuf::Bytes(data)
            }
            5 => {
                let (data, rest) = bytes
                    .split_first_chunk::<4>()
                    .ok_or("the protobuf ends early")?;
                bytes = rest;
                Protobuf::Fixed32(u32::from_le_bytes(*data))
            }
            other => return Err(format!("protobuf wire type {other} is not read")),
        };
        fields.push((key >> 3, value));
    }
    Ok(fields)
}
