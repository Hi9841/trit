//! Byte-level BPE (GPT-2 style) from a Hugging Face `tokenizer.json`, as
//! Whisper and Qwen use it: every byte has a printable stand-in character,
//! and a token is a string of stand-ins.

use std::collections::HashMap;
use std::path::Path;

pub struct Bpe {
    /// Token id to its bytes; special tokens keep their text.
    pieces: Vec<Vec<u8>>,
    special: Vec<bool>,
    ids: HashMap<String, usize>,
    /// Merge rank of each adjacent pair of symbols; lower merges first.
    merges: HashMap<(String, String), usize>,
    /// Special tokens as text, longest first, for `encode`.
    specials: Vec<(String, usize)>,
    split: Split,
}

/// The pre-tokenizer regex that cuts text into words before the merges.
#[derive(Clone, Copy)]
enum Split {
    /// GPT-2 and Whisper: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
    Gpt2,
    /// Qwen2 and Qwen3: contractions any case, `[^\r\n\p{L}\p{N}]?\p{L}+`,
    /// one digit at a time, ` ?[^\s\p{L}\p{N}]+[\r\n]*`, `\s*[\r\n]+`,
    /// `\s+(?!\S)`, `\s+`.
    Qwen,
}

impl Bpe {
    /// The tokenizer of a model folder: `tokenizer.json`, or the older
    /// `vocab.json`, `merges.txt`, and `tokenizer_config.json` (Qwen3-ASR).
    pub fn open_dir(dir: &Path) -> Result<Self, String> {
        let json = dir.join("tokenizer.json");
        if json.is_file() {
            return Self::open(&json);
        }
        let read = |name: &str| {
            std::fs::read_to_string(dir.join(name))
                .map_err(|e| format!("cannot read {}: {e}", dir.join(name).display()))
        };
        let vocab: serde_json::Value = serde_json::from_str(&read("vocab.json")?)
            .map_err(|e| format!("vocab.json is not JSON: {e}"))?;
        let merges = read("merges.txt")?;
        let config: serde_json::Value = serde_json::from_str(&read("tokenizer_config.json")?)
            .map_err(|e| format!("tokenizer_config.json is not JSON: {e}"))?;
        let added = config["added_tokens_decoder"]
            .as_object()
            .into_iter()
            .flatten()
            .filter_map(|(id, token)| {
                let content = token["content"].as_str()?.to_string();
                Some((
                    id.parse().ok()?,
                    content,
                    token["special"].as_bool().unwrap_or(true),
                ))
            })
            .collect();
        let merges = merges
            .lines()
            .filter(|line| !line.starts_with("#version"))
            .filter_map(|line| line.split_once(' '))
            .map(|(a, b)| (a.to_string(), b.to_string()));
        let qwen = config["tokenizer_class"]
            .as_str()
            .is_some_and(|class| class.starts_with("Qwen"));
        Self::build(
            vocab.as_object().ok_or("vocab.json is not a map")?,
            merges,
            added,
            if qwen { Split::Qwen } else { Split::Gpt2 },
        )
    }

    pub fn open(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("tokenizer.json is not JSON: {e}"))?;
        let vocab = json["model"]["vocab"]
            .as_object()
            .ok_or("tokenizer.json has no model.vocab")?;
        let added = json["added_tokens"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|added| {
                let id = added["id"].as_u64()? as usize;
                let content = added["content"].as_str()?.to_string();
                Some((id, content, added["special"].as_bool().unwrap_or(true)))
            })
            .collect();
        // Either ["a", "b"] or "a b".
        let merges = json["model"]["merges"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|merge| match merge {
                serde_json::Value::Array(parts) => parts
                    .first()
                    .and_then(|a| a.as_str())
                    .zip(parts.get(1).and_then(|b| b.as_str()))
                    .map(|(a, b)| (a.to_string(), b.to_string())),
                serde_json::Value::String(text) => text
                    .split_once(' ')
                    .map(|(a, b)| (a.to_string(), b.to_string())),
                _ => None,
            });
        let qwen = json["pre_tokenizer"]
            .to_string()
            .contains(r"[^\\r\\n\\p{L}\\p{N}]?\\p{L}+");
        Self::build(
            vocab,
            merges,
            added,
            if qwen { Split::Qwen } else { Split::Gpt2 },
        )
    }

    /// From the vocabulary (token as stand-ins to id), the merges in rank
    /// order, and the added tokens `(id, text, special)`.
    fn build(
        vocab: &serde_json::Map<String, serde_json::Value>,
        merges: impl Iterator<Item = (String, String)>,
        added: Vec<(usize, String, bool)>,
        split: Split,
    ) -> Result<Self, String> {
        let byte_of = unicode_to_byte();
        let mut entries: Vec<(usize, Vec<u8>, bool, String)> = Vec::with_capacity(vocab.len());
        for (token, id) in vocab {
            let id = id.as_u64().ok_or("the vocabulary has a bad id")? as usize;
            let bytes = token
                .chars()
                .map(|c| byte_of.get(&c).copied().unwrap_or(b'?'))
                .collect();
            entries.push((id, bytes, false, token.clone()));
        }
        let mut specials = Vec::new();
        for (id, content, special) in added {
            specials.push((content.clone(), id));
            entries.push((id, content.as_bytes().to_vec(), special, content));
        }
        specials.sort_by_key(|(text, _)| std::cmp::Reverse(text.len()));
        let size = entries.iter().map(|e| e.0 + 1).max().unwrap_or(0);
        let mut pieces = vec![Vec::new(); size];
        let mut special = vec![false; size];
        let mut ids = HashMap::with_capacity(size);
        for (id, bytes, is_special, text) in entries {
            pieces[id] = bytes;
            special[id] = is_special;
            ids.insert(text, id);
        }
        let mut ranks = HashMap::new();
        for (rank, pair) in merges.enumerate() {
            ranks.entry(pair).or_insert(rank);
        }
        Ok(Self {
            pieces,
            special,
            ids,
            merges: ranks,
            specials,
            split,
        })
    }

    /// The id of a token as written in the vocabulary, for example `<|en|>`.
    pub fn id(&self, token: &str) -> Option<usize> {
        self.ids.get(token).copied()
    }

    /// Text of `ids` without special tokens. Invalid UTF-8 (a character cut
    /// between tokens) becomes U+FFFD.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id < self.pieces.len() && !self.special[id] {
                bytes.extend_from_slice(&self.pieces[id]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Token ids of `text`. Special tokens written in the text (`<|im_start|>`)
    /// become their ids; the rest goes through the pre-tokenizer and merges.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let next = self
                .specials
                .iter()
                .filter_map(|(special, id)| {
                    rest.find(special.as_str())
                        .map(|at| (at, special.len(), *id))
                })
                .min_by_key(|(at, len, _)| (*at, std::cmp::Reverse(*len)));
            let (plain, special) = match next {
                Some((at, len, id)) => (&rest[..at], Some((len, id))),
                None => (rest, None),
            };
            for word in pretokenize(plain, self.split) {
                self.merge_word(word, &mut ids);
            }
            match special {
                Some((len, id)) => {
                    ids.push(id);
                    rest = &rest[plain.len() + len..];
                }
                None => break,
            }
        }
        ids
    }

    /// Token ids of `text` with no special tokens: `<|im_end|>` in the text
    /// stays characters.
    pub fn encode_plain(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        for word in pretokenize(text, self.split) {
            self.merge_word(word, &mut ids);
        }
        ids
    }

    fn merge_word(&self, word: &str, ids: &mut Vec<usize>) {
        let to_char = byte_to_unicode();
        let mut symbols: Vec<String> = word
            .bytes()
            .map(|b| to_char[b as usize].to_string())
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
        for symbol in symbols {
            if let Some(id) = self.ids.get(&symbol) {
                ids.push(*id);
            }
        }
    }
}

/// Splits `text` into the words the merges work on, as the regex of `split`
/// would: each step takes the first alternative that matches here.
fn pretokenize(text: &str, split: Split) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let at = |i: usize| chars.get(i).map(|(_, c)| *c);
    let offset = |i: usize| chars.get(i).map_or(text.len(), |(o, _)| *o);
    let letter = |c: char| c.is_alphabetic();
    let number = |c: char| c.is_numeric();
    let newline = |c: char| c == '\r' || c == '\n';
    let other = |c: char| !c.is_whitespace() && !letter(c) && !number(c);
    let mut words = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i].1;
        let start = i;
        // Contractions: 's 't 're 've 'm 'll 'd (Qwen: any case).
        if c == '\'' {
            let tail: String = chars[i + 1..chars.len().min(i + 3)]
                .iter()
                .map(|(_, c)| *c)
                .collect();
            let tail = match split {
                Split::Qwen => tail.to_lowercase(),
                Split::Gpt2 => tail,
            };
            let len = if ["re", "ve", "ll"].iter().any(|s| tail.starts_with(s)) {
                2
            } else if ["s", "t", "m", "d"].iter().any(|s| tail.starts_with(s)) {
                1
            } else {
                0
            };
            if len > 0 {
                i += 1 + len;
                words.push(&text[offset(start)..offset(i)]);
                continue;
            }
        }
        let run = |mut j: usize, f: &dyn Fn(char) -> bool| {
            while at(j).is_some_and(f) {
                j += 1;
            }
            j
        };
        let end = match split {
            Split::Gpt2 => {
                let lead = usize::from(c == ' ' && at(i + 1).is_some_and(|n| !n.is_whitespace()));
                let first = at(i + lead).unwrap_or(' ');
                if letter(first) {
                    Some(run(i + lead, &letter))
                } else if number(first) {
                    Some(run(i + lead, &number))
                } else if other(first) {
                    Some(run(i + lead, &other))
                } else {
                    None
                }
            }
            Split::Qwen => {
                // [^\r\n\p{L}\p{N}]?\p{L}+
                let lead = usize::from(!newline(c) && !letter(c) && !number(c));
                if at(i + lead).is_some_and(letter) {
                    Some(run(i + lead, &letter))
                } else if number(c) {
                    Some(i + 1)
                } else {
                    // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
                    let lead = usize::from(c == ' ');
                    if at(i + lead).is_some_and(other) {
                        Some(run(run(i + lead, &other), &newline))
                    } else {
                        None
                    }
                }
            }
        };
        i = match end {
            Some(end) => end,
            None => {
                // Whitespace. Qwen first takes `\s*[\r\n]+`: up to the last newline.
                let space_end = run(i, &|c: char| c.is_whitespace());
                let last_newline = (i..space_end).rev().find(|j| at(*j).is_some_and(newline));
                match (split, last_newline) {
                    (Split::Qwen, Some(nl)) => nl + 1,
                    // `\s+(?!\S)`: leave the last space for the next word.
                    _ if space_end < chars.len() && space_end - i > 1 => space_end - 1,
                    _ => space_end,
                }
            }
        };
        words.push(&text[offset(start)..offset(i)]);
    }
    words
}

/// GPT-2's `bytes_to_unicode`: printable bytes stand for themselves, the
/// other 68 map to U+0100 and up.
fn byte_to_unicode() -> [char; 256] {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut table = ['\0'; 256];
    let mut next = 0u32;
    for b in 0..=255u8 {
        table[b as usize] = if printable(b) {
            char::from(b)
        } else {
            next += 1;
            char::from_u32(255 + next).unwrap_or('?')
        };
    }
    table
}

fn unicode_to_byte() -> HashMap<char, u8> {
    byte_to_unicode()
        .iter()
        .enumerate()
        .map(|(b, c)| (*c, b as u8))
        .collect()
}
