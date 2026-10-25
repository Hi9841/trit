//! The character tokenizer of VITS and MMS-TTS (Transformers
//! `VitsTokenizer` without its phonemizer): one id per character of
//! `vocab.json`, and id 0 between characters when `add_blank` is set.
//!
//! The blank goes around every character, as the MMS training code puts it.
//! Transformers' tokenizer differs at the edges of a text: it takes the
//! blank's character ("k" in MMS-TTS) for a special token, and a "k" that
//! starts or ends the text loses its blanks ("kick": 7 ids, not 9).

use std::collections::HashMap;
use std::path::Path;

pub struct Characters {
    ids: HashMap<char, usize>,
    /// Lowercase the text and drop characters that are not in the vocabulary.
    normalize: bool,
    add_blank: bool,
}

impl Characters {
    /// `vocab.json` and `tokenizer_config.json` of a model folder. A model
    /// that reads phonemes (`phonemize: true`) fails: Trit has no phonemizer.
    pub fn open(dir: &Path) -> Result<Self, String> {
        let read = |name: &str| -> Result<serde_json::Value, String> {
            let path = dir.join(name);
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            serde_json::from_str(&text).map_err(|e| format!("{} is not JSON: {e}", path.display()))
        };
        let config = read("tokenizer_config.json")?;
        if config["phonemize"].as_bool() == Some(true) {
            return Err(format!(
                "{} reads phonemes (tokenizer_config.json: phonemize); Trit reads characters only",
                dir.display()
            ));
        }
        let vocab = read("vocab.json")?;
        let mut ids = HashMap::new();
        for (token, id) in vocab.as_object().ok_or("vocab.json is not an object")? {
            let mut chars = token.chars();
            let (Some(c), None, Some(id)) = (chars.next(), chars.next(), id.as_u64()) else {
                // Tokens longer than one character (`<unk>`) never match a character.
                continue;
            };
            ids.insert(c, id as usize);
        }
        Ok(Self {
            ids,
            normalize: config["normalize"].as_bool().unwrap_or(true),
            add_blank: config["add_blank"].as_bool().unwrap_or(true),
        })
    }

    /// The text the model reads: lowercase, only vocabulary characters, and
    /// trimmed, as `VitsTokenizer` prepares it.
    pub fn reads(&self, text: &str) -> String {
        if !self.normalize {
            return text.to_string();
        }
        let mut kept = String::with_capacity(text.len());
        for c in text.chars() {
            // A vocabulary character stays as it is (the reference keeps
            // an uppercase one); any other is lowercased, then must be one.
            if self.ids.contains_key(&c) {
                kept.push(c);
            } else {
                kept.extend(c.to_lowercase().filter(|l| self.ids.contains_key(l)));
            }
        }
        kept.trim().to_string()
    }

    /// The ids of `reads(text)`, with id 0 around and between characters
    /// when the model was trained so. Characters outside the vocabulary are
    /// dropped (with `normalize` off, too).
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let ids: Vec<usize> = self
            .reads(text)
            .chars()
            .filter_map(|c| self.ids.get(&c).copied())
            .collect();
        if !self.add_blank {
            return ids;
        }
        let mut out = vec![0; ids.len() * 2 + 1];
        for (i, id) in ids.into_iter().enumerate() {
            out[2 * i + 1] = id;
        }
        out
    }
}
