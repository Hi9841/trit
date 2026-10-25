//! NLTK's averaged perceptron tagger for English (from textblob-aptagger,
//! MIT): Penn Treebank tags from a word, its neighbors, and the two tags
//! before it. The same features, normalization, and tie-break as NLTK's
//! `PerceptronTagger.tag`, so the tags are NLTK's.

use super::data;
use std::collections::HashMap;

pub struct Tagger {
    /// Feature to `(class, weight)`.
    weights: HashMap<String, Vec<(u16, f64)>>,
    tagdict: HashMap<String, String>,
    classes: Vec<String>,
}

const START: [&str; 2] = ["-START-", "-START2-"];
const END: [&str; 2] = ["-END-", "-END2-"];

impl Tagger {
    pub fn load() -> Result<Self, String> {
        let classes: Vec<String> = serde_json::from_slice(&data::inflate(data::TAGGER_CLASSES)?)
            .map_err(|e| format!("tagger classes: {e}"))?;
        let index: HashMap<&str, u16> = classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i as u16))
            .collect();
        let raw: HashMap<String, HashMap<String, f64>> =
            serde_json::from_slice(&data::inflate(data::TAGGER_WEIGHTS)?)
                .map_err(|e| format!("tagger weights: {e}"))?;
        let weights = raw
            .into_iter()
            .map(|(feature, labels)| {
                let labels = labels
                    .into_iter()
                    .filter_map(|(label, w)| index.get(label.as_str()).map(|i| (*i, w)))
                    .collect();
                (feature, labels)
            })
            .collect();
        let tagdict = serde_json::from_slice(&data::inflate(data::TAGGER_TAGDICT)?)
            .map_err(|e| format!("tagger tag dictionary: {e}"))?;
        Ok(Self {
            weights,
            tagdict,
            classes,
        })
    }

    /// One tag per word.
    pub fn tag(&self, words: &[&str]) -> Vec<String> {
        let mut context: Vec<String> = START.iter().map(|s| s.to_string()).collect();
        context.extend(words.iter().map(|w| normalize(w)));
        context.extend(END.iter().map(|s| s.to_string()));
        let (mut prev, mut prev2) = (START[0].to_string(), START[1].to_string());
        let mut out = Vec::with_capacity(words.len());
        for (i, word) in words.iter().enumerate() {
            let tag = match self.tagdict.get(*word) {
                Some(tag) if !tag.is_empty() => tag.clone(),
                _ => self.predict(&features(i, word, &context, &prev, &prev2)),
            };
            prev2 = std::mem::replace(&mut prev, tag.clone());
            out.push(tag);
        }
        out
    }

    fn predict(&self, features: &[String]) -> String {
        let mut scores = vec![0.0f64; self.classes.len()];
        for feature in features {
            if let Some(labels) = self.weights.get(feature) {
                for (label, weight) in labels {
                    scores[*label as usize] += weight;
                }
            }
        }
        // The highest score; a tie goes to the label that sorts last, as NLTK's max does.
        let mut best = 0;
        for i in 1..self.classes.len() {
            if (scores[i], &self.classes[i]) > (scores[best], &self.classes[best]) {
                best = i;
            }
        }
        self.classes[best].clone()
    }
}

/// NLTK's normalization: hyphenated words, years, other numbers, lowercase.
fn normalize(word: &str) -> String {
    let first = word.chars().next();
    if word.contains('-') && first != Some('-') {
        return "!HYPHEN".into();
    }
    if word.chars().count() == 4 && is_digits(word) {
        return "!YEAR".into();
    }
    if first.is_some_and(|c| c.is_numeric()) {
        return "!DIGITS".into();
    }
    word.to_lowercase()
}

pub fn is_digits(word: &str) -> bool {
    !word.is_empty() && word.chars().all(char::is_numeric)
}

/// The last three characters (Python's `word[-3:]`).
fn suffix(word: &str) -> &str {
    let start = word.char_indices().rev().nth(2).map_or(0, |(i, _)| i);
    &word[start..]
}

/// The features of word `i`, as NLTK's `_get_features` names them; each once.
fn features(i: usize, word: &str, context: &[String], prev: &str, prev2: &str) -> Vec<String> {
    let i = i + START.len();
    let first = word.chars().next().map(String::from).unwrap_or_default();
    vec![
        "bias".to_string(),
        format!("i suffix {}", suffix(word)),
        format!("i pref1 {first}"),
        format!("i-1 tag {prev}"),
        format!("i-2 tag {prev2}"),
        format!("i tag+i-2 tag {prev} {prev2}"),
        format!("i word {}", context[i]),
        format!("i-1 tag+i word {prev} {}", context[i]),
        format!("i-1 word {}", context[i - 1]),
        format!("i-1 suffix {}", suffix(&context[i - 1])),
        format!("i-2 word {}", context[i - 2]),
        format!("i+1 word {}", context[i + 1]),
        format!("i+1 suffix {}", suffix(&context[i + 1])),
        format!("i+2 word {}", context[i + 2]),
    ]
}
