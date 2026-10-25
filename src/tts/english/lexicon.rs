//! misaki's `Lexicon`: words to phonemes from the gold and silver
//! dictionaries, with misaki's special cases, the -s, -ed, and -ing stems,
//! numbers, currency, and stress. Function for function as `misaki/en.py`
//! (0.9.4) writes it; the comments name the Python function.

use super::numbers;
use super::tokens::Token;
use super::{data, Context};
use std::collections::HashMap;

pub const PRIMARY: char = 'ˈ';
pub const SECONDARY: char = 'ˌ';
pub const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
pub const CONSONANTS: &str = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const US_TAUS: &str = "AIOWYiuæɑəɛɪɹʊʌ";
const DIPHTHONGS: &str = "AIOQWYʤʧ";
pub const PUNCTS: &str = ";:,.!?—…\"“”";
pub const NON_QUOTE_PUNCTS: &str = ";:,.!?—…";
const ORDINAL_SUFFIXES: [&str; 4] = ["st", "nd", "rd", "th"];
const CURRENCIES: [(&str, (&str, &str)); 3] = [
    ("$", ("dollar", "cent")),
    ("£", ("pound", "pence")),
    ("€", ("euro", "cent")),
];
const ADD_SYMBOLS: [(&str, &str); 2] = [(".", "dot"), ("/", "slash")];
const SYMBOLS: [(&str, &str); 4] = [("%", "percent"), ("&", "and"), ("+", "plus"), ("@", "at")];

/// A dictionary entry: phonemes, or phonemes by tag (a tag may map to none).
#[derive(Clone, Debug)]
pub enum Entry {
    Word(String),
    Tagged(HashMap<String, Option<String>>),
}

pub type Found = (Option<String>, Option<i32>);

pub struct Lexicon {
    golds: HashMap<String, Entry>,
    silvers: HashMap<String, Entry>,
    british: bool,
}

/// `str.isalpha()`, `lower()`, `upper()`, `capitalize()`, as Python defines them.
pub fn is_alpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(char::is_alphabetic)
}
pub fn lower(s: &str) -> String {
    s.to_lowercase()
}
pub fn upper(s: &str) -> String {
    s.to_uppercase()
}
pub fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first
            .to_uppercase()
            .chain(chars.flat_map(char::to_lowercase))
            .collect(),
        None => String::new(),
    }
}
pub fn is_digit(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}
fn count(s: &str) -> usize {
    s.chars().count()
}
/// `s[i:j]` in characters.
fn slice(s: &str, from: usize, to: usize) -> String {
    s.chars().skip(from).take(to.saturating_sub(from)).collect()
}

/// `grow_dictionary`: a lowercase word also as Capitalized, a Capitalized word also lowercase.
fn grow(d: HashMap<String, Entry>) -> HashMap<String, Entry> {
    let mut grown: HashMap<String, Entry> = HashMap::new();
    for (k, v) in &d {
        if count(k) < 2 {
            continue;
        }
        if *k == lower(k) {
            if *k != capitalize(k) {
                grown.insert(capitalize(k), v.clone());
            }
        } else if *k == capitalize(&lower(k)) {
            grown.insert(lower(k), v.clone());
        }
    }
    grown.extend(d);
    grown
}

/// `apply_stress`.
pub fn apply_stress(ps: Option<String>, stress: Option<f64>) -> Option<String> {
    let ps = ps?;
    let Some(stress) = stress else {
        return Some(ps);
    };
    let has = |c: char| ps.contains(c);
    let any_vowel = ps.chars().any(|c| VOWELS.contains(c));
    let restress = |ps: String| -> String {
        // Each stress mark moves to just before the next vowel.
        let chars: Vec<char> = ps.chars().collect();
        let mut keyed: Vec<(f64, char)> = chars
            .iter()
            .enumerate()
            .map(|(i, c)| (i as f64, *c))
            .collect();
        for (i, c) in chars.iter().enumerate() {
            if *c == PRIMARY || *c == SECONDARY {
                if let Some(j) = chars[i..].iter().position(|v| VOWELS.contains(*v)) {
                    keyed[i].0 = (i + j) as f64 - 0.5;
                }
            }
        }
        // Python sorts the (position, character) pairs: a tie goes by the character.
        keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        keyed.into_iter().map(|(_, c)| c).collect()
    };
    Some(if stress < -1.0 {
        ps.replace([PRIMARY, SECONDARY], "")
    } else if stress == -1.0 || ((stress == 0.0 || stress == -0.5) && has(PRIMARY)) {
        ps.replace(SECONDARY, "")
            .replace(PRIMARY, &SECONDARY.to_string())
    } else if (stress == 0.0 || stress == 0.5 || stress == 1.0) && !has(PRIMARY) && !has(SECONDARY)
    {
        if !any_vowel {
            return Some(ps);
        }
        restress(format!("{SECONDARY}{ps}"))
    } else if stress >= 1.0 && !has(PRIMARY) && has(SECONDARY) {
        ps.replace(SECONDARY, &PRIMARY.to_string())
    } else if stress > 1.0 && !has(PRIMARY) && !has(SECONDARY) {
        if !any_vowel {
            return Some(ps);
        }
        restress(format!("{PRIMARY}{ps}"))
    } else {
        ps
    })
}

/// `stress_weight`.
pub fn stress_weight(ps: &str) -> usize {
    ps.chars()
        .map(|c| if DIPHTHONGS.contains(c) { 2 } else { 1 })
        .sum()
}

fn parent_tag(tag: Option<&str>) -> Option<String> {
    let tag = tag?;
    Some(
        if tag.starts_with("VB") {
            "VERB"
        } else if tag.starts_with("NN") {
            "NOUN"
        } else if tag.starts_with("ADV") || tag.starts_with("RB") {
            "ADV"
        } else if tag.starts_with("ADJ") || tag.starts_with("JJ") {
            "ADJ"
        } else {
            tag
        }
        .to_string(),
    )
}

impl Lexicon {
    pub fn load(british: bool) -> Result<Self, String> {
        let read = |bytes: &[u8]| -> Result<HashMap<String, Entry>, String> {
            let value: serde_json::Value = serde_json::from_slice(&data::inflate(bytes)?)
                .map_err(|e| format!("lexicon: {e}"))?;
            let serde_json::Value::Object(map) = value else {
                return Err("lexicon: not an object".into());
            };
            Ok(map
                .into_iter()
                .filter_map(|(word, v)| {
                    let entry = match v {
                        serde_json::Value::String(s) => Entry::Word(s),
                        serde_json::Value::Object(tags) => Entry::Tagged(
                            tags.into_iter()
                                .map(|(tag, p)| (tag, p.as_str().map(str::to_string)))
                                .collect(),
                        ),
                        _ => return None,
                    };
                    Some((word, entry))
                })
                .collect())
        };
        let (gold, silver) = if british {
            (data::GB_GOLD, data::GB_SILVER)
        } else {
            (data::US_GOLD, data::US_SILVER)
        };
        Ok(Self {
            golds: grow(read(gold)?),
            silvers: grow(read(silver)?),
            british,
        })
    }

    fn gold_word(&self, word: &str) -> Option<String> {
        match self.golds.get(word)? {
            Entry::Word(w) => Some(w.clone()),
            Entry::Tagged(t) => t.get("DEFAULT").cloned().flatten(),
        }
    }

    /// `get_NNP`: the word spelled letter by letter.
    fn nnp(&self, word: &str) -> Found {
        let mut ps = String::new();
        for c in word.chars().filter(|c| c.is_alphabetic()) {
            match self.golds.get(&upper(&c.to_string())) {
                Some(Entry::Word(p)) => ps.push_str(p),
                _ => return (None, None),
            }
        }
        let ps = apply_stress(Some(ps), Some(0.0)).unwrap_or_default();
        let ps = match ps.rfind(SECONDARY) {
            Some(i) => format!("{}{PRIMARY}{}", &ps[..i], &ps[i + SECONDARY.len_utf8()..]),
            None => ps,
        };
        (Some(ps), Some(3))
    }

    /// `get_special_case`.
    fn special(&self, word: &str, tag: &str, stress: Option<f64>, ctx: &Context) -> Found {
        let symbol =
            |list: &[(&str, &'static str)]| list.iter().find(|(k, _)| *k == word).map(|(_, v)| *v);
        if tag == "ADD" {
            if let Some(name) = symbol(&ADD_SYMBOLS) {
                return self.lookup(name, None, Some(-0.5), Some(ctx));
            }
        }
        if let Some(name) = symbol(&SYMBOLS) {
            return self.lookup(name, None, None, Some(ctx));
        }
        let stripped = word.trim_matches('.');
        let undotted: String = word.chars().filter(|c| *c != '.').collect();
        if stripped.contains('.')
            && is_alpha(&undotted)
            && word.split('.').map(count).max().unwrap_or(0) < 3
        {
            return self.nnp(word);
        }
        let gold = |w: &str| self.gold_word(w);
        match word {
            "a" | "A" => {
                return (
                    Some(if tag == "DT" {
                        "ɐ".into()
                    } else {
                        "ˈA".into()
                    }),
                    Some(4),
                )
            }
            "am" | "Am" | "AM" => {
                if tag.starts_with("NN") {
                    return self.nnp(word);
                }
                if ctx.future_vowel.is_none() || word != "am" || stress.is_some_and(|s| s > 0.0) {
                    return (gold("am"), Some(4));
                }
                return (Some("ɐm".into()), Some(4));
            }
            "an" | "An" | "AN" => {
                if word == "AN" && tag.starts_with("NN") {
                    return self.nnp(word);
                }
                return (Some("ɐn".into()), Some(4));
            }
            "I" if tag == "PRP" => return (Some(format!("{SECONDARY}I")), Some(4)),
            "by" | "By" | "BY" if parent_tag(Some(tag)).as_deref() == Some("ADV") => {
                return (Some("bˈI".into()), Some(4));
            }
            _ => {}
        }
        if matches!(word, "to" | "To") || (word == "TO" && matches!(tag, "TO" | "IN")) {
            let ps = match ctx.future_vowel {
                None => gold("to"),
                Some(false) => Some("tə".into()),
                Some(true) => Some("tʊ".into()),
            };
            return (ps, Some(4));
        }
        if matches!(word, "in" | "In") || (word == "IN" && tag != "NNP") {
            let mark = if ctx.future_vowel.is_none() || tag != "IN" {
                PRIMARY.to_string()
            } else {
                String::new()
            };
            return (Some(format!("{mark}ɪn")), Some(4));
        }
        if matches!(word, "the" | "The") || (word == "THE" && tag == "DT") {
            return (
                Some(if ctx.future_vowel == Some(true) {
                    "ði".into()
                } else {
                    "ðə".into()
                }),
                Some(4),
            );
        }
        if tag == "IN" && matches!(lower(word).as_str(), "vs" | "vs.") {
            return self.lookup("versus", None, None, Some(ctx));
        }
        if matches!(word, "used" | "Used" | "USED") {
            let key = if matches!(tag, "VBD" | "JJ") && ctx.future_to {
                "VBD"
            } else {
                "DEFAULT"
            };
            if let Some(Entry::Tagged(t)) = self.golds.get("used") {
                return (t.get(key).cloned().flatten(), Some(4));
            }
        }
        (None, None)
    }

    /// `is_known`.
    fn is_known(&self, word: &str) -> bool {
        if self.golds.contains_key(word)
            || SYMBOLS.iter().any(|(k, _)| *k == word)
            || self.silvers.contains_key(word)
        {
            return true;
        }
        if !is_alpha(word)
            || !word
                .chars()
                .all(|c| c == '\'' || c == '-' || c.is_ascii_alphabetic())
        {
            return false;
        }
        if count(word) == 1 {
            return true;
        }
        if word == upper(word) && self.golds.contains_key(&lower(word)) {
            return true;
        }
        let rest = slice(word, 1, count(word));
        rest == upper(&rest)
    }

    /// `lookup`.
    pub fn lookup(
        &self,
        word: &str,
        tag: Option<&str>,
        stress: Option<f64>,
        ctx: Option<&Context>,
    ) -> Found {
        let mut word = word.to_string();
        let mut is_nnp = None;
        if word == upper(&word) && !self.golds.contains_key(&word) {
            word = lower(&word);
            is_nnp = Some(tag == Some("NNP"));
        }
        let (mut entry, mut rating) = (self.golds.get(&word), 4);
        if entry.is_none() && is_nnp != Some(true) {
            entry = self.silvers.get(&word);
            rating = 3;
        }
        let ps = match entry {
            None => None,
            Some(Entry::Word(w)) => Some(w.clone()),
            Some(Entry::Tagged(by_tag)) => {
                let key = if ctx.is_some_and(|c| c.future_vowel.is_none())
                    && by_tag.contains_key("None")
                {
                    Some("None".to_string())
                } else if tag.is_some_and(|t| by_tag.contains_key(t)) {
                    tag.map(str::to_string)
                } else {
                    parent_tag(tag)
                };
                match key.and_then(|k| by_tag.get(&k)) {
                    Some(value) => value.clone(),
                    None => by_tag.get("DEFAULT").cloned().flatten(),
                }
            }
        };
        if ps.is_none()
            || (is_nnp == Some(true) && !ps.as_ref().is_some_and(|p| p.contains(PRIMARY)))
        {
            let (nnp, nnp_rating) = self.nnp(&word);
            if nnp.is_some() {
                return (nnp, nnp_rating);
            }
        }
        (apply_stress(ps, stress), Some(rating))
    }

    /// `_s`.
    fn suffix_s(&self, stem: Option<String>) -> Option<String> {
        let stem = stem.filter(|s| !s.is_empty())?;
        let last = stem.chars().last()?;
        Some(if "ptkfθ".contains(last) {
            format!("{stem}s")
        } else if "szʃʒʧʤ".contains(last) {
            format!("{stem}{}z", if self.british { 'ɪ' } else { 'ᵻ' })
        } else {
            format!("{stem}z")
        })
    }

    /// `stem_s`.
    fn stem_s(
        &self,
        word: &str,
        tag: Option<&str>,
        stress: Option<f64>,
        ctx: Option<&Context>,
    ) -> Found {
        let n = count(word);
        if n < 3 || !word.ends_with('s') {
            return (None, None);
        }
        let stem = if !word.ends_with("ss") && self.is_known(&slice(word, 0, n - 1)) {
            slice(word, 0, n - 1)
        } else if (word.ends_with("'s")
            || (n > 4 && word.ends_with("es") && !word.ends_with("ies")))
            && self.is_known(&slice(word, 0, n - 2))
        {
            slice(word, 0, n - 2)
        } else if n > 4
            && word.ends_with("ies")
            && self.is_known(&format!("{}y", slice(word, 0, n - 3)))
        {
            format!("{}y", slice(word, 0, n - 3))
        } else {
            return (None, None);
        };
        let (ps, rating) = self.lookup(&stem, tag, stress, ctx);
        (self.suffix_s(ps), rating)
    }

    /// `_ed`.
    fn suffix_ed(&self, stem: Option<String>) -> Option<String> {
        let stem = stem.filter(|s| !s.is_empty())?;
        let chars: Vec<char> = stem.chars().collect();
        let last = *chars.last()?;
        let schwa = if self.british { 'ɪ' } else { 'ᵻ' };
        Some(if "pkfθʃsʧ".contains(last) {
            format!("{stem}t")
        } else if last == 'd' {
            format!("{stem}{schwa}d")
        } else if last != 't' {
            format!("{stem}d")
        } else if self.british || chars.len() < 2 {
            format!("{stem}ɪd")
        } else if US_TAUS.contains(chars[chars.len() - 2]) {
            format!("{}ɾᵻd", chars[..chars.len() - 1].iter().collect::<String>())
        } else {
            format!("{stem}ᵻd")
        })
    }

    /// `stem_ed`.
    fn stem_ed(
        &self,
        word: &str,
        tag: Option<&str>,
        stress: Option<f64>,
        ctx: Option<&Context>,
    ) -> Found {
        let n = count(word);
        if n < 4 || !word.ends_with('d') {
            return (None, None);
        }
        let stem = if !word.ends_with("dd") && self.is_known(&slice(word, 0, n - 1)) {
            slice(word, 0, n - 1)
        } else if n > 4
            && word.ends_with("ed")
            && !word.ends_with("eed")
            && self.is_known(&slice(word, 0, n - 2))
        {
            slice(word, 0, n - 2)
        } else {
            return (None, None);
        };
        let (ps, rating) = self.lookup(&stem, tag, stress, ctx);
        (self.suffix_ed(ps), rating)
    }

    /// `_ing`.
    fn suffix_ing(&self, stem: Option<String>) -> Option<String> {
        let stem = stem.filter(|s| !s.is_empty())?;
        let chars: Vec<char> = stem.chars().collect();
        let last = *chars.last()?;
        if self.british {
            if "əː".contains(last) {
                return None;
            }
        } else if chars.len() > 1 && last == 't' && US_TAUS.contains(chars[chars.len() - 2]) {
            return Some(format!(
                "{}ɾɪŋ",
                chars[..chars.len() - 1].iter().collect::<String>()
            ));
        }
        Some(format!("{stem}ɪŋ"))
    }

    /// `stem_ing`.
    fn stem_ing(
        &self,
        word: &str,
        tag: Option<&str>,
        stress: Option<f64>,
        ctx: Option<&Context>,
    ) -> Found {
        let n = count(word);
        if n < 5 || !word.ends_with("ing") {
            return (None, None);
        }
        let base = slice(word, 0, n - 3);
        let chars: Vec<char> = word.chars().collect();
        let doubled = {
            // ([bcdgklmnprstvxz])\1ing$ or cking$
            let c = chars[n - 4];
            (n >= 5 && "bcdgklmnprstvxz".contains(c) && chars[n - 5] == c)
                || word.ends_with("cking")
        };
        let stem = if n > 5 && self.is_known(&base) {
            base
        } else if self.is_known(&format!("{base}e")) {
            format!("{base}e")
        } else if n > 5 && doubled && self.is_known(&slice(word, 0, n - 4)) {
            slice(word, 0, n - 4)
        } else {
            return (None, None);
        };
        let (ps, rating) = self.lookup(&stem, tag, stress, ctx);
        (self.suffix_ing(ps), rating)
    }

    /// `get_word`.
    fn word(&self, word: &str, tag: &str, stress: Option<f64>, ctx: &Context) -> Found {
        let special = self.special(word, tag, stress, ctx);
        if special.0.is_some() {
            return special;
        }
        let low = lower(word);
        let undashed: String = word.chars().filter(|c| *c != '\'').collect();
        let rest = slice(word, 1, count(word));
        let mut word = word.to_string();
        if count(&word) > 1
            && is_alpha(&undashed)
            && word != low
            && (tag != "NNP" || count(&word) > 7)
            && !self.golds.contains_key(&word)
            && !self.silvers.contains_key(&word)
            && (word == upper(&word) || rest == lower(&rest))
            && (self.golds.contains_key(&low)
                || self.silvers.contains_key(&low)
                || truthy(self.stem_s(&low, Some(tag), stress, Some(ctx)).0)
                || truthy(self.stem_ed(&low, Some(tag), stress, Some(ctx)).0)
                || truthy(self.stem_ing(&low, Some(tag), stress, Some(ctx)).0))
        {
            word = low;
        }
        let tag = Some(tag);
        if self.is_known(&word) {
            return self.lookup(&word, tag, stress, Some(ctx));
        }
        let n = count(&word);
        if word.ends_with("s'") && self.is_known(&format!("{}'s", slice(&word, 0, n - 2))) {
            return self.lookup(
                &format!("{}'s", slice(&word, 0, n - 2)),
                tag,
                stress,
                Some(ctx),
            );
        }
        if word.ends_with('\'') && self.is_known(&slice(&word, 0, n - 1)) {
            return self.lookup(&slice(&word, 0, n - 1), tag, stress, Some(ctx));
        }
        let s = self.stem_s(&word, tag, stress, Some(ctx));
        if s.0.is_some() {
            return s;
        }
        let ed = self.stem_ed(&word, tag, stress, Some(ctx));
        if ed.0.is_some() {
            return ed;
        }
        let ing = self.stem_ing(&word, tag, Some(stress.unwrap_or(0.5)), Some(ctx));
        if ing.0.is_some() {
            return ing;
        }
        (None, None)
    }

    /// `is_currency`.
    fn is_currency(word: &str) -> bool {
        if !word.contains('.') {
            return true;
        }
        if word.matches('.').count() > 1 {
            return false;
        }
        // misaki also compares the set of cent characters with {0}, an int,
        // which is never equal: fewer than 3 cent digits is the rule.
        count(word.split('.').nth(1).unwrap_or("")) < 3
    }

    /// `get_number`.
    fn number(&self, word: &str, currency: Option<&str>, is_head: bool, flags: &str) -> Found {
        // A lowercase suffix ("st", "s", "'s", "ed"...).
        let tail = word
            .char_indices()
            .rev()
            .take_while(|(_, c)| c.is_ascii_lowercase() || *c == '\'')
            .last()
            .map(|(i, _)| i);
        let (mut word, suffix) = match tail {
            Some(i) => (word[..i].to_string(), Some(word[i..].to_string())),
            None => (word.to_string(), None),
        };
        let suffix = suffix.as_deref();
        let mut result: Vec<(String, i32)> = Vec::new();
        if word.starts_with('-') {
            let (ps, r) = self.lookup("minus", None, None, None);
            result.push((ps.unwrap_or_default(), r.unwrap_or(4)));
            word = word[1..].to_string();
        }
        let push = |result: &mut Vec<(String, i32)>, found: Found| {
            result.push((found.0.unwrap_or_default(), found.1.unwrap_or(4)));
        };
        let extend = |result: &mut Vec<(String, i32)>, num: &str, first: bool, escape: bool| {
            let text = if escape {
                num.to_string()
            } else {
                match num.parse::<u128>() {
                    Ok(n) => numbers::cardinal(n),
                    Err(_) => return,
                }
            };
            let splits = numbers::words(&text);
            for (i, w) in splits.iter().enumerate() {
                if w != "and" || flags.contains('&') {
                    if first && i == 0 && splits.len() > 1 && w == "one" && flags.contains('a') {
                        result.push(("ə".into(), 4));
                    } else {
                        let stress = if w == "point" { Some(-2.0) } else { None };
                        push(result, self.lookup(w, None, stress, None));
                    }
                } else if w == "and" && flags.contains('n') && !result.is_empty() {
                    let last = result.len() - 1;
                    result[last].0.push_str("ən");
                }
            }
        };
        let ordinal_suffix = suffix.is_some_and(|s| ORDINAL_SUFFIXES.contains(&s));
        let currency_known = currency.is_some_and(|c| CURRENCIES.iter().any(|(k, _)| *k == c));
        if is_digit(&word) && ordinal_suffix {
            extend(
                &mut result,
                &numbers::ordinal(word.parse().unwrap_or(0)),
                true,
                true,
            );
        } else if result.is_empty() && count(&word) == 4 && !currency_known && is_digit(&word) {
            extend(
                &mut result,
                &numbers::year(word.parse().unwrap_or(0)),
                true,
                true,
            );
        } else if !is_head && !word.contains('.') {
            let num: String = word.chars().filter(|c| *c != ',').collect();
            let digits: Vec<char> = num.chars().collect();
            if digits.first() == Some(&'0') || digits.len() > 3 {
                for d in &digits {
                    extend(&mut result, &d.to_string(), false, false);
                }
            } else if digits.len() == 3 && !num.ends_with("00") {
                extend(&mut result, &digits[0].to_string(), true, false);
                if digits[1] == '0' {
                    push(&mut result, self.lookup("O", None, Some(-2.0), None));
                    extend(&mut result, &digits[2].to_string(), false, false);
                } else {
                    extend(
                        &mut result,
                        &digits[1..].iter().collect::<String>(),
                        false,
                        false,
                    );
                }
            } else {
                extend(&mut result, &num, true, false);
            }
        } else if word.matches('.').count() > 1 || !is_head {
            let mut first = true;
            let plain: String = word.chars().filter(|c| *c != ',').collect();
            for num in plain.split('.') {
                let digits: Vec<char> = num.chars().collect();
                if digits.is_empty() {
                } else if digits[0] == '0'
                    || (digits.len() != 2 && digits[1..].iter().any(|d| *d != '0'))
                {
                    for d in &digits {
                        extend(&mut result, &d.to_string(), false, false);
                    }
                } else {
                    extend(&mut result, num, first, false);
                }
                first = false;
            }
        } else if currency_known && Self::is_currency(&word) {
            let units = CURRENCIES
                .iter()
                .find(|(k, _)| Some(*k) == currency)
                .map(|(_, u)| *u)
                .unwrap_or(("dollar", "cent"));
            let plain: String = word.chars().filter(|c| *c != ',').collect();
            let mut pairs: Vec<(u128, &str)> = plain
                .split('.')
                .zip([units.0, units.1])
                .map(|(num, unit)| (num.parse().unwrap_or(0), unit))
                .collect();
            if pairs.len() > 1 {
                if pairs[1].0 == 0 {
                    pairs.truncate(1);
                } else if pairs[0].0 == 0 {
                    pairs.remove(0);
                }
            }
            for (i, (num, unit)) in pairs.iter().enumerate() {
                if i > 0 {
                    push(&mut result, self.lookup("and", None, None, None));
                }
                extend(&mut result, &num.to_string(), i == 0, false);
                if *num != 1 && *unit != "pence" {
                    push(
                        &mut result,
                        self.stem_s(&format!("{unit}s"), None, None, None),
                    );
                } else {
                    push(&mut result, self.lookup(unit, None, None, None));
                }
            }
        } else {
            let text = if is_digit(&word) {
                numbers::cardinal(word.parse().unwrap_or(0))
            } else if !word.contains('.') {
                let plain: String = word.chars().filter(|c| *c != ',').collect();
                let n: u128 = plain.parse().unwrap_or(0);
                if ordinal_suffix {
                    numbers::ordinal(n)
                } else {
                    numbers::cardinal(n)
                }
            } else {
                let plain: String = word.chars().filter(|c| *c != ',').collect();
                if let Some(rest) = plain.strip_prefix('.') {
                    let digits: Vec<String> = rest
                        .chars()
                        .filter_map(|c| c.to_digit(10))
                        .map(|d| numbers::cardinal(u128::from(d)))
                        .collect();
                    format!("point {}", digits.join(" "))
                } else {
                    match numbers::decimal(&plain) {
                        Some(text) => text,
                        None => return (None, None),
                    }
                }
            };
            extend(&mut result, &text, true, true);
        }
        if result.is_empty() {
            return (None, None);
        }
        let rating = result.iter().map(|(_, r)| *r).min();
        let joined = result
            .into_iter()
            .map(|(p, _)| p)
            .collect::<Vec<_>>()
            .join(" ");
        match suffix {
            Some("s") | Some("'s") => (self.suffix_s(Some(joined)), rating),
            Some("ed") | Some("'d") => (self.suffix_ed(Some(joined)), rating),
            Some("ing") => (self.suffix_ing(Some(joined)), rating),
            _ => (Some(joined), rating),
        }
    }

    /// `append_currency`.
    fn append_currency(&self, ps: String, currency: Option<&str>) -> String {
        let Some(currency) = currency else { return ps };
        let Some((_, (unit, _))) = CURRENCIES.iter().find(|(k, _)| *k == currency) else {
            return ps;
        };
        match self.stem_s(&format!("{unit}s"), None, None, None).0 {
            Some(c) => format!("{ps} {c}"),
            None => ps,
        }
    }

    /// `is_number`.
    fn is_number(word: &str, is_head: bool) -> bool {
        if !word.chars().any(|c| c.is_ascii_digit()) {
            return false;
        }
        let mut word = word;
        for s in ["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"] {
            if let Some(w) = word.strip_suffix(s) {
                word = w;
                break;
            }
        }
        word.chars().enumerate().all(|(i, c)| {
            c.is_ascii_digit() || c == ',' || c == '.' || (is_head && i == 0 && c == '-')
        })
    }

    /// `__call__`: the phonemes and rating of one token in its context.
    pub fn token(&self, tk: &Token, ctx: &Context) -> Found {
        // misaki also applies NFKC and maps non-ASCII digits to ASCII; Trit has
        // no Unicode normalization tables, and ASCII text is the same under both.
        let word = tk
            .alias
            .clone()
            .unwrap_or_else(|| tk.text.clone())
            .replace(['‘', '’'], "'");
        let stress = if word == lower(&word) {
            None
        } else if word == upper(&word) {
            Some(2.0)
        } else {
            Some(0.5)
        };
        let (ps, rating) = self.word(&word, &tk.tag, stress, ctx);
        if let Some(ps) = ps {
            return (
                apply_stress(
                    Some(self.append_currency(ps, tk.currency.as_deref())),
                    tk.stress,
                ),
                rating,
            );
        }
        if Self::is_number(&word, tk.is_head) {
            let (ps, rating) =
                self.number(&word, tk.currency.as_deref(), tk.is_head, &tk.num_flags);
            return (apply_stress(ps, tk.stress), rating);
        }
        (None, None)
    }
}

/// Python's truth value of phonemes: `None` and "" are false.
fn truthy(ps: Option<String>) -> bool {
    ps.is_some_and(|p| !p.is_empty())
}
