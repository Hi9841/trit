//! English text to Kokoro phonemes, as misaki's `en.G2P` makes them
//! (misaki 0.9.4; the tokens and tags come from `tokens.rs` instead of
//! spaCy, the fallback for unknown words from `fallback.rs`). Function for
//! function as `misaki/en.py`; the comments name the Python function.
//!
//! The pronunciation markup of misaki works: `[Kokoro](/kˈOkəɹO/)` gives a
//! word its phonemes, `[word](-1)` its stress, `[1984](#a#)` number flags.

mod data;
mod fallback;
mod lexicon;
mod numbers;
mod tagger;
mod tokens;

use lexicon::{
    apply_stress, stress_weight, Lexicon, CONSONANTS, NON_QUOTE_PUNCTS, PRIMARY, PUNCTS, VOWELS,
};
use std::path::Path;
use std::sync::OnceLock;
use tagger::Tagger;
use tokens::Token;

pub use fallback::Fallback;

/// misaki's `TokenContext`: what follows the word being read.
#[derive(Clone, Copy, Debug, Default)]
pub struct Context {
    /// The next sound is a vowel (`Some(true)`), a consonant, or punctuation (`None`).
    pub future_vowel: Option<bool>,
    pub future_to: bool,
}

const SUBTOKEN_JUNKS: &str = "',-._‘’/";
const PUNCT_TAGS: [&str; 11] = [
    ".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP",
];
const CURRENCY_SYMBOLS: &str = "$£€";
/// Phonemes per pass of Kokoro (its context less the two pads).
pub const MAX_PHONEMES: usize = 510;

fn tagger() -> Result<&'static Tagger, String> {
    static TAGGER: OnceLock<Result<Tagger, String>> = OnceLock::new();
    TAGGER
        .get_or_init(Tagger::load)
        .as_ref()
        .map_err(Clone::clone)
}

pub struct English {
    lexicon: Lexicon,
    fallback: Option<Fallback>,
}

/// One pass of the model: its phonemes (at most `MAX_PHONEMES`).
#[derive(Clone, Debug)]
pub struct Chunk {
    pub phonemes: String,
}

impl English {
    /// American (`british` false) or British English. `fallback` is the
    /// folder of the BART for unknown words; without it they are not spoken.
    pub fn load(british: bool, fallback: Option<&Path>) -> Result<Self, String> {
        tagger()?;
        Ok(Self {
            lexicon: Lexicon::load(british)?,
            fallback: fallback.map(Fallback::load).transpose()?,
        })
    }

    /// The text in passes, as Kokoro's pipeline cuts it: at line breaks,
    /// then at 510 phonemes, at the last sentence end, else clause end,
    /// else comma that keeps a pass under the limit.
    pub fn chunks(&self, text: &str) -> Result<Vec<Chunk>, String> {
        let mut out = Vec::new();
        for line in text
            .trim()
            .split('\n')
            .map(str::trim)
            .filter(|l| !l.is_empty())
        {
            let tokens = self.tokens(line)?;
            out.extend(split(tokens));
        }
        Ok(out)
    }

    /// The phonemes of `text` as one string (`G2P.__call__`'s result).
    pub fn phonemize(&self, text: &str) -> Result<String, String> {
        let tokens = self.tokens(text)?;
        Ok(tokens
            .iter()
            .map(|t| format!("{}{}", t.phonemes.clone().unwrap_or_default(), t.whitespace))
            .collect())
    }

    /// `G2P.__call__`: the tokens of `text` with their phonemes.
    fn tokens(&self, text: &str) -> Result<Vec<Token>, String> {
        let (text, links) = preprocess(text);
        let mut tokens = tokens::tokenize(&text, tagger()?);
        for (span, feature) in &links {
            let inside: Vec<usize> = (0..tokens.len())
                .filter(|i| tokens[*i].span.0 >= span.0 && tokens[*i].span.1 <= span.1)
                .collect();
            for (n, i) in inside.into_iter().enumerate() {
                let tk = &mut tokens[i];
                match feature {
                    Feature::Stress(s) => tk.stress = Some(*s),
                    Feature::Phonemes(p) => {
                        tk.is_head = n == 0;
                        tk.phonemes = Some(if n == 0 { p.clone() } else { String::new() });
                        tk.rating = Some(5);
                    }
                    Feature::Flags(f) => tk.num_flags = f.clone(),
                }
            }
        }
        let tokens = fold_left(tokens);
        let mut words = retokenize(tokens);
        let mut ctx = Context::default();
        for w in words.iter_mut().rev() {
            match w {
                Word::One(tk) => {
                    if tk.phonemes.is_none() {
                        let (ps, rating) = self.lexicon.token(tk, &ctx);
                        tk.phonemes = ps;
                        tk.rating = rating;
                    }
                    if tk.phonemes.is_none() && wordlike(&tk.text) {
                        if let Some(fallback) = &self.fallback {
                            tk.phonemes = fallback.phonemes(&tk.text);
                            tk.rating = Some(1);
                        }
                    }
                    ctx = token_context(ctx, tk.phonemes.as_deref(), tk);
                }
                Word::Many(group) => self.group(group, &mut ctx),
            }
        }
        let mut out: Vec<Token> = words
            .into_iter()
            .map(|w| match w {
                Word::One(tk) => tk,
                Word::Many(group) => merge(&group, Some("")),
            })
            .collect();
        // misaki before version 2.0 writes the flap and the glottal stop as T and t.
        for tk in &mut out {
            if let Some(ps) = &tk.phonemes {
                tk.phonemes = Some(ps.replace('ɾ', "T").replace('ʔ', "t"));
            }
        }
        Ok(out)
    }

    /// The `while left < right` loop of `G2P.__call__` for a group of
    /// subtokens with no space between them: the longest known span from
    /// the right.
    fn group(&self, w: &mut [Token], ctx: &mut Context) {
        // Unlike misaki, a group that is not a word ("5:30pm") does not go
        // to the fallback: the BART reads words, and gives noise for digits
        // and signs, which the lexicon reads well piece by piece.
        let fallback = self
            .fallback
            .as_ref()
            .filter(|_| wordlike(&merge(w, None).text));
        let (mut left, mut right) = (0usize, w.len());
        let mut fall_back = false;
        while left < right {
            let merged = if w[left..right]
                .iter()
                .any(|t| t.alias.is_some() || t.phonemes.is_some())
            {
                None
            } else {
                Some(merge(&w[left..right], None))
            };
            let (ps, rating) = match &merged {
                None => (None, None),
                Some(tk) => self.lexicon.token(tk, ctx),
            };
            if let Some(ps) = ps {
                w[left].phonemes = Some(ps.clone());
                w[left].rating = rating;
                for x in &mut w[left + 1..right] {
                    x.phonemes = Some(String::new());
                }
                if let Some(tk) = &merged {
                    *ctx = token_context(*ctx, Some(&ps), tk);
                }
                right = left;
                left = 0;
            } else if left + 1 < right {
                left += 1;
            } else {
                right -= 1;
                let tk = &mut w[right];
                if tk.phonemes.is_none() {
                    if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                        tk.phonemes = Some(String::new());
                        tk.rating = Some(3);
                    } else if fallback.is_some() {
                        fall_back = true;
                        break;
                    }
                }
                left = 0;
            }
        }
        if fall_back {
            let merged = merge(w, None);
            if let Some(fallback) = fallback {
                w[0].phonemes = fallback.phonemes(&merged.text);
                w[0].rating = Some(1);
                for x in &mut w[1..] {
                    x.phonemes = Some(String::new());
                }
            }
        } else {
            resolve(w);
        }
    }
}

/// Text the fallback can read: letters, with apostrophes and hyphens.
fn wordlike(text: &str) -> bool {
    text.chars().any(char::is_alphabetic)
        && text
            .chars()
            .all(|c| c.is_alphabetic() || matches!(c, '\'' | '’' | '-'))
}

/// A marked word: its character span, and its feature.
type Link = ((usize, usize), Feature);

enum Feature {
    Stress(f64),
    Phonemes(String),
    Flags(String),
}

/// `G2P.preprocess`: the text without the markup, and the character span
/// of each marked word with its feature.
fn preprocess(text: &str) -> (String, Vec<Link>) {
    let text = text.trim_start();
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut out_len = 0usize;
    let mut links = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        // [text](feature): text without ']', feature without ')'.
        if chars[i] == '[' {
            if let Some(close) = (i + 1..chars.len())
                .find(|j| chars[*j] == ']')
                .filter(|j| *j > i + 1)
            {
                if chars.get(close + 1) == Some(&'(') {
                    if let Some(end) = (close + 2..chars.len()).find(|j| chars[*j] == ')') {
                        let word: String = chars[i + 1..close].iter().collect();
                        let feature: String = chars[close + 2..end].iter().collect();
                        let start = out_len;
                        out.push_str(&word);
                        out_len += word.chars().count();
                        if let Some(f) = parse_feature(&feature) {
                            links.push(((start, out_len), f));
                        }
                        i = end + 1;
                        continue;
                    }
                }
            }
        }
        out.push(chars[i]);
        out_len += 1;
        i += 1;
    }
    (out, links)
}

fn parse_feature(f: &str) -> Option<Feature> {
    let digits = f.strip_prefix(['-', '+']).unwrap_or(f);
    if lexicon::is_digit(digits) {
        return f.parse::<f64>().ok().map(Feature::Stress);
    }
    match f {
        "0.5" | "+0.5" => return Some(Feature::Stress(0.5)),
        "-0.5" => return Some(Feature::Stress(-0.5)),
        _ => {}
    }
    let n = f.chars().count();
    if n > 1 && f.starts_with('/') && f.ends_with('/') {
        return Some(Feature::Phonemes(
            f[1..]
                .trim_end_matches('/')
                .trim_start_matches('/')
                .to_string(),
        ));
    }
    if n > 1 && f.starts_with('#') && f.ends_with('#') {
        return Some(Feature::Flags(
            f[1..]
                .trim_end_matches('#')
                .trim_start_matches('#')
                .to_string(),
        ));
    }
    None
}

/// `merge_tokens`.
fn merge(tokens: &[Token], unk: Option<&str>) -> Token {
    let mut stresses: Vec<f64> = Vec::new();
    for s in tokens.iter().filter_map(|t| t.stress) {
        if !stresses.contains(&s) {
            stresses.push(s);
        }
    }
    let currency = tokens.iter().filter_map(|t| t.currency.clone()).max();
    let rating = if tokens.iter().any(|t| t.rating.is_none()) {
        None
    } else {
        tokens.iter().filter_map(|t| t.rating).min()
    };
    let phonemes = unk.map(|unk| {
        let mut ps = String::new();
        for tk in tokens {
            if tk.prespace
                && !ps.is_empty()
                && !ps.ends_with(char::is_whitespace)
                && tk.phonemes.as_deref().is_some_and(|p| !p.is_empty())
            {
                ps.push(' ');
            }
            ps.push_str(tk.phonemes.as_deref().unwrap_or(unk));
        }
        ps
    });
    let weight = |t: &Token| {
        t.text
            .chars()
            .map(|c| {
                if lexicon::lower(&c.to_string()) == c.to_string() {
                    1
                } else {
                    2
                }
            })
            .sum::<usize>()
    };
    // Python's max: the first token of the largest weight.
    let mut tag_from = &tokens[0];
    for t in &tokens[1..] {
        if weight(t) > weight(tag_from) {
            tag_from = t;
        }
    }
    let mut flags: Vec<char> = tokens.iter().flat_map(|t| t.num_flags.chars()).collect();
    flags.sort_unstable();
    flags.dedup();
    let last = &tokens[tokens.len() - 1];
    let mut text: String = tokens[..tokens.len() - 1]
        .iter()
        .map(|t| format!("{}{}", t.text, t.whitespace))
        .collect();
    text.push_str(&last.text);
    Token {
        text,
        tag: tag_from.tag.clone(),
        whitespace: last.whitespace.clone(),
        phonemes,
        is_head: tokens[0].is_head,
        alias: None,
        stress: if stresses.len() == 1 {
            Some(stresses[0])
        } else {
            None
        },
        currency,
        num_flags: flags.into_iter().collect(),
        prespace: tokens[0].prespace,
        rating,
        span: (tokens[0].span.0, last.span.1),
    }
}

/// `G2P.fold_left`: a token that is not a head joins the one before it.
fn fold_left(tokens: Vec<Token>) -> Vec<Token> {
    let mut out: Vec<Token> = Vec::new();
    for tk in tokens {
        if !tk.is_head {
            if let Some(prev) = out.pop() {
                out.push(merge(&[prev, tk], Some("")));
                continue;
            }
        }
        out.push(tk);
    }
    out
}

enum Word {
    One(Token),
    Many(Vec<Token>),
}

/// `subtokenize`: misaki's regex, alternative by alternative (Python's
/// `findall` takes the first alternative that matches at a position).
fn subtokenize(word: &str) -> Vec<String> {
    let c: Vec<char> = word.chars().collect();
    let n = c.len();
    let quote = |x: char| matches!(x, '\'' | '‘' | '’');
    let letter = |i: usize| i < n && c[i].is_alphabetic();
    let digit = |i: usize| i < n && c[i].is_numeric();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let end = 'found: {
            // ^['‘’]+
            if i == 0 && quote(c[0]) {
                let mut j = 0;
                while j < n && quote(c[j]) {
                    j += 1;
                }
                break 'found Some(j);
            }
            // \p{Lu}(?=\p{Lu}\p{Ll})
            if c[i].is_uppercase()
                && i + 2 < n
                && c[i + 1].is_uppercase()
                && c[i + 2].is_lowercase()
            {
                break 'found Some(i + 1);
            }
            // (?:^-)?(?:\d?[,.]?\d)+
            let digits_from = |start: usize| -> Option<usize> {
                let mut j = start;
                loop {
                    let sep = |k: usize| k < n && (c[k] == ',' || c[k] == '.');
                    let step = if digit(j) && sep(j + 1) && digit(j + 2) {
                        3
                    } else if (digit(j) || sep(j)) && digit(j + 1) {
                        2
                    } else if digit(j) {
                        1
                    } else {
                        break;
                    };
                    j += step;
                }
                (j > start).then_some(j)
            };
            if i == 0 && c[0] == '-' {
                if let Some(j) = digits_from(1) {
                    break 'found Some(j);
                }
            }
            if let Some(j) = digits_from(i) {
                break 'found Some(j);
            }
            // [-_]+
            if c[i] == '-' || c[i] == '_' {
                let mut j = i;
                while j < n && (c[j] == '-' || c[j] == '_') {
                    j += 1;
                }
                break 'found Some(j);
            }
            // ['‘’]{2,}
            if quote(c[i]) && i + 1 < n && quote(c[i + 1]) {
                let mut j = i;
                while j < n && quote(c[j]) {
                    j += 1;
                }
                break 'found Some(j);
            }
            // \p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu}): the shortest letters
            // ending lowercase just before an uppercase letter.
            {
                let mut j = i;
                loop {
                    if j + 1 < n && c[j].is_lowercase() && c[j + 1].is_uppercase() {
                        break 'found Some(j + 1);
                    }
                    if letter(j) || (j < n && quote(c[j]) && letter(j + 1) && j > i) {
                        j += 1;
                    } else {
                        break;
                    }
                }
            }
            // \p{L}+(?:['‘’]\p{L})*
            if letter(i) {
                let mut j = i;
                while letter(j) {
                    j += 1;
                }
                while j + 1 < n && quote(c[j]) && letter(j + 1) {
                    j += 2;
                }
                break 'found Some(j);
            }
            // [^-_\p{L}'‘’\d]
            if !(c[i] == '-'
                || c[i] == '_'
                || c[i].is_alphabetic()
                || quote(c[i])
                || c[i].is_numeric())
            {
                break 'found Some(i + 1);
            }
            // ['‘’]+$
            if quote(c[i]) {
                let mut j = i;
                while j < n && quote(c[j]) {
                    j += 1;
                }
                if j == n {
                    break 'found Some(j);
                }
            }
            None
        };
        match end {
            Some(j) if j > i => {
                out.push(c[i..j].iter().collect());
                i = j;
            }
            _ => i += 1,
        }
    }
    out
}

/// `G2P.retokenize`.
fn retokenize(tokens: Vec<Token>) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut currency: Option<String> = None;
    let count = tokens.len();
    let tags: Vec<String> = tokens.iter().map(|t| t.tag.clone()).collect();
    for (i, token) in tokens.into_iter().enumerate() {
        let mut tks: Vec<Token> = if token.alias.is_none() && token.phonemes.is_none() {
            subtokenize(&token.text)
                .into_iter()
                .map(|t| Token {
                    text: t,
                    tag: token.tag.clone(),
                    whitespace: String::new(),
                    phonemes: None,
                    is_head: true,
                    alias: None,
                    stress: token.stress,
                    currency: None,
                    num_flags: token.num_flags.clone(),
                    prespace: false,
                    rating: None,
                    span: token.span,
                })
                .collect()
        } else {
            vec![token.clone()]
        };
        if tks.is_empty() {
            continue;
        }
        let last = tks.len() - 1;
        tks[last].whitespace = token.whitespace.clone();
        let n = tks.len();
        for j in 0..n {
            let next_text_first = tks.get(j + 1).and_then(|t| t.text.chars().next());
            let prev_text_last = if j > 0 {
                tks[j - 1].text.chars().last()
            } else {
                None
            };
            let tk = &mut tks[j];
            if tk.alias.is_some() || tk.phonemes.is_some() {
            } else if tk.tag == "$"
                && CURRENCY_SYMBOLS.contains(tk.text.as_str())
                && tk.text.chars().count() == 1
            {
                currency = Some(tk.text.clone());
                tk.phonemes = Some(String::new());
                tk.rating = Some(4);
            } else if tk.tag == ":" && matches!(tk.text.as_str(), "-" | "–") {
                tk.phonemes = Some("—".into());
                tk.rating = Some(3);
            } else if PUNCT_TAGS.contains(&tk.tag.as_str())
                && !tk
                    .text
                    .chars()
                    .all(|c| c.to_ascii_lowercase().is_ascii_lowercase())
            {
                let ps = match tk.tag.as_str() {
                    "-LRB-" => "(".to_string(),
                    "-RRB-" => ")".to_string(),
                    "``" => "“".to_string(),
                    "\"\"" | "''" => "”".to_string(),
                    _ => tk.text.chars().filter(|c| PUNCTS.contains(*c)).collect(),
                };
                tk.phonemes = Some(ps);
                tk.rating = Some(4);
            } else if currency.is_some() {
                if tk.tag != "CD" {
                    currency = None;
                } else if j + 1 == n && (i + 1 == count || tags[i + 1] != "CD") {
                    tk.currency = currency.clone();
                }
            } else if 0 < j && j + 1 < n && tk.text == "2" && {
                let joined: String = [prev_text_last, next_text_first].iter().flatten().collect();
                joined.chars().count() == 2 && lexicon::is_alpha(&joined)
            } {
                tk.alias = Some("to".into());
            }
        }
        for mut tk in tks {
            if tk.alias.is_some() || tk.phonemes.is_some() {
                words.push(Word::One(tk));
            } else if let Some(Word::Many(group)) = words.last_mut().filter(
                |w| matches!(w, Word::Many(g) if g.last().is_some_and(|t| t.whitespace.is_empty())),
            ) {
                tk.is_head = false;
                group.push(tk);
            } else if tk.whitespace.is_empty() {
                words.push(Word::Many(vec![tk]));
            } else {
                words.push(Word::One(tk));
            }
        }
    }
    words
        .into_iter()
        .map(|w| match w {
            Word::Many(mut g) if g.len() == 1 => Word::One(g.remove(0)),
            w => w,
        })
        .collect()
}

/// `G2P.token_context`.
fn token_context(ctx: Context, ps: Option<&str>, token: &Token) -> Context {
    let mut vowel = ctx.future_vowel;
    if let Some(ps) = ps.filter(|p| !p.is_empty()) {
        if let Some(c) = ps.chars().find(|c| {
            VOWELS.contains(*c) || CONSONANTS.contains(*c) || NON_QUOTE_PUNCTS.contains(*c)
        }) {
            vowel = if NON_QUOTE_PUNCTS.contains(c) {
                None
            } else {
                Some(VOWELS.contains(c))
            };
        }
    }
    let future_to = matches!(token.text.as_str(), "to" | "To")
        || (token.text == "TO" && matches!(token.tag.as_str(), "TO" | "IN"));
    Context {
        future_vowel: vowel,
        future_to,
    }
}

/// `G2P.resolve_tokens`: punctuation of a group, and its stress.
fn resolve(tokens: &mut [Token]) {
    let last = tokens.len() - 1;
    let mut text: String = tokens[..last]
        .iter()
        .map(|t| format!("{}{}", t.text, t.whitespace))
        .collect();
    text.push_str(&tokens[last].text);
    let mut kinds: Vec<u8> = text
        .chars()
        .filter(|c| !SUBTOKEN_JUNKS.contains(*c))
        .map(|c| {
            if c.is_alphabetic() {
                0
            } else if c.is_ascii_digit() {
                1
            } else {
                2
            }
        })
        .collect();
    kinds.sort_unstable();
    kinds.dedup();
    let prespace = text.contains(' ') || text.contains('/') || kinds.len() > 1;
    for (i, tk) in tokens.iter_mut().enumerate() {
        if tk.phonemes.is_none() {
            if i == last && NON_QUOTE_PUNCTS.contains(tk.text.as_str()) {
                tk.phonemes = Some(tk.text.clone());
                tk.rating = Some(3);
            } else if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                tk.phonemes = Some(String::new());
                tk.rating = Some(3);
            }
        } else if i > 0 {
            tk.prespace = prespace;
        }
    }
    if prespace {
        return;
    }
    let mut indices: Vec<(bool, usize, usize)> = tokens
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            let ps = t.phonemes.as_deref().filter(|p| !p.is_empty())?;
            Some((ps.contains(PRIMARY), stress_weight(ps), i))
        })
        .collect();
    if indices.len() == 2 && tokens[indices[0].2].text.chars().count() == 1 {
        let i = indices[1].2;
        tokens[i].phonemes = apply_stress(tokens[i].phonemes.take(), Some(-0.5));
        return;
    }
    if indices.len() < 2 || indices.iter().filter(|x| x.0).count() <= indices.len().div_ceil(2) {
        return;
    }
    indices.sort();
    let half = indices.len() / 2;
    for (_, _, i) in indices.into_iter().take(half) {
        tokens[i].phonemes = apply_stress(tokens[i].phonemes.take(), Some(-0.5));
    }
}

fn to_ps(tokens: &[Token]) -> String {
    tokens
        .iter()
        .map(|t| {
            format!(
                "{}{}",
                t.phonemes.as_deref().unwrap_or(""),
                if t.whitespace.is_empty() { "" } else { " " }
            )
        })
        .collect::<String>()
        .trim()
        .to_string()
}

/// `KPipeline.waterfall_last`: where to cut `tokens` so the rest fits.
fn waterfall_last(tokens: &[Token], next_count: usize) -> usize {
    for marks in ["!.?…", ":;", ",—"] {
        let Some(z) = tokens.iter().rposition(|t| {
            let ps = t.phonemes.as_deref().unwrap_or("");
            ps.chars().count() == 1 && marks.contains(ps)
        }) else {
            continue;
        };
        let mut z = z + 1;
        if z < tokens.len() && matches!(tokens[z].phonemes.as_deref(), Some(")") | Some("”")) {
            z += 1;
        }
        if next_count.saturating_sub(to_ps(&tokens[..z]).chars().count()) <= MAX_PHONEMES {
            return z;
        }
    }
    tokens.len()
}

/// `KPipeline.en_tokenize`: the tokens in passes of at most 510 phonemes.
fn split(tokens: Vec<Token>) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut held: Vec<Token> = Vec::new();
    let mut count = 0usize;
    for mut t in tokens {
        if t.phonemes.is_none() {
            t.phonemes = Some(String::new());
        }
        let mut next = format!(
            "{}{}",
            t.phonemes.as_deref().unwrap_or(""),
            if t.whitespace.is_empty() { "" } else { " " }
        );
        let next_count = count + next.trim_end().chars().count();
        if next_count > MAX_PHONEMES {
            let z = waterfall_last(&held, next_count);
            out.push(Chunk {
                phonemes: to_ps(&held[..z]),
            });
            held.drain(..z);
            count = to_ps(&held).chars().count();
            if held.is_empty() {
                next = next.trim_start().to_string();
            }
        }
        count += next.chars().count();
        held.push(t);
    }
    if !held.is_empty() {
        out.push(Chunk {
            phonemes: to_ps(&held),
        });
    }
    out.into_iter()
        .filter(|c| !c.phonemes.is_empty())
        .map(|mut c| {
            if c.phonemes.chars().count() > MAX_PHONEMES {
                c.phonemes = c.phonemes.chars().take(MAX_PHONEMES).collect();
            }
            c
        })
        .collect()
}
