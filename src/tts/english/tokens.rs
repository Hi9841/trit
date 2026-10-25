//! Tokens of the English phonemizer, and the tokenizer that makes them.
//! misaki reads spaCy's tokens and tags; Trit splits words from their
//! leading and trailing punctuation (the pieces of a glued word go back
//! together in misaki's grouping, so finer splits change nothing), tags
//! them with the perceptron, and corrects the tags misaki depends on most.

use super::tagger::Tagger;

/// A token, as misaki's `MToken` with its `_` fields.
#[derive(Clone, Debug, Default)]
pub struct Token {
    pub text: String,
    pub tag: String,
    /// " " when a space follows, else "".
    pub whitespace: String,
    pub phonemes: Option<String>,
    pub is_head: bool,
    pub alias: Option<String>,
    pub stress: Option<f64>,
    pub currency: Option<String>,
    pub num_flags: String,
    pub prespace: bool,
    pub rating: Option<i32>,
    /// Character span in the text, for the inline pronunciation markup.
    pub span: (usize, usize),
}

const PREFIX: &str = "\"'“‘([{¿¡$£€#";
const SUFFIX: &str = ".,!?;:)]}\"'”’…%";
const ABBREVIATIONS: [&str; 10] = [
    "Dr", "Mr", "Mrs", "Ms", "St", "Jr", "Sr", "Prof", "Mt", "vs",
];

/// `(text, span)` of the parts of one whitespace-free chunk: leading
/// punctuation, the word, trailing punctuation.
fn split_chunk(chunk: &str, at: usize) -> Vec<(String, (usize, usize))> {
    let chars: Vec<char> = chunk.chars().collect();
    let (mut start, mut end) = (0usize, chars.len());
    let mut pre = Vec::new();
    while start < end && end - start > 1 && PREFIX.contains(chars[start]) {
        pre.push(start);
        start += 1;
    }
    let mut post = Vec::new();
    while start < end && SUFFIX.contains(chars[end - 1]) {
        let word: String = chars[start..end].iter().collect();
        let last = chars[end - 1];
        if last == '.' && end - start > 1 && (is_initials(&word) || is_title(&word)) {
            break;
        }
        // The closing apostrophe of a plural possessive (dogs').
        if (last == '\'' || last == '’')
            && end - start > 2
            && matches!(chars[end - 2], 's' | 'S')
            && pre.is_empty()
        {
            break;
        }
        post.push(end - 1);
        end -= 1;
    }
    post.reverse();
    let piece = |a: usize, b: usize| (chars[a..b].iter().collect::<String>(), (at + a, at + b));
    let mut parts: Vec<_> = pre.iter().map(|i| piece(*i, i + 1)).collect();
    if start < end {
        let cut = clitic(&chars[start..end]).map_or(end, |n| end - n);
        parts.push(piece(start, cut));
        if cut < end {
            parts.push(piece(cut, end));
        }
    }
    parts.extend(post.iter().map(|i| piece(*i, i + 1)));
    parts
}

/// The length of a clitic that ends `word` after letters, split as Penn
/// Treebank tokens are (can't -> ca n't, it's -> it 's), so the tagger
/// sees the verb: "ca n't read" is a verb, "can't read" a past tense.
fn clitic(word: &[char]) -> Option<usize> {
    const CLITICS: [&str; 7] = ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m"];
    let plain: String = word
        .iter()
        .map(|c| {
            if *c == '’' {
                '\''
            } else {
                c.to_ascii_lowercase()
            }
        })
        .collect();
    CLITICS.iter().find_map(|clitic| {
        let n = clitic.chars().count();
        let stem_is_letters =
            word.len() > n && word[..word.len() - n].iter().all(|ch| ch.is_alphabetic());
        (plain.ends_with(clitic) && stem_is_letters).then_some(n)
    })
}

/// `U.S.`, `a.m.`: letters each followed by a period.
fn is_initials(word: &str) -> bool {
    let chars: Vec<char> = word.chars().collect();
    chars.len().is_multiple_of(2)
        && chars
            .chunks(2)
            .all(|p| p[0].is_ascii_alphabetic() && p[1] == '.')
}

/// `Dr.`, `Mrs.`: a title before a name (an uppercase letter and up to
/// three lowercase ones, then a period).
fn is_title(word: &str) -> bool {
    word.strip_suffix('.').is_some_and(|w| {
        let mut chars = w.chars();
        chars.next().is_some_and(|c| c.is_ascii_uppercase())
            && w.len() <= 4
            && chars.all(|c| c.is_ascii_lowercase())
            && ABBREVIATIONS.contains(&w)
    })
}

/// Tokens of `text` (spans in characters), tagged.
pub fn tokenize(text: &str, tagger: &Tagger) -> Vec<Token> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens: Vec<Token> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && !chars[i].is_whitespace() {
            i += 1;
        }
        let chunk: String = chars[start..i].iter().collect();
        let space = if i < chars.len() { " " } else { "" };
        let parts = split_chunk(&chunk, start);
        let count = parts.len();
        for (n, (part, span)) in parts.into_iter().enumerate() {
            tokens.push(Token {
                text: part,
                whitespace: if n + 1 == count {
                    space.to_string()
                } else {
                    String::new()
                },
                is_head: true,
                span,
                ..Token::default()
            });
        }
    }
    // Quotes: opening when nothing is glued before, else closing.
    let mut quotes: Vec<Option<&'static str>> = vec![None; tokens.len()];
    for (n, quote) in quotes.iter_mut().enumerate() {
        let t = tokens[n].text.as_str();
        let free = n == 0 || !tokens[n - 1].whitespace.is_empty();
        if matches!(t, "\"" | "“" | "‘") || (t == "'" && free) {
            *quote = Some(if matches!(t, "“" | "‘") || free {
                "``"
            } else {
                "''"
            });
        } else if matches!(t, "\"" | "”" | "’" | "'") {
            *quote = Some("''");
        }
    }
    let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
    let tags: Vec<String> = tagger
        .tag(&words)
        .into_iter()
        .map(|t| match t.as_str() {
            "(" => "-LRB-".to_string(),
            ")" => "-RRB-".to_string(),
            _ => t,
        })
        .collect();
    let mut start = true;
    for n in 0..tokens.len() {
        let prev = if n > 0 { tags[n - 1].as_str() } else { "" };
        let next = tags.get(n + 1).map_or("", String::as_str);
        let next_text = tokens.get(n + 1).map_or("", |t| t.text.as_str());
        let t = tokens[n].text.as_str();
        let mut tag = tags[n].clone();
        if let Some(quote) = quotes[n] {
            tag = quote.to_string();
        } else if is_address(t) && !is_digits_and_dots(t) {
            tag = "ADD".into();
        } else if start
            && matches!(tag.as_str(), "NN" | "NNP" | "RB" | "JJ")
            && t.chars().next().is_some_and(char::is_uppercase)
            && t.chars().all(char::is_alphabetic)
        {
            // The first word of a sentence: often an order ("Use", "Close the door").
            tag = "VB".into();
        } else if t.to_lowercase() == "in"
            && (matches!(next, "TO" | "." | "," | ":" | "") || next_text == "to")
        {
            tag = "RP".into();
        } else if tag == "NNS"
            && matches!(prev, "NN" | "NNS" | "NNP" | "PRP")
            && matches!(next, "DT" | "PRP$" | "IN" | "TO" | "PRP")
        {
            // A plural noun before its object is a verb ("table lives in").
            tag = "VBZ".into();
        }
        if !matches!(tag.as_str(), "``" | "-LRB-") {
            start = matches!(t, "." | "!" | "?");
        }
        tokens[n].tag = tag;
    }
    tokens
}

fn is_digits_and_dots(t: &str) -> bool {
    let digits: String = t.chars().filter(|c| *c != '.').collect();
    super::tagger::is_digits(&digits)
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// An email address, a URL, a host or file name (`take.wav`, `crates.io`).
fn is_address(t: &str) -> bool {
    if let Some(at) = t.find('@') {
        let rest = &t[at + 1..];
        if at > 0
            && rest
                .char_indices()
                .any(|(i, c)| c == '.' && i > 0 && i + 1 < rest.len())
        {
            return true;
        }
    }
    if let Some(scheme) = t.find("://") {
        if scheme > 0 && t[..scheme].chars().all(is_word_char) && t.len() > scheme + 3 {
            return true;
        }
    }
    let segments: Vec<&str> = t.split('.').collect();
    if segments.len() < 2 {
        return false;
    }
    let (first, last) = (segments[0], segments[segments.len() - 1]);
    first.chars().count() >= 2
        && first.chars().all(|c| is_word_char(c) || c == '-')
        && segments[1..segments.len() - 1]
            .iter()
            .all(|s| !s.is_empty() && s.chars().all(|c| is_word_char(c) || c == '-'))
        && last.chars().count() >= 2
        && last.chars().all(|c| c.is_ascii_alphanumeric())
}
