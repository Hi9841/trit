//! Tokenizers: token ids to text and back.

pub mod bpe;
pub mod sentencepiece;

pub(super) fn merge_symbols(symbols: &mut Vec<String>, rank: impl Fn(&str, &str) -> Option<usize>) {
    loop {
        let best = symbols
            .windows(2)
            .enumerate()
            .filter_map(|(index, pair)| rank(&pair[0], &pair[1]).map(|rank| (rank, index)))
            .min();
        let Some((_, at)) = best else {
            break;
        };
        let right = symbols.remove(at + 1);
        symbols[at].push_str(&right);
    }
}

pub(super) fn parse_merge(value: &serde_json::Value) -> Option<(String, String)> {
    match value {
        serde_json::Value::Array(parts) => {
            let left = parts.first().and_then(|part| part.as_str())?;
            let right = parts.get(1).and_then(|part| part.as_str())?;
            Some((left.to_string(), right.to_string()))
        }
        serde_json::Value::String(text) => {
            let (left, right) = text.split_once(' ')?;
            Some((left.to_string(), right.to_string()))
        }
        _ => None,
    }
}
