//! Tokenizers: token ids to text and back.

pub mod bpe;
pub mod sentencepiece;

/// Joins the pair of neighbors with the lowest rank, the leftmost first,
/// until no pair has a rank: byte-pair merges. A heap holds the pairs, so a
/// word of `n` symbols takes `n log n` steps. A scan of all pairs for each
/// merge took `n * n` rank lookups: one word of 10,000 letters took 2 s.
pub(super) fn merge_symbols(symbols: &mut Vec<String>, rank: impl Fn(&str, &str) -> Option<usize>) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    const NONE: usize = usize::MAX;
    let count = symbols.len();
    if count < 2 {
        return;
    }
    // A merge joins a symbol into the one before it. Each symbol knows its
    // neighbors that are still there, and how many times it grew.
    let mut next: Vec<usize> = (1..count).chain([NONE]).collect();
    let mut before: Vec<usize> = [NONE].into_iter().chain(0..count - 1).collect();
    let mut grown = vec![0u32; count];
    let mut gone = vec![false; count];
    // (rank, left symbol, its growth, the growth of its right neighbor).
    // The left symbols keep their order, so the lowest index is the leftmost.
    let mut pairs: BinaryHeap<Reverse<(usize, usize, u32, u32)>> = BinaryHeap::new();
    let offer = |pairs: &mut BinaryHeap<_>,
                 symbols: &[String],
                 next: &[usize],
                 grown: &[u32],
                 left: usize| {
        let right = next[left];
        if right != NONE {
            if let Some(rank) = rank(&symbols[left], &symbols[right]) {
                pairs.push(Reverse((rank, left, grown[left], grown[right])));
            }
        }
    };
    for left in 0..count - 1 {
        offer(&mut pairs, symbols, &next, &grown, left);
    }
    while let Some(Reverse((_, left, grew_left, grew_right))) = pairs.pop() {
        let right = next[left];
        // A pair whose symbols changed after the offer is not there any more.
        if gone[left] || right == NONE || grown[left] != grew_left || grown[right] != grew_right {
            continue;
        }
        let tail = std::mem::take(&mut symbols[right]);
        symbols[left].push_str(&tail);
        gone[right] = true;
        grown[left] += 1;
        next[left] = next[right];
        if next[left] != NONE {
            before[next[left]] = left;
        }
        if before[left] != NONE {
            offer(&mut pairs, symbols, &next, &grown, before[left]);
        }
        offer(&mut pairs, symbols, &next, &grown, left);
    }
    let mut index = 0;
    symbols.retain(|_| {
        index += 1;
        !gone[index - 1]
    });
}

/// `bytes` as text. A last character that is not complete (a decode that
/// the token limit stopped between the pieces of a character) is dropped;
/// other bytes that are not UTF-8 become U+FFFD.
pub(super) fn lossy(bytes: &[u8]) -> String {
    let len = bytes.len();
    let mut keep = len;
    for back in 1..=len.min(4) {
        let byte = bytes[len - back];
        // A continuation byte: the first byte of its character is before it.
        if byte & 0xC0 == 0x80 {
            continue;
        }
        let needs = match byte {
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF7 => 4,
            _ => 1,
        };
        if needs > back {
            keep = len - back;
        }
        break;
    }
    String::from_utf8_lossy(&bytes[..keep]).into_owned()
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
