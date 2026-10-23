//! CTC decoding, shared by the families with a CTC head.

/// Greedy CTC: the best token per frame, repeats merged, blanks dropped.
pub fn greedy(
    logits: &[f32],
    vocab: usize,
    blank: usize,
    banned: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let mut tokens = Vec::new();
    let mut frames = Vec::new();
    let mut previous = blank;
    for (index, frame) in logits.chunks_exact(vocab).enumerate() {
        let mut row = frame.to_vec();
        for &id in banned {
            row[id] = f32::NEG_INFINITY;
        }
        let token = crate::models::layers::argmax(&row);
        if token != blank && token != previous {
            tokens.push(token);
            frames.push(index);
        }
        previous = token;
    }
    (tokens, frames)
}
