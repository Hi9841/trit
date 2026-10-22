//! Beam search for the decoder families, by the rules of one of the two
//! reference implementations (`Rules`). Both keep `width` live sequences
//! and feed them to the decoder as one batch per step.

use crate::cpu::{activation, buf::Buf};

/// Whose beam search to follow. They differ in which candidates they look
/// at, when an ended sequence counts, when they stop, and how they rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rules {
    /// OpenAI Whisper's `BeamSearchDecoder`: each live sequence offers its
    /// `width + 1` best tokens; candidates are walked best first, and an
    /// ended one counts while fewer than `width` live ones are kept; the
    /// search stops when `width` sequences have ended; the best has the
    /// highest log probability per text token. Blocked tokens are removed
    /// before the softmax.
    OpenAi,
    /// Transformers' `generate(num_beams=width)` with its defaults (length
    /// penalty 1, `early_stopping=False`): the `2 * width` best candidates
    /// of all live sequences; an ended one counts only among the first
    /// `width`, scored per generated token (end of text included); the
    /// search stops when `width` have ended and no live sequence can beat
    /// the worst of them. Blocked tokens are removed after the softmax.
    Transformers,
}

/// The tokens of the best sequence and its mean log probability per token
/// (end of text included, as OpenAI counts it).
pub struct Best {
    pub tokens: Vec<usize>,
    pub log_probability: f32,
}

/// One live sequence: its decoder state and the logits for its next token.
struct Live<S> {
    tokens: Vec<usize>,
    score: f32,
    state: S,
    logits: Buf,
}

/// An ended sequence: its tokens (end of text not included), the sum of
/// its log probabilities, and its rank score.
struct Ended {
    tokens: Vec<usize>,
    sum: f32,
    rank: f32,
}

/// Searches with `width` beams by `rules` from `state`, whose next-token
/// logits are `logits`, for at most `limit` tokens. `mask(scores, tokens)`
/// blocks the tokens a sequence may not take next; `step(states, tokens)`
/// feeds `tokens[i]` to `states[i]`, all beams as one batch (the weights are
/// read once per step), and returns the next logits of each. `S` is the
/// decoder's own state (its self-attention caches), cloned when a beam splits.
pub fn search<S: Clone>(
    (width, rules): (usize, Rules),
    (state, logits): (S, Buf),
    (limit, eos): (usize, usize),
    mask: impl Fn(&mut [f32], &[usize]),
    mut step: impl FnMut(&mut [S], &[usize]) -> Vec<Buf>,
) -> Best {
    let mut live = vec![Live {
        tokens: Vec::new(),
        score: 0.0,
        state,
        logits,
    }];
    let mut ended: Vec<Ended> = Vec::new();
    for generated in 0..limit {
        let last = generated + 1 == limit;
        // (score, beam, token), best first.
        let mut candidates: Vec<(f32, usize, usize)> = Vec::new();
        for (b, beam) in live.iter_mut().enumerate() {
            // log softmax(x) = (x - max) - log sum exp(x - max); the order of
            // the logits is the order of their log probabilities.
            let (offered, (max, log_sum)) = match rules {
                Rules::OpenAi => {
                    mask(&mut beam.logits, &beam.tokens);
                    (width + 1, log_sum_exp(&beam.logits))
                }
                Rules::Transformers => {
                    let normalizer = log_sum_exp(&beam.logits);
                    mask(&mut beam.logits, &beam.tokens);
                    (2 * width, normalizer)
                }
            };
            for (token, logit) in top(&beam.logits, offered) {
                candidates.push((beam.score + ((logit - max) - log_sum), b, token));
            }
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        // (beam, token, score) of the sequences that go on.
        let mut kept: Vec<(usize, usize, f32)> = Vec::with_capacity(width);
        match rules {
            Rules::OpenAi => {
                for &(score, b, token) in &candidates {
                    if token == eos {
                        if ended.len() < width {
                            let tokens = live[b].tokens.clone();
                            let rank = score / tokens.len().max(1) as f32;
                            ended.push(Ended {
                                tokens,
                                sum: score,
                                rank,
                            });
                        }
                        continue;
                    }
                    kept.push((b, token, score));
                    if kept.len() == width {
                        break;
                    }
                }
            }
            Rules::Transformers => {
                for (place, &(score, b, token)) in candidates.iter().take(2 * width).enumerate() {
                    // At the token limit every candidate ends.
                    let ends = token == eos || last;
                    if ends && place < width {
                        let mut tokens = live[b].tokens.clone();
                        if token != eos {
                            tokens.push(token);
                        }
                        let rank = score / (live[b].tokens.len() + 1) as f32;
                        ended.push(Ended {
                            tokens,
                            sum: score,
                            rank,
                        });
                    }
                    if !ends && kept.len() < width {
                        kept.push((b, token, score));
                    }
                }
                // The `width` best ended sequences stay.
                ended.sort_by(|a, b| b.rank.total_cmp(&a.rank));
                ended.truncate(width);
            }
        }
        let mut states: Vec<S> = kept
            .iter()
            .map(|(b, _, _)| live[*b].state.clone())
            .collect();
        let fed: Vec<usize> = kept.iter().map(|(_, token, _)| *token).collect();
        let logits = if kept.is_empty() {
            Vec::new()
        } else {
            step(&mut states, &fed)
        };
        live = kept
            .iter()
            .zip(states)
            .zip(logits)
            .map(|((&(b, token, score), state), logits)| {
                let mut tokens = live[b].tokens.clone();
                tokens.push(token);
                Live {
                    tokens,
                    score,
                    state,
                    logits,
                }
            })
            .collect();
        let done = match rules {
            Rules::OpenAi => ended.len() >= width,
            // No live sequence can beat the worst ended one: the best live
            // score per token, if no later token cost anything.
            Rules::Transformers => {
                ended.len() >= width
                    && live.first().is_none_or(|best| {
                        best.score / best.tokens.len() as f32 <= ended[width - 1].rank
                    })
            }
        };
        if done || live.is_empty() {
            break;
        }
    }
    // The token limit ended the search: the live sequences count too.
    if ended.is_empty() {
        ended = live
            .into_iter()
            .map(|l| Ended {
                rank: l.score / l.tokens.len().max(1) as f32,
                sum: l.score,
                tokens: l.tokens,
            })
            .collect();
    }
    ended
        .into_iter()
        .max_by(|a, b| a.rank.total_cmp(&b.rank))
        .map(|best| Best {
            log_probability: best.sum / (best.tokens.len() + 1) as f32,
            tokens: best.tokens,
        })
        .unwrap_or(Best {
            tokens: Vec::new(),
            log_probability: 0.0,
        })
}

/// The maximum of `logits` and `log sum exp(logits - max)`, so that
/// `log softmax(x) = (x - max) - log_sum`. The exponentials go to a buffer
/// that each thread reuses: no vector the size of the vocabulary per call.
fn log_sum_exp(logits: &[f32]) -> (f32, f32) {
    thread_local! {
        static EXPS: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    EXPS.with(|exps| {
        let mut exps = exps.borrow_mut();
        exps.clear();
        exps.extend(logits.iter().map(|x| x - max));
        activation::exp(&mut exps);
        (max, exps.iter().sum::<f32>().ln())
    })
}

/// The `count` largest values with their indices, largest first.
fn top(values: &[f32], count: usize) -> Vec<(usize, f32)> {
    let mut best: Vec<(usize, f32)> = Vec::with_capacity(count + 1);
    for (i, &v) in values.iter().enumerate() {
        if v == f32::NEG_INFINITY || (best.len() == count && v <= best[count - 1].1) {
            continue;
        }
        let at = best.partition_point(|(_, b)| *b >= v);
        best.insert(at, (i, v));
        best.truncate(count);
    }
    best
}
