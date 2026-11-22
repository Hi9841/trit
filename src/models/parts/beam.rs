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

struct Live<S> {
    tokens: Vec<usize>,
    score: f32,
    state: S,
    /// The best next tokens and their scores, best first: filled by the
    /// beam's task of a step. The caller gives it its room, so the task
    /// allocates nothing.
    offers: Vec<(usize, f32)>,
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
/// read once per step), and returns their next logits packed as `beams x vocab`
/// in one contiguous buffer. `S` is the decoder's own state (its self-attention
/// caches), cloned when a beam splits.
pub fn search<S: Clone + Send>(
    (width, rules): (usize, Rules),
    (state, logits): (S, Buf),
    (limit, eos): (usize, usize),
    mask: impl Fn(&mut [f32], &[usize]) + Sync,
    mut step: impl FnMut(&mut [S], &[usize]) -> Buf,
) -> Best {
    let vocab = logits.len();
    if vocab == 0 {
        return Best {
            tokens: Vec::new(),
            log_probability: 0.0,
        };
    }
    // The candidates each sequence offers.
    let offered = match rules {
        Rules::OpenAi => width + 1,
        Rules::Transformers => 2 * width,
    };
    let mut live = vec![Live {
        tokens: Vec::new(),
        score: 0.0,
        state,
        offers: Vec::with_capacity(offered + 1),
    }];
    let mut logits = Some(logits);
    let mut ended: Vec<Ended> = Vec::new();
    for generated in 0..limit {
        let last = generated + 1 == limit;
        // (score, beam, token), best first.
        // One task per sequence: each reads its own logits row (three passes
        // over the vocabulary). One after the other on the caller, 5 beams
        // took 0.65 ms of each Whisper step, with all other threads idle.
        let mut step_logits = logits.take().expect("live beam logits");
        assert_eq!(step_logits.len(), live.len() * vocab);
        let scores = crate::prof::span(crate::prof::Stage::BeamScores);
        let logits_ptr = crate::cpu::team::SendPtr::new(&mut step_logits);
        crate::cpu::team::chunks_mut(&mut live, 1, |b, beam| {
            let beam = &mut beam[0];
            // SAFETY: each task owns a disjoint `vocab`-wide row of `step_logits`,
            // which outlives `chunks_mut`.
            let row = unsafe { logits_ptr.slice_mut(b * vocab, vocab) };
            // log softmax(x) = (x - max) - log sum exp(x - max); the order of
            // the logits is the order of their log probabilities.
            let (max, log_sum) = match rules {
                Rules::OpenAi => {
                    mask(row, &beam.tokens);
                    log_sum_exp(row)
                }
                Rules::Transformers => {
                    let normalizer = log_sum_exp(row);
                    mask(row, &beam.tokens);
                    normalizer
                }
            };
            top(row, offered, &mut beam.offers);
            let score = beam.score;
            for (_, logit) in &mut beam.offers {
                *logit = score + ((*logit - max) - log_sum);
            }
        });
        scores.end();
        // Return the `beams x vocab` buffer to the pool before `step` runs so
        // the next vocab projection reuses it in place.
        drop(step_logits);
        let mut candidates: Vec<(f32, usize, usize)> = Vec::with_capacity(live.len() * offered);
        for (b, beam) in live.iter().enumerate() {
            candidates.extend(beam.offers.iter().map(|(token, score)| (*score, b, *token)));
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
                ended.sort_by(|a, b| b.rank.total_cmp(&a.rank));
                ended.truncate(width);
            }
        }
        // A sequence that goes on in one beam gives its state to that beam.
        // Only a sequence that splits is copied: a copy of every state at
        // every step moved all the self-attention caches of all beams.
        let mut uses = vec![0usize; live.len()];
        for (b, _, _) in &kept {
            uses[*b] += 1;
        }
        let mut parents: Vec<(Vec<usize>, Option<S>)> = std::mem::take(&mut live)
            .into_iter()
            .map(|beam| (beam.tokens, Some(beam.state)))
            .collect();
        let mut states: Vec<S> = Vec::with_capacity(kept.len());
        for (b, _, _) in &kept {
            uses[*b] -= 1;
            let state = match uses[*b] {
                0 => parents[*b].1.take(),
                _ => {
                    let _span = crate::prof::span(crate::prof::Stage::BeamClone);
                    parents[*b].1.clone()
                }
            };
            states.extend(state);
        }
        let fed: Vec<usize> = kept.iter().map(|(_, token, _)| *token).collect();
        logits = if kept.is_empty() {
            None
        } else {
            let _span = crate::prof::span(crate::prof::Stage::BeamStep);
            Some(step(&mut states, &fed))
        };
        live = kept
            .iter()
            .zip(states)
            .map(|(&(b, token, score), state)| {
                let mut tokens = parents[b].0.clone();
                tokens.push(token);
                Live {
                    tokens,
                    score,
                    state,
                    offers: Vec::with_capacity(offered + 1),
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
/// `log softmax(x) = (x - max) - log_sum`. Sums exponentials directly from
/// `logits` in 16 SIMD lanes without writing a vocabulary-sized buffer.
pub(crate) fn log_sum_exp(logits: &[f32]) -> (f32, f32) {
    let max = activation::max(logits);
    (max, activation::exp_sum(logits, max).ln())
}

multiversion! {
    /// The `count` largest values with their indices, largest first, in `best`,
    /// which has room for `count + 1`. Minus infinity (a blocked token) is not
    /// a value.
    pub(crate) fn top(values: &[f32], count: usize, best: &mut Vec<(usize, f32)>) {
        /// Nearly every block of a vocabulary has no value above the current
        /// floor; one branchless vector scan over 32 values skips it.
        const BLOCK: usize = 32;
        best.clear();
        if count == 0 {
            return;
        }
        let (blocks, tail) = values.as_chunks::<BLOCK>();
        for (block, part) in blocks.iter().enumerate() {
            let floor = match best.len() == count {
                true => best[count - 1].1,
                false => f32::NEG_INFINITY,
            };
            if floor.is_nan()
                || part
                    .iter()
                    .fold(false, |hit, &v| hit | (v > floor) | v.is_nan())
            {
                offer(block * BLOCK, part, count, best);
            }
        }
        offer(blocks.len() * BLOCK, tail, count, best);
    }
}

/// Puts the values of `part`, whose indices start at `from`, into `best`:
/// the `count` largest so far, largest first.
#[inline(always)]
fn offer(from: usize, part: &[f32], count: usize, best: &mut Vec<(usize, f32)>) {
    for (i, &v) in part.iter().enumerate() {
        if v == f32::NEG_INFINITY || (best.len() == count && v <= best[count - 1].1) {
            continue;
        }
        let at = best.partition_point(|(_, b)| *b >= v);
        best.insert(at, (from + i, v));
        best.truncate(count);
    }
}
