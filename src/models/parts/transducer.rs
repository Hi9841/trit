//! The transducer head of Parakeet: an LSTM prediction network and a joint
//! network over the encoder frames. TDT (`parakeet_tdt`) also predicts how
//! many frames each token lasts; RNNT (`parakeet_rnnt`) moves one frame per
//! blank. Greedy decoding, and the forward algorithm that scores dictionary
//! terms against the sound (`dictionary.rs`). The structure follows the
//! Apache-2.0 Hugging Face `ParakeetForTDT`.

use super::dictionary::{Candidate, Check, Dictionary, CAPITAL_PRIOR};
use super::fastconformer::{dense, half_dense, Config};
use crate::cpu::activation;
use crate::cpu::buf::Buf;
use crate::cpu::gemm::Dense;
use crate::cpu::half::HalfDense;
use crate::cpu::team;
use crate::cpu::tensors::Tensors;
use crate::models::layers::argmax;
use std::collections::HashMap;

struct Lstm {
    /// `[W_ih | W_hh]`, applied to `[input | hidden]`.
    gates: HalfDense,
    bias: Vec<f32>,
}

impl Lstm {
    /// One step of the cell from the gate products: updates `h` and `c`.
    fn cell(&self, gates: &mut [f32], h: &mut [f32], c: &mut [f32]) {
        let dh = h.len();
        for (g, b) in gates.iter_mut().zip(&self.bias) {
            *g += b;
        }
        // PyTorch gate order: input, forget, cell, output.
        let (input_forget, rest) = gates.split_at_mut(2 * dh);
        let (cell_gate, output) = rest.split_at_mut(dh);
        activation::sigmoid(input_forget);
        activation::tanh(cell_gate);
        activation::sigmoid(output);
        for u in 0..dh {
            c[u] = input_forget[dh + u] * c[u] + input_forget[u] * cell_gate[u];
        }
        h.copy_from_slice(c);
        activation::tanh(h);
        for (value, gate) in h.iter_mut().zip(output.iter()) {
            *value *= gate;
        }
    }
}

#[derive(Clone, Default)]
pub struct DecoderState {
    hidden: Vec<Vec<f32>>,
    cell: Vec<Vec<f32>>,
}

pub(crate) struct Transducer {
    pub(crate) encoder_projector: Dense,
    embedding: Vec<f32>,
    lstm: Vec<Lstm>,
    decoder_projector: HalfDense,
    joint: HalfDense,
}

impl Transducer {
    pub(crate) fn load(t: &Tensors, c: &Config) -> Result<Self, String> {
        let dh = c.decoder_hidden;
        let lstm = (0..c.decoder_layers)
            .map(|l| -> Result<Lstm, String> {
                let b_ih = t.f32_len(&format!("decoder.lstm.bias_ih_l{l}"), 4 * dh)?;
                let b_hh = t.f32_len(&format!("decoder.lstm.bias_hh_l{l}"), 4 * dh)?;
                // LSTM weights carry no `.weight` suffix.
                let w_ih = t.f16_bits(&format!("decoder.lstm.weight_ih_l{l}"), 4 * dh * dh)?;
                let w_hh = t.f16_bits(&format!("decoder.lstm.weight_hh_l{l}"), 4 * dh * dh)?;
                let joined = HalfDense::side_by_side(&w_ih, &w_hh, 4 * dh, dh, dh);
                Ok(Lstm {
                    gates: HalfDense::new(joined, None, 4 * dh, 2 * dh)?,
                    bias: b_ih.iter().zip(&b_hh).map(|(a, b)| a + b).collect(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            encoder_projector: dense(t, "encoder_projector", dh, c.hidden, true)?,
            embedding: t.f32_len("decoder.embedding.weight", c.vocab * dh)?,
            lstm,
            decoder_projector: half_dense(t, "decoder.decoder_projector", dh, dh)?,
            joint: half_dense(t, "joint.head", c.vocab + c.durations.len(), dh)?,
        })
    }
}

impl Transducer {
    fn decoder_step(&self, c: &Config, token: usize, state: &mut DecoderState) -> Buf {
        let dh = c.decoder_hidden;
        let mut input = self.embedding[token * dh..(token + 1) * dh].to_vec();
        let mut joined = vec![0.0f32; 2 * dh];
        for (layer, lstm) in self.lstm.iter().enumerate() {
            joined[..dh].copy_from_slice(&input);
            joined[dh..].copy_from_slice(&state.hidden[layer]);
            let mut gates = lstm.gates.matvec(&joined);
            lstm.cell(&mut gates, &mut state.hidden[layer], &mut state.cell[layer]);
            input.copy_from_slice(&state.hidden[layer]);
        }
        self.decoder_projector.matvec(&input)
    }

    /// `decoder_step` for one token of each state at once: each weight is
    /// read once for all of them. Returns the outputs, one row per state.
    fn decoder_steps(&self, c: &Config, tokens: &[usize], states: &mut [DecoderState]) -> Buf {
        let (dh, n) = (c.decoder_hidden, tokens.len());
        let mut input: Vec<f32> = tokens
            .iter()
            .flat_map(|&t| self.embedding[t * dh..(t + 1) * dh].iter().copied())
            .collect();
        let mut joined = vec![0.0f32; n * 2 * dh];
        for (layer, lstm) in self.lstm.iter().enumerate() {
            for (i, row) in joined.chunks_exact_mut(2 * dh).enumerate() {
                row[..dh].copy_from_slice(&input[i * dh..(i + 1) * dh]);
                row[dh..].copy_from_slice(&states[i].hidden[layer]);
            }
            let mut gates = lstm.gates.forward(&joined, n);
            for (i, (row, state)) in gates
                .chunks_exact_mut(4 * dh)
                .zip(states.iter_mut())
                .enumerate()
            {
                lstm.cell(row, &mut state.hidden[layer], &mut state.cell[layer]);
                input[i * dh..(i + 1) * dh].copy_from_slice(&state.hidden[layer]);
            }
        }
        self.decoder_projector.forward(&input, n)
    }

    /// Greedy TDT decoding over projected encoder frames (`t x decoder_hidden`).
    /// `dictionary` terms then replace the words heard where the sound
    /// allows (see `dictionary.rs`).
    pub(crate) fn decode(
        &self,
        c: &Config,
        projected: &[f32],
        t: usize,
        dictionary: Option<&Dictionary>,
        banned: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        let run = self.greedy(c, projected, t, banned, dictionary.is_some());
        let Greedy {
            mut tokens,
            mut frames,
            trail,
        } = run;
        if let Some(dictionary) = dictionary {
            let heard = Heard {
                projected,
                t,
                banned,
                trail: &trail,
            };
            self.write_terms(c, dictionary, &heard, &mut tokens, &mut frames);
        }
        (tokens, frames)
    }

    /// One greedy pass; `keep` records the decoder before each token for
    /// the dictionary.
    fn greedy(
        &self,
        c: &Config,
        projected: &[f32],
        t: usize,
        banned: &[usize],
        keep: bool,
    ) -> Greedy {
        let dh = c.decoder_hidden;
        let mut state = DecoderState {
            hidden: vec![vec![0.0; dh]; c.decoder_layers],
            cell: vec![vec![0.0; dh]; c.decoder_layers],
        };
        let mut dec = self.decoder_step(c, c.blank, &mut state);
        let mut run = Greedy {
            tokens: Vec::new(),
            frames: Vec::new(),
            trail: Vec::new(),
        };
        let snapshot = |trail: &mut Vec<Snapshot>, state: &DecoderState, dec: &[f32], next| {
            if keep {
                trail.push(Snapshot {
                    state: state.clone(),
                    dec: dec.to_vec(),
                    next,
                });
            }
        };
        snapshot(&mut run.trail, &state, &dec, 0);
        let (mut frame, mut symbols) = (0usize, 0usize);
        let mut joint_in = vec![0.0f32; dh];
        while frame < t {
            let enc = &projected[frame * dh..(frame + 1) * dh];
            for i in 0..dh {
                joint_in[i] = (enc[i] + dec[i]).max(0.0);
            }
            let mut logits = self.joint.matvec(&joint_in);
            for &id in banned {
                logits[id] = f32::NEG_INFINITY;
            }
            let token = argmax(&logits[..c.vocab]);
            // RNNT has no durations: a token stays on its frame.
            let mut duration = match c.durations.is_empty() {
                true => 0,
                false => c.durations[argmax(&logits[c.vocab..])],
            };
            if token == c.blank {
                duration = duration.max(1);
            } else {
                run.tokens.push(token);
                run.frames.push(frame);
                dec = self.decoder_step(c, token, &mut state);
                symbols += 1;
                if duration == 0 && symbols >= c.max_symbols {
                    duration = 1;
                }
                snapshot(&mut run.trail, &state, &dec, frame + duration);
            }
            if duration > 0 {
                symbols = 0;
            }
            frame += duration;
        }
        run
    }

    /// Replaces words of `tokens` with dictionary terms that the sound
    /// supports, the clearest wins first; a term takes the frame of the
    /// first word it replaces.
    fn write_terms(
        &self,
        c: &Config,
        dictionary: &Dictionary,
        heard: &Heard,
        tokens: &mut Vec<usize>,
        frames: &mut Vec<usize>,
    ) {
        let candidates = dictionary.candidates(tokens);
        // Every spelling to score, once: the words heard and each term.
        let mut jobs: Vec<Job> = Vec::new();
        let mut job = |start: usize, spelling: &[usize], end: End| -> usize {
            let found = jobs
                .iter()
                .position(|j| j.start == start && j.tokens == spelling && j.end == end);
            found.unwrap_or_else(|| {
                jobs.push(Job {
                    start,
                    tokens: spelling.to_vec(),
                    end,
                });
                jobs.len() - 1
            })
        };
        // Per candidate: the job of the words heard, the jobs of the term, the prior.
        let mut plans = Vec::new();
        for candidate in &candidates {
            let span = candidate.span.clone();
            let end = match tokens.get(span.end) {
                // A term ends within `SPAN_REACH` frames of the words it
                // replaces: a long pause after them is not scored.
                Some(&token) if frames[span.end] <= frames[span.end - 1] + SPAN_REACH => {
                    End::Token(token, frames[span.end])
                }
                _ => End::Close(heard.t.min(frames[span.end - 1] + SPAN_REACH)),
            };
            let (spellings, prior) = match candidate.check {
                Check::None => (Vec::new(), 0.0),
                Check::Capital => (vec![dictionary.written(candidate)], CAPITAL_PRIOR),
                Check::Sound => (
                    dictionary
                        .spellings(candidate)
                        .iter()
                        .map(Vec::as_slice)
                        .collect(),
                    dictionary.limit(candidate),
                ),
            };
            let words =
                (!spellings.is_empty()).then(|| job(span.start, &tokens[span.clone()], end));
            let terms: Vec<usize> = spellings.iter().map(|s| job(span.start, s, end)).collect();
            plans.push((words, terms, prior));
        }
        let scores = self.score(c, heard, &jobs);
        let mut wins = Vec::new();
        for (candidate, (words, terms, prior)) in candidates.into_iter().zip(plans) {
            let gain = match words {
                None => f32::INFINITY,
                Some(words) => {
                    let term = terms
                        .iter()
                        .map(|&j| scores[j])
                        .fold(f32::NEG_INFINITY, f32::max);
                    term + prior - scores[words]
                }
            };
            if gain >= 0.0 {
                wins.push((gain, candidate));
            }
        }
        // A term that explains more of the words wins over a shorter one
        // inside it: "trip run time" is "Trit Runtime", not "Trit run time".
        wins.sort_by(|a, b| (b.1.span.len().cmp(&a.1.span.len())).then(b.0.total_cmp(&a.0)));
        let mut chosen: Vec<Candidate> = Vec::new();
        for (_, candidate) in wins {
            let free = chosen.iter().all(|other| {
                other.span.end <= candidate.span.start || candidate.span.end <= other.span.start
            });
            if free {
                chosen.push(candidate);
            }
        }
        chosen.sort_by_key(|candidate| std::cmp::Reverse(candidate.span.start));
        for candidate in chosen {
            let (span, term) = (candidate.span.clone(), dictionary.written(&candidate));
            let at = frames[span.start];
            frames.splice(span.clone(), std::iter::repeat_n(at, term.len()));
            tokens.splice(span, term.iter().copied());
        }
    }

    /// The log-probability of each job: that the decoder at its start emits
    /// its tokens and then reaches its end, summed over every alignment to
    /// the frames between (the TDT forward algorithm). Blanks move at least
    /// one frame; tokens may stay on theirs. All jobs share the decoder
    /// steps and one joint product, so each weight is read once.
    fn score(&self, c: &Config, heard: &Heard, jobs: &[Job]) -> Vec<f32> {
        let dh = c.decoder_hidden;
        // The frames each job crosses: from where its start left the
        // decoder to its end, and the anchor token there.
        let spans: Vec<Option<(usize, usize, Option<usize>)>> = jobs
            .iter()
            .map(|job| {
                let from = heard.trail[job.start].next;
                let (last, anchor) = match job.end {
                    End::Token(token, frame) => (frame, Some(token)),
                    End::Close(frame) => (frame, None),
                };
                (last >= from).then(|| (from, last - from, anchor))
            })
            .collect();
        // Jobs that share a start and a first few tokens share the decoder
        // there, and so the joint rows: one node per (start, prefix).
        let mut nodes: Vec<Node> = Vec::new();
        let mut index: HashMap<(usize, &[usize]), usize> = HashMap::new();
        // Per job: its node before each token and after the last.
        let mut paths: Vec<Vec<usize>> = Vec::with_capacity(jobs.len());
        for (job, span) in jobs.iter().zip(&spans) {
            let Some((_, width, anchor)) = *span else {
                paths.push(Vec::new());
                continue;
            };
            let count = job.tokens.len();
            let mut path = Vec::with_capacity(count + 1);
            for u in 0..=count {
                let key = (job.start, &job.tokens[..u]);
                let node = *index.entry(key).or_insert_with(|| {
                    nodes.push(Node {
                        parent: path.last().copied(),
                        token: u.checked_sub(1).map(|p| job.tokens[p]),
                        start: job.start,
                        rows: 0,
                        base: 0,
                    });
                    nodes.len() - 1
                });
                let needed = rows(width, anchor);
                nodes[node].rows = nodes[node].rows.max(needed);
                path.push(node);
            }
            paths.push(path);
        }
        // Decoder output of each node, one batch per depth: parents come first.
        let mut outputs: Vec<Vec<f32>> = vec![Vec::new(); nodes.len()];
        let mut states: Vec<DecoderState> = vec![DecoderState::default(); nodes.len()];
        let mut pending: Vec<usize> = Vec::new();
        for (i, node) in nodes.iter().enumerate() {
            match node.parent {
                None => {
                    outputs[i] = heard.trail[node.start].dec.clone();
                    states[i] = heard.trail[node.start].state.clone();
                }
                Some(_) => pending.push(i),
            }
        }
        while !pending.is_empty() {
            let (ready, wait): (Vec<usize>, Vec<usize>) = pending
                .iter()
                .partition(|&&i| nodes[i].parent.is_some_and(|p| !outputs[p].is_empty()));
            if ready.is_empty() {
                break;
            }
            let step: Vec<usize> = ready.iter().filter_map(|&i| nodes[i].token).collect();
            let mut batch: Vec<DecoderState> = ready
                .iter()
                .filter_map(|&i| nodes[i].parent.map(|p| states[p].clone()))
                .collect();
            let out = self.decoder_steps(c, &step, &mut batch);
            for ((&i, state), row) in ready.iter().zip(batch).zip(out.chunks_exact(dh)) {
                states[i] = state;
                outputs[i] = row.to_vec();
            }
            pending = wait;
        }
        // The joint input of every node and frame.
        let mut input = Vec::new();
        let mut total = 0;
        for (node, output) in nodes.iter_mut().zip(&outputs) {
            let from = heard.trail[node.start].next;
            node.base = total;
            total += node.rows;
            for f in from..from + node.rows {
                let enc = &heard.projected[f * dh..(f + 1) * dh];
                input.extend(enc.iter().zip(output).map(|(e, d)| (e + d).max(0.0)));
            }
        }
        let cols = c.vocab + c.durations.len();
        let mut logits = if total == 0 {
            Buf::scratch(0)
        } else {
            self.joint.forward(&input, total)
        };
        // Every row as log-probabilities, in parallel.
        team::chunks_mut(&mut logits, cols, |_, row| {
            for &id in heard.banned {
                row[id] = f32::NEG_INFINITY;
            }
            let (token_part, duration_part) = row.split_at_mut(c.vocab);
            log_softmax(token_part);
            log_softmax(duration_part);
        });
        let row = |node: usize, f: usize| {
            let at = (nodes[node].base + f) * cols;
            logits[at..at + cols].split_at(c.vocab)
        };
        // RNNT: a blank moves one frame and a token none, both for certain.
        let rnnt = c.durations.is_empty();
        let rnnt_moves = [(0, 0.0), (1, 0.0)].into_iter().filter(|_| rnnt);
        let mut out = Vec::with_capacity(jobs.len());
        for ((job, span), path) in jobs.iter().zip(&spans).zip(&paths) {
            let Some((_, width, anchor)) = *span else {
                out.push(f32::NEG_INFINITY);
                continue;
            };
            let count = job.tokens.len();
            let at = |frame: usize, u: usize| frame * (count + 1) + u;
            let mut alpha = vec![f32::NEG_INFINITY; (width + 1) * (count + 1)];
            alpha[0] = 0.0;
            for (u, &node) in path.iter().enumerate() {
                let next = if u < count {
                    Some(job.tokens[u])
                } else {
                    anchor
                };
                for f in 0..rows(width, anchor) {
                    let here = alpha[at(f, u)];
                    if here == f32::NEG_INFINITY {
                        continue;
                    }
                    let (token_part, duration_part) = row(node, f);
                    let tdt = c
                        .durations
                        .iter()
                        .copied()
                        .zip(duration_part.iter().copied());
                    for (d, p) in tdt.chain(rnnt_moves.clone()) {
                        // A path past the anchor misses it; past the end it closes.
                        let target = if f + d <= width {
                            f + d
                        } else if anchor.is_some() {
                            continue;
                        } else {
                            width
                        };
                        if d > 0 {
                            let cell = &mut alpha[at(target, u)];
                            *cell = log_add(*cell, here + token_part[c.blank] + p);
                        }
                        if let (true, Some(id), true) = (u < count, next, d == 0 || !rnnt) {
                            let cell = &mut alpha[at(target, u + 1)];
                            *cell = log_add(*cell, here + token_part[id] + p);
                        }
                    }
                }
            }
            let end = alpha[at(width, count)];
            out.push(match anchor {
                // The anchor token on its own frame, after the last token.
                Some(id) => end + row(path[count], width).0[id],
                None => end,
            });
        }
        out
    }
}

/// A decoder position shared by the jobs that reach it: their start and a
/// prefix of their tokens.
struct Node {
    parent: Option<usize>,
    /// The last token of the prefix; `None` at the start.
    token: Option<usize>,
    start: usize,
    /// Joint rows: the frames from the start that the jobs here cross.
    rows: usize,
    /// The first of those rows in the joint product.
    base: usize,
}

/// Joint rows of each step of a job: the frames it crosses, and the
/// anchor's own frame.
fn rows(width: usize, anchor: Option<usize>) -> usize {
    // Tokens may stay on the anchor's frame, before the anchor itself.
    if anchor.is_some() {
        width + 1
    } else {
        width
    }
}

/// Frames after the last word of a span that its sound check covers: 2 s.
const SPAN_REACH: usize = 25;

/// A spelling to score: from the decoder before token `start` of the
/// greedy pass, through `tokens`, to `end`.
struct Job {
    start: usize,
    tokens: Vec<usize>,
    end: End,
}

/// The tokens of one greedy pass, the encoder frame of each, and the
/// decoder before each (when kept).
struct Greedy {
    tokens: Vec<usize>,
    frames: Vec<usize>,
    trail: Vec<Snapshot>,
}

/// The decoder after a token, kept for the dictionary's sound check.
struct Snapshot {
    state: DecoderState,
    dec: Vec<f32>,
    /// The frame decoding moved to after the token.
    next: usize,
}

/// What one greedy decode heard, for scoring other words over its frames.
struct Heard<'a> {
    projected: &'a [f32],
    t: usize,
    banned: &'a [usize],
    /// `trail[i]`: the decoder before token `i`.
    trail: &'a [Snapshot],
}

/// Where a scored span ends: the next token heard and its frame, or the
/// end of the window.
#[derive(Clone, Copy, PartialEq, Eq)]
enum End {
    Token(usize, usize),
    /// Every path that reaches this frame, or goes past it, ends.
    Close(usize),
}

thread_local! {
    /// `log_softmax` scratch: a row of the joint, once per thread.
    static EXPS: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn log_softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = EXPS.with(|exps| {
        let mut exps = exps.borrow_mut();
        exps.clear();
        exps.extend(values.iter().map(|v| v - max));
        activation::exp(&mut exps);
        exps.iter().sum()
    });
    let norm = max + sum.ln();
    values.iter_mut().for_each(|v| *v -= norm);
}

/// `ln(e^a + e^b)` without overflow.
fn log_add(a: f32, b: f32) -> f32 {
    let (high, low) = if a > b { (a, b) } else { (b, a) };
    if low == f32::NEG_INFINITY {
        return high;
    }
    high + (low - high).exp().ln_1p()
}
