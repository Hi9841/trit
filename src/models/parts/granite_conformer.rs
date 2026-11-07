//! IBM's Granite Speech Conformer encoder (Granite Speech 4.x, 5.0
//! TurboCTC): log-mel features (with deltas in 5.0) stacked in pairs, an
//! input projection, and blocks whose attention is local to blocks of
//! `context_size` frames with Shaw relative positions. Blocks listed in
//! `subsample_layers` halve the time (a stride-2 conv, the mean of each
//! frame pair as the residual). After half the blocks (or
//! `self_conditioning_layer`) the CTC softmax goes back into the encoder.
//! Granite Speech 4.1 plus also returns an earlier block's output beside the
//! last. The structure follows transcribe.cpp's port of IBM's
//! `GraniteSpeechCTCEncoder`; tensor names are under `encoder.`.

use crate::audio::dft::PowerSpectrum;
use crate::config::Json;
use crate::cpu::{activation, buf::Buf, fgemm, gemm::Dense, team, tensors::Tensors};
use crate::layers::{add_into, folded_batch_norm, linear, Norm};

/// The log floor before the log10, as the reference's `clamp(min=1e-10)`.
const LOG_FLOOR: f32 = 1e-10;

pub struct Config {
    pub d: usize,
    heads: usize,
    head_dim: usize,
    ffn: usize,
    layers: usize,
    input: usize,
    /// Outputs of the CTC head (the vocabulary in 5.0).
    pub ctc_dim: usize,
    kernel: usize,
    /// Channels of the conv module (`d * conv_expansion_factor`).
    inner: usize,
    /// Frames per attention block.
    context: usize,
    max_pos: usize,
    /// After this many blocks the CTC softmax goes back into the encoder.
    self_conditioning: usize,
    /// Blocks that halve the time.
    pub subsample: Vec<usize>,
    /// Blocks whose output (after the self-conditioning) the encoder
    /// returns, channels side by side: with the last block's after them
    /// (4.1 plus), or alone (4.1 NAR, whose list ends with the last block).
    cat_layers: Vec<usize>,
    cat_with_last: bool,
}

impl Config {
    pub fn parse(json: &Json) -> Result<Self, String> {
        let enc = |key: &str| json.num(&format!("encoder_config.{key}"));
        let list = |key: &str| -> Vec<usize> {
            json.get(&format!("encoder_config.{key}"))
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_u64().map(|v| v as usize))
                .collect()
        };
        let d = enc("hidden_size")?;
        let layers = enc("num_hidden_layers")?;
        let config = Self {
            d,
            heads: enc("num_attention_heads")?,
            head_dim: enc("head_dim")?,
            ffn: enc("intermediate_size")?,
            layers,
            input: enc("input_dim")?,
            ctc_dim: enc("output_dim")?,
            kernel: enc("conv_kernel_size")?,
            inner: d * enc("conv_expansion_factor")?,
            context: enc("context_size")?,
            max_pos: enc("max_pos_emb")?,
            self_conditioning: enc("self_conditioning_layer").unwrap_or(layers / 2),
            subsample: list("subsample_layers"),
            cat_layers: list("cat_hidden_layers"),
            cat_with_last: json.flag("encoder_config.cat_with_last") != Some(false),
        };
        if config.heads * config.head_dim != d
            || config.kernel.is_multiple_of(2)
            || config.context == 0
            || config.context > config.max_pos + 1
            || config.cat_layers.iter().any(|l| *l >= layers)
        {
            return Err("config.json describes a Granite encoder this engine does not run".into());
        }
        Ok(config)
    }

    /// Channels of the encoder output: each `cat_layers` block's, and the
    /// last block's unless only the listed ones.
    pub fn width(&self) -> usize {
        let last = if self.cat_layers.is_empty() || self.cat_with_last {
            1
        } else {
            0
        };
        self.d * (self.cat_layers.len() + last)
    }
}

/// Log-mel features as the reference computes them: a centered STFT
/// (reflect padding) with the file's window and filterbank, `log10`, a
/// floor `floor_db` below the take's peak, `(x + 4) / 4`, then (5.0) the
/// delta of each frame (`(next - previous) / 2`), and `stack` frames side
/// by side. 5.0 pads the audio to whole stacks; 4.x drops the frame of the
/// last center and then an odd last frame.
pub struct Frontend {
    n_fft: usize,
    hop: usize,
    pub mels: usize,
    pub stack: usize,
    floor_db: f32,
    pub deltas: bool,
    whole_stacks: bool,
    spectrum: PowerSpectrum,
    /// `mels x bins`.
    filters: Vec<f32>,
}

impl Frontend {
    pub fn load(json: &Json, t: &Tensors) -> Result<Self, String> {
        let fe = |key: &str| json.num(&format!("feature_extractor.{key}"));
        let (n_fft, win, mels) = (fe("n_fft")?, fe("win_length")?, fe("num_mel_bins")?);
        if win > n_fft {
            return Err("config.json: the window is longer than the transform".into());
        }
        let window = t.f32_len("frontend.window", win)?;
        // The window sits in the middle of the transform, as torch.stft.
        let mut padded = vec![0.0f32; n_fft];
        padded[(n_fft - win) / 2..][..win].copy_from_slice(&window);
        Ok(Self {
            n_fft,
            hop: fe("hop_length")?,
            mels,
            stack: fe("stack_factor")?,
            floor_db: json.float("feature_extractor.floor_db").unwrap_or(8.0) as f32,
            deltas: json.flag("feature_extractor.deltas") == Some(true),
            whole_stacks: json.str("feature_extractor.frames") == Some("whole_stacks"),
            spectrum: PowerSpectrum::new(n_fft, &padded),
            filters: t.f32_len("frontend.mel_filterbank", mels * (n_fft / 2 + 1))?,
        })
    }

    /// Values per stacked frame.
    pub fn width(&self) -> usize {
        self.stack * self.mels * if self.deltas { 2 } else { 1 }
    }

    /// `frames x width()`.
    pub fn compute(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        let (hop, stack, mels) = (self.hop, self.stack, self.mels);
        let mel_frames = audio.len() / hop;
        if mel_frames == 0 || audio.len() <= self.n_fft / 2 {
            return (Vec::new(), 0);
        }
        let mut samples = audio.to_vec();
        let frames = if self.whole_stacks {
            // Zeros after the audio up to the last frame's center.
            let frames = stack * mel_frames.div_ceil(stack);
            samples.resize(samples.len().max((frames - 1) * hop + 1), 0.0);
            frames
        } else {
            mel_frames
        };
        let len = samples.len() as isize;
        let reflect = |i: isize| -> f32 {
            let i = if i < 0 { -i } else { i };
            let i = if i >= len { 2 * (len - 1) - i } else { i };
            samples[i.clamp(0, len - 1) as usize]
        };
        let half = (self.n_fft / 2) as isize;
        let mut windows = Buf::scratch(frames * self.n_fft);
        team::chunks_mut(&mut windows, self.n_fft, |f, row| {
            let start = (f * hop) as isize - half;
            for (n, value) in row.iter_mut().enumerate() {
                *value = reflect(start + n as isize);
            }
        });
        let power = self.spectrum.compute(&windows, frames);
        let mel = fgemm::matmul(&power, frames, self.spectrum.bins(), &self.filters, mels);
        let mut logs: Vec<f32> = mel.iter().map(|m| m.max(LOG_FLOOR).log10()).collect();
        let peak = logs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in &mut logs {
            *v = (v.max(peak - self.floor_db) + 4.0) / 4.0;
        }
        let row = |f: usize| &logs[f * mels..(f + 1) * mels];
        let per_frame = self.width() / stack;
        let out_frames = frames / stack;
        let kept = out_frames * stack;
        let mut out = vec![0.0f32; kept * per_frame];
        for f in 0..kept {
            let at = &mut out[f * per_frame..(f + 1) * per_frame];
            at[..mels].copy_from_slice(row(f));
            if self.deltas {
                let (next, prev) = (row((f + 1).min(frames - 1)), row(f.saturating_sub(1)));
                for ((o, n), p) in at[mels..].iter_mut().zip(next).zip(prev) {
                    *o = 0.5 * (n - p);
                }
            }
        }
        (out, out_frames)
    }
}

struct Block {
    norm_ff1: Norm,
    ff1: (Dense, Dense),
    norm_attn: Norm,
    q: Dense,
    /// Keys then values, `2d x d`.
    kv: Dense,
    out: Dense,
    /// The relative position rows of one attention block, distance
    /// `r - (context - 1)` at row `r`: `(2 context - 1) x head_dim`.
    positions: Vec<f32>,
    norm_conv: Norm,
    pw1: Dense,
    /// Depthwise conv weights, `[tap][channel]`.
    dw: Vec<f32>,
    bn_scale: Vec<f32>,
    bn_shift: Vec<f32>,
    pw2: Dense,
    norm_ff2: Norm,
    ff2: (Dense, Dense),
    norm_out: Norm,
}

impl Block {
    fn load(t: &Tensors, i: usize, c: &Config) -> Result<Self, String> {
        let p = format!("encoder.blocks.{i}");
        let (d, inner, hd, k) = (c.d, c.inner, c.head_dim, c.kernel);
        let n = |name: &str| Norm::load(t, &format!("{p}.{name}"), d);
        let l = |name: &str, out: usize, inp: usize| linear(t, &format!("{p}.{name}"), out, inp);
        let table = t.f32_len(
            &format!("{p}.attn.rel_pos_emb.weight"),
            (2 * c.max_pos + 1) * hd,
        )?;
        let positions = (0..2 * c.context - 1)
            .flat_map(|r| {
                let at = r + c.max_pos + 1 - c.context;
                table[at * hd..(at + 1) * hd].iter().copied()
            })
            .collect();
        let bn = |name: &str| t.f32_len(&format!("{p}.conv.bn.{name}"), inner);
        let (weight, bias, mean, var) = (
            bn("weight")?,
            bn("bias")?,
            bn("running_mean")?,
            bn("running_var")?,
        );
        let (bn_scale, bn_shift) =
            folded_batch_norm(&weight, &bias, &mean, &var, &vec![0.0; inner]);
        let dw = t.f32_len(&format!("{p}.conv.depthwise.weight"), inner * k)?;
        Ok(Self {
            norm_ff1: n("norm_ff1")?,
            ff1: (l("ff1.linear1", c.ffn, d)?, l("ff1.linear2", d, c.ffn)?),
            norm_attn: n("norm_attn")?,
            q: l("attn.q", d, d)?,
            kv: l("attn.kv", 2 * d, d)?,
            out: l("attn.out", d, d)?,
            positions,
            norm_conv: n("norm_conv")?,
            pw1: l("conv.pointwise1", 2 * inner, d)?,
            dw: (0..k * inner)
                .map(|i| dw[(i % inner) * k + i / inner])
                .collect(),
            bn_scale,
            bn_shift,
            pw2: l("conv.pointwise2", d, inner)?,
            norm_ff2: n("norm_ff2")?,
            ff2: (l("ff2.linear1", c.ffn, d)?, l("ff2.linear2", d, c.ffn)?),
            norm_out: n("norm_out")?,
        })
    }

    /// The block on `t x d`; a subsampling block returns `t / 2` frames.
    fn forward(&self, mut x: Buf, t: usize, c: &Config, subsample: bool) -> (Buf, usize) {
        let ff = feed_forward(&self.ff1, &self.norm_ff1.apply(&x, c.d), t);
        add_scaled(&mut x, &ff, 0.5);
        let attended = self.attention(&self.norm_attn.apply(&x, c.d), t, c);
        add_into(&mut x, &attended);
        let stride = if subsample { 2 } else { 1 };
        let conv = self.convolution(&self.norm_conv.apply(&x, c.d), t, c, stride);
        let (mut x, t) = if subsample {
            let half = t / 2;
            let mut pooled = Buf::scratch(half * c.d);
            team::chunks_mut(&mut pooled, c.d, |i, row| {
                let (a, b) = (&x[2 * i * c.d..][..c.d], &x[(2 * i + 1) * c.d..][..c.d]);
                for ((o, a), b) in row.iter_mut().zip(a).zip(b) {
                    *o = 0.5 * (a + b);
                }
            });
            add_into(&mut pooled, &conv[..half * c.d]);
            (pooled, half)
        } else {
            add_into(&mut x, &conv);
            (x, t)
        };
        let ff = feed_forward(&self.ff2, &self.norm_ff2.apply(&x, c.d), t);
        add_scaled(&mut x, &ff, 0.5);
        (self.norm_out.apply(&x, c.d), t)
    }

    /// Attention inside blocks of `context` frames, with Shaw relative
    /// positions: score `(q_i . k_j + q_i . E[i - j]) / sqrt(head_dim)`. The
    /// last block is shorter; keys past the take do not exist.
    fn attention(&self, x: &[f32], t: usize, c: &Config) -> Buf {
        let (d, hd, ctx) = (c.d, c.head_dim, c.context);
        let q = self.q.forward(x, t);
        let kv = self.kv.forward(x, t);
        let blocks = t.div_ceil(ctx);
        let scale = 1.0 / (hd as f32).sqrt();
        let heads = team::map(blocks * c.heads, |task| {
            let (block, h) = (task / c.heads, task % c.heads);
            let first = block * ctx;
            let n = ctx.min(t - first);
            let gather = |src: &[f32], stride: usize, offset: usize| -> Vec<f32> {
                (0..n)
                    .flat_map(|i| src[(first + i) * stride + offset..][..hd].iter().copied())
                    .collect()
            };
            let qh = gather(&q, d, h * hd);
            let kh = gather(&kv, 2 * d, h * hd);
            let vh = gather(&kv, 2 * d, d + h * hd);
            let mut out = vec![0.0f32; n * hd];
            let mut scores = vec![0.0f32; n];
            for i in 0..n {
                let qi = &qh[i * hd..(i + 1) * hd];
                for (j, s) in scores.iter_mut().enumerate() {
                    let kj = &kh[j * hd..(j + 1) * hd];
                    let pos = &self.positions[(i + ctx - 1 - j) * hd..][..hd];
                    let dot: f32 = qi
                        .iter()
                        .zip(kj)
                        .zip(pos)
                        .map(|((q, k), p)| q * (k + p))
                        .sum();
                    *s = dot * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let row = &mut out[i * hd..(i + 1) * hd];
                for (j, s) in scores.iter().enumerate() {
                    let w = s / sum;
                    for (o, v) in row.iter_mut().zip(&vh[j * hd..(j + 1) * hd]) {
                        *o += w * v;
                    }
                }
            }
            out
        });
        let mut joined = Buf::zeroed(t * d);
        for (task, out) in heads.iter().enumerate() {
            let (block, h) = (task / c.heads, task % c.heads);
            for (i, row) in out.chunks_exact(hd).enumerate() {
                let at = (block * ctx + i) * d + h * hd;
                joined[at..at + hd].copy_from_slice(row);
            }
        }
        self.out.forward(&joined, t)
    }

    /// Pointwise conv with GLU, depthwise conv over time (stride 1 or 2),
    /// batch norm, SiLU, pointwise conv. Returns `out_t x d`.
    fn convolution(&self, x: &[f32], t: usize, c: &Config, stride: usize) -> Buf {
        let (inner, k) = (c.inner, c.kernel);
        let wide = self.pw1.forward(x, t);
        let mut gated = Buf::scratch(t * inner);
        team::chunks_mut(&mut gated, inner, |i, row| {
            let w = &wide[i * 2 * inner..(i + 1) * 2 * inner];
            row.copy_from_slice(&w[inner..]);
            activation::sigmoid(row);
            for (o, v) in row.iter_mut().zip(&w[..inner]) {
                *o *= v;
            }
        });
        let pad = k / 2;
        let out_t = (t + 2 * pad - k) / stride + 1;
        let mut mixed = Buf::zeroed(out_t * inner);
        team::chunks_mut(&mut mixed, inner, |o, row| {
            for tap in 0..k {
                let Some(src) = (o * stride + tap).checked_sub(pad).filter(|s| *s < t) else {
                    continue;
                };
                let w = &self.dw[tap * inner..(tap + 1) * inner];
                for ((r, w), g) in row.iter_mut().zip(w).zip(&gated[src * inner..][..inner]) {
                    *r += w * g;
                }
            }
            for ((r, s), b) in row.iter_mut().zip(&self.bn_scale).zip(&self.bn_shift) {
                *r = *r * s + b;
            }
            activation::silu(row);
        });
        self.pw2.forward(&mixed, out_t)
    }
}

fn feed_forward((up, down): &(Dense, Dense), x: &[f32], t: usize) -> Buf {
    let mut hidden = up.forward(x, t);
    team::chunks_mut(&mut hidden, 4096, |_, values| activation::silu(values));
    down.forward(&hidden, t)
}

fn add_scaled(x: &mut [f32], y: &[f32], scale: f32) {
    team::chunks_mut(x, 4096, |i, row| {
        for (o, v) in row.iter_mut().zip(&y[i * 4096..]) {
            *o += scale * v;
        }
    });
}

/// What `Encoder::encode_all` returns.
pub struct Encoded {
    pub joined: Buf,
    /// The last block's output, `t x d`.
    pub last: Buf,
    pub t: usize,
    /// The self-conditioning CTC's blank probability per frame (channel 0).
    pub blank: Vec<f32>,
}

pub struct Encoder {
    pub config: Config,
    input: Dense,
    blocks: Vec<Block>,
    /// The CTC head, also the first half of the self-conditioning.
    ctc: Dense,
    /// The CTC softmax back to the model width.
    bypass: Dense,
}

impl Encoder {
    pub fn load(t: &Tensors, config: Config) -> Result<Self, String> {
        let c = &config;
        let blocks = team::map(c.layers, |i| Block::load(t, i, c))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            input: linear(t, "encoder.input_linear", c.d, c.input)?,
            ctc: linear(t, "encoder.ctc_proj", c.ctc_dim, c.d)?,
            bypass: linear(t, "encoder.ctc_bypass", c.d, c.ctc_dim)?,
            blocks,
            config,
        })
    }

    /// The encoder output, `t x width()`, from `frames` stacked features.
    pub fn encode(&self, features: &[f32], frames: usize) -> (Buf, usize) {
        let encoded = self.encode_all(features, frames);
        (encoded.joined, encoded.t)
    }

    /// The output (`joined`, `t x width()`), the last block's output, and
    /// the blank's probability under the self-conditioning CTC per frame.
    pub fn encode_all(&self, features: &[f32], frames: usize) -> Encoded {
        let c = &self.config;
        let (mut x, mut t) = (self.input.forward(features, frames), frames);
        let mut kept = Vec::new();
        let mut blank = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            (x, t) = block.forward(x, t, c, c.subsample.contains(&i));
            if i + 1 == c.self_conditioning {
                let mut soft = self.ctc.forward(&x, t);
                team::chunks_mut(&mut soft, c.ctc_dim, |_, row| softmax(row));
                blank = soft.chunks_exact(c.ctc_dim).map(|row| row[0]).collect();
                add_into(&mut x, &self.bypass.forward(&soft, t));
            }
            if c.cat_layers.contains(&i) {
                kept.push(x.to_vec());
            }
        }
        if kept.is_empty() {
            let mut last = Buf::scratch(x.len());
            last.copy_from_slice(&x);
            return Encoded {
                joined: x,
                last,
                t,
                blank,
            };
        }
        // The kept outputs first, the last block's after, per frame.
        let width = c.width();
        let tail: &[&[f32]] = if c.cat_with_last { &[&x[..]] } else { &[] };
        let parts: Vec<&[f32]> = kept
            .iter()
            .map(Vec::as_slice)
            .chain(tail.iter().copied())
            .collect();
        let mut joined = Buf::scratch(t * width);
        team::chunks_mut(&mut joined, width, |f, row| {
            for (k, part) in parts.iter().enumerate() {
                row[k * c.d..(k + 1) * c.d].copy_from_slice(&part[f * c.d..(f + 1) * c.d]);
            }
        });
        Encoded {
            joined,
            last: x,
            t,
            blank,
        }
    }

    /// CTC logits of encoder output `x` (`t x d`).
    pub fn ctc_logits(&self, x: &[f32], t: usize) -> Buf {
        self.ctc.forward(x, t)
    }
}

fn softmax(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    row.iter_mut().for_each(|v| *v /= sum);
}
