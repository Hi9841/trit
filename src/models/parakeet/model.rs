//! Parakeet TDT: a FastConformer encoder and a token-and-duration transducer.
//! The structure follows the Apache-2.0 Hugging Face `ParakeetForTDT`.

use super::dictionary::{Candidate, Check, Dictionary, CAPITAL_PRIOR};
use super::english::{English, Partial};
use crate::cpu::activation;
use crate::cpu::buf::Buf;
use crate::cpu::fgemm;
use crate::cpu::gemm::Dense;
use crate::cpu::half::HalfDense;
use crate::cpu::kernels::Ternary;
use crate::cpu::team;
use crate::cpu::tensors::Tensors;
use crate::models::transformer::Norm;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

const BATCH_NORM_EPS: f32 = 1e-5;

thread_local! {
    /// One attention head's query (two biases), key, and position columns.
    /// Each grows to the longest take once per thread and is then reused,
    /// so the worker threads allocate nothing per head.
    static HEADS: std::cell::RefCell<[Vec<f32>; 4]> = const {
        std::cell::RefCell::new([Vec::new(), Vec::new(), Vec::new(), Vec::new()])
    };
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub ff: usize,
    pub heads: usize,
    pub layers: usize,
    pub mels: usize,
    pub conv_kernel: usize,
    pub sub_channels: usize,
    pub decoder_hidden: usize,
    pub decoder_layers: usize,
    pub vocab: usize,
    pub blank: usize,
    pub durations: Vec<usize>,
    pub max_symbols: usize,
    pub group: usize,
    /// `sqrt(hidden)` when the encoder scales its input (`scale_input`), else 1.
    pub input_scale: f32,
    pub head: HeadKind,
    /// Before every `encoder.` tensor name: `model.` in Canary files.
    pub prefix: &'static str,
}

/// What follows the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadKind {
    /// TDT decoder and joint network (`parakeet_tdt`).
    Tdt,
    /// One projection per frame (`parakeet_ctc`).
    Ctc,
    /// None: another model (Canary) decodes the encoder output.
    Encoder,
}

impl Config {
    pub fn parse(text: &str) -> Result<Self, String> {
        let json: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("config.json is not JSON: {e}"))?;
        let enc = &json["encoder_config"];
        let num = |value: &serde_json::Value, key: &str| -> Result<usize, String> {
            value[key]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("config.json has no {key}"))
        };
        let head = match json["model_type"].as_str() {
            Some("parakeet_ctc") => HeadKind::Ctc,
            Some("canary") => HeadKind::Encoder,
            _ => HeadKind::Tdt,
        };
        // Only the shape this engine implements is accepted.
        if num(enc, "subsampling_factor")? != 8
            || num(enc, "subsampling_conv_kernel_size")? != 3
            || num(enc, "subsampling_conv_stride")? != 2
        {
            return Err("config.json describes a model shape this engine does not run".into());
        }
        let hidden = num(enc, "hidden_size")?;
        // Only TDT has the decoder fields; a CTC blank is the pad token.
        let tdt = |key: &str| {
            if head == HeadKind::Tdt {
                num(&json, key)
            } else {
                Ok(0)
            }
        };
        let config = Self {
            hidden,
            ff: num(enc, "intermediate_size")?,
            heads: num(enc, "num_attention_heads")?,
            layers: num(enc, "num_hidden_layers")?,
            mels: num(enc, "num_mel_bins")?,
            conv_kernel: num(enc, "conv_kernel_size")?,
            sub_channels: num(enc, "subsampling_conv_channels")?,
            decoder_hidden: tdt("decoder_hidden_size")?,
            decoder_layers: tdt("num_decoder_layers")?,
            vocab: num(&json, "vocab_size")?,
            blank: num(
                &json,
                if head == HeadKind::Tdt {
                    "blank_token_id"
                } else {
                    "pad_token_id"
                },
            )?,
            durations: match json["durations"].as_array() {
                Some(list) => list
                    .iter()
                    .map(|d| d.as_u64().map(|d| d as usize).ok_or("bad duration"))
                    .collect::<Result<_, _>>()?,
                None if head != HeadKind::Tdt => Vec::new(),
                None => return Err("config.json has no durations".into()),
            },
            max_symbols: tdt("max_symbols_per_step")?,
            // Ternary models (Redux) only; dense models have no groups.
            group: json["ternary_group_size"].as_u64().unwrap_or(0) as usize,
            input_scale: if enc["scale_input"].as_bool() == Some(true) {
                (hidden as f32).sqrt()
            } else {
                1.0
            },
            head,
            prefix: if head == HeadKind::Encoder {
                "model."
            } else {
                ""
            },
        };
        if !config.hidden.is_multiple_of(config.heads)
            || !config.mels.is_multiple_of(8)
            || (head == HeadKind::Tdt && config.durations.is_empty())
        {
            return Err("config.json has inconsistent sizes".into());
        }
        Ok(config)
    }
}

/// A projection of the encoder: ternary weights (Parakeet Redux) or dense
/// weights (Parakeet TDT v3), whichever the file holds.
enum Linear {
    Ternary(Ternary),
    Dense(Dense),
}

impl Linear {
    fn forward(&self, x: &[f32], rows: usize) -> Buf {
        match self {
            Self::Ternary(layer) => layer.forward(x, rows),
            Self::Dense(layer) => layer.forward(x, rows),
        }
    }
}

/// `name` as ternary (`.qweight` + `.scales`) or dense (`.weight`, any float
/// dtype, `[out, in]` or `[out, in, 1]` for a pointwise conv).
fn linear(t: &Tensors, name: &str, out: usize, inp: usize, group: usize) -> Result<Linear, String> {
    stacked(t, &[name.to_string()], (out, inp), group)
}

/// Several layers with the same input, stacked by output rows into one.
fn stacked(
    t: &Tensors,
    names: &[String],
    (out, inp): (usize, usize),
    group: usize,
) -> Result<Linear, String> {
    if t.has(&format!("{}.qweight", names[0])) {
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        for name in names {
            packed.extend_from_slice(&t.u8(&format!("{name}.qweight"))?);
            scales.extend(t.f32_len(&format!("{name}.scales"), out * (inp / group))?);
        }
        return Ternary::from_base3(&packed, &scales, names.len() * out, inp, group)
            .map(Linear::Ternary);
    }
    let parts: Vec<(&str, usize)> = names.iter().map(|name| (name.as_str(), out)).collect();
    Dense::load(t, &parts, inp).map(Linear::Dense)
}

/// Decoder layer with f16 weights as stored and an f32 bias.
fn half_dense(t: &Tensors, name: &str, out: usize, inp: usize) -> Result<HalfDense, String> {
    let weight = t.f16_bits(&format!("{name}.weight"), out * inp)?;
    let bias = t.f32_len(&format!("{name}.bias"), out)?;
    HalfDense::new(weight, Some(bias), out, inp)
}

fn dense(t: &Tensors, name: &str, out: usize, inp: usize, bias: bool) -> Result<Dense, String> {
    let weight = t.f32_len(&format!("{name}.weight"), out * inp)?;
    let bias = if bias {
        Some(t.f32_len(&format!("{name}.bias"), out)?)
    } else {
        None
    };
    Dense::new(weight, bias, out, inp)
}

/// Stride-2 convolution stack that turns 10 ms mel frames into 80 ms frames.
struct Subsampling {
    channels: usize,
    /// First conv as `[tap][channel]` weights and bias.
    conv0: (Vec<f32>, Vec<f32>),
    dw1: (Vec<f32>, Vec<f32>),
    pw1: Dense,
    dw2: (Vec<f32>, Vec<f32>),
    pw2: Dense,
    linear: Dense,
}

impl Subsampling {
    fn load(t: &Tensors, c: &Config) -> Result<Self, String> {
        let ch = c.sub_channels;
        let conv = |index: usize| -> Result<(Vec<f32>, Vec<f32>), String> {
            Ok((
                t.f32_len(
                    &format!("{}encoder.subsampling.layers.{index}.weight", c.prefix),
                    ch * 9,
                )?,
                t.f32_len(
                    &format!("{}encoder.subsampling.layers.{index}.bias", c.prefix),
                    ch,
                )?,
            ))
        };
        Ok(Self {
            channels: ch,
            conv0: {
                let (weight, bias) = conv(0)?;
                (
                    (0..9 * ch).map(|i| weight[(i % ch) * 9 + i / ch]).collect(),
                    bias,
                )
            },
            dw1: tap_major(conv(2)?, ch),
            pw1: dense(
                t,
                &format!("{}encoder.subsampling.layers.3", c.prefix),
                ch,
                ch,
                true,
            )?,
            dw2: tap_major(conv(5)?, ch),
            pw2: dense(
                t,
                &format!("{}encoder.subsampling.layers.6", c.prefix),
                ch,
                ch,
                true,
            )?,
            linear: dense(
                t,
                &format!("{}encoder.subsampling.linear", c.prefix),
                c.hidden,
                ch * (c.mels / 8),
                true,
            )?,
        })
    }

    /// `features` is `frames x mels`; returns `(out_frames x hidden, out_frames)`.
    fn forward(&self, features: &[f32], frames: usize, mels: usize) -> (Buf, usize) {
        let ch = self.channels;
        let (x, h2, w2) = self.first_two_convs(features, frames, mels);
        let x = self.pw1.forward_relu(&x, h2 * w2);
        let (x, h3, w3) = depthwise(&x, h2, w2, ch, &self.dw2);
        let x = self.pw2.forward_relu(&x, h3 * w3);
        // Flatten per frame as channel-major: index = c * w3 + w.
        let mut flat = Buf::scratch(h3 * ch * w3);
        team::chunks_mut(&mut flat, ch * w3, |h, row| {
            for w in 0..w3 {
                for c in 0..ch {
                    row[c * w3 + w] = x[(h * w3 + w) * ch + c];
                }
            }
        });
        (self.linear.forward(&flat, h3), h3)
    }
}

impl Subsampling {
    /// conv0 (one input channel) and the first depthwise conv, in strips of
    /// output rows. The full conv0 output is 73 MB for a 22 s take; a strip
    /// only computes the conv0 rows it needs, so that buffer never exists.
    fn first_two_convs(&self, features: &[f32], frames: usize, mels: usize) -> (Buf, usize, usize) {
        let ch = self.channels;
        let (h1, w1) = (half(frames), half(mels));
        let (h2, w2) = (half(h1), half(w1));
        let mut out = Buf::scratch(h2 * w2 * ch);
        // At least four strips per thread, so a short take still spreads out.
        let strip_rows = h2.div_ceil(4 * team::threads()).clamp(1, MAX_STRIP_ROWS);
        team::chunks_mut(&mut out, strip_rows * w2 * ch, |strip, rows| {
            conv_strip(self, features, (frames, mels), strip_rows, strip, rows);
        });
        (out, h2, w2)
    }

    /// One output row of conv0, channels-last, after ReLU.
    #[inline(always)]
    fn conv0_row(&self, features: &[f32], frames: usize, mels: usize, h: usize, row: &mut [f32]) {
        let ch = self.channels;
        let (weight, bias) = &self.conv0;
        for (w, cell) in row.chunks_exact_mut(ch).enumerate() {
            cell.copy_from_slice(bias);
            for tap in 0..9 {
                let Some(v) = sample(features, frames, mels, 2 * h + tap / 3, 2 * w + tap % 3)
                else {
                    continue;
                };
                for (o, wt) in cell.iter_mut().zip(&weight[tap * ch..(tap + 1) * ch]) {
                    *o += wt * v;
                }
            }
            cell.iter_mut().for_each(|o| *o = o.max(0.0));
        }
    }
}

multiversion! {
    /// Output rows `strip * strip_rows ..` of the first depthwise conv, from
    /// the conv0 rows they read. `rows` holds those output rows.
    fn conv_strip(
        sub: &Subsampling,
        features: &[f32],
        frames_mels: (usize, usize),
        strip_rows: usize,
        strip: usize,
        rows: &mut [f32],
    ) {
        let (frames, mels) = frames_mels;
        let ch = sub.channels;
        let (h1, w1) = (half(frames), half(mels));
        let w2 = half(w1);
        let first_out = strip * strip_rows;
        let out_rows = rows.len() / (w2 * ch);
        // Output row oh reads conv0 rows 2oh-1 ..= 2oh+1.
        let first_in = (2 * first_out).saturating_sub(1);
        let last_in = (2 * (first_out + out_rows - 1) + 1).min(h1 - 1);
        let mut conv0 = vec![0.0f32; (last_in - first_in + 1) * w1 * ch];
        for (offset, row) in conv0.chunks_exact_mut(w1 * ch).enumerate() {
            sub.conv0_row(features, frames, mels, first_in + offset, row);
        }
        let (weight, bias) = &sub.dw1;
        for local in 0..out_rows {
            let oh = first_out + local;
            for ow in 0..w2 {
                let cell = &mut rows[(local * w2 + ow) * ch..][..ch];
                cell.copy_from_slice(bias);
                for i in 0..3 {
                    for j in 0..3 {
                        let (ih, iw) = (2 * oh + i, 2 * ow + j);
                        if ih == 0 || iw == 0 || ih > h1 || iw > w1 {
                            continue;
                        }
                        let input = &conv0[((ih - 1 - first_in) * w1 + (iw - 1)) * ch..][..ch];
                        let taps = &weight[(i * 3 + j) * ch..][..ch];
                        for ((o, wt), v) in cell.iter_mut().zip(taps).zip(input) {
                            *o += wt * v;
                        }
                    }
                }
            }
        }
    }
}

/// Most output rows of the first depthwise conv in one parallel task.
const MAX_STRIP_ROWS: usize = 8;

/// `[channel][tap]` depthwise weights as `[tap][channel]`, so a tap is one contiguous row.
fn tap_major((weight, bias): (Vec<f32>, Vec<f32>), ch: usize) -> (Vec<f32>, Vec<f32>) {
    (
        (0..9 * ch).map(|i| weight[(i % ch) * 9 + i / ch]).collect(),
        bias,
    )
}

/// Output length of a kernel-3, stride-2, padding-1 convolution.
fn half(n: usize) -> usize {
    (n - 1) / 2 + 1
}

/// Input value at padded position (row, col), or None in the zero padding.
#[inline(always)]
fn sample(x: &[f32], rows: usize, cols: usize, row: usize, col: usize) -> Option<f32> {
    let (r, c) = (row.checked_sub(1)?, col.checked_sub(1)?);
    (r < rows && c < cols).then(|| x[r * cols + c])
}

/// Depthwise 3x3 stride-2 convolution on channels-last data.
fn depthwise(
    x: &[f32],
    h: usize,
    w: usize,
    ch: usize,
    (weight, bias): &(Vec<f32>, Vec<f32>),
) -> (Buf, usize, usize) {
    let (ho, wo) = (half(h), half(w));
    let mut out = Buf::scratch(ho * wo * ch);
    team::chunks_mut(&mut out, wo * ch, |oh, row| {
        depthwise_row(x, (h, w, ch), weight, bias, oh, row);
    });
    (out, ho, wo)
}

multiversion! {
    /// Output row `oh` of `depthwise`.
    fn depthwise_row(
        x: &[f32],
        shape: (usize, usize, usize),
        weight: &[f32],
        bias: &[f32],
        oh: usize,
        row: &mut [f32],
    ) {
        let (h, w, ch) = shape;
        for (ow, cell) in row.chunks_exact_mut(ch).enumerate() {
            cell.copy_from_slice(bias);
            for i in 0..3 {
                for j in 0..3 {
                    let (ih, iw) = (2 * oh + i, 2 * ow + j);
                    if ih == 0 || iw == 0 || ih > h || iw > w {
                        continue;
                    }
                    let input = &x[((ih - 1) * w + (iw - 1)) * ch..][..ch];
                    let taps = &weight[(i * 3 + j) * ch..][..ch];
                    for ((o, wt), v) in cell.iter_mut().zip(taps).zip(input) {
                        *o += wt * v;
                    }
                }
            }
        }
    }
}

struct Block {
    norm_ff1: Norm,
    ff1: (Linear, Linear),
    norm_att: Norm,
    /// q, k, and v stacked into one `3d x d` layer: one quantization, one product.
    qkv: Linear,
    o: Linear,
    rel_k: Linear,
    /// `rel_k` applied to the position table, kept between takes.
    rel_cache: RelCache,
    bias_u: Vec<f32>,
    bias_v: Vec<f32>,
    norm_conv: Norm,
    pw1: Linear,
    /// Depthwise conv weights as `[tap][channel]`, so a tap is one contiguous row.
    dw_by_tap: Vec<f32>,
    bn_scale: Vec<f32>,
    bn_shift: Vec<f32>,
    pw2: Linear,
    norm_ff2: Norm,
    ff2: (Linear, Linear),
    norm_out: Norm,
}

impl Block {
    fn load(t: &Tensors, c: &Config, index: usize) -> Result<Self, String> {
        let p = format!("{}encoder.layers.{index}", c.prefix);
        let (d, g) = (c.hidden, c.group);
        let layer =
            |name: &str, out: usize, inp: usize| linear(t, &format!("{p}.{name}"), out, inp, g);
        let mean = t.f32_len(&format!("{p}.conv.norm.running_mean"), d)?;
        let var = t.f32_len(&format!("{p}.conv.norm.running_var"), d)?;
        let weight = t.f32_len(&format!("{p}.conv.norm.weight"), d)?;
        let bias = t.f32_len(&format!("{p}.conv.norm.bias"), d)?;
        let bn_scale: Vec<f32> = (0..d)
            .map(|i| weight[i] / (var[i] + BATCH_NORM_EPS).sqrt())
            .collect();
        // A depthwise conv bias `b` adds `b * scale` after the batch norm.
        let dw_bias = match t.has(&format!("{p}.conv.depthwise_conv.bias")) {
            true => t.f32_len(&format!("{p}.conv.depthwise_conv.bias"), d)?,
            false => vec![0.0; d],
        };
        let bn_shift = (0..d)
            .map(|i| bias[i] + (dw_bias[i] - mean[i]) * bn_scale[i])
            .collect();
        Ok(Self {
            norm_ff1: Norm::load(t, &format!("{p}.norm_feed_forward1"), d)?,
            ff1: (
                layer("feed_forward1.linear1", c.ff, d)?,
                layer("feed_forward1.linear2", d, c.ff)?,
            ),
            norm_att: Norm::load(t, &format!("{p}.norm_self_att"), d)?,
            qkv: stacked(
                t,
                &["q_proj", "k_proj", "v_proj"].map(|n| format!("{p}.self_attn.{n}")),
                (d, d),
                g,
            )?,
            o: layer("self_attn.o_proj", d, d)?,
            rel_k: layer("self_attn.relative_k_proj", d, d)?,
            rel_cache: RelCache::default(),
            bias_u: t.f32_len(&format!("{p}.self_attn.bias_u"), d)?,
            bias_v: t.f32_len(&format!("{p}.self_attn.bias_v"), d)?,
            norm_conv: Norm::load(t, &format!("{p}.norm_conv"), d)?,
            pw1: layer("conv.pointwise_conv1", 2 * d, d)?,
            dw_by_tap: {
                let kernel = c.conv_kernel;
                let dw = t.f32_len(&format!("{p}.conv.depthwise_conv.weight"), d * kernel)?;
                (0..kernel * d)
                    .map(|i| dw[(i % d) * kernel + i / d])
                    .collect()
            },
            bn_scale,
            bn_shift,
            pw2: layer("conv.pointwise_conv2", d, d)?,
            norm_ff2: Norm::load(t, &format!("{p}.norm_feed_forward2"), d)?,
            ff2: (
                layer("feed_forward2.linear1", c.ff, d)?,
                layer("feed_forward2.linear2", d, c.ff)?,
            ),
            norm_out: Norm::load(t, &format!("{p}.norm_out"), d)?,
        })
    }

    fn feed_forward((up, down): &(Linear, Linear), x: &[f32], frames: usize) -> Buf {
        let mut hidden = up.forward(x, frames);
        team::chunks_mut(&mut hidden, 4096, |_, values| activation::silu(values));
        down.forward(&hidden, frames)
    }

    fn forward(&self, x: &mut Buf, frames: usize, c: &Config) {
        let ff = Self::feed_forward(&self.ff1, &self.norm_ff1.apply(x, c.hidden), frames);
        let normed = self.norm_att.residual(x, &ff, 0.5);
        let att = self.attention(&normed, frames, c);
        let normed = self.norm_conv.residual(x, &att, 1.0);
        let conv = self.convolution(&normed, frames, c);
        let normed = self.norm_ff2.residual(x, &conv, 1.0);
        let ff = Self::feed_forward(&self.ff2, &normed, frames);
        *x = self.norm_out.residual(x, &ff, 0.5);
    }

    /// Multi-head attention with Transformer-XL relative positions.
    fn attention(&self, x: &[f32], t: usize, c: &Config) -> Buf {
        let (d, heads) = (c.hidden, c.heads);
        let hd = d / heads;
        let qkv = self.qkv.forward(x, t);
        let cache = self
            .rel_cache
            .rows(t, d, |table, rows| self.rel_k.forward(table, rows));
        let r = cache.slice(t, d);
        let scale = 1.0 / (hd as f32).sqrt();
        let p = 2 * t - 1;
        let mut context = Buf::scratch(t * d);
        let per_head: Vec<Buf> = team::map(heads, |h| {
            let cols = h * hd..(h + 1) * hd;
            // Columns of head h from a matrix with `stride` values per row,
            // into `out`, which keeps its memory for the next call.
            let head = |out: &mut Vec<f32>,
                        src: &[f32],
                        stride: usize,
                        offset: usize,
                        rows: usize,
                        bias: Option<&[f32]>| {
                out.clear();
                for i in 0..rows {
                    let row = &src[i * stride + offset..][cols.clone()];
                    match bias {
                        Some(b) => out.extend(row.iter().zip(b).map(|(v, b)| v + b)),
                        None => out.extend_from_slice(row),
                    }
                }
            };
            let (ac, bd) = HEADS.with(|heads| {
                let [qu, qv, k, pos] = &mut *heads.borrow_mut();
                head(qu, &qkv, 3 * d, 0, t, Some(&self.bias_u[cols.clone()]));
                head(qv, &qkv, 3 * d, 0, t, Some(&self.bias_v[cols.clone()]));
                head(k, &qkv, 3 * d, d, t, None);
                head(pos, r, d, 0, p, None);
                (
                    fgemm::matmul(qu, t, hd, &k[..], t),
                    fgemm::matmul(qv, t, hd, &pos[..], p),
                )
            });
            let mut probs = Buf::scratch(t * t);
            for (i, row) in probs.chunks_exact_mut(t).enumerate() {
                // Column t-1-i+j of the position product is relative position i - j.
                let shifted = &bd[i * p + (t - 1 - i)..][..t];
                let mut max = f32::NEG_INFINITY;
                for j in 0..t {
                    row[j] = (ac[i * t + j] + shifted[j]) * scale;
                    max = max.max(row[j]);
                }
                row.iter_mut().for_each(|value| *value -= max);
                activation::exp(row);
                let inv = 1.0 / row.iter().sum::<f32>();
                row.iter_mut().for_each(|value| *value *= inv);
            }
            let mut v_t = Buf::scratch(hd * t);
            for j in 0..t {
                for (e, value) in qkv[j * 3 * d + 2 * d..][cols.clone()].iter().enumerate() {
                    v_t[e * t + j] = *value;
                }
            }
            fgemm::matmul(&probs, t, t, &v_t, hd)
        });
        for (h, out) in per_head.iter().enumerate() {
            for i in 0..t {
                context[i * d + h * hd..i * d + (h + 1) * hd]
                    .copy_from_slice(&out[i * hd..(i + 1) * hd]);
            }
        }
        self.o.forward(&context, t)
    }

    /// Pointwise conv with GLU, depthwise conv over time, batch norm, SiLU, pointwise conv.
    fn convolution(&self, x: &[f32], t: usize, c: &Config) -> Buf {
        let d = c.hidden;
        let kernel = c.conv_kernel;
        let wide = self.pw1.forward(x, t);
        let mut gated = Buf::scratch(t * d);
        team::chunks_mut(&mut gated, d, |index, out| {
            let row = &wide[index * 2 * d..(index + 1) * 2 * d];
            out.copy_from_slice(&row[d..]);
            activation::sigmoid(out);
            for (o, value) in out.iter_mut().zip(&row[..d]) {
                *o *= value;
            }
        });
        // Accumulated with +=: the one buffer here that must start at zero.
        let mut mixed = Buf::zeroed(t * d);
        team::chunks_mut(&mut mixed, d, |i, out| {
            time_conv_row(self, &gated, (t, kernel), i, out);
            activation::silu(out);
        });
        self.pw2.forward(&mixed, t)
    }
}

multiversion! {
    /// Frame `i` of the conv module's depthwise conv over time, then its batch
    /// norm. `out` starts at zero.
    fn time_conv_row(block: &Block, gated: &[f32], t_kernel: (usize, usize), i: usize, out: &mut [f32]) {
        let (t, kernel) = t_kernel;
        let d = out.len();
        let pad = (kernel - 1) / 2;
        for k in 0..kernel {
            let src = i + k;
            if src < pad || src - pad >= t {
                continue;
            }
            let taps = &block.dw_by_tap[k * d..(k + 1) * d];
            for ((o, w), g) in out.iter_mut().zip(taps).zip(&gated[(src - pad) * d..][..d]) {
                *o += w * g;
            }
        }
        for ((o, scale), shift) in out.iter_mut().zip(&block.bn_scale).zip(&block.bn_shift) {
            *o = *o * scale + shift;
        }
    }
}

/// Positions every take reaches without growing the cache: 30 s of 80 ms frames.
pub const POSITION_REACH: usize = 375;

/// Projected relative positions for `reach` down to `-reach`, one row each.
/// A row depends only on its position, never on the take's length or on
/// other rows, so one table serves every take up to `reach + 1` frames.
#[derive(Default)]
struct RelCache {
    table: RwLock<Option<Arc<RelRows>>>,
}

struct RelRows {
    reach: usize,
    rows: Buf,
}

impl RelRows {
    /// Rows for positions t-1 down to -(t-1).
    fn slice(&self, t: usize, d: usize) -> &[f32] {
        let first = self.reach - (t - 1);
        &self.rows[first * d..(first + 2 * t - 1) * d]
    }
}

impl RelCache {
    /// The table, grown first when `t` frames reach past it.
    fn rows(&self, t: usize, d: usize, project: impl Fn(&[f32], usize) -> Buf) -> Arc<RelRows> {
        if let Some(rows) = self.table.read().as_ref().filter(|r| r.reach + 1 >= t) {
            return Arc::clone(rows);
        }
        let mut table = self.table.write();
        if let Some(rows) = table.as_ref().filter(|r| r.reach + 1 >= t) {
            return Arc::clone(rows);
        }
        let reach = (t - 1)
            .max(POSITION_REACH)
            .max(table.as_ref().map_or(0, |r| 2 * r.reach));
        let positions = relative_positions(reach + 1, d);
        let rows = Arc::new(RelRows {
            reach,
            rows: project(&positions, 2 * reach + 1),
        });
        *table = Some(Arc::clone(&rows));
        rows
    }
}

/// Sinusoidal table for relative positions t-1 down to -(t-1), sin and cos interleaved.
fn relative_positions(t: usize, d: usize) -> Vec<f32> {
    let rows = 2 * t - 1;
    let inv_freq: Vec<f32> = (0..d / 2)
        .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / d as f32))
        .collect();
    let mut table = vec![0.0f32; rows * d];
    team::chunks_mut(&mut table, d, |p, row| {
        let position = (t - 1) as f32 - p as f32;
        for (i, freq) in inv_freq.iter().enumerate() {
            let angle = position * freq;
            row[2 * i] = angle.sin();
            row[2 * i + 1] = angle.cos();
        }
    });
    table
}

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

pub struct Parakeet {
    pub config: Config,
    subsampling: Subsampling,
    blocks: Vec<Block>,
    head: Head,
    /// The speech head some checkpoints carry (Redux: `vad_head.*`).
    vad: Option<VadHead>,
}

/// Speech probability per 80 ms frame from the subsampler output, as
/// Kestrel runs it: a 1x1 conv to 128, SiLU, a width-5 conv, SiLU, a 1x1
/// conv to one logit, sigmoid. The subsampler is local, so this costs a
/// fraction of one encoder layer.
struct VadHead {
    proj: Dense,
    ctx: Dense,
    out: Dense,
}

/// Seconds per subsampled frame of the speech head.
const FRAME_SECONDS: f32 = 0.08;
/// Shorter gaps between speech join; shorter speech does not count.
const MIN_GAP_SECONDS: f32 = 0.1;
const MIN_SPEECH_SECONDS: f32 = 0.1;

impl VadHead {
    const KERNEL: usize = 5;

    fn load(t: &Tensors, c: &Config) -> Result<Option<Self>, String> {
        if !t.has("vad_head.proj.weight") {
            return Ok(None);
        }
        let hidden = 128;
        Ok(Some(Self {
            proj: Dense::load(t, &[("vad_head.proj", hidden)], c.hidden)?,
            ctx: Dense::load(t, &[("vad_head.ctx", hidden)], hidden * Self::KERNEL)?,
            out: Dense::load(t, &[("vad_head.out", 1)], hidden)?,
        }))
    }

    /// Whether `t` subsampled frames hold speech: a run of frames with a
    /// probability of 0.5 or more lasting `MIN_SPEECH_SECONDS`, after gaps
    /// under `MIN_GAP_SECONDS` are bridged (Kestrel's `speech_regions`).
    fn speech(&self, x: &[f32], t: usize) -> bool {
        let mut h = self.proj.forward(x, t);
        activation::silu(&mut h);
        let hidden = self.proj.out_features;
        let cols = crate::models::transformer::conv_columns(
            &h,
            (t, hidden),
            (Self::KERNEL, 1, Self::KERNEL / 2),
            t,
        );
        let mut h = self.ctx.forward(&cols, t);
        activation::silu(&mut h);
        let logits = self.out.forward(&h, t);
        // sigmoid(v) >= 0.5 exactly when v >= 0.
        let speaking: Vec<bool> = logits.iter().map(|v| *v >= 0.0).collect();
        let mut regions: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < t {
            if !speaking[i] {
                i += 1;
                continue;
            }
            let start = i;
            while i < t && speaking[i] {
                i += 1;
            }
            match regions.last_mut() {
                Some(last) if (start - last.1) as f32 * FRAME_SECONDS < MIN_GAP_SECONDS => {
                    last.1 = i
                }
                _ => regions.push((start, i)),
            }
        }
        regions
            .iter()
            .any(|(start, end)| (end - start) as f32 * FRAME_SECONDS >= MIN_SPEECH_SECONDS)
    }
}

/// What turns encoder frames into tokens.
enum Head {
    /// Token and duration transducer: LSTM prediction network and joint network.
    Tdt(Tdt),
    /// One projection to token logits per frame, greedy with repeats merged.
    Ctc(Dense),
    /// No head: the encoder output is the result.
    Encoder,
}

struct Tdt {
    encoder_projector: Dense,
    embedding: Vec<f32>,
    lstm: Vec<Lstm>,
    decoder_projector: HalfDense,
    joint: HalfDense,
}

impl Parakeet {
    pub fn load(t: &Tensors, config: Config) -> Result<Self, String> {
        let c = &config;
        let blocks = team::map(c.layers, |index| Block::load(t, c, index))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let head = match c.head {
            HeadKind::Ctc => Head::Ctc(dense(t, "ctc_head", c.vocab, c.hidden, true)?),
            HeadKind::Tdt => Head::Tdt(Tdt::load(t, c)?),
            HeadKind::Encoder => Head::Encoder,
        };
        Ok(Self {
            subsampling: Subsampling::load(t, c)?,
            blocks,
            head,
            vad: VadHead::load(t, c)?,
            config,
        })
    }
}

impl Tdt {
    fn load(t: &Tensors, c: &Config) -> Result<Self, String> {
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

impl Parakeet {
    /// Encoder output after the 24 blocks: `frames x hidden`.
    pub fn encode(&self, features: &[f32], frames: usize) -> (Buf, usize) {
        let (x, t) = self.subsampling.forward(features, frames, self.config.mels);
        self.encode_subsampled(x, t)
    }

    /// `encode`, or `None` when the model's speech head finds no speech
    /// (a model without one always encodes).
    pub fn encode_speech(&self, features: &[f32], frames: usize) -> Option<(Buf, usize)> {
        let (x, t) = self.subsampling.forward(features, frames, self.config.mels);
        if self.vad.as_ref().is_some_and(|vad| !vad.speech(&x, t)) {
            return None;
        }
        Some(self.encode_subsampled(x, t))
    }

    fn encode_subsampled(&self, mut x: Buf, t: usize) -> (Buf, usize) {
        let c = &self.config;
        if c.input_scale != 1.0 {
            let scale = c.input_scale;
            team::chunks_mut(&mut x, c.hidden, |_, row| {
                row.iter_mut().for_each(|v| *v *= scale)
            });
        }
        for block in &self.blocks {
            block.forward(&mut x, t, c);
        }
        (x, t)
    }

    /// Fills every layer's position table up front, so the first take does not wait.
    pub fn warm_positions(&self) {
        let d = self.config.hidden;
        for block in &self.blocks {
            block.rel_cache.rows(POSITION_REACH + 1, d, |table, rows| {
                block.rel_k.forward(table, rows)
            });
        }
    }

    /// Tokens of the encoder output `t x hidden`, each with the encoder
    /// frame it was emitted at. `dictionary` terms replace words they sound
    /// like (TDT only); `banned` tokens are never picked.
    pub fn decode(
        &self,
        encoded: &[f32],
        t: usize,
        dictionary: Option<&Dictionary>,
        english: Option<&English>,
        banned: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        let c = &self.config;
        match &self.head {
            Head::Tdt(tdt) => {
                let projected = tdt.encoder_projector.forward(encoded, t);
                tdt.decode(c, &projected, t, dictionary, english, banned)
            }
            Head::Ctc(head) => ctc_greedy(&head.forward(encoded, t), c.vocab, c.blank, banned),
            Head::Encoder => (Vec::new(), Vec::new()),
        }
    }
}

/// Greedy CTC: the best token per frame, repeats merged, blanks dropped.
fn ctc_greedy(
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
        let token = argmax(&row);
        if token != blank && token != previous {
            tokens.push(token);
            frames.push(index);
        }
        previous = token;
    }
    (tokens, frames)
}

impl Tdt {
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
    /// A transcript mostly in no English word is decoded again with the
    /// `english` prior; `dictionary` terms then replace the words heard
    /// where the sound allows (see `english.rs` and `dictionary.rs`).
    fn decode(
        &self,
        c: &Config,
        projected: &[f32],
        t: usize,
        dictionary: Option<&Dictionary>,
        english: Option<&English>,
        banned: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        let keep = dictionary.is_some();
        let mut run = self.greedy(c, projected, t, None, banned, keep);
        if let Some(english) = english.filter(|e| e.drifted(&run.tokens)) {
            run = self.greedy(c, projected, t, Some(english), banned, keep);
        }
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

    /// One greedy pass; `english` steers unclear words to English, and
    /// `keep` records the decoder before each token for the dictionary.
    fn greedy(
        &self,
        c: &Config,
        projected: &[f32],
        t: usize,
        english: Option<&English>,
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
        let mut word = Partial::default();
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
            let mut token = argmax(&logits[..c.vocab]);
            if let Some(english) = english {
                token = english.pick(&word, &logits[..c.vocab], token, c.blank);
                if token != c.blank {
                    word = english.step(&word, token).0;
                }
            }
            let mut duration = c.durations[argmax(&logits[c.vocab..])];
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
                Some(&token) => End::Token(token, frames[span.end]),
                None => End::Close,
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
                    dictionary.prior,
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
                    End::Close => (heard.t, None),
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
                let needed = rows(u, count, width, anchor);
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
                for f in 0..width {
                    let here = alpha[at(f, u)];
                    if here == f32::NEG_INFINITY {
                        continue;
                    }
                    let (token_part, duration_part) = row(node, f);
                    for (&d, &p) in c.durations.iter().zip(duration_part.iter()) {
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
                        if let (true, Some(id)) = (u < count, next) {
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

/// Joint rows of step `u` of a job with `count` tokens: the frames it
/// crosses, and on the last step also the anchor's own frame.
fn rows(u: usize, count: usize, width: usize, anchor: Option<usize>) -> usize {
    if u == count && anchor.is_some() {
        width + 1
    } else {
        width
    }
}

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
    Close,
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

/// Index of the largest value; the first one wins a tie, as torch.argmax does.
fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}
