//! The FastConformer encoder of Parakeet and Canary, and the head that
//! follows it (`transducer.rs`, CTC, or none). The structure follows the
//! Apache-2.0 Hugging Face `ParakeetEncoder`.

use super::dictionary::Dictionary;
use super::transducer::Transducer;
use crate::cpu::activation;
use crate::cpu::buf::Buf;
use crate::cpu::fgemm;
use crate::cpu::gemm::Dense;
use crate::cpu::half::HalfDense;
use crate::cpu::kernels::Ternary;
use crate::cpu::team;
use crate::cpu::tensors::Tensors;
use crate::models::config::Json;
use crate::models::layers::Norm;
use parking_lot::RwLock;
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
    /// TDT decoder and joint network (`parakeet_tdt`): the joint network
    /// gives each token a duration too.
    Transducer,
    /// RNNT decoder and joint network (`parakeet_rnnt`): a TDT without
    /// durations; a blank moves one frame, a token none.
    Rnnt,
    /// One projection per frame (`parakeet_ctc`).
    Ctc,
    /// None: another model (Canary) decodes the encoder output.
    Encoder,
}

impl Config {
    pub fn parse(json: &Json) -> Result<Self, String> {
        let head = match json.str("model_type") {
            Some("parakeet_ctc") => HeadKind::Ctc,
            Some("parakeet_rnnt") => HeadKind::Rnnt,
            Some("canary") => HeadKind::Encoder,
            _ => HeadKind::Transducer,
        };
        let enc = |key: &str| json.num(&format!("encoder_config.{key}"));
        // Only the shape this engine implements is accepted.
        if enc("subsampling_factor")? != 8
            || enc("subsampling_conv_kernel_size")? != 3
            || enc("subsampling_conv_stride")? != 2
        {
            return Err("config.json describes a model shape this engine does not run".into());
        }
        let hidden = enc("hidden_size")?;
        let transducer = matches!(head, HeadKind::Transducer | HeadKind::Rnnt);
        // Only the transducers have the decoder fields; a CTC blank is the pad token.
        let decoder = |key: &str| if transducer { json.num(key) } else { Ok(0) };
        let config = Self {
            hidden,
            ff: enc("intermediate_size")?,
            heads: enc("num_attention_heads")?,
            layers: enc("num_hidden_layers")?,
            mels: enc("num_mel_bins")?,
            conv_kernel: enc("conv_kernel_size")?,
            sub_channels: enc("subsampling_conv_channels")?,
            decoder_hidden: decoder("decoder_hidden_size")?,
            decoder_layers: decoder("num_decoder_layers")?,
            vocab: json.num("vocab_size")?,
            blank: json.num(if transducer {
                "blank_token_id"
            } else {
                "pad_token_id"
            })?,
            durations: match json.get("durations").as_array() {
                Some(list) => list
                    .iter()
                    .map(|d| d.as_u64().map(|d| d as usize).ok_or("bad duration"))
                    .collect::<Result<_, _>>()?,
                None if head != HeadKind::Transducer => Vec::new(),
                None => return Err("config.json has no durations".into()),
            },
            max_symbols: decoder("max_symbols_per_step")?,
            // Ternary models (Redux) only; dense models have no groups.
            group: json.num("ternary_group_size").unwrap_or(0),
            input_scale: if json.flag("encoder_config.scale_input") == Some(true) {
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
            || (head == HeadKind::Transducer && config.durations.is_empty())
            || (transducer && config.max_symbols == 0)
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
pub(crate) fn half_dense(
    t: &Tensors,
    name: &str,
    out: usize,
    inp: usize,
) -> Result<HalfDense, String> {
    let weight = t.f16_bits(&format!("{name}.weight"), out * inp)?;
    let bias = t.f32_len(&format!("{name}.bias"), out)?;
    HalfDense::new(weight, Some(bias), out, inp)
}

pub(crate) fn dense(
    t: &Tensors,
    name: &str,
    out: usize,
    inp: usize,
    bias: bool,
) -> Result<Dense, String> {
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
        let cols = crate::models::layers::conv_columns(
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
    /// Transducer (TDT, or RNNT without durations): LSTM prediction
    /// network and joint network.
    Transducer(Transducer),
    /// One projection to token logits per frame, greedy with repeats merged.
    Ctc(Dense),
    /// No head: the encoder output is the result.
    Encoder,
}

impl Parakeet {
    pub fn load(t: &Tensors, config: Config) -> Result<Self, String> {
        let c = &config;
        let blocks = team::map(c.layers, |index| Block::load(t, c, index))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let head = match c.head {
            HeadKind::Ctc => Head::Ctc(dense(t, "ctc_head", c.vocab, c.hidden, true)?),
            HeadKind::Transducer | HeadKind::Rnnt => Head::Transducer(Transducer::load(t, c)?),
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
        banned: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        let c = &self.config;
        match &self.head {
            Head::Transducer(tdt) => {
                let projected = tdt.encoder_projector.forward(encoded, t);
                tdt.decode(c, &projected, t, dictionary, banned)
            }
            Head::Ctc(head) => crate::models::parts::ctc::greedy(
                &head.forward(encoded, t),
                c.vocab,
                c.blank,
                banned,
            ),
            Head::Encoder => (Vec::new(), Vec::new()),
        }
    }
}
