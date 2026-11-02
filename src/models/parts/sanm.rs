//! FunASR's SAN-M encoder (SenseVoice, Fun-ASR Nano): the input scaled by
//! `sqrt(d)` plus sinusoids, one block that projects the stacked features
//! to the model width, the main blocks, `after_norm`, the `tp` blocks, and
//! `tp_norm`. A block is pre-norm self-attention with an FSMN memory over
//! the values, then a pre-norm ReLU feed-forward. Tensor names are
//! FunASR's (`encoders0`, `encoders`, `tp_encoders`).

use crate::audio::kaldi_fbank::{FbankShape, KaldiFbank};
use crate::cpu::{buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::models::config::Json;
use crate::models::layers::{add_into, attention, linear, Norm};

/// FunASR's layer norm epsilon.
const NORM_EPS: f32 = 1e-12;

/// The encoder's sizes, from `encoder_config`.
pub struct Shape {
    pub d: usize,
    /// Width of the stacked features.
    pub input: usize,
    heads: usize,
    ffn: usize,
    kernel: usize,
    /// The projecting block and the main stack.
    blocks: usize,
    tp_blocks: usize,
}

impl Shape {
    pub fn parse(json: &Json) -> Result<Self, String> {
        let enc = |key: &str| json.num(&format!("encoder_config.{key}"));
        let shape = Self {
            d: enc("hidden_size")?,
            input: enc("input_size")?,
            heads: enc("num_attention_heads")?,
            ffn: enc("intermediate_size")?,
            kernel: enc("kernel_size")?,
            blocks: enc("num_blocks")?,
            tp_blocks: enc("tp_blocks")?,
        };
        if !shape.d.is_multiple_of(shape.heads)
            || shape.kernel.is_multiple_of(2)
            || shape.blocks == 0
        {
            return Err("config.json describes a SAN-M shape this engine does not run".into());
        }
        Ok(shape)
    }
}

/// The Kaldi filterbank front end of `feature_extractor`, with the file's
/// CMVN (`frontend.cmvn.shift`, `.scale`) unless `cmvn` is false.
pub fn frontend(json: &Json, t: &Tensors, shape: &Shape) -> Result<KaldiFbank, String> {
    let fe = |key: &str| json.num(&format!("feature_extractor.{key}"));
    if json.str("feature_extractor.type") != Some("kaldi_fbank") {
        return Err("config.json: a SAN-M model reads Kaldi filterbank features".into());
    }
    let fbank = FbankShape {
        sample_rate: fe("sample_rate")?,
        win: fe("win_length")?,
        hop: fe("hop_length")?,
        mels: fe("num_mel_bins")?,
        lfr_m: fe("lfr_m")?,
        lfr_n: fe("lfr_n")?,
        upscale: json.flag("feature_extractor.upscale_samples") != Some(false),
    };
    if fbank.lfr_m * fbank.mels != shape.input || fbank.sample_rate != crate::SAMPLE_RATE as usize {
        return Err("config.json: the stacked features do not fit the encoder".into());
    }
    // A model without CMVN (Fun-ASR Nano) feeds the stacked features as they are.
    let (shift, scale) = match json.flag("feature_extractor.cmvn") {
        Some(false) => (vec![0.0; shape.input], vec![1.0; shape.input]),
        _ => (
            t.f32_len("frontend.cmvn.shift", shape.input)?,
            t.f32_len("frontend.cmvn.scale", shape.input)?,
        ),
    };
    KaldiFbank::new(fbank, shift, scale)
}

struct Block {
    /// Width of the input: the stacked features for the first block.
    input: usize,
    norm_attn: Norm,
    qkv: Dense,
    out: Dense,
    /// The FSMN's depthwise conv over time, `[tap][channel]`.
    fsmn: Vec<f32>,
    norm_ffn: Norm,
    fc1: Dense,
    fc2: Dense,
}

impl Block {
    fn load(t: &Tensors, name: &str, input: usize, s: &Shape) -> Result<Self, String> {
        let (d, k) = (s.d, s.kernel);
        let fsmn = t.f32_len(&format!("{name}.self_attn.fsmn_block.weight"), d * k)?;
        Ok(Self {
            input,
            norm_attn: Norm::load(t, &format!("{name}.norm1"), input)?.with_eps(NORM_EPS),
            qkv: linear(t, &format!("{name}.self_attn.linear_q_k_v"), 3 * d, input)?,
            out: linear(t, &format!("{name}.self_attn.linear_out"), d, d)?,
            fsmn: (0..k * d).map(|i| fsmn[(i % d) * k + i / d]).collect(),
            norm_ffn: Norm::load(t, &format!("{name}.norm2"), d)?.with_eps(NORM_EPS),
            fc1: linear(t, &format!("{name}.feed_forward.w_1"), s.ffn, d)?,
            fc2: linear(t, &format!("{name}.feed_forward.w_2"), d, s.ffn)?,
        })
    }

    /// `x` (`t x input`) through the block. The first block's input is wider
    /// than its output, so its attention has no residual.
    fn forward(&self, x: &mut Buf, t: usize, s: &Shape) {
        let attended = self.attend(&self.norm_attn.apply(x, self.input), t, s);
        if self.input == s.d {
            add_into(x, &attended);
        } else {
            *x = attended;
        }
        let mut hidden = self.fc1.forward(&self.norm_ffn.apply(x, s.d), t);
        team::chunks_mut(&mut hidden, 4096, |_, values| {
            values.iter_mut().for_each(|v| *v = v.max(0.0))
        });
        add_into(x, &self.fc2.forward(&hidden, t));
    }

    /// Softmax attention, projected, plus the values and their FSMN memory
    /// (a centered depthwise conv over time).
    fn attend(&self, x: &[f32], t: usize, s: &Shape) -> Buf {
        let (d, k) = (s.d, s.kernel);
        let qkv = self.qkv.forward(x, t);
        let heard = attention(
            (&qkv, 3 * d),
            (&qkv[d..], 3 * d),
            (&qkv[2 * d..], 3 * d),
            (t, t),
            (d, s.heads, s.heads),
            false,
        );
        let mut out = self.out.forward(&heard, t);
        let pad = k / 2;
        let value = |j: usize| &qkv[j * 3 * d + 2 * d..][..d];
        team::chunks_mut(&mut out, d, |i, row| {
            add_into(row, value(i));
            for tap in 0..k {
                let Some(src) = (i + tap).checked_sub(pad).filter(|src| *src < t) else {
                    continue;
                };
                let weights = &self.fsmn[tap * d..][..d];
                for ((o, w), v) in row.iter_mut().zip(weights).zip(value(src)) {
                    *o += w * v;
                }
            }
        });
        out
    }
}

pub struct Encoder {
    pub shape: Shape,
    /// Every block in order: the projecting one, the main stack, and the
    /// `tp` stack after `after_norm`.
    blocks: Vec<Block>,
    after_norm: Norm,
    tp_norm: Norm,
}

impl Encoder {
    /// The encoder under `prefix` (`encoder`, `audio_encoder`).
    pub fn load(t: &Tensors, prefix: &str, shape: Shape) -> Result<Self, String> {
        let s = &shape;
        let names = std::iter::once(("encoders0", 0, s.input))
            .chain((0..s.blocks - 1).map(|i| ("encoders", i, s.d)))
            .chain((0..s.tp_blocks).map(|i| ("tp_encoders", i, s.d)))
            .collect::<Vec<_>>();
        let blocks = team::map(names.len(), |i| {
            let (stack, index, input) = names[i];
            Block::load(t, &format!("{prefix}.{stack}.{index}"), input, s)
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            blocks,
            after_norm: Norm::load(t, &format!("{prefix}.after_norm"), s.d)?.with_eps(NORM_EPS),
            tp_norm: Norm::load(t, &format!("{prefix}.tp_norm"), s.d)?.with_eps(NORM_EPS),
            shape,
        })
    }

    /// `t x d` from `t x input` rows (stacked features, and any prompt rows
    /// before them): scaled by `sqrt(d)` (the model width, not the input's),
    /// plus sinusoids, through every block.
    pub fn encode(&self, mut x: Buf, t: usize) -> Buf {
        let s = &self.shape;
        let scale = (s.d as f32).sqrt();
        let positions = sinusoids(t, s.input);
        team::chunks_mut(&mut x, s.input, |i, row| {
            for (v, p) in row.iter_mut().zip(&positions[i * s.input..][..s.input]) {
                *v = *v * scale + p;
            }
        });
        let (main, tp) = self.blocks.split_at(s.blocks);
        for block in main {
            block.forward(&mut x, t, s);
        }
        let mut x = self.after_norm.apply(&x, s.d);
        for block in tp {
            block.forward(&mut x, t, s);
        }
        self.tp_norm.apply(&x, s.d)
    }
}

/// FunASR's `SinusoidalPositionEncoder`: positions from 1, sines then
/// cosines over `depth / 2` timescales spaced `ln(10000) / (depth / 2 - 1)`.
fn sinusoids(t: usize, depth: usize) -> Vec<f32> {
    let half = depth / 2;
    let step = (10_000f64).ln() / (half - 1) as f64;
    let mut out = vec![0.0f32; t * depth];
    for (i, row) in out.chunks_exact_mut(depth).enumerate() {
        let position = (i + 1) as f64;
        for k in 0..half {
            let angle = position * (-(k as f64) * step).exp();
            row[k] = angle.sin() as f32;
            row[half + k] = angle.cos() as f32;
        }
    }
    out
}
