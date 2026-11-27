//! The audio encoder of Whisper, which Voxtral and MOSS run too: a conv stem
//! (stride 1, then 2) with GELU, learned positions, pre-norm layers, and a
//! final norm.

use crate::config::Json;
use crate::cpu::{buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::layers::{add_into, attention, conv_columns, linear, stacked, Gelu, Norm};

/// The exact GELU, as OpenAI Whisper and Transformers compute it. With 5
/// beams it gives 768 and 479 errors (tiny, base.en) on the 456 English
/// test clips, the tanh form of whisper.cpp 766 and 480; greedy, the tanh
/// form makes 2 fewer errors each. Beam search is the default, so exact.
pub const GELU: Gelu = Gelu::Exact;

struct EncoderLayer {
    norm_attn: Norm,
    /// q, k, v stacked: one product.
    qkv: Dense,
    out: Dense,
    norm_ffn: Norm,
    fc1: Dense,
    fc2: Dense,
}

pub struct Encoder {
    d: usize,
    heads: usize,
    ffn: usize,
    mels: usize,
    positions: usize,
    conv1: Dense,
    conv2: Dense,
    embed_positions: Vec<f32>,
    layers: Vec<EncoderLayer>,
    norm: Norm,
}

/// mel bins, and positions.
pub struct EncoderShape {
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    pub mels: usize,
    pub positions: usize,
}

impl EncoderShape {
    /// The shape under `encoder_config` of a model that reuses this encoder.
    pub fn parse(json: &Json) -> Result<Self, String> {
        let e = |key: &str| json.num(&format!("encoder_config.{key}"));
        Ok(Self {
            d: e("hidden_size")?,
            layers: e("num_hidden_layers")?,
            heads: e("num_attention_heads")?,
            ffn: e("intermediate_size")?,
            mels: e("num_mel_bins")?,
            positions: e("max_source_positions")?,
        })
    }
}

impl Encoder {
    /// The encoder whose tensors are under `prefix` (`model.encoder`).
    pub fn load(t: &Tensors, prefix: &str, shape: EncoderShape) -> Result<Self, String> {
        let EncoderShape {
            d,
            layers,
            heads,
            ffn,
            mels,
            positions,
        } = shape;
        let layers = team::map(layers, |i| -> Result<EncoderLayer, String> {
            let p = format!("{prefix}.layers.{i}");
            let attn = |n: &str| format!("{p}.self_attn.{n}");
            Ok(EncoderLayer {
                norm_attn: Norm::load(t, &format!("{p}.self_attn_layer_norm"), d)?,
                qkv: stacked(
                    t,
                    &[&attn("q_proj"), &attn("k_proj"), &attn("v_proj")],
                    d,
                    d,
                )?,
                out: linear(t, &attn("out_proj"), d, d)?,
                norm_ffn: Norm::load(t, &format!("{p}.final_layer_norm"), d)?,
                fc1: linear(t, &format!("{p}.fc1"), ffn, d)?,
                fc2: linear(t, &format!("{p}.fc2"), d, ffn)?,
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            conv1: linear(t, &format!("{prefix}.conv1"), d, 3 * mels)?,
            conv2: linear(t, &format!("{prefix}.conv2"), d, 3 * d)?,
            embed_positions: t
                .f32_len(&format!("{prefix}.embed_positions.weight"), positions * d)?,
            layers,
            norm: Norm::load(t, &format!("{prefix}.layer_norm"), d)?,
            d,
            heads,
            ffn,
            mels,
            positions,
        })
    }

    /// Encoder output `positions x d` for one window of `2 * positions x mels`
    /// features. Always the full 30 s: encoding only the positions the audio
    /// covers raised whisper-base.en from 3.42 % to 49 % WER on LibriSpeech.
    pub fn encode(&self, mel: &[f32]) -> Buf {
        let (d, frames, positions) = (self.d, 2 * self.positions, self.positions);
        let mut x = {
            let mut h = {
                let cols = conv_columns(mel, (frames, self.mels), (3, 1, 1), frames);
                self.conv1.forward(&cols, frames)
            };
            GELU.apply(&mut h, d);
            let cols = conv_columns(&h, (frames, d), (3, 2, 1), positions);
            drop(h);
            self.conv2.forward(&cols, positions)
        };
        GELU.apply(&mut x, d);
        if self.layers.is_empty() {
            add_into(&mut x, &self.embed_positions);
            return self.norm.apply(&x, d);
        }
        let mut normed = self.layers[0]
            .norm_attn
            .residual(&mut x, &self.embed_positions, 1.0);
        for (i, layer) in self.layers.iter().enumerate() {
            let att_out = {
                let att = {
                    let qkv = layer.qkv.forward(&normed, positions);
                    drop(normed);
                    attention(
                        (&qkv, 3 * d),
                        (&qkv[d..], 3 * d),
                        (&qkv[2 * d..], 3 * d),
                        (positions, positions),
                        (d, self.heads, self.heads),
                        false,
                    )
                };
                layer.out.forward(&att, positions)
            };
            let normed_ffn = layer.norm_ffn.residual(&mut x, &att_out, 1.0);
            drop(att_out);
            let ffn_out = {
                let mut hidden = layer.fc1.forward(&normed_ffn, positions);
                drop(normed_ffn);
                GELU.apply(&mut hidden, self.ffn);
                layer.fc2.forward(&hidden, positions)
            };
            normed = match self.layers.get(i + 1) {
                Some(next) => next.norm_attn.residual(&mut x, &ffn_out, 1.0),
                None => self.norm.residual(&mut x, &ffn_out, 1.0),
            };
        }
        normed
    }

    /// Deterministic synthetic Whisper-tiny stem probe for `examples/perf_budget.rs`:
    /// returns `(peak_live_buf_bytes, conv_columns_tasks, output_len, output_hash)` after one encode.
    /// Resets process-wide counters and requires an idle process.
    #[cfg(feature = "profile")]
    pub(crate) fn whisper_stem_budget_probe() -> Result<(u64, u64, usize, u64), String> {
        let (d, mels, positions) = (384, 80, 1500);
        let mut seed = 0x5EED_CAFE_1234_5678u64;
        let mut next = || -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((seed >> 40) as u32) as f32 / ((1u32 << 24) as f32)) * 0.2 - 0.1
        };
        let mut fill = |len: usize| -> Vec<f32> { (0..len).map(|_| next()).collect() };
        let encoder = Self {
            conv1: Dense::new(fill(d * 3 * mels), Some(fill(d)), d, 3 * mels)?,
            conv2: Dense::new(fill(d * 3 * d), Some(fill(d)), d, 3 * d)?,
            embed_positions: fill(positions * d),
            layers: Vec::new(),
            norm: Norm::new(vec![1.0; d], vec![0.0; d], 1e-5),
            d,
            heads: 6,
            ffn: 1536,
            mels,
            positions,
        };
        let mel = fill(2 * positions * mels);
        let _team = crate::cpu::enter(crate::Spread::AllCores);
        crate::prof::reset_peak_buf_bytes();
        crate::prof::reset_stage(crate::prof::Stage::ConvColumnsTasks);
        let out = encoder.encode(&mel);
        let len = out.len();
        if len != positions * d {
            return Err(format!(
                "unexpected stem output len {len}, expected {}",
                positions * d
            ));
        }
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut nonzero = false;
        for &v in out.iter() {
            if !v.is_finite() {
                return Err("non-finite value in stem output".to_string());
            }
            if v.abs() > 1e-4 {
                nonzero = true;
            }
            hash = (hash ^ u64::from(v.to_bits())).wrapping_mul(0x0000_0100_0000_01b3);
        }
        if !nonzero {
            return Err("stem output is all zeros".to_string());
        }
        let peak = crate::prof::peak_buf_bytes();
        let tasks = crate::prof::counted(crate::prof::Stage::ConvColumnsTasks);
        drop(out);
        Ok((peak, tasks, len, hash))
    }
}
