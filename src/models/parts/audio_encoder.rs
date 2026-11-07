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
        let cols = conv_columns(mel, (frames, self.mels), (3, 1, 1), frames);
        let mut h = self.conv1.forward(&cols, frames);
        GELU.apply(&mut h, d);
        let cols = conv_columns(&h, (frames, d), (3, 2, 1), positions);
        let mut x = self.conv2.forward(&cols, positions);
        GELU.apply(&mut x, d);
        add_into(&mut x, &self.embed_positions);
        for layer in &self.layers {
            let normed = layer.norm_attn.apply(&x, d);
            let qkv = layer.qkv.forward(&normed, positions);
            let att = attention(
                (&qkv, 3 * d),
                (&qkv[d..], 3 * d),
                (&qkv[2 * d..], 3 * d),
                (positions, positions),
                (d, self.heads, self.heads),
                false,
            );
            add_into(&mut x, &layer.out.forward(&att, positions));
            let normed = layer.norm_ffn.apply(&x, d);
            let mut hidden = layer.fc1.forward(&normed, positions);
            GELU.apply(&mut hidden, self.ffn);
            add_into(&mut x, &layer.fc2.forward(&hidden, positions));
        }
        self.norm.apply(&x, d)
    }
}
