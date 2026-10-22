//! The Whisper network (OpenAI Whisper, Distil-Whisper, large-v3-turbo):
//! conv stem and pre-norm transformer encoder over 1500 positions, and a
//! transformer decoder with self- and cross-attention. Tensor names follow
//! the Hugging Face `WhisperForConditionalGeneration` files.

use crate::cpu::{buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::models::transformer::{
    add_into, attention, attention_with, conv_columns, linear, stacked, Gelu, HeadMajor, Norm,
};

pub struct Config {
    pub d: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_heads: usize,
    pub decoder_heads: usize,
    pub encoder_ffn: usize,
    pub decoder_ffn: usize,
    pub mels: usize,
    pub vocab: usize,
    /// Encoder positions: 1500 for a 30 s window.
    pub positions: usize,
    /// Decoder positions: prompt plus text, 448.
    pub max_tokens: usize,
}

/// The exact GELU, as OpenAI Whisper and Transformers compute it. With 5
/// beams it gives 768 and 479 errors (tiny, base.en) on the 456 English
/// test clips, the tanh form of whisper.cpp 766 and 480; greedy, the tanh
/// form makes 2 fewer errors each. Beam search is the default, so exact.
const GELU: Gelu = Gelu::Exact;

impl Config {
    pub fn parse(text: &str) -> Result<Self, String> {
        let json: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("config.json is not JSON: {e}"))?;
        let num = |key: &str| -> Result<usize, String> {
            json[key]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("config.json has no {key}"))
        };
        let config = Self {
            d: num("d_model")?,
            encoder_layers: num("encoder_layers")?,
            decoder_layers: num("decoder_layers")?,
            encoder_heads: num("encoder_attention_heads")?,
            decoder_heads: num("decoder_attention_heads")?,
            encoder_ffn: num("encoder_ffn_dim")?,
            decoder_ffn: num("decoder_ffn_dim")?,
            mels: num("num_mel_bins")?,
            vocab: num("vocab_size")?,
            positions: num("max_source_positions")?,
            max_tokens: num("max_target_positions")?,
        };
        if !config.d.is_multiple_of(config.encoder_heads)
            || !config.d.is_multiple_of(config.decoder_heads)
            || json["scale_embedding"].as_bool() == Some(true)
        {
            return Err("config.json describes a Whisper shape this engine does not run".into());
        }
        Ok(config)
    }
}

struct EncoderLayer {
    norm_attn: Norm,
    /// q, k, v stacked: one product.
    qkv: Dense,
    out: Dense,
    norm_ffn: Norm,
    fc1: Dense,
    fc2: Dense,
}

struct DecoderLayer {
    norm_self: Norm,
    self_qkv: Dense,
    self_out: Dense,
    norm_cross: Norm,
    cross_q: Dense,
    /// Cross-attention k and v, applied once per window to the encoder output.
    cross_kv: Dense,
    cross_out: Dense,
    norm_ffn: Norm,
    fc1: Dense,
    fc2: Dense,
}

pub struct Whisper {
    pub config: Config,
    conv1: Dense,
    conv2: Dense,
    encoder_positions: Vec<f32>,
    encoder: Vec<EncoderLayer>,
    encoder_norm: Norm,
    /// Token embeddings; also the output projection (tied).
    embed: Dense,
    decoder_positions: Vec<f32>,
    decoder: Vec<DecoderLayer>,
    decoder_norm: Norm,
}

/// The decoder's state over one window. A clone shares the cross-attention
/// keys and copies the self-attention caches (a beam that splits).
#[derive(Clone)]
pub struct Session {
    /// Per layer: cross-attention keys and values, laid out per head once
    /// per window.
    cross: std::sync::Arc<Vec<HeadMajor>>,
    /// Per layer: self-attention keys and values so far, `tokens x d` each.
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    pub position: usize,
}

impl Session {
    /// Forgets every token after the first `position`, to decode a window again.
    pub fn rewind(&mut self, position: usize, d: usize) {
        for cache in self.keys.iter_mut().chain(&mut self.values) {
            cache.truncate(position * d);
        }
        self.position = position;
    }
}

impl Whisper {
    pub fn load(t: &Tensors, config: Config) -> Result<Self, String> {
        let c = &config;
        let d = c.d;
        let encoder = team::map(c.encoder_layers, |i| -> Result<EncoderLayer, String> {
            let p = format!("model.encoder.layers.{i}");
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
                fc1: linear(t, &format!("{p}.fc1"), c.encoder_ffn, d)?,
                fc2: linear(t, &format!("{p}.fc2"), d, c.encoder_ffn)?,
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        let decoder = (0..c.decoder_layers)
            .map(|i| -> Result<DecoderLayer, String> {
                let p = format!("model.decoder.layers.{i}");
                let own = |n: &str| format!("{p}.self_attn.{n}");
                let cross = |n: &str| format!("{p}.encoder_attn.{n}");
                Ok(DecoderLayer {
                    norm_self: Norm::load(t, &format!("{p}.self_attn_layer_norm"), d)?,
                    self_qkv: stacked(t, &[&own("q_proj"), &own("k_proj"), &own("v_proj")], d, d)?,
                    self_out: linear(t, &own("out_proj"), d, d)?,
                    norm_cross: Norm::load(t, &format!("{p}.encoder_attn_layer_norm"), d)?,
                    cross_q: linear(t, &cross("q_proj"), d, d)?,
                    cross_kv: stacked(t, &[&cross("k_proj"), &cross("v_proj")], d, d)?,
                    cross_out: linear(t, &cross("out_proj"), d, d)?,
                    norm_ffn: Norm::load(t, &format!("{p}.final_layer_norm"), d)?,
                    fc1: linear(t, &format!("{p}.fc1"), c.decoder_ffn, d)?,
                    fc2: linear(t, &format!("{p}.fc2"), d, c.decoder_ffn)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            conv1: linear(t, "model.encoder.conv1", d, 3 * c.mels)?,
            conv2: linear(t, "model.encoder.conv2", d, 3 * d)?,
            encoder_positions: t
                .f32_len("model.encoder.embed_positions.weight", c.positions * d)?,
            encoder,
            encoder_norm: Norm::load(t, "model.encoder.layer_norm", d)?,
            embed: Dense::table(t, "model.decoder.embed_tokens.weight", c.vocab, d)?,
            decoder_positions: t
                .f32_len("model.decoder.embed_positions.weight", c.max_tokens * d)?,
            decoder,
            decoder_norm: Norm::load(t, "model.decoder.layer_norm", d)?,
            config,
        })
    }

    /// Encoder output `positions x d` for one window of `2 * positions x mels` features.
    /// Encoder output `positions x d` for one window of `2 * positions x mels`
    /// features. Always the full 30 s: encoding only the positions the audio
    /// covers raised whisper-base.en from 3.42 % to 49 % WER on LibriSpeech.
    pub fn encode(&self, mel: &[f32]) -> Buf {
        let c = &self.config;
        let (d, frames, positions) = (c.d, 2 * c.positions, c.positions);
        let cols = conv_columns(mel, (frames, c.mels), (3, 1, 1), frames);
        let mut h = self.conv1.forward(&cols, frames);
        GELU.apply(&mut h, d);
        let cols = conv_columns(&h, (frames, d), (3, 2, 1), positions);
        let mut x = self.conv2.forward(&cols, positions);
        GELU.apply(&mut x, d);
        add_into(&mut x, &self.encoder_positions);
        for layer in &self.encoder {
            let normed = layer.norm_attn.apply(&x, d);
            let qkv = layer.qkv.forward(&normed, positions);
            let att = attention(
                (&qkv, 3 * d),
                (&qkv[d..], 3 * d),
                (&qkv[2 * d..], 3 * d),
                (positions, positions),
                (d, c.encoder_heads, c.encoder_heads),
                false,
            );
            add_into(&mut x, &layer.out.forward(&att, positions));
            let normed = layer.norm_ffn.apply(&x, d);
            let mut hidden = layer.fc1.forward(&normed, positions);
            GELU.apply(&mut hidden, c.encoder_ffn);
            add_into(&mut x, &layer.fc2.forward(&hidden, positions));
        }
        self.encoder_norm.apply(&x, d)
    }

    /// A decoder session over one encoded window.
    pub fn session(&self, encoded: &[f32]) -> Session {
        let c = &self.config;
        Session {
            cross: std::sync::Arc::new(
                self.decoder
                    .iter()
                    .map(|layer| {
                        let kv = layer.cross_kv.forward(encoded, c.positions);
                        let hd = c.d / c.decoder_heads;
                        HeadMajor::new(
                            (&kv, 2 * c.d),
                            (&kv[c.d..], 2 * c.d),
                            c.positions,
                            (hd, c.decoder_heads),
                        )
                    })
                    .collect(),
            ),
            keys: vec![Vec::with_capacity(c.max_tokens * c.d); c.decoder_layers],
            values: vec![Vec::with_capacity(c.max_tokens * c.d); c.decoder_layers],
            position: 0,
        }
    }

    /// Feeds one token and returns the logits for the next one.
    pub fn step(&self, session: &mut Session, token: usize) -> Buf {
        self.feed(session, &[token])
    }

    /// Feeds `tokens[i]` to `sessions[i]` (beams of one window, which share
    /// its cross-attention keys) as one batch, so each layer's weights are
    /// read once for all beams. Returns the next logits of each.
    pub fn step_beams(&self, sessions: &mut [Session], tokens: &[usize]) -> Vec<Buf> {
        let c = &self.config;
        let (d, m) = (c.d, tokens.len());
        let cross = std::sync::Arc::clone(&sessions[0].cross);
        debug_assert!(sessions
            .iter()
            .all(|s| std::sync::Arc::ptr_eq(&s.cross, &cross)));
        let mut x = Buf::scratch(m * d);
        for ((row, &token), session) in x.chunks_exact_mut(d).zip(tokens).zip(sessions.iter()) {
            let pos = session.position.min(c.max_tokens - 1);
            self.embed.row(token, row);
            add_into(row, &self.decoder_positions[pos * d..(pos + 1) * d]);
        }
        for (l, layer) in self.decoder.iter().enumerate() {
            let normed = layer.norm_self.apply(&x, d);
            let qkv = layer.self_qkv.forward(&normed, m);
            let mut att = Buf::scratch(m * d);
            for ((row, out), session) in qkv
                .chunks_exact(3 * d)
                .zip(att.chunks_exact_mut(d))
                .zip(sessions.iter_mut())
            {
                session.keys[l].extend_from_slice(&row[d..2 * d]);
                session.values[l].extend_from_slice(&row[2 * d..]);
                let seen = session.keys[l].len() / d;
                out.copy_from_slice(&attention(
                    (&row[..d], d),
                    (&session.keys[l], d),
                    (&session.values[l], d),
                    (1, seen),
                    (d, c.decoder_heads, c.decoder_heads),
                    false,
                ));
            }
            add_into(&mut x, &layer.self_out.forward(&att, m));
            let normed = layer.norm_cross.apply(&x, d);
            let q = layer.cross_q.forward(&normed, m);
            // Every query sees the same window, so all beams go as one.
            let att = attention_with((&q, d), &cross[l], m, (d, c.decoder_heads), false);
            add_into(&mut x, &layer.cross_out.forward(&att, m));
            let normed = layer.norm_ffn.apply(&x, d);
            let mut hidden = layer.fc1.forward(&normed, m);
            GELU.apply(&mut hidden, c.decoder_ffn);
            add_into(&mut x, &layer.fc2.forward(&hidden, m));
        }
        sessions.iter_mut().for_each(|s| s.position += 1);
        let normed = self.decoder_norm.apply(&x, d);
        let logits = self.embed.forward(&normed, m);
        logits
            .chunks_exact(c.vocab)
            .map(|row| {
                let mut out = Buf::scratch(c.vocab);
                out.copy_from_slice(row);
                out
            })
            .collect()
    }

    /// Feeds `tokens` as one batch (a prompt) and returns the logits after
    /// the last one.
    pub fn feed(&self, session: &mut Session, tokens: &[usize]) -> Buf {
        let c = &self.config;
        let (d, m) = (c.d, tokens.len());
        let mut x = Buf::scratch(m * d);
        for (r, (row, &token)) in x.chunks_exact_mut(d).zip(tokens).enumerate() {
            let pos = (session.position + r).min(c.max_tokens - 1);
            self.embed.row(token, row);
            add_into(row, &self.decoder_positions[pos * d..(pos + 1) * d]);
        }
        for (l, layer) in self.decoder.iter().enumerate() {
            let normed = layer.norm_self.apply(&x, d);
            let qkv = layer.self_qkv.forward(&normed, m);
            for row in qkv.chunks_exact(3 * d) {
                session.keys[l].extend_from_slice(&row[d..2 * d]);
                session.values[l].extend_from_slice(&row[2 * d..]);
            }
            let seen = session.keys[l].len() / d;
            let att = attention(
                (&qkv, 3 * d),
                (&session.keys[l], d),
                (&session.values[l], d),
                (m, seen),
                (d, c.decoder_heads, c.decoder_heads),
                true,
            );
            add_into(&mut x, &layer.self_out.forward(&att, m));
            let normed = layer.norm_cross.apply(&x, d);
            let q = layer.cross_q.forward(&normed, m);
            let att = attention_with((&q, d), &session.cross[l], m, (d, c.decoder_heads), false);
            add_into(&mut x, &layer.cross_out.forward(&att, m));
            let normed = layer.norm_ffn.apply(&x, d);
            let mut hidden = layer.fc1.forward(&normed, m);
            GELU.apply(&mut hidden, c.decoder_ffn);
            add_into(&mut x, &layer.fc2.forward(&hidden, m));
        }
        session.position += m;
        let normed = self.decoder_norm.apply(&x[(m - 1) * d..], d);
        self.embed.forward(&normed, 1)
    }
}
