//! Moonshine (Useful Sensors, tiny and base): a conv stem over raw audio,
//! a transformer encoder and decoder with rotary positions, and a gated
//! SiLU MLP in the decoder. Tensor names follow the Hugging Face
//! `MoonshineForConditionalGeneration` files.

use crate::audio;
use crate::cpu::{activation, buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::models::config::Json;
use crate::models::layers::{
    add_into, argmax, attention, attention_with, conv_columns, linear, stacked, Gelu, HeadMajor,
    Norm, Rotary,
};
use crate::models::parts::beam;
use crate::models::parts::logit_bias::{self as bias, Bias, BiasWeights};
use crate::text::sentencepiece::Tokenizer;
use crate::{Options, Precision};
use parking_lot::RwLock;
use std::path::Path;

/// Moonshine reads at most about 6.5 tokens per second of speech.
const TOKENS_PER_SECOND: f32 = 6.5;
/// Beams unless the caller sets `Options::beam`: on the 456 English test
/// clips 5 beams gave 557 and 415 errors (tiny, base), greedy 653 and 477.
const BEAMS: usize = 5;
/// Whose beam search rules to follow: those of Transformers, Moonshine's
/// reference implementation (557 and 415 errors, OpenAI's 561 and 417).
const BEAM_RULES: crate::models::parts::beam::Rules =
    crate::models::parts::beam::Rules::Transformers;
/// Longest window per pass; the decoder holds 194 positions.
const WINDOW_SAMPLES: usize = 30 * 16_000;

struct Config {
    d: usize,
    ffn: usize,
    heads: usize,
    encoder_layers: usize,
    decoder_layers: usize,
    vocab: usize,
    max_tokens: usize,
    bos: usize,
    eos: usize,
    rotary_dims: usize,
    rope_base: f32,
    /// The encoder MLP activation (`encoder_hidden_act`).
    gelu: Gelu,
}

impl Config {
    fn parse(json: &Json) -> Result<Self, String> {
        let num = |key: &str| json.num(key);
        let d = num("hidden_size")?;
        let heads = num("encoder_num_attention_heads")?;
        if num("decoder_num_attention_heads")? != heads
            || num("encoder_num_key_value_heads")? != heads
            || num("decoder_num_key_value_heads")? != heads
            || !d.is_multiple_of(heads)
        {
            return Err("config.json describes a Moonshine shape this engine does not run".into());
        }
        let factor = json.float("partial_rotary_factor").unwrap_or(1.0);
        Ok(Self {
            d,
            ffn: num("intermediate_size")?,
            heads,
            encoder_layers: num("encoder_num_hidden_layers")?,
            decoder_layers: num("decoder_num_hidden_layers")?,
            vocab: num("vocab_size")?,
            max_tokens: num("max_position_embeddings")?,
            bos: num("decoder_start_token_id")?,
            eos: num("eos_token_id")?,
            // int(head_dim * factor), rounded down to whole pairs.
            rotary_dims: ((d / heads) as f64 * factor) as usize / 2 * 2,
            rope_base: json.float("rope_theta").unwrap_or(10_000.0) as f32,
            gelu: Gelu::parse(json.str("encoder_hidden_act").unwrap_or("gelu"))?,
        })
    }
}

struct EncoderLayer {
    norm_attn: Norm,
    qkv: Dense,
    out: Dense,
    norm_mlp: Norm,
    fc1: Dense,
    fc2: Dense,
}

struct DecoderLayer {
    norm_self: Norm,
    self_qkv: Dense,
    self_out: Dense,
    norm_cross: Norm,
    cross_q: Dense,
    cross_kv: Dense,
    cross_out: Dense,
    norm_mlp: Norm,
    /// `2 * ffn` outputs: the value half, then the gate half.
    fc1: Dense,
    fc2: Dense,
}

pub struct Moonshine {
    config: Config,
    conv1: Dense,
    group_norm: (Vec<f32>, Vec<f32>),
    conv2: Dense,
    conv3: Dense,
    encoder: Vec<EncoderLayer>,
    encoder_norm: Norm,
    decoder: Vec<DecoderLayer>,
    decoder_norm: Norm,
    /// Token embeddings; also the output projection (tied).
    embed: Dense,
    rotary: Rotary,
    tokenizer: Tokenizer,
    dictionary: RwLock<Option<Bias>>,
}

impl Moonshine {
    pub fn load(dir: &Path, config: &Json, precision: Precision) -> Result<Self, String> {
        let c = Config::parse(config)?;
        let t = Tensors::open_dir(dir, precision)?;
        let d = c.d;
        let attn = |p: &str, n: &str| format!("{p}.{n}");
        let encoder = (0..c.encoder_layers)
            .map(|i| -> Result<EncoderLayer, String> {
                let p = format!("model.encoder.layers.{i}");
                let a = format!("{p}.self_attn");
                Ok(EncoderLayer {
                    norm_attn: Norm::load(&t, &format!("{p}.input_layernorm"), d)?,
                    qkv: stacked(
                        &t,
                        &[
                            &attn(&a, "q_proj"),
                            &attn(&a, "k_proj"),
                            &attn(&a, "v_proj"),
                        ],
                        d,
                        d,
                    )?,
                    out: linear(&t, &attn(&a, "o_proj"), d, d)?,
                    norm_mlp: Norm::load(&t, &format!("{p}.post_attention_layernorm"), d)?,
                    fc1: linear(&t, &format!("{p}.mlp.fc1"), c.ffn, d)?,
                    fc2: linear(&t, &format!("{p}.mlp.fc2"), d, c.ffn)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let decoder = (0..c.decoder_layers)
            .map(|i| -> Result<DecoderLayer, String> {
                let p = format!("model.decoder.layers.{i}");
                let own = format!("{p}.self_attn");
                let cross = format!("{p}.encoder_attn");
                Ok(DecoderLayer {
                    norm_self: Norm::load(&t, &format!("{p}.input_layernorm"), d)?,
                    self_qkv: stacked(
                        &t,
                        &[
                            &attn(&own, "q_proj"),
                            &attn(&own, "k_proj"),
                            &attn(&own, "v_proj"),
                        ],
                        d,
                        d,
                    )?,
                    self_out: linear(&t, &attn(&own, "o_proj"), d, d)?,
                    norm_cross: Norm::load(&t, &format!("{p}.post_attention_layernorm"), d)?,
                    cross_q: linear(&t, &attn(&cross, "q_proj"), d, d)?,
                    cross_kv: stacked(
                        &t,
                        &[&attn(&cross, "k_proj"), &attn(&cross, "v_proj")],
                        d,
                        d,
                    )?,
                    cross_out: linear(&t, &attn(&cross, "o_proj"), d, d)?,
                    norm_mlp: Norm::load(&t, &format!("{p}.final_layernorm"), d)?,
                    fc1: linear(&t, &format!("{p}.mlp.fc1"), 2 * c.ffn, d)?,
                    fc2: linear(&t, &format!("{p}.mlp.fc2"), d, c.ffn)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            conv1: linear(&t, "model.encoder.conv1", d, 127)?,
            group_norm: (
                t.f32_len("model.encoder.groupnorm.weight", d)?,
                t.f32_len("model.encoder.groupnorm.bias", d)?,
            ),
            conv2: linear(&t, "model.encoder.conv2", 2 * d, 7 * d)?,
            conv3: linear(&t, "model.encoder.conv3", d, 3 * 2 * d)?,
            encoder,
            encoder_norm: Norm::load(&t, "model.encoder.layer_norm", d)?,
            decoder,
            decoder_norm: Norm::load(&t, "model.decoder.norm", d)?,
            embed: Dense::table(&t, "model.decoder.embed_tokens.weight", c.vocab, d)?,
            rotary: Rotary::interleaved(c.rotary_dims, c.rope_base),
            tokenizer: Tokenizer::open(&dir.join("tokenizer.json"))?,
            dictionary: RwLock::new(None),
            config: c,
        })
    }

    /// Moonshine is English only; `options.language` changes nothing.
    /// Terms to favor while decoding; with terms set, decoding is greedy (the
    /// bonuses follow one sequence). See `parakeet::bias`.
    pub fn set_dictionary(&self, terms: &[String], weights: BiasWeights) -> Vec<String> {
        let (bias, skipped) = bias::build(&self.tokenizer, terms, weights);
        *self.dictionary.write() = bias;
        skipped
    }

    /// The text of each window that holds speech: `(first sample, samples, text)`.
    pub fn pieces(&self, samples: &[f32], options: &Options) -> Vec<(usize, usize, String)> {
        let dictionary = self.dictionary.read();
        let mut pieces = Vec::new();
        for window in audio::windows(samples, WINDOW_SAMPLES) {
            // The conv stem needs at least one output frame.
            if window.len() < 127 + 64 * (6 + 2 * 3) || audio::is_silent(window) {
                continue;
            }
            let encoded = self.encode(window);
            let limit = ((window.len() as f32 / 16_000.0 * TOKENS_PER_SECOND) as usize + 4)
                .min(self.config.max_tokens);
            let ids = self.generate(&encoded, limit, options.beams(BEAMS), dictionary.as_ref());
            let piece = self.tokenizer.decode(&ids);
            let piece = piece.trim();
            if !piece.is_empty() {
                pieces.push((
                    audio::offset(samples, window),
                    window.len(),
                    piece.to_string(),
                ));
            }
        }
        pieces
    }

    /// Encoder output, `frames x d`.
    fn encode(&self, samples: &[f32]) -> (Buf, usize) {
        let c = &self.config;
        let d = c.d;
        let l1 = (samples.len() - 127) / 64 + 1;
        let cols = conv_columns(samples, (samples.len(), 1), (127, 64, 0), l1);
        let mut h = self.conv1.forward(&cols, l1);
        activation::tanh(&mut h);
        // Group norm with one group: statistics over every channel and frame.
        let n = h.len() as f32;
        let mean = h.iter().sum::<f32>() / n;
        let var = h.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + 1e-5).sqrt();
        let (weight, bias) = &self.group_norm;
        team::chunks_mut(&mut h, d, |_, row| {
            for ((v, w), b) in row.iter_mut().zip(weight).zip(bias) {
                *v = (*v - mean) * inv * w + b;
            }
        });
        let l2 = (l1 - 7) / 3 + 1;
        let mut h = self
            .conv2
            .forward(&conv_columns(&h, (l1, d), (7, 3, 0), l2), l2);
        // The conv stem always uses the exact form (`F.gelu`).
        Gelu::Exact.apply(&mut h, 2 * d);
        let frames = (l2 - 3) / 2 + 1;
        let mut x = self
            .conv3
            .forward(&conv_columns(&h, (l2, 2 * d), (3, 2, 0), frames), frames);
        Gelu::Exact.apply(&mut x, d);
        for layer in &self.encoder {
            let normed = layer.norm_attn.apply(&x, d);
            let mut qkv = layer.qkv.forward(&normed, frames);
            self.rotary
                .apply(&mut qkv, (frames, 3 * d), (c.heads, d / c.heads), 0);
            self.rotary
                .apply(&mut qkv[d..], (frames, 3 * d), (c.heads, d / c.heads), 0);
            let att = attention(
                (&qkv, 3 * d),
                (&qkv[d..], 3 * d),
                (&qkv[2 * d..], 3 * d),
                (frames, frames),
                (d, c.heads, c.heads),
                false,
            );
            add_into(&mut x, &layer.out.forward(&att, frames));
            let normed = layer.norm_mlp.apply(&x, d);
            let mut hidden = layer.fc1.forward(&normed, frames);
            c.gelu.apply(&mut hidden, c.ffn);
            add_into(&mut x, &layer.fc2.forward(&hidden, frames));
        }
        (self.encoder_norm.apply(&x, d), frames)
    }

    /// Tokens from the start token until end of text or `limit`: greedy
    /// (with the dictionary, when there is one), or `beam` beams.
    fn generate(
        &self,
        (encoded, frames): &(Buf, usize),
        limit: usize,
        beam: usize,
        dictionary: Option<&Bias>,
    ) -> Vec<usize> {
        let c = &self.config;
        let d = c.d;
        // Cross-attention keys and values, laid out per head once per window.
        let cross: Vec<HeadMajor> = self
            .decoder
            .iter()
            .map(|layer| {
                let kv = layer.cross_kv.forward(encoded, *frames);
                HeadMajor::new(
                    (&kv, 2 * d),
                    (&kv[d..], 2 * d),
                    *frames,
                    (d / c.heads, c.heads),
                )
            })
            .collect();
        let mut cache = Cache {
            keys: vec![Vec::with_capacity(limit * d); c.decoder_layers],
            values: vec![Vec::with_capacity(limit * d); c.decoder_layers],
            position: 0,
        };
        let mut logits = self.step(&cross, &mut cache, c.bos);
        // The dictionary works on greedy decoding; with one, decode greedy.
        if beam > 1 && dictionary.is_none() {
            let step =
                |caches: &mut [Cache], tokens: &[usize]| self.step_beams(&cross, caches, tokens);
            return beam::search(
                (beam, BEAM_RULES),
                (cache, logits),
                (limit, c.eos),
                |_, _| {},
                step,
            )
            .tokens;
        }
        let mut ids = Vec::new();
        let mut run = dictionary.map(bias::Run::new);
        loop {
            let token = match &mut run {
                Some(run) => run.pick(&mut logits),
                None => argmax(&logits),
            };
            if token == c.eos {
                break;
            }
            match &mut run {
                Some(run) => run.push(token),
                None => ids.push(token),
            }
            if run.as_ref().map_or(ids.len(), bias::Run::len) == limit {
                break;
            }
            logits = self.step(&cross, &mut cache, token);
        }
        run.map_or(ids, bias::Run::finish)
    }

    /// Feeds `tokens[i]` to `caches[i]` (beams of one window) as one batch,
    /// so each layer's weights are read once for all beams. Returns the next
    /// logits of each.
    fn step_beams(&self, cross: &[HeadMajor], caches: &mut [Cache], tokens: &[usize]) -> Vec<Buf> {
        let c = &self.config;
        let (d, m) = (c.d, tokens.len());
        let mut x = Buf::scratch(m * d);
        for (row, &token) in x.chunks_exact_mut(d).zip(tokens) {
            self.embed.row(token, row);
        }
        for (l, layer) in self.decoder.iter().enumerate() {
            let normed = layer.norm_self.apply(&x, d);
            let mut qkv = layer.self_qkv.forward(&normed, m);
            let mut att = Buf::scratch(m * d);
            for ((row, out), cache) in qkv
                .chunks_exact_mut(3 * d)
                .zip(att.chunks_exact_mut(d))
                .zip(caches.iter_mut())
            {
                let position = cache.position;
                self.rotary
                    .apply(row, (1, 3 * d), (c.heads, d / c.heads), position);
                self.rotary
                    .apply(&mut row[d..], (1, 3 * d), (c.heads, d / c.heads), position);
                cache.keys[l].extend_from_slice(&row[d..2 * d]);
                cache.values[l].extend_from_slice(&row[2 * d..]);
                out.copy_from_slice(&attention(
                    (&row[..d], d),
                    (&cache.keys[l], d),
                    (&cache.values[l], d),
                    (1, position + 1),
                    (d, c.heads, c.heads),
                    false,
                ));
            }
            add_into(&mut x, &layer.self_out.forward(&att, m));
            let normed = layer.norm_cross.apply(&x, d);
            let q = layer.cross_q.forward(&normed, m);
            let att = attention_with((&q, d), &cross[l], m, (d, c.heads), false);
            add_into(&mut x, &layer.cross_out.forward(&att, m));
            let normed = layer.norm_mlp.apply(&x, d);
            let mut wide = layer.fc1.forward(&normed, m);
            let mut gated = Buf::scratch(m * c.ffn);
            for (row, out) in wide
                .chunks_exact_mut(2 * c.ffn)
                .zip(gated.chunks_exact_mut(c.ffn))
            {
                let (value, gate) = row.split_at_mut(c.ffn);
                activation::silu(gate);
                for ((o, v), g) in out.iter_mut().zip(value.iter()).zip(gate.iter()) {
                    *o = v * g;
                }
            }
            add_into(&mut x, &layer.fc2.forward(&gated, m));
        }
        caches.iter_mut().for_each(|cache| cache.position += 1);
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

    /// Feeds one token and returns the logits for the next one.
    fn step(&self, cross: &[HeadMajor], cache: &mut Cache, token: usize) -> Buf {
        let c = &self.config;
        let d = c.d;
        let position = cache.position;
        let mut x = vec![0.0; d];
        self.embed.row(token, &mut x);
        for (l, layer) in self.decoder.iter().enumerate() {
            let normed = layer.norm_self.apply(&x, d);
            let mut qkv = layer.self_qkv.forward(&normed, 1);
            self.rotary
                .apply(&mut qkv, (1, 3 * d), (c.heads, d / c.heads), position);
            self.rotary
                .apply(&mut qkv[d..], (1, 3 * d), (c.heads, d / c.heads), position);
            cache.keys[l].extend_from_slice(&qkv[d..2 * d]);
            cache.values[l].extend_from_slice(&qkv[2 * d..]);
            let seen = position + 1;
            let att = attention(
                (&qkv[..d], d),
                (&cache.keys[l], d),
                (&cache.values[l], d),
                (1, seen),
                (d, c.heads, c.heads),
                false,
            );
            add_into(&mut x, &layer.self_out.forward(&att, 1));
            let normed = layer.norm_cross.apply(&x, d);
            let q = layer.cross_q.forward(&normed, 1);
            let att = attention_with((&q, d), &cross[l], 1, (d, c.heads), false);
            add_into(&mut x, &layer.cross_out.forward(&att, 1));
            let normed = layer.norm_mlp.apply(&x, d);
            let mut wide = layer.fc1.forward(&normed, 1);
            let (value, gate) = wide.split_at_mut(c.ffn);
            activation::silu(gate);
            for (v, g) in value.iter_mut().zip(gate.iter()) {
                *v *= g;
            }
            add_into(&mut x, &layer.fc2.forward(value, 1));
        }
        cache.position += 1;
        let normed = self.decoder_norm.apply(&x, d);
        self.embed.forward(&normed, 1)
    }
}

/// The decoder's self-attention caches; a clone is a beam that splits.
#[derive(Clone)]
struct Cache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    position: usize,
}
