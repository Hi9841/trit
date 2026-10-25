//! The phonemes of a word the lexicon does not hold, from a small BART
//! (PeterReid/graphemes_to_phonemes_en_us and _gb, the fallback of
//! misaki's main branch): letters in, Kokoro phonemes out, decoded greedily
//! as Transformers' `generate` does with the model's defaults (20 tokens
//! after the start token, the end token forced at the last step).

use crate::cpu::{gemm::Dense, tensors::Tensors};
use crate::models::config::Json;
use crate::models::layers::{attention, linear, Gelu, Norm};
use crate::Precision;
use std::collections::HashMap;
use std::path::Path;

const BOS: usize = 1;
const EOS: usize = 2;
const UNKNOWN: usize = 3;
/// `generate`'s length limit: the default `max_length` (20) plus the length
/// of the decoder input (the start token), which `generate` adds to a
/// default limit.
const MAX_LENGTH: usize = 21;
/// BART's learned positions start at index 2.
const POSITION_OFFSET: usize = 2;

struct Attention {
    q: Dense,
    k: Dense,
    v: Dense,
    out: Dense,
    norm: Norm,
}

impl Attention {
    fn load(t: &Tensors, name: &str, norm: &str, d: usize) -> Result<Self, String> {
        Ok(Self {
            q: linear(t, &format!("{name}.q_proj"), d, d)?,
            k: linear(t, &format!("{name}.k_proj"), d, d)?,
            v: linear(t, &format!("{name}.v_proj"), d, d)?,
            out: linear(t, &format!("{name}.out_proj"), d, d)?,
            norm: Norm::load(t, norm, d)?,
        })
    }

    /// `norm(x + attention(x over memory))`.
    fn forward(
        &self,
        x: &mut [f32],
        rows: usize,
        memory: Option<(&[f32], usize)>,
        heads: usize,
        causal: bool,
    ) -> Vec<f32> {
        let d = self.q.out_features;
        let (source, keys) = memory.unwrap_or((x, rows));
        let (q, k, v) = (
            self.q.forward(x, rows),
            self.k.forward(source, keys),
            self.v.forward(source, keys),
        );
        let mixed = attention(
            (&q, d),
            (&k, d),
            (&v, d),
            (rows, keys),
            (d, heads, heads),
            causal,
        );
        let projected = self.out.forward(&mixed, rows);
        self.norm.residual(x, &projected, 1.0).to_vec()
    }
}

struct Layer {
    self_attention: Attention,
    cross: Option<Attention>,
    fc1: Dense,
    fc2: Dense,
    final_norm: Norm,
}

impl Layer {
    fn load(
        t: &Tensors,
        name: &str,
        (d, ffn): (usize, usize),
        decoder: bool,
    ) -> Result<Self, String> {
        Ok(Self {
            self_attention: Attention::load(
                t,
                &format!("{name}.self_attn"),
                &format!("{name}.self_attn_layer_norm"),
                d,
            )?,
            cross: if decoder {
                Some(Attention::load(
                    t,
                    &format!("{name}.encoder_attn"),
                    &format!("{name}.encoder_attn_layer_norm"),
                    d,
                )?)
            } else {
                None
            },
            fc1: linear(t, &format!("{name}.fc1"), ffn, d)?,
            fc2: linear(t, &format!("{name}.fc2"), d, ffn)?,
            final_norm: Norm::load(t, &format!("{name}.final_layer_norm"), d)?,
        })
    }

    fn forward(
        &self,
        mut x: Vec<f32>,
        rows: usize,
        memory: Option<(&[f32], usize)>,
        heads: usize,
    ) -> Vec<f32> {
        x = self
            .self_attention
            .forward(&mut x, rows, None, heads, memory.is_some());
        if let (Some(cross), Some(memory)) = (&self.cross, memory) {
            x = cross.forward(&mut x, rows, Some(memory), heads, false);
        }
        let mut inner = self.fc1.forward(&x, rows);
        Gelu::Exact.apply(&mut inner, self.fc1.out_features);
        let out = self.fc2.forward(&inner, rows);
        self.final_norm.residual(&mut x, &out, 1.0).to_vec()
    }
}

pub struct Fallback {
    shared: Vec<f32>,
    encoder_positions: Vec<f32>,
    decoder_positions: Vec<f32>,
    encoder_norm: Norm,
    decoder_norm: Norm,
    encoder: Vec<Layer>,
    decoder: Vec<Layer>,
    logits_bias: Vec<f32>,
    graphemes: HashMap<char, usize>,
    phonemes: Vec<String>,
    d: usize,
    heads: usize,
    vocab: usize,
    positions: usize,
}

impl Fallback {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("cannot read {}: {e}", dir.join("config.json").display()))?;
        let c = Json::parse(&text)?;
        let t = Tensors::open_dir(dir, Precision::F32)?;
        let d = c.num("d_model")?;
        let vocab = c.num("vocab_size")?;
        let positions = c.num("max_position_embeddings")?;
        let ffn = c.num("encoder_ffn_dim")?;
        // One character per token: a string, or a list of strings.
        let chars = |key: &str| -> Result<Vec<String>, String> {
            let value = c.get(key);
            if let Some(text) = value.as_str() {
                return Ok(text.chars().map(String::from).collect());
            }
            value
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|v| v.as_str().unwrap_or_default().to_string())
                        .collect()
                })
                .ok_or_else(|| format!("config.json has no {key}"))
        };
        let graphemes = chars("grapheme_chars")?
            .iter()
            .enumerate()
            .filter_map(|(i, g)| {
                let mut it = g.chars();
                match (it.next(), it.next()) {
                    (Some(c), None) => Some((c, i)),
                    _ => None,
                }
            })
            .collect();
        let layers = |part: &str, count: usize, decoder: bool| -> Result<Vec<Layer>, String> {
            (0..count)
                .map(|i| Layer::load(&t, &format!("model.{part}.layers.{i}"), (d, ffn), decoder))
                .collect()
        };
        let position_rows = positions + POSITION_OFFSET;
        Ok(Self {
            shared: t.f32_len("model.shared.weight", vocab * d)?,
            encoder_positions: t
                .f32_len("model.encoder.embed_positions.weight", position_rows * d)?,
            decoder_positions: t
                .f32_len("model.decoder.embed_positions.weight", position_rows * d)?,
            encoder_norm: Norm::load(&t, "model.encoder.layernorm_embedding", d)?,
            decoder_norm: Norm::load(&t, "model.decoder.layernorm_embedding", d)?,
            encoder: layers("encoder", c.num("encoder_layers")?, false)?,
            decoder: layers("decoder", c.num("decoder_layers")?, true)?,
            logits_bias: if t.has("final_logits_bias") {
                t.f32_len("final_logits_bias", vocab)?
            } else {
                vec![0.0; vocab]
            },
            graphemes,
            phonemes: chars("phoneme_chars")?,
            d,
            heads: c.num("encoder_attention_heads")?,
            vocab,
            positions,
        })
    }

    fn embed(&self, ids: &[usize], positions: &[f32], norm: &Norm) -> Vec<f32> {
        let d = self.d;
        let mut x = vec![0.0f32; ids.len() * d];
        for (i, (row, id)) in x.chunks_exact_mut(d).zip(ids).enumerate() {
            let token = &self.shared[id * d..(id + 1) * d];
            let position = &positions[(i + POSITION_OFFSET) * d..(i + POSITION_OFFSET + 1) * d];
            for ((o, a), b) in row.iter_mut().zip(token).zip(position) {
                *o = a + b;
            }
        }
        norm.apply(&x, d).to_vec()
    }

    /// The phonemes of `word`, or `None` when the model gives none.
    pub fn phonemes(&self, word: &str) -> Option<String> {
        let mut ids = vec![BOS];
        ids.extend(
            word.chars()
                .map(|c| *self.graphemes.get(&c).unwrap_or(&UNKNOWN)),
        );
        ids.push(EOS);
        if ids.len() > self.positions {
            return None;
        }
        let mut memory = self.embed(&ids, &self.encoder_positions, &self.encoder_norm);
        for layer in &self.encoder {
            memory = layer.forward(memory, ids.len(), None, self.heads);
        }
        let mut out = vec![BOS];
        while out.len() < MAX_LENGTH {
            let mut x = self.embed(&out, &self.decoder_positions, &self.decoder_norm);
            for layer in &self.decoder {
                x = layer.forward(x, out.len(), Some((&memory, ids.len())), self.heads);
            }
            let last = &x[(out.len() - 1) * self.d..];
            let next = if out.len() == MAX_LENGTH - 1 {
                EOS
            } else {
                let mut best = (f32::NEG_INFINITY, 0usize);
                for v in 0..self.vocab {
                    let row = &self.shared[v * self.d..(v + 1) * self.d];
                    let logit =
                        row.iter().zip(last).map(|(a, b)| a * b).sum::<f32>() + self.logits_bias[v];
                    if logit > best.0 {
                        best = (logit, v);
                    }
                }
                best.1
            };
            out.push(next);
            if next == EOS {
                break;
            }
        }
        let text: String = out
            .iter()
            .filter(|t| **t > UNKNOWN)
            .filter_map(|t| self.phonemes.get(*t).map(String::as_str))
            .collect();
        Some(text)
    }
}
