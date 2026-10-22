//! Qwen3 decoder-only language model (Hugging Face `Qwen3ForCausalLM`):
//! RMS norms, per-head q and k norms, grouped-query attention with rotary
//! positions, and a SwiGLU MLP. Trit uses it for text models that work on
//! a transcript, such as Superwhisper S1-mini.

use crate::cpu::{activation, buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::models::transformer::{add_into, argmax, attention, linear, stacked, RmsNorm, Rotary};
use crate::Precision;
use std::path::Path;

struct Config {
    d: usize,
    ffn: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    vocab: usize,
    eps: f32,
    rope_base: f32,
    tied: bool,
}

impl Config {
    fn parse(text: &str) -> Result<Self, String> {
        let json: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("config.json is not JSON: {e}"))?;
        let num = |key: &str| -> Result<usize, String> {
            json[key]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("config.json has no {key}"))
        };
        let heads = num("num_attention_heads")?;
        let kv_heads = num("num_key_value_heads")?;
        if json["model_type"].as_str() != Some("qwen3") || !heads.is_multiple_of(kv_heads) {
            return Err("config.json does not describe a Qwen3 model this engine runs".into());
        }
        Ok(Self {
            d: num("hidden_size")?,
            ffn: num("intermediate_size")?,
            layers: num("num_hidden_layers")?,
            heads,
            kv_heads,
            head_dim: num("head_dim")?,
            vocab: num("vocab_size")?,
            eps: json["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            rope_base: json["rope_theta"].as_f64().unwrap_or(1_000_000.0) as f32,
            tied: json["tie_word_embeddings"].as_bool().unwrap_or(false),
        })
    }
}

struct Layer {
    norm_attn: RmsNorm,
    /// q (heads), k and v (kv heads) stacked: one product.
    qkv: Dense,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    out: Dense,
    norm_mlp: RmsNorm,
    /// gate and up stacked: `2 * ffn` outputs.
    gate_up: Dense,
    down: Dense,
}

pub struct Qwen3 {
    config: Config,
    layers: Vec<Layer>,
    norm: RmsNorm,
    /// Token embeddings.
    embed: Dense,
    /// The output projection when it is not tied to `embed`.
    head: Option<Dense>,
    rotary: Rotary,
}

/// The caches of a prompt start; see `Qwen3::prefix`.
pub struct Prefix {
    tokens: Vec<usize>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl Qwen3 {
    pub fn load(dir: &Path, precision: Precision) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("cannot read config.json: {e}"))?;
        let c = Config::parse(&text)?;
        let t = Tensors::open_with(&dir.join("model.safetensors"), precision)?;
        let (d, hd) = (c.d, c.head_dim);
        let layers = team::map(c.layers, |i| -> Result<Layer, String> {
            let p = format!("model.layers.{i}");
            let a = |n: &str| format!("{p}.self_attn.{n}");
            let m = |n: &str| format!("{p}.mlp.{n}");
            Ok(Layer {
                norm_attn: RmsNorm::load(&t, &format!("{p}.input_layernorm"), d, c.eps)?,
                qkv: Dense::load(
                    &t,
                    &[
                        (&a("q_proj"), c.heads * hd),
                        (&a("k_proj"), c.kv_heads * hd),
                        (&a("v_proj"), c.kv_heads * hd),
                    ],
                    d,
                )?,
                q_norm: RmsNorm::load(&t, &a("q_norm"), hd, c.eps)?,
                k_norm: RmsNorm::load(&t, &a("k_norm"), hd, c.eps)?,
                out: linear(&t, &a("o_proj"), d, c.heads * hd)?,
                norm_mlp: RmsNorm::load(&t, &format!("{p}.post_attention_layernorm"), d, c.eps)?,
                gate_up: stacked(&t, &[&m("gate_proj"), &m("up_proj")], c.ffn, d)?,
                down: linear(&t, &m("down_proj"), d, c.ffn)?,
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        let head = if c.tied || !t.has("lm_head.weight") {
            None
        } else {
            Some(Dense::table(&t, "lm_head.weight", c.vocab, d)?)
        };
        Ok(Self {
            layers,
            norm: RmsNorm::load(&t, "model.norm", d, c.eps)?,
            embed: Dense::table(&t, "model.embed_tokens.weight", c.vocab, d)?,
            head,
            rotary: Rotary::halves(hd, c.rope_base),
            config: c,
        })
    }

    /// Greedy continuation of `prompt` until a `stop` token or `max_new` tokens.
    /// Runs a prompt start that many calls share (a system prompt) once; a
    /// `generate` whose prompt begins with it starts from its caches.
    pub fn prefix(&self, tokens: &[usize]) -> Prefix {
        let c = &self.config;
        let kv = c.kv_heads * c.head_dim;
        let mut keys = vec![Vec::with_capacity(tokens.len() * kv); c.layers];
        let mut values = vec![Vec::with_capacity(tokens.len() * kv); c.layers];
        if !tokens.is_empty() {
            self.forward(tokens, 0, &mut keys, &mut values);
        }
        Prefix {
            tokens: tokens.to_vec(),
            keys,
            values,
        }
    }

    /// Greedy tokens after `prompt` until a `stop` token or `max_new` tokens.
    pub fn generate(
        &self,
        prompt: &[usize],
        prefix: Option<&Prefix>,
        max_new: usize,
        stop: &[usize],
    ) -> Vec<usize> {
        let c = &self.config;
        let kv = c.kv_heads * c.head_dim;
        let capacity = (prompt.len() + max_new) * kv;
        let cache = |from: Option<&Vec<f32>>| {
            let mut cache = Vec::with_capacity(capacity);
            cache.extend_from_slice(from.map_or(&[][..], Vec::as_slice));
            cache
        };
        // The prefix counts only when the prompt continues it.
        let prefix =
            prefix.filter(|p| prompt.len() > p.tokens.len() && prompt.starts_with(&p.tokens));
        let done = prefix.map_or(0, |p| p.tokens.len());
        let mut keys: Vec<Vec<f32>> = (0..c.layers)
            .map(|l| cache(prefix.map(|p| &p.keys[l])))
            .collect();
        let mut values: Vec<Vec<f32>> = (0..c.layers)
            .map(|l| cache(prefix.map(|p| &p.values[l])))
            .collect();
        // The rest of the prompt goes through as one batch; then one token per step.
        let mut logits = self.forward(&prompt[done..], done, &mut keys, &mut values);
        let mut out = Vec::new();
        while out.len() < max_new {
            let token = argmax(&logits);
            if stop.contains(&token) {
                break;
            }
            out.push(token);
            logits = self.forward(
                &[token],
                prompt.len() + out.len() - 1,
                &mut keys,
                &mut values,
            );
        }
        out
    }

    /// Runs `tokens` at positions `first..`, extends the caches, and returns
    /// the logits after the last token.
    fn forward(
        &self,
        tokens: &[usize],
        first: usize,
        keys: &mut [Vec<f32>],
        values: &mut [Vec<f32>],
    ) -> Buf {
        let c = &self.config;
        let (d, hd, m) = (c.d, c.head_dim, tokens.len());
        let (qd, kv) = (c.heads * hd, c.kv_heads * hd);
        let mut x = Buf::scratch(m * d);
        for (row, &token) in x.chunks_exact_mut(d).zip(tokens) {
            self.embed.row(token, row);
        }
        for (l, layer) in self.layers.iter().enumerate() {
            let normed = layer.norm_attn.apply(&x);
            let mut qkv = layer.qkv.forward(&normed, m);
            team::chunks_mut(&mut qkv, qd + 2 * kv, |_, row| {
                layer.q_norm.apply_in_place(&mut row[..qd]);
                layer.k_norm.apply_in_place(&mut row[qd..qd + kv]);
            });
            self.rotary
                .apply(&mut qkv, (m, qd + 2 * kv), (c.heads, hd), first);
            self.rotary
                .apply(&mut qkv[qd..], (m, qd + 2 * kv), (c.kv_heads, hd), first);
            for row in qkv.chunks_exact(qd + 2 * kv) {
                keys[l].extend_from_slice(&row[qd..qd + kv]);
                values[l].extend_from_slice(&row[qd + kv..]);
            }
            let seen = first + m;
            let att = attention(
                (&qkv, qd + 2 * kv),
                (&keys[l], kv),
                (&values[l], kv),
                (m, seen),
                (qd, c.heads, c.kv_heads),
                true,
            );
            add_into(&mut x, &layer.out.forward(&att, m));
            let normed = layer.norm_mlp.apply(&x);
            let wide = layer.gate_up.forward(&normed, m);
            let mut hidden = Buf::scratch(m * c.ffn);
            team::chunks_mut(&mut hidden, c.ffn, |r, h| {
                let row = &wide[r * 2 * c.ffn..(r + 1) * 2 * c.ffn];
                h.copy_from_slice(&row[..c.ffn]);
                activation::silu(h);
                for (h, up) in h.iter_mut().zip(&row[c.ffn..]) {
                    *h *= up;
                }
            });
            add_into(&mut x, &layer.down.forward(&hidden, m));
        }
        let last = self.norm.apply(&x[(m - 1) * d..]);
        self.head.as_ref().unwrap_or(&self.embed).forward(&last, 1)
    }
}
