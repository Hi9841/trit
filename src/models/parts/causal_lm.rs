//! Decoder-only language models of the Llama plan (Hugging Face
//! `Qwen3ForCausalLM`, `LlamaForCausalLM`, `MistralForCausalLM`,
//! `GraniteForCausalLM`): RMS norms, grouped-query attention with rotary
//! positions (halves), and a SwiGLU MLP. Qwen3 adds per-head q and k norms;
//! Granite scales the embeddings, the attention scores, the residuals, and
//! the logits. Trit uses it for text models that work on a transcript, such
//! as Superwhisper S1-mini, and as the text decoder of the audio-LLM families.

use crate::cpu::{activation, buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::models::config::Json;
use crate::models::files::Files;
use crate::models::layers::{add_into, argmax, attention, linear, stacked, RmsNorm, Rotary};
use crate::{Device, Precision};

struct Config {
    /// Per-head RMS norms on q and k (Qwen3).
    qk_norm: bool,
    /// Granite's multipliers: on the input embeddings, the attention scores
    /// (instead of `1 / sqrt(head_dim)`), each residual branch, and (as a
    /// divisor) the logits. 1 (and `None`) elsewhere.
    embedding_multiplier: f32,
    attention_multiplier: Option<f32>,
    residual_multiplier: f32,
    logits_scaling: f32,
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
    /// The keys under `at` (`""`, or `thinker_config.text_config.` in Qwen3-ASR).
    fn parse(json: &Json, at: &str) -> Result<Self, String> {
        let key = |name: &str| format!("{at}{name}");
        let num = |name: &str| json.num(&key(name));
        let float = |name: &str| json.float(&key(name)).map(|v| v as f32);
        let heads = num("num_attention_heads")?;
        let kv_heads = num("num_key_value_heads")?;
        let qk_norm = match json.str(&key("model_type")) {
            Some("qwen3") => true,
            Some("llama" | "mistral" | "granite") => false,
            _ => {
                return Err(
                    "config.json does not describe a language model this engine runs".into(),
                )
            }
        };
        if !heads.is_multiple_of(kv_heads) {
            return Err("config.json: the query heads do not share the key heads evenly".into());
        }
        let d = num("hidden_size")?;
        Ok(Self {
            qk_norm,
            embedding_multiplier: float("embedding_multiplier").unwrap_or(1.0),
            attention_multiplier: float("attention_multiplier"),
            residual_multiplier: float("residual_multiplier").unwrap_or(1.0),
            logits_scaling: float("logits_scaling").unwrap_or(1.0),
            d,
            ffn: num("intermediate_size")?,
            layers: num("num_hidden_layers")?,
            heads,
            kv_heads,
            head_dim: num("head_dim").unwrap_or(d / heads),
            vocab: num("vocab_size")?,
            eps: json.float(&key("rms_norm_eps")).unwrap_or(1e-6) as f32,
            rope_base: json.float(&key("rope_theta")).unwrap_or(1_000_000.0) as f32,
            tied: json.flag(&key("tie_word_embeddings")).unwrap_or(false),
        })
    }
}

struct Layer {
    norm_attn: RmsNorm,
    /// q (heads), k and v (kv heads) stacked: one product.
    qkv: Dense,
    /// Qwen3's per-head norms on q and k.
    qk_norm: Option<(RmsNorm, RmsNorm)>,
    out: Dense,
    norm_mlp: RmsNorm,
    /// gate and up stacked: `2 * ffn` outputs.
    gate_up: Dense,
    down: Dense,
}

pub struct CausalLm {
    config: Config,
    layers: Vec<Layer>,
    norm: RmsNorm,
    /// Token embeddings.
    embed: Dense,
    /// The output projection when it is not tied to `embed`.
    head: Option<Dense>,
    rotary: Rotary,
    /// Per layer, a scale on the normed input of the MLP (Voxtral Realtime's
    /// delay-conditioned `1 + ada(t)`); none elsewhere.
    ffn_scales: Option<Vec<Vec<f32>>>,
}

/// The caches of a prompt start; see `CausalLm::prefix`.
pub struct Prefix {
    tokens: Vec<usize>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl CausalLm {
    pub fn load(files: &Files, precision: Precision, device: Device) -> Result<Self, String> {
        let device = crate::active_device(device)?;
        let config = Json::parse(&files.read("config.json")?)?;
        let t = files.tensors(precision, device)?;
        Self::from_tensors(&t, &config, ("", ""))
    }

    /// The model whose config is under `config_at` and whose tensor names
    /// start with `prefix` (`thinker.` in Qwen3-ASR).
    pub fn from_tensors(
        t: &Tensors,
        json: &Json,
        (config_at, prefix): (&str, &str),
    ) -> Result<Self, String> {
        let c = Config::parse(json, config_at)?;
        let (d, hd) = (c.d, c.head_dim);
        let layers = team::map(c.layers, |i| -> Result<Layer, String> {
            let p = format!("{prefix}model.layers.{i}");
            let a = |n: &str| format!("{p}.self_attn.{n}");
            let m = |n: &str| format!("{p}.mlp.{n}");
            Ok(Layer {
                norm_attn: RmsNorm::load(t, &format!("{p}.input_layernorm"), d, c.eps)?,
                qkv: Dense::load(
                    t,
                    &[
                        (&a("q_proj"), c.heads * hd),
                        (&a("k_proj"), c.kv_heads * hd),
                        (&a("v_proj"), c.kv_heads * hd),
                    ],
                    d,
                )?,
                qk_norm: match c.qk_norm {
                    true => Some((
                        RmsNorm::load(t, &a("q_norm"), hd, c.eps)?,
                        RmsNorm::load(t, &a("k_norm"), hd, c.eps)?,
                    )),
                    false => None,
                },
                out: linear(t, &a("o_proj"), d, c.heads * hd)?,
                norm_mlp: RmsNorm::load(t, &format!("{p}.post_attention_layernorm"), d, c.eps)?,
                gate_up: stacked(t, &[&m("gate_proj"), &m("up_proj")], c.ffn, d)?,
                down: linear(t, &m("down_proj"), d, c.ffn)?,
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        let head_name = format!("{prefix}lm_head.weight");
        let head = if c.tied || !t.has(&head_name) {
            None
        } else {
            Some(Dense::table(t, &head_name, c.vocab, d)?)
        };
        Ok(Self {
            layers,
            norm: RmsNorm::load(t, &format!("{prefix}model.norm"), d, c.eps)?,
            embed: Dense::table(t, &format!("{prefix}model.embed_tokens.weight"), c.vocab, d)?,
            head,
            rotary: Rotary::halves(hd, c.rope_base),
            ffn_scales: None,
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
        let logits = self.forward(&prompt[done..], done, &mut keys, &mut values);
        self.greedy(
            logits,
            prompt.len(),
            (&mut keys, &mut values),
            max_new,
            stop,
        )
    }

    /// Greedy tokens after a prompt given as embeddings (`rows x width`,
    /// some rows from another model, such as audio frames), until a `stop`
    /// token or `max_new` tokens.
    pub fn generate_rows(&self, rows: Buf, max_new: usize, stop: &[usize]) -> Vec<usize> {
        let c = &self.config;
        let length = rows.len() / c.d;
        let capacity = (length + max_new) * c.kv_heads * c.head_dim;
        let mut keys = vec![Vec::with_capacity(capacity); c.layers];
        let mut values = vec![Vec::with_capacity(capacity); c.layers];
        let logits = self.forward_rows(rows, 0, &mut keys, &mut values);
        self.greedy(logits, length, (&mut keys, &mut values), max_new, stop)
    }

    /// Greedy tokens from `logits`, the next position being `position`.
    fn greedy(
        &self,
        mut logits: Buf,
        position: usize,
        (keys, values): (&mut [Vec<f32>], &mut [Vec<f32>]),
        max_new: usize,
        stop: &[usize],
    ) -> Vec<usize> {
        let mut out = Vec::new();
        while out.len() < max_new {
            let token = argmax(&logits);
            if stop.contains(&token) {
                break;
            }
            out.push(token);
            logits = self.forward(&[token], position + out.len() - 1, keys, values);
        }
        out
    }

    /// Sets a scale on the normed MLP input of each layer, `layers x width`.
    pub fn set_ffn_scales(&mut self, scales: Vec<Vec<f32>>) {
        self.ffn_scales = Some(scales);
    }

    /// The number of layers.
    pub fn layers(&self) -> usize {
        self.config.layers
    }

    /// Greedy tokens with audio added to every input (Voxtral Realtime): the
    /// prompt's embeddings plus `audio` rows `0..prompt.len()`, then each new
    /// token's embedding plus the next row, until a `stop` token or the rows
    /// end. `audio` is `rows x width`.
    pub fn generate_additive(&self, prompt: &[usize], audio: &[f32], stop: &[usize]) -> Vec<usize> {
        let c = &self.config;
        let d = c.d;
        let rows = audio.len() / d;
        let n = prompt.len();
        if n == 0 || rows < n {
            return Vec::new();
        }
        let capacity = (rows + 1) * c.kv_heads * c.head_dim;
        let mut keys = vec![Vec::with_capacity(capacity); c.layers];
        let mut values = vec![Vec::with_capacity(capacity); c.layers];
        let mut x = self.embed(prompt);
        add_into(&mut x, &audio[..n * d]);
        let mut token = argmax(&self.forward_rows(x, 0, &mut keys, &mut values));
        let mut out = Vec::new();
        for position in n..rows {
            if stop.contains(&token) {
                return out;
            }
            out.push(token);
            let mut x = self.embed(&[token]);
            add_into(&mut x, &audio[position * d..(position + 1) * d]);
            token = argmax(&self.forward_rows(x, position, &mut keys, &mut values));
        }
        if !stop.contains(&token) {
            out.push(token);
        }
        out
    }

    /// The width of a token embedding.
    pub fn width(&self) -> usize {
        self.config.d
    }

    /// The embeddings of `tokens`, one row each.
    pub fn embed(&self, tokens: &[usize]) -> Buf {
        let d = self.config.d;
        let mut x = Buf::scratch(tokens.len() * d);
        for (row, &token) in x.chunks_exact_mut(d).zip(tokens) {
            self.embed.row(token, row);
        }
        x
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
        self.forward_rows(self.embed(tokens), first, keys, values)
    }

    /// `forward` from embeddings, `rows x width`.
    fn forward_rows(
        &self,
        x: Buf,
        first: usize,
        keys: &mut [Vec<f32>],
        values: &mut [Vec<f32>],
    ) -> Buf {
        let c = &self.config;
        let d = c.d;
        let m = x.len() / d;
        let x = self.hidden(x, first, (keys, values), true);
        let last = self.norm.apply(&x[(m - 1) * d..]);
        let mut logits = self.head.as_ref().unwrap_or(&self.embed).forward(&last, 1);
        scale(&mut logits, 1.0 / c.logits_scaling);
        logits
    }

    /// The best token of each row from `from` on after one pass over all
    /// `rows` with no causal mask (Granite Speech NAR's editor).
    pub fn bidirectional_tokens(&self, rows: Buf, from: usize) -> Vec<usize> {
        let c = &self.config;
        let d = c.d;
        let m = rows.len() / d;
        let mut keys = vec![Vec::with_capacity(m * c.kv_heads * c.head_dim); c.layers];
        let mut values = vec![Vec::with_capacity(m * c.kv_heads * c.head_dim); c.layers];
        let x = self.hidden(rows, 0, (&mut keys, &mut values), false);
        let normed = self.norm.apply(&x[from * d..]);
        let logits = self
            .head
            .as_ref()
            .unwrap_or(&self.embed)
            .forward(&normed, m - from);
        logits.chunks_exact(c.vocab).map(argmax).collect()
    }

    /// The hidden states after every layer, before the final norm.
    fn hidden(
        &self,
        mut x: Buf,
        first: usize,
        (keys, values): (&mut [Vec<f32>], &mut [Vec<f32>]),
        causal: bool,
    ) -> Buf {
        let c = &self.config;
        let (d, hd, m) = (c.d, c.head_dim, x.len() / c.d);
        let (qd, kv) = (c.heads * hd, c.kv_heads * hd);
        scale(&mut x, c.embedding_multiplier);
        // `attention` divides by sqrt(head_dim); Granite's multiplier replaces that.
        let q_scale = c
            .attention_multiplier
            .map_or(1.0, |multiplier| multiplier * (hd as f32).sqrt());
        for (l, layer) in self.layers.iter().enumerate() {
            let normed = layer.norm_attn.apply(&x);
            let mut qkv = layer.qkv.forward(&normed, m);
            if let Some((q_norm, k_norm)) = &layer.qk_norm {
                team::chunks_mut(&mut qkv, qd + 2 * kv, |_, row| {
                    q_norm.apply_in_place(&mut row[..qd]);
                    k_norm.apply_in_place(&mut row[qd..qd + kv]);
                });
            }
            self.rotary
                .apply(&mut qkv, (m, qd + 2 * kv), (c.heads, hd), first);
            self.rotary
                .apply(&mut qkv[qd..], (m, qd + 2 * kv), (c.kv_heads, hd), first);
            if q_scale != 1.0 {
                team::chunks_mut(&mut qkv, qd + 2 * kv, |_, row| {
                    row[..qd].iter_mut().for_each(|v| *v *= q_scale)
                });
            }
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
                causal,
            );
            add_residual(&mut x, &layer.out.forward(&att, m), c.residual_multiplier);
            let mut normed = layer.norm_mlp.apply(&x);
            if let Some(scales) = &self.ffn_scales {
                let scale = &scales[l];
                team::chunks_mut(&mut normed, d, |_, row| {
                    row.iter_mut().zip(scale).for_each(|(v, s)| *v *= s)
                });
            }
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
            add_residual(
                &mut x,
                &layer.down.forward(&hidden, m),
                c.residual_multiplier,
            );
        }
        x
    }
}

/// `x *= w`; nothing for 1, so a model without multipliers keeps its sums.
fn scale(x: &mut [f32], w: f32) {
    if w != 1.0 {
        team::chunks_mut(x, 4096, |_, values| values.iter_mut().for_each(|v| *v *= w));
    }
}

/// `x += w * y`; `add_into` for 1.
fn add_residual(x: &mut [f32], y: &[f32], w: f32) {
    if w == 1.0 {
        add_into(x, y);
        return;
    }
    team::chunks_mut(x, 4096, |i, row| {
        for (o, v) in row.iter_mut().zip(&y[i * 4096..]) {
            *o += w * v;
        }
    });
}
