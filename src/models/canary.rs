//! Canary (NVIDIA Canary 1B v2): the Parakeet FastConformer encoder and a
//! transformer decoder that writes the text after a prompt of task tokens.
//! Tensor names follow the Hugging Face `CanaryForConditionalGeneration`
//! files; the prompt follows the model's `chat_template.jinja`.

use crate::audio::mel::MelFrontend;
use crate::cpu::{buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::models::parakeet::bias::{self, Bias, BiasWeights};
use crate::models::parakeet::model::{Config as EncoderConfig, Parakeet as Encoder};
use crate::models::transformer::{
    add_into, argmax, attention, attention_with, linear, stacked, HeadMajor, Norm,
};
use crate::text::sentencepiece::Tokenizer;
use crate::{Options, Precision};
use parking_lot::RwLock;
use std::path::Path;

/// Languages Canary 1B v2 reads and writes.
const LANGUAGES: [&str; 25] = [
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt",
    "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
];

struct Config {
    d: usize,
    ffn: usize,
    heads: usize,
    layers: usize,
    vocab: usize,
    max_tokens: usize,
    eos: usize,
}

impl Config {
    fn parse(text: &str) -> Result<Self, String> {
        let json: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("config.json is not JSON: {e}"))?;
        let dec = &json["decoder_config"];
        let num = |value: &serde_json::Value, key: &str| -> Result<usize, String> {
            value[key]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("config.json has no decoder_config.{key}"))
        };
        let heads = num(dec, "num_attention_heads")?;
        if num(dec, "num_key_value_heads")? != heads || dec["hidden_act"].as_str() != Some("relu") {
            return Err("config.json describes a Canary decoder this engine does not run".into());
        }
        Ok(Self {
            d: num(dec, "hidden_size")?,
            ffn: num(dec, "intermediate_size")?,
            heads,
            layers: num(dec, "num_hidden_layers")?,
            vocab: num(dec, "vocab_size")?,
            max_tokens: num(dec, "max_position_embeddings")?,
            eos: num(dec, "eos_token_id")?,
        })
    }
}

struct Layer {
    norm_self: Norm,
    self_qkv: Dense,
    self_out: Dense,
    norm_cross: Norm,
    cross_q: Dense,
    cross_kv: Dense,
    cross_out: Dense,
    norm_mlp: Norm,
    fc1: Dense,
    fc2: Dense,
}

pub struct Canary {
    config: Config,
    encoder: Encoder,
    frontend: MelFrontend,
    layers: Vec<Layer>,
    embed: Dense,
    embed_norm: Norm,
    norm: Norm,
    head: Dense,
    /// Sinusoidal positions, `max_tokens x d`, already divided by `sqrt(d)`.
    positions: Vec<f32>,
    tokenizer: Tokenizer,
    dictionary: RwLock<Option<Bias>>,
}

const WINDOW_SAMPLES: usize = 40 * 16_000;

/// Self-attention keys and values per layer: the state of one beam.
type Caches = (Vec<Vec<f32>>, Vec<Vec<f32>>);

impl Canary {
    pub fn load(dir: &Path, config_text: &str, precision: Precision) -> Result<Self, String> {
        let c = Config::parse(config_text)?;
        let encoder_config = EncoderConfig::parse(config_text)?;
        let t = Tensors::open_with(&dir.join("model.safetensors"), precision)?;
        let frontend = MelFrontend::new(encoder_config.mels);
        let encoder = Encoder::load(&t, encoder_config)?;
        encoder.warm_positions();
        let d = c.d;
        let layers = (0..c.layers)
            .map(|i| -> Result<Layer, String> {
                let p = format!("model.decoder.layers.{i}");
                let own = |n: &str| format!("{p}.self_attn.{n}");
                let cross = |n: &str| format!("{p}.encoder_attn.{n}");
                Ok(Layer {
                    norm_self: Norm::load(&t, &format!("{p}.input_layernorm"), d)?,
                    self_qkv: stacked(&t, &[&own("q_proj"), &own("k_proj"), &own("v_proj")], d, d)?,
                    self_out: linear(&t, &own("o_proj"), d, d)?,
                    norm_cross: Norm::load(&t, &format!("{p}.post_attention_layernorm"), d)?,
                    cross_q: linear(&t, &cross("q_proj"), d, d)?,
                    cross_kv: stacked(&t, &[&cross("k_proj"), &cross("v_proj")], d, d)?,
                    cross_out: linear(&t, &cross("o_proj"), d, d)?,
                    norm_mlp: Norm::load(&t, &format!("{p}.final_layernorm"), d)?,
                    fc1: linear(&t, &format!("{p}.mlp.fc1"), c.ffn, d)?,
                    fc2: linear(&t, &format!("{p}.mlp.fc2"), d, c.ffn)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            encoder,
            frontend,
            layers,
            embed: Dense::table(&t, "model.decoder.embed_tokens.weight", c.vocab, d)?,
            embed_norm: Norm::load(&t, "model.decoder.embedding_layernorm", d)?,
            norm: Norm::load(&t, "model.decoder.norm", d)?,
            head: linear(&t, "proj_out", c.vocab, d)?,
            positions: sinusoids(c.max_tokens, d),
            tokenizer: Tokenizer::open(&dir.join("tokenizer.json"))?,
            dictionary: RwLock::new(None),
            config: c,
        })
    }

    /// Terms to favor while decoding; with terms set, decoding is greedy (the
    /// bonuses follow one sequence). See `parakeet::bias`.
    pub fn set_dictionary(&self, terms: &[String], weights: BiasWeights) -> Vec<String> {
        let (bias, skipped) = bias::build(&self.tokenizer, terms, weights);
        *self.dictionary.write() = bias;
        skipped
    }

    /// `options.language` is the spoken language (one of 25; English when
    /// none is given: Canary cannot detect it); Canary writes the text in
    /// the same language, or in English for `Task::Translate`.
    /// Recordings longer than 40 s (the longest Canary was trained on) go
    /// through in windows cut at quiet points.
    pub fn pieces(&self, samples: &[f32], options: &Options) -> Vec<(usize, usize, String)> {
        crate::audio::windows(samples, WINDOW_SAMPLES)
            .into_iter()
            .map(|window| {
                let text = self.transcribe_window(window, options);
                (crate::audio::offset(samples, window), window.len(), text)
            })
            .filter(|(_, _, text)| !text.is_empty())
            .collect()
    }

    fn transcribe_window(&self, samples: &[f32], options: &Options) -> String {
        let (features, frames) = self.frontend.compute(samples);
        if frames < 8 {
            return String::new();
        }
        let (encoded, t) = self.encoder.encode(&features, frames);
        let language = options
            .language
            .as_deref()
            .filter(|code| LANGUAGES.contains(code))
            .unwrap_or("en");
        let lang = format!("<|{language}|>");
        // The second language token is the language to write.
        let target = match options.task {
            crate::Task::Transcribe => lang.clone(),
            crate::Task::Translate => "<|en|>".to_string(),
        };
        let prompt: Option<Vec<usize>> = [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            &lang,
            &target,
            "<|pnc|>",
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ]
        .iter()
        .map(|token| self.tokenizer.id(token))
        .collect();
        let Some(prompt) = prompt else {
            return String::new();
        };
        let ids = self.generate(
            &encoded,
            t,
            &prompt,
            options.beams(1),
            self.dictionary.read().as_ref(),
        );
        self.tokenizer.decode(&ids).trim().to_string()
    }

    /// Greedy continuation of `prompt` until end of text.
    fn generate(
        &self,
        encoded: &[f32],
        frames: usize,
        prompt: &[usize],
        beam: usize,
        dictionary: Option<&Bias>,
    ) -> Vec<usize> {
        let c = &self.config;
        let d = c.d;
        // Cross-attention keys and values, laid out per head once.
        let cross: Vec<HeadMajor> = self
            .layers
            .iter()
            .map(|layer| {
                let kv = layer.cross_kv.forward(encoded, frames);
                HeadMajor::new(
                    (&kv, 2 * d),
                    (&kv[d..], 2 * d),
                    frames,
                    (d / c.heads, c.heads),
                )
            })
            .collect();
        // Room for every token, so the caches never move while they grow.
        let mut keys = vec![Vec::with_capacity(c.max_tokens * d); c.layers];
        let mut values = vec![Vec::with_capacity(c.max_tokens * d); c.layers];
        let mut logits = self.forward(prompt, 0, &cross, (&mut keys, &mut values));
        // The dictionary works on greedy decoding; with one, decode greedy.
        if beam > 1 && dictionary.is_none() {
            // A beam's state: its caches; their length is its position.
            // One beam at a time: Canary is not in a beam-search hot path.
            let step = |states: &mut [Caches], tokens: &[usize]| {
                states
                    .iter_mut()
                    .zip(tokens)
                    .map(|((keys, values), &token)| {
                        let position = keys[0].len() / d;
                        self.forward(&[token], position, &cross, (keys, values))
                    })
                    .collect()
            };
            let limit = c.max_tokens.saturating_sub(prompt.len());
            return crate::models::beam::search(
                (beam, crate::models::beam::Rules::OpenAi),
                ((keys, values), logits),
                (limit, c.eos),
                |_, _| {},
                step,
            )
            .tokens;
        }
        let mut out = Vec::new();
        let mut run = dictionary.map(bias::Run::new);
        let mut position = prompt.len();
        while position < c.max_tokens {
            let token = match &mut run {
                Some(run) => run.pick(&mut logits),
                None => argmax(&logits),
            };
            if token == c.eos {
                break;
            }
            match &mut run {
                Some(run) => run.push(token),
                None => out.push(token),
            }
            logits = self.forward(&[token], position, &cross, (&mut keys, &mut values));
            position += 1;
        }
        run.map_or(out, bias::Run::finish)
    }

    /// Runs `tokens` at positions `first..` and returns the logits after the last.
    fn forward(
        &self,
        tokens: &[usize],
        first: usize,
        cross: &[HeadMajor],
        (keys, values): (&mut [Vec<f32>], &mut [Vec<f32>]),
    ) -> Buf {
        let c = &self.config;
        let (d, m) = (c.d, tokens.len());
        let mut x = Buf::scratch(m * d);
        for (r, (row, &token)) in x.chunks_exact_mut(d).zip(tokens).enumerate() {
            self.embed.row(token, row);
            add_into(row, &self.positions[(first + r) * d..][..d]);
        }
        let mut x = self.embed_norm.apply(&x, d);
        for (l, layer) in self.layers.iter().enumerate() {
            let normed = layer.norm_self.apply(&x, d);
            let qkv = layer.self_qkv.forward(&normed, m);
            for row in qkv.chunks_exact(3 * d) {
                keys[l].extend_from_slice(&row[d..2 * d]);
                values[l].extend_from_slice(&row[2 * d..]);
            }
            let att = attention(
                (&qkv, 3 * d),
                (&keys[l], d),
                (&values[l], d),
                (m, first + m),
                (d, c.heads, c.heads),
                true,
            );
            add_into(&mut x, &layer.self_out.forward(&att, m));
            let normed = layer.norm_cross.apply(&x, d);
            let q = layer.cross_q.forward(&normed, m);
            let att = attention_with((&q, d), &cross[l], m, (d, c.heads), false);
            add_into(&mut x, &layer.cross_out.forward(&att, m));
            let normed = layer.norm_mlp.apply(&x, d);
            let mut hidden = layer.fc1.forward(&normed, m);
            team::chunks_mut(&mut hidden, c.ffn, |_, row| {
                row.iter_mut().for_each(|v| *v = v.max(0.0))
            });
            add_into(&mut x, &layer.fc2.forward(&hidden, m));
        }
        let last = self.norm.apply(&x[(m - 1) * d..], d);
        self.head.forward(&last, 1)
    }
}

/// NeMo's `FixedPositionalEncoding`: `[sin | cos]` of position times
/// `exp(-i * ln(10000^((d-2)/d)) / (d/2 - 1))`, divided by `sqrt(d)`.
fn sinusoids(length: usize, d: usize) -> Vec<f32> {
    let half = d / 2;
    let max_timescale = 10_000f64.powf((d as f64 - 2.0) / d as f64);
    let increment = max_timescale.ln() / (half as f64 - 1.0);
    let scale = 1.0 / (d as f64).sqrt();
    let mut table = vec![0.0f32; length * d];
    for pos in 0..length {
        for i in 0..half {
            let angle = pos as f64 * (-increment * i as f64).exp();
            table[pos * d + i] = (angle.sin() * scale) as f32;
            table[pos * d + half + i] = (angle.cos() * scale) as f32;
        }
    }
    table
}
