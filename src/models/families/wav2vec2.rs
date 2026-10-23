//! wav2vec 2.0 with a CTC head (Hugging Face `Wav2Vec2ForCTC`): a conv
//! feature encoder over raw audio, a transformer
//! encoder with a convolutional position embedding, and one projection to
//! characters per 20 ms frame, decoded greedily. Both layouts run: post-norm
//! (`feat_extract_norm: group`, the base models) and pre-norm
//! (`do_stable_layer_norm`, the large and XLS-R models).

use crate::audio;
use crate::cpu::{buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::models::config::Json;
use crate::models::layers::{add_into, attention, conv_columns, linear, stacked, Gelu, Norm};
use crate::Precision;
use std::path::Path;

/// Longest window per pass: attention grows with the square of the frames
/// (50 per second).
const WINDOW_SAMPLES: usize = 40 * 16_000;
/// Epsilon of the input normalization (`zero_mean_unit_var_norm`).
const INPUT_EPS: f32 = 1e-7;
/// Epsilon of the group norm of the first conv layer.
const GROUP_NORM_EPS: f32 = 1e-5;

struct Config {
    d: usize,
    heads: usize,
    ffn: usize,
    layers: usize,
    vocab: usize,
    blank: usize,
    /// `(channels, kernel, stride)` per conv layer.
    convs: Vec<(usize, usize, usize)>,
    /// Every conv layer has a layer norm (`layer`), or only the first a
    /// group norm (`group`).
    layer_norm_convs: bool,
    /// Pre-norm layers and a final norm (`do_stable_layer_norm`).
    pre_norm: bool,
    pos_kernel: usize,
    pos_groups: usize,
    gelu: Gelu,
}

impl Config {
    fn parse(json: &Json) -> Result<Self, String> {
        let list = |key: &str| -> Result<Vec<usize>, String> {
            json.get(key)
                .as_array()
                .ok_or_else(|| format!("config.json has no {key}"))?
                .iter()
                .map(|v| v.as_u64().map(|v| v as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| format!("config.json has a bad {key}"))
        };
        let (dims, kernels, strides) = (
            list("conv_dim")?,
            list("conv_kernel")?,
            list("conv_stride")?,
        );
        let d = json.num("hidden_size")?;
        let heads = json.num("num_attention_heads")?;
        let pos_groups = json.num("num_conv_pos_embedding_groups")?;
        let gelu = Gelu::parse(json.str("hidden_act").unwrap_or("gelu"))?;
        if dims.is_empty()
            || dims.len() != kernels.len()
            || dims.len() != strides.len()
            || !d.is_multiple_of(heads)
            || !d.is_multiple_of(pos_groups)
            || json.str("feat_extract_activation").unwrap_or("gelu") != "gelu"
            || json
                .str("position_embeddings_type")
                .is_some_and(|p| p != "absolute")
        {
            return Err(
                "config.json describes a wav2vec 2.0 shape this engine does not run".into(),
            );
        }
        Ok(Self {
            d,
            heads,
            ffn: json.num("intermediate_size")?,
            layers: json.num("num_hidden_layers")?,
            vocab: json.num("vocab_size")?,
            blank: json.num("pad_token_id")?,
            convs: dims
                .into_iter()
                .zip(kernels)
                .zip(strides)
                .map(|((c, k), s)| (c, k, s))
                .collect(),
            layer_norm_convs: json.str("feat_extract_norm") == Some("layer"),
            pre_norm: json.flag("do_stable_layer_norm") == Some(true),
            pos_kernel: json.num("num_conv_pos_embeddings")?,
            pos_groups,
            gelu,
        })
    }
}

struct Conv {
    weight: Dense,
    kernel: usize,
    stride: usize,
    norm: Option<ConvNorm>,
}

enum ConvNorm {
    /// One group per channel: each channel normalized over time.
    Group { weight: Vec<f32>, bias: Vec<f32> },
    /// Each frame normalized over its channels.
    Layer(Norm),
}

struct Layer {
    norm_attn: Norm,
    qkv: Dense,
    out: Dense,
    norm_ffn: Norm,
    fc1: Dense,
    fc2: Dense,
}

pub struct Wav2Vec2 {
    config: Config,
    convs: Vec<Conv>,
    /// The feature projection's layer norm.
    feature_norm: Option<Norm>,
    projection: Dense,
    /// The position conv, one layer per group: `d / groups` channels in
    /// and out, `kernel` taps.
    position: Vec<Dense>,
    encoder_norm: Norm,
    layers: Vec<Layer>,
    head: Dense,
    /// Text per token id; special tokens are empty, the word delimiter a space.
    tokens: Vec<String>,
    /// Scale the input to zero mean and unit variance (`do_normalize`).
    normalize: bool,
}

impl Wav2Vec2 {
    pub fn load(dir: &Path, json: &Json, precision: Precision) -> Result<Self, String> {
        let c = Config::parse(json)?;
        let t = Tensors::open_dir(dir, precision)?;
        let p = "wav2vec2";
        let d = c.d;
        let mut convs = Vec::with_capacity(c.convs.len());
        let mut channels = 1;
        for (i, &(out, kernel, stride)) in c.convs.iter().enumerate() {
            let name = format!("{p}.feature_extractor.conv_layers.{i}");
            let norm = format!("{name}.layer_norm");
            let norm = match (c.layer_norm_convs, i) {
                (true, _) => Some(ConvNorm::Layer(Norm::load(&t, &norm, out)?)),
                (false, 0) => Some(ConvNorm::Group {
                    weight: t.f32_len(&format!("{norm}.weight"), out)?,
                    bias: t.f32_len(&format!("{norm}.bias"), out)?,
                }),
                (false, _) => None,
            };
            convs.push(Conv {
                weight: linear(&t, &format!("{name}.conv"), out, channels * kernel)?,
                kernel,
                stride,
                norm,
            });
            channels = out;
        }
        let feature_norm = format!("{p}.feature_projection.layer_norm");
        let feature_norm = match t.has(&format!("{feature_norm}.weight")) {
            true => Some(Norm::load(&t, &feature_norm, channels)?),
            false => None,
        };
        let layers = team::map(c.layers, |i| -> Result<Layer, String> {
            let l = format!("{p}.encoder.layers.{i}");
            let a = |n: &str| format!("{l}.attention.{n}");
            Ok(Layer {
                norm_attn: Norm::load(&t, &format!("{l}.layer_norm"), d)?,
                qkv: stacked(&t, &[&a("q_proj"), &a("k_proj"), &a("v_proj")], d, d)?,
                out: linear(&t, &a("out_proj"), d, d)?,
                norm_ffn: Norm::load(&t, &format!("{l}.final_layer_norm"), d)?,
                fc1: linear(
                    &t,
                    &format!("{l}.feed_forward.intermediate_dense"),
                    c.ffn,
                    d,
                )?,
                fc2: linear(&t, &format!("{l}.feed_forward.output_dense"), d, c.ffn)?,
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        let normalize = std::fs::read_to_string(dir.join("preprocessor_config.json"))
            .ok()
            .and_then(|text| Json::parse(&text).ok())
            .and_then(|json| json.flag("do_normalize"))
            .unwrap_or(true);
        Ok(Self {
            projection: linear(
                &t,
                &format!("{p}.feature_projection.projection"),
                d,
                channels,
            )?,
            feature_norm,
            position: position_conv(&t, &format!("{p}.encoder.pos_conv_embed.conv"), &c)?,
            encoder_norm: Norm::load(&t, &format!("{p}.encoder.layer_norm"), d)?,
            layers,
            head: linear(&t, "lm_head", c.vocab, d)?,
            tokens: vocabulary(dir, c.vocab)?,
            normalize,
            convs,
            config: c,
        })
    }

    /// The text of each window that holds speech: `(first sample, samples, text)`.
    /// The model writes one language (its fine-tuning's); options change nothing.
    pub fn pieces(&self, samples: &[f32]) -> Vec<(usize, usize, String)> {
        audio::windows(samples, WINDOW_SAMPLES)
            .into_iter()
            .filter(|window| !audio::is_silent(window))
            .filter_map(|window| {
                let text = self.transcribe_window(window);
                (!text.is_empty()).then(|| (audio::offset(samples, window), window.len(), text))
            })
            .collect()
    }

    fn transcribe_window(&self, samples: &[f32]) -> String {
        let Some((features, t)) = self.features(samples) else {
            return String::new();
        };
        let encoded = self.encode(&features, t);
        let logits = self.head.forward(&encoded, t);
        let (ids, _) =
            crate::models::parts::ctc::greedy(&logits, self.config.vocab, self.config.blank, &[]);
        let text: String = ids.iter().map(|&id| self.tokens[id].as_str()).collect();
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// The conv feature encoder: `frames x channels` of the last conv, or
    /// `None` when the audio is shorter than one frame.
    fn features(&self, samples: &[f32]) -> Option<(Buf, usize)> {
        let mut x = Buf::scratch(samples.len());
        x.copy_from_slice(samples);
        if self.normalize {
            let n = x.len() as f32;
            let mean = x.iter().sum::<f32>() / n;
            let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let inv = 1.0 / (var + INPUT_EPS).sqrt();
            x.iter_mut().for_each(|v| *v = (*v - mean) * inv);
        }
        let (mut frames, mut channels) = (samples.len(), 1);
        for conv in &self.convs {
            let out_frames = frames.checked_sub(conv.kernel)? / conv.stride + 1;
            let cols = conv_columns(
                &x,
                (frames, channels),
                (conv.kernel, conv.stride, 0),
                out_frames,
            );
            x = conv.weight.forward(&cols, out_frames);
            channels = conv.weight.out_features;
            frames = out_frames;
            match &conv.norm {
                Some(ConvNorm::Layer(norm)) => x = norm.apply(&x, channels),
                Some(ConvNorm::Group { weight, bias }) => {
                    group_norm(&mut x, channels, weight, bias)
                }
                None => {}
            }
            Gelu::Exact.apply(&mut x, channels);
        }
        Some((x, frames))
    }

    /// The transformer encoder over `t` frames of conv features: `t x d`.
    fn encode(&self, features: &[f32], t: usize) -> Buf {
        let c = &self.config;
        let d = c.d;
        let features = match &self.feature_norm {
            Some(norm) => norm.apply(features, features.len() / t),
            None => {
                let mut copy = Buf::scratch(features.len());
                copy.copy_from_slice(features);
                copy
            }
        };
        let mut x = self.projection.forward(&features, t);
        let position = self.position_embedding(&x, t);
        add_into(&mut x, &position);
        if !c.pre_norm {
            x = self.encoder_norm.apply(&x, d);
        }
        for layer in &self.layers {
            let attend = |input: &[f32]| {
                let qkv = layer.qkv.forward(input, t);
                let att = attention(
                    (&qkv, 3 * d),
                    (&qkv[d..], 3 * d),
                    (&qkv[2 * d..], 3 * d),
                    (t, t),
                    (d, c.heads, c.heads),
                    false,
                );
                layer.out.forward(&att, t)
            };
            let feed = |input: &[f32]| {
                let mut hidden = layer.fc1.forward(input, t);
                c.gelu.apply(&mut hidden, c.ffn);
                layer.fc2.forward(&hidden, t)
            };
            if c.pre_norm {
                let att = attend(&layer.norm_attn.apply(&x, d));
                add_into(&mut x, &att);
                let ffn = feed(&layer.norm_ffn.apply(&x, d));
                add_into(&mut x, &ffn);
            } else {
                let att = attend(&x);
                x = layer.norm_attn.residual(&mut x, &att, 1.0);
                let ffn = feed(&x);
                x = layer.norm_ffn.residual(&mut x, &ffn, 1.0);
            }
        }
        if c.pre_norm {
            x = self.encoder_norm.apply(&x, d);
        }
        x
    }

    /// GELU of the grouped position conv over `x` (`t x d`), padded by half
    /// the kernel on each side; an even kernel's extra last frame is dropped.
    fn position_embedding(&self, x: &[f32], t: usize) -> Buf {
        let c = &self.config;
        let width = c.d / c.pos_groups;
        let mut out = Buf::scratch(t * c.d);
        for (g, conv) in self.position.iter().enumerate() {
            let mut group = Buf::scratch(t * width);
            team::chunks_mut(&mut group, width, |row, values| {
                values.copy_from_slice(&x[row * c.d + g * width..][..width]);
            });
            let cols = conv_columns(&group, (t, width), (c.pos_kernel, 1, c.pos_kernel / 2), t);
            let y = conv.forward(&cols, t);
            team::chunks_mut(&mut out, c.d, |row, values| {
                values[g * width..(g + 1) * width].copy_from_slice(&y[row * width..][..width]);
            });
        }
        c.gelu.apply(&mut out, c.d);
        out
    }
}

/// The position conv's weight from its weight norm, `g * v / |v|` with one
/// norm per tap (`weight_norm(dim=2)`), split into one layer per group.
fn position_conv(t: &Tensors, name: &str, c: &Config) -> Result<Vec<Dense>, String> {
    let (k, width) = (c.pos_kernel, c.d / c.pos_groups);
    // Older files: `weight_g`, `weight_v`; newer: the parametrization's originals.
    let (g, v) = match t.has(&format!("{name}.weight_g")) {
        true => ("weight_g", "weight_v"),
        false => (
            "parametrizations.weight.original0",
            "parametrizations.weight.original1",
        ),
    };
    let g = t.f32_len(&format!("{name}.{g}"), k)?;
    let mut v = t.f32_len(&format!("{name}.{v}"), c.d * width * k)?;
    let bias = t.f32_len(&format!("{name}.bias"), c.d)?;
    for (tap, gain) in g.iter().enumerate() {
        let norm = v
            .iter()
            .skip(tap)
            .step_by(k)
            .map(|w| w * w)
            .sum::<f32>()
            .sqrt();
        let scale = gain / norm;
        v.iter_mut().skip(tap).step_by(k).for_each(|w| *w *= scale);
    }
    // `v` is `[out][in of the group][tap]`: group `i` owns outputs `i * width..`.
    v.chunks_exact(width * width * k)
        .zip(bias.chunks_exact(width))
        .map(|(weight, bias)| Dense::new(weight.to_vec(), Some(bias.to_vec()), width, width * k))
        .collect()
}

/// Each of `channels` columns of `x` normalized over its rows (frames).
fn group_norm(x: &mut [f32], channels: usize, weight: &[f32], bias: &[f32]) {
    let frames = (x.len() / channels) as f32;
    let mut mean = vec![0.0f32; channels];
    let mut square = vec![0.0f32; channels];
    for row in x.chunks_exact(channels) {
        for ((m, s), v) in mean.iter_mut().zip(square.iter_mut()).zip(row) {
            *m += v;
            *s += v * v;
        }
    }
    let scale: Vec<(f32, f32)> = mean
        .iter()
        .zip(&square)
        .zip(weight.iter().zip(bias))
        .map(|((m, s), (w, b))| {
            let mean = m / frames;
            let var = (s / frames - mean * mean).max(0.0);
            let inv = w / (var + GROUP_NORM_EPS).sqrt();
            (inv, b - mean * inv)
        })
        .collect();
    team::chunks_mut(x, channels, |_, row| {
        for (v, (a, b)) in row.iter_mut().zip(&scale) {
            *v = *v * a + b;
        }
    });
}

/// The text of each token id from `vocab.json`: the word delimiter (`|`,
/// or `word_delimiter_token` of `tokenizer_config.json`) is a space, and
/// special tokens (`<pad>`, `<s>`, `<unk>`) are empty.
fn vocabulary(dir: &Path, size: usize) -> Result<Vec<String>, String> {
    let path = dir.join("vocab.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("vocab.json is not JSON: {e}"))?;
    let delimiter = std::fs::read_to_string(dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|text| Json::parse(&text).ok())
        .and_then(|json| json.str("word_delimiter_token").map(str::to_string))
        .unwrap_or_else(|| "|".to_string());
    let mut tokens = vec![String::new(); size];
    for (token, id) in json.as_object().ok_or("vocab.json is not a map")? {
        let Some(slot) = id.as_u64().and_then(|id| tokens.get_mut(id as usize)) else {
            continue;
        };
        let special = token.len() > 2 && token.starts_with('<') && token.ends_with('>');
        *slot = match () {
            _ if *token == delimiter => " ".to_string(),
            _ if special => String::new(),
            _ => token.clone(),
        };
    }
    Ok(tokens)
}
