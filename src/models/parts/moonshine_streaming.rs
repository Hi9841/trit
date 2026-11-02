//! The encoder of Moonshine Streaming (UsefulSensors), and the adapter that
//! feeds Moonshine's decoder: 5 ms frames of raw audio, each normalized,
//! scaled by a learned `k` and compressed with `asinh`, a linear layer with
//! SiLU, two causal stride-2 convs (20 ms frames), then pre-norm blocks whose
//! attention sees a sliding window of frames (no positions), and a final
//! norm. The adapter adds learned positions and projects to the decoder
//! width. The structure follows Transformers' `MoonshineStreamingEncoder`
//! as transcribe.cpp ports it; the norms are stored with their `+ 1`.

use crate::cpu::{activation, buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::models::config::Json;
use crate::models::layers::{add_into, conv_columns, linear, stacked, Gelu, Norm};

/// The causal convs: kernel 5, stride 2, four zeros before.
const KERNEL: usize = 5;
const STRIDE: usize = 2;

pub struct StreamConfig {
    d: usize,
    heads: usize,
    head_dim: usize,
    ffn: usize,
    layers: usize,
    frame_len: usize,
    /// Per layer: keys `left - 1` frames back and `right - 1` ahead.
    windows: Vec<(usize, usize)>,
    cmvn_eps: f32,
    projection: bool,
    max_positions: usize,
}

impl StreamConfig {
    pub fn parse(json: &Json) -> Result<Option<Self>, String> {
        if json.get("streaming_encoder").is_null() {
            return Ok(None);
        }
        let enc = |key: &str| json.num(&format!("streaming_encoder.{key}"));
        let windows = json
            .get("streaming_encoder.sliding_windows")
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|pair| {
                let pair = pair.as_array()?;
                Some((
                    pair.first()?.as_u64()? as usize,
                    pair.get(1)?.as_u64()? as usize,
                ))
            })
            .collect::<Vec<_>>();
        let config = Self {
            d: enc("hidden_size")?,
            heads: enc("num_attention_heads")?,
            head_dim: enc("head_dim")?,
            ffn: enc("intermediate_size")?,
            layers: enc("num_hidden_layers")?,
            frame_len: enc("frame_len")?,
            cmvn_eps: json.float("streaming_encoder.cmvn_eps").unwrap_or(1e-6) as f32,
            projection: json.flag("streaming_encoder.adapter_projection") == Some(true),
            max_positions: enc("max_position_embeddings")?,
            windows,
        };
        if config.windows.len() != config.layers
            || config.windows.iter().any(|(left, _)| *left == 0)
        {
            return Err("config.json: a streaming layer has no attention window".into());
        }
        Ok(Some(config))
    }
}

struct Layer {
    norm_attn: Norm,
    /// q, k, v stacked: `3 * heads * head_dim x d`.
    qkv: Dense,
    out: Dense,
    norm_ffn: Norm,
    fc1: Dense,
    fc2: Dense,
    window: (usize, usize),
}

pub struct StreamingEncoder {
    config: StreamConfig,
    /// `exp(log_k)`: the scale before `asinh`.
    k: f32,
    linear: Dense,
    conv1: Dense,
    conv2: Dense,
    layers: Vec<Layer>,
    norm: Norm,
    /// The adapter's learned positions, `max_positions x d`.
    positions: Vec<f32>,
    projection: Option<Dense>,
}

impl StreamingEncoder {
    /// `decoder_d`: the width the adapter projects to.
    pub fn load(t: &Tensors, config: StreamConfig, decoder_d: usize) -> Result<Self, String> {
        let c = &config;
        let (d, attn) = (c.d, c.heads * c.head_dim);
        let e = "model.encoder";
        let layers = (0..c.layers)
            .map(|i| -> Result<Layer, String> {
                let p = format!("{e}.layers.{i}");
                let a = |n: &str| format!("{p}.self_attn.{n}");
                Ok(Layer {
                    norm_attn: Norm::load(t, &format!("{p}.input_layernorm"), d)?,
                    qkv: stacked(t, &[&a("q_proj"), &a("k_proj"), &a("v_proj")], attn, d)?,
                    out: linear(t, &a("o_proj"), d, attn)?,
                    norm_ffn: Norm::load(t, &format!("{p}.post_attention_layernorm"), d)?,
                    fc1: linear(t, &format!("{p}.mlp.fc1"), c.ffn, d)?,
                    fc2: linear(t, &format!("{p}.mlp.fc2"), d, c.ffn)?,
                    window: c.windows[i],
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let log_k = t.f32_len(&format!("{e}.embedder.comp.log_k"), 1)?[0];
        Ok(Self {
            k: log_k.exp(),
            linear: linear(t, &format!("{e}.embedder.linear"), d, c.frame_len)?,
            conv1: linear(t, &format!("{e}.embedder.conv1"), 2 * d, d * KERNEL)?,
            conv2: linear(t, &format!("{e}.embedder.conv2"), d, 2 * d * KERNEL)?,
            layers,
            norm: Norm::load(t, &format!("{e}.layer_norm"), d)?,
            positions: t.f32_len("model.adapter.pos_emb.weight", c.max_positions * d)?,
            projection: match c.projection {
                true => Some(linear(t, "model.adapter.proj", decoder_d, d)?),
                false if d == decoder_d => None,
                false => return Err("the streaming encoder and the decoder differ in width".into()),
            },
            config,
        })
    }

    /// Samples the encoder needs for one output frame.
    pub fn min_samples(&self) -> usize {
        self.config.frame_len * STRIDE * STRIDE
    }

    /// The decoder's cross-attention input, `frames x decoder_d`.
    pub fn encode(&self, samples: &[f32]) -> (Buf, usize) {
        let c = &self.config;
        let (d, len) = (c.d, c.frame_len);
        let frames = samples.len().div_ceil(len);
        let mut x = Buf::zeroed(frames * len);
        x[..samples.len()].copy_from_slice(samples);
        team::chunks_mut(&mut x, len, |_, row| {
            let mean = row.iter().sum::<f32>() / len as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / len as f32;
            let inv = 1.0 / (var + c.cmvn_eps).sqrt();
            for v in row.iter_mut() {
                let z = (*v - mean) * inv * self.k;
                *v = (z + (z * z + 1.0).sqrt()).ln();
            }
        });
        let mut h = self.linear.forward(&x, frames);
        activation::silu(&mut h);
        let out = |n: usize| (n + KERNEL - 1 - KERNEL) / STRIDE + 1;
        let t1 = out(frames);
        let cols = conv_columns(&h, (frames, d), (KERNEL, STRIDE, KERNEL - 1), t1);
        let mut h = self.conv1.forward(&cols, t1);
        activation::silu(&mut h);
        let t = out(t1);
        let cols = conv_columns(&h, (t1, 2 * d), (KERNEL, STRIDE, KERNEL - 1), t);
        let mut x = self.conv2.forward(&cols, t);
        for layer in &self.layers {
            let qkv = layer.qkv.forward(&layer.norm_attn.apply(&x, d), t);
            let heard = self.window_attention(&qkv, t, layer.window);
            add_into(&mut x, &layer.out.forward(&heard, t));
            let mut hidden = layer.fc1.forward(&layer.norm_ffn.apply(&x, d), t);
            Gelu::Exact.apply(&mut hidden, c.ffn);
            add_into(&mut x, &layer.fc2.forward(&hidden, t));
        }
        let mut x = self.norm.apply(&x, d);
        let reach = t.min(c.max_positions);
        team::chunks_mut(&mut x[..reach * d], d, |i, row| {
            add_into(row, &self.positions[i * d..(i + 1) * d]);
        });
        match &self.projection {
            Some(projection) => (projection.forward(&x, t), t),
            None => (x, t),
        }
    }

    /// Attention of each frame over keys `left - 1` back to `right - 1`
    /// ahead (itself when `right` is 0). `qkv` rows are q, k, v.
    fn window_attention(&self, qkv: &[f32], t: usize, (left, right): (usize, usize)) -> Buf {
        let c = &self.config;
        let (heads, hd) = (c.heads, c.head_dim);
        let attn = heads * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = Buf::zeroed(t * attn);
        team::chunks_mut(&mut out, attn, |i, row| {
            let keys = i.saturating_sub(left - 1)..(i + right.max(1)).min(t);
            let mut scores = vec![0.0f32; keys.len()];
            for h in 0..heads {
                let q = &qkv[i * 3 * attn + h * hd..][..hd];
                for (s, j) in scores.iter_mut().zip(keys.clone()) {
                    let k = &qkv[j * 3 * attn + attn + h * hd..][..hd];
                    *s = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let o = &mut row[h * hd..(h + 1) * hd];
                for (s, j) in scores.iter().zip(keys.clone()) {
                    let v = &qkv[j * 3 * attn + 2 * attn + h * hd..][..hd];
                    let w = s / sum;
                    for (o, v) in o.iter_mut().zip(v) {
                        *o += w * v;
                    }
                }
            }
        });
        out
    }
}
