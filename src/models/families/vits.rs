//! VITS (Transformers `VitsModel`, the MMS-TTS checkpoints): text to
//! speech. A transformer reads the characters, a stochastic duration
//! predictor gives each character its frames, a normalizing flow turns the
//! noisy prior of each frame into a spectrogram latent, and a HiFi-GAN
//! writes the waveform. Inference only: the posterior encoder of training
//! is not loaded.

use crate::cpu::{activation, buf::Buf, team, tensors::Tensors};
use crate::models::config::Json;
use crate::models::layers::{Gelu, Norm};
use crate::models::parts::conv::{Conv1d, Depthwise, Upsample};
use crate::Precision;
use std::cell::RefCell;
use std::path::Path;

/// The leaky ReLU before the last convolution of HiFi-GAN: PyTorch's
/// default slope (the reference calls `leaky_relu` without one there).
const FINAL_SLOPE: f32 = 0.01;
/// Bounds of the rational-quadratic spline bins (the reference's defaults).
const MIN_BIN: f32 = 1e-3;

struct Config {
    hidden: usize,
    heads: usize,
    layers: usize,
    window: usize,
    ffn: usize,
    ffn_kernel: usize,
    flow: usize,
    vocab: usize,
}

/// How one text is spoken.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Durations are divided by it.
    pub speed: f32,
    /// Scale of the prior noise.
    pub noise: f32,
    /// Scale of the duration noise.
    pub noise_duration: f32,
}

/// Standard normal values from a seed: SplitMix64, then Box-Muller. One
/// seed always gives the same values, so one text gives the same audio.
pub struct Noise {
    state: u64,
    spare: Option<f32>,
}

impl Noise {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f32 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform();
        self.spare = Some((radius * angle.sin()) as f32);
        (radius * angle.cos()) as f32
    }
}

pub struct Vits {
    config: Config,
    embed: Vec<f32>,
    encoder: Vec<EncoderLayer>,
    project: Conv1d,
    durations: DurationPredictor,
    flows: Vec<Coupling>,
    decoder: HifiGan,
    /// The defaults of `config.json`.
    pub defaults: Settings,
    pub sample_rate: u32,
}

impl Vits {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("cannot read {}: {e}", dir.join("config.json").display()))?;
        let c = Json::parse(&text)?;
        match c.str("model_type") {
            Some("vits") => {}
            other => {
                return Err(format!(
                    "{} holds a `{}` model; a voice is a `vits` model (MMS-TTS)",
                    dir.display(),
                    other.unwrap_or("unnamed")
                ))
            }
        }
        if c.num("num_speakers").unwrap_or(1) > 1 {
            return Err(
                "this VITS model has several speakers; Trit runs one-speaker models".into(),
            );
        }
        if c.flag("use_stochastic_duration_prediction") == Some(false) {
            return Err("this VITS model has a plain duration predictor; Trit runs the stochastic one (MMS-TTS)".into());
        }
        if c.num("depth_separable_channels")? != 2 {
            return Err("config.json: depth_separable_channels must be 2".into());
        }
        let config = Config {
            hidden: c.num("hidden_size")?,
            heads: c.num("num_attention_heads")?,
            layers: c.num("num_hidden_layers")?,
            window: c.num("window_size")?,
            ffn: c.num("ffn_dim")?,
            ffn_kernel: c.num("ffn_kernel_size")?,
            flow: c.num("flow_size")?,
            vocab: c.num("vocab_size")?,
        };
        if config.ffn_kernel.is_multiple_of(2) || c.str("hidden_act").is_some_and(|a| a != "relu") {
            return Err("config.json: Trit runs VITS with an odd ffn_kernel_size and relu".into());
        }
        let t = Tensors::open_dir(dir, Precision::F32)?;
        let d = config.hidden;
        let embed = t.f32_len("text_encoder.embed_tokens.weight", config.vocab * d)?;
        let encoder = (0..config.layers)
            .map(|i| EncoderLayer::load(&t, &format!("text_encoder.encoder.layers.{i}"), &config))
            .collect::<Result<Vec<_>, _>>()?;
        let project = Conv1d::linear(&t, "text_encoder.project", d, 2 * config.flow)?;
        let durations = DurationPredictor::load(&t, &c, d)?;
        let flows = (0..c.num("prior_encoder_num_flows")?)
            .map(|i| Coupling::load(&t, &format!("flow.flows.{i}"), &c, &config))
            .collect::<Result<Vec<_>, _>>()?;
        let decoder = HifiGan::load(&t, &c, config.flow)?;
        let float = |key: &str, default: f64| c.float(key).unwrap_or(default) as f32;
        Ok(Self {
            defaults: Settings {
                speed: float("speaking_rate", 1.0),
                noise: float("noise_scale", 0.667),
                noise_duration: float("noise_scale_duration", 0.8),
            },
            sample_rate: c.num("sampling_rate").unwrap_or(16_000) as u32,
            config,
            embed,
            encoder,
            project,
            durations,
            flows,
            decoder,
        })
    }

    /// The waveform of token ids (from `Characters::encode`).
    pub fn speak(&self, ids: &[usize], settings: &Settings, noise: &mut Noise) -> Vec<f32> {
        let (d, flow) = (self.config.hidden, self.config.flow);
        let tokens = ids.len();
        if tokens == 0 {
            return Vec::new();
        }
        // Text encoder.
        let scale = (d as f32).sqrt();
        let mut h = Buf::scratch(tokens * d);
        for (row, id) in h.chunks_exact_mut(d).zip(ids) {
            let id = (*id).min(self.config.vocab - 1);
            for (o, e) in row.iter_mut().zip(&self.embed[id * d..(id + 1) * d]) {
                *o = e * scale;
            }
        }
        for layer in &self.encoder {
            h = layer.forward(h, tokens, &self.config);
        }
        let stats = self.project.forward(&h, tokens, None);

        // Durations: frames per token.
        let log_durations = self
            .durations
            .reverse(&h, tokens, settings.noise_duration, noise);
        let length_scale = 1.0 / settings.speed;
        let repeats: Vec<usize> = log_durations
            .iter()
            .map(|w| (w.exp() * length_scale).ceil().max(0.0) as usize)
            .collect();
        let frames = repeats.iter().sum::<usize>().max(1);

        // The prior of each frame, sampled, then the flow in reverse.
        let mut z = Buf::zeroed(frames * flow);
        let mut row_at = 0;
        for (token, repeat) in repeats.iter().enumerate() {
            let means = &stats[token * 2 * flow..][..flow];
            let log_variances = &stats[token * 2 * flow + flow..][..flow];
            for _ in 0..*repeat {
                let row = &mut z[row_at * flow..(row_at + 1) * flow];
                for ((o, m), lv) in row.iter_mut().zip(means).zip(log_variances) {
                    *o = m + noise.normal() * lv.exp() * settings.noise;
                }
                row_at += 1;
            }
        }
        for coupling in self.flows.iter().rev() {
            z.chunks_exact_mut(flow).for_each(<[f32]>::reverse);
            coupling.reverse(&mut z, frames);
        }
        self.decoder.forward(&z, frames)
    }
}

struct EncoderLayer {
    q: Conv1d,
    k: Conv1d,
    v: Conv1d,
    out: Conv1d,
    /// `[2 * window + 1][head dim]`, shared by the heads.
    rel_k: Vec<f32>,
    rel_v: Vec<f32>,
    norm: Norm,
    ffn_in: Conv1d,
    ffn_out: Conv1d,
    final_norm: Norm,
}

thread_local! {
    /// One attention row's scores; it grows to the longest text once per thread.
    static SCORES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

impl EncoderLayer {
    fn load(t: &Tensors, name: &str, c: &Config) -> Result<Self, String> {
        let d = c.hidden;
        let hd = d / c.heads;
        let attention = |part: &str| Conv1d::linear(t, &format!("{name}.attention.{part}"), d, d);
        let relative =
            |part: &str| t.f32_len(&format!("{name}.attention.{part}"), (2 * c.window + 1) * hd);
        Ok(Self {
            q: attention("q_proj")?,
            k: attention("k_proj")?,
            v: attention("v_proj")?,
            out: attention("out_proj")?,
            rel_k: relative("emb_rel_k")?,
            rel_v: relative("emb_rel_v")?,
            norm: Norm::load(t, &format!("{name}.layer_norm"), d)?,
            ffn_in: Conv1d::load(
                t,
                &format!("{name}.feed_forward.conv_1"),
                (d, c.ffn),
                (c.ffn_kernel, 1),
            )?,
            ffn_out: Conv1d::load(
                t,
                &format!("{name}.feed_forward.conv_2"),
                (c.ffn, d),
                (c.ffn_kernel, 1),
            )?,
            final_norm: Norm::load(t, &format!("{name}.final_layer_norm"), d)?,
        })
    }

    /// Post-norm: `x = norm(x + attention(x))`, then `x = norm(x + ffn(x))`.
    fn forward(&self, mut x: Buf, tokens: usize, c: &Config) -> Buf {
        let attended = self.attention(&x, tokens, c);
        let mut x = self.norm.residual(&mut x, &attended, 1.0);
        let inner = self.ffn_in.forward(&x, tokens, None);
        // ReLU, then the second convolution: a leaky ReLU of slope 0.
        let ffn = self.ffn_out.forward(&inner, tokens, Some(0.0));
        self.final_norm.residual(&mut x, &ffn, 1.0)
    }

    /// Self-attention with relative positions: key `j` of query `i` adds
    /// `q_i . rel_k[j - i + window]` to the score and `p_ij * rel_v[j - i + window]`
    /// to the output, for `|j - i| <= window`.
    fn attention(&self, x: &[f32], tokens: usize, c: &Config) -> Buf {
        let (d, heads, window) = (c.hidden, c.heads, c.window);
        let hd = d / heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let (q, k, v) = (
            self.q.forward(x, tokens, None),
            self.k.forward(x, tokens, None),
            self.v.forward(x, tokens, None),
        );
        let mut mixed = Buf::scratch(tokens * d);
        let mixed_ptr = team::SendPtr::new(&mut mixed);
        team::for_each(heads * tokens, |task| {
            let (h, i) = (task / tokens, task % tokens);
            let query = &q[i * d + h * hd..][..hd];
            // SAFETY: row i, head h belongs to this task only; `mixed` outlives the step.
            let out = unsafe { mixed_ptr.slice_mut(i * d + h * hd, hd) };
            SCORES.with(|scores| {
                let mut scores = scores.borrow_mut();
                scores.resize(tokens, 0.0);
                let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
                let near = i.saturating_sub(window)..(i + window + 1).min(tokens);
                for (j, score) in scores.iter_mut().enumerate() {
                    *score = dot(query, &k[j * d + h * hd..][..hd]) * scale;
                }
                for j in near.clone() {
                    let r = j + window - i;
                    scores[j] += dot(query, &self.rel_k[r * hd..][..hd]) * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                scores.iter_mut().for_each(|s| *s -= max);
                activation::exp(&mut scores);
                let inv = 1.0 / scores.iter().sum::<f32>();
                out.fill(0.0);
                for (j, p) in scores.iter().enumerate() {
                    let p = p * inv;
                    for (o, value) in out.iter_mut().zip(&v[j * d + h * hd..][..hd]) {
                        *o += p * value;
                    }
                }
                for j in near {
                    let (r, p) = (j + window - i, scores[j] * inv);
                    for (o, value) in out.iter_mut().zip(&self.rel_v[r * hd..][..hd]) {
                        *o += p * value;
                    }
                }
            });
        });
        self.out.forward(&mixed, tokens, None)
    }
}

/// Dilated depth-separable convolutions (`VitsDilatedDepthSeparableConv`):
/// per layer a depthwise convolution with dilation `kernel^i`, layer norm,
/// GELU, a pointwise convolution, layer norm, GELU, and a residual add.
struct Separable {
    layers: Vec<(Depthwise, Norm, Conv1d, Norm)>,
    channels: usize,
}

impl Separable {
    fn load(t: &Tensors, name: &str, c: &Json, channels: usize) -> Result<Self, String> {
        let kernel = c.num("duration_predictor_kernel_size")?;
        let layers = (0..c.num("depth_separable_num_layers")?)
            .map(|i| {
                let dilation = kernel.pow(i as u32);
                Ok((
                    Depthwise::load(
                        t,
                        &format!("{name}.convs_dilated.{i}"),
                        channels,
                        (kernel, dilation),
                    )?,
                    Norm::load(t, &format!("{name}.norms_1.{i}"), channels)?,
                    Conv1d::linear(
                        t,
                        &format!("{name}.convs_pointwise.{i}"),
                        channels,
                        channels,
                    )?,
                    Norm::load(t, &format!("{name}.norms_2.{i}"), channels)?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self { layers, channels })
    }

    /// `x` (plus `condition` first, when given) through every layer.
    fn forward(&self, x: &mut [f32], frames: usize, condition: Option<&[f32]>) {
        if let Some(condition) = condition {
            x.iter_mut().zip(condition).for_each(|(v, c)| *v += c);
        }
        let channels = self.channels;
        for (depthwise, norm_1, pointwise, norm_2) in &self.layers {
            let h = depthwise.forward(x, frames);
            let mut h = norm_1.apply(&h, channels);
            Gelu::Exact.apply(&mut h, channels);
            let h = pointwise.forward(&h, frames, None);
            let mut h = norm_2.apply(&h, channels);
            Gelu::Exact.apply(&mut h, channels);
            x.iter_mut().zip(h.iter()).for_each(|(v, a)| *v += a);
        }
    }
}

/// A rational-quadratic spline flow on the second of two channels
/// (`VitsConvFlow`), conditioned on the text.
struct ConvFlow {
    pre: Conv1d,
    separable: Separable,
    project: Conv1d,
    bins: usize,
    tail: f32,
    /// `sqrt(filter channels)`: the widths and heights are divided by it.
    divisor: f32,
}

impl ConvFlow {
    fn load(t: &Tensors, name: &str, c: &Json, filter: usize) -> Result<Self, String> {
        let bins = c.num("duration_predictor_flow_bins")?;
        Ok(Self {
            pre: Conv1d::linear(t, &format!("{name}.conv_pre"), 1, filter)?,
            separable: Separable::load(t, &format!("{name}.conv_dds"), c, filter)?,
            project: Conv1d::linear(t, &format!("{name}.conv_proj"), filter, 3 * bins - 1)?,
            bins,
            tail: c.float("duration_predictor_tail_bound").unwrap_or(5.0) as f32,
            divisor: (filter as f32).sqrt(),
        })
    }

    /// The inverse flow on `z` (two channels per frame): the second channel
    /// through the inverse spline that the first channel and `condition`
    /// parameterize.
    fn reverse(&self, z: &mut [[f32; 2]], frames: usize, condition: &[f32]) {
        let first: Vec<f32> = z.iter().map(|pair| pair[0]).collect();
        let mut h = self.pre.forward(&first, frames, None);
        self.separable.forward(&mut h, frames, Some(condition));
        let params = self.project.forward(&h, frames, None);
        let bins = self.bins;
        for (pair, p) in z.iter_mut().zip(params.chunks_exact(3 * bins - 1)) {
            let widths: Vec<f32> = p[..bins].iter().map(|w| w / self.divisor).collect();
            let heights: Vec<f32> = p[bins..2 * bins].iter().map(|h| h / self.divisor).collect();
            pair[1] = inverse_spline(pair[1], &widths, &heights, &p[2 * bins..], self.tail);
        }
    }
}

/// The inverse of the monotonic rational-quadratic spline with `widths.len()`
/// bins on `[-tail, tail]` (identity outside), as `_rational_quadratic_spline`
/// computes it in reverse, in f32.
fn inverse_spline(x: f32, widths: &[f32], heights: &[f32], inner: &[f32], tail: f32) -> f32 {
    if !(-tail..=tail).contains(&x) {
        return x;
    }
    let bins = widths.len();
    let softplus = |v: f32| if v > 20.0 { v } else { v.exp().ln_1p() };
    let edge = ((1.0f32 - MIN_BIN).exp() - 1.0).ln();
    let derivatives: Vec<f32> = std::iter::once(edge)
        .chain(inner.iter().copied())
        .chain(std::iter::once(edge))
        .map(|v| MIN_BIN + softplus(v))
        .collect();
    // Bin edges from a softmax: at least MIN_BIN wide, spanning [-tail, tail].
    let edges = |raw: &[f32]| -> Vec<f32> {
        let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = raw.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let mut cumulative = vec![0.0f32; bins + 1];
        let mut total = 0.0f32;
        for (i, e) in exps.iter().enumerate() {
            total += MIN_BIN + (1.0 - MIN_BIN * bins as f32) * (e / sum);
            cumulative[i + 1] = total;
        }
        for v in &mut cumulative {
            *v = 2.0 * tail * *v - tail;
        }
        cumulative[0] = -tail;
        cumulative[bins] = tail;
        cumulative
    };
    let cum_widths = edges(widths);
    let cum_heights = edges(heights);
    let mut locations = cum_heights.clone();
    locations[bins] += 1e-6;
    let bin = locations
        .iter()
        .filter(|edge| x >= **edge)
        .count()
        .saturating_sub(1);
    let width = cum_widths[bin + 1] - cum_widths[bin];
    let height = cum_heights[bin + 1] - cum_heights[bin];
    let delta = height / width;
    let (d0, d1) = (derivatives[bin], derivatives[bin + 1]);
    let sum = d0 + d1 - 2.0 * delta;
    let offset = x - cum_heights[bin];
    let a = height * (delta - d0) + offset * sum;
    let b = height * d0 - offset * sum;
    let c = -delta * offset;
    let discriminant = (b * b - 4.0 * a * c).max(0.0);
    let root = (2.0 * c) / (-b - discriminant.sqrt());
    root * width + cum_widths[bin]
}

/// `VitsStochasticDurationPredictor`, run in reverse: noise through the
/// inverse flows gives the log duration of each token.
struct DurationPredictor {
    pre: Conv1d,
    separable: Separable,
    project: Conv1d,
    /// The reverse order of the reference: flows 4, 3, 2 (flow 1 is unused
    /// in reverse, as the reference drops it).
    flows: Vec<ConvFlow>,
    /// The elementwise affine flow 0: `(translate, log_scale)` per channel.
    affine: ([f32; 2], [f32; 2]),
}

impl DurationPredictor {
    fn load(t: &Tensors, c: &Json, d: usize) -> Result<Self, String> {
        let name = "duration_predictor";
        let count = c.num("duration_predictor_num_flows")?;
        let flows = (2..=count)
            .rev()
            .map(|i| ConvFlow::load(t, &format!("{name}.flows.{i}"), c, d))
            .collect::<Result<Vec<_>, _>>()?;
        let pair = |part: &str| -> Result<[f32; 2], String> {
            let v = t.f32_len(&format!("{name}.flows.0.{part}"), 2)?;
            Ok([v[0], v[1]])
        };
        Ok(Self {
            pre: Conv1d::linear(t, &format!("{name}.conv_pre"), d, d)?,
            separable: Separable::load(t, &format!("{name}.conv_dds"), c, d)?,
            project: Conv1d::linear(t, &format!("{name}.conv_proj"), d, d)?,
            flows,
            affine: (pair("translate")?, pair("log_scale")?),
        })
    }

    fn reverse(&self, hidden: &[f32], tokens: usize, scale: f32, noise: &mut Noise) -> Vec<f32> {
        let mut x = self.pre.forward(hidden, tokens, None);
        self.separable.forward(&mut x, tokens, None);
        let condition = self.project.forward(&x, tokens, None);
        let mut z: Vec<[f32; 2]> = (0..tokens)
            .map(|_| [noise.normal() * scale, noise.normal() * scale])
            .collect();
        for flow in &self.flows {
            z.iter_mut().for_each(|pair| pair.swap(0, 1));
            flow.reverse(&mut z, tokens, &condition);
        }
        z.iter_mut().for_each(|pair| pair.swap(0, 1));
        let (translate, log_scale) = self.affine;
        z.iter()
            .map(|pair| (pair[0] - translate[0]) * (-log_scale[0]).exp())
            .collect()
    }
}

/// One affine coupling layer of the prior flow (`VitsResidualCouplingLayer`,
/// mean only): a WaveNet on the first half predicts a shift of the second.
struct Coupling {
    pre: Conv1d,
    /// `(input layer, residual and skip layer)` per WaveNet layer.
    wavenet: Vec<(Conv1d, Conv1d)>,
    post: Conv1d,
    hidden: usize,
    half: usize,
}

impl Coupling {
    fn load(t: &Tensors, name: &str, c: &Json, config: &Config) -> Result<Self, String> {
        let (hidden, half) = (config.hidden, config.flow / 2);
        let layers = c.num("prior_encoder_num_wavenet_layers")?;
        let kernel = c.num("wavenet_kernel_size")?;
        let rate = c.num("wavenet_dilation_rate")?;
        let wavenet = (0..layers)
            .map(|i| {
                let skip = if i + 1 < layers { 2 * hidden } else { hidden };
                Ok((
                    Conv1d::load(
                        t,
                        &format!("{name}.wavenet.in_layers.{i}"),
                        (hidden, 2 * hidden),
                        (kernel, rate.pow(i as u32)),
                    )?,
                    Conv1d::linear(
                        t,
                        &format!("{name}.wavenet.res_skip_layers.{i}"),
                        hidden,
                        skip,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            pre: Conv1d::linear(t, &format!("{name}.conv_pre"), half, hidden)?,
            wavenet,
            post: Conv1d::linear(t, &format!("{name}.conv_post"), hidden, half)?,
            hidden,
            half,
        })
    }

    /// `second -= mean(first)` on every row of `z` (`frames x 2 * half`).
    fn reverse(&self, z: &mut [f32], frames: usize) {
        let (hidden, half) = (self.hidden, self.half);
        let first: Vec<f32> = z
            .chunks_exact(2 * half)
            .flat_map(|row| row[..half].to_vec())
            .collect();
        let mut h = self.pre.forward(&first, frames, None);
        let mut skip = Buf::zeroed(frames * hidden);
        for (i, (input, res_skip)) in self.wavenet.iter().enumerate() {
            let mut gate = input.forward(&h, frames, None);
            // tanh of the first half times the sigmoid of the second, per row.
            team::chunks_mut(&mut gate, 2 * hidden * 16, |_, rows| {
                for row in rows.chunks_exact_mut(2 * hidden) {
                    let (a, b) = row.split_at_mut(hidden);
                    activation::tanh(a);
                    activation::sigmoid(b);
                    a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x *= y);
                }
            });
            let acts: Vec<f32> = gate
                .chunks_exact(2 * hidden)
                .flat_map(|row| row[..hidden].to_vec())
                .collect();
            let out = res_skip.forward(&acts, frames, None);
            if i + 1 < self.wavenet.len() {
                for ((hr, sr), or) in h
                    .chunks_exact_mut(hidden)
                    .zip(skip.chunks_exact_mut(hidden))
                    .zip(out.chunks_exact(2 * hidden))
                {
                    hr.iter_mut().zip(&or[..hidden]).for_each(|(x, y)| *x += y);
                    sr.iter_mut().zip(&or[hidden..]).for_each(|(x, y)| *x += y);
                }
            } else {
                skip.iter_mut().zip(out.iter()).for_each(|(x, y)| *x += y);
            }
        }
        let mean = self.post.forward(&skip, frames, None);
        for (row, m) in z.chunks_exact_mut(2 * half).zip(mean.chunks_exact(half)) {
            row[half..].iter_mut().zip(m).for_each(|(x, y)| *x -= y);
        }
    }
}

/// HiFi-GAN (`VitsHifiGan`): upsampling stages, each followed by the mean of
/// residual blocks with several kernel sizes.
struct HifiGan {
    pre: Conv1d,
    stages: Vec<Stage>,
    post: Conv1d,
    slope: f32,
}

/// One upsampling and its residual blocks. A block is its
/// `(dilated conv, conv)` pairs.
struct Stage {
    up: Upsample,
    blocks: Vec<Vec<(Conv1d, Conv1d)>>,
}

impl HifiGan {
    fn load(t: &Tensors, c: &Json, flow: usize) -> Result<Self, String> {
        let list = |key: &str| -> Result<Vec<usize>, String> {
            c.get(key)
                .as_array()
                .and_then(|values| {
                    values
                        .iter()
                        .map(|v| v.as_u64().map(|v| v as usize))
                        .collect()
                })
                .ok_or_else(|| format!("config.json has no {key}"))
        };
        let rates = list("upsample_rates")?;
        let kernels = list("upsample_kernel_sizes")?;
        let block_kernels = list("resblock_kernel_sizes")?;
        let dilations: Vec<Vec<usize>> = c
            .get("resblock_dilation_sizes")
            .as_array()
            .and_then(|rows| {
                rows.iter()
                    .map(|row| {
                        row.as_array()?
                            .iter()
                            .map(|v| v.as_u64().map(|v| v as usize))
                            .collect()
                    })
                    .collect()
            })
            .ok_or("config.json has no resblock_dilation_sizes")?;
        let initial = c.num("upsample_initial_channel")?;
        let mut stages = Vec::new();
        for (i, (rate, kernel)) in rates.iter().zip(&kernels).enumerate() {
            let (inputs, outputs) = (initial >> i, initial >> (i + 1));
            let up = Upsample::load(
                t,
                &format!("decoder.upsampler.{i}"),
                (inputs, outputs),
                (*kernel, *rate),
            )?;
            let blocks = block_kernels
                .iter()
                .zip(&dilations)
                .enumerate()
                .map(|(j, (k, ds))| {
                    let name = format!("decoder.resblocks.{}", i * block_kernels.len() + j);
                    ds.iter()
                        .enumerate()
                        .map(|(n, dilation)| {
                            Ok((
                                Conv1d::load(
                                    t,
                                    &format!("{name}.convs1.{n}"),
                                    (outputs, outputs),
                                    (*k, *dilation),
                                )?,
                                Conv1d::load(
                                    t,
                                    &format!("{name}.convs2.{n}"),
                                    (outputs, outputs),
                                    (*k, 1),
                                )?,
                            ))
                        })
                        .collect::<Result<Vec<_>, String>>()
                })
                .collect::<Result<Vec<_>, String>>()?;
            stages.push(Stage { up, blocks });
        }
        let last = initial >> rates.len();
        Ok(Self {
            pre: Conv1d::load(t, "decoder.conv_pre", (flow, initial), (7, 1))?,
            stages,
            post: Conv1d::load(t, "decoder.conv_post", (last, 1), (7, 1))?,
            slope: c.float("leaky_relu_slope").unwrap_or(0.1) as f32,
        })
    }

    /// The waveform of a spectrogram latent (`frames x flow`), in -1..1.
    fn forward(&self, latent: &[f32], frames: usize) -> Vec<f32> {
        let mut x = self.pre.forward(latent, frames, None);
        let mut frames = frames;
        for Stage { up, blocks } in &self.stages {
            let upsampled = up.forward(&x, frames, self.slope);
            frames *= up.stride;
            let mut sum = Buf::zeroed(frames * up.outputs);
            for block in blocks {
                let mut y = Buf::scratch(upsampled.len());
                y.copy_from_slice(&upsampled);
                for (dilated, plain) in block {
                    let a = dilated.forward(&y, frames, Some(self.slope));
                    let b = plain.forward(&a, frames, Some(self.slope));
                    y.iter_mut().zip(b.iter()).for_each(|(v, r)| *v += r);
                }
                sum.iter_mut().zip(y.iter()).for_each(|(s, v)| *s += v);
            }
            let inv = 1.0 / blocks.len() as f32;
            sum.iter_mut().for_each(|v| *v *= inv);
            x = sum;
        }
        let mut wave = self.post.forward(&x, frames, Some(FINAL_SLOPE));
        activation::tanh(&mut wave);
        wave.to_vec()
    }
}
