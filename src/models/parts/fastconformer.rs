//! The FastConformer encoder of Parakeet and Canary, and the head that
//! follows it (`transducer.rs`, CTC, or none). The structure follows the
//! Apache-2.0 Hugging Face `ParakeetEncoder`.

use super::dictionary::Dictionary;
use super::transducer::Transducer;
use crate::config::Json;
use crate::cpu::activation;
use crate::cpu::buf::Buf;
use crate::cpu::fgemm;
use crate::cpu::gemm::{Dense, Precision};
use crate::cpu::kernels::Ternary;
use crate::cpu::store::Store;
use crate::cpu::team;
use crate::cpu::tensors::Tensors;
use crate::layers::{conv_columns, folded_batch_norm, Norm, Rotary};
use crate::Cancel;
use parking_lot::{Mutex, RwLock};
use std::sync::{Arc, Weak};

thread_local! {
    /// One attention task's queries (two biases), scores, position scores,
    /// and output. It grows to the longest take once per thread and is then
    /// reused, so the worker threads allocate nothing per task.
    static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Queries per attention task: a short take still makes several tasks per
/// head, and a block reads t+31 positions instead of 2t-1 per query.
const ATTENTION_ROWS: usize = 32;

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub ff: usize,
    pub heads: usize,
    pub layers: usize,
    pub mels: usize,
    pub conv_kernel: usize,
    pub sub_channels: usize,
    pub decoder_hidden: usize,
    pub decoder_layers: usize,
    pub vocab: usize,
    pub blank: usize,
    pub durations: Vec<usize>,
    pub max_symbols: usize,
    pub group: usize,
    /// `sqrt(hidden)` when the encoder scales its input (`scale_input`), else 1.
    pub input_scale: f32,
    pub head: HeadKind,
    /// Before every `encoder.` tensor name: `model.` in Canary files.
    pub prefix: &'static str,
    pub attention: Attention,
    /// Frames before and after a frame that the depthwise conv reads:
    /// `(kernel - 1) / 2` each, or `(kernel - 1, 0)` for a causal conv.
    pub conv_context: (usize, usize),
    /// NeMo's `causal_downsampling`: the subsampling convs pad two rows and
    /// columns before and one after, so 128 mels become 17 bins, not 16.
    pub causal_subsampling: bool,
    /// The conv module's norm: NeMo's `conv_norm_type`.
    pub conv_norm: ConvNormKind,
    /// A speaker kernel before layer 0 (NeMo's multitalker models), run
    /// for one speaker: the whole input is the target speaker's.
    pub speaker_kernel: bool,
    /// The language prompt of a prompt-conditioned model (Nemotron 3.5).
    pub prompt: Option<PromptConfig>,
    /// Per-bin normalization of the log-mel features over the take
    /// (`per_feature`), or none (the raw log-mel of a streaming model).
    pub normalize_features: bool,
    pub position: Position,
    pub subsampler: SubsamplerKind,
    /// The predictor's first input is zeros (GigaAM), not the embedding of
    /// the blank (NeMo, where that row is zero anyway).
    pub start_zero: bool,
    /// Every layer norm's epsilon: 1e-5 (PyTorch), 1e-6 (LASR).
    pub norm_eps: f32,
    /// `(w0, w1)`: a feed-forward module gives `w0 * x + w1 * ff(x)`
    /// (`(1, 0.5)` in a Conformer; LASR `(1.5, 0.5)`).
    pub ff_residual: (f32, f32),
    /// `(w0, w1)` for the conv module: `(1, 1)`; LASR `(2, 1)`.
    pub conv_residual: (f32, f32),
    /// A layer norm after the last block (LASR's `out_norm`).
    pub out_norm: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Position {
    /// Transformer-XL relative positions (NeMo): `relative_k_proj`, `bias_u`, `bias_v`.
    Relative,
    /// Rotary positions on the block input before the query and key
    /// projections (GigaAM): halves of each head, this base.
    RotaryInput { base: f32 },
    /// Rotary positions on the projected queries and keys (LASR): halves
    /// of each head, this base.
    Rotary { base: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubsamplerKind {
    /// NeMo's `dw_striding`: a 2-D conv and two depthwise-separable 2-D
    /// convs, each stride 2: 80 ms frames.
    Conv2d,
    /// Two 1-D convs over time, stride 2, with ReLU (GigaAM): 40 ms frames.
    Conv1d { kernel: usize },
    /// LASR's: a dense layer, two unpadded 1-D convs over time with this
    /// kernel and stride (the second to `channels`), then a dense layer
    /// back to the model width; ReLU after the first three.
    Lasr {
        kernel: usize,
        stride: usize,
        channels: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvNormKind {
    Batch,
    Layer,
}

/// NeMo's `EncDecRNNTBPEModelWithPrompt`: a one-hot of the language joins
/// every encoder frame, and an MLP maps the pair back to the model width.
#[derive(Debug, Clone)]
pub struct PromptConfig {
    pub count: usize,
    pub hidden: usize,
    /// Language codes (and aliases: `en`, `en-US`) and their prompt.
    pub dictionary: Vec<(String, usize)>,
    /// The prompt that lets the model find the language itself.
    pub auto: usize,
}

impl PromptConfig {
    fn parse(json: &Json) -> Result<Option<Self>, String> {
        let Some(dictionary) = json.get("prompt.dictionary").as_object() else {
            return Ok(None);
        };
        let dictionary = dictionary
            .iter()
            .map(|(code, index)| {
                index
                    .as_u64()
                    .map(|index| (code.clone(), index as usize))
                    .ok_or("config.json: a prompt index is not a number")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let config = Self {
            count: json.num("prompt.num_prompts")?,
            hidden: json.num("prompt.hidden")?,
            auto: json.num("prompt.auto_id")?,
            dictionary,
        };
        if config.auto >= config.count || config.dictionary.iter().any(|(_, i)| *i >= config.count)
        {
            return Err("config.json: a prompt index past num_prompts".into());
        }
        Ok(Some(config))
    }

    /// The prompt of language `code` (`en`, `en-US`), else the automatic one.
    fn index(&self, code: Option<&str>) -> usize {
        code.and_then(|code| {
            self.dictionary
                .iter()
                .find(|(known, _)| known.eq_ignore_ascii_case(code.trim()))
        })
        .map_or(self.auto, |(_, index)| *index)
    }
}

/// The keys of self-attention, as NeMo's `att_context_size` and
/// `att_context_style` set them. Hugging Face folders give `Full`; a GGUF
/// file of a model trained with a limited context gives the others
/// (`encoder_config.attention_context`, `attention_style`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attention {
    Full,
    /// `regular` local attention: keys `i - left ..= i + right`.
    Band {
        left: usize,
        right: usize,
    },
    /// `chunked_limited` (cache-aware streaming): chunks of `right + 1`
    /// frames; a query reads its own chunk and `left / (right + 1)` before it.
    Chunked {
        chunk: usize,
        left_chunks: usize,
    },
}

impl Attention {
    fn parse(json: &Json) -> Result<Self, String> {
        let Some(sides) = json.get("encoder_config.attention_context").as_array() else {
            return Ok(Self::Full);
        };
        let side = |i: usize| sides.get(i).and_then(|v| v.as_i64()).unwrap_or(-1);
        let (left, right) = (side(0), side(1));
        if left < 0 && right < 0 {
            return Ok(Self::Full);
        }
        if left < 0 || right < 0 {
            return Err("config.json: an attention context with one open side".into());
        }
        let (left, right) = (left as usize, right as usize);
        match json
            .str("encoder_config.attention_style")
            .unwrap_or("regular")
        {
            "regular" => Ok(Self::Band { left, right }),
            "chunked_limited" => Ok(Self::Chunked {
                chunk: right + 1,
                left_chunks: left / (right + 1),
            }),
            other => Err(format!("config.json: attention style `{other}` is not run")),
        }
    }

    /// The keys query `i` reads, as a range of `0..t`.
    fn keys(self, i: usize, t: usize) -> std::ops::Range<usize> {
        match self {
            Self::Full => 0..t,
            Self::Band { left, right } => i.saturating_sub(left)..(i + right + 1).min(t),
            Self::Chunked { chunk, left_chunks } => {
                let own = i / chunk;
                own.saturating_sub(left_chunks) * chunk..((own + 1) * chunk).min(t)
            }
        }
    }
}

/// The fewest mel frames the subsampling takes: one output frame per stage.
pub const MIN_FRAMES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadKind {
    /// TDT decoder and joint network (`parakeet_tdt`): the joint network
    /// gives each token a duration too.
    Transducer,
    /// RNNT decoder and joint network (`parakeet_rnnt`): a TDT without
    /// durations; a blank moves one frame, a token none.
    Rnnt,
    /// One projection per frame (`parakeet_ctc`).
    Ctc,
    /// None: another model (Canary) decodes the encoder output.
    Encoder,
}

impl Config {
    pub fn parse(json: &Json) -> Result<Self, String> {
        let head = match json.str("model_type") {
            Some("parakeet_tdt") => HeadKind::Transducer,
            Some("parakeet_rnnt" | "gigaam_rnnt") => HeadKind::Rnnt,
            Some("parakeet_ctc" | "gigaam_ctc" | "medasr") => HeadKind::Ctc,
            Some("canary" | "canary_qwen") => HeadKind::Encoder,
            other => return Err(format!("config.json: no encoder for `{other:?}`")),
        };
        let enc = |key: &str| json.num(&format!("encoder_config.{key}"));
        // Only the shapes this engine implements are accepted.
        let (factor, kernel, stride) = (
            enc("subsampling_factor")?,
            enc("subsampling_conv_kernel_size")?,
            enc("subsampling_conv_stride")?,
        );
        let subsampler = match json.str("encoder_config.subsampling") {
            None | Some("dw_striding") if (factor, kernel, stride) == (8, 3, 2) => {
                SubsamplerKind::Conv2d
            }
            Some("conv1d") if factor == 4 && stride == 2 && !kernel.is_multiple_of(2) => {
                SubsamplerKind::Conv1d { kernel }
            }
            Some("lasr") if factor == stride * stride && stride > 1 => SubsamplerKind::Lasr {
                kernel,
                stride,
                channels: enc("subsampling_conv_channels")?,
            },
            _ => return Err("config.json describes a model shape this engine does not run".into()),
        };
        let hidden = enc("hidden_size")?;
        let transducer = matches!(head, HeadKind::Transducer | HeadKind::Rnnt);
        // Only the transducers have the decoder fields; a CTC blank is the pad token.
        let decoder = |key: &str| if transducer { json.num(key) } else { Ok(0) };
        let config = Self {
            hidden,
            ff: enc("intermediate_size")?,
            heads: enc("num_attention_heads")?,
            layers: enc("num_hidden_layers")?,
            mels: enc("num_mel_bins")?,
            conv_kernel: enc("conv_kernel_size")?,
            sub_channels: match subsampler {
                SubsamplerKind::Conv2d => enc("subsampling_conv_channels")?,
                SubsamplerKind::Conv1d { .. } | SubsamplerKind::Lasr { .. } => hidden,
            },
            subsampler,
            position: match json.str("encoder_config.position_embedding") {
                None | Some("relative") => Position::Relative,
                Some("rotary_input") => Position::RotaryInput {
                    base: json.float("encoder_config.rotary_base").unwrap_or(10_000.0) as f32,
                },
                Some("rotary") => Position::Rotary {
                    base: json.float("encoder_config.rotary_base").unwrap_or(10_000.0) as f32,
                },
                Some(other) => return Err(format!("config.json: positions `{other}` are not run")),
            },
            start_zero: json.str("decoder_start") == Some("zeros"),
            norm_eps: json.float("encoder_config.layer_norm_eps").unwrap_or(1e-5) as f32,
            ff_residual: weights(json, "encoder_config.ff_residual", (1.0, 0.5))?,
            conv_residual: weights(json, "encoder_config.conv_residual", (1.0, 1.0))?,
            out_norm: json.flag("encoder_config.output_norm") == Some(true),
            decoder_hidden: decoder("decoder_hidden_size")?,
            decoder_layers: decoder("num_decoder_layers")?,
            vocab: json.num("vocab_size")?,
            blank: json.num(if transducer {
                "blank_token_id"
            } else {
                "pad_token_id"
            })?,
            durations: match json.get("durations").as_array() {
                Some(list) => list
                    .iter()
                    .map(|d| d.as_u64().map(|d| d as usize).ok_or("bad duration"))
                    .collect::<Result<_, _>>()?,
                None if head != HeadKind::Transducer => Vec::new(),
                None => return Err("config.json has no durations".into()),
            },
            max_symbols: decoder("max_symbols_per_step")?,
            // Ternary models (Redux) only; dense models have no groups.
            group: json.num("ternary_group_size").unwrap_or(0),
            input_scale: if json.flag("encoder_config.scale_input") == Some(true) {
                (hidden as f32).sqrt()
            } else {
                1.0
            },
            head,
            prefix: if head == HeadKind::Encoder {
                "model."
            } else {
                ""
            },
            attention: Attention::parse(json)?,
            causal_subsampling: json.flag("encoder_config.causal_downsampling") == Some(true),
            conv_norm: match json.str("encoder_config.conv_norm_type") {
                None | Some("batch_norm") => ConvNormKind::Batch,
                Some("layer_norm") => ConvNormKind::Layer,
                Some(other) => return Err(format!("config.json: conv norm `{other}` is not run")),
            },
            speaker_kernel: match json.get("encoder_config.speaker_kernel_layers").as_array() {
                None => false,
                Some(layers) if layers.iter().map(|v| v.as_u64()).eq([Some(0)]) => true,
                Some(_) => {
                    return Err("config.json: a speaker kernel after layer 0 is not run".into())
                }
            },
            prompt: PromptConfig::parse(json)?,
            normalize_features: match json.str("feature_normalize") {
                None | Some("per_feature") => true,
                Some("none") => false,
                Some(other) => {
                    return Err(format!(
                        "config.json: feature normalization `{other}` is not run"
                    ))
                }
            },
            conv_context: match json.get("encoder_config.conv_context").as_array() {
                Some(sides) => (
                    sides.first().and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                    sides.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                ),
                None => {
                    let half = (enc("conv_kernel_size")? - 1) / 2;
                    (half, half)
                }
            },
        };
        if config.heads == 0
            || config.hidden == 0
            || !config.hidden.is_multiple_of(config.heads)
            || (config.subsampler == SubsamplerKind::Conv2d && !config.mels.is_multiple_of(8))
            || (config.position != Position::Relative && config.attention != Attention::Full)
            || (head == HeadKind::Transducer && config.durations.is_empty())
            || (transducer && config.max_symbols == 0)
            || config.blank >= config.vocab
            || config.conv_context.0 + config.conv_context.1 + 1 != config.conv_kernel
        {
            return Err("config.json has inconsistent sizes".into());
        }
        Ok(config)
    }

    /// Seconds per encoder frame: 10 ms mel frames, subsampled 8 or 4 times.
    pub fn frame_seconds(&self) -> f32 {
        match self.subsampler {
            SubsamplerKind::Conv2d => 0.08,
            SubsamplerKind::Conv1d { .. } => 0.04,
            SubsamplerKind::Lasr { stride, .. } => 0.01 * (stride * stride) as f32,
        }
    }

    /// Whether the GPU encoder runs this shape: full attention, a centered
    /// conv with batch norm, and no speaker kernel.
    #[cfg(feature = "gpu")]
    fn gpu_shape(&self) -> bool {
        self.attention == Attention::Full
            && self.position == Position::Relative
            && self.norm_eps == 1e-5
            && self.ff_residual == (1.0, 0.5)
            && self.conv_residual == (1.0, 1.0)
            && !self.out_norm
            && self.conv_context.0 == self.conv_context.1
            && self.conv_norm == ConvNormKind::Batch
            && !self.speaker_kernel
    }
}

/// `(w0, w1)` residual weights at `path`, else `default`.
fn weights(json: &Json, path: &str, default: (f32, f32)) -> Result<(f32, f32), String> {
    match json.get(path).as_array().map(|pair| pair.as_slice()) {
        None => Ok(default),
        Some([w0, w1]) => match (w0.as_f64(), w1.as_f64()) {
            (Some(w0), Some(w1)) => Ok((w0 as f32, w1 as f32)),
            _ => Err(format!("config.json: {path} is not two numbers")),
        },
        Some(_) => Err(format!("config.json: {path} is not two numbers")),
    }
}

/// A projection of the encoder: ternary weights (Parakeet Redux) or dense
/// weights (Parakeet TDT v3), whichever the file holds.
enum Linear {
    Ternary(Ternary),
    Dense(Dense),
}

impl Linear {
    fn forward(&self, x: &[f32], rows: usize) -> Buf {
        match self {
            Self::Ternary(layer) => layer.forward(x, rows),
            Self::Dense(layer) => layer.forward(x, rows),
        }
    }
}

/// `name` as ternary (`.qweight` + `.scales`) or dense (`.weight`, any float
/// dtype, `[out, in]` or `[out, in, 1]` for a pointwise conv).
fn linear(t: &Tensors, name: &str, out: usize, inp: usize, group: usize) -> Result<Linear, String> {
    stacked(t, &[name.to_string()], (out, inp), group)
}

/// Several layers with the same input, stacked by output rows into one.
fn stacked(
    t: &Tensors,
    names: &[String],
    (out, inp): (usize, usize),
    group: usize,
) -> Result<Linear, String> {
    if t.has(&format!("{}.qweight", names[0])) {
        if group == 0 {
            return Err("config.json has no ternary_group_size for the ternary layers".into());
        }
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        for name in names {
            packed.extend_from_slice(&t.u8(&format!("{name}.qweight"))?);
            scales.extend(t.f32_len(&format!("{name}.scales"), out * (inp / group))?);
        }
        return Ternary::from_base3(&packed, &scales, names.len() * out, inp, group)
            .and_then(|layer| layer.on_gpu(t.device()))
            .map(Linear::Ternary);
    }
    let parts: Vec<(&str, usize)> = names.iter().map(|name| (name.as_str(), out)).collect();
    Dense::load(t, &parts, inp).map(Linear::Dense)
}

/// `name` (`.weight`, `.bias`): f32 under `Precision::Auto`, as this model's
/// accuracy was measured, else the precision the caller asked for.
fn dense(t: &Tensors, name: &str, out: usize, inp: usize) -> Result<Dense, String> {
    dense_in(t, name, out, inp, true)
}

/// `dense` with no GPU copy. The subsampler and the heads stay on the CPU.
pub(crate) fn dense_host(t: &Tensors, name: &str, out: usize, inp: usize) -> Result<Dense, String> {
    dense_in(t, name, out, inp, false)
}

fn dense_in(t: &Tensors, name: &str, out: usize, inp: usize, gpu: bool) -> Result<Dense, String> {
    if t.precision_of(&format!("{name}.weight")) != Precision::Auto {
        return if gpu {
            Dense::load(t, &[(name, out)], inp)
        } else {
            Dense::load_cpu(t, &[(name, out)], inp)
        };
    }
    let weight = t.f32_len(&format!("{name}.weight"), out * inp)?;
    let bias = t.f32_len(&format!("{name}.bias"), out)?;
    let layer = Dense::new(weight, Some(bias), out, inp)?;
    if gpu {
        layer.on_gpu(t.device())
    } else {
        Ok(layer)
    }
}

enum Subsampler {
    Conv2d(Subsampling),
    Conv1d(Conv1dSubsampling),
    Lasr(LasrSubsampling),
}

impl Subsampler {
    fn load(t: &Tensors, c: &Config) -> Result<Self, String> {
        Ok(match c.subsampler {
            SubsamplerKind::Conv2d => Self::Conv2d(Subsampling::load(t, c)?),
            SubsamplerKind::Lasr {
                kernel,
                stride,
                channels,
            } => Self::Lasr(LasrSubsampling::load(t, c, (kernel, stride, channels))?),
            SubsamplerKind::Conv1d { kernel } => {
                Self::Conv1d(Conv1dSubsampling::load(t, c, kernel)?)
            }
        })
    }

    /// `features` is `frames x mels`; returns `(out_frames x hidden, out_frames)`.
    fn forward(&self, features: &[f32], frames: usize, mels: usize) -> (Buf, usize) {
        match self {
            Self::Conv2d(sub) => sub.forward(features, frames, mels),
            Self::Conv1d(sub) => sub.forward(features, frames, mels),
            Self::Lasr(sub) => sub.forward(features, frames, mels),
        }
    }
}

/// LASR's subsampler (MedASR): `dense_0` from the mels to the model width,
/// two unpadded 1-D convs over time (`conv_0`, `conv_1`), and `dense_1`
/// back to the model width; ReLU after all but the last.
struct LasrSubsampling {
    kernel: usize,
    stride: usize,
    hidden: usize,
    dense0: Dense,
    convs: [Dense; 2],
    dense1: Dense,
}

impl LasrSubsampling {
    fn load(
        t: &Tensors,
        c: &Config,
        (kernel, stride, channels): (usize, usize, usize),
    ) -> Result<Self, String> {
        let p = format!("{}encoder.subsampling", c.prefix);
        let d = c.hidden;
        let load = |name: &str, out: usize, inp: usize| {
            Dense::load_cpu(t, &[(&format!("{p}.{name}"), out)], inp)
        };
        Ok(Self {
            kernel,
            stride,
            hidden: d,
            dense0: load("dense_0", d, c.mels)?,
            convs: [
                load("conv_0", d, d * kernel)?,
                load("conv_1", channels, d * kernel)?,
            ],
            dense1: load("dense_1", d, channels)?,
        })
    }

    fn forward(&self, features: &[f32], frames: usize, _mels: usize) -> (Buf, usize) {
        let (k, s) = (self.kernel, self.stride);
        let out = |n: usize| if n < k { 0 } else { (n - k) / s + 1 };
        let (t1, t2) = (out(frames), out(out(frames)));
        if t2 == 0 {
            return (Buf::scratch(0), 0);
        }
        let x = self.dense0.forward_relu(features, frames);
        let cols = conv_columns(&x, (frames, self.hidden), (k, s, 0), t1);
        let x = self.convs[0].forward_relu(&cols, t1);
        let cols = conv_columns(&x, (t1, self.hidden), (k, s, 0), t2);
        let x = self.convs[1].forward_relu(&cols, t2);
        (self.dense1.forward(&x, t2), t2)
    }
}

/// Two 1-D convs over time, stride 2, centered padding, each with ReLU:
/// 10 ms mel frames into 40 ms frames (GigaAM's `Conv1dSubsampling`).
struct Conv1dSubsampling {
    kernel: usize,
    hidden: usize,
    convs: [Dense; 2],
}

impl Conv1dSubsampling {
    fn load(t: &Tensors, c: &Config, kernel: usize) -> Result<Self, String> {
        let p = format!("{}encoder.subsampling.layers", c.prefix);
        let d = c.hidden;
        Ok(Self {
            kernel,
            hidden: d,
            convs: [
                Dense::load_cpu(t, &[(&format!("{p}.0"), d)], c.mels * kernel)?,
                Dense::load_cpu(t, &[(&format!("{p}.2"), d)], d * kernel)?,
            ],
        })
    }

    fn forward(&self, features: &[f32], frames: usize, mels: usize) -> (Buf, usize) {
        let (k, pad) = (self.kernel, self.kernel / 2);
        let out = |n: usize| (n + 2 * pad - k) / 2 + 1;
        let (t1, t2) = (out(frames), out(out(frames)));
        let cols = conv_columns(features, (frames, mels), (k, 2, pad), t1);
        let x = self.convs[0].forward_relu(&cols, t1);
        let cols = conv_columns(&x, (t1, self.hidden), (k, 2, pad), t2);
        (self.convs[1].forward_relu(&cols, t2), t2)
    }
}

/// Stride-2 convolution stack that turns 10 ms mel frames into 80 ms frames.
struct Subsampling {
    channels: usize,
    /// Rows and columns of zeros before the input of each conv: 1, or 2
    /// for a causal model. One after, either way.
    pad: usize,
    /// First conv as `[tap][channel]` weights and bias.
    conv0: (Vec<f32>, Vec<f32>),
    dw1: (Vec<f32>, Vec<f32>),
    pw1: Dense,
    dw2: (Vec<f32>, Vec<f32>),
    pw2: Dense,
    linear: Dense,
}

impl Subsampling {
    fn load(t: &Tensors, c: &Config) -> Result<Self, String> {
        let ch = c.sub_channels;
        let conv = |index: usize| -> Result<(Vec<f32>, Vec<f32>), String> {
            Ok((
                t.f32_len(
                    &format!("{}encoder.subsampling.layers.{index}.weight", c.prefix),
                    ch * 9,
                )?,
                t.f32_len(
                    &format!("{}encoder.subsampling.layers.{index}.bias", c.prefix),
                    ch,
                )?,
            ))
        };
        let pad = if c.causal_subsampling { 2 } else { 1 };
        let bins = out_len(out_len(out_len(c.mels, pad), pad), pad);
        Ok(Self {
            channels: ch,
            pad,
            conv0: {
                let (weight, bias) = conv(0)?;
                (
                    (0..9 * ch).map(|i| weight[(i % ch) * 9 + i / ch]).collect(),
                    bias,
                )
            },
            dw1: tap_major(conv(2)?, ch),
            pw1: dense_host(
                t,
                &format!("{}encoder.subsampling.layers.3", c.prefix),
                ch,
                ch,
            )?,
            dw2: tap_major(conv(5)?, ch),
            pw2: dense_host(
                t,
                &format!("{}encoder.subsampling.layers.6", c.prefix),
                ch,
                ch,
            )?,
            linear: dense_host(
                t,
                &format!("{}encoder.subsampling.linear", c.prefix),
                c.hidden,
                ch * bins,
            )?,
        })
    }

    /// `features` is `frames x mels`; returns `(out_frames x hidden, out_frames)`.
    fn forward(&self, features: &[f32], frames: usize, mels: usize) -> (Buf, usize) {
        let ch = self.channels;
        let (x, h2, w2) = self.first_two_convs(features, frames, mels);
        let x = self.pw1.forward_relu(&x, h2 * w2);
        let (x, h3, w3) = depthwise(&x, (h2, w2, ch), self.pad, &self.dw2);
        let x = self.pw2.forward_relu(&x, h3 * w3);
        // Flatten per frame as channel-major: index = c * w3 + w.
        let mut flat = Buf::scratch(h3 * ch * w3);
        team::chunks_mut(&mut flat, ch * w3, |h, row| {
            for w in 0..w3 {
                for c in 0..ch {
                    row[c * w3 + w] = x[(h * w3 + w) * ch + c];
                }
            }
        });
        (self.linear.forward(&flat, h3), h3)
    }
}

impl Subsampling {
    /// conv0 (one input channel) and the first depthwise conv, in strips of
    /// output rows. The full conv0 output is 73 MB for a 22 s take; a strip
    /// only computes the conv0 rows it needs, so that buffer never exists.
    fn first_two_convs(&self, features: &[f32], frames: usize, mels: usize) -> (Buf, usize, usize) {
        let (ch, pad) = (self.channels, self.pad);
        let (h1, w1) = (out_len(frames, pad), out_len(mels, pad));
        let (h2, w2) = (out_len(h1, pad), out_len(w1, pad));
        let mut out = Buf::scratch(h2 * w2 * ch);
        // At least four strips per thread, so a short take still spreads out.
        let strip_rows = h2.div_ceil(4 * team::threads()).clamp(1, MAX_STRIP_ROWS);
        team::chunks_mut(&mut out, strip_rows * w2 * ch, |strip, rows| {
            conv_strip(self, features, (frames, mels), strip_rows, strip, rows);
        });
        (out, h2, w2)
    }

    /// One output row of conv0, channels-last, after ReLU.
    #[inline(always)]
    fn conv0_row(&self, features: &[f32], frames: usize, mels: usize, h: usize, row: &mut [f32]) {
        let ch = self.channels;
        let (weight, bias) = &self.conv0;
        for (w, cell) in row.chunks_exact_mut(ch).enumerate() {
            cell.copy_from_slice(bias);
            for tap in 0..9 {
                let Some(v) = sample(
                    features,
                    (frames, mels, self.pad),
                    2 * h + tap / 3,
                    2 * w + tap % 3,
                ) else {
                    continue;
                };
                for (o, wt) in cell.iter_mut().zip(&weight[tap * ch..(tap + 1) * ch]) {
                    *o += wt * v;
                }
            }
            cell.iter_mut().for_each(|o| *o = o.max(0.0));
        }
    }
}

multiversion! {
    /// Output rows `strip * strip_rows ..` of the first depthwise conv, from
    /// the conv0 rows they read. `rows` holds those output rows.
    fn conv_strip(
        sub: &Subsampling,
        features: &[f32],
        frames_mels: (usize, usize),
        strip_rows: usize,
        strip: usize,
        rows: &mut [f32],
    ) {
        let (frames, mels) = frames_mels;
        let (ch, pad) = (sub.channels, sub.pad);
        let (h1, w1) = (out_len(frames, pad), out_len(mels, pad));
        let w2 = out_len(w1, pad);
        let first_out = strip * strip_rows;
        let out_rows = rows.len() / (w2 * ch);
        // Output row oh reads conv0 rows 2oh-pad ..= 2oh+2-pad.
        let first_in = (2 * first_out).saturating_sub(pad);
        let last_in = (2 * (first_out + out_rows - 1) + 2 - pad).min(h1 - 1);
        let len = (last_in - first_in + 1) * w1 * ch;
        // This thread's buffer, so a task allocates nothing; `conv0_row`
        // writes every value it reads.
        let mut conv0 = CONV0.take();
        if conv0.len() < len {
            conv0.resize(len, 0.0);
        }
        for (offset, row) in conv0[..len].chunks_exact_mut(w1 * ch).enumerate() {
            sub.conv0_row(features, frames, mels, first_in + offset, row);
        }
        let (weight, bias) = &sub.dw1;
        for local in 0..out_rows {
            let oh = first_out + local;
            for ow in 0..w2 {
                let cell = &mut rows[(local * w2 + ow) * ch..][..ch];
                cell.copy_from_slice(bias);
                for i in 0..3 {
                    for j in 0..3 {
                        let (ih, iw) = (2 * oh + i, 2 * ow + j);
                        if ih < pad || iw < pad || ih - pad >= h1 || iw - pad >= w1 {
                            continue;
                        }
                        let input = &conv0[((ih - pad - first_in) * w1 + (iw - pad)) * ch..][..ch];
                        let taps = &weight[(i * 3 + j) * ch..][..ch];
                        for ((o, wt), v) in cell.iter_mut().zip(taps).zip(input) {
                            *o += wt * v;
                        }
                    }
                }
            }
        }
        CONV0.set(conv0);
    }
}

thread_local! {
    /// The conv0 rows of one `conv_strip` task, up to 1 MB. It grows once
    /// per thread and is then reused.
    static CONV0: std::cell::Cell<Vec<f32>> = const { std::cell::Cell::new(Vec::new()) };
}

/// Most output rows of the first depthwise conv in one parallel task.
const MAX_STRIP_ROWS: usize = 8;

/// `[channel][tap]` depthwise weights as `[tap][channel]`, so a tap is one contiguous row.
fn tap_major((weight, bias): (Vec<f32>, Vec<f32>), ch: usize) -> (Vec<f32>, Vec<f32>) {
    (
        (0..9 * ch).map(|i| weight[(i % ch) * 9 + i / ch]).collect(),
        bias,
    )
}

/// Output length of a kernel-3, stride-2 convolution with `pad` zeros
/// before the input and one after.
fn out_len(n: usize, pad: usize) -> usize {
    (n + pad - 2) / 2 + 1
}

/// Input value at padded position (row, col), or None in the zero padding.
#[inline(always)]
fn sample(
    x: &[f32],
    (rows, cols, pad): (usize, usize, usize),
    row: usize,
    col: usize,
) -> Option<f32> {
    let (r, c) = (row.checked_sub(pad)?, col.checked_sub(pad)?);
    (r < rows && c < cols).then(|| x[r * cols + c])
}

/// Depthwise 3x3 stride-2 convolution on channels-last data.
fn depthwise(
    x: &[f32],
    (h, w, ch): (usize, usize, usize),
    pad: usize,
    (weight, bias): &(Vec<f32>, Vec<f32>),
) -> (Buf, usize, usize) {
    let (ho, wo) = (out_len(h, pad), out_len(w, pad));
    let mut out = Buf::scratch(ho * wo * ch);
    team::chunks_mut(&mut out, wo * ch, |oh, row| {
        depthwise_row(x, (h, w, ch, pad), weight, bias, oh, row);
    });
    (out, ho, wo)
}

multiversion! {
    fn depthwise_row(
        x: &[f32],
        shape: (usize, usize, usize, usize),
        weight: &[f32],
        bias: &[f32],
        oh: usize,
        row: &mut [f32],
    ) {
        let (h, w, ch, pad) = shape;
        for (ow, cell) in row.chunks_exact_mut(ch).enumerate() {
            cell.copy_from_slice(bias);
            for i in 0..3 {
                for j in 0..3 {
                    let (ih, iw) = (2 * oh + i, 2 * ow + j);
                    if ih < pad || iw < pad || ih - pad >= h || iw - pad >= w {
                        continue;
                    }
                    let input = &x[((ih - pad) * w + (iw - pad)) * ch..][..ch];
                    let taps = &weight[(i * 3 + j) * ch..][..ch];
                    for ((o, wt), v) in cell.iter_mut().zip(taps).zip(input) {
                        *o += wt * v;
                    }
                }
            }
        }
    }
}

struct Block {
    norm_ff1: Norm,
    ff1: (Linear, Linear),
    norm_att: Norm,
    /// q, k, and v stacked into one `3d x d` layer: one quantization, one
    /// product. With rotary positions, only q and k (`2d x d`).
    qkv: Linear,
    o: Linear,
    positions: Positions,
    norm_conv: Norm,
    pw1: Linear,
    /// Depthwise conv weights as `[tap][channel]`, so a tap is one contiguous row.
    dw_by_tap: Vec<f32>,
    conv_norm: ConvNorm,
    pw2: Linear,
    norm_ff2: Norm,
    ff2: (Linear, Linear),
    norm_out: Norm,
}

impl Block {
    fn load(t: &Tensors, c: &Config, index: usize) -> Result<Self, String> {
        let p = format!("{}encoder.layers.{index}", c.prefix);
        let (d, g) = (c.hidden, c.group);
        let layer =
            |name: &str, out: usize, inp: usize| linear(t, &format!("{p}.{name}"), out, inp, g);
        let weight = t.f32_len(&format!("{p}.conv.norm.weight"), d)?;
        let bias = t.f32_len(&format!("{p}.conv.norm.bias"), d)?;
        let dw_bias = match t.has(&format!("{p}.conv.depthwise_conv.bias")) {
            true => t.f32_len(&format!("{p}.conv.depthwise_conv.bias"), d)?,
            false => vec![0.0; d],
        };
        let conv_norm = match c.conv_norm {
            ConvNormKind::Batch => {
                let mean = t.f32_len(&format!("{p}.conv.norm.running_mean"), d)?;
                let var = t.f32_len(&format!("{p}.conv.norm.running_var"), d)?;
                let (scale, shift) = folded_batch_norm(&weight, &bias, &mean, &var, &dw_bias);
                ConvNorm::Batch { scale, shift }
            }
            ConvNormKind::Layer => ConvNorm::Layer {
                dw_bias,
                weight,
                bias,
            },
        };
        Ok(Self {
            norm_ff1: Norm::load(t, &format!("{p}.norm_feed_forward1"), d)?.with_eps(c.norm_eps),
            ff1: (
                layer("feed_forward1.linear1", c.ff, d)?,
                layer("feed_forward1.linear2", d, c.ff)?,
            ),
            norm_att: Norm::load(t, &format!("{p}.norm_self_att"), d)?.with_eps(c.norm_eps),
            qkv: match c.position {
                Position::Relative | Position::Rotary { .. } => stacked(
                    t,
                    &["q_proj", "k_proj", "v_proj"].map(|n| format!("{p}.self_attn.{n}")),
                    (d, d),
                    g,
                )?,
                Position::RotaryInput { .. } => stacked(
                    t,
                    &["q_proj", "k_proj"].map(|n| format!("{p}.self_attn.{n}")),
                    (d, d),
                    g,
                )?,
            },
            o: layer("self_attn.o_proj", d, d)?,
            positions: match c.position {
                Position::Relative => Positions::Relative {
                    rel_k: layer("self_attn.relative_k_proj", d, d)?,
                    rel_cache: RelCache::default(),
                    bias_u: t.f32_len(&format!("{p}.self_attn.bias_u"), d)?,
                    bias_v: t.f32_len(&format!("{p}.self_attn.bias_v"), d)?,
                },
                Position::RotaryInput { base } => Positions::RotaryInput {
                    rotary: Rotary::halves(d / c.heads, base),
                    v: layer("self_attn.v_proj", d, d)?,
                },
                Position::Rotary { base } => Positions::Rotary {
                    rotary: Rotary::halves(d / c.heads, base),
                },
            },
            norm_conv: Norm::load(t, &format!("{p}.norm_conv"), d)?.with_eps(c.norm_eps),
            pw1: layer("conv.pointwise_conv1", 2 * d, d)?,
            dw_by_tap: {
                let kernel = c.conv_kernel;
                let dw = t.f32_len(&format!("{p}.conv.depthwise_conv.weight"), d * kernel)?;
                (0..kernel * d)
                    .map(|i| dw[(i % d) * kernel + i / d])
                    .collect()
            },
            conv_norm,
            pw2: layer("conv.pointwise_conv2", d, d)?,
            norm_ff2: Norm::load(t, &format!("{p}.norm_feed_forward2"), d)?.with_eps(c.norm_eps),
            ff2: (
                layer("feed_forward2.linear1", c.ff, d)?,
                layer("feed_forward2.linear2", d, c.ff)?,
            ),
            norm_out: Norm::load(t, &format!("{p}.norm_out"), d)?.with_eps(c.norm_eps),
        })
    }

    fn feed_forward((up, down): &(Linear, Linear), x: &[f32], frames: usize) -> Buf {
        let mut hidden = up.forward(x, frames);
        team::chunks_mut(&mut hidden, 4096, |_, values| activation::silu(values));
        down.forward(&hidden, frames)
    }

    /// One block. The GPU encoder runs the same steps in `Encoder::record`.
    fn forward(&self, x: &mut Buf, frames: usize, c: &Config) {
        let (ff_w0, ff_w1) = c.ff_residual;
        let (conv_w0, conv_w1) = c.conv_residual;
        let ff = Self::feed_forward(&self.ff1, &self.norm_ff1.apply(x, c.hidden), frames);
        scale(x, ff_w0);
        let normed = self.norm_att.residual(x, &ff, ff_w1);
        let att = self.attention(&normed, frames, c);
        let normed = self.norm_conv.residual(x, &att, 1.0);
        let conv = self.convolution(&normed, frames, c);
        scale(x, conv_w0);
        let normed = self.norm_ff2.residual(x, &conv, conv_w1);
        let ff = Self::feed_forward(&self.ff2, &normed, frames);
        scale(x, ff_w0);
        *x = self.norm_out.residual(x, &ff, ff_w1);
    }

    /// Multi-head attention with Transformer-XL relative positions: one task
    /// per head and block of `ATTENTION_ROWS` queries. The products are
    /// `fgemm::matmul_into`, so every value is the one a whole-head product
    /// gives; a change in a sum's order changes the text of some clips.
    fn attention(&self, x: &[f32], t: usize, c: &Config) -> Buf {
        let (rel_k, rel_cache, all_bias_u, all_bias_v) = match &self.positions {
            Positions::Relative {
                rel_k,
                rel_cache,
                bias_u,
                bias_v,
            } => (rel_k, rel_cache, bias_u, bias_v),
            Positions::RotaryInput { rotary, v } => {
                return self.rotary_input_attention(x, t, c, rotary, v)
            }
            Positions::Rotary { rotary } => return self.rotary_attention(x, t, c, rotary),
        };
        let (d, heads) = (c.hidden, c.heads);
        let hd = d / heads;
        let qkv = self.qkv.forward(x, t);
        // Per head, from the pool on the calling thread: K as rows (t x hd)
        // and V transposed (hd x t), so each product reads whole rows.
        let mut keys = Buf::scratch(heads * 2 * t * hd);
        team::chunks_mut(&mut keys, 2 * t * hd, |h, head| {
            let (k, v_t) = head.split_at_mut(t * hd);
            for j in 0..t {
                let row = &qkv[j * 3 * d + h * hd..];
                k[j * hd..(j + 1) * hd].copy_from_slice(&row[d..][..hd]);
                for (e, value) in row[2 * d..][..hd].iter().enumerate() {
                    v_t[e * t + j] = *value;
                }
            }
        });
        let positions = rel_cache.rows(t, (d, heads), Store::owned, |table, rows| {
            rel_k.forward(table, rows)
        });
        let scale = 1.0 / (hd as f32).sqrt();
        let chunk = ATTENTION_ROWS.min(t);
        let chunks = t.div_ceil(chunk);
        let mut context = Buf::scratch(t * d);
        let out = team::SendPtr::new(&mut context);
        team::for_each(heads * chunks, |task| {
            let (h, block) = (task / chunks, task % chunks);
            let rows = block * chunk..((block + 1) * chunk).min(t);
            let m = rows.len();
            let (k, v_t) = keys[h * 2 * t * hd..][..2 * t * hd].split_at(t * hd);
            // Query i reads relative positions i down to i-(t-1): t of the
            // take's 2t-1, from t-1-i. The block's rows read t+m-1 of them,
            // from t-rows.end.
            let width = t + m - 1;
            let first = positions.first(t) + t - rows.end;
            let band = &positions.head(h, hd)[first * hd..][..width * hd];
            let (bias_u, bias_v) = (&all_bias_u[h * hd..][..hd], &all_bias_v[h * hd..][..hd]);
            SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                let need = m * (3 * hd + t + width);
                if scratch.len() < need {
                    scratch.resize(need, 0.0);
                }
                let (qu, rest) = scratch.split_at_mut(m * hd);
                let (qv, rest) = rest.split_at_mut(m * hd);
                let (scores, rest) = rest.split_at_mut(m * t);
                let (shifted, rest) = rest.split_at_mut(m * width);
                let head_out = &mut rest[..m * hd];
                for ((u, v), i) in qu
                    .chunks_exact_mut(hd)
                    .zip(qv.chunks_exact_mut(hd))
                    .zip(rows.clone())
                {
                    let q = &qkv[i * 3 * d + h * hd..][..hd];
                    for e in 0..hd {
                        u[e] = q[e] + bias_u[e];
                        v[e] = q[e] + bias_v[e];
                    }
                }
                fgemm::matmul_into(qu, m, hd, k, t, scores);
                fgemm::matmul_into(qv, m, hd, band, width, shifted);
                for (row, i) in scores.chunks_exact_mut(t).zip(rows.clone()) {
                    let local = i - rows.start;
                    let position = &shifted[local * width + (rows.end - 1 - i)..][..t];
                    let mut max = f32::NEG_INFINITY;
                    for (value, p) in row.iter_mut().zip(position) {
                        *value = (*value + p) * scale;
                        max = max.max(*value);
                    }
                    let keys = c.attention.keys(i, t);
                    if keys != (0..t) {
                        // Keys outside the context get no weight; the
                        // largest score is the largest of the ones kept.
                        row[..keys.start].fill(f32::NEG_INFINITY);
                        row[keys.end..].fill(f32::NEG_INFINITY);
                        max = row[keys.clone()]
                            .iter()
                            .fold(f32::NEG_INFINITY, |m, v| m.max(*v));
                    }
                    row.iter_mut().for_each(|value| *value -= max);
                    activation::exp(row);
                    let inv = 1.0 / row.iter().sum::<f32>();
                    row.iter_mut().for_each(|value| *value *= inv);
                }
                fgemm::matmul_into(scores, m, t, v_t, hd, head_out);
                for (part, i) in head_out.chunks_exact(hd).zip(rows.clone()) {
                    // SAFETY: row i, head h belongs to this task only; `context` outlives the step.
                    unsafe { out.slice_mut(i * d + h * hd, hd) }.copy_from_slice(part);
                }
            });
        });
        self.o.forward(&context, t)
    }

    /// Softmax attention with rotary positions (GigaAM): the input, rotated
    /// per head, gives the queries and keys; the input as it is, the values.
    fn rotary_input_attention(
        &self,
        x: &[f32],
        t: usize,
        c: &Config,
        rotary: &Rotary,
        v: &Linear,
    ) -> Buf {
        let (d, heads) = (c.hidden, c.heads);
        let mut rotated = Buf::scratch(t * d);
        rotated.copy_from_slice(x);
        team::chunks_mut(&mut rotated, d, |i, row| {
            rotary.apply(row, (1, d), (heads, d / heads), i)
        });
        let qk = self.qkv.forward(&rotated, t);
        let values = v.forward(x, t);
        let out = crate::layers::attention(
            (&qk, 2 * d),
            (&qk[d..], 2 * d),
            (&values, d),
            (t, t),
            (d, heads, heads),
            false,
        );
        self.o.forward(&out, t)
    }

    /// Softmax attention with rotary positions on the projected queries and
    /// keys (LASR).
    fn rotary_attention(&self, x: &[f32], t: usize, c: &Config, rotary: &Rotary) -> Buf {
        let (d, heads) = (c.hidden, c.heads);
        let mut qkv = self.qkv.forward(x, t);
        team::chunks_mut(&mut qkv, 3 * d, |i, row| {
            rotary.apply(&mut row[..2 * d], (1, 2 * d), (2 * heads, d / heads), i)
        });
        let out = crate::layers::attention(
            (&qkv, 3 * d),
            (&qkv[d..], 3 * d),
            (&qkv[2 * d..], 3 * d),
            (t, t),
            (d, heads, heads),
            false,
        );
        self.o.forward(&out, t)
    }

    fn convolution(&self, x: &[f32], t: usize, c: &Config) -> Buf {
        let d = c.hidden;
        let kernel = c.conv_kernel;
        let wide = self.pw1.forward(x, t);
        let mut gated = Buf::scratch(t * d);
        team::chunks_mut(&mut gated, d, |index, out| {
            let row = &wide[index * 2 * d..(index + 1) * 2 * d];
            out.copy_from_slice(&row[d..]);
            activation::sigmoid(out);
            for (o, value) in out.iter_mut().zip(&row[..d]) {
                *o *= value;
            }
        });
        // Accumulated with +=: the one buffer here that must start at zero.
        let mut mixed = Buf::zeroed(t * d);
        team::chunks_mut(&mut mixed, d, |i, out| {
            time_conv_row(self, &gated, (t, kernel, c.conv_context.0), i, out);
            activation::silu(out);
        });
        self.pw2.forward(&mixed, t)
    }
}

multiversion! {
    /// `out` starts at zero.
    fn time_conv_row(block: &Block, gated: &[f32], t_kernel: (usize, usize, usize), i: usize, out: &mut [f32]) {
        // `pad`: the frames before frame i that tap 0 reads (`conv_context.0`).
        let (t, kernel, pad) = t_kernel;
        let d = out.len();
        for k in 0..kernel {
            let src = i + k;
            if src < pad || src - pad >= t {
                continue;
            }
            let taps = &block.dw_by_tap[k * d..(k + 1) * d];
            for ((o, w), g) in out.iter_mut().zip(taps).zip(&gated[(src - pad) * d..][..d]) {
                *o += w * g;
            }
        }
        match &block.conv_norm {
            ConvNorm::Batch { scale, shift } => {
                for ((o, scale), shift) in out.iter_mut().zip(scale).zip(shift) {
                    *o = *o * scale + shift;
                }
            }
            ConvNorm::Layer { dw_bias, weight, bias } => {
                out.iter_mut().zip(dw_bias).for_each(|(o, b)| *o += b);
                let mean = out.iter().sum::<f32>() / d as f32;
                let var = out.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
                let inv = 1.0 / (var + LAYER_NORM_EPS).sqrt();
                for ((o, w), b) in out.iter_mut().zip(weight).zip(bias) {
                    *o = (*o - mean) * inv * w + b;
                }
            }
        }
    }
}

enum Positions {
    Relative {
        rel_k: Linear,
        /// `rel_k` applied to the position table, kept between takes.
        rel_cache: RelCache,
        bias_u: Vec<f32>,
        bias_v: Vec<f32>,
    },
    /// The values' projection is apart: it reads the input unrotated.
    RotaryInput { rotary: Rotary, v: Linear },
    /// q, k, and v stacked; q and k rotated after the projection.
    Rotary { rotary: Rotary },
}

/// Nothing for 1, so a Conformer's sums keep their order.
fn scale(x: &mut [f32], w: f32) {
    if w != 1.0 {
        team::chunks_mut(x, 4096, |_, values| values.iter_mut().for_each(|v| *v *= w));
    }
}

/// The conv module's norm after the depthwise conv.
enum ConvNorm {
    /// Batch norm with its statistics (and the conv bias) folded in.
    Batch { scale: Vec<f32>, shift: Vec<f32> },
    /// Layer norm over the channels of a frame, after the conv bias.
    Layer {
        dw_bias: Vec<f32>,
        weight: Vec<f32>,
        bias: Vec<f32>,
    },
}

const LAYER_NORM_EPS: f32 = 1e-5;

/// NeMo's speaker kernel at the input of layer 0 (`SpeakerKernelMixin`),
/// for one speaker: `x += W3 relu(W0 x + b0) + b3`, then the background
/// kernel of a zero input, a constant row.
struct SpeakerKernel {
    up: Dense,
    down: Dense,
    background: Vec<f32>,
}

impl SpeakerKernel {
    fn load(t: &Tensors, c: &Config) -> Result<Option<Self>, String> {
        if !c.speaker_kernel {
            return Ok(None);
        }
        let d = c.hidden;
        let p = format!("{}encoder", c.prefix);
        let up = Dense::load_cpu(t, &[(&format!("{p}.spk_kernel.0.0"), d)], d)?;
        let down = Dense::load_cpu(t, &[(&format!("{p}.spk_kernel.0.3"), d)], d)?;
        let mut hidden = t.f32_len(&format!("{p}.bg_spk_kernel.0.0.bias"), d)?;
        hidden.iter_mut().for_each(|v| *v = v.max(0.0));
        let background = Dense::load_cpu(t, &[(&format!("{p}.bg_spk_kernel.0.3"), d)], d)?
            .forward(&hidden, 1)
            .to_vec();
        Ok(Some(Self {
            up,
            down,
            background,
        }))
    }

    fn apply(&self, x: &mut Buf, t: usize) {
        let mut hidden = self.up.forward(x, t);
        team::chunks_mut(&mut hidden, 4096, |_, values| {
            values.iter_mut().for_each(|v| *v = v.max(0.0))
        });
        let kernel = self.down.forward(&hidden, t);
        let d = self.background.len();
        team::chunks_mut(x, d, |i, row| {
            for ((o, k), b) in row
                .iter_mut()
                .zip(&kernel[i * d..][..d])
                .zip(&self.background)
            {
                *o += k + b;
            }
        });
    }
}

/// The language prompt's MLP: `W2 relu(W0 [x; one_hot] + b0) + b2`.
struct Prompt {
    config: PromptConfig,
    first: Dense,
    second: Dense,
}

impl Prompt {
    fn load(t: &Tensors, c: &Config) -> Result<Option<Self>, String> {
        let Some(config) = c.prompt.clone() else {
            return Ok(None);
        };
        let (d, hidden) = (c.hidden, config.hidden);
        Ok(Some(Self {
            first: Dense::load_cpu(t, &[("prompt_kernel.0", hidden)], d + config.count)?,
            second: Dense::load_cpu(t, &[("prompt_kernel.2", d)], hidden)?,
            config,
        }))
    }

    fn apply(&self, x: &[f32], t: usize, language: Option<&str>) -> Buf {
        let d = x.len() / t.max(1);
        let count = self.config.count;
        let index = self.config.index(language);
        let mut joined = Buf::zeroed(t * (d + count));
        team::chunks_mut(&mut joined, d + count, |i, row| {
            row[..d].copy_from_slice(&x[i * d..][..d]);
            row[d + index] = 1.0;
        });
        let mut hidden = self.first.forward(&joined, t);
        team::chunks_mut(&mut hidden, 4096, |_, values| {
            values.iter_mut().for_each(|v| *v = v.max(0.0))
        });
        self.second.forward(&hidden, t)
    }
}

/// Positions every take reaches without growing the cache: 30 s of 80 ms frames.
pub const POSITION_REACH: usize = 375;

/// Projected relative positions for `reach` down to `-reach`, one row each.
/// A row depends only on its position, never on the take's length or on
/// other rows, so one table serves every take up to `reach + 1` frames.
/// The table of the load stays. A longer take builds a table apart, in
/// `grown`, and `release_positions` drops it: about 148 MB for a take over
/// 30 s, and most takes never need it again.
#[derive(Default)]
struct RelCache {
    table: RwLock<Option<Arc<RelRows>>>,
    grown: Arc<GrownRows>,
}

type GrownRows = RwLock<Option<Arc<RelRows>>>;

/// The tables that grew since the last `release_positions`, of every model.
static GROWN: Mutex<Vec<Weak<GrownRows>>> = Mutex::new(Vec::new());

/// Drops the position tables that long takes built. A take that needs one
/// builds it again; a row depends only on its position, so the text stays.
pub(crate) fn release_positions() {
    for grown in GROWN.lock().drain(..).filter_map(|grown| grown.upgrade()) {
        *grown.write() = None;
    }
}

struct RelRows {
    reach: usize,
    /// Positions `reach` down to `-reach`, per head: `[head][position][hd]`,
    /// so the positions of a block of queries are consecutive rows.
    rows: Store<f32>,
}

impl RelRows {
    /// Head `h`'s rows, one per position.
    fn head(&self, h: usize, hd: usize) -> &[f32] {
        let size = (2 * self.reach + 1) * hd;
        &self.rows[h * size..][..size]
    }

    /// The table column of position t-1, the first of a take of `t` frames.
    fn first(&self, t: usize) -> usize {
        self.reach - (t - 1)
    }
}

impl RelCache {
    /// A table for `t` frames, built first when `t` frames reach past both.
    /// `keep` stores a new table: `Store::weights` at load, `Store::owned`
    /// after.
    fn rows(
        &self,
        t: usize,
        (d, heads): (usize, usize),
        keep: fn(Vec<f32>) -> Store<f32>,
        project: impl Fn(&[f32], usize) -> Buf,
    ) -> Arc<RelRows> {
        let fits = |rows: &&Arc<RelRows>| rows.reach + 1 >= t;
        if let Some(rows) = self.table.read().as_ref().filter(fits) {
            return Arc::clone(rows);
        }
        if let Some(rows) = self.grown.read().as_ref().filter(fits) {
            return Arc::clone(rows);
        }
        // One builder at a time: a second one finds the table of the first.
        let mut table = self.table.write();
        if let Some(rows) = table.as_ref().filter(fits) {
            return Arc::clone(rows);
        }
        if let Some(rows) = self.grown.read().as_ref().filter(fits) {
            return Arc::clone(rows);
        }
        let largest = self
            .grown
            .read()
            .as_ref()
            .or(table.as_ref())
            .map_or(0, |r| r.reach);
        let reach = (t - 1).max(POSITION_REACH).max(2 * largest);
        let count = 2 * reach + 1;
        let projected = project(&relative_positions(reach + 1, d), count);
        let hd = d / heads;
        let mut by_head = Buf::scratch(count * d);
        team::chunks_mut(&mut by_head, count * hd, |h, head| {
            for (row, out) in projected.chunks_exact(d).zip(head.chunks_exact_mut(hd)) {
                out.copy_from_slice(&row[h * hd..][..hd]);
            }
        });
        let rows = Arc::new(RelRows {
            reach,
            rows: keep(by_head.into_vec()),
        });
        if table.is_none() {
            *table = Some(Arc::clone(&rows));
        } else {
            *self.grown.write() = Some(Arc::clone(&rows));
            GROWN.lock().push(Arc::downgrade(&self.grown));
        }
        rows
    }
}

/// Sinusoidal table for relative positions t-1 down to -(t-1), sin and cos interleaved.
fn relative_positions(t: usize, d: usize) -> Vec<f32> {
    let rows = 2 * t - 1;
    let inv_freq: Vec<f32> = (0..d / 2)
        .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / d as f32))
        .collect();
    let mut table = vec![0.0f32; rows * d];
    team::chunks_mut(&mut table, d, |p, row| {
        let position = (t - 1) as f32 - p as f32;
        for (i, freq) in inv_freq.iter().enumerate() {
            let angle = position * freq;
            row[2 * i] = angle.sin();
            row[2 * i + 1] = angle.cos();
        }
    });
    table
}

pub struct Parakeet {
    pub config: Config,
    subsampling: Subsampler,
    blocks: Vec<Block>,
    head: Head,
    /// The speech head some checkpoints carry (Redux: `vad_head.*`).
    vad: Option<VadHead>,
    speaker_kernel: Option<SpeakerKernel>,
    prompt: Option<Prompt>,
    /// LASR's norm after the last block.
    out_norm: Option<Norm>,
    /// The conformer, when this model was loaded on the GPU.
    #[cfg(feature = "gpu")]
    gpu_encoder: Option<crate::gpu::encoder::Encoder>,
}

/// Speech probability per 80 ms frame from the subsampler output, as
/// Kestrel runs it: a 1x1 conv to 128, SiLU, a width-5 conv, SiLU, a 1x1
/// conv to one logit, sigmoid. The subsampler is local, so this costs a
/// fraction of one encoder layer.
struct VadHead {
    proj: Dense,
    ctx: Dense,
    out: Dense,
}

/// Seconds per subsampled frame of the speech head.
const FRAME_SECONDS: f32 = 0.08;
/// Shorter gaps between speech join; shorter speech does not count.
const MIN_GAP_SECONDS: f32 = 0.1;
const MIN_SPEECH_SECONDS: f32 = 0.1;

impl VadHead {
    const KERNEL: usize = 5;

    fn load(t: &Tensors, c: &Config) -> Result<Option<Self>, String> {
        if !t.has("vad_head.proj.weight") {
            return Ok(None);
        }
        let hidden = 128;
        Ok(Some(Self {
            proj: Dense::load_cpu(t, &[("vad_head.proj", hidden)], c.hidden)?,
            ctx: Dense::load_cpu(t, &[("vad_head.ctx", hidden)], hidden * Self::KERNEL)?,
            out: Dense::load_cpu(t, &[("vad_head.out", 1)], hidden)?,
        }))
    }

    /// Whether `t` subsampled frames hold speech: a run of frames with a
    /// probability of 0.5 or more lasting `MIN_SPEECH_SECONDS`, after gaps
    /// under `MIN_GAP_SECONDS` are bridged (Kestrel's `speech_regions`).
    fn speech(&self, x: &[f32], t: usize) -> bool {
        let mut h = self.proj.forward(x, t);
        activation::silu(&mut h);
        let hidden = self.proj.out_features;
        let cols =
            crate::layers::conv_columns(&h, (t, hidden), (Self::KERNEL, 1, Self::KERNEL / 2), t);
        let mut h = self.ctx.forward(&cols, t);
        activation::silu(&mut h);
        let logits = self.out.forward(&h, t);
        // sigmoid(v) >= 0.5 exactly when v >= 0.
        let speaking: Vec<bool> = logits.iter().map(|v| *v >= 0.0).collect();
        let mut regions: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < t {
            if !speaking[i] {
                i += 1;
                continue;
            }
            let start = i;
            while i < t && speaking[i] {
                i += 1;
            }
            match regions.last_mut() {
                Some(last) if (start - last.1) as f32 * FRAME_SECONDS < MIN_GAP_SECONDS => {
                    last.1 = i
                }
                _ => regions.push((start, i)),
            }
        }
        regions
            .iter()
            .any(|(start, end)| (end - start) as f32 * FRAME_SECONDS >= MIN_SPEECH_SECONDS)
    }
}

enum Head {
    /// Transducer (TDT, or RNNT without durations): LSTM prediction
    /// network and joint network.
    Transducer(Box<Transducer>),
    /// One projection to token logits per frame, greedy with repeats merged.
    Ctc(Dense),
    /// No head: the encoder output is the result.
    Encoder,
}

impl Parakeet {
    pub fn load(t: &Tensors, config: Config) -> Result<Self, String> {
        let c = &config;
        let blocks = team::map(c.layers, |index| Block::load(t, c, index))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let head = match c.head {
            HeadKind::Ctc => Head::Ctc(dense(t, "ctc_head", c.vocab, c.hidden)?),
            HeadKind::Transducer | HeadKind::Rnnt => {
                Head::Transducer(Box::new(Transducer::load(t, c)?))
            }
            HeadKind::Encoder => Head::Encoder,
        };
        #[cfg(feature = "gpu")]
        let gpu_encoder = if t.device() == crate::Device::Gpu && !c.gpu_shape() {
            crate::gpu::log_fallback(
                "this encoder shape (masked attention, a causal conv, a layer-norm conv) runs on the CPU",
            );
            None
        } else if t.device() == crate::Device::Gpu {
            match gpu_encoder(t, &blocks, c) {
                Ok(encoder) => Some(encoder),
                Err(error) => {
                    crate::gpu::log_fallback(&error);
                    None
                }
            }
        } else {
            None
        };
        Ok(Self {
            subsampling: Subsampler::load(t, c)?,
            blocks,
            head,
            vad: VadHead::load(t, c)?,
            speaker_kernel: SpeakerKernel::load(t, c)?,
            prompt: Prompt::load(t, c)?,
            out_norm: match c.out_norm {
                true => Some(
                    Norm::load(t, &format!("{}encoder.out_norm", c.prefix), c.hidden)?
                        .with_eps(c.norm_eps),
                ),
                false => None,
            },
            config,
            #[cfg(feature = "gpu")]
            gpu_encoder,
        })
    }
}

/// Frames the first GPU position table fits. About 41 s of audio.
#[cfg(feature = "gpu")]
const GPU_REACH: usize = 511;

/// Shorter takes stay on the CPU. The GPU tile does not fill the device.
#[cfg(feature = "gpu")]
const GPU_MIN_FRAMES: usize = 48;

/// Keeps products on the CPU until dropped.
#[cfg(feature = "gpu")]
struct HostProducts;

#[cfg(feature = "gpu")]
impl HostProducts {
    fn new() -> Self {
        crate::gpu::use_host_products(true);
        Self
    }
}

#[cfg(feature = "gpu")]
impl Drop for HostProducts {
    fn drop(&mut self) {
        crate::gpu::use_host_products(false);
    }
}

#[cfg(feature = "gpu")]
fn gpu_mat(linear: &Linear) -> Result<crate::gpu::encoder::Mat, String> {
    let layer = match linear {
        Linear::Ternary(layer) => layer.gpu_layer(),
        Linear::Dense(layer) => layer.gpu_layer(),
    };
    layer
        .map(crate::gpu::encoder::Mat::from_layer)
        .ok_or_else(|| "encoder layer has no GPU weights".to_string())
}

#[cfg(feature = "gpu")]
fn gpu_encoder(
    t: &Tensors,
    blocks: &[Block],
    c: &Config,
) -> Result<crate::gpu::encoder::Encoder, String> {
    let mut layers = Vec::with_capacity(blocks.len());
    for (index, block) in blocks.iter().enumerate() {
        let p = format!("{}encoder.layers.{index}", c.prefix);
        let mut norms = Vec::with_capacity(5);
        for name in [
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ] {
            let full = format!("{p}.{name}");
            let weight = t.f32_len(&format!("{full}.weight"), c.hidden)?;
            let bias_name = format!("{full}.bias");
            let bias = if t.has(&bias_name) {
                t.f32_len(&bias_name, c.hidden)?
            } else {
                vec![0.0; c.hidden]
            };
            norms.push((weight, bias));
        }
        layers.push(crate::gpu::encoder::LayerUpload {
            norms,
            mats: vec![
                gpu_mat(&block.ff1.0)?,
                gpu_mat(&block.ff1.1)?,
                gpu_mat(&block.qkv)?,
                gpu_mat(&block.o)?,
                gpu_mat(&block.pw1)?,
                gpu_mat(&block.pw2)?,
                gpu_mat(&block.ff2.0)?,
                gpu_mat(&block.ff2.1)?,
            ],
            rel_k: match &block.positions {
                Positions::Relative { rel_k, .. } => gpu_mat(rel_k)?,
                Positions::RotaryInput { .. } | Positions::Rotary { .. } => {
                    return Err("rotary attention runs on the CPU".into())
                }
            },
            bias_u: match &block.positions {
                Positions::Relative { bias_u, .. } => bias_u.clone(),
                Positions::RotaryInput { .. } | Positions::Rotary { .. } => {
                    return Err("rotary attention runs on the CPU".into())
                }
            },
            bias_v: match &block.positions {
                Positions::Relative { bias_v, .. } => bias_v.clone(),
                Positions::RotaryInput { .. } | Positions::Rotary { .. } => {
                    return Err("rotary attention runs on the CPU".into())
                }
            },
            dw: block.dw_by_tap.clone(),
            bn_scale: match &block.conv_norm {
                ConvNorm::Batch { scale, .. } => scale.clone(),
                ConvNorm::Layer { .. } => return Err("a layer-norm conv runs on the CPU".into()),
            },
            bn_shift: match &block.conv_norm {
                ConvNorm::Batch { shift, .. } => shift.clone(),
                ConvNorm::Layer { .. } => return Err("a layer-norm conv runs on the CPU".into()),
            },
        });
    }
    let sine = relative_positions(GPU_REACH + 1, c.hidden);
    crate::gpu::encoder::Encoder::build(
        crate::gpu::encoder::Cfg {
            d: u32::try_from(c.hidden).map_err(|_| "hidden size does not fit".to_string())?,
            f: u32::try_from(c.ff).map_err(|_| "feed-forward size does not fit".to_string())?,
            heads: u32::try_from(c.heads).map_err(|_| "head count does not fit".to_string())?,
            kernel: u32::try_from(c.conv_kernel)
                .map_err(|_| "conv kernel does not fit".to_string())?,
            input_scale: c.input_scale,
            min_reach: POSITION_REACH as u32,
            ff_w1: c.ff_residual.1,
            conv_w1: c.conv_residual.1,
            norm_eps: c.norm_eps,
        },
        layers,
        &sine,
        GPU_REACH as u32,
    )
}

impl Parakeet {
    /// Encoder output after the 24 blocks: `frames x hidden`.
    pub fn encode(&self, features: &[f32], frames: usize) -> (Buf, usize) {
        let (x, t) = self.subsampling.forward(features, frames, self.config.mels);
        self.encode_subsampled(x, t, None)
    }

    /// `encode`, or `None` when the model's speech head finds no speech
    /// (a model without one always encodes). After `cancel` the output is
    /// not valid: the caller must look at the flag.
    pub fn encode_speech(
        &self,
        features: &[f32],
        frames: usize,
        cancel: Option<&Cancel>,
    ) -> Option<(Buf, usize)> {
        let (x, t) = self.subsampling.forward(features, frames, self.config.mels);
        // A take too short for one frame of this subsampler holds no speech.
        if t == 0 || self.vad.as_ref().is_some_and(|vad| !vad.speech(&x, t)) {
            return None;
        }
        Some(self.encode_subsampled(x, t, cancel))
    }

    fn encode_subsampled(&self, mut x: Buf, t: usize, cancel: Option<&Cancel>) -> (Buf, usize) {
        // No frames: the attention has no rows to split into tasks.
        if t == 0 {
            return (x, 0);
        }
        // A GPU result returns before the speaker kernel, cancel, and out_norm below.
        // `Encoder::record` does not run those steps. `gpu_shape` keeps them off.
        #[cfg(feature = "gpu")]
        if let Some(encoder) = &self.gpu_encoder {
            if t >= GPU_MIN_FRAMES && crate::gpu::usable() {
                let d = self.config.hidden;
                match encoder.encode(&x, t, |reach| relative_positions(reach + 1, d)) {
                    Ok(encoded) => {
                        crate::gpu::worked();
                        return (encoded, t);
                    }
                    Err(error) => crate::gpu::failed(&error),
                }
            }
        }
        // A short take, or a GPU failure, uses the CPU products.
        #[cfg(feature = "gpu")]
        let _host = HostProducts::new();
        let c = &self.config;
        if c.input_scale != 1.0 {
            let scale = c.input_scale;
            team::chunks_mut(&mut x, c.hidden, |_, row| {
                row.iter_mut().for_each(|v| *v *= scale)
            });
        }
        if let Some(kernel) = &self.speaker_kernel {
            kernel.apply(&mut x, t);
        }
        for block in &self.blocks {
            if cancel.is_some_and(Cancel::is_cancelled) {
                break;
            }
            block.forward(&mut x, t, c);
        }
        if let Some(norm) = &self.out_norm {
            x = norm.apply(&x, c.hidden);
        }
        (x, t)
    }

    /// The encoder output with language `code`'s prompt, for a
    /// prompt-conditioned model (else unchanged): `None` or a code the
    /// model does not list lets it find the language.
    pub fn prompted(&self, encoded: Buf, t: usize, code: Option<&str>) -> Buf {
        match &self.prompt {
            Some(prompt) => prompt.apply(&encoded, t, code),
            None => encoded,
        }
    }

    /// Fills every layer's position table up front, so the first take does not wait.
    pub fn warm_positions(&self) {
        // Project these rows on the CPU. The GPU product changes the sum order.
        #[cfg(feature = "gpu")]
        let _host = HostProducts::new();
        let shape = (self.config.hidden, self.config.heads);
        for block in &self.blocks {
            if let Positions::Relative {
                rel_k, rel_cache, ..
            } = &block.positions
            {
                rel_cache.rows(POSITION_REACH + 1, shape, Store::weights, |table, rows| {
                    rel_k.forward(table, rows)
                });
            }
        }
    }

    /// Tokens of the encoder output `t x hidden`, each with the encoder
    /// frame it was emitted at. `dictionary` terms replace words they sound
    /// like (TDT only); `banned` tokens are never picked. `cancel` gives no
    /// tokens (transducer heads only).
    pub fn decode(
        &self,
        encoded: &[f32],
        t: usize,
        dictionary: Option<&Dictionary>,
        banned: &[usize],
        cancel: Option<&Cancel>,
    ) -> (Vec<usize>, Vec<usize>) {
        let c = &self.config;
        match &self.head {
            Head::Transducer(tdt) => {
                let projected = tdt.encoder_projector.forward(encoded, t);
                tdt.decode(c, &projected, t, dictionary, banned, cancel)
            }
            Head::Ctc(head) => crate::models::parts::ctc::greedy(
                &head.forward(encoded, t),
                c.vocab,
                c.blank,
                banned,
            ),
            Head::Encoder => (Vec::new(), Vec::new()),
        }
    }
}
