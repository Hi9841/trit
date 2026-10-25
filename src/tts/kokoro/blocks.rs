//! The style-conditioned blocks of StyleTTS 2 and iSTFTNet: AdaIN and
//! AdaLayerNorm (a norm whose scale and shift come from the voice's style
//! vector), the AdaIN residual block with optional 2x upsampling, and the
//! Snake residual block of the generator. Signals are time-major
//! (`frames x channels`).

use crate::cpu::{buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::models::layers::linear;
use crate::models::parts::conv::{Conv1d, DepthwiseUpsample};

/// Epsilon of the instance and layer norms (PyTorch's default).
const EPS: f64 = 1e-5;
/// The leaky ReLU slope of the AdaIN residual blocks.
const SLOPE: f32 = 0.2;

/// `(1 + gamma)` and `beta` of one norm for one style.
fn scale_shift(fc: &Dense, style: &[f32], channels: usize) -> (Vec<f32>, Vec<f32>) {
    let h = fc.forward(style, 1);
    let scale = h[..channels].iter().map(|g| 1.0 + g).collect();
    (scale, h[channels..2 * channels].to_vec())
}

/// AdaIN: instance norm over time per channel (no learned affine: the
/// checkpoint has none), then `(1 + gamma) * x + beta` from the style.
pub struct AdaIn {
    fc: Dense,
    channels: usize,
}

impl AdaIn {
    pub fn load(t: &Tensors, name: &str, style: usize, channels: usize) -> Result<Self, String> {
        Ok(Self {
            fc: linear(t, &format!("{name}.fc"), 2 * channels, style)?,
            channels,
        })
    }

    pub fn apply(&self, x: &[f32], frames: usize, style: &[f32]) -> Buf {
        let c = self.channels;
        let (scale, shift) = scale_shift(&self.fc, style, c);
        // Mean and variance per channel, in f64, column by column.
        let mut mean = vec![0.0f64; c];
        let mut square = vec![0.0f64; c];
        for row in x.chunks_exact(c) {
            for ((m, s), v) in mean.iter_mut().zip(square.iter_mut()).zip(row) {
                let v = f64::from(*v);
                *m += v;
                *s += v * v;
            }
        }
        let n = frames as f64;
        let factor: Vec<(f32, f32)> = (0..c)
            .map(|j| {
                let m = mean[j] / n;
                let var = (square[j] / n - m * m).max(0.0);
                let inv = 1.0 / (var + EPS).sqrt();
                // y = scale * (x - m) * inv + shift = a * x + b.
                let a = f64::from(scale[j]) * inv;
                (a as f32, (f64::from(shift[j]) - a * m) as f32)
            })
            .collect();
        let mut out = Buf::scratch(x.len());
        team::chunks_mut(&mut out, c * 256, |index, block| {
            for (r, row) in block.chunks_exact_mut(c).enumerate() {
                let source = &x[(index * 256 + r) * c..][..c];
                for ((o, v), (a, b)) in row.iter_mut().zip(source).zip(&factor) {
                    *o = a * v + b;
                }
            }
        });
        out
    }
}

/// AdaLayerNorm: layer norm over the channels of each frame (no learned
/// affine), then `(1 + gamma) * x + beta` from the style.
pub struct AdaLayerNorm {
    fc: Dense,
    channels: usize,
}

impl AdaLayerNorm {
    pub fn load(t: &Tensors, name: &str, style: usize, channels: usize) -> Result<Self, String> {
        Ok(Self {
            fc: linear(t, &format!("{name}.fc"), 2 * channels, style)?,
            channels,
        })
    }

    pub fn apply(&self, x: &[f32], style: &[f32]) -> Buf {
        let c = self.channels;
        let (scale, shift) = scale_shift(&self.fc, style, c);
        let mut out = Buf::scratch(x.len());
        team::chunks_mut(&mut out, c, |index, row| {
            let source = &x[index * c..(index + 1) * c];
            let n = c as f64;
            let mean = source.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
            let var = source
                .iter()
                .map(|v| (f64::from(*v) - mean).powi(2))
                .sum::<f64>()
                / n;
            let inv = 1.0 / (var + EPS).sqrt();
            for ((o, v), (s, b)) in row.iter_mut().zip(source).zip(scale.iter().zip(&shift)) {
                *o = s * ((f64::from(*v) - mean) * inv) as f32 + b;
            }
        });
        out
    }
}

/// `max(v, 0.2 v)` on every value.
fn leaky(x: &mut [f32]) {
    team::chunks_mut(x, 8192, |_, part| {
        part.iter_mut().for_each(|v| *v = v.max(*v * SLOPE));
    });
}

/// Each frame twice (nearest-neighbor 2x upsampling).
fn repeat2(x: &[f32], channels: usize) -> Buf {
    let mut out = Buf::scratch(x.len() * 2);
    for (pair, row) in out
        .chunks_exact_mut(2 * channels)
        .zip(x.chunks_exact(channels))
    {
        pair[..channels].copy_from_slice(row);
        pair[channels..].copy_from_slice(row);
    }
    out
}

/// StyleTTS 2's `AdainResBlk1d`: AdaIN, leaky ReLU, (2x transposed
/// depthwise upsampling), conv 3, AdaIN, leaky ReLU, conv 3; plus a
/// shortcut (2x repeat, 1x1 conv when the width changes); the sum over
/// sqrt 2.
pub struct AdaInResBlock {
    norm1: AdaIn,
    conv1: Conv1d,
    norm2: AdaIn,
    conv2: Conv1d,
    shortcut: Option<Conv1d>,
    pool: Option<DepthwiseUpsample>,
    inputs: usize,
}

impl AdaInResBlock {
    pub fn load(
        t: &Tensors,
        name: &str,
        (inputs, outputs, style): (usize, usize, usize),
        upsample: bool,
    ) -> Result<Self, String> {
        let shortcut = format!("{name}.conv1x1");
        Ok(Self {
            norm1: AdaIn::load(t, &format!("{name}.norm1"), style, inputs)?,
            conv1: Conv1d::load(t, &format!("{name}.conv1"), (inputs, outputs), (3, 1))?,
            norm2: AdaIn::load(t, &format!("{name}.norm2"), style, outputs)?,
            conv2: Conv1d::load(t, &format!("{name}.conv2"), (outputs, outputs), (3, 1))?,
            shortcut: if inputs != outputs {
                Some(Conv1d::load(t, &shortcut, (inputs, outputs), (1, 1))?)
            } else {
                None
            },
            pool: if upsample {
                Some(DepthwiseUpsample::load(
                    t,
                    &format!("{name}.pool"),
                    inputs,
                    (3, 2, 1, 1),
                )?)
            } else {
                None
            },
            inputs,
        })
    }

    /// `x` (`frames x inputs`) to `(out, out_frames)`.
    pub fn forward(&self, x: &[f32], frames: usize, style: &[f32]) -> (Buf, usize) {
        let mut h = self.norm1.apply(x, frames, style);
        let (h, out_frames) = match &self.pool {
            Some(pool) => {
                leaky(&mut h);
                (
                    self.conv1
                        .forward(&pool.forward(&h, frames), 2 * frames, None),
                    2 * frames,
                )
            }
            None => (self.conv1.forward(&h, frames, Some(SLOPE)), frames),
        };
        let h = self.norm2.apply(&h, out_frames, style);
        let mut out = self.conv2.forward(&h, out_frames, Some(SLOPE));
        let short = if self.pool.is_some() {
            repeat2(x, self.inputs)
        } else {
            let mut copy = Buf::scratch(x.len());
            copy.copy_from_slice(x);
            copy
        };
        let short = match &self.shortcut {
            Some(conv) => conv.forward(&short, out_frames, None),
            None => short,
        };
        let inv = 1.0 / 2.0f32.sqrt();
        team::chunks_mut(&mut out, 8192, |index, part| {
            for (o, s) in part.iter_mut().zip(&short[index * 8192..]) {
                *o = (*o + s) * inv;
            }
        });
        (out, out_frames)
    }
}

/// `x + sin(alpha x)^2 / alpha`, alpha per channel (Snake).
fn snake(x: &mut [f32], alpha: &[f32]) {
    let c = alpha.len();
    team::chunks_mut(x, c * 64, |_, block| {
        for row in block.chunks_exact_mut(c) {
            for (v, a) in row.iter_mut().zip(alpha) {
                let s = (a * *v).sin();
                *v += (1.0 / a) * (s * s);
            }
        }
    });
}

/// iSTFTNet's `AdaINResBlock1`: per dilation, AdaIN, Snake, conv; AdaIN,
/// Snake, conv; plus the input.
pub struct SnakeBlock {
    layers: Vec<SnakeLayer>,
}

struct SnakeLayer {
    norm1: AdaIn,
    alpha1: Vec<f32>,
    conv1: Conv1d,
    norm2: AdaIn,
    alpha2: Vec<f32>,
    conv2: Conv1d,
}

impl SnakeBlock {
    pub fn load(
        t: &Tensors,
        name: &str,
        (channels, style): (usize, usize),
        (kernel, dilations): (usize, &[usize]),
    ) -> Result<Self, String> {
        let layers = dilations
            .iter()
            .enumerate()
            .map(|(i, d)| {
                Ok(SnakeLayer {
                    norm1: AdaIn::load(t, &format!("{name}.adain1.{i}"), style, channels)?,
                    alpha1: t.f32_len(&format!("{name}.alpha1.{i}"), channels)?,
                    conv1: Conv1d::load(
                        t,
                        &format!("{name}.convs1.{i}"),
                        (channels, channels),
                        (kernel, *d),
                    )?,
                    norm2: AdaIn::load(t, &format!("{name}.adain2.{i}"), style, channels)?,
                    alpha2: t.f32_len(&format!("{name}.alpha2.{i}"), channels)?,
                    conv2: Conv1d::load(
                        t,
                        &format!("{name}.convs2.{i}"),
                        (channels, channels),
                        (kernel, 1),
                    )?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self { layers })
    }

    pub fn forward(&self, x: &[f32], frames: usize, style: &[f32]) -> Buf {
        let mut x = {
            let mut copy = Buf::scratch(x.len());
            copy.copy_from_slice(x);
            copy
        };
        for layer in &self.layers {
            let mut h = layer.norm1.apply(&x, frames, style);
            snake(&mut h, &layer.alpha1);
            let h = layer.conv1.forward(&h, frames, None);
            let mut h = layer.norm2.apply(&h, frames, style);
            snake(&mut h, &layer.alpha2);
            let h = layer.conv2.forward(&h, frames, None);
            team::chunks_mut(&mut x, 8192, |index, part| {
                for (v, a) in part.iter_mut().zip(&h[index * 8192..]) {
                    *v += a;
                }
            });
        }
        x
    }
}
