//! The style-conditioned blocks of StyleTTS 2 and iSTFTNet: AdaIN and
//! AdaLayerNorm (a norm whose scale and shift come from the voice's style
//! vector), the AdaIN residual block with optional 2x upsampling, and the
//! Snake residual block of the generator. Signals are time-major
//! (`frames x channels`).

use crate::cpu::{buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::layers::{balanced_chunk, balanced_rows, linear};
use crate::tts::conv::{Conv1d, DepthwiseUpsample};

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

    fn affine(&self, x: &[f32], frames: usize, style: &[f32]) -> Vec<(f32, f32)> {
        let c = self.channels;
        let (scale, shift) = scale_shift(&self.fc, style, c);
        let mut factor = vec![(0.0f32, 0.0f32); c];
        let n = frames as f64;
        if frames * c <= 32_768 || c < 32 {
            let mut mean = vec![0.0f64; c];
            let mut square = vec![0.0f64; c];
            for row in x.chunks_exact(c) {
                for ((m, s), v) in mean.iter_mut().zip(square.iter_mut()).zip(row) {
                    let v = f64::from(*v);
                    *m += v;
                    *s += v * v;
                }
            }
            for j in 0..c {
                let m = mean[j] / n;
                let var = (square[j] / n - m * m).max(0.0);
                let inv = 1.0 / (var + EPS).sqrt();
                let a = f64::from(scale[j]) * inv;
                factor[j] = (a as f32, (f64::from(shift[j]) - a * m) as f32);
            }
            return factor;
        }
        // Channel-blocked reduction preserves exact frame order (0..frames) per channel.
        let ch_block = c.div_ceil(team::threads().max(1)).max(16);
        team::chunks_mut(&mut factor, ch_block, |block, out| {
            let start = block * ch_block;
            let len = out.len();
            let mut mean = [0.0f64; 256];
            let mut square = [0.0f64; 256];
            let mut off = 0;
            while off < len {
                let take = (len - off).min(256);
                let m = &mut mean[..take];
                let s = &mut square[..take];
                m.fill(0.0);
                s.fill(0.0);
                let c0 = start + off;
                for row in x.chunks_exact(c) {
                    for ((mm, ss), &v) in m.iter_mut().zip(s.iter_mut()).zip(&row[c0..c0 + take]) {
                        let v = f64::from(v);
                        *mm += v;
                        *ss += v * v;
                    }
                }
                for j in 0..take {
                    let mj = m[j] / n;
                    let var = (s[j] / n - mj * mj).max(0.0);
                    let inv = 1.0 / (var + EPS).sqrt();
                    let a = f64::from(scale[c0 + j]) * inv;
                    out[off + j] = (a as f32, (f64::from(shift[c0 + j]) - a * mj) as f32);
                }
                off += take;
            }
        });
        factor
    }

    pub fn apply(&self, x: &[f32], frames: usize, style: &[f32]) -> Buf {
        let c = self.channels;
        assert_eq!(Some(x.len()), frames.checked_mul(c));
        if frames == 0 || c == 0 {
            return Buf::scratch(0);
        }
        let factor = self.affine(x, frames, style);
        let mut out = Buf::scratch(x.len());
        let rows_per_task = balanced_rows(frames, c);
        team::chunks_mut(&mut out, c * rows_per_task, |index, block| {
            for (r, row) in block.chunks_exact_mut(c).enumerate() {
                let source = &x[(index * rows_per_task + r) * c..][..c];
                for ((o, v), (a, b)) in row.iter_mut().zip(source).zip(&factor) {
                    *o = a * v + b;
                }
            }
        });
        out
    }

    /// Fused AdaIN + Snake (`y = a * x + b; out = y + inv_alpha * sin(alpha * y)^2`),
    /// avoiding a second full-tensor pass and team barrier in `SnakeBlock`.
    pub fn apply_snake(
        &self,
        x: &[f32],
        frames: usize,
        style: &[f32],
        alpha: &[f32],
        inv_alpha: &[f32],
    ) -> Buf {
        let c = self.channels;
        assert_eq!(Some(x.len()), frames.checked_mul(c));
        assert!(alpha.len() == c && inv_alpha.len() == c);
        if frames == 0 || c == 0 {
            return Buf::scratch(0);
        }
        let factor = self.affine(x, frames, style);
        let mut out = Buf::scratch(x.len());
        let rows_per_task = balanced_rows(frames, c);
        team::chunks_mut(&mut out, c * rows_per_task, |index, block| {
            for (r, row) in block.chunks_exact_mut(c).enumerate() {
                let source = &x[(index * rows_per_task + r) * c..][..c];
                for ((((o, &v), &(a, b)), &al), &inv_al) in row
                    .iter_mut()
                    .zip(source)
                    .zip(&factor)
                    .zip(alpha)
                    .zip(inv_alpha)
                {
                    let y = a * v + b;
                    let s = (al * y).sin();
                    *o = y + inv_al * (s * s);
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
        let (h, out_frames) = {
            let mut h = self.norm1.apply(x, frames, style);
            match &self.pool {
                Some(pool) => {
                    leaky(&mut h);
                    let up = pool.forward(&h, frames);
                    drop(h);
                    (self.conv1.forward(&up, 2 * frames, None), 2 * frames)
                }
                None => (self.conv1.forward(&h, frames, Some(SLOPE)), frames),
            }
        };
        let mut out = {
            let h2 = self.norm2.apply(&h, out_frames, style);
            drop(h);
            self.conv2.forward(&h2, out_frames, Some(SLOPE))
        };
        let repeated = self.pool.is_some().then(|| repeat2(x, self.inputs));
        let short_in: &[f32] = repeated.as_deref().unwrap_or(x);
        let projected = self
            .shortcut
            .as_ref()
            .map(|conv| conv.forward(short_in, out_frames, None));
        let short: &[f32] = projected.as_deref().unwrap_or(short_in);
        let inv = 1.0 / 2.0f32.sqrt();
        let chunk = balanced_chunk(out.len(), 4096);
        team::chunks_mut(&mut out, chunk, |index, part| {
            add_and_scale(part, &short[index * chunk..index * chunk + part.len()], inv);
        });
        (out, out_frames)
    }
}

multiversion! {
    fn add_and_scale(dst: &mut [f32], src: &[f32], scale: f32) {
        for (o, &s) in dst.iter_mut().zip(src) {
            *o = (*o + s) * scale;
        }
    }
}

multiversion! {
    fn add_in_place(dst: &mut [f32], src: &[f32]) {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d += s;
        }
    }
}

/// iSTFTNet's `AdaINResBlock1`: per dilation, AdaIN, Snake, conv; AdaIN,
/// Snake, conv; plus the input.
pub struct SnakeBlock {
    layers: Vec<SnakeLayer>,
}

struct SnakeLayer {
    norm1: AdaIn,
    alpha1: Vec<f32>,
    inv_alpha1: Vec<f32>,
    conv1: Conv1d,
    norm2: AdaIn,
    alpha2: Vec<f32>,
    inv_alpha2: Vec<f32>,
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
                let alpha1 = t.f32_len(&format!("{name}.alpha1.{i}"), channels)?;
                let inv_alpha1 = alpha1.iter().map(|&a| 1.0 / a).collect();
                let alpha2 = t.f32_len(&format!("{name}.alpha2.{i}"), channels)?;
                let inv_alpha2 = alpha2.iter().map(|&a| 1.0 / a).collect();
                Ok(SnakeLayer {
                    norm1: AdaIn::load(t, &format!("{name}.adain1.{i}"), style, channels)?,
                    alpha1,
                    inv_alpha1,
                    conv1: Conv1d::load(
                        t,
                        &format!("{name}.convs1.{i}"),
                        (channels, channels),
                        (kernel, *d),
                    )?,
                    norm2: AdaIn::load(t, &format!("{name}.adain2.{i}"), style, channels)?,
                    alpha2,
                    inv_alpha2,
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

    pub fn forward(&self, x_in: &[f32], frames: usize, style: &[f32]) -> Buf {
        let Some((first, rest)) = self.layers.split_first() else {
            let mut copy = Buf::scratch(x_in.len());
            copy.copy_from_slice(x_in);
            return copy;
        };
        let chunk = balanced_chunk(x_in.len(), 4096);
        let mut x = {
            let h = {
                let h =
                    first
                        .norm1
                        .apply_snake(x_in, frames, style, &first.alpha1, &first.inv_alpha1);
                first.conv1.forward(&h, frames, None)
            };
            let h = first
                .norm2
                .apply_snake(&h, frames, style, &first.alpha2, &first.inv_alpha2);
            first.conv2.forward(&h, frames, None)
        };
        team::chunks_mut(&mut x, chunk, |index, part| {
            add_in_place(part, &x_in[index * chunk..index * chunk + part.len()]);
        });
        for layer in rest {
            let h = {
                let h = {
                    let h = layer.norm1.apply_snake(
                        &x,
                        frames,
                        style,
                        &layer.alpha1,
                        &layer.inv_alpha1,
                    );
                    layer.conv1.forward(&h, frames, None)
                };
                let h =
                    layer
                        .norm2
                        .apply_snake(&h, frames, style, &layer.alpha2, &layer.inv_alpha2);
                layer.conv2.forward(&h, frames, None)
            };
            team::chunks_mut(&mut x, chunk, |index, part| {
                add_in_place(part, &h[index * chunk..index * chunk + part.len()]);
            });
        }
        x
    }
}
