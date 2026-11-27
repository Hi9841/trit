//! 1-D convolutions over time-major signals (`frames x channels`): plain
//! and dilated, depthwise, and transposed (the upsampling of a vocoder).
//! Weights stay f32: a vocoder writes every sample from them, and the
//! networks that use them are small.
//!
//! A step splits the frames into blocks, one task each. A task gathers the
//! inputs of its block (im2col, tap-major, so each tap copies whole input
//! rows) into thread-local scratch and multiplies them on its own thread,
//! so no step allocates a column matrix of the whole signal.

use crate::cpu::{buf::Buf, fgemm, team, tensors::Tensors};
use std::cell::RefCell;

thread_local! {
    /// One task's gathered inputs; it grows to the largest task once per thread.
    static COLUMNS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Output frames per task, at most. Fewer when the signal is short, so every
/// thread gets work.
const MAX_BLOCK: usize = 96;
/// Smaller frame cap for wide im2col gathers (`kernel * cin >= 512`) so the
/// per-thread `COLUMNS` scratch stays L2-friendly alongside the weight matrix.
const WIDE_MAX_BLOCK: usize = 48;

/// Frames per task for `frames` frames: about two tasks per thread, a
/// multiple of the 6 rows of the product tiles. Call it on the calling
/// thread, before the step: inside a task, `team::threads()` counts the
/// team of the worker, which can be another team than the caller's.
fn block_frames(frames: usize) -> usize {
    block_frames_capped(frames, MAX_BLOCK)
}

fn block_frames_capped(frames: usize, cap: usize) -> usize {
    let per_task = frames.div_ceil(2 * team::threads());
    per_task.div_ceil(6).saturating_mul(6).clamp(6, cap)
}

/// Leaky ReLU for `0 <= slope <= 1`.
#[inline(always)]
fn leaky(v: f32, slope: f32) -> f32 {
    v.max(v * slope)
}

multiversion! {
    fn copy_leaky(dst: &mut [f32], src: &[f32], slope: f32) {
        for (o, &v) in dst.iter_mut().zip(src) {
            *o = leaky(v, slope);
        }
    }
}

multiversion! {
    fn add_bias_rows(out: &mut [f32], bias: &[f32]) {
        let cout = bias.len();
        for row in out.chunks_exact_mut(cout) {
            for (v, &b) in row.iter_mut().zip(bias) {
                *v += b;
            }
        }
    }
}

multiversion! {
    fn add_weighted(out: &mut [f32], input: &[f32], tap_weights: &[f32]) {
        for ((o, &v), &w) in out.iter_mut().zip(input).zip(tap_weights) {
            *o += v * w;
        }
    }
}

multiversion! {
    fn add_slice(out: &mut [f32], part: &[f32]) {
        for (o, &p) in out.iter_mut().zip(part) {
            *o += p;
        }
    }
}

/// The weight of `name`, `[out][in][kernel]`: `name.weight` as stored, or
/// the weight norm `g * v / |v|` of `name.weight_g` and `name.weight_v`
/// (also under the names of `torch.nn.utils.parametrizations`).
fn weight(t: &Tensors, name: &str, len: usize, out: usize) -> Result<Vec<f32>, String> {
    let plain = format!("{name}.weight");
    if t.has(&plain) {
        return t.f32_len(&plain, len);
    }
    let (g, v) = if t.has(&format!("{name}.weight_g")) {
        (format!("{name}.weight_g"), format!("{name}.weight_v"))
    } else {
        (
            format!("{name}.parametrizations.weight.original0"),
            format!("{name}.parametrizations.weight.original1"),
        )
    };
    let g = t.f32_len(&g, out)?;
    let mut v = t.f32_len(&v, len)?;
    let row = len / out;
    for (values, g) in v.chunks_exact_mut(row).zip(&g) {
        let norm = values
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        let scale = (f64::from(*g) / norm) as f32;
        values.iter_mut().for_each(|x| *x *= scale);
    }
    Ok(v)
}

/// The bias of `name`, or zeros.
fn bias(t: &Tensors, name: &str, out: usize) -> Result<Vec<f32>, String> {
    let name = format!("{name}.bias");
    if t.has(&name) {
        t.f32_len(&name, out)
    } else {
        Ok(vec![0.0; out])
    }
}

/// A convolution: "same" padding and stride 1 (`load`), or any stride and
/// padding (`load_strided`).
pub struct Conv1d {
    /// `[tap][in][out]`: row `tap * inputs + c` holds the weights of input
    /// `c` at `tap` for every output, as the gathered columns are ordered.
    weight: Vec<f32>,
    bias: Vec<f32>,
    pub inputs: usize,
    pub outputs: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    /// Zeros before the first frame and after the last.
    pad: usize,
}

impl Conv1d {
    /// `name.weight` (or its weight norm) `[outputs][inputs][kernel]`, and
    /// `name.bias` (zeros when absent). The padding is `dilation * (kernel - 1) / 2`
    /// on both sides.
    pub fn load(
        t: &Tensors,
        name: &str,
        (inputs, outputs): (usize, usize),
        (kernel, dilation): (usize, usize),
    ) -> Result<Self, String> {
        let pad = dilation * (kernel - 1) / 2;
        Self::build(t, name, (inputs, outputs), (kernel, dilation, 1, pad))
    }

    /// `load` with a stride and `pad` zeros on both sides (dilation 1).
    pub fn load_strided(
        t: &Tensors,
        name: &str,
        (inputs, outputs): (usize, usize),
        (kernel, stride, pad): (usize, usize, usize),
    ) -> Result<Self, String> {
        Self::build(t, name, (inputs, outputs), (kernel, 1, stride, pad))
    }

    fn build(
        t: &Tensors,
        name: &str,
        (inputs, outputs): (usize, usize),
        (kernel, dilation, stride, pad): (usize, usize, usize, usize),
    ) -> Result<Self, String> {
        let stored = weight(t, name, outputs * inputs * kernel, outputs)?;
        let mut weight = vec![0.0f32; stored.len()];
        for o in 0..outputs {
            for c in 0..inputs {
                for tap in 0..kernel {
                    weight[(tap * inputs + c) * outputs + o] =
                        stored[(o * inputs + c) * kernel + tap];
                }
            }
        }
        Ok(Self {
            weight,
            bias: bias(t, name, outputs)?,
            inputs,
            outputs,
            kernel,
            dilation,
            stride,
            pad,
        })
    }

    pub fn out_frames(&self, frames: usize) -> usize {
        (frames + 2 * self.pad).saturating_sub(self.dilation * (self.kernel - 1) + 1) / self.stride
            + 1
    }

    /// A linear layer `name.weight` `[outputs][inputs]`: a kernel of 1.
    pub fn linear(t: &Tensors, name: &str, inputs: usize, outputs: usize) -> Result<Self, String> {
        Self::load(t, name, (inputs, outputs), (1, 1))
    }

    /// The convolution of `x` (`frames x inputs`): `out_frames(frames)` rows.
    /// With `slope`, the input passes a leaky ReLU first (`max(v, slope * v)`;
    /// 0 is ReLU).
    pub fn forward(&self, x: &[f32], frames: usize, slope: Option<f32>) -> Buf {
        let (cin, cout, kernel) = (self.inputs, self.outputs, self.kernel);
        assert_eq!(Some(x.len()), frames.checked_mul(cin));
        if frames == 0 || cout == 0 {
            return Buf::scratch(0);
        }
        let (pad, stride) = (self.pad, self.stride);
        let width = kernel * cin;
        let out_frames = self.out_frames(frames);
        let cap = if width >= 512 {
            WIDE_MAX_BLOCK
        } else {
            MAX_BLOCK
        };
        let block = block_frames_capped(out_frames, cap);
        let identity_1x1 =
            kernel == 1 && stride == 1 && pad == 0 && slope.is_none() && out_frames == frames;
        let mut out = Buf::scratch(out_frames * cout);
        let out_ptr = team::SendPtr::new(&mut out);
        team::for_each(out_frames.div_ceil(block), |task| {
            let first = task * block;
            let rows = block.min(out_frames - first);
            // SAFETY: the rows of this block belong to this task; `out` outlives the step.
            let out = unsafe { out_ptr.slice_mut(first * cout, rows * cout) };
            if identity_1x1 {
                fgemm::matmul_nn_into(
                    &x[first * cin..(first + rows) * cin],
                    rows,
                    cin,
                    &self.weight,
                    cout,
                    out,
                );
            } else {
                COLUMNS.with(|columns| {
                    let mut columns = columns.borrow_mut();
                    if columns.len() < rows * width {
                        columns.resize(rows * width, 0.0);
                    }
                    let columns = &mut columns[..rows * width];
                    for (r, row) in columns.chunks_exact_mut(width).enumerate() {
                        for (tap, part) in row.chunks_exact_mut(cin).enumerate() {
                            let source = ((first + r) * stride + tap * self.dilation)
                                .checked_sub(pad)
                                .filter(|s| *s < frames);
                            match source {
                                None => part.fill(0.0),
                                Some(s) => {
                                    let input = &x[s * cin..(s + 1) * cin];
                                    match slope {
                                        None => part.copy_from_slice(input),
                                        Some(slope) => copy_leaky(part, input, slope),
                                    }
                                }
                            }
                        }
                    }
                    fgemm::matmul_nn_into(columns, rows, width, &self.weight, cout, out);
                });
            }
            add_bias_rows(out, &self.bias);
        });
        out
    }
}

/// A depthwise convolution (`groups = channels`) with "same" padding.
pub struct Depthwise {
    /// `[tap][channel]` so each tap reads a contiguous SIMD vector of weights.
    weight: Vec<f32>,
    bias: Vec<f32>,
    channels: usize,
    kernel: usize,
    dilation: usize,
}

impl Depthwise {
    pub fn load(
        t: &Tensors,
        name: &str,
        channels: usize,
        (kernel, dilation): (usize, usize),
    ) -> Result<Self, String> {
        let stored = weight(t, name, channels * kernel, channels)?;
        let weight = (0..kernel * channels)
            .map(|i| stored[(i % channels) * kernel + i / channels])
            .collect();
        Ok(Self {
            weight,
            bias: bias(t, name, channels)?,
            channels,
            kernel,
            dilation,
        })
    }

    pub fn forward(&self, x: &[f32], frames: usize) -> Buf {
        let (channels, kernel) = (self.channels, self.kernel);
        let pad = self.dilation * (kernel - 1) / 2;
        let mut out = Buf::scratch(frames * channels);
        let rows = block_frames(frames);
        team::chunks_mut(&mut out, rows * channels, |task, block| {
            let first = task * rows;
            for (r, row) in block.chunks_exact_mut(channels).enumerate() {
                row.copy_from_slice(&self.bias);
                for tap in 0..kernel {
                    let Some(s) = (first + r + tap * self.dilation)
                        .checked_sub(pad)
                        .filter(|s| *s < frames)
                    else {
                        continue;
                    };
                    let input = &x[s * channels..(s + 1) * channels];
                    let taps = &self.weight[tap * channels..(tap + 1) * channels];
                    add_weighted(row, input, taps);
                }
            }
        });
        out
    }
}

/// A depthwise transposed convolution (`groups = channels`): `frames` in,
/// `(frames - 1) * stride - 2 * pad + kernel + extra` out, where `extra` is
/// PyTorch's `output_padding`.
pub struct DepthwiseUpsample {
    /// `[tap][channel]` so each tap reads a contiguous SIMD vector of weights.
    weight: Vec<f32>,
    bias: Vec<f32>,
    channels: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
    extra: usize,
}

impl DepthwiseUpsample {
    pub fn load(
        t: &Tensors,
        name: &str,
        channels: usize,
        (kernel, stride, pad, extra): (usize, usize, usize, usize),
    ) -> Result<Self, String> {
        let stored = weight(t, name, channels * kernel, channels)?;
        let weight = (0..kernel * channels)
            .map(|i| stored[(i % channels) * kernel + i / channels])
            .collect();
        Ok(Self {
            weight,
            bias: bias(t, name, channels)?,
            channels,
            kernel,
            stride,
            pad,
            extra,
        })
    }

    pub fn out_frames(&self, frames: usize) -> usize {
        ((frames - 1) * self.stride + self.kernel + self.extra).saturating_sub(2 * self.pad)
    }

    /// Output `y` gets input `s` through tap `y + pad - s * stride`.
    pub fn forward(&self, x: &[f32], frames: usize) -> Buf {
        let (channels, kernel, stride) = (self.channels, self.kernel, self.stride);
        let out_frames = self.out_frames(frames);
        let rows = block_frames(out_frames);
        let mut out = Buf::scratch(out_frames * channels);
        team::chunks_mut(&mut out, rows * channels, |task, block| {
            let first = task * rows;
            for (r, row) in block.chunks_exact_mut(channels).enumerate() {
                let y = first + r + self.pad;
                row.copy_from_slice(&self.bias);
                for tap in 0..kernel {
                    let Some(from) = y.checked_sub(tap).filter(|v| v % stride == 0) else {
                        continue;
                    };
                    let s = from / stride;
                    if s >= frames {
                        continue;
                    }
                    let input = &x[s * channels..(s + 1) * channels];
                    let taps = &self.weight[tap * channels..(tap + 1) * channels];
                    add_weighted(row, input, taps);
                }
            }
        });
        out
    }
}

/// A transposed convolution with `stride` and padding `(kernel - stride) / 2`:
/// `frames` in, `frames * stride` out (HiFi-GAN's upsampling).
pub struct Upsample {
    /// `[in][tap][out]`: input `c` times this row gives the outputs of every tap.
    weight: Vec<f32>,
    bias: Vec<f32>,
    inputs: usize,
    pub outputs: usize,
    kernel: usize,
    pub stride: usize,
}

impl Upsample {
    /// `name.weight` `[inputs][outputs][kernel]` (PyTorch's layout for a
    /// transposed convolution) and `name.bias`.
    pub fn load(
        t: &Tensors,
        name: &str,
        (inputs, outputs): (usize, usize),
        (kernel, stride): (usize, usize),
    ) -> Result<Self, String> {
        if kernel < stride || (kernel - stride) % 2 != 0 {
            return Err(format!(
                "{name}: an upsampling kernel of {kernel} with stride {stride} has no symmetric padding"
            ));
        }
        let stored = weight(t, name, inputs * outputs * kernel, inputs)?;
        let mut weight = vec![0.0f32; stored.len()];
        for c in 0..inputs {
            for o in 0..outputs {
                for tap in 0..kernel {
                    weight[(c * kernel + tap) * outputs + o] =
                        stored[(c * outputs + o) * kernel + tap];
                }
            }
        }
        Ok(Self {
            weight,
            bias: bias(t, name, outputs)?,
            inputs,
            outputs,
            kernel,
            stride,
        })
    }

    /// The upsampled `x` (`frames x inputs`), after a leaky ReLU with `slope`.
    /// Output `y` gets input `s` through tap `y + pad - s * stride`; a task
    /// multiplies the inputs its block of outputs reads, then gathers.
    pub fn forward(&self, x: &[f32], frames: usize, slope: f32) -> Buf {
        let (cin, cout, kernel, stride) = (self.inputs, self.outputs, self.kernel, self.stride);
        let pad = (kernel - stride) / 2;
        let out_frames = frames * stride;
        let width = kernel * cout;
        let block = block_frames(out_frames).div_ceil(stride) * stride;
        let mut out = Buf::scratch(out_frames * cout);
        let out_ptr = team::SendPtr::new(&mut out);
        team::for_each(out_frames.div_ceil(block), |task| {
            let first = task * block;
            let rows = block.min(out_frames - first);
            // The inputs that reach outputs first..first + rows.
            let low = (first + pad + 1).saturating_sub(kernel).div_ceil(stride);
            let high = ((first + rows - 1 + pad) / stride).min(frames - 1);
            let count = high + 1 - low;
            // SAFETY: the rows of this block belong to this task; `out` outlives the step.
            let out = unsafe { out_ptr.slice_mut(first * cout, rows * cout) };
            COLUMNS.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                let need = count * cin + count * width;
                if scratch.len() < need {
                    scratch.resize(need, 0.0);
                }
                let (inputs, products) = scratch.split_at_mut(count * cin);
                let products = &mut products[..count * width];
                copy_leaky(inputs, &x[low * cin..(high + 1) * cin], slope);
                fgemm::matmul_nn_into(inputs, count, cin, &self.weight, width, products);
                for (r, row) in out.chunks_exact_mut(cout).enumerate() {
                    let y = first + r;
                    row.copy_from_slice(&self.bias);
                    for s in low..=high {
                        let Some(tap) = (y + pad).checked_sub(s * stride).filter(|t| *t < kernel)
                        else {
                            continue;
                        };
                        let part = &products[(s - low) * width + tap * cout..][..cout];
                        add_slice(row, part);
                    }
                }
            });
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((*state >> 40) as u32) as f32 / ((1u32 << 24) as f32);
        bits * 2.0 - 1.0
    }

    #[test]
    fn depthwise_tap_major_matches_reference() {
        let mut seed = 0x1111_2222_3333_4444u64;
        for (channels, kernel, dilation, frames) in [
            (1usize, 3usize, 1usize, 17usize),
            (19, 7, 1, 130),
            (64, 5, 3, 200),
        ] {
            let stored: Vec<f32> = (0..channels * kernel).map(|_| lcg(&mut seed)).collect();
            let weight: Vec<f32> = (0..kernel * channels)
                .map(|i| stored[(i % channels) * kernel + i / channels])
                .collect();
            let bias: Vec<f32> = (0..channels).map(|_| lcg(&mut seed) * 0.25).collect();
            let x: Vec<f32> = (0..frames * channels).map(|_| lcg(&mut seed)).collect();
            let conv = Depthwise {
                weight,
                bias: bias.clone(),
                channels,
                kernel,
                dilation,
            };
            let got = conv.forward(&x, frames);
            let pad = dilation * (kernel - 1) / 2;
            for t in 0..frames {
                for c in 0..channels {
                    let mut sum = f64::from(bias[c]);
                    for tap in 0..kernel {
                        if let Some(s) = (t + tap * dilation)
                            .checked_sub(pad)
                            .filter(|&s| s < frames)
                        {
                            sum += f64::from(x[s * channels + c])
                                * f64::from(stored[c * kernel + tap]);
                        }
                    }
                    let want = sum as f32;
                    let g = got[t * channels + c];
                    assert!(
                        (g - want).abs() <= 1e-5,
                        "depthwise c={channels} k={kernel} d={dilation} t={t} ch={c}: got={g} want={want}"
                    );
                }
            }
        }
    }

    #[test]
    fn depthwise_upsample_tap_major_matches_reference() {
        let mut seed = 0x5555_6666_7777_8888u64;
        for (channels, kernel, stride, pad, extra, frames) in [
            (1usize, 4usize, 2usize, 1usize, 0usize, 11usize),
            (19, 7, 3, 2, 1, 75),
            (64, 16, 8, 4, 0, 40),
        ] {
            let stored: Vec<f32> = (0..channels * kernel).map(|_| lcg(&mut seed)).collect();
            let weight: Vec<f32> = (0..kernel * channels)
                .map(|i| stored[(i % channels) * kernel + i / channels])
                .collect();
            let bias: Vec<f32> = (0..channels).map(|_| lcg(&mut seed) * 0.25).collect();
            let x: Vec<f32> = (0..frames * channels).map(|_| lcg(&mut seed)).collect();
            let up = DepthwiseUpsample {
                weight,
                bias: bias.clone(),
                channels,
                kernel,
                stride,
                pad,
                extra,
            };
            let out_frames = up.out_frames(frames);
            let got = up.forward(&x, frames);
            assert_eq!(got.len(), out_frames * channels);
            for y_out in 0..out_frames {
                let y = y_out + pad;
                for c in 0..channels {
                    let mut sum = f64::from(bias[c]);
                    for tap in 0..kernel {
                        if let Some(from) = y.checked_sub(tap).filter(|v| v % stride == 0) {
                            let s = from / stride;
                            if s < frames {
                                sum += f64::from(x[s * channels + c])
                                    * f64::from(stored[c * kernel + tap]);
                            }
                        }
                    }
                    let want = sum as f32;
                    let g = got[y_out * channels + c];
                    assert!(
                        (g - want).abs() <= 1e-5,
                        "depthwise_up c={channels} k={kernel} s={stride} y={y_out} ch={c}: got={g} want={want}"
                    );
                }
            }
        }
    }
}
