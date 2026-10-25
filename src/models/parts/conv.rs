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

/// Frames per task for `frames` frames: about two tasks per thread, a
/// multiple of the 6 rows of the product tiles. Call it on the calling
/// thread, before the step: inside a task, `team::threads()` counts the
/// team of the worker, which can be another team than the caller's.
fn block_frames(frames: usize) -> usize {
    let per_task = frames.div_ceil(2 * team::threads());
    per_task.div_ceil(6).saturating_mul(6).clamp(6, MAX_BLOCK)
}

/// `max(v, slope * v)`: leaky ReLU for `0 <= slope <= 1`.
#[inline(always)]
fn leaky(v: f32, slope: f32) -> f32 {
    v.max(v * slope)
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

/// A convolution with stride 1 and "same" padding: `frames` in, `frames` out.
pub struct Conv1d {
    /// `[tap][in][out]`: row `tap * inputs + c` holds the weights of input
    /// `c` at `tap` for every output, as the gathered columns are ordered.
    weight: Vec<f32>,
    bias: Vec<f32>,
    pub inputs: usize,
    pub outputs: usize,
    kernel: usize,
    dilation: usize,
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
        })
    }

    /// A linear layer `name.weight` `[outputs][inputs]`: a kernel of 1.
    pub fn linear(t: &Tensors, name: &str, inputs: usize, outputs: usize) -> Result<Self, String> {
        Self::load(t, name, (inputs, outputs), (1, 1))
    }

    /// The convolution of `x` (`frames x inputs`). With `slope`, the input
    /// passes a leaky ReLU first (`max(v, slope * v)`; 0 is ReLU).
    pub fn forward(&self, x: &[f32], frames: usize, slope: Option<f32>) -> Buf {
        let (cin, cout, kernel) = (self.inputs, self.outputs, self.kernel);
        debug_assert_eq!(x.len(), frames * cin);
        let pad = self.dilation * (kernel - 1) / 2;
        let width = kernel * cin;
        let block = block_frames(frames);
        let mut out = Buf::scratch(frames * cout);
        let out_ptr = team::SendPtr::new(&mut out);
        team::for_each(frames.div_ceil(block), |task| {
            let first = task * block;
            let rows = block.min(frames - first);
            // SAFETY: the rows of this block belong to this task; `out` outlives the step.
            let out = unsafe { out_ptr.slice_mut(first * cout, rows * cout) };
            COLUMNS.with(|columns| {
                let mut columns = columns.borrow_mut();
                if columns.len() < rows * width {
                    columns.resize(rows * width, 0.0);
                }
                let columns = &mut columns[..rows * width];
                for (r, row) in columns.chunks_exact_mut(width).enumerate() {
                    for (tap, part) in row.chunks_exact_mut(cin).enumerate() {
                        let source = (first + r + tap * self.dilation)
                            .checked_sub(pad)
                            .filter(|s| *s < frames);
                        match source {
                            None => part.fill(0.0),
                            Some(s) => {
                                let input = &x[s * cin..(s + 1) * cin];
                                match slope {
                                    None => part.copy_from_slice(input),
                                    Some(slope) => {
                                        for (p, v) in part.iter_mut().zip(input) {
                                            *p = leaky(*v, slope);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                fgemm::matmul_nn_into(columns, rows, width, &self.weight, cout, out);
            });
            for row in out.chunks_exact_mut(cout) {
                for (v, b) in row.iter_mut().zip(&self.bias) {
                    *v += b;
                }
            }
        });
        out
    }
}

/// A depthwise convolution (`groups = channels`) with "same" padding.
pub struct Depthwise {
    /// `[channel][kernel]`, as stored.
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
        Ok(Self {
            weight: weight(t, name, channels * kernel, channels)?,
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
                    for (c, (o, v)) in row.iter_mut().zip(input).enumerate() {
                        *o += v * self.weight[c * kernel + tap];
                    }
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
                for (i, v) in inputs.iter_mut().zip(&x[low * cin..(high + 1) * cin]) {
                    *i = leaky(*v, slope);
                }
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
                        for (o, p) in row.iter_mut().zip(part) {
                            *o += p;
                        }
                    }
                }
            });
        });
        out
    }
}
