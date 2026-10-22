//! Matrix products for the speech model. Weights are row-major `[out][in]`,
//! as the model file stores them, so every output is a dot product of two
//! contiguous rows.

use super::buf::Buf;
use super::fgemm;
use super::int8::Int8;
use super::team;

/// Output rows per task of the bias and ReLU pass.
const EPILOGUE_ROWS: usize = 64;

/// Dense layer with f32, f16, or int8 weights, `[out][in]`.
pub struct Dense {
    pub out_features: usize,
    pub in_features: usize,
    weight: Weights,
    bias: Option<Vec<f32>>,
}

enum Weights {
    F32(Vec<f32>),
    /// f16 bits, as stored; widened to f32 inside the product.
    F16(Vec<u16>),
    /// Int8 with a scale per row and 128 inputs; i8 activations.
    Int8(Int8),
    /// Int8 storage with a scale per row and 128 inputs, widened to f32 for
    /// f32 math: half the memory of f16, the accuracy of f16 (the error of
    /// int8 products came from quantizing the activations).
    Q8(Q8),
}

/// Rows of int8 values with one f32 scale per `Q8_GROUP` values.
pub struct Q8 {
    values: Vec<i8>,
    scales: Vec<f32>,
    groups: usize,
}

const Q8_GROUP: usize = 32;

impl Q8 {
    fn quantize(weight: &[f32], n: usize, k: usize) -> Self {
        let groups = k.div_ceil(Q8_GROUP);
        let mut values = vec![0i8; n * k];
        let mut scales = vec![0.0f32; n * groups];
        let (all_v, all_s) = (
            team::SendPtr::new(&mut values),
            team::SendPtr::new(&mut scales),
        );
        team::for_each(n, |row| {
            // SAFETY: each row belongs to one task; the buffers outlive the step.
            let (v, s) = unsafe {
                (
                    all_v.slice_mut(row * k, k),
                    all_s.slice_mut(row * groups, groups),
                )
            };
            let src = &weight[row * k..(row + 1) * k];
            for ((span, out), scale) in src
                .chunks(Q8_GROUP)
                .zip(v.chunks_mut(Q8_GROUP))
                .zip(s.iter_mut())
            {
                let max = span.iter().fold(0.0f32, |m, w| m.max(w.abs()));
                *scale = if max > 0.0 { max / 127.0 } else { 1.0 };
                for (o, w) in out.iter_mut().zip(span) {
                    *o = (w / *scale).round_ties_even().clamp(-127.0, 127.0) as i8;
                }
            }
        });
        Self {
            values,
            scales,
            groups,
        }
    }

    fn matmul(&self, x: &[f32], m: usize, k: usize, n: usize) -> Buf {
        fgemm::matmul_widened(x, m, k, n, |cols, panel| {
            for (j, row) in cols.enumerate() {
                widen_q8(
                    &self.values[row * k..(row + 1) * k],
                    &self.scales[row * self.groups..(row + 1) * self.groups],
                    &mut panel[j * k..(j + 1) * k],
                );
            }
        })
    }
}

multiversion! {
    /// One row: `out = value * scale` per group of `Q8_GROUP`.
    fn widen_q8(values: &[i8], scales: &[f32], out: &mut [f32]) {
        for ((group, out), scale) in values.chunks(Q8_GROUP).zip(out.chunks_mut(Q8_GROUP)).zip(scales) {
            for (o, v) in out.iter_mut().zip(group) {
                *o = f32::from(*v) * scale;
            }
        }
    }
}

impl Dense {
    pub fn new(
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self, String> {
        let len = weight.len();
        Self::checked(Weights::F32(weight), len, bias, out_features, in_features)
    }

    /// A layer with f16 weights (as bits); half the memory of `new`.
    pub fn half(
        weight: Vec<u16>,
        bias: Option<Vec<f32>>,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self, String> {
        let len = weight.len();
        Self::checked(Weights::F16(weight), len, bias, out_features, in_features)
    }

    /// A layer from the model file: `parts` are tensor names (without
    /// `.weight`) with their output rows, stacked in order; a missing bias is
    /// zero, and no bias at all gives a layer without one. Weights are read
    /// at the file's precision and stored as `Tensors::precision` says
    /// (see `Precision`).
    pub fn load(
        t: &super::tensors::Tensors,
        parts: &[(&str, usize)],
        in_features: usize,
    ) -> Result<Self, String> {
        let out_features: usize = parts.iter().map(|(_, rows)| rows).sum();
        let mut bias = Vec::with_capacity(out_features);
        let mut any_bias = false;
        for (name, rows) in parts {
            if t.has(&format!("{name}.bias")) {
                bias.extend(t.f32_len(&format!("{name}.bias"), *rows)?);
                any_bias = true;
            } else {
                bias.extend(std::iter::repeat_n(0.0, *rows));
            }
        }
        let bias = any_bias.then_some(bias);
        let precision = t.precision();
        let int8 = Int8::fits(out_features, in_features)
            && match precision {
                Precision::Int8 => true,
                Precision::Auto => out_features.min(in_features) >= INT8_MIN_WIDTH,
                Precision::F16 | Precision::F32 => false,
            };
        let len = out_features * in_features;
        if !int8 && matches!(precision, Precision::Auto | Precision::F16) {
            // f16: read as f16 bits directly; no f32 copy of the layer exists.
            let mut half = Vec::with_capacity(len);
            for (name, rows) in parts {
                half.extend(t.f16_bits(&format!("{name}.weight"), rows * in_features)?);
            }
            return Self::half(half, bias, out_features, in_features);
        }
        let mut weight = Vec::with_capacity(len);
        for (name, rows) in parts {
            weight.extend(t.f32_len(&format!("{name}.weight"), rows * in_features)?);
        }
        let weight = if int8 {
            Weights::Int8(Int8::quantize(out_features, in_features, |row, out| {
                out.copy_from_slice(&weight[row * in_features..(row + 1) * in_features]);
            }))
        } else if precision == Precision::Int8 {
            // A shape the int8 kernels cannot take: int8 storage, f32 math.
            Weights::Q8(Q8::quantize(&weight, out_features, in_features))
        } else {
            Weights::F32(weight)
        };
        Self::checked(weight, len, bias, out_features, in_features)
    }

    /// A token table `[rows][cols]` (embedding, output projection) without
    /// bias: f32 under `Precision::F32`, int8 storage with f32 math under
    /// `Precision::Int8`, else f16. Never int8 activations: the logits
    /// decide every token.
    pub fn table(
        t: &super::tensors::Tensors,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        let len = rows * cols;
        let weight = match t.precision() {
            Precision::F32 => Weights::F32(t.f32_len(name, len)?),
            Precision::Int8 => Weights::Q8(Q8::quantize(&t.f32_len(name, len)?, rows, cols)),
            Precision::Auto if cols >= INT8_TABLE_WIDTH => {
                Weights::Q8(Q8::quantize(&t.f32_len(name, len)?, rows, cols))
            }
            Precision::Auto | Precision::F16 => Weights::F16(t.f16_bits(name, len)?),
        };
        Self::checked(weight, len, None, rows, cols)
    }

    /// Row `index` of the weights as f32 (a token embedding).
    pub fn row(&self, index: usize, out: &mut [f32]) {
        let k = self.in_features;
        let range = index * k..(index + 1) * k;
        match &self.weight {
            Weights::F32(w) => out.copy_from_slice(&w[range]),
            Weights::F16(w) => {
                for (o, h) in out.iter_mut().zip(&w[range]) {
                    *o = super::tensors::f16_to_f32(*h);
                }
            }
            Weights::Q8(q) => widen_q8(
                &q.values[range],
                &q.scales[index * q.groups..(index + 1) * q.groups],
                out,
            ),
            Weights::Int8(_) => unreachable!("token tables are never int8 activations"),
        }
    }

    fn checked(
        weight: Weights,
        len: usize,
        bias: Option<Vec<f32>>,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self, String> {
        if len != out_features * in_features
            || bias.as_ref().is_some_and(|b| b.len() != out_features)
        {
            return Err(format!(
                "dense layer {out_features}x{in_features} has {len} weights"
            ));
        }
        Ok(Self {
            out_features,
            in_features,
            weight,
            bias,
        })
    }

    /// `x` is `rows x in_features`; the result is `rows x out_features`.
    pub fn forward(&self, x: &[f32], rows: usize) -> Buf {
        self.forward_then(x, rows, false)
    }

    /// `forward`, then `max(0, v)` on every output.
    pub fn forward_relu(&self, x: &[f32], rows: usize) -> Buf {
        self.forward_then(x, rows, true)
    }

    fn forward_then(&self, x: &[f32], rows: usize, relu: bool) -> Buf {
        let n = self.out_features;
        let mut out = match &self.weight {
            Weights::F32(w) => fgemm::matmul(x, rows, self.in_features, w, n),
            Weights::F16(w) => fgemm::matmul(x, rows, self.in_features, w, n),
            Weights::Int8(w) => w.forward(x, rows),
            Weights::Q8(w) => w.matmul(x, rows, self.in_features, n),
        };
        if self.bias.is_none() && !relu {
            return out;
        }
        team::chunks_mut(&mut out, EPILOGUE_ROWS * n, |_, block| {
            for row in block.chunks_exact_mut(n) {
                if let Some(bias) = &self.bias {
                    for (value, b) in row.iter_mut().zip(bias) {
                        *value += b;
                    }
                }
                if relu {
                    row.iter_mut().for_each(|v| *v = v.max(0.0));
                }
            }
        });
        out
    }
}

/// `Precision::Auto` gives int8 weights and activations to layers this wide
/// and more, f16 to the rest. With beam search on the 456 English test clips,
/// int8 from 384 wide gave Whisper tiny 760 and base.en 468 errors (f16:
/// 768 and 479) at 64 and 111 ms per clip (f16: 87 and 174). Moonshine tiny
/// (288 wide) lost accuracy to int8 (568 against 557), so it stays f16.
const INT8_MIN_WIDTH: usize = 384;
/// `Precision::Auto` keeps token tables this wide and more as int8 (f32
/// math): a decoder reads the whole output projection for every token.
const INT8_TABLE_WIDTH: usize = 1024;

/// How a model stores its dense layers and token tables.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    /// Int8 where it keeps the accuracy (layers 384 wide and more), f16
    /// elsewhere. The fastest choice that keeps the reference accuracy.
    #[default]
    Auto,
    /// Int8 everywhere: int8 activations where the kernels take the shape,
    /// int8 storage with f32 math elsewhere. The least memory; narrow
    /// models lose some accuracy.
    Int8,
    /// f16 weights, f32 math: half the memory of f32, the same text.
    F16,
    /// The weights as the reference runtimes hold them; the slowest.
    F32,
}

impl Precision {
    pub const NAMES: [&str; 4] = ["auto", "int8", "f16", "f32"];

    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "auto" => Ok(Self::Auto),
            "int8" => Ok(Self::Int8),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            other => Err(format!(
                "precision `{other}`: use one of {}",
                Self::NAMES.join(", ")
            )),
        }
    }

    pub fn name(self) -> &'static str {
        Self::NAMES[self as usize]
    }
}
