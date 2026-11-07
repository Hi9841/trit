//! Matrix products for the speech model. Weights are row-major `[out][in]`,
//! as the model file stores them, so every output is a dot product of two
//! contiguous rows.

use super::buf::Buf;
use super::fgemm;
use super::int8::{Int8, Q8_0_BLOCK};
use super::store::Store;
use super::team;

/// Output rows per task of the bias and ReLU pass.
const EPILOGUE_ROWS: usize = 64;

/// Dense layer with f32, f16, or int8 weights, `[out][in]`.
pub struct Dense {
    pub out_features: usize,
    pub in_features: usize,
    weight: Weights,
    bias: Option<Vec<f32>>,
    /// A GPU copy of the weights, present when the load asked for the GPU.
    #[cfg(feature = "gpu")]
    gpu: Option<Box<crate::gpu::Layer>>,
}

enum Weights {
    F32(Store<f32>),
    /// f16 bits, as stored; widened to f32 inside the product.
    F16(Store<u16>),
    /// Int8 with a scale per row and 128 inputs; i8 activations.
    Int8(Int8),
    /// Int8 storage with a scale per row and 128 inputs, widened to f32 for
    /// f32 math: half the memory of f16, the accuracy of f16 (the error of
    /// int8 products came from quantizing the activations).
    Q8(Q8),
}

/// Rows of int8 values with one f32 scale per `Q8_GROUP` values.
pub struct Q8 {
    values: Store<i8>,
    scales: Store<f32>,
    groups: usize,
}

const Q8_GROUP: usize = Q8_0_BLOCK;

/// ggml Q8_0 blocks of a row-major matrix: `Q8_0_BLOCK` codes per block and
/// the block's scale (stored as f16 in a GGUF file, widened to f32 here).
pub struct Blocks {
    pub codes: Vec<i8>,
    pub scales: Vec<f32>,
}

impl Blocks {
    /// Quantizes as ggml's `quantize_row_q8_0_ref`: per block `d = max|x| / 127`,
    /// codes `round(x * (1 / d))` (half away from zero) from the f32 `d`, and
    /// `d` stored as f16. A GGUF converter writes the same bytes from the
    /// same f32 weights. `values.len()` is a multiple of `Q8_0_BLOCK`.
    pub fn quantize(values: &[f32]) -> Self {
        let mut codes = vec![0i8; values.len()];
        let mut scales = Vec::with_capacity(values.len() / Q8_0_BLOCK);
        for (block, out) in values
            .as_chunks::<Q8_0_BLOCK>()
            .0
            .iter()
            .zip(codes.as_chunks_mut::<Q8_0_BLOCK>().0)
        {
            let max = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = max / 127.0;
            let inverse = if d != 0.0 { 1.0 / d } else { 0.0 };
            for (o, v) in out.iter_mut().zip(block) {
                *o = (v * inverse).round() as i8;
            }
            scales.push(super::tensors::f16_to_f32(super::tensors::f32_to_f16(d)));
        }
        Self { codes, scales }
    }

    /// `a` (`rows x ka`) and `b` (`rows x kb`) joined row by row; both widths
    /// are whole blocks, so every block stays as it is.
    pub fn side_by_side(a: &Self, b: &Self, rows: usize, ka: usize, kb: usize) -> Self {
        let (ga, gb) = (ka / Q8_0_BLOCK, kb / Q8_0_BLOCK);
        let mut codes = Vec::with_capacity(rows * (ka + kb));
        let mut scales = Vec::with_capacity(rows * (ga + gb));
        for row in 0..rows {
            codes.extend_from_slice(&a.codes[row * ka..(row + 1) * ka]);
            codes.extend_from_slice(&b.codes[row * kb..(row + 1) * kb]);
            scales.extend_from_slice(&a.scales[row * ga..(row + 1) * ga]);
            scales.extend_from_slice(&b.scales[row * gb..(row + 1) * gb]);
        }
        Self { codes, scales }
    }

    /// The values as f32: `code * scale`.
    pub fn to_f32(&self) -> Vec<f32> {
        self.codes
            .as_chunks::<Q8_0_BLOCK>()
            .0
            .iter()
            .zip(&self.scales)
            .flat_map(|(block, scale)| block.iter().map(move |c| f32::from(*c) * scale))
            .collect()
    }
}

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
                *scale = super::int8::group_scale(max);
                for (o, w) in out.iter_mut().zip(span) {
                    *o = super::int8::quantize_code(*w, *scale);
                }
            }
        });
        Self {
            values: Store::weights(values),
            scales: Store::weights(scales),
            groups,
        }
    }

    #[cfg(feature = "gpu")]
    fn to_f32(&self, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n * k];
        for row in 0..n {
            widen_q8(
                &self.values[row * k..(row + 1) * k],
                &self.scales[row * self.groups..(row + 1) * self.groups],
                &mut out[row * k..(row + 1) * k],
            );
        }
        out
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
        Self::checked(
            Weights::F32(Store::weights(weight)),
            len,
            bias,
            out_features,
            in_features,
        )
    }

    /// A layer with f16 weights (as bits); half the memory of `new`.
    pub fn half(
        weight: Vec<u16>,
        bias: Option<Vec<f32>>,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self, String> {
        let len = weight.len();
        Self::checked(
            Weights::F16(Store::weights(weight)),
            len,
            bias,
            out_features,
            in_features,
        )
    }

    /// A layer with ggml Q8_0 weights, as stored: int8 activations in groups
    /// of 32 where the int8 kernels take the shape, f32 math elsewhere.
    pub fn q8_0(
        blocks: Blocks,
        bias: Option<Vec<f32>>,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self, String> {
        let len = blocks.codes.len();
        let weight = if Int8::fits_group(out_features, in_features, Q8_0_BLOCK) {
            Weights::Int8(Int8::from_q8_0(
                out_features,
                in_features,
                &blocks.codes,
                &blocks.scales,
            ))
        } else {
            Weights::Q8(Q8 {
                values: Store::weights(blocks.codes),
                scales: Store::weights(blocks.scales),
                groups: in_features / Q8_GROUP,
            })
        };
        Self::checked(weight, len, bias, out_features, in_features)
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
        Self::load_in(t, parts, in_features, true)
    }

    /// `load`, with no GPU copy. The subsampler and the speech head use this.
    pub(crate) fn load_cpu(
        t: &super::tensors::Tensors,
        parts: &[(&str, usize)],
        in_features: usize,
    ) -> Result<Self, String> {
        Self::load_in(t, parts, in_features, false)
    }

    fn load_in(
        t: &super::tensors::Tensors,
        parts: &[(&str, usize)],
        in_features: usize,
        gpu: bool,
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
        let precision = t.precision_of(&format!("{}.weight", parts[0].0));
        let len = out_features * in_features;
        if precision == Precision::Q8_0 && in_features.is_multiple_of(Q8_0_BLOCK) {
            let mut blocks = Blocks {
                codes: Vec::with_capacity(len),
                scales: Vec::with_capacity(len / Q8_0_BLOCK),
            };
            for (name, rows) in parts {
                let part = t.q8_0(&format!("{name}.weight"), rows * in_features)?;
                blocks.codes.extend(part.codes);
                blocks.scales.extend(part.scales);
            }
            let layer = Self::q8_0(blocks, bias, out_features, in_features)?;
            return if gpu {
                layer.on_gpu(t.device())
            } else {
                Ok(layer)
            };
        }
        if !Self::int8(out_features, in_features, precision)
            && matches!(precision, Precision::Auto | Precision::F16)
        {
            // f16: read as f16 bits directly; no f32 copy of the layer exists.
            let mut half = Vec::with_capacity(len);
            for (name, rows) in parts {
                half.extend(t.f16_bits(&format!("{name}.weight"), rows * in_features)?);
            }
            let layer = Self::half(half, bias, out_features, in_features)?;
            return if gpu {
                layer.on_gpu(t.device())
            } else {
                Ok(layer)
            };
        }
        let mut weight = Vec::with_capacity(len);
        for (name, rows) in parts {
            weight.extend(t.f32_len(&format!("{name}.weight"), rows * in_features)?);
        }
        let layer = Self::with_precision(weight, bias, out_features, in_features, precision)?;
        if gpu {
            layer.on_gpu(t.device())
        } else {
            Ok(layer)
        }
    }

    /// Whether `load` stores a layer of this shape with int8 activations.
    fn int8(out_features: usize, in_features: usize, precision: Precision) -> bool {
        Int8::fits(out_features, in_features)
            && match precision {
                Precision::Int8 => true,
                Precision::Auto => out_features.min(in_features) >= INT8_MIN_WIDTH,
                Precision::F16 | Precision::F32 | Precision::Q8_0 => false,
            }
    }

    /// f32 weights (`out x in`) stored as `precision` says, by the rules of
    /// `load`: for a layer that is not one tensor of the file.
    pub fn with_precision(
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        out_features: usize,
        in_features: usize,
        precision: Precision,
    ) -> Result<Self, String> {
        let len = weight.len();
        if precision == Precision::Q8_0 && in_features.is_multiple_of(Q8_0_BLOCK) {
            return Self::q8_0(Blocks::quantize(&weight), bias, out_features, in_features);
        }
        let int8 = Self::int8(out_features, in_features, precision);
        if !int8 && matches!(precision, Precision::Auto | Precision::F16) {
            let half = weight
                .iter()
                .map(|v| super::tensors::f32_to_f16(*v))
                .collect();
            return Self::half(half, bias, out_features, in_features);
        }
        let weight = if int8 {
            Weights::Int8(Int8::quantize(out_features, in_features, |row, out| {
                out.copy_from_slice(&weight[row * in_features..(row + 1) * in_features]);
            }))
        } else if precision == Precision::Int8 {
            // A shape the int8 kernels cannot take: int8 storage, f32 math.
            Weights::Q8(Q8::quantize(&weight, out_features, in_features))
        } else {
            Weights::F32(Store::weights(weight))
        };
        Self::checked(weight, len, bias, out_features, in_features)
    }

    /// A token table `[rows][cols]` (embedding, output projection) without
    /// bias: f32 under `Precision::F32`, int8 storage with f32 math under
    /// `Precision::Int8`, Q8_0 blocks with f32 math under `Precision::Q8_0`,
    /// else f16. Never int8 activations: the logits decide every token.
    pub fn table(
        t: &super::tensors::Tensors,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        let len = rows * cols;
        let weight = match t.precision_of(name) {
            Precision::Q8_0 if cols.is_multiple_of(Q8_0_BLOCK) => {
                let blocks = t.q8_0(name, len)?;
                Weights::Q8(Q8 {
                    values: Store::weights(blocks.codes),
                    scales: Store::weights(blocks.scales),
                    groups: cols / Q8_GROUP,
                })
            }
            Precision::F32 => Weights::F32(Store::weights(t.f32_len(name, len)?)),
            Precision::Int8 => Weights::Q8(Q8::quantize(&t.f32_len(name, len)?, rows, cols)),
            Precision::Auto if cols >= INT8_TABLE_WIDTH => {
                Weights::Q8(Q8::quantize(&t.f32_len(name, len)?, rows, cols))
            }
            Precision::Auto | Precision::F16 | Precision::Q8_0 => {
                Weights::F16(Store::weights(t.f16_bits(name, len)?))
            }
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
            #[cfg(feature = "gpu")]
            gpu: None,
        })
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_layer(&self) -> Option<&crate::gpu::Layer> {
        self.gpu.as_deref()
    }

    /// Uploads a GPU copy when `device` is `Gpu`. The CPU weights stay.
    pub(crate) fn on_gpu(self, device: crate::Device) -> Result<Self, String> {
        #[cfg(feature = "gpu")]
        if device == crate::Device::Gpu {
            let uploaded = self.upload_gpu()?;
            let mut layer = self;
            layer.gpu = Some(Box::new(uploaded));
            return Ok(layer);
        }
        #[cfg(not(feature = "gpu"))]
        let _ = device;
        Ok(self)
    }

    #[cfg(feature = "gpu")]
    fn upload_gpu(&self) -> Result<crate::gpu::Layer, String> {
        let bias = self.bias.as_deref();
        let (n, k) = (self.out_features, self.in_features);
        match &self.weight {
            Weights::F32(weight) => crate::gpu::Layer::f32(n, k, weight, bias),
            Weights::F16(weight) => crate::gpu::Layer::f16_bits(n, k, weight, bias),
            Weights::Q8(weight) => upload_wide(&weight.to_f32(n, k), n, k, bias),
            Weights::Int8(weight) => upload_wide(&weight.dequant_f32(), n, k, bias),
        }
    }

    /// `x` is `rows x in_features`; the result is `rows x out_features`.
    pub fn forward(&self, x: &[f32], rows: usize) -> Buf {
        self.forward_then(x, rows, false)
    }

    pub fn forward_relu(&self, x: &[f32], rows: usize) -> Buf {
        self.forward_then(x, rows, true)
    }

    fn forward_then(&self, x: &[f32], rows: usize, relu: bool) -> Buf {
        #[cfg(feature = "gpu")]
        if let Some(layer) = &self.gpu {
            if rows >= crate::gpu::MIN_ROWS && !crate::gpu::host_products() && crate::gpu::usable()
            {
                match layer.forward(x, rows, relu) {
                    Ok(out) => {
                        crate::gpu::worked();
                        return out;
                    }
                    Err(error) => crate::gpu::failed(&error),
                }
            }
        }
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

/// Int8 and Q8 weights become f16 on the GPU when the adapter has
/// `SHADER_F16`, and f32 otherwise.
#[cfg(feature = "gpu")]
fn upload_wide(
    weight: &[f32],
    n: usize,
    k: usize,
    bias: Option<&[f32]>,
) -> Result<crate::gpu::Layer, String> {
    if crate::gpu::Gpu::get()?.shader_f16() {
        let bits: Vec<u16> = weight
            .iter()
            .copied()
            .map(super::tensors::f32_to_f16)
            .collect();
        crate::gpu::Layer::f16_bits(n, k, &bits, bias)
    } else {
        crate::gpu::Layer::f32(n, k, weight, bias)
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
    /// ggml's Q8_0, as llama.cpp and transcribe.cpp run it: weights in
    /// blocks of 32 int8 values with an f16 scale, activations quantized the
    /// same way, i8 x i8 products. A GGUF file's Q8_0 tensors are used as
    /// stored; other weights are quantized at load as ggml quantizes them.
    Q8_0,
}

impl Precision {
    pub const NAMES: [&str; 5] = ["auto", "int8", "f16", "f32", "q8_0"];

    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "auto" => Ok(Self::Auto),
            "int8" => Ok(Self::Int8),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            "q8_0" => Ok(Self::Q8_0),
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
