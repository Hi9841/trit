//! Transformer pieces that several families share: layer norm, GELU,
//! rotary positions, multi-head attention, and layer loading.

use crate::cpu::{activation, buf::Buf, fgemm, gemm::Dense, team, tensors::Tensors};

const LAYER_NORM_EPS: f32 = 1e-5;

/// Layers with the same input and `out` rows each, stacked by output rows
/// (`Dense::load`: int8 where the shape allows). A missing bias is zero.
pub fn stacked(t: &Tensors, names: &[&str], out: usize, inp: usize) -> Result<Dense, String> {
    let parts: Vec<(&str, usize)> = names.iter().map(|name| (*name, out)).collect();
    Dense::load(t, &parts, inp)
}

pub fn linear(t: &Tensors, name: &str, out: usize, inp: usize) -> Result<Dense, String> {
    stacked(t, &[name], out, inp)
}

/// Layer norm over rows of `d`; the bias is optional (Moonshine has none).
pub struct Norm {
    weight: Vec<f32>,
    /// Zeros when the model has no bias.
    bias: Vec<f32>,
}

impl Norm {
    pub fn load(t: &Tensors, name: &str, d: usize) -> Result<Self, String> {
        let bias = format!("{name}.bias");
        Ok(Self {
            weight: t.f32_len(&format!("{name}.weight"), d)?,
            bias: if t.has(&bias) {
                t.f32_len(&bias, d)?
            } else {
                vec![0.0; d]
            },
        })
    }

    /// Rows of `x`, normalized; a row is `d` (the weight's size) values.
    pub fn apply(&self, x: &[f32], d: usize) -> Buf {
        debug_assert_eq!(d, self.weight.len());
        let mut out = Buf::scratch(x.len());
        team::chunks_mut(&mut out, d, |index, o| {
            normalize_row(&x[index * d..(index + 1) * d], &self.weight, &self.bias, o);
        });
        out
    }

    /// `x += scale * y`, then the normalized `x`, in one pass per row.
    pub fn residual(&self, x: &mut [f32], y: &[f32], scale: f32) -> Buf {
        let size = self.weight.len();
        let mut out = Buf::scratch(x.len());
        let rows = team::SendPtr::new(x);
        team::chunks_mut(&mut out, size, |index, o| {
            // SAFETY: each row of `x` belongs to the task of the same output row.
            let row = unsafe { rows.slice_mut(index * size, size) };
            add_scaled(row, &y[index * size..(index + 1) * size], scale);
            normalize_row(row, &self.weight, &self.bias, o);
        });
        out
    }
}

multiversion! {
    /// `out = (row - mean) / sqrt(var + eps) * weight + bias`.
    fn normalize_row(row: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32]) {
        let (mean, var) = moments(row);
        let inv = 1.0 / (var + LAYER_NORM_EPS).sqrt();
        for (((o, v), w), b) in out.iter_mut().zip(row).zip(weight).zip(bias) {
            *o = (v - mean) * inv * w + b;
        }
    }
}

multiversion! {
    /// `row += scale * add`.
    fn add_scaled(row: &mut [f32], add: &[f32], scale: f32) {
        for (v, a) in row.iter_mut().zip(add) {
            *v += scale * a;
        }
    }
}

/// Mean and variance of a row, two passes, each summed in 32 lanes so the
/// adds do not wait on each other: eight 4-wide chains even on plain SSE2.
/// A lane adds the same values in the same order at any vector width.
#[inline(always)]
fn moments(row: &[f32]) -> (f32, f32) {
    let n = row.len() as f32;
    let mean = lane_sum(row, |v| v) / n;
    let var = lane_sum(row, |v| (v - mean) * (v - mean)) / n;
    (mean, var)
}

#[inline(always)]
fn lane_sum(row: &[f32], term: impl Fn(f32) -> f32) -> f32 {
    let mut lanes = [0.0f32; 32];
    let (chunks, tail) = row.as_chunks::<32>();
    for chunk in chunks {
        for (lane, v) in lanes.iter_mut().zip(chunk) {
            *lane += term(*v);
        }
    }
    lanes.iter().sum::<f32>() + tail.iter().map(|v| term(*v)).sum::<f32>()
}

/// RMS norm over rows of `d`: `x * weight / sqrt(mean(x^2) + eps)`.
pub struct RmsNorm {
    weight: Vec<f32>,
    eps: f32,
}

impl RmsNorm {
    pub fn load(t: &Tensors, name: &str, d: usize, eps: f32) -> Result<Self, String> {
        Ok(Self {
            weight: t.f32_len(&format!("{name}.weight"), d)?,
            eps,
        })
    }

    /// Normalizes every `d`-wide row of `x` in place.
    pub fn apply_in_place(&self, x: &mut [f32]) {
        let d = self.weight.len();
        for row in x.chunks_exact_mut(d) {
            let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / d as f32 + self.eps).sqrt();
            for (v, w) in row.iter_mut().zip(&self.weight) {
                *v *= inv * w;
            }
        }
    }

    pub fn apply(&self, x: &[f32]) -> Buf {
        let d = self.weight.len();
        let mut out = Buf::scratch(x.len());
        out.copy_from_slice(x);
        team::chunks_mut(&mut out, d, |_, row| self.apply_in_place(row));
        out
    }
}

/// The two GELU forms. The reference models (PyTorch, Transformers) use
/// the exact one; whisper.cpp uses tanh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gelu {
    /// `x * (1 + erf(x / sqrt 2)) / 2`: config `gelu`.
    Exact,
    /// The tanh approximation: config `gelu_new`, `gelu_pytorch_tanh`, `gelu_fast`.
    Tanh,
}

impl Gelu {
    /// The form a config's activation name asks for.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "gelu" => Ok(Self::Exact),
            "gelu_new" | "gelu_pytorch_tanh" | "gelu_fast" => Ok(Self::Tanh),
            other => Err(format!(
                "config.json asks for activation `{other}`; Trit runs gelu forms here"
            )),
        }
    }

    /// GELU on `values`, row by row (`d` values per row).
    pub fn apply(self, values: &mut [f32], d: usize) {
        const C: f32 = 0.797_884_6; // sqrt(2 / pi)
        team::chunks_mut(values, d, |_, row| match self {
            Self::Exact => activation::gelu(row),
            Self::Tanh => {
                // In stack blocks: a task allocates nothing.
                let mut inner = [0.0f32; 256];
                for block in row.chunks_mut(inner.len()) {
                    let inner = &mut inner[..block.len()];
                    for (t, x) in inner.iter_mut().zip(block.iter()) {
                        *t = C * (x + 0.044_715 * x * x * x);
                    }
                    activation::tanh(inner);
                    for (x, t) in block.iter_mut().zip(inner.iter()) {
                        *x *= 0.5 * (1.0 + t);
                    }
                }
            }
        });
    }
}

/// `x += y`.
pub fn add_into(x: &mut [f32], y: &[f32]) {
    team::chunks_mut(x, 4096, |chunk, xs| {
        for (a, b) in xs.iter_mut().zip(&y[chunk * 4096..]) {
            *a += b;
        }
    });
}

/// Rotary positions on the first `dims` values of each head. Pair `i`
/// turns by `position * base^(-2i / dims)`. Interleaved pairs are values
/// `2i` and `2i + 1` (Moonshine); halves pairs are `i` and `i + dims / 2`
/// (Llama, Qwen).
pub struct Rotary {
    dims: usize,
    inv_freq: Vec<f32>,
    interleaved: bool,
}

impl Rotary {
    pub fn interleaved(dims: usize, base: f32) -> Self {
        Self::new(dims, base, true)
    }

    pub fn halves(dims: usize, base: f32) -> Self {
        Self::new(dims, base, false)
    }

    fn new(dims: usize, base: f32, interleaved: bool) -> Self {
        Self {
            dims,
            inv_freq: (0..dims / 2)
                .map(|i| base.powf(-((2 * i) as f32) / dims as f32))
                .collect(),
            interleaved,
        }
    }

    /// Rotates every head of `rows` rows of `x` (row stride `stride`, heads
    /// of `hd` from offset 0); row `r` is at position `first + r`.
    pub fn apply(
        &self,
        x: &mut [f32],
        (rows, stride): (usize, usize),
        (heads, hd): (usize, usize),
        first: usize,
    ) {
        for r in 0..rows {
            let position = (first + r) as f32;
            for h in 0..heads {
                let head = &mut x[r * stride + h * hd..][..self.dims];
                let half = self.dims / 2;
                for (i, f) in self.inv_freq.iter().enumerate() {
                    let (sin, cos) = (position * f).sin_cos();
                    let (ia, ib) = if self.interleaved {
                        (2 * i, 2 * i + 1)
                    } else {
                        (i, i + half)
                    };
                    let (a, b) = (head[ia], head[ib]);
                    head[ia] = a * cos - b * sin;
                    head[ib] = b * cos + a * sin;
                }
            }
        }
    }
}

/// Softmax attention of `tq` queries over `tk` keys for every head.
/// `q`, `k`, `v` are row-major with row strides `qs`, `ks`, `vs`; query head
/// `h` uses columns `h * hd..(h + 1) * hd` of the first `d`, and key and
/// value head `h / (heads / kv_heads)` (grouped-query attention when
/// `kv_heads < heads`). With `causal`, query `i` sees keys up to
/// `tk - tq + i` (the queries are the last `tq` positions). Returns `tq x d`.
pub fn attention(
    (q, qs): (&[f32], usize),
    (k, ks): (&[f32], usize),
    (v, vs): (&[f32], usize),
    (tq, tk): (usize, usize),
    (d, heads, kv_heads): (usize, usize, usize),
    causal: bool,
) -> Buf {
    let kv = HeadMajor::new((k, ks), (v, vs), tk, (d / heads, kv_heads));
    attention_with((q, qs), &kv, tq, (d, heads), causal)
}

/// Keys and values per key/value head, laid out for `attention_with`: K
/// transposed (hd x tk) and V as it is (tk x hd), so both products are
/// `matmul_nn`, whose tiles need no horizontal sums. Build it once for keys
/// that do not change (Whisper's cross-attention over one window).
pub struct HeadMajor {
    kt: Buf,
    vh: Buf,
    tk: usize,
    hd: usize,
    kv_heads: usize,
}

impl HeadMajor {
    /// From `tk` rows of keys and values (row strides `ks`, `vs`).
    pub fn new(
        (k, ks): (&[f32], usize),
        (v, vs): (&[f32], usize),
        tk: usize,
        (hd, kv_heads): (usize, usize),
    ) -> Self {
        // From the pool, on the calling thread: the workers allocate nothing,
        // so their allocator heaps do not grow with every clip.
        let (mut kt, mut vh) = (
            Buf::scratch(kv_heads * hd * tk),
            Buf::scratch(kv_heads * tk * hd),
        );
        let (kt_ptr, vh_ptr) = (team::SendPtr::new(&mut kt), team::SendPtr::new(&mut vh));
        team::for_each(kv_heads, |g| {
            // SAFETY: head g owns its range of both buffers, which outlive the step.
            let (kt, vh) = unsafe {
                (
                    kt_ptr.slice_mut(g * hd * tk, hd * tk),
                    vh_ptr.slice_mut(g * tk * hd, tk * hd),
                )
            };
            for j in 0..tk {
                for (e, value) in k[j * ks + g * hd..][..hd].iter().enumerate() {
                    kt[e * tk + j] = *value;
                }
                vh[j * hd..(j + 1) * hd].copy_from_slice(&v[j * vs + g * hd..][..hd]);
            }
        });
        Self {
            kt,
            vh,
            tk,
            hd,
            kv_heads,
        }
    }
}

/// `attention` over keys and values already laid out per head.
pub fn attention_with(
    (q, qs): (&[f32], usize),
    kv: &HeadMajor,
    tq: usize,
    (d, heads): (usize, usize),
    causal: bool,
) -> Buf {
    let (hd, tk) = (kv.hd, kv.tk);
    assert_eq!(hd * heads, d);
    let group = heads / kv.kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    // One task per head and block of queries: 20 heads alone would leave
    // part of a 16-thread team idle for the second round.
    let chunk = ATTENTION_ROWS.min(tq);
    let chunks = tq.div_ceil(chunk);
    let mut out = Buf::scratch(tq * d);
    let out_ptr = team::SendPtr::new(&mut out);
    team::for_each(heads * chunks, |task| {
        let (h, c) = (task / chunks, task % chunks);
        let rows = c * chunk..((c + 1) * chunk).min(tq);
        let g = h / group;
        let (kt, vh) = (
            &kv.kt[g * hd * tk..][..hd * tk],
            &kv.vh[g * tk * hd..][..tk * hd],
        );
        SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let m = rows.len();
            let need = m * (2 * hd + tk);
            if scratch.len() < need {
                scratch.resize(need, 0.0);
            }
            let (qh, rest) = scratch.split_at_mut(m * hd);
            let (probs, rest) = rest.split_at_mut(m * tk);
            let heads_out = &mut rest[..m * hd];
            for (row, i) in qh.chunks_exact_mut(hd).zip(rows.clone()) {
                for (o, x) in row.iter_mut().zip(&q[i * qs + h * hd..][..hd]) {
                    *o = x * scale;
                }
            }
            fgemm::matmul_nn_into(qh, m, hd, kt, tk, probs);
            for (i, row) in rows.clone().zip(probs.chunks_exact_mut(tk)) {
                let visible = if causal { tk - tq + i + 1 } else { tk };
                row[visible..]
                    .iter_mut()
                    .for_each(|x| *x = f32::NEG_INFINITY);
                let max = row[..visible]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                row.iter_mut().for_each(|x| *x -= max);
                activation::exp(row);
                let inv = 1.0 / row.iter().sum::<f32>();
                row.iter_mut().for_each(|x| *x *= inv);
            }
            fgemm::matmul_nn_into(probs, m, tk, vh, hd, heads_out);
            for (part, i) in heads_out.chunks_exact(hd).zip(rows.clone()) {
                // SAFETY: row i, head h belongs to this task only; `out` outlives the step.
                unsafe { out_ptr.slice_mut(i * d + h * hd, hd) }.copy_from_slice(part);
            }
        });
    });
    out
}

thread_local! {
    /// One attention task's queries, probabilities, and head output. It grows
    /// to the largest task once per thread and is then reused.
    static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Queries per attention task.
const ATTENTION_ROWS: usize = 128;

/// Index of the largest value; the first one wins a tie.
pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// Im2col for a 1-D convolution over time-major `x` (`frames x channels`):
/// output row `t` holds `x[t * stride + tap - pad][c]` at `c * kernel + tap`,
/// the order of a `[out][in][kernel]` weight. Positions outside are zero.
pub fn conv_columns(
    x: &[f32],
    (frames, channels): (usize, usize),
    (kernel, stride, pad): (usize, usize, usize),
    out_frames: usize,
) -> Buf {
    let mut cols = Buf::scratch(out_frames * channels * kernel);
    team::chunks_mut(&mut cols, channels * kernel, |t, row| {
        for tap in 0..kernel {
            let src = (t * stride + tap).checked_sub(pad).filter(|s| *s < frames);
            for c in 0..channels {
                row[c * kernel + tap] = src.map_or(0.0, |s| x[s * channels + c]);
            }
        }
    });
    cols
}
