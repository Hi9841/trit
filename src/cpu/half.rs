//! Matrix-vector product with f16 weights, for the decoder. Each decoding
//! step reads every weight once, so the step is bound by memory traffic;
//! keeping the weights as stored (f16) halves it at no loss of precision.

use super::buf::Buf;
use super::fgemm::{level, Level};
use super::store::Store;
use super::team;
use super::tensors::f16_to_f32;

/// Below this total multiply-add count, run `matvec` inline.
const INLINE_WORK: usize = 131_072;
/// Above this total multiply-add count, give each thread two tasks to absorb
/// wake jitter while keeping task counts low.
const TWO_WAVE_WORK: usize = 1_000_000;

pub struct HalfDense {
    pub out_features: usize,
    pub in_features: usize,
    weight: Store<u16>,
    bias: Option<Vec<f32>>,
}

impl HalfDense {
    /// `weight` is row-major `out x in` f16 bits.
    pub fn new(
        weight: Vec<u16>,
        bias: Option<Vec<f32>>,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self, String> {
        if weight.len() != out_features * in_features
            || bias.as_ref().is_some_and(|b| b.len() != out_features)
        {
            return Err(format!(
                "f16 layer {out_features}x{in_features} has {} weights",
                weight.len()
            ));
        }
        Ok(Self {
            out_features,
            in_features,
            weight: Store::weights(weight),
            bias,
        })
    }

    /// Rows of `a` followed by rows of `b`, side by side: `[a | b]`, so one
    /// product covers `a * x + b * y` for the input `[x | y]`.
    pub fn side_by_side<T: Copy>(
        a: &[T],
        b: &[T],
        rows: usize,
        a_cols: usize,
        b_cols: usize,
    ) -> Vec<T> {
        let mut out = Vec::with_capacity(rows * (a_cols + b_cols));
        for r in 0..rows {
            out.extend_from_slice(&a[r * a_cols..(r + 1) * a_cols]);
            out.extend_from_slice(&b[r * b_cols..(r + 1) * b_cols]);
        }
        out
    }

    /// `rows` input vectors at once (`rows x in_features`), for the few
    /// decoder products that cover many frames: one pass over the weights.
    pub fn forward(&self, x: &[f32], rows: usize) -> Buf {
        super::fgemm::matmul_then(
            x,
            rows,
            self.in_features,
            &self.weight,
            self.out_features,
            self.bias.as_deref(),
            false,
        )
    }

    pub fn matvec(&self, x: &[f32]) -> Buf {
        assert_eq!(x.len(), self.in_features);
        let (n, k) = (self.out_features, self.in_features);
        let _product = crate::prof::product(crate::prof::Product::F16Vector, n, k, 1);
        if super::v2::float_enabled() {
            let mut out = super::v2::float::matmul(x, 1, k, &self.weight, n);
            if let Some(bias) = &self.bias {
                for (o, b) in out.iter_mut().zip(bias.iter()) {
                    *o += b;
                }
            }
            return out;
        }
        let mut out = Buf::scratch(n);
        if n == 0 {
            return out;
        }
        let level = level();
        let threads = team::threads().max(1);
        let work = n.saturating_mul(k);
        let row_groups = n.div_ceil(8);
        let tasks = if threads == 1 || work <= INLINE_WORK {
            1
        } else if team::is_heterogeneous() {
            // Finer chunks on mixed-capacity teams let fast cores steal from slow cores.
            (4 * threads).min(row_groups).min((work / 65_536).max(1))
        } else if work < TWO_WAVE_WORK {
            threads.min(row_groups)
        } else {
            (2 * threads).min(row_groups)
        };
        let out_ptr = team::SendPtr::new(&mut out);
        let bias = self.bias.as_deref();
        team::for_each(tasks, |task| {
            let first = (task * row_groups / tasks) * 8;
            let end = (((task + 1) * row_groups / tasks) * 8).min(n);
            let count = end - first;
            // SAFETY: tasks cover disjoint row spans of `out`; `out` outlives `for_each`.
            let rows = unsafe { out_ptr.slice_mut(first, count) };
            let weights = &self.weight[first * k..end * k];
            let row_bias = bias.map(|b| &b[first..end]);
            match level {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: the level was detected from the CPU.
                Level::Avx512 => unsafe { avx512_rows(weights, k, x, row_bias, rows) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: the level was detected from the CPU.
                Level::Avx2 => unsafe { avx2_rows(weights, k, x, row_bias, rows) },
                #[cfg(target_arch = "aarch64")]
                Level::Neon => neon_rows(weights, k, x, row_bias, rows),
                Level::Scalar => scalar_rows(weights, k, x, row_bias, rows),
            }
        });
        out
    }
}

fn scalar_rows(weights: &[u16], k: usize, x: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
    for (r, (row, o)) in weights.chunks_exact(k).zip(out.iter_mut()).enumerate() {
        let mut total: f32 = row.iter().zip(x).map(|(w, v)| f16_to_f32(*w) * v).sum();
        if let Some(bias) = bias {
            total += bias[r];
        }
        *o = total;
    }
}

/// Rows per pass of the 256-bit and 128-bit SIMD paths. One row is one chain
/// of dependent FMAs, which waits on their latency; four chains overlap and
/// share the loads of `x`. Each row's sum keeps its order, so the values do
/// not change.
const ROWS_AT_ONCE: usize = 4;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn avx512_rows(weights: &[u16], k: usize, x: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
    let mut first = 0;
    // 8 rows keep both Zen 4 FMA pipes saturated (8 chains >= 4-cycle latency x 2/cycle)
    // and halve `x` loads; drop to 4 when the f16 row stride is a 4 KB L1 alias.
    if !k.is_multiple_of(2048) {
        while first + 8 <= out.len() {
            let b = bias.map(|b| &b[first..]);
            avx512_tile::<8>(&weights[first * k..], k, x, b, &mut out[first..]);
            first += 8;
        }
    }
    while first + ROWS_AT_ONCE <= out.len() {
        let b = bias.map(|b| &b[first..]);
        avx512_tile::<ROWS_AT_ONCE>(&weights[first * k..], k, x, b, &mut out[first..]);
        first += ROWS_AT_ONCE;
    }
    while first < out.len() {
        let b = bias.map(|b| &b[first..]);
        avx512_tile::<1>(&weights[first * k..], k, x, b, &mut out[first..]);
        first += 1;
    }
}

/// `R` rows of `weights` from its start into `out[..R]`, adding `bias[..R]` if present.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::needless_range_loop)]
unsafe fn avx512_tile<const R: usize>(
    weights: &[u16],
    k: usize,
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    assert!(
        weights.len() >= R * k
            && out.len() >= R
            && x.len() >= k
            && bias.is_none_or(|b| b.len() >= R)
    );
    let mut acc = [_mm512_setzero_ps(); R];
    let mut i = 0;
    while i + 16 <= k {
        let xv = _mm512_loadu_ps(x.as_ptr().add(i));
        for r in 0..R {
            let w = _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(r * k + i).cast()));
            acc[r] = _mm512_fmadd_ps(w, xv, acc[r]);
        }
        i += 16;
    }
    for r in 0..R {
        let mut total = _mm512_reduce_add_ps(acc[r]);
        for j in i..k {
            total += f16_to_f32(weights[r * k + j]) * x[j];
        }
        if let Some(bias) = bias {
            total += bias[r];
        }
        out[r] = total;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn avx2_rows(weights: &[u16], k: usize, x: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
    let mut first = 0;
    while first + ROWS_AT_ONCE <= out.len() {
        let b = bias.map(|b| &b[first..]);
        avx2_tile::<ROWS_AT_ONCE>(&weights[first * k..], k, x, b, &mut out[first..]);
        first += ROWS_AT_ONCE;
    }
    while first < out.len() {
        let b = bias.map(|b| &b[first..]);
        avx2_tile::<1>(&weights[first * k..], k, x, b, &mut out[first..]);
        first += 1;
    }
}

/// `R` rows of `weights` from its start into `out[..R]`, adding `bias[..R]` if present.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::needless_range_loop)]
unsafe fn avx2_tile<const R: usize>(
    weights: &[u16],
    k: usize,
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    assert!(
        weights.len() >= R * k
            && out.len() >= R
            && x.len() >= k
            && bias.is_none_or(|b| b.len() >= R)
    );
    let mut acc = [_mm256_setzero_ps(); R];
    let mut i = 0;
    while i + 8 <= k {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        for r in 0..R {
            let w = _mm256_cvtph_ps(_mm_loadu_si128(weights.as_ptr().add(r * k + i).cast()));
            acc[r] = _mm256_fmadd_ps(w, xv, acc[r]);
        }
        i += 8;
    }
    for r in 0..R {
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc[r]);
        let mut total: f32 = lanes.iter().sum();
        for j in i..k {
            total += f16_to_f32(weights[r * k + j]) * x[j];
        }
        if let Some(bias) = bias {
            total += bias[r];
        }
        out[r] = total;
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_rows(weights: &[u16], k: usize, x: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
    let mut first = 0;
    while first + ROWS_AT_ONCE <= out.len() {
        let b = bias.map(|b| &b[first..]);
        neon_tile::<ROWS_AT_ONCE>(&weights[first * k..], k, x, b, &mut out[first..]);
        first += ROWS_AT_ONCE;
    }
    while first < out.len() {
        let b = bias.map(|b| &b[first..]);
        neon_tile::<1>(&weights[first * k..], k, x, b, &mut out[first..]);
        first += 1;
    }
}

/// `R` rows of `weights` from its start into `out[..R]`, adding `bias[..R]` if present.
/// Two accumulators per row, so four rows keep eight FMA chains in flight.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)]
fn neon_tile<const R: usize>(
    weights: &[u16],
    k: usize,
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    use super::fgemm::f16x4;
    use std::arch::aarch64::*;
    assert!(
        weights.len() >= R * k
            && out.len() >= R
            && x.len() >= k
            && bias.is_none_or(|b| b.len() >= R)
    );
    // SAFETY: the assert keeps every load below inside `weights` and `x`.
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); 2]; R];
        let mut i = 0;
        while i + 8 <= k {
            let xv = [
                vld1q_f32(x.as_ptr().add(i)),
                vld1q_f32(x.as_ptr().add(i + 4)),
            ];
            for r in 0..R {
                let w = weights.as_ptr().add(r * k + i);
                acc[r][0] = vfmaq_f32(acc[r][0], f16x4(w), xv[0]);
                acc[r][1] = vfmaq_f32(acc[r][1], f16x4(w.add(4)), xv[1]);
            }
            i += 8;
        }
        for r in 0..R {
            let mut total = vaddvq_f32(vaddq_f32(acc[r][0], acc[r][1]));
            for j in i..k {
                total += f16_to_f32(weights[r * k + j]) * x[j];
            }
            if let Some(bias) = bias {
                total += bias[r];
            }
            out[r] = total;
        }
    }
}
