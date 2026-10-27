//! Matrix-vector product with f16 weights, for the decoder. Each decoding
//! step reads every weight once, so the step is bound by memory traffic;
//! keeping the weights as stored (f16) halves it at no loss of precision.

use super::buf::Buf;
use super::fgemm::{level, Level};
use super::team;
use super::tensors::f16_to_f32;

/// Output rows per parallel task.
const ROWS_PER_TASK: usize = 128;

pub struct HalfDense {
    pub out_features: usize,
    pub in_features: usize,
    weight: Vec<u16>,
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
            weight,
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
        let (n, k) = (self.out_features, self.in_features);
        let mut out = super::fgemm::matmul(x, rows, k, &self.weight, n);
        if let Some(bias) = &self.bias {
            for row in out.chunks_exact_mut(n) {
                for (o, b) in row.iter_mut().zip(bias) {
                    *o += b;
                }
            }
        }
        out
    }

    /// One input vector of `in_features` values.
    pub fn matvec(&self, x: &[f32]) -> Buf {
        assert_eq!(x.len(), self.in_features);
        let (n, k) = (self.out_features, self.in_features);
        let mut out = Buf::scratch(n);
        let level = level();
        team::chunks_mut(&mut out, ROWS_PER_TASK, |task, rows| {
            let first = task * ROWS_PER_TASK;
            let weights = &self.weight[first * k..(first + rows.len()) * k];
            match level {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: the level was detected from the CPU.
                Level::Avx512 => unsafe { avx512_rows(weights, k, x, rows) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: the level was detected from the CPU.
                Level::Avx2 => unsafe { avx2_rows(weights, k, x, rows) },
                Level::Scalar => scalar_rows(weights, k, x, rows),
            }
            if let Some(bias) = &self.bias {
                for (o, b) in rows.iter_mut().zip(&bias[first..]) {
                    *o += b;
                }
            }
        });
        out
    }
}

fn scalar_rows(weights: &[u16], k: usize, x: &[f32], out: &mut [f32]) {
    for (row, o) in weights.chunks_exact(k).zip(out.iter_mut()) {
        *o = row.iter().zip(x).map(|(w, v)| f16_to_f32(*w) * v).sum();
    }
}

/// Rows per pass of the SIMD paths. One row is one chain of dependent
/// FMAs, which waits on their latency; four chains overlap and share the
/// loads of `x`. Each row's sum keeps its order, so the values do not change.
const ROWS_AT_ONCE: usize = 4;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn avx512_rows(weights: &[u16], k: usize, x: &[f32], out: &mut [f32]) {
    let mut first = 0;
    while first + ROWS_AT_ONCE <= out.len() {
        avx512_tile::<ROWS_AT_ONCE>(&weights[first * k..], k, x, &mut out[first..]);
        first += ROWS_AT_ONCE;
    }
    while first < out.len() {
        avx512_tile::<1>(&weights[first * k..], k, x, &mut out[first..]);
        first += 1;
    }
}

/// `R` rows of `weights` from its start into `out[..R]`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::needless_range_loop)]
unsafe fn avx512_tile<const R: usize>(weights: &[u16], k: usize, x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    assert!(weights.len() >= R * k && out.len() >= R && x.len() >= k);
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
        out[r] = total;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn avx2_rows(weights: &[u16], k: usize, x: &[f32], out: &mut [f32]) {
    let mut first = 0;
    while first + ROWS_AT_ONCE <= out.len() {
        avx2_tile::<ROWS_AT_ONCE>(&weights[first * k..], k, x, &mut out[first..]);
        first += ROWS_AT_ONCE;
    }
    while first < out.len() {
        avx2_tile::<1>(&weights[first * k..], k, x, &mut out[first..]);
        first += 1;
    }
}

/// `R` rows of `weights` from its start into `out[..R]`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::needless_range_loop)]
unsafe fn avx2_tile<const R: usize>(weights: &[u16], k: usize, x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    assert!(weights.len() >= R * k && out.len() >= R && x.len() >= k);
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
        out[r] = total;
    }
}
