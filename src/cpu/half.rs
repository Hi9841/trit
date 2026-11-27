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
    /// Maps a previously cached f16 layer directly from the active read cache.
    pub fn load_cached(
        tag: &str,
        bias: Option<Vec<f32>>,
        out_features: usize,
        in_features: usize,
    ) -> Option<Self> {
        if bias.as_ref().is_some_and(|b| b.len() != out_features) {
            return None;
        }
        let weight = Store::<u16>::mapped(tag, out_features * in_features)?;
        Some(Self {
            out_features,
            in_features,
            weight,
            bias,
        })
    }

    /// `weight` is row-major `out x in` f16 bits.
    pub fn new_named(
        tag: Option<&str>,
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
        let weight = match tag {
            Some(tag) => Store::named(tag, weight),
            None => Store::weights(weight),
        };
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
        matvec_slice(
            &self.weight,
            self.out_features,
            self.in_features,
            x,
            self.bias.as_deref(),
        )
    }
}

/// Matrix-vector product `out = w * x + bias` for row-major f16 weights `w`
/// (`n x k`). Uses 8-row FMA tiles (AVX-512 and NEON) and coarse thread tasks
/// (`<= 2 * threads`), avoiding the hundreds of tiny tasks and 3-column tiles
/// that `fgemm::matmul_then` runs for single-row decoder projections and token
/// tables.
pub(super) fn matvec_slice(
    weight: &[u16],
    n: usize,
    k: usize,
    x: &[f32],
    bias: Option<&[f32]>,
) -> Buf {
    assert_eq!(x.len(), k);
    assert!(weight.len() >= n * k && bias.is_none_or(|b| b.len() >= n));
    let _product = crate::prof::product(crate::prof::Product::F16Vector, n, k, 1);
    if super::v2::float_enabled() {
        let mut out = super::v2::float::matmul(x, 1, k, weight, n);
        if let Some(bias) = bias {
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
    team::for_each(tasks, |task| {
        let first = (task * row_groups / tasks) * 8;
        let end = (((task + 1) * row_groups / tasks) * 8).min(n);
        let count = end - first;
        // SAFETY: tasks cover disjoint row spans of `out`; `out` outlives `for_each`.
        let rows = unsafe { out_ptr.slice_mut(first, count) };
        let weights = &weight[first * k..end * k];
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
    if i == k && (R == 8 || R == 4) {
        for chunk in (0..R).step_by(4) {
            let mut sum = super::fgemm::avx512_hsum4(
                acc[chunk],
                acc[chunk + 1],
                acc[chunk + 2],
                acc[chunk + 3],
            );
            if let Some(bias) = bias {
                sum = _mm_add_ps(sum, _mm_loadu_ps(bias.as_ptr().add(chunk)));
            }
            _mm_storeu_ps(out.as_mut_ptr().add(chunk), sum);
        }
        return;
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
    // 8 rows (8 accumulators + 1 x + 1 w = 10 <= 16 YMMs) halve `x` loads and
    // saturate both FMA pipes; drop to 4 on 4 KB f16 row strides.
    if !k.is_multiple_of(2048) {
        while first + 8 <= out.len() {
            let b = bias.map(|b| &b[first..]);
            avx2_tile::<8>(&weights[first * k..], k, x, b, &mut out[first..]);
            first += 8;
        }
    }
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
    if i == k && (R == 8 || R == 4) {
        for chunk in (0..R).step_by(4) {
            let mut sum = super::fgemm::avx2_hsum4(
                acc[chunk],
                acc[chunk + 1],
                acc[chunk + 2],
                acc[chunk + 3],
            );
            if let Some(bias) = bias {
                sum = _mm_add_ps(sum, _mm_loadu_ps(bias.as_ptr().add(chunk)));
            }
            _mm_storeu_ps(out.as_mut_ptr().add(chunk), sum);
        }
        return;
    }
    for r in 0..R {
        let mut total = super::fgemm::avx2_hsum(acc[r]);
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
    // 8 rows (16 accumulators, 20 of 32 NEON registers) keep Apple Silicon's
    // 4 NEON FMA pipes saturated and halve `x` loads; drop to 4 on 4 KB row strides.
    if !k.is_multiple_of(2048) {
        while first + 8 <= out.len() {
            let b = bias.map(|b| &b[first..]);
            neon_tile::<8>(&weights[first * k..], k, x, b, &mut out[first..]);
            first += 8;
        }
    }
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
/// Two accumulators per row, loading 8 f16 weights per 128-bit vector (`fcvtl` + `fcvtl2`).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)]
fn neon_tile<const R: usize>(
    weights: &[u16],
    k: usize,
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    use super::fgemm::f16x8;
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
                let (w0, w1) = f16x8(weights.as_ptr().add(r * k + i));
                acc[r][0] = vfmaq_f32(acc[r][0], w0, xv[0]);
                acc[r][1] = vfmaq_f32(acc[r][1], w1, xv[1]);
            }
            i += 8;
        }
        if i == k && (R == 8 || R == 4) {
            for chunk in (0..R).step_by(4) {
                let s0 = vaddq_f32(acc[chunk][0], acc[chunk][1]);
                let s1 = vaddq_f32(acc[chunk + 1][0], acc[chunk + 1][1]);
                let s2 = vaddq_f32(acc[chunk + 2][0], acc[chunk + 2][1]);
                let s3 = vaddq_f32(acc[chunk + 3][0], acc[chunk + 3][1]);
                let mut sum = vpaddq_f32(vpaddq_f32(s0, s1), vpaddq_f32(s2, s3));
                if let Some(bias) = bias {
                    sum = vaddq_f32(sum, vld1q_f32(bias.as_ptr().add(chunk)));
                }
                vst1q_f32(out.as_mut_ptr().add(chunk), sum);
            }
            return;
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

#[cfg(test)]
mod tests {
    use super::super::tensors::f32_to_f16;
    use super::*;

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((*state >> 40) as u32) as f32 / ((1u32 << 24) as f32);
        bits * 2.0 - 1.0
    }

    #[test]
    fn half_dense_matvec_and_forward_cover_all_row_and_k_paths() {
        let mut seed = 0xdead_beef_cafe_1234u64;
        // Includes non-2048 k (8-row SIMD tiles), k = 2048 (4-row alias guard),
        // odd k tails, and every row remainder 0..=17 plus multi-task n = 131.
        for (n, k) in [
            (0usize, 16usize),
            (1, 1),
            (3, 7),
            (4, 8),
            (5, 15),
            (7, 16),
            (8, 17),
            (9, 32),
            (15, 33),
            (16, 64),
            (17, 640),
            (131, 640),
            (17, 2048),
        ] {
            let w_bits: Vec<u16> = (0..n * k).map(|_| f32_to_f16(lcg(&mut seed))).collect();
            let bias: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * 0.5).collect();
            let x: Vec<f32> = (0..k).map(|_| lcg(&mut seed)).collect();
            let tol = (k as f32).sqrt() * 2e-5 + 1e-4;

            for use_bias in [false, true] {
                let b_opt = use_bias.then(|| bias.clone());
                let layer = HalfDense::new_named(None, w_bits.clone(), b_opt, n, k)
                    .expect("valid HalfDense");
                let got = layer.matvec(&x);
                let mut want = vec![0.0f32; n];
                scalar_rows(
                    &w_bits,
                    k,
                    &x,
                    use_bias.then_some(bias.as_slice()),
                    &mut want,
                );
                assert_eq!(got.len(), n);
                for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        g.is_finite() && (g - w).abs() <= tol,
                        "matvec n={n} k={k} bias={use_bias} at {i}: got={g} want={w}"
                    );
                }

                // Multi-row forward must match row-by-row matvec.
                if n > 0 && k <= 64 {
                    let rows = 5;
                    let batch_x: Vec<f32> = (0..rows * k).map(|_| lcg(&mut seed)).collect();
                    let batch_got = layer.forward(&batch_x, rows);
                    for r in 0..rows {
                        let single = layer.matvec(&batch_x[r * k..(r + 1) * k]);
                        for (c, (&bg, &sg)) in batch_got[r * n..(r + 1) * n]
                            .iter()
                            .zip(single.iter())
                            .enumerate()
                        {
                            assert!(
                                (bg - sg).abs() <= tol,
                                "forward n={n} k={k} r={r} c={c}: got={bg} want={sg}"
                            );
                        }
                    }
                }
            }
        }
    }
}
