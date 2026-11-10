//! `out = x * w` (m x n) for row-major `x` (m x k) and `w` (k x n), on the
//! calling thread: every output row is a sum of `w` rows scaled by `x`
//! values (attention against the values, where `k` is short).
//!
//! An output starts at 0 and adds `x[i][e] * w[e][j]` with one rounding (a
//! fused multiply-add) for `e` in order, on every path and in every column.
//! So every value is the same, bit for bit, on every CPU. The first kernel
//! (`fgemm.rs`) does this in its SIMD tiles, but adds the edge columns of
//! AVX2 and NEON, and every value of the scalar path, without fusing.

use crate::cpu::fgemm::{level, Level};

/// Rows of a tile: `x` values broadcast against one strip of a `w` row.
const ROWS: usize = 6;

pub fn matmul_nn_into(x: &[f32], m: usize, k: usize, w: &[f32], n: usize, out: &mut [f32]) {
    assert!(x.len() >= m * k && w.len() >= k * n && out.len() >= m * n);
    let (x, w, out) = (x.as_ptr(), w.as_ptr(), out.as_mut_ptr());
    // SAFETY: the assert keeps every access inside the three buffers; the
    // level was detected from the CPU.
    unsafe {
        match level() {
            #[cfg(target_arch = "x86_64")]
            Level::Avx512 => wide::block(x, m, k, w, n, out),
            #[cfg(target_arch = "x86_64")]
            Level::Avx2 => half::block(x, m, k, w, n, out),
            #[cfg(target_arch = "aarch64")]
            Level::Neon => neon::block(x, m, k, w, n, out),
            Level::Scalar => scalar(x, m, k, w, n, out),
        }
    }
}

/// The scalar path: with FMA on x86-64, `mul_add` is one instruction.
unsafe fn scalar(x: *const f32, m: usize, k: usize, w: *const f32, n: usize, out: *mut f32) {
    #[cfg(target_arch = "x86_64")]
    if super::float::fused() {
        // SAFETY: the CPU has FMA.
        return scalar_fma(x, m, k, w, n, out);
    }
    edge(x, 0..m, k, w, n, 0..n, out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn scalar_fma(x: *const f32, m: usize, k: usize, w: *const f32, n: usize, out: *mut f32) {
    edge(x, 0..m, k, w, n, 0..n, out);
}

/// Columns `cols` of rows `rows`, one value at a time.
#[inline(always)]
unsafe fn edge(
    x: *const f32,
    rows: std::ops::Range<usize>,
    k: usize,
    w: *const f32,
    n: usize,
    cols: std::ops::Range<usize>,
    out: *mut f32,
) {
    for i in rows {
        for j in cols.clone() {
            let mut total = 0.0f32;
            for e in 0..k {
                total = (*x.add(i * k + e)).mul_add(*w.add(e * n + j), total);
            }
            *out.add(i * n + j) = total;
        }
    }
}

/// Strips of `ROWS` rows by `$width` columns: `tile::<R>` for each full
/// strip of columns, then `last` for the columns that remain.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
macro_rules! strips {
    ($width:literal) => {
        #[inline(always)]
        unsafe fn strips(
            x: *const f32,
            m: usize,
            k: usize,
            w: *const f32,
            n: usize,
            out: *mut f32,
        ) {
            let mut i = 0;
            while i < m {
                let r = (m - i).min(super::ROWS);
                let mut j = 0;
                while j + $width <= n {
                    match r {
                        6 => tile::<6>(x, k, w, n, i, j, $width, out),
                        5 => tile::<5>(x, k, w, n, i, j, $width, out),
                        4 => tile::<4>(x, k, w, n, i, j, $width, out),
                        3 => tile::<3>(x, k, w, n, i, j, $width, out),
                        2 => tile::<2>(x, k, w, n, i, j, $width, out),
                        _ => tile::<1>(x, k, w, n, i, j, $width, out),
                    }
                    j += $width;
                }
                if j < n {
                    last(x, i..i + r, k, w, n, j, out);
                }
                i += r;
            }
        }
    };
}

/// AVX-512: 32 columns (two registers) per strip; the last strip is masked.
#[cfg(target_arch = "x86_64")]
mod wide {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f,fma")]
    pub unsafe fn block(x: *const f32, m: usize, k: usize, w: *const f32, n: usize, out: *mut f32) {
        strips(x, m, k, w, n, out);
    }

    strips!(32);

    /// The columns from `j`, fewer than 32, as one masked tile.
    #[inline(always)]
    unsafe fn last(
        x: *const f32,
        rows: std::ops::Range<usize>,
        k: usize,
        w: *const f32,
        n: usize,
        j: usize,
        out: *mut f32,
    ) {
        let (i, c) = (rows.start, n - j);
        match rows.len() {
            6 => tile::<6>(x, k, w, n, i, j, c, out),
            5 => tile::<5>(x, k, w, n, i, j, c, out),
            4 => tile::<4>(x, k, w, n, i, j, c, out),
            3 => tile::<3>(x, k, w, n, i, j, c, out),
            2 => tile::<2>(x, k, w, n, i, j, c, out),
            _ => tile::<1>(x, k, w, n, i, j, c, out),
        }
    }

    /// `R` rows by `c` columns (at most 32) from column `j`.
    #[inline(always)]
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    unsafe fn tile<const R: usize>(
        x: *const f32,
        k: usize,
        w: *const f32,
        n: usize,
        i: usize,
        j: usize,
        c: usize,
        out: *mut f32,
    ) {
        let mask = |from: usize| -> __mmask16 {
            match c.saturating_sub(from) {
                0 => 0,
                left if left >= 16 => 0xFFFF,
                left => ((1u32 << left) - 1) as u16,
            }
        };
        let (low, high) = (mask(0), mask(16));
        let mut sums = [[_mm512_setzero_ps(); 2]; R];
        for e in 0..k {
            let row = w.add(e * n + j);
            let b0 = _mm512_maskz_loadu_ps(low, row);
            // With a mask of 0 this points past the buffer, and nothing loads.
            let b1 = _mm512_maskz_loadu_ps(high, row.wrapping_add(16));
            for r in 0..R {
                let a = _mm512_set1_ps(*x.add((i + r) * k + e));
                sums[r][0] = _mm512_fmadd_ps(a, b0, sums[r][0]);
                sums[r][1] = _mm512_fmadd_ps(a, b1, sums[r][1]);
            }
        }
        for r in 0..R {
            let o = out.add((i + r) * n + j);
            _mm512_mask_storeu_ps(o, low, sums[r][0]);
            _mm512_mask_storeu_ps(o.wrapping_add(16), high, sums[r][1]);
        }
    }
}

/// AVX2: 16 columns (two registers) per strip; the edge columns are fused
/// one at a time.
#[cfg(target_arch = "x86_64")]
mod half {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn block(x: *const f32, m: usize, k: usize, w: *const f32, n: usize, out: *mut f32) {
        strips(x, m, k, w, n, out);
    }

    strips!(16);

    /// The columns from `j`, one value at a time.
    #[inline(always)]
    unsafe fn last(
        x: *const f32,
        rows: std::ops::Range<usize>,
        k: usize,
        w: *const f32,
        n: usize,
        j: usize,
        out: *mut f32,
    ) {
        super::edge(x, rows, k, w, n, j..n, out);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    unsafe fn tile<const R: usize>(
        x: *const f32,
        k: usize,
        w: *const f32,
        n: usize,
        i: usize,
        j: usize,
        _c: usize,
        out: *mut f32,
    ) {
        let mut sums = [[_mm256_setzero_ps(); 2]; R];
        for e in 0..k {
            let row = w.add(e * n + j);
            let (b0, b1) = (_mm256_loadu_ps(row), _mm256_loadu_ps(row.add(8)));
            for r in 0..R {
                let a = _mm256_set1_ps(*x.add((i + r) * k + e));
                sums[r][0] = _mm256_fmadd_ps(a, b0, sums[r][0]);
                sums[r][1] = _mm256_fmadd_ps(a, b1, sums[r][1]);
            }
        }
        for r in 0..R {
            let o = out.add((i + r) * n + j);
            _mm256_storeu_ps(o, sums[r][0]);
            _mm256_storeu_ps(o.add(8), sums[r][1]);
        }
    }
}

/// NEON: 16 columns (four registers) per strip, 24 sums; the edge columns
/// are fused one at a time.
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    pub unsafe fn block(x: *const f32, m: usize, k: usize, w: *const f32, n: usize, out: *mut f32) {
        strips(x, m, k, w, n, out);
    }

    strips!(16);

    /// The columns from `j`, one value at a time.
    #[inline(always)]
    unsafe fn last(
        x: *const f32,
        rows: std::ops::Range<usize>,
        k: usize,
        w: *const f32,
        n: usize,
        j: usize,
        out: *mut f32,
    ) {
        super::edge(x, rows, k, w, n, j..n, out);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    unsafe fn tile<const R: usize>(
        x: *const f32,
        k: usize,
        w: *const f32,
        n: usize,
        i: usize,
        j: usize,
        _c: usize,
        out: *mut f32,
    ) {
        let mut sums = [[vdupq_n_f32(0.0); 4]; R];
        for e in 0..k {
            let row = w.add(e * n + j);
            let b = [
                vld1q_f32(row),
                vld1q_f32(row.add(4)),
                vld1q_f32(row.add(8)),
                vld1q_f32(row.add(12)),
            ];
            for r in 0..R {
                let a = vdupq_n_f32(*x.add((i + r) * k + e));
                for q in 0..4 {
                    sums[r][q] = vfmaq_f32(sums[r][q], a, b[q]);
                }
            }
        }
        for r in 0..R {
            let o = out.add((i + r) * n + j);
            for (q, v) in sums[r].iter().enumerate() {
                vst1q_f32(o.add(4 * q), *v);
            }
        }
    }
}
