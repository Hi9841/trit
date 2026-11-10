//! f32 product `out = x * w^T` for row-major `x` (m x k) and `w` (n x k),
//! with `w` as f32 or f16.
//!
//! An output is one dot product along k, with **one sum order on every
//! path**: 16 lanes, lane `i` adds `x[16j + i] * w[16j + i]` with one
//! rounding (a fused multiply-add) in the order of `j`; the last 16 values
//! are padded with zeros; then lane `i` adds lane `i + 8`, `i + 4`, `i + 2`,
//! and the last two (the tree of `_mm512_reduce_add_ps`). AVX-512 holds the
//! 16 lanes in one register, AVX2 in two, NEON in four, and the scalar path
//! in an array. So every value is the same, bit for bit, on every CPU, and
//! the text cannot change with the CPU. The first kernel (`fgemm.rs`) adds
//! in 16 lanes on AVX-512 but in 8 on AVX2 and 4 on NEON, with the k tail
//! value by value: on Parakeet Ultra, AVX-512 gives "land?" for a clip
//! where AVX2 gives "land.". On AVX-512 the two kernels give the same bits.
//!
//! Other differences from the first kernel:
//!
//! - **f16 weights widen in the tile, always.** The first kernel writes an
//!   f32 copy of a 64-row weight panel for each block of 5 rows or more. On
//!   Zen 4 the conversion does not use the two multiply pipes, so the copy
//!   only adds memory traffic: 5 rows (a decoder step with 5 beams) ran at
//!   44 % of the multiply-add peak with the copy, and run at 70 % without.
//! - **5 rows are one strip** on AVX-512 (5 x 3), so a decoder step reads
//!   each weight vector once. The first kernel runs 4 x 3 and then 1 x 3.
//! - **The k tail is one masked load** on AVX-512, not one value at a time.
//!
//! Tested and not kept: wider tiles for 1, 2, and 3 rows (1 x 6, 2 x 6,
//! 3 x 4). They gave no gain for f16 and lost 15 to 30 % for f32.

use crate::cpu::buf::Buf;
use crate::cpu::fgemm::{level, Level};
use crate::cpu::team;

/// Rows and columns of a task. 63 columns are whole tiles of 3.
const ROWS: usize = 64;
const COLS: usize = 63;

/// Lanes of every sum.
const LANES: usize = 16;

/// A weight: f32, or f16 as its bits. The loads are `inline(always)`, so
/// they compile with the target features of the block that calls them.
pub trait Element: Copy + Send + Sync + 'static {
    fn as_f32(self) -> f32;
    #[cfg(target_arch = "x86_64")]
    unsafe fn wide16(at: *const Self) -> std::arch::x86_64::__m512;
    /// The first `count` values (fewer than 16) as f32, zeros after them.
    #[cfg(target_arch = "x86_64")]
    unsafe fn wide16_first(at: *const Self, count: usize) -> std::arch::x86_64::__m512;
    #[cfg(target_arch = "x86_64")]
    unsafe fn wide8(at: *const Self) -> std::arch::x86_64::__m256;
    #[cfg(target_arch = "aarch64")]
    unsafe fn wide4(at: *const Self) -> std::arch::aarch64::float32x4_t;
}

impl Element for f32 {
    #[inline(always)]
    fn as_f32(self) -> f32 {
        self
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn wide16(at: *const Self) -> std::arch::x86_64::__m512 {
        std::arch::x86_64::_mm512_loadu_ps(at)
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn wide16_first(at: *const Self, count: usize) -> std::arch::x86_64::__m512 {
        std::arch::x86_64::_mm512_maskz_loadu_ps((1u32 << count) as u16 - 1, at)
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn wide8(at: *const Self) -> std::arch::x86_64::__m256 {
        std::arch::x86_64::_mm256_loadu_ps(at)
    }
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn wide4(at: *const Self) -> std::arch::aarch64::float32x4_t {
        std::arch::aarch64::vld1q_f32(at)
    }
}

/// f16 bits. The CPUs of the AVX2 and AVX-512 levels have F16C; every
/// arm64 CPU has `fcvtl`.
impl Element for u16 {
    #[inline(always)]
    fn as_f32(self) -> f32 {
        crate::cpu::tensors::f16_to_f32(self)
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn wide16(at: *const Self) -> std::arch::x86_64::__m512 {
        use std::arch::x86_64::*;
        _mm512_cvtph_ps(_mm256_loadu_si256(at.cast()))
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn wide16_first(at: *const Self, count: usize) -> std::arch::x86_64::__m512 {
        use std::arch::x86_64::*;
        _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(
            (1u32 << count) as u16 - 1,
            at.cast(),
        ))
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn wide8(at: *const Self) -> std::arch::x86_64::__m256 {
        use std::arch::x86_64::*;
        _mm256_cvtph_ps(_mm_loadu_si128(at.cast()))
    }
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn wide4(at: *const Self) -> std::arch::aarch64::float32x4_t {
        crate::cpu::fgemm::f16x4(at)
    }
}

#[derive(Clone, Copy)]
struct OutPtr(*mut f32);

impl OutPtr {
    /// The pointer; a method, so closures capture the `Send` wrapper.
    fn get(self) -> *mut f32 {
        self.0
    }
}
// SAFETY: tasks write disjoint output blocks; the buffer outlives the step.
unsafe impl Send for OutPtr {}
// SAFETY: as for `Send`: tasks write disjoint output blocks.
unsafe impl Sync for OutPtr {}

/// `out[i][j] = sum_k x[i][k] * w[j][k]`, parallel over blocks.
pub fn matmul<E: Element>(x: &[f32], m: usize, k: usize, w: &[E], n: usize) -> Buf {
    assert!(x.len() >= m * k && w.len() >= n * k);
    let mut out = Buf::scratch(m * n);
    let col_blocks = n.div_ceil(COLS);
    let ptr = OutPtr(out.as_mut_ptr());
    let level = level();
    team::for_each(m.div_ceil(ROWS) * col_blocks, |task| {
        let (i, j) = ((task / col_blocks) * ROWS, (task % col_blocks) * COLS);
        let (rows, cols) = (ROWS.min(m - i), COLS.min(n - j));
        // SAFETY: the block is inside all three buffers and belongs to this
        // task only; the level was detected from the CPU.
        unsafe {
            block(
                level,
                x.as_ptr().add(i * k),
                rows,
                w.as_ptr().add(j * k),
                cols,
                k,
                ptr.get().add(i * n + j),
                n,
            );
        }
    });
    out
}

/// `matmul` into `out` (`m x n`), on the calling thread, for a task that
/// is itself one of many parallel tasks.
pub fn matmul_into(x: &[f32], m: usize, k: usize, w: &[f32], n: usize, out: &mut [f32]) {
    assert!(x.len() >= m * k && w.len() >= n * k && out.len() >= m * n);
    // SAFETY: the block covers `out` exactly; the level was detected from the CPU.
    unsafe {
        block(
            level(),
            x.as_ptr(),
            m,
            w.as_ptr(),
            n,
            k,
            out.as_mut_ptr(),
            n,
        )
    };
}

thread_local! {
    /// One f32 weight panel per worker, for `matmul_widened`.
    static PANEL: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// `matmul` where the weights come from `widen(cols, panel)`, which writes
/// rows `cols` of `w` as f32 into a `cols.len() x k` panel (int8 storage
/// with f32 math). Each block widens its columns once.
pub fn matmul_widened(
    x: &[f32],
    m: usize,
    k: usize,
    n: usize,
    widen: impl Fn(std::ops::Range<usize>, &mut [f32]) + Sync,
) -> Buf {
    assert!(x.len() >= m * k);
    let mut out = Buf::scratch(m * n);
    let col_blocks = n.div_ceil(COLS);
    let ptr = OutPtr(out.as_mut_ptr());
    let level = level();
    team::for_each(m.div_ceil(ROWS) * col_blocks, |task| {
        let (i, j) = ((task / col_blocks) * ROWS, (task % col_blocks) * COLS);
        let (rows, cols) = (ROWS.min(m - i), COLS.min(n - j));
        PANEL.with(|panel| {
            let mut panel = panel.borrow_mut();
            panel.resize(cols * k, 0.0);
            widen(j..j + cols, &mut panel);
            // SAFETY: as in `matmul`; the panel holds this block's columns.
            unsafe {
                block(
                    level,
                    x.as_ptr().add(i * k),
                    rows,
                    panel.as_ptr(),
                    cols,
                    k,
                    ptr.get().add(i * n + j),
                    n,
                );
            }
        });
    });
    out
}

/// `rows` rows of `x` by `cols` rows of `w`, on the calling thread. `out`
/// is the first value of the block, in a matrix `n` wide.
#[allow(clippy::too_many_arguments)]
unsafe fn block<E: Element>(
    level: Level,
    x: *const f32,
    rows: usize,
    w: *const E,
    cols: usize,
    k: usize,
    out: *mut f32,
    n: usize,
) {
    match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 => wide::block(x, rows, w, cols, k, out, n),
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 => half::block(x, rows, w, cols, k, out, n),
        #[cfg(target_arch = "aarch64")]
        Level::Neon => neon::block(x, rows, w, cols, k, out, n),
        Level::Scalar => scalar::block(x, rows, w, cols, k, out, n),
    }
}

/// Whether the CPU has FMA: then `mul_add` is one instruction.
#[cfg(target_arch = "x86_64")]
pub(super) fn fused() -> bool {
    static FMA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FMA.get_or_init(|| is_x86_feature_detected!("fma"))
}

/// The 16 lanes of a sum to one value, in the order of the tree.
#[inline(always)]
fn reduce(lanes: [f32; LANES]) -> f32 {
    let mut half = [0.0f32; 8];
    for (i, v) in half.iter_mut().enumerate() {
        *v = lanes[i] + lanes[i + 8];
    }
    let quarter = [
        half[0] + half[4],
        half[1] + half[5],
        half[2] + half[6],
        half[3] + half[7],
    ];
    let pair = [quarter[0] + quarter[2], quarter[1] + quarter[3]];
    pair[0] + pair[1]
}

/// The AVX2 and NEON tiles: up to 2 rows by 3 columns. Their k tail reads
/// copies padded with zeros to 16, which is what the masked loads of
/// AVX-512 give: the weight tails are made once per block of columns, the
/// row tails once per strip of rows.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
trait Padded {
    /// `R x C` dot products of length k; `x_tail` and `w_tail` are the
    /// padded tails, 16 values each, of the tile's rows and columns.
    #[allow(clippy::too_many_arguments)]
    unsafe fn tile<E: Element, const R: usize, const C: usize>(
        x: *const f32,
        w: *const E,
        k: usize,
        out: *mut f32,
        n: usize,
        x_tail: *const f32,
        w_tail: *const f32,
    );
}

/// Columns whose tails one block holds.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const TAIL_COLS: usize = 64;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn padded_block<P: Padded, E: Element>(
    x: *const f32,
    rows: usize,
    w: *const E,
    cols: usize,
    k: usize,
    out: *mut f32,
    n: usize,
) {
    let at = k - k % LANES;
    let mut j = 0;
    while j < cols {
        let span = (cols - j).min(TAIL_COLS);
        let mut w_tail = [[0.0f32; LANES]; TAIL_COLS];
        for (c, tail) in w_tail.iter_mut().enumerate().take(span) {
            for (t, v) in tail.iter_mut().enumerate().take(k - at) {
                *v = (*w.add((j + c) * k + at + t)).as_f32();
            }
        }
        let mut i = 0;
        while i < rows {
            let (xs, ws, os) = (x.add(i * k), w.add(j * k), out.add(i * n + j));
            if rows - i >= 2 {
                padded_strip::<P, E, 2>(xs, ws, span, k, os, n, &w_tail);
                i += 2;
            } else {
                padded_strip::<P, E, 1>(xs, ws, span, k, os, n, &w_tail);
                i += 1;
            }
        }
        j += span;
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn padded_strip<P: Padded, E: Element, const R: usize>(
    x: *const f32,
    w: *const E,
    cols: usize,
    k: usize,
    out: *mut f32,
    n: usize,
    w_tail: &[[f32; LANES]; TAIL_COLS],
) {
    let at = k - k % LANES;
    let mut x_tail = [[0.0f32; LANES]; R];
    for (r, tail) in x_tail.iter_mut().enumerate() {
        for (t, v) in tail.iter_mut().enumerate().take(k - at) {
            *v = *x.add(r * k + at + t);
        }
    }
    let (xt, wt) = (x_tail.as_ptr().cast::<f32>(), w_tail.as_ptr().cast::<f32>());
    let mut c = 0;
    while c + 3 <= cols {
        P::tile::<E, R, 3>(x, w.add(c * k), k, out.add(c), n, xt, wt.add(c * LANES));
        c += 3;
    }
    match cols - c {
        1 => P::tile::<E, R, 1>(x, w.add(c * k), k, out.add(c), n, xt, wt.add(c * LANES)),
        2 => P::tile::<E, R, 2>(x, w.add(c * k), k, out.add(c), n, xt, wt.add(c * LANES)),
        _ => {}
    }
}

/// `$tile::<E, R, C>` over all the columns of one strip of `R` rows: tiles
/// of `$full` columns, then one tile of the columns that remain.
#[cfg(target_arch = "x86_64")]
macro_rules! strip {
    ($name:ident, $rows:literal, $full:literal, [$($rest:literal)*]) => {
        #[inline(always)]
        unsafe fn $name<E: Element>(
            x: *const f32,
            w: *const E,
            cols: usize,
            k: usize,
            out: *mut f32,
            n: usize,
        ) {
            let mut j = 0;
            while j + $full <= cols {
                tile::<E, $rows, $full>(x, w.add(j * k), k, out.add(j), n);
                j += $full;
            }
            match cols - j {
                $($rest => tile::<E, $rows, $rest>(x, w.add(j * k), k, out.add(j), n),)*
                _ => {}
            }
        }
    };
}

/// AVX-512: strips of 1 to 5 rows by 3 columns. A strip reads `R + 3` rows
/// at once, 8 at most: rows of a power-of-two length fall into one set of
/// the 8-way L1 cache.
#[cfg(target_arch = "x86_64")]
mod wide {
    use super::Element;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma,f16c")]
    pub unsafe fn block<E: Element>(
        x: *const f32,
        rows: usize,
        w: *const E,
        cols: usize,
        k: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut i = 0;
        while i < rows {
            let (xs, os) = (x.add(i * k), out.add(i * n));
            // 5 rows are one strip: a decoder step has one row for each beam.
            let strip = match rows - i {
                left @ (1..=3 | 5) => left,
                _ => 4,
            };
            match strip {
                1 => strip1(xs, w, cols, k, os, n),
                2 => strip2(xs, w, cols, k, os, n),
                3 => strip3(xs, w, cols, k, os, n),
                5 => strip5(xs, w, cols, k, os, n),
                _ => strip4(xs, w, cols, k, os, n),
            }
            i += strip;
        }
    }

    strip!(strip1, 1, 3, [1 2]);
    strip!(strip2, 2, 3, [1 2]);
    strip!(strip3, 3, 3, [1 2]);
    strip!(strip4, 4, 3, [1 2]);
    strip!(strip5, 5, 3, [1 2]);

    /// R x C dot products of length k; the k tail uses masked loads.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile<E: Element, const R: usize, const C: usize>(
        x: *const f32,
        w: *const E,
        k: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut sums = [[_mm512_setzero_ps(); C]; R];
        let mut at = 0;
        while at + 16 <= k {
            let mut weights = [_mm512_setzero_ps(); C];
            for c in 0..C {
                weights[c] = E::wide16(w.add(c * k + at));
            }
            for r in 0..R {
                let row = _mm512_loadu_ps(x.add(r * k + at));
                for c in 0..C {
                    sums[r][c] = _mm512_fmadd_ps(row, weights[c], sums[r][c]);
                }
            }
            at += 16;
        }
        if at < k {
            let mask: __mmask16 = (1u32 << (k - at)) as u16 - 1;
            let mut weights = [_mm512_setzero_ps(); C];
            for c in 0..C {
                weights[c] = E::wide16_first(w.add(c * k + at), k - at);
            }
            for r in 0..R {
                let row = _mm512_maskz_loadu_ps(mask, x.add(r * k + at));
                for c in 0..C {
                    sums[r][c] = _mm512_fmadd_ps(row, weights[c], sums[r][c]);
                }
            }
        }
        for r in 0..R {
            for c in 0..C {
                // The tree of `super::reduce` (checked against it).
                *out.add(r * n + c) = _mm512_reduce_add_ps(sums[r][c]);
            }
        }
    }
}

/// AVX2: the 16 lanes of a sum are two registers, so 2 x 3 fills the 16
/// registers (12 sums, 3 weight vectors, 1 row). The first kernel fits
/// 4 x 3 with 8 lanes; for f32 weights and many rows, 2 x 3 is about 10 %
/// slower on Zen 4.
#[cfg(target_arch = "x86_64")]
mod half {
    use super::{Element, Padded, LANES};
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn block<E: Element>(
        x: *const f32,
        rows: usize,
        w: *const E,
        cols: usize,
        k: usize,
        out: *mut f32,
        n: usize,
    ) {
        super::padded_block::<Avx2, E>(x, rows, w, cols, k, out, n);
    }

    /// 16 values of `R` rows times 16 of `C` columns.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn add<V: Element, const R: usize, const C: usize>(
        sums: &mut [[[__m256; 2]; C]; R],
        x: *const f32,
        x_step: usize,
        w: *const V,
        w_step: usize,
    ) {
        for h in 0..2 {
            let mut weights = [_mm256_setzero_ps(); C];
            for c in 0..C {
                weights[c] = V::wide8(w.add(c * w_step + 8 * h));
            }
            for r in 0..R {
                let row = _mm256_loadu_ps(x.add(r * x_step + 8 * h));
                for c in 0..C {
                    sums[r][c][h] = _mm256_fmadd_ps(row, weights[c], sums[r][c][h]);
                }
            }
        }
    }

    struct Avx2;

    impl Padded for Avx2 {
        #[inline(always)]
        #[allow(clippy::needless_range_loop)]
        unsafe fn tile<E: Element, const R: usize, const C: usize>(
            x: *const f32,
            w: *const E,
            k: usize,
            out: *mut f32,
            n: usize,
            x_tail: *const f32,
            w_tail: *const f32,
        ) {
            let mut sums = [[[_mm256_setzero_ps(); 2]; C]; R];
            let mut at = 0;
            while at + 16 <= k {
                add::<E, R, C>(&mut sums, x.add(at), k, w.add(at), k);
                at += 16;
            }
            if at < k {
                add::<f32, R, C>(&mut sums, x_tail, LANES, w_tail, LANES);
            }
            for r in 0..R {
                for c in 0..C {
                    // Lanes i and i + 8, then the tree of `super::reduce`.
                    let v = _mm256_add_ps(sums[r][c][0], sums[r][c][1]);
                    let four = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
                    let two = _mm_add_ps(four, _mm_movehl_ps(four, four));
                    *out.add(r * n + c) =
                        _mm_cvtss_f32(_mm_add_ss(two, _mm_shuffle_ps::<1>(two, two)));
                }
            }
        }
    }
}

/// NEON: the 16 lanes of a sum are four registers, so 2 x 3 takes 28 of
/// the 32 registers (24 sums, 3 weight vectors, 1 row).
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{Element, Padded, LANES};
    use std::arch::aarch64::*;

    pub unsafe fn block<E: Element>(
        x: *const f32,
        rows: usize,
        w: *const E,
        cols: usize,
        k: usize,
        out: *mut f32,
        n: usize,
    ) {
        super::padded_block::<Neon, E>(x, rows, w, cols, k, out, n);
    }

    /// 16 values of `R` rows times 16 of `C` columns.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn add<V: Element, const R: usize, const C: usize>(
        sums: &mut [[[float32x4_t; 4]; C]; R],
        x: *const f32,
        x_step: usize,
        w: *const V,
        w_step: usize,
    ) {
        for q in 0..4 {
            let mut weights = [vdupq_n_f32(0.0); C];
            for c in 0..C {
                weights[c] = V::wide4(w.add(c * w_step + 4 * q));
            }
            for r in 0..R {
                let row = vld1q_f32(x.add(r * x_step + 4 * q));
                for c in 0..C {
                    sums[r][c][q] = vfmaq_f32(sums[r][c][q], row, weights[c]);
                }
            }
        }
    }

    struct Neon;

    impl Padded for Neon {
        #[inline(always)]
        #[allow(clippy::needless_range_loop)]
        unsafe fn tile<E: Element, const R: usize, const C: usize>(
            x: *const f32,
            w: *const E,
            k: usize,
            out: *mut f32,
            n: usize,
            x_tail: *const f32,
            w_tail: *const f32,
        ) {
            let mut sums = [[[vdupq_n_f32(0.0); 4]; C]; R];
            let mut at = 0;
            while at + 16 <= k {
                add::<E, R, C>(&mut sums, x.add(at), k, w.add(at), k);
                at += 16;
            }
            if at < k {
                add::<f32, R, C>(&mut sums, x_tail, LANES, w_tail, LANES);
            }
            for r in 0..R {
                for c in 0..C {
                    // Lanes i and i + 8, then the tree of `super::reduce`.
                    let [a, b, c2, d] = sums[r][c];
                    let four = vaddq_f32(vaddq_f32(a, c2), vaddq_f32(b, d));
                    let two = vadd_f32(vget_low_f32(four), vget_high_f32(four));
                    *out.add(r * n + c) = vget_lane_f32::<0>(two) + vget_lane_f32::<1>(two);
                }
            }
        }
    }
}

/// Scalar: the 16 lanes in an array, with `mul_add` (one rounding, as the
/// SIMD paths). A weight row widens once for all the rows of `x`.
mod scalar {
    use super::{reduce, Element, LANES};

    thread_local! {
        /// One weight row as f32, padded with zeros to a multiple of 16.
        static ROW: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    pub unsafe fn block<E: Element>(
        x: *const f32,
        rows: usize,
        w: *const E,
        cols: usize,
        k: usize,
        out: *mut f32,
        n: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        if super::fused() {
            // SAFETY: the CPU has FMA.
            return block_fma(x, rows, w, cols, k, out, n);
        }
        block_any(x, rows, w, cols, k, out, n);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "fma")]
    unsafe fn block_fma<E: Element>(
        x: *const f32,
        rows: usize,
        w: *const E,
        cols: usize,
        k: usize,
        out: *mut f32,
        n: usize,
    ) {
        block_any(x, rows, w, cols, k, out, n);
    }

    #[inline(always)]
    unsafe fn block_any<E: Element>(
        x: *const f32,
        rows: usize,
        w: *const E,
        cols: usize,
        k: usize,
        out: *mut f32,
        n: usize,
    ) {
        let padded = k.next_multiple_of(LANES);
        let mut wide = ROW.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
        wide.resize(padded, 0.0);
        wide[k..].fill(0.0);
        for j in 0..cols {
            for (t, v) in wide[..k].iter_mut().enumerate() {
                *v = (*w.add(j * k + t)).as_f32();
            }
            let weights = wide.as_chunks::<LANES>().0;
            for i in 0..rows {
                let (full, rest) = std::slice::from_raw_parts(x.add(i * k), k).as_chunks::<LANES>();
                let mut lanes = [0.0f32; LANES];
                let mut add = |x: &[f32; LANES], w: &[f32; LANES]| {
                    for ((lane, x), w) in lanes.iter_mut().zip(x).zip(w) {
                        *lane = x.mul_add(*w, *lane);
                    }
                };
                for (x, w) in full.iter().zip(weights) {
                    add(x, w);
                }
                if !rest.is_empty() {
                    let mut last = [0.0f32; LANES];
                    last[..rest.len()].copy_from_slice(rest);
                    add(&last, &weights[full.len()]);
                }
                *out.add(i * n + j) = reduce(lanes);
            }
        }
        ROW.with(|cell| *cell.borrow_mut() = wide);
    }
}
