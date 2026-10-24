//! f32 matrix product `out = x * w^T` for row-major `x` (m x k) and `w`
//! (n x k). Both operands are read along k, so every output is a dot
//! product. Register tiles keep 4 x 4 (AVX-512) or 3 x 4 (AVX2) dot
//! products in flight and reduce each one once at the end. `w` holds f32
//! or f16 values (`Weight`); f16 rows widen to f32 as they load, so a model
//! stored in f16 keeps half the memory and reads half the bytes.

use super::buf::Buf;
use super::team;
use std::sync::OnceLock;

/// Rows and columns per parallel task.
const BLOCK: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Level {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Level {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Level::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Level::Avx2 => "avx2",
            #[cfg(target_arch = "x86_64")]
            Level::Avx512 => "avx512",
        }
    }
}

/// The widest float path this CPU runs. `TRIT_ISA=scalar|avx2` forces a slower one.
pub(crate) fn level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        let forced = std::env::var("TRIT_ISA").unwrap_or_default();
        #[cfg(target_arch = "x86_64")]
        {
            // F16C comes with every AVX2 CPU; the f16 decoder layers need it.
            let avx2 = is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("f16c");
            match forced.as_str() {
                "scalar" => return Level::Scalar,
                "avx2" if avx2 => return Level::Avx2,
                _ => {}
            }
            if is_x86_feature_detected!("avx512f") {
                return Level::Avx512;
            }
            if avx2 {
                return Level::Avx2;
            }
        }
        let _ = forced;
        Level::Scalar
    })
}

/// A weight element: f32, or f16 as its bits. The loads are `inline(always)`
/// so they compile with the target features of the kernel that calls them.
pub trait Weight: Copy + Send + Sync + 'static {
    /// True for f16: a block of many rows widens its weight panel to f32
    /// once, instead of once per register tile.
    const WIDEN: bool;
    fn to_f32(self) -> f32;
    /// `dst = src` as f32; the caller sizes `dst` like `src`.
    fn widen(src: &[Self], dst: &mut [f32]);
    /// 16 values from `p`.
    #[cfg(target_arch = "x86_64")]
    unsafe fn load16(p: *const Self) -> std::arch::x86_64::__m512;
    /// 8 values from `p`.
    #[cfg(target_arch = "x86_64")]
    unsafe fn load8(p: *const Self) -> std::arch::x86_64::__m256;
}

impl Weight for f32 {
    const WIDEN: bool = false;
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    fn widen(src: &[Self], dst: &mut [f32]) {
        dst.copy_from_slice(src);
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load16(p: *const Self) -> std::arch::x86_64::__m512 {
        std::arch::x86_64::_mm512_loadu_ps(p)
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load8(p: *const Self) -> std::arch::x86_64::__m256 {
        std::arch::x86_64::_mm256_loadu_ps(p)
    }
}

/// f16 bits. The CPUs of the AVX2 and AVX-512 paths have F16C (see `level`).
impl Weight for u16 {
    const WIDEN: bool = true;
    #[inline(always)]
    fn to_f32(self) -> f32 {
        super::tensors::f16_to_f32(self)
    }
    fn widen(src: &[Self], dst: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if level() != Level::Scalar {
            // SAFETY: the AVX2 and AVX-512 levels imply F16C (see `level`).
            unsafe { widen_f16c(src, dst) };
            return;
        }
        for (d, s) in dst.iter_mut().zip(src) {
            *d = s.to_f32();
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load16(p: *const Self) -> std::arch::x86_64::__m512 {
        use std::arch::x86_64::*;
        _mm512_cvtph_ps(_mm256_loadu_si256(p.cast()))
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load8(p: *const Self) -> std::arch::x86_64::__m256 {
        use std::arch::x86_64::*;
        _mm256_cvtph_ps(_mm_loadu_si128(p.cast()))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn widen_f16c(src: &[u16], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let (chunks, tail) = src.as_chunks::<8>();
    for (i, chunk) in chunks.iter().enumerate() {
        let wide = _mm256_cvtph_ps(_mm_loadu_si128(chunk.as_ptr().cast()));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i * 8), wide);
    }
    let done = chunks.len() * 8;
    for (d, s) in dst[done..].iter_mut().zip(tail) {
        *d = s.to_f32();
    }
}

thread_local! {
    /// One f32 weight panel per worker, reused across blocks.
    static PANEL: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy)]
struct OutPtr(*mut f32);

impl OutPtr {
    /// The pointer; a method, so closures capture the `Send` wrapper.
    fn get(self) -> *mut f32 {
        self.0
    }
}
// SAFETY: tasks write disjoint output blocks; the buffer outlives the parallel scope.
unsafe impl Send for OutPtr {}
// SAFETY: as for `Send`: tasks write disjoint output blocks.
unsafe impl Sync for OutPtr {}

/// `out[i][j] = sum_k x[i][k] * w[j][k]`, parallel over 64 x 64 blocks.
pub fn matmul<W: Weight>(x: &[f32], m: usize, k: usize, w: &[W], n: usize) -> Buf {
    assert!(x.len() >= m * k && w.len() >= n * k);
    let mut out = Buf::scratch(m * n);
    let (row_blocks, col_blocks) = (m.div_ceil(BLOCK), n.div_ceil(BLOCK));
    let ptr = OutPtr(out.as_mut_ptr());
    let level = level();
    team::for_each(row_blocks * col_blocks, |task| {
        let rows = (task / col_blocks) * BLOCK..((task / col_blocks + 1) * BLOCK).min(m);
        let cols = (task % col_blocks) * BLOCK..((task % col_blocks + 1) * BLOCK).min(n);
        if W::WIDEN && rows.len() > 4 {
            // Widen this block's weight rows once; every row tile reuses them.
            PANEL.with(|panel| {
                let mut panel = panel.borrow_mut();
                panel.resize(cols.len() * k, 0.0);
                W::widen(&w[cols.start * k..cols.end * k], &mut panel);
                let shifted = OutPtr(ptr.0.wrapping_add(cols.start));
                // SAFETY: as below; the panel holds this block's columns from 0,
                // so the output pointer moves to the block's first column.
                unsafe { block(level, x, k, &panel[..], n, rows, 0..cols.len(), shifted) };
            });
            return;
        }
        // SAFETY: the block is inside all three buffers and belongs to this task only;
        // the level was detected from the CPU.
        unsafe { block(level, x, k, w, n, rows, cols, ptr) };
    });
    out
}

/// `out[i][j] = sum_k x[i][k] * w[j][k]` where the weights come from
/// `widen(cols, panel)`, which writes rows `cols` of `w` as f32 into a
/// `cols.len() x k` panel. Each 64 x 64 block widens its columns once, then
/// runs the f32 tiles; for int8 storage with f32 math.
pub fn matmul_widened(
    x: &[f32],
    m: usize,
    k: usize,
    n: usize,
    widen: impl Fn(std::ops::Range<usize>, &mut [f32]) + Sync,
) -> Buf {
    assert!(x.len() >= m * k);
    let mut out = Buf::scratch(m * n);
    let (row_blocks, col_blocks) = (m.div_ceil(BLOCK), n.div_ceil(BLOCK));
    let ptr = OutPtr(out.as_mut_ptr());
    let level = level();
    team::for_each(row_blocks * col_blocks, |task| {
        let rows = (task / col_blocks) * BLOCK..((task / col_blocks + 1) * BLOCK).min(m);
        let cols = (task % col_blocks) * BLOCK..((task % col_blocks + 1) * BLOCK).min(n);
        PANEL.with(|panel| {
            let mut panel = panel.borrow_mut();
            panel.resize(cols.len() * k, 0.0);
            widen(cols.clone(), &mut panel);
            let shifted = OutPtr(ptr.get().wrapping_add(cols.start));
            // SAFETY: the panel holds this block's columns from 0, so the output
            // pointer moves to the block's first column; the block belongs to
            // this task only; the level was detected from the CPU.
            unsafe { block(level, x, k, &panel[..], n, rows, 0..cols.len(), shifted) };
        });
    });
    out
}

/// One block, on the calling thread.
#[allow(clippy::too_many_arguments)]
unsafe fn block<W: Weight>(
    level: Level,
    x: &[f32],
    k: usize,
    w: &[W],
    n: usize,
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
    out: OutPtr,
) {
    match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 => avx512::block(x, k, w, n, rows, cols, out.0),
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 => avx2::block(x, k, w, n, rows, cols, out.0),
        Level::Scalar => {
            for i in rows {
                for j in cols.clone() {
                    *out.0.add(i * n + j) =
                        dot_scalar(&x[i * k..(i + 1) * k], &w[j * k..(j + 1) * k]);
                }
            }
        }
    }
}

fn dot_scalar<W: Weight>(a: &[f32], b: &[W]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for ((lane, xv), yv) in lanes.iter_mut().zip(x).zip(y) {
            *lane += xv * yv.to_f32();
        }
    }
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y.to_f32()).sum();
    lanes.iter().sum::<f32>() + tail
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::Weight;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f,avx2,fma,f16c")]
    pub unsafe fn block<W: Weight>(
        x: &[f32],
        k: usize,
        w: &[W],
        n: usize,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
        out: *mut f32,
    ) {
        let mut i = rows.start;
        while i < rows.end {
            let r = (rows.end - i).min(4);
            let mut j = cols.start;
            while j < cols.end {
                let c = (cols.end - j).min(4);
                let xp = x.as_ptr().add(i * k);
                let wp = w.as_ptr().add(j * k);
                let op = out.add(i * n + j);
                match (r, c) {
                    (4, 4) => tile::<4, 4, W>(xp, wp, k, op, n),
                    (4, 3) => tile::<4, 3, W>(xp, wp, k, op, n),
                    (4, 2) => tile::<4, 2, W>(xp, wp, k, op, n),
                    (4, 1) => tile::<4, 1, W>(xp, wp, k, op, n),
                    (3, 4) => tile::<3, 4, W>(xp, wp, k, op, n),
                    (3, 3) => tile::<3, 3, W>(xp, wp, k, op, n),
                    (3, 2) => tile::<3, 2, W>(xp, wp, k, op, n),
                    (3, 1) => tile::<3, 1, W>(xp, wp, k, op, n),
                    (2, 4) => tile::<2, 4, W>(xp, wp, k, op, n),
                    (2, 3) => tile::<2, 3, W>(xp, wp, k, op, n),
                    (2, 2) => tile::<2, 2, W>(xp, wp, k, op, n),
                    (2, 1) => tile::<2, 1, W>(xp, wp, k, op, n),
                    (1, 4) => tile::<1, 4, W>(xp, wp, k, op, n),
                    (1, 3) => tile::<1, 3, W>(xp, wp, k, op, n),
                    (1, 2) => tile::<1, 2, W>(xp, wp, k, op, n),
                    _ => tile::<1, 1, W>(xp, wp, k, op, n),
                }
                j += c;
            }
            i += r;
        }
    }

    /// R x C dot products of length k; the k tail uses masked loads.
    #[target_feature(enable = "avx512f,avx2,fma,f16c")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile<const R: usize, const C: usize, W: Weight>(
        x: *const f32,
        w: *const W,
        k: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut acc = [[_mm512_setzero_ps(); C]; R];
        let mut kk = 0;
        while kk + 16 <= k {
            let mut wv = [_mm512_setzero_ps(); C];
            for (c, v) in wv.iter_mut().enumerate() {
                *v = W::load16(w.add(c * k + kk));
            }
            for (r, row) in acc.iter_mut().enumerate() {
                let xv = _mm512_loadu_ps(x.add(r * k + kk));
                for (c, a) in row.iter_mut().enumerate() {
                    *a = _mm512_fmadd_ps(xv, wv[c], *a);
                }
            }
            kk += 16;
        }
        if kk < k {
            let mask: __mmask16 = (1u32 << (k - kk)) as u16 - 1;
            let mut wv = [_mm512_setzero_ps(); C];
            for (c, v) in wv.iter_mut().enumerate() {
                // The tail is short and rare (k is a multiple of 16 in
                // every model), so widen it value by value.
                let mut tail = [0.0f32; 16];
                for (t, value) in tail.iter_mut().enumerate().take(k - kk) {
                    *value = (*w.add(c * k + kk + t)).to_f32();
                }
                *v = _mm512_loadu_ps(tail.as_ptr());
            }
            for (r, row) in acc.iter_mut().enumerate() {
                let xv = _mm512_maskz_loadu_ps(mask, x.add(r * k + kk));
                for (c, a) in row.iter_mut().enumerate() {
                    *a = _mm512_fmadd_ps(xv, wv[c], *a);
                }
            }
        }
        for (r, row) in acc.iter().enumerate() {
            for (c, a) in row.iter().enumerate() {
                *out.add(r * n + c) = _mm512_reduce_add_ps(*a);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::Weight;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn block<W: Weight>(
        x: &[f32],
        k: usize,
        w: &[W],
        n: usize,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
        out: *mut f32,
    ) {
        let mut i = rows.start;
        while i < rows.end {
            let r = (rows.end - i).min(3);
            let mut j = cols.start;
            while j < cols.end {
                let c = (cols.end - j).min(4);
                let xp = x.as_ptr().add(i * k);
                let wp = w.as_ptr().add(j * k);
                let op = out.add(i * n + j);
                match (r, c) {
                    (3, 4) => tile::<3, 4, W>(xp, wp, k, op, n),
                    (3, 3) => tile::<3, 3, W>(xp, wp, k, op, n),
                    (3, 2) => tile::<3, 2, W>(xp, wp, k, op, n),
                    (3, 1) => tile::<3, 1, W>(xp, wp, k, op, n),
                    (2, 4) => tile::<2, 4, W>(xp, wp, k, op, n),
                    (2, 3) => tile::<2, 3, W>(xp, wp, k, op, n),
                    (2, 2) => tile::<2, 2, W>(xp, wp, k, op, n),
                    (2, 1) => tile::<2, 1, W>(xp, wp, k, op, n),
                    (1, 4) => tile::<1, 4, W>(xp, wp, k, op, n),
                    (1, 3) => tile::<1, 3, W>(xp, wp, k, op, n),
                    (1, 2) => tile::<1, 2, W>(xp, wp, k, op, n),
                    _ => tile::<1, 1, W>(xp, wp, k, op, n),
                }
                j += c;
            }
            i += r;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let pair = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let quad = _mm_add_ps(pair, _mm_movehl_ps(pair, pair));
        _mm_cvtss_f32(_mm_add_ss(quad, _mm_shuffle_ps(quad, quad, 1)))
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile<const R: usize, const C: usize, W: Weight>(
        x: *const f32,
        w: *const W,
        k: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut acc = [[_mm256_setzero_ps(); C]; R];
        let mut kk = 0;
        while kk + 8 <= k {
            let mut wv = [_mm256_setzero_ps(); C];
            for (c, v) in wv.iter_mut().enumerate() {
                *v = W::load8(w.add(c * k + kk));
            }
            for (r, row) in acc.iter_mut().enumerate() {
                let xv = _mm256_loadu_ps(x.add(r * k + kk));
                for (c, a) in row.iter_mut().enumerate() {
                    *a = _mm256_fmadd_ps(xv, wv[c], *a);
                }
            }
            kk += 8;
        }
        for r in 0..R {
            for c in 0..C {
                let mut total = hsum(acc[r][c]);
                for t in kk..k {
                    total += *x.add(r * k + t) * (*w.add(c * k + t)).to_f32();
                }
                *out.add(r * n + c) = total;
            }
        }
    }
}

/// `out = x * w` (m x n) for row-major `x` (m x k) and `w` (k x n), on the
/// calling thread, for a task that is itself one of many parallel tasks:
/// every output row is a sum of `w` rows scaled by `x` values. The tiles
/// broadcast one `x` value and add it times a strip of a `w` row to 6 x 32
/// (AVX-512) or 6 x 16 (AVX2) accumulators, so a short `k` (an attention
/// head) costs no horizontal sums.
pub fn matmul_nn_into(x: &[f32], m: usize, k: usize, w: &[f32], n: usize, out: &mut [f32]) {
    assert!(x.len() >= m * k && w.len() >= k * n && out.len() >= m * n);
    match level() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the block covers `out` exactly; the level was detected from the CPU.
        Level::Avx512 => unsafe { nn_avx512::block(x, k, w, n, 0..m, 0..n, out.as_mut_ptr()) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above.
        Level::Avx2 => unsafe { nn_avx2::block(x, k, w, n, 0..m, 0..n, out.as_mut_ptr()) },
        Level::Scalar => {
            for (i, row) in out.chunks_exact_mut(n).take(m).enumerate() {
                row.fill(0.0);
                for e in 0..k {
                    let a = x[i * k + e];
                    for (o, w) in row.iter_mut().zip(&w[e * n..(e + 1) * n]) {
                        *o += a * w;
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod nn_avx512 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f")]
    pub unsafe fn block(
        x: &[f32],
        k: usize,
        w: &[f32],
        n: usize,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
        out: *mut f32,
    ) {
        let mut i = rows.start;
        while i < rows.end {
            let r = (rows.end - i).min(6);
            let mut j = cols.start;
            while j < cols.end {
                let c = (cols.end - j).min(32);
                match r {
                    6 => tile::<6>(x, k, w, n, i, j, c, out),
                    5 => tile::<5>(x, k, w, n, i, j, c, out),
                    4 => tile::<4>(x, k, w, n, i, j, c, out),
                    3 => tile::<3>(x, k, w, n, i, j, c, out),
                    2 => tile::<2>(x, k, w, n, i, j, c, out),
                    _ => tile::<1>(x, k, w, n, i, j, c, out),
                }
                j += c;
            }
            i += r;
        }
    }

    /// R rows by up to 32 columns (`c`), masked at the right edge.
    #[target_feature(enable = "avx512f")]
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    unsafe fn tile<const R: usize>(
        x: &[f32],
        k: usize,
        w: &[f32],
        n: usize,
        i: usize,
        j: usize,
        c: usize,
        out: *mut f32,
    ) {
        let mask_lo: __mmask16 = if c >= 16 {
            0xFFFF
        } else {
            ((1u32 << c) - 1) as u16
        };
        let mask_hi: __mmask16 = if c >= 32 {
            0xFFFF
        } else if c > 16 {
            ((1u32 << (c - 16)) - 1) as u16
        } else {
            0
        };
        let mut lo = [_mm512_setzero_ps(); R];
        let mut hi = [_mm512_setzero_ps(); R];
        let wp = w.as_ptr().add(j);
        let xp = x.as_ptr().add(i * k);
        for e in 0..k {
            let row = wp.add(e * n);
            let b0 = _mm512_maskz_loadu_ps(mask_lo, row);
            let b1 = _mm512_maskz_loadu_ps(mask_hi, row.add(16));
            for r in 0..R {
                let a = _mm512_set1_ps(*xp.add(r * k + e));
                lo[r] = _mm512_fmadd_ps(a, b0, lo[r]);
                hi[r] = _mm512_fmadd_ps(a, b1, hi[r]);
            }
        }
        for r in 0..R {
            let o = out.add((i + r) * n + j);
            _mm512_mask_storeu_ps(o, mask_lo, lo[r]);
            _mm512_mask_storeu_ps(o.add(16), mask_hi, hi[r]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod nn_avx2 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn block(
        x: &[f32],
        k: usize,
        w: &[f32],
        n: usize,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
        out: *mut f32,
    ) {
        let mut i = rows.start;
        while i < rows.end {
            let r = (rows.end - i).min(6);
            let mut j = cols.start;
            while j < cols.end {
                let c = (cols.end - j).min(16);
                if c == 16 {
                    match r {
                        6 => tile::<6>(x, k, w, n, i, j, out),
                        5 => tile::<5>(x, k, w, n, i, j, out),
                        4 => tile::<4>(x, k, w, n, i, j, out),
                        3 => tile::<3>(x, k, w, n, i, j, out),
                        2 => tile::<2>(x, k, w, n, i, j, out),
                        _ => tile::<1>(x, k, w, n, i, j, out),
                    }
                } else {
                    // The right edge, under 16 columns: plain loops.
                    for row in i..i + r {
                        for col in j..j + c {
                            let mut total = 0.0f32;
                            for e in 0..k {
                                total += x[row * k + e] * w[e * n + col];
                            }
                            *out.add(row * n + col) = total;
                        }
                    }
                }
                j += c;
            }
            i += r;
        }
    }

    /// R rows by 16 columns.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile<const R: usize>(
        x: &[f32],
        k: usize,
        w: &[f32],
        n: usize,
        i: usize,
        j: usize,
        out: *mut f32,
    ) {
        let mut lo = [_mm256_setzero_ps(); R];
        let mut hi = [_mm256_setzero_ps(); R];
        let wp = w.as_ptr().add(j);
        let xp = x.as_ptr().add(i * k);
        for e in 0..k {
            let row = wp.add(e * n);
            let b0 = _mm256_loadu_ps(row);
            let b1 = _mm256_loadu_ps(row.add(8));
            for r in 0..R {
                let a = _mm256_broadcast_ss(&*xp.add(r * k + e));
                lo[r] = _mm256_fmadd_ps(a, b0, lo[r]);
                hi[r] = _mm256_fmadd_ps(a, b1, hi[r]);
            }
        }
        for r in 0..R {
            let o = out.add((i + r) * n + j);
            _mm256_storeu_ps(o, lo[r]);
            _mm256_storeu_ps(o.add(8), hi[r]);
        }
    }
}
