//! f32 matrix product `out = x * w^T` for row-major `x` (m x k) and `w`
//! (n x k). Both operands are read along k, so every output is a dot
//! product. Register tiles keep 4 x 3 dot products in flight (AVX-512 and
//! AVX2) and reduce each one once at the end. Two limits pick the shape:
//! a tile reads R + C rows at once, and rows of a power-of-two length
//! (1024, 4096) fall into one set of the 8-way L1 cache, so 8 rows or more
//! evict each other (4 x 4 and 6 x 4 were slower); and AVX2 has 16
//! registers, which 4 x 3 fills exactly (12 sums, 3 weight rows, 1 input
//! row) where 3 x 4 needed 17. A value's sum does not depend on the tile. `w` holds f32
//! or f16 values (`Weight`); f16 rows widen to f32 as they load, so a model
//! stored in f16 keeps half the memory and reads half the bytes.

use super::buf::Buf;
use super::team;

const ROW_BLOCK: usize = 64;
/// Least multiply-adds per column task when splitting a single row block
/// across worker threads.
const MIN_COL_WORK: usize = 32_768;

/// Column register-tile width and column panel size for `(level, k)`.
/// AVX-512 widens from 3 to 4 columns whenever `k` is not a multiple of 512
/// (no 4 KB L1d 8-way set conflict), cutting input loads by 25 % and filling
/// all four lanes of `hsum4` with a single 16-byte vector store.
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
fn tile_layout(level: Level, k: usize) -> (usize, usize) {
    match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 if !k.is_multiple_of(512) => (4, 64),
        #[cfg(target_arch = "aarch64")]
        Level::Neon => (4, 64),
        _ => (3, 66),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Level {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    /// Every arm64 CPU.
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Level {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Level::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Level::Avx2 => "avx2",
            #[cfg(target_arch = "x86_64")]
            Level::Avx512 => "avx512",
            #[cfg(target_arch = "aarch64")]
            Level::Neon => "neon",
        }
    }

    /// Whether the tiles of this level widen f16 weights as they load them,
    /// for a block of any height. The other levels widen a 64-column panel
    /// to f32 once for a block of more than 4 rows.
    fn widens_in_tile(self) -> bool {
        #[cfg(target_arch = "x86_64")]
        return self == Level::Avx512;
        #[cfg(target_arch = "aarch64")]
        return self == Level::Neon;
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        false
    }
}

/// The widest float path this CPU runs. `TRIT_ISA=scalar|avx2` forces a slower one;
/// on arm64, `TRIT_ISA=scalar` forces the scalar path.
pub(crate) fn level() -> Level {
    super::isa::isa_choice().float_level
}

/// A weight element: f32, or f16 as its bits. The loads are `inline(always)`
/// so they compile with the target features of the kernel that calls them.
pub trait Weight: Copy + Send + Sync + 'static + super::v2::float::Element {
    /// True for f16: a block of many rows widens its weight panel to f32
    /// once, instead of once per register tile.
    const WIDEN: bool;
    /// The name of this weight type in the profile's table of products.
    const PRODUCT: crate::prof::Product;
    fn to_f32(self) -> f32;
    /// `dst = src` as f32; the caller sizes `dst` like `src`.
    fn widen(src: &[Self], dst: &mut [f32]);
    #[cfg(target_arch = "x86_64")]
    unsafe fn load16(p: *const Self) -> std::arch::x86_64::__m512;
    #[cfg(target_arch = "x86_64")]
    unsafe fn load8(p: *const Self) -> std::arch::x86_64::__m256;
    #[cfg(target_arch = "aarch64")]
    unsafe fn load4(p: *const Self) -> std::arch::aarch64::float32x4_t;
}

impl Weight for f32 {
    const WIDEN: bool = false;
    const PRODUCT: crate::prof::Product = crate::prof::Product::F32;
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
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn load4(p: *const Self) -> std::arch::aarch64::float32x4_t {
        std::arch::aarch64::vld1q_f32(p)
    }
}

/// f16 bits. The CPUs of the AVX2 and AVX-512 paths have F16C (see `level`).
impl Weight for u16 {
    const WIDEN: bool = true;
    const PRODUCT: crate::prof::Product = crate::prof::Product::F16;
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
        #[cfg(target_arch = "aarch64")]
        if level() == Level::Neon {
            let (chunks, tail) = src.as_chunks::<8>();
            for (i, chunk) in chunks.iter().enumerate() {
                // SAFETY: `dst` is sized like `src`, so 8 values fit at `i * 8`.
                unsafe {
                    let (lo, hi) = f16x8(chunk.as_ptr());
                    let p = dst.as_mut_ptr().add(i * 8);
                    std::arch::aarch64::vst1q_f32(p, lo);
                    std::arch::aarch64::vst1q_f32(p.add(4), hi);
                }
            }
            let done = chunks.len() * 8;
            for (d, s) in dst[done..].iter_mut().zip(tail) {
                *d = s.to_f32();
            }
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
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn load4(p: *const Self) -> std::arch::aarch64::float32x4_t {
        f16x4(p)
    }
}

/// 4 f16 values from `p` as f32, with `fcvtl` (every arm64 CPU has it).
/// Inline assembly: Rust's f16 vector types are not stable yet.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn f16x4(p: *const u16) -> std::arch::aarch64::float32x4_t {
    let half = std::arch::aarch64::vld1_u16(p);
    let wide: std::arch::aarch64::float32x4_t;
    std::arch::asm!(
        "fcvtl {wide:v}.4s, {half:v}.4h",
        wide = lateout(vreg) wide,
        half = in(vreg) half,
        options(pure, nomem, nostack, preserves_flags),
    );
    wide
}

/// 8 f16 values from `p` as two f32x4 vectors with one 128-bit load and
/// `fcvtl` + `fcvtl2` (every arm64 CPU has both).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn f16x8(
    p: *const u16,
) -> (
    std::arch::aarch64::float32x4_t,
    std::arch::aarch64::float32x4_t,
) {
    let half = std::arch::aarch64::vld1q_u16(p);
    let lo: std::arch::aarch64::float32x4_t;
    let hi: std::arch::aarch64::float32x4_t;
    std::arch::asm!(
        "fcvtl {lo:v}.4s, {half:v}.4h",
        "fcvtl2 {hi:v}.4s, {half:v}.8h",
        // `fcvtl2` still needs `half` after `fcvtl` writes `lo`.
        lo = out(vreg) lo,
        hi = lateout(vreg) hi,
        half = in(vreg) half,
        options(pure, nomem, nostack, preserves_flags),
    );
    (lo, hi)
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

/// `out[i][j] = sum_k x[i][k] * w[j][k]`, parallel over tile-aligned blocks.
pub fn matmul<W: Weight>(x: &[f32], m: usize, k: usize, w: &[W], n: usize) -> Buf {
    matmul_then(x, m, k, w, n, None, false)
}

/// `matmul` with an optional per-column `bias` and ReLU folded into each tile
/// store, so output rows are written once.
pub fn matmul_then<W: Weight>(
    x: &[f32],
    m: usize,
    k: usize,
    w: &[W],
    n: usize,
    bias: Option<&[f32]>,
    relu: bool,
) -> Buf {
    let _span = crate::prof::span(crate::prof::Stage::Float);
    let _product = crate::prof::product(W::PRODUCT, n, k, m);
    if super::v2::float_enabled() {
        let mut out = super::v2::float::matmul(x, m, k, w, n);
        apply_epilogue(&mut out, n, bias, relu);
        return out;
    }
    assert!(x.len() >= m * k && w.len() >= n * k && bias.is_none_or(|b| b.len() >= n));
    if m == 0 || n == 0 {
        return Buf::scratch(0);
    }
    let mut out = Buf::scratch(m * n);
    let level = level();
    let (tile_c, col_block) = tile_layout(level, k);
    let (row_blocks, col_blocks) = block_counts(m, n, k, tile_c, col_block);
    let ptr = OutPtr(out.as_mut_ptr());
    team::for_each(row_blocks * col_blocks, |task| {
        let (rows, cols) = block_span(task, row_blocks, col_blocks, m, n, tile_c);
        // On the AVX-512 level the tile widens f16 weights itself: the
        // conversion does not wait for the multiply-add (Zen 4), and the
        // f32 copy of the panel cost 35 % of a 5-row product.
        if W::WIDEN && rows.len() > 4 && !level.widens_in_tile() {
            PANEL.with(|panel| {
                let mut panel = panel.borrow_mut();
                panel.resize(cols.len() * k, 0.0);
                W::widen(&w[cols.start * k..cols.end * k], &mut panel);
                let shifted = OutPtr(ptr.0.wrapping_add(cols.start));
                let block_bias = bias.map(|b| &b[cols.start..cols.end]);
                // SAFETY: as below; the panel holds this block's columns from 0,
                // so the output pointer moves to the block's first column.
                unsafe {
                    block(
                        level,
                        x,
                        k,
                        &panel[..],
                        n,
                        rows,
                        0..cols.len(),
                        block_bias,
                        relu,
                        shifted,
                    )
                };
            });
            return;
        }
        // SAFETY: the block is inside all buffers and belongs to this task only;
        // the level was detected from the CPU.
        unsafe { block(level, x, k, w, n, rows, cols, bias, relu, ptr) };
    });
    out
}

/// `matmul` into `out` (`m x n`), on the calling thread, for a task that
/// is itself one of many parallel tasks. Each value is the one `matmul`
/// gives: a value's sum does not depend on the tile around it.
pub fn matmul_into(x: &[f32], m: usize, k: usize, w: &[f32], n: usize, out: &mut [f32]) {
    let _product = crate::prof::product(crate::prof::Product::F32Into, n, k, m);
    if super::v2::float_enabled() {
        return super::v2::float::matmul_into(x, m, k, w, n, out);
    }
    assert!(x.len() >= m * k && w.len() >= n * k && out.len() >= m * n);
    // SAFETY: the block covers `out` exactly; the level was detected from the CPU.
    unsafe {
        block(
            level(),
            x,
            k,
            w,
            n,
            0..m,
            0..n,
            None,
            false,
            OutPtr(out.as_mut_ptr()),
        )
    };
}

/// `matmul_then` into `out` (`m x n`), on the calling thread, for a task that
/// is itself one of many parallel tasks. Each value is the one `matmul_then`
/// gives: a value's sum does not depend on the tile around it.
#[allow(clippy::too_many_arguments)]
pub fn matmul_into_then<W: Weight>(
    x: &[f32],
    m: usize,
    k: usize,
    w: &[W],
    n: usize,
    bias: Option<&[f32]>,
    relu: bool,
    out: &mut [f32],
) {
    let _product = crate::prof::product(W::PRODUCT, n, k, m);
    assert!(
        x.len() >= m * k
            && w.len() >= n * k
            && out.len() >= m * n
            && bias.is_none_or(|b| b.len() >= n)
    );
    if m == 0 || n == 0 {
        return;
    }
    if super::v2::float_enabled() {
        super::v2::float::matmul_into(x, m, k, w, n, out);
        if bias.is_some() || relu {
            for row in out[..m * n].chunks_exact_mut(n) {
                if let Some(bias) = bias {
                    for (value, b) in row.iter_mut().zip(bias) {
                        *value += b;
                    }
                }
                if relu {
                    row.iter_mut().for_each(|v| *v = v.max(0.0));
                }
            }
        }
        return;
    }
    // SAFETY: the block covers `out` exactly; the level was detected from the CPU.
    unsafe {
        block(
            level(),
            x,
            k,
            w,
            n,
            0..m,
            0..n,
            bias,
            relu,
            OutPtr(out.as_mut_ptr()),
        )
    };
}

/// `out[i][j] = sum_k x[i][k] * w[j][k]` where the weights come from
/// `widen(cols, panel)`, which writes rows `cols` of `w` as f32 into a
/// `cols.len() x k` panel. Each block widens its columns once, then runs
/// the f32 tiles with optional `bias` and ReLU; for int8 storage with f32 math.
pub fn matmul_widened(
    x: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    relu: bool,
    widen: impl Fn(std::ops::Range<usize>, &mut [f32]) + Sync,
) -> Buf {
    let _product = crate::prof::product(crate::prof::Product::Q8Widened, n, k, m);
    if super::v2::float_enabled() {
        let mut out = super::v2::float::matmul_widened(x, m, k, n, widen);
        apply_epilogue(&mut out, n, bias, relu);
        return out;
    }
    assert!(x.len() >= m * k && bias.is_none_or(|b| b.len() >= n));
    if m == 0 || n == 0 {
        return Buf::scratch(0);
    }
    let mut out = Buf::scratch(m * n);
    let level = level();
    let (tile_c, col_block) = tile_layout(level, k);
    let (row_blocks, col_blocks) = block_counts(m, n, k, tile_c, col_block);
    let ptr = OutPtr(out.as_mut_ptr());
    team::for_each(row_blocks * col_blocks, |task| {
        let (rows, cols) = block_span(task, row_blocks, col_blocks, m, n, tile_c);
        PANEL.with(|panel| {
            let mut panel = panel.borrow_mut();
            panel.resize(cols.len() * k, 0.0);
            widen(cols.clone(), &mut panel);
            let shifted = OutPtr(ptr.get().wrapping_add(cols.start));
            let block_bias = bias.map(|b| &b[cols.start..cols.end]);
            // SAFETY: the panel holds this block's columns from 0, so the output
            // pointer moves to the block's first column; the block belongs to
            // this task only; the level was detected from the CPU.
            unsafe {
                block(
                    level,
                    x,
                    k,
                    &panel[..],
                    n,
                    rows,
                    0..cols.len(),
                    block_bias,
                    relu,
                    shifted,
                )
            };
        });
    });
    out
}

fn apply_epilogue(out: &mut [f32], n: usize, bias: Option<&[f32]>, relu: bool) {
    if bias.is_none() && !relu {
        return;
    }
    team::chunks_mut(out, ROW_BLOCK * n, |_, block| {
        for row in block.chunks_exact_mut(n) {
            if let Some(bias) = bias {
                for (value, b) in row.iter_mut().zip(bias) {
                    *value += b;
                }
            }
            if relu {
                row.iter_mut().for_each(|v| *v = v.max(0.0));
            }
        }
    });
}

fn block_counts(m: usize, n: usize, k: usize, tile_c: usize, col_block: usize) -> (usize, usize) {
    let row_blocks = m.div_ceil(ROW_BLOCK);
    let col_tiles = n.div_ceil(tile_c);
    let base_cols = n.div_ceil(col_block);
    let threads = team::threads().max(1);
    let hetero = team::is_heterogeneous();
    // Heterogeneous teams (e.g. Intel P+E, Apple M5 Super+Performance, ARM big.LITTLE)
    // keep >= 4 waves in the queue so fast cores dynamically steal work from slower cores.
    let min_waves = if hetero { 4 } else { 1 };
    let col_blocks = if threads == 1 {
        base_cols
    } else if row_blocks == 1 {
        let max_by_work = (m * n * k / MIN_COL_WORK).max(1);
        let limit = col_tiles.min(max_by_work);
        if hetero {
            base_cols.max(min_waves * threads).min(limit)
        } else if base_cols <= 6 * threads {
            let waves = (base_cols + threads / 2) / threads;
            (waves.max(1) * threads).min(limit)
        } else {
            base_cols.min(col_tiles)
        }
    } else if row_blocks * base_cols < min_waves * threads {
        let max_by_work = (m * n * k / MIN_COL_WORK).max(1);
        (min_waves * threads)
            .div_ceil(row_blocks)
            .min(col_tiles)
            .min(max_by_work.div_ceil(row_blocks).max(1))
    } else {
        base_cols.min(col_tiles)
    };
    (row_blocks, col_blocks.max(1))
}

fn block_span(
    task: usize,
    row_blocks: usize,
    col_blocks: usize,
    m: usize,
    n: usize,
    tile_c: usize,
) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    let row = task / col_blocks;
    let col = task % col_blocks;
    let rows = if row_blocks == 1 {
        0..m
    } else {
        let row_tiles = m.div_ceil(4);
        let start = (row * row_tiles / row_blocks) * 4;
        let end = (((row + 1) * row_tiles / row_blocks) * 4).min(m);
        start..end
    };
    let col_tiles = n.div_ceil(tile_c);
    let col_start = (col * col_tiles / col_blocks) * tile_c;
    let col_end = (((col + 1) * col_tiles / col_blocks) * tile_c).min(n);
    (rows, col_start..col_end)
}

/// On the calling thread.
#[allow(clippy::too_many_arguments)]
unsafe fn block<W: Weight>(
    level: Level,
    x: &[f32],
    k: usize,
    w: &[W],
    n: usize,
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
    bias: Option<&[f32]>,
    relu: bool,
    out: OutPtr,
) {
    match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 => avx512::block(x, k, w, n, rows, cols, bias, relu, out.0),
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 => avx2::block(x, k, w, n, rows, cols, bias, relu, out.0),
        #[cfg(target_arch = "aarch64")]
        Level::Neon => neon::block(x, k, w, n, rows, cols, bias, relu, out.0),
        Level::Scalar => {
            for i in rows {
                for j in cols.clone() {
                    let mut total = dot_scalar(&x[i * k..(i + 1) * k], &w[j * k..(j + 1) * k]);
                    if let Some(bias) = bias {
                        total += bias[j];
                    }
                    if relu {
                        total = total.max(0.0);
                    }
                    *out.0.add(i * n + j) = total;
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
pub(super) use avx2::{hsum as avx2_hsum, hsum4 as avx2_hsum4};
#[cfg(target_arch = "x86_64")]
pub(super) use avx512::hsum4 as avx512_hsum4;

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::Weight;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma,f16c")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn block<W: Weight>(
        x: &[f32],
        k: usize,
        w: &[W],
        n: usize,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
        bias: Option<&[f32]>,
        relu: bool,
        out: *mut f32,
    ) {
        let max_c = if k.is_multiple_of(512) { 3 } else { 4 };
        let mut i = rows.start;
        while i < rows.end {
            // When k is not a 4 KB L1d alias (`max_c == 4`), widen to 6 x 4
            // (24 acc + 4 w + 1 x = 29 <= 32 ZMMs), cutting weight loads and
            // f16 conversions by 33 %; step 4/5 near the tail so m = 16, 32, 64
            // never emit 1-3 row runt tiles. Keep <= 5 x 3 on 4 KB strides so
            // R + C <= 8 stays within the 8-way L1d.
            let r = if max_c == 4 {
                match rows.end - i {
                    left @ 1..=6 => left,
                    7 | 8 => 4,
                    9 => 5,
                    _ => 6,
                }
            } else {
                match rows.end - i {
                    left @ (1..=3 | 5) => left,
                    _ => 4,
                }
            };
            let mut j = cols.start;
            while j < cols.end {
                let c = (cols.end - j).min(max_c);
                let xp = x.as_ptr().add(i * k);
                let wp = w.as_ptr().add(j * k);
                let bp = bias.map(|b| b.as_ptr().add(j));
                let op = out.add(i * n + j);
                match (r, c) {
                    (6, 4) => tile::<6, 4, W>(xp, wp, k, op, n, bp, relu),
                    (6, 3) => tile::<6, 3, W>(xp, wp, k, op, n, bp, relu),
                    (6, 2) => tile::<6, 2, W>(xp, wp, k, op, n, bp, relu),
                    (6, 1) => tile::<6, 1, W>(xp, wp, k, op, n, bp, relu),
                    (5, 4) => tile::<5, 4, W>(xp, wp, k, op, n, bp, relu),
                    (5, 3) => tile::<5, 3, W>(xp, wp, k, op, n, bp, relu),
                    (5, 2) => tile::<5, 2, W>(xp, wp, k, op, n, bp, relu),
                    (5, 1) => tile::<5, 1, W>(xp, wp, k, op, n, bp, relu),
                    (4, 4) => tile::<4, 4, W>(xp, wp, k, op, n, bp, relu),
                    (4, 3) => tile::<4, 3, W>(xp, wp, k, op, n, bp, relu),
                    (4, 2) => tile::<4, 2, W>(xp, wp, k, op, n, bp, relu),
                    (4, 1) => tile::<4, 1, W>(xp, wp, k, op, n, bp, relu),
                    (3, 4) => tile::<3, 4, W>(xp, wp, k, op, n, bp, relu),
                    (3, 3) => tile::<3, 3, W>(xp, wp, k, op, n, bp, relu),
                    (3, 2) => tile::<3, 2, W>(xp, wp, k, op, n, bp, relu),
                    (3, 1) => tile::<3, 1, W>(xp, wp, k, op, n, bp, relu),
                    (2, 4) => tile::<2, 4, W>(xp, wp, k, op, n, bp, relu),
                    (2, 3) => tile::<2, 3, W>(xp, wp, k, op, n, bp, relu),
                    (2, 2) => tile::<2, 2, W>(xp, wp, k, op, n, bp, relu),
                    (2, 1) => tile::<2, 1, W>(xp, wp, k, op, n, bp, relu),
                    (1, 4) => tile::<1, 4, W>(xp, wp, k, op, n, bp, relu),
                    (1, 3) => tile::<1, 3, W>(xp, wp, k, op, n, bp, relu),
                    (1, 2) => tile::<1, 2, W>(xp, wp, k, op, n, bp, relu),
                    _ => tile::<1, 1, W>(xp, wp, k, op, n, bp, relu),
                }
                j += c;
            }
            i += r;
        }
    }

    /// Reduces four 512-bit accumulators to one 128-bit vector `[sum(a0), sum(a1), sum(a2), sum(a3)]`,
    /// preserving the exact binary reduction tree `((s0 + s2) + (s1 + s3))` of `_mm512_reduce_add_ps`.
    /// Folds the four ZMMs in 512-bit with base `avx512f` `vshuff32x4` and `_mm512_extractf64x4_pd`
    /// so it always inlines into `avx512f`-only callers with zero out-of-line calls or `vzeroupper`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) unsafe fn hsum4(a0: __m512, a1: __m512, a2: __m512, a3: __m512) -> __m128 {
        let sum01 = _mm512_add_ps(
            _mm512_shuffle_f32x4::<0x44>(a0, a1),
            _mm512_shuffle_f32x4::<0xEE>(a0, a1),
        );
        let sum23 = _mm512_add_ps(
            _mm512_shuffle_f32x4::<0x44>(a2, a3),
            _mm512_shuffle_f32x4::<0xEE>(a2, a3),
        );
        let q = _mm512_add_ps(
            _mm512_shuffle_f32x4::<0x88>(sum01, sum23),
            _mm512_shuffle_f32x4::<0xDD>(sum01, sum23),
        );
        let q_lo = _mm512_castps512_ps256(q);
        let q_hi = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(q)));
        let s = _mm256_add_ps(
            _mm256_unpacklo_ps(q_lo, q_hi),
            _mm256_unpackhi_ps(q_lo, q_hi),
        );
        let s_lo = _mm256_castps256_ps128(s);
        let s_hi = _mm256_extractf128_ps::<1>(s);
        _mm_add_ps(_mm_unpacklo_ps(s_lo, s_hi), _mm_unpackhi_ps(s_lo, s_hi))
    }

    /// R x C dot products of length k; the k tail uses masked loads.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma,f16c")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile<const R: usize, const C: usize, W: Weight>(
        x: *const f32,
        w: *const W,
        k: usize,
        out: *mut f32,
        n: usize,
        bias: Option<*const f32>,
        relu: bool,
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
                // One masked load: Moonshine's first layer has k = 127 and
                // Whisper's mel filters k = 201, so the tail is not rare.
                *v = W::wide16_first(w.add(c * k + kk), k - kk);
            }
            for (r, row) in acc.iter_mut().enumerate() {
                let xv = _mm512_maskz_loadu_ps(mask, x.add(r * k + kk));
                for (c, a) in row.iter_mut().enumerate() {
                    *a = _mm512_fmadd_ps(xv, wv[c], *a);
                }
            }
        }
        if C == 4 {
            let bv = bias.map(|b| _mm_loadu_ps(b));
            let zero = _mm_setzero_ps();
            for (r, row) in acc.iter().enumerate() {
                let mut sum = hsum4(row[0], row[1], row[2], row[3]);
                if let Some(bv) = bv {
                    sum = _mm_add_ps(sum, bv);
                }
                if relu {
                    sum = _mm_max_ps(sum, zero);
                }
                _mm_storeu_ps(out.add(r * n), sum);
            }
            return;
        }
        if C == 3 {
            let bv = bias.map(|b| _mm_setr_ps(*b, *b.add(1), *b.add(2), 0.0));
            let zero = _mm_setzero_ps();
            let z512 = _mm512_setzero_ps();
            for (r, row) in acc.iter().enumerate() {
                let mut sum = hsum4(row[0], row[1], row[2], z512);
                if let Some(bv) = bv {
                    sum = _mm_add_ps(sum, bv);
                }
                if relu {
                    sum = _mm_max_ps(sum, zero);
                }
                let dst = out.add(r * n);
                std::ptr::write_unaligned(
                    dst.cast::<i64>(),
                    _mm_cvtsi128_si64(_mm_castps_si128(sum)),
                );
                *dst.add(2) = _mm_cvtss_f32(_mm_movehl_ps(sum, sum));
            }
            return;
        }
        match (bias, relu) {
            (None, false) => {
                for (r, row) in acc.iter().enumerate() {
                    for (c, a) in row.iter().enumerate() {
                        *out.add(r * n + c) = _mm512_reduce_add_ps(*a);
                    }
                }
            }
            (Some(bias), false) => {
                let mut b = [0.0f32; C];
                for (c, slot) in b.iter_mut().enumerate() {
                    *slot = *bias.add(c);
                }
                for (r, row) in acc.iter().enumerate() {
                    for (c, a) in row.iter().enumerate() {
                        *out.add(r * n + c) = _mm512_reduce_add_ps(*a) + b[c];
                    }
                }
            }
            (None, true) => {
                for (r, row) in acc.iter().enumerate() {
                    for (c, a) in row.iter().enumerate() {
                        *out.add(r * n + c) = _mm512_reduce_add_ps(*a).max(0.0);
                    }
                }
            }
            (Some(bias), true) => {
                let mut b = [0.0f32; C];
                for (c, slot) in b.iter_mut().enumerate() {
                    *slot = *bias.add(c);
                }
                for (r, row) in acc.iter().enumerate() {
                    for (c, a) in row.iter().enumerate() {
                        *out.add(r * n + c) = (_mm512_reduce_add_ps(*a) + b[c]).max(0.0);
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::Weight;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma,f16c")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn block<W: Weight>(
        x: &[f32],
        k: usize,
        w: &[W],
        n: usize,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
        bias: Option<&[f32]>,
        relu: bool,
        out: *mut f32,
    ) {
        let mut i = rows.start;
        while i < rows.end {
            let r = (rows.end - i).min(4);
            let mut j = cols.start;
            while j < cols.end {
                let c = (cols.end - j).min(3);
                let xp = x.as_ptr().add(i * k);
                let wp = w.as_ptr().add(j * k);
                let bp = bias.map(|b| b.as_ptr().add(j));
                let op = out.add(i * n + j);
                match (r, c) {
                    (4, 3) => tile::<4, 3, W>(xp, wp, k, op, n, bp, relu),
                    (4, 2) => tile::<4, 2, W>(xp, wp, k, op, n, bp, relu),
                    (4, 1) => tile::<4, 1, W>(xp, wp, k, op, n, bp, relu),
                    (3, 3) => tile::<3, 3, W>(xp, wp, k, op, n, bp, relu),
                    (3, 2) => tile::<3, 2, W>(xp, wp, k, op, n, bp, relu),
                    (3, 1) => tile::<3, 1, W>(xp, wp, k, op, n, bp, relu),
                    (2, 3) => tile::<2, 3, W>(xp, wp, k, op, n, bp, relu),
                    (2, 2) => tile::<2, 2, W>(xp, wp, k, op, n, bp, relu),
                    (2, 1) => tile::<2, 1, W>(xp, wp, k, op, n, bp, relu),
                    (1, 3) => tile::<1, 3, W>(xp, wp, k, op, n, bp, relu),
                    (1, 2) => tile::<1, 2, W>(xp, wp, k, op, n, bp, relu),
                    _ => tile::<1, 1, W>(xp, wp, k, op, n, bp, relu),
                }
                j += c;
            }
            i += r;
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) unsafe fn hsum(v: __m256) -> f32 {
        let pair = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let quad = _mm_add_ps(pair, _mm_movehl_ps(pair, pair));
        _mm_cvtss_f32(_mm_add_ss(quad, _mm_shuffle_ps(quad, quad, 1)))
    }

    /// Reduces four 256-bit accumulators to `[hsum(a0), hsum(a1), hsum(a2), hsum(a3)]`,
    /// preserving the exact binary reduction tree of `hsum`.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) unsafe fn hsum4(a0: __m256, a1: __m256, a2: __m256, a3: __m256) -> __m128 {
        macro_rules! fold {
            ($v:expr) => {
                _mm_add_ps(_mm256_castps256_ps128($v), _mm256_extractf128_ps::<1>($v))
            };
        }
        let (v0, v1, v2, v3) = (fold!(a0), fold!(a1), fold!(a2), fold!(a3));
        let s01 = _mm_add_ps(_mm_unpacklo_ps(v0, v1), _mm_unpackhi_ps(v0, v1));
        let s23 = _mm_add_ps(_mm_unpacklo_ps(v2, v3), _mm_unpackhi_ps(v2, v3));
        _mm_add_ps(_mm_movelh_ps(s01, s23), _mm_movehl_ps(s23, s01))
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile<const R: usize, const C: usize, W: Weight>(
        x: *const f32,
        w: *const W,
        k: usize,
        out: *mut f32,
        n: usize,
        bias: Option<*const f32>,
        relu: bool,
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
        if kk == k && C == 3 {
            let bv = bias.map(|b| _mm_setr_ps(*b, *b.add(1), *b.add(2), 0.0));
            let zero = _mm_setzero_ps();
            let z256 = _mm256_setzero_ps();
            for r in 0..R {
                let mut sum = hsum4(acc[r][0], acc[r][1], acc[r][2], z256);
                if let Some(bv) = bv {
                    sum = _mm_add_ps(sum, bv);
                }
                if relu {
                    sum = _mm_max_ps(sum, zero);
                }
                let dst = out.add(r * n);
                std::ptr::write_unaligned(
                    dst.cast::<i64>(),
                    _mm_cvtsi128_si64(_mm_castps_si128(sum)),
                );
                *dst.add(2) = _mm_cvtss_f32(_mm_movehl_ps(sum, sum));
            }
            return;
        }
        for r in 0..R {
            for c in 0..C {
                let mut total = hsum(acc[r][c]);
                for t in kk..k {
                    total += *x.add(r * k + t) * (*w.add(c * k + t)).to_f32();
                }
                if let Some(bias) = bias {
                    total += *bias.add(c);
                }
                if relu {
                    total = total.max(0.0);
                }
                *out.add(r * n + c) = total;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::Weight;
    use std::arch::aarch64::*;

    /// 6 x 4 tiles (24 accumulators, 4 weight vectors, 1 input = 29 of 32 NEON
    /// registers, cutting weight loads and f16 widens by 33 %) plus 5 x 4 for
    /// 5-beam decoder steps.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn block<W: Weight>(
        x: &[f32],
        k: usize,
        w: &[W],
        n: usize,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
        bias: Option<&[f32]>,
        relu: bool,
        out: *mut f32,
    ) {
        let mut i = rows.start;
        while i < rows.end {
            let r = match rows.end - i {
                left @ 1..=6 => left,
                7 | 8 => 4,
                9 => 5,
                _ => 6,
            };
            let mut j = cols.start;
            while j < cols.end {
                let c = (cols.end - j).min(4);
                let xp = x.as_ptr().add(i * k);
                let wp = w.as_ptr().add(j * k);
                let bp = bias.map(|b| b.as_ptr().add(j));
                let op = out.add(i * n + j);
                match (r, c) {
                    (6, 4) => tile::<6, 4, W>(xp, wp, k, op, n, bp, relu),
                    (6, 3) => tile::<6, 3, W>(xp, wp, k, op, n, bp, relu),
                    (6, 2) => tile::<6, 2, W>(xp, wp, k, op, n, bp, relu),
                    (6, 1) => tile::<6, 1, W>(xp, wp, k, op, n, bp, relu),
                    (5, 4) => tile::<5, 4, W>(xp, wp, k, op, n, bp, relu),
                    (5, 3) => tile::<5, 3, W>(xp, wp, k, op, n, bp, relu),
                    (5, 2) => tile::<5, 2, W>(xp, wp, k, op, n, bp, relu),
                    (5, 1) => tile::<5, 1, W>(xp, wp, k, op, n, bp, relu),
                    (4, 4) => tile::<4, 4, W>(xp, wp, k, op, n, bp, relu),
                    (4, 3) => tile::<4, 3, W>(xp, wp, k, op, n, bp, relu),
                    (4, 2) => tile::<4, 2, W>(xp, wp, k, op, n, bp, relu),
                    (4, 1) => tile::<4, 1, W>(xp, wp, k, op, n, bp, relu),
                    (3, 4) => tile::<3, 4, W>(xp, wp, k, op, n, bp, relu),
                    (3, 3) => tile::<3, 3, W>(xp, wp, k, op, n, bp, relu),
                    (3, 2) => tile::<3, 2, W>(xp, wp, k, op, n, bp, relu),
                    (3, 1) => tile::<3, 1, W>(xp, wp, k, op, n, bp, relu),
                    (2, 4) => tile::<2, 4, W>(xp, wp, k, op, n, bp, relu),
                    (2, 3) => tile::<2, 3, W>(xp, wp, k, op, n, bp, relu),
                    (2, 2) => tile::<2, 2, W>(xp, wp, k, op, n, bp, relu),
                    (2, 1) => tile::<2, 1, W>(xp, wp, k, op, n, bp, relu),
                    (1, 4) => tile::<1, 4, W>(xp, wp, k, op, n, bp, relu),
                    (1, 3) => tile::<1, 3, W>(xp, wp, k, op, n, bp, relu),
                    (1, 2) => tile::<1, 2, W>(xp, wp, k, op, n, bp, relu),
                    _ => tile::<1, 1, W>(xp, wp, k, op, n, bp, relu),
                }
                j += c;
            }
            i += r;
        }
    }

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile<const R: usize, const C: usize, W: Weight>(
        x: *const f32,
        w: *const W,
        k: usize,
        out: *mut f32,
        n: usize,
        bias: Option<*const f32>,
        relu: bool,
    ) {
        let mut acc = [[vdupq_n_f32(0.0); C]; R];
        let mut kk = 0;
        while kk + 4 <= k {
            let mut wv = [vdupq_n_f32(0.0); C];
            for (c, v) in wv.iter_mut().enumerate() {
                *v = W::load4(w.add(c * k + kk));
            }
            for (r, row) in acc.iter_mut().enumerate() {
                let xv = vld1q_f32(x.add(r * k + kk));
                for (c, a) in row.iter_mut().enumerate() {
                    *a = vfmaq_f32(*a, xv, wv[c]);
                }
            }
            kk += 4;
        }
        if kk == k && C == 4 {
            let bv = bias.map(|b| vld1q_f32(b));
            let zero = vdupq_n_f32(0.0);
            for r in 0..R {
                let p01 = vpaddq_f32(acc[r][0], acc[r][1]);
                let p23 = vpaddq_f32(acc[r][2], acc[r][3]);
                let mut sum = vpaddq_f32(p01, p23);
                if let Some(bv) = bv {
                    sum = vaddq_f32(sum, bv);
                }
                if relu {
                    sum = vmaxnmq_f32(sum, zero);
                }
                vst1q_f32(out.add(r * n), sum);
            }
            return;
        }
        for r in 0..R {
            for c in 0..C {
                let mut total = vaddvq_f32(acc[r][c]);
                for t in kk..k {
                    total += *x.add(r * k + t) * (*w.add(c * k + t)).to_f32();
                }
                if let Some(bias) = bias {
                    total += *bias.add(c);
                }
                if relu {
                    total = total.max(0.0);
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
    let _product = crate::prof::product(crate::prof::Product::F32Broadcast, n, k, m);
    if super::v2::float_enabled() {
        return super::v2::nn::matmul_nn_into(x, m, k, w, n, out);
    }
    assert!(x.len() >= m * k && w.len() >= k * n && out.len() >= m * n);
    match level() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the block covers `out` exactly; the level was detected from the CPU.
        Level::Avx512 => unsafe { nn_avx512::block(x, k, w, n, 0..m, 0..n, out.as_mut_ptr()) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above.
        Level::Avx2 => unsafe { nn_avx2::block(x, k, w, n, 0..m, 0..n, out.as_mut_ptr()) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: as above; NEON is part of every arm64 CPU.
        Level::Neon => unsafe { nn_neon::block(x, k, w, n, m, out.as_mut_ptr()) },
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

/// Columns of one strip of `matmul_nn_strips_into` on this CPU, or `0` when
/// no strip kernel applies (scalar or v2 float path).
///
/// Matches each SIMD level's register tile width so every strip row is read
/// sequentially with zero skipped bytes: 32 floats (128 B, 2 x86 lines) on
/// AVX-512, 16 floats (64 B, 1 x86 line / half an Apple Silicon line) on
/// AVX2 and NEON.
pub fn nn_strip_cols() -> usize {
    if super::v2::float_enabled() {
        return 0;
    }
    match level() {
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 => 32,
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 => 16,
        #[cfg(target_arch = "aarch64")]
        Level::Neon => 16,
        Level::Scalar => 0,
    }
}

/// `matmul_nn_into` for weights in strips of `S = nn_strip_cols()` columns:
/// strip `s` holds columns `s * S..` of the `k` weight rows, one row after
/// the other (`[strip][k][S]`; the last strip can have fewer columns, and its
/// other places are not read). A tile then reads one contiguous run of bytes.
/// With `k x n` weights a tile reads `k` runs that are `n` values apart, and
/// keys that are in no cache came 1.5 times slower that way: the
/// cross-attention of a Whisper decoder step, after the token table went
/// through the caches. Each value is the one `matmul_nn_into` gives.
/// Only when `nn_strip_cols() > 0`.
pub fn matmul_nn_strips_into(x: &[f32], m: usize, k: usize, w: &[f32], n: usize, out: &mut [f32]) {
    let _product = crate::prof::product(crate::prof::Product::F32Broadcast, n, k, m);
    let strip = nn_strip_cols();
    assert!(strip > 0, "this CPU has no kernel for weights in strips");
    assert!(x.len() >= m * k && w.len() >= n.div_ceil(strip) * k * strip);
    assert!(out.len() >= m * n);
    match level() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the asserts keep every access inside the three buffers; level was detected from the CPU.
        Level::Avx512 => unsafe {
            nn_avx512::strips(x.as_ptr(), m, k, w.as_ptr(), n, out.as_mut_ptr());
        },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above.
        Level::Avx2 => unsafe {
            nn_avx2::strips(x.as_ptr(), m, k, w.as_ptr(), n, out.as_mut_ptr());
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: as above; NEON is part of every arm64 CPU.
        Level::Neon => unsafe {
            nn_neon::strips(x.as_ptr(), m, k, w.as_ptr(), n, out.as_mut_ptr());
        },
        Level::Scalar => unreachable!("nn_strip_cols is 0 for Scalar"),
    }
}

#[cfg(target_arch = "x86_64")]
mod nn_avx512 {
    use std::arch::x86_64::*;

    /// `$tile::<R>` for `$r` rows (1..=6).
    macro_rules! by_rows {
        ($tile:ident, $r:expr, $($arg:expr),*) => {
            match $r {
                6 => $tile::<6>($($arg),*),
                5 => $tile::<5>($($arg),*),
                4 => $tile::<4>($($arg),*),
                3 => $tile::<3>($($arg),*),
                2 => $tile::<2>($($arg),*),
                _ => $tile::<1>($($arg),*),
            }
        };
    }

    /// `$tile::<R>` for `$r` rows (1..=12): `12 x 32` uses 24 accumulators + 2 `b` + 1 `a` = 27 <= 32 ZMMs,
    /// halving weight strip loads vs `6 x 32`.
    macro_rules! by_rows12 {
        ($tile:ident, $r:expr, $($arg:expr),*) => {
            match $r {
                12 => $tile::<12>($($arg),*),
                11 => $tile::<11>($($arg),*),
                10 => $tile::<10>($($arg),*),
                9 => $tile::<9>($($arg),*),
                8 => $tile::<8>($($arg),*),
                7 => $tile::<7>($($arg),*),
                6 => $tile::<6>($($arg),*),
                5 => $tile::<5>($($arg),*),
                4 => $tile::<4>($($arg),*),
                3 => $tile::<3>($($arg),*),
                2 => $tile::<2>($($arg),*),
                _ => $tile::<1>($($arg),*),
            }
        };
    }

    #[target_feature(enable = "avx512f,avx512vl,fma")]
    pub unsafe fn block(
        x: &[f32],
        k: usize,
        w: &[f32],
        n: usize,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
        out: *mut f32,
    ) {
        let (x, w) = (x.as_ptr(), w.as_ptr().add(cols.start));
        let c = cols.len();
        let mut i = rows.start;
        if c >= 64 {
            while i < rows.end {
                let r = (rows.end - i).min(6);
                let o = out.add(i * n + cols.start);
                columns_wide(r, x.add(i * k), k, w, n, c, o, n);
                i += r;
            }
        } else {
            while i < rows.end {
                let r = (rows.end - i).min(12);
                let o = out.add(i * n + cols.start);
                columns12(r, x.add(i * k), k, w, n, c, o, n);
                i += r;
            }
        }
    }

    /// The product for weights in strips of 32 columns (`[strip][k][32]`).
    /// Uses `12 x 32` tiles (27 of 32 ZMMs) to halve weight strip reads vs `6 x 32`.
    #[target_feature(enable = "avx512f,avx512vl,fma")]
    pub unsafe fn strips(
        x: *const f32,
        m: usize,
        k: usize,
        w: *const f32,
        n: usize,
        out: *mut f32,
    ) {
        let mut i = 0;
        while i < m {
            let r = (m - i).min(12);
            let xr = x.add(i * k);
            let or = out.add(i * n);
            let mut j = 0;
            while j + 32 <= n {
                // Strip `j / 32` starts at `j / 32 * k * 32`, which is `j * k`.
                by_rows12!(pair_full, r, xr, k, w.add(j * k), 32, or.add(j), n);
                j += 32;
            }
            if j < n {
                columns12(r, xr, k, w.add(j * k), 32, n - j, or.add(j), n);
            }
            i += r;
        }
    }

    /// `r` rows (at most 6) by `c >= 64` columns: runs 64-column `quad` tiles
    /// (`6 x 64` = 29 of 32 ZMMs) so `hd = 64` attention (`probs * V`) reads `x` once.
    #[target_feature(enable = "avx512f,avx512vl,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn columns_wide(
        r: usize,
        x: *const f32,
        k: usize,
        w: *const f32,
        stride: usize,
        c: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut j = 0;
        while j + 64 <= c {
            by_rows!(quad, r, x, k, w.add(j), stride, out.add(j), n);
            j += 64;
        }
        if j < c {
            columns(r, x, k, w.add(j), stride, c - j, out.add(j), n);
        }
    }

    /// `r` rows (at most 12) by `c` columns.
    #[target_feature(enable = "avx512f,avx512vl,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn columns12(
        r: usize,
        x: *const f32,
        k: usize,
        w: *const f32,
        stride: usize,
        c: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut j = 0;
        while j < c {
            let part = match c - j {
                left @ (1..=16 | 25..=32) => left,
                17..=24 => 16,
                _ => 32,
            };
            let (w, o) = (w.add(j), out.add(j));
            match part {
                32 => by_rows12!(pair_full, r, x, k, w, stride, o, n),
                17..=31 => by_rows12!(pair, r, x, k, w, stride, part, o, n),
                9..=16 => by_rows12!(single, r, x, k, w, stride, part, o, n),
                _ => by_rows12!(half, r, x, k, w, stride, part, o, n),
            }
            j += part;
        }
    }

    /// `r` rows (at most 6) by `c` columns. `w` has `k` rows that are
    /// `stride` values apart; the rows of `out` are `n` apart.
    #[target_feature(enable = "avx512f,avx512vl,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn columns(
        r: usize,
        x: *const f32,
        k: usize,
        w: *const f32,
        stride: usize,
        c: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut j = 0;
        while j < c {
            // Strips of 32 columns in two registers. The columns that
            // remain take one register (up to 16), half of one (up to
            // 8), or the two in turn (17 to 24): a multiply-add on a
            // masked register costs as much as on a full one, and a
            // head of 36 values (Moonshine) ran its last 4 columns as 32.
            let part = match c - j {
                left @ (1..=16 | 25..=32) => left,
                17..=24 => 16,
                _ => 32,
            };
            let (w, o) = (w.add(j), out.add(j));
            match part {
                32 => by_rows!(pair_full, r, x, k, w, stride, o, n),
                17..=31 => by_rows!(pair, r, x, k, w, stride, part, o, n),
                9..=16 => by_rows!(single, r, x, k, w, stride, part, o, n),
                _ => by_rows!(half, r, x, k, w, stride, part, o, n),
            }
            j += part;
        }
    }

    /// R rows (at most 6) by 64 columns: 24 accumulators + 4 `b` + 1 `a` = 29 <= 32 ZMMs.
    #[target_feature(enable = "avx512f")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn quad<const R: usize>(
        x: *const f32,
        k: usize,
        w: *const f32,
        stride: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut c0 = [_mm512_setzero_ps(); R];
        let mut c1 = [_mm512_setzero_ps(); R];
        let mut c2 = [_mm512_setzero_ps(); R];
        let mut c3 = [_mm512_setzero_ps(); R];
        for e in 0..k {
            let row = w.add(e * stride);
            let b0 = _mm512_loadu_ps(row);
            let b1 = _mm512_loadu_ps(row.add(16));
            let b2 = _mm512_loadu_ps(row.add(32));
            let b3 = _mm512_loadu_ps(row.add(48));
            for r in 0..R {
                let a = _mm512_set1_ps(*x.add(r * k + e));
                c0[r] = _mm512_fmadd_ps(a, b0, c0[r]);
                c1[r] = _mm512_fmadd_ps(a, b1, c1[r]);
                c2[r] = _mm512_fmadd_ps(a, b2, c2[r]);
                c3[r] = _mm512_fmadd_ps(a, b3, c3[r]);
            }
        }
        for r in 0..R {
            let o = out.add(r * n);
            _mm512_storeu_ps(o, c0[r]);
            _mm512_storeu_ps(o.add(16), c1[r]);
            _mm512_storeu_ps(o.add(32), c2[r]);
            _mm512_storeu_ps(o.add(48), c3[r]);
        }
    }

    /// R rows (at most 12) by 32 full columns without right-edge masking.
    #[target_feature(enable = "avx512f")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn pair_full<const R: usize>(
        x: *const f32,
        k: usize,
        w: *const f32,
        stride: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut lo = [_mm512_setzero_ps(); R];
        let mut hi = [_mm512_setzero_ps(); R];
        for e in 0..k {
            let row = w.add(e * stride);
            let b0 = _mm512_loadu_ps(row);
            let b1 = _mm512_loadu_ps(row.add(16));
            for r in 0..R {
                let a = _mm512_set1_ps(*x.add(r * k + e));
                lo[r] = _mm512_fmadd_ps(a, b0, lo[r]);
                hi[r] = _mm512_fmadd_ps(a, b1, hi[r]);
            }
        }
        for r in 0..R {
            let o = out.add(r * n);
            _mm512_storeu_ps(o, lo[r]);
            _mm512_storeu_ps(o.add(16), hi[r]);
        }
    }

    /// R rows by 17 to 32 columns (`c`), masked at the right edge.
    #[target_feature(enable = "avx512f")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn pair<const R: usize>(
        x: *const f32,
        k: usize,
        w: *const f32,
        stride: usize,
        c: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mask_hi: __mmask16 = ((1u32 << (c - 16)) - 1) as u16;
        let mut lo = [_mm512_setzero_ps(); R];
        let mut hi = [_mm512_setzero_ps(); R];
        for e in 0..k {
            let row = w.add(e * stride);
            let b0 = _mm512_loadu_ps(row);
            let b1 = _mm512_maskz_loadu_ps(mask_hi, row.add(16));
            for r in 0..R {
                let a = _mm512_set1_ps(*x.add(r * k + e));
                lo[r] = _mm512_fmadd_ps(a, b0, lo[r]);
                hi[r] = _mm512_fmadd_ps(a, b1, hi[r]);
            }
        }
        for r in 0..R {
            let o = out.add(r * n);
            _mm512_storeu_ps(o, lo[r]);
            _mm512_mask_storeu_ps(o.add(16), mask_hi, hi[r]);
        }
    }

    /// `$name::<R>`: R rows by up to `c` columns in one register for each
    /// row, masked at the right edge.
    macro_rules! narrow {
        ($name:ident, $mask:ty, $zero:ident, $load:ident, $set1:ident, $fmadd:ident, $store:ident) => {
            #[target_feature(enable = "avx512f,avx512vl,fma")]
            #[allow(clippy::needless_range_loop)]
            unsafe fn $name<const R: usize>(
                x: *const f32,
                k: usize,
                w: *const f32,
                stride: usize,
                c: usize,
                out: *mut f32,
                n: usize,
            ) {
                let mask = ((1u32 << c) - 1) as $mask;
                let mut acc = [$zero(); R];
                for e in 0..k {
                    let b = $load(mask, w.add(e * stride));
                    for r in 0..R {
                        acc[r] = $fmadd($set1(*x.add(r * k + e)), b, acc[r]);
                    }
                }
                for r in 0..R {
                    $store(out.add(r * n), mask, acc[r]);
                }
            }
        };
    }

    narrow!(
        single,
        __mmask16,
        _mm512_setzero_ps,
        _mm512_maskz_loadu_ps,
        _mm512_set1_ps,
        _mm512_fmadd_ps,
        _mm512_mask_storeu_ps
    );
    narrow!(
        half,
        __mmask8,
        _mm256_setzero_ps,
        _mm256_maskz_loadu_ps,
        _mm256_set1_ps,
        _mm256_fmadd_ps,
        _mm256_mask_storeu_ps
    );
}

#[cfg(target_arch = "x86_64")]
mod nn_avx2 {
    use std::arch::x86_64::*;

    macro_rules! by_rows {
        ($tile:ident, $r:expr, $($arg:expr),*) => {
            match $r {
                6 => $tile::<6>($($arg),*),
                5 => $tile::<5>($($arg),*),
                4 => $tile::<4>($($arg),*),
                3 => $tile::<3>($($arg),*),
                2 => $tile::<2>($($arg),*),
                _ => $tile::<1>($($arg),*),
            }
        };
    }

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
        let (x, w) = (x.as_ptr(), w.as_ptr().add(cols.start));
        let mut i = rows.start;
        while i < rows.end {
            let r = (rows.end - i).min(6);
            let o = out.add(i * n + cols.start);
            columns(r, x.add(i * k), k, w, n, cols.len(), o, n);
            i += r;
        }
    }

    /// The product for weights in strips of 16 columns (`[strip][k][16]`,
    /// 64 bytes per row = one x86 cache line, matching one `tile16` width).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn strips(
        x: *const f32,
        m: usize,
        k: usize,
        w: *const f32,
        n: usize,
        out: *mut f32,
    ) {
        let full = n & !15;
        let tail = n - full;
        let mut i = 0;
        while i < m {
            let r = (m - i).min(6);
            let xr = x.add(i * k);
            let or = out.add(i * n);
            let mut j = 0;
            while j < full {
                // Strip `j / 16` starts at `j / 16 * k * 16`, which is `j * k`.
                by_rows!(tile16_strip, r, xr, k, w.add(j * k), or.add(j), n);
                j += 16;
            }
            if tail > 0 {
                columns(r, xr, k, w.add(full * k), 16, tail, or.add(full), n);
            }
            i += r;
        }
    }

    /// `r` rows (at most 6) by `c` columns. `w` has `k` rows `stride` values
    /// apart; the rows of `out` are `n` apart.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn columns(
        r: usize,
        x: *const f32,
        k: usize,
        w: *const f32,
        stride: usize,
        c: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut j = 0;
        while j + 16 <= c {
            by_rows!(tile16, r, x, k, w.add(j), stride, out.add(j), n);
            j += 16;
        }
        if j + 8 <= c {
            by_rows!(tile8, r, x, k, w.add(j), stride, out.add(j), n);
            j += 8;
        }
        if j < c {
            by_rows!(tile_masked, r, x, k, w.add(j), stride, c - j, out.add(j), n);
        }
    }

    /// R rows by 16 contiguous strip columns (constant stride 16 = 64 bytes).
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile16_strip<const R: usize>(
        x: *const f32,
        k: usize,
        w: *const f32,
        out: *mut f32,
        n: usize,
    ) {
        let mut lo = [_mm256_setzero_ps(); R];
        let mut hi = [_mm256_setzero_ps(); R];
        let mut row = w;
        for e in 0..k {
            let b0 = _mm256_loadu_ps(row);
            let b1 = _mm256_loadu_ps(row.add(8));
            row = row.add(16);
            for r in 0..R {
                let a = _mm256_broadcast_ss(&*x.add(r * k + e));
                lo[r] = _mm256_fmadd_ps(a, b0, lo[r]);
                hi[r] = _mm256_fmadd_ps(a, b1, hi[r]);
            }
        }
        for r in 0..R {
            let o = out.add(r * n);
            _mm256_storeu_ps(o, lo[r]);
            _mm256_storeu_ps(o.add(8), hi[r]);
        }
    }

    /// R rows by 16 columns.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile16<const R: usize>(
        x: *const f32,
        k: usize,
        w: *const f32,
        stride: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut lo = [_mm256_setzero_ps(); R];
        let mut hi = [_mm256_setzero_ps(); R];
        for e in 0..k {
            let row = w.add(e * stride);
            let b0 = _mm256_loadu_ps(row);
            let b1 = _mm256_loadu_ps(row.add(8));
            for r in 0..R {
                let a = _mm256_broadcast_ss(&*x.add(r * k + e));
                lo[r] = _mm256_fmadd_ps(a, b0, lo[r]);
                hi[r] = _mm256_fmadd_ps(a, b1, hi[r]);
            }
        }
        for r in 0..R {
            let o = out.add(r * n);
            _mm256_storeu_ps(o, lo[r]);
            _mm256_storeu_ps(o.add(8), hi[r]);
        }
    }

    /// R rows by 8 columns.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile8<const R: usize>(
        x: *const f32,
        k: usize,
        w: *const f32,
        stride: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut acc = [_mm256_setzero_ps(); R];
        for e in 0..k {
            let b = _mm256_loadu_ps(w.add(e * stride));
            for r in 0..R {
                let a = _mm256_broadcast_ss(&*x.add(r * k + e));
                acc[r] = _mm256_fmadd_ps(a, b, acc[r]);
            }
        }
        for r in 0..R {
            _mm256_storeu_ps(out.add(r * n), acc[r]);
        }
    }

    /// R rows by 1..=7 columns, masked so unused lanes never fault or touch out-of-bounds memory.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile_masked<const R: usize>(
        x: *const f32,
        k: usize,
        w: *const f32,
        stride: usize,
        c: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut mask_lanes = [0i32; 8];
        for lane in &mut mask_lanes[..c] {
            *lane = -1;
        }
        let mask = _mm256_loadu_si256(mask_lanes.as_ptr().cast());
        let mut acc = [_mm256_setzero_ps(); R];
        for e in 0..k {
            let b = _mm256_maskload_ps(w.add(e * stride), mask);
            for r in 0..R {
                let a = _mm256_broadcast_ss(&*x.add(r * k + e));
                acc[r] = _mm256_fmadd_ps(a, b, acc[r]);
            }
        }
        for r in 0..R {
            _mm256_maskstore_ps(out.add(r * n), mask, acc[r]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod nn_neon {
    use std::arch::aarch64::*;

    macro_rules! by_rows {
        ($tile:ident, $r:expr, $($arg:expr),*) => {
            match $r {
                6 => $tile::<6>($($arg),*),
                5 => $tile::<5>($($arg),*),
                4 => $tile::<4>($($arg),*),
                3 => $tile::<3>($($arg),*),
                2 => $tile::<2>($($arg),*),
                _ => $tile::<1>($($arg),*),
            }
        };
    }

    /// 6 rows by 16 columns: 24 accumulators, 4 strips of `w`, 1 input.
    pub unsafe fn block(x: &[f32], k: usize, w: &[f32], n: usize, m: usize, out: *mut f32) {
        let (x, w) = (x.as_ptr(), w.as_ptr());
        let mut i = 0;
        while i < m {
            let r = (m - i).min(6);
            columns(r, x.add(i * k), k, w, n, n, out.add(i * n), n);
            i += r;
        }
    }

    /// The product for weights in strips of 16 columns (`[strip][k][16]`,
    /// 64 bytes per row: contiguous sequential reads matching `tile16` width,
    /// two rows per 128-byte Apple Silicon cache line).
    pub unsafe fn strips(
        x: *const f32,
        m: usize,
        k: usize,
        w: *const f32,
        n: usize,
        out: *mut f32,
    ) {
        let full = n & !15;
        let tail = n - full;
        let mut i = 0;
        while i < m {
            let r = (m - i).min(6);
            let xr = x.add(i * k);
            let or = out.add(i * n);
            let mut j = 0;
            while j < full {
                // Strip `j / 16` starts at `j / 16 * k * 16`, which is `j * k`.
                by_rows!(tile16_strip, r, xr, k, w.add(j * k), or.add(j), n);
                j += 16;
            }
            if tail > 0 {
                columns(r, xr, k, w.add(full * k), 16, tail, or.add(full), n);
            }
            i += r;
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn columns(
        r: usize,
        x: *const f32,
        k: usize,
        w: *const f32,
        stride: usize,
        c: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut j = 0;
        while j + 16 <= c {
            by_rows!(tile16, r, x, k, w.add(j), stride, out.add(j), n);
            j += 16;
        }
        if j + 8 <= c {
            by_rows!(tile8, r, x, k, w.add(j), stride, out.add(j), n);
            j += 8;
        }
        if j + 4 <= c {
            by_rows!(tile4, r, x, k, w.add(j), stride, out.add(j), n);
            j += 4;
        }
        if j < c {
            let rem = c - j;
            for row in 0..r {
                for col in 0..rem {
                    let mut total = 0.0f32;
                    for e in 0..k {
                        total = (*x.add(row * k + e)).mul_add(*w.add(e * stride + j + col), total);
                    }
                    *out.add(row * n + j + col) = total;
                }
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile16_strip<const R: usize>(
        x: *const f32,
        k: usize,
        w: *const f32,
        out: *mut f32,
        n: usize,
    ) {
        let mut acc = [[vdupq_n_f32(0.0); 4]; R];
        let mut row = w;
        for e in 0..k {
            let b = [
                vld1q_f32(row),
                vld1q_f32(row.add(4)),
                vld1q_f32(row.add(8)),
                vld1q_f32(row.add(12)),
            ];
            row = row.add(16);
            for r in 0..R {
                let a = vld1q_dup_f32(x.add(r * k + e));
                for q in 0..4 {
                    acc[r][q] = vfmaq_f32(acc[r][q], a, b[q]);
                }
            }
        }
        for r in 0..R {
            let o = out.add(r * n);
            for q in 0..4 {
                vst1q_f32(o.add(q * 4), acc[r][q]);
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile16<const R: usize>(
        x: *const f32,
        k: usize,
        w: *const f32,
        stride: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut acc = [[vdupq_n_f32(0.0); 4]; R];
        for e in 0..k {
            let row = w.add(e * stride);
            let b = [
                vld1q_f32(row),
                vld1q_f32(row.add(4)),
                vld1q_f32(row.add(8)),
                vld1q_f32(row.add(12)),
            ];
            for r in 0..R {
                let a = vld1q_dup_f32(x.add(r * k + e));
                for q in 0..4 {
                    acc[r][q] = vfmaq_f32(acc[r][q], a, b[q]);
                }
            }
        }
        for r in 0..R {
            let o = out.add(r * n);
            for q in 0..4 {
                vst1q_f32(o.add(q * 4), acc[r][q]);
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile8<const R: usize>(
        x: *const f32,
        k: usize,
        w: *const f32,
        stride: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut acc = [[vdupq_n_f32(0.0); 2]; R];
        for e in 0..k {
            let row = w.add(e * stride);
            let b = [vld1q_f32(row), vld1q_f32(row.add(4))];
            for r in 0..R {
                let a = vld1q_dup_f32(x.add(r * k + e));
                acc[r][0] = vfmaq_f32(acc[r][0], a, b[0]);
                acc[r][1] = vfmaq_f32(acc[r][1], a, b[1]);
            }
        }
        for r in 0..R {
            let o = out.add(r * n);
            vst1q_f32(o, acc[r][0]);
            vst1q_f32(o.add(4), acc[r][1]);
        }
    }

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile4<const R: usize>(
        x: *const f32,
        k: usize,
        w: *const f32,
        stride: usize,
        out: *mut f32,
        n: usize,
    ) {
        let mut acc = [vdupq_n_f32(0.0); R];
        for e in 0..k {
            let b = vld1q_f32(w.add(e * stride));
            for r in 0..R {
                let a = vld1q_dup_f32(x.add(r * k + e));
                acc[r] = vfmaq_f32(acc[r], a, b);
            }
        }
        for r in 0..R {
            vst1q_f32(out.add(r * n), acc[r]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tensors::{f16_to_f32, f32_to_f16};
    use super::*;

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((*state >> 40) as u32) as f32 / ((1u32 << 24) as f32);
        bits * 2.0 - 1.0
    }

    fn ref_matmul(
        x: &[f32],
        m: usize,
        k: usize,
        w: &[f32],
        n: usize,
        bias: Option<&[f32]>,
        relu: bool,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut sum = 0.0f64;
                for i in 0..k {
                    sum += f64::from(x[r * k + i]) * f64::from(w[c * k + i]);
                }
                let mut v = sum as f32;
                if let Some(b) = bias {
                    v += b[c];
                }
                if relu {
                    v = v.max(0.0);
                }
                out[r * n + c] = v;
            }
        }
        out
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, tag: &str) {
        assert_eq!(got.len(), want.len(), "{tag}: length mismatch");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(g.is_finite(), "{tag}: non-finite {g} at {i}");
            let diff = (g - w).abs();
            assert!(
                diff <= tol,
                "{tag}: at {i} got={g} want={w} diff={diff} > {tol}"
            );
        }
    }

    #[test]
    fn matmul_and_epilogue_cover_all_tile_and_k_paths() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        assert!(matmul::<f32>(&[], 0, 16, &[0.0; 128], 8).is_empty());
        assert!(matmul::<f32>(&[0.0; 64], 4, 16, &[], 0).is_empty());

        // Covers k % 512 == 0 (3-col AVX-512 path) and non-512 k (6x4 AVX-512/NEON path),
        // odd k tails, every row remainder 1..=10 plus multi-block m = 65, and odd n.
        let shapes = [
            (1, 1, 1),
            (2, 2, 7),
            (3, 3, 15),
            (4, 4, 16),
            (5, 5, 17),
            (6, 7, 33),
            (7, 13, 64),
            (8, 19, 127),
            (9, 4, 32),
            (10, 11, 512),
            (13, 67, 48),
            (16, 17, 512),
            (32, 9, 24),
            (65, 70, 31),
        ];
        for (m, n, k) in shapes {
            let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed)).collect();
            let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
            let bias: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * 0.5).collect();
            let w_f16: Vec<u16> = w.iter().map(|&v| f32_to_f16(v)).collect();
            let w_f16_ref: Vec<f32> = w_f16.iter().map(|&b| f16_to_f32(b)).collect();
            let tol = (k as f32).sqrt() * 2e-5 + 1e-4;

            for use_bias in [false, true] {
                let b_opt = use_bias.then_some(bias.as_slice());
                for relu in [false, true] {
                    let tag = format!("m={m} n={n} k={k} bias={use_bias} relu={relu}");
                    let want32 = ref_matmul(&x, m, k, &w, n, b_opt, relu);
                    let got32 = matmul_then(&x, m, k, &w, n, b_opt, relu);
                    assert_close(&got32, &want32, tol, &format!("f32 {tag}"));

                    let mut into32 = vec![f32::NAN; m * n];
                    matmul_into_then(&x, m, k, &w, n, b_opt, relu, &mut into32);
                    assert_close(&into32, &want32, tol, &format!("f32_into {tag}"));

                    let want16 = ref_matmul(&x, m, k, &w_f16_ref, n, b_opt, relu);
                    let got16 = matmul_then(&x, m, k, &w_f16, n, b_opt, relu);
                    assert_close(&got16, &want16, tol, &format!("f16 {tag}"));

                    let mut into16 = vec![f32::NAN; m * n];
                    matmul_into_then(&x, m, k, &w_f16, n, b_opt, relu, &mut into16);
                    assert_close(&into16, &want16, tol, &format!("f16_into {tag}"));
                }
            }
            let mut plain = vec![f32::NAN; m * n];
            matmul_into(&x, m, k, &w, n, &mut plain);
            let want = ref_matmul(&x, m, k, &w, n, None, false);
            assert_close(
                &plain,
                &want,
                tol,
                &format!("matmul_into m={m} n={n} k={k}"),
            );
        }

        // ReLU must clamp negatives, -0.0, and NaN to 0.0 across C = 1, 2, 3, 4 tiles.
        for k in [16, 512] {
            for n in [1, 2, 3, 4, 7] {
                let m = 6;
                let mut x = vec![-1.0f32; m * k];
                x[0] = f32::NAN;
                x[k] = -0.0;
                let w = vec![1.0f32; n * k];
                let got = matmul_then(&x, m, k, &w, n, None, true);
                for (i, &v) in got.iter().enumerate() {
                    assert!(v == 0.0 && !v.is_nan(), "relu k={k} n={n} at {i}: got {v}");
                }
            }
        }
    }

    #[test]
    fn matmul_nn_and_strips_match_reference() {
        let mut seed = 0xfeed_face_cafe_beefu64;
        let strip = nn_strip_cols();
        for &m in &[1, 2, 5, 6, 7, 12, 13] {
            for &k in &[1, 7, 19] {
                for &n in &[
                    1, 3, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 40, 63, 64, 65, 68,
                ] {
                    let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed)).collect();
                    let w: Vec<f32> = (0..k * n).map(|_| lcg(&mut seed)).collect();
                    let mut want = vec![0.0f32; m * n];
                    for r in 0..m {
                        for c in 0..n {
                            let mut s = 0.0f64;
                            for e in 0..k {
                                s += f64::from(x[r * k + e]) * f64::from(w[e * n + c]);
                            }
                            want[r * n + c] = s as f32;
                        }
                    }
                    let mut got = vec![f32::NAN; m * n];
                    matmul_nn_into(&x, m, k, &w, n, &mut got);
                    assert_close(&got, &want, 1e-4, &format!("nn m={m} k={k} n={n}"));

                    if strip > 0 {
                        // Fill unused trailing strip lanes with NaN to verify they are never read.
                        let strips = n.div_ceil(strip);
                        let mut packed = vec![f32::NAN; strips * k * strip];
                        for s in 0..strips {
                            let col0 = s * strip;
                            let cols = (n - col0).min(strip);
                            for e in 0..k {
                                packed[(s * k + e) * strip..][..cols]
                                    .copy_from_slice(&w[e * n + col0..][..cols]);
                            }
                        }
                        let mut got_s = vec![f32::NAN; m * n];
                        matmul_nn_strips_into(&x, m, k, &packed, n, &mut got_s);
                        assert_close(&got_s, &want, 1e-4, &format!("nn_strips m={m} k={k} n={n}"));
                    }
                }
            }
        }
    }
}
