//! Int8 weights with one scale per row and group of inputs: 128-value
//! groups quantized at load from f32 or f16 (`Precision::Int8`), or ggml's
//! Q8_0 blocks of 32 with f16 scales, taken as stored (`Precision::Q8_0`).
//! Activations go through the same i8 quantizer as the ternary layers
//! (`kernels::Quantized`), in groups of the same size, so a product is
//! i8 x i8 dot products in i32, one f32 scale pair per group. Q8_0 rounds
//! the activation scales to f16, as ggml's `vec_dot_q8_0_q8_0` reads them.
//!
//! VNNI (AVX-512, or 256-bit AVX-VNNI on Intel client CPUs without
//! AVX-512): `vpdpbusd` multiplies unsigned by signed bytes, so each weight
//! byte is stored flipped, as `w + 128`, and `128 * sum(x)` per group (the
//! quantizer computes the sums) comes off again. AVX2: the sign
//! trick, `maddubs(|w|, sign(x, w))`; weights and activations stay within
//! +-127, so a pair sum fits in i16. Arm: `sdot` with the dot product
//! feature, widening multiplies on plain NEON; both take signed bytes.

use super::buf::Buf;
#[cfg(target_arch = "aarch64")]
use super::kernels::neon;
use super::kernels::{Isa, Quantized, GROUP};
use super::store::Store;
use super::team;

#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: tasks write disjoint output columns; the buffer outlives the step.
unsafe impl Send for OutPtr {}
// SAFETY: as for `Send`: tasks write disjoint output columns.
unsafe impl Sync for OutPtr {}

/// The product kernel. The activation quantizer follows `Isa`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Path {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// 8 outputs per block with 256-bit `vpdpbusd` (VEX form, AVX-VNNI).
    #[cfg(target_arch = "x86_64")]
    Vnni256Vex,
    /// 8 outputs per block with 256-bit `vpdpbusd` (EVEX form, AVX-512 VNNI + VL).
    #[cfg(target_arch = "x86_64")]
    Vnni256Evex,
    #[cfg(target_arch = "x86_64")]
    Vnni512,
    /// 8 outputs per block, widening multiplies (every arm64 CPU).
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// 8 outputs per block with `sdot` (Armv8.2 dot product, every Apple M chip).
    #[cfg(target_arch = "aarch64")]
    NeonDot,
    /// 8 outputs per block with `smmla` (Armv8.6 int8 matrix multiply: Apple
    /// M2 and later, recent Arm cores): 2 x 2 dot products of 8 bytes per
    /// instruction, twice the products of `sdot`. `TRIT_ISA=dotprod` forces
    /// `NeonDot` on such a CPU.
    #[cfg(target_arch = "aarch64")]
    NeonMmla,
}

impl Path {
    fn rows(self) -> usize {
        match self {
            Self::Scalar => 1,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 | Self::Vnni256Vex | Self::Vnni256Evex => 8,
            #[cfg(target_arch = "x86_64")]
            Self::Vnni512 => 16,
            #[cfg(target_arch = "aarch64")]
            Self::Neon | Self::NeonDot | Self::NeonMmla => 8,
        }
    }

    /// The index of weight (`row`, `kk`) of a layer `k` wide in the packed
    /// bytes. Scalar: row-major. `NeonMmla`: `[n / 8][k / 8][8][8]`, so one
    /// 16-byte load holds 8 inputs of two rows. Other SIMD paths:
    /// `[n / rows][k / 4][rows][4]`, so one load holds four inputs of
    /// `rows` outputs.
    fn at(self, k: usize, row: usize, kk: usize) -> usize {
        let rows = self.rows();
        let (block, r) = (row / rows, row % rows);
        match self {
            Self::Scalar => row * k + kk,
            #[cfg(target_arch = "aarch64")]
            Self::NeonMmla => ((block * (k / 8) + kk / 8) * rows + r) * 8 + kk % 8,
            _ => ((block * (k / 4) + kk / 4) * rows + r) * 4 + kk % 4,
        }
    }

    fn flipped(self) -> bool {
        #[cfg(target_arch = "x86_64")]
        return matches!(self, Self::Vnni256Vex | Self::Vnni256Evex | Self::Vnni512);
        #[cfg(not(target_arch = "x86_64"))]
        false
    }
}

/// Defines `2 blocks (16 outputs) x F frames` 256-bit VNNI tiles and a 2-chain
/// single-frame fast path, parameterized by the VEX (`_mm256_dpbusd_avx_epi32`)
/// or EVEX (`_mm256_dpbusd_epi32`) intrinsic so every tile unrolls `0..F` into
/// YMM registers with zero stack spills inside the `k4` loop (`8 dots + 2 w + 1 x = 11 <= 16`).
#[cfg(target_arch = "x86_64")]
macro_rules! define_vnni256 {
    ($task:ident, $tile_k:ident, $tile:ident, $m1:ident, $feature:literal, $dpbusd:path) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $feature)]
        unsafe fn $task(
            &self,
            q: &Quantized,
            task: usize,
            frames: std::ops::Range<usize>,
            bias: Option<&[f32]>,
            out: OutPtr,
        ) {
            use std::arch::x86_64::__m256;
            // 16 rows * k bytes overflows 32 KB L1d at k >= 2048 (32 KB at 2048, 64 KB at 4096).
            // K-blocking in 1024-byte chunks (16 KB weights + 4 KB x + 2 KB stack acc = 22 KB)
            // keeps weights L1d-hot across 8 micro-tiles of 4 frames.
            const K_BYTES: usize = 1024;
            const STRIP: usize = 32;
            let groups = self.k / self.group;
            let k_groups = (K_BYTES / self.group).max(1);
            if self.k >= 2048 && groups > k_groups && frames.end - frames.start > 4 {
                let mut buf = std::mem::MaybeUninit::<[[__m256; STRIP]; 2]>::uninit();
                let base0 = buf.as_mut_ptr().cast::<__m256>();
                let base1 = base0.add(STRIP);
                macro_rules! run_strip {
                    ($first:literal, $last:literal, $strip_start:expr, $strip_len:expr, $g0:expr, $g1:expr) => {{
                        let mut off = 0;
                        while off + 4 <= $strip_len {
                            self.$tile_k::<4, $first, $last>(
                                q,
                                task,
                                $strip_start + off,
                                $g0..$g1,
                                base0.add(off),
                                base1.add(off),
                                bias,
                                out,
                            );
                            off += 4;
                        }
                        match $strip_len - off {
                            1 => self.$tile_k::<1, $first, $last>(
                                q,
                                task,
                                $strip_start + off,
                                $g0..$g1,
                                base0.add(off),
                                base1.add(off),
                                bias,
                                out,
                            ),
                            2 => self.$tile_k::<2, $first, $last>(
                                q,
                                task,
                                $strip_start + off,
                                $g0..$g1,
                                base0.add(off),
                                base1.add(off),
                                bias,
                                out,
                            ),
                            3 => self.$tile_k::<3, $first, $last>(
                                q,
                                task,
                                $strip_start + off,
                                $g0..$g1,
                                base0.add(off),
                                base1.add(off),
                                bias,
                                out,
                            ),
                            _ => {}
                        }
                    }};
                }
                let mut strip_start = frames.start;
                while strip_start < frames.end {
                    let strip_len = (frames.end - strip_start).min(STRIP);
                    let mut g0 = 0;
                    while g0 < groups {
                        let g1 = (g0 + k_groups).min(groups);
                        match (g0 == 0, g1 == groups) {
                            (true, false) => {
                                run_strip!(true, false, strip_start, strip_len, g0, g1)
                            }
                            (false, false) => {
                                run_strip!(false, false, strip_start, strip_len, g0, g1)
                            }
                            (false, true) => {
                                run_strip!(false, true, strip_start, strip_len, g0, g1)
                            }
                            (true, true) => run_strip!(true, true, strip_start, strip_len, g0, g1),
                        }
                        g0 = g1;
                    }
                    strip_start += strip_len;
                }
                return;
            }
            let mut frame = frames.start;
            while frame + 4 <= frames.end {
                self.$tile::<4>(q, task, frame, bias, out);
                frame += 4;
            }
            match frames.end - frame {
                1 => self.$m1(q, task, frame, bias, out),
                2 => self.$tile::<2>(q, task, frame, bias, out),
                3 => self.$tile::<3>(q, task, frame, bias, out),
                _ => {}
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $feature)]
        unsafe fn $m1(
            &self,
            q: &Quantized,
            task: usize,
            frame: usize,
            bias: Option<&[f32]>,
            out: OutPtr,
        ) {
            use std::arch::x86_64::*;
            let (k, k4s, groups) = (self.k, self.k / 4, self.k / self.group);
            let quads = self.group / 4;
            debug_assert!(quads.is_multiple_of(2));
            let blocks = [2 * task, 2 * task + 1];
            let w0_base = self.weights.as_ptr().add(blocks[0] * k4s * 32);
            let w1_base = self.weights.as_ptr().add(blocks[1] * k4s * 32);
            let s0_base = self.scales.as_ptr().add(blocks[0] * groups * 8);
            let s1_base = self.scales.as_ptr().add(blocks[1] * groups * 8);
            let x_row = q.values.as_ptr().add(frame * k);
            let x_scales = q.scales.as_ptr().add(frame * groups);
            let x_sums = q.sums.as_ptr().add(frame * groups);
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            for g in 0..groups {
                let neg_bias = _mm256_set1_epi32((-128i32).wrapping_mul(*x_sums.add(g)));
                let (mut d0a, mut d0b) = (neg_bias, _mm256_setzero_si256());
                let (mut d1a, mut d1b) = (neg_bias, _mm256_setzero_si256());
                let mut k4 = g * quads;
                let k4_end = k4 + quads;
                while k4 + 2 <= k4_end {
                    let w0a = _mm256_loadu_si256(w0_base.add(k4 * 32).cast());
                    let w1a = _mm256_loadu_si256(w1_base.add(k4 * 32).cast());
                    let xa = _mm256_set1_epi32(std::ptr::read_unaligned(
                        x_row.add(k4 * 4).cast::<i32>(),
                    ));
                    d0a = $dpbusd(d0a, w0a, xa);
                    d1a = $dpbusd(d1a, w1a, xa);

                    let w0b = _mm256_loadu_si256(w0_base.add((k4 + 1) * 32).cast());
                    let w1b = _mm256_loadu_si256(w1_base.add((k4 + 1) * 32).cast());
                    let xb = _mm256_set1_epi32(std::ptr::read_unaligned(
                        x_row.add((k4 + 1) * 4).cast::<i32>(),
                    ));
                    d0b = $dpbusd(d0b, w0b, xb);
                    d1b = $dpbusd(d1b, w1b, xb);
                    k4 += 2;
                }
                let d0 = _mm256_add_epi32(d0a, d0b);
                let d1 = _mm256_add_epi32(d1a, d1b);
                let xscale = _mm256_set1_ps(*x_scales.add(g));
                let wscale0 = _mm256_loadu_ps(s0_base.add(g * 8));
                let wscale1 = _mm256_loadu_ps(s1_base.add(g * 8));
                acc0 =
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), _mm256_mul_ps(wscale0, xscale), acc0);
                acc1 =
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1), _mm256_mul_ps(wscale1, xscale), acc1);
            }
            if let Some(bias) = bias {
                acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(bias.as_ptr().add(blocks[0] * 8)));
                acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(bias.as_ptr().add(blocks[1] * 8)));
            }
            let dst = out.0.add(frame * self.n);
            _mm256_storeu_ps(dst.add(blocks[0] * 8), acc0);
            _mm256_storeu_ps(dst.add(blocks[1] * 8), acc1);
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $feature)]
        #[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
        unsafe fn $tile_k<const F: usize, const FIRST: bool, const LAST: bool>(
            &self,
            q: &Quantized,
            task: usize,
            frame0: usize,
            g_range: std::ops::Range<usize>,
            buf0: *mut std::arch::x86_64::__m256,
            buf1: *mut std::arch::x86_64::__m256,
            bias: Option<&[f32]>,
            out: OutPtr,
        ) {
            use std::arch::x86_64::*;
            let (k, k4s, groups) = (self.k, self.k / 4, self.k / self.group);
            let quads = self.group / 4;
            debug_assert!(quads.is_multiple_of(2));
            let blocks = [2 * task, 2 * task + 1];
            let w0_base = self.weights.as_ptr().add(blocks[0] * k4s * 32);
            let w1_base = self.weights.as_ptr().add(blocks[1] * k4s * 32);
            let s0_base = self.scales.as_ptr().add(blocks[0] * groups * 8);
            let s1_base = self.scales.as_ptr().add(blocks[1] * groups * 8);
            let x_base = q.values.as_ptr().add(frame0 * k);
            let x_scales = q.scales.as_ptr().add(frame0 * groups);
            let x_sums = q.sums.as_ptr().add(frame0 * groups);
            let mut acc = [[_mm256_setzero_ps(); F]; 2];
            if !FIRST {
                for f in 0..F {
                    acc[0][f] = *buf0.add(f);
                    acc[1][f] = *buf1.add(f);
                }
            }
            for g in g_range {
                let mut dots = [[_mm256_setzero_si256(); F]; 2];
                for f in 0..F {
                    let neg_bias =
                        _mm256_set1_epi32((-128i32).wrapping_mul(*x_sums.add(f * groups + g)));
                    dots[0][f] = neg_bias;
                    dots[1][f] = neg_bias;
                }
                for k4 in g * quads..(g + 1) * quads {
                    let w0 = _mm256_loadu_si256(w0_base.add(k4 * 32).cast());
                    let w1 = _mm256_loadu_si256(w1_base.add(k4 * 32).cast());
                    let off = k4 * 4;
                    for f in 0..F {
                        let x = _mm256_set1_epi32(std::ptr::read_unaligned(
                            x_base.add(f * k + off).cast::<i32>(),
                        ));
                        dots[0][f] = $dpbusd(dots[0][f], w0, x);
                        dots[1][f] = $dpbusd(dots[1][f], w1, x);
                    }
                }
                let wscale0 = _mm256_loadu_ps(s0_base.add(g * 8));
                let wscale1 = _mm256_loadu_ps(s1_base.add(g * 8));
                for f in 0..F {
                    let xscale = _mm256_set1_ps(*x_scales.add(f * groups + g));
                    let scale0 = _mm256_mul_ps(wscale0, xscale);
                    let scale1 = _mm256_mul_ps(wscale1, xscale);
                    acc[0][f] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dots[0][f]), scale0, acc[0][f]);
                    acc[1][f] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dots[1][f]), scale1, acc[1][f]);
                }
            }
            if LAST {
                for (b, block) in blocks.iter().enumerate() {
                    if let Some(bias) = bias {
                        let shift = _mm256_loadu_ps(bias.as_ptr().add(block * 8));
                        for sum in acc[b].iter_mut() {
                            *sum = _mm256_add_ps(*sum, shift);
                        }
                    }
                    for f in 0..F {
                        _mm256_storeu_ps(out.0.add((frame0 + f) * self.n + block * 8), acc[b][f]);
                    }
                }
            } else {
                for f in 0..F {
                    *buf0.add(f) = acc[0][f];
                    *buf1.add(f) = acc[1][f];
                }
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $feature)]
        unsafe fn $tile<const F: usize>(
            &self,
            q: &Quantized,
            task: usize,
            frame0: usize,
            bias: Option<&[f32]>,
            out: OutPtr,
        ) {
            self.$tile_k::<F, true, true>(
                q,
                task,
                frame0,
                0..self.k / self.group,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                bias,
                out,
            );
        }
    };
}

pub fn path_name() -> &'static str {
    match super::isa::isa_choice().int8 {
        Path::Scalar => "scalar",
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => "avx2",
        #[cfg(target_arch = "x86_64")]
        Path::Vnni256Vex | Path::Vnni256Evex => "vnni256",
        #[cfg(target_arch = "x86_64")]
        Path::Vnni512 => "vnni512",
        #[cfg(target_arch = "aarch64")]
        Path::Neon => "neon",
        #[cfg(target_arch = "aarch64")]
        Path::NeonDot => "neon-dotprod",
        #[cfg(target_arch = "aarch64")]
        Path::NeonMmla => "neon-i8mm",
    }
}

pub struct Int8 {
    isa: Isa,
    path: Path,
    pub n: usize,
    pub k: usize,
    /// Inputs per scale: `GROUP`, or `Q8_0_BLOCK` for ggml Q8_0 weights.
    group: usize,
    /// Round each activation scale to f16 (Q8_0).
    half_scales: bool,
    /// Packed for the path (`Path::at`).
    weights: Store<i8>,
    /// `[n / rows][k / group][rows]`.
    scales: Store<f32>,
    /// The layer in the second kernel set (`TRIT_KERNELS=v2`); `weights`
    /// and `scales` are then empty.
    v2: Option<Box<super::v2::product::IntProduct>>,
}

/// Values per block of ggml's Q8_0.
pub const Q8_0_BLOCK: usize = 32;

pub(super) fn group_scale(max: f32) -> f32 {
    if max > 0.0 {
        max / 127.0
    } else {
        1.0
    }
}

pub(super) fn quantize_code(value: f32, scale: f32) -> i8 {
    (value / scale).round_ties_even().clamp(-127.0, 127.0) as i8
}

/// `group_scale` of the group's largest magnitude, and `quantize_code` of
/// each value into `codes`, four lanes at a time: the same values for
/// finite inputs (the maximum does not depend on the order, `frintn` rounds
/// half to even). A row with a NaN or an infinity takes the scalar path: there
/// the order of the maximum matters.
#[cfg(target_arch = "aarch64")]
fn quantize_group_neon(span: &[f32], codes: &mut [i8; GROUP]) -> f32 {
    use std::arch::aarch64::*;
    assert_eq!(span.len(), GROUP);
    // SAFETY: NEON is part of every arm64 CPU; `span` and `codes` hold
    // GROUP values, read and written 16 at a time.
    unsafe {
        let p = span.as_ptr();
        let mut most = [vdupq_n_f32(0.0); 4];
        for i in 0..GROUP / 16 {
            for (j, most) in most.iter_mut().enumerate() {
                *most = vmaxnmq_f32(*most, vabsq_f32(vld1q_f32(p.add(16 * i + 4 * j))));
            }
        }
        let most = vmaxnmq_f32(vmaxnmq_f32(most[0], most[1]), vmaxnmq_f32(most[2], most[3]));
        let scale = group_scale(vmaxnmvq_f32(most));
        let divisor = vdupq_n_f32(scale);
        let (low, high) = (vdupq_n_f32(-127.0), vdupq_n_f32(127.0));
        for i in 0..GROUP / 16 {
            let mut q = [vdupq_n_s32(0); 4];
            for (j, q) in q.iter_mut().enumerate() {
                let v = vrndnq_f32(vdivq_f32(vld1q_f32(p.add(16 * i + 4 * j)), divisor));
                *q = vcvtq_s32_f32(vminq_f32(vmaxq_f32(v, low), high));
            }
            let first = vcombine_s16(vmovn_s32(q[0]), vmovn_s32(q[1]));
            let second = vcombine_s16(vmovn_s32(q[2]), vmovn_s32(q[3]));
            vst1q_s8(
                codes.as_mut_ptr().add(16 * i),
                vcombine_s8(vmovn_s16(first), vmovn_s16(second)),
            );
        }
        scale
    }
}

impl Int8 {
    /// Whether a `n x k` layer fits the kernels: whole groups and whole tiles.
    pub fn fits(n: usize, k: usize) -> bool {
        Self::fits_group(n, k, GROUP)
    }

    pub fn fits_group(n: usize, k: usize, group: usize) -> bool {
        k.is_multiple_of(group) && n.is_multiple_of(32)
    }

    /// A layer that the second kernel set holds.
    fn second(isa: Isa, path: Path, layer: super::v2::product::IntProduct) -> Self {
        Self {
            isa,
            path,
            n: layer.n,
            k: layer.k,
            group: layer.group(),
            half_scales: false,
            weights: Store::owned(Vec::new()),
            scales: Store::owned(Vec::new()),
            v2: Some(Box::new(layer)),
        }
    }

    /// Maps a previously cached int8 layer directly from the active read cache
    /// without reading or quantizing the source weights.
    pub fn load_cached(
        tag: &str,
        n: usize,
        k: usize,
        group: usize,
        half_scales: bool,
    ) -> Option<Self> {
        if super::v2::enabled() {
            return None;
        }
        let choice = super::isa::isa_choice();
        let (isa, path) = (choice.ternary, choice.int8);
        let groups = k / group;
        let weights = Store::<i8>::mapped(&format!("{tag}:w"), n * k)?;
        let scales = Store::<f32>::mapped(&format!("{tag}:s"), n * groups)?;
        Some(Self {
            isa,
            path,
            n,
            k,
            group,
            half_scales,
            weights,
            scales,
            v2: None,
        })
    }

    /// ggml Q8_0 weights as stored: row-major `n x k` codes and one scale
    /// per block of `Q8_0_BLOCK` (the f16 scale widened to f32). The codes
    /// are packed for this CPU's kernel, not changed.
    pub fn from_q8_0(n: usize, k: usize, codes: &[i8], block_scales: &[f32]) -> Self {
        Self::from_q8_0_named(None, n, k, codes, block_scales)
    }

    pub fn from_q8_0_named(
        tag: Option<&str>,
        n: usize,
        k: usize,
        codes: &[i8],
        block_scales: &[f32],
    ) -> Self {
        let choice = super::isa::isa_choice();
        let (isa, path) = (choice.ternary, choice.int8);
        if super::v2::enabled() {
            let layer = super::v2::product::IntProduct::int8_from_q8_0(n, k, codes, block_scales);
            return Self::second(isa, path, layer);
        }
        let rows = path.rows();
        let groups = k / Q8_0_BLOCK;
        let mut weights = vec![0i8; n * k];
        let mut scales = vec![0.0f32; n * groups];
        for row in 0..n {
            let (block, r) = (row / rows, row % rows);
            for g in 0..groups {
                scales[(block * groups + g) * rows + r] = block_scales[row * groups + g];
            }
            for kk in 0..k {
                let code = codes[row * k + kk];
                let code = if path.flipped() { code ^ i8::MIN } else { code };
                weights[path.at(k, row, kk)] = code;
            }
        }
        let (weights, scales) = match tag {
            Some(tag) => (
                Store::named(&format!("{tag}:w"), weights),
                Store::named(&format!("{tag}:s"), scales),
            ),
            None => (Store::weights(weights), Store::weights(scales)),
        };
        Self {
            isa,
            path,
            n,
            k,
            group: Q8_0_BLOCK,
            half_scales: true,
            weights,
            scales,
            v2: None,
        }
    }

    /// Quantizes row-major `n x k` weights; `row(r, out)` writes row `r` as f32.
    pub fn quantize(n: usize, k: usize, row: impl Fn(usize, &mut [f32]) + Sync) -> Self {
        Self::quantize_named(None, n, k, row)
    }

    pub fn quantize_named(
        tag: Option<&str>,
        n: usize,
        k: usize,
        row: impl Fn(usize, &mut [f32]) + Sync,
    ) -> Self {
        let choice = super::isa::isa_choice();
        let (isa, path) = (choice.ternary, choice.int8);
        if super::v2::enabled() {
            let layer = super::v2::product::IntProduct::int8_from_f32(n, k, row);
            return Self::second(isa, path, layer);
        }
        let rows = path.rows();
        let groups = k / GROUP;
        let mut weights = vec![0i8; n * k];
        let mut scales = vec![0.0f32; n * groups];
        let (all_w, all_s) = (
            team::SendPtr::new(&mut weights),
            team::SendPtr::new(&mut scales),
        );
        team::for_each(n / rows, |block| {
            let mut values = vec![0.0f32; k];
            for r in 0..rows {
                let out_row = block * rows + r;
                row(out_row, &mut values);
                #[cfg(target_arch = "aarch64")]
                {
                    // A row with a NaN or an infinity takes the scalar path below.
                    if values.iter().all(|v| v.is_finite()) {
                        let mut codes = [0i8; GROUP];
                        for g in 0..groups {
                            let span = &values[g * GROUP..(g + 1) * GROUP];
                            let scale = quantize_group_neon(span, &mut codes);
                            // SAFETY: each block belongs to one task; the buffers outlive the step.
                            unsafe {
                                *all_s
                                    .slice_mut((block * groups + g) * rows + r, 1)
                                    .as_mut_ptr() = scale
                            };
                            // Every arm64 layout keeps 4 inputs of a row together.
                            for (i, four) in codes.as_chunks::<4>().0.iter().enumerate() {
                                let at = path.at(k, out_row, g * GROUP + 4 * i);
                                // SAFETY: as above.
                                unsafe { all_w.slice_mut(at, 4) }.copy_from_slice(four);
                            }
                        }
                        continue;
                    }
                }
                for g in 0..groups {
                    let span = &values[g * GROUP..(g + 1) * GROUP];
                    let max = span.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let scale = group_scale(max);
                    // SAFETY: each block belongs to one task; the buffers outlive the step.
                    unsafe {
                        *all_s
                            .slice_mut((block * groups + g) * rows + r, 1)
                            .as_mut_ptr() = scale
                    };
                    for (i, v) in span.iter().enumerate() {
                        let code = quantize_code(*v, scale);
                        // VNNI reads the byte as unsigned `w + 128`.
                        let code = if path.flipped() { code ^ i8::MIN } else { code };
                        let kk = g * GROUP + i;
                        let at = path.at(k, out_row, kk);
                        // SAFETY: as above.
                        unsafe { *all_w.slice_mut(at, 1).as_mut_ptr() = code };
                    }
                }
            }
        });
        let (weights, scales) = match tag {
            Some(tag) => (
                Store::named(&format!("{tag}:w"), weights),
                Store::named(&format!("{tag}:s"), scales),
            ),
            None => (Store::weights(weights), Store::weights(scales)),
        };
        Self {
            isa,
            path,
            n,
            k,
            group: GROUP,
            half_scales: false,
            weights,
            scales,
            v2: None,
        }
    }

    /// The weights as f32, one scale per group. The VNNI byte flip is undone.
    #[cfg(feature = "gpu")]
    pub(crate) fn dequant_f32(&self) -> Vec<f32> {
        let (n, k, group) = (self.n, self.k, self.group);
        let groups = k / group;
        let rows = self.path.rows();
        let flipped = self.path.flipped();
        let mut out = vec![0.0f32; n * k];
        if let Some(layer) = &self.v2 {
            for (at, value) in out.iter_mut().enumerate() {
                let (row, kk) = (at / k, at % k);
                *value = f32::from(layer.int8_code(row, kk)) * layer.scale(row, kk / group);
            }
            return out;
        }
        for block in 0..n / rows {
            for r in 0..rows {
                let row = block * rows + r;
                for g in 0..groups {
                    let scale = self.scales[(block * groups + g) * rows + r];
                    for i in 0..group {
                        let kk = g * group + i;
                        let mut code = self.weights[self.path.at(k, row, kk)];
                        if flipped {
                            code ^= i8::MIN;
                        }
                        out[row * k + kk] = f32::from(code) * scale;
                    }
                }
            }
        }
        out
    }

    /// `x` is `m x k` f32; returns `m x n`.
    pub fn forward(&self, x: &[f32], m: usize) -> Buf {
        self.forward_biased(x, m, None)
    }

    /// True when `forward_biased` adds a bias itself: every path of the first
    /// kernel set.
    pub fn adds_bias(&self) -> bool {
        self.v2.is_none()
    }

    /// `forward` with `bias` (`n` values) added to every output row. The
    /// tile adds it to a sum before it stores the sum, so each output is
    /// written once: a pass over the output to add the bias moved up to
    /// 9 MB through memory again (a Whisper encoder layer has 1,500 rows)
    /// and took 7 % of a Whisper tiny clip. Only when `adds_bias()`.
    pub fn forward_biased(&self, x: &[f32], m: usize, bias: Option<&[f32]>) -> Buf {
        let _span = crate::prof::span(crate::prof::Stage::Int8);
        let _product = crate::prof::product(crate::prof::Product::Int8, self.n, self.k, m);
        assert!(bias.is_none_or(|bias| self.adds_bias() && bias.len() == self.n));
        if let Some(layer) = &self.v2 {
            return layer.forward(x, m);
        }
        let mut out = Buf::scratch(m * self.n);
        let q = Quantized::grouped(self.isa, x, m, self.k, self.group, self.half_scales);
        let ptr = OutPtr(out.as_mut_ptr());
        let rows = self.path.rows();
        // The multiply-adds of one block of `rows` outputs over the frames.
        let block_work = q.m * self.k * rows;
        let inline = inline_work(self.path);
        match self.path {
            Path::Scalar => grouped_with(self.n, q.m * self.k, inline, |row| {
                self.scalar_row(&q, row, bias, ptr)
            }),
            #[cfg(target_arch = "x86_64")]
            Path::Vnni512 => dispatch_2d_with(
                self.n / (2 * rows),
                2 * block_work,
                q.m,
                6,
                inline,
                |unit, f| {
                    // SAFETY: `isa_choice` confirmed AVX-512 VNNI; each (unit, f) writes disjoint outputs.
                    unsafe { self.vnni_task(&q, unit, f, bias, ptr) }
                },
            ),
            #[cfg(target_arch = "x86_64")]
            Path::Vnni256Vex => dispatch_2d_with(
                self.n / (2 * rows),
                2 * block_work,
                q.m,
                4,
                inline,
                |unit, f| {
                    // SAFETY: `isa_choice` confirmed AVX2 + FMA + AVX-VNNI.
                    unsafe { self.vnni256_vex(&q, unit, f, bias, ptr) }
                },
            ),
            #[cfg(target_arch = "x86_64")]
            Path::Vnni256Evex => dispatch_2d_with(
                self.n / (2 * rows),
                2 * block_work,
                q.m,
                4,
                inline,
                |unit, f| {
                    // SAFETY: `isa_choice` confirmed AVX2 + FMA + AVX-512F/BW/VL/VNNI.
                    unsafe { self.vnni256_evex(&q, unit, f, bias, ptr) }
                },
            ),
            #[cfg(target_arch = "x86_64")]
            Path::Avx2 => dispatch_2d_with(
                self.n / (2 * rows),
                2 * block_work,
                q.m,
                4,
                inline,
                |unit, f| {
                    // SAFETY: `isa_choice` confirmed AVX2 + FMA.
                    unsafe { self.avx2_task(&q, unit, f, bias, ptr) }
                },
            ),
            #[cfg(target_arch = "aarch64")]
            Path::Neon => {
                let tile = if self.group >= 128 { 8 } else { 4 };
                dispatch_2d_with(self.n / rows, block_work, q.m, tile, inline, |block, f| {
                    // SAFETY: NEON is part of every arm64 CPU.
                    unsafe { self.neon_task::<neon::Smull>(&q, block, f, bias, ptr) }
                });
            }
            #[cfg(target_arch = "aarch64")]
            Path::NeonDot => {
                let tile = if self.group >= 128 { 8 } else { 4 };
                dispatch_2d_with(self.n / rows, block_work, q.m, tile, inline, |block, f| {
                    // SAFETY: `isa_choice` confirmed the arm64 dotprod feature.
                    unsafe { self.sdot_task(&q, block, f, bias, ptr) }
                });
            }
            #[cfg(target_arch = "aarch64")]
            Path::NeonMmla => {
                let x = paired(&q, self.k, self.k / self.group);
                // A tile reads whole frame pairs: an odd last frame costs a pair.
                let work = x.pairs * 2 * self.k * rows;
                let tile = if self.group >= 128 { 4 } else { 2 };
                dispatch_2d_with(
                    self.n / rows,
                    work,
                    x.pairs,
                    tile,
                    inline,
                    |block, pairs| {
                        // SAFETY: `isa_choice` confirmed arm64 i8mm; each (block, pairs) writes disjoint outputs.
                        unsafe { self.mmla_task(&x, q.m, block, pairs, bias, ptr) }
                    },
                );
            }
        }
        out
    }

    fn scalar_row(&self, q: &Quantized, row: usize, bias: Option<&[f32]>, out: OutPtr) {
        let (k, group) = (self.k, self.group);
        let groups = k / group;
        let shift = bias.map(|b| b[row]);
        for frame in 0..q.m {
            let mut total = 0.0f32;
            for g in 0..groups {
                let mut dot = 0i32;
                for kk in g * group..(g + 1) * group {
                    dot +=
                        i32::from(self.weights[row * k + kk]) * i32::from(q.values[frame * k + kk]);
                }
                total += dot as f32 * self.scales[row * groups + g] * q.scales[frame * groups + g];
            }
            if let Some(shift) = shift {
                total += shift;
            }
            // SAFETY: each output element belongs to one task.
            unsafe { *out.0.add(frame * self.n + row) = total };
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn vnni_task(
        &self,
        q: &Quantized,
        task: usize,
        frames: std::ops::Range<usize>,
        bias: Option<&[f32]>,
        out: OutPtr,
    ) {
        use std::arch::x86_64::__m512;
        // Int8 weights are 32 * k bytes per task (64 KB at k = 2048, 128 KB at k = 4096),
        // overflowing a 32 KB L1d on every 6-frame tile. K-blocking in 512-byte chunks
        // (16 KB weights + 3 KB x + 6 KB stack acc = 25 KB < 32 KB L1d) keeps weights L1d-hot
        // across 8 micro-tiles while preserving the exact left-to-right group accumulation order.
        const K_BYTES: usize = 512;
        const STRIP: usize = 48;
        let groups = self.k / self.group;
        let k_groups = (K_BYTES / self.group).max(1);
        if self.k >= 1024 && groups > k_groups && frames.end - frames.start > 6 {
            let mut buf = std::mem::MaybeUninit::<[[__m512; STRIP]; 2]>::uninit();
            let base0 = buf.as_mut_ptr().cast::<__m512>();
            let base1 = base0.add(STRIP);
            macro_rules! run_strip {
                ($first:literal, $last:literal, $strip_start:expr, $strip_len:expr, $g0:expr, $g1:expr) => {{
                    let mut off = 0;
                    while off + 6 <= $strip_len {
                        self.vnni_tile_k::<6, $first, $last>(
                            q,
                            task,
                            $strip_start + off,
                            $g0..$g1,
                            base0.add(off),
                            base1.add(off),
                            bias,
                            out,
                        );
                        off += 6;
                    }
                    match $strip_len - off {
                        1 => self.vnni_tile_k::<1, $first, $last>(
                            q,
                            task,
                            $strip_start + off,
                            $g0..$g1,
                            base0.add(off),
                            base1.add(off),
                            bias,
                            out,
                        ),
                        2 => self.vnni_tile_k::<2, $first, $last>(
                            q,
                            task,
                            $strip_start + off,
                            $g0..$g1,
                            base0.add(off),
                            base1.add(off),
                            bias,
                            out,
                        ),
                        3 => self.vnni_tile_k::<3, $first, $last>(
                            q,
                            task,
                            $strip_start + off,
                            $g0..$g1,
                            base0.add(off),
                            base1.add(off),
                            bias,
                            out,
                        ),
                        4 => self.vnni_tile_k::<4, $first, $last>(
                            q,
                            task,
                            $strip_start + off,
                            $g0..$g1,
                            base0.add(off),
                            base1.add(off),
                            bias,
                            out,
                        ),
                        5 => self.vnni_tile_k::<5, $first, $last>(
                            q,
                            task,
                            $strip_start + off,
                            $g0..$g1,
                            base0.add(off),
                            base1.add(off),
                            bias,
                            out,
                        ),
                        _ => {}
                    }
                }};
            }
            let mut strip_start = frames.start;
            while strip_start < frames.end {
                let strip_len = (frames.end - strip_start).min(STRIP);
                let mut g0 = 0;
                while g0 < groups {
                    let g1 = (g0 + k_groups).min(groups);
                    match (g0 == 0, g1 == groups) {
                        (true, false) => run_strip!(true, false, strip_start, strip_len, g0, g1),
                        (false, false) => run_strip!(false, false, strip_start, strip_len, g0, g1),
                        (false, true) => run_strip!(false, true, strip_start, strip_len, g0, g1),
                        (true, true) => run_strip!(true, true, strip_start, strip_len, g0, g1),
                    }
                    g0 = g1;
                }
                strip_start += strip_len;
            }
            return;
        }
        let mut frame = frames.start;
        while frame + 6 <= frames.end {
            self.vnni_tile::<6>(q, task, frame, bias, out);
            frame += 6;
        }
        match frames.end - frame {
            1 => self.vnni_tile_m1(q, task, frame, bias, out),
            2 => self.vnni_tile::<2>(q, task, frame, bias, out),
            3 => self.vnni_tile::<3>(q, task, frame, bias, out),
            4 => self.vnni_tile::<4>(q, task, frame, bias, out),
            5 => self.vnni_tile::<5>(q, task, frame, bias, out),
            _ => {}
        }
    }

    /// Single-frame fast path: 32 outputs with two independent `k4` chains per
    /// 16-row block so 4 `vpdpbusd` chains overlap across the 4-cycle latency.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn vnni_tile_m1(
        &self,
        q: &Quantized,
        task: usize,
        frame: usize,
        bias: Option<&[f32]>,
        out: OutPtr,
    ) {
        use std::arch::x86_64::*;
        let (k, k4s, groups) = (self.k, self.k / 4, self.k / self.group);
        let quads = self.group / 4;
        debug_assert!(quads.is_multiple_of(2));
        let blocks = [2 * task, 2 * task + 1];
        debug_assert!(frame < q.m);
        debug_assert!(q.values.len() >= (frame + 1) * k);
        debug_assert!(q.scales.len() >= (frame + 1) * groups);
        debug_assert!(q.sums.len() >= (frame + 1) * groups);
        let w0_base = self.weights.as_ptr().add(blocks[0] * k4s * 64);
        let w1_base = self.weights.as_ptr().add(blocks[1] * k4s * 64);
        let s0_base = self.scales.as_ptr().add(blocks[0] * groups * 16);
        let s1_base = self.scales.as_ptr().add(blocks[1] * groups * 16);
        let x_row = q.values.as_ptr().add(frame * k);
        let x_scales = q.scales.as_ptr().add(frame * groups);
        let x_sums = q.sums.as_ptr().add(frame * groups);
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        for g in 0..groups {
            // (w + 128) . x - 128 * sum(x) = -128 * sum(x) + (w + 128) . x
            let neg_bias = _mm512_set1_epi32((-128i32).wrapping_mul(*x_sums.add(g)));
            let (mut d0a, mut d0b) = (neg_bias, _mm512_setzero_si512());
            let (mut d1a, mut d1b) = (neg_bias, _mm512_setzero_si512());
            let mut k4 = g * quads;
            let k4_end = k4 + quads;
            while k4 + 2 <= k4_end {
                let w0a = _mm512_loadu_si512(w0_base.add(k4 * 64).cast());
                let w1a = _mm512_loadu_si512(w1_base.add(k4 * 64).cast());
                let xa =
                    _mm512_set1_epi32(std::ptr::read_unaligned(x_row.add(k4 * 4).cast::<i32>()));
                d0a = _mm512_dpbusd_epi32(d0a, w0a, xa);
                d1a = _mm512_dpbusd_epi32(d1a, w1a, xa);

                let w0b = _mm512_loadu_si512(w0_base.add((k4 + 1) * 64).cast());
                let w1b = _mm512_loadu_si512(w1_base.add((k4 + 1) * 64).cast());
                let xb = _mm512_set1_epi32(std::ptr::read_unaligned(
                    x_row.add((k4 + 1) * 4).cast::<i32>(),
                ));
                d0b = _mm512_dpbusd_epi32(d0b, w0b, xb);
                d1b = _mm512_dpbusd_epi32(d1b, w1b, xb);
                k4 += 2;
            }
            let d0 = _mm512_add_epi32(d0a, d0b);
            let d1 = _mm512_add_epi32(d1a, d1b);
            let xscale = _mm512_set1_ps(*x_scales.add(g));
            let wscale0 = _mm512_loadu_ps(s0_base.add(g * 16));
            let wscale1 = _mm512_loadu_ps(s1_base.add(g * 16));
            acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d0), _mm512_mul_ps(wscale0, xscale), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d1), _mm512_mul_ps(wscale1, xscale), acc1);
        }
        if let Some(bias) = bias {
            acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(bias.as_ptr().add(blocks[0] * 16)));
            acc1 = _mm512_add_ps(acc1, _mm512_loadu_ps(bias.as_ptr().add(blocks[1] * 16)));
        }
        let dst = out.0.add(frame * self.n);
        _mm512_storeu_ps(dst.add(blocks[0] * 16), acc0);
        _mm512_storeu_ps(dst.add(blocks[1] * 16), acc1);
    }

    /// 32 outputs (two 16-row blocks) by F frames over `g_range`, each plus its `bias` on `LAST`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    #[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
    unsafe fn vnni_tile_k<const F: usize, const FIRST: bool, const LAST: bool>(
        &self,
        q: &Quantized,
        task: usize,
        frame0: usize,
        g_range: std::ops::Range<usize>,
        buf0: *mut std::arch::x86_64::__m512,
        buf1: *mut std::arch::x86_64::__m512,
        bias: Option<&[f32]>,
        out: OutPtr,
    ) {
        use std::arch::x86_64::*;
        let (k, k4s, groups) = (self.k, self.k / 4, self.k / self.group);
        let quads = self.group / 4;
        let blocks = [2 * task, 2 * task + 1];
        debug_assert!(frame0 + F <= q.m);
        debug_assert!(q.values.len() >= (frame0 + F) * k);
        debug_assert!(q.scales.len() >= (frame0 + F) * groups);
        debug_assert!(q.sums.len() >= (frame0 + F) * groups);
        let w0_base = self.weights.as_ptr().add(blocks[0] * k4s * 64);
        let w1_base = self.weights.as_ptr().add(blocks[1] * k4s * 64);
        let s0_base = self.scales.as_ptr().add(blocks[0] * groups * 16);
        let s1_base = self.scales.as_ptr().add(blocks[1] * groups * 16);
        let x_base = q.values.as_ptr().add(frame0 * k);
        let x_scales = q.scales.as_ptr().add(frame0 * groups);
        let x_sums = q.sums.as_ptr().add(frame0 * groups);
        let mut acc = [[_mm512_setzero_ps(); F]; 2];
        if !FIRST {
            for f in 0..F {
                acc[0][f] = *buf0.add(f);
                acc[1][f] = *buf1.add(f);
            }
        }
        for g in g_range {
            // Start each dot product at -128 * sum(x) so (w + 128) . x gives w . x
            // directly without a post-loop subtraction.
            let mut dots = [[_mm512_setzero_si512(); F]; 2];
            for f in 0..F {
                let neg_bias =
                    _mm512_set1_epi32((-128i32).wrapping_mul(*x_sums.add(f * groups + g)));
                dots[0][f] = neg_bias;
                dots[1][f] = neg_bias;
            }
            for k4 in g * quads..(g + 1) * quads {
                let w0 = _mm512_loadu_si512(w0_base.add(k4 * 64).cast());
                let w1 = _mm512_loadu_si512(w1_base.add(k4 * 64).cast());
                let off = k4 * 4;
                for f in 0..F {
                    let x = _mm512_set1_epi32(std::ptr::read_unaligned(
                        x_base.add(f * k + off).cast::<i32>(),
                    ));
                    dots[0][f] = _mm512_dpbusd_epi32(dots[0][f], w0, x);
                    dots[1][f] = _mm512_dpbusd_epi32(dots[1][f], w1, x);
                }
            }
            let wscale0 = _mm512_loadu_ps(s0_base.add(g * 16));
            let wscale1 = _mm512_loadu_ps(s1_base.add(g * 16));
            for f in 0..F {
                let xscale = _mm512_set1_ps(*x_scales.add(f * groups + g));
                let scale0 = _mm512_mul_ps(wscale0, xscale);
                let scale1 = _mm512_mul_ps(wscale1, xscale);
                acc[0][f] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dots[0][f]), scale0, acc[0][f]);
                acc[1][f] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dots[1][f]), scale1, acc[1][f]);
            }
        }
        if LAST {
            for (b, block) in blocks.iter().enumerate() {
                if let Some(bias) = bias {
                    // No bias is not a bias of zero: a sum of -0.0 stays -0.0.
                    let shift = _mm512_loadu_ps(bias.as_ptr().add(block * 16));
                    for sum in acc[b].iter_mut() {
                        *sum = _mm512_add_ps(*sum, shift);
                    }
                }
                for f in 0..F {
                    _mm512_storeu_ps(out.0.add((frame0 + f) * self.n + block * 16), acc[b][f]);
                }
            }
        } else {
            for f in 0..F {
                *buf0.add(f) = acc[0][f];
                *buf1.add(f) = acc[1][f];
            }
        }
    }

    /// 32 outputs (two 16-row blocks) by F frames, each plus its `bias`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn vnni_tile<const F: usize>(
        &self,
        q: &Quantized,
        task: usize,
        frame0: usize,
        bias: Option<&[f32]>,
        out: OutPtr,
    ) {
        self.vnni_tile_k::<F, true, true>(
            q,
            task,
            frame0,
            0..self.k / self.group,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            bias,
            out,
        );
    }

    #[cfg(target_arch = "x86_64")]
    define_vnni256!(
        vnni256_vex,
        vnni256_tile_vex_k,
        vnni256_tile_vex,
        vnni256_tile_vex_m1,
        "avx2,fma,avxvnni",
        std::arch::x86_64::_mm256_dpbusd_avx_epi32
    );

    #[cfg(target_arch = "x86_64")]
    define_vnni256!(
        vnni256_evex,
        vnni256_tile_evex_k,
        vnni256_tile_evex,
        vnni256_tile_evex_m1,
        "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni",
        std::arch::x86_64::_mm256_dpbusd_epi32
    );

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_task(
        &self,
        q: &Quantized,
        task: usize,
        frames: std::ops::Range<usize>,
        bias: Option<&[f32]>,
        out: OutPtr,
    ) {
        let mut frame = frames.start;
        while frame + 4 <= frames.end {
            self.avx2_tile::<4>(q, task, frame, bias, out);
            frame += 4;
        }
        match frames.end - frame {
            1 => self.avx2_tile::<1>(q, task, frame, bias, out),
            2 => self.avx2_tile::<2>(q, task, frame, bias, out),
            3 => self.avx2_tile::<3>(q, task, frame, bias, out),
            _ => {}
        }
    }

    /// 16 outputs (two 8-row blocks) by F frames: shares each broadcast `x` across
    /// both blocks (`8 dots + 1 ones + 4 w/mag + 1 x + 1 tmp = 15 <= 16 YMMs`).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn avx2_tile<const F: usize>(
        &self,
        q: &Quantized,
        task: usize,
        frame0: usize,
        bias: Option<&[f32]>,
        out: OutPtr,
    ) {
        use std::arch::x86_64::*;
        let (k, k4s, groups) = (self.k, self.k / 4, self.k / self.group);
        let quads = self.group / 4;
        let blocks = [2 * task, 2 * task + 1];
        let w0_base = self.weights.as_ptr().add(blocks[0] * k4s * 32);
        let w1_base = self.weights.as_ptr().add(blocks[1] * k4s * 32);
        let s0_base = self.scales.as_ptr().add(blocks[0] * groups * 8);
        let s1_base = self.scales.as_ptr().add(blocks[1] * groups * 8);
        let x_base = q.values.as_ptr().add(frame0 * k);
        let x_scales = q.scales.as_ptr().add(frame0 * groups);
        let ones = _mm256_set1_epi16(1);
        let mut acc = [[_mm256_setzero_ps(); F]; 2];
        for g in 0..groups {
            let mut dots = [[_mm256_setzero_si256(); F]; 2];
            for k4 in g * quads..(g + 1) * quads {
                let w0 = _mm256_loadu_si256(w0_base.add(k4 * 32).cast());
                let w1 = _mm256_loadu_si256(w1_base.add(k4 * 32).cast());
                let mag0 = _mm256_abs_epi8(w0);
                let mag1 = _mm256_abs_epi8(w1);
                let off = k4 * 4;
                for f in 0..F {
                    let x = _mm256_set1_epi32(std::ptr::read_unaligned(
                        x_base.add(f * k + off).cast::<i32>(),
                    ));
                    let p0 = _mm256_maddubs_epi16(mag0, _mm256_sign_epi8(x, w0));
                    let p1 = _mm256_maddubs_epi16(mag1, _mm256_sign_epi8(x, w1));
                    dots[0][f] = _mm256_add_epi32(dots[0][f], _mm256_madd_epi16(p0, ones));
                    dots[1][f] = _mm256_add_epi32(dots[1][f], _mm256_madd_epi16(p1, ones));
                }
            }
            let wscale0 = _mm256_loadu_ps(s0_base.add(g * 8));
            let wscale1 = _mm256_loadu_ps(s1_base.add(g * 8));
            for f in 0..F {
                let xscale = _mm256_set1_ps(*x_scales.add(f * groups + g));
                let scale0 = _mm256_mul_ps(wscale0, xscale);
                let scale1 = _mm256_mul_ps(wscale1, xscale);
                acc[0][f] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dots[0][f]), scale0, acc[0][f]);
                acc[1][f] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dots[1][f]), scale1, acc[1][f]);
            }
        }
        for (b, block) in blocks.iter().enumerate() {
            if let Some(bias) = bias {
                let shift = _mm256_loadu_ps(bias.as_ptr().add(block * 8));
                for sum in acc[b].iter_mut() {
                    *sum = _mm256_add_ps(*sum, shift);
                }
            }
            for f in 0..F {
                _mm256_storeu_ps(out.0.add((frame0 + f) * self.n + block * 8), acc[b][f]);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "dotprod")]
    unsafe fn sdot_task(
        &self,
        q: &Quantized,
        block: usize,
        frames: std::ops::Range<usize>,
        bias: Option<&[f32]>,
        out: OutPtr,
    ) {
        self.neon_task::<neon::Sdot>(q, block, frames, bias, out);
    }

    /// One 8-output block over `frames`: 8 frames per tile when `group >= 128`
    /// (`16 dots + 8 x + 2 w = 26 <= 32` NEON registers, halving weight loads),
    /// or 4 frames per tile for short Q8_0 (`group = 32`) groups so `acc` stays in registers.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon_task<D: neon::Dot>(
        &self,
        q: &Quantized,
        block: usize,
        frames: std::ops::Range<usize>,
        bias: Option<&[f32]>,
        out: OutPtr,
    ) {
        let mut frame = frames.start;
        if self.group >= 128 {
            while frame + 8 <= frames.end {
                self.neon_tile::<8, D>(q, block, frame, bias, out);
                frame += 8;
            }
            match frames.end - frame {
                1 => self.neon_tile::<1, D>(q, block, frame, bias, out),
                2 => self.neon_tile::<2, D>(q, block, frame, bias, out),
                3 => self.neon_tile::<3, D>(q, block, frame, bias, out),
                4 => self.neon_tile::<4, D>(q, block, frame, bias, out),
                5 => self.neon_tile::<5, D>(q, block, frame, bias, out),
                6 => self.neon_tile::<6, D>(q, block, frame, bias, out),
                7 => self.neon_tile::<7, D>(q, block, frame, bias, out),
                _ => {}
            }
            return;
        }
        while frame + 4 <= frames.end {
            self.neon_tile::<4, D>(q, block, frame, bias, out);
            frame += 4;
        }
        match frames.end - frame {
            1 => self.neon_tile::<1, D>(q, block, frame, bias, out),
            2 => self.neon_tile::<2, D>(q, block, frame, bias, out),
            3 => self.neon_tile::<3, D>(q, block, frame, bias, out),
            _ => {}
        }
    }

    /// 8 outputs (two 4-row halves) by F frames. One 16-byte load of a frame
    /// holds the inputs of four k4 steps; step `s` reads lane `s`. The f32
    /// steps are those of the AVX2 path, so both give the same values.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn neon_tile<const F: usize, D: neon::Dot>(
        &self,
        q: &Quantized,
        block: usize,
        frame0: usize,
        bias: Option<&[f32]>,
        out: OutPtr,
    ) {
        use std::arch::aarch64::*;
        /// Step `S` of a k16 span: both halves of the block, every frame.
        #[inline(always)]
        unsafe fn step<const S: i32, const F: usize, D: neon::Dot>(
            dots: &mut [[int32x4_t; 2]; F],
            w: *const i8,
            x: &[int8x16_t; F],
        ) {
            let w = [
                vld1q_s8(w.add(S as usize * 32)),
                vld1q_s8(w.add(S as usize * 32 + 16)),
            ];
            for f in 0..F {
                dots[f][0] = D::dot::<S>(dots[f][0], w[0], x[f]);
                dots[f][1] = D::dot::<S>(dots[f][1], w[1], x[f]);
            }
        }
        let (k, k4s, groups) = (self.k, self.k / 4, self.k / self.group);
        let spans = self.group / 16;
        let mut acc = [[vdupq_n_f32(0.0); 2]; F];
        for g in 0..groups {
            let mut dots = [[vdupq_n_s32(0); 2]; F];
            for k16 in g * spans..(g + 1) * spans {
                let mut x = [vdupq_n_s8(0); F];
                for f in 0..F {
                    x[f] = vld1q_s8(q.values.as_ptr().add((frame0 + f) * k + k16 * 16));
                }
                let w = self.weights.as_ptr().add((block * k4s + k16 * 4) * 32);
                step::<0, F, D>(&mut dots, w, &x);
                step::<1, F, D>(&mut dots, w, &x);
                step::<2, F, D>(&mut dots, w, &x);
                step::<3, F, D>(&mut dots, w, &x);
            }
            let scales = self.scales.as_ptr().add((block * groups + g) * 8);
            let wscale = [vld1q_f32(scales), vld1q_f32(scales.add(4))];
            for f in 0..F {
                let xscale = q.scales[(frame0 + f) * groups + g];
                for half in 0..2 {
                    let scale = vmulq_n_f32(wscale[half], xscale);
                    acc[f][half] = vfmaq_f32(acc[f][half], vcvtq_f32_s32(dots[f][half]), scale);
                }
            }
        }
        if let Some(bias) = bias {
            let b = bias.as_ptr().add(block * 8);
            let shift = [vld1q_f32(b), vld1q_f32(b.add(4))];
            for sum in acc.iter_mut() {
                sum[0] = vaddq_f32(sum[0], shift[0]);
                sum[1] = vaddq_f32(sum[1], shift[1]);
            }
        }
        for f in 0..F {
            let dst = out.0.add((frame0 + f) * self.n + block * 8);
            vst1q_f32(dst, acc[f][0]);
            vst1q_f32(dst.add(4), acc[f][1]);
        }
    }

    /// One 8-output block over `pairs`: 4 pairs (8 frames) per tile when `group >= 128`
    /// (`16 dots + 4 w + 1 x = 21 <= 32` NEON registers, halving weight loads on Apple M2-M5),
    /// or 2 pairs (4 frames) for short Q8_0 (`group = 32`) groups so `acc` stays in registers.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "i8mm")]
    unsafe fn mmla_task(
        &self,
        x: &Paired,
        m: usize,
        block: usize,
        pairs: std::ops::Range<usize>,
        bias: Option<&[f32]>,
        out: OutPtr,
    ) {
        let mut pair = pairs.start;
        if self.group >= 128 {
            while pair + 4 <= pairs.end {
                self.mmla_tile::<4>(x, m, block, pair, bias, out);
                pair += 4;
            }
            match pairs.end - pair {
                1 => self.mmla_tile::<1>(x, m, block, pair, bias, out),
                2 => self.mmla_tile::<2>(x, m, block, pair, bias, out),
                3 => self.mmla_tile::<3>(x, m, block, pair, bias, out),
                _ => {}
            }
            return;
        }
        while pair + 2 <= pairs.end {
            self.mmla_tile::<2>(x, m, block, pair, bias, out);
            pair += 2;
        }
        if pair < pairs.end {
            self.mmla_tile::<1>(x, m, block, pair, bias, out);
        }
    }

    /// 8 outputs (four row pairs) by `P` frame pairs from `pair0`. `smmla`
    /// takes 8 inputs of two frames and 8 of two rows, and adds the four dot
    /// products, `[f0 r0, f0 r1, f1 r0, f1 r1]`. A frame past `m` (the second
    /// of the last pair) reads as zeros and is not stored. The f32 steps are
    /// those of `neon_tile`, so both give the same values.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn mmla_tile<const P: usize>(
        &self,
        x: &Paired,
        m: usize,
        block: usize,
        pair0: usize,
        bias: Option<&[f32]>,
        out: OutPtr,
    ) {
        use std::arch::aarch64::*;
        let (k, k8s, groups) = (self.k, self.k / 8, self.k / self.group);
        let spans = self.group / 16;
        let codes = x.values.as_ptr().add(pair0 * 2 * k);
        let weights = self.weights.as_ptr().add(block * k8s * 64);
        let mut acc = [[vdupq_n_f32(0.0); 4]; P];
        for g in 0..groups {
            let mut dots = [[vdupq_n_s32(0); 4]; P];
            for k16 in g * spans..(g + 1) * spans {
                for half in 0..2 {
                    let w = weights.add((k16 * 2 + half) * 64);
                    let w = [
                        vld1q_s8(w),
                        vld1q_s8(w.add(16)),
                        vld1q_s8(w.add(32)),
                        vld1q_s8(w.add(48)),
                    ];
                    for p in 0..P {
                        let pair = vld1q_s8(codes.add(p * 2 * k + k16 * 32 + half * 16));
                        for r in 0..4 {
                            dots[p][r] = smmla(dots[p][r], pair, w[r]);
                        }
                    }
                }
            }
            let scales = self.scales.as_ptr().add((block * groups + g) * 8);
            let mut xs = [vdupq_n_f32(0.0); P];
            for p in 0..P {
                xs[p] = vld1q_f32(x.scales.as_ptr().add(((pair0 + p) * groups + g) * 4));
            }
            for r in 0..4 {
                // [w 2r, w 2r+1] twice.
                let ws = vreinterpretq_f32_u64(vdupq_n_u64(std::ptr::read_unaligned(
                    scales.add(2 * r).cast::<u64>(),
                )));
                for p in 0..P {
                    let scale = vmulq_f32(ws, xs[p]);
                    acc[p][r] = vfmaq_f32(acc[p][r], vcvtq_f32_s32(dots[p][r]), scale);
                }
            }
        }
        if let Some(bias) = bias {
            let b = bias.as_ptr().add(block * 8);
            for r in 0..4 {
                let shift = vreinterpretq_f32_u64(vdupq_n_u64(std::ptr::read_unaligned(
                    b.add(2 * r).cast::<u64>(),
                )));
                for p in 0..P {
                    acc[p][r] = vaddq_f32(acc[p][r], shift);
                }
            }
        }
        for p in 0..P {
            let first = 2 * (pair0 + p);
            for r in 0..4 {
                let column = block * 8 + 2 * r;
                vst1_f32(out.0.add(first * self.n + column), vget_low_f32(acc[p][r]));
                if first + 1 < m {
                    vst1_f32(
                        out.0.add((first + 1) * self.n + column),
                        vget_high_f32(acc[p][r]),
                    );
                }
            }
        }
    }
}

/// The quantized frames in pairs for `smmla`: for pair `p` and inputs
/// `8j..8j+8`, 16 bytes at `(p * k / 8 + j) * 16`, the codes of frame 2p
/// and then those of frame 2p + 1 (zeros past the last frame). Scales as
/// `[pair][group]` of `[x0, x0, x1, x1]`, the order of `smmla`'s results.
#[cfg(target_arch = "aarch64")]
pub(super) struct Paired {
    pub(super) pairs: usize,
    pub(super) values: Vec<i8>,
    pub(super) scales: Vec<f32>,
}

#[cfg(target_arch = "aarch64")]
type PairedBufs = (Vec<i8>, Vec<f32>);

#[cfg(target_arch = "aarch64")]
thread_local! {
    static PAIRED: std::cell::RefCell<PairedBufs> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

#[cfg(target_arch = "aarch64")]
impl Drop for Paired {
    fn drop(&mut self) {
        let bufs = (
            std::mem::take(&mut self.values),
            std::mem::take(&mut self.scales),
        );
        PAIRED.with(|cell| *cell.borrow_mut() = bufs);
    }
}

/// Below this total multiply-add count, run inline on the calling thread
/// rather than paying cross-thread synchronization overhead.
const INLINE_WORK: usize = 250_000;
/// Least work per task on small/medium decoder steps.
const MIN_DECODER_TASK_WORK: usize = 180_000;
/// Least work of a large product task, in multiply-adds of whole tiles. Keeps
/// task claims cheap on slower or arm64 cores while giving every thread plenty
/// of chunks on large encoders.
const MIN_TASK_WORK: usize = 800_000;

/// Scales the inline threshold with the kernel's throughput so fast SIMD paths
/// (e.g. AVX-512 VNNI running Whisper's 384x384 x 5 = 737K MACs in ~1.2 us)
/// do not pay cross-thread wake latency for sub-2 us products.
fn inline_work(path: Path) -> usize {
    match path {
        #[cfg(target_arch = "x86_64")]
        Path::Vnni512 => 1_000_000,
        #[cfg(target_arch = "x86_64")]
        Path::Vnni256Vex | Path::Vnni256Evex => 500_000,
        #[cfg(target_arch = "aarch64")]
        Path::NeonMmla => 500_000,
        _ => INLINE_WORK,
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Splits an int8/ternary product over `(unit, frame_span)` when column units
/// alone do not balance across the active team (or on heterogeneous CPUs that
/// need finer chunks for fast cores to steal dynamically); otherwise delegates
/// to `grouped_with`.
pub(super) fn dispatch_2d(
    units: usize,
    work_per_unit: usize,
    extent: usize,
    tile: usize,
    run: impl Fn(usize, std::ops::Range<usize>) + Sync,
) {
    dispatch_2d_with(units, work_per_unit, extent, tile, INLINE_WORK, run);
}

fn dispatch_2d_with(
    units: usize,
    work_per_unit: usize,
    extent: usize,
    tile: usize,
    inline_threshold: usize,
    run: impl Fn(usize, std::ops::Range<usize>) + Sync,
) {
    if units == 0 || extent == 0 {
        return;
    }
    let threads = team::threads().max(1);
    let hetero = team::is_heterogeneous();
    let total = units.saturating_mul(work_per_unit);
    let spans = if threads <= 1 || total <= inline_threshold {
        1
    } else if hetero {
        let target = 4 * threads;
        if units < target && extent >= 8 * tile {
            target.div_ceil(units).min(extent / (4 * tile)).max(1)
        } else {
            1
        }
    } else {
        let min_spans = threads / gcd(units, threads);
        if min_spans > 1 && units < 6 * threads && extent >= min_spans * (4 * tile) {
            if units * min_spans < 2 * threads && extent >= min_spans * (8 * tile) {
                min_spans * 2
            } else {
                min_spans
            }
        } else {
            1
        }
    };
    if spans > 1 {
        let tile_groups = extent.div_ceil(tile);
        team::for_each(units * spans, |task| {
            let (unit, span) = (task / spans, task % spans);
            let start = (span * tile_groups / spans) * tile;
            let end = (((span + 1) * tile_groups / spans) * tile).min(extent);
            if start < end {
                run(unit, start..end);
            }
        });
    } else {
        grouped_with(units, work_per_unit, inline_threshold, |unit| {
            run(unit, 0..extent)
        });
    }
}

/// Runs `run` for each of `units` units of `work` multiply-adds, running tiny
/// products inline and spreading larger products across threads (with finer
/// chunks on heterogeneous teams so fast cores steal work from slower cores).
fn grouped_with(units: usize, work: usize, inline_threshold: usize, run: impl Fn(usize) + Sync) {
    if units == 0 {
        return;
    }
    let threads = team::threads().max(1);
    let hetero = team::is_heterogeneous();
    let total = units.saturating_mul(work);
    let tasks = if threads == 1 || total <= inline_threshold {
        1
    } else if total < 8_000_000 {
        let cap = if hetero { 4 * threads } else { threads };
        total
            .div_ceil(MIN_DECODER_TASK_WORK)
            .min(cap)
            .clamp(1, units)
    } else {
        let waves = if hetero { 4 } else { 2 };
        let per_task = MIN_TASK_WORK
            .div_ceil(work.max(1))
            .clamp(1, (units / (waves * threads)).max(1));
        units.div_ceil(per_task).clamp(1, units)
    };
    team::for_each(tasks, |task| {
        let start = task * units / tasks;
        let end = (task + 1) * units / tasks;
        for unit in start..end {
            run(unit);
        }
    });
}

/// Frame pairs that one task of `paired` writes.
#[cfg(target_arch = "aarch64")]
const PAIRS_PER_TASK: usize = 4;

#[cfg(target_arch = "aarch64")]
pub(super) fn paired(q: &Quantized, k: usize, groups: usize) -> Paired {
    use std::arch::aarch64::*;
    let (m, pairs) = (q.m, q.m.div_ceil(2));
    // Reuse this thread's buffers: fresh ones would fault in new pages.
    let (mut values, mut scales) = PAIRED.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    if values.len() < pairs * 2 * k {
        values.resize(pairs * 2 * k, 0);
    }
    if scales.len() < pairs * groups * 4 {
        scales.resize(pairs * groups * 4, 0.0);
    }
    let (all_values, all_scales) = (
        team::SendPtr::new(&mut values),
        team::SendPtr::new(&mut scales),
    );
    team::for_each(pairs.div_ceil(PAIRS_PER_TASK), |task| {
        for p in task * PAIRS_PER_TASK..((task + 1) * PAIRS_PER_TASK).min(pairs) {
            let (f0, f1) = (2 * p, 2 * p + 1);
            // SAFETY: each pair belongs to one task; the buffers outlive the step.
            let (dst, scales) = unsafe {
                (
                    all_values.slice_mut(p * 2 * k, 2 * k),
                    all_scales.slice_mut(p * groups * 4, groups * 4),
                )
            };
            let a = &q.values[f0 * k..(f0 + 1) * k];
            let b = (f1 < m).then(|| &q.values[f1 * k..(f1 + 1) * k]);
            for j in 0..k / 16 {
                // SAFETY: NEON is part of every arm64 CPU; `a` and `b` hold k
                // bytes and `dst` 2k, so 16 bytes at 16j and 32 at 32j fit.
                unsafe {
                    let x0 = vreinterpretq_s64_s8(vld1q_s8(a.as_ptr().add(16 * j)));
                    let x1 = match b {
                        Some(b) => vreinterpretq_s64_s8(vld1q_s8(b.as_ptr().add(16 * j))),
                        None => vdupq_n_s64(0),
                    };
                    let at = dst.as_mut_ptr().add(32 * j);
                    vst1q_s8(at, vreinterpretq_s8_s64(vzip1q_s64(x0, x1)));
                    vst1q_s8(at.add(16), vreinterpretq_s8_s64(vzip2q_s64(x0, x1)));
                }
            }
            for g in 0..groups {
                let x0 = q.scales[f0 * groups + g];
                let x1 = if f1 < m {
                    q.scales[f1 * groups + g]
                } else {
                    0.0
                };
                scales[4 * g..4 * g + 4].copy_from_slice(&[x0, x0, x1, x1]);
            }
        }
    });
    Paired {
        pairs,
        values,
        scales,
    }
}

/// `acc + a * b^T` for 2 x 8 byte matrices `a` and `b` (Armv8.6 `smmla`).
/// Inline assembly: the intrinsic is not stable yet. Callers run with the
/// `i8mm` feature on.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(super) unsafe fn smmla(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut acc = acc;
    std::arch::asm!(
        "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack, preserves_flags),
    );
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((*state >> 40) as u32) as f32 / ((1u32 << 24) as f32);
        bits * 2.0 - 1.0
    }

    #[allow(clippy::too_many_arguments)]
    fn ref_int8(
        x: &[f32],
        m: usize,
        k: usize,
        n: usize,
        group: usize,
        half_scales: bool,
        w_codes: &[i8],
        w_scales: &[f32],
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        let q = Quantized::grouped(Isa::Scalar, x, m, k, group, half_scales);
        let groups = k / group;
        let mut out = vec![0.0f32; m * n];
        for frame in 0..m {
            for row in 0..n {
                let mut total = 0.0f64;
                for g in 0..groups {
                    let mut dot = 0i32;
                    for i in 0..group {
                        let kk = g * group + i;
                        dot +=
                            i32::from(w_codes[row * k + kk]) * i32::from(q.values[frame * k + kk]);
                    }
                    total += f64::from(dot)
                        * f64::from(w_scales[row * groups + g])
                        * f64::from(q.scales[frame * groups + g]);
                }
                let mut v = total as f32;
                if let Some(b) = bias {
                    v += b[row];
                }
                out[frame * n + row] = v;
            }
        }
        out
    }

    #[test]
    fn int8_and_q8_0_match_reference_across_k_blocks_and_strips() {
        let mut seed = 0x7777_8888_9999_aaaau64;
        // k = 1024 and 2048 with m = 49 cross K_BYTES blocking and STRIP boundaries
        // (STRIP = 48 on AVX-512 VNNI, STRIP = 32 on AVX-VNNI/AVX2) plus m = 1 fast paths.
        for (n, k) in [(32usize, 128usize), (64, 1024), (32, 2048)] {
            let weights_f32: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
            let bias: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * 0.5).collect();

            // 1. Int8::quantize (128-value groups).
            let groups128 = k / GROUP;
            let mut w_codes = vec![0i8; n * k];
            let mut w_scales = vec![0.0f32; n * groups128];
            for row in 0..n {
                for g in 0..groups128 {
                    let span = &weights_f32[row * k + g * GROUP..][..GROUP];
                    let max = span.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let scale = group_scale(max);
                    w_scales[row * groups128 + g] = scale;
                    for (i, &v) in span.iter().enumerate() {
                        w_codes[row * k + g * GROUP + i] = quantize_code(v, scale);
                    }
                }
            }
            let layer128 = Int8::quantize(n, k, |r, dst| {
                dst.copy_from_slice(&weights_f32[r * k..(r + 1) * k]);
            });

            // 2. Int8::from_q8_0 (32-value groups, f16 activation scales).
            let groups32 = k / Q8_0_BLOCK;
            let q_codes: Vec<i8> = (0..n * k)
                .map(|_| (lcg(&mut seed) * 127.0).round().clamp(-127.0, 127.0) as i8)
                .collect();
            let q_scales: Vec<f32> = (0..n * groups32)
                .map(|_| lcg(&mut seed).abs() * 0.05 + 1e-3)
                .collect();
            let layer32 = Int8::from_q8_0(n, k, &q_codes, &q_scales);

            for &m in &[1usize, 2, 3, 4, 5, 6, 7, 11, 33, 49] {
                let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed) * 2.0).collect();
                let tol = (k as f32).sqrt() * 2e-4 + 1e-3;
                for use_bias in [false, true] {
                    let b_opt = use_bias.then_some(bias.as_slice());

                    let got128 = layer128.forward_biased(&x, m, b_opt);
                    let want128 = ref_int8(&x, m, k, n, GROUP, false, &w_codes, &w_scales, b_opt);
                    for (i, (&g, &w)) in got128.iter().zip(&want128).enumerate() {
                        assert!(
                            g.is_finite() && (g - w).abs() <= tol,
                            "int8 n={n} k={k} m={m} bias={use_bias} at {i}: got={g} want={w}"
                        );
                    }

                    let got32 = layer32.forward_biased(&x, m, b_opt);
                    let want32 =
                        ref_int8(&x, m, k, n, Q8_0_BLOCK, true, &q_codes, &q_scales, b_opt);
                    for (i, (&g, &w)) in got32.iter().zip(&want32).enumerate() {
                        assert!(
                            g.is_finite() && (g - w).abs() <= tol,
                            "q8_0 n={n} k={k} m={m} bias={use_bias} at {i}: got={g} want={w}"
                        );
                    }
                }
            }
        }
    }
}
