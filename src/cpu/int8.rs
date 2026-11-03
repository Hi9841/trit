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
enum Path {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// 8 outputs per block with 256-bit `vpdpbusd`: the VEX form (AVX-VNNI),
    /// or the EVEX form where AVX-512 VNNI runs it (`TRIT_ISA=avxvnni`, to
    /// check this path on such a CPU).
    #[cfg(target_arch = "x86_64")]
    Vnni256,
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
    fn detect(isa: Isa) -> Self {
        match isa {
            Isa::Scalar => Self::Scalar,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 if is_x86_feature_detected!("avxvnni") => Self::Vnni256,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => Self::Avx2,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni if std::env::var("TRIT_ISA").as_deref() == Ok("avxvnni") => {
                Self::Vnni256
            }
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni => Self::Vnni512,
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => Self::Neon,
            #[cfg(target_arch = "aarch64")]
            Isa::NeonDot
                if std::arch::is_aarch64_feature_detected!("i8mm")
                    && std::env::var("TRIT_ISA").as_deref() != Ok("dotprod") =>
            {
                Self::NeonMmla
            }
            #[cfg(target_arch = "aarch64")]
            Isa::NeonDot => Self::NeonDot,
        }
    }

    /// Outputs per packed block.
    fn rows(self) -> usize {
        match self {
            Self::Scalar => 1,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 | Self::Vnni256 => 8,
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
        return matches!(self, Self::Vnni256 | Self::Vnni512);
        #[cfg(not(target_arch = "x86_64"))]
        false
    }
}

/// One 8-output block of `Path::Vnni256` over every frame, 4 frames per
/// tile, with `$dpbusd` the VEX or the EVEX instruction (the same result).
#[cfg(target_arch = "x86_64")]
macro_rules! vnni256_task {
    ($self:ident, $q:ident, $block:ident, $out:ident, $dpbusd:ident) => {{
        use std::arch::x86_64::*;
        let (k, k4s, groups) = ($self.k, $self.k / 4, $self.k / $self.group);
        let quads = $self.group / 4;
        let mut frame0 = 0;
        while frame0 < $q.m {
            let frames = ($q.m - frame0).min(4);
            let mut acc = [_mm256_setzero_ps(); 4];
            for g in 0..groups {
                let mut dots = [_mm256_setzero_si256(); 4];
                for k4 in g * quads..(g + 1) * quads {
                    let w = _mm256_loadu_si256(
                        $self.weights.as_ptr().add(($block * k4s + k4) * 32).cast(),
                    );
                    for f in 0..frames {
                        let src = $q.values.as_ptr().add((frame0 + f) * k + k4 * 4);
                        let x = _mm256_set1_epi32(std::ptr::read_unaligned(src.cast::<i32>()));
                        dots[f] = $dpbusd(dots[f], w, x);
                    }
                }
                let wscale = _mm256_loadu_ps($self.scales.as_ptr().add(($block * groups + g) * 8));
                for f in 0..frames {
                    let index = (frame0 + f) * groups + g;
                    // (w + 128) . x = w . x + 128 * sum(x)
                    let corrected =
                        _mm256_sub_epi32(dots[f], _mm256_set1_epi32(128 * $q.sums[index]));
                    let scale = _mm256_mul_ps(wscale, _mm256_set1_ps($q.scales[index]));
                    acc[f] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(corrected), scale, acc[f]);
                }
            }
            for f in 0..frames {
                _mm256_storeu_ps($out.0.add((frame0 + f) * $self.n + $block * 8), acc[f]);
            }
            frame0 += frames;
        }
    }};
}

/// The int8 product path of this CPU, for `cpu_path`.
pub fn path_name() -> &'static str {
    match Path::detect(Isa::detect()) {
        Path::Scalar => "scalar",
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => "avx2",
        #[cfg(target_arch = "x86_64")]
        Path::Vnni256 => "vnni256",
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
}

/// Values per block of ggml's Q8_0.
pub const Q8_0_BLOCK: usize = 32;

impl Int8 {
    /// Whether a `n x k` layer fits the kernels: whole groups and whole tiles.
    pub fn fits(n: usize, k: usize) -> bool {
        Self::fits_group(n, k, GROUP)
    }

    /// `fits` for groups of `group` inputs.
    pub fn fits_group(n: usize, k: usize, group: usize) -> bool {
        k.is_multiple_of(group) && n.is_multiple_of(32)
    }

    /// ggml Q8_0 weights as stored: row-major `n x k` codes and one scale
    /// per block of `Q8_0_BLOCK` (the f16 scale widened to f32). The codes
    /// are packed for this CPU's kernel, not changed.
    pub fn from_q8_0(n: usize, k: usize, codes: &[i8], block_scales: &[f32]) -> Self {
        let isa = Isa::detect();
        let path = Path::detect(isa);
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
        Self {
            isa,
            path,
            n,
            k,
            group: Q8_0_BLOCK,
            half_scales: true,
            weights: Store::weights(weights),
            scales: Store::weights(scales),
        }
    }

    /// Quantizes row-major `n x k` weights; `row(r, out)` writes row `r` as f32.
    pub fn quantize(n: usize, k: usize, row: impl Fn(usize, &mut [f32]) + Sync) -> Self {
        let isa = Isa::detect();
        let path = Path::detect(isa);
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
                for g in 0..groups {
                    let span = &values[g * GROUP..(g + 1) * GROUP];
                    let max = span.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
                    // SAFETY: each block belongs to one task; the buffers outlive the step.
                    unsafe {
                        *all_s
                            .slice_mut((block * groups + g) * rows + r, 1)
                            .as_mut_ptr() = scale
                    };
                    for (i, v) in span.iter().enumerate() {
                        let code = (v / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
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
        Self {
            isa,
            path,
            n,
            k,
            group: GROUP,
            half_scales: false,
            weights: Store::weights(weights),
            scales: Store::weights(scales),
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
        let mut out = Buf::scratch(m * self.n);
        let q = Quantized::grouped(self.isa, x, m, self.k, self.group, self.half_scales);
        let ptr = OutPtr(out.as_mut_ptr());
        let rows = self.path.rows();
        match self.path {
            Path::Scalar => team::for_each(self.n, |row| self.scalar_row(&q, row, ptr)),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: Path::detect confirmed the CPU features.
            Path::Vnni512 => team::for_each(self.n / (2 * rows), |task| unsafe {
                self.vnni_task(&q, task, ptr)
            }),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: Path::detect confirmed the CPU features: AVX-VNNI, or
            // AVX-512 VNNI (with VL), which runs the EVEX form.
            Path::Vnni256 => team::for_each(self.n / rows, |block| unsafe {
                if is_x86_feature_detected!("avxvnni") {
                    self.vnni256_vex(&q, block, ptr)
                } else {
                    self.vnni256_evex(&q, block, ptr)
                }
            }),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: Isa::detect confirmed the CPU features.
            Path::Avx2 => team::for_each(self.n / rows, |block| unsafe {
                self.avx2_task(&q, block, ptr)
            }),
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is part of every arm64 CPU.
            Path::Neon => team::for_each(self.n / rows, |block| unsafe {
                self.neon_task::<neon::Smull>(&q, block, ptr)
            }),
            #[cfg(target_arch = "aarch64")]
            // SAFETY: Isa::detect confirmed the dot product feature.
            Path::NeonDot => team::for_each(self.n / rows, |block| unsafe {
                self.sdot_task(&q, block, ptr)
            }),
            #[cfg(target_arch = "aarch64")]
            // SAFETY: Path::detect confirmed the int8 matrix multiply feature.
            Path::NeonMmla => team::for_each(self.n / rows, |block| unsafe {
                self.mmla_task(&q, block, ptr)
            }),
        }
        out
    }

    fn scalar_row(&self, q: &Quantized, row: usize, out: OutPtr) {
        let (k, group) = (self.k, self.group);
        let groups = k / group;
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
            // SAFETY: each output element belongs to one task.
            unsafe { *out.0.add(frame * self.n + row) = total };
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn vnni_task(&self, q: &Quantized, task: usize, out: OutPtr) {
        let mut frame = 0;
        while frame + 6 <= q.m {
            self.vnni_tile::<6>(q, task, frame, out);
            frame += 6;
        }
        match q.m - frame {
            1 => self.vnni_tile::<1>(q, task, frame, out),
            2 => self.vnni_tile::<2>(q, task, frame, out),
            3 => self.vnni_tile::<3>(q, task, frame, out),
            4 => self.vnni_tile::<4>(q, task, frame, out),
            5 => self.vnni_tile::<5>(q, task, frame, out),
            _ => {}
        }
    }

    /// 32 outputs (two 16-row blocks) by F frames.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn vnni_tile<const F: usize>(
        &self,
        q: &Quantized,
        task: usize,
        frame0: usize,
        out: OutPtr,
    ) {
        use std::arch::x86_64::*;
        let (k, k4s, groups) = (self.k, self.k / 4, self.k / self.group);
        let quads = self.group / 4;
        let blocks = [2 * task, 2 * task + 1];
        let mut acc = [[_mm512_setzero_ps(); F]; 2];
        for g in 0..groups {
            let mut dots = [[_mm512_setzero_si512(); F]; 2];
            for k4 in g * quads..(g + 1) * quads {
                let w0 = _mm512_loadu_si512(
                    self.weights
                        .as_ptr()
                        .add((blocks[0] * k4s + k4) * 64)
                        .cast(),
                );
                let w1 = _mm512_loadu_si512(
                    self.weights
                        .as_ptr()
                        .add((blocks[1] * k4s + k4) * 64)
                        .cast(),
                );
                for f in 0..F {
                    let src = q.values.as_ptr().add((frame0 + f) * k + k4 * 4);
                    let x = _mm512_set1_epi32(std::ptr::read_unaligned(src.cast::<i32>()));
                    dots[0][f] = _mm512_dpbusd_epi32(dots[0][f], w0, x);
                    dots[1][f] = _mm512_dpbusd_epi32(dots[1][f], w1, x);
                }
            }
            for (b, block) in blocks.iter().enumerate() {
                let wscale = _mm512_loadu_ps(self.scales.as_ptr().add((block * groups + g) * 16));
                for f in 0..F {
                    let index = (frame0 + f) * groups + g;
                    // (w + 128) . x = w . x + 128 * sum(x)
                    let corrected =
                        _mm512_sub_epi32(dots[b][f], _mm512_set1_epi32(128 * q.sums[index]));
                    let scale = _mm512_mul_ps(wscale, _mm512_set1_ps(q.scales[index]));
                    acc[b][f] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(corrected), scale, acc[b][f]);
                }
            }
        }
        for (b, block) in blocks.iter().enumerate() {
            for f in 0..F {
                _mm512_storeu_ps(out.0.add((frame0 + f) * self.n + block * 16), acc[b][f]);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,avxvnni")]
    unsafe fn vnni256_vex(&self, q: &Quantized, block: usize, out: OutPtr) {
        use std::arch::x86_64::_mm256_dpbusd_avx_epi32 as dpbusd;
        vnni256_task!(self, q, block, out, dpbusd);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,avx512vl,avx512vnni")]
    unsafe fn vnni256_evex(&self, q: &Quantized, block: usize, out: OutPtr) {
        use std::arch::x86_64::_mm256_dpbusd_epi32 as dpbusd;
        vnni256_task!(self, q, block, out, dpbusd);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_task(&self, q: &Quantized, block: usize, out: OutPtr) {
        let mut frame = 0;
        while frame + 4 <= q.m {
            self.avx2_tile::<4>(q, block, frame, out);
            frame += 4;
        }
        match q.m - frame {
            1 => self.avx2_tile::<1>(q, block, frame, out),
            2 => self.avx2_tile::<2>(q, block, frame, out),
            3 => self.avx2_tile::<3>(q, block, frame, out),
            _ => {}
        }
    }

    /// 8 outputs by F frames.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn avx2_tile<const F: usize>(
        &self,
        q: &Quantized,
        block: usize,
        frame0: usize,
        out: OutPtr,
    ) {
        use std::arch::x86_64::*;
        let (k, k4s, groups) = (self.k, self.k / 4, self.k / self.group);
        let quads = self.group / 4;
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); F];
        for g in 0..groups {
            let mut dots = [_mm256_setzero_si256(); F];
            for k4 in g * quads..(g + 1) * quads {
                let w =
                    _mm256_loadu_si256(self.weights.as_ptr().add((block * k4s + k4) * 32).cast());
                let magnitude = _mm256_abs_epi8(w);
                for f in 0..F {
                    let src = q.values.as_ptr().add((frame0 + f) * k + k4 * 4);
                    let x = _mm256_set1_epi32(std::ptr::read_unaligned(src.cast::<i32>()));
                    let pairs = _mm256_maddubs_epi16(magnitude, _mm256_sign_epi8(x, w));
                    dots[f] = _mm256_add_epi32(dots[f], _mm256_madd_epi16(pairs, ones));
                }
            }
            let wscale = _mm256_loadu_ps(self.scales.as_ptr().add((block * groups + g) * 8));
            for f in 0..F {
                let scale =
                    _mm256_mul_ps(wscale, _mm256_set1_ps(q.scales[(frame0 + f) * groups + g]));
                acc[f] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dots[f]), scale, acc[f]);
            }
        }
        for f in 0..F {
            _mm256_storeu_ps(out.0.add((frame0 + f) * self.n + block * 8), acc[f]);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "dotprod")]
    unsafe fn sdot_task(&self, q: &Quantized, block: usize, out: OutPtr) {
        self.neon_task::<neon::Sdot>(q, block, out);
    }

    /// One 8-output block over every frame, 4 frames per tile, as on AVX2.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon_task<D: neon::Dot>(&self, q: &Quantized, block: usize, out: OutPtr) {
        let mut frame = 0;
        while frame + 4 <= q.m {
            self.neon_tile::<4, D>(q, block, frame, out);
            frame += 4;
        }
        match q.m - frame {
            1 => self.neon_tile::<1, D>(q, block, frame, out),
            2 => self.neon_tile::<2, D>(q, block, frame, out),
            3 => self.neon_tile::<3, D>(q, block, frame, out),
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
        for f in 0..F {
            let dst = out.0.add((frame0 + f) * self.n + block * 8);
            vst1q_f32(dst, acc[f][0]);
            vst1q_f32(dst.add(4), acc[f][1]);
        }
    }

    /// One 8-output block over every frame, 4 frames (two pairs) per tile.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "i8mm")]
    unsafe fn mmla_task(&self, q: &Quantized, block: usize, out: OutPtr) {
        let mut frame = 0;
        while frame + 4 <= q.m {
            self.mmla_tile::<2>(q, block, frame, 4, out);
            frame += 4;
        }
        match q.m - frame {
            0 => {}
            frames @ (1 | 2) => self.mmla_tile::<1>(q, block, frame, frames, out),
            frames => self.mmla_tile::<2>(q, block, frame, frames, out),
        }
    }

    /// 8 outputs (four row pairs) by `frames` frames, at most `2 * P`, in `P`
    /// frame pairs. `smmla` takes 8 inputs of two frames and 8 of two rows,
    /// and adds the four dot products, `[f0 r0, f0 r1, f1 r0, f1 r1]`. A
    /// missing frame of the last pair reads as zeros and is not stored. The
    /// f32 steps are those of `neon_tile`, so both give the same values.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn mmla_tile<const P: usize>(
        &self,
        q: &Quantized,
        block: usize,
        frame0: usize,
        frames: usize,
        out: OutPtr,
    ) {
        use std::arch::aarch64::*;
        let (k, k8s, groups) = (self.k, self.k / 8, self.k / self.group);
        let spans = self.group / 16;
        let zero = vdupq_n_s8(0);
        let mut acc = [[vdupq_n_f32(0.0); 4]; P];
        for g in 0..groups {
            let mut dots = [[vdupq_n_s32(0); 4]; P];
            for k16 in g * spans..(g + 1) * spans {
                let load = |f: usize| {
                    if f < frames {
                        vld1q_s8(q.values.as_ptr().add((frame0 + f) * k + k16 * 16))
                    } else {
                        zero
                    }
                };
                // Pair p, half h: inputs 8h..8h+8 of frames 2p and 2p + 1.
                let mut x = [[zero; 2]; P];
                for p in 0..P {
                    let (a, b) = (
                        vreinterpretq_s64_s8(load(2 * p)),
                        vreinterpretq_s64_s8(load(2 * p + 1)),
                    );
                    x[p] = [
                        vreinterpretq_s8_s64(vzip1q_s64(a, b)),
                        vreinterpretq_s8_s64(vzip2q_s64(a, b)),
                    ];
                }
                for half in 0..2 {
                    let w = self
                        .weights
                        .as_ptr()
                        .add((block * k8s + k16 * 2 + half) * 64);
                    let w = [
                        vld1q_s8(w),
                        vld1q_s8(w.add(16)),
                        vld1q_s8(w.add(32)),
                        vld1q_s8(w.add(48)),
                    ];
                    for p in 0..P {
                        for r in 0..4 {
                            dots[p][r] = smmla(dots[p][r], x[p][half], w[r]);
                        }
                    }
                }
            }
            let scales = self.scales.as_ptr().add((block * groups + g) * 8);
            let xscale = |f: usize| {
                if f < frames {
                    q.scales[(frame0 + f) * groups + g]
                } else {
                    0.0
                }
            };
            for p in 0..P {
                let xs = vcombine_f32(vdup_n_f32(xscale(2 * p)), vdup_n_f32(xscale(2 * p + 1)));
                for r in 0..4 {
                    let ws = vld1_f32(scales.add(2 * r));
                    let scale = vmulq_f32(vcombine_f32(ws, ws), xs);
                    acc[p][r] = vfmaq_f32(acc[p][r], vcvtq_f32_s32(dots[p][r]), scale);
                }
            }
        }
        for p in 0..P {
            for r in 0..4 {
                let column = block * 8 + 2 * r;
                let first = frame0 + 2 * p;
                vst1_f32(out.0.add(first * self.n + column), vget_low_f32(acc[p][r]));
                if 2 * p + 1 < frames {
                    vst1_f32(
                        out.0.add((first + 1) * self.n + column),
                        vget_high_f32(acc[p][r]),
                    );
                }
            }
        }
    }
}

/// `acc + a * b^T` for 2 x 8 byte matrices `a` and `b` (Armv8.6 `smmla`).
/// Inline assembly: the intrinsic is not stable yet. Callers run with the
/// `i8mm` feature on.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn smmla(
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
