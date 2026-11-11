//! Ternary matrix product with 8-bit activations.
//!
//! Weights stay as codes 0, 1, 2 (`w = scale * (code - 1)`). Each 128-value
//! group of an activation row is quantized to i8 with its own scale. A group
//! dot product is then `sum(code * q) - sum(q)`, done in i32 by
//! `vpdpbusd` (AVX-512 VNNI), `vpmaddubsw` (AVX2), `sdot` (Arm dot product)
//! or `smull` (Arm NEON), and scaled into f32 by the weight scale and the
//! activation scale of that group.
//!
//! The SIMD paths keep four codes per byte. A short take reads every weight
//! once from memory, so at one byte per code the product waits on memory;
//! two shifts and a mask per weight vector cost less than the bytes they save.

use super::buf::Buf;
use super::store::Store;
use super::team;
use std::cell::RefCell;

pub const GROUP: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Isa {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512Vnni,
    /// Armv8.0 NEON: every arm64 CPU.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// Armv8.2 dot product (`sdot`): most phones since 2018.
    #[cfg(target_arch = "aarch64")]
    NeonDot,
}

impl Isa {
    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => "avx2",
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni => "avx512-vnni",
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => "neon",
            #[cfg(target_arch = "aarch64")]
            Isa::NeonDot => "neon-dotprod",
        }
    }

    pub(super) fn rows(self) -> usize {
        match self {
            Isa::Scalar => 1,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => 8,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni => 16,
            #[cfg(target_arch = "aarch64")]
            Isa::Neon | Isa::NeonDot => 8,
        }
    }

    /// The fastest path this CPU runs. `TRIT_ISA=scalar|avx2|neon` forces a slower one.
    pub fn detect() -> Self {
        super::isa::isa_choice().ternary
    }
}

/// Codes packed for one instruction set. Scalar: `[n][k / 4][1][4]`, one
/// code per byte. SIMD: `[n / rows][k / 16][rows][4]` bytes, where bits
/// `2s..2s+2` of a byte hold the code of k4 step `4 * (k / 16 index) + s`.
pub struct Ternary {
    pub isa: Isa,
    pub n: usize,
    pub k: usize,
    codes: Store<u8>,
    /// `[n / rows][k / GROUP][rows]`.
    scales: Store<f32>,
    /// The layer in the second kernel set (`TRIT_KERNELS=v2`); `codes` and
    /// `scales` are then empty.
    v2: Option<Box<super::v2::product::IntProduct>>,
    /// A GPU copy of the codes and scales, present when the load asked for the GPU.
    #[cfg(feature = "gpu")]
    gpu: Option<Box<crate::gpu::Layer>>,
}

impl Ternary {
    /// A layer as the model file stores it: `packed` holds 5 base-3 digits
    /// per byte, least significant first, `ceil(k / 5)` bytes per row, and
    /// `w = scale[row][col / GROUP] * (code - 1)`.
    pub fn from_base3(
        packed: &[u8],
        scales: &[f32],
        n: usize,
        k: usize,
        group: usize,
    ) -> Result<Self, String> {
        let row_bytes = k.div_ceil(5);
        if packed.len() != n * row_bytes
            || group != GROUP
            || !k.is_multiple_of(GROUP)
            || scales.len() != n * (k / GROUP)
        {
            return Err(format!("ternary layer {n}x{k} does not match its tensors"));
        }
        if let Some(row) = packed
            .chunks_exact(row_bytes)
            .position(|row| row.iter().any(|b| *b >= 243))
        {
            return Err(format!("ternary byte above 242 in row {row}"));
        }
        Ok(Self::from_rows(
            Isa::detect(),
            scales,
            n,
            k,
            |row, codes| {
                let packed_row = &packed[row * row_bytes..(row + 1) * row_bytes];
                // A copy of exactly 5 bytes for each file byte: a copy of a
                // length the compiler does not know was most of a load.
                let (fives, rest) = codes.as_chunks_mut::<5>();
                for (five, byte) in fives.iter_mut().zip(packed_row) {
                    *five = DIGITS[usize::from(*byte)];
                }
                if let Some(byte) = packed_row.get(fives.len()) {
                    rest.copy_from_slice(&DIGITS[usize::from(*byte)][..rest.len()]);
                }
            },
        ))
    }

    /// `codes` is row-major `n x k` (values 0..=2), `scales` is `n x (k / GROUP)`.
    /// `row(r, codes)` writes the `k` codes (values 0..=2) of row `r`;
    /// `scales` is `n x (k / GROUP)`. Rows come one block at a time, so the
    /// whole byte-per-code matrix never exists.
    pub fn from_rows(
        isa: Isa,
        scales: &[f32],
        n: usize,
        k: usize,
        row: impl Fn(usize, &mut [u8]),
    ) -> Self {
        if super::v2::enabled() && k.is_multiple_of(GROUP) {
            return Self {
                isa,
                n,
                k,
                codes: Store::owned(Vec::new()),
                scales: Store::owned(Vec::new()),
                v2: Some(Box::new(super::v2::product::IntProduct::ternary(
                    scales, n, k, GROUP, row,
                ))),
                #[cfg(feature = "gpu")]
                gpu: None,
            };
        }
        let rows = isa.rows();
        let isa = if k.is_multiple_of(GROUP) && n.is_multiple_of(2 * rows) {
            isa
        } else {
            Isa::Scalar
        };
        let rows = isa.rows();
        let groups = k / GROUP;
        // One row per block: the scalar layout is the row-major codes.
        let scalar = isa == Isa::Scalar;
        let mut packed = vec![0u8; if scalar { n * k } else { n * k / 4 }];
        let mut packed_scales = vec![0.0f32; n * groups];
        let mut codes = vec![0u8; rows * k];
        for block in 0..n / rows {
            for (r, codes) in codes.chunks_exact_mut(k).enumerate() {
                row(block * rows + r, codes);
            }
            if scalar {
                packed[block * k..(block + 1) * k].copy_from_slice(&codes);
            } else {
                for k16 in 0..k / 16 {
                    for r in 0..rows {
                        let src = &codes[r * k + k16 * 16..][..16];
                        let dst = ((block * (k / 16) + k16) * rows + r) * 4;
                        packed[dst..dst + 4].copy_from_slice(&pack_span(src));
                    }
                }
            }
            for g in 0..groups {
                for r in 0..rows {
                    packed_scales[(block * groups + g) * rows + r] =
                        scales[(block * rows + r) * groups + g];
                }
            }
        }
        Self {
            isa,
            n,
            k,
            codes: Store::weights(packed),
            scales: Store::weights(packed_scales),
            v2: None,
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_layer(&self) -> Option<&crate::gpu::Layer> {
        self.gpu.as_deref()
    }

    /// Uploads codes (16 per `u32`) and scales when `device` is `Gpu`.
    pub(crate) fn on_gpu(self, device: crate::Device) -> Result<Self, String> {
        #[cfg(feature = "gpu")]
        if device == crate::Device::Gpu {
            let (codes, scales) = self.gpu_pack();
            let uploaded = crate::gpu::Layer::ternary(self.n, self.k, &codes, &scales, None)?;
            let mut layer = self;
            layer.gpu = Some(Box::new(uploaded));
            return Ok(layer);
        }
        #[cfg(not(feature = "gpu"))]
        let _ = device;
        Ok(self)
    }

    /// Row-major codes, 16 two-bit values per `u32`, and row-major scales.
    #[cfg(feature = "gpu")]
    fn gpu_pack(&self) -> (Vec<u32>, Vec<f32>) {
        let rows = self.isa.rows();
        let groups = self.k / GROUP;
        let words = self.k / 16;
        let mut codes = vec![0u32; self.n * words];
        let mut scales = vec![0.0f32; self.n * groups];
        for row in 0..self.n {
            let block = row / rows;
            let r = row % rows;
            for group in 0..groups {
                scales[row * groups + group] = match &self.v2 {
                    Some(layer) => layer.scale(row, group),
                    None => self.scales[(block * groups + group) * rows + r],
                };
            }
            for word in 0..words {
                let mut packed = 0u32;
                for lane in 0..16 {
                    let code = u32::from(match &self.v2 {
                        Some(layer) => layer.ternary_code(row, word * 16 + lane),
                        None => self.code_at(block, r, word * 16 + lane),
                    });
                    packed |= code << (lane * 2);
                }
                codes[row * words + word] = packed;
            }
        }
        (codes, scales)
    }

    #[cfg(feature = "gpu")]
    fn code_at(&self, block: usize, r: usize, kk: usize) -> u8 {
        let rows = self.isa.rows();
        if self.isa == Isa::Scalar {
            return self.codes[block * self.k + kk];
        }
        let k16 = kk / 16;
        let lane = kk % 16;
        let byte = ((block * (self.k / 16) + k16) * rows + r) * 4 + lane % 4;
        (self.codes[byte] >> ((lane / 4) * 2)) & 3
    }

    /// `x` is `m x k` f32; returns `m x n`.
    pub fn forward(&self, x: &[f32], m: usize) -> Buf {
        #[cfg(feature = "gpu")]
        if let Some(layer) = &self.gpu {
            if !crate::gpu::host_products() && crate::gpu::usable() {
                match layer.forward(x, m, false) {
                    Ok(out) => {
                        crate::gpu::worked();
                        return out;
                    }
                    Err(error) => crate::gpu::failed(&error),
                }
            }
        }
        if let Some(layer) = &self.v2 {
            return layer.forward(x, m);
        }
        let mut out = Buf::scratch(m * self.n);
        let q = Quantized::new(self.isa, x, m, self.k);
        let ptr = OutPtr(out.as_mut_ptr());
        let rows = self.isa.rows();
        let tasks = self.n / (2 * rows).max(1);
        let tasks = if self.isa == Isa::Scalar {
            self.n
        } else {
            tasks
        };
        team::for_each(tasks, |task| match self.isa {
            Isa::Scalar => self.scalar_rows(&q, task, ptr),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: Isa::detect confirmed the CPU features.
            Isa::Avx2 => unsafe { self.avx2_task(&q, task, ptr) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: Isa::detect confirmed the CPU features.
            Isa::Avx512Vnni => unsafe { self.vnni_task(&q, task, ptr) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is part of every arm64 CPU.
            Isa::Neon => unsafe { self.neon_task::<neon::Smull>(&q, task, ptr) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: Isa::detect confirmed the dot product feature.
            Isa::NeonDot => unsafe { self.sdot_task(&q, task, ptr) },
        });
        out
    }

    fn scalar_rows(&self, q: &Quantized, row: usize, out: OutPtr) {
        let (k, groups) = (self.k, self.k / GROUP);
        let rows = self.isa.rows();
        let (block, r) = (row / rows, row % rows);
        for frame in 0..q.m {
            let mut total = 0.0f32;
            for g in 0..groups {
                let mut sum = 0i32;
                for kk in g * GROUP..(g + 1) * GROUP {
                    let code = self.codes[((block * (k / 4) + kk / 4) * rows + r) * 4 + kk % 4];
                    sum += i32::from(code) * i32::from(q.values[frame * k + kk]);
                }
                let dot = sum - q.sums[frame * groups + g];
                total += dot as f32
                    * self.scales[(block * groups + g) * rows + r]
                    * q.scales[frame * groups + g];
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
        let (k, k4s, groups) = (self.k, self.k / 4, self.k / GROUP);
        let k16s = k4s / 4;
        let blocks = [2 * task, 2 * task + 1];
        let low = _mm512_set1_epi8(3);
        let mut acc = [[_mm512_setzero_ps(); F]; 2];
        for g in 0..groups {
            let mut dots = [[_mm512_setzero_si512(); F]; 2];
            for quad in 0..GROUP / 16 {
                let k16 = g * (GROUP / 16) + quad;
                let z0 = _mm512_loadu_si512(
                    self.codes
                        .as_ptr()
                        .add((blocks[0] * k16s + k16) * 64)
                        .cast(),
                );
                let z1 = _mm512_loadu_si512(
                    self.codes
                        .as_ptr()
                        .add((blocks[1] * k16s + k16) * 64)
                        .cast(),
                );
                // A 16-bit shift is safe: the mask keeps the low two bits of each byte.
                let w0 = [
                    _mm512_and_si512(z0, low),
                    _mm512_and_si512(_mm512_srli_epi16::<2>(z0), low),
                    _mm512_and_si512(_mm512_srli_epi16::<4>(z0), low),
                    _mm512_and_si512(_mm512_srli_epi16::<6>(z0), low),
                ];
                let w1 = [
                    _mm512_and_si512(z1, low),
                    _mm512_and_si512(_mm512_srli_epi16::<2>(z1), low),
                    _mm512_and_si512(_mm512_srli_epi16::<4>(z1), low),
                    _mm512_and_si512(_mm512_srli_epi16::<6>(z1), low),
                ];
                for s in 0..4 {
                    let k4 = 4 * k16 + s;
                    for f in 0..F {
                        let src = q.values.as_ptr().add((frame0 + f) * k + k4 * 4);
                        let x = _mm512_set1_epi32(std::ptr::read_unaligned(src.cast::<i32>()));
                        dots[0][f] = _mm512_dpbusd_epi32(dots[0][f], w0[s], x);
                        dots[1][f] = _mm512_dpbusd_epi32(dots[1][f], w1[s], x);
                    }
                }
            }
            for (b, block) in blocks.iter().enumerate() {
                let wscale = _mm512_loadu_ps(self.scales.as_ptr().add((block * groups + g) * 16));
                for f in 0..F {
                    let index = (frame0 + f) * groups + g;
                    let corrected = _mm512_sub_epi32(dots[b][f], _mm512_set1_epi32(q.sums[index]));
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
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_task(&self, q: &Quantized, task: usize, out: OutPtr) {
        for block in [2 * task, 2 * task + 1] {
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
        let (k, k4s, groups) = (self.k, self.k / 4, self.k / GROUP);
        let k16s = k4s / 4;
        let ones = _mm256_set1_epi16(1);
        let low = _mm256_set1_epi8(3);
        let mut acc = [_mm256_setzero_ps(); F];
        for g in 0..groups {
            let mut dots = [_mm256_setzero_si256(); F];
            for quad in 0..GROUP / 16 {
                let k16 = g * (GROUP / 16) + quad;
                let z =
                    _mm256_loadu_si256(self.codes.as_ptr().add((block * k16s + k16) * 32).cast());
                // A 16-bit shift is safe: the mask keeps the low two bits of each byte.
                let ws = [
                    _mm256_and_si256(z, low),
                    _mm256_and_si256(_mm256_srli_epi16::<2>(z), low),
                    _mm256_and_si256(_mm256_srli_epi16::<4>(z), low),
                    _mm256_and_si256(_mm256_srli_epi16::<6>(z), low),
                ];
                for (s, w) in ws.iter().enumerate() {
                    let k4 = 4 * k16 + s;
                    for f in 0..F {
                        let src = q.values.as_ptr().add((frame0 + f) * k + k4 * 4);
                        let x = _mm256_set1_epi32(std::ptr::read_unaligned(src.cast::<i32>()));
                        // Codes are at most 2 and |q| at most 127: pair sums fit in i16.
                        let pairs = _mm256_maddubs_epi16(*w, x);
                        dots[f] = _mm256_add_epi32(dots[f], _mm256_madd_epi16(pairs, ones));
                    }
                }
            }
            let wscale = _mm256_loadu_ps(self.scales.as_ptr().add((block * groups + g) * 8));
            for f in 0..F {
                let index = (frame0 + f) * groups + g;
                let corrected = _mm256_sub_epi32(dots[f], _mm256_set1_epi32(q.sums[index]));
                let scale = _mm256_mul_ps(wscale, _mm256_set1_ps(q.scales[index]));
                acc[f] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(corrected), scale, acc[f]);
            }
        }
        for f in 0..F {
            _mm256_storeu_ps(out.0.add((frame0 + f) * self.n + block * 8), acc[f]);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "dotprod")]
    unsafe fn sdot_task(&self, q: &Quantized, task: usize, out: OutPtr) {
        self.neon_task::<neon::Sdot>(q, task, out);
    }

    /// Two 8-row blocks, 4 frames at a time, as on AVX2.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon_task<D: neon::Dot>(&self, q: &Quantized, task: usize, out: OutPtr) {
        for block in [2 * task, 2 * task + 1] {
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
    }

    /// 8 outputs (two 4-row halves) by F frames. One 16-byte load of a
    /// frame covers the four k4 steps of a k16 span; step `s` reads lane `s`.
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
        let (k, k16s, groups) = (self.k, self.k / 16, self.k / GROUP);
        let low = vdupq_n_u8(3);
        let mut acc = [[vdupq_n_f32(0.0); 2]; F];
        for g in 0..groups {
            let mut dots = [[vdupq_n_s32(0); 2]; F];
            for quad in 0..GROUP / 16 {
                let k16 = g * (GROUP / 16) + quad;
                let mut x = [vdupq_n_s8(0); F];
                for f in 0..F {
                    x[f] = vld1q_s8(q.values.as_ptr().add((frame0 + f) * k + k16 * 16));
                }
                let codes = self.codes.as_ptr().add((block * k16s + k16) * 32);
                for half in 0..2 {
                    let z = vld1q_u8(codes.add(half * 16));
                    let w = [
                        vreinterpretq_s8_u8(vandq_u8(z, low)),
                        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(z), low)),
                        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(z), low)),
                        vreinterpretq_s8_u8(vshrq_n_u8::<6>(z)),
                    ];
                    for f in 0..F {
                        let mut d = dots[f][half];
                        d = D::dot::<0>(d, w[0], x[f]);
                        d = D::dot::<1>(d, w[1], x[f]);
                        d = D::dot::<2>(d, w[2], x[f]);
                        d = D::dot::<3>(d, w[3], x[f]);
                        dots[f][half] = d;
                    }
                }
            }
            let scales = self.scales.as_ptr().add((block * groups + g) * 8);
            let wscale = [vld1q_f32(scales), vld1q_f32(scales.add(4))];
            for f in 0..F {
                let index = (frame0 + f) * groups + g;
                let sum = vdupq_n_s32(q.sums[index]);
                for half in 0..2 {
                    let corrected = vcvtq_f32_s32(vsubq_s32(dots[f][half], sum));
                    let scale = vmulq_n_f32(wscale[half], q.scales[index]);
                    acc[f][half] = vfmaq_f32(acc[f][half], corrected, scale);
                }
            }
        }
        for f in 0..F {
            let dst = out.0.add((frame0 + f) * self.n + block * 8);
            vst1q_f32(dst, acc[f][0]);
            vst1q_f32(dst.add(4), acc[f][1]);
        }
    }
}

/// The two Arm dot products: four codes (one 4-byte lane per output row)
/// times the four activations in lane `S` of `x`, added to `acc`. The
/// int8 layers use them too (`int8.rs`).
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    use std::arch::aarch64::*;

    pub trait Dot {
        unsafe fn dot<const S: i32>(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t;
    }

    /// `sdot`. Callers run with the `dotprod` feature on.
    pub struct Sdot;

    impl Dot for Sdot {
        #[inline(always)]
        unsafe fn dot<const S: i32>(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
            vdotq_laneq_s32::<S>(acc, w, x)
        }
    }

    /// Armv8.0: widening products, then two pairwise adds per row. Both
    /// factors are i8, so each product fits in i16; the pair sums widen to i32.
    pub struct Smull;

    impl Dot for Smull {
        #[inline(always)]
        unsafe fn dot<const S: i32>(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
            let x = vreinterpretq_s8_s32(vdupq_laneq_s32::<S>(vreinterpretq_s32_s8(x)));
            let low = vpaddlq_s16(vmull_s8(vget_low_s8(w), vget_low_s8(x)));
            let high = vpaddlq_s16(vmull_high_s8(w, x));
            vaddq_s32(acc, vpaddq_s32(low, high))
        }
    }
}

/// The five base-3 digits of every packed byte, least significant first.
/// A table copy per byte replaces a division per code: model load spent
/// most of its time on those divisions.
/// The 4 bytes of a 16-code span: byte j holds codes j, 4 + j, 8 + j, and
/// 12 + j, two bits each. Four codes move as one word: a code is 0 to 2, so
/// no bit leaves its byte.
pub(super) fn pack_span(codes: &[u8]) -> [u8; 4] {
    let word =
        |at: usize| u32::from_le_bytes([codes[at], codes[at + 1], codes[at + 2], codes[at + 3]]);
    (word(0) | word(4) << 2 | word(8) << 4 | word(12) << 6).to_le_bytes()
}

const DIGITS: [[u8; 5]; 243] = {
    let mut table = [[0u8; 5]; 243];
    let mut byte = 0;
    while byte < 243 {
        let (mut value, mut digit) = (byte, 0);
        while digit < 5 {
            table[byte][digit] = (value % 3) as u8;
            value /= 3;
            digit += 1;
        }
        byte += 1;
    }
    table
};

/// Activations as i8 with one scale and one sum per 128-value group.
pub struct Quantized {
    pub(super) m: usize,
    pub(super) values: Vec<i8>,
    pub(super) scales: Vec<f32>,
    pub(super) sums: Vec<i32>,
}

impl Quantized {
    pub fn new(isa: Isa, x: &[f32], m: usize, k: usize) -> Self {
        Self::grouped(isa, x, m, k, GROUP, false)
    }

    /// `new` with `group` values per scale (a multiple of 32). With
    /// `half_scales` each scale is rounded to f16 after the codes are made
    /// from the f32 scale, as ggml's `quantize_row_q8_0` stores it.
    pub fn grouped(
        isa: Isa,
        x: &[f32],
        m: usize,
        k: usize,
        group: usize,
        half_scales: bool,
    ) -> Self {
        let groups = k / group;
        // Reuse this thread's buffer: a fresh one would fault in new pages.
        let mut values = QUANTIZED.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
        // The quantizer writes every value of the first m x k, so a buffer that
        // is long enough is used as it is: no fill on the calling thread.
        if values.len() < m * k {
            values.resize(m * k, 0);
        }
        let mut scales = vec![0.0f32; m * groups];
        let mut sums = vec![0i32; m * groups];
        let (all_values, all_scales, all_sums) = (
            team::SendPtr::new(&mut values),
            team::SendPtr::new(&mut scales),
            team::SendPtr::new(&mut sums),
        );
        team::for_each(m.div_ceil(QUANTIZE_ROWS), |block| {
            for row in block * QUANTIZE_ROWS..((block + 1) * QUANTIZE_ROWS).min(m) {
                // SAFETY: each row belongs to one task; the buffers outlive the step.
                let (values, scales, sums) = unsafe {
                    (
                        all_values.slice_mut(row * k, k),
                        all_scales.slice_mut(row * groups, groups),
                        all_sums.slice_mut(row * groups, groups),
                    )
                };
                let src = &x[row * k..(row + 1) * k];
                for g in 0..groups {
                    let span = g * group..(g + 1) * group;
                    let (scale, sum) = match isa {
                        #[cfg(target_arch = "x86_64")]
                        // SAFETY: Isa::detect confirmed the CPU features.
                        Isa::Avx512Vnni => unsafe {
                            quantize_avx512(&src[span.clone()], &mut values[span])
                        },
                        #[cfg(target_arch = "x86_64")]
                        // SAFETY: Isa::detect confirmed the CPU features.
                        Isa::Avx2 => unsafe {
                            quantize_avx2(&src[span.clone()], &mut values[span])
                        },
                        #[cfg(target_arch = "aarch64")]
                        // SAFETY: NEON is part of every arm64 CPU.
                        Isa::Neon | Isa::NeonDot => unsafe {
                            quantize_neon(&src[span.clone()], &mut values[span])
                        },
                        Isa::Scalar => quantize_scalar(&src[span.clone()], &mut values[span]),
                    };
                    scales[g] = if half_scales {
                        super::tensors::f16_to_f32(super::tensors::f32_to_f16(scale))
                    } else {
                        scale
                    };
                    sums[g] = sum;
                }
            }
        });
        Self {
            m,
            values,
            scales,
            sums,
        }
    }
}

/// Rows per quantization task: a short take still splits, a row is never shared.
const QUANTIZE_ROWS: usize = 8;

thread_local! {
    static QUANTIZED: RefCell<Vec<i8>> = const { RefCell::new(Vec::new()) };
}

impl Drop for Quantized {
    fn drop(&mut self) {
        let values = std::mem::take(&mut self.values);
        QUANTIZED.with(|cell| *cell.borrow_mut() = values);
    }
}

fn group_scale(max: f32) -> f32 {
    if max > 0.0 {
        max / 127.0
    } else {
        1.0
    }
}

/// One group (a multiple of 32 values) to i8, rounding half to even like `cvtps2dq`.
fn quantize_scalar(src: &[f32], dst: &mut [i8]) -> (f32, i32) {
    let scale = group_scale(src.iter().fold(0.0f32, |m, v| m.max(v.abs())));
    let inv = 1.0 / scale;
    let mut sum = 0i32;
    for (d, v) in dst.iter_mut().zip(src) {
        let code = (v * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
        *d = code;
        sum += i32::from(code);
    }
    (scale, sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn quantize_avx512(src: &[f32], dst: &mut [i8]) -> (f32, i32) {
    use std::arch::x86_64::*;
    let p = src.as_ptr();
    let mut vmax = _mm512_setzero_ps();
    for i in 0..src.len() / 16 {
        vmax = _mm512_max_ps(vmax, _mm512_abs_ps(_mm512_loadu_ps(p.add(i * 16))));
    }
    let scale = group_scale(_mm512_reduce_max_ps(vmax));
    let inv = _mm512_set1_ps(1.0 / scale);
    let limit = _mm512_set1_epi32(127);
    let mut vsum = _mm512_setzero_si512();
    for i in 0..src.len() / 16 {
        let q = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(p.add(i * 16)), inv));
        let q = _mm512_max_epi32(
            _mm512_min_epi32(q, limit),
            _mm512_sub_epi32(_mm512_setzero_si512(), limit),
        );
        vsum = _mm512_add_epi32(vsum, q);
        _mm_storeu_si128(
            dst.as_mut_ptr().add(i * 16).cast(),
            _mm512_cvtsepi32_epi8(q),
        );
    }
    (scale, _mm512_reduce_add_epi32(vsum))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_avx2(src: &[f32], dst: &mut [i8]) -> (f32, i32) {
    use std::arch::x86_64::*;
    let p = src.as_ptr();
    let sign = _mm256_set1_ps(-0.0);
    let mut vmax = _mm256_setzero_ps();
    for i in 0..src.len() / 8 {
        vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign, _mm256_loadu_ps(p.add(i * 8))));
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), vmax);
    let scale = group_scale(lanes.iter().fold(0.0f32, |m, v| m.max(*v)));
    let inv = _mm256_set1_ps(1.0 / scale);
    let limit = _mm256_set1_epi32(127);
    let low = _mm256_set1_epi32(-127);
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let mut vsum = _mm256_setzero_si256();
    for i in 0..src.len() / 32 {
        let mut q = [_mm256_setzero_si256(); 4];
        for (j, lane) in q.iter_mut().enumerate() {
            let v = _mm256_mul_ps(_mm256_loadu_ps(p.add(i * 32 + j * 8)), inv);
            *lane = _mm256_max_epi32(_mm256_min_epi32(_mm256_cvtps_epi32(v), limit), low);
            vsum = _mm256_add_epi32(vsum, *lane);
        }
        let words = _mm256_packs_epi16(
            _mm256_packs_epi32(q[0], q[1]),
            _mm256_packs_epi32(q[2], q[3]),
        );
        _mm256_storeu_si256(
            dst.as_mut_ptr().add(i * 32).cast(),
            _mm256_permutevar8x32_epi32(words, order),
        );
    }
    let mut sums = [0i32; 8];
    _mm256_storeu_si256(sums.as_mut_ptr().cast(), vsum);
    (scale, sums.iter().sum())
}

/// As `quantize_scalar` (a multiple of 32 values: 128 for ternary and int8
/// layers, 32 for Q8_0); `fcvtns` also rounds half to even.
#[cfg(target_arch = "aarch64")]
unsafe fn quantize_neon(src: &[f32], dst: &mut [i8]) -> (f32, i32) {
    use std::arch::aarch64::*;
    let p = src.as_ptr();
    let mut vmax = vdupq_n_f32(0.0);
    for i in 0..src.len() / 4 {
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(p.add(i * 4))));
    }
    let scale = group_scale(vmaxvq_f32(vmax));
    let inv = vdupq_n_f32(1.0 / scale);
    let (limit, low) = (vdupq_n_s32(127), vdupq_n_s32(-127));
    let mut vsum = vdupq_n_s32(0);
    for i in 0..src.len() / 16 {
        let mut q = [vdupq_n_s32(0); 4];
        for (j, lane) in q.iter_mut().enumerate() {
            let v = vmulq_f32(vld1q_f32(p.add(i * 16 + j * 4)), inv);
            *lane = vmaxq_s32(vminq_s32(vcvtnq_s32_f32(v), limit), low);
            vsum = vaddq_s32(vsum, *lane);
        }
        let first = vcombine_s16(vmovn_s32(q[0]), vmovn_s32(q[1]));
        let second = vcombine_s16(vmovn_s32(q[2]), vmovn_s32(q[3]));
        vst1q_s8(
            dst.as_mut_ptr().add(i * 16),
            vcombine_s8(vmovn_s16(first), vmovn_s16(second)),
        );
    }
    (scale, vaddvq_s32(vsum))
}

#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: tasks write disjoint output elements; the buffer outlives the parallel scope.
unsafe impl Send for OutPtr {}
// SAFETY: as for `Send`: tasks write disjoint output elements.
unsafe impl Sync for OutPtr {}
