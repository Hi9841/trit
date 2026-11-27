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
    /// 256-bit `vpdpbusd` (VEX form, AVX-VNNI on Intel 12th+ gen / Arrow Lake).
    #[cfg(target_arch = "x86_64")]
    AvxVnniVex,
    /// 256-bit `vpdpbusd` (EVEX form, AVX-512 VNNI + VL).
    #[cfg(target_arch = "x86_64")]
    AvxVnniEvex,
    #[cfg(target_arch = "x86_64")]
    Avx512Vnni,
    /// Armv8.0 NEON: every arm64 CPU.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// Armv8.2 dot product (`sdot`): Apple M1 and most phones since 2018.
    #[cfg(target_arch = "aarch64")]
    NeonDot,
    /// Armv8.6 int8 matrix multiply (`smmla`): Apple M2 and later, Neoverse N2/V1/V2.
    #[cfg(target_arch = "aarch64")]
    NeonMmla,
}

impl Isa {
    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => "avx2",
            #[cfg(target_arch = "x86_64")]
            Isa::AvxVnniVex | Isa::AvxVnniEvex => "avx-vnni",
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni => "avx512-vnni",
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => "neon",
            #[cfg(target_arch = "aarch64")]
            Isa::NeonDot => "neon-dotprod",
            #[cfg(target_arch = "aarch64")]
            Isa::NeonMmla => "neon-i8mm",
        }
    }

    pub(super) fn rows(self) -> usize {
        match self {
            Isa::Scalar => 1,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 | Isa::AvxVnniVex | Isa::AvxVnniEvex => 8,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni => 16,
            #[cfg(target_arch = "aarch64")]
            Isa::Neon | Isa::NeonDot | Isa::NeonMmla => 8,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn is_mmla(self) -> bool {
        self == Isa::NeonMmla
    }

    /// The fastest path this CPU runs. `TRIT_ISA=scalar|avx2|avxvnni|neon|dotprod` forces a slower one.
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
    /// Maps a previously cached ternary layer directly from the active read
    /// cache without unpacking base-3 weights.
    pub fn load_cached(tag: &str, n: usize, k: usize, group: usize) -> Option<Self> {
        if group != GROUP || !k.is_multiple_of(GROUP) || super::v2::enabled() {
            return None;
        }
        let detected = Isa::detect();
        let rows = detected.rows();
        let isa = if k.is_multiple_of(GROUP) && n.is_multiple_of(2 * rows) {
            detected
        } else {
            Isa::Scalar
        };
        let scalar = isa == Isa::Scalar;
        let groups = k / GROUP;
        let code_len = if scalar { n * k } else { n * k / 4 };
        let codes = Store::<u8>::mapped(&format!("{tag}:c"), code_len)?;
        let scales = Store::<f32>::mapped(&format!("{tag}:s"), n * groups)?;
        Some(Self {
            isa,
            n,
            k,
            codes,
            scales,
            v2: None,
            #[cfg(feature = "gpu")]
            gpu: None,
        })
    }

    pub fn from_base3_named(
        tag: Option<&str>,
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
        Ok(Self::from_rows_named(
            tag,
            Isa::detect(),
            scales,
            n,
            k,
            |row, codes| {
                let packed_row = &packed[row * row_bytes..(row + 1) * row_bytes];
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
        Self::from_rows_named(None, isa, scales, n, k, row)
    }

    fn from_rows_named(
        tag: Option<&str>,
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
                #[cfg(target_arch = "aarch64")]
                let mmla = isa.is_mmla();
                #[cfg(not(target_arch = "aarch64"))]
                let mmla = false;
                if mmla {
                    // `[n / 8][k / 32][4 row-pairs][16 bytes]`: one 16-byte vector holds
                    // 4 consecutive 2x8 `smmla` weight tiles packed 4 codes per byte.
                    for k32 in 0..k / 32 {
                        for rp in 0..4 {
                            let dst = ((block * (k / 32) + k32) * 4 + rp) * 16;
                            for rr in 0..2 {
                                let row_codes = &codes[(rp * 2 + rr) * k + k32 * 32..][..32];
                                for j in 0..8 {
                                    packed[dst + rr * 8 + j] = row_codes[j]
                                        | (row_codes[j + 8] << 2)
                                        | (row_codes[j + 16] << 4)
                                        | (row_codes[j + 24] << 6);
                                }
                            }
                        }
                    }
                } else {
                    for k16 in 0..k / 16 {
                        for r in 0..rows {
                            let src = &codes[r * k + k16 * 16..][..16];
                            let dst = ((block * (k / 16) + k16) * rows + r) * 4;
                            packed[dst..dst + 4].copy_from_slice(&pack_span(src));
                        }
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
        let (codes, scales) = match tag {
            Some(tag) => (
                Store::named(&format!("{tag}:c"), packed),
                Store::named(&format!("{tag}:s"), packed_scales),
            ),
            None => (Store::weights(packed), Store::weights(packed_scales)),
        };
        Self {
            isa,
            n,
            k,
            codes,
            scales,
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
        #[cfg(target_arch = "aarch64")]
        if self.isa.is_mmla() {
            let k32 = kk / 32;
            let rem = kk % 32;
            let s = rem / 8;
            let j = rem % 8;
            let rp = r / 2;
            let rr = r % 2;
            let byte = ((block * (self.k / 32) + k32) * 4 + rp) * 16 + rr * 8 + j;
            return (self.codes[byte] >> (s * 2)) & 3;
        }
        let k16 = kk / 16;
        let lane = kk % 16;
        let byte = ((block * (self.k / 16) + k16) * rows + r) * 4 + lane % 4;
        (self.codes[byte] >> ((lane / 4) * 2)) & 3
    }

    /// `x` is `m x k` f32; returns `m x n`.
    pub fn forward(&self, x: &[f32], m: usize) -> Buf {
        let _span = crate::prof::span(crate::prof::Stage::Ternary);
        let _product = crate::prof::product(crate::prof::Product::Ternary, self.n, self.k, m);
        #[cfg(feature = "gpu")]
        if let Some(layer) = &self.gpu {
            if let Some(out) = crate::gpu::try_product(|| layer.forward(x, m, false)) {
                return out;
            }
        }
        if let Some(layer) = &self.v2 {
            return layer.forward(x, m);
        }
        let mut out = Buf::scratch(m * self.n);
        let q = Quantized::new(self.isa, x, m, self.k);
        let ptr = OutPtr(out.as_mut_ptr());
        let rows = self.isa.rows();
        let unit_work = q.m * self.k * (2 * rows);
        match self.isa {
            Isa::Scalar => team::for_each(self.n, |row| self.scalar_rows(&q, row, ptr)),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => {
                super::int8::dispatch_2d(self.n / (2 * rows), unit_work, q.m, 4, |unit, frames| {
                    // SAFETY: Isa::detect confirmed AVX2 + FMA; each (unit, frames) writes disjoint outputs.
                    unsafe { self.avx2_task(&q, unit, frames, ptr) }
                })
            }
            #[cfg(target_arch = "x86_64")]
            Isa::AvxVnniVex => {
                super::int8::dispatch_2d(self.n / (2 * rows), unit_work, q.m, 4, |unit, frames| {
                    // SAFETY: Isa::detect confirmed AVX2 + FMA + AVX-VNNI.
                    unsafe { self.vnni256_vex_task(&q, unit, frames, ptr) }
                })
            }
            #[cfg(target_arch = "x86_64")]
            Isa::AvxVnniEvex => {
                super::int8::dispatch_2d(self.n / (2 * rows), unit_work, q.m, 4, |unit, frames| {
                    // SAFETY: Isa::detect confirmed AVX-512F/BW/VL/VNNI.
                    unsafe { self.vnni256_evex_task(&q, unit, frames, ptr) }
                })
            }
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni => {
                super::int8::dispatch_2d(self.n / (2 * rows), unit_work, q.m, 6, |unit, frames| {
                    // SAFETY: Isa::detect confirmed AVX-512 VNNI.
                    unsafe { self.vnni_task(&q, unit, frames, ptr) }
                })
            }
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => {
                super::int8::dispatch_2d(self.n / (2 * rows), unit_work, q.m, 8, |unit, frames| {
                    // SAFETY: NEON is part of every arm64 CPU.
                    unsafe { self.neon_task::<neon::Smull>(&q, unit, frames, ptr) }
                })
            }
            #[cfg(target_arch = "aarch64")]
            Isa::NeonDot => {
                super::int8::dispatch_2d(self.n / (2 * rows), unit_work, q.m, 8, |unit, frames| {
                    // SAFETY: Isa::detect confirmed the dot product feature.
                    unsafe { self.sdot_task(&q, unit, frames, ptr) }
                })
            }
            #[cfg(target_arch = "aarch64")]
            Isa::NeonMmla => {
                let paired = super::int8::paired(&q, self.k, self.k / GROUP);
                let block_work = paired.pairs * 2 * self.k * rows;
                super::int8::dispatch_2d(
                    self.n / rows,
                    block_work,
                    paired.pairs,
                    4,
                    |block, pairs| {
                        // SAFETY: Isa::detect confirmed arm64 i8mm; each (block, pairs) writes disjoint outputs.
                        unsafe { self.mmla_task(&paired, &q, block, pairs, ptr) }
                    },
                );
            }
        }
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
    unsafe fn vnni_task(
        &self,
        q: &Quantized,
        task: usize,
        frames: std::ops::Range<usize>,
        out: OutPtr,
    ) {
        let mut frame = frames.start;
        while frame + 6 <= frames.end {
            self.vnni_tile::<6>(q, task, frame, out);
            frame += 6;
        }
        match frames.end - frame {
            1 => self.vnni_tile::<1>(q, task, frame, out),
            2 => self.vnni_tile::<2>(q, task, frame, out),
            3 => self.vnni_tile::<3>(q, task, frame, out),
            4 => self.vnni_tile::<4>(q, task, frame, out),
            5 => self.vnni_tile::<5>(q, task, frame, out),
            _ => {}
        }
    }

    /// 32 outputs (two 16-row blocks) by F frames.
    /// With F = 6, `acc` (12) + `dots` (12) + `z0, z1, w0, w1, low, x` (6) = 30 <= 32 ZMMs,
    /// keeping both accumulators and integer dots completely spill-free.
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
        debug_assert!(frame0 + F <= q.m);
        debug_assert!(q.values.len() >= (frame0 + F) * k);
        debug_assert!(q.scales.len() >= (frame0 + F) * groups);
        debug_assert!(q.sums.len() >= (frame0 + F) * groups);
        let c0_base = self.codes.as_ptr().add(blocks[0] * k16s * 64);
        let c1_base = self.codes.as_ptr().add(blocks[1] * k16s * 64);
        let s0_base = self.scales.as_ptr().add(blocks[0] * groups * 16);
        let s1_base = self.scales.as_ptr().add(blocks[1] * groups * 16);
        let x_base = q.values.as_ptr().add(frame0 * k);
        let x_scales = q.scales.as_ptr().add(frame0 * groups);
        let x_sums = q.sums.as_ptr().add(frame0 * groups);
        let low = _mm512_set1_epi8(3);
        let mut acc = [[_mm512_setzero_ps(); F]; 2];
        for g in 0..groups {
            // Start each dot product at -sum(x) so (code . x) gives (code - 1) . x
            // directly without a post-loop subtraction.
            let mut dots = [[_mm512_setzero_si512(); F]; 2];
            for f in 0..F {
                let neg_sum = _mm512_set1_epi32((*x_sums.add(f * groups + g)).wrapping_neg());
                dots[0][f] = neg_sum;
                dots[1][f] = neg_sum;
            }
            for quad in 0..GROUP / 16 {
                let k16 = g * (GROUP / 16) + quad;
                let z0 = _mm512_loadu_si512(c0_base.add(k16 * 64).cast());
                let z1 = _mm512_loadu_si512(c1_base.add(k16 * 64).cast());
                // A 16-bit shift is safe: the mask keeps the low two bits of each byte.
                for s in 0..4 {
                    let (w0, w1) = match s {
                        0 => (_mm512_and_si512(z0, low), _mm512_and_si512(z1, low)),
                        1 => (
                            _mm512_and_si512(_mm512_srli_epi16::<2>(z0), low),
                            _mm512_and_si512(_mm512_srli_epi16::<2>(z1), low),
                        ),
                        2 => (
                            _mm512_and_si512(_mm512_srli_epi16::<4>(z0), low),
                            _mm512_and_si512(_mm512_srli_epi16::<4>(z1), low),
                        ),
                        _ => (
                            _mm512_and_si512(_mm512_srli_epi16::<6>(z0), low),
                            _mm512_and_si512(_mm512_srli_epi16::<6>(z1), low),
                        ),
                    };
                    let off = (4 * k16 + s) * 4;
                    for f in 0..F {
                        let x = _mm512_set1_epi32(std::ptr::read_unaligned(
                            x_base.add(f * k + off).cast::<i32>(),
                        ));
                        dots[0][f] = _mm512_dpbusd_epi32(dots[0][f], w0, x);
                        dots[1][f] = _mm512_dpbusd_epi32(dots[1][f], w1, x);
                    }
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
        for (b, block) in blocks.iter().enumerate() {
            for f in 0..F {
                _mm512_storeu_ps(out.0.add((frame0 + f) * self.n + block * 16), acc[b][f]);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,avxvnni")]
    unsafe fn vnni256_vex_task(
        &self,
        q: &Quantized,
        task: usize,
        frames: std::ops::Range<usize>,
        out: OutPtr,
    ) {
        let mut frame = frames.start;
        while frame + 4 <= frames.end {
            self.vnni256_tile_vex::<4>(q, task, frame, out);
            frame += 4;
        }
        match frames.end - frame {
            1 => self.vnni256_tile_vex::<1>(q, task, frame, out),
            2 => self.vnni256_tile_vex::<2>(q, task, frame, out),
            3 => self.vnni256_tile_vex::<3>(q, task, frame, out),
            _ => {}
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
    unsafe fn vnni256_evex_task(
        &self,
        q: &Quantized,
        task: usize,
        frames: std::ops::Range<usize>,
        out: OutPtr,
    ) {
        let mut frame = frames.start;
        while frame + 4 <= frames.end {
            self.vnni256_tile_evex::<4>(q, task, frame, out);
            frame += 4;
        }
        match frames.end - frame {
            1 => self.vnni256_tile_evex::<1>(q, task, frame, out),
            2 => self.vnni256_tile_evex::<2>(q, task, frame, out),
            3 => self.vnni256_tile_evex::<3>(q, task, frame, out),
            _ => {}
        }
    }

    /// 16 outputs (two 8-row blocks) by F frames using 256-bit VEX `vpdpbusd`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,avxvnni")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn vnni256_tile_vex<const F: usize>(
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
        let c0_base = self.codes.as_ptr().add(blocks[0] * k16s * 32);
        let c1_base = self.codes.as_ptr().add(blocks[1] * k16s * 32);
        let s0_base = self.scales.as_ptr().add(blocks[0] * groups * 8);
        let s1_base = self.scales.as_ptr().add(blocks[1] * groups * 8);
        let x_base = q.values.as_ptr().add(frame0 * k);
        let x_scales = q.scales.as_ptr().add(frame0 * groups);
        let x_sums = q.sums.as_ptr().add(frame0 * groups);
        let low = _mm256_set1_epi8(3);
        let mut acc = [[_mm256_setzero_ps(); F]; 2];
        for g in 0..groups {
            let mut dots = [[_mm256_setzero_si256(); F]; 2];
            for f in 0..F {
                let neg_sum = _mm256_set1_epi32((*x_sums.add(f * groups + g)).wrapping_neg());
                dots[0][f] = neg_sum;
                dots[1][f] = neg_sum;
            }
            for quad in 0..GROUP / 16 {
                let k16 = g * (GROUP / 16) + quad;
                let z0 = _mm256_loadu_si256(c0_base.add(k16 * 32).cast());
                let z1 = _mm256_loadu_si256(c1_base.add(k16 * 32).cast());
                for s in 0..4 {
                    let (w0, w1) = match s {
                        0 => (_mm256_and_si256(z0, low), _mm256_and_si256(z1, low)),
                        1 => (
                            _mm256_and_si256(_mm256_srli_epi16::<2>(z0), low),
                            _mm256_and_si256(_mm256_srli_epi16::<2>(z1), low),
                        ),
                        2 => (
                            _mm256_and_si256(_mm256_srli_epi16::<4>(z0), low),
                            _mm256_and_si256(_mm256_srli_epi16::<4>(z1), low),
                        ),
                        _ => (
                            _mm256_and_si256(_mm256_srli_epi16::<6>(z0), low),
                            _mm256_and_si256(_mm256_srli_epi16::<6>(z1), low),
                        ),
                    };
                    let off = (4 * k16 + s) * 4;
                    for f in 0..F {
                        let x = _mm256_set1_epi32(std::ptr::read_unaligned(
                            x_base.add(f * k + off).cast::<i32>(),
                        ));
                        dots[0][f] = _mm256_dpbusd_avx_epi32(dots[0][f], w0, x);
                        dots[1][f] = _mm256_dpbusd_avx_epi32(dots[1][f], w1, x);
                    }
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
            for f in 0..F {
                _mm256_storeu_ps(out.0.add((frame0 + f) * self.n + block * 8), acc[b][f]);
            }
        }
    }

    /// 16 outputs (two 8-row blocks) by F frames using 256-bit EVEX `vpdpbusd`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn vnni256_tile_evex<const F: usize>(
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
        let c0_base = self.codes.as_ptr().add(blocks[0] * k16s * 32);
        let c1_base = self.codes.as_ptr().add(blocks[1] * k16s * 32);
        let s0_base = self.scales.as_ptr().add(blocks[0] * groups * 8);
        let s1_base = self.scales.as_ptr().add(blocks[1] * groups * 8);
        let x_base = q.values.as_ptr().add(frame0 * k);
        let x_scales = q.scales.as_ptr().add(frame0 * groups);
        let x_sums = q.sums.as_ptr().add(frame0 * groups);
        let low = _mm256_set1_epi8(3);
        let mut acc = [[_mm256_setzero_ps(); F]; 2];
        for g in 0..groups {
            let mut dots = [[_mm256_setzero_si256(); F]; 2];
            for f in 0..F {
                let neg_sum = _mm256_set1_epi32((*x_sums.add(f * groups + g)).wrapping_neg());
                dots[0][f] = neg_sum;
                dots[1][f] = neg_sum;
            }
            for quad in 0..GROUP / 16 {
                let k16 = g * (GROUP / 16) + quad;
                let z0 = _mm256_loadu_si256(c0_base.add(k16 * 32).cast());
                let z1 = _mm256_loadu_si256(c1_base.add(k16 * 32).cast());
                for s in 0..4 {
                    let (w0, w1) = match s {
                        0 => (_mm256_and_si256(z0, low), _mm256_and_si256(z1, low)),
                        1 => (
                            _mm256_and_si256(_mm256_srli_epi16::<2>(z0), low),
                            _mm256_and_si256(_mm256_srli_epi16::<2>(z1), low),
                        ),
                        2 => (
                            _mm256_and_si256(_mm256_srli_epi16::<4>(z0), low),
                            _mm256_and_si256(_mm256_srli_epi16::<4>(z1), low),
                        ),
                        _ => (
                            _mm256_and_si256(_mm256_srli_epi16::<6>(z0), low),
                            _mm256_and_si256(_mm256_srli_epi16::<6>(z1), low),
                        ),
                    };
                    let off = (4 * k16 + s) * 4;
                    for f in 0..F {
                        let x = _mm256_set1_epi32(std::ptr::read_unaligned(
                            x_base.add(f * k + off).cast::<i32>(),
                        ));
                        dots[0][f] = _mm256_dpbusd_epi32(dots[0][f], w0, x);
                        dots[1][f] = _mm256_dpbusd_epi32(dots[1][f], w1, x);
                    }
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
            for f in 0..F {
                _mm256_storeu_ps(out.0.add((frame0 + f) * self.n + block * 8), acc[b][f]);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_task(
        &self,
        q: &Quantized,
        task: usize,
        frames: std::ops::Range<usize>,
        out: OutPtr,
    ) {
        let mut frame = frames.start;
        while frame + 4 <= frames.end {
            self.avx2_tile::<4>(q, task, frame, out);
            frame += 4;
        }
        match frames.end - frame {
            1 => self.avx2_tile::<1>(q, task, frame, out),
            2 => self.avx2_tile::<2>(q, task, frame, out),
            3 => self.avx2_tile::<3>(q, task, frame, out),
            _ => {}
        }
    }

    /// 16 outputs (two 8-row blocks) by F frames.
    /// Codes are 0..=2 and |q| <= 127, so each `vpmaddubsw` pair sum lies in
    /// [-508, +508]. Across all 32 `k4` steps of a 128-value group, the i16
    /// sum stays within [-16256, +16256] (never overflows i16), so we widen
    /// to i32 with a single `vpmaddwd` once per group instead of 32 times.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn avx2_tile<const F: usize>(
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
        let c0_base = self.codes.as_ptr().add(blocks[0] * k16s * 32);
        let c1_base = self.codes.as_ptr().add(blocks[1] * k16s * 32);
        let s0_base = self.scales.as_ptr().add(blocks[0] * groups * 8);
        let s1_base = self.scales.as_ptr().add(blocks[1] * groups * 8);
        let x_base = q.values.as_ptr().add(frame0 * k);
        let x_scales = q.scales.as_ptr().add(frame0 * groups);
        let x_sums = q.sums.as_ptr().add(frame0 * groups);
        let ones = _mm256_set1_epi16(1);
        let low = _mm256_set1_epi8(3);
        let mut acc = [[_mm256_setzero_ps(); F]; 2];
        for g in 0..groups {
            let mut dots16 = [[_mm256_setzero_si256(); F]; 2];
            for quad in 0..GROUP / 16 {
                let k16 = g * (GROUP / 16) + quad;
                let z0 = _mm256_loadu_si256(c0_base.add(k16 * 32).cast());
                let z1 = _mm256_loadu_si256(c1_base.add(k16 * 32).cast());
                // A 16-bit shift is safe: the mask keeps the low two bits of each byte.
                for s in 0..4 {
                    let (w0, w1) = match s {
                        0 => (_mm256_and_si256(z0, low), _mm256_and_si256(z1, low)),
                        1 => (
                            _mm256_and_si256(_mm256_srli_epi16::<2>(z0), low),
                            _mm256_and_si256(_mm256_srli_epi16::<2>(z1), low),
                        ),
                        2 => (
                            _mm256_and_si256(_mm256_srli_epi16::<4>(z0), low),
                            _mm256_and_si256(_mm256_srli_epi16::<4>(z1), low),
                        ),
                        _ => (
                            _mm256_and_si256(_mm256_srli_epi16::<6>(z0), low),
                            _mm256_and_si256(_mm256_srli_epi16::<6>(z1), low),
                        ),
                    };
                    let off = (4 * k16 + s) * 4;
                    for f in 0..F {
                        let x = _mm256_set1_epi32(std::ptr::read_unaligned(
                            x_base.add(f * k + off).cast::<i32>(),
                        ));
                        dots16[0][f] = _mm256_add_epi16(dots16[0][f], _mm256_maddubs_epi16(w0, x));
                        dots16[1][f] = _mm256_add_epi16(dots16[1][f], _mm256_maddubs_epi16(w1, x));
                    }
                }
            }
            let wscale0 = _mm256_loadu_ps(s0_base.add(g * 8));
            let wscale1 = _mm256_loadu_ps(s1_base.add(g * 8));
            for f in 0..F {
                let sum = _mm256_set1_epi32(*x_sums.add(f * groups + g));
                let xscale = _mm256_set1_ps(*x_scales.add(f * groups + g));
                let d0 = _mm256_sub_epi32(_mm256_madd_epi16(dots16[0][f], ones), sum);
                let d1 = _mm256_sub_epi32(_mm256_madd_epi16(dots16[1][f], ones), sum);
                acc[0][f] = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(d0),
                    _mm256_mul_ps(wscale0, xscale),
                    acc[0][f],
                );
                acc[1][f] = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(d1),
                    _mm256_mul_ps(wscale1, xscale),
                    acc[1][f],
                );
            }
        }
        for (b, block) in blocks.iter().enumerate() {
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
        task: usize,
        frames: std::ops::Range<usize>,
        out: OutPtr,
    ) {
        self.neon_task::<neon::Sdot>(q, task, frames, out);
    }

    /// Two 8-row blocks, 8 frames at a time (`acc` sits in L1 stack across
    /// 128-K groups so 16 `dots` + 8 `x` + `z` + `w` + `low` = 27 <= 32 NEON
    /// registers, halving weight loads and 2-bit unpacks vs 4 frames).
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon_task<D: neon::Dot>(
        &self,
        q: &Quantized,
        task: usize,
        frames: std::ops::Range<usize>,
        out: OutPtr,
    ) {
        for block in [2 * task, 2 * task + 1] {
            let mut frame = frames.start;
            while frame + 8 <= frames.end {
                self.neon_tile::<8, D>(q, block, frame, out);
                frame += 8;
            }
            match frames.end - frame {
                1 => self.neon_tile::<1, D>(q, block, frame, out),
                2 => self.neon_tile::<2, D>(q, block, frame, out),
                3 => self.neon_tile::<3, D>(q, block, frame, out),
                4 => self.neon_tile::<4, D>(q, block, frame, out),
                5 => self.neon_tile::<5, D>(q, block, frame, out),
                6 => self.neon_tile::<6, D>(q, block, frame, out),
                7 => self.neon_tile::<7, D>(q, block, frame, out),
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
            for f in 0..F {
                let neg_sum = vdupq_n_s32(q.sums[(frame0 + f) * groups + g].wrapping_neg());
                dots[f][0] = neg_sum;
                dots[f][1] = neg_sum;
            }
            for quad in 0..GROUP / 16 {
                let k16 = g * (GROUP / 16) + quad;
                let mut x = [vdupq_n_s8(0); F];
                for f in 0..F {
                    x[f] = vld1q_s8(q.values.as_ptr().add((frame0 + f) * k + k16 * 16));
                }
                let codes = self.codes.as_ptr().add((block * k16s + k16) * 32);
                for half in 0..2 {
                    let z = vld1q_u8(codes.add(half * 16));
                    let w0 = vreinterpretq_s8_u8(vandq_u8(z, low));
                    for f in 0..F {
                        dots[f][half] = D::dot::<0>(dots[f][half], w0, x[f]);
                    }
                    let w1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(z), low));
                    for f in 0..F {
                        dots[f][half] = D::dot::<1>(dots[f][half], w1, x[f]);
                    }
                    let w2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(z), low));
                    for f in 0..F {
                        dots[f][half] = D::dot::<2>(dots[f][half], w2, x[f]);
                    }
                    let w3 = vreinterpretq_s8_u8(vshrq_n_u8::<6>(z));
                    for f in 0..F {
                        dots[f][half] = D::dot::<3>(dots[f][half], w3, x[f]);
                    }
                }
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

    /// Armv8.6 `smmla` ternary kernel (Apple M2 and later, Neoverse N2/V1/V2):
    /// one 8-output block over `pairs`, up to 4 frame pairs (8 frames) per tile.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "i8mm")]
    unsafe fn mmla_task(
        &self,
        x: &super::int8::Paired,
        q: &Quantized,
        block: usize,
        pairs: std::ops::Range<usize>,
        out: OutPtr,
    ) {
        let mut pair = pairs.start;
        while pair + 4 <= pairs.end {
            self.mmla_tile::<4>(x, q, block, pair, out);
            pair += 4;
        }
        match pairs.end - pair {
            1 => self.mmla_tile::<1>(x, q, block, pair, out),
            2 => self.mmla_tile::<2>(x, q, block, pair, out),
            3 => self.mmla_tile::<3>(x, q, block, pair, out),
            _ => {}
        }
    }

    /// 8 outputs (four row pairs) by `P` frame pairs (`2 * P` frames).
    /// Each packed 16-byte vector holds four 2x8 ternary weight tiles (`0..=2`),
    /// decoded directly into `int8x16_t` for `smmla` (32 MACs/instruction, twice
    /// the throughput of `sdot`).
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn mmla_tile<const P: usize>(
        &self,
        x: &super::int8::Paired,
        q: &Quantized,
        block: usize,
        pair0: usize,
        out: OutPtr,
    ) {
        use std::arch::aarch64::*;
        let (k, k32s, groups) = (self.k, self.k / 32, self.k / GROUP);
        let codes = x.values.as_ptr().add(pair0 * 2 * k);
        let weights = self.codes.as_ptr().add(block * k32s * 64);
        let low = vdupq_n_u8(3);
        let mut acc = [[vdupq_n_f32(0.0); 4]; P];
        for g in 0..groups {
            let mut dots = [[vdupq_n_s32(0); 4]; P];
            for p in 0..P {
                let f0 = 2 * (pair0 + p);
                let f1 = f0 + 1;
                let s0 = q.sums[f0 * groups + g].wrapping_neg();
                let s1 = if f1 < q.m {
                    q.sums[f1 * groups + g].wrapping_neg()
                } else {
                    0
                };
                let neg_sums = vcombine_s32(vdup_n_s32(s0), vdup_n_s32(s1));
                for r in 0..4 {
                    dots[p][r] = neg_sums;
                }
            }
            for span in 0..GROUP / 32 {
                let k32 = g * (GROUP / 32) + span;
                let wp = weights.add(k32 * 64);
                let z = [
                    vld1q_u8(wp),
                    vld1q_u8(wp.add(16)),
                    vld1q_u8(wp.add(32)),
                    vld1q_u8(wp.add(48)),
                ];
                for s in 0..4 {
                    let w = match s {
                        0 => [
                            vreinterpretq_s8_u8(vandq_u8(z[0], low)),
                            vreinterpretq_s8_u8(vandq_u8(z[1], low)),
                            vreinterpretq_s8_u8(vandq_u8(z[2], low)),
                            vreinterpretq_s8_u8(vandq_u8(z[3], low)),
                        ],
                        1 => [
                            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(z[0]), low)),
                            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(z[1]), low)),
                            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(z[2]), low)),
                            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(z[3]), low)),
                        ],
                        2 => [
                            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(z[0]), low)),
                            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(z[1]), low)),
                            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(z[2]), low)),
                            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(z[3]), low)),
                        ],
                        _ => [
                            vreinterpretq_s8_u8(vshrq_n_u8::<6>(z[0])),
                            vreinterpretq_s8_u8(vshrq_n_u8::<6>(z[1])),
                            vreinterpretq_s8_u8(vshrq_n_u8::<6>(z[2])),
                            vreinterpretq_s8_u8(vshrq_n_u8::<6>(z[3])),
                        ],
                    };
                    let x_off = (k32 * 4 + s) * 16;
                    for p in 0..P {
                        let pair = vld1q_s8(codes.add(p * 2 * k + x_off));
                        for r in 0..4 {
                            dots[p][r] = super::int8::smmla(dots[p][r], pair, w[r]);
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
                let ws = vreinterpretq_f32_u64(vdupq_n_u64(std::ptr::read_unaligned(
                    scales.add(2 * r).cast::<u64>(),
                )));
                for p in 0..P {
                    let scale = vmulq_f32(ws, xs[p]);
                    acc[p][r] = vfmaq_f32(acc[p][r], vcvtq_f32_s32(dots[p][r]), scale);
                }
            }
        }
        for p in 0..P {
            let first = 2 * (pair0 + p);
            for r in 0..4 {
                let column = block * 8 + 2 * r;
                vst1_f32(out.0.add(first * self.n + column), vget_low_f32(acc[p][r]));
                if first + 1 < q.m {
                    vst1_f32(
                        out.0.add((first + 1) * self.n + column),
                        vget_high_f32(acc[p][r]),
                    );
                }
            }
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
        let _span = crate::prof::span(crate::prof::Stage::Quantize);
        let groups = k / group;
        // Reuse this thread's buffers: a fresh one would fault in new pages,
        // and two new vectors for each product were 3 of 4 allocations of a
        // Parakeet call.
        let (mut values, mut scales, mut sums) =
            QUANTIZED.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
        // The quantizer writes every value, scale, and sum for the first m rows,
        // so buffers that are long enough are reused as they are without zero-fill.
        if values.len() < m * k {
            values.resize(m * k, 0);
        }
        if scales.len() < m * groups {
            scales.resize(m * groups, 0.0);
        }
        if sums.len() < m * groups {
            sums.resize(m * groups, 0);
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let hw_f16 = half_scales && super::fgemm::level() != super::fgemm::Level::Scalar;
        let (all_values, all_scales, all_sums) = (
            team::SendPtr::new(&mut values),
            team::SendPtr::new(&mut scales),
            team::SendPtr::new(&mut sums),
        );
        let threads = team::threads().max(1);
        let work = m.saturating_mul(k);
        let tasks = if threads <= 1 || work <= 32_768 {
            1
        } else {
            let target = if team::is_heterogeneous() {
                4 * threads
            } else {
                threads
            };
            target.min(m).min((work / 8_192).max(1))
        };
        team::for_each(tasks, |task| {
            let start = task * m / tasks;
            let end = (task + 1) * m / tasks;
            for row in start..end {
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
                        Isa::Avx2 | Isa::AvxVnniVex | Isa::AvxVnniEvex => unsafe {
                            quantize_avx2(&src[span.clone()], &mut values[span])
                        },
                        #[cfg(target_arch = "aarch64")]
                        // SAFETY: NEON is part of every arm64 CPU.
                        Isa::Neon | Isa::NeonDot | Isa::NeonMmla => unsafe {
                            quantize_neon(&src[span.clone()], &mut values[span])
                        },
                        Isa::Scalar => quantize_scalar(&src[span.clone()], &mut values[span]),
                    };
                    scales[g] = if half_scales {
                        #[cfg(target_arch = "x86_64")]
                        if hw_f16 {
                            // SAFETY: `fgemm::level() != Scalar` implies F16C.
                            unsafe { round_f16_f16c(scale) }
                        } else {
                            super::tensors::f16_to_f32(super::tensors::f32_to_f16(scale))
                        }
                        #[cfg(target_arch = "aarch64")]
                        if hw_f16 {
                            // SAFETY: scalar `fcvt h, s` / `fcvt s, h` are part of base arm64.
                            unsafe { round_f16_neon(scale) }
                        } else {
                            super::tensors::f16_to_f32(super::tensors::f32_to_f16(scale))
                        }
                        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
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

/// The values, scales, and sums of the last `Quantized` of this thread.
type Reused = (Vec<i8>, Vec<f32>, Vec<i32>);

thread_local! {
    static QUANTIZED: RefCell<Reused> = const { RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
}

impl Drop for Quantized {
    fn drop(&mut self) {
        let reused = (
            std::mem::take(&mut self.values),
            std::mem::take(&mut self.scales),
            std::mem::take(&mut self.sums),
        );
        QUANTIZED.with(|cell| *cell.borrow_mut() = reused);
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
#[target_feature(enable = "f16c")]
unsafe fn round_f16_f16c(value: f32) -> f32 {
    use std::arch::x86_64::*;
    let half = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm_set_ss(value));
    _mm_cvtss_f32(_mm_cvtph_ps(half))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn round_f16_neon(value: f32) -> f32 {
    let out: f32;
    std::arch::asm!(
        "fcvt {h:h}, {v:s}",
        "fcvt {out:s}, {h:h}",
        v = in(vreg) value,
        h = out(vreg) _,
        out = lateout(vreg) out,
        options(pure, nomem, nostack, preserves_flags),
    );
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn quantize_avx512(src: &[f32], dst: &mut [i8]) -> (f32, i32) {
    use std::arch::x86_64::*;
    assert!(src.len() == dst.len() && src.len().is_multiple_of(16));
    let p = src.as_ptr();
    let d = dst.as_mut_ptr();
    let f_high = _mm512_set1_ps(127.0);
    let f_low = _mm512_set1_ps(-127.0);
    // Clamp in float before conversion (+Inf -> 127, -Inf -> -127) and mask out
    // unordered lanes (NaN -> 0) so every float matches `quantize_scalar` and NEON.
    macro_rules! quant16 {
        ($x:expr, $inv:expr) => {{
            let v = _mm512_mul_ps($x, $inv);
            let ord = _mm512_cmp_ps_mask::<_CMP_ORD_Q>(v, v);
            let clamped = _mm512_min_ps(_mm512_max_ps(v, f_low), f_high);
            _mm512_maskz_cvtps_epi32(ord, clamped)
        }};
    }
    if src.len() == 128 {
        // Standard 128-value group: hold all 8 vectors in ZMM registers so pass 2 never reloads.
        let x0 = _mm512_loadu_ps(p);
        let x1 = _mm512_loadu_ps(p.add(16));
        let x2 = _mm512_loadu_ps(p.add(32));
        let x3 = _mm512_loadu_ps(p.add(48));
        let x4 = _mm512_loadu_ps(p.add(64));
        let x5 = _mm512_loadu_ps(p.add(80));
        let x6 = _mm512_loadu_ps(p.add(96));
        let x7 = _mm512_loadu_ps(p.add(112));
        let zero = _mm512_setzero_ps();
        let m0 = _mm512_max_ps(_mm512_abs_ps(x0), zero);
        let m1 = _mm512_max_ps(_mm512_abs_ps(x1), zero);
        let m2 = _mm512_max_ps(_mm512_abs_ps(x2), zero);
        let m3 = _mm512_max_ps(_mm512_abs_ps(x3), zero);
        let m0 = _mm512_max_ps(_mm512_abs_ps(x4), m0);
        let m1 = _mm512_max_ps(_mm512_abs_ps(x5), m1);
        let m2 = _mm512_max_ps(_mm512_abs_ps(x6), m2);
        let m3 = _mm512_max_ps(_mm512_abs_ps(x7), m3);
        let vmax = _mm512_max_ps(_mm512_max_ps(m0, m1), _mm512_max_ps(m2, m3));
        let scale = group_scale(_mm512_reduce_max_ps(vmax));
        let inv = _mm512_set1_ps(1.0 / scale);
        let q0 = quant16!(x0, inv);
        let q1 = quant16!(x1, inv);
        let q2 = quant16!(x2, inv);
        let q3 = quant16!(x3, inv);
        let q4 = quant16!(x4, inv);
        let q5 = quant16!(x5, inv);
        let q6 = quant16!(x6, inv);
        let q7 = quant16!(x7, inv);
        _mm_storeu_si128(d.cast(), _mm512_cvtepi32_epi8(q0));
        _mm_storeu_si128(d.add(16).cast(), _mm512_cvtepi32_epi8(q1));
        _mm_storeu_si128(d.add(32).cast(), _mm512_cvtepi32_epi8(q2));
        _mm_storeu_si128(d.add(48).cast(), _mm512_cvtepi32_epi8(q3));
        _mm_storeu_si128(d.add(64).cast(), _mm512_cvtepi32_epi8(q4));
        _mm_storeu_si128(d.add(80).cast(), _mm512_cvtepi32_epi8(q5));
        _mm_storeu_si128(d.add(96).cast(), _mm512_cvtepi32_epi8(q6));
        _mm_storeu_si128(d.add(112).cast(), _mm512_cvtepi32_epi8(q7));
        let s0 = _mm512_add_epi32(_mm512_add_epi32(q0, q1), _mm512_add_epi32(q2, q3));
        let s1 = _mm512_add_epi32(_mm512_add_epi32(q4, q5), _mm512_add_epi32(q6, q7));
        return (scale, _mm512_reduce_add_epi32(_mm512_add_epi32(s0, s1)));
    }
    if src.len() == 32 {
        // ggml Q8_0 block: keep both 16-lane vectors in registers so pass 2 never reloads.
        // Put the new value first in `max_ps(val, acc)` so x86 `vmaxps` keeps the finite
        // accumulator on NaN, matching `f32::max` in `quantize_scalar`.
        let x0 = _mm512_loadu_ps(p);
        let x1 = _mm512_loadu_ps(p.add(16));
        let vmax0 = _mm512_max_ps(_mm512_abs_ps(x0), _mm512_setzero_ps());
        let vmax = _mm512_max_ps(_mm512_abs_ps(x1), vmax0);
        let scale = group_scale(_mm512_reduce_max_ps(vmax));
        let inv = _mm512_set1_ps(1.0 / scale);
        let q0 = quant16!(x0, inv);
        let q1 = quant16!(x1, inv);
        _mm_storeu_si128(d.cast(), _mm512_cvtepi32_epi8(q0));
        _mm_storeu_si128(d.add(16).cast(), _mm512_cvtepi32_epi8(q1));
        return (scale, _mm512_reduce_add_epi32(_mm512_add_epi32(q0, q1)));
    }
    let pairs = src.len() / 32;
    let has_tail = src.len() % 32 == 16;
    let mut vmax0 = _mm512_setzero_ps();
    let mut vmax1 = _mm512_setzero_ps();
    for i in 0..pairs {
        vmax0 = _mm512_max_ps(_mm512_abs_ps(_mm512_loadu_ps(p.add(i * 32))), vmax0);
        vmax1 = _mm512_max_ps(_mm512_abs_ps(_mm512_loadu_ps(p.add(i * 32 + 16))), vmax1);
    }
    if has_tail {
        vmax0 = _mm512_max_ps(_mm512_abs_ps(_mm512_loadu_ps(p.add(pairs * 32))), vmax0);
    }
    let scale = group_scale(_mm512_reduce_max_ps(_mm512_max_ps(vmax0, vmax1)));
    let inv = _mm512_set1_ps(1.0 / scale);
    let mut vsum0 = _mm512_setzero_si512();
    let mut vsum1 = _mm512_setzero_si512();
    for i in 0..pairs {
        let q0 = quant16!(_mm512_loadu_ps(p.add(i * 32)), inv);
        let q1 = quant16!(_mm512_loadu_ps(p.add(i * 32 + 16)), inv);
        vsum0 = _mm512_add_epi32(vsum0, q0);
        vsum1 = _mm512_add_epi32(vsum1, q1);
        _mm_storeu_si128(d.add(i * 32).cast(), _mm512_cvtepi32_epi8(q0));
        _mm_storeu_si128(d.add(i * 32 + 16).cast(), _mm512_cvtepi32_epi8(q1));
    }
    if has_tail {
        let q0 = quant16!(_mm512_loadu_ps(p.add(pairs * 32)), inv);
        vsum0 = _mm512_add_epi32(vsum0, q0);
        _mm_storeu_si128(d.add(pairs * 32).cast(), _mm512_cvtepi32_epi8(q0));
    }
    (
        scale,
        _mm512_reduce_add_epi32(_mm512_add_epi32(vsum0, vsum1)),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_avx2(src: &[f32], dst: &mut [i8]) -> (f32, i32) {
    use std::arch::x86_64::*;
    assert!(src.len() == dst.len() && src.len().is_multiple_of(32));
    let p = src.as_ptr();
    let sign = _mm256_set1_ps(-0.0);
    let mut vmax0 = _mm256_setzero_ps();
    let mut vmax1 = _mm256_setzero_ps();
    for i in 0..src.len() / 16 {
        vmax0 = _mm256_max_ps(
            _mm256_andnot_ps(sign, _mm256_loadu_ps(p.add(i * 16))),
            vmax0,
        );
        vmax1 = _mm256_max_ps(
            _mm256_andnot_ps(sign, _mm256_loadu_ps(p.add(i * 16 + 8))),
            vmax1,
        );
    }
    let vmax = _mm256_max_ps(vmax0, vmax1);
    let m4 = _mm_max_ps(
        _mm256_extractf128_ps::<1>(vmax),
        _mm256_castps256_ps128(vmax),
    );
    let m2 = _mm_max_ps(_mm_movehl_ps(m4, m4), m4);
    let m1 = _mm_max_ss(_mm_shuffle_ps::<1>(m2, m2), m2);
    let scale = group_scale(_mm_cvtss_f32(m1));
    let inv = _mm256_set1_ps(1.0 / scale);
    let f_high = _mm256_set1_ps(127.0);
    let f_low = _mm256_set1_ps(-127.0);
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let mut vsum = _mm256_setzero_si256();
    for i in 0..src.len() / 32 {
        let mut q = [_mm256_setzero_si256(); 4];
        for (j, lane) in q.iter_mut().enumerate() {
            let v = _mm256_mul_ps(_mm256_loadu_ps(p.add(i * 32 + j * 8)), inv);
            let ord = _mm256_cmp_ps::<_CMP_ORD_Q>(v, v);
            let clamped = _mm256_min_ps(_mm256_max_ps(v, f_low), f_high);
            *lane = _mm256_and_si256(_mm256_cvtps_epi32(clamped), _mm256_castps_si256(ord));
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
    let s128 = _mm_add_epi32(
        _mm256_castsi256_si128(vsum),
        _mm256_extracti128_si256::<1>(vsum),
    );
    let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
    let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
    (scale, _mm_cvtsi128_si32(s32))
}

/// As `quantize_scalar` (a multiple of 32 values: 128 for ternary and int8
/// layers, 32 for Q8_0); `fcvtns` also rounds half to even.
#[cfg(target_arch = "aarch64")]
unsafe fn quantize_neon(src: &[f32], dst: &mut [i8]) -> (f32, i32) {
    use std::arch::aarch64::*;
    assert!(src.len() == dst.len() && src.len().is_multiple_of(32));
    let p = src.as_ptr();
    let mut vmax0 = vdupq_n_f32(0.0);
    let mut vmax1 = vdupq_n_f32(0.0);
    let mut vmax2 = vdupq_n_f32(0.0);
    let mut vmax3 = vdupq_n_f32(0.0);
    for i in 0..src.len() / 16 {
        let a0 = vabsq_f32(vld1q_f32(p.add(i * 16)));
        let a1 = vabsq_f32(vld1q_f32(p.add(i * 16 + 4)));
        let a2 = vabsq_f32(vld1q_f32(p.add(i * 16 + 8)));
        let a3 = vabsq_f32(vld1q_f32(p.add(i * 16 + 12)));
        // `a > vmax` is false for both qNaN and sNaN (`vmaxnm` propagates sNaN).
        vmax0 = vbslq_f32(vcgtq_f32(a0, vmax0), a0, vmax0);
        vmax1 = vbslq_f32(vcgtq_f32(a1, vmax1), a1, vmax1);
        vmax2 = vbslq_f32(vcgtq_f32(a2, vmax2), a2, vmax2);
        vmax3 = vbslq_f32(vcgtq_f32(a3, vmax3), a3, vmax3);
    }
    let vmax01 = vbslq_f32(vcgtq_f32(vmax0, vmax1), vmax0, vmax1);
    let vmax23 = vbslq_f32(vcgtq_f32(vmax2, vmax3), vmax2, vmax3);
    let vmax = vbslq_f32(vcgtq_f32(vmax01, vmax23), vmax01, vmax23);
    let scale = group_scale(vmaxvq_f32(vmax));
    let inv = vdupq_n_f32(1.0 / scale);
    let (limit, low) = (vdupq_n_s32(127), vdupq_n_s32(-127));
    let mut vsum0 = vdupq_n_s32(0);
    let mut vsum1 = vdupq_n_s32(0);
    for i in 0..src.len() / 16 {
        let mut q = [vdupq_n_s32(0); 4];
        for (j, lane) in q.iter_mut().enumerate() {
            let v = vmulq_f32(vld1q_f32(p.add(i * 16 + j * 4)), inv);
            *lane = vmaxq_s32(vminq_s32(vcvtnq_s32_f32(v), limit), low);
        }
        vsum0 = vaddq_s32(vsum0, vaddq_s32(q[0], q[1]));
        vsum1 = vaddq_s32(vsum1, vaddq_s32(q[2], q[3]));
        let first = vcombine_s16(vmovn_s32(q[0]), vmovn_s32(q[1]));
        let second = vcombine_s16(vmovn_s32(q[2]), vmovn_s32(q[3]));
        vst1q_s8(
            dst.as_mut_ptr().add(i * 16),
            vcombine_s8(vmovn_s16(first), vmovn_s16(second)),
        );
    }
    (scale, vaddvq_s32(vaddq_s32(vsum0, vsum1)))
}

#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: tasks write disjoint output elements; the buffer outlives the parallel scope.
unsafe impl Send for OutPtr {}
// SAFETY: as for `Send`: tasks write disjoint output elements.
unsafe impl Sync for OutPtr {}

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

    #[test]
    fn quantized_matches_scalar_and_clamps_special_floats() {
        let mut seed = 0x51a7_0001_beef_c0deu64;
        let isa = Isa::detect();
        let k = 256;
        let m = 6;
        let mut x = vec![0.0f32; m * k];
        // Row 0: normal finite range.
        for v in &mut x[..k] {
            *v = lcg(&mut seed) * 5.0;
        }
        // Row 1: zeros and -0.0.
        for (i, v) in x[k..2 * k].iter_mut().enumerate() {
            *v = if i % 2 == 0 { 0.0 } else { -0.0 };
        }
        // Row 2: NaNs interspersed with finite values.
        for (i, v) in x[2 * k..3 * k].iter_mut().enumerate() {
            *v = if i % 11 == 0 {
                f32::NAN
            } else {
                lcg(&mut seed) * 3.0
            };
        }
        // Row 3: +Inf and -Inf mixed with finite values.
        for (i, v) in x[3 * k..4 * k].iter_mut().enumerate() {
            *v = match i % 17 {
                0 => f32::INFINITY,
                1 => f32::NEG_INFINITY,
                _ => lcg(&mut seed),
            };
        }
        // Row 4: subnormals and zeros.
        for (i, v) in x[4 * k..5 * k].iter_mut().enumerate() {
            *v = match i % 3 {
                0 => 1e-40,
                1 => -1e-40,
                _ => 0.0,
            };
        }
        // Row 5: huge finite values near f32::MAX.
        for (i, v) in x[5 * k..6 * k].iter_mut().enumerate() {
            *v = if i % 2 == 0 { 1e38 } else { -1e38 };
        }

        for group in [32usize, 128] {
            let groups = k / group;
            for half_scales in [false, true] {
                let got = Quantized::grouped(isa, &x, m, k, group, half_scales);
                let want = Quantized::grouped(Isa::Scalar, &x, m, k, group, half_scales);
                assert_eq!(got.m, m);
                assert_eq!(
                    &got.values[..m * k],
                    &want.values[..m * k],
                    "values mismatch group={group} half={half_scales}"
                );
                assert_eq!(
                    &got.sums[..m * groups],
                    &want.sums[..m * groups],
                    "sums mismatch group={group} half={half_scales}"
                );
                for (i, (&gs, &ws)) in got.scales[..m * groups]
                    .iter()
                    .zip(&want.scales[..m * groups])
                    .enumerate()
                {
                    assert_eq!(
                        gs.to_bits(),
                        ws.to_bits(),
                        "scale mismatch at {i} group={group} half={half_scales}: got={gs} want={ws}"
                    );
                }
                for row in 0..m {
                    for g in 0..groups {
                        let span = &got.values[row * k + g * group..][..group];
                        for &c in span {
                            assert!(
                                (-127..=127).contains(&c),
                                "code {c} out of [-127, 127] at row={row} g={g}"
                            );
                        }
                        let sum: i32 = span.iter().map(|&c| i32::from(c)).sum();
                        assert_eq!(got.sums[row * groups + g], sum);
                    }
                }
                // NaN lanes in row 2 must quantize to 0.
                for i in (0..k).step_by(11) {
                    assert_eq!(got.values[2 * k + i], 0);
                }
            }
        }
    }

    #[test]
    fn ternary_forward_matches_scalar() {
        let mut seed = 0xabcd_ef01_2345_6789u64;
        let isa = Isa::detect();
        for (n, k) in [(32usize, 128usize), (64, 256), (64, 1024), (64, 4096)] {
            let groups = k / GROUP;
            let codes: Vec<u8> = (0..n * k)
                .map(|_| ((lcg(&mut seed) + 1.0) * 1.5).clamp(0.0, 2.0) as u8)
                .collect();
            let scales: Vec<f32> = (0..n * groups)
                .map(|_| lcg(&mut seed).abs() + 0.05)
                .collect();
            let simd = Ternary::from_rows(isa, &scales, n, k, |r, dst| {
                dst.copy_from_slice(&codes[r * k..(r + 1) * k]);
            });
            let scalar = Ternary::from_rows(Isa::Scalar, &scales, n, k, |r, dst| {
                dst.copy_from_slice(&codes[r * k..(r + 1) * k]);
            });
            let ms: &[usize] = if k >= 2048 {
                &[1, 6, 7, 49]
            } else {
                &[1, 2, 3, 4, 5, 6, 7, 13]
            };
            let tol = (k as f32).sqrt() * 2e-4 + 1e-3;
            for &m in ms {
                let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed) * 2.0).collect();
                let got = simd.forward(&x, m);
                let want = scalar.forward(&x, m);
                assert_eq!(got.len(), want.len());
                for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                    let diff = (g - w).abs();
                    assert!(
                        diff <= tol,
                        "ternary n={n} k={k} m={m} at {i}: got={g} want={w} diff={diff}"
                    );
                }
            }
        }
    }
}
