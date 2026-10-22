//! Ternary matrix product with 8-bit activations.
//!
//! Weights stay as codes 0, 1, 2 (`w = scale * (code - 1)`). Each 128-value
//! group of an activation row is quantized to i8 with its own scale. A group
//! dot product is then `sum(code * q) - sum(q)`, done in i32 by
//! `vpdpbusd` (AVX-512 VNNI) or `vpmaddubsw` (AVX2), and scaled into f32 by
//! the weight scale and the activation scale of that group.
//!
//! The SIMD paths keep four codes per byte. A short take reads every weight
//! once from memory, so at one byte per code the product waits on memory;
//! two shifts and a mask per weight vector cost less than the bytes they save.

use super::buf::Buf;
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
}

impl Isa {
    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => "avx2",
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni => "avx512-vnni",
        }
    }

    /// Rows per packed block.
    pub(super) fn rows(self) -> usize {
        match self {
            Isa::Scalar => 1,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => 8,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni => 16,
        }
    }

    /// The fastest path this CPU runs. `TRIT_ISA=scalar|avx2` forces a slower one.
    pub fn detect() -> Self {
        let forced = std::env::var("TRIT_ISA").unwrap_or_default();
        #[cfg(target_arch = "x86_64")]
        {
            let vnni = is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vnni");
            let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
            match forced.as_str() {
                "scalar" => return Isa::Scalar,
                "avx2" if avx2 => return Isa::Avx2,
                _ => {}
            }
            if vnni {
                return Isa::Avx512Vnni;
            }
            if avx2 {
                return Isa::Avx2;
            }
        }
        let _ = forced;
        Isa::Scalar
    }
}

/// Codes packed for one instruction set. Scalar: `[n][k / 4][1][4]`, one
/// code per byte. SIMD: `[n / rows][k / 16][rows][4]` bytes, where bits
/// `2s..2s+2` of a byte hold the code of k4 step `4 * (k / 16 index) + s`.
pub struct Ternary {
    pub isa: Isa,
    pub n: usize,
    pub k: usize,
    codes: Vec<u8>,
    /// `[n / rows][k / GROUP][rows]`.
    scales: Vec<f32>,
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
                for (five, byte) in codes.chunks_mut(5).zip(packed_row) {
                    five.copy_from_slice(&DIGITS[usize::from(*byte)][..five.len()]);
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
        let rows = isa.rows();
        // The SIMD tiles need whole blocks; other shapes run the scalar path.
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
                // Byte j of a row's 16-code span holds codes j, 4 + j, 8 + j, 12 + j.
                for k16 in 0..k / 16 {
                    for r in 0..rows {
                        let src = &codes[r * k + k16 * 16..][..16];
                        let dst = ((block * (k / 16) + k16) * rows + r) * 4;
                        for j in 0..4 {
                            packed[dst + j] =
                                src[j] | src[4 + j] << 2 | src[8 + j] << 4 | src[12 + j] << 6;
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
        Self {
            isa,
            n,
            k,
            codes: packed,
            scales: packed_scales,
        }
    }

    /// `x` is `m x k` f32; returns `m x n`.
    pub fn forward(&self, x: &[f32], m: usize) -> Buf {
        let mut out = Buf::scratch(m * self.n);
        let q = Quantized::new(self.isa, x, m, self.k);
        let ptr = OutPtr(out.as_mut_ptr());
        let rows = self.isa.rows();
        // Two blocks per task: the SIMD tiles cover 2 * rows outputs.
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
}

/// The five base-3 digits of every packed byte, least significant first.
/// A table copy per byte replaces a division per code: model load spent
/// most of its time on those divisions.
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
        let groups = k / GROUP;
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
                    let span = g * GROUP..(g + 1) * GROUP;
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
                        Isa::Scalar => quantize_scalar(&src[span.clone()], &mut values[span]),
                    };
                    scales[g] = scale;
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

/// One 128-value group to i8, rounding half to even like `cvtps2dq`.
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
    for i in 0..GROUP / 16 {
        vmax = _mm512_max_ps(vmax, _mm512_abs_ps(_mm512_loadu_ps(p.add(i * 16))));
    }
    let scale = group_scale(_mm512_reduce_max_ps(vmax));
    let inv = _mm512_set1_ps(1.0 / scale);
    let limit = _mm512_set1_epi32(127);
    let mut vsum = _mm512_setzero_si512();
    for i in 0..GROUP / 16 {
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
    for i in 0..GROUP / 8 {
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
    for i in 0..GROUP / 32 {
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

#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: tasks write disjoint output elements; the buffer outlives the parallel scope.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}
