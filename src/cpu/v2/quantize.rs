//! Activations as i8 for the integer products. Each group of a frame has
//! its own scale, as in the first kernels, so the codes are the same.
//!
//! The sum of a group is stored as the value its dot product starts from
//! (`-bias * sum`), so the product has no correction step.
//!
//! Tested and not kept: the frames of a tile stored side by side
//! (`[inputs / 4][frames][4]`), so that a tile reads one stream. The copy
//! into that order cost more than the stream gave.

use crate::cpu::team;
use std::cell::RefCell;

/// The instructions that round a group. Every one gives the same codes for
/// finite inputs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Rounder {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

pub(super) struct Quantized {
    /// `m x k` codes.
    pub values: Vec<i8>,
    /// `m x groups`.
    pub scales: Vec<f32>,
    /// `m x groups`: `-bias * sum` of the group's codes.
    pub starts: Vec<i32>,
}

thread_local! {
    /// The buffers of the calling thread's last product: a new one would
    /// fault in new pages.
    static LAST: RefCell<(Vec<i8>, Vec<f32>, Vec<i32>)> = const {
        RefCell::new((Vec::new(), Vec::new(), Vec::new()))
    };
}

impl Drop for Quantized {
    fn drop(&mut self) {
        let buffers = (
            std::mem::take(&mut self.values),
            std::mem::take(&mut self.scales),
            std::mem::take(&mut self.starts),
        );
        LAST.with(|cell| *cell.borrow_mut() = buffers);
    }
}

/// What one product asks of the quantizer.
#[derive(Clone, Copy)]
pub(super) struct Plan {
    pub rounder: Rounder,
    /// Inputs per scale: a multiple of 16 (128, or 32 for ggml Q8_0).
    pub group: usize,
    /// The product computes `(w + bias) . x`, so it starts at `-bias * sum(x)`.
    pub bias: i32,
    /// Round each scale to f16 after the codes are made (ggml Q8_0).
    pub half_scales: bool,
}

/// Frames per task: a short take still splits, a frame is never shared.
const FRAMES: usize = 8;

/// `x` is `m x k` f32, with `k` a multiple of `plan.group`.
pub(super) fn quantize(plan: Plan, x: &[f32], m: usize, k: usize) -> Quantized {
    let Plan {
        rounder,
        group,
        bias,
        half_scales,
    } = plan;
    assert!(x.len() >= m * k && k.is_multiple_of(group) && group.is_multiple_of(16));
    let groups = k / group;
    let (mut values, mut scales, mut starts) =
        LAST.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    // Every value that the product reads is written below, so a buffer that
    // is long enough is used as it is.
    if values.len() < m * k {
        values.resize(m * k, 0);
    }
    if scales.len() < m * groups {
        scales.resize(m * groups, 0.0);
        starts.resize(m * groups, 0);
    }
    let (all_values, all_scales, all_starts) = (
        team::SendPtr::new(&mut values),
        team::SendPtr::new(&mut scales),
        team::SendPtr::new(&mut starts),
    );
    team::for_each(m.div_ceil(FRAMES), |task| {
        for frame in task * FRAMES..((task + 1) * FRAMES).min(m) {
            // SAFETY: each frame belongs to one task; the buffers outlive the step.
            let (values, scales, starts) = unsafe {
                (
                    all_values.slice_mut(frame * k, k),
                    all_scales.slice_mut(frame * groups, groups),
                    all_starts.slice_mut(frame * groups, groups),
                )
            };
            let row = &x[frame * k..][..k];
            for (g, (src, codes)) in row
                .chunks_exact(group)
                .zip(values.chunks_exact_mut(group))
                .enumerate()
            {
                let (scale, sum) = match rounder {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: the path was detected from the CPU.
                    Rounder::Avx512 => unsafe { round_avx512(src, codes) },
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: the path was detected from the CPU.
                    Rounder::Avx2 => unsafe { round_avx2(src, codes) },
                    #[cfg(target_arch = "aarch64")]
                    // SAFETY: NEON is part of every arm64 CPU.
                    Rounder::Neon => unsafe { round_neon(src, codes) },
                    Rounder::Scalar => round_scalar(src, codes),
                };
                scales[g] = if half_scales {
                    crate::cpu::tensors::f16_to_f32(crate::cpu::tensors::f32_to_f16(scale))
                } else {
                    scale
                };
                starts[g] = (-bias).wrapping_mul(sum);
            }
        }
    });
    Quantized {
        values,
        scales,
        starts,
    }
}

/// The scale of a group with largest magnitude `peak`: its codes fill -127..=127.
fn step(peak: f32) -> f32 {
    if peak > 0.0 {
        peak / 127.0
    } else {
        1.0
    }
}

/// `cvtps2dq`, then the limits: half to even, and a value that is no i32
/// (NaN) becomes `i32::MIN`, as the instruction gives it.
fn code_of(scaled: f32) -> i32 {
    let rounded = scaled.round_ties_even();
    let whole = if (-2_147_483_648.0..2_147_483_648.0).contains(&rounded) {
        rounded as i32
    } else {
        i32::MIN
    };
    whole.clamp(-127, 127)
}

/// One group to i8: the scale and the sum of the codes.
fn round_scalar(src: &[f32], dst: &mut [i8]) -> (f32, i32) {
    let scale = step(src.iter().fold(0.0f32, |peak, v| peak.max(v.abs())));
    let inverse = 1.0 / scale;
    let mut sum = 0i32;
    for (d, v) in dst.iter_mut().zip(src) {
        let code = code_of(v * inverse);
        *d = code as i8;
        sum += code;
    }
    (scale, sum)
}

/// As `round_scalar`; the group is a multiple of 16 values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn round_avx512(src: &[f32], dst: &mut [i8]) -> (f32, i32) {
    use std::arch::x86_64::*;
    let (chunks, _) = src.as_chunks::<16>();
    let mut peak = _mm512_setzero_ps();
    for chunk in chunks {
        peak = _mm512_max_ps(peak, _mm512_abs_ps(_mm512_loadu_ps(chunk.as_ptr())));
    }
    let scale = step(_mm512_reduce_max_ps(peak));
    let inverse = _mm512_set1_ps(1.0 / scale);
    let (high, low) = (_mm512_set1_epi32(127), _mm512_set1_epi32(-127));
    let mut sum = _mm512_setzero_si512();
    for (chunk, out) in chunks.iter().zip(dst.as_chunks_mut::<16>().0) {
        let scaled = _mm512_mul_ps(_mm512_loadu_ps(chunk.as_ptr()), inverse);
        let code = _mm512_max_epi32(_mm512_min_epi32(_mm512_cvtps_epi32(scaled), high), low);
        sum = _mm512_add_epi32(sum, code);
        _mm_storeu_si128(out.as_mut_ptr().cast(), _mm512_cvtepi32_epi8(code));
    }
    (scale, _mm512_reduce_add_epi32(sum))
}

/// As `round_scalar`; the group is a multiple of 8 values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn round_avx2(src: &[f32], dst: &mut [i8]) -> (f32, i32) {
    use std::arch::x86_64::*;
    let (chunks, _) = src.as_chunks::<8>();
    let sign = _mm256_set1_ps(-0.0);
    let mut peak = _mm256_setzero_ps();
    for chunk in chunks {
        peak = _mm256_max_ps(
            peak,
            _mm256_andnot_ps(sign, _mm256_loadu_ps(chunk.as_ptr())),
        );
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), peak);
    let scale = step(lanes.iter().fold(0.0f32, |peak, v| peak.max(*v)));
    let inverse = _mm256_set1_ps(1.0 / scale);
    let (high, low) = (_mm256_set1_epi32(127), _mm256_set1_epi32(-127));
    // The low byte of each of the four values of a 128-bit half.
    let low_bytes = _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let mut sum = _mm256_setzero_si256();
    for (chunk, out) in chunks.iter().zip(dst.as_chunks_mut::<8>().0) {
        let scaled = _mm256_mul_ps(_mm256_loadu_ps(chunk.as_ptr()), inverse);
        let code = _mm256_max_epi32(_mm256_min_epi32(_mm256_cvtps_epi32(scaled), high), low);
        sum = _mm256_add_epi32(sum, code);
        let bytes = _mm256_shuffle_epi8(code, low_bytes);
        let eight = _mm_unpacklo_epi32(
            _mm256_castsi256_si128(bytes),
            _mm256_extracti128_si256::<1>(bytes),
        );
        _mm_storel_epi64(out.as_mut_ptr().cast(), eight);
    }
    let mut sums = [0i32; 8];
    _mm256_storeu_si256(sums.as_mut_ptr().cast(), sum);
    (scale, sums.iter().sum())
}

/// As `round_scalar`; the group is a multiple of 16 values. `fcvtns` also
/// rounds half to even.
#[cfg(target_arch = "aarch64")]
unsafe fn round_neon(src: &[f32], dst: &mut [i8]) -> (f32, i32) {
    use std::arch::aarch64::*;
    let (chunks, _) = src.as_chunks::<16>();
    let mut peak = vdupq_n_f32(0.0);
    for chunk in chunks {
        for four in chunk.as_chunks::<4>().0 {
            peak = vmaxq_f32(peak, vabsq_f32(vld1q_f32(four.as_ptr())));
        }
    }
    let scale = step(vmaxvq_f32(peak));
    let inverse = vdupq_n_f32(1.0 / scale);
    let (high, low) = (vdupq_n_s32(127), vdupq_n_s32(-127));
    let mut sum = vdupq_n_s32(0);
    for (chunk, out) in chunks.iter().zip(dst.as_chunks_mut::<16>().0) {
        let mut codes = [vdupq_n_s32(0); 4];
        for (code, four) in codes.iter_mut().zip(chunk.as_chunks::<4>().0) {
            let scaled = vmulq_f32(vld1q_f32(four.as_ptr()), inverse);
            *code = vmaxq_s32(vminq_s32(vcvtnq_s32_f32(scaled), high), low);
            sum = vaddq_s32(sum, *code);
        }
        let first = vcombine_s16(vmovn_s32(codes[0]), vmovn_s32(codes[1]));
        let second = vcombine_s16(vmovn_s32(codes[2]), vmovn_s32(codes[3]));
        vst1q_s8(
            out.as_mut_ptr(),
            vcombine_s8(vmovn_s16(first), vmovn_s16(second)),
        );
    }
    (scale, vaddvq_s32(sum))
}
