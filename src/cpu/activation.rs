//! Element-wise exp, SiLU, sigmoid, tanh, and GELU on whole slices. The Windows target
//! has no vector math library, so `f32::exp` runs one value at a time. These
//! run one polynomial (Cephes `expf`, within about 2 ulp) on 16 or 8 lanes,
//! and the scalar path runs the same polynomial.

use super::fgemm::{level, Level};

pub(super) const LOG2E: f32 = std::f32::consts::LOG2_E;
/// ln 2 split in two, so `n * LN2_HI` is exact for every `n` in range.
pub(super) const LN2_HI: f32 = 0.693_359_4;
pub(super) const LN2_LO: f32 = -2.121_944_4e-4;
/// `exp` of these stays a normal f32, and `n` stays in -126..=127.
pub(super) const LOW: f32 = -87.3;
pub(super) const HIGH: f32 = 88.0;
/// `erfc(z) = t * exp(-z^2 + P(t))`, `t = 1 / (1 + z / 2)`, for `z >= 0`:
/// Numerical Recipes' `erfcc`, within 1.2e-7 relative everywhere, so GELU
/// stays exact also where `x` is far below 0 and `erfc` is small.
pub(super) const ERFC: [f32; 10] = [
    0.170_872_77,
    -0.822_152_23,
    1.488_515_9,
    -1.135_204,
    0.278_868_07,
    -0.186_288_06,
    0.096_784_18,
    0.374_091_96,
    1.000_023_7,
    -1.265_512_2,
];
pub(super) const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
pub(super) const POLY: [f32; 6] = [
    1.987_569_1e-4,
    1.398_199_9e-3,
    8.333_452e-3,
    4.166_579_6e-2,
    0.166_666_65,
    0.5,
];

#[derive(Clone, Copy)]
pub(super) enum Op {
    Exp,
    Silu,
    Sigmoid,
    Tanh,
    Gelu,
}

pub fn exp(values: &mut [f32]) {
    apply(values, Op::Exp);
}

/// `v = v / (1 + e^-v)`.
pub fn silu(values: &mut [f32]) {
    apply(values, Op::Silu);
}

/// `v = 1 / (1 + e^-v)`.
pub fn sigmoid(values: &mut [f32]) {
    apply(values, Op::Sigmoid);
}

/// `v = tanh(v)`, as `2 / (1 + e^-2v) - 1`: within 2e-7 of tanh, absolute.
pub fn tanh(values: &mut [f32]) {
    apply(values, Op::Tanh);
}

/// `v = v * (1 + erf(v / sqrt 2)) / 2`, the exact GELU (PyTorch's default),
/// as `v * erfc(-v / sqrt 2) / 2`.
pub fn gelu(values: &mut [f32]) {
    apply(values, Op::Gelu);
}

/// The largest of `values`; minus infinity for no values. A NaN is not a
/// value, as for `f32::max`. A maximum does not depend on the order of its
/// values, so it comes from lanes; one chain over the values takes 2 to 3
/// cycles for each value. Only a zero can come out with the other sign.
pub fn max(values: &[f32]) -> f32 {
    match level() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the level was detected from the CPU.
        Level::Avx512 => unsafe { avx512::max(values) },
        _ => {
            let (chunks, tail) = values.as_chunks::<16>();
            let mut best = [f32::NEG_INFINITY; 16];
            for chunk in chunks {
                for (best, value) in best.iter_mut().zip(chunk) {
                    // One instruction for each 4 or 8 values; `f32::max` is 5.
                    if *value > *best {
                        *best = *value;
                    }
                }
            }
            let lanes = best.iter().chain(tail).copied();
            lanes.fold(f32::NEG_INFINITY, f32::max)
        }
    }
}

/// The rows that `exp_sums` takes side by side. The sum of a row is a chain:
/// each add waits for the one before it. A CPU runs about 8 chains at once.
pub const SIDE: usize = 8;

/// `v = exp(v - shifts[r])` for each value of row `r` of `rows` (rows of
/// `width`), and each new value added to `sums[r]`, in the order of the row:
/// the bits of an `exp` pass and then a sum pass, without the wait between
/// them. A softmax is this with the maximum of each row as its shift.
pub fn exp_sums(rows: &mut [f32], width: usize, shifts: &[f32], sums: &mut [f32]) {
    assert!(rows.len() == shifts.len() * width && sums.len() == shifts.len());
    if width == 0 {
        return;
    }
    let groups = rows.chunks_mut(SIDE * width);
    for ((group, shifts), sums) in groups.zip(shifts.chunks(SIDE)).zip(sums.chunks_mut(SIDE)) {
        match level() {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: the level was detected from the CPU.
            Level::Avx512 => unsafe { avx512::exp_sums(group, width, shifts, sums) },
            _ => {
                for (row, shift) in group.chunks_exact_mut(width).zip(shifts) {
                    row.iter_mut().for_each(|v| *v -= shift);
                    apply(row, Op::Exp);
                }
                add_columns(group, width, sums);
            }
        }
    }
}

/// Adds the values of each row of `group` (rows of `width`) to the sum of
/// that row: for each column, one add of each row, so the chains of the rows
/// run side by side.
fn add_columns(group: &[f32], width: usize, sums: &mut [f32]) {
    assert!(group.len() == sums.len() * width);
    if let Ok(kept) = <&mut [f32; SIDE]>::try_from(&mut *sums) {
        // A constant count of sums in a local array: they stay in registers.
        let mut sum = *kept;
        for column in 0..width {
            for (r, sum) in sum.iter_mut().enumerate() {
                // SAFETY: `group` has `SIDE` rows of `width`, and `column < width`.
                *sum += unsafe { *group.get_unchecked(r * width + column) };
            }
        }
        *kept = sum;
    } else {
        for column in 0..width {
            for (r, sum) in sums.iter_mut().enumerate() {
                *sum += group[r * width + column];
            }
        }
    }
}

fn apply(values: &mut [f32], op: Op) {
    match level() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the level was detected from the CPU.
        Level::Avx512 => unsafe { avx512::apply(values, op) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the level was detected from the CPU.
        Level::Avx2 => unsafe { avx2::apply(values, op) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of every arm64 CPU.
        Level::Neon => unsafe { neon::apply(values, op) },
        Level::Scalar if super::v2::float_enabled() => super::v2::activation::apply(values, op),
        Level::Scalar => values.iter_mut().for_each(|v| *v = scalar(*v, op)),
    }
}

fn scalar(v: f32, op: Op) -> f32 {
    match op {
        Op::Exp => exp_scalar(v),
        Op::Silu => v / (1.0 + exp_scalar(-v)),
        Op::Sigmoid => 1.0 / (1.0 + exp_scalar(-v)),
        Op::Tanh => 2.0 / (1.0 + exp_scalar(-2.0 * v)) - 1.0,
        Op::Gelu => {
            let z = v.abs() * FRAC_1_SQRT_2;
            let t = 1.0 / (1.0 + 0.5 * z);
            let p = ERFC.iter().fold(0.0f32, |p, c| p * t + c);
            let tail = t * exp_scalar(p - z * z);
            0.5 * v * if v >= 0.0 { 2.0 - tail } else { tail }
        }
    }
}

fn exp_scalar(x: f32) -> f32 {
    let x = x.clamp(LOW, HIGH);
    let n = (x * LOG2E).round_ties_even();
    let r = x - n * LN2_HI - n * LN2_LO;
    let p = POLY.iter().fold(0.0f32, |p, c| p * r + c);
    let two_n = f32::from_bits(((n as i32 + 127) as u32) << 23);
    (p * r * r + r + 1.0) * two_n
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::{Op, ERFC, FRAC_1_SQRT_2, HIGH, LN2_HI, LN2_LO, LOG2E, LOW, POLY, SIDE};
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f")]
    pub unsafe fn apply(values: &mut [f32], op: Op) {
        let (chunks, tail) = values.as_chunks_mut::<16>();
        for chunk in chunks {
            let v = _mm512_loadu_ps(chunk.as_ptr());
            _mm512_storeu_ps(chunk.as_mut_ptr(), compute(v, op));
        }
        if !tail.is_empty() {
            let mask: __mmask16 = ((1u32 << tail.len()) - 1) as u16;
            let v = _mm512_maskz_loadu_ps(mask, tail.as_ptr());
            _mm512_mask_storeu_ps(tail.as_mut_ptr(), mask, compute(v, op));
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn max(values: &[f32]) -> f32 {
        let (chunks, tail) = values.as_chunks::<16>();
        let (fours, rest) = chunks.as_chunks::<4>();
        // `vmaxps` gives its second operand when the first is NaN: the best so far.
        let mut best = [_mm512_set1_ps(f32::NEG_INFINITY); 4];
        for four in fours {
            for (best, chunk) in best.iter_mut().zip(four) {
                *best = _mm512_max_ps(_mm512_loadu_ps(chunk.as_ptr()), *best);
            }
        }
        for chunk in rest {
            best[0] = _mm512_max_ps(_mm512_loadu_ps(chunk.as_ptr()), best[0]);
        }
        if !tail.is_empty() {
            let mask: __mmask16 = ((1u32 << tail.len()) - 1) as u16;
            let v = _mm512_mask_loadu_ps(best[0], mask, tail.as_ptr());
            best[0] = _mm512_max_ps(v, best[0]);
        }
        let best = _mm512_max_ps(
            _mm512_max_ps(best[0], best[1]),
            _mm512_max_ps(best[2], best[3]),
        );
        _mm512_reduce_max_ps(best)
    }

    /// `super::exp_sums` for the rows of `group`, at most `SIDE`. The sums
    /// of the rows are the 8 lanes of one register: the 16 exponentials of
    /// each row turn into 16 columns (one value of each row), and the
    /// columns are added in sequence, so each row adds its values in order.
    /// One add for each value from memory took 1.5 cycles; this takes 48
    /// operations for 128 values, and the next `exp` runs beside the adds.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn exp_sums(group: &mut [f32], width: usize, shifts: &[f32], sums: &mut [f32]) {
        let rows = shifts.len();
        assert!(rows <= SIDE && group.len() == rows * width && sums.len() == rows);
        let mut held = [0.0f32; SIDE];
        held[..rows].copy_from_slice(sums);
        let mut sum = _mm256_loadu_ps(held.as_ptr());
        // A row that is not there is zeros, and its sum is not kept.
        let mut e = [_mm512_setzero_ps(); SIDE];
        let (steps, tail) = (width / 16, width % 16);
        for step in 0..steps {
            for r in 0..rows {
                // SAFETY: row `r` has `width` values, and `step * 16 + 16 <= width`.
                let at = group.as_mut_ptr().add(r * width + step * 16);
                e[r] = exp(_mm512_sub_ps(
                    _mm512_loadu_ps(at),
                    _mm512_set1_ps(shifts[r]),
                ));
                _mm512_storeu_ps(at, e[r]);
            }
            for column in columns(&e) {
                sum = _mm256_add_ps(sum, column);
            }
        }
        if tail > 0 {
            let mask: __mmask16 = ((1u32 << tail) - 1) as u16;
            for r in 0..rows {
                // SAFETY: the mask keeps the load and the store inside row `r`.
                let at = group.as_mut_ptr().add(r * width + steps * 16);
                let v = _mm512_maskz_loadu_ps(mask, at);
                e[r] = exp(_mm512_sub_ps(v, _mm512_set1_ps(shifts[r])));
                _mm512_mask_storeu_ps(at, mask, e[r]);
            }
            for column in &columns(&e)[..tail] {
                sum = _mm256_add_ps(sum, *column);
            }
        }
        _mm256_storeu_ps(held.as_mut_ptr(), sum);
        sums.copy_from_slice(&held[..rows]);
    }

    /// The 16 columns of 8 rows of 16 values: lane `r` of column `j` is
    /// value `j` of row `r`.
    #[target_feature(enable = "avx512f")]
    unsafe fn columns(rows: &[__m512; SIDE]) -> [__m256; 16] {
        // Each 128-bit part `a` of a register is on its own in these steps.
        // Two rows: values 0 and 1 of both, then values 2 and 3.
        let mut two = [_mm512_setzero_ps(); 8];
        for pair in 0..4 {
            two[2 * pair] = _mm512_unpacklo_ps(rows[2 * pair], rows[2 * pair + 1]);
            two[2 * pair + 1] = _mm512_unpackhi_ps(rows[2 * pair], rows[2 * pair + 1]);
        }
        // Four rows: `four[4 * half + b]` has value `4 * a + b` of rows 0 to 3
        // (`half` 0) or 4 to 7 (`half` 1) in its part `a`.
        let mut four = [_mm512_setzero_ps(); 8];
        for half in 0..2 {
            let t = &two[4 * half..];
            four[4 * half] = _mm512_shuffle_ps::<0x44>(t[0], t[2]);
            four[4 * half + 1] = _mm512_shuffle_ps::<0xEE>(t[0], t[2]);
            four[4 * half + 2] = _mm512_shuffle_ps::<0x44>(t[1], t[3]);
            four[4 * half + 3] = _mm512_shuffle_ps::<0xEE>(t[1], t[3]);
        }
        // Eight rows: parts 0 and 1 of both halves side by side, then 2 and 3.
        let first = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
        let second =
            _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
        let high = |v: __m512| _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(v)));
        let mut columns = [_mm256_setzero_ps(); 16];
        for b in 0..4 {
            let low = _mm512_permutex2var_ps(four[b], first, four[4 + b]);
            let rest = _mm512_permutex2var_ps(four[b], second, four[4 + b]);
            columns[b] = _mm512_castps512_ps256(low);
            columns[4 + b] = high(low);
            columns[8 + b] = _mm512_castps512_ps256(rest);
            columns[12 + b] = high(rest);
        }
        columns
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn compute(v: __m512, op: Op) -> __m512 {
        let one = _mm512_set1_ps(1.0);
        match op {
            Op::Exp => exp(v),
            Op::Silu => _mm512_div_ps(v, _mm512_add_ps(one, exp(neg(v)))),
            Op::Sigmoid => _mm512_div_ps(one, _mm512_add_ps(one, exp(neg(v)))),
            Op::Tanh => {
                let two = _mm512_set1_ps(2.0);
                let e = exp(neg(_mm512_mul_ps(two, v)));
                _mm512_sub_ps(_mm512_div_ps(two, _mm512_add_ps(one, e)), one)
            }
            Op::Gelu => {
                let z = _mm512_mul_ps(_mm512_abs_ps(v), _mm512_set1_ps(FRAC_1_SQRT_2));
                let t = _mm512_div_ps(one, _mm512_fmadd_ps(z, _mm512_set1_ps(0.5), one));
                let mut p = _mm512_setzero_ps();
                for c in ERFC {
                    p = _mm512_fmadd_ps(p, t, _mm512_set1_ps(c));
                }
                let tail = _mm512_mul_ps(t, exp(_mm512_fnmadd_ps(z, z, p)));
                let positive = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(v, _mm512_setzero_ps());
                let erfc =
                    _mm512_mask_blend_ps(positive, tail, _mm512_sub_ps(_mm512_set1_ps(2.0), tail));
                _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.5), v), erfc)
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn neg(v: __m512) -> __m512 {
        _mm512_sub_ps(_mm512_setzero_ps(), v)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn exp(x: __m512) -> __m512 {
        let x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(LOW)), _mm512_set1_ps(HIGH));
        let n_int = _mm512_cvtps_epi32(_mm512_mul_ps(x, _mm512_set1_ps(LOG2E)));
        let n = _mm512_cvtepi32_ps(n_int);
        let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_HI), x);
        let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_LO), r);
        let mut p = _mm512_setzero_ps();
        for c in POLY {
            p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(c));
        }
        let y = _mm512_fmadd_ps(
            p,
            _mm512_mul_ps(r, r),
            _mm512_add_ps(r, _mm512_set1_ps(1.0)),
        );
        let two_n = _mm512_slli_epi32::<23>(_mm512_add_epi32(n_int, _mm512_set1_epi32(127)));
        _mm512_mul_ps(y, _mm512_castsi512_ps(two_n))
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{Op, ERFC, FRAC_1_SQRT_2, HIGH, LN2_HI, LN2_LO, LOG2E, LOW, POLY};
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn apply(values: &mut [f32], op: Op) {
        let (chunks, tail) = values.as_chunks_mut::<8>();
        for chunk in chunks {
            let v = _mm256_loadu_ps(chunk.as_ptr());
            _mm256_storeu_ps(chunk.as_mut_ptr(), compute(v, op));
        }
        if !tail.is_empty() {
            // The tail goes through the same lanes, so every value gets the same rounding.
            let mut padded = [0.0f32; 8];
            padded[..tail.len()].copy_from_slice(tail);
            let v = compute(_mm256_loadu_ps(padded.as_ptr()), op);
            _mm256_storeu_ps(padded.as_mut_ptr(), v);
            tail.copy_from_slice(&padded[..tail.len()]);
        }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn compute(v: __m256, op: Op) -> __m256 {
        let one = _mm256_set1_ps(1.0);
        let neg = |v| _mm256_sub_ps(_mm256_setzero_ps(), v);
        match op {
            Op::Exp => exp(v),
            Op::Silu => _mm256_div_ps(v, _mm256_add_ps(one, exp(neg(v)))),
            Op::Sigmoid => _mm256_div_ps(one, _mm256_add_ps(one, exp(neg(v)))),
            Op::Tanh => {
                let two = _mm256_set1_ps(2.0);
                let e = exp(neg(_mm256_mul_ps(two, v)));
                _mm256_sub_ps(_mm256_div_ps(two, _mm256_add_ps(one, e)), one)
            }
            Op::Gelu => {
                let abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0), v);
                let z = _mm256_mul_ps(abs, _mm256_set1_ps(FRAC_1_SQRT_2));
                let t = _mm256_div_ps(one, _mm256_fmadd_ps(z, _mm256_set1_ps(0.5), one));
                let mut p = _mm256_setzero_ps();
                for c in ERFC {
                    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(c));
                }
                let tail = _mm256_mul_ps(t, exp(_mm256_fnmadd_ps(z, z, p)));
                let positive = _mm256_cmp_ps::<_CMP_GE_OQ>(v, _mm256_setzero_ps());
                let erfc =
                    _mm256_blendv_ps(tail, _mm256_sub_ps(_mm256_set1_ps(2.0), tail), positive);
                _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5), v), erfc)
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp(x: __m256) -> __m256 {
        let x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(LOW)), _mm256_set1_ps(HIGH));
        let n_int = _mm256_cvtps_epi32(_mm256_mul_ps(x, _mm256_set1_ps(LOG2E)));
        let n = _mm256_cvtepi32_ps(n_int);
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HI), x);
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LO), r);
        let mut p = _mm256_setzero_ps();
        for c in POLY {
            p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(c));
        }
        let y = _mm256_fmadd_ps(
            p,
            _mm256_mul_ps(r, r),
            _mm256_add_ps(r, _mm256_set1_ps(1.0)),
        );
        let two_n = _mm256_slli_epi32::<23>(_mm256_add_epi32(n_int, _mm256_set1_epi32(127)));
        _mm256_mul_ps(y, _mm256_castsi256_ps(two_n))
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{Op, ERFC, FRAC_1_SQRT_2, HIGH, LN2_HI, LN2_LO, LOG2E, LOW, POLY};
    use std::arch::aarch64::*;

    pub unsafe fn apply(values: &mut [f32], op: Op) {
        // Four vectors at a time, step by step: one vector's steps form a
        // chain of about 30 dependent operations, and the core keeps too
        // few in flight to overlap the chains of the next vectors.
        let (quads, rest) = values.as_chunks_mut::<16>();
        for quad in quads {
            let at = quad.as_mut_ptr();
            let mut v = [vdupq_n_f32(0.0); 4];
            for (i, v) in v.iter_mut().enumerate() {
                *v = vld1q_f32(at.add(4 * i));
            }
            let v = compute4(v, op);
            for (i, v) in v.iter().enumerate() {
                vst1q_f32(at.add(4 * i), *v);
            }
        }
        let (chunks, tail) = rest.as_chunks_mut::<4>();
        for chunk in chunks {
            let v = vld1q_f32(chunk.as_ptr());
            vst1q_f32(chunk.as_mut_ptr(), compute(v, op));
        }
        if !tail.is_empty() {
            // The tail goes through the same lanes, so every value gets the same rounding.
            let mut padded = [0.0f32; 4];
            padded[..tail.len()].copy_from_slice(tail);
            let v = compute(vld1q_f32(padded.as_ptr()), op);
            vst1q_f32(padded.as_mut_ptr(), v);
            tail.copy_from_slice(&padded[..tail.len()]);
        }
    }

    #[inline(always)]
    unsafe fn compute(v: float32x4_t, op: Op) -> float32x4_t {
        let one = vdupq_n_f32(1.0);
        match op {
            Op::Exp => exp(v),
            Op::Silu => vdivq_f32(v, vaddq_f32(one, exp(vnegq_f32(v)))),
            Op::Sigmoid => vdivq_f32(one, vaddq_f32(one, exp(vnegq_f32(v)))),
            Op::Tanh => {
                let two = vdupq_n_f32(2.0);
                let e = exp(vnegq_f32(vmulq_f32(two, v)));
                vsubq_f32(vdivq_f32(two, vaddq_f32(one, e)), one)
            }
            Op::Gelu => {
                let z = vmulq_n_f32(vabsq_f32(v), FRAC_1_SQRT_2);
                let t = vdivq_f32(one, vfmaq_n_f32(one, z, 0.5));
                let mut p = vdupq_n_f32(0.0);
                for c in ERFC {
                    p = vfmaq_f32(vdupq_n_f32(c), p, t);
                }
                let tail = vmulq_f32(t, exp(vfmsq_f32(p, z, z)));
                let positive = vcgeq_f32(v, vdupq_n_f32(0.0));
                let erfc = vbslq_f32(positive, vsubq_f32(vdupq_n_f32(2.0), tail), tail);
                vmulq_f32(vmulq_n_f32(v, 0.5), erfc)
            }
        }
    }

    /// `compute` on four vectors, each step on all four before the next:
    /// every lane gets the same operations in the same order.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn compute4(v: [float32x4_t; 4], op: Op) -> [float32x4_t; 4] {
        let one = vdupq_n_f32(1.0);
        let mut out = [one; 4];
        match op {
            Op::Exp => return exp4(v),
            Op::Silu => {
                let mut neg = v;
                for i in 0..4 {
                    neg[i] = vnegq_f32(v[i]);
                }
                let e = exp4(neg);
                for i in 0..4 {
                    out[i] = vdivq_f32(v[i], vaddq_f32(one, e[i]));
                }
            }
            Op::Sigmoid => {
                let mut neg = v;
                for i in 0..4 {
                    neg[i] = vnegq_f32(v[i]);
                }
                let e = exp4(neg);
                for i in 0..4 {
                    out[i] = vdivq_f32(one, vaddq_f32(one, e[i]));
                }
            }
            Op::Tanh => {
                let two = vdupq_n_f32(2.0);
                let mut neg = v;
                for i in 0..4 {
                    neg[i] = vnegq_f32(vmulq_f32(two, v[i]));
                }
                let e = exp4(neg);
                for i in 0..4 {
                    out[i] = vsubq_f32(vdivq_f32(two, vaddq_f32(one, e[i])), one);
                }
            }
            Op::Gelu => {
                for i in 0..4 {
                    out[i] = compute(v[i], op);
                }
            }
        }
        out
    }

    /// `exp` on four vectors, step by step.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn exp4(x: [float32x4_t; 4]) -> [float32x4_t; 4] {
        let mut x = x;
        let mut n_int = [vdupq_n_s32(0); 4];
        let mut n = [vdupq_n_f32(0.0); 4];
        let mut r = [vdupq_n_f32(0.0); 4];
        for i in 0..4 {
            x[i] = vminq_f32(vmaxq_f32(x[i], vdupq_n_f32(LOW)), vdupq_n_f32(HIGH));
        }
        for i in 0..4 {
            n_int[i] = vcvtnq_s32_f32(vmulq_n_f32(x[i], LOG2E));
        }
        for i in 0..4 {
            n[i] = vcvtq_f32_s32(n_int[i]);
        }
        for i in 0..4 {
            r[i] = vfmsq_n_f32(x[i], n[i], LN2_HI);
        }
        for i in 0..4 {
            r[i] = vfmsq_n_f32(r[i], n[i], LN2_LO);
        }
        let mut p = [vdupq_n_f32(0.0); 4];
        for c in POLY {
            for i in 0..4 {
                p[i] = vfmaq_f32(vdupq_n_f32(c), p[i], r[i]);
            }
        }
        let mut y = [vdupq_n_f32(0.0); 4];
        for i in 0..4 {
            y[i] = vfmaq_f32(
                vaddq_f32(r[i], vdupq_n_f32(1.0)),
                p[i],
                vmulq_f32(r[i], r[i]),
            );
        }
        for i in 0..4 {
            let two_n = vshlq_n_s32::<23>(vaddq_s32(n_int[i], vdupq_n_s32(127)));
            y[i] = vmulq_f32(y[i], vreinterpretq_f32_s32(two_n));
        }
        y
    }

    #[inline(always)]
    unsafe fn exp(x: float32x4_t) -> float32x4_t {
        let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(LOW)), vdupq_n_f32(HIGH));
        let n_int = vcvtnq_s32_f32(vmulq_n_f32(x, LOG2E));
        let n = vcvtq_f32_s32(n_int);
        let r = vfmsq_n_f32(x, n, LN2_HI);
        let r = vfmsq_n_f32(r, n, LN2_LO);
        let mut p = vdupq_n_f32(0.0);
        for c in POLY {
            p = vfmaq_f32(vdupq_n_f32(c), p, r);
        }
        let y = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.0)), p, vmulq_f32(r, r));
        let two_n = vshlq_n_s32::<23>(vaddq_s32(n_int, vdupq_n_s32(127)));
        vmulq_f32(y, vreinterpretq_f32_s32(two_n))
    }
}
