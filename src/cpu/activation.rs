//! Element-wise exp, SiLU, sigmoid, tanh, and GELU on whole slices. The Windows target
//! has no vector math library, so `f32::exp` runs one value at a time. These
//! run one polynomial (Cephes `expf`, within about 2 ulp) on 16 or 8 lanes,
//! and the scalar path runs the same polynomial.

use super::fgemm::{level, Level};

const LOG2E: f32 = std::f32::consts::LOG2_E;
/// ln 2 split in two, so `n * LN2_HI` is exact for every `n` in range.
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;
/// `exp` of these stays a normal f32, and `n` stays in -126..=127.
const LOW: f32 = -87.3;
const HIGH: f32 = 88.0;
/// `erfc(z) = t * exp(-z^2 + P(t))`, `t = 1 / (1 + z / 2)`, for `z >= 0`:
/// Numerical Recipes' `erfcc`, within 1.2e-7 relative everywhere, so GELU
/// stays exact also where `x` is far below 0 and `erfc` is small.
const ERFC: [f32; 10] = [
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
const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
const POLY: [f32; 6] = [
    1.987_569_1e-4,
    1.398_199_9e-3,
    8.333_452e-3,
    4.166_579_6e-2,
    0.166_666_65,
    0.5,
];

#[derive(Clone, Copy)]
enum Op {
    Exp,
    Silu,
    Sigmoid,
    Tanh,
    Gelu,
}

/// `v = e^v`.
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

fn apply(values: &mut [f32], op: Op) {
    match level() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the level was detected from the CPU.
        Level::Avx512 => unsafe { avx512::apply(values, op) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the level was detected from the CPU.
        Level::Avx2 => unsafe { avx2::apply(values, op) },
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
    use super::{Op, ERFC, FRAC_1_SQRT_2, HIGH, LN2_HI, LN2_LO, LOG2E, LOW, POLY};
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
