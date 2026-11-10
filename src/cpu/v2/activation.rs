//! The scalar path of the activations (`activation.rs`), with the steps of
//! its SIMD paths: each multiply-add that they fuse is one `mul_add` here,
//! in the same order. So the scalar path gives the bits of AVX-512, AVX2,
//! and NEON, which agree with each other. The first scalar path does not
//! fuse, and adds the last terms of `exp` in another order.

use crate::cpu::activation::{Op, ERFC, FRAC_1_SQRT_2, HIGH, LN2_HI, LN2_LO, LOG2E, LOW, POLY};

pub(in crate::cpu) fn apply(values: &mut [f32], op: Op) {
    #[cfg(target_arch = "x86_64")]
    if super::float::fused() {
        // SAFETY: the CPU has FMA.
        unsafe { apply_fma(values, op) };
        return;
    }
    values.iter_mut().for_each(|v| *v = one(*v, op));
}

/// With FMA on, `mul_add` is one instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn apply_fma(values: &mut [f32], op: Op) {
    values.iter_mut().for_each(|v| *v = one(*v, op));
}

#[inline(always)]
fn one(v: f32, op: Op) -> f32 {
    match op {
        Op::Exp => exp(v),
        Op::Silu => v / (1.0 + exp(0.0 - v)),
        Op::Sigmoid => 1.0 / (1.0 + exp(0.0 - v)),
        Op::Tanh => 2.0 / (1.0 + exp(0.0 - 2.0 * v)) - 1.0,
        Op::Gelu => {
            let z = v.abs() * FRAC_1_SQRT_2;
            let t = 1.0 / z.mul_add(0.5, 1.0);
            let p = ERFC.iter().fold(0.0f32, |p, c| p.mul_add(t, *c));
            let tail = t * exp((-z).mul_add(z, p));
            (0.5 * v) * if v >= 0.0 { 2.0 - tail } else { tail }
        }
    }
}

#[inline(always)]
// `max` then `min`, as `maxps` then `minps`: NaN becomes `LOW`, where
// `clamp` keeps it.
#[allow(clippy::manual_clamp)]
fn exp(x: f32) -> f32 {
    let x = x.max(LOW).min(HIGH);
    let n_int = (x * LOG2E).round_ties_even() as i32;
    let n = n_int as f32;
    let r = (-n).mul_add(LN2_HI, x);
    let r = (-n).mul_add(LN2_LO, r);
    let p = POLY.iter().fold(0.0f32, |p, c| p.mul_add(r, *c));
    let y = p.mul_add(r * r, r + 1.0);
    y * f32::from_bits(((n_int + 127) as u32) << 23)
}
