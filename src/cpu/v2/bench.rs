//! Times the two kernel sets on the same inputs, inside the engine's team,
//! and compares their outputs value for value.

use crate::cpu::fgemm;
use crate::cpu::int8::Int8;
use crate::cpu::kernels::{Isa, Ternary, GROUP};
#[cfg(target_arch = "x86_64")]
use crate::cpu::team;
use crate::cpu::v2::product::IntProduct;
use crate::cpu::{enter, Spread};
use std::time::Instant;

pub(crate) struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A value in -1..1.
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    /// Activations like a layer's: a sum of three values, some rows louder.
    pub fn activations(&mut self, m: usize, k: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; m * k];
        for row in x.chunks_exact_mut(k) {
            let gain = 0.25 + 4.0 * self.unit().abs();
            for v in row {
                *v = gain * (self.unit() + self.unit() + self.unit());
            }
        }
        x
    }
}

/// The least and the median time of `run`, in ms, after two runs to warm up.
pub(crate) fn time_ms(rounds: usize, mut run: impl FnMut()) -> (f64, f64) {
    run();
    run();
    let mut times: Vec<f64> = (0..rounds)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    (times[0], times[times.len() / 2])
}

fn spread_of(name: &str) -> Spread {
    if name == "all" {
        Spread::AllCores
    } else {
        Spread::OneCache
    }
}

/// `vpdpbusd` alone, 12 sums in registers: the most int8 multiply-adds this
/// CPU does. Returns a value that depends on every sum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dpbusd_only(iterations: usize, seed: i32) -> i32 {
    use std::arch::x86_64::*;
    let w = _mm512_set1_epi8(std::hint::black_box(2));
    let x = _mm512_set1_epi32(std::hint::black_box(seed));
    let mut acc = [_mm512_setzero_si512(); 12];
    for _ in 0..iterations {
        for a in acc.iter_mut() {
            *a = _mm512_dpbusd_epi32(*a, w, x);
        }
    }
    let mut total = _mm512_setzero_si512();
    for a in acc {
        total = _mm512_add_epi32(total, a);
    }
    _mm512_reduce_add_epi32(total)
}

/// `vfmadd231ps` alone, 12 sums in registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn fma_only(iterations: usize, seed: f32) -> f32 {
    use std::arch::x86_64::*;
    let w = _mm512_set1_ps(std::hint::black_box(1.0e-3));
    let x = _mm512_set1_ps(std::hint::black_box(seed));
    let mut acc = [_mm512_setzero_ps(); 12];
    for _ in 0..iterations {
        for a in acc.iter_mut() {
            *a = _mm512_fmadd_ps(w, x, *a);
        }
    }
    let mut total = _mm512_setzero_ps();
    for a in acc {
        total = _mm512_add_ps(total, a);
    }
    _mm512_reduce_add_ps(total)
}

/// A chain of dependent shifts and exclusive-ors: 6 cycles for each step on
/// every x86-64 CPU, so the time of `steps` gives the clock.
#[cfg(target_arch = "x86_64")]
fn clock_chain(steps: usize, seed: u64) -> u64 {
    let mut x = std::hint::black_box(seed | 1);
    for _ in 0..steps {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
    }
    x
}

/// The team with every thread busy: the clock in GHz, then the products per
/// cycle and core of `vpdpbusd` alone and of `vfmadd231ps` alone.
#[cfg(target_arch = "x86_64")]
pub(crate) fn peak(spread: Spread) -> (f64, f64, f64) {
    let _team = enter(spread);
    let threads = team::threads();
    let tasks = threads * 8;
    let sink = std::sync::atomic::AtomicU64::new(0);
    let relaxed = std::sync::atomic::Ordering::Relaxed;
    let steps = 300_000usize;
    let (clock_ms, _) = time_ms(9, || {
        team::for_each(tasks, |task| {
            sink.fetch_add(clock_chain(steps, task as u64), relaxed);
        });
    });
    let iterations = 150_000usize;
    let (int8_ms, _) = time_ms(9, || {
        team::for_each(tasks, |task| {
            // SAFETY: the caller checked the CPU features in `run`.
            let value = unsafe { dpbusd_only(iterations, task as i32) };
            sink.fetch_add(value as u64, relaxed);
        });
    });
    let (float_ms, _) = time_ms(9, || {
        team::for_each(tasks, |task| {
            // SAFETY: as above.
            let value = unsafe { fma_only(iterations, task as f32) };
            sink.fetch_add(value as u64, relaxed);
        });
    });
    let cycles = |ms: f64| (tasks * steps * 6) as f64 / clock_ms * ms / threads as f64;
    let products = (tasks * iterations * 12) as f64 / threads as f64;
    let ghz = (tasks * steps * 6) as f64 / threads as f64 / (clock_ms * 1e-3) / 1e9;
    (ghz, products / cycles(int8_ms), products / cycles(float_ms))
}

/// `first` and `second` in turns, `rounds` times each, after two runs each
/// to warm up: the least time of each in ms, and the median of the
/// second-to-first ratios of the rounds.
pub(crate) fn in_turns(
    rounds: usize,
    mut first: impl FnMut(),
    mut second: impl FnMut(),
) -> (f64, f64, f64) {
    for _ in 0..2 {
        first();
        second();
    }
    let mut times = (Vec::new(), Vec::new());
    for round in 0..rounds {
        // The side that runs first changes each round.
        for side in 0..2 {
            let start = Instant::now();
            if (side + round) % 2 == 0 {
                first();
                times.0.push(start.elapsed().as_secs_f64() * 1e3);
            } else {
                second();
                times.1.push(start.elapsed().as_secs_f64() * 1e3);
            }
        }
    }
    let mut ratios: Vec<f64> = times.0.iter().zip(&times.1).map(|(a, b)| b / a).collect();
    ratios.sort_by(f64::total_cmp);
    let least = |times: &[f64]| times.iter().copied().fold(f64::INFINITY, f64::min);
    (least(&times.0), least(&times.1), ratios[ratios.len() / 2])
}

/// How two outputs differ: the count of values with other bits, and the
/// largest difference relative to the largest value.
fn difference(a: &[f32], b: &[f32]) -> (usize, f32) {
    let peak = a.iter().fold(0.0f32, |peak, v| peak.max(v.abs()));
    let other = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let gap = a
        .iter()
        .zip(b)
        .fold(0.0f32, |gap, (x, y)| gap.max((x - y).abs()));
    (
        other + a.len().abs_diff(b.len()),
        if peak > 0.0 { gap / peak } else { gap },
    )
}

/// Rounds for a product of `work` multiply-adds: about 0.3 s of them.
fn rounds_for(work: usize) -> usize {
    (60_000_000_000usize / work.max(1)).clamp(15, 4000)
}

fn compared(
    name: &str,
    work: usize,
    first: impl Fn() -> crate::cpu::buf::Buf,
    second: impl Fn() -> crate::cpu::buf::Buf,
) -> String {
    let reference = second();
    let (other, gap) = difference(&first(), &reference);
    // FNV-1a of the second set's bits: equal on every path and CPU when the
    // second set gives the same values everywhere.
    let hash = reference.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
        (h ^ u64::from(v.to_bits())).wrapping_mul(0x0100_0000_01b3)
    });
    let (a, b, ratio) = in_turns(
        rounds_for(work),
        || {
            std::hint::black_box(first());
        },
        || {
            std::hint::black_box(second());
        },
    );
    let rate = |ms: f64| work as f64 / (ms * 1e-3) / 1e12;
    format!(
        "{name}: first {a:.4} ms ({:.3} T/s), second {b:.4} ms ({:.3} T/s), second/first least {:.3}, median of rounds {ratio:.3}; values with other bits {other} (largest gap {gap:.1e}); second hash {hash:016x}",
        rate(a),
        rate(b),
        b / a
    )
}

fn random_ternary(rng: &mut Rng, n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let codes: Vec<u8> = (0..n * k).map(|_| (rng.next() % 3) as u8).collect();
    let scales: Vec<f32> = (0..n * (k / GROUP))
        .map(|_| 0.01 + 0.02 * rng.unit().abs())
        .collect();
    (codes, scales)
}

fn numbers(args: &[String], from: usize) -> Vec<usize> {
    args.iter()
        .skip(from)
        .filter_map(|a| a.parse().ok())
        .collect()
}

/// One line per result. `args`: `peak [one|all]` (the clock and the most
/// products of a core), `ternary|int8|f32|f16 <one|all> <n> <k> <m>...` (the
/// first set alone), or `ternary2|int82|q8_02|f322|f162 <one|all> <n> <k>
/// <m>...` (the first set against the second, in turns). `one` is the team
/// of one L3 domain, `all` the team of all cores. Run it with `TRIT_KERNELS`
/// not set, so that the first set is the first set.
pub fn run(args: &[String]) -> Vec<String> {
    let mut lines = Vec::new();
    let what = args.first().map(String::as_str).unwrap_or("help");
    let spread = spread_of(args.get(1).map(String::as_str).unwrap_or("one"));
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    match what {
        #[cfg(target_arch = "x86_64")]
        "peak" => {
            if !is_x86_feature_detected!("avx512vnni") {
                return vec!["peak: this CPU has no AVX-512 VNNI".into()];
            }
            let (ghz, int8, float) = peak(spread);
            let _team = enter(spread);
            let threads = team::threads();
            lines.push(format!(
                "peak {threads} threads: clock {ghz:.2} GHz, vpdpbusd {int8:.2} per cycle and core ({:.3} T/s), vfmadd {float:.2} ({:.3} T/s)",
                int8 * 64.0 * ghz * threads as f64 / 1e3,
                float * 16.0 * ghz * threads as f64 / 1e3,
            ));
        }
        "ternary" => {
            let shape = numbers(args, 2);
            let (n, k) = (shape[0], shape[1]);
            let (codes, scales) = random_ternary(&mut rng, n, k);
            let layer = Ternary::from_rows(Isa::detect(), &scales, n, k, |row, out| {
                out.copy_from_slice(&codes[row * k..(row + 1) * k]);
            });
            let _team = enter(spread);
            for &m in &shape[2..] {
                let x = rng.activations(m, k);
                let (least, median) = time_ms(30, || {
                    std::hint::black_box(layer.forward(&x, m));
                });
                let rate = (m * n * k) as f64 / (least * 1e-3) / 1e12;
                lines.push(format!(
                    "ternary first {n}x{k} m={m}: least {least:.3} ms, median {median:.3} ms, {rate:.3} T/s"
                ));
            }
        }
        "int8" => {
            let shape = numbers(args, 2);
            let (n, k) = (shape[0], shape[1]);
            let weights: Vec<f32> = (0..n * k).map(|_| 0.05 * rng.unit()).collect();
            let _team = enter(spread);
            let layer = Int8::quantize(n, k, |row, out| {
                out.copy_from_slice(&weights[row * k..(row + 1) * k]);
            });
            for &m in &shape[2..] {
                let x = rng.activations(m, k);
                let (least, median) = time_ms(30, || {
                    std::hint::black_box(layer.forward(&x, m));
                });
                let rate = (m * n * k) as f64 / (least * 1e-3) / 1e12;
                lines.push(format!(
                    "int8 first {n}x{k} m={m}: least {least:.3} ms, median {median:.3} ms, {rate:.3} T/s"
                ));
            }
        }
        "f32" | "f16" => {
            let shape = numbers(args, 2);
            let (n, k) = (shape[0], shape[1]);
            let weights: Vec<f32> = (0..n * k).map(|_| 0.05 * rng.unit()).collect();
            let half: Vec<u16> = weights
                .iter()
                .map(|v| crate::cpu::tensors::f32_to_f16(*v))
                .collect();
            let _team = enter(spread);
            for &m in &shape[2..] {
                let x = rng.activations(m, k);
                let (least, median) = time_ms(30, || {
                    if what == "f32" {
                        std::hint::black_box(fgemm::matmul(&x, m, k, &weights, n));
                    } else {
                        std::hint::black_box(fgemm::matmul(&x, m, k, &half, n));
                    }
                });
                let rate = (m * n * k) as f64 / (least * 1e-3) / 1e12;
                lines.push(format!(
                    "{what} first {n}x{k} m={m}: least {least:.3} ms, median {median:.3} ms, {rate:.3} T/s"
                ));
            }
        }
        "ternary2" => {
            let shape = numbers(args, 2);
            let (n, k) = (shape[0], shape[1]);
            let (codes, scales) = random_ternary(&mut rng, n, k);
            let row = |row: usize, out: &mut [u8]| {
                out.copy_from_slice(&codes[row * k..(row + 1) * k]);
            };
            let first = Ternary::from_rows(Isa::detect(), &scales, n, k, row);
            let second = IntProduct::ternary(&scales, n, k, GROUP, row);
            let _team = enter(spread);
            for &m in &shape[2..] {
                let x = rng.activations(m, k);
                lines.push(compared(
                    &format!("ternary {n}x{k} m={m}"),
                    m * n * k,
                    || first.forward(&x, m),
                    || second.forward(&x, m),
                ));
            }
        }
        "int82" | "q8_02" => {
            let shape = numbers(args, 2);
            let (n, k) = (shape[0], shape[1]);
            let weights: Vec<f32> = (0..n * k).map(|_| 0.05 * rng.unit()).collect();
            let _team = enter(spread);
            let (first, second) = if what == "int82" {
                let row = |row: usize, out: &mut [f32]| {
                    out.copy_from_slice(&weights[row * k..(row + 1) * k]);
                };
                (Int8::quantize(n, k, row), IntProduct::int8_from_f32(n, k, row))
            } else {
                let blocks = crate::cpu::gemm::Blocks::quantize(&weights);
                (
                    Int8::from_q8_0(n, k, &blocks.codes, &blocks.scales),
                    IntProduct::int8_from_q8_0(n, k, &blocks.codes, &blocks.scales),
                )
            };
            for &m in &shape[2..] {
                let x = rng.activations(m, k);
                lines.push(compared(
                    &format!("{} {n}x{k} m={m}", &what[..what.len() - 1]),
                    m * n * k,
                    || first.forward(&x, m),
                    || second.forward(&x, m),
                ));
            }
        }
        "f322" | "f162" => {
            let shape = numbers(args, 2);
            let (n, k) = (shape[0], shape[1]);
            let weights: Vec<f32> = (0..n * k).map(|_| 0.05 * rng.unit()).collect();
            let half: Vec<u16> = weights
                .iter()
                .map(|v| crate::cpu::tensors::f32_to_f16(*v))
                .collect();
            let _team = enter(spread);
            for &m in &shape[2..] {
                let x = rng.activations(m, k);
                let name = format!("{} {n}x{k} m={m}", &what[..3]);
                lines.push(if what == "f322" {
                    compared(
                        &name,
                        m * n * k,
                        || fgemm::matmul(&x, m, k, &weights, n),
                        || crate::cpu::v2::float::matmul(&x, m, k, &weights, n),
                    )
                } else {
                    compared(
                        &name,
                        m * n * k,
                        || fgemm::matmul(&x, m, k, &half, n),
                        || crate::cpu::v2::float::matmul(&x, m, k, &half, n),
                    )
                });
            }
        }
        _ => lines.push(
            "usage: kernel_bench peak [one|all] | ternary2|int82|q8_02|f322|f162 <one|all> <n> <k> <m>..."
                .into(),
        ),
    }
    lines
}
