//! Checks `trit::resample` with tones: `cargo run --release --example resample_check`.
//! No model. Prints one line for each rate and ends with exit code 1 when a
//! check fails: the level of tones below the cutoff, the level of tones that
//! 16 kHz audio cannot hold, the length, a constant input, the rates that
//! fail, the time, and a hash of the output (the same on every CPU path:
//! compare runs with `TRIT_ISA=scalar` and `TRIT_ISA=avx2`).
#![allow(clippy::print_stdout)]

use std::f64::consts::TAU;
use std::process::ExitCode;

fn tone(rate: u32, hz: f64, seconds: f64) -> Vec<f32> {
    let count = (f64::from(rate) * seconds) as usize;
    (0..count)
        .map(|n| (0.5 * (TAU * hz * n as f64 / f64::from(rate)).sin()) as f32)
        .collect()
}

/// The level of the middle half of `samples`, in dB against a tone of 0.5.
fn level(samples: &[f32]) -> f64 {
    let middle = &samples[samples.len() / 4..samples.len() * 3 / 4];
    let power = middle.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>() / middle.len() as f64;
    10.0 * (power / 0.125 + 1e-30).log10()
}

/// A fixed stream of numbers in -0.5 to 0.5.
fn noise(count: usize) -> Vec<f32> {
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

fn hash(samples: &[f32]) -> u64 {
    samples.iter().fold(0xcbf2_9ce4_8422_2325, |hash, sample| {
        (hash ^ u64::from(sample.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn main() -> ExitCode {
    let mut failed = false;
    let mut check = |ok: bool, what: String| {
        if !ok {
            failed = true;
            println!("  FAIL: {what}");
        }
    };
    for rate in [
        8_000u32, 11_025, 22_050, 24_000, 32_000, 44_100, 48_000, 96_000, 192_000,
    ] {
        // The lower of the two Nyquist rates: the audio that both rates hold.
        let nyquist = f64::from(rate.min(16_000)) / 2.0;
        let mut worst_pass = 0.0f64;
        for part in [0.02, 0.1, 0.25, 0.5, 0.75] {
            let out = trit::resample(&tone(rate, part * nyquist, 1.0), rate).unwrap();
            worst_pass = worst_pass.max(level(&out).abs());
        }
        // Tones that 16 kHz audio cannot hold, past the transition band.
        let mut worst_stop = f64::NEG_INFINITY;
        for hz in [9_500.0, 10_000.0, 12_000.0, 15_000.0, 20_000.0, 40_000.0] {
            if hz < f64::from(rate) / 2.0 {
                let out = trit::resample(&tone(rate, hz, 1.0), rate).unwrap();
                worst_stop = worst_stop.max(level(&out));
            }
        }
        let flat = trit::resample(&vec![0.5f32; rate as usize], rate).unwrap();
        let middle = flat[flat.len() / 2];
        let odd = trit::resample(&vec![0.0f32; 12_345], rate).unwrap();
        let length = (12_345u64 * 16_000).div_ceil(u64::from(rate)) as usize;
        let input = noise(rate as usize * 30);
        let started = std::time::Instant::now();
        let out = trit::resample(&input, rate).unwrap();
        let ms = started.elapsed().as_secs_f64() * 1000.0 / 30.0;
        println!(
            "{rate:6} Hz: tones below the cutoff within {worst_pass:.3} dB, tones from 9.5 kHz at {worst_stop:.0} dB, \
             constant 0.5 gives {middle:.6}, {ms:.3} ms for each second, hash {:016x}",
            hash(&out)
        );
        check(
            worst_pass < 0.05,
            format!("a tone below the cutoff changed its level by {worst_pass:.3} dB"),
        );
        check(
            worst_stop < -70.0,
            format!("a tone from 9.5 kHz is at {worst_stop:.0} dB"),
        );
        check(
            (middle - 0.5).abs() < 1e-6,
            format!("a constant 0.5 gives {middle}"),
        );
        check(
            odd.len() == length,
            format!("12345 samples give {} samples, not {length}", odd.len()),
        );
        // A forced path (`TRIT_ISA`) is slow on purpose: its time is no fault.
        let forced = std::env::var_os("TRIT_ISA").is_some_and(|value| !value.is_empty());
        check(
            forced || ms < 0.5,
            format!("{ms:.3} ms for each second of audio"),
        );
    }
    for rate in [0u32, 3_999, 384_001, 44_101] {
        check(
            trit::resample(&[0.0; 16], rate).is_err(),
            format!("{rate} Hz did not fail"),
        );
    }
    let same = noise(1_000);
    check(
        trit::resample(&same, 16_000).unwrap() == same,
        "16000 Hz changed the samples".into(),
    );
    check(
        trit::resample(&[], 48_000).unwrap().is_empty(),
        "no samples gave samples".into(),
    );
    if failed {
        ExitCode::FAILURE
    } else {
        println!("resample: all checks passed");
        ExitCode::SUCCESS
    }
}
