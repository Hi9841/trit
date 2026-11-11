//! Audio at another rate, as the 16 kHz samples that the models take.
//!
//! A polyphase filter: a sinc with a Kaiser window, as
//! `scipy.signal.resample_poly` and SoX build theirs. The ratio of the two
//! rates is a ratio of whole numbers (48 kHz is 1 to 3, 44.1 kHz is 160 to
//! 441), so each output sample has one of `up` exact tap sets, and no tap is
//! interpolated.
//!
//! Each tap set is one dot product of the second kernel set (`cpu::v2`),
//! which adds in one order on every CPU. So every CPU gives the same
//! samples, and so the same text.

use parking_lot::Mutex;
use std::sync::Arc;

/// Zero crossings of the sinc on each side of an output sample. The cutoff
/// is the lower of the two Nyquist rates, as in `resample_poly`: for 48 kHz
/// audio the level is full up to 6.8 kHz, half at 8 kHz, and silence from
/// 9.3 kHz. A cutoff at 0.95 of that rate kept more of the audio above
/// 8 kHz out, and cost Moonshine tiny 6 errors more on 100 clips (63 for
/// 57): the models learned from audio with the band up to 8 kHz.
const ZEROS: usize = 16;
/// The Kaiser window for about 80 dB below the pass band.
const BETA: f64 = 7.857;
/// The rates a recording or a microphone has: 4 kHz to 384 kHz.
const RATES: std::ops::RangeInclusive<u32> = 4_000..=384_000;
/// The most taps of all tap sets together (4 MB). The usual rates need
/// under 40,000. A rate that shares no factor with 16,000 (44,101 Hz) would
/// need 16,000 tap sets.
const MAX_TAPS: usize = 1 << 20;
/// The lanes of the dot product: a tap set is a whole number of them.
const LANES: usize = 16;

/// `samples` at `rate` Hz as 16 kHz samples, for `Trit::transcribe`. Mono
/// in, mono out; the length is `samples.len() * 16000 / rate`, rounded up.
/// Fails for a rate outside 4,000 to 384,000 Hz, and for a rate that has
/// too few factors in common with 16,000 (not a rate of audio hardware).
pub fn resample(samples: &[f32], rate: u32) -> Result<Vec<f32>, String> {
    if rate == crate::SAMPLE_RATE {
        return Ok(samples.to_vec());
    }
    let filter = Filter::of(rate)?;
    let count = (samples.len() as u64 * filter.up as u64).div_ceil(filter.down as u64) as usize;
    let mut out = vec![0.0f32; count];
    filter.run(samples, &mut out);
    Ok(out)
}

/// The filter of one rate: `up` tap sets of `taps` taps each.
struct Filter {
    rate: u32,
    /// 16000 / rate as a ratio of whole numbers with no common factor.
    up: usize,
    down: usize,
    /// Input samples before the sample of an output that its tap set reads.
    reach: usize,
    taps: usize,
    /// `up x taps`.
    table: Vec<f32>,
}

/// The filter of the last rate. A program has one microphone rate, and the
/// table of 44.1 kHz takes 2 ms to build: too long for each short take.
static LAST: Mutex<Option<Arc<Filter>>> = Mutex::new(None);

impl Filter {
    fn of(rate: u32) -> Result<Arc<Self>, String> {
        if let Some(filter) = LAST.lock().as_ref().filter(|filter| filter.rate == rate) {
            return Ok(Arc::clone(filter));
        }
        let filter = Arc::new(Self::new(rate)?);
        *LAST.lock() = Some(Arc::clone(&filter));
        Ok(filter)
    }

    fn new(rate: u32) -> Result<Self, String> {
        if !RATES.contains(&rate) {
            return Err(format!(
                "{rate} Hz is not a sample rate of audio; Trit takes {} to {} Hz",
                RATES.start(),
                RATES.end()
            ));
        }
        let common = gcd(rate, crate::SAMPLE_RATE);
        let up = (crate::SAMPLE_RATE / common) as usize;
        let down = (rate / common) as usize;
        // The filter runs at `rate * up` samples per second. One zero
        // crossing of its sinc is `widest` of those samples.
        let widest = up.max(down);
        let half = ZEROS * widest;
        let reach = half / up;
        let taps = (2 * reach + 2).next_multiple_of(LANES);
        if up.saturating_mul(taps) > MAX_TAPS {
            return Err(format!(
                "Trit cannot resample {rate} Hz; give 16000 Hz, or a usual rate (8000, 22050, 24000, 32000, 44100, 48000, 96000 Hz)"
            ));
        }
        Ok(Self {
            rate,
            up,
            down,
            reach,
            taps,
            table: tap_sets(up, widest, half, reach, taps),
        })
    }

    /// Each output sample: the dot product of its tap set and the input
    /// around it. Output `n` sits at input position `n * down / up`: sample
    /// `at` with phase `phase`. Its tap set reads from input `at - reach`.
    fn run(&self, samples: &[f32], out: &mut [f32]) {
        let (whole, part) = (self.down / self.up, self.down % self.up);
        let (mut at, mut phase) = (0usize, 0usize);
        // The inputs of an output near the start or the end, with zeros
        // where the audio has no sample.
        let mut edge = vec![0.0f32; self.taps];
        for value in out.iter_mut() {
            let set = &self.table[phase * self.taps..][..self.taps];
            let inside = at >= self.reach && at - self.reach + self.taps <= samples.len();
            let window = if inside {
                &samples[at - self.reach..][..self.taps]
            } else {
                for (t, slot) in edge.iter_mut().enumerate() {
                    let index = (at + t).checked_sub(self.reach);
                    *slot = index.and_then(|i| samples.get(i)).copied().unwrap_or(0.0);
                }
                &edge[..]
            };
            // The dot product of the second kernel set: one sum order on
            // every CPU, so the samples do not change with the CPU.
            crate::cpu::v2::float::matmul_into(
                window,
                1,
                self.taps,
                set,
                1,
                std::slice::from_mut(value),
            );
            at += whole;
            phase += part;
            if phase >= self.up {
                phase -= self.up;
                at += 1;
            }
        }
    }
}

/// The taps of each phase, `up x taps`. Tap `t` of phase `p` weighs input
/// `at - reach + t`, which is `p + (reach - t) * up` filter samples before
/// the output. Each set sums to 1, so the level does not change with the phase.
fn tap_sets(up: usize, widest: usize, half: usize, reach: usize, taps: usize) -> Vec<f32> {
    let cutoff = 1.0 / widest as f64;
    let window_peak = bessel_i0(BETA);
    let mut table = vec![0.0f32; up * taps];
    let mut set = vec![0.0f64; taps];
    for (phase, row) in table.chunks_exact_mut(taps).enumerate() {
        for (t, tap) in set.iter_mut().enumerate() {
            let distance = phase as f64 + (reach as f64 - t as f64) * up as f64;
            let edge = distance / half as f64;
            *tap = if edge.abs() >= 1.0 {
                0.0
            } else {
                let window = bessel_i0(BETA * (1.0 - edge * edge).sqrt()) / window_peak;
                sinc(cutoff * distance) * window
            };
        }
        let sum: f64 = set.iter().sum();
        for (out, tap) in row.iter_mut().zip(&set) {
            *out = (tap / sum) as f32;
        }
    }
    table
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let angle = std::f64::consts::PI * x;
    angle.sin() / angle
}

/// The modified Bessel function of the first kind, order 0, by its series.
fn bessel_i0(x: f64) -> f64 {
    let quarter = x * x / 4.0;
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..64 {
        term *= quarter / (k * k) as f64;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}
