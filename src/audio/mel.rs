//! Log-mel features as the Parakeet feature extractor computes them:
//! preemphasis, centered STFT with a Hann window, power spectrum, Slaney mel
//! filters, log, and per-bin normalization over the take.

use crate::cpu::team;
use crate::layers::balanced_rows;

pub const N_FFT: usize = 512;
pub const WIN_LENGTH: usize = 400;
pub const HOP: usize = 160;
const SAMPLE_RATE: f64 = 16_000.0;
pub const PREEMPHASIS: f32 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8; // 2^-24
const NORM_EPSILON: f32 = 1e-5;
const BINS: usize = N_FFT / 2 + 1;
const HALF_FFT: usize = N_FFT / 2;

pub struct MelFrontend {
    /// Per mel bin, the first FFT bin and the weights of its nonzero span.
    filters: Vec<(usize, Vec<f32>)>,
    window: Box<[f64; N_FFT]>,
    fft: Box<Rfft512>,
    /// off for a model fed raw log-mel.
    normalize: bool,
}

impl MelFrontend {
    pub fn new(mels: usize) -> Self {
        let mut window = Box::new([0.0f64; N_FFT]);
        let offset = (N_FFT - WIN_LENGTH) / 2;
        for n in 0..WIN_LENGTH {
            let phase = 2.0 * std::f64::consts::PI * n as f64 / (WIN_LENGTH - 1) as f64;
            window[offset + n] = 0.5 - 0.5 * phase.cos();
        }
        Self {
            filters: spans(&slaney_filters(mels, N_FFT)),
            window,
            fft: Box::new(Rfft512::new()),
            normalize: true,
        }
    }

    pub fn normalized(self, normalize: bool) -> Self {
        Self { normalize, ..self }
    }

    /// `frames x mels`, frames = samples / 160. Empty for less than one hop.
    pub fn compute(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        let _span = crate::prof::span(crate::prof::Stage::Features);
        let frames = audio.len() / HOP;
        if frames < 2 {
            return (Vec::new(), 0);
        }
        let mut emphasized = vec![0.0f32; audio.len() + N_FFT];
        emphasized[HALF_FFT] = audio[0];
        if audio.len() > 1 {
            preemphasize(audio, &mut emphasized[HALF_FFT + 1..HALF_FFT + audio.len()]);
        }

        let mels = self.filters.len();
        let mut out = vec![0.0f32; frames * mels];
        let rows_per_task = balanced_rows(frames, mels);
        team::chunks_mut(&mut out, rows_per_task * mels, |task, rows| {
            let first = task * rows_per_task;
            compute_mel_rows(self, &emphasized, first, rows);
        });
        if self.normalize {
            normalize(&mut out, frames, mels);
        }
        (out, frames)
    }
}

multiversion! {
    fn preemphasize(audio: &[f32], dst: &mut [f32]) {
        for ((out, &cur), &prev) in dst.iter_mut().zip(&audio[1..]).zip(audio) {
            *out = cur - PREEMPHASIS * prev;
        }
    }
}

multiversion! {
    fn compute_mel_rows(fe: &MelFrontend, emphasized: &[f32], first_frame: usize, rows: &mut [f32]) {
        let mels = fe.filters.len();
        let mut re = [0.0f64; HALF_FFT];
        let mut im = [0.0f64; HALF_FFT];
        let mut power = [0.0f32; BINS];
        for (offset, row) in rows.chunks_exact_mut(mels).enumerate() {
            let start = (first_frame + offset) * HOP;
            fe.fft.power(&emphasized[start..start + N_FFT], &fe.window, &mut re, &mut im, &mut power);
            for (value, (first, weights)) in row.iter_mut().zip(&fe.filters) {
                let mut energy = 0.0f32;
                for (&w, &p) in weights.iter().zip(&power[*first..]) {
                    energy += w * p;
                }
                *value = (energy + LOG_GUARD).ln();
            }
        }
    }
}

/// Per-bin normalization over the take. Accumulates 16 contiguous mel bins per
/// block in exact frame order so each value takes the same steps in the same
/// order as a single-bin loop while reading contiguous cache lines.
fn normalize(features: &mut [f32], frames: usize, mels: usize) {
    const MEL_BLOCK: usize = 16;
    let blocks = mels.div_ceil(MEL_BLOCK);
    let mut stats = vec![(0.0f32, 1.0f32); mels];
    let stats_ptr = team::SendPtr::new(&mut stats);
    team::for_each(blocks, |block| {
        let mel0 = block * MEL_BLOCK;
        let width = (mels - mel0).min(MEL_BLOCK);
        // SAFETY: each block owns disjoint `mel0..mel0 + width` slots of `stats`.
        let out = unsafe { stats_ptr.slice_mut(mel0, width) };
        mel_block_stats(features, frames, mels, mel0, out);
    });
    let rows_per_task = balanced_rows(frames, mels);
    team::chunks_mut(features, rows_per_task * mels, |_, rows| {
        normalize_rows(rows, mels, &stats);
    });
}

multiversion! {
    fn mel_block_stats(
        features: &[f32],
        frames: usize,
        mels: usize,
        mel0: usize,
        out: &mut [(f32, f32)],
    ) {
        let width = out.len();
        let mut sum = [0.0f64; 16];
        for f in 0..frames {
            let row = &features[f * mels + mel0..][..width];
            for j in 0..width {
                sum[j] += f64::from(row[j]);
            }
        }
        let inv_n = 1.0 / frames as f64;
        let mut mean = [0.0f64; 16];
        for j in 0..width {
            mean[j] = sum[j] * inv_n;
        }
        let mut var = [0.0f64; 16];
        for f in 0..frames {
            let row = &features[f * mels + mel0..][..width];
            for j in 0..width {
                let d = f64::from(row[j]) - mean[j];
                var[j] += d * d;
            }
        }
        let inv_nm1 = 1.0 / (frames - 1) as f64;
        for j in 0..width {
            out[j] = (mean[j] as f32, (var[j] * inv_nm1).sqrt() as f32 + NORM_EPSILON);
        }
    }
}

multiversion! {
    fn normalize_rows(rows: &mut [f32], mels: usize, stats: &[(f32, f32)]) {
        for row in rows.chunks_exact_mut(mels) {
            for (value, &(mean, spread)) in row.iter_mut().zip(stats) {
                *value = (*value - mean) / spread;
            }
        }
    }
}

fn hz_to_mel(hz: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        hz / f_sp
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_sp * mel
    }
}

/// librosa.filters.mel(sr=16000, n_fft, fmin=0, fmax=8000, htk=False,
/// norm="slaney"): `mels x (n_fft / 2 + 1)`. Parakeet and Whisper both use it.
pub fn slaney_filters(mels: usize, n_fft: usize) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let fft_freqs: Vec<f64> = (0..bins)
        .map(|bin| bin as f64 * SAMPLE_RATE / n_fft as f64)
        .collect();
    let max_mel = hz_to_mel(SAMPLE_RATE / 2.0);
    let points: Vec<f64> = (0..mels + 2)
        .map(|i| mel_to_hz(max_mel * i as f64 / (mels + 1) as f64))
        .collect();
    let mut filters = vec![0.0f32; mels * bins];
    for mel in 0..mels {
        let lower_width = points[mel + 1] - points[mel];
        let upper_width = points[mel + 2] - points[mel + 1];
        let norm = 2.0 / (points[mel + 2] - points[mel]);
        for (bin, freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - points[mel]) / lower_width;
            let upper = (points[mel + 2] - freq) / upper_width;
            filters[mel * bins + bin] = (lower.min(upper).max(0.0) * norm) as f32;
        }
    }
    filters
}

/// the sum stays the same, in the same order.
fn spans(filters: &[f32]) -> Vec<(usize, Vec<f32>)> {
    filters
        .as_chunks::<BINS>()
        .0
        .iter()
        .map(|row| {
            let first = row.iter().position(|w| *w != 0.0).unwrap_or(0);
            let last = row.iter().rposition(|w| *w != 0.0).unwrap_or(0);
            (first, row[first..=last.max(first)].to_vec())
        })
        .collect()
}

/// 512-point real-to-complex FFT via a 256-point radix-4 complex FFT
/// (`z[k] = x[2k] + i * x[2k + 1]`), cutting complex butterflies by more than
/// half while keeping full `f64` precision.
struct Rfft512 {
    rev4: [u8; HALF_FFT],
    /// Contiguous twiddles for radix-4 stages `m = 4, 16, 64` (lengths 4, 16, 64).
    tw4: StageTwiddles<4>,
    tw16: StageTwiddles<16>,
    tw64: StageTwiddles<64>,
    /// Unpack twiddles `W_512^k = exp(-i * 2*pi*k / 512)` for `k in 0..128`.
    unpack_cos: [f64; HALF_FFT / 2],
    unpack_sin: [f64; HALF_FFT / 2],
}

struct StageTwiddles<const M: usize> {
    w1r: [f64; M],
    w1i: [f64; M],
    w2r: [f64; M],
    w2i: [f64; M],
    w3r: [f64; M],
    w3i: [f64; M],
}

impl<const M: usize> StageTwiddles<M> {
    fn new() -> Self {
        let mut t = Self {
            w1r: [0.0; M],
            w1i: [0.0; M],
            w2r: [0.0; M],
            w2i: [0.0; M],
            w3r: [0.0; M],
            w3i: [0.0; M],
        };
        let step = -std::f64::consts::TAU / (4 * M) as f64;
        for j in 0..M {
            let a1 = step * j as f64;
            let a2 = a1 * 2.0;
            let a3 = a1 * 3.0;
            t.w1r[j] = a1.cos();
            t.w1i[j] = a1.sin();
            t.w2r[j] = a2.cos();
            t.w2i[j] = a2.sin();
            t.w3r[j] = a3.cos();
            t.w3i[j] = a3.sin();
        }
        t
    }
}

impl Rfft512 {
    fn new() -> Self {
        let mut rev4 = [0u8; HALF_FFT];
        for (i, slot) in rev4.iter_mut().enumerate() {
            *slot =
                (((i & 3) << 6) | (((i >> 2) & 3) << 4) | (((i >> 4) & 3) << 2) | ((i >> 6) & 3))
                    as u8;
        }
        let mut unpack_cos = [0.0f64; HALF_FFT / 2];
        let mut unpack_sin = [0.0f64; HALF_FFT / 2];
        let step = -std::f64::consts::TAU / N_FFT as f64;
        for k in 0..HALF_FFT / 2 {
            let a = step * k as f64;
            unpack_cos[k] = a.cos();
            unpack_sin[k] = a.sin();
        }
        Self {
            rev4,
            tw4: StageTwiddles::new(),
            tw16: StageTwiddles::new(),
            tw64: StageTwiddles::new(),
            unpack_cos,
            unpack_sin,
        }
    }

    #[inline(always)]
    fn radix4_stage<const M: usize>(
        re: &mut [f64; HALF_FFT],
        im: &mut [f64; HALF_FFT],
        tw: &StageTwiddles<M>,
    ) {
        let stride = 4 * M;
        let mut start = 0;
        while start < HALF_FFT {
            for j in 0..M {
                let (i0, i1, i2, i3) = (
                    start + j,
                    start + j + M,
                    start + j + 2 * M,
                    start + j + 3 * M,
                );
                let (a0r, a0i) = (re[i0], im[i0]);
                let (b1r, b1i) = (re[i1], im[i1]);
                let (b2r, b2i) = (re[i2], im[i2]);
                let (b3r, b3i) = (re[i3], im[i3]);
                let a1r = b1r * tw.w1r[j] - b1i * tw.w1i[j];
                let a1i = b1r * tw.w1i[j] + b1i * tw.w1r[j];
                let a2r = b2r * tw.w2r[j] - b2i * tw.w2i[j];
                let a2i = b2r * tw.w2i[j] + b2i * tw.w2r[j];
                let a3r = b3r * tw.w3r[j] - b3i * tw.w3i[j];
                let a3i = b3r * tw.w3i[j] + b3i * tw.w3r[j];

                let (t0r, t0i) = (a0r + a2r, a0i + a2i);
                let (t1r, t1i) = (a0r - a2r, a0i - a2i);
                let (t2r, t2i) = (a1r + a3r, a1i + a3i);
                let (t3r, t3i) = (a1r - a3r, a1i - a3i);

                re[i0] = t0r + t2r;
                im[i0] = t0i + t2i;
                re[i1] = t1r + t3i;
                im[i1] = t1i - t3r;
                re[i2] = t0r - t2r;
                im[i2] = t0i - t2i;
                re[i3] = t1r - t3i;
                im[i3] = t1i + t3r;
            }
            start += stride;
        }
    }

    #[inline(always)]
    fn power(
        &self,
        input: &[f32],
        window: &[f64; N_FFT],
        re: &mut [f64; HALF_FFT],
        im: &mut [f64; HALF_FFT],
        power: &mut [f32; BINS],
    ) {
        // Pack even/odd real samples directly in base-4 digit-reversed order.
        for k in 0..HALF_FFT {
            let j = usize::from(self.rev4[k]);
            re[j] = f64::from(input[2 * k]) * window[2 * k];
            im[j] = f64::from(input[2 * k + 1]) * window[2 * k + 1];
        }
        // Stage 0 (m = 1): all twiddles are 1.
        for start in (0..HALF_FFT).step_by(4) {
            let (a0r, a0i) = (re[start], im[start]);
            let (a1r, a1i) = (re[start + 1], im[start + 1]);
            let (a2r, a2i) = (re[start + 2], im[start + 2]);
            let (a3r, a3i) = (re[start + 3], im[start + 3]);
            let (t0r, t0i) = (a0r + a2r, a0i + a2i);
            let (t1r, t1i) = (a0r - a2r, a0i - a2i);
            let (t2r, t2i) = (a1r + a3r, a1i + a3i);
            let (t3r, t3i) = (a1r - a3r, a1i - a3i);
            re[start] = t0r + t2r;
            im[start] = t0i + t2i;
            re[start + 1] = t1r + t3i;
            im[start + 1] = t1i - t3r;
            re[start + 2] = t0r - t2r;
            im[start + 2] = t0i - t2i;
            re[start + 3] = t1r - t3i;
            im[start + 3] = t1i + t3r;
        }
        Self::radix4_stage(re, im, &self.tw4);
        Self::radix4_stage(re, im, &self.tw16);
        Self::radix4_stage(re, im, &self.tw64);

        // Unpack the 256-point complex spectrum into the 257 real power bins.
        let dc = re[0] + im[0];
        let nyq = re[0] - im[0];
        power[0] = (dc * dc) as f32;
        power[HALF_FFT] = (nyq * nyq) as f32;
        power[HALF_FFT / 2] =
            (re[HALF_FFT / 2] * re[HALF_FFT / 2] + im[HALF_FFT / 2] * im[HALF_FFT / 2]) as f32;
        for k in 1..HALF_FFT / 2 {
            let (ar, ai) = (re[k], im[k]);
            let (br, bi) = (re[HALF_FFT - k], im[HALF_FFT - k]);
            let er = 0.5 * (ar + br);
            let ei = 0.5 * (ai - bi);
            let or = 0.5 * (ai + bi);
            let oi = 0.5 * (br - ar);
            let (wr, wi) = (self.unpack_cos[k], self.unpack_sin[k]);
            let tr = or * wr - oi * wi;
            let ti = or * wi + oi * wr;
            let (xkr, xki) = (er + tr, ei + ti);
            let (xnr, xni) = (er - tr, ti - ei);
            power[k] = (xkr * xkr + xki * xki) as f32;
            power[HALF_FFT - k] = (xnr * xnr + xni * xni) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_power(input: &[f32], window: &[f64; N_FFT]) -> [f32; BINS] {
        let mut out = [0.0f32; BINS];
        let step = -std::f64::consts::TAU / N_FFT as f64;
        for (k, slot) in out.iter_mut().enumerate() {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for n in 0..N_FFT {
                let x = f64::from(input[n]) * window[n];
                let angle = step * (k * n) as f64;
                re += x * angle.cos();
                im += x * angle.sin();
            }
            *slot = (re * re + im * im) as f32;
        }
        out
    }

    #[test]
    fn rfft512_matches_direct_dft() {
        let fe = MelFrontend::new(80);
        let mut re = [0.0f64; HALF_FFT];
        let mut im = [0.0f64; HALF_FFT];
        let mut got = [0.0f32; BINS];

        let mut impulse = [0.0f32; N_FFT];
        impulse[137] = 1.5;
        let dc = [0.75f32; N_FFT];
        let nyq: [f32; N_FFT] = std::array::from_fn(|n| if n % 2 == 0 { 1.0 } else { -1.0 });
        let multi: [f32; N_FFT] = std::array::from_fn(|n| {
            let t = n as f32 / SAMPLE_RATE as f32;
            0.6 * (std::f32::consts::TAU * 440.0 * t).sin()
                + 0.3 * (std::f32::consts::TAU * 2317.0 * t).cos()
                - 0.15 * (std::f32::consts::TAU * 5123.0 * t).sin()
        });
        let silence = [0.0f32; N_FFT];

        for (name, signal) in [
            ("silence", &silence),
            ("impulse", &impulse),
            ("dc", &dc),
            ("nyq", &nyq),
            ("multi", &multi),
        ] {
            fe.fft.power(signal, &fe.window, &mut re, &mut im, &mut got);
            let want = direct_power(signal, &fe.window);
            for (bin, (&g, &w)) in got.iter().zip(&want).enumerate() {
                let tol = 1e-4 + 1e-5 * w.abs();
                assert!(
                    (g - w).abs() <= tol,
                    "rfft512 {name} bin {bin}: got={g} want={w}"
                );
            }
        }
    }

    #[test]
    fn mel_frontend_blocked_normalize_matches_naive_and_edge_lengths() {
        let fe_raw = MelFrontend::new(128).normalized(false);
        let fe_norm = MelFrontend::new(128);
        assert_eq!(fe_norm.compute(&[]), (Vec::new(), 0));
        assert_eq!(fe_norm.compute(&[0.0; HOP]), (Vec::new(), 0));

        let audio: Vec<f32> = (0..24_000)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                0.4 * (std::f32::consts::TAU * 320.0 * t).sin()
                    + 0.25 * (std::f32::consts::TAU * 1470.0 * t).cos()
            })
            .collect();
        let (mut expected, frames) = fe_raw.compute(&audio);
        let (got, got_frames) = fe_norm.compute(&audio);
        assert_eq!(frames, 24_000 / HOP);
        assert_eq!(got_frames, frames);
        assert_eq!(got.len(), frames * 128);

        // Naive single-bin f64 mean/std normalization.
        for mel in 0..128 {
            let mean: f64 = (0..frames)
                .map(|f| f64::from(expected[f * 128 + mel]))
                .sum::<f64>()
                / frames as f64;
            let var: f64 = (0..frames)
                .map(|f| {
                    let d = f64::from(expected[f * 128 + mel]) - mean;
                    d * d
                })
                .sum::<f64>()
                / (frames - 1) as f64;
            let (mean, spread) = (mean as f32, var.sqrt() as f32 + NORM_EPSILON);
            for f in 0..frames {
                expected[f * 128 + mel] = (expected[f * 128 + mel] - mean) / spread;
            }
        }
        for (i, (&g, &w)) in got.iter().zip(&expected).enumerate() {
            assert!(
                g.is_finite() && (g - w).abs() <= 1e-6,
                "mel norm at {i}: got={g} want={w}"
            );
        }
    }
}
