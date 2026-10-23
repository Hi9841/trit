//! Log-mel features as Whisper computes them: audio padded to 30 s, centered
//! STFT (400-point periodic Hann window, hop 160, reflect padding), power
//! spectrum, Slaney mel filters, log10, a floor 8 below the peak, then
//! `(x + 4) / 4`. 3000 frames per 30 s window.

use super::mel::slaney_filters;
use crate::cpu::{buf::Buf, fgemm, team};

pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
/// Samples in one 30 s window; Whisper always sees a full window.
pub const WINDOW_SAMPLES: usize = 480_000;
/// Frames in one window.
pub const FRAMES: usize = WINDOW_SAMPLES / HOP;
const BINS: usize = N_FFT / 2 + 1;
const LOG_FLOOR: f32 = 1e-10;

pub struct WhisperMel {
    mels: usize,
    /// `mels x BINS` Slaney filters.
    filters: Vec<f32>,
    /// `2 * BINS x N_FFT`: rows `2b` and `2b + 1` are the windowed cosine
    /// and sine of bin `b`, so one product gives every frame's spectrum.
    basis: Vec<f32>,
}

impl WhisperMel {
    pub fn new(mels: usize) -> Self {
        let mut basis = vec![0.0f32; 2 * BINS * N_FFT];
        for bin in 0..BINS {
            for n in 0..N_FFT {
                let tau = 2.0 * std::f64::consts::PI;
                let window = 0.5 - 0.5 * (tau * n as f64 / N_FFT as f64).cos();
                let angle = tau * (bin * n % N_FFT) as f64 / N_FFT as f64;
                basis[(2 * bin) * N_FFT + n] = (window * angle.cos()) as f32;
                basis[(2 * bin + 1) * N_FFT + n] = (-window * angle.sin()) as f32;
            }
        }
        Self {
            mels,
            filters: slaney_filters(mels, N_FFT),
            basis,
        }
    }

    /// `FRAMES x mels` for up to 30 s of audio, zero padded to 30 s.
    pub fn compute(&self, audio: &[f32]) -> Vec<f32> {
        self.compute_frames(&audio[..audio.len().min(WINDOW_SAMPLES)], FRAMES)
    }

    /// `frames x mels` for `audio` zero padded to `frames` hops (Qwen3-ASR:
    /// `audio.len() / HOP` frames, no padding to 30 s).
    pub fn compute_frames(&self, audio: &[f32], frames: usize) -> Vec<f32> {
        // Frames that touch audio; later frames see only padding and are
        // all equal to the frame of silence.
        let live = ((audio.len() + N_FFT / 2).div_ceil(HOP) + 1).min(frames);
        let mut windows = Buf::scratch(live * N_FFT);
        let sample = |i: isize| -> f32 {
            // Reflect at the start (torch.stft center=True); the end is
            // padding, so samples past the audio are zero.
            let i = if i < 0 { -i } else { i } as usize;
            audio.get(i).copied().unwrap_or(0.0)
        };
        team::chunks_mut(&mut windows, N_FFT, |f, row| {
            let start = (f * HOP) as isize - (N_FFT / 2) as isize;
            for (n, value) in row.iter_mut().enumerate() {
                *value = sample(start + n as isize);
            }
        });
        let spectrum = fgemm::matmul(&windows, live, N_FFT, &self.basis, 2 * BINS);
        let mut power = Buf::scratch(live * BINS);
        team::chunks_mut(&mut power, BINS, |f, row| {
            let pairs = &spectrum[f * 2 * BINS..(f + 1) * 2 * BINS];
            for (b, p) in row.iter_mut().enumerate() {
                *p = pairs[2 * b] * pairs[2 * b] + pairs[2 * b + 1] * pairs[2 * b + 1];
            }
        });
        let mel = fgemm::matmul(&power, live, BINS, &self.filters, self.mels);
        let mut out = vec![LOG_FLOOR.log10(); frames * self.mels];
        for (o, m) in out.iter_mut().zip(mel.iter()) {
            *o = m.max(LOG_FLOOR).log10();
        }
        let peak = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for value in &mut out {
            *value = (value.max(peak - 8.0) + 4.0) / 4.0;
        }
        out
    }
}
