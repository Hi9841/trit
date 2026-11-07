//! Log-mel features as Whisper computes them: audio padded to 30 s, centered
//! STFT (400-point periodic Hann window, hop 160, reflect padding), power
//! spectrum, Slaney mel filters, log10, a floor 8 below the peak, then
//! `(x + 4) / 4`. 3000 frames per 30 s window.

use super::dft::PowerSpectrum;
use super::mel::slaney_filters;
use crate::cpu::{buf::Buf, fgemm, team};

pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const WINDOW_SAMPLES: usize = 480_000;
pub const FRAMES: usize = WINDOW_SAMPLES / HOP;
const BINS: usize = N_FFT / 2 + 1;
const LOG_FLOOR: f32 = 1e-10;

pub struct WhisperMel {
    mels: usize,
    /// `mels x BINS` Slaney filters.
    filters: Vec<f32>,
    spectrum: PowerSpectrum,
}

impl WhisperMel {
    pub fn new(mels: usize) -> Self {
        Self {
            mels,
            filters: slaney_filters(mels, N_FFT),
            spectrum: PowerSpectrum::hann(N_FFT),
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
        let power = self.spectrum.compute(&windows, live);
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
