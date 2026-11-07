//! Kaldi filterbank features as FunASR's `WavFrontend` computes them
//! (SenseVoice): samples scaled to 16-bit range, frames of 25 ms every
//! 10 ms with snipped edges, the mean removed, pre-emphasis 0.97, a Hamming
//! window, a power spectrum over the next power of two, HTK mel filters from
//! 20 Hz, and the natural log. Then low frame rate stacking: `lfr_m` frames
//! side by side every `lfr_n` frames, the first frame repeated
//! `(lfr_m - 1) / 2` times before, and the model's CMVN,
//! `(x + shift) * scale`.

use super::dft::PowerSpectrum;
use crate::cpu::{buf::Buf, fgemm, team};

const PREEMPHASIS: f32 = 0.97;
const LOW_HZ: f64 = 20.0;
/// The log floor: f32 epsilon, as Kaldi.
const FLOOR: f32 = f32::EPSILON;

pub struct KaldiFbank {
    win: usize,
    hop: usize,
    mels: usize,
    lfr_m: usize,
    lfr_n: usize,
    upscale: f32,
    spectrum: PowerSpectrum,
    /// `mels x bins`.
    filters: Vec<f32>,
    /// Per value of a stacked frame (`lfr_m * mels`).
    cmvn_shift: Vec<f32>,
    cmvn_scale: Vec<f32>,
}

pub struct FbankShape {
    pub sample_rate: usize,
    pub win: usize,
    pub hop: usize,
    pub mels: usize,
    pub lfr_m: usize,
    pub lfr_n: usize,
    /// Samples times 32768 first (Kaldi's 16-bit range).
    pub upscale: bool,
}

impl KaldiFbank {
    pub fn new(
        shape: FbankShape,
        cmvn_shift: Vec<f32>,
        cmvn_scale: Vec<f32>,
    ) -> Result<Self, String> {
        let FbankShape {
            sample_rate,
            win,
            hop,
            mels,
            lfr_m,
            lfr_n,
            upscale,
        } = shape;
        if cmvn_shift.len() != lfr_m * mels
            || cmvn_scale.len() != lfr_m * mels
            || hop == 0
            || lfr_n == 0
        {
            return Err("the front end's CMVN does not fit its stacked frames".into());
        }
        let n_fft = win.next_power_of_two();
        let a = std::f64::consts::TAU / (win - 1) as f64;
        let window: Vec<f32> = (0..win)
            .map(|i| (0.54 - 0.46 * (a * i as f64).cos()) as f32)
            .collect();
        Ok(Self {
            win,
            hop,
            mels,
            lfr_m,
            lfr_n,
            upscale: if upscale { 32768.0 } else { 1.0 },
            spectrum: PowerSpectrum::new(n_fft, &window),
            filters: htk_filters(sample_rate, n_fft, mels),
            cmvn_shift,
            cmvn_scale,
        })
    }

    /// `frames x (lfr_m * mels)` stacked frames; none for less than one window.
    pub fn compute(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        if audio.len() < self.win {
            return (Vec::new(), 0);
        }
        let frames = (audio.len() - self.win) / self.hop + 1;
        let mut windows = Buf::scratch(frames * self.win);
        team::chunks_mut(&mut windows, self.win, |f, row| {
            for (o, s) in row.iter_mut().zip(&audio[f * self.hop..][..self.win]) {
                *o = s * self.upscale;
            }
            let mean = (row.iter().map(|v| *v as f64).sum::<f64>() / self.win as f64) as f32;
            row.iter_mut().for_each(|v| *v -= mean);
            for i in (1..row.len()).rev() {
                row[i] -= PREEMPHASIS * row[i - 1];
            }
            row[0] *= 1.0 - PREEMPHASIS;
        });
        let power = self.spectrum.compute(&windows, frames);
        let mel = fgemm::matmul(
            &power,
            frames,
            self.spectrum.bins(),
            &self.filters,
            self.mels,
        );
        let logs: Vec<f32> = mel.iter().map(|m| m.max(FLOOR).ln()).collect();
        let (mels, width) = (self.mels, self.lfr_m * self.mels);
        let left = (self.lfr_m - 1) / 2;
        let stacked = frames.div_ceil(self.lfr_n);
        let mut out = vec![0.0f32; stacked * width];
        for (t, row) in out.chunks_exact_mut(width).enumerate() {
            for k in 0..self.lfr_m {
                let src = (t * self.lfr_n + k).saturating_sub(left).min(frames - 1);
                row[k * mels..(k + 1) * mels].copy_from_slice(&logs[src * mels..(src + 1) * mels]);
            }
            for ((v, shift), scale) in row.iter_mut().zip(&self.cmvn_shift).zip(&self.cmvn_scale) {
                *v = (*v + shift) * scale;
            }
        }
        (out, stacked)
    }
}

fn htk_filters(sample_rate: usize, n_fft: usize, mels: usize) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let mel = |hz: f64| 1127.0 * (1.0 + hz / 700.0).ln();
    let hz = |m: f64| 700.0 * ((m / 1127.0).exp() - 1.0);
    let (low, high) = (mel(LOW_HZ), mel(sample_rate as f64 / 2.0));
    let step = (high - low) / (mels + 1) as f64;
    let width = sample_rate as f64 / n_fft as f64;
    let mut filters = vec![0.0f32; mels * bins];
    for m in 0..mels {
        let (left, center, right) = (
            hz(low + m as f64 * step),
            hz(low + (m + 1) as f64 * step),
            hz(low + (m + 2) as f64 * step),
        );
        for b in 0..bins {
            let f = b as f64 * width;
            if f > left && f < right {
                filters[m * bins + b] = if f <= center {
                    (f - left) / (center - left)
                } else {
                    (right - f) / (right - center)
                } as f32;
            }
        }
    }
    filters
}
