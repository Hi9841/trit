//! The power spectrum of windowed frames as one matrix product: a DFT
//! basis with the window folded in, for a transform of any size (GigaAM's
//! 320 points, Kaldi's 512 over 400 samples).

use crate::cpu::{buf::Buf, fgemm, team};

pub struct PowerSpectrum {
    win: usize,
    bins: usize,
    /// `2 * bins x win`: rows `2b` and `2b + 1` are the window times the
    /// cosine and minus the sine of bin `b` over `n_fft` points.
    basis: Vec<f32>,
}

impl PowerSpectrum {
    /// A frame is `window.len()` samples, then zeros up to `n_fft`.
    pub fn new(n_fft: usize, window: &[f32]) -> Self {
        let (win, bins) = (window.len(), n_fft / 2 + 1);
        let tau = std::f64::consts::TAU;
        let mut basis = vec![0.0f32; 2 * bins * win];
        for bin in 0..bins {
            for (n, w) in window.iter().enumerate() {
                let angle = tau * ((bin * n) % n_fft) as f64 / n_fft as f64;
                basis[(2 * bin) * win + n] = (*w as f64 * angle.cos()) as f32;
                basis[(2 * bin + 1) * win + n] = (-*w as f64 * angle.sin()) as f32;
            }
        }
        Self { win, bins, basis }
    }

    pub fn bins(&self) -> usize {
        self.bins
    }

    /// `frames x bins` power from `frames x win` samples (before the window).
    pub fn compute(&self, samples: &[f32], frames: usize) -> Buf {
        let bins = self.bins;
        let spectrum = fgemm::matmul(samples, frames, self.win, &self.basis, 2 * bins);
        let mut power = Buf::scratch(frames * bins);
        team::chunks_mut(&mut power, bins, |f, row| {
            let pairs = &spectrum[f * 2 * bins..(f + 1) * 2 * bins];
            for (b, p) in row.iter_mut().enumerate() {
                *p = pairs[2 * b] * pairs[2 * b] + pairs[2 * b + 1] * pairs[2 * b + 1];
            }
        });
        power
    }
}
