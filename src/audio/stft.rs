//! A short-time Fourier transform and its inverse, as `torch.stft` and
//! `torch.istft` compute them with `center=True` (reflect padding) and a
//! periodic Hann window as long as the transform. For the small transforms
//! of a vocoder (iSTFTNet: 20 points), a direct DFT with tables is exact
//! enough and needs no FFT.

use crate::cpu::team;
use crate::layers::balanced_rows;

pub struct Stft {
    n_fft: usize,
    hop: usize,
    window: Vec<f64>,
    /// `cos(2 pi k n / n_fft)` and `sin(...)`, `[bin][n]`.
    cos: Vec<f64>,
    sin: Vec<f64>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        let bins = n_fft / 2 + 1;
        let tau = std::f64::consts::TAU;
        let window = (0..n_fft)
            .map(|n| super::dft::periodic_hann(n, n_fft))
            .collect();
        let mut cos = Vec::with_capacity(bins * n_fft);
        let mut sin = Vec::with_capacity(bins * n_fft);
        for k in 0..bins {
            for n in 0..n_fft {
                // Exact at the quarter turns: a sine that is 0 must be 0, so a
                // phase of +-pi keeps the sign the FFT gives it.
                let step = (k * n) % n_fft;
                let (c, s) = match (4 * step).checked_div(n_fft) {
                    _ if !(4 * step).is_multiple_of(n_fft) => {
                        let angle = tau * step as f64 / n_fft as f64;
                        (angle.cos(), angle.sin())
                    }
                    Some(0) => (1.0, 0.0),
                    Some(1) => (0.0, 1.0),
                    Some(2) => (-1.0, 0.0),
                    _ => (0.0, -1.0),
                };
                cos.push(c);
                sin.push(s);
            }
        }
        Self {
            n_fft,
            hop,
            window,
            cos,
            sin,
        }
    }

    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    pub fn frames(&self, samples: usize) -> usize {
        1 + samples / self.hop
    }

    pub fn forward(&self, signal: &[f32]) -> Vec<f32> {
        let bins = self.bins();
        let len = signal.len();
        let frames = self.frames(len);
        let mut out = vec![0.0f32; frames * 2 * bins];
        let rows_per_task = balanced_rows(frames, 2 * bins);
        let mut scratch = vec![0.0f64; frames.div_ceil(rows_per_task) * self.n_fft];
        let scratch = team::SendPtr::new(&mut scratch);
        team::chunks_mut(&mut out, rows_per_task * 2 * bins, |task, chunk| {
            // SAFETY: each task uses its own frame buffer within `scratch`.
            let frame = unsafe { scratch.slice_mut(task * self.n_fft, self.n_fft) };
            stft_forward_rows(self, signal, task * rows_per_task, frame, chunk);
        });
        out
    }

    /// `magnitude` and `phase` (each
    /// `frames x bins`, read by `at(t, k)`). `hop * (frames - 1)` samples.
    pub fn inverse(
        &self,
        frames: usize,
        mut at: impl FnMut(usize, usize) -> (f32, f32),
    ) -> Vec<f32> {
        let (n_fft, bins, hop) = (self.n_fft, self.bins(), self.hop);
        let total = n_fft + hop * (frames - 1);
        let mut sum = vec![0.0f64; total];
        let mut envelope = vec![0.0f64; total];
        let mut re = vec![0.0f64; bins];
        let mut im = vec![0.0f64; bins];
        let mut frame = vec![0.0f64; n_fft];
        let inv_n = 1.0 / n_fft as f64;
        for t in 0..frames {
            for k in 0..bins {
                let (magnitude, phase) = at(t, k);
                let (magnitude, phase) = (f64::from(magnitude), f64::from(phase));
                re[k] = magnitude * phase.cos();
                im[k] = magnitude * phase.sin();
            }
            istft_frame_add(
                self,
                &re,
                &im,
                inv_n,
                &mut frame,
                &mut sum[t * hop..t * hop + n_fft],
                &mut envelope[t * hop..t * hop + n_fft],
            );
        }
        let half = n_fft / 2;
        (half..half + hop * (frames - 1))
            .map(|i| {
                if envelope[i] > 1e-11 {
                    (sum[i] / envelope[i]) as f32
                } else {
                    0.0
                }
            })
            .collect()
    }
}

multiversion! {
    fn stft_forward_rows(stft: &Stft, signal: &[f32], first_frame: usize, frame: &mut [f64], out: &mut [f32]) {
        let (n_fft, bins) = (stft.n_fft, stft.bins());
        let half = n_fft / 2;
        let len = signal.len();
        for (offset, row) in out.chunks_exact_mut(2 * bins).enumerate() {
            let t = first_frame + offset;
            for (n, v) in frame.iter_mut().enumerate() {
                let mut j = (t * stft.hop + n) as isize - half as isize;
                if j < 0 {
                    j = -j;
                }
                if j >= len as isize {
                    j = 2 * (len as isize - 1) - j;
                }
                *v = f64::from(signal[j.clamp(0, len as isize - 1) as usize]) * stft.window[n];
            }
            for k in 0..bins {
                let cos_k = &stft.cos[k * n_fft..(k + 1) * n_fft];
                let sin_k = &stft.sin[k * n_fft..(k + 1) * n_fft];
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for ((&v, &c), &s) in frame.iter().zip(cos_k).zip(sin_k) {
                    re += v * c;
                    im -= v * s;
                }
                if k == 0 || 2 * k == n_fft {
                    im = 0.0;
                }
                row[k] = (re * re + im * im).sqrt() as f32;
                row[bins + k] = im.atan2(re) as f32;
            }
        }
    }
}

multiversion! {
    fn istft_frame_add(
        stft: &Stft,
        re: &[f64],
        im: &[f64],
        inv_n: f64,
        frame: &mut [f64],
        sum: &mut [f64],
        envelope: &mut [f64],
    ) {
        let (n_fft, bins) = (stft.n_fft, stft.bins());
        frame.fill(re[0]);
        for k in 1..bins {
            let weight = if 2 * k == n_fft { 1.0 } else { 2.0 };
            let wr = weight * re[k];
            let wi = weight * im[k];
            let cos_k = &stft.cos[k * n_fft..(k + 1) * n_fft];
            let sin_k = &stft.sin[k * n_fft..(k + 1) * n_fft];
            for ((dst, &c), &s) in frame.iter_mut().zip(cos_k).zip(sin_k) {
                *dst += wr * c - wi * s;
            }
        }
        for (((s, e), &v), &w) in sum.iter_mut().zip(envelope.iter_mut()).zip(frame.iter()).zip(&stft.window) {
            *s += (v * inv_n) * w;
            *e += w * w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_forward_and_inverse_round_trip() {
        let len = 512;
        let signal: Vec<f32> = (0..len)
            .map(|i| {
                let x = i as f32 / 24_000.0;
                0.5 * (std::f32::consts::TAU * 350.0 * x).sin()
                    + 0.3 * (std::f32::consts::TAU * 1850.0 * x).cos()
                    - 0.2 * (std::f32::consts::TAU * 4200.0 * x).sin()
            })
            .collect();
        for (n_fft, hop) in [(20usize, 5usize), (32, 8)] {
            let stft = Stft::new(n_fft, hop);
            let bins = stft.bins();
            let frames = stft.frames(len);
            let spec = stft.forward(&signal);
            assert_eq!(spec.len(), frames * 2 * bins);
            let recon = stft.inverse(frames, |t, k| {
                let row = &spec[t * 2 * bins..(t + 1) * 2 * bins];
                (row[k], row[bins + k])
            });
            assert_eq!(recon.len(), hop * (frames - 1));
            for i in n_fft..recon.len().saturating_sub(n_fft) {
                let diff = (recon[i] - signal[i]).abs();
                assert!(
                    diff <= 1e-5,
                    "stft ({n_fft}, {hop}) sample {i}: got={} want={} diff={diff}",
                    recon[i],
                    signal[i]
                );
            }
        }
    }
}
