//! A short-time Fourier transform and its inverse, as `torch.stft` and
//! `torch.istft` compute them with `center=True` (reflect padding) and a
//! periodic Hann window as long as the transform. For the small transforms
//! of a vocoder (iSTFTNet: 20 points), a direct DFT with tables is exact
//! enough and needs no FFT.

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
            .map(|n| 0.5 - 0.5 * (tau * n as f64 / n_fft as f64).cos())
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

    /// Frames of `samples`: `1 + samples / hop`.
    pub fn frames(&self, samples: usize) -> usize {
        1 + samples / self.hop
    }

    /// The magnitude and phase of each frame: row `t` holds the magnitudes
    /// of every bin, then their phases (`atan2(im, re)`).
    pub fn forward(&self, signal: &[f32]) -> Vec<f32> {
        let (n_fft, bins) = (self.n_fft, self.bins());
        let half = n_fft / 2;
        let len = signal.len();
        // Reflect padding of n_fft / 2 on both sides, as torch.stft pads.
        let at = |i: isize| -> f64 {
            let mut j = i - half as isize;
            if j < 0 {
                j = -j;
            }
            if j >= len as isize {
                j = 2 * (len as isize - 1) - j;
            }
            f64::from(signal[j.clamp(0, len as isize - 1) as usize])
        };
        let frames = self.frames(len);
        let mut out = vec![0.0f32; frames * 2 * bins];
        let mut frame = vec![0.0f64; n_fft];
        for t in 0..frames {
            for (n, v) in frame.iter_mut().enumerate() {
                *v = at((t * self.hop + n) as isize) * self.window[n];
            }
            let row = &mut out[t * 2 * bins..(t + 1) * 2 * bins];
            for k in 0..bins {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (n, v) in frame.iter().enumerate() {
                    re += v * self.cos[k * n_fft + n];
                    im -= v * self.sin[k * n_fft + n];
                }
                // A real FFT gives the first and the middle bin an imaginary part of exactly 0.
                if k == 0 || 2 * k == n_fft {
                    im = 0.0;
                }
                row[k] = (re * re + im * im).sqrt() as f32;
                row[bins + k] = im.atan2(re) as f32;
            }
        }
        out
    }

    /// The signal of `frames` frames of `magnitude` and `phase` (each
    /// `frames x bins`, read by `at(t, k)`): an inverse real DFT per frame,
    /// windowed, overlapped and added, divided by the summed squared window,
    /// with the center padding cut. `hop * (frames - 1)` samples.
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
        for t in 0..frames {
            for k in 0..bins {
                let (magnitude, phase) = at(t, k);
                let (magnitude, phase) = (f64::from(magnitude), f64::from(phase));
                re[k] = magnitude * phase.cos();
                im[k] = magnitude * phase.sin();
            }
            for n in 0..n_fft {
                // irfft: the imaginary parts of the first and the middle bin do not count.
                let mut value = re[0];
                for k in 1..bins {
                    let weight = if 2 * k == n_fft { 1.0 } else { 2.0 };
                    value += weight
                        * (re[k] * self.cos[k * n_fft + n] - im[k] * self.sin[k * n_fft + n]);
                }
                value /= n_fft as f64;
                sum[t * hop + n] += value * self.window[n];
                envelope[t * hop + n] += self.window[n] * self.window[n];
            }
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
