//! Power mels of a centered STFT from a model file's own window and
//! filterbank (Granite Speech, Voxtral Realtime): frames centered on every
//! `hop`-th sample with reflected samples past each end (torch.stft with
//! `center=True`), the window in the middle of the transform, the power
//! spectrum, and the filterbank. The caller takes the log.

use super::dft::PowerSpectrum;
use crate::config::Json;
use crate::cpu::{buf::Buf, fgemm, team, tensors::Tensors};

pub struct CenteredMel {
    pub n_fft: usize,
    pub hop: usize,
    pub mels: usize,
    spectrum: PowerSpectrum,
    /// `mels x bins`.
    filters: Vec<f32>,
}

impl CenteredMel {
    /// `n_fft`, `win_length`, `num_mel_bins`, and `hop_length` of
    /// `feature_extractor`; the tensors `frontend.window` and
    /// `frontend.mel_filterbank`.
    pub fn load(json: &Json, t: &Tensors) -> Result<Self, String> {
        let fe = |key: &str| json.num(&format!("feature_extractor.{key}"));
        let (n_fft, win, mels) = (fe("n_fft")?, fe("win_length")?, fe("num_mel_bins")?);
        if win > n_fft {
            return Err("config.json: the window is longer than the transform".into());
        }
        let window = t.f32_len("frontend.window", win)?;
        let mut padded = vec![0.0f32; n_fft];
        padded[(n_fft - win) / 2..][..win].copy_from_slice(&window);
        Ok(Self {
            n_fft,
            hop: fe("hop_length")?,
            mels,
            spectrum: PowerSpectrum::new(n_fft, &padded),
            filters: t.f32_len("frontend.mel_filterbank", mels * (n_fft / 2 + 1))?,
        })
    }

    /// `frames x mels`. `samples` must be longer than `n_fft / 2`.
    pub fn power(&self, samples: &[f32], frames: usize) -> Buf {
        let len = samples.len() as isize;
        let reflect = |i: isize| -> f32 {
            let i = if i < 0 { -i } else { i };
            let i = if i >= len { 2 * (len - 1) - i } else { i };
            samples[i.clamp(0, len - 1) as usize]
        };
        let half = (self.n_fft / 2) as isize;
        let mut windows = Buf::scratch(frames * self.n_fft);
        team::chunks_mut(&mut windows, self.n_fft, |f, row| {
            let start = (f * self.hop) as isize - half;
            for (n, value) in row.iter_mut().enumerate() {
                *value = reflect(start + n as isize);
            }
        });
        let power = self.spectrum.compute(&windows, frames);
        fgemm::matmul(
            &power,
            frames,
            self.spectrum.bins(),
            &self.filters,
            self.mels,
        )
    }
}
