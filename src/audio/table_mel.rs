//! Log-mel features from a model file's own filterbank and window (GigaAM):
//! frames of `win` samples every `hop` with no centering, the window, zeros
//! up to `n_fft`, the power spectrum, the filterbank, and
//! `ln(clamp(x, floor, ceiling))`. No normalization.

use super::dft::PowerSpectrum;
use crate::cpu::{buf::Buf, fgemm, team};

pub struct TableMel {
    hop: usize,
    win: usize,
    mels: usize,
    spectrum: PowerSpectrum,
    /// `mels x bins`.
    filters: Vec<f32>,
    floor: f32,
    ceiling: f32,
}

/// The shape of the front end: transform size, hop, window length, and the
/// clamp before the log.
pub struct TableShape {
    pub n_fft: usize,
    pub hop: usize,
    pub win: usize,
    pub floor: f32,
    pub ceiling: f32,
}

impl TableMel {
    /// `filters` is `mels x (n_fft / 2 + 1)`, `window` has `win` values.
    pub fn new(shape: TableShape, filters: Vec<f32>, window: &[f32]) -> Result<Self, String> {
        let TableShape {
            n_fft,
            hop,
            win,
            floor,
            ceiling,
        } = shape;
        let bins = n_fft / 2 + 1;
        if win > n_fft || window.len() != win || hop == 0 || !filters.len().is_multiple_of(bins) {
            return Err("the front end's window or filterbank does not fit its transform".into());
        }
        Ok(Self {
            hop,
            win,
            mels: filters.len() / bins,
            spectrum: PowerSpectrum::new(n_fft, window),
            filters,
            floor,
            ceiling,
        })
    }

    /// `frames x mels`, frames = `(samples - win) / hop + 1`; none for less
    /// than one window.
    pub fn compute(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        if audio.len() < self.win {
            return (Vec::new(), 0);
        }
        let frames = (audio.len() - self.win) / self.hop + 1;
        let mut windows = Buf::scratch(frames * self.win);
        team::chunks_mut(&mut windows, self.win, |f, row| {
            row.copy_from_slice(&audio[f * self.hop..][..self.win]);
        });
        let power = self.spectrum.compute(&windows, frames);
        let mel = fgemm::matmul(
            &power,
            frames,
            self.spectrum.bins(),
            &self.filters,
            self.mels,
        );
        let out = mel
            .iter()
            .map(|m| m.clamp(self.floor, self.ceiling).ln())
            .collect();
        (out, frames)
    }
}
