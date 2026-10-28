//! Audio in: WAV files, the log-mel frontends of the model families, and
//! long recordings cut into windows.

pub mod mel;
pub mod stft;
pub mod wav;
pub mod whisper_mel;

use std::borrow::Cow;

/// Windows of at most `max` samples, cut at the quietest 100 ms in the last
/// third of each window, so a cut rarely splits a word.
pub fn windows(samples: &[f32], max: usize) -> Vec<&[f32]> {
    const FRAME: usize = 1600;
    let mut out = Vec::new();
    let mut rest = samples;
    while rest.len() > max {
        let energy =
            |start: usize| -> f32 { rest[start..start + FRAME].iter().map(|s| s * s).sum() };
        let cut = (max * 2 / 3..max.saturating_sub(FRAME))
            .step_by(FRAME / 2)
            .min_by(|a, b| energy(*a).total_cmp(&energy(*b)))
            .map_or(max, |start| start + FRAME / 2);
        out.push(&rest[..cut]);
        rest = &rest[cut..];
    }
    if !rest.is_empty() {
        out.push(rest);
    }
    out
}

/// Where `part` (a window of `all`) starts in `all`, in samples.
pub fn offset(all: &[f32], part: &[f32]) -> usize {
    (part.as_ptr() as usize - all.as_ptr() as usize) / std::mem::size_of::<f32>()
}

/// Samples every frontend can read: NaN and infinity become 0, and values
/// past `PEAK` become `PEAK`. One bad sample from a driver or a resampler
/// otherwise turns the features of its whole window into NaN, and the take
/// into no text. Every sample in range (any real recording) is borrowed.
pub fn sanitized(samples: &[f32]) -> Cow<'_, [f32]> {
    // Far above any PCM scale (32-bit full scale is 2.1e9), far below the
    // level where a power spectrum overflows f32 (about 1e16).
    const PEAK: f32 = 1e10;
    // NaN fails the comparison too.
    if samples.iter().all(|s| s.abs() <= PEAK) {
        return Cow::Borrowed(samples);
    }
    samples
        .iter()
        .map(|s| {
            if s.is_finite() {
                s.clamp(-PEAK, PEAK)
            } else {
                0.0
            }
        })
        .collect()
}

/// Below -60 dBFS RMS a window holds no speech. Whisper writes a guess for
/// such a window ("you"); its own no-speech check misses digital silence
/// with large-v3-turbo, so the families with a text decoder skip it here.
pub fn is_silent(samples: &[f32]) -> bool {
    const SILENT_RMS: f32 = 1e-3;
    let energy: f32 = samples.iter().map(|s| s * s).sum();
    samples.is_empty() || energy / (samples.len() as f32) < SILENT_RMS * SILENT_RMS
}
