//! WAV files: in, to the 16 kHz mono samples that `Trit::transcribe`
//! takes; out, from the samples that `Voice::speak` gives.

use std::path::Path;

/// WAVE_FORMAT_EXTENSIBLE: the real format code is in the SubFormat field.
const EXTENSIBLE: u16 = 0xFFFE;

pub fn read_wav(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let not_wav = || format!("{} is not a WAV file", path.display());
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(not_wav());
    }
    let mut format = None;
    let mut data = None;
    let mut offset = 12usize;
    while offset.checked_add(8).is_some_and(|end| end <= bytes.len()) {
        let id = &bytes[offset..offset + 4];
        let size = le_u32(&bytes, offset + 4) as usize;
        let body = offset + 8;
        // A writer that streams sets the data size too large; read what is there.
        let end = body.saturating_add(size).min(bytes.len());
        if id == b"fmt " && end - body >= 16 {
            let field = |at: usize| u16::from_le_bytes([bytes[body + at], bytes[body + at + 1]]);
            let rate = le_u32(&bytes, body + 4);
            // WAVE_FORMAT_EXTENSIBLE: the format code is the start of the
            // SubFormat field (1 PCM, 3 IEEE float), not the bit depth.
            let tag = match field(0) {
                EXTENSIBLE if end - body >= 26 => field(24),
                tag => tag,
            };
            format = Some((tag, field(2), rate, field(14)));
        } else if id == b"data" {
            data = Some(&bytes[body..end]);
        }
        offset = body.saturating_add(size).saturating_add(size % 2);
    }
    let (tag, channels, rate, bits) = format.ok_or_else(not_wav)?;
    let data = data.ok_or_else(not_wav)?;
    if rate != crate::SAMPLE_RATE {
        return Err(format!(
            "{} is {rate} Hz; Trit needs 16000 Hz. Convert it: ffmpeg -i \"{}\" -ar 16000 -ac 1 out.wav",
            path.display(),
            path.display()
        ));
    }
    // 1 = PCM, 3 = IEEE float.
    let sample: fn(&[u8]) -> f32 = match (tag, bits) {
        (1, 16) => |b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0,
        (1, 24) => |b| i32::from_le_bytes([0, b[0], b[1], b[2]]) as f32 / 2_147_483_648.0,
        (1, 32) => |b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0,
        (3, 32) => |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        _ => {
            return Err(format!(
                "{} is {bits}-bit format {tag}; Trit reads 16, 24, or 32-bit PCM and 32-bit float. Convert it: ffmpeg -i \"{}\" -ar 16000 -ac 1 -c:a pcm_s16le out.wav",
                path.display(),
                path.display()
            ))
        }
    };
    let width = usize::from(bits / 8);
    let channels = usize::from(channels.max(1));
    Ok(data
        .chunks_exact(width * channels)
        .map(|frame| frame.chunks_exact(width).map(sample).sum::<f32>() / channels as f32)
        .collect())
}

pub fn write_wav(path: &Path, samples: &[f32], rate: u32) -> Result<(), String> {
    let data = u32::try_from(samples.len() * 2)
        .ok()
        .filter(|bytes| *bytes <= u32::MAX - 36)
        .ok_or("the audio is too long for one WAV file")?;
    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&(rate * 2).to_le_bytes()); // bytes per second
    bytes.extend_from_slice(&2u16.to_le_bytes()); // bytes per frame
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data.to_le_bytes());
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// The caller has checked that four bytes are there.
fn le_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}
