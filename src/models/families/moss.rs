//! MOSS-Transcribe-Diarize (OpenMOSS): Whisper medium's encoder over 30 s
//! chunks (each trimmed to the frames its audio covers), 4 frames joined
//! and passed through an adaptor (linear, SiLU, linear, layer norm), and
//! Qwen3 0.6B. The adaptor rows fill the audio span of the model's prompt,
//! with the seconds written in digit tokens every 5 s. The model writes
//! `[start][S01]text[end]`; the text keeps the words of each turn. The
//! structure follows transcribe.cpp's port of the OpenMOSS code.

use crate::audio::{self, whisper_mel::WhisperMel};
use crate::cpu::{activation, buf::Buf, gemm::Dense};
use crate::models::config::Json;
use crate::models::families::whisper::model::{Encoder, EncoderShape};
use crate::models::files::Files;
use crate::models::layers::{linear, Norm};
use crate::models::parts::causal_lm::CausalLm;
use crate::text::bpe::Bpe;
use crate::{Device, Options, Precision};

/// Audio per encoder pass.
const CHUNK_SAMPLES: usize = crate::audio::whisper_mel::WINDOW_SAMPLES;
/// Longest window per prompt.
const WINDOW_SAMPLES: usize = 120 * 16_000;
const MIN_NEW_TOKENS: usize = 256;
const TOKENS_PER_SECOND: usize = 12;
/// The adaptor's layer norm epsilon.
const ADAPTOR_EPS: f32 = 1e-6;

pub struct Moss {
    mel: WhisperMel,
    encoder: Encoder,
    d_encoder: usize,
    merge: usize,
    fc1: Dense,
    fc2: Dense,
    norm: Norm,
    decoder: CausalLm,
    bpe: Bpe,
    audio_token: usize,
    tokens_per_second: f32,
    marker_seconds: usize,
    prefix: Vec<usize>,
    suffix: Vec<usize>,
    digits: Vec<usize>,
    eos: usize,
}

impl Moss {
    pub fn load(
        files: &Files,
        json: &Json,
        precision: Precision,
        device: Device,
    ) -> Result<Self, String> {
        let e = |key: &str| json.num(&format!("encoder_config.{key}"));
        let shape = EncoderShape {
            d: e("hidden_size")?,
            layers: e("num_hidden_layers")?,
            heads: e("num_attention_heads")?,
            ffn: e("intermediate_size")?,
            mels: e("num_mel_bins")?,
            positions: e("max_source_positions")?,
        };
        let (d_encoder, mels) = (shape.d, shape.mels);
        let merge = json.num("audio_merge_size")?;
        if json.num("adaptor_config.input_dim")? != merge * d_encoder {
            return Err("config.json: the adaptor does not fit the encoder".into());
        }
        let ids = |key: &str| -> Vec<usize> {
            json.get(key)
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_u64().map(|v| v as usize))
                .collect()
        };
        let digits = ids("digit_tokens");
        if digits.len() != 10 {
            return Err("config.json: MOSS needs its ten digit tokens".into());
        }
        let t = files.tensors(precision, device)?;
        let encoder = Encoder::load(&t, "model.encoder", shape)?;
        let decoder = CausalLm::from_tensors(&t, json, ("llm_config.", "language_model."))?;
        let width = decoder.width();
        let model = Self {
            mel: WhisperMel::new(mels),
            encoder,
            d_encoder,
            merge,
            fc1: linear(&t, "adaptor.fc1", width, merge * d_encoder)?,
            fc2: linear(&t, "adaptor.fc2", width, width)?,
            norm: Norm::load(&t, "adaptor.norm_out", width)?.with_eps(ADAPTOR_EPS),
            decoder,
            bpe: Bpe::parse(&files.read("tokenizer.json")?)?,
            audio_token: json.num("audio_token_id")?,
            tokens_per_second: json.float("audio_tokens_per_second").unwrap_or(12.5) as f32,
            marker_seconds: json.num("time_marker_every_seconds").unwrap_or(0),
            prefix: ids("prompt_prefix_tokens"),
            suffix: ids("prompt_suffix_tokens"),
            digits,
            eos: json.num("eos_token_id")?,
        };
        Ok(model)
    }

    /// Adaptor rows of `samples`, `n x width`: every 30 s chunk encoded
    /// whole, trimmed to the frames its audio covers, then joined.
    fn audio_rows(&self, samples: &[f32]) -> (Buf, usize) {
        let d = self.d_encoder;
        let stride = crate::audio::whisper_mel::HOP * 2 * self.merge;
        let mut frames = Vec::new();
        for chunk in samples.chunks(CHUNK_SAMPLES) {
            let tokens = (chunk.len() - 1) / stride + 1;
            let encoded = self.encoder.encode(&self.mel.compute(chunk));
            let keep = (tokens * self.merge).min(encoded.len() / d);
            frames.extend_from_slice(&encoded[..keep * d]);
        }
        let n = frames.len() / (d * self.merge);
        let mut h = self.fc1.forward(&frames[..n * d * self.merge], n);
        activation::silu(&mut h);
        let x = self.fc2.forward(&h, n);
        (self.norm.apply(&x, self.decoder.width()), n)
    }

    /// The prompt with `n` audio tokens, the seconds written every
    /// `marker_seconds`: `(ids, positions of the audio tokens)`.
    fn prompt(&self, n: usize) -> (Vec<usize>, Vec<usize>) {
        let mut ids = self.prefix.clone();
        let mut audio = Vec::with_capacity(n);
        let mut push_audio = |ids: &mut Vec<usize>, count: usize| {
            for _ in 0..count {
                audio.push(ids.len());
                ids.push(self.audio_token);
            }
        };
        let per_marker = (self.tokens_per_second * self.marker_seconds as f32) as usize;
        if self.marker_seconds == 0 || per_marker == 0 {
            push_audio(&mut ids, n);
        } else {
            let duration = (n as f32 / self.tokens_per_second) as usize;
            let mut used = 0;
            let mut second = self.marker_seconds;
            while second <= duration {
                let at = (second / self.marker_seconds) * per_marker;
                push_audio(&mut ids, at.saturating_sub(used));
                used = used.max(at);
                for digit in second.to_string().bytes() {
                    ids.push(self.digits[(digit - b'0') as usize]);
                }
                second += self.marker_seconds;
            }
            push_audio(&mut ids, n.saturating_sub(used));
        }
        ids.extend_from_slice(&self.suffix);
        (ids, audio)
    }

    /// The text of each window that holds speech: `(first sample, samples, text)`.
    pub fn pieces(&self, samples: &[f32], _: &Options) -> Vec<(usize, usize, String)> {
        let mut pieces = Vec::new();
        for window in audio::windows(samples, WINDOW_SAMPLES) {
            if window.is_empty() || audio::is_silent(window) {
                continue;
            }
            let (audio, n) = self.audio_rows(window);
            let (ids, positions) = self.prompt(n);
            let d = self.decoder.width();
            let mut rows = self.decoder.embed(&ids);
            for (row, at) in positions.iter().enumerate() {
                rows[at * d..(at + 1) * d].copy_from_slice(&audio[row * d..(row + 1) * d]);
            }
            let seconds = window.len() / crate::SAMPLE_RATE as usize;
            let max_new = MIN_NEW_TOKENS.max(seconds * TOKENS_PER_SECOND);
            let out = self.decoder.generate_rows(rows, max_new, &[self.eos]);
            let text = plain_text(&self.bpe.decode(&out));
            if !text.is_empty() {
                pieces.push((audio::offset(samples, window), window.len(), text));
            }
        }
        pieces
    }
}

/// The words of `[0.48][S01]Welcome[1.66]`-style output: the text between
/// the time and speaker markers, each piece trimmed, joined by spaces.
fn plain_text(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut pieces = Vec::new();
    let (mut start, mut i) = (0, 0);
    while i < bytes.len() {
        if bytes[i] == b'[' {
            if let Some(end) = marker_end(bytes, i) {
                pieces.push(raw[start..i].trim());
                start = end;
                i = end;
                continue;
            }
        }
        i += 1;
    }
    pieces.push(raw[start..].trim());
    pieces.retain(|piece| !piece.is_empty());
    pieces.join(" ")
}

/// The end of a `[1.25]` or `[S01]` marker at `at`, if one starts there.
fn marker_end(bytes: &[u8], at: usize) -> Option<usize> {
    let mut i = at + 1;
    let speaker = bytes.get(i) == Some(&b'S');
    if speaker {
        i += 1;
    }
    let digits = |i: &mut usize| {
        let first = *i;
        while bytes.get(*i).is_some_and(u8::is_ascii_digit) {
            *i += 1;
        }
        *i > first
    };
    if !digits(&mut i) {
        return None;
    }
    if !speaker && bytes.get(i) == Some(&b'.') {
        i += 1;
        digits(&mut i);
    }
    (bytes.get(i) == Some(&b']')).then_some(i + 1)
}
