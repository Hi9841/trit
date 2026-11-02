//! Canary-Qwen 2.5B (NVIDIA SALM): the Canary 1B flash FastConformer, a
//! projection to the language model's width, and Qwen3 1.7B. The encoder
//! frames take the place of the `<|audioplaceholder|>` token of a chat
//! prompt ("Transcribe the following: ..."), and Qwen3 writes the text. The
//! prompt follows transcribe.cpp's port of NeMo's `SALM`.

use crate::audio::{self, mel::MelFrontend};
use crate::cpu::{buf::Buf, gemm::Dense};
use crate::models::config::Json;
use crate::models::files::Files;
use crate::models::layers::linear;
use crate::models::parts::{fastconformer as conformer, qwen3::Qwen3};
use crate::text::bpe::Bpe;
use crate::{Device, Options, Precision};

/// The model's limit per pass (its card: 40 s).
const WINDOW_SAMPLES: usize = 40 * 16_000;
const MIN_NEW_TOKENS: usize = 128;
const TOKENS_PER_SECOND: usize = 8;

pub struct CanaryQwen {
    frontend: MelFrontend,
    encoder: conformer::Parakeet,
    /// From the encoder width to the language model's.
    projection: Dense,
    decoder: Qwen3,
    bpe: Bpe,
    /// The prompt before and after the audio frames.
    head: Vec<usize>,
    tail: Vec<usize>,
    stop: Vec<usize>,
}

impl CanaryQwen {
    pub fn load(
        files: &Files,
        json: &Json,
        precision: Precision,
        device: Device,
    ) -> Result<Self, String> {
        let t = files.tensors(precision, device)?;
        let config = conformer::Config::parse(json)?;
        let frontend = MelFrontend::new(config.mels).normalized(config.normalize_features);
        let width = config.hidden;
        let encoder = conformer::Parakeet::load(&t, config)?;
        let decoder = Qwen3::from_tensors(&t, json, ("llm_config.", "llm."))?;
        let projection = linear(&t, "model.encoder_decoder_proj", decoder.width(), width)?;
        drop(t);
        encoder.warm_positions();
        let bpe = Bpe::parse(&files.read("tokenizer.json")?)?;
        let id = |token: &str| {
            bpe.id(token)
                .ok_or_else(|| format!("the tokenizer has no {token}"))
        };
        let (start, end) = (id("<|im_start|>")?, id("<|im_end|>")?);
        let mut head = vec![start];
        head.extend(bpe.encode_plain("user\nTranscribe the following: "));
        let mut tail = vec![end];
        tail.extend(bpe.encode_plain("\n"));
        tail.push(start);
        tail.extend(bpe.encode_plain("assistant\n"));
        let stop = vec![end, id("<|endoftext|>")?];
        Ok(Self {
            frontend,
            encoder,
            projection,
            decoder,
            bpe,
            head,
            tail,
            stop,
        })
    }

    /// The text of each window that holds speech: `(first sample, samples, text)`.
    pub fn pieces(&self, samples: &[f32], _: &Options) -> Vec<(usize, usize, String)> {
        let mut pieces = Vec::new();
        for window in audio::windows(samples, WINDOW_SAMPLES) {
            if audio::is_silent(window) {
                continue;
            }
            let (features, frames) = self.frontend.compute(window);
            // The subsampling needs at least one output frame per stage.
            if frames < 8 {
                continue;
            }
            let (encoded, t) = self.encoder.encode(&features, frames);
            let audio = self.projection.forward(&encoded, t);
            let d = self.decoder.width();
            let (head, tail) = (self.head.len(), self.tail.len());
            let mut rows = Buf::scratch((head + t + tail) * d);
            rows[..head * d].copy_from_slice(&self.decoder.embed(&self.head));
            rows[head * d..(head + t) * d].copy_from_slice(&audio);
            rows[(head + t) * d..].copy_from_slice(&self.decoder.embed(&self.tail));
            let seconds = window.len() / crate::SAMPLE_RATE as usize;
            let max_new = MIN_NEW_TOKENS.max(seconds * TOKENS_PER_SECOND);
            let ids = self.decoder.generate_rows(rows, max_new, &self.stop);
            let text = self.bpe.decode(&ids).trim().to_string();
            if !text.is_empty() {
                pieces.push((audio::offset(samples, window), window.len(), text));
            }
        }
        pieces
    }
}
