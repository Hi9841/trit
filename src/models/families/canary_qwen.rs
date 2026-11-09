//! Canary-Qwen 2.5B (NVIDIA SALM): the Canary 1B flash FastConformer, a
//! projection to the language model's width, and Qwen3 1.7B. The encoder
//! frames take the place of the `<|audioplaceholder|>` token of a chat
//! prompt ("Transcribe the following: ..."), and Qwen3 writes the text. The
//! prompt follows transcribe.cpp's port of NeMo's `SALM`.

use crate::audio::mel::MelFrontend;
use crate::config::Json;
use crate::cpu::gemm::Dense;
use crate::layers::linear;
use crate::models::files::Files;
use crate::models::parts::{
    causal_lm::{CausalLm, TokenBudget},
    fastconformer as conformer,
};
use crate::models::{speech_windows, Family};
use crate::text::bpe::Bpe;
use crate::{Device, Options, Precision, Segment};

/// The model's limit per pass (its card: 40 s).
const WINDOW_SAMPLES: usize = 40 * crate::SAMPLE_RATE as usize;
const BUDGET: TokenBudget = TokenBudget {
    least: 128,
    per_second: 8,
};

pub struct CanaryQwen {
    frontend: MelFrontend,
    encoder: conformer::Parakeet,
    /// From the encoder width to the language model's.
    projection: Dense,
    decoder: CausalLm,
    bpe: Bpe,
    /// The prompt before and after the audio frames.
    head: Vec<usize>,
    tail: Vec<usize>,
    stop: Vec<usize>,
}

impl Family for CanaryQwen {
    fn load(
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
        let decoder = CausalLm::from_tensors(&t, json, ("llm_config.", "llm."))?;
        let projection = linear(&t, "model.encoder_decoder_proj", decoder.width(), width)?;
        drop(t);
        encoder.warm_positions();
        let bpe = Bpe::parse(&files.read("tokenizer.json")?)?;
        let id = |token: &str| bpe.token(token);
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

    fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment> {
        speech_windows(samples, WINDOW_SAMPLES, options, |window| {
            let (features, frames) = self.frontend.compute(window);
            if frames < conformer::MIN_FRAMES {
                return String::new();
            }
            let (encoded, t) = self.encoder.encode(&features, frames);
            let audio = self.projection.forward(&encoded, t);
            let budget = BUDGET.tokens(window.len());
            let ids = self
                .decoder
                .answer(&self.head, &audio, &self.tail, budget, &self.stop);
            self.bpe.decode(&ids).trim().to_string()
        })
    }
}
