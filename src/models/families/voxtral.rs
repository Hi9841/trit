//! Voxtral (Mistral AI): Whisper large's encoder over 30 s windows of
//! log-mel features, a projector that joins 4 encoder frames and maps them
//! (GELU between two linear layers) to the language model's width, and a
//! Mistral language model (`parts::causal_lm`). The 375 rows of a window take
//! the place of the `[AUDIO]` tokens of Mistral's transcription request:
//! `<s>[INST][BEGIN_AUDIO][AUDIO]...[/INST]lang:en[TRANSCRIBE]`. The structure
//! follows transcribe.cpp's port of `VoxtralForConditionalGeneration`.

use crate::audio::{self, whisper_mel::WhisperMel};
use crate::cpu::{activation, buf::Buf, gemm::Dense};
use crate::models::config::Json;
use crate::models::families::whisper::model::{Encoder, EncoderShape};
use crate::models::files::Files;
use crate::models::layers::linear;
use crate::models::parts::causal_lm::CausalLm;
use crate::text::bpe::Bpe;
use crate::{Device, Options, Precision};

/// One encoder window: 30 s.
const WINDOW_SAMPLES: usize = crate::audio::whisper_mel::WINDOW_SAMPLES;
const MIN_NEW_TOKENS: usize = 128;
const TOKENS_PER_SECOND: usize = 8;

pub struct Voxtral {
    mel: WhisperMel,
    encoder: Encoder,
    d_encoder: usize,
    /// Encoder frames per language model row.
    join: usize,
    linear1: Dense,
    linear2: Dense,
    decoder: CausalLm,
    bpe: Bpe,
    languages: Vec<String>,
    tokens: Tokens,
}

/// The request's control tokens.
struct Tokens {
    bos: usize,
    eos: usize,
    inst: usize,
    inst_end: usize,
    begin_audio: usize,
    transcribe: usize,
}

impl Voxtral {
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
        let join = json.num("projector_config.downsample_factor")?;
        if json.num("projector_config.input_dim")? != join * d_encoder {
            return Err("config.json: the projector does not fit the encoder".into());
        }
        let t = files.tensors(precision, device)?;
        let encoder = Encoder::load(&t, "model.encoder", shape)?;
        let decoder = CausalLm::from_tensors(&t, json, ("llm_config.", "language_model."))?;
        let width = decoder.width();
        let linear1 = linear(
            &t,
            "multi_modal_projector.linear_1",
            width,
            join * d_encoder,
        )?;
        let linear2 = linear(&t, "multi_modal_projector.linear_2", width, width)?;
        drop(t);
        let bpe = Bpe::parse(&files.read("tokenizer.json")?)?;
        let id = |token: &str| {
            bpe.id(token)
                .ok_or_else(|| format!("the tokenizer has no {token}"))
        };
        let tokens = Tokens {
            bos: json.num("bos_token_id")?,
            eos: json.num("eos_token_id")?,
            inst: id("[INST]")?,
            inst_end: id("[/INST]")?,
            begin_audio: id("[BEGIN_AUDIO]")?,
            transcribe: id("[TRANSCRIBE]")?,
        };
        let languages = json
            .get("languages")
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|code| code.as_str().map(str::to_string))
            .collect();
        Ok(Self {
            mel: WhisperMel::new(mels),
            encoder,
            d_encoder,
            join,
            linear1,
            linear2,
            decoder,
            bpe,
            languages,
            tokens,
        })
    }

    /// The text of each window that holds speech: `(first sample, samples, text)`.
    /// A language the model lists goes into the request (`lang:fr`); none
    /// lets the model find it.
    pub fn pieces(&self, samples: &[f32], options: &Options) -> Vec<(usize, usize, String)> {
        let language = options.language.as_deref().map(str::trim).filter(|code| {
            self.languages
                .iter()
                .any(|known| known.eq_ignore_ascii_case(code))
        });
        let k = &self.tokens;
        let head = [k.bos, k.inst, k.begin_audio];
        let mut tail = vec![k.inst_end];
        if let Some(code) = language {
            tail.extend(self.bpe.encode_plain(&format!("lang:{code}")));
        }
        tail.push(k.transcribe);
        let mut pieces = Vec::new();
        for window in audio::windows(samples, WINDOW_SAMPLES) {
            if audio::is_silent(window) {
                continue;
            }
            let encoded = self.encoder.encode(&self.mel.compute(window));
            // Each row of the projector's input is `join` encoder frames side by side.
            let n = encoded.len() / (self.join * self.d_encoder);
            let mut h = self.linear1.forward(&encoded, n);
            activation::gelu(&mut h);
            let audio = self.linear2.forward(&h, n);
            let d = self.decoder.width();
            let mut rows = Buf::scratch((head.len() + n + tail.len()) * d);
            rows[..head.len() * d].copy_from_slice(&self.decoder.embed(&head));
            rows[head.len() * d..(head.len() + n) * d].copy_from_slice(&audio);
            rows[(head.len() + n) * d..].copy_from_slice(&self.decoder.embed(&tail));
            let seconds = window.len() / crate::SAMPLE_RATE as usize;
            let max_new = MIN_NEW_TOKENS.max(seconds * TOKENS_PER_SECOND);
            let ids = self.decoder.generate_rows(rows, max_new, &[k.eos]);
            let text = self.bpe.decode(&ids).trim().to_string();
            if !text.is_empty() {
                pieces.push((audio::offset(samples, window), window.len(), text));
            }
        }
        pieces
    }
}
