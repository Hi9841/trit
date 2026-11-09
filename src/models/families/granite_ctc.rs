//! Granite Speech 5.0 TurboCTC (IBM): the Granite Conformer
//! (`parts::granite_conformer`: features with deltas, two blocks that halve
//! the time, self-conditioning after block 8) and its CTC head over 16,384
//! BPE tokens. The structure follows transcribe.cpp's port of IBM's
//! `GraniteSpeechCTCEncoder`.

use crate::config::Json;
use crate::models::files::Files;
use crate::models::parts::granite_conformer::{Config, Encoder, Frontend};
use crate::models::{speech_windows, Family};
use crate::text::{bpe::Bpe, sentencepiece::Tokenizer};
use crate::{Device, Options, Precision, Segment};

/// Longest window per pass. Attention is local, so this only bounds memory.
const WINDOW_SAMPLES: usize = 60 * crate::SAMPLE_RATE as usize;

pub struct GraniteCtc {
    frontend: Frontend,
    encoder: Encoder,
    blank: usize,
    text: Text,
}

/// The tokenizer: byte-level BPE (the Apache release) or SentencePiece BPE
/// (the NC release), as `tokenizer.json` names its decoder.
enum Text {
    Bytes(Bpe),
    Pieces(Tokenizer),
}

impl Text {
    fn parse(json: &str) -> Result<Self, String> {
        match json.contains("\"ByteLevel\"") {
            true => Bpe::parse(json).map(Self::Bytes),
            false => Tokenizer::parse(json).map(Self::Pieces),
        }
    }

    fn decode(&self, ids: &[usize]) -> String {
        match self {
            Self::Bytes(bpe) => bpe.decode(ids),
            Self::Pieces(tokenizer) => tokenizer.decode(ids),
        }
    }
}

impl Family for GraniteCtc {
    fn load(
        files: &Files,
        json: &Json,
        precision: Precision,
        device: Device,
    ) -> Result<Self, String> {
        let config = Config::parse(json)?;
        let t = files.tensors(precision, device)?;
        let frontend = Frontend::load(json, &t)?;
        let blank = json.num("blank_token_id")?;
        if blank >= config.ctc_dim {
            return Err("config.json: the blank is past the CTC head".into());
        }
        Ok(Self {
            encoder: Encoder::load(&t, config)?,
            frontend,
            blank,
            text: Text::parse(&files.read("tokenizer.json")?)?,
        })
    }

    fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment> {
        speech_windows(samples, WINDOW_SAMPLES, options, |window| {
            let (features, frames) = self.frontend.compute(window);
            // Each subsampling block needs a frame to halve.
            if frames >> self.encoder.config.subsample.len() == 0 {
                return String::new();
            }
            let (x, t) = self.encoder.encode(&features, frames);
            let vocab = self.encoder.config.ctc_dim;
            let logits = self.encoder.ctc_logits(&x, t);
            let (ids, _) = crate::models::parts::ctc::greedy(&logits, vocab, self.blank, &[]);
            self.text.decode(&ids).trim().to_string()
        })
    }
}
