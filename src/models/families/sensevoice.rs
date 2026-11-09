//! SenseVoice Small (FunAudioLLM): Kaldi filterbank features stacked to
//! 60 ms frames, four prompt rows before them (language, event, emotion,
//! text normalization), a SAN-M encoder of 1 + 49 + 20 blocks
//! (`parts::sanm`), and a CTC head over 25,055 SentencePiece tokens. The
//! model writes its tags (`<|en|>`, `<|NEUTRAL|>`, `<|withitn|>`) as special
//! tokens, which the text drops. The structure follows FunASR's
//! `SenseVoiceSmall` and transcribe.cpp's port; tensor names are FunASR's.

use crate::audio::kaldi_fbank::KaldiFbank;
use crate::config::Json;
use crate::cpu::{buf::Buf, gemm::Dense};
use crate::layers::linear;
use crate::models::files::Files;
use crate::models::parts::sanm;
use crate::models::{speech_windows, Family};
use crate::text::sentencepiece::Tokenizer;
use crate::{Device, Options, Precision, Segment};

/// The model's limit per pass (its card: 30 s per direct call).
const WINDOW_SAMPLES: usize = 30 * crate::SAMPLE_RATE as usize;

struct Config {
    vocab: usize,
    blank: usize,
    prompt_rows: usize,
    /// Language codes (`auto` too) and their prompt row.
    languages: Vec<(String, usize)>,
    event: usize,
    emotion: usize,
    /// Text normalization on (digits, punctuation), as transcribe.cpp's default.
    with_itn: usize,
}

impl Config {
    fn parse(json: &Json) -> Result<Self, String> {
        let prompt = |key: &str| json.num(&format!("prompt.{key}"));
        let languages = json
            .get("prompt.languages")
            .as_object()
            .ok_or("config.json has no prompt.languages")?
            .iter()
            .filter_map(|(code, row)| Some((code.clone(), row.as_u64()? as usize)))
            .collect();
        let config = Self {
            vocab: json.num("vocab_size")?,
            blank: json.num("blank_token_id")?,
            prompt_rows: prompt("rows")?,
            languages,
            event: prompt("event_speech")?,
            emotion: prompt("emotion_neutral")?,
            with_itn: prompt("with_itn")?,
        };
        let rows = [config.event, config.emotion, config.with_itn];
        if config.blank >= config.vocab
            || rows
                .iter()
                .chain(config.languages.iter().map(|(_, row)| row))
                .any(|row| *row >= config.prompt_rows)
        {
            return Err("config.json describes a SenseVoice shape this engine does not run".into());
        }
        Ok(config)
    }

    /// The prompt row of language `code`; none or an unknown code: `auto`.
    fn language(&self, code: Option<&str>) -> usize {
        let find = |code: &str| {
            self.languages
                .iter()
                .find(|(known, _)| known.eq_ignore_ascii_case(code.trim()))
                .map(|(_, row)| *row)
        };
        code.and_then(find).or_else(|| find("auto")).unwrap_or(0)
    }
}

pub struct SenseVoice {
    config: Config,
    frontend: KaldiFbank,
    /// The prompt rows, `prompt_rows x input`.
    prompts: Vec<f32>,
    encoder: sanm::Encoder,
    head: Dense,
    tokenizer: Tokenizer,
}

impl SenseVoice {
    /// CTC logits, `(4 + frames) x vocab`: the prompt rows, then the features.
    fn logits(&self, features: &[f32], frames: usize, language: usize) -> Buf {
        let c = &self.config;
        let (input, t) = (self.encoder.shape.input, frames + 4);
        let mut x = Buf::scratch(t * input);
        for (i, row) in [language, c.event, c.emotion, c.with_itn]
            .into_iter()
            .enumerate()
        {
            x[i * input..(i + 1) * input].copy_from_slice(&self.prompts[row * input..][..input]);
        }
        x[4 * input..].copy_from_slice(features);
        self.head.forward(&self.encoder.encode(x, t), t)
    }
}

impl Family for SenseVoice {
    fn load(
        files: &Files,
        json: &Json,
        precision: Precision,
        device: Device,
    ) -> Result<Self, String> {
        let config = Config::parse(json)?;
        let shape = sanm::Shape::parse(json)?;
        let t = files.tensors(precision, device)?;
        let frontend = sanm::frontend(json, &t, &shape)?;
        let (d, input) = (shape.d, shape.input);
        let model = Self {
            prompts: t.f32_len("embed.weight", config.prompt_rows * input)?,
            encoder: sanm::Encoder::load(&t, "encoder", shape)?,
            head: linear(&t, "ctc.ctc_lo", config.vocab, d)?,
            tokenizer: Tokenizer::parse(&files.read("tokenizer.json")?)?,
            frontend,
            config,
        };
        if model.tokenizer.len() < model.config.vocab {
            return Err("the tokenizer has fewer pieces than the CTC head".into());
        }
        Ok(model)
    }

    fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment> {
        let language = self.config.language(options.language.as_deref());
        speech_windows(samples, WINDOW_SAMPLES, options, |window| {
            let (features, frames) = self.frontend.compute(window);
            if frames == 0 {
                return String::new();
            }
            let logits = self.logits(&features, frames, language);
            let c = &self.config;
            let (ids, _) = crate::models::parts::ctc::greedy(&logits, c.vocab, c.blank, &[]);
            self.tokenizer.decode(&ids).trim().to_string()
        })
    }
}
