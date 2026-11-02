//! SenseVoice Small (FunAudioLLM): Kaldi filterbank features stacked to
//! 60 ms frames, four prompt rows before them (language, event, emotion,
//! text normalization), a SAN-M encoder of 1 + 49 + 20 blocks
//! (self-attention plus an FSMN memory over the values), and a CTC head over
//! 25,055 SentencePiece tokens. The model writes its tags (`<|en|>`,
//! `<|NEUTRAL|>`, `<|withitn|>`) as special tokens, which the text drops.
//! The structure follows FunASR's `SenseVoiceSmall` and transcribe.cpp's
//! port; tensor names are FunASR's.

use crate::audio::{
    self,
    kaldi_fbank::{FbankShape, KaldiFbank},
};
use crate::cpu::{buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::models::config::Json;
use crate::models::files::Files;
use crate::models::layers::{add_into, attention, linear, Norm};
use crate::text::sentencepiece::Tokenizer;
use crate::{Device, Options, Precision};

/// The model's limit per pass (its card: 30 s per direct call).
const WINDOW_SAMPLES: usize = 30 * 16_000;
/// FunASR's layer norm epsilon.
const NORM_EPS: f32 = 1e-12;

struct Config {
    d: usize,
    input: usize,
    heads: usize,
    ffn: usize,
    kernel: usize,
    blocks: usize,
    tp_blocks: usize,
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
        let enc = |key: &str| json.num(&format!("encoder_config.{key}"));
        let prompt = |key: &str| json.num(&format!("prompt.{key}"));
        let languages = json
            .get("prompt.languages")
            .as_object()
            .ok_or("config.json has no prompt.languages")?
            .iter()
            .filter_map(|(code, row)| Some((code.clone(), row.as_u64()? as usize)))
            .collect();
        let config = Self {
            d: enc("hidden_size")?,
            input: enc("input_size")?,
            heads: enc("num_attention_heads")?,
            ffn: enc("intermediate_size")?,
            kernel: enc("kernel_size")?,
            blocks: enc("num_blocks")?,
            tp_blocks: enc("tp_blocks")?,
            vocab: json.num("vocab_size")?,
            blank: json.num("blank_token_id")?,
            prompt_rows: prompt("rows")?,
            languages,
            event: prompt("event_speech")?,
            emotion: prompt("emotion_neutral")?,
            with_itn: prompt("with_itn")?,
        };
        let rows = [config.event, config.emotion, config.with_itn];
        if !config.d.is_multiple_of(config.heads)
            || config.kernel.is_multiple_of(2)
            || config.blocks == 0
            || config.blank >= config.vocab
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

/// One SAN-M block: pre-norm self-attention with an FSMN memory, then a
/// pre-norm ReLU feed-forward.
struct Block {
    /// Width of the input: the stacked features for the first block.
    input: usize,
    norm_attn: Norm,
    qkv: Dense,
    out: Dense,
    /// The FSMN's depthwise conv over time, `[tap][channel]`.
    fsmn: Vec<f32>,
    norm_ffn: Norm,
    fc1: Dense,
    fc2: Dense,
}

impl Block {
    fn load(t: &Tensors, name: &str, input: usize, c: &Config) -> Result<Self, String> {
        let (d, k) = (c.d, c.kernel);
        let fsmn = t.f32_len(&format!("{name}.self_attn.fsmn_block.weight"), d * k)?;
        Ok(Self {
            input,
            norm_attn: Norm::load(t, &format!("{name}.norm1"), input)?.with_eps(NORM_EPS),
            qkv: linear(t, &format!("{name}.self_attn.linear_q_k_v"), 3 * d, input)?,
            out: linear(t, &format!("{name}.self_attn.linear_out"), d, d)?,
            fsmn: (0..k * d).map(|i| fsmn[(i % d) * k + i / d]).collect(),
            norm_ffn: Norm::load(t, &format!("{name}.norm2"), d)?.with_eps(NORM_EPS),
            fc1: linear(t, &format!("{name}.feed_forward.w_1"), c.ffn, d)?,
            fc2: linear(t, &format!("{name}.feed_forward.w_2"), d, c.ffn)?,
        })
    }

    /// `x` (`t x input`) through the block. The first block's input is wider
    /// than its output, so its attention has no residual.
    fn forward(&self, x: &mut Buf, t: usize, c: &Config) {
        let attended = self.attend(&self.norm_attn.apply(x, self.input), t, c);
        if self.input == c.d {
            add_into(x, &attended);
        } else {
            *x = attended;
        }
        let mut hidden = self.fc1.forward(&self.norm_ffn.apply(x, c.d), t);
        team::chunks_mut(&mut hidden, 4096, |_, values| {
            values.iter_mut().for_each(|v| *v = v.max(0.0))
        });
        add_into(x, &self.fc2.forward(&hidden, t));
    }

    /// Softmax attention, projected, plus the values and their FSMN memory
    /// (a centered depthwise conv over time).
    fn attend(&self, x: &[f32], t: usize, c: &Config) -> Buf {
        let (d, k) = (c.d, c.kernel);
        let qkv = self.qkv.forward(x, t);
        let heard = attention(
            (&qkv, 3 * d),
            (&qkv[d..], 3 * d),
            (&qkv[2 * d..], 3 * d),
            (t, t),
            (d, c.heads, c.heads),
            false,
        );
        let mut out = self.out.forward(&heard, t);
        let pad = k / 2;
        let value = |j: usize| &qkv[j * 3 * d + 2 * d..][..d];
        team::chunks_mut(&mut out, d, |i, row| {
            add_into(row, value(i));
            for tap in 0..k {
                let Some(src) = (i + tap).checked_sub(pad).filter(|src| *src < t) else {
                    continue;
                };
                let weights = &self.fsmn[tap * d..][..d];
                for ((o, w), v) in row.iter_mut().zip(weights).zip(value(src)) {
                    *o += w * v;
                }
            }
        });
        out
    }
}

pub struct SenseVoice {
    config: Config,
    frontend: KaldiFbank,
    /// The prompt rows, `prompt_rows x input`.
    prompts: Vec<f32>,
    /// Every block in order: the projecting one, the main stack, and the
    /// `tp` stack after `after_norm`.
    blocks: Vec<Block>,
    after_norm: Norm,
    tp_norm: Norm,
    head: Dense,
    tokenizer: Tokenizer,
}

impl SenseVoice {
    pub fn load(
        files: &Files,
        json: &Json,
        precision: Precision,
        device: Device,
    ) -> Result<Self, String> {
        let config = Config::parse(json)?;
        let t = files.tensors(precision, device)?;
        let c = &config;
        let fe = |key: &str| json.num(&format!("feature_extractor.{key}"));
        if json.str("feature_extractor.type") != Some("kaldi_fbank") {
            return Err("config.json: SenseVoice reads Kaldi filterbank features".into());
        }
        let shape = FbankShape {
            sample_rate: fe("sample_rate")?,
            win: fe("win_length")?,
            hop: fe("hop_length")?,
            mels: fe("num_mel_bins")?,
            lfr_m: fe("lfr_m")?,
            lfr_n: fe("lfr_n")?,
            upscale: json.flag("feature_extractor.upscale_samples") != Some(false),
        };
        if shape.lfr_m * shape.mels != c.input || shape.sample_rate != crate::SAMPLE_RATE as usize {
            return Err("config.json: the stacked features do not fit the encoder".into());
        }
        let frontend = KaldiFbank::new(
            shape,
            t.f32_len("frontend.cmvn.shift", c.input)?,
            t.f32_len("frontend.cmvn.scale", c.input)?,
        )?;
        let names = std::iter::once(("encoders0".to_string(), 0, c.input))
            .chain((0..c.blocks - 1).map(|i| ("encoders".to_string(), i, c.d)))
            .chain((0..c.tp_blocks).map(|i| ("tp_encoders".to_string(), i, c.d)))
            .collect::<Vec<_>>();
        let blocks = team::map(names.len(), |i| {
            let (stack, index, input) = &names[i];
            Block::load(&t, &format!("encoder.{stack}.{index}"), *input, c)
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        let model = Self {
            prompts: t.f32_len("embed.weight", c.prompt_rows * c.input)?,
            blocks,
            after_norm: Norm::load(&t, "encoder.after_norm", c.d)?.with_eps(NORM_EPS),
            tp_norm: Norm::load(&t, "encoder.tp_norm", c.d)?.with_eps(NORM_EPS),
            head: linear(&t, "ctc.ctc_lo", c.vocab, c.d)?,
            tokenizer: Tokenizer::parse(&files.read("tokenizer.json")?)?,
            frontend,
            config,
        };
        if model.tokenizer.len() < model.config.vocab {
            return Err("the tokenizer has fewer pieces than the CTC head".into());
        }
        Ok(model)
    }

    /// The text of each window that holds speech: `(first sample, samples, text)`.
    pub fn pieces(&self, samples: &[f32], options: &Options) -> Vec<(usize, usize, String)> {
        let language = self.config.language(options.language.as_deref());
        let mut pieces = Vec::new();
        for window in audio::windows(samples, WINDOW_SAMPLES) {
            let (features, frames) = self.frontend.compute(window);
            if frames == 0 || audio::is_silent(window) {
                continue;
            }
            let logits = self.logits(&features, frames, language);
            let c = &self.config;
            let (ids, _) = crate::models::parts::ctc::greedy(&logits, c.vocab, c.blank, &[]);
            let text = self.tokenizer.decode(&ids).trim().to_string();
            if !text.is_empty() {
                pieces.push((audio::offset(samples, window), window.len(), text));
            }
        }
        pieces
    }

    /// CTC logits, `(4 + frames) x vocab`: the prompt rows, then the features.
    fn logits(&self, features: &[f32], frames: usize, language: usize) -> Buf {
        let c = &self.config;
        let (input, t) = (c.input, frames + 4);
        let mut x = Buf::scratch(t * input);
        for (i, row) in [language, c.event, c.emotion, c.with_itn]
            .into_iter()
            .enumerate()
        {
            x[i * input..(i + 1) * input].copy_from_slice(&self.prompts[row * input..][..input]);
        }
        x[4 * input..].copy_from_slice(features);
        // Scaled by sqrt of the model width (not the input's), plus sinusoids.
        let scale = (c.d as f32).sqrt();
        let positions = sinusoids(t, input);
        team::chunks_mut(&mut x, input, |i, row| {
            for (v, p) in row.iter_mut().zip(&positions[i * input..][..input]) {
                *v = *v * scale + p;
            }
        });
        let (main, tp) = self.blocks.split_at(c.blocks);
        for block in main {
            block.forward(&mut x, t, c);
        }
        let mut x = self.after_norm.apply(&x, c.d);
        for block in tp {
            block.forward(&mut x, t, c);
        }
        self.head.forward(&self.tp_norm.apply(&x, c.d), t)
    }
}

/// FunASR's `SinusoidalPositionEncoder`: positions from 1, sines then
/// cosines over `depth / 2` timescales spaced `ln(10000) / (depth / 2 - 1)`.
fn sinusoids(t: usize, depth: usize) -> Vec<f32> {
    let half = depth / 2;
    let step = (10_000f64).ln() / (half - 1) as f64;
    let mut out = vec![0.0f32; t * depth];
    for (i, row) in out.chunks_exact_mut(depth).enumerate() {
        let position = (i + 1) as f64;
        for k in 0..half {
            let angle = position * (-(k as f64) * step).exp();
            row[k] = angle.sin() as f32;
            row[half + k] = angle.cos() as f32;
        }
    }
    out
}
