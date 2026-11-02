//! Fun-ASR Nano (FunAudioLLM, `Fun-ASR-Nano-2512` and its multilingual
//! sibling): Kaldi filterbank features stacked to 60 ms frames, FunASR's
//! SAN-M encoder (`parts::sanm`), a transformer adaptor to the language
//! model's width, and Qwen3. The adaptor rows fill the speech span of a
//! chat prompt ("语音转写：", "transcribe the speech:") and Qwen3 writes the
//! text. The prompt follows FunASR's `FunASRNano.get_prompt` as
//! transcribe.cpp's port renders it.

use crate::audio::{self, kaldi_fbank::KaldiFbank};
use crate::cpu::{buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::models::config::Json;
use crate::models::files::Files;
use crate::models::layers::{add_into, attention, linear, Norm};
use crate::models::parts::{causal_lm::CausalLm, sanm};
use crate::text::bpe::Bpe;
use crate::{Device, Options, Precision};

/// Longest window per pass.
const WINDOW_SAMPLES: usize = 30 * 16_000;
const MIN_NEW_TOKENS: usize = 256;
const TOKENS_PER_SECOND: usize = 12;
const NORM_EPS: f32 = 1e-12;
/// 语音转写 ("transcribe the speech"), then ，不进行文本规整 ("no text
/// normalization") when that is off, then a full-width colon.
const INSTRUCTION: &str = "语音转写";
const NO_ITN: &str = "，不进行文本规整";

struct AdaptorBlock {
    norm_attn: Norm,
    q: Dense,
    k: Dense,
    v: Dense,
    out: Dense,
    norm_ffn: Norm,
    fc1: Dense,
    fc2: Dense,
}

/// From the encoder width to the language model's: `linear1`, ReLU,
/// `linear2`, then pre-norm transformer blocks with ReLU feed-forwards.
struct Adaptor {
    linear1: Dense,
    linear2: Dense,
    blocks: Vec<AdaptorBlock>,
    d: usize,
    heads: usize,
    /// Fewer rows go to the language model than the adaptor makes.
    low_frame_rate: bool,
}

impl Adaptor {
    fn load(t: &Tensors, json: &Json) -> Result<Self, String> {
        let a = |key: &str| json.num(&format!("adaptor_config.{key}"));
        let (inp, d, pre, ffn) = (
            a("encoder_dim")?,
            a("llm_dim")?,
            a("pre_ffn_dim")?,
            a("ffn_dim")?,
        );
        let heads = a("num_attention_heads")?;
        if heads * a("head_dim")? != d {
            return Err("config.json: the adaptor's heads do not fill its width".into());
        }
        let blocks = (0..a("num_blocks")?)
            .map(|i| {
                let p = format!("audio_adaptor.blocks.{i}");
                let l = |name: &str, out: usize, inp: usize| {
                    linear(t, &format!("{p}.{name}"), out, inp)
                };
                Ok(AdaptorBlock {
                    norm_attn: Norm::load(t, &format!("{p}.norm1"), d)?.with_eps(NORM_EPS),
                    q: l("self_attn.linear_q", d, d)?,
                    k: l("self_attn.linear_k", d, d)?,
                    v: l("self_attn.linear_v", d, d)?,
                    out: l("self_attn.linear_out", d, d)?,
                    norm_ffn: Norm::load(t, &format!("{p}.norm2"), d)?.with_eps(NORM_EPS),
                    fc1: l("feed_forward.w_1", ffn, d)?,
                    fc2: l("feed_forward.w_2", d, ffn)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            linear1: linear(t, "audio_adaptor.linear1", pre, inp)?,
            linear2: linear(t, "audio_adaptor.linear2", d, pre)?,
            blocks,
            d,
            heads,
            low_frame_rate: json.flag("adaptor_config.low_frame_rate") == Some(true),
        })
    }

    /// The rows for the language model, `n x d`, from `t x encoder_dim`.
    fn forward(&self, x: &[f32], t: usize) -> (Buf, usize) {
        let d = self.d;
        let mut h = self.linear1.forward(x, t);
        relu(&mut h);
        let mut x = self.linear2.forward(&h, t);
        for b in &self.blocks {
            let n = b.norm_attn.apply(&x, d);
            let (q, k, v) = (b.q.forward(&n, t), b.k.forward(&n, t), b.v.forward(&n, t));
            let heard = attention(
                (&q, d),
                (&k, d),
                (&v, d),
                (t, t),
                (d, self.heads, self.heads),
                false,
            );
            add_into(&mut x, &b.out.forward(&heard, t));
            let mut h = b.fc1.forward(&b.norm_ffn.apply(&x, d), t);
            relu(&mut h);
            add_into(&mut x, &b.fc2.forward(&h, t));
        }
        // FunASR's `fake_token_len`: the length after three stride-2 convs
        // the adaptor does not run; the first rows go to the language model.
        let n = match self.low_frame_rate {
            true => {
                let o1 = 1 + (t + 2).saturating_sub(3) / 2;
                let o2 = 1 + (o1 + 2).saturating_sub(3) / 2;
                (o2 - 1) / 2 + 1
            }
            false => t,
        };
        let n = n.min(t);
        let mut rows = Buf::scratch(n * d);
        rows.copy_from_slice(&x[..n * d]);
        (rows, n)
    }
}

fn relu(x: &mut [f32]) {
    team::chunks_mut(x, 4096, |_, values| {
        values.iter_mut().for_each(|v| *v = v.max(0.0))
    });
}

pub struct FunAsr {
    frontend: KaldiFbank,
    encoder: sanm::Encoder,
    adaptor: Adaptor,
    decoder: CausalLm,
    bpe: Bpe,
    head: Vec<usize>,
    tail: Vec<usize>,
    stop: Vec<usize>,
}

impl FunAsr {
    pub fn load(
        files: &Files,
        json: &Json,
        precision: Precision,
        device: Device,
    ) -> Result<Self, String> {
        let shape = sanm::Shape::parse(json)?;
        let t = files.tensors(precision, device)?;
        let frontend = sanm::frontend(json, &t, &shape)?;
        let encoder = sanm::Encoder::load(&t, "audio_encoder", shape)?;
        let adaptor = Adaptor::load(&t, json)?;
        let decoder = CausalLm::from_tensors(&t, json, ("llm_config.", "llm."))?;
        drop(t);
        if decoder.width() != adaptor.d {
            return Err("config.json: the adaptor and the language model differ in width".into());
        }
        let bpe = Bpe::parse(&files.read("tokenizer.json")?)?;
        let id = |token: &str| {
            bpe.id(token)
                .ok_or_else(|| format!("the tokenizer has no {token}"))
        };
        // No language named: the model finds it, as FunASR without `language`.
        let mut instruction = INSTRUCTION.to_string();
        if json.flag("itn") != Some(true) {
            instruction.push_str(NO_ITN);
        }
        instruction.push('：');
        let head = bpe.encode(&format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}"
        ));
        let tail = bpe.encode("<|im_end|>\n<|im_start|>assistant\n");
        let stop = vec![id("<|im_end|>")?, id("<|endoftext|>")?];
        Ok(Self {
            frontend,
            encoder,
            adaptor,
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
            let (features, frames) = self.frontend.compute(window);
            if frames == 0 || audio::is_silent(window) {
                continue;
            }
            let mut x = Buf::scratch(features.len());
            x.copy_from_slice(&features);
            let encoded = self.encoder.encode(x, frames);
            let (audio, n) = self.adaptor.forward(&encoded, frames);
            let d = self.decoder.width();
            let (head, tail) = (self.head.len(), self.tail.len());
            let mut rows = Buf::scratch((head + n + tail) * d);
            rows[..head * d].copy_from_slice(&self.decoder.embed(&self.head));
            rows[head * d..(head + n) * d].copy_from_slice(&audio);
            rows[(head + n) * d..].copy_from_slice(&self.decoder.embed(&self.tail));
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
