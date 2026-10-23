//! Qwen3-ASR (0.6B and 1.7B): an audio encoder and the Qwen3 language
//! model. The encoder reads Whisper log-mel features in chunks of one
//! second through three stride-2 2-D convolutions (13 frames per second),
//! then a pre-norm transformer that attends within windows of eight
//! chunks. Its frames take the place of the `<|audio_pad|>` tokens of a
//! chat prompt, and Qwen3 writes `language <name><asr_text><text>`. The
//! structure follows the Apache-2.0 `qwen-asr` package
//! (`modeling_qwen3_asr.py`); tensor names are its `thinker.*` names.

use crate::audio::{self, whisper_mel::WhisperMel};
use crate::cpu::{buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::models::config::Json;
use crate::models::layers::{add_into, attention, linear, stacked, Gelu, Norm};
use crate::models::parts::qwen3::Qwen3;
use crate::text::bpe::Bpe;
use crate::{Options, Precision};
use parking_lot::RwLock;
use std::path::Path;

/// Longest window per pass. The reference takes up to 20 minutes at once;
/// two minutes keep the decoder's caches near 100 MB.
const WINDOW_SAMPLES: usize = 120 * 16_000;
/// The reference's `max_new_tokens`; longer windows get more.
const MIN_NEW_TOKENS: usize = 512;
const TOKENS_PER_SECOND: usize = 8;
/// What the model writes between the language and the transcript.
const TEXT_TAG: &str = "<asr_text>";
/// Where the config of each part is.
const AUDIO: &str = "thinker_config.audio_config.";
const TEXT: &str = "thinker_config.text_config.";

/// ISO 639-1 (and 639-2 for Cantonese and Filipino) codes of the languages
/// Qwen3-ASR names in its prompt.
const LANGUAGE_NAMES: [(&str, &str); 31] = [
    ("zh", "Chinese"),
    ("en", "English"),
    ("yue", "Cantonese"),
    ("ar", "Arabic"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ko", "Korean"),
    ("ru", "Russian"),
    ("th", "Thai"),
    ("vi", "Vietnamese"),
    ("ja", "Japanese"),
    ("tr", "Turkish"),
    ("hi", "Hindi"),
    ("ms", "Malay"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("pl", "Polish"),
    ("cs", "Czech"),
    ("fil", "Filipino"),
    ("tl", "Filipino"),
    ("fa", "Persian"),
    ("el", "Greek"),
    ("ro", "Romanian"),
    ("hu", "Hungarian"),
    ("mk", "Macedonian"),
];

struct AudioConfig {
    d: usize,
    heads: usize,
    ffn: usize,
    mels: usize,
    /// Channels of the three convolutions.
    channels: usize,
    /// Mel frames per chunk (`2 * n_window`).
    chunk: usize,
    /// Chunks per attention window (`n_window_infer / chunk`).
    window_chunks: usize,
    layers: usize,
    /// Output width: the language model's.
    out: usize,
    max_positions: usize,
}

impl AudioConfig {
    fn parse(json: &Json) -> Result<Self, String> {
        let num = |key: &str| json.num(&format!("{AUDIO}{key}"));
        let chunk = 2 * num("n_window")?;
        let config = Self {
            d: num("d_model")?,
            heads: num("encoder_attention_heads")?,
            ffn: num("encoder_ffn_dim")?,
            mels: num("num_mel_bins")?,
            channels: num("downsample_hidden_size")?,
            chunk,
            window_chunks: num("n_window_infer")? / chunk,
            layers: num("encoder_layers")?,
            out: num("output_dim")?,
            max_positions: num("max_source_positions")?,
        };
        let activation = json.str(&format!("{AUDIO}activation_function"));
        if !config.d.is_multiple_of(config.heads)
            || config.window_chunks == 0
            || activation.unwrap_or("gelu") != "gelu"
        {
            return Err(
                "config.json describes a Qwen3-ASR encoder this engine does not run".into(),
            );
        }
        Ok(config)
    }

    /// Mel bins after the three convolutions.
    fn bins(&self) -> usize {
        half(half(half(self.mels)))
    }
}

/// Length after one stride-2 convolution with padding 1 and kernel 3.
fn half(n: usize) -> usize {
    n.div_ceil(2)
}

struct EncoderLayer {
    norm_attn: Norm,
    qkv: Dense,
    out: Dense,
    norm_ffn: Norm,
    fc1: Dense,
    fc2: Dense,
}

struct AudioEncoder {
    config: AudioConfig,
    /// Three 3x3 convolutions, stride 2, padding 1; weights `[out][in][3][3]`.
    convs: [Dense; 3],
    /// Channels and bins after the convolutions to the model width.
    conv_out: Dense,
    /// Sinusoids, `max_positions x d`; each chunk starts at position 0.
    positions: Vec<f32>,
    layers: Vec<EncoderLayer>,
    norm: Norm,
    proj1: Dense,
    proj2: Dense,
}

pub struct Qwen3Asr {
    encoder: AudioEncoder,
    decoder: Qwen3,
    mel: WhisperMel,
    bpe: Bpe,
    /// `<|audio_pad|>`, replaced by the audio frames.
    audio_pad: usize,
    stop: Vec<usize>,
    /// The languages of `support_languages`, as the prompt names them.
    languages: Vec<String>,
    /// The dictionary as the system prompt's context, as the reference's
    /// `context` argument.
    context: RwLock<String>,
}

impl Qwen3Asr {
    pub fn load(dir: &Path, json: &Json, precision: Precision) -> Result<Self, String> {
        let t = Tensors::open_dir(dir, precision)?;
        let encoder = AudioEncoder::load(&t, AudioConfig::parse(json)?)?;
        let decoder = Qwen3::from_tensors(&t, json, (TEXT, "thinker."))?;
        drop(t);
        if decoder.width() != encoder.config.out {
            return Err(
                "config.json: the audio encoder and the language model differ in width".into(),
            );
        }
        let bpe = Bpe::open_dir(dir)?;
        let id = |token: &str| {
            bpe.id(token)
                .ok_or_else(|| format!("the tokenizer has no {token}"))
        };
        let audio_pad = id("<|audio_pad|>")?;
        let stop = vec![id("<|im_end|>")?, id("<|endoftext|>")?];
        let languages = json
            .get("support_languages")
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|name| name.as_str().map(str::to_string))
            .collect();
        Ok(Self {
            mel: WhisperMel::new(encoder.config.mels),
            encoder,
            decoder,
            bpe,
            audio_pad,
            stop,
            languages,
            context: RwLock::new(String::new()),
        })
    }

    /// Terms to favor, as the context of the system prompt: Qwen3-ASR is
    /// trained to spell words as its context does. No term is skipped.
    pub fn set_dictionary(&self, terms: &[String]) -> Vec<String> {
        let terms: Vec<&str> = terms
            .iter()
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .collect();
        *self.context.write() = terms.join(", ");
        Vec::new()
    }

    /// The text of each window that holds speech: `(first sample, samples, text)`.
    /// `options.language` (one of 30) makes the model write that language;
    /// without one, it detects the language.
    pub fn pieces(&self, samples: &[f32], options: &Options) -> Vec<(usize, usize, String)> {
        let language = options.language.as_deref().and_then(|code| {
            LANGUAGE_NAMES
                .iter()
                .find(|(c, _)| c.eq_ignore_ascii_case(code.trim()))
                .map(|(_, name)| *name)
                .filter(|name| self.languages.iter().any(|l| l == name))
        });
        let context = self.context.read().clone();
        audio::windows(samples, WINDOW_SAMPLES)
            .into_iter()
            .filter(|window| !audio::is_silent(window))
            .filter_map(|window| {
                let text = self.transcribe_window(window, &context, language);
                (!text.is_empty()).then(|| (audio::offset(samples, window), window.len(), text))
            })
            .collect()
    }

    fn transcribe_window(&self, samples: &[f32], context: &str, language: Option<&str>) -> String {
        let frames = samples.len() / crate::audio::whisper_mel::HOP;
        if frames == 0 {
            return String::new();
        }
        let mel = self.mel.compute_frames(samples, frames);
        let (audio, count) = self.encoder.encode(&mel, frames);
        let head =
            format!("<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>");
        let mut tail = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n".to_string();
        if let Some(name) = language {
            tail.push_str(&format!("language {name}{TEXT_TAG}"));
        }
        let (head, tail) = (self.bpe.encode(&head), self.bpe.encode(&tail));
        let d = self.decoder.width();
        let mut rows = Buf::scratch((head.len() + count + tail.len()) * d);
        rows[..head.len() * d].copy_from_slice(&self.decoder.embed(&head));
        rows[head.len() * d..(head.len() + count) * d].copy_from_slice(&audio);
        rows[(head.len() + count) * d..].copy_from_slice(&self.decoder.embed(&tail));
        debug_assert!(!head.contains(&self.audio_pad));
        let seconds = samples.len() / crate::SAMPLE_RATE as usize;
        let max_new = MIN_NEW_TOKENS.max(seconds * TOKENS_PER_SECOND);
        let ids = self.decoder.generate_rows(rows, max_new, &self.stop);
        let raw = self.bpe.decode(&ids);
        transcript(&raw, language.is_some())
    }
}

/// The text of the model's output: after `<asr_text>` when it named the
/// language itself (`language None` and no text when it hears no speech).
fn transcript(raw: &str, forced: bool) -> String {
    let raw = raw.trim();
    match raw.split_once(TEXT_TAG) {
        Some((_, text)) if !forced => text.trim().to_string(),
        _ => raw.to_string(),
    }
}

impl AudioEncoder {
    fn load(t: &Tensors, config: AudioConfig) -> Result<Self, String> {
        let c = &config;
        let p = "thinker.audio_tower";
        let (d, ch) = (c.d, c.channels);
        let layers = team::map(c.layers, |i| -> Result<EncoderLayer, String> {
            let l = format!("{p}.layers.{i}");
            let a = |n: &str| format!("{l}.self_attn.{n}");
            Ok(EncoderLayer {
                norm_attn: Norm::load(t, &format!("{l}.self_attn_layer_norm"), d)?,
                qkv: stacked(t, &[&a("q_proj"), &a("k_proj"), &a("v_proj")], d, d)?,
                out: linear(t, &a("out_proj"), d, d)?,
                norm_ffn: Norm::load(t, &format!("{l}.final_layer_norm"), d)?,
                fc1: linear(t, &format!("{l}.fc1"), c.ffn, d)?,
                fc2: linear(t, &format!("{l}.fc2"), d, c.ffn)?,
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            convs: [
                linear(t, &format!("{p}.conv2d1"), ch, 9)?,
                linear(t, &format!("{p}.conv2d2"), ch, ch * 9)?,
                linear(t, &format!("{p}.conv2d3"), ch, ch * 9)?,
            ],
            conv_out: linear(t, &format!("{p}.conv_out"), d, ch * c.bins())?,
            positions: sinusoids(c.max_positions, d),
            layers,
            norm: Norm::load(t, &format!("{p}.ln_post"), d)?,
            proj1: linear(t, &format!("{p}.proj1"), d, d)?,
            proj2: linear(t, &format!("{p}.proj2"), c.out, d)?,
            config,
        })
    }

    /// Frames for the language model (`count x out`) from `frames x mels`
    /// log-mel features.
    fn encode(&self, mel: &[f32], frames: usize) -> (Buf, usize) {
        let c = &self.config;
        let d = c.d;
        // Every chunk is padded with zeros to the longest one, as the
        // reference batches them; the frames of the padding are dropped.
        let chunks: Vec<(usize, usize)> = (0..frames)
            .step_by(c.chunk)
            .map(|start| (start, c.chunk.min(frames - start)))
            .collect();
        let longest = chunks.iter().map(|(_, n)| *n).max().unwrap_or(0);
        let padded = half(half(half(longest)));
        let mut x = Vec::new();
        for &(start, length) in &chunks {
            let embedded = self.chunk(&mel[start * c.mels..(start + length) * c.mels], longest);
            let keep = half(half(half(length)));
            x.extend_from_slice(&embedded[..keep * d]);
        }
        let t = x.len() / d;
        let mut x = {
            let mut buf = Buf::scratch(x.len());
            buf.copy_from_slice(&x);
            buf
        };
        let window = padded * c.window_chunks;
        for layer in &self.layers {
            let normed = layer.norm_attn.apply(&x, d);
            let qkv = layer.qkv.forward(&normed, t);
            let mut att = Buf::scratch(t * d);
            for start in (0..t).step_by(window) {
                let n = window.min(t - start);
                let rows = &qkv[start * 3 * d..];
                let part = attention(
                    (rows, 3 * d),
                    (&rows[d..], 3 * d),
                    (&rows[2 * d..], 3 * d),
                    (n, n),
                    (d, c.heads, c.heads),
                    false,
                );
                att[start * d..(start + n) * d].copy_from_slice(&part);
            }
            add_into(&mut x, &layer.out.forward(&att, t));
            let normed = layer.norm_ffn.apply(&x, d);
            let mut hidden = layer.fc1.forward(&normed, t);
            Gelu::Exact.apply(&mut hidden, c.ffn);
            add_into(&mut x, &layer.fc2.forward(&hidden, t));
        }
        let normed = self.norm.apply(&x, d);
        let mut hidden = self.proj1.forward(&normed, t);
        Gelu::Exact.apply(&mut hidden, d);
        (self.proj2.forward(&hidden, t), t)
    }

    /// One chunk (`length x mels`, zero padded to `frames`) through the
    /// convolutions, `conv_out`, and the positions: `half^3(frames) x d`.
    fn chunk(&self, mel: &[f32], frames: usize) -> Buf {
        let c = &self.config;
        // Maps are `[bin][time][channel]`; the first has one channel.
        let mut map = Buf::zeroed(c.mels * frames);
        let length = mel.len() / c.mels;
        team::chunks_mut(&mut map, frames, |bin, row| {
            for (time, value) in row[..length].iter_mut().enumerate() {
                *value = mel[time * c.mels + bin];
            }
        });
        let (mut bins, mut times, mut channels) = (c.mels, frames, 1);
        for conv in &self.convs {
            map = conv2d(&map, (bins, times, channels), conv);
            (bins, times, channels) = (half(bins), half(times), conv.out_features);
            Gelu::Exact.apply(&mut map, channels);
        }
        // Per time step: channel-major, then bin (`permute(0, 3, 1, 2)`).
        let mut columns = Buf::scratch(times * channels * bins);
        team::chunks_mut(&mut columns, channels * bins, |time, row| {
            for bin in 0..bins {
                let from = &map[(bin * times + time) * channels..][..channels];
                for (channel, value) in from.iter().enumerate() {
                    row[channel * bins + bin] = *value;
                }
            }
        });
        let mut x = self.conv_out.forward(&columns, times);
        add_into(&mut x, &self.positions[..times * c.d]);
        x
    }
}

/// A 3x3 convolution, stride 2, padding 1, over a `[bin][time][channel]`
/// map, as im2col and one product: `[bin'][time'][out]`.
fn conv2d(map: &[f32], (bins, times, channels): (usize, usize, usize), conv: &Dense) -> Buf {
    let (out_bins, out_times) = (half(bins), half(times));
    let width = channels * 9;
    let mut columns = Buf::scratch(out_bins * out_times * width);
    team::chunks_mut(&mut columns, width, |position, row| {
        let (bin, time) = (position / out_times, position % out_times);
        for tap in 0..9 {
            let (dy, dx) = (tap / 3, tap % 3);
            let source = (2 * bin + dy)
                .checked_sub(1)
                .zip((2 * time + dx).checked_sub(1))
                .filter(|(y, x)| *y < bins && *x < times);
            for channel in 0..channels {
                row[channel * 9 + tap] =
                    source.map_or(0.0, |(y, x)| map[(y * times + x) * channels + channel]);
            }
        }
    });
    conv.forward(&columns, out_bins * out_times)
}

/// Whisper's sinusoids: `[sin | cos]` of position times
/// `exp(-i * ln(10000) / (d/2 - 1))`.
fn sinusoids(length: usize, d: usize) -> Vec<f32> {
    let half = d / 2;
    let increment = 10_000f64.ln() / (half as f64 - 1.0);
    let mut table = vec![0.0f32; length * d];
    for position in 0..length {
        for i in 0..half {
            let angle = position as f64 * (-increment * i as f64).exp();
            table[position * d + i] = angle.sin() as f32;
            table[position * d + half + i] = angle.cos() as f32;
        }
    }
    table
}
