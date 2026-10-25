//! Kokoro-82M (hexgrad, StyleTTS 2 with an iSTFTNet vocoder): phonemes to
//! 24 kHz speech, as the `kokoro` package's `KModel` runs it. ALBERT reads
//! the phonemes; a prosody predictor gives each phoneme its frames, the
//! pitch (F0) and the energy of each frame; a text encoder and a decoder
//! turn them into the inputs of the generator, which writes the waveform
//! with a harmonic source and an inverse STFT. A voice is a style vector
//! per phoneme count (`voices/<name>.pt`).

mod albert;
mod blocks;
mod generator;

use crate::cpu::{buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::models::config::Json;
use crate::models::layers::{linear, Norm};
use crate::models::parts::conv::Conv1d;
use crate::models::parts::lstm::BiLstm;
use crate::tts::vits::Noise;
use crate::Precision;
use albert::Albert;
use blocks::{AdaInResBlock, AdaLayerNorm};
use generator::Decoder;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The style vectors of a voice: one row of `2 * style` per phoneme count.
const PACK_ROWS: usize = 510;

pub struct Kokoro {
    vocab: HashMap<char, usize>,
    albert: Albert,
    bert_encoder: Dense,
    durations: DurationEncoder,
    lstm: BiLstm,
    duration_proj: Dense,
    shared: BiLstm,
    f0: Vec<AdaInResBlock>,
    n: Vec<AdaInResBlock>,
    f0_proj: Conv1d,
    n_proj: Conv1d,
    text: TextEncoder,
    decoder: Decoder,
    style: usize,
    hidden: usize,
    /// The phonemes of one pass, at most (the context of ALBERT less the two pads).
    pub max_phonemes: usize,
    voices_dir: PathBuf,
    packs: Mutex<HashMap<String, Arc<Vec<f32>>>>,
    pub sample_rate: u32,
}

/// True when `dir` holds a Kokoro checkpoint: a config with `istftnet` and a `.pth` file.
pub fn is_kokoro(dir: &Path) -> bool {
    checkpoint(dir).is_some()
        && std::fs::read_to_string(dir.join("config.json"))
            .is_ok_and(|c| c.contains("\"istftnet\""))
}

fn checkpoint(dir: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "pth"))
        .collect();
    found.sort();
    found.into_iter().next()
}

impl Kokoro {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("cannot read {}: {e}", dir.join("config.json").display()))?;
        let c = Json::parse(&text)?;
        let weights =
            checkpoint(dir).ok_or_else(|| format!("{} has no .pth checkpoint", dir.display()))?;
        let t = Tensors::open_torch(&[(&weights, "")], Precision::F32)?;
        let vocab_size = c.num("n_token")?;
        let hidden = c.num("hidden_dim")?;
        let style = c.num("style_dim")?;
        let layers = c.num("n_layer")?;
        let max_dur = c.num("max_dur")?;
        let vocab = c
            .get("vocab")
            .as_object()
            .ok_or("config.json has no vocab")?
            .iter()
            .filter_map(|(k, v)| {
                let mut chars = k.chars();
                match (chars.next(), chars.next(), v.as_u64()) {
                    (Some(ch), None, Some(id)) => Some((ch, id as usize)),
                    _ => None,
                }
            })
            .collect();
        let albert = Albert::load(&t, &c, vocab_size)?;
        let block = |name: &str, (i, o): (usize, usize), up: bool| {
            AdaInResBlock::load(&t, name, (i, o, style), up)
        };
        let half = hidden / 2;
        Ok(Self {
            vocab,
            bert_encoder: linear(&t, "bert_encoder", hidden, albert.hidden)?,
            max_phonemes: c.num("plbert.max_position_embeddings")? - 2,
            albert,
            durations: DurationEncoder::load(&t, (hidden, style, layers))?,
            lstm: BiLstm::load(&t, "predictor.lstm", hidden + style, half)?,
            duration_proj: linear(&t, "predictor.duration_proj.linear_layer", max_dur, hidden)?,
            shared: BiLstm::load(&t, "predictor.shared", hidden + style, half)?,
            f0: vec![
                block("predictor.F0.0", (hidden, hidden), false)?,
                block("predictor.F0.1", (hidden, half), true)?,
                block("predictor.F0.2", (half, half), false)?,
            ],
            n: vec![
                block("predictor.N.0", (hidden, hidden), false)?,
                block("predictor.N.1", (hidden, half), true)?,
                block("predictor.N.2", (half, half), false)?,
            ],
            f0_proj: Conv1d::load(&t, "predictor.F0_proj", (half, 1), (1, 1))?,
            n_proj: Conv1d::load(&t, "predictor.N_proj", (half, 1), (1, 1))?,
            text: TextEncoder::load(&t, &c, (vocab_size, hidden, layers))?,
            decoder: Decoder::load(&t, &c, (hidden, style))?,
            style,
            hidden,
            voices_dir: dir.join("voices"),
            packs: Mutex::new(HashMap::new()),
            sample_rate: 24_000,
        })
    }

    /// The voices of the folder (`voices/<name>.pt`), sorted.
    pub fn voices(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.voices_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let path = e.path();
                if path.extension()? != "pt" {
                    return None;
                }
                Some(path.file_stem()?.to_string_lossy().into_owned())
            })
            .collect();
        names.sort();
        names
    }

    /// The style rows of `voice`: one voice (`af_heart`), or the mean of
    /// several (`af_bella,af_sky`), as the `kokoro` package blends them.
    pub fn pack(&self, voice: &str) -> Result<Arc<Vec<f32>>, String> {
        if let Some(pack) = self.packs.lock().get(voice) {
            return Ok(pack.clone());
        }
        let width = 2 * self.style;
        let names: Vec<&str> = voice
            .split(',')
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .collect();
        if names.is_empty() {
            return Err("no voice named".into());
        }
        let mut sum = vec![0.0f32; PACK_ROWS * width];
        for name in &names {
            let path = self.voices_dir.join(format!("{name}.pt"));
            if !path.is_file() {
                return Err(format!(
                    "voice {name} is not in {}; voices: {}",
                    self.voices_dir.display(),
                    self.voices().join(", ")
                ));
            }
            let t = Tensors::open_torch(&[(&path, "")], Precision::F32)?;
            let values = t.f32_len("", PACK_ROWS * width)?;
            for (s, v) in sum.iter_mut().zip(values) {
                *s += v;
            }
        }
        let n = names.len() as f32;
        sum.iter_mut().for_each(|v| *v /= n);
        let pack = Arc::new(sum);
        self.packs.lock().insert(voice.to_string(), pack.clone());
        Ok(pack)
    }

    /// The model's ids of `phonemes`; characters outside its vocabulary drop.
    pub fn ids(&self, phonemes: &str) -> Vec<usize> {
        phonemes
            .chars()
            .filter_map(|c| self.vocab.get(&c).copied())
            .collect()
    }

    /// The speech of one phoneme string (at most `max_phonemes`), mono at
    /// 24 kHz. `noise` scales the noise of the harmonic source (1 is the model's).
    pub fn speak(
        &self,
        phonemes: &str,
        pack: &[f32],
        speed: f32,
        noise: f32,
        rng: &mut Noise,
    ) -> Result<Vec<f32>, String> {
        let width = 2 * self.style;
        let row = phonemes.chars().count().clamp(1, PACK_ROWS) - 1;
        let reference = &pack[row * width..(row + 1) * width];
        let (decoder_style, prosody_style) = reference.split_at(self.style);
        let mut ids = vec![0];
        ids.extend(self.ids(phonemes));
        ids.push(0);
        let tokens = ids.len();
        if tokens > self.max_phonemes + 2 {
            return Err(format!(
                "{} phonemes are more than the {} of one pass",
                tokens - 2,
                self.max_phonemes
            ));
        }
        let (hidden, style) = (self.hidden, self.style);

        // Durations: frames per phoneme.
        let bert = self.albert.forward(&ids)?;
        let d_en = self.bert_encoder.forward(&bert, tokens);
        let d = self.durations.forward(&d_en, tokens, prosody_style);
        let x = self.lstm.forward(&d, tokens);
        let mut logits = self.duration_proj.forward(&x, tokens);
        crate::cpu::activation::sigmoid(&mut logits);
        let repeats: Vec<usize> = logits
            .chunks_exact(self.duration_proj.out_features)
            .map(|row| {
                let total = row.iter().sum::<f32>() / speed;
                total.round_ties_even().max(1.0) as usize
            })
            .collect();
        let frames: usize = repeats.iter().sum();

        // Pitch and energy, two values per frame.
        let en = expand(&d, hidden + style, &repeats);
        let shared = self.shared.forward(&en, frames);
        let curve = |blocks: &[AdaInResBlock], proj: &Conv1d| -> Vec<f32> {
            let mut x = {
                let mut copy = Buf::scratch(shared.len());
                copy.copy_from_slice(&shared);
                copy
            };
            let mut length = frames;
            for block in blocks {
                (x, length) = block.forward(&x, length, prosody_style);
            }
            proj.forward(&x, length, None).to_vec()
        };
        let f0 = curve(&self.f0, &self.f0_proj);
        let energy = curve(&self.n, &self.n_proj);

        // The text, frame by frame, then the decoder.
        let t_en = self.text.forward(&ids);
        let asr = expand(&t_en, hidden, &repeats);
        Ok(self
            .decoder
            .forward(&asr, frames, (&f0, &energy), decoder_style, noise, rng))
    }
}

/// Row `i` of `x` (`rows x width`) `repeats[i]` times.
fn expand(x: &[f32], width: usize, repeats: &[usize]) -> Buf {
    let frames: usize = repeats.iter().sum();
    let mut out = Buf::scratch(frames * width);
    let mut at = 0;
    for (row, n) in x.chunks_exact(width).zip(repeats) {
        for _ in 0..*n {
            out[at * width..(at + 1) * width].copy_from_slice(row);
            at += 1;
        }
    }
    out
}

/// `x` (`rows x width`) with `extra` appended to every row.
fn append(x: &[f32], width: usize, extra: &[f32]) -> Buf {
    let rows = x.len() / width;
    let wide = width + extra.len();
    let mut out = Buf::scratch(rows * wide);
    for (o, row) in out.chunks_exact_mut(wide).zip(x.chunks_exact(width)) {
        o[..width].copy_from_slice(row);
        o[width..].copy_from_slice(extra);
    }
    out
}

/// The prosody predictor's text encoder: BiLSTM and AdaLayerNorm in turn,
/// the style appended to every frame before each LSTM.
struct DurationEncoder {
    layers: Vec<(BiLstm, AdaLayerNorm)>,
    hidden: usize,
}

impl DurationEncoder {
    fn load(t: &Tensors, (hidden, style, layers): (usize, usize, usize)) -> Result<Self, String> {
        let name = "predictor.text_encoder.lstms";
        let layers = (0..layers)
            .map(|i| {
                Ok((
                    BiLstm::load(t, &format!("{name}.{}", 2 * i), hidden + style, hidden / 2)?,
                    AdaLayerNorm::load(t, &format!("{name}.{}", 2 * i + 1), style, hidden)?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self { layers, hidden })
    }

    /// `tokens x (hidden + style)`.
    fn forward(&self, x: &[f32], tokens: usize, style: &[f32]) -> Buf {
        let mut x = append(x, self.hidden, style);
        for (lstm, norm) in &self.layers {
            let h = lstm.forward(&x, tokens);
            let h = norm.apply(&h, style);
            x = append(&h, self.hidden, style);
        }
        x
    }
}

/// Kokoro's text encoder: embedding, convolutions with layer norm and leaky
/// ReLU, and a BiLSTM.
struct TextEncoder {
    embedding: Vec<f32>,
    convs: Vec<(Conv1d, Norm)>,
    lstm: BiLstm,
    hidden: usize,
}

impl TextEncoder {
    fn load(
        t: &Tensors,
        c: &Json,
        (vocab, hidden, layers): (usize, usize, usize),
    ) -> Result<Self, String> {
        let kernel = c.num("text_encoder_kernel_size")?;
        let convs = (0..layers)
            .map(|i| {
                let norm = format!("text_encoder.cnn.{i}.1");
                Ok((
                    Conv1d::load(
                        t,
                        &format!("text_encoder.cnn.{i}.0"),
                        (hidden, hidden),
                        (kernel, 1),
                    )?,
                    Norm::new(
                        t.f32_len(&format!("{norm}.gamma"), hidden)?,
                        t.f32_len(&format!("{norm}.beta"), hidden)?,
                        1e-5,
                    ),
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            embedding: t.f32_len("text_encoder.embedding.weight", vocab * hidden)?,
            convs,
            lstm: BiLstm::load(t, "text_encoder.lstm", hidden, hidden / 2)?,
            hidden,
        })
    }

    fn forward(&self, ids: &[usize]) -> Buf {
        let d = self.hidden;
        let tokens = ids.len();
        let mut x = Buf::scratch(tokens * d);
        for (row, id) in x.chunks_exact_mut(d).zip(ids) {
            row.copy_from_slice(&self.embedding[id * d..(id + 1) * d]);
        }
        for (conv, norm) in &self.convs {
            let h = conv.forward(&x, tokens, None);
            x = norm.apply(&h, d);
            team::chunks_mut(&mut x, 8192, |_, part| {
                part.iter_mut().for_each(|v| *v = v.max(*v * 0.2))
            });
        }
        self.lstm.forward(&x, tokens)
    }
}
