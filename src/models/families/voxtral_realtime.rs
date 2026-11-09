//! Voxtral Mini 4B Realtime (Mistral AI), run offline on the whole clip as
//! Mistral's offline mode does: the audio padded into its 80 ms token grid,
//! log-mel features with a fixed peak, a causal encoder (causal conv stem,
//! RMS norms, rotary positions, attention over a sliding window of the
//! frames before), a projector that joins 4 frames, and a Ministral
//! language model whose every input gets the next audio row added. Each
//! layer's MLP input is scaled by a constant from the transcription delay.
//! The structure follows transcribe.cpp's port of
//! `VoxtralRealtimeForConditionalGeneration`.

use crate::audio::dft::PowerSpectrum;
use crate::config::Json;
use crate::cpu::{activation, buf::Buf, fgemm, gemm::Dense, team, tensors::Tensors};
use crate::layers::{add_into, conv_columns, linear, RmsNorm, Rotary};
use crate::models::files::Files;
use crate::models::parts::causal_lm::CausalLm;
use crate::models::{speech_windows, Family};
use crate::text::bpe::Bpe;
use crate::{Device, Options, Precision, Segment};

const WINDOW_SAMPLES: usize = 60 * crate::SAMPLE_RATE as usize;
/// Mistral's best offline delay, in 80 ms tokens (transcribe.cpp's offline default).
const OFFLINE_DELAY: usize = 30;
/// Tokens of silence before the audio (tekken's `streaming_n_left_pad_tokens`).
const LEFT_PAD: usize = 32;
/// Tokens of silence after it beyond the delay and BOS (`OFFLINE_STREAMING_BUFFER_TOKENS`).
const BUFFER: usize = 10;
const LOG_FLOOR: f32 = 1e-10;
const QUERY_BLOCK: usize = 256;

/// Log-mel features with a fixed peak: a centered STFT with the file's
/// window and filterbank, `log10`, a floor 8 below `peak`, `(x + 4) / 4`,
/// and no frame for the last center.
struct Frontend {
    n_fft: usize,
    hop: usize,
    mels: usize,
    peak: f32,
    spectrum: PowerSpectrum,
    filters: Vec<f32>,
}

impl Frontend {
    fn load(json: &Json, t: &Tensors) -> Result<Self, String> {
        let fe = |key: &str| json.num(&format!("feature_extractor.{key}"));
        let (n_fft, win, mels) = (fe("n_fft")?, fe("win_length")?, fe("num_mel_bins")?);
        if win > n_fft {
            return Err("config.json: the window is longer than the transform".into());
        }
        let window = t.f32_len("frontend.window", win)?;
        let mut padded = vec![0.0f32; n_fft];
        padded[(n_fft - win) / 2..][..win].copy_from_slice(&window);
        Ok(Self {
            n_fft,
            hop: fe("hop_length")?,
            mels,
            peak: json.float("feature_extractor.log_mel_max").unwrap_or(1.5) as f32,
            spectrum: PowerSpectrum::new(n_fft, &padded),
            filters: t.f32_len("frontend.mel_filterbank", mels * (n_fft / 2 + 1))?,
        })
    }

    /// `frames x mels`, frames = samples / hop.
    fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let frames = samples.len() / self.hop;
        let len = samples.len() as isize;
        if frames == 0 || len <= (self.n_fft / 2) as isize {
            return (Vec::new(), 0);
        }
        let reflect = |i: isize| -> f32 {
            let i = if i < 0 { -i } else { i };
            let i = if i >= len { 2 * (len - 1) - i } else { i };
            samples[i.clamp(0, len - 1) as usize]
        };
        let half = (self.n_fft / 2) as isize;
        let mut windows = Buf::scratch(frames * self.n_fft);
        team::chunks_mut(&mut windows, self.n_fft, |f, row| {
            let start = (f * self.hop) as isize - half;
            for (n, value) in row.iter_mut().enumerate() {
                *value = reflect(start + n as isize);
            }
        });
        let power = self.spectrum.compute(&windows, frames);
        let mel = fgemm::matmul(
            &power,
            frames,
            self.spectrum.bins(),
            &self.filters,
            self.mels,
        );
        let floor = self.peak - 8.0;
        let out = mel
            .iter()
            .map(|m| (m.max(LOG_FLOOR).log10().max(floor) + 4.0) / 4.0)
            .collect();
        (out, frames)
    }
}

struct Layer {
    norm_attn: RmsNorm,
    /// q, k, v stacked: `3 * heads * head_dim x d` (k has no bias).
    qkv: Dense,
    out: Dense,
    norm_ffn: RmsNorm,
    /// gate and up stacked.
    gate_up: Dense,
    down: Dense,
}

struct Encoder {
    d: usize,
    heads: usize,
    head_dim: usize,
    ffn: usize,
    window: usize,
    conv1: Dense,
    conv2: Dense,
    layers: Vec<Layer>,
    norm: RmsNorm,
    rotary: Rotary,
}

impl Encoder {
    fn load(t: &Tensors, json: &Json) -> Result<Self, String> {
        let e = |key: &str| json.num(&format!("encoder_config.{key}"));
        let (d, heads, hd, ffn, mels) = (
            e("hidden_size")?,
            e("num_attention_heads")?,
            e("head_dim")?,
            e("intermediate_size")?,
            e("num_mel_bins")?,
        );
        let eps = json.float("encoder_config.rms_norm_eps").unwrap_or(1e-5) as f32;
        let attn = heads * hd;
        let p = "audio_tower";
        let layers = team::map(e("num_hidden_layers")?, |i| -> Result<Layer, String> {
            let b = format!("{p}.layers.{i}");
            let a = |n: &str| format!("{b}.self_attn.{n}");
            let m = |n: &str| format!("{b}.mlp.{n}");
            Ok(Layer {
                norm_attn: RmsNorm::load(t, &format!("{b}.input_layernorm"), d, eps)?,
                qkv: Dense::load(
                    t,
                    &[
                        (&a("q_proj"), attn),
                        (&a("k_proj"), attn),
                        (&a("v_proj"), attn),
                    ],
                    d,
                )?,
                out: linear(t, &a("o_proj"), d, attn)?,
                norm_ffn: RmsNorm::load(t, &format!("{b}.post_attention_layernorm"), d, eps)?,
                gate_up: Dense::load(t, &[(&m("gate_proj"), ffn), (&m("up_proj"), ffn)], d)?,
                down: linear(t, &m("down_proj"), d, ffn)?,
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            d,
            heads,
            head_dim: hd,
            ffn,
            window: e("sliding_window")?,
            conv1: linear(t, &format!("{p}.conv1"), d, 3 * mels)?,
            conv2: linear(t, &format!("{p}.conv2"), d, 3 * d)?,
            layers,
            norm: RmsNorm::load(t, &format!("{p}.norm"), d, eps)?,
            rotary: Rotary::halves(
                hd,
                json.float("encoder_config.rope_theta").unwrap_or(1e6) as f32,
            ),
        })
    }

    /// `t x d` from `frames x mels`: causal convs (stride 1, then 2), then the layers.
    fn encode(&self, mel: &[f32], frames: usize, mels: usize) -> (Buf, usize) {
        let d = self.d;
        let cols = conv_columns(mel, (frames, mels), (3, 1, 2), frames);
        let mut h = self.conv1.forward(&cols, frames);
        activation::gelu(&mut h);
        let t = (frames - 2) / 2 + 1;
        let cols = conv_columns(&h, (frames, d), (3, 2, 1), t);
        let mut x = self.conv2.forward(&cols, t);
        activation::gelu(&mut x);
        let (heads, hd) = (self.heads, self.head_dim);
        let attn = heads * hd;
        for layer in &self.layers {
            let mut qkv = layer.qkv.forward(&layer.norm_attn.apply(&x), t);
            self.rotary.apply(&mut qkv, (t, 3 * attn), (heads, hd), 0);
            self.rotary
                .apply(&mut qkv[attn..], (t, 3 * attn), (heads, hd), 0);
            let heard = self.window_attention(&qkv, t);
            add_into(&mut x, &layer.out.forward(&heard, t));
            let wide = layer.gate_up.forward(&layer.norm_ffn.apply(&x), t);
            let ffn = self.ffn;
            let mut hidden = Buf::scratch(t * ffn);
            team::chunks_mut(&mut hidden, ffn, |r, row| {
                let w = &wide[r * 2 * ffn..(r + 1) * 2 * ffn];
                row.copy_from_slice(&w[..ffn]);
                activation::silu(row);
                row.iter_mut().zip(&w[ffn..]).for_each(|(g, u)| *g *= u);
            });
            add_into(&mut x, &layer.down.forward(&hidden, t));
        }
        (self.norm.apply(&x), t)
    }

    /// Causal attention over the last `window` frames, head by head, in
    /// blocks of queries against the keys they reach.
    fn window_attention(&self, qkv: &[f32], t: usize) -> Buf {
        let (heads, hd) = (self.heads, self.head_dim);
        let attn = heads * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = Buf::zeroed(t * attn);
        for h in 0..heads {
            let gather = |offset: usize| -> Vec<f32> {
                (0..t)
                    .flat_map(|i| qkv[i * 3 * attn + offset + h * hd..][..hd].iter().copied())
                    .collect()
            };
            let (q, k) = (gather(0), gather(attn));
            let v = gather(2 * attn);
            for first_q in (0..t).step_by(QUERY_BLOCK) {
                let n = QUERY_BLOCK.min(t - first_q);
                let first_k = (first_q + 1).saturating_sub(self.window);
                let m = first_q + n - first_k;
                let mut scores = fgemm::matmul(
                    &q[first_q * hd..(first_q + n) * hd],
                    n,
                    hd,
                    &k[first_k * hd..(first_k + m) * hd],
                    m,
                );
                team::chunks_mut(&mut scores, m, |r, row| {
                    let i = first_q + r;
                    let (lo, hi) = ((i + 1).saturating_sub(self.window) - first_k, i - first_k);
                    let keys = &mut row[lo..=hi];
                    let max = keys
                        .iter()
                        .fold(f32::NEG_INFINITY, |a, v| a.max(*v * scale));
                    let mut sum = 0.0;
                    for s in keys.iter_mut() {
                        *s = (*s * scale - max).exp();
                        sum += *s;
                    }
                    keys.iter_mut().for_each(|s| *s /= sum);
                    row[..lo].fill(0.0);
                    row[hi + 1..].fill(0.0);
                });
                // V of these keys, transposed: `hd x m`.
                let mut v_t = vec![0.0f32; hd * m];
                for j in 0..m {
                    for e in 0..hd {
                        v_t[e * m + j] = v[(first_k + j) * hd + e];
                    }
                }
                let heard = fgemm::matmul(&scores, n, m, &v_t, hd);
                for (r, row) in heard.chunks_exact(hd).enumerate() {
                    out[(first_q + r) * attn + h * hd..][..hd].copy_from_slice(row);
                }
            }
        }
        out
    }
}

pub struct VoxtralRealtime {
    frontend: Frontend,
    encoder: Encoder,
    /// Encoder frames per language model row.
    join: usize,
    /// Mel frames per 80 ms token, times the hop: samples per token.
    samples_per_token: usize,
    linear1: Dense,
    linear2: Dense,
    decoder: CausalLm,
    bpe: Bpe,
    bos: usize,
    eos: usize,
    pad: usize,
}

impl Family for VoxtralRealtime {
    fn load(
        files: &Files,
        json: &Json,
        precision: Precision,
        device: Device,
    ) -> Result<Self, String> {
        let t = files.tensors(precision, device)?;
        let frontend = Frontend::load(json, &t)?;
        let encoder = Encoder::load(&t, json)?;
        let join = json.num("projector_config.downsample_factor")?;
        let mut decoder = CausalLm::from_tensors(&t, json, ("llm_config.", "language_model."))?;
        let width = decoder.width();
        let linear1 = linear(
            &t,
            "multi_modal_projector.linear_1",
            width,
            join * encoder.d,
        )?;
        let linear2 = linear(&t, "multi_modal_projector.linear_2", width, width)?;
        // Each layer's MLP scale, `1 + ada(t_cond)`, from the offline delay.
        let inv_freq = t.f32_len("time_embed.inv_freq", width / 2)?;
        let mut t_cond = vec![0.0f32; width];
        for (i, f) in inv_freq.iter().enumerate() {
            let e = OFFLINE_DELAY as f64 * *f as f64;
            t_cond[i] = e.cos() as f32;
            t_cond[width / 2 + i] = e.sin() as f32;
        }
        let ada_hidden = json.num("llm_config.ada_hidden")?;
        let scales = (0..decoder.layers())
            .map(|l| -> Result<Vec<f32>, String> {
                let p = format!("language_model.model.layers.{l}.ada");
                let mut h =
                    linear(&t, &format!("{p}.linear_1"), ada_hidden, width)?.forward(&t_cond, 1);
                activation::gelu(&mut h);
                let ada = linear(&t, &format!("{p}.linear_2"), width, ada_hidden)?.forward(&h, 1);
                Ok(ada.iter().map(|a| 1.0 + a).collect())
            })
            .collect::<Result<Vec<_>, _>>()?;
        decoder.set_ffn_scales(scales);
        let samples_per_token = json.num("projector_config.audio_length_per_tok")? * frontend.hop;
        drop(t);
        Ok(Self {
            frontend,
            encoder,
            join,
            samples_per_token,
            linear1,
            linear2,
            decoder,
            bpe: Bpe::parse(&files.read("tokenizer.json")?)?,
            bos: json.num("bos_token_id")?,
            eos: json.num("eos_token_id")?,
            pad: json.num("streaming_pad_token_id")?,
        })
    }

    fn segments(&self, samples: &[f32], options: &Options) -> Vec<Segment> {
        let per = self.samples_per_token;
        let mut prompt = vec![self.bos];
        prompt.resize(1 + LEFT_PAD + OFFLINE_DELAY, self.pad);
        speech_windows(samples, WINDOW_SAMPLES, options, |window| {
            // Mistral's offline layout: silence, the audio up to whole
            // tokens, silence for the delay, BOS, and a buffer.
            let left = LEFT_PAD * per;
            let mut padded = vec![
                0.0f32;
                left + window.len().div_ceil(per) * per
                    + (OFFLINE_DELAY + 1 + BUFFER) * per
            ];
            padded[left..left + window.len()].copy_from_slice(window);
            let (mel, frames) = self.frontend.compute(&padded);
            if frames < 2 {
                return String::new();
            }
            let (encoded, t) = self.encoder.encode(&mel, frames, self.frontend.mels);
            let rows = t / self.join;
            let mut h = self
                .linear1
                .forward(&encoded[..rows * self.join * self.encoder.d], rows);
            activation::gelu(&mut h);
            let audio = self.linear2.forward(&h, rows);
            let ids = self.decoder.generate_additive(&prompt, &audio, &[self.eos]);
            self.bpe.decode(&ids).trim().to_string()
        })
    }
}
