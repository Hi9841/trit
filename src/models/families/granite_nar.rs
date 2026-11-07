//! Granite Speech 4.1 2B NAR (IBM's NLE editor): the Granite Conformer
//! (`parts::granite_conformer`) gives a first hypothesis from a BPE CTC head
//! over windows of 4 frames (each frame weighted by how little the
//! self-conditioning CTC hears a blank); end-of-text tokens go between and
//! around its tokens; a projector turns 4 captured encoder layers into 3
//! rows per 15 frames; and the Granite language model reads the rows and the
//! hypothesis in one bidirectional pass. The best token of each hypothesis
//! position, repeats and end-of-text dropped, is the text. The structure
//! follows transcribe.cpp's port of `NLENARDecoder`.

use crate::config::Json;
use crate::cpu::{activation, buf::Buf, gemm::Dense, team, tensors::Tensors};
use crate::layers::{add_into, attention, linear, Gelu, Norm};
use crate::models::files::Files;
use crate::models::parts::causal_lm::CausalLm;
use crate::models::parts::granite_conformer::{Config, Encoder, Frontend};
use crate::models::{speech_windows, Family};
use crate::text::bpe::Bpe;
use crate::{Device, Options, Precision, Segment};

/// Longest window per pass (the editor reads all of it at once).
const WINDOW_SAMPLES: usize = 60 * crate::SAMPLE_RATE as usize;

struct ProjectorLayer {
    norm_attn: Norm,
    q: Dense,
    k: Dense,
    v: Dense,
    o: Dense,
    norm_ffn: Norm,
    fc1: Dense,
    fc2: Dense,
}

/// IBM's `EncoderProjectorQFormer` of the NLE model: a norm per captured
/// layer, a GELU projection of their joined channels, then per window of
/// `block` frames (with learned positions) queries that start as the mean
/// of each `rate` frames plus a learned row, pre-norm cross-attention and
/// SiLU feed-forward layers, a final norm, and a linear projection.
struct Projector {
    norms: Vec<Norm>,
    projection: Dense,
    window_positions: Vec<f32>,
    query: Vec<f32>,
    layers: Vec<ProjectorLayer>,
    out_norm: Norm,
    out: Dense,
    d: usize,
    heads: usize,
    block: usize,
    rate: usize,
    encoder_d: usize,
}

impl Projector {
    fn load(t: &Tensors, json: &Json) -> Result<Self, String> {
        let p = |key: &str| json.num(&format!("projector_config.{key}"));
        let (d, heads, block, rate) = (
            p("hidden_size")?,
            p("num_attention_heads")?,
            p("block_size")?,
            p("downsample_rate")?,
        );
        let (encoder_d, captured) = (p("encoder_dim")?, p("num_encoder_layers")?);
        let ffn = d * p("mlp_ratio")?;
        if !d.is_multiple_of(heads) || rate == 0 || !block.is_multiple_of(rate) {
            return Err("config.json: the NAR projector's shape does not fit".into());
        }
        let eps = json
            .float("projector_config.layer_norm_eps")
            .unwrap_or(1e-6) as f32;
        let n = |name: &str, width: usize| -> Result<Norm, String> {
            Ok(Norm::load(t, &format!("nar_projector.{name}"), width)?.with_eps(eps))
        };
        let norms = (0..captured)
            .map(|j| n(&format!("layer_norms.{j}"), encoder_d))
            .collect::<Result<Vec<_>, _>>()?;
        let layers = (0..p("num_hidden_layers")?)
            .map(|i| -> Result<ProjectorLayer, String> {
                let b = format!("nar_projector.blocks.{i}");
                let l = |name: &str, out: usize, inp: usize| {
                    linear(t, &format!("{b}.{name}"), out, inp)
                };
                Ok(ProjectorLayer {
                    norm_attn: n(&format!("blocks.{i}.norm_attn"), d)?,
                    q: l("cross_attn_q", d, d)?,
                    k: l("cross_attn_k", d, d)?,
                    v: l("cross_attn_v", d, d)?,
                    o: l("cross_attn_o", d, d)?,
                    norm_ffn: n(&format!("blocks.{i}.norm_ffn"), d)?,
                    fc1: l("ffn_fc1", ffn, d)?,
                    fc2: l("ffn_fc2", d, ffn)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            norms,
            projection: linear(t, "nar_projector.layer_projector", d, captured * encoder_d)?,
            window_positions: t.f32_len("nar_projector.window_positions", block * d)?,
            query: t.f32_len("nar_projector.query", block / rate * d)?,
            layers,
            out_norm: n("out_norm", d)?,
            out: linear(t, "nar_projector.out_linear", p("llm_dim")?, d)?,
            d,
            heads,
            block,
            rate,
            encoder_d,
        })
    }

    /// Rows for the language model from the captured layers (`t x
    /// captured * encoder_d`): `t / rate` of them (the rows of the zero
    /// padding are dropped, as the reference).
    fn forward(&self, captured: &[f32], t: usize) -> (Buf, usize) {
        let (d, e) = (self.d, self.encoder_d);
        let width = self.norms.len() * e;
        let mut normed = Buf::scratch(t * width);
        team::chunks_mut(&mut normed, width, |f, row| {
            for (j, norm) in self.norms.iter().enumerate() {
                let x = norm.apply(&captured[f * width + j * e..][..e], e);
                row[j * e..(j + 1) * e].copy_from_slice(&x);
            }
        });
        let mut h = self.projection.forward(&normed, t);
        // transcribe.cpp's `ggml_gelu`: the tanh form.
        Gelu::Tanh.apply(&mut h, d);
        let windows = t.div_ceil(self.block);
        let mut frames = vec![0.0f32; windows * self.block * d];
        frames[..t * d].copy_from_slice(&h);
        let count = self.block / self.rate;
        let mut tokens = Vec::with_capacity(windows * count * d);
        for window in frames.chunks_exact(self.block * d) {
            let mut kv = window.to_vec();
            add_into(&mut kv, &self.window_positions);
            // Each query starts as the mean of its `rate` frames.
            let mut x = Buf::scratch(count * d);
            for (q, row) in x.chunks_exact_mut(d).enumerate() {
                row.fill(0.0);
                for f in 0..self.rate {
                    add_into(row, &window[(q * self.rate + f) * d..][..d]);
                }
                row.iter_mut().for_each(|v| *v /= self.rate as f32);
            }
            add_into(&mut x, &self.query);
            for layer in &self.layers {
                let q = layer.q.forward(&layer.norm_attn.apply(&x, d), count);
                let (k, v) = (
                    layer.k.forward(&kv, self.block),
                    layer.v.forward(&kv, self.block),
                );
                let heard = attention(
                    (&q, d),
                    (&k, d),
                    (&v, d),
                    (count, self.block),
                    (d, self.heads, self.heads),
                    false,
                );
                add_into(&mut x, &layer.o.forward(&heard, count));
                let mut mid = layer.fc1.forward(&layer.norm_ffn.apply(&x, d), count);
                activation::silu(&mut mid);
                add_into(&mut x, &layer.fc2.forward(&mid, count));
            }
            tokens.extend_from_slice(&self.out_norm.apply(&x, d));
        }
        let rows = (t / self.rate).min(windows * count);
        (self.out.forward(&tokens[..rows * d], rows), rows)
    }
}

pub struct GraniteNar {
    frontend: Frontend,
    encoder: Encoder,
    /// The BPE CTC head over pooled frames.
    bpe_head: Dense,
    bpe_blank: usize,
    pool: usize,
    projector: Projector,
    decoder: CausalLm,
    bpe: Bpe,
    eos: usize,
    /// The projector's rows are divided by the language model's embedding
    /// multiplier, which the model then applies to every row.
    embedding_divisor: f32,
}

impl GraniteNar {
    /// The first hypothesis: per window of `pool` frames, the BPE head on
    /// the frames weighted by `1 - p(blank)`, repeats and blanks dropped. A
    /// window the CTC hears as all blank is a blank.
    fn hypothesis(&self, last: &[f32], blank: &[f32], t: usize) -> Vec<usize> {
        let d = self.encoder.config.d;
        let mut ids = Vec::new();
        let mut previous = None;
        for first in (0..t).step_by(self.pool) {
            let frames = first..(first + self.pool).min(t);
            let weights: Vec<f32> = frames.clone().map(|f| 1.0 - blank[f]).collect();
            let mass: f32 = weights.iter().sum();
            let id = if mass <= 1e-9 {
                self.bpe_blank
            } else {
                let mut pooled = vec![0.0f32; d];
                for (f, w) in frames.zip(&weights) {
                    for (p, v) in pooled.iter_mut().zip(&last[f * d..(f + 1) * d]) {
                        *p += w / mass * v;
                    }
                }
                crate::layers::argmax(&self.bpe_head.forward(&pooled, 1))
            };
            if id != self.bpe_blank && Some(id) != previous {
                // A blank at id 0 shifts the ids by one (a head one wider than the vocabulary).
                ids.push(if self.bpe_blank == 0 { id - 1 } else { id });
            }
            previous = Some(id);
        }
        ids
    }
}

impl Family for GraniteNar {
    fn load(
        files: &Files,
        json: &Json,
        precision: Precision,
        device: Device,
    ) -> Result<Self, String> {
        let config = Config::parse(json)?;
        let d = config.d;
        let t = files.tensors(precision, device)?;
        let frontend = Frontend::load(json, &t)?;
        let encoder = Encoder::load(&t, config)?;
        let decoder = CausalLm::from_tensors(&t, json, ("llm_config.", "language_model."))?;
        let embedding_divisor = match json.flag("scale_projected_embeddings") {
            Some(true) => json.float("llm_config.embedding_multiplier").unwrap_or(1.0) as f32,
            _ => 1.0,
        };
        Ok(Self {
            bpe_head: linear(&t, "encoder.ctc_bpe", json.num("bpe_vocab")?, d)?,
            bpe_blank: json.num("bpe_blank_id")?,
            pool: json.num("bpe_pool_window")?,
            projector: Projector::load(&t, json)?,
            frontend,
            encoder,
            decoder,
            bpe: Bpe::parse(&files.read("tokenizer.json")?)?,
            eos: json.num("eos_token_id")?,
            embedding_divisor,
        })
    }

    fn segments(&self, samples: &[f32], _: &Options) -> Vec<Segment> {
        speech_windows(samples, WINDOW_SAMPLES, |window| {
            let (features, frames) = self.frontend.compute(window);
            if frames == 0 {
                return String::new();
            }
            let encoded = self.encoder.encode_all(&features, frames);
            let t = encoded.t;
            let hypothesis = self.hypothesis(&encoded.last, &encoded.blank, t);
            if hypothesis.is_empty() {
                return String::new();
            }
            // End-of-text between and around the tokens, at least 8 positions.
            let n = hypothesis.len();
            let mut text_ids = vec![self.eos; (2 * n + 1).max(8)];
            for (i, id) in hypothesis.iter().enumerate() {
                text_ids[2 * i + 1] = *id;
            }
            let (mut audio, rows) = self.projector.forward(&encoded.joined, t);
            audio.iter_mut().for_each(|v| *v /= self.embedding_divisor);
            let d = self.decoder.width();
            let mut input = Buf::scratch((rows + text_ids.len()) * d);
            input[..rows * d].copy_from_slice(&audio);
            input[rows * d..].copy_from_slice(&self.decoder.embed(&text_ids));
            let best = self.decoder.bidirectional_tokens(input, rows);
            let mut ids = Vec::new();
            let mut previous = None;
            for id in best {
                if id != self.eos && Some(id) != previous {
                    ids.push(id);
                }
                previous = Some(id);
            }
            self.bpe.decode(&ids).trim().to_string()
        })
    }
}
