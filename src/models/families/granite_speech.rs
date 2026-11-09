//! Granite Speech 4.0 and 4.1 (IBM): the Granite Conformer
//! (`parts::granite_conformer`), a Q-Former that reads windows of 15
//! encoder frames with 3 learned queries each, a projection to the language
//! model's width, and a Granite language model (`parts::causal_lm`). The
//! query rows take the place of the `<|audio|>` token of IBM's prompt
//! ("USER: <|audio|>can you transcribe the speech into a written format?\n
//! ASSISTANT:", or the Granite 4 chat roles of the plus release). The
//! structure follows transcribe.cpp's port of `GraniteSpeechForConditionalGeneration`.

use crate::config::Json;
use crate::cpu::{activation, buf::Buf, gemm::Dense, tensors::Tensors};
use crate::layers::{add_into, attention, linear, Norm};
use crate::models::files::Files;
use crate::models::parts::causal_lm::{CausalLm, TokenBudget};
use crate::models::parts::granite_conformer::{Config, Encoder, Frontend};
use crate::models::{speech_windows, Family};
use crate::text::bpe::Bpe;
use crate::{Device, Options, Precision, Segment};

const WINDOW_SAMPLES: usize = 30 * crate::SAMPLE_RATE as usize;
const BUDGET: TokenBudget = TokenBudget {
    least: 128,
    per_second: 8,
};

struct QFormerLayer {
    self_q: Dense,
    self_k: Dense,
    self_v: Dense,
    self_out: Dense,
    norm_self: Norm,
    cross_q: Dense,
    cross_k: Dense,
    cross_v: Dense,
    cross_out: Dense,
    norm_cross: Norm,
    up: Dense,
    down: Dense,
    norm_ffn: Norm,
}

/// BLIP-2's Q-Former as Granite Speech uses it: learned queries, normed,
/// then post-norm layers of self-attention, cross-attention into one window
/// of encoder frames, and a GELU feed-forward; then a linear projection.
struct QFormer {
    /// The learned queries after `final_norm`, `queries x d`.
    queries: Vec<f32>,
    count: usize,
    layers: Vec<QFormerLayer>,
    projection: Dense,
    d: usize,
    heads: usize,
    window: usize,
}

impl QFormer {
    fn load(
        t: &Tensors,
        json: &Json,
        encoder_width: usize,
        llm_width: usize,
    ) -> Result<Self, String> {
        let p = |key: &str| json.num(&format!("projector_config.{key}"));
        let (d, ffn, heads) = (
            p("hidden_size")?,
            p("intermediate_size")?,
            p("num_attention_heads")?,
        );
        let (window, rate) = (p("window_size")?, p("downsample_rate")?);
        let enc = p("encoder_hidden_size")?;
        if enc != encoder_width
            || !d.is_multiple_of(heads)
            || rate == 0
            || !window.is_multiple_of(rate)
        {
            return Err("config.json: the projector does not fit the encoder".into());
        }
        let eps = json
            .float("projector_config.layer_norm_eps")
            .unwrap_or(1e-12) as f32;
        let count = window / rate;
        let norm = Norm::load(t, "projector.qformer.final_norm", d)?.with_eps(eps);
        let queries = norm
            .apply(&t.f32_len("projector.query", count * d)?, d)
            .to_vec();
        let layers = (0..p("num_hidden_layers")?)
            .map(|i| -> Result<QFormerLayer, String> {
                let b = format!("projector.qformer.blocks.{i}");
                let l = |name: &str, out: usize, inp: usize| {
                    linear(t, &format!("{b}.{name}"), out, inp)
                };
                let n = |name: &str| {
                    Ok::<_, String>(Norm::load(t, &format!("{b}.{name}"), d)?.with_eps(eps))
                };
                Ok(QFormerLayer {
                    self_q: l("self_attn.q", d, d)?,
                    self_k: l("self_attn.k", d, d)?,
                    self_v: l("self_attn.v", d, d)?,
                    self_out: l("self_attn.out", d, d)?,
                    norm_self: n("norm_self_attn")?,
                    cross_q: l("cross_attn.q", d, d)?,
                    cross_k: l("cross_attn.k", d, enc)?,
                    cross_v: l("cross_attn.v", d, enc)?,
                    cross_out: l("cross_attn.out", d, d)?,
                    norm_cross: n("norm_cross_attn")?,
                    up: l("ffn.up", ffn, d)?,
                    down: l("ffn.down", d, ffn)?,
                    norm_ffn: n("norm_ffn")?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            queries,
            count,
            layers,
            projection: linear(t, "projector.linear", llm_width, d)?,
            d,
            heads,
            window,
        })
    }

    /// Rows for the language model, `count` per window of encoder frames;
    /// the last window is padded with zero frames, as the reference.
    fn forward(&self, encoded: &[f32], t: usize, width: usize) -> (Buf, usize) {
        let (d, n) = (self.d, self.count);
        let windows = t.div_ceil(self.window);
        let mut padded = vec![0.0f32; windows * self.window * width];
        padded[..t * width].copy_from_slice(&encoded[..t * width]);
        let mut tokens = Vec::with_capacity(windows * n * d);
        for frames in padded.chunks_exact(self.window * width) {
            let mut x = Buf::scratch(n * d);
            x.copy_from_slice(&self.queries);
            for layer in &self.layers {
                let (q, k, v) = (
                    layer.self_q.forward(&x, n),
                    layer.self_k.forward(&x, n),
                    layer.self_v.forward(&x, n),
                );
                let heard = attention(
                    (&q, d),
                    (&k, d),
                    (&v, d),
                    (n, n),
                    (d, self.heads, self.heads),
                    false,
                );
                let mut y = layer.self_out.forward(&heard, n);
                add_into(&mut y, &x);
                let x1 = layer.norm_self.apply(&y, d);
                let q = layer.cross_q.forward(&x1, n);
                let (k, v) = (
                    layer.cross_k.forward(frames, self.window),
                    layer.cross_v.forward(frames, self.window),
                );
                let heard = attention(
                    (&q, d),
                    (&k, d),
                    (&v, d),
                    (n, self.window),
                    (d, self.heads, self.heads),
                    false,
                );
                let mut y = layer.cross_out.forward(&heard, n);
                add_into(&mut y, &x1);
                let x2 = layer.norm_cross.apply(&y, d);
                let mut h = layer.up.forward(&x2, n);
                activation::gelu(&mut h);
                let mut y = layer.down.forward(&h, n);
                add_into(&mut y, &x2);
                x = layer.norm_ffn.apply(&y, d);
            }
            tokens.extend_from_slice(&x);
        }
        let rows = windows * n;
        (self.projection.forward(&tokens, rows), rows)
    }
}

pub struct GraniteSpeech {
    frontend: Frontend,
    encoder: Encoder,
    projector: QFormer,
    decoder: CausalLm,
    bpe: Bpe,
    head: Vec<usize>,
    tail: Vec<usize>,
    stop: Vec<usize>,
}

impl Family for GraniteSpeech {
    fn load(
        files: &Files,
        json: &Json,
        precision: Precision,
        device: Device,
    ) -> Result<Self, String> {
        let config = Config::parse(json)?;
        let t = files.tensors(precision, device)?;
        let frontend = Frontend::load(json, &t)?;
        let width = config.width();
        let encoder = Encoder::load(&t, config)?;
        let decoder = CausalLm::from_tensors(&t, json, ("llm_config.", "language_model."))?;
        let projector = QFormer::load(&t, json, width, decoder.width())?;
        drop(t);
        let bpe = Bpe::parse(&files.read("tokenizer.json")?)?;
        let id = |token: &str| bpe.token(token);
        let instruction = json.str("instruction").unwrap_or_default();
        let (head, tail) = match json.str("chat") {
            Some("roles") => {
                let (start, end, text_end) = (
                    id("<|start_of_role|>")?,
                    id("<|end_of_role|>")?,
                    id("<|end_of_text|>")?,
                );
                let mut head = vec![start];
                head.extend(bpe.encode_plain("system"));
                head.push(end);
                head.extend(bpe.encode_plain(json.str("system").unwrap_or_default()));
                head.push(text_end);
                head.extend(bpe.encode_plain("\n"));
                head.push(start);
                head.extend(bpe.encode_plain("user"));
                head.push(end);
                let mut tail = bpe.encode_plain(instruction);
                tail.push(text_end);
                tail.extend(bpe.encode_plain("\n"));
                tail.push(start);
                tail.extend(bpe.encode_plain("assistant"));
                tail.push(end);
                (head, tail)
            }
            _ => (
                bpe.encode_plain("USER: "),
                bpe.encode_plain(&format!("{instruction}\n ASSISTANT:")),
            ),
        };
        let stop = vec![json.num("eos_token_id")?];
        Ok(Self {
            frontend,
            encoder,
            projector,
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
            if frames == 0 {
                return String::new();
            }
            let (encoded, t) = self.encoder.encode(&features, frames);
            let (audio, _) = self
                .projector
                .forward(&encoded, t, self.encoder.config.width());
            let budget = BUDGET.tokens(window.len());
            let ids = self
                .decoder
                .answer(&self.head, &audio, &self.tail, budget, &self.stop);
            self.bpe.decode(&ids).trim().to_string()
        })
    }
}
