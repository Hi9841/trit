//! ALBERT (Transformers `AlbertModel`), the phoneme encoder of Kokoro:
//! 128-wide embeddings mapped to the hidden width, then one layer whose
//! weights all layers share, run `layers` times. Post-norm, epsilon 1e-12,
//! the tanh GELU (`gelu_new`, ALBERT's default).

use crate::cpu::{buf::Buf, gemm::Dense, tensors::Tensors};
use crate::models::config::Json;
use crate::models::layers::{attention, linear, Gelu, Norm};

const EPS: f32 = 1e-12;

pub struct Albert {
    word: Vec<f32>,
    position: Vec<f32>,
    token_type: Vec<f32>,
    embed_norm: Norm,
    mapping: Dense,
    query: Dense,
    key: Dense,
    value: Dense,
    dense: Dense,
    attention_norm: Norm,
    ffn: Dense,
    ffn_output: Dense,
    full_norm: Norm,
    embedding: usize,
    pub hidden: usize,
    heads: usize,
    layers: usize,
    positions: usize,
}

impl Albert {
    /// `bert.*` of a Kokoro checkpoint; the sizes from `config.json`'s `plbert`.
    pub fn load(t: &Tensors, c: &Json, vocab: usize) -> Result<Self, String> {
        let hidden = c.num("plbert.hidden_size")?;
        let ffn = c.num("plbert.intermediate_size")?;
        let positions = c.num("plbert.max_position_embeddings")?;
        let embedding = c.num("plbert.embedding_size").unwrap_or(128);
        let layer = "bert.encoder.albert_layer_groups.0.albert_layers.0";
        let norm = |name: &str, d: usize| Norm::load(t, name, d).map(|n| n.with_eps(EPS));
        Ok(Self {
            word: t.f32_len("bert.embeddings.word_embeddings.weight", vocab * embedding)?,
            position: t.f32_len(
                "bert.embeddings.position_embeddings.weight",
                positions * embedding,
            )?,
            token_type: t.f32("bert.embeddings.token_type_embeddings.weight")?,
            embed_norm: norm("bert.embeddings.LayerNorm", embedding)?,
            mapping: linear(
                t,
                "bert.encoder.embedding_hidden_mapping_in",
                hidden,
                embedding,
            )?,
            query: linear(t, &format!("{layer}.attention.query"), hidden, hidden)?,
            key: linear(t, &format!("{layer}.attention.key"), hidden, hidden)?,
            value: linear(t, &format!("{layer}.attention.value"), hidden, hidden)?,
            dense: linear(t, &format!("{layer}.attention.dense"), hidden, hidden)?,
            attention_norm: norm(&format!("{layer}.attention.LayerNorm"), hidden)?,
            ffn: linear(t, &format!("{layer}.ffn"), ffn, hidden)?,
            ffn_output: linear(t, &format!("{layer}.ffn_output"), hidden, ffn)?,
            full_norm: norm(&format!("{layer}.full_layer_layer_norm"), hidden)?,
            embedding,
            hidden,
            heads: c.num("plbert.num_attention_heads")?,
            layers: c.num("plbert.num_hidden_layers")?,
            positions,
        })
    }

    /// The last hidden state of `ids` (`tokens x hidden`).
    pub fn forward(&self, ids: &[usize]) -> Result<Buf, String> {
        let (e, d) = (self.embedding, self.hidden);
        let tokens = ids.len();
        if tokens > self.positions {
            return Err(format!(
                "{tokens} phonemes are more than the {} the model reads",
                self.positions
            ));
        }
        let mut x = Buf::scratch(tokens * e);
        for (i, (row, id)) in x.chunks_exact_mut(e).zip(ids).enumerate() {
            let word = &self.word[id * e..(id + 1) * e];
            let position = &self.position[i * e..(i + 1) * e];
            for (((o, w), p), t) in row
                .iter_mut()
                .zip(word)
                .zip(position)
                .zip(&self.token_type[..e])
            {
                *o = w + p + t;
            }
        }
        let x = self.embed_norm.apply(&x, e);
        let mut h = self.mapping.forward(&x, tokens);
        for _ in 0..self.layers {
            let (q, k, v) = (
                self.query.forward(&h, tokens),
                self.key.forward(&h, tokens),
                self.value.forward(&h, tokens),
            );
            let mixed = attention(
                (&q, d),
                (&k, d),
                (&v, d),
                (tokens, tokens),
                (d, self.heads, self.heads),
                false,
            );
            let projected = self.dense.forward(&mixed, tokens);
            let mut attended = self.attention_norm.residual(&mut h, &projected, 1.0);
            let mut inner = self.ffn.forward(&attended, tokens);
            Gelu::Tanh.apply(&mut inner, self.ffn.out_features);
            let out = self.ffn_output.forward(&inner, tokens);
            h = self.full_norm.residual(&mut attended, &out, 1.0);
        }
        Ok(h)
    }
}
