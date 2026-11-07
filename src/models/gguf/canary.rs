//! Canary GGUF files (`general.architecture` `canary`, from transcribe.cpp's
//! `scripts/convert-canary.py`): the `config.json` of the Hugging Face
//! Canary checkpoint (plus the languages, the prompt format, and the
//! encoder width, which Canary 1B v2's folder does not need), and the
//! tensor names. The encoder is Parakeet's FastConformer, so its names
//! come from the Parakeet table with `model.` before them.

use super::{parakeet, Gguf, Name};
use crate::models::package::{encoder_block, EncShape};
use serde_json::{json, Value};

pub fn config(g: &Gguf) -> Result<Value, String> {
    let c = |key: &str| g.num(&format!("stt.canary.{key}"));
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    if text("stt.canary.decoder.activation") != "relu"
        || g.get("stt.canary.decoder.pre_ln").as_bool() == Some(false)
        || g.get("stt.canary.decoder.learn_positional_encodings")
            .as_bool()
            == Some(true)
    {
        return Err("its decoder differs from Canary's (ReLU, pre-norm, fixed positions)".into());
    }
    let format = match text("stt.canary.tokenizer.prompt_format") {
        "" | "canary2" => "canary2",
        "canary" => "canary",
        other => return Err(format!("prompt format `{other}` is not read")),
    };
    let heads = c("decoder.n_heads")?;
    Ok(json!({
        "model_type": "canary",
        "vocab_size": c("decoder.vocab_size")?,
        "pad_token_id": c("special.pad_id")?,
        "eos_token_id": c("special.endoftext_id")?,
        "decoder_config": {
            "hidden_size": c("decoder.d_model")?,
            "intermediate_size": c("decoder.d_ff")?,
            "num_attention_heads": heads,
            "num_key_value_heads": heads,
            "num_hidden_layers": c("decoder.n_layers")?,
            "vocab_size": c("decoder.vocab_size")?,
            "max_position_embeddings": c("decoder.max_position")?,
            "eos_token_id": c("special.endoftext_id")?,
            "hidden_act": "relu",
        },
        "encoder_config": encoder_block(EncShape {
            hidden: c("encoder.d_model")?,
            intermediate: c("encoder.d_ff")?,
            heads: c("encoder.n_heads")?,
            layers: c("encoder.n_layers")?,
            mels: g.num("stt.frontend.num_mels")?,
            conv_kernel: c("encoder.conv_kernel")?,
            subsample: c("encoder.subsampling_factor")?,
            sub_channels: c("encoder.subsampling_channels")?,
            scale_input: g
                .get("stt.canary.encoder.xscaling")
                .as_bool()
                .unwrap_or(false),
        }),
        "languages": g.get("general.languages"),
        "prompt_format": format,
    }))
}

/// A transcribe.cpp Canary tensor name as the Hugging Face checkpoint names it.
pub fn hf_name(name: &str) -> Option<Name> {
    if name == "dec.embed.pos_enc" {
        // NeMo's own position table (sine and cosine interleaved).
        return Some(Name::To("model.decoder.pos_enc".into()));
    }
    if let Some(rest) = name.strip_prefix("enc.proj.") {
        return Some(Name::To(format!("model.encoder_decoder_proj.{rest}")));
    }
    if name.starts_with("enc.") {
        return parakeet::rename(name, "model.", "conv.bn.");
    }
    const TOP: [(&str, &str); 4] = [
        ("dec.embed.token.", "model.decoder.embed_tokens."),
        ("dec.embed.norm.", "model.decoder.embedding_layernorm."),
        ("dec.norm.", "model.decoder.norm."),
        ("dec.head.", "proj_out."),
    ];
    if let Some(mapped) = super::map_suffix(name, &TOP) {
        return Some(Name::To(mapped));
    }
    let (layer, part) = name.strip_prefix("dec.layer.")?.split_once('.')?;
    const FROM: [&str; 13] = [
        "norm1.",
        "self_attn.q.",
        "self_attn.k.",
        "self_attn.v.",
        "self_attn.o.",
        "norm2.",
        "cross_attn.q.",
        "cross_attn.k.",
        "cross_attn.v.",
        "cross_attn.o.",
        "norm3.",
        "ffn.up.",
        "ffn.down.",
    ];
    let suffix = decoder_suffix(part, &FROM)?;
    Some(Name::To(format!("model.decoder.layers.{layer}.{suffix}")))
}

/// The Hugging Face suffixes of a Canary-plan decoder layer. Canary and
/// Cohere name the same 13 roles with different source tokens.
pub(super) const DECODER_TO: [&str; 13] = [
    "input_layernorm.",
    "self_attn.q_proj.",
    "self_attn.k_proj.",
    "self_attn.v_proj.",
    "self_attn.o_proj.",
    "post_attention_layernorm.",
    "encoder_attn.q_proj.",
    "encoder_attn.k_proj.",
    "encoder_attn.v_proj.",
    "encoder_attn.o_proj.",
    "final_layernorm.",
    "mlp.fc1.",
    "mlp.fc2.",
];

pub(super) fn decoder_suffix(part: &str, from: &[&str; 13]) -> Option<String> {
    let index = from.iter().position(|from| part.starts_with(from))?;
    let rest = &part[from[index].len()..];
    Some(format!("{}{rest}", DECODER_TO[index]))
}
