//! Canary GGUF files (`general.architecture` `canary`, from transcribe.cpp's
//! `scripts/convert-canary.py`): the `config.json` of the Hugging Face
//! Canary checkpoint (plus the languages, the prompt format, and the
//! encoder width, which Canary 1B v2's folder does not need), and the
//! tensor names. The encoder is Parakeet's FastConformer, so its names
//! come from the Parakeet table with `model.` before them.

use super::{parakeet, Gguf, Name};
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
        "encoder_config": {
            "hidden_size": c("encoder.d_model")?,
            "intermediate_size": c("encoder.d_ff")?,
            "num_attention_heads": c("encoder.n_heads")?,
            "num_hidden_layers": c("encoder.n_layers")?,
            "num_mel_bins": g.num("stt.frontend.num_mels")?,
            "conv_kernel_size": c("encoder.conv_kernel")?,
            "subsampling_factor": c("encoder.subsampling_factor")?,
            "subsampling_conv_channels": c("encoder.subsampling_channels")?,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "scale_input": g.get("stt.canary.encoder.xscaling").as_bool().unwrap_or(false),
        },
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
        return parakeet::hf_name(name).map(|hf| Name::To(format!("model.{hf}")));
    }
    const TOP: [(&str, &str); 4] = [
        ("dec.embed.token.", "model.decoder.embed_tokens."),
        ("dec.embed.norm.", "model.decoder.embedding_layernorm."),
        ("dec.norm.", "model.decoder.norm."),
        ("dec.head.", "proj_out."),
    ];
    for (from, to) in TOP {
        if let Some(rest) = name.strip_prefix(from) {
            return Some(Name::To(format!("{to}{rest}")));
        }
    }
    let (layer, part) = name.strip_prefix("dec.layer.")?.split_once('.')?;
    const LAYER: [(&str, &str); 13] = [
        ("norm1.", "input_layernorm."),
        ("self_attn.q.", "self_attn.q_proj."),
        ("self_attn.k.", "self_attn.k_proj."),
        ("self_attn.v.", "self_attn.v_proj."),
        ("self_attn.o.", "self_attn.o_proj."),
        ("norm2.", "post_attention_layernorm."),
        ("cross_attn.q.", "encoder_attn.q_proj."),
        ("cross_attn.k.", "encoder_attn.k_proj."),
        ("cross_attn.v.", "encoder_attn.v_proj."),
        ("cross_attn.o.", "encoder_attn.o_proj."),
        ("norm3.", "final_layernorm."),
        ("ffn.up.", "mlp.fc1."),
        ("ffn.down.", "mlp.fc2."),
    ];
    let (from, to) = LAYER.iter().find(|(from, _)| part.starts_with(from))?;
    Some(Name::To(format!(
        "model.decoder.layers.{layer}.{to}{}",
        &part[from.len()..]
    )))
}
