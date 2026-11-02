//! Moonshine GGUF files (`general.architecture` `moonshine`, from
//! transcribe.cpp's `scripts/convert-moonshine.py`): the `config.json` of
//! the Hugging Face checkpoint, and its tensor names.

use super::{Gguf, Name};
use serde_json::{json, Value};

pub fn config(g: &Gguf) -> Result<Value, String> {
    let m = |key: &str| g.num(&format!("stt.moonshine.{key}"));
    let d = m("encoder.d_model")?;
    let ffn = m("encoder.ffn_dim")?;
    if m("decoder.d_model")? != d || m("decoder.ffn_dim")? != ffn {
        return Err("the encoder and decoder differ in shape; Trit runs one width".into());
    }
    if g.get("stt.frontend.type").as_str() != Some("raw")
        || g.num("stt.frontend.sample_rate")? != 16_000
    {
        return Err("its audio input differs from Moonshine's (raw 16 kHz)".into());
    }
    Ok(json!({
        "model_type": "moonshine",
        "hidden_size": d,
        "intermediate_size": ffn,
        "encoder_num_attention_heads": m("encoder.n_heads")?,
        "decoder_num_attention_heads": m("decoder.n_heads")?,
        "encoder_num_key_value_heads": m("encoder.n_kv_heads")?,
        "decoder_num_key_value_heads": m("decoder.n_kv_heads")?,
        "encoder_num_hidden_layers": m("encoder.n_layers")?,
        "decoder_num_hidden_layers": m("decoder.n_layers")?,
        "vocab_size": m("decoder.vocab_size")?,
        "max_position_embeddings": m("decoder.max_position_embeddings")?,
        "decoder_start_token_id": m("decoder_start_token_id")?,
        "eos_token_id": m("eos_token_id")?,
        "partial_rotary_factor": g.get("stt.moonshine.partial_rotary_factor").as_f64().unwrap_or(1.0),
        "rope_theta": g.get("stt.moonshine.rope_theta").as_f64().unwrap_or(10_000.0),
        "encoder_hidden_act": g.get("stt.moonshine.encoder.activation").as_str().unwrap_or("gelu"),
        "languages": g.get("general.languages"),
    }))
}

/// A transcribe.cpp Moonshine tensor name as the Hugging Face checkpoint names it.
pub fn hf_name(name: &str) -> Option<Name> {
    const TOP: [(&str, &str); 7] = [
        ("enc.conv.0.", "model.encoder.conv1."),
        ("enc.conv.1.", "model.encoder.conv2."),
        ("enc.conv.2.", "model.encoder.conv3."),
        ("enc.conv.norm.", "model.encoder.groupnorm."),
        ("enc.final_norm.", "model.encoder.layer_norm."),
        ("dec.token_embd.", "model.decoder.embed_tokens."),
        ("dec.final_norm.", "model.decoder.norm."),
    ];
    for (from, to) in TOP {
        if let Some(rest) = name.strip_prefix(from) {
            return Some(Name::To(format!("{to}{rest}")));
        }
    }
    // The same GGUF suffix names a different norm in the two stacks.
    const ENCODER: [(&str, &str); 7] = [
        ("norm_attn.", "input_layernorm."),
        ("attn.q.", "self_attn.q_proj."),
        ("attn.k.", "self_attn.k_proj."),
        ("attn.v.", "self_attn.v_proj."),
        ("attn.out.", "self_attn.o_proj."),
        ("norm_ffn.", "post_attention_layernorm."),
        ("ffn.", "mlp."),
    ];
    const DECODER: [(&str, &str); 11] = [
        ("norm_self.", "input_layernorm."),
        ("self_attn.q.", "self_attn.q_proj."),
        ("self_attn.k.", "self_attn.k_proj."),
        ("self_attn.v.", "self_attn.v_proj."),
        ("self_attn.out.", "self_attn.o_proj."),
        ("norm_cross.", "post_attention_layernorm."),
        ("cross_attn.q.", "encoder_attn.q_proj."),
        ("cross_attn.k.", "encoder_attn.k_proj."),
        ("cross_attn.v.", "encoder_attn.v_proj."),
        ("cross_attn.out.", "encoder_attn.o_proj."),
        ("norm_ffn.", "final_layernorm."),
    ];
    let (side, rest, table): (&str, &str, &[(&str, &str)]) =
        if let Some(rest) = name.strip_prefix("enc.blocks.") {
            ("encoder", rest, &ENCODER)
        } else {
            ("decoder", name.strip_prefix("dec.blocks.")?, &DECODER)
        };
    let (layer, part) = rest.split_once('.')?;
    let (from, to) = table
        .iter()
        .copied()
        .chain([("ffn.", "mlp.")])
        .find(|(from, _)| part.starts_with(from))?;
    Some(Name::To(format!(
        "model.{side}.layers.{layer}.{to}{}",
        &part[from.len()..]
    )))
}
