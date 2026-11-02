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

/// Moonshine Streaming files (`general.architecture` `moonshine_streaming`):
/// Moonshine's decoder keys, and the streaming encoder under
/// `streaming_encoder`.
pub fn streaming_config(g: &Gguf) -> Result<Value, String> {
    let m = |key: &str| g.num(&format!("stt.moonshine_streaming.{key}"));
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    let heads = m("decoder.n_heads")?;
    if text("stt.moonshine_streaming.decoder.activation") != "silu"
        || text("stt.moonshine_streaming.encoder.activation") != "gelu"
        || g.get("stt.moonshine_streaming.attention_bias").as_bool() == Some(true)
        || m("pad_head_dim_to_multiple_of").unwrap_or(0) != 0
        || g.get("stt.moonshine_streaming.encoder_layernorm_unit_offset")
            .as_bool()
            != Some(true)
    {
        return Err(
            "it differs from Moonshine Streaming's (SiLU decoder, GELU encoder, no bias)".into(),
        );
    }
    let windows: Vec<Value> = g
        .get("stt.moonshine_streaming.encoder.sliding_windows")
        .as_array()
        .map(|flat| flat.chunks(2).map(|pair| json!(pair)).collect())
        .unwrap_or_default();
    Ok(json!({
        "model_type": "moonshine_streaming",
        "hidden_size": m("decoder.d_model")?,
        "intermediate_size": m("decoder.ffn_dim")?,
        "encoder_num_attention_heads": heads,
        "decoder_num_attention_heads": heads,
        "encoder_num_key_value_heads": heads,
        "decoder_num_key_value_heads": m("decoder.n_kv_heads")?,
        "encoder_num_hidden_layers": m("encoder.n_layers")?,
        "decoder_num_hidden_layers": m("decoder.n_layers")?,
        "vocab_size": m("decoder.vocab_size")?,
        "max_position_embeddings": m("decoder.max_position_embeddings")?,
        "decoder_start_token_id": m("decoder_start_token_id")?,
        "eos_token_id": m("eos_token_id")?,
        "partial_rotary_factor": g.get("stt.moonshine_streaming.partial_rotary_factor").as_f64().unwrap_or(1.0),
        "rope_theta": g.get("stt.moonshine_streaming.rope_theta").as_f64().unwrap_or(10_000.0),
        "tie_word_embeddings": g.get("stt.moonshine_streaming.decoder.tie_word_embeddings").as_bool().unwrap_or(true),
        "languages": g.get("general.languages"),
        "streaming_encoder": {
            "hidden_size": m("encoder.d_model")?,
            "num_attention_heads": m("encoder.n_heads")?,
            "head_dim": m("encoder.head_dim")?,
            "intermediate_size": m("encoder.ffn_dim")?,
            "num_hidden_layers": m("encoder.n_layers")?,
            "frame_len": m("encoder.frame_len")?,
            "sliding_windows": windows,
            "cmvn_eps": g.get("stt.moonshine_streaming.cmvn_eps").as_f64(),
            "adapter_projection": g.get("stt.moonshine_streaming.adapter_has_proj").as_bool().unwrap_or(false),
            "max_position_embeddings": m("decoder.max_position_embeddings")?,
        },
    }))
}

/// A transcribe.cpp Moonshine tensor name as the Hugging Face checkpoint names it.
pub fn hf_name(name: &str) -> Option<Name> {
    const TOP: [(&str, &str); 10] = [
        ("enc.embedder.", "model.encoder.embedder."),
        ("adapter.", "model.adapter."),
        ("dec.lm_head.", "proj_out."),
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
