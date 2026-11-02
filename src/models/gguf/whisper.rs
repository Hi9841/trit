//! Whisper GGUF files (`general.architecture` `whisper`, from
//! transcribe.cpp's `scripts/convert-whisper.py`): the `config.json` and
//! `generation_config.json` of the Hugging Face checkpoint, and its tensor
//! names. Trit computes the mel filterbank and window itself, as for a
//! Hugging Face folder, so the file's copies are skipped.

use super::{Gguf, Name};
use serde_json::{json, Map, Value};

pub fn config(g: &Gguf) -> Result<Value, String> {
    let w = |key: &str| g.num(&format!("stt.whisper.{key}"));
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    let d = w("encoder.d_model")?;
    if w("decoder.d_model")? != d {
        return Err("the encoder and decoder differ in width; Trit runs one width".into());
    }
    if text("stt.frontend.normalize") != "whisper_logmel"
        || g.num("stt.frontend.n_fft")? != 400
        || g.num("stt.frontend.hop_length")? != 160
        || g.num("stt.frontend.sample_rate")? != 16_000
    {
        return Err("its audio features differ from Whisper's".into());
    }
    for (key, want) in [
        ("stt.whisper.encoder.activation", "gelu"),
        ("stt.whisper.decoder.activation", "gelu"),
    ] {
        if !matches!(text(key), "" | "gelu") {
            return Err(format!("{key} is not {want}"));
        }
    }
    Ok(json!({
        "model_type": "whisper",
        "d_model": d,
        "encoder_layers": w("encoder.n_layers")?,
        "decoder_layers": w("decoder.n_layers")?,
        "encoder_attention_heads": w("encoder.n_heads")?,
        "decoder_attention_heads": w("decoder.n_heads")?,
        "encoder_ffn_dim": w("encoder.ffn_dim")?,
        "decoder_ffn_dim": w("decoder.ffn_dim")?,
        "num_mel_bins": w("encoder.num_mel_bins")?,
        "vocab_size": w("decoder.vocab_size")?,
        "max_source_positions": w("encoder.max_source_positions")?,
        "max_target_positions": w("decoder.max_target_positions")?,
        "scale_embedding": g.get("stt.whisper.decoder.scale_embedding").as_bool().unwrap_or(false),
        "activation_function": "gelu",
    }))
}

/// The decoding rules of `generation_config.json`: the special tokens, the
/// languages, and the suppressed tokens.
pub fn generation(g: &Gguf) -> Result<Value, String> {
    let languages = g
        .get("general.languages")
        .as_array()
        .cloned()
        .unwrap_or_default();
    let mut lang_to_id = Map::new();
    for code in languages.iter().filter_map(Value::as_str) {
        let tag = format!("<|{code}|>");
        if let Some(id) = g.token(&tag) {
            lang_to_id.insert(tag, json!(id));
        }
    }
    Ok(json!({
        "decoder_start_token_id": g.num("stt.whisper.decoder_start_token_id")?,
        "eos_token_id": g.num("tokenizer.ggml.eos_token_id")?,
        "no_timestamps_token_id": g.num("stt.whisper.no_timestamps_token_id")?,
        "is_multilingual": languages.len() > 1,
        "lang_to_id": lang_to_id,
        "suppress_tokens": g.get("stt.whisper.suppress_tokens"),
        "begin_suppress_tokens": g.get("stt.whisper.begin_suppress_tokens"),
    }))
}

/// A transcribe.cpp Whisper tensor name as the Hugging Face checkpoint names it.
pub fn hf_name(name: &str) -> Option<Name> {
    if name.starts_with("frontend.") {
        return Some(Name::Skip);
    }
    const TOP: [(&str, &str); 7] = [
        ("enc.conv.0.", "model.encoder.conv1."),
        ("enc.conv.1.", "model.encoder.conv2."),
        ("enc.pos_emb.", "model.encoder.embed_positions."),
        ("enc.final_norm.", "model.encoder.layer_norm."),
        ("dec.token_embd.", "model.decoder.embed_tokens."),
        ("dec.pos_emb.", "model.decoder.embed_positions."),
        ("dec.final_norm.", "model.decoder.layer_norm."),
    ];
    for (from, to) in TOP {
        if let Some(rest) = name.strip_prefix(from) {
            return Some(Name::To(format!("{to}{rest}")));
        }
    }
    let (side, rest) = if let Some(rest) = name.strip_prefix("enc.blocks.") {
        ("encoder", rest)
    } else {
        ("decoder", name.strip_prefix("dec.blocks.")?)
    };
    let (layer, part) = rest.split_once('.')?;
    const BLOCK: [(&str, &str); 16] = [
        ("norm_attn.", "self_attn_layer_norm."),
        ("norm_self.", "self_attn_layer_norm."),
        ("attn.q.", "self_attn.q_proj."),
        ("attn.k.", "self_attn.k_proj."),
        ("attn.v.", "self_attn.v_proj."),
        ("attn.out.", "self_attn.out_proj."),
        ("self_attn.q.", "self_attn.q_proj."),
        ("self_attn.k.", "self_attn.k_proj."),
        ("self_attn.v.", "self_attn.v_proj."),
        ("self_attn.out.", "self_attn.out_proj."),
        ("norm_cross.", "encoder_attn_layer_norm."),
        ("cross_attn.q.", "encoder_attn.q_proj."),
        ("cross_attn.k.", "encoder_attn.k_proj."),
        ("cross_attn.v.", "encoder_attn.v_proj."),
        ("cross_attn.out.", "encoder_attn.out_proj."),
        ("norm_ffn.", "final_layer_norm."),
    ];
    let (from, to) = BLOCK
        .iter()
        .copied()
        .chain([("ffn.fc1.", "fc1."), ("ffn.fc2.", "fc2.")])
        .find(|(from, _)| part.starts_with(from))?;
    Some(Name::To(format!(
        "model.{side}.layers.{layer}.{to}{}",
        &part[from.len()..]
    )))
}
