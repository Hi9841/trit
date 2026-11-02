//! Voxtral Realtime GGUF files (`general.architecture` `voxtral_realtime`,
//! from transcribe.cpp's `scripts/convert-voxtral_realtime.py`): the causal
//! audio encoder (`audio_tower`), the projector, and the Ministral language
//! model with its delay-conditioned `ada` layers (`language_model`).

use super::{qwen3_asr, Gguf, Name};
use serde_json::{json, Value};

pub fn config(g: &Gguf) -> Result<Value, String> {
    let v = |key: &str| g.num(&format!("stt.voxtral_realtime.{key}"));
    let vf = |key: &str| g.get(&format!("stt.voxtral_realtime.{key}")).as_f64();
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    if text("stt.voxtral_realtime.decoder.hidden_act") != "silu"
        || text("stt.voxtral_realtime.encoder.hidden_act") != "silu"
        || text("stt.voxtral_realtime.projector.hidden_act") != "gelu"
        || v("encoder.n_kv_heads")? != v("encoder.n_heads")?
    {
        return Err("it differs from Voxtral Realtime's (SwiGLU encoder and decoder)".into());
    }
    if text("stt.frontend.type") != "mel" || text("stt.frontend.normalize") != "global" {
        return Err("its audio features differ from Voxtral Realtime's (fixed peak)".into());
    }
    Ok(json!({
        "model_type": "voxtral_realtime",
        "bos_token_id": g.num("tokenizer.ggml.bos_token_id")?,
        "eos_token_id": g.num("tokenizer.ggml.eos_token_id")?,
        "streaming_pad_token_id": v("streaming_pad_token_id")?,
        "encoder_config": {
            "hidden_size": v("encoder.d_model")?,
            "num_hidden_layers": v("encoder.n_layers")?,
            "num_attention_heads": v("encoder.n_heads")?,
            "head_dim": v("encoder.head_dim")?,
            "intermediate_size": v("encoder.ffn_dim")?,
            "num_mel_bins": v("encoder.num_mel_bins")?,
            "sliding_window": v("encoder.sliding_window")?,
            "rope_theta": vf("encoder.rope_theta"),
            "rms_norm_eps": vf("encoder.rms_norm_eps"),
        },
        "projector_config": {
            "downsample_factor": v("projector.downsample_factor")?,
            "audio_length_per_tok": v("projector.audio_length_per_tok")?,
        },
        "llm_config": {
            "model_type": "mistral",
            "hidden_size": v("decoder.hidden_size")?,
            "intermediate_size": v("decoder.intermediate_size")?,
            "num_hidden_layers": v("decoder.n_layers")?,
            "num_attention_heads": v("decoder.n_heads")?,
            "num_key_value_heads": v("decoder.n_kv_heads")?,
            "head_dim": v("decoder.head_dim")?,
            "vocab_size": v("decoder.vocab_size")?,
            "rms_norm_eps": vf("decoder.rms_norm_eps"),
            "rope_theta": vf("decoder.rope_theta"),
            "tie_word_embeddings": g.get("stt.voxtral_realtime.decoder.tie_word_embeddings").as_bool().unwrap_or(true),
            "ada_hidden": v("time.ada_hidden")?,
        },
        "feature_extractor": {
            "n_fft": g.num("stt.frontend.n_fft")?,
            "win_length": g.num("stt.frontend.win_length")?,
            "hop_length": g.num("stt.frontend.hop_length")?,
            "num_mel_bins": g.num("stt.frontend.num_mels")?,
            "log_mel_max": g.get("stt.frontend.global_log_mel_max").as_f64(),
        },
    }))
}

/// A transcribe.cpp Voxtral Realtime tensor name as Trit names it.
pub fn hf_name(name: &str) -> Option<Name> {
    if name.starts_with("frontend.") {
        return Some(Name::To(name.to_string()));
    }
    const TOP: [(&str, &str); 5] = [
        ("proj.", "multi_modal_projector."),
        ("dec.time_embed.", "time_embed."),
        ("enc.conv.0.", "audio_tower.conv1."),
        ("enc.conv.1.", "audio_tower.conv2."),
        ("enc.final_norm.", "audio_tower.norm."),
    ];
    for (from, to) in TOP {
        if let Some(rest) = name.strip_prefix(from) {
            return Some(Name::To(format!("{to}{rest}")));
        }
    }
    if let Some((layer, part)) = name
        .strip_prefix("dec.blocks.")
        .and_then(|rest| rest.split_once('.'))
    {
        if let Some(rest) = part.strip_prefix("ada.") {
            return Some(Name::To(format!(
                "language_model.model.layers.{layer}.ada.{rest}"
            )));
        }
    }
    if name.starts_with("dec.") {
        return qwen3_asr::language_model_name(name, "language_model.model");
    }
    let (layer, part) = name.strip_prefix("enc.blocks.")?.split_once('.')?;
    const BLOCK: [(&str, &str); 9] = [
        ("norm_attn.", "input_layernorm."),
        ("attn.q.", "self_attn.q_proj."),
        ("attn.k.", "self_attn.k_proj."),
        ("attn.v.", "self_attn.v_proj."),
        ("attn.out.", "self_attn.o_proj."),
        ("norm_ffn.", "post_attention_layernorm."),
        ("ffn.gate.", "mlp.gate_proj."),
        ("ffn.up.", "mlp.up_proj."),
        ("ffn.down.", "mlp.down_proj."),
    ];
    let (from, to) = BLOCK.iter().find(|(from, _)| part.starts_with(from))?;
    Some(Name::To(format!(
        "audio_tower.layers.{layer}.{to}{}",
        &part[from.len()..]
    )))
}
