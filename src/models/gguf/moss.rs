//! MOSS-Transcribe-Diarize GGUF files (`general.architecture` `moss`, from
//! transcribe.cpp's `scripts/convert-moss.py`): a Whisper medium encoder, an
//! adaptor over 4 joined frames, and Qwen3 0.6B. The prompt's tokens are in
//! the file. The encoder takes Whisper's names, the adaptor `adaptor.`, the
//! language model the Qwen3 names under `language_model`.

use super::{qwen3_asr, whisper, Gguf, Name};
use serde_json::{json, Value};

pub fn config(g: &Gguf) -> Result<Value, String> {
    let m = |key: &str| g.num(&format!("stt.moss.{key}"));
    let mf = |key: &str| g.get(&format!("stt.moss.{key}")).as_f64();
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    if text("stt.moss.decoder.hidden_act") != "silu"
        || text("stt.moss.encoder.activation") != "gelu"
        || g.get("stt.moss.encoder.scale_embedding").as_bool() == Some(true)
    {
        return Err(
            "it differs from MOSS-Transcribe-Diarize's (GELU encoder, SiLU decoder)".into(),
        );
    }
    if text("stt.frontend.type") != "mel" || text("stt.frontend.normalize") != "per_utterance" {
        return Err("its audio features differ from Whisper's".into());
    }
    Ok(json!({
        "model_type": "moss",
        "eos_token_id": g.num("tokenizer.ggml.eos_token_id")?,
        "audio_token_id": m("audio_token_id")?,
        "audio_merge_size": m("audio_merge_size")?,
        "audio_tokens_per_second": mf("audio_tokens_per_second"),
        "time_marker_every_seconds": match g.get("stt.moss.enable_time_marker").as_bool() {
            Some(true) => m("time_marker_every_seconds")?,
            _ => 0,
        },
        "prompt_prefix_tokens": g.get("stt.moss.prompt_prefix_tokens"),
        "prompt_suffix_tokens": g.get("stt.moss.prompt_suffix_tokens"),
        "digit_tokens": g.get("stt.moss.digit_tokens"),
        "encoder_config": {
            "hidden_size": m("encoder.d_model")?,
            "num_hidden_layers": m("encoder.n_layers")?,
            "num_attention_heads": m("encoder.n_heads")?,
            "intermediate_size": m("encoder.ffn_dim")?,
            "num_mel_bins": m("encoder.num_mel_bins")?,
            "max_source_positions": m("encoder.max_source_positions")?,
        },
        "adaptor_config": {
            "input_dim": m("adaptor.input_dim")?,
        },
        "llm_config": {
            "model_type": "qwen3",
            "hidden_size": m("decoder.hidden_size")?,
            "intermediate_size": m("decoder.intermediate_size")?,
            "num_hidden_layers": m("decoder.n_layers")?,
            "num_attention_heads": m("decoder.n_heads")?,
            "num_key_value_heads": m("decoder.n_kv_heads")?,
            "head_dim": m("decoder.head_dim")?,
            "vocab_size": m("decoder.vocab_size")?,
            "rms_norm_eps": mf("decoder.rms_norm_eps"),
            "rope_theta": mf("decoder.rope_theta"),
            "tie_word_embeddings": g.get("stt.moss.decoder.tie_word_embeddings").as_bool().unwrap_or(true),
        },
    }))
}

/// A transcribe.cpp MOSS tensor name as Trit names it.
pub fn hf_name(name: &str) -> Option<Name> {
    if name.starts_with("frontend.") {
        // Trit computes Whisper's filterbank and window itself.
        return Some(Name::Skip);
    }
    if let Some(rest) = name.strip_prefix("adaptor.") {
        return Some(Name::To(format!("adaptor.{rest}")));
    }
    if name.starts_with("dec.") {
        return qwen3_asr::language_model_name(name, "language_model.model");
    }
    whisper::hf_name(name)
}
