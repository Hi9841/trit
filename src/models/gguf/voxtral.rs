//! Voxtral GGUF files (`general.architecture` `voxtral`, from transcribe.cpp's
//! `scripts/convert-voxtral.py`): a Whisper large encoder, a projector that
//! joins 4 frames, and a Mistral language model. The encoder takes Whisper's
//! Hugging Face names, the projector `multi_modal_projector`, the language
//! model the Llama names under `language_model`.

use super::{qwen3_asr, whisper, Gguf, Name};
use serde_json::{json, Value};

pub fn config(g: &Gguf) -> Result<Value, String> {
    let e = |key: &str| g.num(&format!("stt.voxtral.encoder.{key}"));
    let q = |key: &str| g.num(&format!("stt.voxtral.decoder.{key}"));
    let qf = |key: &str| g.get(&format!("stt.voxtral.decoder.{key}")).as_f64();
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    if text("stt.voxtral.decoder.hidden_act") != "silu"
        || text("stt.voxtral.encoder.activation") != "gelu"
        || text("stt.voxtral.projector.hidden_act") != "gelu"
    {
        return Err("it differs from Voxtral's (GELU encoder and projector, SiLU decoder)".into());
    }
    if text("stt.frontend.type") != "mel" || text("stt.frontend.normalize") != "per_utterance" {
        return Err("its audio features differ from Whisper's".into());
    }
    Ok(json!({
        "model_type": "voxtral",
        "audio_token_id": g.num("stt.voxtral.audio_token_id")?,
        "bos_token_id": g.num("tokenizer.ggml.bos_token_id")?,
        "eos_token_id": g.num("tokenizer.ggml.eos_token_id")?,
        "languages": g.get("general.languages"),
        "encoder_config": {
            "hidden_size": e("d_model")?,
            "num_hidden_layers": e("n_layers")?,
            "num_attention_heads": e("n_heads")?,
            "intermediate_size": e("ffn_dim")?,
            "num_mel_bins": e("num_mel_bins")?,
            "max_source_positions": e("max_source_positions")?,
        },
        "projector_config": {
            "downsample_factor": g.num("stt.voxtral.projector.downsample_factor")?,
            "input_dim": g.num("stt.voxtral.projector.input_dim")?,
        },
        "llm_config": {
            "model_type": "mistral",
            "hidden_size": q("hidden_size")?,
            "intermediate_size": q("intermediate_size")?,
            "num_hidden_layers": q("n_layers")?,
            "num_attention_heads": q("n_heads")?,
            "num_key_value_heads": q("n_kv_heads")?,
            "head_dim": q("head_dim")?,
            "vocab_size": q("vocab_size")?,
            "rms_norm_eps": qf("rms_norm_eps"),
            "rope_theta": qf("rope_theta"),
            "tie_word_embeddings": g.get("stt.voxtral.decoder.tie_word_embeddings").as_bool().unwrap_or(false),
        },
    }))
}

/// A transcribe.cpp Voxtral tensor name as Trit names it.
pub fn hf_name(name: &str) -> Option<Name> {
    if let Some(rest) = name.strip_prefix("proj.") {
        return Some(Name::To(format!("multi_modal_projector.{rest}")));
    }
    if let Some(rest) = name.strip_prefix("dec.output.") {
        return Some(Name::To(format!("language_model.lm_head.{rest}")));
    }
    if name.starts_with("dec.") {
        return qwen3_asr::language_model_name(name, "language_model.model");
    }
    // Whisper's encoder; its final norm is `ln_post` here.
    whisper::hf_name(&name.replace("enc.ln_post.", "enc.final_norm."))
}
