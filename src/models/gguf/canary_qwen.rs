//! Canary-Qwen GGUF files (`general.architecture` `canary_qwen`, from
//! transcribe.cpp's `scripts/convert-canary-qwen.py`): NVIDIA's SALM, the
//! Canary 1B flash FastConformer, a projection to the language model's
//! width, and Qwen3 1.7B (its LoRA merged at conversion). The encoder takes
//! Canary's names, the language model the Hugging Face Qwen3 names under
//! `llm.model`.

use super::blocks::{fastconformer, llm};
use super::{canary, qwen3_asr, Gguf, Name};
use serde_json::{json, Value};

pub fn config(g: &Gguf) -> Result<Value, String> {
    let q = |key: &str| g.num(&format!("stt.canary_qwen.decoder.{key}"));
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    if text("stt.canary_qwen.decoder.hidden_act") != "silu"
        || g.get("stt.canary_qwen.decoder.tie_word_embeddings")
            .as_bool()
            == Some(false)
    {
        return Err("its language model differs from Qwen3's (SiLU, tied head)".into());
    }
    if text("stt.frontend.type") != "mel" || text("stt.frontend.normalize") != "per_feature" {
        return Err("its audio features differ from NeMo's".into());
    }
    let vocab = q("vocab_size")?;
    Ok(json!({
        "model_type": "canary_qwen",
        // The FastConformer reads these; the encoder has no head of its own.
        "vocab_size": vocab,
        "pad_token_id": 0,
        "audio_locator_id": g.num("stt.canary_qwen.perception.audio_locator_id")?,
        "projection_size": g.num("stt.canary_qwen.perception.output_dim")?,
        "encoder_config": fastconformer(
            g,
            "stt.canary_qwen.encoder",
            g.num("stt.frontend.num_mels")?,
            false,
        )?,
        "llm_config": llm(g, "stt.canary_qwen.decoder", "qwen3", true)?,
    }))
}

/// A transcribe.cpp Canary-Qwen tensor name as Trit names it.
pub fn hf_name(name: &str) -> Option<Name> {
    if name.starts_with("frontend.") {
        // Trit computes NeMo's filterbank and window itself.
        return Some(Name::Skip);
    }
    if name.starts_with("dec.") {
        return qwen3_asr::language_model_name(name, "llm.");
    }
    canary::hf_name(name)
}
