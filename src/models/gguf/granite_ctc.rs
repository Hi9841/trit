//! Granite Speech 5.0 TurboCTC GGUF files (`general.architecture`
//! `granite_speech5_ctc`, from transcribe.cpp's
//! `scripts/convert-granite5_ctc.py`): the encoder config, the front end,
//! and the tensor names (transcribe.cpp's, under `encoder.`). The file's
//! mel filterbank and window feed the front end as published.

use super::{Gguf, Name};
use serde_json::{json, Value};

pub fn config(g: &Gguf) -> Result<Value, String> {
    let e = |key: &str| g.num(&format!("stt.granite5_ctc.encoder.{key}"));
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    let fe = |key: &str| g.num(&format!("stt.frontend.{key}"));
    if text("stt.granite5_ctc.encoder.hidden_act") != "silu"
        || g.get("stt.granite5_ctc.encoder.attention_bias").as_bool() == Some(false)
    {
        return Err(
            "its encoder differs from Granite Speech 5.0's (SiLU, attention output bias)".into(),
        );
    }
    if text("stt.frontend.type") != "mel"
        || text("stt.frontend.normalize") != "per_utterance"
        || text("stt.frontend.pad_mode") != "reflect"
        || g.get("stt.frontend.deltas").as_bool() != Some(true)
        || fe("delta_win_length")? != 3
        || fe("sample_rate")? != 16_000
    {
        return Err("its audio features differ from Granite Speech 5.0's".into());
    }
    Ok(json!({
        "model_type": "granite_ctc",
        "vocab_size": e("output_dim")?,
        "blank_token_id": g.num("stt.granite5_ctc.blank_id")?,
        "encoder_config": {
            "hidden_size": e("hidden")?,
            "num_attention_heads": e("n_heads")?,
            "head_dim": e("head_dim")?,
            "intermediate_size": e("intermediate")?,
            "num_hidden_layers": e("n_layers")?,
            "input_dim": e("input_dim")?,
            "output_dim": e("output_dim")?,
            "conv_kernel_size": e("conv_kernel_size")?,
            "conv_expansion_factor": e("conv_expansion")?,
            "context_size": e("context_size")?,
            "max_pos_emb": e("max_pos_emb")?,
            "self_conditioning_layer": e("self_cond_layer")?,
            "subsample_layers": g.get("stt.granite5_ctc.encoder.subsample_layers"),
        },
        "feature_extractor": super::blocks::granite_front(
            g,
            fe("stack_factor")?,
            g.get("stt.frontend.logmel_floor_db").as_f64().unwrap_or(8.0),
            true,
            "whole_stacks",
        )?,
    }))
}

/// A transcribe.cpp Granite TurboCTC tensor name as Trit names it.
pub fn hf_name(name: &str) -> Option<Name> {
    if name.starts_with("frontend.") {
        return Some(Name::To(name.to_string()));
    }
    name.strip_prefix("enc.")
        .map(|rest| Name::To(format!("encoder.{rest}")))
}
