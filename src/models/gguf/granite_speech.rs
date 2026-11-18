//! Granite Speech 4.0 and 4.1 GGUF files (`general.architecture`
//! `granite_speech`, from transcribe.cpp's `scripts/convert-granite.py`):
//! the Granite Conformer, a Q-Former projector, and a Granite language
//! model. The encoder takes the names of `parts::granite_conformer`, the
//! projector stays under `projector.`, and the language model takes the
//! Hugging Face Granite names under `language_model`.

use super::blocks::{granite_encoder, llm};
use super::{qwen3_asr, Gguf, Name};
use serde_json::{json, Value};

pub fn config(g: &Gguf) -> Result<Value, String> {
    let e = |key: &str| g.num(&format!("stt.granite.encoder.{key}"));
    let p = |key: &str| g.num(&format!("stt.granite.projector.{key}"));
    let qf = |key: &str| g.get(&format!("stt.granite.decoder.{key}")).as_f64();
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    if text("stt.granite.decoder.hidden_act") != "silu"
        || text("stt.granite.projector.hidden_act") != "gelu"
        || p("cross_attn_frequency")? != 1
        || text("stt.granite.projector.position_embedding_type") != "absolute"
    {
        return Err(
            "it differs from Granite Speech 4's (SiLU language model, GELU Q-Former)".into(),
        );
    }
    if text("stt.frontend.type") != "mel" || text("stt.frontend.normalize") != "per_utterance" {
        return Err("its audio features differ from Granite Speech's".into());
    }
    let d = e("hidden")?;
    let mut encoder = granite_encoder(g, "stt.granite.encoder", d, d * e("feedforward_mult")?)?;
    encoder["cat_hidden_layers"] = g.get("stt.granite.encoder.cat_hidden_layers").clone();
    let mut llm = llm(g, "stt.granite.decoder", "granite", false)?;
    llm["embedding_multiplier"] = json!(qf("embedding_multiplier"));
    llm["attention_multiplier"] = json!(qf("attention_multiplier"));
    llm["residual_multiplier"] = json!(qf("residual_multiplier"));
    llm["logits_scaling"] = json!(qf("logits_scaling"));
    let variant = text("stt.variant");
    // IBM's instruction for each release (transcribe.cpp's prompts).
    let (instruction, chat) = match variant {
        "granite-speech-4.1-2b" => (
            "transcribe the speech with proper punctuation and capitalization.",
            "plain",
        ),
        "granite-speech-4.1-2b-plus" => (
            " can you transcribe the speech into a written format?",
            "roles",
        ),
        _ => (
            "can you transcribe the speech into a written format?",
            "plain",
        ),
    };
    Ok(json!({
        "model_type": "granite_speech",
        "audio_token_id": g.num("stt.granite.audio_token_id")?,
        "eos_token_id": g.num("tokenizer.ggml.eos_token_id")?,
        "instruction": instruction,
        "chat": chat,
        // The system turn of the plus release's chat template.
        "system": "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant",
        "encoder_config": encoder,
        "projector_config": {
            "hidden_size": p("hidden")?,
            "intermediate_size": p("intermediate")?,
            "num_attention_heads": p("n_heads")?,
            "num_hidden_layers": p("n_layers")?,
            "encoder_hidden_size": p("encoder_hidden_size")?,
            "layer_norm_eps": g.get("stt.granite.projector.layer_norm_eps").as_f64(),
            "window_size": g.num("stt.granite.window_size")?,
            "downsample_rate": g.num("stt.granite.downsample_rate")?,
        },
        "llm_config": llm,
        "feature_extractor": front(g)?,
    }))
}

/// The mel front end of Granite Speech 4.x.
pub(super) fn front(g: &Gguf) -> Result<Value, String> {
    super::blocks::granite_front(g, 2, 8.0, false, "even")
}

/// A transcribe.cpp Granite Speech tensor name as Trit names it.
pub fn hf_name(name: &str) -> Option<Name> {
    if name.starts_with("frontend.") {
        return Some(Name::To(name.to_string()));
    }
    if let Some(rest) = name.strip_prefix("proj.") {
        return Some(Name::To(format!("projector.{rest}")));
    }
    if let Some(rest) = name.strip_prefix("dec.output.") {
        return Some(Name::To(format!("language_model.lm_head.{rest}")));
    }
    if name.starts_with("dec.") {
        return qwen3_asr::language_model_name(name, "language_model.");
    }
    // The block names of Granite Speech 5.0 (`parts::granite_conformer`).
    let rest = name.strip_prefix("enc.")?;
    let rest = super::replace_all(
        rest,
        &[
            (".ff1.up.", ".ff1.linear1."),
            (".ff1.down.", ".ff1.linear2."),
            (".ff2.up.", ".ff2.linear1."),
            (".ff2.down.", ".ff2.linear2."),
            (".norm_post.", ".norm_out."),
        ],
    );
    Some(Name::To(format!("encoder.{rest}")))
}
