//! Qwen3-ASR GGUF files (`general.architecture` `qwen3_asr`, from
//! transcribe.cpp's `scripts/convert-qwen3_asr.py`): the `config.json` of
//! the Hugging Face checkpoint and its `thinker.*` tensor names. The
//! language model's output head is its token table (tied), as in the
//! checkpoint. Trit computes the mel filterbank and window itself.

use super::blocks::{llm_block, LlmShape};
use super::{Gguf, Name};
use crate::models::families::qwen3_asr::LANGUAGE_NAMES;
use crate::models::parts::causal_lm::CausalLm;
use serde_json::{json, Value};

pub fn config(g: &Gguf) -> Result<Value, String> {
    let q = |key: &str| g.num(&format!("stt.qwen3_asr.{key}"));
    let float = |key: &str| g.get(&format!("stt.qwen3_asr.{key}")).as_f64();
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    if text("stt.qwen3_asr.decoder.hidden_act") != "silu"
        || g.get("stt.qwen3_asr.decoder.tie_word_embeddings").as_bool() == Some(false)
    {
        return Err("its language model differs from Qwen3-ASR's (SiLU, tied head)".into());
    }
    if text("stt.frontend.type") != "mel" || !super::blocks::whisper_mel(g)? {
        return Err("its audio features differ from Whisper's".into());
    }
    // The prompt names a language in English ("English"); the file gives codes.
    let languages: Vec<&str> = g
        .get("general.languages")
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter_map(|code| {
            LANGUAGE_NAMES
                .iter()
                .find(|(known, _)| *known == code)
                .map(|(_, name)| *name)
        })
        .collect();
    Ok(json!({
        "model_type": "qwen3_asr",
        "support_languages": languages,
        "thinker_config": {
            "audio_token_id": q("audio_token_id")?,
            "audio_start_token_id": q("audio_start_token_id")?,
            "audio_end_token_id": q("audio_end_token_id")?,
            "audio_config": {
                "d_model": q("encoder.d_model")?,
                "encoder_attention_heads": q("encoder.n_heads")?,
                "encoder_ffn_dim": q("encoder.ffn_dim")?,
                "encoder_layers": q("encoder.n_layers")?,
                "num_mel_bins": q("encoder.num_mel_bins")?,
                "downsample_hidden_size": q("encoder.downsample_hidden")?,
                "output_dim": q("encoder.output_dim")?,
                "max_source_positions": q("encoder.max_source_positions")?,
                "n_window": q("encoder.n_window")?,
                "n_window_infer": q("encoder.n_window_infer")?,
                "activation_function": text("stt.qwen3_asr.encoder.activation"),
            },
            "text_config": llm_block(LlmShape {
                model_type: "qwen3",
                hidden: q("decoder.hidden_size")?,
                intermediate: q("decoder.intermediate_size")?,
                layers: q("decoder.n_layers")?,
                heads: q("decoder.n_heads")?,
                kv_heads: q("decoder.n_kv_heads")?,
                head_dim: q("decoder.head_dim")?,
                vocab: q("decoder.vocab_size")?,
                rms_norm_eps: float("decoder.rms_norm_eps"),
                rope_theta: float("decoder.rope_theta"),
                tie: true,
            }),
        },
    }))
}

/// A transcribe.cpp Qwen3-ASR tensor name as the Hugging Face checkpoint names it.
pub fn hf_name(name: &str) -> Option<Name> {
    if name.starts_with("frontend.") {
        return Some(Name::Skip);
    }
    if name.starts_with("dec.") {
        return language_model_name(name, "thinker.");
    }
    const TOP: [(&str, &str); 7] = [
        ("enc.conv.0.", "thinker.audio_tower.conv2d1."),
        ("enc.conv.1.", "thinker.audio_tower.conv2d2."),
        ("enc.conv.2.", "thinker.audio_tower.conv2d3."),
        ("enc.conv_out.", "thinker.audio_tower.conv_out."),
        ("enc.ln_post.", "thinker.audio_tower.ln_post."),
        ("enc.proj1.", "thinker.audio_tower.proj1."),
        ("enc.proj2.", "thinker.audio_tower.proj2."),
    ];
    if let Some(mapped) = super::map_suffix(name, &TOP) {
        return Some(Name::To(mapped));
    }
    const ENCODER: [(&str, &str); 8] = [
        ("norm_attn.", "self_attn_layer_norm."),
        ("attn.q.", "self_attn.q_proj."),
        ("attn.k.", "self_attn.k_proj."),
        ("attn.v.", "self_attn.v_proj."),
        ("attn.out.", "self_attn.out_proj."),
        ("norm_ffn.", "final_layer_norm."),
        ("ffn.fc1.", "fc1."),
        ("ffn.fc2.", "fc2."),
    ];
    let (layer, part) = name.strip_prefix("enc.blocks.")?.split_once('.')?;
    let mapped = super::map_suffix(part, &ENCODER)?;
    Some(Name::To(format!(
        "thinker.audio_tower.layers.{layer}.{mapped}"
    )))
}

/// A transcribe.cpp Qwen3 language-model name (`dec.*`) under `prefix`
/// (`llm.`, `language_model.`, or `thinker.`). `CausalLm::model_tensor` adds
/// the `model.` segment.
pub fn language_model_name(name: &str, prefix: &str) -> Option<Name> {
    if let Some(mapped) = super::map_suffix(
        name,
        &[
            ("dec.token_embd.", "embed_tokens."),
            ("dec.output_norm.", "norm."),
        ],
    ) {
        return Some(Name::To(CausalLm::model_tensor(prefix, &mapped)));
    }
    const DECODER: [(&str, &str); 11] = [
        ("norm_attn.", "input_layernorm."),
        ("norm_ffn.", "post_attention_layernorm."),
        ("attn.q_norm.", "self_attn.q_norm."),
        ("attn.k_norm.", "self_attn.k_norm."),
        ("attn.q.", "self_attn.q_proj."),
        ("attn.k.", "self_attn.k_proj."),
        ("attn.v.", "self_attn.v_proj."),
        ("attn.o.", "self_attn.o_proj."),
        ("ffn.gate.", "mlp.gate_proj."),
        ("ffn.up.", "mlp.up_proj."),
        ("ffn.down.", "mlp.down_proj."),
    ];
    // Fun-ASR files say `dec.layers`, the others `dec.blocks`.
    let (layer, part) = name
        .strip_prefix("dec.blocks.")
        .or_else(|| name.strip_prefix("dec.layers."))?
        .split_once('.')?;
    let mapped = super::map_suffix(part, &DECODER)?;
    Some(Name::To(CausalLm::model_tensor(
        prefix,
        &format!("layers.{layer}.{mapped}"),
    )))
}
