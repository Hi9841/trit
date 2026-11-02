//! Granite Speech 4.1 2B NAR GGUF files (`general.architecture`
//! `granite_speech_nar`, from transcribe.cpp's `scripts/convert-granite_nar.py`):
//! the Granite Conformer with a pooled BPE CTC head, IBM's NLE projector, and
//! a Granite language model run as a bidirectional editor. The encoder takes
//! the names of `parts::granite_conformer`, the projector `nar_projector.`,
//! the language model the Granite names under `language_model`.

use super::{granite_speech, qwen3_asr, Gguf, Name};
use serde_json::{json, Value};

pub fn config(g: &Gguf) -> Result<Value, String> {
    let e = |key: &str| g.num(&format!("stt.granite_nar.encoder.{key}"));
    let p = |key: &str| g.num(&format!("stt.granite_nar.projector.{key}"));
    let q = |key: &str| g.num(&format!("stt.granite_nar.text.{key}"));
    let qf = |key: &str| g.get(&format!("stt.granite_nar.text.{key}")).as_f64();
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    if text("stt.granite_nar.text.hidden_act") != "silu" {
        return Err("its language model differs from Granite's (SiLU)".into());
    }
    if text("stt.frontend.type") != "mel" || text("stt.frontend.normalize") != "per_utterance" {
        return Err("its audio features differ from Granite Speech's".into());
    }
    let layers = e("n_layers")?;
    // `layer_indices` count hidden states from the input projection: block
    // `i - 1`; a negative one counts from the end.
    let captured: Vec<u64> = g
        .get("stt.granite_nar.encoder.layer_indices")
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_i64)
        .map(|i| match i {
            i if i > 0 => i as u64 - 1,
            i if i < 0 => (layers as i64 + i) as u64,
            _ => 0,
        })
        .collect();
    let d = e("hidden")?;
    Ok(json!({
        "model_type": "granite_speech_nar",
        "eos_token_id": q("eos_id")?,
        "bpe_blank_id": e("bpe_blank_id")?,
        "bpe_pool_window": e("bpe_pool_window")?,
        "bpe_vocab": e("bpe_output_dim")?,
        "scale_projected_embeddings": g.get("stt.granite_nar.scale_projected_embeddings").as_bool().unwrap_or(false),
        "encoder_config": {
            "hidden_size": d,
            "num_attention_heads": e("n_heads")?,
            "head_dim": e("head_dim")?,
            "intermediate_size": d * e("feedforward_mult")?,
            "num_hidden_layers": layers,
            "input_dim": e("input_dim")?,
            "output_dim": e("output_dim")?,
            "conv_kernel_size": e("conv_kernel_size")?,
            "conv_expansion_factor": e("conv_expansion")?,
            "context_size": e("context_size")?,
            "max_pos_emb": e("max_pos_emb")?,
            "self_conditioning_layer": e("self_cond_layer")?,
            "cat_hidden_layers": captured,
            "cat_with_last": false,
        },
        "projector_config": {
            "hidden_size": p("hidden")?,
            "mlp_ratio": p("mlp_ratio")?,
            "num_attention_heads": p("n_heads")?,
            "num_hidden_layers": p("n_layers")?,
            "encoder_dim": p("encoder_dim")?,
            "num_encoder_layers": p("num_encoder_layers")?,
            "block_size": p("block_size")?,
            "downsample_rate": p("downsample_rate")?,
            "llm_dim": p("llm_dim")?,
            "layer_norm_eps": g.get("stt.granite_nar.projector.layernorm_eps").as_f64(),
        },
        "llm_config": {
            "model_type": "granite",
            "hidden_size": q("hidden")?,
            "intermediate_size": q("intermediate")?,
            "num_hidden_layers": q("n_layers")?,
            "num_attention_heads": q("n_heads")?,
            "num_key_value_heads": q("n_kv_heads")?,
            "head_dim": q("head_dim")?,
            "vocab_size": q("vocab_size")?,
            "rms_norm_eps": qf("rms_norm_eps"),
            "rope_theta": qf("rope_theta"),
            "tie_word_embeddings": g.get("stt.granite_nar.text.tie_word_embeddings").as_bool().unwrap_or(true),
            "embedding_multiplier": qf("embedding_multiplier"),
            "attention_multiplier": qf("attention_multiplier"),
            "residual_multiplier": qf("residual_multiplier"),
            "logits_scaling": qf("logits_scaling"),
        },
        "feature_extractor": {
            "n_fft": g.num("stt.frontend.n_fft")?,
            "win_length": g.num("stt.frontend.win_length")?,
            "hop_length": g.num("stt.frontend.hop_length")?,
            "num_mel_bins": g.num("stt.frontend.num_mels")?,
            "stack_factor": 2,
            "floor_db": 8.0,
            "deltas": false,
            "frames": "even",
        },
    }))
}

/// A transcribe.cpp Granite NAR tensor name as Trit names it.
pub fn hf_name(name: &str) -> Option<Name> {
    if let Some(rest) = name.strip_prefix("prj.") {
        return Some(Name::To(format!("nar_projector.{rest}")));
    }
    if name.starts_with("dec.") {
        return qwen3_asr::language_model_name(name, "language_model.model");
    }
    // The block names of the AR release, which this file spells with `_`.
    let name = [
        ("ff1_up.", "ff1.up."),
        ("ff1_down.", "ff1.down."),
        ("ff2_up.", "ff2.up."),
        ("ff2_down.", "ff2.down."),
        ("attn_q.", "attn.q."),
        ("attn_kv.", "attn.kv."),
        ("attn_out.", "attn.out."),
        ("attn_rel_pos_emb.", "attn.rel_pos_emb."),
        ("conv_pointwise1.", "conv.pointwise1."),
        ("conv_depthwise.", "conv.depthwise."),
        ("conv_bn.", "conv.bn."),
        ("conv_pointwise2.", "conv.pointwise2."),
    ]
    .iter()
    .fold(name.to_string(), |name, (from, to)| name.replace(from, to));
    granite_speech::hf_name(&name)
}
