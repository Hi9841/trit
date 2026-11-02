//! Cohere Transcribe GGUF files (`general.architecture` `cohere_asr`, from
//! transcribe.cpp's `scripts/convert-cohere.py`): a FastConformer and a
//! transformer decoder of Canary's plan (stored positions, an embedding
//! norm, ReLU feed-forwards, Canary 2's prompt after a `▁` start token), its
//! head the token table plus a bias. The config is Canary's; the tensors
//! take the Canary names.

use super::{canary, Gguf, Name};
use serde_json::{json, Value};

pub fn config(g: &Gguf) -> Result<Value, String> {
    let e = |key: &str| g.num(&format!("stt.cohere.encoder.{key}"));
    let q = |key: &str| g.num(&format!("stt.cohere.decoder.{key}"));
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    if text("stt.cohere.decoder.activation") != "relu"
        || g.get("stt.cohere.head.tied_weights").as_bool() == Some(false)
    {
        return Err("its decoder differs from Cohere Transcribe's (ReLU, tied head)".into());
    }
    if text("stt.frontend.type") != "mel" || text("stt.frontend.normalize") != "per_feature" {
        return Err("its audio features differ from NeMo's".into());
    }
    let vocab = g
        .get("tokenizer.ggml.tokens")
        .as_array()
        .map_or(0, Vec::len);
    let heads = q("n_heads")?;
    Ok(json!({
        "model_type": "canary",
        "vocab_size": vocab,
        "pad_token_id": g.num("tokenizer.ggml.padding_token_id")?,
        "eos_token_id": g.num("tokenizer.ggml.eos_token_id")?,
        "prompt_start_token_id": g.num("stt.cohere.decoder_start_token_id")?,
        "decoder_config": {
            "hidden_size": q("hidden_size")?,
            "intermediate_size": q("inner_size")?,
            "num_attention_heads": heads,
            "num_key_value_heads": heads,
            "num_hidden_layers": q("n_layers")?,
            "vocab_size": vocab,
            "max_position_embeddings": q("max_seq_len")?,
            "eos_token_id": g.num("tokenizer.ggml.eos_token_id")?,
            "hidden_act": "relu",
        },
        "encoder_config": {
            "hidden_size": e("d_model")?,
            "intermediate_size": e("d_ff")?,
            "num_attention_heads": e("n_heads")?,
            "num_hidden_layers": e("n_layers")?,
            "num_mel_bins": g.num("stt.frontend.num_mels")?,
            "conv_kernel_size": e("conv_kernel")?,
            "subsampling_factor": e("subsampling_factor")?,
            "subsampling_conv_channels": e("subsampling_channels")?,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "scale_input": false,
        },
        "languages": g.get("general.languages"),
        "prompt_format": "canary2",
    }))
}

/// A transcribe.cpp Cohere tensor name as the Canary loader names it.
pub fn hf_name(name: &str) -> Option<Name> {
    if name.starts_with("frontend.") {
        // Trit computes NeMo's filterbank and window itself.
        return Some(Name::Skip);
    }
    const TOP: [(&str, &str); 3] = [
        ("enc_dec_proj.", "model.encoder_decoder_proj."),
        ("dec.final_norm.", "model.decoder.norm."),
        // The head is the token table; its bias goes beside the table.
        ("head.", "model.decoder.embed_tokens."),
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
        const LAYER: [(&str, &str); 13] = [
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
            ("norm_ff.", "final_layernorm."),
            ("ff.dense_in.", "mlp.fc1."),
            ("ff.dense_out.", "mlp.fc2."),
        ];
        let (from, to) = LAYER.iter().find(|(from, _)| part.starts_with(from))?;
        return Some(Name::To(format!(
            "model.decoder.layers.{layer}.{to}{}",
            &part[from.len()..]
        )));
    }
    canary::hf_name(name)
}
