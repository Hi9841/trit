//! Cohere Transcribe GGUF files (`general.architecture` `cohere_asr`, from
//! transcribe.cpp's `scripts/convert-cohere.py`): a FastConformer and a
//! transformer decoder of Canary's plan (stored positions, an embedding
//! norm, ReLU feed-forwards, Canary 2's prompt after a `▁` start token), its
//! head the token table plus a bias. The config is Canary's; the tensors
//! take the Canary names.

use super::{canary, Gguf, Name};
use crate::models::package::{encoder_block, EncShape};
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
        "encoder_config": encoder_block(EncShape {
            hidden: e("d_model")?,
            intermediate: e("d_ff")?,
            heads: e("n_heads")?,
            layers: e("n_layers")?,
            mels: g.num("stt.frontend.num_mels")?,
            conv_kernel: e("conv_kernel")?,
            subsample: e("subsampling_factor")?,
            sub_channels: e("subsampling_channels")?,
            scale_input: false,
        }),
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
    if let Some(mapped) = super::map_suffix(name, &TOP) {
        return Some(Name::To(mapped));
    }
    if let Some((layer, part)) = name
        .strip_prefix("dec.blocks.")
        .and_then(|rest| rest.split_once('.'))
    {
        const FROM: [&str; 13] = [
            "norm_self.",
            "self_attn.q.",
            "self_attn.k.",
            "self_attn.v.",
            "self_attn.out.",
            "norm_cross.",
            "cross_attn.q.",
            "cross_attn.k.",
            "cross_attn.v.",
            "cross_attn.out.",
            "norm_ff.",
            "ff.dense_in.",
            "ff.dense_out.",
        ];
        let suffix = canary::decoder_suffix(part, &FROM)?;
        return Some(Name::To(format!("model.decoder.layers.{layer}.{suffix}")));
    }
    canary::hf_name(name)
}
