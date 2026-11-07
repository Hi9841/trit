//! GigaAM v3 GGUF files (`general.architecture` `gigaam`, from
//! transcribe.cpp's `scripts/convert-gigaam.py`): a Conformer with rotary
//! attention, conv1d subsampling by 4, and a layer-norm conv module, then a
//! CTC or RNN-T head. The config names these parts for the FastConformer;
//! the tensors take the Parakeet names, and the file's mel filterbank and
//! window feed the front end as published.

use super::{parakeet, Gguf, Name};
use serde_json::{json, Value};

/// transcribe.cpp's rotary base (GigaAM's `RotaryPositionalEmbedding`).
const ROTARY_BASE: f64 = 5000.0;
/// GigaAM's greedy RNN-T decoding.
const MAX_SYMBOLS: u64 = 10;

pub fn config(g: &Gguf) -> Result<Value, String> {
    let e = |key: &str| g.num(&format!("stt.gigaam.encoder.{key}"));
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    let float = |key: &str| g.get(key).as_f64();
    if text("stt.gigaam.encoder.self_attention_model") != "rotary"
        || text("stt.gigaam.encoder.conv_norm_type") != "layer_norm"
        || e("subsampling_factor")? != 4
    {
        return Err(
            "its encoder differs from GigaAM v3's (rotary, layer norm, conv1d by 4)".into(),
        );
    }
    if text("stt.frontend.type") != "mel"
        || g.get("stt.frontend.center").as_bool() != Some(false)
        || text("stt.frontend.normalize") != "none"
        || float("stt.frontend.pre_emphasis").unwrap_or(0.0) != 0.0
        || g.num("stt.frontend.sample_rate")? != 16_000
    {
        return Err("its audio features differ from GigaAM's".into());
    }
    let d = e("d_model")?;
    let blank = g.num("tokenizer.ggml.blank_token_id")?;
    let mut config = json!({
        "encoder_config": {
            "hidden_size": d,
            "intermediate_size": e("d_ff")?,
            "num_attention_heads": e("n_heads")?,
            "num_hidden_layers": e("n_layers")?,
            "num_mel_bins": e("feat_in")?,
            "conv_kernel_size": e("conv_kernel")?,
            "subsampling": "conv1d",
            "subsampling_factor": 4,
            "subsampling_conv_kernel_size": e("subs_kernel_size")?,
            "subsampling_conv_stride": 2,
            "position_embedding": "rotary_input",
            "rotary_base": ROTARY_BASE,
            "conv_norm_type": "layer_norm",
        },
        "feature_extractor": {
            "type": "table",
            "n_fft": g.num("stt.frontend.n_fft")?,
            "win_length": g.num("stt.frontend.win_length")?,
            "hop_length": g.num("stt.frontend.hop_length")?,
            "log_floor": float("stt.frontend.log_clamp_min"),
            "log_ceiling": float("stt.frontend.log_clamp_max").unwrap_or(1e9),
        },
    });
    match text("stt.gigaam.head_kind") {
        "ctc" => {
            config["model_type"] = json!("gigaam_ctc");
            config["vocab_size"] = json!(g.num("stt.gigaam.head.num_classes")?);
            config["pad_token_id"] = json!(blank);
        }
        "rnnt" => {
            let hidden = g.num("stt.gigaam.predictor.hidden")?;
            if g.num("stt.gigaam.joint.hidden")? != hidden
                || text("stt.gigaam.joint.activation") != "relu"
            {
                return Err(
                    "its joint network differs from a ReLU joint as wide as the predictor".into(),
                );
            }
            config["model_type"] = json!("gigaam_rnnt");
            config["vocab_size"] = json!(g.num("stt.gigaam.joint.num_classes")?);
            config["blank_token_id"] = json!(blank);
            config["decoder_hidden_size"] = json!(hidden);
            config["num_decoder_layers"] = json!(g.num("stt.gigaam.predictor.n_layers")?);
            config["max_symbols_per_step"] = json!(MAX_SYMBOLS);
            // GigaAM's predictor starts from zeros, not the blank's embedding.
            config["decoder_start"] = json!("zeros");
        }
        other => return Err(format!("head `{other}` is not read")),
    }
    Ok(config)
}

/// A transcribe.cpp GigaAM tensor name as Trit's Parakeet loader names it.
pub fn hf_name(name: &str) -> Option<Name> {
    if name.starts_with("frontend.") {
        // The filterbank and window, read by the front end.
        return Some(Name::To(name.to_string()));
    }
    // GigaAM's conv-module norm is the layer norm transcribe.cpp names `ln`.
    parakeet::rename(name, "", "conv.ln.")
}
