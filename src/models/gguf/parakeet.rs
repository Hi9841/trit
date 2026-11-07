//! Parakeet GGUF files (`general.architecture` `parakeet`, from
//! transcribe.cpp's `scripts/convert-parakeet.py`): the config of the
//! Hugging Face Parakeet checkpoint, and its tensor names.

use super::{Gguf, Name};
use crate::audio::mel;
use crate::models::package::{encoder_block, EncShape};
use serde_json::{json, Value};

/// The config of the Hugging Face checkpoint of this model, after
/// checking that it has the shape the FastConformer runs.
pub fn config(g: &Gguf) -> Result<Value, String> {
    let unsupported = |what: &str| what.to_string();
    let head = g.get("stt.parakeet.head_kind").as_str().unwrap_or("tdt");
    let model_type = match head {
        "tdt" => "parakeet_tdt",
        "rnnt" => "parakeet_rnnt",
        "ctc" => "parakeet_ctc",
        other => return Err(unsupported(&format!("head `{other}` is not read"))),
    };
    let enc = |key: &str| g.num(&format!("stt.parakeet.encoder.{key}"));
    let side = |key: &str| {
        g.get(&format!("stt.parakeet.encoder.{key}"))
            .as_i64()
            .unwrap_or(-1)
    };
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    let float = |key: &str| g.get(key).as_f64().unwrap_or(f64::NAN);
    let style = text("stt.parakeet.encoder.att_context_style");
    // Offline, a chunked_limited_with_rc model (Unified) attends to
    // the whole clip; its chunk menu is for streaming.
    let style = match style {
        "" | "regular" | "chunked_limited_with_rc" => "regular",
        // Cache-aware streaming (Nemotron): run offline with its chunk mask.
        "chunked_limited" => "chunked_limited",
        other => {
            return Err(unsupported(&format!(
                "attention style `{other}` is not run"
            )))
        }
    };
    let kernel = enc("conv_kernel")?;
    let conv = |key: &str, default: u64| {
        g.get(&format!("stt.parakeet.encoder.{key}"))
            .as_i64()
            .filter(|v| *v >= 0)
            .map_or(default, |v| v as u64)
    };
    let conv_context = [
        conv("conv_context_left", (kernel - 1) / 2),
        conv("conv_context_right", (kernel - 1) / 2),
    ];
    let conv_norm = match text("stt.parakeet.encoder.conv_norm_type") {
        "" | "batch_norm" => "batch_norm",
        "layer_norm" => "layer_norm",
        other => return Err(unsupported(&format!("conv norm `{other}` is not run"))),
    };
    let speaker_kernel = g.get("stt.parakeet.encoder.spk_kernel_layers").clone();
    if !speaker_kernel.is_null()
        && (text("stt.parakeet.encoder.spk_kernel_type") != "ff"
            || g.get("stt.parakeet.encoder.add_bg_spk_kernel").as_bool() != Some(true))
    {
        return Err(unsupported(
            "its speaker kernel differs from NeMo's FF kernel",
        ));
    }
    let mels = g.num("stt.frontend.num_mels")?;
    if text("stt.frontend.type") != "mel"
        || g.num("stt.frontend.sample_rate")? != u64::from(crate::SAMPLE_RATE)
        || g.num("stt.frontend.n_fft")? != mel::N_FFT as u64
        || g.num("stt.frontend.win_length")? != mel::WIN_LENGTH as u64
        || g.num("stt.frontend.hop_length")? != mel::HOP as u64
        || text("stt.frontend.window") != "hann"
        || !matches!(text("stt.frontend.normalize"), "per_feature" | "none")
        || (float("stt.frontend.pre_emphasis") - f64::from(mel::PREEMPHASIS)).abs() > 1e-6
    {
        return Err(unsupported("its audio features differ from Parakeet's"));
    }
    let blank = g.num("tokenizer.ggml.blank_token_id")?;
    let mut encoder_config = encoder_block(EncShape {
        hidden: enc("d_model")?,
        intermediate: enc("d_ff")?,
        heads: enc("n_heads")?,
        layers: enc("n_layers")?,
        mels,
        conv_kernel: enc("conv_kernel")?,
        subsample: enc("subsampling_factor")?,
        sub_channels: enc("subsampling_channels")?,
        scale_input: g
            .get("stt.parakeet.encoder.xscaling")
            .as_bool()
            .unwrap_or(false),
    });
    encoder_config["attention_context"] =
        json!([side("att_context_left"), side("att_context_right")]);
    encoder_config["attention_style"] = json!(style);
    encoder_config["conv_context"] = json!(conv_context);
    // transcribe.cpp's rule: a chunked_limited model subsamples causally.
    encoder_config["causal_downsampling"] = json!(style == "chunked_limited");
    encoder_config["conv_norm_type"] = json!(conv_norm);
    encoder_config["speaker_kernel_layers"] = speaker_kernel;
    let mut config = json!({
        "model_type": model_type,
        "encoder_config": encoder_config,
        "feature_normalize": text("stt.frontend.normalize"),
    });
    if !g.get("stt.parakeet.prompt.num_prompts").is_null() {
        if text("stt.parakeet.prompt.activation") != "relu" {
            return Err(unsupported("its prompt MLP is not ReLU"));
        }
        let locales = g.get("stt.parakeet.prompt.dictionary.locales").as_array();
        let indices = g.get("stt.parakeet.prompt.dictionary.indices").as_array();
        let dictionary: serde_json::Map<String, Value> = locales
            .into_iter()
            .flatten()
            .zip(indices.into_iter().flatten())
            .filter_map(|(code, index)| Some((code.as_str()?.to_string(), index.clone())))
            .collect();
        config["prompt"] = json!({
            "num_prompts": g.num("stt.parakeet.prompt.num_prompts")?,
            "hidden": g.num("stt.parakeet.prompt.hidden")?,
            "auto_id": g.num("stt.parakeet.prompt.auto_id")?,
            "dictionary": dictionary,
        });
    }
    if model_type == "parakeet_ctc" {
        // The CTC head writes one logit per token and one for the blank,
        // the last id, which Trit calls the pad token.
        config["vocab_size"] = json!(blank + 1);
        config["pad_token_id"] = json!(blank);
        return Ok(config);
    }
    config["vocab_size"] = json!(g.num("stt.parakeet.predictor.vocab")?);
    config["blank_token_id"] = json!(blank);
    config["decoder_hidden_size"] = json!(g.num("stt.parakeet.predictor.hidden")?);
    config["num_decoder_layers"] = json!(g.num("stt.parakeet.predictor.n_layers")?);
    config["max_symbols_per_step"] =
        json!(g.get("stt.parakeet.tdt.max_symbols").as_u64().unwrap_or(10));
    if model_type == "parakeet_tdt" {
        config["durations"] = g.get("stt.parakeet.tdt.durations").clone();
    }
    Ok(config)
}

/// A transcribe.cpp tensor name as the Hugging Face Parakeet checkpoint
/// names it (`scripts/convert-parakeet.py` names them from NeMo's), or
/// `None` for a tensor Trit does not run.
pub fn hf_name(name: &str) -> Option<Name> {
    if name.starts_with("sortformer.") {
        // A multitalker bundle's Sortformer diarizer: Trit runs one speaker.
        return Some(Name::Skip);
    }
    rename(name, "", "conv.bn.")
}

/// `head` is `""` or `"model."` (Canary). `conv_norm` is the file's conv-norm
/// token (`conv.bn.`, or GigaAM's `conv.ln.`); both become `conv.norm.`.
pub(super) fn rename(name: &str, head: &str, conv_norm: &str) -> Option<Name> {
    let to = |rest: String| Some(Name::To(format!("{head}{rest}")));
    if let Some(rest) = name.strip_prefix("enc.pre_encode.conv.") {
        return to(format!("encoder.subsampling.layers.{rest}"));
    }
    if let Some(rest) = name.strip_prefix("enc.pre_encode.out.") {
        return to(format!("encoder.subsampling.linear.{rest}"));
    }
    if let Some(rest) = name.strip_prefix("enc.blocks.") {
        let (layer, part) = rest.split_once('.')?;
        const BLOCK: [(&str, &str); 17] = [
            ("norm_ff1.", "norm_feed_forward1."),
            ("ff1.", "feed_forward1."),
            ("norm_attn.", "norm_self_att."),
            ("attn.linear_q.", "self_attn.q_proj."),
            ("attn.linear_k.", "self_attn.k_proj."),
            ("attn.linear_v.", "self_attn.v_proj."),
            ("attn.linear_out.", "self_attn.o_proj."),
            ("attn.linear_pos.", "self_attn.relative_k_proj."),
            ("attn.pos_bias_u", "self_attn.bias_u"),
            ("attn.pos_bias_v", "self_attn.bias_v"),
            ("norm_conv.", "norm_conv."),
            ("conv.pointwise1.", "conv.pointwise_conv1."),
            ("conv.depthwise.", "conv.depthwise_conv."),
            ("conv.pointwise2.", "conv.pointwise_conv2."),
            ("norm_ff2.", "norm_feed_forward2."),
            ("ff2.", "feed_forward2."),
            ("norm_out.", "norm_out."),
        ];
        let mapped = super::map_suffix(part, &BLOCK).or_else(|| {
            let rest = part.strip_prefix(conv_norm)?;
            Some(format!("conv.norm.{rest}"))
        })?;
        return to(format!("encoder.layers.{layer}.{mapped}"));
    }
    if name == "pred.embed.weight" {
        return to("decoder.embedding.weight".into());
    }
    if let Some(rest) = name.strip_prefix("pred.lstm.") {
        let (layer, part) = rest.split_once('.')?;
        let hf = match part {
            "Wx" => "weight_ih_l",
            "Wh" => "weight_hh_l",
            // The sum of PyTorch's two biases; the loader adds a zero bias_hh.
            "bias" => "bias_ih_l",
            _ => return None,
        };
        return to(format!("decoder.lstm.{hf}{layer}"));
    }
    let mapped = super::map_suffix(
        name,
        &[
            ("joint.enc.", "encoder_projector."),
            ("joint.pred.", "decoder.decoder_projector."),
            ("joint.out.", "joint.head."),
            ("head.ctc.", "ctc_head."),
            ("vad.", "vad_head."),
            ("enc.spk_kernel.", "encoder.spk_kernel."),
            ("enc.bg_spk_kernel.", "encoder.bg_spk_kernel."),
            ("prompt.mlp.", "prompt_kernel."),
        ],
    )?;
    to(mapped)
}
