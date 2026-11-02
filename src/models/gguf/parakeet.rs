//! Parakeet GGUF files (`general.architecture` `parakeet`, from
//! transcribe.cpp's `scripts/convert-parakeet.py`): the config of the
//! Hugging Face Parakeet checkpoint, and its tensor names.

use super::Gguf;
use serde_json::{json, Value};

/// The config of the Hugging Face checkpoint of this model, after
/// checking that it has the shape the FastConformer runs.
pub fn config(g: &Gguf) -> Result<Value, String> {
    let unsupported = |what: &str| format!("{}: {what}", g.path.display());
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
        || g.num("stt.frontend.sample_rate")? != 16_000
        || g.num("stt.frontend.n_fft")? != 512
        || g.num("stt.frontend.win_length")? != 400
        || g.num("stt.frontend.hop_length")? != 160
        || text("stt.frontend.window") != "hann"
        || !matches!(text("stt.frontend.normalize"), "per_feature" | "none")
        || (float("stt.frontend.pre_emphasis") - 0.97).abs() > 1e-6
    {
        return Err(unsupported("its audio features differ from Parakeet's"));
    }
    let blank = g.num("tokenizer.ggml.blank_token_id")?;
    let mut config = json!({
        "model_type": model_type,
        "encoder_config": {
            "hidden_size": enc("d_model")?,
            "intermediate_size": enc("d_ff")?,
            "num_attention_heads": enc("n_heads")?,
            "num_hidden_layers": enc("n_layers")?,
            "num_mel_bins": mels,
            "conv_kernel_size": enc("conv_kernel")?,
            "subsampling_factor": enc("subsampling_factor")?,
            "subsampling_conv_channels": enc("subsampling_channels")?,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "scale_input": g.get("stt.parakeet.encoder.xscaling").as_bool().unwrap_or(false),
            "attention_context": [side("att_context_left"), side("att_context_right")],
            "attention_style": style,
            "conv_context": conv_context,
            // transcribe.cpp's rule: a chunked_limited model subsamples causally.
            "causal_downsampling": style == "chunked_limited",
            "conv_norm_type": conv_norm,
            "speaker_kernel_layers": speaker_kernel,
        },
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
pub fn hf_name(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("enc.pre_encode.conv.") {
        return Some(format!("encoder.subsampling.layers.{rest}"));
    }
    if let Some(rest) = name.strip_prefix("enc.pre_encode.out.") {
        return Some(format!("encoder.subsampling.linear.{rest}"));
    }
    if let Some(rest) = name.strip_prefix("enc.blocks.") {
        let (layer, part) = rest.split_once('.')?;
        const BLOCK: [(&str, &str); 18] = [
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
            ("conv.bn.", "conv.norm."),
            ("norm_ff2.", "norm_feed_forward2."),
            ("ff2.", "feed_forward2."),
            ("norm_out.", "norm_out."),
        ];
        let (from, to) = BLOCK.iter().find(|(from, _)| part.starts_with(from))?;
        return Some(format!(
            "encoder.layers.{layer}.{to}{}",
            &part[from.len()..]
        ));
    }
    if name == "pred.embed.weight" {
        return Some("decoder.embedding.weight".into());
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
        return Some(format!("decoder.lstm.{hf}{layer}"));
    }
    for (from, to) in [
        ("joint.enc.", "encoder_projector."),
        ("joint.pred.", "decoder.decoder_projector."),
        ("joint.out.", "joint.head."),
        ("head.ctc.", "ctc_head."),
        ("vad.", "vad_head."),
        ("enc.spk_kernel.", "encoder.spk_kernel."),
        ("enc.bg_spk_kernel.", "encoder.bg_spk_kernel."),
        ("prompt.mlp.", "prompt_kernel."),
    ] {
        if let Some(rest) = name.strip_prefix(from) {
            return Some(format!("{to}{rest}"));
        }
    }
    None
}
