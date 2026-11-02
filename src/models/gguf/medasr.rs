//! MedASR GGUF files (`general.architecture` `medasr`, from transcribe.cpp's
//! `scripts/convert-medasr.py`): Google's LASR CTC Conformer. The config
//! names its parts for the FastConformer (rotary attention on the
//! projected queries and keys, weighted residuals, a dense and 1-D conv
//! subsampler, a final norm); the tensors take the Parakeet names, and the
//! file's mel filterbank and window feed the front end as published.

use super::{Gguf, Name};
use serde_json::{json, Value};

pub fn config(g: &Gguf) -> Result<Value, String> {
    let e = |key: &str| g.num(&format!("stt.medasr.encoder.{key}"));
    let float = |key: &str| g.get(&format!("stt.medasr.encoder.{key}")).as_f64();
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    let d = e("hidden")?;
    let kernel = e("conv_kernel")?;
    if text("stt.medasr.encoder.hidden_act") != "silu"
        || text("stt.medasr.encoder.rope_type") != "default"
        || e("n_kv_heads")? != e("n_heads")?
        || e("head_dim")? * e("n_heads")? != d
        || e("sub_n_layers")? != 2
    {
        return Err(
            "its encoder differs from LASR's (SiLU, plain RoPE, two subsampling convs)".into(),
        );
    }
    if text("stt.frontend.type") != "mel"
        || text("stt.frontend.normalize") != "none"
        || g.num("stt.frontend.sample_rate")? != 16_000
    {
        return Err("its audio features differ from MedASR's".into());
    }
    let pair = |a: &str, b: &str| -> Result<Value, String> {
        Ok(json!([
            float(a).ok_or_else(|| format!("no stt.medasr.encoder.{a}"))?,
            float(b).ok_or_else(|| format!("no stt.medasr.encoder.{b}"))?
        ]))
    };
    let stride = e("sub_stride")?;
    Ok(json!({
        "model_type": "medasr",
        "vocab_size": g.num("stt.medasr.ctc.vocab_size")?,
        "pad_token_id": g.num("stt.medasr.ctc.blank_id")?,
        "encoder_config": {
            "hidden_size": d,
            "intermediate_size": e("intermediate")?,
            "num_attention_heads": e("n_heads")?,
            "num_hidden_layers": e("n_layers")?,
            "num_mel_bins": e("num_mel_bins")?,
            "conv_kernel_size": kernel,
            // An even kernel: one more frame after than before.
            "conv_context": [(kernel - 1) / 2, kernel - 1 - (kernel - 1) / 2],
            "subsampling": "lasr",
            "subsampling_factor": stride * stride,
            "subsampling_conv_kernel_size": e("sub_kernel")?,
            "subsampling_conv_stride": stride,
            "subsampling_conv_channels": e("sub_channels")?,
            "position_embedding": "rotary",
            "rotary_base": float("rope_theta").unwrap_or(10_000.0),
            "conv_norm_type": "batch_norm",
            "layer_norm_eps": float("layer_norm_eps").unwrap_or(1e-6),
            "ff_residual": pair("ff_residual_w0", "ff_residual_w1")?,
            "conv_residual": pair("conv_residual_w0", "conv_residual_w1")?,
            "output_norm": true,
        },
        "feature_extractor": {
            "type": "table",
            "n_fft": g.num("stt.frontend.n_fft")?,
            "win_length": g.num("stt.frontend.win_length")?,
            "hop_length": g.num("stt.frontend.hop_length")?,
            "log_floor": g.get("stt.frontend.log_clamp_min").as_f64(),
            // transcribe.cpp writes this filterbank bins first.
            "filterbank_layout": "bins_mels",
        },
    }))
}

/// A transcribe.cpp MedASR tensor name as Trit's Parakeet loader names it.
pub fn hf_name(name: &str) -> Option<Name> {
    if name.starts_with("frontend.") {
        return Some(Name::To(name.to_string()));
    }
    const TOP: [(&str, &str); 3] = [
        ("enc.subsampling.", "encoder.subsampling."),
        ("enc.out_norm.", "encoder.out_norm."),
        ("ctc.proj.", "ctc_head."),
    ];
    for (from, to) in TOP {
        if let Some(rest) = name.strip_prefix(from) {
            return Some(Name::To(format!("{to}{rest}")));
        }
    }
    let (layer, part) = name.strip_prefix("enc.blocks.")?.split_once('.')?;
    const BLOCK: [(&str, &str); 17] = [
        ("norm_ff1.", "norm_feed_forward1."),
        ("ff1_up.", "feed_forward1.linear1."),
        ("ff1_down.", "feed_forward1.linear2."),
        ("norm_attn.", "norm_self_att."),
        ("attn_q.", "self_attn.q_proj."),
        ("attn_k.", "self_attn.k_proj."),
        ("attn_v.", "self_attn.v_proj."),
        ("attn_o.", "self_attn.o_proj."),
        ("norm_conv.", "norm_conv."),
        ("conv_pointwise1.", "conv.pointwise_conv1."),
        ("conv_depthwise.", "conv.depthwise_conv."),
        ("conv_bn.", "conv.norm."),
        ("conv_pointwise2.", "conv.pointwise_conv2."),
        ("norm_ff2.", "norm_feed_forward2."),
        ("ff2_up.", "feed_forward2.linear1."),
        ("ff2_down.", "feed_forward2.linear2."),
        ("norm_post.", "norm_out."),
    ];
    let (from, to) = BLOCK.iter().find(|(from, _)| part.starts_with(from))?;
    Some(Name::To(format!(
        "encoder.layers.{layer}.{to}{}",
        &part[from.len()..]
    )))
}
