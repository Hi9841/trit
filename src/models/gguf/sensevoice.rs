//! SenseVoice GGUF files (`general.architecture` `sensevoice`, from
//! transcribe.cpp's `scripts/convert-sensevoice.py`): the config of the
//! encoder, the prompt rows, and the Kaldi front end, and the tensor names
//! of FunASR's `SenseVoiceSmall` state dict. The file's CMVN feeds the
//! front end as published.

use super::blocks::{kaldi_front, KaldiFront};
use super::{Gguf, Name};
use serde_json::{json, Value};

pub fn config(g: &Gguf) -> Result<Value, String> {
    let e = |key: &str| g.num(&format!("stt.sensevoice.encoder.{key}"));
    let special = |key: &str| g.num(&format!("stt.sensevoice.special.{key}"));
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    if text("stt.sensevoice.encoder.attention_type") != "sanm"
        || e("sanm_shift")? != 0
        || g.get("stt.sensevoice.encoder.normalize_before").as_bool() != Some(true)
    {
        return Err(
            "its encoder differs from SenseVoice Small's (SAN-M, no shift, pre-norm)".into(),
        );
    }
    if text("stt.frontend.type") != "kaldi_fbank_lfr"
        || text("stt.frontend.window") != "hamming"
        || g.get("stt.frontend.snip_edges").as_bool() != Some(true)
        || g.get("stt.frontend.dither").as_f64().unwrap_or(0.0) != 0.0
    {
        return Err("its audio features differ from FunASR's WavFrontend".into());
    }
    let input = e("d_input")?;
    // The prompt rows: the size of the embedding table.
    let rows = g
        .tensors
        .iter()
        .find(|(name, dtype, ..)| name == "enc.embed.weight" && dtype == "F32")
        .map(|(_, _, start, end)| (end - start) / 4 / input as usize)
        .ok_or("the prompt embedding is not F32")?;
    let mut languages = serde_json::Map::new();
    for code in ["auto", "zh", "en", "yue", "ja", "ko", "nospeech"] {
        languages.insert(code.into(), json!(special(&format!("lang_{code}"))?));
    }
    Ok(json!({
        "model_type": "sensevoice",
        "vocab_size": g.get("tokenizer.ggml.tokens").as_array().map_or(0, Vec::len),
        "blank_token_id": g.num("tokenizer.ggml.blank_token_id")?,
        "encoder_config": {
            "hidden_size": e("d_model")?,
            "input_size": input,
            "num_attention_heads": e("n_heads")?,
            "intermediate_size": e("d_ff")?,
            "kernel_size": e("kernel_size")?,
            "num_blocks": e("n_blocks")?,
            "tp_blocks": e("tp_blocks")?,
        },
        "prompt": {
            "rows": rows,
            "languages": languages,
            "event_speech": special("event_speech")?,
            "emotion_neutral": special("emotion_neutral")?,
            "with_itn": special("withitn")?,
        },
        "feature_extractor": kaldi_front(KaldiFront {
            sample_rate: g.num("stt.frontend.sample_rate")?,
            win: g.num("stt.frontend.win_length")?,
            hop: g.num("stt.frontend.hop_length")?,
            mels: g.num("stt.frontend.num_mels")?,
            lfr_m: g.num("stt.frontend.lfr_m")?,
            lfr_n: g.num("stt.frontend.lfr_n")?,
            upscale: g
                .get("stt.frontend.upscale_samples")
                .as_bool()
                .unwrap_or(true),
            cmvn: None,
        }),
    }))
}

/// A transcribe.cpp SenseVoice tensor name as FunASR's state dict names it.
pub fn hf_name(name: &str) -> Option<Name> {
    const TOP: [(&str, &str); 3] = [
        ("frontend.cmvn.", "frontend.cmvn."),
        ("enc.embed.", "embed."),
        ("ctc.head.", "ctc.ctc_lo."),
    ];
    if let Some(mapped) = super::map_suffix(name, &TOP) {
        return Some(Name::To(mapped));
    }
    encoder_name(name, "encoder")
}

/// A transcribe.cpp SAN-M encoder name (`enc.encoders0.0...`, as SenseVoice
/// and Fun-ASR files name it) as FunASR names it under `prefix`.
pub fn encoder_name(name: &str, prefix: &str) -> Option<Name> {
    if let Some(rest) = name.strip_prefix("enc.after_norm.") {
        return Some(Name::To(format!("{prefix}.after_norm.{rest}")));
    }
    if let Some(rest) = name.strip_prefix("enc.tp_norm.") {
        return Some(Name::To(format!("{prefix}.tp_norm.{rest}")));
    }
    let rest = name.strip_prefix("enc.")?;
    let (stack, rest) = rest.split_once('.')?;
    if !matches!(stack, "encoders0" | "encoders" | "tp_encoders") {
        return None;
    }
    let (layer, part) = rest.split_once('.')?;
    const BLOCK: [(&str, &str); 7] = [
        ("attn.qkv.", "self_attn.linear_q_k_v."),
        ("attn.out.", "self_attn.linear_out."),
        ("attn.fsmn.", "self_attn.fsmn_block."),
        ("ffn.fc1.", "feed_forward.w_1."),
        ("ffn.fc2.", "feed_forward.w_2."),
        ("norm_attn.", "norm1."),
        ("norm_ffn.", "norm2."),
    ];
    let mapped = super::map_suffix(part, &BLOCK)?;
    Some(Name::To(format!("{prefix}.{stack}.{layer}.{mapped}")))
}
