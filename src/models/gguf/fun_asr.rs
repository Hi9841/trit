//! Fun-ASR Nano GGUF files (`general.architecture` `funasr_nano`, from
//! transcribe.cpp's `scripts/convert-funasr_nano.py`): FunASR's SAN-M audio
//! encoder, a transformer adaptor, and a Qwen3 language model. Tensors take
//! FunASR's names (`audio_encoder`, `audio_adaptor`) and the Hugging Face
//! Qwen3 names under `llm.model`.

use super::{qwen3_asr, sensevoice, Gguf, Name};
use serde_json::{json, Value};

pub fn config(g: &Gguf) -> Result<Value, String> {
    let e = |key: &str| g.num(&format!("stt.funasr_nano.encoder.{key}"));
    let a = |key: &str| g.num(&format!("stt.funasr_nano.adaptor.{key}"));
    let q = |key: &str| g.num(&format!("stt.funasr_nano.decoder.{key}"));
    let float = |key: &str| g.get(&format!("stt.funasr_nano.decoder.{key}")).as_f64();
    let text = |key: &str| g.get(key).as_str().unwrap_or_default();
    if text("stt.funasr_nano.encoder.attention_type") != "sanm"
        || e("sanm_shift")? != 0
        || text("stt.funasr_nano.adaptor.activation") != "relu"
        || a("downsample_rate")? != 1
        || text("stt.funasr_nano.decoder.activation") != "silu"
    {
        return Err("its encoder or adaptor differs from Fun-ASR Nano's".into());
    }
    if text("stt.frontend.type") != "kaldi_fbank_lfr" || text("stt.frontend.window") != "hamming" {
        return Err("its audio features differ from FunASR's WavFrontend".into());
    }
    let vocab = q("vocab_size")?;
    Ok(json!({
        "model_type": "fun_asr_nano",
        // FunASR's default: text normalization on for the multilingual model only.
        "itn": text("stt.variant").contains("mlt"),
        "encoder_config": {
            "hidden_size": e("d_model")?,
            "input_size": e("d_input")?,
            "num_attention_heads": e("n_heads")?,
            "intermediate_size": e("d_ff")?,
            "kernel_size": e("kernel_size")?,
            "num_blocks": e("n_blocks")?,
            "tp_blocks": e("tp_blocks")?,
        },
        "adaptor_config": {
            "encoder_dim": a("encoder_dim")?,
            "llm_dim": a("llm_dim")?,
            "pre_ffn_dim": a("pre_ffn_dim")?,
            "ffn_dim": a("block_ffn_dim")?,
            "num_attention_heads": a("n_heads")?,
            "head_dim": a("d_head")?,
            "num_blocks": a("n_blocks")?,
            "low_frame_rate": g.get("stt.funasr_nano.adaptor.use_low_frame_rate").as_bool().unwrap_or(false),
        },
        "llm_config": {
            "model_type": "qwen3",
            "hidden_size": q("hidden_size")?,
            "intermediate_size": q("intermediate_size")?,
            "num_hidden_layers": q("n_layers")?,
            "num_attention_heads": q("n_heads")?,
            "num_key_value_heads": q("n_kv_heads")?,
            "head_dim": q("head_dim")?,
            "vocab_size": vocab,
            "rms_norm_eps": float("rms_norm_eps"),
            "rope_theta": float("rope_theta"),
            "tie_word_embeddings": g.get("stt.funasr_nano.decoder.tie_word_embeddings").as_bool().unwrap_or(true),
        },
        "feature_extractor": {
            "type": "kaldi_fbank",
            "sample_rate": g.num("stt.frontend.sample_rate")?,
            "win_length": g.num("stt.frontend.win_length")?,
            "hop_length": g.num("stt.frontend.hop_length")?,
            "num_mel_bins": g.num("stt.frontend.num_mels")?,
            "lfr_m": g.num("stt.frontend.lfr_m")?,
            "lfr_n": g.num("stt.frontend.lfr_n")?,
            "upscale_samples": g.get("stt.frontend.upscale_samples").as_bool().unwrap_or(true),
            "cmvn": g.get("stt.frontend.apply_cmvn").as_bool().unwrap_or(true),
        },
    }))
}

/// A transcribe.cpp Fun-ASR tensor name as Trit names it.
pub fn hf_name(name: &str) -> Option<Name> {
    if name.starts_with("dec.") {
        return qwen3_asr::language_model_name(name, "llm.model");
    }
    if let Some(rest) = name.strip_prefix("adaptor.") {
        if let Some((layer, part)) = rest.strip_prefix("blocks.").and_then(|r| r.split_once('.')) {
            const BLOCK: [(&str, &str); 8] = [
                ("attn.q.", "self_attn.linear_q."),
                ("attn.k.", "self_attn.linear_k."),
                ("attn.v.", "self_attn.linear_v."),
                ("attn.out.", "self_attn.linear_out."),
                ("ffn.fc1.", "feed_forward.w_1."),
                ("ffn.fc2.", "feed_forward.w_2."),
                ("norm_attn.", "norm1."),
                ("norm_ffn.", "norm2."),
            ];
            let (from, to) = BLOCK.iter().find(|(from, _)| part.starts_with(from))?;
            return Some(Name::To(format!(
                "audio_adaptor.blocks.{layer}.{to}{}",
                &part[from.len()..]
            )));
        }
        return Some(Name::To(format!("audio_adaptor.{rest}")));
    }
    sensevoice::encoder_name(name, "audio_encoder")
}
