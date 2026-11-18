//! The `config.json` blocks that more than one GGUF architecture writes.

use super::Gguf;
use crate::models::package::{encoder_block, EncShape};
use serde_json::{json, Value};

/// A Llama-plan language model (`CausalLm::parse` reads these keys).
pub(crate) struct LlmShape<'a> {
    pub model_type: &'a str,
    pub hidden: u64,
    pub intermediate: u64,
    pub layers: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub vocab: u64,
    pub rms_norm_eps: Option<f64>,
    pub rope_theta: Option<f64>,
    pub tie: bool,
}

/// The language model block of a transcribe.cpp file, from the keys under
/// `prefix` (`stt.voxtral.decoder`). `tie` is the tie of the head when the
/// file has no `tie_word_embeddings`.
pub(crate) fn llm(g: &Gguf, prefix: &str, model_type: &str, tie: bool) -> Result<Value, String> {
    let num = |key: &str| g.num(&format!("{prefix}.{key}"));
    let float = |key: &str| g.get(&format!("{prefix}.{key}")).as_f64();
    Ok(llm_block(LlmShape {
        model_type,
        hidden: num("hidden_size")?,
        intermediate: num("intermediate_size")?,
        layers: num("n_layers")?,
        heads: num("n_heads")?,
        kv_heads: num("n_kv_heads")?,
        head_dim: num("head_dim")?,
        vocab: num("vocab_size")?,
        rms_norm_eps: float("rms_norm_eps"),
        rope_theta: float("rope_theta"),
        tie: g
            .get(&format!("{prefix}.tie_word_embeddings"))
            .as_bool()
            .unwrap_or(tie),
    }))
}

/// The FastConformer encoder block of a transcribe.cpp file, from the keys
/// under `prefix` (`stt.parakeet.encoder`).
pub(crate) fn fastconformer(
    g: &Gguf,
    prefix: &str,
    mels: u64,
    scale_input: bool,
) -> Result<Value, String> {
    let num = |key: &str| g.num(&format!("{prefix}.{key}"));
    Ok(encoder_block(EncShape {
        hidden: num("d_model")?,
        intermediate: num("d_ff")?,
        heads: num("n_heads")?,
        layers: num("n_layers")?,
        mels,
        conv_kernel: num("conv_kernel")?,
        subsample: num("subsampling_factor")?,
        sub_channels: num("subsampling_channels")?,
        scale_input,
    }))
}

pub(crate) fn llm_block(shape: LlmShape<'_>) -> Value {
    json!({
        "model_type": shape.model_type,
        "hidden_size": shape.hidden,
        "intermediate_size": shape.intermediate,
        "num_hidden_layers": shape.layers,
        "num_attention_heads": shape.heads,
        "num_key_value_heads": shape.kv_heads,
        "head_dim": shape.head_dim,
        "vocab_size": shape.vocab,
        "rms_norm_eps": shape.rms_norm_eps,
        "rope_theta": shape.rope_theta,
        "tie_word_embeddings": shape.tie,
    })
}

/// Granite Speech's mel front end.
pub(crate) fn granite_front(
    g: &Gguf,
    stack: u64,
    floor_db: f64,
    deltas: bool,
    frames: &str,
) -> Result<Value, String> {
    Ok(json!({
        "n_fft": g.num("stt.frontend.n_fft")?,
        "win_length": g.num("stt.frontend.win_length")?,
        "hop_length": g.num("stt.frontend.hop_length")?,
        "num_mel_bins": g.num("stt.frontend.num_mels")?,
        "stack_factor": stack,
        "floor_db": floor_db,
        "deltas": deltas,
        "frames": frames,
    }))
}

/// FunASR's Kaldi filterbank. `cmvn` is omitted when `None` (SenseVoice).
pub(crate) struct KaldiFront {
    pub sample_rate: u64,
    pub win: u64,
    pub hop: u64,
    pub mels: u64,
    pub lfr_m: u64,
    pub lfr_n: u64,
    pub upscale: bool,
    pub cmvn: Option<bool>,
}

pub(crate) fn kaldi_front(front: KaldiFront) -> Value {
    let mut value = json!({
        "type": "kaldi_fbank",
        "sample_rate": front.sample_rate,
        "win_length": front.win,
        "hop_length": front.hop,
        "num_mel_bins": front.mels,
        "lfr_m": front.lfr_m,
        "lfr_n": front.lfr_n,
        "upscale_samples": front.upscale,
    });
    if let Some(cmvn) = front.cmvn {
        value["cmvn"] = json!(cmvn);
    }
    value
}

/// Whether the file's front end is Whisper's mel (FFT of 400, hop of 160).
pub(crate) fn whisper_mel(g: &Gguf) -> Result<bool, String> {
    Ok(
        g.num("stt.frontend.sample_rate")? == u64::from(crate::SAMPLE_RATE)
            && g.num("stt.frontend.n_fft")? == 400
            && g.num("stt.frontend.hop_length")? == 160,
    )
}
