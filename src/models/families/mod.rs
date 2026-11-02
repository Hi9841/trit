//! One module per checkpoint family: it reads the family's files and wires
//! the parts its `config.json` names. The voices (text to speech) are in
//! `src/tts/`.

pub mod canary;
pub mod canary_qwen;
pub mod fun_asr;
pub mod granite_ctc;
pub mod granite_nar;
pub mod granite_speech;
pub mod moonshine;
pub mod moss;
pub mod parakeet;
pub mod qwen3_asr;
pub mod sensevoice;
pub mod voxtral;
pub mod voxtral_realtime;
pub mod wav2vec2;
pub mod whisper;
