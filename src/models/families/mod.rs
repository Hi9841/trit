//! One module per checkpoint family: it reads the family's files and wires
//! the parts its `config.json` names. The voices (text to speech) are in
//! `src/tts/`.

pub mod canary;
pub mod canary_qwen;
pub mod fun_asr;
pub mod granite_ctc;
pub mod moonshine;
pub mod parakeet;
pub mod qwen3_asr;
pub mod sensevoice;
pub mod wav2vec2;
pub mod whisper;
