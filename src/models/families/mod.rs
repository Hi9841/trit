//! One module per checkpoint family: it reads the family's files and wires
//! the parts its `config.json` names.

pub mod canary;
pub mod moonshine;
pub mod parakeet;
pub mod qwen3_asr;
pub mod wav2vec2;
pub mod whisper;
