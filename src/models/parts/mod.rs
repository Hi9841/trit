//! Building blocks that several families share. A family's `config.json`
//! picks them: the FastConformer encoder with a transducer or CTC head, the
//! Qwen3 language model, beam search, and the two dictionaries.

pub mod audio_encoder;
pub mod beam;
pub mod causal_lm;
pub mod ctc;
pub mod dictionary;
pub mod fastconformer;
pub mod granite_conformer;
pub mod logit_bias;
pub mod moonshine_streaming;
pub mod sanm;
pub mod transducer;
