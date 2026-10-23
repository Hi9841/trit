//! Building blocks that several families share. A family's `config.json`
//! picks them: the FastConformer encoder with a transducer or CTC head, the
//! Qwen3 language model, beam search, and the two dictionaries.

pub mod beam;
pub mod ctc;
pub mod dictionary;
pub mod fastconformer;
pub mod logit_bias;
pub mod qwen3;
pub mod transducer;
