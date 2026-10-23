//! Whisper: OpenAI Whisper (tiny to large-v3, turbo), Distil-Whisper, and
//! other `WhisperForConditionalGeneration` checkpoints on Hugging Face.
//! Greedy decoding without timestamps, as one dictation take needs.

mod model;

use crate::audio::{
    self,
    whisper_mel::{WhisperMel, WINDOW_SAMPLES},
};
use crate::cpu::{buf::Buf, tensors::Tensors};
use crate::models::config::Json;
use crate::models::layers::argmax;
use crate::text::bpe::Bpe;
use crate::{Options, Precision, Task};
use parking_lot::RwLock;
use std::path::Path;

/// OpenAI Whisper's rule for a window of silence or noise: when the model
/// gives `<|nospeech|>` more than this probability after the start token...
const NO_SPEECH_PROBABILITY: f32 = 0.6;
/// ...and the text it writes averages less than this log probability per
/// token, the text is a guess ("you", "Thank you.") and is dropped.
const LOW_LOG_PROBABILITY: f32 = -1.0;

pub struct Whisper {
    model: model::Whisper,
    mel: WhisperMel,
    bpe: Bpe,
    tokens: Tokens,
    /// The dictionary as a previous-text prompt (`<|startofprev|>` and the
    /// terms), as OpenAI's `initial_prompt`; empty without one.
    previous: RwLock<Vec<usize>>,
}

/// Special tokens and decoding rules from `generation_config.json`.
struct Tokens {
    sot: usize,
    eos: usize,
    transcribe: Option<usize>,
    translate: Option<usize>,
    start_of_previous: Option<usize>,
    no_timestamps: usize,
    /// `<|nospeech|>` (`<|nocaptions|>` in older tokenizers).
    no_speech: Option<usize>,
    /// `(code, token)` per language; empty for English-only models.
    languages: Vec<(String, usize)>,
    suppress: Vec<usize>,
    begin_suppress: Vec<usize>,
}

impl Whisper {
    pub fn load(dir: &Path, config: &Json, precision: Precision) -> Result<Self, String> {
        let config = model::Config::parse(config)?;
        let mel = WhisperMel::new(config.mels);
        let tensors = Tensors::open_dir(dir, precision)?;
        let model = model::Whisper::load(&tensors, config)?;
        drop(tensors);
        let bpe = Bpe::open(&dir.join("tokenizer.json"))?;
        let generation = std::fs::read_to_string(dir.join("generation_config.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or(serde_json::Value::Null);
        let tokens = Tokens::new(&bpe, &generation)?;
        Ok(Self {
            model,
            mel,
            bpe,
            tokens,
            previous: RwLock::new(Vec::new()),
        })
    }

    /// Terms to favor, as a prompt of earlier text ("Kubernetes, PostgreSQL"):
    /// Whisper tends to spell words as its context does. Every term can be
    /// spelled, so none is skipped.
    pub fn set_dictionary(&self, terms: &[String]) -> Vec<String> {
        let terms: Vec<&str> = terms
            .iter()
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .collect();
        let mut previous = Vec::new();
        if let (Some(start), false) = (self.tokens.start_of_previous, terms.is_empty()) {
            let ids = self.bpe.encode(&format!(" {}", terms.join(", ")));
            // OpenAI keeps the last half of the context minus one.
            let keep = self.model.config.max_tokens / 2 - 1;
            previous.push(start);
            previous.extend_from_slice(&ids[ids.len().saturating_sub(keep)..]);
        }
        *self.previous.write() = previous;
        Vec::new()
    }

    /// The text of each window that holds speech: `(first sample, samples, text)`.
    pub fn pieces(&self, samples: &[f32], options: &Options) -> Vec<(usize, usize, String)> {
        let previous = self.previous.read();
        let task = match options.task {
            Task::Transcribe => self.tokens.transcribe,
            Task::Translate => self.tokens.translate,
        };
        let mut language = options.language.as_deref().and_then(|code| {
            self.tokens
                .languages
                .iter()
                .find(|(c, _)| c.eq_ignore_ascii_case(code.trim()))
                .map(|(_, id)| *id)
        });
        let mut pieces = Vec::new();
        for window in audio::windows(samples, WINDOW_SAMPLES) {
            if audio::is_silent(window) {
                continue;
            }
            let encoded = self.model.encode(&self.mel.compute(window));
            let mut session = self.model.session(&encoded);
            let logits = self.model.step(&mut session, self.tokens.sot);
            let no_speech = self
                .tokens
                .no_speech
                .map_or(0.0, |id| log_softmax(&logits, id).exp());
            let mut prompt = Vec::with_capacity(previous.len() + 4);
            if !previous.is_empty() {
                // The previous text goes before the start of the transcript:
                // decode again from position 0 with it.
                session.rewind(0, self.model.config.d);
                prompt.extend_from_slice(&previous);
                prompt.push(self.tokens.sot);
            }
            if !self.tokens.languages.is_empty() {
                // Detect once, on the first window, then keep it.
                let lang = *language.get_or_insert_with(|| {
                    let ids = self.tokens.languages.iter().map(|(_, id)| *id);
                    ids.max_by(|a, b| logits[*a].total_cmp(&logits[*b]))
                        .unwrap_or(self.tokens.sot)
                });
                prompt.push(lang);
                prompt.extend(task);
            }
            prompt.push(self.tokens.no_timestamps);
            let first = self.model.feed(&mut session, &prompt);
            let after_prompt = session.position;
            // OpenAI's rule: decode at temperature 0; while the text repeats
            // itself (compresses too well) or the model is unsure, decode
            // again at the next temperature. Silence ends the search.
            let mut best: Option<Decoded> = None;
            for (attempt, temperature) in TEMPERATURES.iter().enumerate() {
                if attempt > 0 {
                    session.rewind(after_prompt, self.model.config.d);
                }
                let mut logits = Buf::scratch(first.len());
                logits.copy_from_slice(&first);
                let decoded =
                    self.generate(&mut session, logits, *temperature, options.beams(BEAMS));
                let unsure = decoded.log_probability < LOW_LOG_PROBABILITY;
                let silent = no_speech > NO_SPEECH_PROBABILITY && unsure;
                let done = silent || (!decoded.repeats() && !unsure) || !options.fallback();
                // When every temperature fails: the most probable text that
                // does not repeat itself (as faster-whisper picks).
                let better = best.as_ref().is_none_or(|b| {
                    (b.repeats(), -b.log_probability)
                        > (decoded.repeats(), -decoded.log_probability)
                });
                if done || better {
                    best = Some(decoded);
                }
                if done {
                    break;
                }
            }
            let Some(decoded) = best else { continue };
            if no_speech > NO_SPEECH_PROBABILITY && decoded.log_probability < LOW_LOG_PROBABILITY {
                continue;
            }
            let piece = decoded.text;
            let piece = piece.trim();
            if !piece.is_empty() {
                pieces.push((
                    audio::offset(samples, window),
                    window.len(),
                    piece.to_string(),
                ));
            }
        }
        pieces
    }

    /// Tokens until end of text, the token limit, or a loop: greedy (or
    /// `beam` beams) at temperature 0, else sampled with a fixed seed, so a
    /// result repeats.
    fn generate(
        &self,
        session: &mut model::Session,
        mut logits: Buf,
        temperature: f32,
        beam: usize,
    ) -> Decoded {
        let t = &self.tokens;
        let limit = self.model.config.max_tokens;
        if beam > 1 && temperature == 0.0 {
            let best = crate::models::parts::beam::search(
                (beam, BEAM_RULES),
                (session.clone(), logits),
                (limit.saturating_sub(session.position), t.eos),
                |logits, tokens| self.mask(logits, tokens.is_empty()),
                |sessions, tokens| self.model.step_beams(sessions, tokens),
            );
            let text = self.bpe.decode(&best.tokens);
            return Decoded {
                compression: compression_ratio(&text),
                log_probability: best.log_probability,
                text,
            };
        }
        let mut ids = Vec::new();
        let mut total = 0.0f32;
        let mut random = Random(SEED);
        while session.position < limit {
            self.mask(&mut logits, ids.is_empty());
            let next = if temperature > 0.0 {
                random.sample(&logits, temperature)
            } else {
                argmax(&logits)
            };
            total += log_softmax(&logits, next);
            if next == t.eos {
                break;
            }
            ids.push(next);
            if let Some(keep) = looping(&ids) {
                ids.truncate(keep);
                break;
            }
            logits = self.model.step(session, next);
        }
        let count = (ids.len() + 1) as f32;
        let text = self.bpe.decode(&ids);
        Decoded {
            compression: compression_ratio(&text),
            log_probability: total / count,
            text,
        }
    }
}

impl Whisper {
    /// Blocks the tokens text may not hold: every special and timestamp id
    /// (above end of text), the suppress list, and at the start the
    /// begin-suppress list.
    fn mask(&self, logits: &mut [f32], first: bool) {
        let t = &self.tokens;
        logits[t.eos + 1..]
            .iter_mut()
            .for_each(|x| *x = f32::NEG_INFINITY);
        let begin = if first { &t.begin_suppress[..] } else { &[] };
        for &id in t.suppress.iter().chain(begin) {
            if let Some(x) = logits.get_mut(id) {
                *x = f32::NEG_INFINITY;
            }
        }
    }
}

impl Tokens {
    fn new(bpe: &Bpe, generation: &serde_json::Value) -> Result<Self, String> {
        let config_id = |key: &str| generation[key].as_u64().map(|v| v as usize);
        let token = |text: &str| bpe.id(text);
        let list = |key: &str| -> Vec<usize> {
            generation[key]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_u64().map(|v| v as usize))
                .collect()
        };
        let sot = config_id("decoder_start_token_id")
            .or_else(|| token("<|startoftranscript|>"))
            .ok_or("tokenizer has no <|startoftranscript|>")?;
        let eos = config_id("eos_token_id")
            .or_else(|| token("<|endoftext|>"))
            .ok_or("tokenizer has no <|endoftext|>")?;
        let no_timestamps = config_id("no_timestamps_token_id")
            .or_else(|| token("<|notimestamps|>"))
            .ok_or("tokenizer has no <|notimestamps|>")?;
        let multilingual = generation["is_multilingual"].as_bool().unwrap_or(true);
        let mut languages: Vec<(String, usize)> = generation["lang_to_id"]
            .as_object()
            .into_iter()
            .flatten()
            .filter_map(|(key, id)| {
                let code = key.strip_prefix("<|")?.strip_suffix("|>")?;
                Some((code.to_string(), id.as_u64()? as usize))
            })
            .collect();
        languages.sort_by_key(|(_, id)| *id);
        if !multilingual {
            languages.clear();
        }
        Ok(Self {
            sot,
            eos,
            transcribe: token("<|transcribe|>").filter(|_| multilingual),
            translate: token("<|translate|>").filter(|_| multilingual),
            start_of_previous: token("<|startofprev|>"),
            no_timestamps,
            no_speech: token("<|nospeech|>").or_else(|| token("<|nocaptions|>")),
            languages,
            suppress: list("suppress_tokens"),
            begin_suppress: match list("begin_suppress_tokens") {
                // No generation config: OpenAI's `suppress_blank`, a space
                // ("Ġ") or end of text as the first token.
                ids if ids.is_empty() => token("Ġ").into_iter().chain([eos]).collect(),
                ids => ids,
            },
        })
    }
}

/// One decode of a window.
struct Decoded {
    text: String,
    /// Mean log probability per token, end of text included (as OpenAI counts it).
    log_probability: f32,
    /// Text bytes over zlib bytes; a loop compresses well.
    compression: f32,
}

impl Decoded {
    fn repeats(&self) -> bool {
        self.compression > MAX_COMPRESSION
    }
}

/// Beams unless the caller sets `Options::beam`: on the 456 English test
/// clips 5 beams gave 768 and 479 errors (tiny, base.en), greedy 894 and 519.
const BEAMS: usize = 5;
/// Whose beam search rules to follow: Transformers' gave 768 and 479 errors
/// (tiny, base.en), OpenAI's 782 and 472; Transformers' runtime made 772 and 479.
const BEAM_RULES: crate::models::parts::beam::Rules =
    crate::models::parts::beam::Rules::Transformers;

/// The temperatures OpenAI Whisper, faster-whisper, and whisper.cpp try.
const TEMPERATURES: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
/// A text that compresses better than this repeats itself.
const MAX_COMPRESSION: f32 = 2.4;
const SEED: u64 = 0x5EED;

/// `len(text) / len(zlib.compress(text))`, as OpenAI computes it.
fn compression_ratio(text: &str) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    let packed = miniz_oxide::deflate::compress_to_vec_zlib(text.as_bytes(), 6);
    text.len() as f32 / packed.len() as f32
}

/// SplitMix64: a small generator with a fixed seed, so a sampled decode repeats.
struct Random(u64);

impl Random {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A token drawn from `softmax(logits / temperature)`.
    fn sample(&mut self, logits: &[f32], temperature: f32) -> usize {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weight = |x: f32| ((x - max) / temperature).exp();
        let total: f32 = logits.iter().map(|x| weight(*x)).sum();
        let mut left = self.next() * total;
        for (id, x) in logits.iter().enumerate() {
            left -= weight(*x);
            if left <= 0.0 {
                return id;
            }
        }
        argmax(logits)
    }
}

/// When the tail of `ids` is one span of 1 to 16 tokens repeated four times,
/// the length that keeps one copy. Greedy decoding can fall into such loops.
fn looping(ids: &[usize]) -> Option<usize> {
    for span in 1..=16 {
        let need = 4 * span;
        if ids.len() < need {
            break;
        }
        let tail = &ids[ids.len() - need..];
        if tail.chunks_exact(span).all(|c| c == &tail[..span]) {
            return Some(ids.len() - need + span);
        }
    }
    None
}

/// `log softmax(logits)[id]`; masked logits (minus infinity) count as zero.
fn log_softmax(logits: &[f32], id: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|x| (x - max).exp()).sum();
    logits[id] - max - sum.ln()
}
