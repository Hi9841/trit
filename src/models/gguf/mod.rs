//! GGUF files (ggml's model format, as transcribe.cpp writes them), read as
//! published: the metadata gives the files of a Hugging Face folder
//! (`config.json`, `tokenizer.json`, ...), built by the module of the
//! file's `general.architecture`, and the tensors take the Hugging Face
//! names and keep their stored types under `Precision::Auto` (a Q8_0 file
//! runs as Q8_0). The family loaders then run as for a Hugging Face folder.
//!
//! Read: `parakeet` (TDT, RNNT, CTC; full or local attention), `whisper`,
//! `moonshine`, `canary`, and `qwen3_asr`.

use crate::cpu::tensors::Tensors;
use crate::{Device, Precision};

mod canary;
mod canary_qwen;
mod cohere;
mod fun_asr;
mod gigaam;
mod granite_ctc;
mod granite_speech;
mod medasr;
mod moonshine;
mod moss;
mod parakeet;
mod qwen3_asr;
mod sensevoice;
mod voxtral;
mod voxtral_realtime;
mod whisper;

/// The architectures Trit reads, by `general.architecture`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arch {
    Parakeet,
    Whisper,
    Moonshine,
    Canary,
    Qwen3Asr,
    Gigaam,
    Sensevoice,
    Medasr,
    GraniteCtc,
    CanaryQwen,
    FunAsr,
    MoonshineStreaming,
    GraniteSpeech,
    Voxtral,
    Cohere,
    VoxtralRealtime,
    Moss,
}

/// What a tensor of the file becomes.
pub(super) enum Name {
    /// The Hugging Face name.
    To(String),
    /// A tensor Trit computes itself (a mel filterbank, a window).
    Skip,
}
use serde_json::{json, Map, Value};
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

/// The `.gguf` file of a model folder: the one GGUF file in a folder that
/// has no `config.json`.
pub fn find(dir: &Path) -> Option<PathBuf> {
    if dir.join("config.json").is_file() {
        return None;
    }
    let mut files = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "gguf"));
    let first = files.next()?;
    files.next().is_none().then_some(first)
}

pub struct Gguf {
    path: PathBuf,
    meta: Map<String, Value>,
    /// `(name, dtype, start, end)`, byte ranges of the file.
    tensors: Vec<(String, String, usize, usize)>,
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self, String> {
        let error = |e: std::io::Error| format!("Could not read {}: {e}", path.display());
        let file = File::open(path).map_err(error)?;
        let total = file.metadata().map_err(error)?.len();
        let mut reader = Reader(BufReader::with_capacity(1 << 20, file));
        let fail = |e: String| format!("{}: {e}", path.display());
        let mut magic = [0u8; 4];
        reader.bytes(&mut magic).map_err(fail)?;
        if &magic != MAGIC {
            return Err(fail("not a GGUF file".into()));
        }
        let version = reader.u32().map_err(fail)?;
        if !(2..=3).contains(&version) {
            return Err(fail(format!("GGUF version {version}; Trit reads 2 and 3")));
        }
        let tensor_count = reader.u64().map_err(fail)?;
        let kv_count = reader.u64().map_err(fail)?;
        let mut meta = Map::new();
        for _ in 0..kv_count {
            let key = reader.string().map_err(fail)?;
            let kind = reader.u32().map_err(fail)?;
            meta.insert(key, reader.value(kind).map_err(fail)?);
        }
        let mut infos = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = reader.string().map_err(fail)?;
            let dims = reader.u32().map_err(fail)?;
            let mut count = 1u64;
            for _ in 0..dims {
                count = count.saturating_mul(reader.u64().map_err(fail)?);
            }
            let kind = reader.u32().map_err(fail)?;
            let offset = reader.u64().map_err(fail)?;
            infos.push((name, kind, count, offset));
        }
        let alignment = meta
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT);
        let here = reader.position().map_err(fail)?;
        let data = here.div_ceil(alignment) * alignment;
        let mut tensors = Vec::with_capacity(infos.len());
        for (name, kind, count, offset) in infos {
            let block = crate::cpu::ggml_quants::BlockType::from_ggml(kind);
            let (dtype, bytes) = match (kind, block) {
                (0, _) => ("F32", count * 4),
                (1, _) => ("F16", count * 2),
                (8, _) if count % 32 == 0 => ("Q8_0", count / 32 * 34),
                (30, _) => ("BF16", count * 2),
                (_, Some((block, name))) if count % block.block().0 as u64 == 0 => {
                    let (values, size) = block.block();
                    (name, count / values as u64 * size as u64)
                }
                (other, _) => {
                    return Err(fail(format!(
                        "tensor {name} has ggml type {other}; Trit reads F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, and Q6_K"
                    )))
                }
            };
            let start = data + offset;
            if start + bytes > total {
                return Err(fail(format!("tensor {name} runs past the end of the file")));
            }
            tensors.push((
                name,
                dtype.to_string(),
                start as usize,
                (start + bytes) as usize,
            ));
        }
        Ok(Self {
            path: path.to_path_buf(),
            meta,
            tensors,
        })
    }

    fn arch(&self) -> Result<Arch, String> {
        match self.get("general.architecture").as_str() {
            Some("parakeet") => Ok(Arch::Parakeet),
            Some("whisper") => Ok(Arch::Whisper),
            Some("moonshine") => Ok(Arch::Moonshine),
            Some("canary") => Ok(Arch::Canary),
            Some("qwen3_asr") => Ok(Arch::Qwen3Asr),
            Some("gigaam") => Ok(Arch::Gigaam),
            Some("sensevoice") => Ok(Arch::Sensevoice),
            Some("medasr") => Ok(Arch::Medasr),
            Some("granite_speech5_ctc") => Ok(Arch::GraniteCtc),
            Some("canary_qwen") => Ok(Arch::CanaryQwen),
            Some("funasr_nano") => Ok(Arch::FunAsr),
            Some("moonshine_streaming") => Ok(Arch::MoonshineStreaming),
            Some("granite_speech") => Ok(Arch::GraniteSpeech),
            Some("voxtral") => Ok(Arch::Voxtral),
            Some("cohere_asr") => Ok(Arch::Cohere),
            Some("voxtral_realtime") => Ok(Arch::VoxtralRealtime),
            Some("moss") => Ok(Arch::Moss),
            other => Err(format!(
                "{}: GGUF architecture {} is not read; Trit reads parakeet, whisper, moonshine, canary, qwen3_asr, gigaam, sensevoice, medasr, granite_speech5_ctc, canary_qwen, funasr_nano, moonshine_streaming, granite_speech, voxtral, cohere_asr, voxtral_realtime, and moss",
                self.path.display(),
                other.unwrap_or("(none)")
            )),
        }
    }

    pub(super) fn get(&self, key: &str) -> &Value {
        self.meta.get(key).unwrap_or(&Value::Null)
    }

    pub(super) fn num(&self, key: &str) -> Result<u64, String> {
        self.get(key)
            .as_u64()
            .ok_or_else(|| format!("{} has no {key}", self.path.display()))
    }

    /// The id of token `text` in `tokenizer.ggml.tokens`.
    pub(super) fn token(&self, text: &str) -> Option<usize> {
        self.get("tokenizer.ggml.tokens")
            .as_array()?
            .iter()
            .position(|token| token.as_str() == Some(text))
    }

    /// The tensors under their Hugging Face names, in their stored types.
    pub fn tensors(&self, precision: Precision, device: Device) -> Result<Tensors, String> {
        let arch = self.arch()?;
        let mut renamed = Vec::with_capacity(self.tensors.len());
        for (name, dtype, start, end) in &self.tensors {
            let to = match arch {
                // A multitalker bundle's Sortformer diarizer: Trit runs one speaker.
                Arch::Parakeet if name.starts_with("sortformer.") => Some(Name::Skip),
                Arch::Parakeet => parakeet::hf_name(name).map(Name::To),
                Arch::Whisper => whisper::hf_name(name),
                Arch::Moonshine => moonshine::hf_name(name),
                Arch::Canary => canary::hf_name(name),
                Arch::Qwen3Asr => qwen3_asr::hf_name(name),
                Arch::Gigaam => gigaam::hf_name(name),
                Arch::Sensevoice => sensevoice::hf_name(name),
                Arch::Medasr => medasr::hf_name(name),
                Arch::GraniteCtc => granite_ctc::hf_name(name),
                Arch::CanaryQwen => canary_qwen::hf_name(name),
                Arch::FunAsr => fun_asr::hf_name(name),
                Arch::MoonshineStreaming => moonshine::hf_name(name),
                Arch::GraniteSpeech => granite_speech::hf_name(name),
                Arch::Voxtral => voxtral::hf_name(name),
                Arch::Cohere => cohere::hf_name(name),
                Arch::VoxtralRealtime => voxtral_realtime::hf_name(name),
                Arch::Moss => moss::hf_name(name),
            };
            match to {
                Some(Name::To(hf)) => renamed.push((hf, dtype.clone(), *start, *end)),
                Some(Name::Skip) => {}
                None => {
                    return Err(format!(
                        "{}: Trit does not read tensor {name}",
                        self.path.display()
                    ))
                }
            }
        }
        Tensors::open_gguf(&self.path, renamed, precision, device)
    }

    /// The files a Hugging Face folder of this model holds.
    pub fn provides(&self, name: &str) -> bool {
        matches!(
            (self.arch(), name),
            (Ok(_), "config.json" | "tokenizer.json")
                | (Ok(Arch::Whisper), "generation_config.json")
        )
    }

    /// File `name` of a Hugging Face folder of this model, from the metadata.
    pub fn file(&self, name: &str) -> Result<String, String> {
        let arch = self.arch()?;
        let fail = |e: String| format!("{}: {e}", self.path.display());
        match (arch, name) {
            (Arch::Parakeet, "config.json") => parakeet::config(self).map(|v| v.to_string()),
            (Arch::Whisper, "config.json") => {
                whisper::config(self).map(|v| v.to_string()).map_err(fail)
            }
            (Arch::Moonshine, "config.json") => {
                moonshine::config(self).map(|v| v.to_string()).map_err(fail)
            }
            (Arch::Canary, "config.json") => {
                canary::config(self).map(|v| v.to_string()).map_err(fail)
            }
            (Arch::Moss, "config.json") => moss::config(self).map(|v| v.to_string()).map_err(fail),
            (Arch::VoxtralRealtime, "config.json") => voxtral_realtime::config(self)
                .map(|v| v.to_string())
                .map_err(fail),
            (Arch::Cohere, "config.json") => {
                cohere::config(self).map(|v| v.to_string()).map_err(fail)
            }
            (Arch::Voxtral, "config.json") => {
                voxtral::config(self).map(|v| v.to_string()).map_err(fail)
            }
            (Arch::GraniteSpeech, "config.json") => granite_speech::config(self)
                .map(|v| v.to_string())
                .map_err(fail),
            (Arch::MoonshineStreaming, "config.json") => moonshine::streaming_config(self)
                .map(|v| v.to_string())
                .map_err(fail),
            (Arch::FunAsr, "config.json") => {
                fun_asr::config(self).map(|v| v.to_string()).map_err(fail)
            }
            (Arch::CanaryQwen, "config.json") => canary_qwen::config(self)
                .map(|v| v.to_string())
                .map_err(fail),
            (Arch::GraniteCtc, "config.json") => granite_ctc::config(self)
                .map(|v| v.to_string())
                .map_err(fail),
            (Arch::Medasr, "config.json") => {
                medasr::config(self).map(|v| v.to_string()).map_err(fail)
            }
            (Arch::Sensevoice, "config.json") => sensevoice::config(self)
                .map(|v| v.to_string())
                .map_err(fail),
            (Arch::Gigaam, "config.json") => {
                gigaam::config(self).map(|v| v.to_string()).map_err(fail)
            }
            (Arch::Qwen3Asr, "config.json") => {
                qwen3_asr::config(self).map(|v| v.to_string()).map_err(fail)
            }
            (Arch::Whisper, "generation_config.json") => whisper::generation(self)
                .map(|v| v.to_string())
                .map_err(fail),
            (_, "tokenizer.json") => self.tokenizer_json(),
            _ => Err(format!("{} has no {name}", self.path.display())),
        }
    }

    /// `tokenizer.json` from the `tokenizer.ggml.*` lists. A file with
    /// `merges` gives them in rank order; one without (Parakeet's
    /// SentencePiece models) gives scores, and the merges come from them.
    /// Every token is in the vocabulary; unknown, control, and user-defined
    /// tokens are also added tokens (special for the first two), as is every
    /// token after the first control token: the base vocabulary comes first,
    /// and a file marks an added token that is not special (Qwen's
    /// `<asr_text>`) as normal.
    fn tokenizer_json(&self) -> Result<String, String> {
        if self.get("tokenizer.ggml.merges").is_null() {
            return self.sentencepiece_json();
        }
        let tokens = self
            .get("tokenizer.ggml.tokens")
            .as_array()
            .ok_or_else(|| format!("{} has no tokenizer.ggml.tokens", self.path.display()))?;
        let kinds = self.get("tokenizer.ggml.token_type").as_array();
        let kind = |id: usize| {
            kinds
                .and_then(|kinds| kinds.get(id))
                .and_then(Value::as_u64)
                .unwrap_or(1)
        };
        // The first control token after the base vocabulary (a leading
        // `<|blank|>` is part of it).
        let first_control = (0..tokens.len())
            .position(|id| kind(id) == 1)
            .and_then(|normal| (normal..tokens.len()).find(|&id| kind(id) == 3));
        let mut vocab = Map::new();
        let mut added = Vec::new();
        for (id, token) in tokens.iter().enumerate() {
            let text = token.as_str().unwrap_or_default();
            vocab.insert(text.to_string(), json!(id));
            if matches!(kind(id), 2..=4) || first_control.is_some_and(|first| id >= first) {
                added.push(
                    json!({ "id": id, "content": text, "special": matches!(kind(id), 2 | 3) }),
                );
            }
        }
        let mut json = json!({
            "model": { "type": "BPE", "vocab": vocab, "merges": self.get("tokenizer.ggml.merges") },
            "added_tokens": added,
        });
        // llama.cpp's `gpt2` model is byte-level BPE; `bpe` with merges is a
        // SentencePiece BPE (`▁` and `<0x..>` pieces, Moonshine).
        if self.get("tokenizer.ggml.model").as_str() == Some("gpt2") {
            json["decoder"] = json!({ "type": "ByteLevel" });
        }
        // Qwen's pre-tokenizer, which `Bpe::parse` knows by its pattern. Mistral's
        // Tekken splits text the same way where Trit encodes it (`lang:en`).
        if matches!(
            self.get("tokenizer.ggml.pre").as_str(),
            Some("qwen2" | "tekken")
        ) {
            json["pre_tokenizer"] = json!({ "type": "Split", "pattern": {
                "Regex": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            }});
        }
        Ok(json.to_string())
    }

    /// The tokenizer from `tokenizer.ggml.tokens`, `scores`, and `token_type`.
    fn sentencepiece_json(&self) -> Result<String, String> {
        let list = |key: &str| {
            self.get(key)
                .as_array()
                .ok_or_else(|| format!("{} has no {key}", self.path.display()))
        };
        let pieces = list("tokenizer.ggml.tokens")?
            .iter()
            .map(|piece| piece.as_str().unwrap_or_default().to_string())
            .collect::<Vec<_>>();
        let scores = list("tokenizer.ggml.scores")?
            .iter()
            .map(|score| score.as_f64().unwrap_or(0.0) as f32)
            .collect::<Vec<_>>();
        let kinds = list("tokenizer.ggml.token_type")?
            .iter()
            .map(|kind| kind.as_u64().unwrap_or(1))
            .collect::<Vec<_>>();
        crate::text::sentencepiece::tokenizer_json(&pieces, &scores, &kinds)
            .map_err(|e| format!("{}: {e}", self.path.display()))
    }
}

/// Little-endian reads from the start of a GGUF file.
struct Reader(BufReader<File>);

impl Reader {
    fn bytes(&mut self, out: &mut [u8]) -> Result<(), String> {
        self.0
            .read_exact(out)
            .map_err(|e| format!("the header ends early: {e}"))
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let mut out = [0u8; N];
        self.bytes(&mut out)?;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn string(&mut self) -> Result<String, String> {
        let len = self.u64()?;
        if len > 1 << 24 {
            return Err("a header string is too long".into());
        }
        let mut bytes = vec![0u8; len as usize];
        self.bytes(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| "a header string is not UTF-8".into())
    }

    fn position(&mut self) -> Result<u64, String> {
        self.0
            .stream_position()
            .map_err(|e| format!("cannot read the header: {e}"))
    }

    /// One metadata value of GGUF type `kind`.
    fn value(&mut self, kind: u32) -> Result<Value, String> {
        Ok(match kind {
            0 => json!(self.array::<1>()?[0]),
            1 => json!(i8::from_le_bytes(self.array()?)),
            2 => json!(u16::from_le_bytes(self.array()?)),
            3 => json!(i16::from_le_bytes(self.array()?)),
            4 => json!(self.u32()?),
            5 => json!(i32::from_le_bytes(self.array()?)),
            6 => json!(f32::from_le_bytes(self.array()?)),
            7 => json!(self.array::<1>()?[0] != 0),
            8 => Value::String(self.string()?),
            9 => {
                let item = self.u32()?;
                let count = self.u64()?;
                if count > 1 << 24 {
                    return Err("a header array is too long".into());
                }
                Value::Array(
                    (0..count)
                        .map(|_| self.value(item))
                        .collect::<Result<_, _>>()?,
                )
            }
            10 => json!(self.u64()?),
            11 => json!(i64::from_le_bytes(self.array()?)),
            12 => json!(f64::from_le_bytes(self.array()?)),
            other => return Err(format!("metadata type {other} is not GGUF")),
        })
    }
}
