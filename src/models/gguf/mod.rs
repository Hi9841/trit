//! GGUF files (ggml's model format, as transcribe.cpp writes them), read as
//! published: the metadata gives the files of a Hugging Face folder
//! (`config.json`, `tokenizer.json`, ...), built by the module of the
//! file's `general.architecture`, and the tensors take the Hugging Face
//! names and keep their stored types under `Precision::Auto` (a Q8_0 file
//! runs as Q8_0). The family loaders then run as for a Hugging Face folder.
//! The architectures are the rows of `ARCH`.

use super::package::Package;
use crate::cpu::tensors::Tensors;
use crate::{Device, Precision};

pub(crate) mod blocks;

mod canary;
mod canary_qwen;
mod cohere;
mod fun_asr;
mod gigaam;
mod granite_ctc;
mod granite_nar;
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

/// What a tensor of the file becomes.
pub(super) enum Name {
    /// The Hugging Face name.
    To(String),
    /// A tensor Trit computes itself (a mel filterbank, a window).
    Skip,
}

type Build = fn(&Gguf) -> Result<Value, String>;

/// One `general.architecture`: who writes its config, who renames its tensors,
/// and the files it has besides `config.json` and `tokenizer.json`.
struct Arch {
    name: &'static str,
    config: Build,
    rename: fn(&str) -> Option<Name>,
    extra: &'static [(&'static str, Build)],
}

const fn arch(name: &'static str, config: Build, rename: fn(&str) -> Option<Name>) -> Arch {
    Arch {
        name,
        config,
        rename,
        extra: &[],
    }
}

const ARCH: &[Arch] = &[
    arch("parakeet", parakeet::config, parakeet::hf_name),
    Arch {
        extra: &[("generation_config.json", whisper::generation)],
        ..arch("whisper", whisper::config, whisper::hf_name)
    },
    arch("moonshine", moonshine::config, moonshine::hf_name),
    arch("canary", canary::config, canary::hf_name),
    arch("qwen3_asr", qwen3_asr::config, qwen3_asr::hf_name),
    arch("gigaam", gigaam::config, gigaam::hf_name),
    arch("sensevoice", sensevoice::config, sensevoice::hf_name),
    arch("medasr", medasr::config, medasr::hf_name),
    arch(
        "granite_speech5_ctc",
        granite_ctc::config,
        granite_ctc::hf_name,
    ),
    arch("canary_qwen", canary_qwen::config, canary_qwen::hf_name),
    arch("funasr_nano", fun_asr::config, fun_asr::hf_name),
    arch(
        "moonshine_streaming",
        moonshine::streaming_config,
        moonshine::hf_name,
    ),
    arch(
        "granite_speech",
        granite_speech::config,
        granite_speech::hf_name,
    ),
    arch("voxtral", voxtral::config, voxtral::hf_name),
    arch("cohere_asr", cohere::config, cohere::hf_name),
    arch(
        "voxtral_realtime",
        voxtral_realtime::config,
        voxtral_realtime::hf_name,
    ),
    arch("moss", moss::config, moss::hf_name),
    arch(
        "granite_speech_nar",
        granite_nar::config,
        granite_nar::hf_name,
    ),
];

/// The first table row whose source prefix `part` has, with that prefix
/// replaced. A shorter source must come after a longer one (`attn.q` after
/// `attn.q_norm`).
pub(super) fn map_suffix(part: &str, table: &[(&str, &str)]) -> Option<String> {
    let (from, to) = table.iter().find(|(from, _)| part.starts_with(from))?;
    Some(format!("{to}{}", &part[from.len()..]))
}

/// Every `from` in `name`, in order. A later row sees the earlier replacements.
pub(super) fn replace_all(name: &str, table: &[(&str, &str)]) -> String {
    table
        .iter()
        .fold(name.to_string(), |name, (from, to)| name.replace(from, to))
}

fn arch_names() -> String {
    let names: Vec<&str> = ARCH.iter().map(|row| row.name).collect();
    let (last, head) = names.split_last().expect("ARCH has a row");
    format!("{}, and {last}", head.join(", "))
}
use serde_json::{json, Map, Value};
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
/// The smallest tensor entry of the header: an empty name, no dimensions, a
/// type, and an offset.
const MIN_TENSOR_ENTRY: u64 = 24;
/// Arrays inside arrays. GGUF writers use one level; each level recurses.
const MAX_ARRAY_DEPTH: usize = 4;

/// The `.gguf` file of a model folder: the one GGUF file in a folder that
/// has no `config.json`.
pub fn find(dir: &Path) -> Option<PathBuf> {
    if dir.join("config.json").is_file() {
        return None;
    }
    super::package::only_file(dir, "gguf")
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
            meta.insert(key, reader.value(kind, 0).map_err(fail)?);
        }
        if tensor_count > total / MIN_TENSOR_ENTRY {
            return Err(fail("the tensor list runs past the end of the file".into()));
        }
        let mut infos = Vec::new();
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
        if alignment == 0 {
            return Err(fail("general.alignment is 0".into()));
        }
        let here = reader.position().map_err(fail)?;
        let data = here
            .checked_next_multiple_of(alignment)
            .ok_or_else(|| fail("the tensor data runs past the end of the file".into()))?;
        let mut tensors = Vec::with_capacity(infos.len());
        for (name, kind, count, offset) in infos {
            let Some((dtype, bytes)) = crate::cpu::ggml_quants::layout(kind, count) else {
                return Err(fail(format!(
                    "tensor {name} has ggml type {kind}; Trit reads {}",
                    crate::cpu::ggml_quants::type_list()
                )));
            };
            let span = data.checked_add(offset).and_then(|start| {
                let end = start.checked_add(bytes?)?;
                (end <= total).then_some((start, end))
            });
            let Some((start, end)) = span else {
                return Err(fail(format!("tensor {name} runs past the end of the file")));
            };
            tensors.push((name, dtype.to_string(), start as usize, end as usize));
        }
        Ok(Self {
            path: path.to_path_buf(),
            meta,
            tensors,
        })
    }

    fn row(&self) -> Result<&Arch, String> {
        let name = self.get("general.architecture").as_str();
        ARCH.iter()
            .find(|row| Some(row.name) == name)
            .ok_or_else(|| {
                format!(
                    "{}: GGUF architecture {} is not read; Trit reads {}",
                    self.path.display(),
                    name.unwrap_or("(none)"),
                    arch_names()
                )
            })
    }

    pub(super) fn get(&self, key: &str) -> &Value {
        self.meta.get(key).unwrap_or(&Value::Null)
    }

    pub(super) fn num(&self, key: &str) -> Result<u64, String> {
        self.get(key).as_u64().ok_or_else(|| format!("no {key}"))
    }

    /// The id of token `text` in `tokenizer.ggml.tokens`.
    pub(super) fn token(&self, text: &str) -> Option<usize> {
        self.get("tokenizer.ggml.tokens")
            .as_array()?
            .iter()
            .position(|token| token.as_str() == Some(text))
    }
}

impl Package for Gguf {
    fn provides(&self, name: &str) -> bool {
        let Ok(row) = self.row() else {
            return false;
        };
        name == "config.json"
            || name == "tokenizer.json"
            || row.extra.iter().any(|(file, _)| *file == name)
    }

    fn file(&self, name: &str) -> Result<String, String> {
        let row = self.row()?;
        let fail = |e: String| format!("{}: {e}", self.path.display());
        if name == "tokenizer.json" {
            return self.tokenizer_json();
        }
        if name == "config.json" {
            return (row.config)(self).map(|v| v.to_string()).map_err(fail);
        }
        if let Some((_, build)) = row.extra.iter().find(|(file, _)| *file == name) {
            return build(self).map(|v| v.to_string()).map_err(fail);
        }
        Err(format!("{} has no {name}", self.path.display()))
    }

    fn tensors(&self, precision: Precision, device: Device) -> Result<Tensors, String> {
        let row = self.row()?;
        let mut renamed = Vec::with_capacity(self.tensors.len());
        for (name, dtype, start, end) in &self.tensors {
            match (row.rename)(name) {
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
}

impl Gguf {
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

    /// One metadata value of GGUF type `kind`, inside `depth` arrays.
    fn value(&mut self, kind: u32, depth: usize) -> Result<Value, String> {
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
                if depth >= MAX_ARRAY_DEPTH {
                    return Err("the header arrays are nested too deep".into());
                }
                let item = self.u32()?;
                let count = self.u64()?;
                if count > 1 << 24 {
                    return Err("a header array is too long".into());
                }
                Value::Array(
                    (0..count)
                        .map(|_| self.value(item, depth + 1))
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
