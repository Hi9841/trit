//! GGUF files (ggml's model format, as transcribe.cpp writes them for
//! Parakeet), read as published: the metadata gives the config and the
//! tokenizer, and the tensors keep their stored types (F32, F16, Q8_0)
//! under `Precision::Auto`, so a Q8_0 file runs as Q8_0
//! (`Precision::Q8_0`). The tensor names become the names of the Hugging
//! Face Parakeet checkpoint, so the Parakeet loader runs unchanged.
//!
//! Read: Parakeet TDT and RNNT with full attention (`general.architecture`
//! `parakeet`). Other architectures and streaming-only models are an error.

use crate::cpu::tensors::Tensors;
use crate::models::config::Json;
use crate::text::sentencepiece::Tokenizer;
use crate::{Device, Precision};
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
            let (dtype, bytes) = match kind {
                0 => ("F32", count * 4),
                1 => ("F16", count * 2),
                8 if count % 32 == 0 => ("Q8_0", count / 32 * 34),
                30 => ("BF16", count * 2),
                other => {
                    return Err(fail(format!(
                        "tensor {name} has ggml type {other}; Trit reads F32, F16, BF16, and Q8_0"
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

    fn get(&self, key: &str) -> &Value {
        self.meta.get(key).unwrap_or(&Value::Null)
    }

    fn num(&self, key: &str) -> Result<u64, String> {
        self.get(key)
            .as_u64()
            .ok_or_else(|| format!("{} has no {key}", self.path.display()))
    }

    /// The config of the Hugging Face checkpoint of this model, after
    /// checking that it has the shape the FastConformer runs.
    pub fn config(&self) -> Result<Json, String> {
        let unsupported = |what: &str| format!("{}: {what}", self.path.display());
        if self.get("general.architecture").as_str() != Some("parakeet") {
            return Err(unsupported(
                "not a Parakeet GGUF; Trit reads Parakeet TDT and RNNT GGUF files",
            ));
        }
        let head = self.get("stt.parakeet.head_kind").as_str().unwrap_or("tdt");
        let model_type = match head {
            "tdt" => "parakeet_tdt",
            "rnnt" => "parakeet_rnnt",
            other => return Err(unsupported(&format!("head `{other}` is not read"))),
        };
        let enc = |key: &str| self.num(&format!("stt.parakeet.encoder.{key}"));
        let full = ["att_context_left", "att_context_right"].iter().all(|key| {
            self.get(&format!("stt.parakeet.encoder.{key}"))
                .as_i64()
                .is_none_or(|side| side == -1)
        });
        let text = |key: &str| self.get(key).as_str().unwrap_or_default();
        let float = |key: &str| self.get(key).as_f64().unwrap_or(f64::NAN);
        if !full {
            return Err(unsupported(
                "its attention context is limited (a streaming model)",
            ));
        }
        if !matches!(
            text("stt.parakeet.encoder.conv_norm_type"),
            "" | "batch_norm"
        ) || self
            .tensors
            .iter()
            .any(|(name, ..)| name.starts_with("prompt."))
        {
            return Err(unsupported("its encoder layers differ from Parakeet's"));
        }
        let mels = self.num("stt.frontend.num_mels")?;
        if text("stt.frontend.type") != "mel"
            || self.num("stt.frontend.sample_rate")? != 16_000
            || self.num("stt.frontend.n_fft")? != 512
            || self.num("stt.frontend.win_length")? != 400
            || self.num("stt.frontend.hop_length")? != 160
            || text("stt.frontend.window") != "hann"
            || text("stt.frontend.normalize") != "per_feature"
            || (float("stt.frontend.pre_emphasis") - 0.97).abs() > 1e-6
        {
            return Err(unsupported("its audio features differ from Parakeet's"));
        }
        let mut config = json!({
            "model_type": model_type,
            "vocab_size": self.num("stt.parakeet.predictor.vocab")?,
            "blank_token_id": self.num("tokenizer.ggml.blank_token_id")?,
            "decoder_hidden_size": self.num("stt.parakeet.predictor.hidden")?,
            "num_decoder_layers": self.num("stt.parakeet.predictor.n_layers")?,
            "max_symbols_per_step": self.get("stt.parakeet.tdt.max_symbols").as_u64().unwrap_or(10),
            "encoder_config": {
                "hidden_size": enc("d_model")?,
                "intermediate_size": enc("d_ff")?,
                "num_attention_heads": enc("n_heads")?,
                "num_hidden_layers": enc("n_layers")?,
                "num_mel_bins": mels,
                "conv_kernel_size": enc("conv_kernel")?,
                "subsampling_factor": enc("subsampling_factor")?,
                "subsampling_conv_channels": enc("subsampling_channels")?,
                "subsampling_conv_kernel_size": 3,
                "subsampling_conv_stride": 2,
                "scale_input": self.get("stt.parakeet.encoder.xscaling").as_bool().unwrap_or(false),
            },
        });
        if model_type == "parakeet_tdt" {
            config["durations"] = self.get("stt.parakeet.tdt.durations").clone();
        }
        Ok(Json::from_value(config))
    }

    /// The tensors under their Hugging Face names, in their stored types.
    pub fn tensors(&self, precision: Precision, device: Device) -> Result<Tensors, String> {
        let mut renamed = Vec::with_capacity(self.tensors.len());
        for (name, dtype, start, end) in &self.tensors {
            let hf = hf_name(name).ok_or_else(|| {
                format!("{}: Trit does not read tensor {name}", self.path.display())
            })?;
            renamed.push((hf, dtype.clone(), *start, *end));
        }
        Tensors::open_gguf(&self.path, renamed, precision, device)
    }

    /// The tokenizer from `tokenizer.ggml.tokens`, `scores`, and `token_type`.
    pub fn tokenizer(&self) -> Result<Tokenizer, String> {
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
        if scores.len() != pieces.len() || kinds.len() != pieces.len() {
            return Err(format!(
                "{}: the tokenizer lists differ in length",
                self.path.display()
            ));
        }
        Tokenizer::from_pieces(pieces, &scores, &kinds)
    }
}

/// A transcribe.cpp tensor name as the Hugging Face Parakeet checkpoint
/// names it (`scripts/convert-parakeet.py` names them from NeMo's), or
/// `None` for a tensor Trit does not run.
fn hf_name(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("enc.pre_encode.conv.") {
        return Some(format!("encoder.subsampling.layers.{rest}"));
    }
    if let Some(rest) = name.strip_prefix("enc.pre_encode.out.") {
        return Some(format!("encoder.subsampling.linear.{rest}"));
    }
    if let Some(rest) = name.strip_prefix("enc.blocks.") {
        let (layer, part) = rest.split_once('.')?;
        const BLOCK: [(&str, &str); 18] = [
            ("norm_ff1.", "norm_feed_forward1."),
            ("ff1.", "feed_forward1."),
            ("norm_attn.", "norm_self_att."),
            ("attn.linear_q.", "self_attn.q_proj."),
            ("attn.linear_k.", "self_attn.k_proj."),
            ("attn.linear_v.", "self_attn.v_proj."),
            ("attn.linear_out.", "self_attn.o_proj."),
            ("attn.linear_pos.", "self_attn.relative_k_proj."),
            ("attn.pos_bias_u", "self_attn.bias_u"),
            ("attn.pos_bias_v", "self_attn.bias_v"),
            ("norm_conv.", "norm_conv."),
            ("conv.pointwise1.", "conv.pointwise_conv1."),
            ("conv.depthwise.", "conv.depthwise_conv."),
            ("conv.pointwise2.", "conv.pointwise_conv2."),
            ("conv.bn.", "conv.norm."),
            ("norm_ff2.", "norm_feed_forward2."),
            ("ff2.", "feed_forward2."),
            ("norm_out.", "norm_out."),
        ];
        let (from, to) = BLOCK.iter().find(|(from, _)| part.starts_with(from))?;
        return Some(format!(
            "encoder.layers.{layer}.{to}{}",
            &part[from.len()..]
        ));
    }
    if name == "pred.embed.weight" {
        return Some("decoder.embedding.weight".into());
    }
    if let Some(rest) = name.strip_prefix("pred.lstm.") {
        let (layer, part) = rest.split_once('.')?;
        let hf = match part {
            "Wx" => "weight_ih_l",
            "Wh" => "weight_hh_l",
            // The sum of PyTorch's two biases; the loader adds a zero bias_hh.
            "bias" => "bias_ih_l",
            _ => return None,
        };
        return Some(format!("decoder.lstm.{hf}{layer}"));
    }
    for (from, to) in [
        ("joint.enc.", "encoder_projector."),
        ("joint.pred.", "decoder.decoder_projector."),
        ("joint.out.", "joint.head."),
        ("vad.", "vad_head."),
    ] {
        if let Some(rest) = name.strip_prefix(from) {
            return Some(format!("{to}{rest}"));
        }
    }
    None
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
