//! NVIDIA NeMo archives (`.nemo`), read as NVIDIA publishes them: a tar of
//! `model_config.yaml`, the weights as a PyTorch checkpoint
//! (`model_weights.ckpt`), and the SentencePiece tokenizer
//! (`<hash>_tokenizer.model`). Nothing is unpacked: the checkpoint is read
//! where it lies in the tar.
//!
//! The archive becomes what Transformers' `convert_nemo_to_hf.py` would
//! write for an RNNT model: the config of `ParakeetRNNTConfig`, the tensor
//! names of `NEMO_TO_HF_WEIGHT_MAPPING` and `NEMO_TDT_WEIGHT_MAPPING`, and
//! the tokenizer's pieces and merges. The Parakeet loader then runs as for a
//! Hugging Face folder. Check, 2026-11-02: nvidia/parakeet-unified-en-0.6b
//! gives the same text as its converted folder on 502 of 502 clips.

use crate::cpu::tensors::Tensors;
use crate::models::config::Json;
use crate::{Device, Precision};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

const BLOCK: usize = 512;
/// What `convert_nemo_to_hf.py` writes for every transducer.
const MAX_SYMBOLS_PER_STEP: u64 = 10;

/// The `.nemo` file of a model folder: the one archive in a folder that has
/// no `config.json`, as a Hugging Face snapshot of an NVIDIA NeMo repository is.
pub fn find(dir: &Path) -> Option<PathBuf> {
    if dir.join("config.json").is_file() {
        return None;
    }
    let mut archives = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "nemo"));
    let first = archives.next()?;
    // Two archives: no way to tell which one the caller means.
    archives.next().is_none().then_some(first)
}

pub struct Archive {
    path: PathBuf,
    file: File,
    /// Member name (no leading `./`) to its byte range in the archive.
    members: HashMap<String, (usize, usize)>,
    /// `model_config.yaml`, as JSON values.
    yaml: Value,
}

impl Archive {
    pub fn open(path: &Path) -> Result<Self, String> {
        let error = |e: std::io::Error| format!("Could not read {}: {e}", path.display());
        let file = File::open(path).map_err(error)?;
        let total = file.metadata().map_err(error)?.len() as usize;
        let members = read_tar(&file, total).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut archive = Self {
            path: path.to_path_buf(),
            file,
            members,
            yaml: Value::Null,
        };
        let text = archive.text("model_config.yaml")?;
        archive.yaml = parse_yaml(&text).map_err(|e| format!("model_config.yaml: {e}"))?;
        Ok(archive)
    }

    /// The config that `convert_nemo_to_hf.py` writes for this model.
    pub fn config(&self) -> Result<Json, String> {
        let config = hf_config(&self.yaml, &self.path)?;
        // TDT archives (token durations) are not read yet: no TDT archive has
        // been checked against its converted folder.
        if config.get("durations").is_array() {
            return Err(format!(
                "{}: it is a TDT model; Trit runs this NeMo shape only through a Hugging Face folder",
                self.path.display()
            ));
        }
        Ok(config)
    }

    /// The weights, under the names `convert_nemo_to_hf.py` gives them.
    pub fn tensors(&self, precision: Precision, device: Device) -> Result<Tensors, String> {
        let range = self.member("model_weights.ckpt")?;
        Tensors::open_torch_in(&self.path, range, hf_name, precision, device)
    }

    /// The files a Hugging Face folder of this model holds.
    pub fn provides(&self, name: &str) -> bool {
        matches!(name, "config.json" | "tokenizer.json")
    }

    /// `config.json`, or `tokenizer.json` from the SentencePiece model.
    pub fn file(&self, name: &str) -> Result<String, String> {
        match name {
            "config.json" => Ok(self.config()?.to_text()),
            "tokenizer.json" => self.tokenizer_json(),
            _ => Err(format!("{} has no {name}", self.path.display())),
        }
    }

    /// The SentencePiece model that `tokenizer.model_path` names.
    fn tokenizer_json(&self) -> Result<String, String> {
        let name = self.yaml["tokenizer"]["model_path"]
            .as_str()
            .map(|path| path.trim_start_matches("nemo:").to_string())
            .ok_or("model_config.yaml has no tokenizer.model_path")?;
        let range = self.member(&name)?;
        crate::text::sentencepiece::sentencepiece_json(&self.read(range)?)
            .map_err(|e| format!("{}: {name}: {e}", self.path.display()))
    }

    fn member(&self, name: &str) -> Result<(usize, usize), String> {
        self.members
            .get(name)
            .copied()
            .ok_or_else(|| format!("{} has no {name}", self.path.display()))
    }

    fn read(&self, (start, len): (usize, usize)) -> Result<Vec<u8>, String> {
        let mut bytes = vec![0u8; len];
        crate::cpu::tensors::read_at(&self.file, &mut bytes, start as u64)
            .map_err(|e| format!("Could not read {}: {e}", self.path.display()))?;
        Ok(bytes)
    }

    fn text(&self, name: &str) -> Result<String, String> {
        String::from_utf8(self.read(self.member(name)?)?)
            .map_err(|_| format!("{name} in {} is not UTF-8", self.path.display()))
    }
}

/// The config that `convert_nemo_to_hf.py` writes for NeMo config `y`
/// (`model_config.yaml` as JSON) of the model at `path`, after checking
/// that the model has the shape the FastConformer runs: RNNT, or TDT when
/// the config lists token durations.
pub fn hf_config(y: &Value, path: &Path) -> Result<Json, String> {
    let unsupported = |what: &str| {
        format!(
            "{}: {what}; Trit runs this NeMo shape only through a Hugging Face folder",
            path.display()
        )
    };
    let encoder = &y["encoder"];
    let text = |v: &Value| v.as_str().unwrap_or_default().to_string();
    let num = |v: &Value, what: &str| {
        v.as_u64()
            .ok_or_else(|| format!("model_config.yaml has no {what}"))
    };
    if !text(&encoder["_target_"]).ends_with("ConformerEncoder") {
        return Err(unsupported("the encoder is not a Conformer"));
    }
    if text(&encoder["subsampling"]) != "dw_striding"
        || encoder["causal_downsampling"].as_bool() == Some(true)
        || text(&encoder["self_attention_model"]) != "rel_pos"
        || encoder["untie_biases"].as_bool() == Some(false)
        || !matches!(text(&encoder["conv_norm_type"]).as_str(), "" | "batch_norm")
    {
        return Err(unsupported("the encoder layers differ from Parakeet's"));
    }
    // Offline transcription attends to the whole clip: [-1, -1]. A
    // limited left or right context is a streaming model.
    let context = &encoder["att_context_size"];
    let full = match context.as_array() {
        None => true,
        Some(pair) => pair.iter().all(|side| side.as_i64() == Some(-1)),
    };
    if !full {
        return Err(unsupported(
            "its attention context is limited (a streaming model)",
        ));
    }
    let pre = &y["preprocessor"];
    let float = |v: &Value, default: f64| v.as_f64().unwrap_or(default);
    if num(&pre["sample_rate"], "preprocessor.sample_rate")? != 16_000
        || num(&pre["n_fft"], "preprocessor.n_fft")? != 512
        || (float(&pre["window_size"], 0.0) - 0.025).abs() > 1e-9
        || (float(&pre["window_stride"], 0.0) - 0.01).abs() > 1e-9
        || !matches!(text(&pre["window"]).as_str(), "" | "hann")
        || text(&pre["normalize"]) != "per_feature"
        || (float(&pre["preemph"], 0.97) - 0.97).abs() > 1e-9
        || pre["features"] != encoder["feat_in"]
    {
        return Err(unsupported("its audio features differ from Parakeet's"));
    }
    let hidden = num(&encoder["d_model"], "encoder.d_model")?;
    let encoder_config = json!({
        "hidden_size": hidden,
        "intermediate_size": hidden * num(&encoder["ff_expansion_factor"], "encoder.ff_expansion_factor")?,
        "num_attention_heads": num(&encoder["n_heads"], "encoder.n_heads")?,
        "num_hidden_layers": num(&encoder["n_layers"], "encoder.n_layers")?,
        "num_mel_bins": num(&encoder["feat_in"], "encoder.feat_in")?,
        "conv_kernel_size": num(&encoder["conv_kernel_size"], "encoder.conv_kernel_size")?,
        "subsampling_factor": num(&encoder["subsampling_factor"], "encoder.subsampling_factor")?,
        "subsampling_conv_channels": num(&encoder["subsampling_conv_channels"], "encoder.subsampling_conv_channels")?,
        // NeMo's dw_striding stack: 3x3 kernels, stride 2.
        "subsampling_conv_kernel_size": 3,
        "subsampling_conv_stride": 2,
        "scale_input": encoder["xscaling"].as_bool().unwrap_or(true),
    });
    let decoder = &y["decoder"];
    if !text(&decoder["_target_"]).ends_with("RNNTDecoder")
        || !text(&y["joint"]["_target_"]).ends_with("RNNTJoint")
    {
        return Err(unsupported("it has no RNNT decoder"));
    }
    let labels = y["labels"]
        .as_array()
        .or_else(|| y["joint"]["vocabulary"].as_array())
        .ok_or("model_config.yaml has no labels")?;
    // The blank comes after the labels.
    let blank = labels.len();
    let prednet = &decoder["prednet"];
    let durations = y["decoding"]["durations"]
        .as_array()
        .or_else(|| y["model_defaults"]["tdt_durations"].as_array())
        .filter(|list| !list.is_empty());
    let mut config = json!({
        "model_type": if durations.is_some() { "parakeet_tdt" } else { "parakeet_rnnt" },
        "vocab_size": blank + 1,
        "blank_token_id": blank,
        "decoder_hidden_size": prednet["pred_hidden"].as_u64().unwrap_or(640),
        "num_decoder_layers": prednet["pred_rnn_layers"].as_u64().unwrap_or(2),
        "max_symbols_per_step": MAX_SYMBOLS_PER_STEP,
        "encoder_config": encoder_config,
    });
    if let Some(durations) = durations {
        config["durations"] = json!(durations);
    }
    Ok(Json::from_value(config))
}

/// A NeMo tensor name as `convert_nemo_to_hf.py` renames it (its two
/// mappings, applied in their order), or `None` for the feature extractor's
/// buffers, which Trit computes itself.
fn hf_name(nemo: &str) -> Option<String> {
    if nemo.starts_with("preprocessor.") {
        return None;
    }
    let mut name = nemo
        .replacen("encoder.pre_encode.conv.", "encoder.subsampling.layers.", 1)
        .replacen("encoder.pre_encode.out.", "encoder.subsampling.linear.", 1)
        .replacen("encoder.pos_enc.", "encoder.encode_positions.", 1)
        .replace(".conv.batch_norm.", ".conv.norm.")
        .replace("linear_k", "k_proj")
        .replace("linear_v", "v_proj")
        .replace("linear_out", "o_proj")
        .replace("linear_q", "q_proj")
        .replace("pos_bias_u", "bias_u")
        .replace("pos_bias_v", "bias_v")
        .replace("linear_pos", "relative_k_proj");
    for (from, to) in [
        ("decoder.prediction.embed.", "decoder.embedding."),
        ("decoder.prediction.dec_rnn.lstm.", "decoder.lstm."),
        ("joint.enc.", "encoder_projector."),
        ("joint.pred.", "decoder.decoder_projector."),
        ("joint.joint_net.2.", "joint.head."),
    ] {
        if let Some(rest) = name.strip_prefix(from) {
            name = format!("{to}{rest}");
        }
    }
    Some(name)
}

/// Member name to byte range, from ustar headers with PAX and GNU long names.
fn read_tar(file: &File, total: usize) -> Result<HashMap<String, (usize, usize)>, String> {
    let read = |at: usize, len: usize| -> Result<Vec<u8>, String> {
        let mut bytes = vec![0u8; len];
        crate::cpu::tensors::read_at(file, &mut bytes, at as u64)
            .map_err(|e| format!("cannot read the archive: {e}"))?;
        Ok(bytes)
    };
    let mut members = HashMap::new();
    let mut at = 0;
    // A name or size that a PAX or GNU header gives the next member.
    let mut long_name: Option<String> = None;
    let mut long_size: Option<usize> = None;
    while at + BLOCK <= total {
        let header = read(at, BLOCK)?;
        if header.iter().all(|b| *b == 0) {
            break;
        }
        if &header[257..262] != b"ustar" {
            return Err("not a tar archive (a .nemo file is one)".into());
        }
        let field = |range: std::ops::Range<usize>| {
            let raw = &header[range];
            let end = raw.iter().position(|b| *b == 0).unwrap_or(raw.len());
            String::from_utf8_lossy(&raw[..end]).into_owned()
        };
        let size = match long_size.take() {
            Some(size) => size,
            None => octal(&header[124..136])?,
        };
        let data = at + BLOCK;
        // Checked: a base-256 size near 2^64 wrapped past this test.
        if data.checked_add(size).is_none_or(|end| end > total) {
            return Err("a tar member runs past the end of the file".into());
        }
        match header[156] {
            b'x' => {
                for (key, value) in pax_records(&read(data, size)?) {
                    match key.as_str() {
                        "path" => long_name = Some(value),
                        "size" => long_size = value.parse().ok(),
                        _ => {}
                    }
                }
            }
            b'L' => {
                let raw = read(data, size)?;
                let end = raw.iter().position(|b| *b == 0).unwrap_or(raw.len());
                long_name = Some(String::from_utf8_lossy(&raw[..end]).into_owned());
            }
            b'0' | 0 => {
                let name = long_name.take().unwrap_or_else(|| {
                    let prefix = field(345..500);
                    let name = field(0..100);
                    if prefix.is_empty() {
                        name
                    } else {
                        format!("{prefix}/{name}")
                    }
                });
                let name = name.trim_start_matches("./").to_string();
                members.insert(name, (data, size));
            }
            _ => long_name = None,
        }
        at = data + size.div_ceil(BLOCK) * BLOCK;
    }
    Ok(members)
}

/// A tar number: octal digits, or base-256 when the high bit is set.
fn octal(raw: &[u8]) -> Result<usize, String> {
    if raw[0] & 0x80 != 0 {
        return Ok(raw[1..]
            .iter()
            .fold(0usize, |n, b| (n << 8) | usize::from(*b)));
    }
    let digits: String = raw
        .iter()
        .take_while(|b| **b != 0 && **b != b' ')
        .map(|b| char::from(*b))
        .collect();
    let digits = digits.trim();
    if digits.is_empty() {
        return Ok(0);
    }
    usize::from_str_radix(digits, 8).map_err(|_| "a tar header has a bad size".to_string())
}

/// `<length> <key>=<value>\n` records of a PAX header.
fn pax_records(data: &[u8]) -> Vec<(String, String)> {
    let text = String::from_utf8_lossy(data);
    text.lines()
        .filter_map(|line| {
            let (_, record) = line.split_once(' ')?;
            let (key, value) = record.split_once('=')?;
            Some((key.to_string(), value.to_string()))
        })
        .collect()
}

/// The YAML that OmegaConf writes for a NeMo config: block mappings, block
/// sequences (also nested, `- - 70`), and plain or quoted scalars. Other
/// YAML (anchors, flow collections with items, block scalars) is an error.
fn parse_yaml(text: &str) -> Result<Value, String> {
    let mut lines: Vec<(usize, String)> = text
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        })
        .map(|line| {
            let indent = line.len() - line.trim_start_matches(' ').len();
            (indent, line[indent..].trim_end().to_string())
        })
        .collect();
    let mut at = 0;
    if lines.is_empty() {
        return Ok(Value::Null);
    }
    let indent = lines[0].0;
    let value = block(&mut lines, &mut at, indent)?;
    if at != lines.len() {
        return Err(format!("line {} is not indented as YAML", at + 1));
    }
    Ok(value)
}

fn is_item(text: &str) -> bool {
    text == "-" || text.starts_with("- ")
}

/// The mapping or sequence whose lines start at `indent`.
fn block(lines: &mut [(usize, String)], at: &mut usize, indent: usize) -> Result<Value, String> {
    if is_item(&lines[*at].1) {
        let mut items = Vec::new();
        while *at < lines.len() && lines[*at].0 == indent && is_item(&lines[*at].1) {
            let rest = lines[*at].1[1..].trim_start().to_string();
            if rest.is_empty() {
                *at += 1;
                let child = lines.get(*at).map(|line| line.0).filter(|i| *i > indent);
                items.push(match child {
                    Some(child) => block(lines, at, child)?,
                    None => Value::Null,
                });
            } else if is_item(&rest) || key_value(&rest).is_some() {
                // `- - 70` or `- key: value`: the rest opens a block two columns in.
                lines[*at] = (indent + 2, rest);
                items.push(block(lines, at, indent + 2)?);
            } else {
                items.push(scalar(&rest)?);
                *at += 1;
            }
        }
        return Ok(Value::Array(items));
    }
    let mut map = serde_json::Map::new();
    while *at < lines.len() && lines[*at].0 == indent && !is_item(&lines[*at].1) {
        let line = lines[*at].1.clone();
        let (key, value) =
            key_value(&line).ok_or_else(|| format!("line {} is not `key: value`", *at + 1))?;
        *at += 1;
        let value = if value.is_empty() {
            match lines.get(*at) {
                Some((child, _)) if *child > indent => block(lines, at, *child)?,
                // OmegaConf writes a sequence at the indent of its key.
                Some((child, text)) if *child == indent && is_item(text) => {
                    block(lines, at, indent)?
                }
                _ => Value::Null,
            }
        } else {
            scalar(value)?
        };
        map.insert(unquote(key), value);
    }
    Ok(Value::Object(map))
}

/// `key: value` or `key:`, split at the first `: ` outside quotes.
fn key_value(line: &str) -> Option<(&str, &str)> {
    if line.starts_with(['\'', '"']) {
        let quote = line.chars().next()?;
        let end = line[1..].find(quote)? + 1;
        let rest = line[end + 1..].strip_prefix(':')?;
        return Some((&line[..=end], rest.trim()));
    }
    if let Some(key) = line.strip_suffix(':') {
        return (!key.contains(": ")).then_some((key, ""));
    }
    let (key, value) = line.split_once(": ")?;
    Some((key, value.trim()))
}

fn unquote(text: &str) -> String {
    let bytes = text.as_bytes();
    match bytes.first() {
        Some(b'\'') if text.len() >= 2 && text.ends_with('\'') => {
            text[1..text.len() - 1].replace("''", "'")
        }
        Some(b'"') if text.len() >= 2 && text.ends_with('"') => text[1..text.len() - 1]
            .replace("\\\"", "\"")
            .replace("\\\\", "\\"),
        _ => text.to_string(),
    }
}

fn scalar(text: &str) -> Result<Value, String> {
    Ok(match text {
        "null" | "~" => Value::Null,
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        "[]" => Value::Array(Vec::new()),
        "{}" => Value::Object(serde_json::Map::new()),
        _ if text.starts_with(['\'', '"']) => Value::String(unquote(text)),
        _ if text.starts_with(['[', '{', '&', '*', '|', '>']) => {
            return Err(format!("`{text}`: this YAML form is not read"));
        }
        _ => {
            if let Ok(int) = text.parse::<i64>() {
                json!(int)
            } else if let Some(float) = text.parse::<f64>().ok().filter(|f| f.is_finite()) {
                json!(float)
            } else {
                Value::String(text.to_string())
            }
        }
    })
}
