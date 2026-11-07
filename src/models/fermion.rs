//! Fermion containers, as Fermion Research publishes Phonon-2 (a Parakeet
//! TDT 0.6B v3 with a five-value encoder): a folder with `model.fermion`
//! and the NeMo config as `config.json`.
//!
//! `model.fermion` is a little-endian u64 header length, a JSON header
//! (`format`, and an `index` of records: name `n`, kind `k`, `shape`, byte
//! count `b`), then the records in index order. The records carry the
//! names of Transformers' `ParakeetForTDT`; a `five_value` record is the
//! `.weight` of the module it names. `cpu::fermion` expands them.
//!
//! The publisher runs the container with the config, processor, and
//! tokenizer of nvidia/parakeet-tdt-0.6b-v3. Trit gives the same: the
//! config from the NeMo config (`nemo::hf_config`), the tokenizer from its
//! `labels`, which are the base model's pieces in order.

use crate::config::Json;
use crate::cpu::fermion::{self, Record};
use crate::cpu::tensors::Tensors;
use crate::{Device, Precision};
use serde_json::Value;
use std::path::{Path, PathBuf};

const CONTAINER: &str = "model.fermion";
const FORMAT: &str = "fermion-five-value-parakeet-v1";

/// The container of a model folder, when it has one.
///
/// Phonon ships `model.fermion` next to `config.json`. NeMo and GGUF skip a
/// folder that has `config.json`. This check does not, or a Phonon folder
/// would look like a plain folder.
pub fn find(dir: &Path) -> Option<PathBuf> {
    let path = dir.join(CONTAINER);
    path.is_file().then_some(path)
}

pub struct Container {
    path: PathBuf,
    /// The NeMo config (`config.json` beside the container).
    nemo: Value,
    records: Vec<Record>,
}

impl Container {
    pub fn open(path: &Path) -> Result<Self, String> {
        let error = |e: std::io::Error| format!("Could not read {}: {e}", path.display());
        let file = std::fs::File::open(path).map_err(error)?;
        let total = file.metadata().map_err(error)?.len() as usize;
        let mut length = [0u8; 8];
        crate::cpu::tensors::read_at(&file, &mut length, 0).map_err(error)?;
        let header_len = usize::try_from(u64::from_le_bytes(length))
            .ok()
            .filter(|len| 8 + len <= total)
            .ok_or_else(|| format!("{}: the header runs past the end", path.display()))?;
        let mut header = vec![0u8; header_len];
        crate::cpu::tensors::read_at(&file, &mut header, 8).map_err(error)?;
        let header: Value = serde_json::from_slice(&header)
            .map_err(|e| format!("{}: the header is not JSON: {e}", path.display()))?;
        if header["format"] != FORMAT {
            return Err(format!(
                "{}: format {} is not read (only {FORMAT})",
                path.display(),
                header["format"]
            ));
        }
        let mut records = Vec::new();
        let mut at = 8 + header_len;
        for entry in header["index"].as_array().into_iter().flatten() {
            let bad = || format!("{}: a header record is malformed", path.display());
            let name = entry["n"].as_str().ok_or_else(bad)?;
            let kind = entry["k"].as_str().ok_or_else(bad)?;
            let len = entry["b"].as_u64().ok_or_else(bad)? as usize;
            let shape: Vec<usize> = entry["shape"]
                .as_array()
                .ok_or_else(bad)?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize).ok_or_else(bad))
                .collect::<Result<_, _>>()?;
            if fermion::dtype(kind).is_none() {
                return Err(format!(
                    "{}: record kind {kind} is not read",
                    path.display()
                ));
            }
            // A five_value record holds a matrix; an intN record a row per
            // output, whatever the rest of its shape.
            let rows = shape.first().copied().unwrap_or(1);
            let count: usize = shape.iter().product();
            if (kind == "five_value" && shape.len() != 2) || rows == 0 {
                return Err(format!(
                    "{}: record {name} has shape {shape:?}",
                    path.display()
                ));
            }
            let end = at
                .checked_add(len)
                .filter(|end| *end <= total)
                .ok_or_else(|| format!("{}: record {name} runs past the end", path.display()))?;
            records.push(Record {
                name: match kind {
                    "five_value" => format!("{name}.weight"),
                    _ => name.to_string(),
                },
                kind: kind.to_string(),
                matrix: (rows, count / rows),
                start: at,
                end,
            });
            at = end;
        }
        let config = path.with_file_name("config.json");
        let text = std::fs::read_to_string(&config)
            .map_err(|e| format!("Could not read {}: {e}", config.display()))?;
        let nemo = serde_json::from_str(&text)
            .map_err(|e| format!("{} is not JSON: {e}", config.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            nemo,
            records,
        })
    }

    fn config(&self) -> Result<Json, String> {
        let config = crate::models::nemo::hf_config(&self.nemo, &self.path)?;
        if !config.get("durations").is_array() {
            return Err(format!(
                "{}: the config lists no TDT durations",
                self.path.display()
            ));
        }
        Ok(config)
    }

    /// The base model's SentencePiece tokenizer from `labels` and the blank
    /// after them. Pieces are ranked by id, as the base model's scores rank
    /// them: only encoding (dictionary terms) reads the rank.
    fn tokenizer_json(&self) -> Result<String, String> {
        let labels = self.nemo["labels"]
            .as_array()
            .ok_or_else(|| format!("{}: config.json has no labels", self.path.display()))?;
        let mut pieces: Vec<String> = labels
            .iter()
            .map(|piece| piece.as_str().unwrap_or_default().to_string())
            .collect();
        pieces.push("<blank>".into());
        let scores: Vec<f32> = (0..pieces.len()).map(|id| -(id as f32)).collect();
        // SentencePiece kinds: 2 unknown, 3 control, 1 normal.
        let kinds: Vec<u64> = pieces
            .iter()
            .map(|piece| match piece.as_str() {
                "<unk>" => 2,
                "<blank>" => 3,
                _ => 1,
            })
            .collect();
        crate::text::sentencepiece::tokenizer_json(&pieces, &scores, &kinds)
            .map_err(|e| format!("{}: {e}", self.path.display()))
    }
}

impl super::package::Package for Container {
    fn provides(&self, name: &str) -> bool {
        matches!(name, "config.json" | "tokenizer.json")
    }

    fn file(&self, name: &str) -> Result<String, String> {
        match name {
            "config.json" => Ok(self.config()?.to_text()),
            "tokenizer.json" => self.tokenizer_json(),
            _ => Err(format!("{} has no {name}", self.path.display())),
        }
    }

    fn tensors(&self, precision: Precision, device: Device) -> Result<Tensors, String> {
        Tensors::open_fermion(&self.path, &self.records, precision, device)
    }
}
