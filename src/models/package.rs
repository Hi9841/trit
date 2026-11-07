//! A model that one published file holds, in place of `config.json` and
//! `model.safetensors`: an NVIDIA NeMo archive, a GGUF file, or a Fermion
//! container. Each gives the config, tensor names, and tokenizer of the
//! Hugging Face checkpoint.

use super::{fermion, gguf, nemo};
use crate::cpu::tensors::Tensors;
use crate::{Device, Precision};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub trait Package {
    /// Whether the package gives file `name` of a Hugging Face folder.
    fn provides(&self, name: &str) -> bool;
    /// File `name` of a Hugging Face folder of this model.
    fn file(&self, name: &str) -> Result<String, String>;
    fn tensors(&self, precision: Precision, device: Device) -> Result<Tensors, String>;
}

/// The package of a model folder: its `model.fermion`, or its one `.nemo` or
/// `.gguf` file when the folder has no `config.json`.
pub fn find(dir: &Path) -> Result<Option<Box<dyn Package>>, String> {
    fn boxed(package: impl Package + 'static) -> Option<Box<dyn Package>> {
        Some(Box::new(package))
    }
    match (fermion::find(dir), nemo::find(dir), gguf::find(dir)) {
        (Some(path), None, None) => fermion::Container::open(&path).map(boxed),
        (None, Some(path), None) => nemo::Archive::open(&path).map(boxed),
        (None, None, Some(path)) => gguf::Gguf::open(&path).map(boxed),
        (None, None, None) => Ok(None),
        _ => Err(format!(
            "{} holds more than one model file (.fermion, .nemo, .gguf); keep one model per folder",
            dir.display()
        )),
    }
}

pub fn in_folder(dir: &Path) -> bool {
    let found = [
        fermion::find(dir).is_some(),
        nemo::find(dir).is_some(),
        gguf::find(dir).is_some(),
    ];
    found.iter().filter(|found| **found).count() == 1
}

/// The one file in `dir` whose extension is `extension`. Two files of that
/// extension give no way to tell which one the caller means.
pub(crate) fn only_file(dir: &Path, extension: &str) -> Option<PathBuf> {
    let mut files = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == extension));
    let first = files.next()?;
    files.next().is_none().then_some(first)
}

/// The FastConformer encoder block (`subsampling_conv_kernel_size` 3,
/// `subsampling_conv_stride` 2: NeMo's dw_striding stack).
pub(crate) struct EncShape {
    pub hidden: u64,
    pub intermediate: u64,
    pub heads: u64,
    pub layers: u64,
    pub mels: u64,
    pub conv_kernel: u64,
    pub subsample: u64,
    pub sub_channels: u64,
    pub scale_input: bool,
}

pub(crate) fn encoder_block(shape: EncShape) -> Value {
    json!({
        "hidden_size": shape.hidden,
        "intermediate_size": shape.intermediate,
        "num_attention_heads": shape.heads,
        "num_hidden_layers": shape.layers,
        "num_mel_bins": shape.mels,
        "conv_kernel_size": shape.conv_kernel,
        "subsampling_factor": shape.subsample,
        "subsampling_conv_channels": shape.sub_channels,
        "subsampling_conv_kernel_size": 3,
        "subsampling_conv_stride": 2,
        "scale_input": shape.scale_input,
    })
}
