//! The files of one model, as the family loaders read them: a folder as
//! Hugging Face publishes it (`config.json`, `model.safetensors`,
//! `tokenizer.json`, ...), or a folder with one published `.nemo` or `.gguf`
//! file, which gives the same files from its metadata (`Package::file`) and
//! its tensors under the Hugging Face names. A family reads the same file
//! names either way, so one loader serves both.

use crate::cpu::tensors::Tensors;
use crate::models::Package;
use crate::{Device, Precision};
use std::path::{Path, PathBuf};

pub enum Files {
    Folder(PathBuf),
    Package(PathBuf, Package),
}

impl Files {
    pub fn open(dir: &Path) -> Result<Self, String> {
        Ok(match Package::find(dir)? {
            Some(package) => Self::Package(dir.to_path_buf(), package),
            None => Self::Folder(dir.to_path_buf()),
        })
    }

    /// Whether the model has file `name`.
    pub fn has(&self, name: &str) -> bool {
        match self {
            Self::Folder(dir) => dir.join(name).is_file(),
            Self::Package(_, package) => package.provides(name),
        }
    }

    /// The text of file `name`.
    pub fn read(&self, name: &str) -> Result<String, String> {
        match self {
            Self::Folder(dir) => std::fs::read_to_string(dir.join(name))
                .map_err(|e| format!("cannot read {}: {e}", dir.join(name).display())),
            Self::Package(dir, package) if package.provides(name) => package.file(name),
            Self::Package(dir, _) => Err(format!("{} has no {name}", dir.display())),
        }
    }

    /// File `name` as JSON.
    pub fn json(&self, name: &str) -> Result<serde_json::Value, String> {
        serde_json::from_str(&self.read(name)?).map_err(|e| format!("{name} is not JSON: {e}"))
    }

    /// The weights, under the Hugging Face names.
    pub fn tensors(&self, precision: Precision, device: Device) -> Result<Tensors, String> {
        match self {
            Self::Folder(dir) => Tensors::open_dir(dir, precision, device),
            Self::Package(_, package) => package.tensors(precision, device),
        }
    }
}
