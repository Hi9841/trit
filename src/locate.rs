//! Where models are on this computer. One `Lookup` per kind of model.

use crate::catalog::{known, Known};
use crate::cleanup::CLEANUP_REPO;
use crate::cpu::tensors::{SHARD_INDEX, WEIGHTS};
use crate::tts::{is_kokoro, VOICE_REPOS};
use std::path::{Path, PathBuf};

/// The Hugging Face repository of the default model.
pub const MODEL_REPO: &str = "moondream/parakeet-redux";
/// The files every model folder holds; a family may need more. A large
/// model may hold `model.safetensors.index.json` and its shards instead of
/// `model.safetensors`.
pub const MODEL_FILES: [&str; 2] = ["config.json", "model.safetensors"];

/// How to find one kind of model.
struct Lookup {
    /// As messages name it.
    what: &'static str,
    /// The variable that names its folder.
    variable: &'static str,
    /// The repositories to search, in order; the first is the one to get.
    repos: &'static [&'static str],
    holds: fn(&Path) -> bool,
}

const SPEECH: Lookup = Lookup {
    what: "model",
    variable: "TRIT_MODEL",
    repos: &[MODEL_REPO],
    holds: is_model,
};

const CLEANUP: Lookup = Lookup {
    what: "cleanup model",
    variable: "TRIT_CLEANUP_MODEL",
    repos: &[CLEANUP_REPO],
    holds: is_model,
};

const VOICE: Lookup = Lookup {
    what: "voice",
    variable: "TRIT_VOICE",
    repos: &VOICE_REPOS,
    holds: is_voice,
};

impl Lookup {
    /// `explicit` if given (a folder, or a `CATALOG` name found in the
    /// places of `candidates`), else the folder of `variable`, else the
    /// first complete folder of `repos`.
    fn find(&self, explicit: Option<&Path>) -> Option<PathBuf> {
        if let Some(dir) = explicit {
            if (self.holds)(dir) {
                return Some(dir.to_path_buf());
            }
            let entry = known(&dir.to_string_lossy())?;
            return candidates(entry.repo)
                .into_iter()
                .find(|dir| (self.holds)(dir));
        }
        if let Some(dir) = std::env::var_os(self.variable) {
            let dir = PathBuf::from(dir);
            return (self.holds)(&dir).then_some(dir);
        }
        self.repos
            .iter()
            .flat_map(|repo| candidates(repo))
            .find(|dir| (self.holds)(dir))
    }

    /// Why `find(explicit)` found nothing, with the command that fixes it.
    fn missing(&self, explicit: Option<&Path>) -> String {
        match explicit.map(|dir| (dir, known(&dir.to_string_lossy()))) {
            Some((_, Some(entry))) => format!(
                "{} is not on this computer; get it: {}",
                entry.name,
                entry.download_command()
            ),
            Some((dir, None)) => format!(
                "{} is not a {} folder ({}) or a name from `trit models`",
                dir.display(),
                self.what,
                MODEL_FILES.join(", ")
            ),
            None => match std::env::var_os(self.variable) {
                // `find` reads only the variable when it is set.
                Some(dir) => format!(
                    "{}={} is not a {} folder ({}); correct the variable or remove it",
                    self.variable,
                    Path::new(&dir).display(),
                    self.what,
                    MODEL_FILES.join(", ")
                ),
                None => format!(
                    "no {} found; get it: {}",
                    self.what,
                    known(self.repos[0]).map_or_else(String::new, Known::download_command)
                ),
            },
        }
    }
}

/// The model folder: `explicit` if given (a folder, or a `CATALOG` name found
/// in the places of `candidates`), else `TRIT_MODEL`, else the first complete
/// `parakeet-redux` folder of `candidates`.
pub fn find_model(explicit: Option<&Path>) -> Option<PathBuf> {
    SPEECH.find(explicit)
}

/// Why `find_model(explicit)` found nothing, with the command that fixes it.
pub fn missing_model(explicit: Option<&Path>) -> String {
    SPEECH.missing(explicit)
}

/// The cleanup model folder: `explicit` as for `find_model`, else
/// `TRIT_CLEANUP_MODEL`, else the first complete `s1-mini` folder of `candidates`.
pub fn find_cleanup_model(explicit: Option<&Path>) -> Option<PathBuf> {
    CLEANUP.find(explicit)
}

/// Why `find_cleanup_model(explicit)` found nothing, with the command that
/// fixes it.
pub fn missing_cleanup_model(explicit: Option<&Path>) -> String {
    CLEANUP.missing(explicit)
}

/// The voice folder: `explicit` as for `find_model`, else `TRIT_VOICE`, else
/// the first complete folder of `VOICE_REPOS` (Kokoro-82M, then MMS-TTS English).
pub fn find_voice(explicit: Option<&Path>) -> Option<PathBuf> {
    VOICE.find(explicit)
}

/// Why `find_voice(explicit)` found nothing, with the command that fixes it.
pub fn missing_voice(explicit: Option<&Path>) -> String {
    VOICE.missing(explicit)
}

/// The folders `find_model` searches, in order, when no folder is named.
pub fn model_candidates() -> Vec<PathBuf> {
    candidates(MODEL_REPO)
}

/// The places a model of Hugging Face repository `repo` (`owner/name`) may
/// be: `models/<name>` under the current folder, its parent, the folder of
/// the executable and two and three folders up, the TDT app data, then the
/// snapshots of the Hugging Face cache.
pub fn candidates(repo: &str) -> Vec<PathBuf> {
    let name = repo.rsplit('/').next().unwrap_or(repo);
    // The catalog's folder name too, where it differs (Kokoro-82M is models/kokoro-82m).
    let names: Vec<&str> = std::iter::once(name)
        .chain(known(repo).map(|k| k.name).filter(|n| *n != name))
        .collect();
    let mut roots = vec![PathBuf::from("models"), PathBuf::from("../models")];
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        roots.push(exe_dir.join("models"));
        roots.push(exe_dir.join("../../models"));
        roots.push(exe_dir.join("../../../models"));
    }
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        roots.push(PathBuf::from(local).join("TDT").join("models"));
    }
    let mut dirs: Vec<PathBuf> = roots
        .iter()
        .flat_map(|root| names.iter().map(move |name| root.join(name)))
        .collect();
    let hub = std::env::var_os("HF_HOME")
        .map(|home| PathBuf::from(home).join("hub"))
        .or_else(|| {
            std::env::var_os("USERPROFILE")
                .or_else(|| std::env::var_os("HOME"))
                .map(|home| PathBuf::from(home).join(".cache/huggingface/hub"))
        });
    if let Some(hub) = hub {
        let snapshots = hub
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots");
        if let Ok(entries) = std::fs::read_dir(snapshots) {
            // The newest snapshot first: the order of a folder is not fixed,
            // and two snapshots of one repository can hold two revisions.
            let mut found: Vec<(std::time::SystemTime, PathBuf)> = entries
                .flatten()
                .map(|entry| {
                    let changed = entry.metadata().and_then(|meta| meta.modified());
                    (changed.unwrap_or(std::time::UNIX_EPOCH), entry.path())
                })
                .collect();
            found.sort_by(|a, b| b.cmp(a));
            dirs.extend(found.into_iter().map(|(_, dir)| dir));
        }
    }
    dirs
}

/// True when `dir` holds a voice: a VITS folder (as `is_model`) or a
/// Kokoro folder (`config.json` with `istftnet` and a `.pth` checkpoint).
pub fn is_voice(dir: &Path) -> bool {
    is_model(dir) || is_kokoro(dir)
}

/// True when `dir` holds every file of `MODEL_FILES` (the weights as one
/// file or as shards), one NVIDIA NeMo archive (`.nemo`) or GGUF file
/// (`.gguf`) and no `config.json`, or a Fermion container (`model.fermion`).
pub fn is_model(dir: &Path) -> bool {
    let weights = dir.join(WEIGHTS).is_file() || dir.join(SHARD_INDEX).is_file();
    (dir.join("config.json").is_file() && weights) || crate::models::package::in_folder(dir)
}
