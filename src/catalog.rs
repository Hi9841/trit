//! Models that Trit has been run with, and where to get them. Any other
//! checkpoint of a supported family (`MODEL_TYPES`) loads too; these are
//! the ones known to work.

/// One downloadable model.
pub struct Known {
    /// Short name; also the folder name under `models/`.
    pub name: &'static str,
    /// Hugging Face repository.
    pub repo: &'static str,
    /// `model_type` of its `config.json`, or `cleanup` for S1-mini.
    pub kind: &'static str,
    /// Languages it writes.
    pub languages: &'static str,
    /// Size of `model.safetensors` in MB.
    pub mb: u32,
}

pub const CATALOG: &[Known] = &[
    Known {
        name: "parakeet-redux",
        repo: "moondream/parakeet-redux",
        kind: "parakeet_tdt",
        languages: "25 European",
        mb: 178,
    },
    Known {
        name: "parakeet-tdt-0.6b-v3",
        repo: "nvidia/parakeet-tdt-0.6b-v3",
        kind: "parakeet_tdt",
        languages: "25 European",
        mb: 2508,
    },
    Known {
        name: "parakeet-ctc-0.6b",
        repo: "nvidia/parakeet-ctc-0.6b",
        kind: "parakeet_ctc",
        languages: "en",
        mb: 2436,
    },
    Known {
        name: "canary-1b-v2",
        repo: "nvidia/canary-1b-v2",
        kind: "canary",
        languages: "25 European",
        mb: 3916,
    },
    Known {
        name: "whisper-tiny",
        repo: "openai/whisper-tiny",
        kind: "whisper",
        languages: "99",
        mb: 151,
    },
    Known {
        name: "whisper-base.en",
        repo: "openai/whisper-base.en",
        kind: "whisper",
        languages: "en",
        mb: 290,
    },
    Known {
        name: "whisper-small.en",
        repo: "openai/whisper-small.en",
        kind: "whisper",
        languages: "en",
        mb: 967,
    },
    Known {
        name: "whisper-large-v3-turbo",
        repo: "openai/whisper-large-v3-turbo",
        kind: "whisper",
        languages: "99",
        mb: 1618,
    },
    Known {
        name: "whisper-large-v3",
        repo: "openai/whisper-large-v3",
        kind: "whisper",
        languages: "99",
        mb: 3087,
    },
    Known {
        name: "distil-large-v3.5",
        repo: "distil-whisper/distil-large-v3.5",
        kind: "whisper",
        languages: "en",
        mb: 3026,
    },
    Known {
        name: "moonshine-tiny",
        repo: "UsefulSensors/moonshine-tiny",
        kind: "moonshine",
        languages: "en",
        mb: 108,
    },
    Known {
        name: "moonshine-base",
        repo: "UsefulSensors/moonshine-base",
        kind: "moonshine",
        languages: "en",
        mb: 246,
    },
    Known {
        name: "s1-mini",
        repo: "superwhisper/s1-mini",
        kind: "cleanup",
        languages: "en",
        mb: 1503,
    },
];

impl Known {
    /// The command that downloads only the files Trit reads.
    pub fn download_command(&self) -> String {
        format!(
            "hf download {} --include \"*.json\" model.safetensors --local-dir models/{}",
            self.repo, self.name
        )
    }
}

/// A catalog entry by short name or repository.
pub fn known(name: &str) -> Option<&'static Known> {
    CATALOG.iter().find(|k| k.name == name || k.repo == name)
}
