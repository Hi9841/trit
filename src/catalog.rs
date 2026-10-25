//! Models that Trit has been run with, and where to get them. Any other
//! checkpoint of a supported family (`MODEL_TYPES`) loads too; these are
//! the ones known to work.

/// One downloadable model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Known {
    /// Short name; also the folder name under `models/`.
    pub name: &'static str,
    /// Hugging Face repository.
    pub repo: &'static str,
    /// `model_type` of its `config.json`; `vits` and `kokoro` are voices
    /// (text to speech), `g2p` the fallback of the English phonemizer, and
    /// `cleanup` S1-mini.
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
        name: "parakeet-rnnt-0.6b",
        repo: "nvidia/parakeet-rnnt-0.6b",
        kind: "parakeet_rnnt",
        languages: "en",
        mb: 2467,
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
        name: "qwen3-asr-0.6b",
        repo: "Qwen/Qwen3-ASR-0.6B",
        kind: "qwen3_asr",
        languages: "30",
        mb: 1876,
    },
    Known {
        name: "wav2vec2-base-960h",
        repo: "facebook/wav2vec2-base-960h",
        kind: "wav2vec2",
        languages: "en",
        mb: 378,
    },
    Known {
        name: "s1-mini",
        repo: "superwhisper/s1-mini",
        kind: "cleanup",
        languages: "en",
        mb: 1503,
    },
    Known {
        name: "kokoro-82m",
        repo: "hexgrad/Kokoro-82M",
        kind: "kokoro",
        languages: "en (US, GB)",
        mb: 327,
    },
    Known {
        name: "graphemes_to_phonemes_en_us",
        repo: "PeterReid/graphemes_to_phonemes_en_us",
        kind: "g2p",
        languages: "en (US)",
        mb: 3,
    },
    Known {
        name: "graphemes_to_phonemes_en_gb",
        repo: "PeterReid/graphemes_to_phonemes_en_gb",
        kind: "g2p",
        languages: "en (GB)",
        mb: 3,
    },
    Known {
        name: "mms-tts-eng",
        repo: "facebook/mms-tts-eng",
        kind: "vits",
        languages: "en",
        mb: 145,
    },
];

impl Known {
    /// The files Trit reads from a folder of this kind.
    pub fn files(&self) -> &'static [&'static str] {
        match self.kind {
            "whisper" => &[
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "model.safetensors",
            ],
            "wav2vec2" => &[
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
                "vocab.json",
                "model.safetensors",
            ],
            "kokoro" => &[
                "config.json",
                "kokoro-v1_0.pth",
                "voices/af_alloy.pt",
                "voices/af_aoede.pt",
                "voices/af_bella.pt",
                "voices/af_heart.pt",
                "voices/af_jessica.pt",
                "voices/af_kore.pt",
                "voices/af_nicole.pt",
                "voices/af_nova.pt",
                "voices/af_river.pt",
                "voices/af_sarah.pt",
                "voices/af_sky.pt",
                "voices/am_adam.pt",
                "voices/am_echo.pt",
                "voices/am_eric.pt",
                "voices/am_fenrir.pt",
                "voices/am_liam.pt",
                "voices/am_michael.pt",
                "voices/am_onyx.pt",
                "voices/am_puck.pt",
                "voices/am_santa.pt",
                "voices/bf_alice.pt",
                "voices/bf_emma.pt",
                "voices/bf_isabella.pt",
                "voices/bf_lily.pt",
                "voices/bm_daniel.pt",
                "voices/bm_fable.pt",
                "voices/bm_george.pt",
                "voices/bm_lewis.pt",
            ],
            "g2p" => &["config.json", "generation_config.json", "model.safetensors"],
            "vits" => &[
                "config.json",
                "tokenizer_config.json",
                "vocab.json",
                "model.safetensors",
            ],
            "qwen3_asr" => &[
                "config.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "model.safetensors",
            ],
            _ => &["config.json", "tokenizer.json", "model.safetensors"],
        }
    }

    /// The command that downloads only the files Trit reads. The files are
    /// named one by one: on Windows `hf` expands a quoted `*.json` against
    /// the current folder, so an `--include` glob is not reliable.
    pub fn download_command(&self) -> String {
        format!(
            "hf download {} {} --local-dir models/{}",
            self.repo,
            self.files().join(" "),
            self.name
        )
    }
}

/// A catalog entry by short name or repository.
pub fn known(name: &str) -> Option<&'static Known> {
    CATALOG.iter().find(|k| k.name == name || k.repo == name)
}
