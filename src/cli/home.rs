//! The home view: what this tool is, its live state, and the next commands.

use super::output::{print_help, tilde, toon};

pub(super) const DESCRIPTION: &str = "Speech to text on the CPU, offline: Parakeet, Canary, Whisper, and Moonshine models from Hugging Face, in any language they speak; 16 kHz WAV in, text out. Text to speech with Kokoro-82M (28 English voices) and MMS-TTS";

pub(super) const HOME_HELP: [&str; 7] = [
    "Run `trit transcribe <file.wav>` to get the text of a recording",
    "Run `trit models` to list the speech models (Parakeet, Canary, Whisper, Moonshine) and get one",
    "Run `trit transcribe <folder> --model <name> --format json` for every WAV file in a folder",
    "Run `trit clean \"<raw transcript>\"` to rewrite text as clean written English (S1-mini)",
    "Run `trit speak \"<text>\" --out speech.wav` to speak text (Kokoro: 28 English voices; MMS-TTS)",
    "Run `trit phonemes \"<text>\"` to see the phonemes Kokoro speaks for text",
    "Run `trit doctor` to check the models, the CPU, and the GPU",
];

pub(super) fn print() {
    println!(
        "bin: {}",
        toon(&tilde(&std::env::current_exe().unwrap_or_default()))
    );
    println!("description: {}", toon(DESCRIPTION));
    println!("version: {}", trit::VERSION);
    let model = trit::find_model(None);
    match &model {
        Some(dir) => println!("model: {}", toon(&tilde(dir))),
        None => println!("model: not found"),
    }
    println!("{}", cpu_line());
    let mut help = Vec::new();
    if let (None, Some(entry)) = (&model, trit::known(trit::MODEL_REPO)) {
        help.push(format!(
            "Run `{}` to get the model ({} MB)",
            entry.download_command(),
            entry.mb
        ));
    }
    help.extend(HOME_HELP.iter().map(|line| line.to_string()));
    print_help(&help);
}

pub(super) fn cpu_line() -> String {
    format!(
        "cpu: {}, {} threads ({} for Whisper)",
        trit::cpu_path(),
        trit::threads(trit::Spread::OneCache),
        trit::threads(trit::Spread::AllCores)
    )
}
