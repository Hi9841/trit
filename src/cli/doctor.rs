//! `trit models` and `trit doctor`.

use super::home::cpu_line;
use super::output::{print_help, tilde, toon};
use super::{Command, Outcome, Parsed};
use std::path::PathBuf;
use std::process::ExitCode;

const MODELS_HELP: &str = "\
usage: trit models
  Lists the models Trit is known to run, whether each is on this computer, and how to get it.
  Pass a name to --model: trit transcribe <file.wav> --model <name>
examples:
  trit models";

const DOCTOR_HELP: &str = "\
usage: trit doctor
  Checks the model folder, the CPU path, and the GPU. Prints the command that fixes each problem.
examples:
  trit doctor";

pub(super) const MODELS: Command = Command {
    name: "models",
    help: MODELS_HELP,
    values: &[],
    switches: &[],
    in_skill: false,
    run: models,
};

pub(super) const DOCTOR: Command = Command {
    name: "doctor",
    help: DOCTOR_HELP,
    values: &[],
    switches: &[],
    in_skill: false,
    run: doctor,
};

fn models(_: &Parsed) -> Outcome {
    let found = |entry: &trit::Known| {
        trit::candidates(entry.repo)
            .into_iter()
            .find(|dir| trit::is_voice(dir))
    };
    let rows: Vec<_> = trit::CATALOG
        .iter()
        .map(|entry| (entry, found(entry)))
        .collect();
    let present = rows.iter().filter(|(_, dir)| dir.is_some()).count();
    println!(
        "models: {present} of {} on this computer; families: {}",
        rows.len(),
        trit::MODEL_TYPES.join(", ")
    );
    println!("models[{}]{{name,kind,languages,mb,folder}}:", rows.len());
    for (entry, dir) in &rows {
        println!(
            "  {},{},{},{},{}",
            entry.name,
            entry.kind,
            toon(entry.languages),
            entry.mb,
            dir.as_deref()
                .map_or("absent".to_string(), |d| toon(&tilde(d)))
        );
    }
    let mut help = vec![
        "Run `trit transcribe <file.wav> --model <name>` to use one; any folder of a listed family works too".to_string(),
    ];
    if let Some((entry, _)) = rows.iter().find(|(_, dir)| dir.is_none()) {
        help.push(format!(
            "Run `{}` to get {} (the others the same way)",
            entry.download_command(),
            entry.name
        ));
    }
    print_help(&help);
    Ok(ExitCode::SUCCESS)
}

fn doctor(_: &Parsed) -> Outcome {
    println!("version: {}", trit::VERSION);
    println!("{}", cpu_line());
    println!("gpu: {}", trit::gpu_path());
    let candidates = trit::model_candidates();
    let model = trit::find_model(None);
    match &model {
        Some(dir) => println!("model: ok, {}", toon(&tilde(dir))),
        None => println!("model: not found"),
    }
    match trit::find_cleanup_model(None) {
        Some(dir) => println!("cleanup_model: ok, {}", toon(&tilde(&dir))),
        None => println!("cleanup_model: not found (optional; for `trit clean` and --cleanup)"),
    }
    match trit::find_voice(None) {
        Some(dir) => println!("voice: ok, {}", toon(&tilde(&dir))),
        None => println!("voice: not found (optional; for `trit speak`)"),
    }
    for (accent, repo) in trit::PHONEMIZER_FALLBACKS {
        let found = trit::candidates(repo)
            .into_iter()
            .find(|d| trit::is_model(d));
        match found {
            Some(dir) => println!("phonemizer_fallback_{accent}: ok, {}", toon(&tilde(&dir))),
            None => println!(
                "phonemizer_fallback_{accent}: not found (optional; Kokoro skips words that no dictionary holds)"
            ),
        }
    }
    println!("families: {}", trit::MODEL_TYPES.join(", "));
    println!("searched[{}]{{folder,status}}:", candidates.len() + 1);
    let env = std::env::var_os("TRIT_MODEL").map(PathBuf::from);
    match &env {
        Some(dir) => println!(
            "  {},{}",
            toon(&format!("TRIT_MODEL={}", dir.display())),
            if trit::is_model(dir) {
                "ok"
            } else {
                "incomplete"
            }
        ),
        None => println!("  TRIT_MODEL,unset"),
    }
    for dir in &candidates {
        let status = if trit::is_model(dir) {
            "ok"
        } else if dir.is_dir() {
            "incomplete"
        } else {
            "absent"
        };
        println!("  {},{status}", toon(&tilde(dir)));
    }
    let mut problems = Vec::new();
    if model.is_none() {
        problems.push(trit::missing_model(None));
    }
    if let Some(problem) = trit::isa_problem() {
        problems.push(problem);
    }
    if trit::cpu_path() == "scalar" {
        problems.push(
            "This CPU has no AVX2; Trit runs on the slow scalar path (expect about 10x slower)"
                .into(),
        );
    }
    if problems.is_empty() {
        println!("problems: 0, ready");
        print_help(&[
            "Run `trit transcribe <file.wav>` to get the text of a recording".into(),
            "Run `trit models` for other models: Whisper, Moonshine, Parakeet v3".into(),
        ]);
        Ok(ExitCode::SUCCESS)
    } else {
        print_help(&problems);
        if model.is_none() {
            Err(ExitCode::FAILURE)
        } else {
            Ok(ExitCode::SUCCESS)
        }
    }
}
