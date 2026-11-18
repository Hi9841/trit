//! The installable agent skill.

use super::home::{DESCRIPTION, HOME_HELP};
use super::output::toon;
use super::{error, Command, Outcome, Parsed, COMMANDS};
use std::process::ExitCode;

const HELP: &str = "\
usage: trit skill [--check <file>] [--check-docs <file>]
  Prints the agent skill (SKILL.md) for Trit. It comes from the same text as the home view.
  --check <file>  exit 1 when <file> differs from the skill this version prints
  --check-docs <file>  exit 1 when the model table of <file> (docs/reference.md) names other models, repositories, or sizes than `trit models`
examples:
  trit skill > skill/SKILL.md
  trit skill --check skill/SKILL.md
  trit skill --check-docs docs/reference.md";

pub(super) const COMMAND: Command = Command {
    name: "skill",
    help: HELP,
    values: &["--check", "--check-docs"],
    switches: &[],
    in_skill: false,
    run,
};

/// The installable agent skill. Built from the home view text so the two
/// cannot drift; `--check` fails when a committed copy is stale.
fn skill_text() -> String {
    let mut text = String::new();
    text.push_str("---\nname: trit\ndescription: Transcribe speech from WAV files to text on the CPU, offline, with the Trit engine and a Parakeet, Canary, Whisper, or Moonshine model, in any language the model speaks; clean a raw transcript with S1-mini; speak text into a WAV file with Kokoro (28 English voices) or MMS-TTS. Use when an agent must turn a recording into text or text into speech, or add speech to text or text to speech to an app in Rust, C, C++, Python, C#, Node, or Bun.\n---\n\n");
    text.push_str("# Trit\n\n");
    text.push_str(DESCRIPTION);
    text.push_str(". `--device gpu` or `--device auto` runs the matrix products on the GPU. `auto` uses the GPU only for the Parakeet models, and the CPU when the computer has no adapter. Build with `--features gpu` for that path.\n\n## Commands\n\n");
    for line in HOME_HELP {
        text.push_str(&format!("- {}\n", line.trim_start_matches("Run ")));
    }
    text.push_str("- `trit skill --check <file>` to check a copy of this skill\n");
    text.push_str("\nOutput is TOON on stdout. Errors are on stdout too, as `error:` and `help:` lines. Exit codes: 0 success, 1 error, 2 usage error.\n\n");
    for command in COMMANDS.iter().filter(|command| command.in_skill) {
        text.push_str(&format!(
            "## {}\n\n```text\n{}\n```\n\n",
            command.name,
            command.help_text()
        ));
    }
    text.push_str("## Get a model\n\n`trit models` lists the known models, where each is, and its download command. Pass a name or a folder to `--model`.\n\n| Name | Family | Languages | MB |\n|---|---|---|---:|\n");
    for entry in trit::CATALOG {
        text.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            entry.name, entry.kind, entry.languages, entry.mb
        ));
    }
    if let Some(default) = trit::known(trit::MODEL_REPO) {
        text.push_str(&format!(
            "\nExample: `{}`. The default model is {}; `TRIT_MODEL=<dir>` sets another.\n\n",
            default.download_command(),
            default.name
        ));
    }
    text.push_str("## Use it from code\n\nRead the README (https://github.com/Hi9841/trit): the Rust API, the C ABI (`include/trit.h`), and bindings for Python, C#, and Node. Any other language with a C FFI uses `include/trit.h` directly.\n");
    text
}

/// The rows of the model table (`| Name | Repository | Family | Languages |
/// MB |`) of a reference text that differ from `CATALOG`: a name it lacks,
/// a name that is not there, or another repository or size. The family and
/// languages columns are prose and stay free.
fn docs_problems(docs: &str) -> Vec<String> {
    let rows: Vec<Vec<&str>> = docs
        .lines()
        .skip_while(|line| !line.starts_with("| Name | Repository |"))
        .skip(2)
        .take_while(|line| line.starts_with('|'))
        .map(|line| line.trim_matches('|').split('|').map(str::trim).collect())
        .collect();
    let mut problems = Vec::new();
    for entry in trit::CATALOG {
        let mb = entry.mb.to_string();
        match rows.iter().find(|row| row.first() == Some(&entry.name)) {
            None => problems.push(format!("no row for {}", entry.name)),
            Some(row) if row.get(1) != Some(&entry.repo) || row.last() != Some(&mb.as_str()) => {
                problems.push(format!("{}: want {} and {} MB", entry.name, entry.repo, mb));
            }
            Some(_) => {}
        }
    }
    for row in &rows {
        if !trit::CATALOG
            .iter()
            .any(|entry| row.first() == Some(&entry.name))
        {
            problems.push(format!(
                "{} is not in `trit models`",
                row.first().unwrap_or(&"")
            ));
        }
    }
    problems
}

fn check_docs(file: &str) -> Result<(), ExitCode> {
    let docs = std::fs::read_to_string(file).map_err(|e| {
        error(
            &format!("cannot read {file}: {e}"),
            "trit skill --check-docs docs/reference.md",
        )
    })?;
    let problems = docs_problems(&docs);
    if problems.is_empty() {
        println!("docs: the model table of {} is current", toon(file));
        return Ok(());
    }
    Err(error(
        &format!(
            "the model table of {file} differs from `trit models`: {}",
            problems.join("; ")
        ),
        "change the rows of the table to the entries of src/catalog.rs",
    ))
}

fn run(parsed: &Parsed) -> Outcome {
    if let Some(file) = parsed.value("--check-docs") {
        check_docs(file)?;
        if parsed.value("--check").is_none() {
            return Ok(ExitCode::SUCCESS);
        }
    }
    let text = skill_text();
    let Some(file) = parsed.value("--check") else {
        print!("{text}");
        return Ok(ExitCode::SUCCESS);
    };
    match std::fs::read_to_string(file) {
        Ok(committed) if committed.replace("\r\n", "\n") == text => {
            println!("skill: {} is current", toon(file));
            Ok(ExitCode::SUCCESS)
        }
        Ok(_) => Err(error(
            &format!("{file} is stale"),
            &format!("trit skill > {file}"),
        )),
        Err(e) => Err(error(
            &format!("cannot read {file}: {e}"),
            &format!("trit skill > {file}"),
        )),
    }
}
