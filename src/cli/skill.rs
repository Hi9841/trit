//! The installable agent skill.

use super::home::{DESCRIPTION, HOME_HELP};
use super::output::toon;
use super::{error, Command, Outcome, Parsed, COMMANDS};
use std::process::ExitCode;

const HELP: &str = "\
usage: trit skill [--check <file>]
  Prints the agent skill (SKILL.md) for Trit. It comes from the same text as the home view.
  --check <file>  exit 1 when <file> differs from the skill this version prints
examples:
  trit skill > skill/SKILL.md
  trit skill --check skill/SKILL.md";

pub(super) const COMMAND: Command = Command {
    name: "skill",
    help: HELP,
    values: &["--check"],
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
            command.name, command.help
        ));
    }
    text.push_str("## Get a model\n\n`trit models` lists the known models, where each is, and its download command. Pass a name or a folder to `--model`.\n\n| Name | Family | Languages | MB |\n|---|---|---|---:|\n");
    for entry in trit::CATALOG {
        text.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            entry.name, entry.kind, entry.languages, entry.mb
        ));
    }
    text.push_str(&format!(
        "\nExample: `{}`. The default model is {}; `TRIT_MODEL=<dir>` sets another.\n\n",
        trit::CATALOG[0].download_command(),
        trit::CATALOG[0].name
    ));
    text.push_str("## Use it from code\n\nRead the README (https://github.com/Hi9841/trit): the Rust API, the C ABI (`include/trit.h`), and bindings for Python, C#, and Node. Any other language with a C FFI uses `include/trit.h` directly.\n");
    text
}

fn run(parsed: &Parsed) -> Outcome {
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
