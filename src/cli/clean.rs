//! `trit clean`, and the cleanup flags of `trit transcribe`.

use super::output::toon;
use super::{error, load_options, usage_error, Command, Format, Outcome, Parsed};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

const HELP: &str = "\
usage: trit clean <text>... [--model <dir>] [--styling <s>] [--structure <s>] [--context <c>] [--precision <p>] [--device <d>] [--format toon|json|text]
  Rewrites a raw transcript as written text with Superwhisper S1-mini (English): removes
  fillers, resolves self-corrections, adds punctuation and casing, writes numbers and addresses.
  <text>          the raw transcript; several arguments join with spaces
  --model <dir>   S1-mini folder (default: TRIT_CLEANUP_MODEL, models/s1-mini, the Hugging Face cache)
  --styling       casual, semi-casual, semi-formal (default), formal
  --structure     prose (default), lists
  --context       general (default), email
  --precision     auto (default), int8, f16, f32, q8_0; as for `trit transcribe`
  --device        cpu (default), gpu, auto; as for `trit transcribe`. TRIT_DEVICE overrides
  --format        toon (default), json, text
examples:
  trit clean \"so um send the report by friday no wait thursday\"
  trit clean \"hey sarah can you send the numbers thanks john\" --context email";

pub(super) const COMMAND: Command = Command {
    name: "clean",
    help: HELP,
    values: &[
        "--model",
        "--styling",
        "--structure",
        "--context",
        "--format",
        "--precision",
        "--device",
    ],
    switches: &[],
    in_skill: true,
    run,
};

pub(super) fn style(parsed: &Parsed) -> Result<trit::Style, ExitCode> {
    trit::Style::named(
        parsed.value("--styling"),
        parsed.value("--structure"),
        parsed.value("--context"),
    )
    .map_err(|unknown| {
        usage_error(
            &format!("unknown {} `{}`", unknown.setting, unknown.value),
            &format!(
                "valid values for --{}: {}",
                unknown.setting,
                unknown.valid.join(", ")
            ),
        )
    })
}

pub(super) fn load_cleaner(
    explicit: Option<&str>,
    options: &trit::LoadOptions,
) -> Result<trit::Cleaner, ExitCode> {
    let explicit = explicit.map(Path::new);
    let Some(dir) = trit::find_cleanup_model(explicit) else {
        return Err(error(
            &trit::missing_cleanup_model(explicit),
            "Run `trit doctor` to see where Trit looks",
        ));
    };
    trit::Cleaner::load_with(&dir, options).map_err(|e| {
        error(
            &format!("cannot load the cleanup model: {e}"),
            "Run `trit doctor`",
        )
    })
}

fn run(parsed: &Parsed) -> Outcome {
    let format = Format::of(parsed, &["toon", "json", "text"])?;
    let style = style(parsed)?;
    if parsed.positional.is_empty() {
        return Err(usage_error(
            "clean needs the text to clean",
            "trit clean \"<raw transcript>\" [--styling semi-formal]",
        ));
    }
    let raw = parsed.positional.join(" ");
    let options = load_options(parsed)?;
    let cleaner = load_cleaner(parsed.value("--model"), &options)?;
    let started = Instant::now();
    let text = cleaner.clean(&raw, &style);
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    match format {
        Format::Text => println!("{text}"),
        Format::Json => println!(
            "{}",
            serde_json::json!({"text": text, "ms": ms, "styling": style.styling.name(),
                "structure": style.structure.name(), "context": style.context.name()})
        ),
        Format::Toon => {
            println!("ms: {ms:.0}");
            println!(
                "style: {} {} {}",
                style.styling.name(),
                style.structure.name(),
                style.context.name()
            );
            if text.is_empty() {
                println!("text: \"\" (the input was only fillers or noise)");
            } else {
                println!("text: {}", toon(&text));
            }
        }
    }
    Ok(ExitCode::SUCCESS)
}
