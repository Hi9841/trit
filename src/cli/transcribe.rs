//! `trit transcribe`.

use super::clean::{load_cleaner, style};
use super::output::{print_help, slashes, tilde, toon};
use super::{error, load_options, usage_error, Command, Format, Outcome, Parsed};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

/// Longer transcripts in a list of several files are cut here; `--full` keeps them.
const PREVIEW_CHARS: usize = 500;

pub(super) const HELP: &str = "\
usage: trit transcribe <path>... [--model <dir>] [--language <code>] [--dictionary <file>] [--task <t>] [--timestamps] [--cleanup] [--precision <p>] [--device <d>] [--beam <n>] [--format toon|json|text] [--full]
  <path>          a WAV file (16 kHz, or another rate: Trit resamples it), or a folder: every .wav file in it and its subfolders
  --model <dir>   speech model folder or `trit models` name: Parakeet, Canary, Whisper, Moonshine (default: TRIT_MODEL, then the places `trit doctor` lists)
  --language <code>  spoken language, ISO 639-1 (en, de, ...); default: the model detects it
  --dictionary <file>  terms to favor, one per line (names, product words)
  --task <t>      transcribe (default), or translate into English (multilingual Whisper, Canary)
  --timestamps    timed segments too: sentences for Parakeet, windows of up to 30 s (40 s Canary) for the others
  --cleanup       clean each transcript with S1-mini (see `trit clean --help`); ms then includes it
  --cleanup-model <dir>, --styling, --structure, --context  as for `trit clean`
  --precision     auto (default: int8 for layers 384 wide and more, f16 below; a GGUF file keeps its stored types), int8 (least memory), f16, f32 (as the reference runtimes), q8_0 (ggml's Q8_0, as llama.cpp and transcribe.cpp run it)
  --device        cpu (default), gpu, auto. auto uses the GPU for the Parakeet models when the computer has an adapter, and the CPU for the other families (slower on the GPU). TRIT_DEVICE overrides
  --beam <n>      beams for Whisper, Moonshine, and Canary (default: 5 for Whisper and Moonshine, 1 (greedy) for Canary; at most 16)
  --format        toon (default), json (full text, for programs), text (one transcript per line)
  --full          do not cut transcripts longer than 500 characters in a list of several files
examples:
  trit transcribe meeting.wav
  trit transcribe clips/ --format json
  trit transcribe take.wav --dictionary terms.txt --cleanup
  trit transcribe interview.wav --model models/whisper-large-v3-turbo --language de
  trit transcribe meeting.wav --device gpu";

pub(super) const COMMAND: Command = Command {
    name: "transcribe",
    help: HELP,
    values: &[
        "--model",
        "--language",
        "--dictionary",
        "--format",
        "--cleanup-model",
        "--styling",
        "--structure",
        "--context",
        "--precision",
        "--device",
        "--beam",
        "--task",
    ],
    switches: &["--full", "--cleanup", "--timestamps"],
    in_skill: true,
    run,
};

struct Row {
    file: String,
    seconds: f64,
    ms: f64,
    text: String,
    /// With `--timestamps` only.
    segments: Vec<trit::Segment>,
}

fn run(parsed: &Parsed) -> Outcome {
    let format = Format::of(parsed, &["toon", "json", "text"])?;
    if parsed.positional.is_empty() {
        return Err(usage_error(
            "transcribe needs a WAV file or a folder",
            "trit transcribe <file.wav> [--format json]",
        ));
    }
    let mut files = Vec::new();
    for path in &parsed.positional {
        let path = PathBuf::from(path);
        if path.is_dir() {
            wav_files(&path, &mut files);
        } else if path.is_file() {
            files.push(path);
        } else {
            return Err(error(
                &format!("{} does not exist", path.display()),
                "give a .wav file or a folder of them: trit transcribe <path>",
            ));
        }
    }
    if files.is_empty() {
        println!(
            "files: 0 WAV files found in {}",
            toon(&parsed.positional.join(" "))
        );
        return Ok(ExitCode::SUCCESS);
    }
    let terms = parsed
        .value("--dictionary")
        .map(|file| {
            std::fs::read_to_string(file).map_err(|e| {
                error(
                    &format!("cannot read dictionary {file}: {e}"),
                    "give a text file with one term per line: --dictionary <file>",
                )
            })
        })
        .transpose()?
        .map(|text| text.lines().map(str::to_string).collect::<Vec<_>>());
    let options = trit::Options {
        language: parsed.value("--language").and_then(trit::language_code),
        beam: match parsed.value("--beam").map(str::parse::<usize>) {
            None => 0,
            Some(Ok(beam)) if beam >= 1 => beam,
            Some(_) => {
                return Err(usage_error(
                    "--beam needs a whole number of 1 or more",
                    "--beam 5",
                ))
            }
        },
        task: match parsed.value("--task").map(trit::Task::parse) {
            None => trit::Task::Transcribe,
            Some(Some(task)) => task,
            Some(None) => {
                return Err(usage_error(
                    "unknown task",
                    &format!("valid tasks: {}", trit::Task::NAMES.join(", ")),
                ))
            }
        },
        ..trit::Options::default()
    };
    let style = style(parsed)?;
    let load = load_options(parsed)?;
    let cleaner = (parsed.switch("--cleanup") || parsed.value("--cleanup-model").is_some())
        .then(|| load_cleaner(parsed.value("--cleanup-model"), &load))
        .transpose()?;
    let explicit = parsed.value("--model").map(Path::new);
    let Some(model_dir) = trit::find_model(explicit) else {
        return Err(error(
            &trit::missing_model(explicit),
            "Run `trit doctor` to see where Trit looks",
        ));
    };
    let engine = trit::Trit::load_with(&model_dir, &load)
        .map_err(|e| error(&format!("cannot load the model: {e}"), "Run `trit doctor`"))?;
    trit::release_memory();
    engine
        .check(&options)
        .map_err(|e| usage_error(&e, "--language en; without --language the model detects it"))?;
    let skipped = terms
        .map(|terms| engine.set_dictionary(&terms, trit::BiasWeights::default()))
        .unwrap_or_default();

    let mut rows = Vec::new();
    let mut failed = Vec::new();
    for file in &files {
        let samples = match trit::read_wav(file) {
            Ok(samples) => samples,
            Err(e) => {
                failed.push((slashes(file), e));
                continue;
            }
        };
        #[cfg(feature = "profile")]
        let before = profile::Marks::now();
        let started = Instant::now();
        let (mut text, segments) =
            engine.transcript(&samples, &options, parsed.switch("--timestamps"));
        #[cfg(feature = "profile")]
        before.line(samples.len(), started);
        if let Some(cleaner) = &cleaner {
            text = cleaner.clean(&text, &style);
        }
        rows.push(Row {
            file: slashes(file),
            seconds: samples.len() as f64 / f64::from(trit::SAMPLE_RATE),
            ms: started.elapsed().as_secs_f64() * 1000.0,
            text,
            segments,
        });
    }

    #[cfg(feature = "profile")]
    profile::table(&rows);
    match format {
        Format::Text => {
            for row in &rows {
                if row.segments.is_empty() {
                    println!("{}", row.text);
                }
                for segment in &row.segments {
                    println!("[{:.2}-{:.2}] {}", segment.start, segment.end, segment.text);
                }
            }
            for (file, e) in &failed {
                println!("error: {file}: {e}");
            }
        }
        Format::Json => print_json(&model_dir, &rows, &failed, &skipped),
        Format::Toon => print_toon(
            &model_dir,
            &rows,
            &failed,
            &skipped,
            parsed.switch("--full"),
        ),
    }
    if failed.is_empty() {
        Ok(ExitCode::SUCCESS)
    } else {
        Err(ExitCode::FAILURE)
    }
}

/// The lines of a profile build, on stderr: stdout stays the result.
#[cfg(feature = "profile")]
#[allow(clippy::print_stderr)]
mod profile {
    use std::sync::atomic::Ordering::Relaxed;
    use trit::prof::{counted, Stage, ALLOCATED_BYTES, ALLOCATIONS};

    /// The counters before one call.
    pub struct Marks(u64, u64, u64, u64);

    impl Marks {
        pub fn now() -> Self {
            Self(
                ALLOCATIONS.load(Relaxed),
                ALLOCATED_BYTES.load(Relaxed),
                counted(Stage::Moves),
                counted(Stage::Jobs),
            )
        }

        /// One line for the call that ran since `now`.
        pub fn line(&self, samples: usize, started: std::time::Instant) {
            let after = Self::now();
            eprintln!(
                "call {:.2} s {:.2} ms allocations {} bytes {} moves {} jobs {} here {} away {} home {}",
                samples as f64 / f64::from(trit::SAMPLE_RATE),
                started.elapsed().as_secs_f64() * 1000.0,
                after.0 - self.0,
                after.1 - self.1,
                after.2 - self.2,
                after.3 - self.3,
                counted(Stage::BusyHere),
                counted(Stage::BusyAway),
                counted(Stage::Home)
            );
        }
    }

    pub fn table(rows: &[super::Row]) {
        let busy: f64 = rows.iter().map(|row| row.ms).sum();
        eprint!(
            "profile: {} calls, {busy:.1} ms\n{}",
            rows.len(),
            trit::prof::report()
        );
    }
}

fn wav_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            wav_files(&path, files);
        } else if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
        {
            files.push(path);
        }
    }
}

fn summary(rows: &[Row]) -> (f64, f64) {
    // fold from 0.0: a float `sum` of nothing is -0.0.
    let audio = rows.iter().fold(0.0, |total, row| total + row.seconds);
    let busy = rows.iter().fold(0.0, |total, row| total + row.ms) / 1000.0;
    (audio, busy)
}

fn print_toon(
    model: &Path,
    rows: &[Row],
    failed: &[(String, String)],
    skipped: &[String],
    full: bool,
) {
    let (audio, busy) = summary(rows);
    println!("model: {}", toon(&tilde(model)));
    println!(
        "files: {} transcribed, {} failed, {audio:.1} s of audio in {busy:.2} s ({:.0}x realtime)",
        rows.len(),
        failed.len(),
        if busy > 0.0 { audio / busy } else { 0.0 }
    );
    if !skipped.is_empty() {
        println!(
            "dictionary_skipped[{}]: {}",
            skipped.len(),
            skipped
                .iter()
                .map(|t| toon(t))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    let mut cut = 0;
    if !rows.is_empty() {
        println!("results[{}]{{file,seconds,ms,text}}:", rows.len());
        for row in rows {
            let mut text = row.text.clone();
            if rows.len() > 1 && !full && text.chars().count() > PREVIEW_CHARS {
                let total = text.chars().count();
                text = text.chars().take(PREVIEW_CHARS).collect::<String>()
                    + &format!("... (cut, {total} chars)");
                cut += 1;
            }
            println!(
                "  {},{:.1},{:.0},{}",
                toon(&row.file),
                row.seconds,
                row.ms,
                toon(&text)
            );
        }
    }
    let segments: Vec<(&str, &trit::Segment)> = rows
        .iter()
        .flat_map(|row| row.segments.iter().map(|s| (row.file.as_str(), s)))
        .collect();
    if !segments.is_empty() {
        println!("segments[{}]{{file,start,end,text}}:", segments.len());
        for (file, s) in segments {
            println!(
                "  {},{:.2},{:.2},{}",
                toon(file),
                s.start,
                s.end,
                toon(&s.text)
            );
        }
    }
    if !failed.is_empty() {
        println!("errors[{}]{{file,error}}:", failed.len());
        for (file, e) in failed {
            println!("  {},{}", toon(file), toon(e));
        }
    }
    if cut > 0 {
        print_help(&[format!(
            "{cut} transcripts were cut; add --full or --format json for the whole text"
        )]);
    }
}

fn print_json(model: &Path, rows: &[Row], failed: &[(String, String)], skipped: &[String]) {
    let (audio, busy) = summary(rows);
    let value = serde_json::json!({
        "model": model.display().to_string(),
        "audio_seconds": audio,
        "busy_seconds": busy,
        "results": rows.iter().map(|row| serde_json::json!({
            "file": row.file,
            "seconds": row.seconds,
            "ms": row.ms,
            "text": row.text,
            "segments": row.segments.iter().map(trit::Segment::json).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "errors": failed.iter().map(|(file, e)| serde_json::json!({"file": file, "error": e})).collect::<Vec<_>>(),
        "dictionary_skipped": skipped,
    });
    println!("{value}");
}
