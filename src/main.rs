//! `trit`: the command line of the Trit speech engine, built for agents
//! (https://axi.md): TOON on stdout, errors on stdout with the fix, exit
//! 0 / 1 / 2 for success / error / usage error, no prompts.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const DESCRIPTION: &str = "Speech to text on the CPU, offline: Parakeet, Canary, Whisper, and Moonshine models from Hugging Face, in any language they speak; 16 kHz WAV in, text out";
/// Longer transcripts in a list of several files are cut here; `--full` keeps them.
const PREVIEW_CHARS: usize = 500;

const TRANSCRIBE_HELP: &str = "\
usage: trit transcribe <path>... [--model <dir>] [--language <code>] [--dictionary <file>] [--task <t>] [--timestamps] [--cleanup] [--precision <p>] [--beam <n>] [--format toon|json|text] [--full]
  <path>          a 16 kHz WAV file, or a folder: every .wav file in it and its subfolders
  --model <dir>   speech model folder or `trit models` name: Parakeet, Canary, Whisper, Moonshine (default: TRIT_MODEL, then the places `trit doctor` lists)
  --language <code>  spoken language, ISO 639-1 (en, de, ...); default: the model detects it
  --dictionary <file>  terms to favor, one per line (names, product words)
  --task <t>      transcribe (default), or translate into English (multilingual Whisper, Canary)
  --timestamps    timed segments too: sentences for Parakeet, windows of up to 30 s (40 s Canary) for the others
  --cleanup       clean each transcript with S1-mini (see `trit clean --help`); ms then includes it
  --cleanup-model <dir>, --styling, --structure, --context  as for `trit clean`
  --precision     auto (default: int8 for layers 384 wide and more, f16 below), int8 (least memory), f16, f32 (as the reference runtimes)
  --beam <n>      beams for Whisper, Moonshine, and Canary (default: 5 for Whisper and Moonshine, 1 (greedy) for Canary)
  --format        toon (default), json (full text, for programs), text (one transcript per line)
  --full          do not cut transcripts longer than 500 characters in a list of several files
examples:
  trit transcribe meeting.wav
  trit transcribe clips/ --format json
  trit transcribe take.wav --dictionary terms.txt --cleanup
  trit transcribe interview.wav --model models/whisper-large-v3-turbo --language de";

const CLEAN_HELP: &str = "\
usage: trit clean <text>... [--model <dir>] [--styling <s>] [--structure <s>] [--context <c>] [--precision <p>] [--format toon|json|text]
  Rewrites a raw transcript as written text with Superwhisper S1-mini (English): removes
  fillers, resolves self-corrections, adds punctuation and casing, writes numbers and addresses.
  <text>          the raw transcript; several arguments join with spaces
  --model <dir>   S1-mini folder (default: TRIT_CLEANUP_MODEL, models/s1-mini, the Hugging Face cache)
  --styling       casual, semi-casual, semi-formal (default), formal
  --structure     prose (default), lists
  --context       general (default), email
  --precision     auto (default), int8, f16, f32; as for `trit transcribe`
  --format        toon (default), json, text
examples:
  trit clean \"so um send the report by friday no wait thursday\"
  trit clean \"hey sarah can you send the numbers thanks john\" --context email";

const MODELS_HELP: &str = "\
usage: trit models
  Lists the models Trit is known to run, whether each is on this computer, and how to get it.
  Pass a name to --model: trit transcribe <file.wav> --model <name>
examples:
  trit models";

const DOCTOR_HELP: &str = "\
usage: trit doctor
  Checks the model folder and the CPU path; prints the command that fixes each problem.
examples:
  trit doctor";

const SKILL_HELP: &str = "\
usage: trit skill [--check <file>]
  Prints the agent skill (SKILL.md) for Trit. It comes from the same text as the home view.
  --check <file>  exit 1 when <file> differs from the skill this version prints
examples:
  trit skill > skill/SKILL.md
  trit skill --check skill/SKILL.md";

enum Format {
    Toon,
    Json,
    Text,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [only] = args.as_slice() {
        if matches!(only.as_str(), "--version" | "-v" | "-V") {
            println!("{}", trit::VERSION);
            return ExitCode::SUCCESS;
        }
    }
    let Some((command, rest)) = args.split_first() else {
        home();
        return ExitCode::SUCCESS;
    };
    match command.as_str() {
        "--help" | "-h" | "help" => {
            home();
            ExitCode::SUCCESS
        }
        "transcribe" => transcribe(rest),
        "clean" => clean(rest),
        "models" => models(rest),
        "doctor" => doctor(rest),
        "skill" => skill(rest),
        other => usage_error(
            &format!("unknown command `{other}`"),
            "commands: transcribe, clean, models, doctor, skill. Run `trit` for the home view",
        ),
    }
}

fn usage_error(message: &str, help: &str) -> ExitCode {
    println!("error: {}", toon(message));
    println!("help: {}", toon(help));
    ExitCode::from(2)
}

fn error(message: &str, help: &str) -> ExitCode {
    println!("error: {}", toon(message));
    println!("help: {}", toon(help));
    ExitCode::FAILURE
}

/// Flags of one command: names that take a value, and names that do not.
struct Parsed {
    positional: Vec<String>,
    values: Vec<(String, String)>,
    switches: Vec<String>,
    help: bool,
}

impl Parsed {
    fn value(&self, name: &str) -> Option<&str> {
        self.values
            .iter()
            .rev()
            .find(|(flag, _)| flag == name)
            .map(|(_, value)| value.as_str())
    }

    fn switch(&self, name: &str) -> bool {
        self.switches.iter().any(|flag| flag == name)
    }
}

/// Splits `args`; an unknown flag or a flag without its value is a usage error
/// that names the valid flags.
fn parse(
    command: &str,
    args: &[String],
    with_value: &[&str],
    switches: &[&str],
) -> Result<Parsed, ExitCode> {
    let valid = || {
        let all: Vec<&str> = with_value.iter().chain(switches).copied().collect();
        if all.is_empty() {
            format!("`{command}` takes no flags (--help always allowed)")
        } else {
            format!(
                "valid flags for `{command}`: {} (--help always allowed)",
                all.join(", ")
            )
        }
    };
    let mut parsed = Parsed {
        positional: Vec::new(),
        values: Vec::new(),
        switches: Vec::new(),
        help: false,
    };
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--help" || arg == "-h" {
            parsed.help = true;
        } else if with_value.contains(&arg.as_str()) {
            let Some(value) = iter.next() else {
                return Err(usage_error(&format!("{arg} needs a value"), &valid()));
            };
            parsed.values.push((arg.clone(), value.clone()));
        } else if switches.contains(&arg.as_str()) {
            parsed.switches.push(arg.clone());
        } else if arg.starts_with('-') && arg.len() > 1 {
            return Err(usage_error(
                &format!("unknown flag {arg} for `{command}`"),
                &valid(),
            ));
        } else {
            parsed.positional.push(arg.clone());
        }
    }
    Ok(parsed)
}

/// The home view: what this tool is, its live state, and the next commands.
fn home() {
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
    println!(
        "cpu: {}, {} threads ({} for Whisper)",
        trit::cpu_path(),
        trit::threads(trit::Spread::OneCache),
        trit::threads(trit::Spread::AllCores)
    );
    let mut help = Vec::new();
    if model.is_none() {
        help.push(format!(
            "Run `hf download {} --local-dir models/parakeet-redux` to get the model (178 MB)",
            trit::MODEL_REPO
        ));
    }
    help.extend(HOME_HELP.iter().map(|line| line.to_string()));
    print_help(&help);
}

const HOME_HELP: [&str; 5] = [
    "Run `trit transcribe <file.wav>` to get the text of a recording",
    "Run `trit models` to list the speech models (Parakeet, Canary, Whisper, Moonshine) and get one",
    "Run `trit transcribe <folder> --model <name> --format json` for every WAV file in a folder",
    "Run `trit clean \"<raw transcript>\"` to rewrite text as clean written English (S1-mini)",
    "Run `trit doctor` to check the models and the CPU",
];

fn print_help(lines: &[String]) {
    println!("help[{}]:", lines.len());
    for line in lines {
        println!("  {line}");
    }
}

struct Row {
    file: String,
    seconds: f64,
    ms: f64,
    text: String,
    /// With `--timestamps` only.
    segments: Vec<trit::Segment>,
}

fn transcribe(args: &[String]) -> ExitCode {
    let parsed = match parse(
        "transcribe",
        args,
        &[
            "--model",
            "--language",
            "--dictionary",
            "--format",
            "--cleanup-model",
            "--styling",
            "--structure",
            "--context",
            "--precision",
            "--beam",
            "--task",
        ],
        &["--full", "--cleanup", "--timestamps"],
    ) {
        Ok(parsed) => parsed,
        Err(code) => return code,
    };
    if parsed.help {
        println!("{TRANSCRIBE_HELP}");
        return ExitCode::SUCCESS;
    }
    let format = match parsed.value("--format").unwrap_or("toon") {
        "toon" => Format::Toon,
        "json" => Format::Json,
        "text" => Format::Text,
        other => {
            return usage_error(
                &format!("unknown format `{other}`"),
                "valid formats: toon, json, text",
            )
        }
    };
    if parsed.positional.is_empty() {
        return usage_error(
            "transcribe needs a WAV file or a folder",
            "trit transcribe <file.wav> [--format json]",
        );
    }
    // Check every input before the model loads, so a typo costs no load time.
    let mut files = Vec::new();
    for path in &parsed.positional {
        let path = PathBuf::from(path);
        if path.is_dir() {
            wav_files(&path, &mut files);
        } else if path.is_file() {
            files.push(path);
        } else {
            return error(
                &format!("{} does not exist", path.display()),
                "give a .wav file or a folder of them: trit transcribe <path>",
            );
        }
    }
    if files.is_empty() {
        println!(
            "files: 0 WAV files found in {}",
            toon(&parsed.positional.join(" "))
        );
        return ExitCode::SUCCESS;
    }
    let terms = match parsed.value("--dictionary") {
        Some(file) => match std::fs::read_to_string(file) {
            Ok(text) => Some(text.lines().map(str::to_string).collect::<Vec<_>>()),
            Err(e) => {
                return error(
                    &format!("cannot read dictionary {file}: {e}"),
                    "give a text file with one term per line: --dictionary <file>",
                )
            }
        },
        None => None,
    };
    let options = trit::Options {
        language: parsed.value("--language").map(str::to_lowercase),
        beam: match parsed.value("--beam").map(str::parse::<usize>) {
            None => 0,
            Some(Ok(beam)) if beam >= 1 => beam,
            Some(_) => return usage_error("--beam needs a whole number of 1 or more", "--beam 5"),
        },
        task: match parsed.value("--task").map(trit::Task::parse) {
            None => trit::Task::Transcribe,
            Some(Some(task)) => task,
            Some(None) => {
                return usage_error(
                    "unknown task",
                    &format!("valid tasks: {}", trit::Task::NAMES.join(", ")),
                )
            }
        },
        ..trit::Options::default()
    };
    let style = match style(&parsed) {
        Ok(style) => style,
        Err(code) => return code,
    };
    let load = match load_options(&parsed) {
        Ok(load) => load,
        Err(code) => return code,
    };
    let cleaner = if parsed.switch("--cleanup") || parsed.value("--cleanup-model").is_some() {
        match load_cleaner(parsed.value("--cleanup-model"), &load) {
            Ok(cleaner) => Some(cleaner),
            Err(code) => return code,
        }
    } else {
        None
    };
    let explicit = parsed.value("--model").map(Path::new);
    let Some(model_dir) = trit::find_model(explicit) else {
        return error(
            &trit::missing_model(explicit),
            "Run `trit doctor` to see where Trit looks",
        );
    };
    let engine = match trit::Trit::load_with(&model_dir, &load) {
        Ok(engine) => engine,
        Err(e) => return error(&format!("cannot load the model: {e}"), "Run `trit doctor`"),
    };
    trit::release_memory();
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
        let started = Instant::now();
        let segments = if parsed.switch("--timestamps") {
            engine.segments(&samples, &options)
        } else {
            Vec::new()
        };
        let mut text = if parsed.switch("--timestamps") {
            let texts: Vec<&str> = segments.iter().map(|s| s.text.as_str()).collect();
            texts.join(" ")
        } else {
            engine.transcribe_with(&samples, &options)
        };
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
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
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
            "segments": row.segments.iter().map(|s| serde_json::json!({
                "start": s.start,
                "end": s.end,
                "text": s.text,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "errors": failed.iter().map(|(file, e)| serde_json::json!({"file": file, "error": e})).collect::<Vec<_>>(),
        "dictionary_skipped": skipped,
    });
    println!("{value}");
}

/// The cleanup style from `--styling`, `--structure`, and `--context`.
fn style(parsed: &Parsed) -> Result<trit::Style, ExitCode> {
    let pick = |flag: &str, names: &[&str]| -> Result<Option<String>, ExitCode> {
        match parsed.value(flag) {
            Some(value) if !names.contains(&value) => Err(usage_error(
                &format!("unknown {} `{value}`", flag.trim_start_matches('-')),
                &format!("valid values for {flag}: {}", names.join(", ")),
            )),
            value => Ok(value.map(str::to_string)),
        }
    };
    let mut style = trit::Style::default();
    if let Some(v) = pick("--styling", &trit::Styling::NAMES)? {
        style.styling = trit::Styling::parse(&v).unwrap_or_default();
    }
    if let Some(v) = pick("--structure", &trit::Structure::NAMES)? {
        style.structure = trit::Structure::parse(&v).unwrap_or_default();
    }
    if let Some(v) = pick("--context", &trit::Context::NAMES)? {
        style.context = trit::Context::parse(&v).unwrap_or_default();
    }
    Ok(style)
}

/// `--precision`, as load options.
fn load_options(parsed: &Parsed) -> Result<trit::LoadOptions, ExitCode> {
    let precision = match parsed.value("--precision") {
        Some(name) => trit::Precision::parse(name)
            .map_err(|e| usage_error(&e, "--precision auto (the default) fits most uses"))?,
        None => trit::Precision::Auto,
    };
    Ok(trit::LoadOptions { precision })
}

fn load_cleaner(
    explicit: Option<&str>,
    options: &trit::LoadOptions,
) -> Result<trit::Cleaner, ExitCode> {
    let explicit = explicit.map(Path::new);
    let Some(dir) = trit::find_cleanup_model(explicit) else {
        return Err(error(
            &match explicit {
                Some(dir) => format!("{} is not a model folder", dir.display()),
                None => "no cleanup model found".to_string(),
            },
            &format!(
                "Run `hf download {} --local-dir models/s1-mini` to get it (1.5 GB), or pass --model <dir>",
                trit::CLEANUP_REPO
            ),
        ));
    };
    trit::Cleaner::load_with(&dir, options).map_err(|e| {
        error(
            &format!("cannot load the cleanup model: {e}"),
            "Run `trit doctor`",
        )
    })
}

fn clean(args: &[String]) -> ExitCode {
    let parsed = match parse(
        "clean",
        args,
        &[
            "--model",
            "--styling",
            "--structure",
            "--context",
            "--format",
            "--precision",
        ],
        &[],
    ) {
        Ok(parsed) => parsed,
        Err(code) => return code,
    };
    if parsed.help {
        println!("{CLEAN_HELP}");
        return ExitCode::SUCCESS;
    }
    let format = parsed.value("--format").unwrap_or("toon");
    if !["toon", "json", "text"].contains(&format) {
        return usage_error(
            &format!("unknown format `{format}`"),
            "valid formats: toon, json, text",
        );
    }
    let style = match style(&parsed) {
        Ok(style) => style,
        Err(code) => return code,
    };
    if parsed.positional.is_empty() {
        return usage_error(
            "clean needs the text to clean",
            "trit clean \"<raw transcript>\" [--styling semi-formal]",
        );
    }
    let raw = parsed.positional.join(" ");
    let options = match load_options(&parsed) {
        Ok(options) => options,
        Err(code) => return code,
    };
    let cleaner = match load_cleaner(parsed.value("--model"), &options) {
        Ok(cleaner) => cleaner,
        Err(code) => return code,
    };
    let started = Instant::now();
    let text = cleaner.clean(&raw, &style);
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    match format {
        "text" => println!("{text}"),
        "json" => println!(
            "{}",
            serde_json::json!({"text": text, "ms": ms, "styling": style.styling.name(),
                "structure": style.structure.name(), "context": style.context.name()})
        ),
        _ => {
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
    ExitCode::SUCCESS
}

fn models(args: &[String]) -> ExitCode {
    let parsed = match parse("models", args, &[], &[]) {
        Ok(parsed) => parsed,
        Err(code) => return code,
    };
    if parsed.help {
        println!("{MODELS_HELP}");
        return ExitCode::SUCCESS;
    }
    let found = |entry: &trit::Known| {
        trit::candidates(entry.repo)
            .into_iter()
            .find(|dir| trit::is_model(dir))
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
    ExitCode::SUCCESS
}

fn doctor(args: &[String]) -> ExitCode {
    let parsed = match parse("doctor", args, &[], &[]) {
        Ok(parsed) => parsed,
        Err(code) => return code,
    };
    if parsed.help {
        println!("{DOCTOR_HELP}");
        return ExitCode::SUCCESS;
    }
    println!("version: {}", trit::VERSION);
    let cpu = trit::cpu_path();
    println!(
        "cpu: {cpu}, {} threads ({} for Whisper)",
        trit::threads(trit::Spread::OneCache),
        trit::threads(trit::Spread::AllCores)
    );
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
    if cpu == "scalar" {
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
        ExitCode::SUCCESS
    } else {
        print_help(&problems);
        if model.is_none() {
            ExitCode::FAILURE
        } else {
            ExitCode::SUCCESS
        }
    }
}

/// The installable agent skill. Built from the home view text so the two
/// cannot drift; `--check` fails when a committed copy is stale.
fn skill_text() -> String {
    let mut text = String::new();
    text.push_str("---\nname: trit\ndescription: Transcribe speech from WAV files to text on the CPU, offline, with the Trit engine and a Parakeet, Canary, Whisper, or Moonshine model, in any language the model speaks; clean a raw transcript with S1-mini. Use when an agent must turn a recording into text, or add speech to text to an app in Rust, C, C++, Python, C#, Node, or Bun.\n---\n\n");
    text.push_str("# Trit\n\n");
    text.push_str(DESCRIPTION);
    text.push_str(".\n\n## Commands\n\n");
    for line in HOME_HELP {
        text.push_str(&format!("- {}\n", line.trim_start_matches("Run ")));
    }
    text.push_str("- `trit skill --check <file>` to check a copy of this skill\n");
    text.push_str("\nOutput is TOON on stdout. Errors are on stdout too, as `error:` and `help:` lines. Exit codes: 0 success, 1 error, 2 usage error.\n\n");
    text.push_str("## transcribe\n\n```text\n");
    text.push_str(TRANSCRIBE_HELP);
    text.push_str("\n```\n\n## clean\n\n```text\n");
    text.push_str(CLEAN_HELP);
    text.push_str("\n```\n\n## Get a model\n\n`trit models` lists the known models, where each is, and its download command. Pass a name or a folder to `--model`.\n\n| Name | Family | Languages | MB |\n|---|---|---|---:|\n");
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
    text.push_str("## Use it from code\n\nRead `trit/README.md`: the Rust API, the C ABI (`include/trit.h`), and bindings for Python, C#, and Node. Any other language with a C FFI uses `include/trit.h` directly.\n");
    text
}

fn skill(args: &[String]) -> ExitCode {
    let parsed = match parse("skill", args, &["--check"], &[]) {
        Ok(parsed) => parsed,
        Err(code) => return code,
    };
    if parsed.help {
        println!("{SKILL_HELP}");
        return ExitCode::SUCCESS;
    }
    let text = skill_text();
    let Some(file) = parsed.value("--check") else {
        print!("{text}");
        return ExitCode::SUCCESS;
    };
    match std::fs::read_to_string(file) {
        Ok(committed) if committed.replace("\r\n", "\n") == text => {
            println!("skill: {} is current", toon(file));
            ExitCode::SUCCESS
        }
        Ok(_) => error(&format!("{file} is stale"), &format!("trit skill > {file}")),
        Err(e) => error(
            &format!("cannot read {file}: {e}"),
            &format!("trit skill > {file}"),
        ),
    }
}

/// The path, absolute when it exists, with `/` separators and the home
/// folder written as `~`. Windows accepts `/`, and TOON then needs no escapes.
fn tilde(path: &Path) -> String {
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let text = slashes(&path);
    let text = text.strip_prefix("//?/").unwrap_or(&text);
    match std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) {
        Ok(home) if !home.is_empty() && text.starts_with(&slashes(Path::new(&home))) => {
            format!("~{}", &text[home.len()..])
        }
        _ => text.to_string(),
    }
}

fn slashes(path: &Path) -> String {
    path.display()
        .to_string()
        .replace(std::path::MAIN_SEPARATOR, "/")
}

/// A TOON string value: bare when it cannot be read as anything else,
/// quoted with `\\ \" \n \r \t` escapes otherwise (TOON spec, section 7).
fn toon(value: &str) -> String {
    let numeric = {
        let body = value.strip_prefix('-').unwrap_or(value);
        !body.is_empty()
            && body
                .chars()
                .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'))
            && body.starts_with(|c: char| c.is_ascii_digit())
    };
    let quote = value.is_empty()
        || value != value.trim()
        || matches!(value, "true" | "false" | "null")
        || numeric
        || value.starts_with('-')
        || value
            .chars()
            .any(|c| matches!(c, ':' | '"' | '\\' | '[' | ']' | '{' | '}' | ',') || c.is_control());
    if !quote {
        return value.to_string();
    }
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push(' '),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
