//! The commands of `trit`. A command is one `Command` value in its own file.

// The CLI writes its results to stdout. The library must not print: its
// messages go through `trit::set_log`.
#![allow(clippy::print_stdout)]

mod clean;
mod doctor;
mod home;
mod output;
mod skill;
mod speak;
mod transcribe;

use output::toon;
use std::process::ExitCode;

/// `Err` is a failure already printed.
type Outcome = Result<ExitCode, ExitCode>;

struct Command {
    name: &'static str,
    help: &'static str,
    /// Flags that take a value.
    values: &'static [&'static str],
    switches: &'static [&'static str],
    /// The skill prints its help.
    in_skill: bool,
    run: fn(&Parsed) -> Outcome,
}

const COMMANDS: [Command; 8] = [
    transcribe::COMMAND,
    clean::COMMAND,
    speak::SPEAK,
    speak::PHONEMES,
    speak::VOICES,
    doctor::MODELS,
    doctor::DOCTOR,
    skill::COMMAND,
];

pub fn run() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [only] = args.as_slice() {
        if matches!(only.as_str(), "--version" | "-v" | "-V") {
            println!("{}", trit::VERSION);
            return ExitCode::SUCCESS;
        }
    }
    let Some((name, rest)) = args.split_first() else {
        home::print();
        return ExitCode::SUCCESS;
    };
    if matches!(name.as_str(), "--help" | "-h" | "help") {
        home::print();
        return ExitCode::SUCCESS;
    }
    let Some(command) = COMMANDS.iter().find(|command| command.name == name) else {
        let names: Vec<&str> = COMMANDS.iter().map(|command| command.name).collect();
        return usage_error(
            &format!("unknown command `{name}`"),
            &format!(
                "commands: {}. Run `trit` for the home view",
                names.join(", ")
            ),
        );
    };
    let parsed = match parse(command, rest) {
        Ok(parsed) => parsed,
        Err(code) => return code,
    };
    if parsed.help {
        println!("{}", command.help);
        return ExitCode::SUCCESS;
    }
    (command.run)(&parsed).unwrap_or_else(|code| code)
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

fn parse(command: &Command, args: &[String]) -> Result<Parsed, ExitCode> {
    let name = command.name;
    let valid = || {
        let all: Vec<&str> = command
            .values
            .iter()
            .chain(command.switches)
            .copied()
            .collect();
        if all.is_empty() {
            format!("`{name}` takes no flags (--help always allowed)")
        } else {
            format!(
                "valid flags for `{name}`: {} (--help always allowed)",
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
        } else if command.values.contains(&arg.as_str()) {
            let Some(value) = iter.next() else {
                return Err(usage_error(&format!("{arg} needs a value"), &valid()));
            };
            parsed.values.push((arg.clone(), value.clone()));
        } else if command.switches.contains(&arg.as_str()) {
            parsed.switches.push(arg.clone());
        } else if arg.starts_with('-') && arg.len() > 1 {
            return Err(usage_error(
                &format!("unknown flag {arg} for `{name}`"),
                &valid(),
            ));
        } else {
            parsed.positional.push(arg.clone());
        }
    }
    Ok(parsed)
}

#[derive(Clone, Copy)]
enum Format {
    Toon,
    Json,
    Text,
}

impl Format {
    fn of(parsed: &Parsed, valid: &[&str]) -> Result<Self, ExitCode> {
        match parsed.value("--format").unwrap_or("toon") {
            name if !valid.contains(&name) => Err(usage_error(
                &format!("unknown format `{name}`"),
                &format!("valid formats: {}", valid.join(", ")),
            )),
            "json" => Ok(Self::Json),
            "text" => Ok(Self::Text),
            _ => Ok(Self::Toon),
        }
    }
}

fn load_options(parsed: &Parsed) -> Result<trit::LoadOptions, ExitCode> {
    let precision = parsed
        .value("--precision")
        .map(trit::Precision::parse)
        .transpose()
        .map_err(|e| usage_error(&e, "--precision auto (the default) fits most uses"))?;
    let device = parsed
        .value("--device")
        .map(trit::Device::parse)
        .transpose()
        .map_err(|e| usage_error(&e, &format!("--device {}", trit::Device::NAMES.join("|"))))?;
    Ok(trit::LoadOptions {
        precision: precision.unwrap_or_default(),
        device: device.unwrap_or_default(),
        cache: false,
    })
}
