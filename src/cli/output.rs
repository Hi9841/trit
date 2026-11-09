//! Text on stdout: TOON values and paths.

use std::path::Path;

pub(super) fn print_help(lines: &[String]) {
    println!("help[{}]:", lines.len());
    for line in lines {
        println!("  {line}");
    }
}

/// The path, absolute when it exists, with `/` separators and the home
/// folder written as `~`. Windows accepts `/`, and TOON then needs no escapes.
pub(super) fn tilde(path: &Path) -> String {
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let text = slashes(&path);
    let text = text.strip_prefix("//?/").unwrap_or(&text);
    match std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) {
        // The home folder, or a path below it: not a folder beside it
        // whose name starts the same (`/Users/hi2`).
        Ok(home)
            if !home.is_empty()
                && text
                    .strip_prefix(&slashes(Path::new(&home)))
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('/')) =>
        {
            format!("~{}", &text[home.len()..])
        }
        _ => text.to_string(),
    }
}

pub(super) fn slashes(path: &Path) -> String {
    path.display()
        .to_string()
        .replace(std::path::MAIN_SEPARATOR, "/")
}

/// A TOON string value: bare when it cannot be read as anything else,
/// quoted with `\\ \" \n \r \t` escapes otherwise (TOON spec, section 7).
pub(super) fn toon(value: &str) -> String {
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
