//! `trit speak`, `trit phonemes`, and `trit voices`.

use super::output::{print_help, slashes, tilde, toon};
use super::{error, usage_error, Command, Format, Outcome, Parsed};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const SPEAK_HELP: &str = "\
usage: trit speak <text>... [--out <file.wav>] [--model <dir>] [--voice <name>] [--phonemes] [--speed <x>] [--noise <x>] [--noise-duration <x>] [--seed <n>] [--format toon|json]
  Speaks text into a mono 16-bit WAV file: Kokoro-82M (24 kHz, 28 English voices, numbers and
  symbols read; the default when found) or a VITS voice (Meta MMS-TTS, 16 kHz, reads letters only).
  <text>          the text; several arguments join with spaces
  --out <file>    the WAV file to write (default: speech.wav)
  --model <dir>   voice folder or `trit models` name (default: TRIT_VOICE, then kokoro-82m, then mms-tts-eng)
  --voice <name>  Kokoro voice: af_heart (default), bm_george, ...; see `trit voices`; a,b blends two
  --phonemes      Kokoro: the text is phonemes (as `trit phonemes` writes them), one pass per line
  --speed <x>     speaking rate: 1 is the voice's pace (default), 1.25 faster, 0.8 slower
  --noise <x>     how much the voice varies: VITS noise_scale (default 0.667); Kokoro source noise (default 1)
  --noise-duration <x>  VITS: how much the timing varies (default 0.8)
  --seed <n>      seed of the noise (default 0); one text and one seed give the same audio
  --format        toon (default), json
examples:
  trit speak \"The build passed on March 3rd, 2026.\" --out done.wav
  trit speak \"Good morning from London.\" --voice bf_emma
  trit speak \"Ship it.\" --model mms-tts-eng --speed 0.9";

const PHONEMES_HELP: &str = "\
usage: trit phonemes <text>... [--british] [--no-fallback] [--file <path>]
  The phonemes Kokoro speaks for English text: misaki's lexicon, numbers, and stress, one line
  per line. Put a word's own phonemes in the text as [word](/phonemes/).
  <text>          the text; several arguments join with spaces
  --british       British English (the b… voices); default American (a… voices)
  --no-fallback   no phonemes for words that no dictionary holds (default: the BART fallback
                  PeterReid/graphemes_to_phonemes_en_us or _gb, when it is found)
  --file <path>   read the text from a file (UTF-8)
examples:
  trit phonemes \"The build passed on March 3rd.\"
  trit phonemes \"[Kokoro](/kˈOkəɹO/) speaks.\" --british";

const VOICES_HELP: &str = "\
usage: trit voices [--model <dir>]
  Lists the voices of a Kokoro folder (the default voice folder when none is named), with
  their accent: af/am American female/male, bf/bm British female/male.
examples:
  trit voices
  trit speak \"Hello.\" --voice bm_george";

pub(super) const SPEAK: Command = Command {
    name: "speak",
    help: SPEAK_HELP,
    values: &[
        "--out",
        "--model",
        "--speed",
        "--noise",
        "--noise-duration",
        "--seed",
        "--format",
        "--voice",
    ],
    switches: &["--phonemes"],
    in_skill: true,
    run: speak,
};

pub(super) const PHONEMES: Command = Command {
    name: "phonemes",
    help: PHONEMES_HELP,
    values: &["--file"],
    switches: &["--british", "--no-fallback"],
    in_skill: true,
    run: phonemes,
};

pub(super) const VOICES: Command = Command {
    name: "voices",
    help: VOICES_HELP,
    values: &["--model"],
    switches: &[],
    in_skill: true,
    run: voices,
};

const SPEAK_EXAMPLE: &str = "trit speak \"<text>\" --speed 1.1 --noise 0.5 --seed 7";

fn load_voice(parsed: &Parsed) -> Result<(PathBuf, trit::Voice), ExitCode> {
    let explicit = parsed.value("--model").map(Path::new);
    let Some(dir) = trit::find_voice(explicit) else {
        return Err(error(
            &trit::missing_voice(explicit),
            "Run `trit models` to list the voices",
        ));
    };
    let voice = trit::Voice::load(&dir)
        .map_err(|e| error(&format!("cannot load the voice: {e}"), "Run `trit doctor`"))?;
    Ok((dir, voice))
}

fn voicing(parsed: &Parsed) -> Result<trit::Voicing, ExitCode> {
    let number = |flag: &str| -> Result<Option<f32>, ExitCode> {
        parsed
            .value(flag)
            .map(|value| {
                value.parse::<f32>().map_err(|_| {
                    usage_error(&format!("{flag} {value} is not a number"), SPEAK_EXAMPLE)
                })
            })
            .transpose()
    };
    Ok(trit::Voicing {
        speed: number("--speed")?,
        noise: number("--noise")?,
        noise_duration: number("--noise-duration")?,
        seed: match parsed.value("--seed").map(str::parse::<u64>) {
            None => 0,
            Some(Ok(seed)) => seed,
            Some(Err(_)) => {
                return Err(usage_error(
                    "--seed must be a whole number of 0 or more",
                    SPEAK_EXAMPLE,
                ))
            }
        },
        voice: parsed.value("--voice").map(str::to_string),
        phonemes: parsed.switch("--phonemes"),
    })
}

fn speak(parsed: &Parsed) -> Outcome {
    let format = Format::of(parsed, &["toon", "json"])?;
    if parsed.positional.is_empty() {
        return Err(usage_error(
            "speak needs the text to speak",
            "trit speak \"<text>\" [--out speech.wav]",
        ));
    }
    let text = parsed.positional.join(" ");
    let voicing = voicing(parsed)?;
    let out = PathBuf::from(parsed.value("--out").unwrap_or("speech.wav"));
    let (_, voice) = load_voice(parsed)?;
    let reads = voice
        .phonemes(&text, &voicing)
        .map_err(|e| usage_error(&e, SPEAK_EXAMPLE))?;
    if reads.is_empty() {
        return Err(error(
            "the voice reads none of this text",
            voice.reads_nothing_help(),
        ));
    }
    let started = Instant::now();
    let samples = voice
        .speak_with(&text, &voicing)
        .map_err(|e| usage_error(&e, SPEAK_EXAMPLE))?;
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    trit::write_wav(&out, &samples, voice.sample_rate())
        .map_err(|e| error(&e, "pass --out <file.wav> in a folder you can write"))?;
    let seconds = samples.len() as f64 / f64::from(voice.sample_rate());
    let file = slashes(&out);
    match format {
        Format::Json => println!(
            "{}",
            serde_json::json!({"file": file, "seconds": trit::hundredths(seconds as f32),
                "ms": ms, "sample_rate": voice.sample_rate(), "reads": reads})
        ),
        _ => {
            println!("file: {}", toon(&file));
            println!("seconds: {seconds:.2}");
            println!("ms: {ms:.0}");
            println!("reads: {}", toon(&reads));
            print_help(&[format!(
                "Run `trit transcribe {}` to check what a speech model hears",
                toon(&file)
            )]);
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn phonemes(parsed: &Parsed) -> Outcome {
    let text = match parsed.value("--file") {
        Some(file) => std::fs::read_to_string(file).map_err(|e| {
            error(
                &format!("cannot read {file}: {e}"),
                "pass a UTF-8 text file",
            )
        })?,
        None if parsed.positional.is_empty() => {
            return Err(usage_error(
                "phonemes needs the text",
                "trit phonemes \"<text>\"",
            ))
        }
        None => parsed.positional.join(" "),
    };
    let phonemes = trit::phonemize(
        &text,
        parsed.switch("--british"),
        !parsed.switch("--no-fallback"),
    )
    .map_err(|e| error(&e, "Run `trit doctor`"))?;
    println!("{phonemes}");
    Ok(ExitCode::SUCCESS)
}

fn voices(parsed: &Parsed) -> Outcome {
    let (dir, voice) = load_voice(parsed)?;
    let names = voice.voices();
    println!("model: {}", toon(&tilde(&dir)));
    println!("kind: {}", voice.kind());
    if names.is_empty() {
        println!("voices: 0 (a VITS model has one voice)");
        return Ok(ExitCode::SUCCESS);
    }
    println!("voices[{}]{{name,accent}}:", names.len());
    for name in &names {
        println!("  {name},{}", trit::voice_accent(name));
    }
    print_help(&["Run `trit speak \"<text>\" --voice <name>` to use one".to_string()]);
    Ok(ExitCode::SUCCESS)
}
