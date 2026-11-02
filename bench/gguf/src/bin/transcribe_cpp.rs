//! transcribe.cpp on a GGUF file, as Hex 3.2.0 calls it on the CPU
//! (`src/transcription/gguf_session.rs`, `local_transcriber.rs`): one
//! model, a new session for every clip, segment timestamps when the model
//! has them, language "en". `HEXENG_THREADS` sets the session's threads;
//! `HEXENG_LANGUAGE` the language (`auto`: the model detects it).
use transcribe_cpp::{Backend, Model, ModelOptions, RunOptions, SessionOptions, TimestampKind};

fn main() {
    let args = gguf_bench::Args::parse();
    let gguf = std::fs::read_dir(&args.model)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.extension().is_some_and(|x| x == "gguf"))
        .expect("no .gguf file in the model folder");
    let threads = std::env::var("HEXENG_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let started = std::time::Instant::now();
    let model = Model::load_with(
        &gguf,
        &ModelOptions {
            backend: Backend::Cpu,
            ..Default::default()
        },
    )
    .unwrap();
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let capabilities = model.capabilities().unwrap();
    let options = RunOptions {
        timestamps: if capabilities.max_timestamp_kind == TimestampKind::None {
            TimestampKind::None
        } else {
            TimestampKind::Segment
        },
        language: match std::env::var("HEXENG_LANGUAGE").as_deref() {
            Ok("auto") => None,
            Ok(code) => Some(code.to_string()),
            Err(_) => Some("en".into()),
        },
        ..Default::default()
    };
    let session_options = SessionOptions {
        n_threads: threads,
        ..Default::default()
    };
    eprintln!(
        "transcribe.cpp: {} {}, {threads} threads",
        model.arch(),
        model.variant()
    );
    // Hex makes a new session for every run (the old one's scratch stays
    // sized to its input otherwise), so each clip's time includes it.
    let mut session_ms = 0.0;
    gguf_bench::run(&args, "transcribe.cpp", load_ms, |pcm| {
        let made = std::time::Instant::now();
        let mut session = model.session_with(&session_options).unwrap();
        session_ms += made.elapsed().as_secs_f64() * 1000.0;
        session
            .run(pcm, &options)
            .map(|t| t.text)
            .unwrap_or_else(|e| format!("<error {e}>"))
    });
    eprintln!("transcribe.cpp: {session_ms:.0} ms of all clip times went to new sessions");
}
