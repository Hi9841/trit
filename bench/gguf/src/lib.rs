//! What both engines of the GGUF benchmark share, so they differ only in the
//! engine: the clips, the warm-up, the timing of one call per clip, the
//! memory figures, and one JSON line per clip.
//!
//! usage: <engine> <model> <clips root> <out.jsonl> <set>...

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct Args {
    pub model: PathBuf,
    root: PathBuf,
    out: PathBuf,
    sets: Vec<String>,
}

impl Args {
    pub fn parse() -> Self {
        let args: Vec<String> = std::env::args().collect();
        if args.len() < 5 {
            eprintln!(
                "usage: {} <model> <clips root> <out.jsonl> <set>...",
                args[0]
            );
            std::process::exit(2);
        }
        Self {
            model: PathBuf::from(&args[1]),
            root: PathBuf::from(&args[2]),
            out: PathBuf::from(&args[3]),
            sets: args[4..].to_vec(),
        }
    }
}

/// 16 kHz mono 16-bit samples as f32 in -1..1.
fn read(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap();
    let spec = reader.spec();
    assert_eq!(
        (spec.sample_rate, spec.channels),
        (16_000, 1),
        "{}",
        path.display()
    );
    reader
        .samples::<i16>()
        .map(|x| f32::from(x.unwrap()) / 32768.0)
        .collect()
}

fn clips(root: &Path, set: &str) -> Vec<PathBuf> {
    let mut clips: Vec<_> = std::fs::read_dir(root.join(set))
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "wav"))
        .collect();
    clips.sort();
    clips
}

/// Working set and private bytes of this process, in MB.
pub fn memory_mb() -> (u64, u64) {
    #[cfg(windows)]
    // SAFETY: the counters struct is sized for the call; the handle is this process's.
    unsafe {
        use windows_sys::Win32::System::ProcessStatus::{
            GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut c).cast(), c.cb);
        (
            (c.WorkingSetSize >> 20) as u64,
            (c.PrivateUsage >> 20) as u64,
        )
    }
    #[cfg(not(windows))]
    (0, 0)
}

/// Runs `transcribe` on every clip of every set, after one warm-up call, and
/// writes one JSON line per clip, then one line with the memory figures.
/// `load_ms` and the memory after load come from the caller.
pub fn run(args: &Args, engine: &str, load_ms: f64, mut transcribe: impl FnMut(&[f32]) -> String) {
    let loaded = memory_mb();
    let mut out = std::fs::File::create(&args.out).unwrap();
    let first = clips(&args.root, &args.sets[0]);
    transcribe(&read(&first[0]));
    for set in &args.sets {
        for clip in clips(&args.root, set) {
            let pcm = read(&clip);
            let started = Instant::now();
            let text = transcribe(&pcm);
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            let line = serde_json::json!({
                "set": set,
                "clip": clip.file_stem().unwrap().to_string_lossy(),
                "hyp": text.trim(),
                "ms": (ms * 10.0).round() / 10.0,
                "audio_s": pcm.len() as f64 / 16_000.0,
            });
            writeln!(out, "{line}").unwrap();
        }
        eprintln!("{engine}: {set} done");
    }
    let done = memory_mb();
    let line = serde_json::json!({
        "engine": engine,
        "load_ms": (load_ms * 10.0).round() / 10.0,
        "load_ws_mb": loaded.0, "load_private_mb": loaded.1,
        "run_ws_mb": done.0, "run_private_mb": done.1,
    });
    writeln!(out, "{line}").unwrap();
}
