//! Trit on a model folder (a GGUF file, a `.nemo`, or a Hugging Face
//! folder), as TDT runs it: one engine, `Options::language("en")`, mimalloc.
//! `TRIT_LANGUAGE` sets another language (`auto`: the model detects it).
//! `TRIT_PRECISION` picks the precision (default `auto`: a GGUF file's
//! stored types); `TRIT_ISA` the kernel path.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let args = gguf_bench::Args::parse();
    let precision = std::env::var("TRIT_PRECISION").unwrap_or_else(|_| "auto".into());
    let started = std::time::Instant::now();
    let engine = trit::Trit::load_with(
        &args.model,
        &trit::LoadOptions {
            precision: trit::Precision::parse(&precision).unwrap(),
            ..Default::default()
        },
    )
    .unwrap();
    trit::release_memory();
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    eprintln!("trit: {} {precision}, {}", engine.kind(), trit::cpu_path());
    let options = match std::env::var("TRIT_LANGUAGE").as_deref() {
        Ok("auto") => trit::Options::default(),
        Ok(code) => trit::Options::language(code),
        Err(_) => trit::Options::language("en"),
    };
    gguf_bench::run(&args, "trit", load_ms, |pcm| {
        engine.transcribe_with(pcm, &options)
    });
}
