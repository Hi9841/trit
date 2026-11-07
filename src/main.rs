//! `trit`: the command line of the Trit speech engine, built for agents
//! (https://axi.md): TOON on stdout, errors on stdout with the fix, exit
//! 0 / 1 / 2 for success / error / usage error, no prompts.

mod cli;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> std::process::ExitCode {
    cli::run()
}
