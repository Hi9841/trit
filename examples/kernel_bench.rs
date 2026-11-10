//! Times the kernels of `src/cpu/` in isolation: `cargo run --release --example kernel_bench -- peak`.
#![allow(clippy::print_stdout)]

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    for line in trit::kernel_bench(&args) {
        println!("{line}");
    }
}
