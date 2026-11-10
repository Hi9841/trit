//! The kernels written again from nothing, next to the first set
//! (`kernels.rs`, `int8.rs`, `fgemm.rs`), for a comparison of the two.
//! `TRIT_KERNELS=v2` makes the engine use this set (`v2-int` or `v2-float`:
//! one product only); the default is the first set. This set has the paths
//! of the first: AVX-512, AVX2 (with or without VNNI), NEON (with or
//! without `sdot` and `smmla`), and scalar on every other CPU.
//!
//! - `product.rs`: the integer product of ternary and int8 layers.
//! - `quantize.rs`: the activations as i8 for it.
//! - `float.rs`: the f32 product with f32 or f16 weights (the f16
//!   matrix-vector product of `half.rs` too).
//! - `nn.rs`: the f32 product with `w` as k x n (`matmul_nn_into`).
//! - `activation.rs`: the scalar path of the activations.
//!
//! Every path of this set gives the same bits on every CPU: the integer
//! products, the float products (one sum order of 16 lanes), and the
//! activations. The text of a model does not change with the CPU.
//! - `bench.rs`: both sets on the same inputs (`examples/kernel_bench.rs`).

pub(crate) mod activation;
pub mod bench;
pub mod float;
pub mod nn;
pub mod product;
mod quantize;

/// What `TRIT_KERNELS` asks for: `v2` (both products), `v2-int`, or `v2-float`.
fn asked() -> &'static str {
    static ASKED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    ASKED.get_or_init(|| std::env::var("TRIT_KERNELS").unwrap_or_default())
}

/// Whether the ternary and int8 layers that load from now on use this set.
pub fn enabled() -> bool {
    matches!(asked(), "v2" | "v2-int")
}

/// Whether the float products use this set.
pub fn float_enabled() -> bool {
    matches!(asked(), "v2" | "v2-float")
}
