//! Plain loops compiled for wider vectors. The crate targets the x86-64
//! baseline, so a loop like `*o += w * v` over 256 channels compiles to 4-lane
//! SSE2. `multiversion!` compiles one function body three times (AVX-512,
//! AVX2, baseline) and calls the one this CPU runs. Rust never fuses a
//! multiply and an add on its own, so every version rounds alike and the
//! results are bit-identical.

/// Wraps a free function with plain-identifier arguments and no return value.
/// Parallel tasks call through `dyn Fn` and do not inherit target features,
/// so call the wrapped function from inside the task.
macro_rules! multiversion {
    ($(#[$meta:meta])* $vis:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $body:block) => {
        $(#[$meta])*
        $vis fn $name($($arg: $ty),*) {
            #[inline(always)]
            fn imp($($arg: $ty),*) $body
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
            unsafe fn avx512($($arg: $ty),*) {
                imp($($arg),*)
            }
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn avx2($($arg: $ty),*) {
                imp($($arg),*)
            }
            match $crate::cpu::fgemm::level() {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: the level was detected from the CPU.
                $crate::cpu::fgemm::Level::Avx512 => unsafe { avx512($($arg),*) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: the level was detected from the CPU.
                $crate::cpu::fgemm::Level::Avx2 => unsafe { avx2($($arg),*) },
                $crate::cpu::fgemm::Level::Scalar => imp($($arg),*),
            }
        }
    };
}
