//! One read of `TRIT_ISA` for the float level, the ternary ISA, and the int8 path.
//! Float `avx2` needs F16C. Ternary `avx2` does not. `avxvnni` selects 256-bit
//! VNNI for both int8 and ternary; `dotprod` selects `sdot` instead of `smmla`
//! on arm64 i8mm CPUs for both int8 and ternary.

use super::fgemm::Level;
use super::int8::Path;
use super::kernels::Isa;
use std::sync::OnceLock;

pub(super) struct IsaChoice {
    pub(super) float_level: Level,
    pub(super) ternary: Isa,
    pub(super) int8: Path,
}

pub(super) fn isa_choice() -> &'static IsaChoice {
    static CHOICE: OnceLock<IsaChoice> = OnceLock::new();
    CHOICE.get_or_init(choose)
}

fn choose() -> IsaChoice {
    let forced = std::env::var("TRIT_ISA").unwrap_or_default();
    let ternary = ternary_of(&forced);
    IsaChoice {
        float_level: float_of(&forced),
        ternary,
        int8: int8_of(&forced, ternary),
    }
}

fn float_of(forced: &str) -> Level {
    #[cfg(target_arch = "x86_64")]
    {
        // F16C comes with every AVX2 CPU; the f16 decoder layers need it.
        let avx2 = is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c");
        match forced {
            "scalar" => return Level::Scalar,
            "avx2" if avx2 => return Level::Avx2,
            _ => {}
        }
        if avx2
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            return Level::Avx512;
        }
        if avx2 {
            return Level::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if forced != "scalar" {
        return Level::Neon;
    }
    let _ = forced;
    Level::Scalar
}

fn ternary_of(forced: &str) -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        let vnni512 = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni");
        let vnni256_vex = avx2 && is_x86_feature_detected!("avxvnni");
        let vnni256_evex = avx2 && vnni512 && is_x86_feature_detected!("avx512vl");
        match forced {
            "scalar" => return Isa::Scalar,
            "avx2" if avx2 => return Isa::Avx2,
            "avxvnni" if vnni256_vex => return Isa::AvxVnniVex,
            "avxvnni" if vnni256_evex => return Isa::AvxVnniEvex,
            "avxvnni" if avx2 => return Isa::Avx2,
            _ => {}
        }
        if vnni512 {
            return Isa::Avx512Vnni;
        }
        if vnni256_vex {
            return Isa::AvxVnniVex;
        }
        if avx2 {
            return Isa::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    return match forced {
        "scalar" => Isa::Scalar,
        "neon" => Isa::Neon,
        "dotprod" if std::arch::is_aarch64_feature_detected!("dotprod") => Isa::NeonDot,
        _ if std::arch::is_aarch64_feature_detected!("i8mm") => Isa::NeonMmla,
        _ if std::arch::is_aarch64_feature_detected!("dotprod") => Isa::NeonDot,
        _ => Isa::Neon,
    };
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = forced;
        Isa::Scalar
    }
}

fn int8_of(forced: &str, isa: Isa) -> Path {
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        let vnni512 = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni");
        let vnni256_vex = avx2 && is_x86_feature_detected!("avxvnni");
        let vnni256_evex = avx2 && vnni512 && is_x86_feature_detected!("avx512vl");
        match forced {
            "scalar" => Path::Scalar,
            // Forced `avx2` always selects plain AVX2 so AVX-VNNI hosts can test it.
            "avx2" if avx2 => Path::Avx2,
            "avxvnni" if vnni256_vex => Path::Vnni256Vex,
            "avxvnni" if vnni256_evex => Path::Vnni256Evex,
            "avxvnni" if avx2 => Path::Avx2,
            _ => match isa {
                Isa::Scalar => Path::Scalar,
                Isa::Avx512Vnni if vnni512 => Path::Vnni512,
                Isa::AvxVnniVex if vnni256_vex => Path::Vnni256Vex,
                Isa::AvxVnniEvex if vnni256_evex => Path::Vnni256Evex,
                Isa::Avx2 if avx2 => Path::Avx2,
                _ => Path::Scalar,
            },
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let _ = forced;
        match isa {
            Isa::Scalar => Path::Scalar,
            Isa::Neon => Path::Neon,
            Isa::NeonMmla => Path::NeonMmla,
            Isa::NeonDot => Path::NeonDot,
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (forced, isa);
        Path::Scalar
    }
}
