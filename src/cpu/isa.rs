//! One read of `TRIT_ISA` for the float level, the ternary ISA, and the int8 path.
//! Float `avx2` needs F16C. Ternary `avx2` does not. `avxvnni` and `dotprod`
//! change only the int8 path.

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
        let vnni = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni");
        let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        match forced {
            "scalar" => return Isa::Scalar,
            "avx2" if avx2 => return Isa::Avx2,
            _ => {}
        }
        if vnni {
            return Isa::Avx512Vnni;
        }
        if avx2 {
            return Isa::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    return match forced {
        "scalar" => Isa::Scalar,
        "neon" => Isa::Neon,
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
    match isa {
        Isa::Scalar => Path::Scalar,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 if is_x86_feature_detected!("avxvnni") => Path::Vnni256,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => Path::Avx2,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512Vnni if forced == "avxvnni" => Path::Vnni256,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512Vnni => Path::Vnni512,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => Path::Neon,
        #[cfg(target_arch = "aarch64")]
        Isa::NeonDot if std::arch::is_aarch64_feature_detected!("i8mm") && forced != "dotprod" => {
            Path::NeonMmla
        }
        #[cfg(target_arch = "aarch64")]
        Isa::NeonDot => Path::NeonDot,
    }
}
