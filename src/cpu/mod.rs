//! Kernels and the worker team that every model family shares.

#[macro_use]
mod simd;

pub mod activation;
pub mod buf;
pub mod fgemm;
pub mod gemm;
pub mod half;
pub mod int8;
pub mod kernels;
pub mod team;
pub mod tensors;

/// The kernels this CPU runs, for example `ternary avx512-vnni, int8
/// vnni512, float avx512`: the ternary and int8 products (`avx512-vnni`,
/// `avx2`, `scalar`; int8 also `vnni256` on AVX-VNNI CPUs) and the f32/f16
/// products (`avx512`, `avx2`, `scalar`). `TRIT_ISA` forces a slower path.
pub fn cpu_path() -> &'static str {
    static PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PATH.get_or_init(|| {
        format!(
            "ternary {}, int8 {}, float {}",
            kernels::Isa::detect().name(),
            int8::path_name(),
            fgemm::level().name()
        )
    })
}

/// How far the worker team spreads over the CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spread {
    /// The physical cores of one L3 domain, pinned. Best for models whose
    /// products have few rows (Parakeet: about 90 for a 7 s take), where
    /// every exchange across L3 domains costs more than the extra cores give.
    OneCache,
    /// Every physical core. Best for models with tall products (Whisper's
    /// encoder always has 1500 rows).
    AllCores,
}

/// Starts the worker team of `spread` once (later calls do nothing) and
/// runs the calling thread's steps on it until the guard drops. Each spread
/// has its own team, so models of both kinds in one process each run on
/// theirs. A second thread on a core shares its vector units and only adds
/// contention, so a team has one thread per core.
pub fn enter(spread: Spread) -> team::Entered {
    let slot = spread as usize;
    team::start(slot, || engine_threads(spread));
    team::enter(slot)
}

/// Worker threads of a team with `spread`. `TRIT_THREADS` overrides it.
pub fn threads(spread: Spread) -> usize {
    engine_threads(spread).0
}

/// Team size and the affinity mask of its L3 domain (0: no pinning).
/// `TRIT_THREADS` overrides the physical core count and turns pinning off.
fn engine_threads(spread: Spread) -> (usize, usize) {
    if let Some(threads) = std::env::var("TRIT_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|t| *t > 0)
    {
        return (threads, 0);
    }
    let fallback = || {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        (threads, 0)
    };
    match (spread, topology()) {
        (Spread::OneCache, Some((cores, l3))) => pick_domain(&cores, &l3).unwrap_or_else(fallback),
        (Spread::AllCores, Some((cores, _))) if !cores.is_empty() => (cores.len(), 0),
        _ => fallback(),
    }
}

/// Physical cores that share one L3 cache: 8 on a two-CCD Ryzen, all cores
/// on a CPU with one shared L3. Threads on another L3 domain pay for every
/// exchange of the small per-layer matrices, so with several domains the
/// team stays on one: the one with the most cores, then the largest L3
/// (the V-cache CCD of an X3D part).
/// `cores` holds one processor mask per physical core, `l3` a mask and a size per L3 cache.
fn pick_domain(cores: &[usize], l3: &[(usize, u32)]) -> Option<(usize, usize)> {
    let count = |domain: usize| cores.iter().filter(|core| *core & domain != 0).count();
    let best = l3
        .iter()
        .max_by_key(|(domain, size)| (count(*domain), *size))
        .filter(|(domain, _)| count(*domain) > 0);
    match best {
        Some((domain, _)) if l3.len() > 1 => Some((count(*domain), *domain)),
        Some((domain, _)) => Some((count(*domain), 0)),
        None => Some((cores.len(), 0)).filter(|(n, _)| *n > 0),
    }
}

/// One processor mask per physical core, and a mask and a size per L3 cache.
type Topology = (Vec<usize>, Vec<(usize, u32)>);

#[cfg(windows)]
fn topology() -> Option<Topology> {
    use windows::Win32::System::SystemInformation::{
        CacheUnified, GetLogicalProcessorInformation, RelationCache, RelationProcessorCore,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION,
    };
    let mut bytes = 0u32;
    // SAFETY: the first call only reports the buffer size.
    let _ = unsafe { GetLogicalProcessorInformation(None, &mut bytes) };
    let count = bytes as usize / std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>();
    if count == 0 {
        return None;
    }
    let mut info = vec![SYSTEM_LOGICAL_PROCESSOR_INFORMATION::default(); count];
    // SAFETY: the buffer holds `bytes` bytes, as the first call asked.
    unsafe { GetLogicalProcessorInformation(Some(info.as_mut_ptr()), &mut bytes) }.ok()?;
    let cores: Vec<usize> = info
        .iter()
        .filter(|entry| entry.Relationship == RelationProcessorCore)
        .map(|entry| entry.ProcessorMask)
        .collect();
    // SAFETY: `Cache` is the active union member for RelationCache entries.
    let l3: Vec<(usize, u32)> = info
        .iter()
        .filter(|entry| entry.Relationship == RelationCache)
        .filter_map(|entry| unsafe {
            let cache = entry.Anonymous.Cache;
            (cache.Level == 3 && cache.Type == CacheUnified)
                .then_some((entry.ProcessorMask, cache.Size))
        })
        .collect();
    Some((cores, l3))
}

#[cfg(target_os = "linux")]
fn topology() -> Option<Topology> {
    sysfs_topology(std::path::Path::new("/sys/devices/system/cpu"))
}

/// Linux: one mask per physical core from `cpuN/topology/thread_siblings_list`,
/// and a mask and a size per L3 cache from `cpuN/cache/index*/` (level 3).
/// Masks hold the first 64 processors, as on Windows.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn sysfs_topology(root: &std::path::Path) -> Option<Topology> {
    let read = |path: std::path::PathBuf| std::fs::read_to_string(path).ok();
    let mut cores = Vec::new();
    let mut l3 = Vec::new();
    for cpu in 0..usize::BITS as usize {
        let dir = root.join(format!("cpu{cpu}"));
        if !dir.exists() {
            continue;
        }
        if let Some(mask) =
            read(dir.join("topology/thread_siblings_list")).and_then(|t| cpu_mask(&t))
        {
            if !cores.contains(&mask) {
                cores.push(mask);
            }
        }
        for index in 0..8 {
            let cache = dir.join(format!("cache/index{index}"));
            if read(cache.join("level")).as_deref().map(str::trim) != Some("3") {
                continue;
            }
            let mask = read(cache.join("shared_cpu_list")).and_then(|t| cpu_mask(&t));
            let size = read(cache.join("size")).and_then(|t| cache_bytes(&t));
            if let (Some(mask), Some(size)) = (mask, size) {
                if !l3.iter().any(|(m, _)| *m == mask) {
                    l3.push((mask, size));
                }
            }
        }
    }
    (!cores.is_empty()).then_some((cores, l3))
}

/// "0-3,8-11" as a bit mask; processors past 63 are left out.
fn cpu_mask(list: &str) -> Option<usize> {
    let mut mask = 0usize;
    for part in list.trim().split(',').filter(|p| !p.is_empty()) {
        let (first, last) = match part.split_once('-') {
            Some((a, b)) => (a.parse::<usize>().ok()?, b.parse::<usize>().ok()?),
            None => (part.parse().ok()?, part.parse().ok()?),
        };
        for cpu in first..=last.min(usize::BITS as usize - 1) {
            mask |= 1 << cpu;
        }
    }
    (mask != 0).then_some(mask)
}

/// "32768K" or "32M" as bytes.
fn cache_bytes(text: &str) -> Option<u32> {
    let text = text.trim();
    let (digits, unit) = text.split_at(
        text.find(|c: char| !c.is_ascii_digit())
            .unwrap_or(text.len()),
    );
    let value: u32 = digits.parse().ok()?;
    value.checked_mul(match unit {
        "K" => 1 << 10,
        "M" => 1 << 20,
        _ => 1,
    })
}

/// macOS: the performance cores only (`hw.perflevel0.physicalcpu`); a team
/// that waits on an efficiency core at every step runs at its speed. One
/// domain, no pinning (macOS has no affinity masks).
#[cfg(target_os = "macos")]
fn topology() -> Option<Topology> {
    extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            old: *mut std::ffi::c_void,
            old_len: *mut usize,
            new: *mut std::ffi::c_void,
            new_len: usize,
        ) -> i32;
    }
    let mut count: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    // SAFETY: the name is NUL-terminated and `count` holds `len` bytes.
    let status = unsafe {
        sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr(),
            (&mut count as *mut i32).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    let count = usize::try_from(count)
        .ok()
        .filter(|c| status == 0 && *c > 0)?;
    let cores: Vec<usize> = (0..count.min(usize::BITS as usize))
        .map(|i| 1 << i)
        .collect();
    let all = cores.iter().fold(0, |m, c| m | c);
    Some((cores, vec![(all, 0)]))
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
fn topology() -> Option<Topology> {
    None
}
