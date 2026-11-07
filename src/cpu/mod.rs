//! Kernels and the worker team that every model family shares.

#[macro_use]
mod simd;
mod isa;

pub mod activation;
pub mod buf;
pub mod fermion;
pub mod fgemm;
pub mod gemm;
pub mod ggml_quants;
pub mod half;
pub mod int8;
pub mod kernels;
pub mod store;
pub mod team;
pub mod tensors;
mod torch_file;

/// The kernels this CPU runs, for example `ternary avx512-vnni, int8
/// vnni512, float avx512`: the ternary and int8 products (`avx512-vnni`,
/// `avx2`, `scalar`; int8 also `vnni256` on AVX-VNNI CPUs; on arm64
/// `neon-dotprod` or `neon`, and int8 `neon-i8mm` on Apple M2 and later)
/// and the f32/f16 products (`avx512`, `avx2`, `neon`, `scalar`).
/// `TRIT_ISA` forces a slower path.
pub fn cpu_path() -> &'static str {
    static PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PATH.get_or_init(|| {
        let choice = isa::isa_choice();
        format!(
            "ternary {}, int8 {}, float {}",
            choice.ternary.name(),
            int8::path_name(),
            choice.float_level.name()
        )
    })
}

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

impl Spread {
    /// Teams a process may start.
    pub const TEAMS: usize = 2;

    pub const fn slot(self) -> usize {
        match self {
            Self::OneCache => 0,
            Self::AllCores => 1,
        }
    }
}

/// Starts the worker team of `spread` once (later calls do nothing) and
/// runs the calling thread's steps on it until the guard drops. Each spread
/// has its own team, so models of both kinds in one process each run on
/// theirs. A second thread on a core shares its vector units and only adds
/// contention, so a team has one thread per core.
pub fn enter(spread: Spread) -> team::Entered {
    let slot = spread.slot();
    team::start(slot, || {
        let (threads, domain) = engine_threads(spread);
        (threads, domain, equal_domains(domain))
    });
    team::enter(slot)
}

/// Worker threads of a team with `spread`. `TRIT_THREADS` overrides it.
pub fn threads(spread: Spread) -> usize {
    engine_threads(spread).0
}

/// Team size and the affinity mask of its cores (0: no pinning). The team
/// has no more threads than the process has CPUs: more threads than CPUs
/// spin while the task that a step waits for has no CPU.
/// `TRIT_THREADS` overrides the physical core count and turns pinning off.
fn engine_threads(spread: Spread) -> (usize, usize) {
    if let Some(threads) = std::env::var("TRIT_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|t| *t > 0)
    {
        return (threads, 0);
    }
    #[cfg(target_os = "android")]
    if let Some(mask) = fast_cores(std::path::Path::new("/sys/devices/system/cpu")) {
        return (mask.count_ones() as usize, mask);
    }
    let fallback = || {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        (threads, 0)
    };
    let layout = topology().map(|(cores, l3)| (inside_process(cores), l3));
    // Not measured on a hybrid CPU yet, so `TRIT_FAST_CORES=1` turns it on:
    // with few fast cores, all the cores can be the faster team.
    #[cfg(windows)]
    if std::env::var("TRIT_FAST_CORES").is_ok_and(|value| value == "1") {
        if let Some(team) = layout.as_ref().and_then(|(cores, _)| fastest_cores(cores)) {
            return team;
        }
    }
    let (threads, domain) = match (spread, layout) {
        (Spread::OneCache, Some((cores, l3))) => pick_domain(&cores, &l3).unwrap_or_else(fallback),
        (Spread::AllCores, Some((cores, _))) if !cores.is_empty() => (cores.len(), 0),
        _ => fallback(),
    };
    // A CPU quota of a container is in no mask.
    #[cfg(target_os = "linux")]
    let threads = threads.min(std::thread::available_parallelism().map_or(threads, |n| n.get()));
    (threads, domain & allowed_mask())
}

/// The processors this process may run on, all bits set when unknown.
fn allowed_mask() -> usize {
    match team::process_mask() {
        0 => usize::MAX,
        mask => mask,
    }
}

/// The physical cores with a processor that this process may run on.
fn inside_process(cores: Vec<usize>) -> Vec<usize> {
    let allowed = allowed_mask();
    cores
        .into_iter()
        .filter(|core| core & allowed != 0)
        .collect()
}

/// A hybrid CPU: the cores of the highest efficiency class, as the team size
/// and their mask. A step ends with its last task, so a team with a task on
/// an E-core runs at the E-core's speed. `None` on a CPU with one class, or
/// with fewer than `MIN_FAST_CORES` fast cores: the slow cores then still
/// add more than they cost.
#[cfg(windows)]
fn fastest_cores(cores: &[usize]) -> Option<(usize, usize)> {
    const MIN_FAST_CORES: usize = 4;
    let classes: Vec<(usize, u8)> = core_classes()?
        .into_iter()
        .filter(|(mask, _)| cores.contains(mask))
        .collect();
    let highest = classes.iter().map(|(_, class)| *class).max()?;
    let lowest = classes.iter().map(|(_, class)| *class).min()?;
    let fast: Vec<usize> = classes
        .iter()
        .filter(|(_, class)| *class == highest)
        .map(|(mask, _)| *mask)
        .collect();
    if highest == lowest || fast.len() < MIN_FAST_CORES {
        return None;
    }
    let mask = fast.iter().fold(0, |all, core| all | core);
    Some((fast.len(), mask & allowed_mask()))
}

/// The processor mask and the efficiency class of each physical core. `None`
/// when the system does not say, or with several processor groups: masks of
/// different groups do not compare, and no hybrid CPU has more than one.
#[cfg(windows)]
fn core_classes() -> Option<Vec<(usize, u8)>> {
    use windows::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationProcessorCore,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    let mut bytes = 0u32;
    // SAFETY: the first call only reports the buffer size.
    let _ = unsafe { GetLogicalProcessorInformationEx(RelationProcessorCore, None, &mut bytes) };
    // u64 words: the entries hold pointer-sized masks. One more entry of
    // room: a core entry is shorter than the type, and a reference to the
    // last one must stay inside the buffer.
    let room = bytes as usize + std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>();
    let mut buffer = vec![0u64; room.div_ceil(8)];
    let first = buffer
        .as_mut_ptr()
        .cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>();
    // SAFETY: the buffer holds `bytes` bytes, as the first call asked.
    unsafe { GetLogicalProcessorInformationEx(RelationProcessorCore, Some(first), &mut bytes) }
        .ok()?;
    let mut classes = Vec::new();
    let mut at = 0usize;
    while at < bytes as usize {
        // SAFETY: the system fills the buffer with whole entries, each `Size`
        // bytes long, and sizes are multiples of 8. `Processor` is the union
        // member of the relation that the call asked for.
        let (size, processor) = unsafe {
            let entry = &*buffer
                .as_ptr()
                .add(at / 8)
                .cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>();
            (entry.Size as usize, entry.Anonymous.Processor)
        };
        let group = processor.GroupMask[0];
        if size == 0 || !size.is_multiple_of(8) || processor.GroupCount != 1 || group.Group != 0 {
            return None;
        }
        classes.push((group.Mask, processor.EfficiencyClass));
        at += size;
    }
    Some(classes)
}

/// The L3 domains the team of `domain` can move to, `domain` included: those
/// with as many cores, inside the process mask. Parakeet runs at the same
/// speed on both CCDs of a 7950X3D, so a larger L3 is no reason to stay.
fn equal_domains(domain: usize) -> Vec<usize> {
    let Some((cores, l3)) = topology().filter(|_| domain != 0) else {
        return Vec::new();
    };
    let count = |mask: usize| cores.iter().filter(|core| *core & mask != 0).count();
    let allowed = team::process_mask();
    let domains: Vec<usize> = l3
        .iter()
        .map(|(mask, _)| *mask)
        .filter(|mask| count(*mask) == count(domain) && mask & !allowed == 0)
        .collect();
    if domains.contains(&domain) {
        domains
    } else {
        Vec::new()
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

/// Android: the mask of the fast cores, those with at least half the
/// capacity of the fastest (`cpuN/cpu_capacity`, else
/// `cpufreq/cpuinfo_max_freq`). Phones mix big and little cores; a step
/// ends with its last task, so a team with a task on a little core runs
/// at the little core's speed, and spinning little cores cost battery.
/// `None` when sysfs tells nothing.
#[cfg(target_os = "android")]
fn fast_cores(root: &std::path::Path) -> Option<usize> {
    let read = |cpu: usize, file: &str| -> Option<u64> {
        let path = root.join(format!("cpu{cpu}/{file}"));
        std::fs::read_to_string(path).ok()?.trim().parse().ok()
    };
    // One source for every core, so the speeds compare.
    let speeds_from = |file: &str| -> Vec<(usize, u64)> {
        (0..usize::BITS as usize)
            .filter_map(|cpu| read(cpu, file).map(|speed| (cpu, speed)))
            .collect()
    };
    let mut speeds = speeds_from("cpu_capacity");
    if speeds.is_empty() {
        speeds = speeds_from("cpufreq/cpuinfo_max_freq");
    }
    let fastest = speeds.iter().map(|(_, speed)| *speed).max()?;
    let mask = speeds
        .iter()
        .filter(|(_, speed)| 2 * speed >= fastest)
        .fold(0usize, |mask, (cpu, _)| mask | 1 << cpu);
    (mask != 0).then_some(mask)
}
