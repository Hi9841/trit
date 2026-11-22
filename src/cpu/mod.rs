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
pub mod v2;

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
        let set = match (v2::enabled(), v2::float_enabled()) {
            (true, true) => ", kernels v2",
            (true, false) => ", kernels v2-int",
            (false, true) => ", kernels v2-float",
            (false, false) => "",
        };
        format!(
            "ternary {}, int8 {}, float {}{set}",
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
pub fn enter(spread: Spread) -> Entered {
    let slot = spread.slot();
    team::start(slot, || team_config(spread));
    Entered {
        _team: team::enter(slot),
        _pool: buf::in_call(),
    }
}

/// A load or a call that runs: its thread is on the team, and the buffer
/// pool keeps the buffers of its passes.
pub struct Entered {
    _team: team::Entered,
    _pool: buf::InCall,
}

/// Worker threads of a team with `spread`. `TRIT_THREADS` overrides it.
pub fn threads(spread: Spread) -> usize {
    team_config(spread).threads
}

/// The most threads `TRIT_THREADS` gives: a typing error (`TRIT_THREADS=100000`)
/// must not start that many threads.
const MAX_THREADS: usize = 256;

/// The values `TRIT_ISA` takes on this kind of CPU.
#[cfg(target_arch = "x86_64")]
const ISA_NAMES: [&str; 3] = ["scalar", "avx2", "avxvnni"];
#[cfg(target_arch = "aarch64")]
const ISA_NAMES: [&str; 3] = ["scalar", "neon", "dotprod"];
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const ISA_NAMES: [&str; 1] = ["scalar"];

/// What is wrong with `TRIT_ISA`, when it has a value that selects no path
/// (a typing error: the engine then runs its fastest path). For `trit doctor`.
pub fn isa_problem() -> Option<String> {
    let forced = std::env::var("TRIT_ISA").ok().filter(|v| !v.is_empty())?;
    (!ISA_NAMES.contains(&forced.as_str())).then(|| {
        format!(
            "TRIT_ISA={forced} selects no kernel path and has no effect; valid values: {}",
            ISA_NAMES.join(", ")
        )
    })
}

/// One physical core: its Windows processor group (0 elsewhere), its bitmask
/// inside the active 64-bit group (0 when pinning is unavailable or the host
/// uses multi-word CPU lists), and its relative performance class/capacity
/// (higher is faster).
#[derive(Clone, Copy)]
struct CoreInfo {
    group: u16,
    mask: usize,
    class: u64,
}

/// Best single-L3 team when a host exceeds a single 64-bit mask (multi-group
/// Windows or >64-CPU Linux).
struct WideOneCache {
    threads: usize,
    pin_group: Option<(u16, usize)>,
    pin_cpus: Vec<usize>,
}

/// Physical cores and L3 cache domains `(mask, size_bytes)`. When CPUs span
/// multiple Windows processor groups or Linux CPU ids >= 64, `wide_one_cache`
/// carries the best single-L3 core count and its group/CPU affinity target so
/// `Spread::OneCache` preserves L3 locality.
struct Topology {
    cores: Vec<CoreInfo>,
    l3: Vec<(usize, u32)>,
    wide_one_cache: Option<WideOneCache>,
}

/// Team size, pinning mask, equal L3 domains for steering, heterogeneity flag,
/// and per-physical-core ideal CPUs. Respects `TRIT_THREADS` and the process
/// affinity mask on Windows, Linux, and Android, and enumerates non-Efficiency
/// performance tiers on macOS (including Apple M5 Pro/Max Super + Performance).
fn team_config(spread: Spread) -> team::Config {
    let layout = topology();
    #[cfg(target_os = "android")]
    let android_fast = fast_cores(std::path::Path::new("/sys/devices/system/cpu"));
    let core_cpus = layout
        .as_ref()
        .map(|t| physical_core_cpus(&t.cores))
        .unwrap_or_default();
    if let Some(threads) = std::env::var("TRIT_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|t| *t > 0)
    {
        let threads = threads.min(MAX_THREADS);
        // `TRIT_THREADS` disables fast-core pinning, so on Android check speed
        // differences across all CPUs rather than only the fast subset.
        #[cfg(target_os = "android")]
        let heterogeneous = threads > 1 && android_fast.is_some_and(|(_, _, _, all_h)| all_h);
        #[cfg(not(target_os = "android"))]
        let heterogeneous = threads > 1
            && layout
                .as_ref()
                .is_some_and(|t| cores_heterogeneous(&t.cores, None, 0));
        return team::Config {
            threads,
            domain: 0,
            domains: Vec::new(),
            heterogeneous,
            core_cpus,
            pin_cpus: Vec::new(),
            pin_group: None,
        };
    }
    #[cfg(target_os = "android")]
    if let Some((threads, mask, fast_hetero, _)) = android_fast {
        return team::Config {
            threads,
            domain: mask,
            domains: Vec::new(),
            heterogeneous: fast_hetero,
            core_cpus: Vec::new(),
            pin_cpus: Vec::new(),
            pin_group: None,
        };
    }
    let fallback = || {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        (threads, 0usize, Vec::new(), None)
    };
    // Not measured on a hybrid CPU yet, so `TRIT_FAST_CORES=1` turns it on:
    // with few fast cores, all the cores can be the faster team.
    #[cfg(windows)]
    if std::env::var("TRIT_FAST_CORES").is_ok_and(|value| value == "1") {
        if let Some((threads, domain)) = layout.as_ref().and_then(|t| fastest_cores(&t.cores)) {
            return team::Config {
                threads,
                domain,
                domains: Vec::new(),
                heterogeneous: false,
                core_cpus,
                pin_cpus: Vec::new(),
                pin_group: None,
            };
        }
    }
    let (threads, domain, pin_cpus, pin_group) = match (spread, layout.as_ref()) {
        (Spread::OneCache, Some(t)) => {
            if let Some(wide) = &t.wide_one_cache {
                (wide.threads, 0, wide.pin_cpus.clone(), wide.pin_group)
            } else {
                pick_domain(&t.cores, &t.l3)
                    .map(|(n, d)| (n, d, Vec::new(), None))
                    .unwrap_or_else(fallback)
            }
        }
        (Spread::AllCores, Some(t)) if !t.cores.is_empty() => (t.cores.len(), 0, Vec::new(), None),
        _ => fallback(),
    };
    let threads = threads.clamp(1, MAX_THREADS);
    // A CPU quota of a container is in no mask.
    #[cfg(target_os = "linux")]
    let threads = threads.min(std::thread::available_parallelism().map_or(threads, |n| n.get()));
    let domain = domain & allowed_mask();
    let heterogeneous = threads > 1
        && layout
            .as_ref()
            .is_some_and(|t| cores_heterogeneous(&t.cores, pin_group, domain));
    let domains = layout
        .as_ref()
        .map(|t| equal_domains(t, domain))
        .unwrap_or_default();
    team::Config {
        threads,
        domain,
        domains,
        heterogeneous,
        core_cpus,
        pin_cpus,
        pin_group,
    }
}

/// One logical CPU per physical core inside `allowed_mask()`, ordered fastest
/// class first so hybrid CPUs with asymmetric SMT (e.g. 8P+16E) never place
/// two workers on SMT siblings of the same core.
fn physical_core_cpus(cores: &[CoreInfo]) -> Vec<u32> {
    if cores.iter().any(|c| c.group != 0) {
        return Vec::new();
    }
    let allowed = allowed_mask();
    let mut ordered: Vec<(u64, u32)> = cores
        .iter()
        .filter_map(|core| {
            let masked = core.mask & allowed;
            (masked != 0).then_some((core.class, masked.trailing_zeros()))
        })
        .collect();
    ordered.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    ordered.into_iter().map(|(_, cpu)| cpu).collect()
}

/// True when the physical cores selected by `pin_group` or `domain` (0: all
/// cores) span performance classes differing by more than 10 %.
fn cores_heterogeneous(cores: &[CoreInfo], pin_group: Option<(u16, usize)>, domain: usize) -> bool {
    let mut min_class = u64::MAX;
    let mut max_class = 0u64;
    let mut seen = 0usize;
    for core in cores {
        if let Some((g, mask)) = pin_group {
            if core.group != g || (core.mask != 0 && core.mask & mask == 0) {
                continue;
            }
        } else if domain != 0 && core.mask != 0 && core.mask & domain == 0 {
            continue;
        }
        min_class = min_class.min(core.class);
        max_class = max_class.max(core.class);
        seen += 1;
    }
    if seen < 2 || min_class == max_class {
        return false;
    }
    // Small integer tiers (Windows EfficiencyClass 0/1/2, macOS perflevels 1/2/3)
    // always differ cleanly; continuous capacities/frequencies require >10 % gap.
    if max_class <= 16 {
        true
    } else {
        min_class.saturating_mul(10) < max_class.saturating_mul(9)
    }
}

/// The processors this process may run on, all bits set when unknown.
fn allowed_mask() -> usize {
    match team::process_mask() {
        0 => usize::MAX,
        mask => mask,
    }
}

/// A hybrid CPU: the cores of the highest efficiency class, as the team size
/// and their mask. A step ends with its last task, so a team with a task on
/// an E-core runs at the E-core's speed. `None` on a CPU with one class, or
/// with fewer than `MIN_FAST_CORES` fast cores: the slow cores then still
/// add more than they cost.
#[cfg(windows)]
fn fastest_cores(cores: &[CoreInfo]) -> Option<(usize, usize)> {
    const MIN_FAST_CORES: usize = 4;
    if cores.iter().any(|c| c.mask == 0) {
        return None;
    }
    let highest = cores.iter().map(|c| c.class).max()?;
    let lowest = cores.iter().map(|c| c.class).min()?;
    let fast: Vec<usize> = cores
        .iter()
        .filter(|c| c.class == highest)
        .map(|c| c.mask)
        .collect();
    if highest == lowest || fast.len() < MIN_FAST_CORES {
        return None;
    }
    let mask = fast.iter().fold(0, |all, core| all | core);
    Some((fast.len(), mask & allowed_mask()))
}

/// The L3 domains the team of `domain` can move to, `domain` included: those
/// with as many cores, inside the process mask. Parakeet runs at the same
/// speed on both CCDs of a 7950X3D. Moonshine tiny does not (17.8 ms with
/// 96 MB of L3, 24.4 ms with 32 MB), so the team goes back to `domain` when
/// the load that sent it away is gone (`Team::steer`).
fn equal_domains(layout: &Topology, domain: usize) -> Vec<usize> {
    if domain == 0 {
        return Vec::new();
    }
    let count = |mask: usize| layout.cores.iter().filter(|c| c.mask & mask != 0).count();
    let allowed = allowed_mask();
    let domains: Vec<usize> = layout
        .l3
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
fn pick_domain(cores: &[CoreInfo], l3: &[(usize, u32)]) -> Option<(usize, usize)> {
    let count = |domain: usize| cores.iter().filter(|c| c.mask & domain != 0).count();
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

#[cfg(windows)]
fn topology() -> Option<Topology> {
    use windows::Win32::System::SystemInformation::{
        CacheUnified, GetLogicalProcessorInformationEx, RelationCache, RelationProcessorCore,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };

    fn query(
        relation: windows::Win32::System::SystemInformation::LOGICAL_PROCESSOR_RELATIONSHIP,
    ) -> Option<(Vec<u64>, usize)> {
        let mut bytes = 0u32;
        // SAFETY: the first call only reports the required buffer size.
        let _ = unsafe { GetLogicalProcessorInformationEx(relation, None, &mut bytes) };
        if bytes == 0 {
            return None;
        }
        // One extra struct of room so a reference to the last short entry stays in bounds.
        let room = bytes as usize + std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>();
        let mut buffer = vec![0u64; room.div_ceil(8)];
        let first = buffer
            .as_mut_ptr()
            .cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>();
        // SAFETY: `buffer` holds at least `bytes` bytes as requested.
        unsafe { GetLogicalProcessorInformationEx(relation, Some(first), &mut bytes) }.ok()?;
        Some((buffer, bytes as usize))
    }

    let (core_buf, core_bytes) = query(RelationProcessorCore)?;
    let mut raw_cores = Vec::new();
    let mut at = 0usize;
    while at < core_bytes {
        // SAFETY: the OS fills whole entries aligned to 8 bytes with `Processor` active.
        let (size, processor) = unsafe {
            let entry = &*core_buf
                .as_ptr()
                .add(at / 8)
                .cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>();
            (entry.Size as usize, entry.Anonymous.Processor)
        };
        if size == 0 || !size.is_multiple_of(8) {
            break;
        }
        let group = processor.GroupMask[0];
        raw_cores.push((
            group.Group,
            group.Mask,
            u64::from(processor.EfficiencyClass),
        ));
        at += size;
    }
    if raw_cores.is_empty() {
        return None;
    }

    // Query the processor groups this process is permitted to run on.
    let allowed_groups = process_groups();
    if let Some(groups) = &allowed_groups {
        raw_cores.retain(|(group, _, _)| groups.contains(group));
        if raw_cores.is_empty() {
            return None;
        }
    }

    let first_group = raw_cores[0].0;
    let single_group = raw_cores.iter().all(|(g, _, _)| *g == first_group);

    let mut raw_l3: Vec<(u16, usize, u32)> = Vec::new();
    if let Some((cache_buf, cache_bytes)) = query(RelationCache) {
        let mut at = 0usize;
        while at < cache_bytes {
            // SAFETY: the OS fills whole entries aligned to 8 bytes with `Cache` active.
            let (size, cache) = unsafe {
                let entry = &*cache_buf
                    .as_ptr()
                    .add(at / 8)
                    .cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>();
                (entry.Size as usize, entry.Anonymous.Cache)
            };
            if size == 0 || !size.is_multiple_of(8) {
                break;
            }
            if cache.Level == 3 && cache.Type == CacheUnified {
                // SAFETY: `GroupMask` is valid for single-group cache entries.
                let group = unsafe { cache.Anonymous.GroupMask };
                if allowed_groups
                    .as_ref()
                    .is_none_or(|groups| groups.contains(&group.Group))
                {
                    raw_l3.push((group.Group, group.Mask, cache.CacheSize));
                }
            }
            at += size;
        }
    }

    if !single_group {
        // Multi-group process (>64 logical CPUs): count only cores in allowed
        // groups, and pin `Spread::OneCache` to the best single L3 domain's
        // `(group, mask)` via `SetThreadGroupAffinity`.
        let multi_l3 = raw_l3.len() > 1;
        let wide_one_cache = raw_l3
            .iter()
            .filter_map(|&(g, mask, size)| {
                let n = raw_cores
                    .iter()
                    .filter(|(cg, cm, _)| *cg == g && (*cm & mask) != 0)
                    .count();
                (n > 0).then_some((n, size, g, mask))
            })
            .max_by_key(|&(n, size, _, _)| (n, size))
            .map(|(n, _, g, mask)| WideOneCache {
                threads: n,
                pin_group: multi_l3.then_some((g, mask)),
                pin_cpus: Vec::new(),
            });
        let cores = raw_cores
            .into_iter()
            .map(|(group, mask, class)| CoreInfo { group, mask, class })
            .collect();
        return Some(Topology {
            cores,
            l3: Vec::new(),
            wide_one_cache,
        });
    }

    let allowed = allowed_mask();
    let cores: Vec<CoreInfo> = raw_cores
        .into_iter()
        .filter(|(_, mask, _)| mask & allowed != 0)
        .map(|(group, mask, class)| CoreInfo { group, mask, class })
        .collect();
    if cores.is_empty() {
        return None;
    }

    let mut l3 = Vec::new();
    for (group, mask, size) in raw_l3 {
        if group == first_group {
            let masked = mask & allowed;
            if masked != 0 && !l3.iter().any(|(m, _)| *m == masked) {
                l3.push((masked, size));
            }
        }
    }
    Some(Topology {
        cores,
        l3,
        wide_one_cache: None,
    })
}

/// Windows: processor groups this process may run on (`GetProcessGroupAffinity`).
#[cfg(windows)]
fn process_groups() -> Option<Vec<u16>> {
    use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessGroupAffinity};
    let mut count = 0u16;
    // SAFETY: first call reports required group count when buffer is null.
    let _ =
        unsafe { GetProcessGroupAffinity(GetCurrentProcess(), &mut count, std::ptr::null_mut()) };
    if count == 0 {
        return None;
    }
    let mut groups = vec![0u16; usize::from(count)];
    // SAFETY: `groups` holds `count` elements as requested.
    if unsafe { GetProcessGroupAffinity(GetCurrentProcess(), &mut count, groups.as_mut_ptr()) }
        .ok()
        .is_err()
    {
        return None;
    }
    groups.truncate(usize::from(count));
    (!groups.is_empty()).then_some(groups)
}

#[cfg(target_os = "linux")]
fn topology() -> Option<Topology> {
    let allowed = team::process_cpus();
    sysfs_topology(
        std::path::Path::new("/sys/devices/system/cpu"),
        allowed.as_deref(),
    )
}

/// Linux: enumerates all `cpuN` entries in `root`, intersects them with
/// `allowed` (when known, supporting >64 CPUs), deduplicates physical cores by
/// sibling list, reads per-core capacity or max frequency for heterogeneity
/// detection, and intersects L3 cache domains with `allowed`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn sysfs_topology(root: &std::path::Path, allowed: Option<&[usize]>) -> Option<Topology> {
    let read = |path: std::path::PathBuf| std::fs::read_to_string(path).ok();
    let is_allowed = |cpu: usize| allowed.is_none_or(|list| list.binary_search(&cpu).is_ok());

    let mut cpu_ids = Vec::new();
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let name = entry.file_name();
        let Some(rest) = name.to_str().and_then(|s| s.strip_prefix("cpu")) else {
            continue;
        };
        if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        if let Ok(cpu) = rest.parse::<usize>() {
            if is_allowed(cpu) {
                cpu_ids.push(cpu);
            }
        }
    }
    cpu_ids.sort_unstable();
    if cpu_ids.is_empty() {
        return None;
    }

    let fits_mask = cpu_ids.iter().all(|&cpu| cpu < usize::BITS as usize);
    let mut seen_cores: Vec<Vec<usize>> = Vec::new();
    let mut cores = Vec::new();
    let mut l3_lists: Vec<(Vec<usize>, u32)> = Vec::new();
    let capacity_source = cpu_ids
        .iter()
        .any(|&cpu| root.join(format!("cpu{cpu}/cpu_capacity")).exists());

    for &cpu in &cpu_ids {
        let dir = root.join(format!("cpu{cpu}"));
        let mut siblings = read(dir.join("topology/thread_siblings_list"))
            .and_then(|t| parse_cpu_list(&t))
            .unwrap_or_else(|| vec![cpu]);
        siblings.retain(|&c| is_allowed(c));
        siblings.sort_unstable();
        siblings.dedup();
        if !siblings.is_empty() && !seen_cores.contains(&siblings) {
            let class = if capacity_source {
                read(dir.join("cpu_capacity"))
            } else {
                read(dir.join("cpufreq/cpuinfo_max_freq"))
            }
            .and_then(|t| t.trim().parse::<u64>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1);
            let mask = if fits_mask {
                siblings.iter().fold(0usize, |m, &c| m | (1usize << c))
            } else {
                0
            };
            seen_cores.push(siblings);
            cores.push(CoreInfo {
                group: 0,
                mask,
                class,
            });
        }
        for index in 0..8 {
            let cache = dir.join(format!("cache/index{index}"));
            if read(cache.join("level")).as_deref().map(str::trim) != Some("3") {
                continue;
            }
            let Some(mut shared) =
                read(cache.join("shared_cpu_list")).and_then(|t| parse_cpu_list(&t))
            else {
                continue;
            };
            shared.retain(|&c| is_allowed(c));
            shared.sort_unstable();
            shared.dedup();
            let Some(size) = read(cache.join("size")).and_then(|t| cache_bytes(&t)) else {
                continue;
            };
            if !shared.is_empty() && !l3_lists.iter().any(|(list, _)| *list == shared) {
                l3_lists.push((shared, size));
            }
        }
    }
    if cores.is_empty() {
        return None;
    }
    if fits_mask {
        let l3 = l3_lists
            .into_iter()
            .map(|(list, size)| {
                let mask = list.into_iter().fold(0usize, |m, c| m | (1usize << c));
                (mask, size)
            })
            .collect();
        return Some(Topology {
            cores,
            l3,
            wide_one_cache: None,
        });
    }
    let multi_l3 = l3_lists.len() > 1;
    let wide_one_cache = l3_lists
        .into_iter()
        .filter_map(|(list, size)| {
            let n = seen_cores
                .iter()
                .filter(|sibs| sibs.iter().any(|c| list.binary_search(c).is_ok()))
                .count();
            (n > 0).then_some((n, size, list))
        })
        .max_by_key(|(n, size, _)| (*n, *size))
        .map(|(n, _, list)| WideOneCache {
            threads: n,
            pin_group: None,
            pin_cpus: if multi_l3 { list } else { Vec::new() },
        });
    Some(Topology {
        cores,
        l3: Vec::new(),
        wide_one_cache,
    })
}

/// Parses a Linux sysfs CPU list such as `"0-3,8-11"` into CPU indices.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_cpu_list(list: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in list.trim().split(',').filter(|p| !p.is_empty()) {
        let (first, last) = match part.split_once('-') {
            Some((a, b)) => (a.parse::<usize>().ok()?, b.parse::<usize>().ok()?),
            None => (part.parse().ok()?, part.parse().ok()?),
        };
        if first > last {
            return None;
        }
        cpus.extend(first..=last);
    }
    (!cpus.is_empty()).then_some(cpus)
}

/// "32768K" or "32M" as bytes.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
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

/// macOS: enumerates `hw.perflevel0..hw.nperflevels`. Level 0 is always
/// included; higher levels are included unless named `"Efficiency"` (case-
/// insensitive). On M1 to M4 and base M5 (`Performance`/`Super` + `Efficiency`)
/// this selects only the fast tier; on M5 Pro and M5 Max (`Super` + `Performance`,
/// no Efficiency cores) it uses both tiers and marks the team heterogeneous so
/// fast Super cores steal work dynamically from Performance cores. Falls back to
/// `hw.physicalcpu` on Intel Macs.
#[cfg(target_os = "macos")]
fn topology() -> Option<Topology> {
    use std::ffi::CString;
    extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            old: *mut std::ffi::c_void,
            old_len: *mut usize,
            new: *mut std::ffi::c_void,
            new_len: usize,
        ) -> i32;
    }

    fn sysctl_i32(name: &std::ffi::CStr) -> Option<i32> {
        let mut value: i32 = 0;
        let mut len = std::mem::size_of::<i32>();
        // SAFETY: `name` is NUL-terminated and `value` holds `len` bytes.
        let status = unsafe {
            sysctlbyname(
                name.as_ptr(),
                (&mut value as *mut i32).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (status == 0 && len == std::mem::size_of::<i32>()).then_some(value)
    }

    fn sysctl_str(name: &std::ffi::CStr) -> Option<String> {
        let mut buf = [0u8; 64];
        let mut len = buf.len();
        // SAFETY: `name` is NUL-terminated and `buf` holds `len` bytes.
        let status = unsafe {
            sysctlbyname(
                name.as_ptr(),
                buf.as_mut_ptr().cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if status != 0 || len == 0 {
            return None;
        }
        let end = buf[..len.min(buf.len())]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(len.min(buf.len()));
        std::str::from_utf8(&buf[..end]).ok().map(str::to_owned)
    }

    let mut cores = Vec::new();
    if let Some(levels) = sysctl_i32(c"hw.nperflevels").filter(|&n| n > 0) {
        for level in 0..levels {
            let count_key = CString::new(format!("hw.perflevel{level}.physicalcpu")).ok()?;
            let Some(count) = sysctl_i32(&count_key).and_then(|c| usize::try_from(c).ok()) else {
                continue;
            };
            if count == 0 {
                continue;
            }
            let include = if level == 0 {
                true
            } else {
                let name_key = CString::new(format!("hw.perflevel{level}.name")).ok()?;
                sysctl_str(&name_key).is_some_and(|name| {
                    !name.is_empty() && !name.to_ascii_lowercase().starts_with("efficiency")
                })
            };
            if include {
                let class = (levels - level) as u64;
                for _ in 0..count {
                    cores.push(CoreInfo {
                        group: 0,
                        mask: 0,
                        class,
                    });
                }
            }
        }
    }
    if cores.is_empty() {
        let count = sysctl_i32(c"hw.physicalcpu")
            .and_then(|c| usize::try_from(c).ok())
            .filter(|&c| c > 0)?;
        for _ in 0..count {
            cores.push(CoreInfo {
                group: 0,
                mask: 0,
                class: 1,
            });
        }
    }
    Some(Topology {
        cores,
        l3: Vec::new(),
        wide_one_cache: None,
    })
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
fn topology() -> Option<Topology> {
    None
}

/// Android: the fast cores, those with at least half the capacity of the
/// fastest (`cpuN/cpu_capacity`, else `cpufreq/cpuinfo_max_freq`), along with
/// heterogeneity flags for the selected fast subset (`fast_hetero`, e.g.
/// Snapdragon Prime + Performance) and for all cores (`all_hetero`, used when
/// `TRIT_THREADS` disables fast-core pinning).
#[cfg(target_os = "android")]
fn fast_cores(root: &std::path::Path) -> Option<(usize, usize, bool, bool)> {
    let read = |cpu: usize, file: &str| -> Option<u64> {
        let path = root.join(format!("cpu{cpu}/{file}"));
        std::fs::read_to_string(path).ok()?.trim().parse().ok()
    };
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
    let slowest_all = speeds
        .iter()
        .map(|(_, speed)| *speed)
        .min()
        .unwrap_or(fastest);
    let all_hetero = speeds.len() > 1 && slowest_all.saturating_mul(10) < fastest.saturating_mul(9);
    let selected: Vec<(usize, u64)> = speeds
        .into_iter()
        .filter(|(_, speed)| 2 * speed >= fastest)
        .collect();
    if selected.is_empty() {
        return None;
    }
    let mask = selected
        .iter()
        .fold(0usize, |mask, (cpu, _)| mask | (1usize << cpu));
    let min_speed = selected.iter().map(|(_, s)| *s).min().unwrap_or(fastest);
    let fast_hetero =
        selected.len() > 1 && min_speed.saturating_mul(10) < fastest.saturating_mul(9);
    Some((selected.len(), mask, fast_hetero, all_hetero))
}
