//! Worker threads for the speech engine.
//!
//! A take runs about 1,200 short parallel steps. A thread pool that puts
//! idle threads to sleep pays about 20 µs to wake them for each step after a
//! short serial gap. These workers spin on a counter while steps keep coming,
//! so a step starts in about a microsecond, and they park after
//! `SPIN_FOR` without work, so an idle app uses no CPU.
//!
//! A step ends when its tasks are done, not when every worker has looked at
//! it. With more ready threads than CPUs (a build on every core), a worker
//! can wait a long time for a CPU; the threads that run take its tasks, and
//! it finds nothing to do when it wakes. Waiting for every worker cost 3 to
//! 5 s of a 6 to 9 s take.
//!
//! Priority does not help against a load on the second hardware thread of a
//! team core: the load takes half of the core and the scheduler has nothing
//! to preempt. A full load on the team's L3 domain cost 1.9 times the quiet
//! time, and on a free domain the same take cost 1.3 times. So a team with a
//! choice of domains moves to the quieter one between runs (`steer`).

use parking_lot::{Condvar, Mutex};
use std::any::Any;
use std::cell::Cell;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Longer than the gaps inside a take, short enough to be invisible as idle CPU.
const SPIN_FOR: Duration = Duration::from_millis(3);

/// A shorter window shows the team's own last run, not the load.
const MIN_LOAD_WINDOW: Duration = Duration::from_millis(200);
/// A longer window shows an idle gap, not the load now.
const MAX_LOAD_WINDOW: Duration = Duration::from_secs(30);
/// The team moves when its domain is this much busier than another, as a
/// fraction of all the domain's processors. One hardware thread on each core
/// is 0.5, and a move costs the warm caches, so noise must not start one.
const MOVE_AT_BUSY_GAP: f64 = 0.25;
/// A team thread takes much less than this to start a task, even under load.
/// A longer wait means that the workers do not exist: a forked child keeps
/// only the thread that called `fork`.
const ALL_THREADS_WAIT: Duration = Duration::from_secs(1);

struct Job {
    run: *const (dyn Fn(usize) + Sync),
}

/// Most tasks in one job: the claim word holds 16 bits of task numbers.
const MAX_TASKS: usize = u16::MAX as usize;

/// The claim word: the job number, its task count, and its next task. A
/// thread claims a task with one compare-and-swap on this word, so it can
/// only claim a task of the job that is current at that moment.
fn claim_word(job: u32, tasks: usize, next: usize) -> u64 {
    (u64::from(job) << 32) | ((tasks as u64) << 16) | next as u64
}

struct Shared {
    /// Changes with each job; parked workers wait for it.
    generation: AtomicUsize,
    job: AtomicPtr<Job>,
    /// See `claim_word`.
    claim: AtomicU64,
    done: AtomicUsize,
    panicked: AtomicBool,
    parked: AtomicUsize,
    /// The generation when the last call ended (`Entered` stores it). A
    /// worker with no job after it parks at once: the 3 ms of spin after the
    /// last step of a call were 21 ms of CPU for each Parakeet call.
    rest: AtomicUsize,
    lock: Mutex<()>,
    wake: Condvar,
}

pub struct Team {
    shared: Arc<Shared>,
    workers: usize,
    /// The worker threads, kept to move them to another domain.
    threads: Vec<JoinHandle<()>>,
    /// Affinity mask of the team's L3 domain, 0 when the team is not pinned.
    domain: AtomicUsize,
    /// Domains of the same size that the team can move to, its own included.
    domains: Vec<usize>,
    /// When and what the last look at the processors saw.
    load: Mutex<Option<(Instant, Vec<CpuTimes>)>>,
    /// One caller at a time; a nested call from inside a task runs inline.
    submit: Mutex<()>,
}

thread_local! {
    static INSIDE: Cell<bool> = const { Cell::new(false) };
    /// The team this thread's steps run on (see `enter`).
    static CURRENT: Cell<usize> = const { Cell::new(0) };
}

impl Team {
    /// `threads` includes the calling thread, which works on every job too.
    /// `slot` names the threads: a process can have both teams.
    fn new(slot: usize, threads: usize, domain: usize, domains: Vec<usize>) -> Self {
        let shared = Arc::new(Shared {
            generation: AtomicUsize::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            claim: AtomicU64::new(0),
            done: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            parked: AtomicUsize::new(0),
            // No generation has this value, so no worker rests before a call ends.
            rest: AtomicUsize::new(usize::MAX),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        let workers = threads.saturating_sub(1);
        let mut handles = Vec::with_capacity(workers);
        for index in 0..workers {
            let for_worker = Arc::clone(&shared);
            let spawned = std::thread::Builder::new()
                .name(format!("trit-speech-{slot}-{index}"))
                .spawn(move || {
                    set_affinity(domain);
                    set_priority(ABOVE_NORMAL);
                    keep_full_speed();
                    set_ideal(ideal_cpu(domain, index + 1, threads));
                    work(&for_worker);
                });
            match spawned {
                Ok(handle) => handles.push(handle),
                // Fewer threads only make the engine slower.
                Err(_) => break,
            }
        }
        Self {
            shared,
            workers: handles.len(),
            threads: handles,
            domain: AtomicUsize::new(domain),
            domains,
            load: Mutex::new(None),
            submit: Mutex::new(()),
        }
    }

    /// Moves the team to a quieter domain when the load of the last window
    /// shows one. Two runs of a take give the window, so the work before
    /// the end of the take does the looking and the moving.
    fn steer(&self) {
        if self.domains.len() < 2 {
            return;
        }
        let Some(mut load) = self.load.try_lock() else {
            return;
        };
        let Some(times) = cpu_times() else {
            return;
        };
        let now = Instant::now();
        let window = load.as_ref().map(|(at, _)| now.duration_since(*at));
        if window.is_some_and(|window| window < MIN_LOAD_WINDOW) {
            return;
        }
        let before = load.replace((now, times));
        let (Some((_, before)), Some((_, times))) = (before, load.as_ref()) else {
            return;
        };
        if window.is_some_and(|window| window > MAX_LOAD_WINDOW) {
            return;
        }
        let current = self.domain.load(Ordering::Relaxed);
        let busy = |domain: usize| busy_fraction(&before, times, domain);
        let quietest = self
            .domains
            .iter()
            .copied()
            .min_by(|a, b| busy(*a).total_cmp(&busy(*b)));
        if let Some(domain) = quietest {
            if busy(current) - busy(domain) >= MOVE_AT_BUSY_GAP {
                self.move_to(domain);
            }
        }
    }

    fn move_to(&self, domain: usize) {
        self.domain.store(domain, Ordering::Relaxed);
        let threads = self.workers + 1;
        for (index, handle) in self.threads.iter().enumerate() {
            pin_thread(handle, domain, ideal_cpu(domain, index + 1, threads));
        }
    }

    fn for_each(&self, tasks: usize, run: &(dyn Fn(usize) + Sync)) {
        if tasks <= 1 || self.workers == 0 || INSIDE.get() {
            (0..tasks).for_each(run);
            return;
        }
        if tasks > MAX_TASKS {
            for start in (0..tasks).step_by(MAX_TASKS) {
                let count = MAX_TASKS.min(tasks - start);
                self.for_each(count, &|task| run(start + task));
            }
            return;
        }
        let _one_caller = self.submit.lock();
        INSIDE.set(true);
        // SAFETY: a thread calls `run` only for a task it claimed while
        // `claim` named this job, and this frame returns only when every
        // claimed task has returned (done == tasks below), so erasing the
        // closure's lifetime cannot outlive the borrow.
        let run: &'static (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(run) };
        let job = Job { run };
        let s = &*self.shared;
        // Only the caller, under `submit`, changes the generation.
        let generation = s.generation.load(Ordering::Relaxed).wrapping_add(1);
        s.job
            .store(&job as *const Job as *mut Job, Ordering::Relaxed);
        s.done.store(0, Ordering::Relaxed);
        // Release: a thread that claims a task sees the job and `done` above.
        s.claim
            .store(claim_word(generation as u32, tasks, 0), Ordering::Release);
        s.generation.store(generation, Ordering::SeqCst);
        if s.parked.load(Ordering::SeqCst) > 0 {
            let _lock = s.lock.lock();
            s.wake.notify_all();
        }
        let caller = run_tasks(s);
        while s.done.load(Ordering::Acquire) < tasks {
            std::hint::spin_loop();
        }
        INSIDE.set(false);
        let panicked = s.panicked.swap(false, Ordering::Relaxed);
        if let Some(panic) = caller {
            resume_unwind(panic);
        }
        // A worker's panic goes to the caller, where the C ABI catches it.
        if panicked {
            panic!("a speech worker thread panicked");
        }
    }
}

/// Claims and runs tasks of the current job until none is left. Returns the
/// first panic of a task this thread ran; every panic also sets `panicked`
/// before its task counts as done, so the caller sees it.
fn run_tasks(s: &Shared) -> Option<Box<dyn Any + Send>> {
    let mut first_panic = None;
    let mut word = s.claim.load(Ordering::Acquire);
    loop {
        let (tasks, next) = ((word >> 16) & 0xFFFF, word & 0xFFFF);
        if next >= tasks {
            return first_panic;
        }
        if let Err(current) =
            s.claim
                .compare_exchange_weak(word, word + 1, Ordering::Acquire, Ordering::Acquire)
        {
            word = current;
            continue;
        }
        // SAFETY: the claim succeeded on a word that the job's caller published
        // after it stored `job`, and that caller returns only after this task
        // counts in `done`, so the job and its closure are alive.
        let run = unsafe { &*(*s.job.load(Ordering::Acquire)).run };
        if let Err(panic) = catch_unwind(AssertUnwindSafe(|| run(next as usize))) {
            s.panicked.store(true, Ordering::Relaxed);
            first_panic.get_or_insert(panic);
        }
        s.done.fetch_add(1, Ordering::Release);
        word = s.claim.load(Ordering::Acquire);
    }
}

fn work(s: &Shared) {
    INSIDE.set(true);
    // A fixed start, so a worker that starts late still sees the first job.
    let mut seen = 0;
    loop {
        let mut idle_since = Instant::now();
        let mut spins = 0u32;
        loop {
            let generation = s.generation.load(Ordering::Acquire);
            if generation != seen {
                seen = generation;
                break;
            }
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(256)
                && (s.rest.load(Ordering::Relaxed) == seen || idle_since.elapsed() > SPIN_FOR)
            {
                let mut lock = s.lock.lock();
                s.parked.fetch_add(1, Ordering::SeqCst);
                while s.generation.load(Ordering::SeqCst) == seen {
                    s.wake.wait(&mut lock);
                }
                s.parked.fetch_sub(1, Ordering::SeqCst);
                idle_since = Instant::now();
            } else {
                std::hint::spin_loop();
            }
        }
        // A late worker finds a later job or no task left: both are safe.
        // Its panics are in `panicked` for the caller.
        let _ = run_tasks(s);
    }
}

/// The teams a process may start, one per spread of `cpu::Spread`. A
/// process that runs Parakeet and Whisper gets both; each parks when idle.
pub const TEAMS: usize = super::Spread::TEAMS;
static TEAM: [OnceLock<Team>; TEAMS] = [const { OnceLock::new() }; TEAMS];

/// Starts team `slot` with the threads (the caller included), the affinity
/// mask (0: none), and the masks it can move to that `size` gives. Later
/// calls do nothing.
pub fn start(slot: usize, size: impl FnOnce() -> (usize, usize, Vec<usize>)) {
    TEAM[slot].get_or_init(|| {
        let (threads, domain, domains) = size();
        Team::new(slot, threads.max(1), domain, domains)
    });
}

/// Runs the calling thread's steps on team `slot`, pinned to its L3 domain,
/// until the guard drops.
pub fn enter(slot: usize) -> Entered {
    let previous = CURRENT.replace(slot);
    let team = team();
    team.steer();
    let domain = team.domain.load(Ordering::Relaxed);
    Entered {
        previous,
        throttle: keep_full_speed(),
        affinity: set_affinity(domain),
        priority: set_priority(ABOVE_NORMAL),
        ideal: set_ideal(ideal_cpu(domain, 0, team.workers + 1)),
    }
}

/// The caller's previous team, power throttling (`None`: nothing to
/// restore), affinity mask (0: nothing to restore), priority, and ideal
/// processor (`None`: nothing to restore).
pub struct Entered {
    previous: usize,
    throttle: Option<Throttle>,
    affinity: usize,
    priority: Option<i32>,
    ideal: Option<u32>,
}

impl Drop for Entered {
    fn drop(&mut self) {
        // The call is over: its workers park now, not after `SPIN_FOR`. A
        // call of another thread that still runs wakes them with its next step.
        if let Some(team) = TEAM[CURRENT.get()].get() {
            let ended = team.shared.generation.load(Ordering::Relaxed);
            team.shared.rest.store(ended, Ordering::Relaxed);
        }
        set_ideal(self.ideal);
        if let Some(priority) = self.priority {
            set_priority(priority);
        }
        set_affinity(self.affinity);
        if let Some(throttle) = self.throttle {
            set_throttle(throttle);
        }
        CURRENT.set(self.previous);
    }
}

/// The ideal processor of team thread `index` (0: the caller) of `threads`:
/// a processor of `domain` (of the process when 0), every other one when
/// there are two per thread (one per core with SMT). A raised thread
/// preempts lower-priority work only on its ideal or last processor, and
/// Windows assigns ideal processors round-robin per process: two team
/// threads with one ideal processor made some processes wait a time slice
/// per step.
fn ideal_cpu(domain: usize, index: usize, threads: usize) -> Option<u32> {
    let mask = if domain != 0 { domain } else { process_mask() };
    let cpus: Vec<u32> = (0..usize::BITS)
        .filter(|bit| mask >> bit & 1 == 1)
        .collect();
    if cpus.is_empty() {
        return None;
    }
    let stride = if cpus.len() >= 2 * threads { 2 } else { 1 };
    Some(cpus[(index * stride) % cpus.len()])
}

/// The time one processor spent, in the system's ticks.
#[derive(Clone, Copy)]
struct CpuTimes {
    busy: i64,
    total: i64,
}

/// The fraction of the processors of `domain` that were busy between the
/// two looks. Each list holds one entry per processor, in processor order.
fn busy_fraction(before: &[CpuTimes], after: &[CpuTimes], domain: usize) -> f64 {
    let (mut busy, mut total) = (0, 0);
    for (cpu, (first, last)) in before.iter().zip(after).enumerate() {
        if cpu < usize::BITS as usize && domain >> cpu & 1 == 1 {
            busy += last.busy - first.busy;
            total += last.total - first.total;
        }
    }
    if total > 0 {
        busy as f64 / total as f64
    } else {
        0.0
    }
}

/// The time of each processor of this processor group, or `None` when the
/// system does not say. Without it a team stays on its first domain.
#[cfg(windows)]
fn cpu_times() -> Option<Vec<CpuTimes>> {
    /// SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION. `kernel` includes `idle`.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Performance {
        idle: i64,
        kernel: i64,
        user: i64,
        dpc: i64,
        interrupt: i64,
        interrupts: u32,
    }
    /// SystemProcessorPerformanceInformation.
    const PROCESSOR_PERFORMANCE: u32 = 8;
    #[link(name = "ntdll")]
    extern "system" {
        fn NtQuerySystemInformation(
            class: u32,
            info: *mut std::ffi::c_void,
            length: u32,
            returned: *mut u32,
        ) -> i32;
    }
    let mut info = [Performance::default(); usize::BITS as usize];
    let mut returned = 0u32;
    // SAFETY: `info` holds the length given, and both pointers name locals
    // that live through the call. The call fills whole entries.
    let status = unsafe {
        NtQuerySystemInformation(
            PROCESSOR_PERFORMANCE,
            info.as_mut_ptr().cast(),
            std::mem::size_of_val(&info) as u32,
            &mut returned,
        )
    };
    if status < 0 {
        return None;
    }
    let count = returned as usize / std::mem::size_of::<Performance>();
    Some(
        info[..count]
            .iter()
            .map(|cpu| CpuTimes {
                busy: cpu.kernel + cpu.user - cpu.idle,
                total: cpu.kernel + cpu.user,
            })
            .collect(),
    )
}

#[cfg(not(windows))]
fn cpu_times() -> Option<Vec<CpuTimes>> {
    None
}

/// Puts a worker on `domain`. It needs no wake: a parked worker starts there,
/// and a running one moves at once.
#[cfg(windows)]
fn pin_thread(thread: &JoinHandle<()>, domain: usize, ideal: Option<u32>) {
    use std::os::windows::io::AsRawHandle;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::Threading::{SetThreadAffinityMask, SetThreadIdealProcessor};
    let handle = HANDLE(thread.as_raw_handle());
    // SAFETY: the `JoinHandle` keeps the thread's handle open through both
    // calls, and neither call takes a pointer. A failed call leaves the
    // thread where it was, which only costs speed.
    unsafe {
        SetThreadAffinityMask(handle, domain);
        if let Some(ideal) = ideal {
            SetThreadIdealProcessor(handle, ideal);
        }
    }
}

#[cfg(not(windows))]
fn pin_thread(_thread: &JoinHandle<()>, _domain: usize, _ideal: Option<u32>) {}

/// The processors this process may run on (0 when unknown).
#[cfg(windows)]
pub(super) fn process_mask() -> usize {
    use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessAffinityMask};
    let (mut process, mut system) = (0usize, 0usize);
    // SAFETY: both pointers name locals that live through the call.
    match unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut process, &mut system) } {
        Ok(()) => process,
        Err(_) => 0,
    }
}

#[cfg(not(windows))]
pub(super) fn process_mask() -> usize {
    0
}

/// Sets this thread's ideal processor and returns the previous one; `None`
/// for `None` or when the call failed.
#[cfg(windows)]
fn set_ideal(cpu: Option<u32>) -> Option<u32> {
    use windows::Win32::System::Threading::{GetCurrentThread, SetThreadIdealProcessor};
    // SAFETY: the pseudo handle names the calling thread; the call takes no pointer.
    let previous = unsafe { SetThreadIdealProcessor(GetCurrentThread(), cpu?) };
    (previous != u32::MAX).then_some(previous)
}

#[cfg(not(windows))]
fn set_ideal(_cpu: Option<u32>) -> Option<u32> {
    None
}

/// THREAD_PRIORITY_ABOVE_NORMAL. A thread that holds a task delays its step
/// until it runs again; at normal priority, a build on every core paused
/// team threads for hundreds of ms per take. The team parks when idle, so
/// the priority costs other programs nothing then.
#[cfg(not(target_os = "macos"))]
const ABOVE_NORMAL: i32 = 1;

/// macOS: QOS_CLASS_USER_INTERACTIVE, the class macOS keeps on the
/// performance cores. At the default class a team thread can run on an
/// efficiency core, and every step then waits for it.
#[cfg(target_os = "macos")]
const ABOVE_NORMAL: i32 = 0x21;

/// The power throttling of a thread: the control mask and the state mask
/// of `THREAD_POWER_THROTTLING_STATE`.
#[derive(Clone, Copy)]
#[cfg_attr(not(windows), allow(dead_code))]
struct Throttle(u32, u32);

/// Keeps the calling thread out of Windows power throttling, and returns
/// the state it had (`None` when the system does not say: nothing to
/// restore). An app in the background counts as low importance: Windows can
/// lower the clock of its threads, and on a hybrid CPU move them to the slow
/// cores, and every step then waits for them. The state stays with the
/// thread: a team thread keeps it (it only runs while there is work), and
/// the caller's thread gets its own state back when its `Entered` drops.
#[cfg(windows)]
fn keep_full_speed() -> Option<Throttle> {
    use windows::Win32::System::Threading::{
        GetCurrentThread, GetThreadInformation, ThreadPowerThrottling,
        THREAD_POWER_THROTTLING_CURRENT_VERSION, THREAD_POWER_THROTTLING_EXECUTION_SPEED,
        THREAD_POWER_THROTTLING_STATE,
    };
    let mut before = THREAD_POWER_THROTTLING_STATE {
        Version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: 0,
        StateMask: 0,
    };
    // SAFETY: the pseudo handle names the calling thread, and `before` lives
    // through the call with the size given. A failed call leaves it as it is.
    let known = unsafe {
        GetThreadInformation(
            GetCurrentThread(),
            ThreadPowerThrottling,
            std::ptr::from_mut(&mut before).cast(),
            std::mem::size_of_val(&before) as u32,
        )
    }
    .is_ok();
    set_throttle(Throttle(THREAD_POWER_THROTTLING_EXECUTION_SPEED, 0));
    known.then_some(Throttle(before.ControlMask, before.StateMask))
}

#[cfg(windows)]
fn set_throttle(Throttle(control, state): Throttle) {
    use windows::Win32::System::Threading::{
        GetCurrentThread, SetThreadInformation, ThreadPowerThrottling,
        THREAD_POWER_THROTTLING_CURRENT_VERSION, THREAD_POWER_THROTTLING_STATE,
    };
    let state = THREAD_POWER_THROTTLING_STATE {
        Version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: control,
        StateMask: state,
    };
    // SAFETY: the pseudo handle names the calling thread, and `state` lives
    // through the call with the size given. A failed call changes nothing.
    let _ = unsafe {
        SetThreadInformation(
            GetCurrentThread(),
            ThreadPowerThrottling,
            std::ptr::from_ref(&state).cast(),
            std::mem::size_of_val(&state) as u32,
        )
    };
}

#[cfg(not(windows))]
fn keep_full_speed() -> Option<Throttle> {
    None
}

#[cfg(not(windows))]
fn set_throttle(_throttle: Throttle) {}

/// Sets this thread's priority and returns the previous one; `None` when
/// the thread already had it, or the call failed.
#[cfg(windows)]
fn set_priority(priority: i32) -> Option<i32> {
    use windows::Win32::System::Threading::{
        GetCurrentThread, GetThreadPriority, SetThreadPriority, THREAD_PRIORITY,
    };
    // SAFETY: the pseudo handle names the calling thread; neither call takes a pointer.
    unsafe {
        let previous = GetThreadPriority(GetCurrentThread());
        if previous == priority {
            return None;
        }
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY(priority)).ok()?;
        Some(previous)
    }
}

/// macOS: the same with the thread's QoS class (`priority` is a
/// `qos_class_t`). A thread may set its own class without privileges.
#[cfg(target_os = "macos")]
fn set_priority(priority: i32) -> Option<i32> {
    extern "C" {
        fn pthread_self() -> usize;
        fn pthread_get_qos_class_np(thread: usize, class: *mut u32, relative: *mut i32) -> i32;
        fn pthread_set_qos_class_self_np(class: u32, relative: i32) -> i32;
    }
    // QOS_CLASS_DEFAULT: what an unspecified class (0) restores to; macOS
    // refuses to set 0.
    const DEFAULT: u32 = 0x15;
    let (mut previous, mut relative) = (0u32, 0i32);
    // SAFETY: `pthread_self` names the calling thread; both pointers name
    // locals that live through the call.
    unsafe {
        if pthread_get_qos_class_np(pthread_self(), &mut previous, &mut relative) != 0 {
            return None;
        }
        if previous == priority as u32 || pthread_set_qos_class_self_np(priority as u32, 0) != 0 {
            return None;
        }
    }
    Some(if previous == 0 { DEFAULT } else { previous } as i32)
}

/// Raising a thread's priority needs privileges on Linux and Android.
#[cfg(not(any(windows, target_os = "macos")))]
fn set_priority(_priority: i32) -> Option<i32> {
    None
}

/// Sets this thread's affinity mask and returns the previous one. A mask of
/// 0 changes nothing and returns 0.
#[cfg(windows)]
fn set_affinity(mask: usize) -> usize {
    use windows::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
    if mask == 0 {
        return 0;
    }
    // SAFETY: the pseudo handle names the calling thread. A mask outside the
    // process mask fails, returns 0, and leaves the thread as it was.
    unsafe { SetThreadAffinityMask(GetCurrentThread(), mask) }
}

/// Linux and Android: the same, with `sched_setaffinity` on the calling thread.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn set_affinity(mask: usize) -> usize {
    extern "C" {
        fn sched_getaffinity(pid: i32, size: usize, mask: *mut usize) -> i32;
        fn sched_setaffinity(pid: i32, size: usize, mask: *const usize) -> i32;
    }
    if mask == 0 {
        return 0;
    }
    let size = std::mem::size_of::<usize>();
    let mut previous = 0usize;
    // SAFETY: pid 0 is the calling thread; both masks are `size` bytes. A
    // mask outside the allowed processors fails and changes nothing. A
    // previous mask with processors past the first 64 reads as 0: nothing
    // to restore.
    unsafe {
        if sched_getaffinity(0, size, &mut previous) != 0 || sched_setaffinity(0, size, &mask) != 0
        {
            return 0;
        }
    }
    previous
}

#[cfg(not(any(windows, target_os = "linux", target_os = "android")))]
fn set_affinity(_mask: usize) -> usize {
    0
}

/// Runs `run` once on every thread of every started team, the workers
/// included: work that must happen on each thread, such as returning the
/// memory a thread's allocator keeps. Each task waits until every thread of
/// its team holds one, so no thread takes two. When the threads do not all
/// come within `ALL_THREADS_WAIT`, `run` runs on the caller only, so it must
/// be safe to repeat there.
pub fn on_every_thread(run: impl Fn() + Sync) {
    // Inside a task the team runs inline on this one thread, and the tasks
    // below would wait for each other forever.
    if INSIDE.get() {
        run();
        return;
    }
    for team in TEAM.iter().filter_map(OnceLock::get) {
        let threads = team.workers + 1;
        let started = AtomicUsize::new(0);
        let gave_up = AtomicBool::new(false);
        team.for_each(threads, &|_| {
            started.fetch_add(1, Ordering::AcqRel);
            let since = Instant::now();
            while started.load(Ordering::Acquire) < threads {
                if gave_up.load(Ordering::Relaxed) || since.elapsed() > ALL_THREADS_WAIT {
                    gave_up.store(true, Ordering::Relaxed);
                    return;
                }
                std::hint::spin_loop();
            }
            run();
        });
        if gave_up.load(Ordering::Relaxed) {
            run();
        }
    }
    if TEAM.iter().all(|team| team.get().is_none()) {
        run();
    }
}

/// The calling thread's team; one of every core, unpinned, if `start` did
/// not run for it.
fn team() -> &'static Team {
    TEAM[CURRENT.get()].get_or_init(|| {
        Team::new(
            CURRENT.get(),
            std::thread::available_parallelism().map_or(4, |n| n.get()),
            0,
            Vec::new(),
        )
    })
}

pub fn threads() -> usize {
    team().workers + 1
}

pub fn for_each(tasks: usize, run: impl Fn(usize) + Sync) {
    // A step inside a task (on a worker or the caller) runs inline.
    if INSIDE.get() {
        (0..tasks).for_each(run);
        return;
    }
    team().for_each(tasks, &run);
}

/// A buffer that parallel tasks write in disjoint ranges.
pub struct SendPtr<T>(*mut T);
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}
// SAFETY: every task touches its own disjoint range.
unsafe impl<T: Send> Send for SendPtr<T> {}
// SAFETY: as for `Send`: a task only reaches its own range, so no element is shared.
unsafe impl<T: Send> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    pub fn new(data: &mut [T]) -> Self {
        Self(data.as_mut_ptr())
    }

    /// # Safety
    /// The range must be inside the buffer, no other task may use it at the
    /// same time, and the buffer must outlive the returned slice.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice_mut<'a>(&self, start: usize, len: usize) -> &'a mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(start), len)
    }
}

pub fn chunks_mut<T: Send>(data: &mut [T], chunk: usize, run: impl Fn(usize, &mut [T]) + Sync) {
    let (len, ptr) = (data.len(), SendPtr::new(data));
    let chunk = chunk.max(1);
    for_each(len.div_ceil(chunk), |index| {
        let start = index * chunk;
        let end = (start + chunk).min(len);
        // SAFETY: chunks are disjoint and inside `data`, which outlives this call.
        let slice = unsafe { ptr.slice_mut(start, end - start) };
        run(index, slice);
    });
}

pub fn map<T: Send>(tasks: usize, run: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let mut slots: Vec<Option<T>> = (0..tasks).map(|_| None).collect();
    chunks_mut(&mut slots, 1, |index, slot| slot[0] = Some(run(index)));
    slots
        .into_iter()
        .map(|slot| slot.expect("every task ran"))
        .collect()
}
