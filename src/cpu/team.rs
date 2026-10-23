//! Worker threads for the speech engine.
//!
//! A take runs about 1,200 short parallel steps. A thread pool that puts
//! idle threads to sleep pays about 20 µs to wake them for each step after a
//! short serial gap. These workers spin on a counter while steps keep coming,
//! so a step starts in about a microsecond, and they park after
//! `SPIN_FOR` without work, so an idle app uses no CPU.

use parking_lot::{Condvar, Mutex};
use std::cell::Cell;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// Longer than the gaps inside a take, short enough to be invisible as idle CPU.
const SPIN_FOR: Duration = Duration::from_millis(3);

struct Job {
    run: *const (dyn Fn(usize) + Sync),
    tasks: usize,
}

struct Shared {
    generation: AtomicUsize,
    job: AtomicPtr<Job>,
    next: AtomicUsize,
    /// Workers that have not finished the current job yet.
    busy: AtomicUsize,
    /// A task of the current job panicked on a worker.
    panicked: AtomicBool,
    parked: AtomicUsize,
    lock: Mutex<()>,
    wake: Condvar,
}

pub struct Team {
    shared: Arc<Shared>,
    workers: usize,
    /// Affinity mask of the team's L3 domain, 0 when the team is not pinned.
    domain: usize,
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
    fn new(threads: usize, domain: usize) -> Self {
        let shared = Arc::new(Shared {
            generation: AtomicUsize::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            next: AtomicUsize::new(0),
            busy: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            parked: AtomicUsize::new(0),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        let workers = threads.saturating_sub(1);
        for index in 0..workers {
            let for_worker = Arc::clone(&shared);
            let spawned = std::thread::Builder::new()
                .name(format!("tdt-speech-{index}"))
                .spawn(move || {
                    set_affinity(domain);
                    work(&for_worker);
                });
            if spawned.is_err() {
                // Fewer threads only make the engine slower.
                return Self::with_workers(shared, index, domain);
            }
        }
        Self::with_workers(shared, workers, domain)
    }

    fn with_workers(shared: Arc<Shared>, workers: usize, domain: usize) -> Self {
        Self {
            shared,
            workers,
            domain,
            submit: Mutex::new(()),
        }
    }

    fn for_each(&self, tasks: usize, run: &(dyn Fn(usize) + Sync)) {
        if tasks <= 1 || self.workers == 0 || INSIDE.get() {
            (0..tasks).for_each(run);
            return;
        }
        let _one_caller = self.submit.lock();
        INSIDE.set(true);
        // SAFETY: the job only lives until every worker has left it (busy == 0
        // below), so erasing the closure's lifetime cannot outlive the borrow.
        let run: &'static (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(run) };
        let job = Job { run, tasks };
        let s = &*self.shared;
        s.job
            .store(&job as *const Job as *mut Job, Ordering::Relaxed);
        s.next.store(0, Ordering::Relaxed);
        s.busy.store(self.workers, Ordering::Relaxed);
        s.generation.fetch_add(1, Ordering::SeqCst);
        if s.parked.load(Ordering::SeqCst) > 0 {
            let _lock = s.lock.lock();
            s.wake.notify_all();
        }
        // The workers read `job` from this stack frame: a panic here must
        // wait for them before it unwinds the frame.
        let caller = catch_unwind(AssertUnwindSafe(|| run_tasks(s, &job)));
        while s.busy.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        INSIDE.set(false);
        if let Err(panic) = caller {
            resume_unwind(panic);
        }
        // A worker's panic goes to the caller, where the C ABI catches it.
        if s.panicked.swap(false, Ordering::Relaxed) {
            panic!("a speech worker thread panicked");
        }
    }
}

fn run_tasks(s: &Shared, job: &Job) {
    // SAFETY: the caller keeps the closure alive until every worker is done.
    let run = unsafe { &*job.run };
    loop {
        let task = s.next.fetch_add(1, Ordering::Relaxed);
        if task >= job.tasks {
            break;
        }
        run(task);
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
            if spins.is_multiple_of(256) && idle_since.elapsed() > SPIN_FOR {
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
        // SAFETY: the job pointer was published before the generation changed.
        let job = unsafe { &*s.job.load(Ordering::Relaxed) };
        // A worker that dies without the decrement would hang the caller.
        if catch_unwind(AssertUnwindSafe(|| run_tasks(s, job))).is_err() {
            s.panicked.store(true, Ordering::Relaxed);
        }
        s.busy.fetch_sub(1, Ordering::Release);
    }
}

/// The teams a process may start, one per spread of `cpu::Spread`. A
/// process that runs Parakeet and Whisper gets both; each parks when idle.
pub const TEAMS: usize = 2;
static TEAM: [OnceLock<Team>; TEAMS] = [OnceLock::new(), OnceLock::new()];

/// Starts team `slot` with the threads (the caller included) and the
/// affinity mask (0: none) that `size` gives. Later calls do nothing.
pub fn start(slot: usize, size: impl FnOnce() -> (usize, usize)) {
    TEAM[slot].get_or_init(|| {
        let (threads, domain) = size();
        Team::new(threads.max(1), domain)
    });
}

/// Runs the calling thread's steps on team `slot`, pinned to its L3 domain,
/// until the guard drops.
pub fn enter(slot: usize) -> Entered {
    let previous = CURRENT.replace(slot);
    Entered {
        previous,
        affinity: set_affinity(team().domain),
    }
}

/// The caller's previous team and affinity mask (0: nothing to restore).
pub struct Entered {
    previous: usize,
    affinity: usize,
}

impl Drop for Entered {
    fn drop(&mut self) {
        set_affinity(self.affinity);
        CURRENT.set(self.previous);
    }
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

/// Linux: the same, with `sched_setaffinity` on the calling thread.
#[cfg(target_os = "linux")]
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

#[cfg(not(any(windows, target_os = "linux")))]
fn set_affinity(_mask: usize) -> usize {
    0
}

/// Runs `run` once on every thread of every started team, the workers
/// included: work that must happen on each thread, such as returning the
/// memory a thread's allocator keeps. Each task waits until every thread of
/// its team holds one, so no thread takes two.
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
        team.for_each(threads, &|_| {
            started.fetch_add(1, Ordering::AcqRel);
            while started.load(Ordering::Acquire) < threads {
                std::hint::spin_loop();
            }
            run();
        });
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
            std::thread::available_parallelism().map_or(4, |n| n.get()),
            0,
        )
    })
}

/// Threads that work on each step, the caller included.
pub fn threads() -> usize {
    team().workers + 1
}

/// `run(task)` for every task in `0..tasks`, spread over the team.
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
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

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

/// `run(index, chunk)` for consecutive chunks of `chunk` elements.
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

/// `run(task)` for every task, results in task order.
pub fn map<T: Send>(tasks: usize, run: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let mut slots: Vec<Option<T>> = (0..tasks).map(|_| None).collect();
    chunks_mut(&mut slots, 1, |index, slot| slot[0] = Some(run(index)));
    slots
        .into_iter()
        .map(|slot| slot.expect("every task ran"))
        .collect()
}
