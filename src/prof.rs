//! Stage timers and counters of a profile build:
//! `cargo build --release --features profile`, then `trit transcribe`
//! prints the table on stderr. Without the feature every function here is
//! empty and a span has no size, so the engine pays nothing.
//!
//! A span on the calling thread measures wall time ("wall"). A span inside
//! a task measures the time of that task, summed over all threads
//! ("threads"). Spans can lie inside each other: `Ternary` holds `Quantize`.

/// What a span or a count belongs to.
#[derive(Clone, Copy, Debug)]
pub enum Stage {
    Features,
    Encoder,
    Decoder,
    Quantize,
    Ternary,
    Int8,
    Float,
    Attention,
    Scores,
    Softmax,
    Values,
    StepAttention,
    HeadMajor,
    BufFill,
    BeamScores,
    BeamStep,
    BeamClone,
    Jobs,
    Moves,
}

#[cfg(feature = "profile")]
impl Stage {
    const COUNT: usize = Stage::Moves as usize + 1;

    const NAMES: [&'static str; Stage::COUNT] = [
        "features (mel), wall",
        "encoder, wall",
        "decoder, wall",
        "quantize the activations, wall",
        "ternary product with its quantize, wall",
        "int8 product with its quantize, wall",
        "float product, wall",
        "attention with a head-major copy, wall",
        "attention: scores, threads",
        "attention: softmax, threads",
        "attention: values, threads",
        "self-attention of a decoder step, wall",
        "head-major copy of keys and values, wall",
        "zero fill of a new pool buffer, wall",
        "beam search: score job, wall",
        "beam search: decoder step, wall",
        "beam search: copy of a state that splits, wall",
        "jobs of the team (count)",
        "moves of the team to another L3 domain (count)",
    ];
}

#[cfg(feature = "profile")]
mod on {
    use super::Stage;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NANOS: [AtomicU64; Stage::COUNT] = [const { AtomicU64::new(0) }; Stage::COUNT];
    static COUNTS: [AtomicU64; Stage::COUNT] = [const { AtomicU64::new(0) }; Stage::COUNT];
    /// Allocations of the process and their bytes; the CLI's allocator counts them.
    pub static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
    pub static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
    /// Allocations and bytes by size: class `c` holds sizes below `2^c` bytes.
    static BY_SIZE: [[AtomicU64; 2]; 40] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; 40];

    /// One allocation of `bytes`, from the allocator of the CLI.
    pub fn allocated(bytes: usize) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        let class = (usize::BITS - bytes.leading_zeros()).min(39) as usize;
        BY_SIZE[class][0].fetch_add(1, Ordering::Relaxed);
        BY_SIZE[class][1].fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// The time from `span` to the drop of this value.
    #[derive(Debug)]
    pub struct Span(Stage, std::time::Instant);

    pub fn span(stage: Stage) -> Span {
        Span(stage, std::time::Instant::now())
    }

    impl Span {
        /// Ends the span here, before the end of its scope.
        pub fn end(self) {}
    }

    impl Drop for Span {
        fn drop(&mut self) {
            let nanos = self.1.elapsed().as_nanos() as u64;
            NANOS[self.0 as usize].fetch_add(nanos, Ordering::Relaxed);
            COUNTS[self.0 as usize].fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn count(stage: Stage, by: u64) {
        COUNTS[stage as usize].fetch_add(by, Ordering::Relaxed);
    }

    /// The count of `stage` up to now.
    pub fn counted(stage: Stage) -> u64 {
        COUNTS[stage as usize].load(Ordering::Relaxed)
    }

    /// One line for each stage that ran: its time and its count.
    pub fn report() -> String {
        let mut out = String::new();
        for (index, name) in Stage::NAMES.iter().enumerate() {
            let nanos = NANOS[index].load(Ordering::Relaxed);
            let count = COUNTS[index].load(Ordering::Relaxed);
            if count > 0 {
                out += &format!("  {name:48} {:10.1} ms {count:10}\n", nanos as f64 / 1e6);
            }
        }
        out += &format!(
            "  {:48} {:10.1} MB {:10}\n",
            "allocations of the process",
            ALLOCATED_BYTES.load(Ordering::Relaxed) as f64 / (1 << 20) as f64,
            ALLOCATIONS.load(Ordering::Relaxed)
        );
        for (class, [count, bytes]) in BY_SIZE.iter().enumerate() {
            let (count, bytes) = (count.load(Ordering::Relaxed), bytes.load(Ordering::Relaxed));
            if count > 0 {
                let name = format!("  allocations below {} bytes", 1u64 << class);
                out += &format!(
                    "  {name:48} {:10.1} MB {count:10}\n",
                    bytes as f64 / (1 << 20) as f64
                );
            }
        }
        out
    }
}

#[cfg(feature = "profile")]
pub use on::{allocated, count, counted, report, span, Span, ALLOCATED_BYTES, ALLOCATIONS};

/// Nothing: this build has no profile.
#[cfg(not(feature = "profile"))]
#[derive(Debug)]
pub struct Span;

#[cfg(not(feature = "profile"))]
impl Span {
    #[inline(always)]
    pub fn end(self) {}
}

#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn span(_stage: Stage) -> Span {
    Span
}

#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn count(_stage: Stage, _by: u64) {}
