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
    Bias,
    Gelu,
    Norm,
    Add,
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
    BusyHere,
    BusyAway,
    Home,
}

/// The kernel of a matrix product, for the table of products by shape.
#[derive(Clone, Copy, Debug)]
pub enum Product {
    Ternary,
    Int8,
    F32,
    F16,
    /// Q8_0 weights widened to f32 for each block (`fgemm::matmul_widened`).
    Q8Widened,
    /// `fgemm::matmul_into`, on the thread of a task.
    F32Into,
    /// `fgemm::matmul_nn_into`, on the thread of a task.
    F32Broadcast,
    /// `Half::matvec`: f16 weights and one input row.
    F16Vector,
}

#[cfg(feature = "profile")]
impl Stage {
    const COUNT: usize = Stage::Home as usize + 1;

    const NAMES: [&'static str; Stage::COUNT] = [
        "features (mel), wall",
        "encoder, wall",
        "decoder, wall",
        "quantize the activations, wall",
        "ternary product with its quantize, wall",
        "int8 product with its quantize, wall",
        "float product, wall",
        "bias and ReLU pass after a product, wall",
        "GELU pass, wall",
        "layer norm pass, wall",
        "residual add pass, wall",
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
        "last look: other load on the team's domain, percent",
        "last look: load on the quietest other domain, percent",
        "last look: 1 when the team is on its first domain",
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

    /// Keeps `value` as the count of `stage`: a gauge, not a sum.
    pub fn set(stage: Stage, value: u64) {
        COUNTS[stage as usize].store(value, Ordering::Relaxed);
    }

    /// The count of `stage` up to now.
    pub fn counted(stage: Stage) -> u64 {
        COUNTS[stage as usize].load(Ordering::Relaxed)
    }

    /// The products of one kind, shape, and thread count. `key` is 0 for a
    /// slot that no product took.
    struct Slot {
        key: AtomicU64,
        calls: AtomicU64,
        rows: AtomicU64,
        nanos: AtomicU64,
    }

    const SLOTS: usize = 4096;
    static PRODUCTS: [Slot; SLOTS] = [const {
        Slot {
            key: AtomicU64::new(0),
            calls: AtomicU64::new(0),
            rows: AtomicU64::new(0),
            nanos: AtomicU64::new(0),
        }
    }; SLOTS];

    /// The time of one product, from `product` to the drop of this value.
    #[derive(Debug)]
    pub struct ProductSpan {
        key: u64,
        rows: u64,
        start: std::time::Instant,
    }

    /// Row counts up to 16 are their own class; above, a class is the next
    /// power of two, so that the clips of a run share lines.
    fn row_class(m: usize) -> u64 {
        if m <= 16 {
            m as u64
        } else {
            16 + u64::from(m.next_power_of_two().trailing_zeros())
        }
    }

    /// One product of `m` input rows with an `n x k` weight matrix. On the
    /// thread of a task the time is thread time of one thread; on the
    /// calling thread it is wall time of the whole team.
    pub fn product(kind: super::Product, n: usize, k: usize, m: usize) -> ProductSpan {
        let threads = if crate::cpu::team::inside() {
            1
        } else {
            crate::cpu::team::threads().min(63) as u64
        };
        let key = 1
            | (kind as u64) << 1
            | threads << 5
            | row_class(m) << 11
            | (k.min(0xFFFF) as u64) << 17
            | (n.min(0xFF_FFFF) as u64) << 33;
        ProductSpan {
            key,
            rows: m as u64,
            start: std::time::Instant::now(),
        }
    }

    impl Drop for ProductSpan {
        fn drop(&mut self) {
            let nanos = self.start.elapsed().as_nanos() as u64;
            let first = (self.key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 52) as usize;
            for index in first..first + SLOTS {
                let slot = &PRODUCTS[index % SLOTS];
                let held = match slot.key.compare_exchange(
                    0,
                    self.key,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => self.key,
                    Err(held) => held,
                };
                if held == self.key {
                    slot.calls.fetch_add(1, Ordering::Relaxed);
                    slot.rows.fetch_add(self.rows, Ordering::Relaxed);
                    slot.nanos.fetch_add(nanos, Ordering::Relaxed);
                    return;
                }
            }
        }
    }

    /// The products by shape, the longest first, with the floor of each
    /// line. The peak of this CPU is measured here, in the same process and
    /// under the same load as the products.
    pub fn products() -> String {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512vnni") {
            let (clock_ghz, int8_peak, float_peak) =
                crate::cpu::v2::bench::peak(crate::cpu::Spread::OneCache);
            return table(clock_ghz, int8_peak, float_peak);
        }
        "products by shape: the floor needs AVX-512 VNNI (`kernel_bench peak`)\n".into()
    }

    /// `clock_ghz` and the peak counts of a core (`vpdpbusd` and
    /// `vfmadd231ps` for each cycle) give each line its floor: the time of
    /// its multiply-adds alone, on its threads. "over" is the time above
    /// that floor: what a kernel at the peak rate would save.
    #[cfg(target_arch = "x86_64")]
    fn table(clock_ghz: f64, int8_peak: f64, float_peak: f64) -> String {
        const KINDS: [&str; 8] = [
            "ternary",
            "int8",
            "f32",
            "f16",
            "q8 wide",
            "f32 into",
            "f32 bcast",
            "f16 vec",
        ];
        let mut lines: Vec<(u64, String)> = Vec::new();
        let (mut wall, mut wall_over, mut thread, mut thread_over) = (0.0, 0.0, 0.0, 0.0);
        for slot in &PRODUCTS {
            let key = slot.key.load(Ordering::Relaxed);
            if key == 0 {
                continue;
            }
            let kind = (key >> 1 & 0xF) as usize;
            let threads = key >> 5 & 0x3F;
            let class = key >> 11 & 0x3F;
            let (k, n) = (key >> 17 & 0xFFFF, key >> 33);
            let calls = slot.calls.load(Ordering::Relaxed);
            let rows = slot.rows.load(Ordering::Relaxed);
            let nanos = slot.nanos.load(Ordering::Relaxed);
            let work = (rows * n * k) as f64;
            let lanes = if kind < 2 {
                64.0 * int8_peak
            } else {
                16.0 * float_peak
            };
            let floor_ms = work / (lanes * clock_ghz * threads as f64) / 1e6;
            let ms = nanos as f64 / 1e6;
            let over = ms - floor_ms;
            if threads == 1 {
                thread += ms;
                thread_over += over;
            } else {
                wall += ms;
                wall_over += over;
            }
            let rows_of = if class <= 16 {
                format!("{class}")
            } else {
                format!("<={}", 1u64 << (class - 16))
            };
            lines.push((
                nanos,
                format!(
                    "  {:9} {n:>6} x {k:<5} rows {rows_of:>7} threads {threads:2} calls {calls:8} {ms:9.1} ms  floor {floor_ms:9.1} ms ({:3.0} %)  over {over:9.1} ms  {:6.1} MB\n",
                    KINDS[kind],
                    100.0 * floor_ms / ms.max(1e-9),
                    (n * k) as f64 * [0.25, 1.0, 4.0, 2.0, 1.0, 4.0, 4.0, 2.0][kind] / (1 << 20) as f64,
                ),
            ));
        }
        lines.sort_by_key(|line| std::cmp::Reverse(line.0));
        let mut out = format!(
            "products by shape (floor: the multiply-adds at {clock_ghz:.2} GHz, {int8_peak:.2} vpdpbusd and {float_peak:.2} vfmadd for each cycle and core)\n"
        );
        for (_, line) in lines.iter().take(60) {
            out += line;
        }
        out += &format!(
            "  products on the calling thread: {wall:.1} ms wall, {wall_over:.1} ms over the floor\n  products inside tasks: {thread:.1} ms thread time, {thread_over:.1} ms over the floor\n"
        );
        out
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
pub use on::{
    allocated, count, counted, product, products, report, set, span, ProductSpan, Span,
    ALLOCATED_BYTES, ALLOCATIONS,
};

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
pub fn product(_kind: Product, _n: usize, _k: usize, _m: usize) -> Span {
    Span
}

#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn count(_stage: Stage, _by: u64) {}

#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn set(_stage: Stage, _value: u64) {}
