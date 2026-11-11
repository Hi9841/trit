//! One integer product for ternary and int8 weights.
//!
//! `out[frame][row] = sum over groups of (w . x) * weight scale * frame scale`,
//! where a group is 128 inputs (32 for ggml Q8_0), `x` is the frame as i8
//! (`quantize.rs`), and `w` is the row as bytes. The float steps of a value
//! are those of the first kernels (`kernels.rs`, `int8.rs`), in the same
//! order, so every value is the same, bit for bit.
//!
//! Differences from the first kernels:
//!
//! - **One tile.** A ternary row and an int8 row are both "unsigned bytes
//!   times signed bytes plus a start value", so one tile runs both, on every
//!   path. Ternary layers get the 256-bit `vpdpbusd` path too.
//! - **No correction step.** A dot product starts at `-bias * sum(x)`.
//! - **Frames split too.** A layer with few rows (384 give 12 tasks) splits
//!   its frames, so each thread of a large team has a task.
//! - **Every path gives the same bits,** the scalar path too.
//! - **Any number of rows.** The last unit of 16 rows can be partial.
//! - **arm64: 16 sums in flight** (one unit by 4 frames, `sdot`, `smull`, or
//!   `smmla`), where the first arm64 tiles keep 8.
//!
//! The tile itself has the shape of the first kernels (two 16-row blocks by
//! 6 frames, codes unpacked in registers). On Zen 4 that tile runs at the
//! limit of its arithmetic: 32 `vpdpbusd` and 2 float steps for each group
//! go through the two multiply pipes, and the first ternary kernel runs
//! at 94 to 95 % of the `vpdpbusd` peak. Tested and not kept: codes unpacked
//! once into a byte panel, with 14 frames side by side. It was 5 to 30 %
//! slower: the panel is 4 times the bytes, and the unpack in registers
//! uses pipes that the products do not.

use super::quantize::{quantize, Plan, Quantized, Rounder};
use crate::cpu::buf::Buf;
use crate::cpu::store::Store;
use crate::cpu::team;

/// Rows of a unit: the weights of 16 rows are stored side by side.
const UNIT: usize = 16;
/// Bytes of a unit for 4 inputs at one byte per weight, or for 16 inputs
/// at 2 bits per weight.
const STEP: usize = UNIT * 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Path {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// 256-bit `vpdpbusd`, VEX form (AVX-VNNI).
    #[cfg(target_arch = "x86_64")]
    Vnni256Vex,
    /// 256-bit `vpdpbusd`, EVEX form (AVX-512 VNNI with VL).
    #[cfg(target_arch = "x86_64")]
    Vnni256Evex,
    #[cfg(target_arch = "x86_64")]
    Vnni512,
    /// Armv8.0: widening products (`smull`) and pairwise adds.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// `sdot` (Armv8.2 dot product).
    #[cfg(target_arch = "aarch64")]
    Sdot,
    /// `smmla` (Armv8.6 i8mm) for int8 layers; ternary layers use `sdot`.
    #[cfg(target_arch = "aarch64")]
    Mmla,
}

impl Path {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            use crate::cpu::int8::Path as First;
            match crate::cpu::isa::isa_choice().int8 {
                First::Scalar => Self::Scalar,
                First::Avx2 => Self::Avx2,
                First::Vnni256 if is_x86_feature_detected!("avxvnni") => Self::Vnni256Vex,
                First::Vnni256 if is_x86_feature_detected!("avx512vl") => Self::Vnni256Evex,
                First::Vnni256 => Self::Avx2,
                First::Vnni512 => Self::Vnni512,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            use crate::cpu::int8::Path as First;
            match crate::cpu::isa::isa_choice().int8 {
                First::Scalar => Self::Scalar,
                First::Neon => Self::Neon,
                First::NeonDot => Self::Sdot,
                First::NeonMmla => Self::Mmla,
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        Self::Scalar
    }

    /// `vpdpbusd` reads the weight byte as unsigned.
    fn dpbusd(self) -> bool {
        #[cfg(target_arch = "x86_64")]
        return matches!(self, Self::Vnni256Vex | Self::Vnni256Evex | Self::Vnni512);
        #[cfg(not(target_arch = "x86_64"))]
        false
    }

    /// The int8 bytes of a unit are `[k / 8][16 rows][8]` for `smmla`,
    /// else `[k / 4][16 rows][4]`.
    fn pairs_of_eight(self) -> bool {
        #[cfg(target_arch = "aarch64")]
        return self == Self::Mmla;
        #[cfg(not(target_arch = "aarch64"))]
        false
    }

    /// Frames of a full tile.
    fn frames(self) -> usize {
        match self {
            Self::Scalar => 1,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 | Self::Vnni256Vex | Self::Vnni256Evex => 4,
            #[cfg(target_arch = "x86_64")]
            Self::Vnni512 => 6,
            #[cfg(target_arch = "aarch64")]
            Self::Neon | Self::Sdot | Self::Mmla => 4,
        }
    }

    fn rounder(self) -> Rounder {
        match self {
            Self::Scalar => Rounder::Scalar,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 | Self::Vnni256Vex | Self::Vnni256Evex => Rounder::Avx2,
            #[cfg(target_arch = "x86_64")]
            Self::Vnni512 => Rounder::Avx512,
            #[cfg(target_arch = "aarch64")]
            Self::Neon | Self::Sdot | Self::Mmla => Rounder::Neon,
        }
    }
}

/// Where the int8 weight of row `r` (of 16) for input `kk` is in its unit.
fn int8_offset(path: Path, r: usize, kk: usize) -> usize {
    if path.pairs_of_eight() {
        ((kk / 8) * UNIT + r) * 8 + kk % 8
    } else {
        ((kk / 4) * UNIT + r) * 4 + kk % 4
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// 2-bit codes 0, 1, 2 for weights -1, 0, 1: `[units][k / 16][16 rows][4]`,
    /// where bits `2s..2s + 2` of byte `j` hold input `16 * (k / 16 index) + 4 * s + j`.
    Ternary,
    /// One byte per weight, `[units][k / 4][16 rows][4]` (`[k / 8][16][8]`
    /// for `smmla`): `w + 128` for `vpdpbusd`, else `w`.
    Int8,
}

pub struct IntProduct {
    path: Path,
    kind: Kind,
    pub n: usize,
    pub k: usize,
    /// Inputs per scale.
    group: usize,
    half_scales: bool,
    weights: Store<u8>,
    /// `[units][k / group][16 rows]`.
    scales: Store<f32>,
}

impl std::fmt::Debug for IntProduct {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntProduct")
            .field("path", &self.path)
            .field("kind", &self.kind)
            .field("n", &self.n)
            .field("k", &self.k)
            .finish()
    }
}

thread_local! {
    /// The weights of one row, for the scalar path.
    static ROW: std::cell::RefCell<Vec<i8>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: tasks write disjoint output elements; the buffer outlives the step.
unsafe impl Send for OutPtr {}
// SAFETY: as for `Send`: tasks write disjoint output elements.
unsafe impl Sync for OutPtr {}

impl IntProduct {
    /// Ternary weights. `row(r, codes)` writes the `k` codes (0, 1, 2) of
    /// row `r`; `scales` is row-major `n x (k / group)`; `k` is a multiple
    /// of `group`, and `group` of 16.
    pub fn ternary(
        scales: &[f32],
        n: usize,
        k: usize,
        group: usize,
        row: impl Fn(usize, &mut [u8]),
    ) -> Self {
        assert!(k.is_multiple_of(group) && group.is_multiple_of(16));
        let units = n.div_ceil(UNIT);
        let mut packed = vec![0u8; units * (k / 16) * STEP];
        let mut codes = vec![1u8; UNIT * k];
        for unit in 0..units {
            for (r, codes) in codes.chunks_exact_mut(k).enumerate() {
                if unit * UNIT + r < n {
                    row(unit * UNIT + r, codes);
                } else {
                    // A row past the end has weight 0 everywhere: code 1.
                    codes.fill(1);
                }
            }
            for k16 in 0..k / 16 {
                for r in 0..UNIT {
                    let src = &codes[r * k + k16 * 16..][..16];
                    let dst = ((unit * (k / 16) + k16) * UNIT + r) * 4;
                    packed[dst..dst + 4].copy_from_slice(&crate::cpu::kernels::pack_span(src));
                }
            }
        }
        Self {
            path: Path::detect(),
            kind: Kind::Ternary,
            n,
            k,
            group,
            half_scales: false,
            weights: Store::weights(packed),
            scales: Store::weights(unit_scales(scales, n, k / group)),
        }
    }

    /// Int8 weights from f32 rows, 128 inputs per scale. `row(r, out)`
    /// writes row `r`; `k` is a multiple of 128.
    pub fn int8_from_f32(n: usize, k: usize, row: impl Fn(usize, &mut [f32]) + Sync) -> Self {
        const GROUP: usize = 128;
        assert!(k.is_multiple_of(GROUP));
        let path = Path::detect();
        let flip = if path.dpbusd() { 0x80 } else { 0 };
        let groups = k / GROUP;
        let units = n.div_ceil(UNIT);
        let mut weights = vec![flip; units * k * UNIT];
        let mut scales = vec![0.0f32; units * groups * UNIT];
        let (all_weights, all_scales) = (
            team::SendPtr::new(&mut weights),
            team::SendPtr::new(&mut scales),
        );
        team::for_each(units, |unit| {
            // SAFETY: each unit belongs to one task; the buffers outlive the step.
            let (weights, scales) = unsafe {
                (
                    all_weights.slice_mut(unit * k * UNIT, k * UNIT),
                    all_scales.slice_mut(unit * groups * UNIT, groups * UNIT),
                )
            };
            let mut values = vec![0.0f32; k];
            for r in 0..UNIT.min(n - unit * UNIT) {
                row(unit * UNIT + r, &mut values);
                for (g, span) in values.as_chunks::<GROUP>().0.iter().enumerate() {
                    let peak = span.iter().fold(0.0f32, |peak, v| peak.max(v.abs()));
                    let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
                    scales[g * UNIT + r] = scale;
                    for (i, v) in span.iter().enumerate() {
                        let code = (v / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
                        weights[int8_offset(path, r, g * GROUP + i)] = code as u8 ^ flip;
                    }
                }
            }
        });
        Self {
            path,
            kind: Kind::Int8,
            n,
            k,
            group: GROUP,
            half_scales: false,
            weights: Store::weights(weights),
            scales: Store::weights(scales),
        }
    }

    /// ggml Q8_0 weights as stored: row-major `n x k` codes and one scale
    /// per block of 32 (the f16 scale as f32). `k` is a multiple of 32.
    pub fn int8_from_q8_0(n: usize, k: usize, codes: &[i8], block_scales: &[f32]) -> Self {
        const BLOCK: usize = 32;
        assert!(k.is_multiple_of(BLOCK) && codes.len() >= n * k);
        let path = Path::detect();
        let flip = if path.dpbusd() { 0x80 } else { 0 };
        let units = n.div_ceil(UNIT);
        let mut weights = vec![flip; units * k * UNIT];
        for row in 0..n {
            let (unit, r) = (row / UNIT, row % UNIT);
            for (kk, code) in codes[row * k..(row + 1) * k].iter().enumerate() {
                weights[unit * k * UNIT + int8_offset(path, r, kk)] = *code as u8 ^ flip;
            }
        }
        Self {
            path,
            kind: Kind::Int8,
            n,
            k,
            group: BLOCK,
            half_scales: true,
            weights: Store::weights(weights),
            scales: Store::weights(unit_scales(block_scales, n, k / BLOCK)),
        }
    }

    /// The code (0, 1, 2) of a ternary weight.
    pub fn ternary_code(&self, row: usize, kk: usize) -> u8 {
        debug_assert!(self.kind == Kind::Ternary);
        let (unit, r) = (row / UNIT, row % UNIT);
        let byte = ((unit * (self.k / 16) + kk / 16) * UNIT + r) * 4 + kk % 4;
        (self.weights[byte] >> ((kk % 16) / 4 * 2)) & 3
    }

    /// The code of an int8 weight.
    pub fn int8_code(&self, row: usize, kk: usize) -> i8 {
        debug_assert!(self.kind == Kind::Int8);
        let (unit, r) = (row / UNIT, row % UNIT);
        let byte = self.weights[unit * self.k * UNIT + int8_offset(self.path, r, kk)];
        let flip = if self.path.dpbusd() { 0x80 } else { 0 };
        (byte ^ flip) as i8
    }

    /// The scale of `row` for group `g`.
    pub fn scale(&self, row: usize, g: usize) -> f32 {
        let groups = self.k / self.group;
        self.scales[((row / UNIT) * groups + g) * UNIT + row % UNIT]
    }

    /// Inputs per scale.
    pub fn group(&self) -> usize {
        self.group
    }

    /// The weight that row `row` gives input `kk`, before its scale.
    fn weight(&self, row: usize, kk: usize) -> i8 {
        match self.kind {
            Kind::Ternary => self.ternary_code(row, kk) as i8 - 1,
            Kind::Int8 => self.int8_code(row, kk),
        }
    }

    /// The product computes `(w + bias) . x` on this path.
    fn bias(&self) -> i32 {
        match self.kind {
            Kind::Ternary if self.path == Path::Scalar => 0,
            Kind::Ternary => 1,
            Kind::Int8 if self.path.dpbusd() => 128,
            Kind::Int8 => 0,
        }
    }

    /// `x` is `m x k` f32; returns `m x n`.
    pub fn forward(&self, x: &[f32], m: usize) -> Buf {
        let mut out = Buf::scratch(m * self.n);
        let plan = Plan {
            rounder: self.path.rounder(),
            group: self.group,
            bias: self.bias(),
            half_scales: self.half_scales,
        };
        let q = quantize(plan, x, m, self.k);
        let ptr = OutPtr(out.as_mut_ptr());
        let pairs = self.n.div_ceil(UNIT).div_ceil(2);
        // Few rows and many frames: split the frames too, in whole tiles,
        // so that each thread has about two tasks.
        let tile = self.path.frames();
        let wanted = (2 * team::threads()).div_ceil(pairs.max(1));
        let span = m.div_ceil(wanted).next_multiple_of(4 * tile).max(1);
        let spans = m.div_ceil(span);
        team::for_each(pairs * spans, |task| {
            let (pair, at) = (task / spans, (task % spans) * span);
            let frames = at..(at + span).min(m);
            match self.path {
                Path::Scalar => self.scalar_task(&q, pair, frames, ptr),
                #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                // SAFETY: the path was detected from the CPU.
                _ => unsafe { self.simd_task(&q, pair, frames, ptr) },
            }
        });
        out
    }

    /// The rows of pair `pair` (32 rows) for `frames`, one value at a time,
    /// with the float steps of the SIMD paths: `dot as f32` times the
    /// rounded product of the two scales, added with one rounding.
    fn scalar_task(&self, q: &Quantized, pair: usize, frames: std::ops::Range<usize>, out: OutPtr) {
        let (k, group) = (self.k, self.group);
        let groups = k / group;
        let mut weights = ROW.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
        weights.resize(k, 0);
        for row in pair * 2 * UNIT..((pair + 1) * 2 * UNIT).min(self.n) {
            for (kk, w) in weights.iter_mut().enumerate() {
                *w = self.weight(row, kk);
            }
            for frame in frames.clone() {
                let values = &q.values[frame * k..][..k];
                let mut total = 0.0f32;
                for (g, (w, x)) in weights
                    .chunks_exact(group)
                    .zip(values.chunks_exact(group))
                    .enumerate()
                {
                    let dot = w
                        .iter()
                        .zip(x)
                        .fold(q.starts[frame * groups + g], |dot, (w, x)| {
                            dot + i32::from(*w) * i32::from(*x)
                        });
                    let scale = self.scale(row, g) * q.scales[frame * groups + g];
                    total = (dot as f32).mul_add(scale, total);
                }
                // SAFETY: each output element belongs to one task.
                unsafe { *out.0.add(frame * self.n + row) = total };
            }
        }
        ROW.with(|cell| *cell.borrow_mut() = weights);
    }

    /// The rows of pair `pair` (two units) for `frames`.
    #[cfg(target_arch = "x86_64")]
    unsafe fn simd_task(
        &self,
        q: &Quantized,
        pair: usize,
        frames: std::ops::Range<usize>,
        out: OutPtr,
    ) {
        let (k, groups) = (self.k, self.k / self.group);
        let units = self.n.div_ceil(UNIT);
        let first = 2 * pair;
        let count = (units - first).min(2);
        let unit_bytes = match self.kind {
            Kind::Ternary => (k / 16) * STEP,
            Kind::Int8 => (k / 4) * STEP,
        };
        let bytes = |unit: usize| self.weights.as_ptr().add(unit * unit_bytes);
        let scales = |unit: usize| self.scales.as_ptr().add(unit * groups * UNIT);
        let live = |unit: usize, half: usize, rows: usize| {
            (self.n.saturating_sub(unit * UNIT + half * rows)).min(rows)
        };
        let shape = Shape {
            groups,
            spans: self.group / 16,
            k,
            n: self.n,
        };
        let ternary = self.kind == Kind::Ternary;
        match self.path {
            Path::Vnni512 if count == 2 => {
                let lanes = Lanes {
                    bytes: [bytes(first), bytes(first + 1)],
                    scales: [scales(first), scales(first + 1)],
                    out: [out.0.add(first * UNIT), out.0.add((first + 1) * UNIT)],
                    live: [live(first, 0, 16), live(first + 1, 0, 16)],
                };
                if ternary {
                    sweep_512_ternary_2(&lanes, q, &shape, frames);
                } else {
                    sweep_512_bytes_2(&lanes, q, &shape, frames);
                }
            }
            Path::Vnni512 => {
                let lanes = Lanes {
                    bytes: [bytes(first)],
                    scales: [scales(first)],
                    out: [out.0.add(first * UNIT)],
                    live: [live(first, 0, 16)],
                };
                if ternary {
                    sweep_512_ternary_1(&lanes, q, &shape, frames);
                } else {
                    sweep_512_bytes_1(&lanes, q, &shape, frames);
                }
            }
            // 256-bit paths: the two halves of each unit, one after the other.
            path => {
                for unit in first..first + count {
                    for half in 0..2 {
                        let rows = live(unit, half, 8);
                        if rows == 0 {
                            continue;
                        }
                        let lanes = Lanes {
                            bytes: [bytes(unit).add(half * STEP / 2)],
                            scales: [scales(unit).add(half * UNIT / 2)],
                            out: [out.0.add(unit * UNIT + half * UNIT / 2)],
                            live: [rows],
                        };
                        let frames = frames.clone();
                        match (path, ternary) {
                            (Path::Vnni256Vex, true) => {
                                sweep_vex_ternary(&lanes, q, &shape, frames)
                            }
                            (Path::Vnni256Vex, false) => sweep_vex_bytes(&lanes, q, &shape, frames),
                            (Path::Vnni256Evex, true) => {
                                sweep_evex_ternary(&lanes, q, &shape, frames)
                            }
                            (Path::Vnni256Evex, false) => {
                                sweep_evex_bytes(&lanes, q, &shape, frames)
                            }
                            (_, true) => sweep_avx2_ternary(&lanes, q, &shape, frames),
                            (_, false) => sweep_avx2_bytes(&lanes, q, &shape, frames),
                        }
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl IntProduct {
    /// The rows of pair `pair` (two units) for `frames`, one unit at a time.
    unsafe fn simd_task(
        &self,
        q: &Quantized,
        pair: usize,
        frames: std::ops::Range<usize>,
        out: OutPtr,
    ) {
        let (k, groups) = (self.k, self.k / self.group);
        let shape = Shape {
            groups,
            spans: self.group / 16,
            k,
            n: self.n,
        };
        let unit_bytes = match self.kind {
            Kind::Ternary => (k / 16) * STEP,
            Kind::Int8 => (k / 4) * STEP,
        };
        for unit in 2 * pair..(2 * pair + 2).min(self.n.div_ceil(UNIT)) {
            let lanes = Lanes {
                bytes: [self.weights.as_ptr().add(unit * unit_bytes)],
                scales: [self.scales.as_ptr().add(unit * groups * UNIT)],
                out: [out.0.add(unit * UNIT)],
                live: [(self.n - unit * UNIT).min(UNIT)],
            };
            let frames = frames.clone();
            match (self.path, self.kind) {
                (Path::Neon, Kind::Ternary) => arm::sweep_smull_ternary(&lanes, q, &shape, frames),
                (Path::Neon, Kind::Int8) => arm::sweep_smull_bytes(&lanes, q, &shape, frames),
                (Path::Mmla, Kind::Int8) => arm::sweep_mmla(&lanes, q, &shape, frames),
                (_, Kind::Ternary) => arm::sweep_sdot_ternary(&lanes, q, &shape, frames),
                (_, Kind::Int8) => arm::sweep_sdot_bytes(&lanes, q, &shape, frames),
            }
        }
    }
}

/// Row-major `n x groups` scales as `[units][groups][16 rows]`; a row past
/// the end has scale 0.
fn unit_scales(scales: &[f32], n: usize, groups: usize) -> Vec<f32> {
    let units = n.div_ceil(UNIT);
    let mut out = vec![0.0f32; units * groups * UNIT];
    for row in 0..n {
        for g in 0..groups {
            out[((row / UNIT) * groups + g) * UNIT + row % UNIT] = scales[row * groups + g];
        }
    }
    out
}

/// The sizes of a layer that a tile needs.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
struct Shape {
    groups: usize,
    /// Inputs of a group, divided by 16.
    spans: usize,
    k: usize,
    n: usize,
}

/// The `B` vectors of rows that a tile computes side by side.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
struct Lanes<const B: usize> {
    /// The stored weights of each vector's rows.
    bytes: [*const u8; B],
    /// The weight scales of each vector: `UNIT` values on for each group.
    scales: [*const f32; B],
    /// Where frame 0 of each vector goes.
    out: [*mut f32; B],
    /// Rows of each vector that are inside the layer.
    live: [usize; B],
}

/// The vector operations of one path. Every method is `inline(always)`, so
/// it compiles with the target features of the sweep that calls it.
#[cfg(target_arch = "x86_64")]
trait Simd {
    type Int: Copy;
    type Float: Copy;
    unsafe fn zero() -> Self::Float;
    /// The weights of this vector's rows for 16 inputs, as four vectors of
    /// 4 inputs each: 2-bit codes unpacked (`TERNARY`), or bytes as stored.
    unsafe fn span<const TERNARY: bool>(at: *const u8) -> [Self::Int; 4];
    /// One 4-byte value in every lane.
    unsafe fn all(at: *const i32) -> Self::Int;
    /// `sum + w . x` in each 4-byte lane.
    unsafe fn dot(sum: Self::Int, w: Self::Int, x: Self::Int) -> Self::Int;
    unsafe fn scales(at: *const f32) -> Self::Float;
    unsafe fn all_f32(at: *const f32) -> Self::Float;
    unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float;
    /// `sum as f32 * scale + total`, with one rounding.
    unsafe fn add_scaled(sum: Self::Int, scale: Self::Float, total: Self::Float) -> Self::Float;
    /// The first `live` rows of `total` to `at`.
    unsafe fn store(at: *mut f32, total: Self::Float, live: usize);
}

/// `B` vectors of rows by the `F` frames from `frame`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(clippy::needless_range_loop)]
unsafe fn tile<S: Simd, const TERNARY: bool, const B: usize, const F: usize>(
    lanes: &Lanes<B>,
    q: &Quantized,
    shape: &Shape,
    frame: usize,
) {
    let Shape {
        groups,
        spans,
        k,
        n,
    } = *shape;
    let x = q.values.as_ptr().add(frame * k);
    let x_scales = q.scales.as_ptr().add(frame * groups);
    let starts = q.starts.as_ptr().add(frame * groups);
    // Bytes of a vector's weights for 16 inputs.
    let stride = if TERNARY { STEP } else { 4 * STEP };
    let mut totals = [[S::zero(); F]; B];
    for g in 0..groups {
        let mut sums = [[S::all(starts.add(g)); F]; B];
        for f in 1..F {
            let start = S::all(starts.add(f * groups + g));
            for b in 0..B {
                sums[b][f] = start;
            }
        }
        for span in g * spans..(g + 1) * spans {
            let mut w = [S::span::<TERNARY>(lanes.bytes[0].add(span * stride)); B];
            for b in 1..B {
                w[b] = S::span::<TERNARY>(lanes.bytes[b].add(span * stride));
            }
            for s in 0..4 {
                for f in 0..F {
                    let value = S::all(x.add(f * k + span * 16 + s * 4).cast());
                    for b in 0..B {
                        sums[b][f] = S::dot(sums[b][f], w[b][s], value);
                    }
                }
            }
        }
        for b in 0..B {
            let weight_scale = S::scales(lanes.scales[b].add(g * UNIT));
            for f in 0..F {
                let scale = S::mul(weight_scale, S::all_f32(x_scales.add(f * groups + g)));
                totals[b][f] = S::add_scaled(sums[b][f], scale, totals[b][f]);
            }
        }
    }
    for b in 0..B {
        for f in 0..F {
            S::store(
                lanes.out[b].add((frame + f) * n),
                totals[b][f],
                lanes.live[b],
            );
        }
    }
}

/// `frames` for `B` vectors of rows: tiles of `$full` frames, then one
/// tile of the frames that remain.
#[cfg(target_arch = "x86_64")]
macro_rules! sweep {
    ($(#[$meta:meta])* $name:ident, $simd:ty, $ternary:literal, $vectors:literal, $full:literal, [$($rest:literal)*]) => {
        $(#[$meta])*
        unsafe fn $name(
            lanes: &Lanes<$vectors>,
            q: &Quantized,
            shape: &Shape,
            frames: std::ops::Range<usize>,
        ) {
            let mut at = frames.start;
            while at + $full <= frames.end {
                tile::<$simd, $ternary, $vectors, $full>(lanes, q, shape, at);
                at += $full;
            }
            match frames.end - at {
                $($rest => tile::<$simd, $ternary, $vectors, $rest>(lanes, q, shape, at),)*
                _ => {}
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! sweeps {
    ($(#[$meta:meta])* $simd:ty, $vectors:literal, $full:literal, $rest:tt, $ternary:ident, $bytes:ident) => {
        sweep!($(#[$meta])* $ternary, $simd, true, $vectors, $full, $rest);
        sweep!($(#[$meta])* $bytes, $simd, false, $vectors, $full, $rest);
    };
}

// AVX-512: 2 x 6 sums, 2 x 4 weight vectors, the mask, and 1 frame value.
#[cfg(target_arch = "x86_64")]
sweeps!(
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    Vnni512, 2, 6, [1 2 3 4 5], sweep_512_ternary_2, sweep_512_bytes_2
);
#[cfg(target_arch = "x86_64")]
sweeps!(
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    Vnni512, 1, 6, [1 2 3 4 5], sweep_512_ternary_1, sweep_512_bytes_1
);
// 256 bits, 16 registers: 4 sums, 4 weight vectors, the mask, 1 frame value.
#[cfg(target_arch = "x86_64")]
sweeps!(
    #[target_feature(enable = "avx2,fma,avxvnni")]
    Vnni256Vex, 1, 4, [1 2 3], sweep_vex_ternary, sweep_vex_bytes
);
#[cfg(target_arch = "x86_64")]
sweeps!(
    #[target_feature(enable = "avx2,fma,avx512vl,avx512vnni")]
    Vnni256Evex, 1, 4, [1 2 3], sweep_evex_ternary, sweep_evex_bytes
);
#[cfg(target_arch = "x86_64")]
sweep!(
    #[target_feature(enable = "avx2,fma")]
    sweep_avx2_ternary, Avx2Codes, true, 1, 4, [1 2 3]
);
#[cfg(target_arch = "x86_64")]
sweep!(
    #[target_feature(enable = "avx2,fma")]
    sweep_avx2_bytes, Avx2Signed, false, 1, 4, [1 2 3]
);

#[cfg(target_arch = "x86_64")]
struct Vnni512;

#[cfg(target_arch = "x86_64")]
impl Simd for Vnni512 {
    type Int = std::arch::x86_64::__m512i;
    type Float = std::arch::x86_64::__m512;
    #[inline(always)]
    unsafe fn zero() -> Self::Float {
        std::arch::x86_64::_mm512_setzero_ps()
    }
    #[inline(always)]
    unsafe fn span<const TERNARY: bool>(at: *const u8) -> [Self::Int; 4] {
        use std::arch::x86_64::*;
        if TERNARY {
            let codes = _mm512_loadu_si512(at.cast());
            let two_bits = _mm512_set1_epi8(3);
            // A 16-bit shift is safe: the mask keeps the low two bits of each byte.
            [
                _mm512_and_si512(codes, two_bits),
                _mm512_and_si512(_mm512_srli_epi16::<2>(codes), two_bits),
                _mm512_and_si512(_mm512_srli_epi16::<4>(codes), two_bits),
                _mm512_and_si512(_mm512_srli_epi16::<6>(codes), two_bits),
            ]
        } else {
            [
                _mm512_loadu_si512(at.cast()),
                _mm512_loadu_si512(at.add(STEP).cast()),
                _mm512_loadu_si512(at.add(2 * STEP).cast()),
                _mm512_loadu_si512(at.add(3 * STEP).cast()),
            ]
        }
    }
    #[inline(always)]
    unsafe fn all(at: *const i32) -> Self::Int {
        std::arch::x86_64::_mm512_set1_epi32(at.read_unaligned())
    }
    #[inline(always)]
    unsafe fn dot(sum: Self::Int, w: Self::Int, x: Self::Int) -> Self::Int {
        std::arch::x86_64::_mm512_dpbusd_epi32(sum, w, x)
    }
    #[inline(always)]
    unsafe fn scales(at: *const f32) -> Self::Float {
        std::arch::x86_64::_mm512_loadu_ps(at)
    }
    #[inline(always)]
    unsafe fn all_f32(at: *const f32) -> Self::Float {
        std::arch::x86_64::_mm512_set1_ps(at.read_unaligned())
    }
    #[inline(always)]
    unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float {
        std::arch::x86_64::_mm512_mul_ps(a, b)
    }
    #[inline(always)]
    unsafe fn add_scaled(sum: Self::Int, scale: Self::Float, total: Self::Float) -> Self::Float {
        use std::arch::x86_64::*;
        _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum), scale, total)
    }
    #[inline(always)]
    unsafe fn store(at: *mut f32, total: Self::Float, live: usize) {
        use std::arch::x86_64::*;
        if live == 16 {
            _mm512_storeu_ps(at, total);
        } else {
            _mm512_mask_storeu_ps(at, (1u32 << live) as u16 - 1, total);
        }
    }
}

/// The operations of a 256-bit path: 8 rows, the low or the high half of a
/// unit. `$dot` is the one that differs.
#[cfg(target_arch = "x86_64")]
macro_rules! simd256 {
    ($name:ident, |$sum:ident, $w:ident, $x:ident| $dot:expr) => {
        #[cfg(target_arch = "x86_64")]
        struct $name;

        #[cfg(target_arch = "x86_64")]
        impl Simd for $name {
            type Int = std::arch::x86_64::__m256i;
            type Float = std::arch::x86_64::__m256;
            #[inline(always)]
            unsafe fn zero() -> Self::Float {
                std::arch::x86_64::_mm256_setzero_ps()
            }
            #[inline(always)]
            unsafe fn span<const TERNARY: bool>(at: *const u8) -> [Self::Int; 4] {
                use std::arch::x86_64::*;
                if TERNARY {
                    let codes = _mm256_loadu_si256(at.cast());
                    let two_bits = _mm256_set1_epi8(3);
                    // A 16-bit shift is safe: the mask keeps the low two bits of each byte.
                    [
                        _mm256_and_si256(codes, two_bits),
                        _mm256_and_si256(_mm256_srli_epi16::<2>(codes), two_bits),
                        _mm256_and_si256(_mm256_srli_epi16::<4>(codes), two_bits),
                        _mm256_and_si256(_mm256_srli_epi16::<6>(codes), two_bits),
                    ]
                } else {
                    [
                        _mm256_loadu_si256(at.cast()),
                        _mm256_loadu_si256(at.add(STEP).cast()),
                        _mm256_loadu_si256(at.add(2 * STEP).cast()),
                        _mm256_loadu_si256(at.add(3 * STEP).cast()),
                    ]
                }
            }
            #[inline(always)]
            unsafe fn all(at: *const i32) -> Self::Int {
                std::arch::x86_64::_mm256_set1_epi32(at.read_unaligned())
            }
            #[inline(always)]
            unsafe fn dot($sum: Self::Int, $w: Self::Int, $x: Self::Int) -> Self::Int {
                #[allow(unused_imports)]
                use std::arch::x86_64::*;
                $dot
            }
            #[inline(always)]
            unsafe fn scales(at: *const f32) -> Self::Float {
                std::arch::x86_64::_mm256_loadu_ps(at)
            }
            #[inline(always)]
            unsafe fn all_f32(at: *const f32) -> Self::Float {
                std::arch::x86_64::_mm256_set1_ps(at.read_unaligned())
            }
            #[inline(always)]
            unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float {
                std::arch::x86_64::_mm256_mul_ps(a, b)
            }
            #[inline(always)]
            unsafe fn add_scaled(
                sum: Self::Int,
                scale: Self::Float,
                total: Self::Float,
            ) -> Self::Float {
                use std::arch::x86_64::*;
                _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), scale, total)
            }
            #[inline(always)]
            unsafe fn store(at: *mut f32, total: Self::Float, live: usize) {
                use std::arch::x86_64::*;
                if live == 8 {
                    _mm256_storeu_ps(at, total);
                } else {
                    let mut rows = [0.0f32; 8];
                    _mm256_storeu_ps(rows.as_mut_ptr(), total);
                    std::ptr::copy_nonoverlapping(rows.as_ptr(), at, live);
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
simd256!(Vnni256Vex, |sum, w, x| _mm256_dpbusd_avx_epi32(sum, w, x));
#[cfg(target_arch = "x86_64")]
simd256!(Vnni256Evex, |sum, w, x| _mm256_dpbusd_epi32(sum, w, x));
// Codes are at most 2 and |x| at most 127: a pair sum fits in i16.
#[cfg(target_arch = "x86_64")]
simd256!(Avx2Codes, |sum, w, x| _mm256_add_epi32(
    sum,
    _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), _mm256_set1_epi16(1))
));
// The sign trick: |w| is unsigned, and x takes the sign of w. Both stay
// within +-127, so a pair sum fits in i16.
#[cfg(target_arch = "x86_64")]
simd256!(Avx2Signed, |sum, w, x| _mm256_add_epi32(
    sum,
    _mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w)),
        _mm256_set1_epi16(1)
    )
));

/// arm64: one unit (16 rows, 4 vectors of 4 rows) by 4 frames. The 16
/// sums in flight cover the latency of `sdot` and `smmla` on 4 pipes;
/// the first kernels keep 8. The float steps are those of the x86 paths,
/// so every path gives the same values.
#[cfg(target_arch = "aarch64")]
mod arm {
    use super::{Lanes, Quantized, Shape, STEP, UNIT};
    use std::arch::aarch64::*;

    /// Four rows (one 4-byte lane each) times the 4 frame bytes in lane `S`
    /// of `x`, added to `sum`.
    trait Dot {
        unsafe fn dot<const S: i32>(sum: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t;
    }

    /// `sdot`. Callers run with the `dotprod` feature on.
    struct Sdot;

    impl Dot for Sdot {
        #[inline(always)]
        unsafe fn dot<const S: i32>(sum: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
            vdotq_laneq_s32::<S>(sum, w, x)
        }
    }

    /// Armv8.0: widening products, then two pairwise adds. Both factors fit
    /// in i8, so a product fits in i16.
    struct Smull;

    impl Dot for Smull {
        #[inline(always)]
        unsafe fn dot<const S: i32>(sum: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
            let x = vreinterpretq_s8_s32(vdupq_laneq_s32::<S>(vreinterpretq_s32_s8(x)));
            let low = vpaddlq_s16(vmull_s8(vget_low_s8(w), vget_low_s8(x)));
            let high = vpaddlq_s16(vmull_high_s8(w, x));
            vaddq_s32(sum, vpaddq_s32(low, high))
        }
    }

    /// Step `S` of a 16-input span: inputs `4S..4S + 4` of the 16 rows, for
    /// every frame. Ternary codes come from `codes`; bytes load from `at`.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn step<D: Dot, const TERNARY: bool, const S: i32, const F: usize>(
        sums: &mut [[int32x4_t; 4]; F],
        codes: &[uint8x16_t; 4],
        at: *const u8,
        x: &[int8x16_t; F],
    ) {
        let three = vdupq_n_u8(3);
        let mut w = [vdupq_n_s8(0); 4];
        for j in 0..4 {
            w[j] = if TERNARY {
                vreinterpretq_s8_u8(match S {
                    0 => vandq_u8(codes[j], three),
                    1 => vandq_u8(vshrq_n_u8::<2>(codes[j]), three),
                    2 => vandq_u8(vshrq_n_u8::<4>(codes[j]), three),
                    _ => vshrq_n_u8::<6>(codes[j]),
                })
            } else {
                vld1q_s8(at.add(S as usize * STEP + j * 16).cast())
            };
        }
        for f in 0..F {
            for j in 0..4 {
                sums[f][j] = D::dot::<S>(sums[f][j], w[j], x[f]);
            }
        }
    }

    /// The first `live` rows of a unit to `at`.
    #[inline(always)]
    unsafe fn store(at: *mut f32, rows: &[float32x4_t; 4], live: usize) {
        if live == UNIT {
            for (j, v) in rows.iter().enumerate() {
                vst1q_f32(at.add(4 * j), *v);
            }
        } else {
            let mut all = [0.0f32; UNIT];
            for (j, v) in rows.iter().enumerate() {
                vst1q_f32(all.as_mut_ptr().add(4 * j), *v);
            }
            std::ptr::copy_nonoverlapping(all.as_ptr(), at, live);
        }
    }

    /// One unit by the `F` frames from `frame`.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn tile<D: Dot, const TERNARY: bool, const F: usize>(
        lanes: &Lanes<1>,
        q: &Quantized,
        shape: &Shape,
        frame: usize,
    ) {
        let Shape {
            groups,
            spans,
            k,
            n,
        } = *shape;
        let x = q.values.as_ptr().add(frame * k);
        let x_scales = q.scales.as_ptr().add(frame * groups);
        let starts = q.starts.as_ptr().add(frame * groups);
        let stride = if TERNARY { STEP } else { 4 * STEP };
        let mut totals = [[vdupq_n_f32(0.0); 4]; F];
        for g in 0..groups {
            let mut sums = [[vdupq_n_s32(0); 4]; F];
            for f in 0..F {
                sums[f] = [vdupq_n_s32(*starts.add(f * groups + g)); 4];
            }
            for span in g * spans..(g + 1) * spans {
                let at = lanes.bytes[0].add(span * stride);
                let mut xs = [vdupq_n_s8(0); F];
                for f in 0..F {
                    xs[f] = vld1q_s8(x.add(f * k + span * 16));
                }
                let mut codes = [vdupq_n_u8(0); 4];
                if TERNARY {
                    for (j, c) in codes.iter_mut().enumerate() {
                        *c = vld1q_u8(at.add(j * 16));
                    }
                }
                step::<D, TERNARY, 0, F>(&mut sums, &codes, at, &xs);
                step::<D, TERNARY, 1, F>(&mut sums, &codes, at, &xs);
                step::<D, TERNARY, 2, F>(&mut sums, &codes, at, &xs);
                step::<D, TERNARY, 3, F>(&mut sums, &codes, at, &xs);
            }
            let weight_scales = lanes.scales[0].add(g * UNIT);
            for f in 0..F {
                let x_scale = *x_scales.add(f * groups + g);
                for j in 0..4 {
                    let scale = vmulq_n_f32(vld1q_f32(weight_scales.add(4 * j)), x_scale);
                    totals[f][j] = vfmaq_f32(totals[f][j], vcvtq_f32_s32(sums[f][j]), scale);
                }
            }
        }
        for f in 0..F {
            store(lanes.out[0].add((frame + f) * n), &totals[f], lanes.live[0]);
        }
    }

    /// `frames`: tiles of 4, then one tile of the frames that remain.
    macro_rules! sweep {
        ($(#[$meta:meta])* $name:ident, $dot:ty, $ternary:literal) => {
            $(#[$meta])*
            pub(super) unsafe fn $name(
                lanes: &Lanes<1>,
                q: &Quantized,
                shape: &Shape,
                frames: std::ops::Range<usize>,
            ) {
                let mut at = frames.start;
                while at + 4 <= frames.end {
                    tile::<$dot, $ternary, 4>(lanes, q, shape, at);
                    at += 4;
                }
                match frames.end - at {
                    1 => tile::<$dot, $ternary, 1>(lanes, q, shape, at),
                    2 => tile::<$dot, $ternary, 2>(lanes, q, shape, at),
                    3 => tile::<$dot, $ternary, 3>(lanes, q, shape, at),
                    _ => {}
                }
            }
        };
    }

    sweep!(
        #[target_feature(enable = "dotprod")]
        sweep_sdot_ternary,
        Sdot,
        true
    );
    sweep!(
        #[target_feature(enable = "dotprod")]
        sweep_sdot_bytes,
        Sdot,
        false
    );
    sweep!(sweep_smull_ternary, Smull, true);
    sweep!(sweep_smull_bytes, Smull, false);

    /// `sum + a * b^T` for 2 x 8 byte matrices `a` and `b` (`smmla`), as
    /// `[a0 b0, a0 b1, a1 b0, a1 b1]`. Inline assembly: the intrinsic is
    /// not stable yet. Callers run with the `i8mm` feature on.
    #[inline(always)]
    unsafe fn smmla(sum: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut sum = sum;
        std::arch::asm!(
            "smmla {sum:v}.4s, {a:v}.16b, {b:v}.16b",
            sum = inout(vreg) sum,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
        sum
    }

    /// One int8 unit by `frames` frames (at most `2 * P`), in `P` frame
    /// pairs. Weights are `[k / 8][16 rows][8]`, so vector `i` holds rows
    /// `2i` and `2i + 1`; each sum holds `[f0 r0, f0 r1, f1 r0, f1 r1]`. A
    /// missing frame of the last pair reads as zeros and is not stored.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn mmla_tile<const P: usize>(
        lanes: &Lanes<1>,
        q: &Quantized,
        shape: &Shape,
        frame: usize,
        frames: usize,
    ) {
        let Shape {
            groups,
            spans,
            k,
            n,
        } = *shape;
        let zero = vdupq_n_s8(0);
        let start = |f: usize, g: usize| {
            if f < frames {
                q.starts[(frame + f) * groups + g]
            } else {
                0
            }
        };
        let x_scale = |f: usize, g: usize| {
            if f < frames {
                q.scales[(frame + f) * groups + g]
            } else {
                0.0
            }
        };
        let load = |f: usize, span: usize| {
            vreinterpretq_s64_s8(if f < frames {
                vld1q_s8(q.values.as_ptr().add((frame + f) * k + span * 16))
            } else {
                zero
            })
        };
        let mut totals = [[vdupq_n_f32(0.0); 8]; P];
        for g in 0..groups {
            let mut sums = [[vdupq_n_s32(0); 8]; P];
            for p in 0..P {
                let both =
                    vcombine_s32(vdup_n_s32(start(2 * p, g)), vdup_n_s32(start(2 * p + 1, g)));
                sums[p] = [both; 8];
            }
            for span in g * spans..(g + 1) * spans {
                // Pair p, half h: inputs 8h..8h + 8 of frames 2p and 2p + 1.
                let mut xs = [[zero; 2]; P];
                for p in 0..P {
                    let (a, b) = (load(2 * p, span), load(2 * p + 1, span));
                    xs[p] = [
                        vreinterpretq_s8_s64(vzip1q_s64(a, b)),
                        vreinterpretq_s8_s64(vzip2q_s64(a, b)),
                    ];
                }
                for half in 0..2 {
                    let at = lanes.bytes[0].add((span * 2 + half) * 2 * STEP);
                    let mut w = [zero; 8];
                    for (i, w) in w.iter_mut().enumerate() {
                        *w = vld1q_s8(at.add(16 * i).cast());
                    }
                    for p in 0..P {
                        for i in 0..8 {
                            sums[p][i] = smmla(sums[p][i], xs[p][half], w[i]);
                        }
                    }
                }
            }
            let weight_scales = lanes.scales[0].add(g * UNIT);
            for p in 0..P {
                let x_scales = vcombine_f32(
                    vdup_n_f32(x_scale(2 * p, g)),
                    vdup_n_f32(x_scale(2 * p + 1, g)),
                );
                for i in 0..8 {
                    let pair = vld1_f32(weight_scales.add(2 * i));
                    let scale = vmulq_f32(vcombine_f32(pair, pair), x_scales);
                    totals[p][i] = vfmaq_f32(totals[p][i], vcvtq_f32_s32(sums[p][i]), scale);
                }
            }
        }
        for p in 0..P {
            for f in [2 * p, 2 * p + 1] {
                if f >= frames {
                    continue;
                }
                let mut rows = [0.0f32; UNIT];
                for i in 0..8 {
                    let two = if f % 2 == 0 {
                        vget_low_f32(totals[p][i])
                    } else {
                        vget_high_f32(totals[p][i])
                    };
                    vst1_f32(rows.as_mut_ptr().add(2 * i), two);
                }
                std::ptr::copy_nonoverlapping(
                    rows.as_ptr(),
                    lanes.out[0].add((frame + f) * n),
                    lanes.live[0],
                );
            }
        }
    }

    #[target_feature(enable = "dotprod,i8mm")]
    pub(super) unsafe fn sweep_mmla(
        lanes: &Lanes<1>,
        q: &Quantized,
        shape: &Shape,
        frames: std::ops::Range<usize>,
    ) {
        let mut at = frames.start;
        while at + 4 <= frames.end {
            mmla_tile::<2>(lanes, q, shape, at, 4);
            at += 4;
        }
        match frames.end - at {
            0 => {}
            left @ (1 | 2) => mmla_tile::<1>(lanes, q, shape, at, left),
            left => mmla_tile::<2>(lanes, q, shape, at, left),
        }
    }
}
