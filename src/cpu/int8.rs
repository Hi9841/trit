//! Int8 weights with one scale per row and 128-value group, quantized at
//! load from f32 or f16. Activations go through the same i8 quantizer as
//! the ternary layers (`kernels::Quantized`), so a product is i8 x i8 dot
//! products in i32, one f32 scale pair per group.
//!
//! VNNI (AVX-512, or 256-bit AVX-VNNI on Intel client CPUs without
//! AVX-512): `vpdpbusd` multiplies unsigned by signed bytes, so each weight
//! byte is stored flipped, as `w + 128`, and `128 * sum(x)` per group (the
//! quantizer computes the sums) comes off again. AVX2: the sign
//! trick, `maddubs(|w|, sign(x, w))`; weights and activations stay within
//! +-127, so a pair sum fits in i16.

use super::buf::Buf;
use super::kernels::{Isa, Quantized, GROUP};
use super::team;

#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: tasks write disjoint output columns; the buffer outlives the step.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// The product kernel. The activation quantizer follows `Isa`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Path {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// 8 outputs per block with 256-bit `vpdpbusd`: the VEX form (AVX-VNNI),
    /// or the EVEX form where AVX-512 VNNI runs it (`TRIT_ISA=avxvnni`, to
    /// check this path on such a CPU).
    #[cfg(target_arch = "x86_64")]
    Vnni256,
    #[cfg(target_arch = "x86_64")]
    Vnni512,
}

impl Path {
    fn detect(isa: Isa) -> Self {
        match isa {
            Isa::Scalar => Self::Scalar,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 if is_x86_feature_detected!("avxvnni") => Self::Vnni256,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => Self::Avx2,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni if std::env::var("TRIT_ISA").as_deref() == Ok("avxvnni") => {
                Self::Vnni256
            }
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni => Self::Vnni512,
        }
    }

    /// Outputs per packed block.
    fn rows(self) -> usize {
        match self {
            Self::Scalar => 1,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 | Self::Vnni256 => 8,
            #[cfg(target_arch = "x86_64")]
            Self::Vnni512 => 16,
        }
    }

    fn flipped(self) -> bool {
        #[cfg(target_arch = "x86_64")]
        return matches!(self, Self::Vnni256 | Self::Vnni512);
        #[cfg(not(target_arch = "x86_64"))]
        false
    }
}

/// One 8-output block of `Path::Vnni256` over every frame, 4 frames per
/// tile, with `$dpbusd` the VEX or the EVEX instruction (the same result).
#[cfg(target_arch = "x86_64")]
macro_rules! vnni256_task {
    ($self:ident, $q:ident, $block:ident, $out:ident, $dpbusd:ident) => {{
        use std::arch::x86_64::*;
        let (k, k4s, groups) = ($self.k, $self.k / 4, $self.k / GROUP);
        let mut frame0 = 0;
        while frame0 < $q.m {
            let frames = ($q.m - frame0).min(4);
            let mut acc = [_mm256_setzero_ps(); 4];
            for g in 0..groups {
                let mut dots = [_mm256_setzero_si256(); 4];
                for k4 in g * (GROUP / 4)..(g + 1) * (GROUP / 4) {
                    let w = _mm256_loadu_si256(
                        $self.weights.as_ptr().add(($block * k4s + k4) * 32).cast(),
                    );
                    for f in 0..frames {
                        let src = $q.values.as_ptr().add((frame0 + f) * k + k4 * 4);
                        let x = _mm256_set1_epi32(std::ptr::read_unaligned(src.cast::<i32>()));
                        dots[f] = $dpbusd(dots[f], w, x);
                    }
                }
                let wscale = _mm256_loadu_ps($self.scales.as_ptr().add(($block * groups + g) * 8));
                for f in 0..frames {
                    let index = (frame0 + f) * groups + g;
                    // (w + 128) . x = w . x + 128 * sum(x)
                    let corrected =
                        _mm256_sub_epi32(dots[f], _mm256_set1_epi32(128 * $q.sums[index]));
                    let scale = _mm256_mul_ps(wscale, _mm256_set1_ps($q.scales[index]));
                    acc[f] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(corrected), scale, acc[f]);
                }
            }
            for f in 0..frames {
                _mm256_storeu_ps($out.0.add((frame0 + f) * $self.n + $block * 8), acc[f]);
            }
            frame0 += frames;
        }
    }};
}

/// The int8 product path of this CPU, for `cpu_path`.
pub fn path_name() -> &'static str {
    match Path::detect(Isa::detect()) {
        Path::Scalar => "scalar",
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => "avx2",
        #[cfg(target_arch = "x86_64")]
        Path::Vnni256 => "vnni256",
        #[cfg(target_arch = "x86_64")]
        Path::Vnni512 => "vnni512",
    }
}

pub struct Int8 {
    isa: Isa,
    path: Path,
    pub n: usize,
    pub k: usize,
    /// SIMD: `[n / rows][k / 4][rows][4]` bytes, so one load holds four
    /// consecutive inputs of `rows` outputs. Scalar: row-major `n x k`.
    weights: Vec<i8>,
    /// `[n / rows][k / GROUP][rows]`.
    scales: Vec<f32>,
}

impl Int8 {
    /// Whether a `n x k` layer fits the kernels: whole groups and whole tiles.
    pub fn fits(n: usize, k: usize) -> bool {
        k.is_multiple_of(GROUP) && n.is_multiple_of(32)
    }

    /// Quantizes row-major `n x k` weights; `row(r, out)` writes row `r` as f32.
    pub fn quantize(n: usize, k: usize, row: impl Fn(usize, &mut [f32]) + Sync) -> Self {
        let isa = Isa::detect();
        let path = Path::detect(isa);
        let rows = path.rows();
        let groups = k / GROUP;
        let mut weights = vec![0i8; n * k];
        let mut scales = vec![0.0f32; n * groups];
        let (all_w, all_s) = (
            team::SendPtr::new(&mut weights),
            team::SendPtr::new(&mut scales),
        );
        team::for_each(n / rows, |block| {
            let mut values = vec![0.0f32; k];
            for r in 0..rows {
                let out_row = block * rows + r;
                row(out_row, &mut values);
                for g in 0..groups {
                    let span = &values[g * GROUP..(g + 1) * GROUP];
                    let max = span.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
                    // SAFETY: each block belongs to one task; the buffers outlive the step.
                    unsafe {
                        *all_s
                            .slice_mut((block * groups + g) * rows + r, 1)
                            .as_mut_ptr() = scale
                    };
                    for (i, v) in span.iter().enumerate() {
                        let code = (v / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
                        // VNNI reads the byte as unsigned `w + 128`.
                        let code = if path.flipped() { code ^ i8::MIN } else { code };
                        let kk = g * GROUP + i;
                        let at = if path == Path::Scalar {
                            out_row * k + kk
                        } else {
                            ((block * (k / 4) + kk / 4) * rows + r) * 4 + kk % 4
                        };
                        // SAFETY: as above.
                        unsafe { *all_w.slice_mut(at, 1).as_mut_ptr() = code };
                    }
                }
            }
        });
        Self {
            isa,
            path,
            n,
            k,
            weights,
            scales,
        }
    }

    /// `x` is `m x k` f32; returns `m x n`.
    pub fn forward(&self, x: &[f32], m: usize) -> Buf {
        let mut out = Buf::scratch(m * self.n);
        let q = Quantized::new(self.isa, x, m, self.k);
        let ptr = OutPtr(out.as_mut_ptr());
        let rows = self.path.rows();
        match self.path {
            Path::Scalar => team::for_each(self.n, |row| self.scalar_row(&q, row, ptr)),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: Path::detect confirmed the CPU features.
            Path::Vnni512 => team::for_each(self.n / (2 * rows), |task| unsafe {
                self.vnni_task(&q, task, ptr)
            }),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: Path::detect confirmed the CPU features: AVX-VNNI, or
            // AVX-512 VNNI (with VL), which runs the EVEX form.
            Path::Vnni256 => team::for_each(self.n / rows, |block| unsafe {
                if is_x86_feature_detected!("avxvnni") {
                    self.vnni256_vex(&q, block, ptr)
                } else {
                    self.vnni256_evex(&q, block, ptr)
                }
            }),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: Isa::detect confirmed the CPU features.
            Path::Avx2 => team::for_each(self.n / rows, |block| unsafe {
                self.avx2_task(&q, block, ptr)
            }),
        }
        out
    }

    fn scalar_row(&self, q: &Quantized, row: usize, out: OutPtr) {
        let (k, groups) = (self.k, self.k / GROUP);
        for frame in 0..q.m {
            let mut total = 0.0f32;
            for g in 0..groups {
                let mut dot = 0i32;
                for kk in g * GROUP..(g + 1) * GROUP {
                    dot +=
                        i32::from(self.weights[row * k + kk]) * i32::from(q.values[frame * k + kk]);
                }
                total += dot as f32 * self.scales[row * groups + g] * q.scales[frame * groups + g];
            }
            // SAFETY: each output element belongs to one task.
            unsafe { *out.0.add(frame * self.n + row) = total };
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn vnni_task(&self, q: &Quantized, task: usize, out: OutPtr) {
        let mut frame = 0;
        while frame + 6 <= q.m {
            self.vnni_tile::<6>(q, task, frame, out);
            frame += 6;
        }
        match q.m - frame {
            1 => self.vnni_tile::<1>(q, task, frame, out),
            2 => self.vnni_tile::<2>(q, task, frame, out),
            3 => self.vnni_tile::<3>(q, task, frame, out),
            4 => self.vnni_tile::<4>(q, task, frame, out),
            5 => self.vnni_tile::<5>(q, task, frame, out),
            _ => {}
        }
    }

    /// 32 outputs (two 16-row blocks) by F frames.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn vnni_tile<const F: usize>(
        &self,
        q: &Quantized,
        task: usize,
        frame0: usize,
        out: OutPtr,
    ) {
        use std::arch::x86_64::*;
        let (k, k4s, groups) = (self.k, self.k / 4, self.k / GROUP);
        let blocks = [2 * task, 2 * task + 1];
        let mut acc = [[_mm512_setzero_ps(); F]; 2];
        for g in 0..groups {
            let mut dots = [[_mm512_setzero_si512(); F]; 2];
            for k4 in g * (GROUP / 4)..(g + 1) * (GROUP / 4) {
                let w0 = _mm512_loadu_si512(
                    self.weights
                        .as_ptr()
                        .add((blocks[0] * k4s + k4) * 64)
                        .cast(),
                );
                let w1 = _mm512_loadu_si512(
                    self.weights
                        .as_ptr()
                        .add((blocks[1] * k4s + k4) * 64)
                        .cast(),
                );
                for f in 0..F {
                    let src = q.values.as_ptr().add((frame0 + f) * k + k4 * 4);
                    let x = _mm512_set1_epi32(std::ptr::read_unaligned(src.cast::<i32>()));
                    dots[0][f] = _mm512_dpbusd_epi32(dots[0][f], w0, x);
                    dots[1][f] = _mm512_dpbusd_epi32(dots[1][f], w1, x);
                }
            }
            for (b, block) in blocks.iter().enumerate() {
                let wscale = _mm512_loadu_ps(self.scales.as_ptr().add((block * groups + g) * 16));
                for f in 0..F {
                    let index = (frame0 + f) * groups + g;
                    // (w + 128) . x = w . x + 128 * sum(x)
                    let corrected =
                        _mm512_sub_epi32(dots[b][f], _mm512_set1_epi32(128 * q.sums[index]));
                    let scale = _mm512_mul_ps(wscale, _mm512_set1_ps(q.scales[index]));
                    acc[b][f] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(corrected), scale, acc[b][f]);
                }
            }
        }
        for (b, block) in blocks.iter().enumerate() {
            for f in 0..F {
                _mm512_storeu_ps(out.0.add((frame0 + f) * self.n + block * 16), acc[b][f]);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,avxvnni")]
    unsafe fn vnni256_vex(&self, q: &Quantized, block: usize, out: OutPtr) {
        use std::arch::x86_64::_mm256_dpbusd_avx_epi32 as dpbusd;
        vnni256_task!(self, q, block, out, dpbusd);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,avx512vl,avx512vnni")]
    unsafe fn vnni256_evex(&self, q: &Quantized, block: usize, out: OutPtr) {
        use std::arch::x86_64::_mm256_dpbusd_epi32 as dpbusd;
        vnni256_task!(self, q, block, out, dpbusd);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_task(&self, q: &Quantized, block: usize, out: OutPtr) {
        let mut frame = 0;
        while frame + 4 <= q.m {
            self.avx2_tile::<4>(q, block, frame, out);
            frame += 4;
        }
        match q.m - frame {
            1 => self.avx2_tile::<1>(q, block, frame, out),
            2 => self.avx2_tile::<2>(q, block, frame, out),
            3 => self.avx2_tile::<3>(q, block, frame, out),
            _ => {}
        }
    }

    /// 8 outputs by F frames.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn avx2_tile<const F: usize>(
        &self,
        q: &Quantized,
        block: usize,
        frame0: usize,
        out: OutPtr,
    ) {
        use std::arch::x86_64::*;
        let (k, k4s, groups) = (self.k, self.k / 4, self.k / GROUP);
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); F];
        for g in 0..groups {
            let mut dots = [_mm256_setzero_si256(); F];
            for k4 in g * (GROUP / 4)..(g + 1) * (GROUP / 4) {
                let w =
                    _mm256_loadu_si256(self.weights.as_ptr().add((block * k4s + k4) * 32).cast());
                let magnitude = _mm256_abs_epi8(w);
                for f in 0..F {
                    let src = q.values.as_ptr().add((frame0 + f) * k + k4 * 4);
                    let x = _mm256_set1_epi32(std::ptr::read_unaligned(src.cast::<i32>()));
                    let pairs = _mm256_maddubs_epi16(magnitude, _mm256_sign_epi8(x, w));
                    dots[f] = _mm256_add_epi32(dots[f], _mm256_madd_epi16(pairs, ones));
                }
            }
            let wscale = _mm256_loadu_ps(self.scales.as_ptr().add((block * groups + g) * 8));
            for f in 0..F {
                let scale =
                    _mm256_mul_ps(wscale, _mm256_set1_ps(q.scales[(frame0 + f) * groups + g]));
                acc[f] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dots[f]), scale, acc[f]);
            }
        }
        for f in 0..F {
            _mm256_storeu_ps(out.0.add((frame0 + f) * self.n + block * 8), acc[f]);
        }
    }
}
