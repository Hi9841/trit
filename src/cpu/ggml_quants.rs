//! ggml's block quantizations, expanded to f32 at load as ggml's reference
//! `dequantize_row_*` functions (`ggml-quants.c`) expand them, in the same
//! float order. Trit has no kernel for these types: a GGUF file's Q4_0,
//! Q4_1, Q5_0, Q5_1, Q4_K, Q5_K and Q6_K tensors become f32 here and are then
//! held as `Precision::Q8_0` under `Precision::Auto` (`Tensors::precision_of`).
//! Q8_0 has its own path (`Tensors::q8_0`).

use super::tensors::f16_to_f32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q4K,
    Q5K,
    Q6K,
}

/// One ggml type id: its dtype name, values per block, and bytes per block.
/// F32, F16, BF16, and Q8_0 have no [`BlockType`] (`Q8_0` expands on its own path).
struct GgmlType {
    id: u32,
    name: &'static str,
    values: usize,
    bytes: usize,
    block: Option<BlockType>,
}

/// Bytes of one ggml Q8_0 block: an f16 scale and 32 int8 codes.
pub const Q8_0_BYTES: usize = 34;

const fn kind(
    id: u32,
    name: &'static str,
    values: usize,
    bytes: usize,
    block: Option<BlockType>,
) -> GgmlType {
    GgmlType {
        id,
        name,
        values,
        bytes,
        block,
    }
}

/// Ids in the order the GGUF reader lists them.
const TYPES: &[GgmlType] = &[
    kind(0, "F32", 1, 4, None),
    kind(1, "F16", 1, 2, None),
    kind(30, "BF16", 1, 2, None),
    kind(8, "Q8_0", 32, Q8_0_BYTES, None),
    kind(2, "Q4_0", 32, 18, Some(BlockType::Q4_0)),
    kind(3, "Q4_1", 32, 20, Some(BlockType::Q4_1)),
    kind(6, "Q5_0", 32, 22, Some(BlockType::Q5_0)),
    kind(7, "Q5_1", 32, 24, Some(BlockType::Q5_1)),
    kind(12, "Q4_K", 256, 144, Some(BlockType::Q4K)),
    kind(13, "Q5_K", 256, 176, Some(BlockType::Q5K)),
    kind(14, "Q6_K", 256, 210, Some(BlockType::Q6K)),
];

/// The dtype name of ggml type `id`, and the bytes of `count` values: `None`
/// when `count` overflows or is not a whole number of blocks. `None` for an
/// unknown id.
pub fn layout(id: u32, count: u64) -> Option<(&'static str, Option<u64>)> {
    let kind = TYPES.iter().find(|kind| kind.id == id)?;
    let (values, bytes) = (kind.values as u64, kind.bytes as u64);
    if !count.is_multiple_of(values) {
        return None;
    }
    Some((kind.name, (count / values).checked_mul(bytes)))
}

/// `F32, F16, BF16, Q8_0, ... and Q6_K`, the list the GGUF reader prints.
pub fn type_list() -> String {
    let names: Vec<&str> = TYPES.iter().map(|kind| kind.name).collect();
    let Some((last, head)) = names.split_last() else {
        return String::new();
    };
    format!("{}, and {last}", head.join(", "))
}

impl BlockType {
    pub fn from_name(name: &str) -> Option<Self> {
        TYPES
            .iter()
            .find(|kind| kind.name == name)
            .and_then(|kind| kind.block)
    }

    /// Values per block, and bytes per block.
    pub fn block(self) -> (usize, usize) {
        let kind = TYPES
            .iter()
            .find(|kind| kind.block == Some(self))
            .expect("every block type is in TYPES");
        (kind.values, kind.bytes)
    }

    pub fn expand(self, bytes: &[u8]) -> Vec<f32> {
        let (values, size) = self.block();
        let mut out = vec![0.0f32; bytes.len() / size * values];
        for (block, y) in bytes.chunks_exact(size).zip(out.chunks_exact_mut(values)) {
            match self {
                Self::Q4_0 => q4_0(block, y),
                Self::Q4_1 => q4_1(block, y),
                Self::Q5_0 => q5_0(block, y),
                Self::Q5_1 => q5_1(block, y),
                Self::Q4K => q4_k(block, y),
                Self::Q5K => q5_k(block, y),
                Self::Q6K => q6_k(block, y),
            }
        }
        out
    }
}

fn half(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

fn q4_0(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    let qs = &b[2..18];
    for j in 0..16 {
        y[j] = f32::from(i16::from(qs[j] & 0x0F) - 8) * d;
        y[j + 16] = f32::from(i16::from(qs[j] >> 4) - 8) * d;
    }
}

fn q4_1(b: &[u8], y: &mut [f32]) {
    let (d, m) = (half(b, 0), half(b, 2));
    let qs = &b[4..20];
    for j in 0..16 {
        y[j] = f32::from(qs[j] & 0x0F) * d + m;
        y[j + 16] = f32::from(qs[j] >> 4) * d + m;
    }
}

fn q5_0(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    let qs = &b[6..22];
    for j in 0..16 {
        let xh0 = (((qh >> j) << 4) & 0x10) as u8;
        let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
        y[j] = (i32::from((qs[j] & 0x0F) | xh0) - 16) as f32 * d;
        y[j + 16] = (i32::from((qs[j] >> 4) | xh1) - 16) as f32 * d;
    }
}

fn q5_1(b: &[u8], y: &mut [f32]) {
    let (d, m) = (half(b, 0), half(b, 2));
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let qs = &b[8..24];
    for j in 0..16 {
        let xh0 = (((qh >> j) << 4) & 0x10) as u8;
        let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
        y[j] = f32::from((qs[j] & 0x0F) | xh0) * d + m;
        y[j + 16] = f32::from((qs[j] >> 4) | xh1) * d + m;
    }
}

/// The 6-bit scale and min `j` of a K-quant block's 12 scale bytes.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn q4_k(b: &[u8], y: &mut [f32]) {
    let (d, min) = (half(b, 0), half(b, 2));
    let scales = &b[4..16];
    let mut q = &b[16..144];
    let mut out = 0;
    for is in (0..8).step_by(2) {
        let (sc, m) = scale_min_k4(is, scales);
        let (d1, m1) = (d * f32::from(sc), min * f32::from(m));
        let (sc, m) = scale_min_k4(is + 1, scales);
        let (d2, m2) = (d * f32::from(sc), min * f32::from(m));
        for l in 0..32 {
            y[out + l] = d1 * f32::from(q[l] & 0xF) - m1;
            y[out + 32 + l] = d2 * f32::from(q[l] >> 4) - m2;
        }
        out += 64;
        q = &q[32..];
    }
}

fn q5_k(b: &[u8], y: &mut [f32]) {
    let (d, min) = (half(b, 0), half(b, 2));
    let scales = &b[4..16];
    let qh = &b[16..48];
    let mut ql = &b[48..176];
    let (mut u1, mut u2) = (1u8, 2u8);
    let mut out = 0;
    for is in (0..8).step_by(2) {
        let (sc, m) = scale_min_k4(is, scales);
        let (d1, m1) = (d * f32::from(sc), min * f32::from(m));
        let (sc, m) = scale_min_k4(is + 1, scales);
        let (d2, m2) = (d * f32::from(sc), min * f32::from(m));
        for l in 0..32 {
            let high1 = if qh[l] & u1 != 0 { 16 } else { 0 };
            let high2 = if qh[l] & u2 != 0 { 16 } else { 0 };
            y[out + l] = d1 * f32::from((ql[l] & 0xF) + high1) - m1;
            y[out + 32 + l] = d2 * f32::from((ql[l] >> 4) + high2) - m2;
        }
        out += 64;
        ql = &ql[32..];
        u1 <<= 2;
        u2 <<= 2;
    }
}

fn q6_k(b: &[u8], y: &mut [f32]) {
    let d = half(b, 208);
    let (mut ql, mut qh) = (&b[0..128], &b[128..192]);
    let mut sc: Vec<i8> = b[192..208].iter().map(|v| *v as i8).collect();
    let mut out = 0;
    for _ in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
            y[out + l] = d * f32::from(sc[is]) * f32::from(q1);
            y[out + l + 32] = d * f32::from(sc[is + 2]) * f32::from(q2);
            y[out + l + 64] = d * f32::from(sc[is + 4]) * f32::from(q3);
            y[out + l + 96] = d * f32::from(sc[is + 6]) * f32::from(q4);
        }
        out += 128;
        ql = &ql[64..];
        qh = &qh[32..];
        sc.drain(..8);
    }
}
