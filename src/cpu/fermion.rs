//! The weight records of a Fermion container (`fermion-five-value-parakeet-v1`,
//! Fermion Research's Phonon-2), expanded to f32 as the publisher's
//! `fermion_container.py` expands them.
//!
//! - `five_value`, an `[o, i]` matrix: per row `(i + 4) / 5` bytes of
//!   base-3 codes, 5 per byte, lowest digit first (code 1 is zero, the sign
//!   is `code - 1`); then one bit per nonzero of the whole matrix, row-major
//!   and low bit first (1 picks the row's high magnitude); then the low and
//!   the high magnitude of each row as f16.
//! - `int6` and `int8`: the codes of all `o * cols` values, then an f16
//!   scale per row. int6 packs 4 values in 3 bytes, 6 bits each from the
//!   low bit, stored plus 32.

use super::tensors::f16_to_f32;

/// One record of a container: its tensor name, kind (`dtype`), rows and
/// values per row, and byte range in the file.
pub struct Record {
    pub name: String,
    pub kind: String,
    pub matrix: (usize, usize),
    pub start: usize,
    pub end: usize,
}

/// The safetensors dtype of a record kind. `fp16` is stored as F16. The
/// other three keep their name and [`expand`] reads them. `None` is a kind
/// Trit does not read.
pub fn dtype(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "fp16" => "F16",
        "five_value" => "five_value",
        "int6" => "int6",
        "int8" => "int8",
        _ => return None,
    })
}

/// Record `bytes` of `kind` as `rows x cols` f32 values.
pub fn expand(kind: &str, bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    // A record has values: with no columns, the loops below have no end
    // that the record's length sets.
    if rows == 0 || cols == 0 || rows.checked_mul(cols).is_none() {
        return Err(format!("a Fermion record has the shape {rows}x{cols}"));
    }
    match kind {
        "five_value" => five_value(bytes, rows, cols),
        "int6" => int_n(bytes, rows, cols, 6),
        "int8" => int_n(bytes, rows, cols, 8),
        other => Err(format!("Fermion record kind {other} is not read")),
    }
}

fn halves(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
}

fn five_value(bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    let row_bytes = cols.div_ceil(5);
    let codes_len = rows * row_bytes;
    let codes = bytes
        .get(..codes_len)
        .ok_or("a five_value record ends in its codes")?;
    // Digit k of each byte, lowest first.
    let digit =
        |row: usize, col: usize| (codes[row * row_bytes + col / 5] / 3u8.pow((col % 5) as u32)) % 3;
    let nonzero = (0..rows)
        .flat_map(|row| (0..cols).map(move |col| (row, col)))
        .filter(|&(row, col)| digit(row, col) != 1)
        .count();
    let mask_len = nonzero.div_ceil(8);
    if bytes.len() != codes_len + mask_len + 4 * rows {
        return Err("a five_value record has the wrong length".into());
    }
    let mask = &bytes[codes_len..codes_len + mask_len];
    let magnitudes = &bytes[codes_len + mask_len..];
    let low: Vec<f32> = halves(&magnitudes[..2 * rows]).collect();
    let high: Vec<f32> = halves(&magnitudes[2 * rows..]).collect();
    let mut values = Vec::with_capacity(rows * cols);
    let mut bit = 0;
    for row in 0..rows {
        for col in 0..cols {
            let code = digit(row, col);
            if code == 1 {
                values.push(0.0);
                continue;
            }
            let is_high = mask[bit / 8] >> (bit % 8) & 1 == 1;
            bit += 1;
            let magnitude = if is_high { high[row] } else { low[row] };
            values.push(if code == 2 { magnitude } else { -magnitude });
        }
    }
    Ok(values)
}

fn int_n(bytes: &[u8], rows: usize, cols: usize, bits: u32) -> Result<Vec<f32>, String> {
    let total = rows * cols;
    let body_len = match bits {
        6 => total.div_ceil(4) * 3,
        _ => total,
    };
    if bytes.len() != body_len + 2 * rows {
        return Err(format!("an int{bits} record has the wrong length"));
    }
    let (body, scales) = bytes.split_at(body_len);
    let codes: Vec<i32> = match bits {
        6 => body
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|b| {
                let packed = u32::from(b[0]) | u32::from(b[1]) << 8 | u32::from(b[2]) << 16;
                (0..4).map(move |k| ((packed >> (6 * k)) & 0x3f) as i32 - 32)
            })
            .take(total)
            .collect(),
        _ => body
            .iter()
            .map(|b| i32::from(i8::from_ne_bytes([*b])))
            .collect(),
    };
    let scales: Vec<f32> = halves(scales).collect();
    Ok(codes
        .chunks_exact(cols)
        .zip(&scales)
        .flat_map(|(row, scale)| row.iter().map(move |q| *q as f32 * scale))
        .collect())
}
