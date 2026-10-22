//! Reads a `.safetensors` file: a little-endian u64 header length, a JSON
//! header with dtype, shape, and byte range per tensor, then the raw data.
//! Tensors are read from the file one at a time. Reading the whole 178 MB
//! file into one buffer kept that memory in the process after load.

use super::gemm::Precision;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub struct Tensors {
    source: Source,
    data_start: usize,
    entries: HashMap<String, Entry>,
    precision: Precision,
}

enum Source {
    File(File),
}

struct Entry {
    dtype: String,
    start: usize,
    end: usize,
}

impl Tensors {
    /// Opens a file whose dense layers `Dense::load` stores as `precision` says.
    pub fn open_with(path: &Path, precision: Precision) -> Result<Self, String> {
        let error = |e: std::io::Error| format!("Could not read {}: {e}", path.display());
        let file = File::open(path).map_err(error)?;
        let total = file.metadata().map_err(error)?.len() as usize;
        let mut length = [0u8; 8];
        read_at(&file, &mut length, 0).map_err(error)?;
        let header_len = u64::from_le_bytes(length) as usize;
        let data_start = 8usize
            .checked_add(header_len)
            .filter(|end| *end <= total)
            .ok_or("safetensors header runs past the end of the file")?;
        let mut header = vec![0u8; header_len];
        read_at(&file, &mut header, 8).map_err(error)?;
        let entries = parse_header(&header, data_start, total)?;
        Ok(Self {
            source: Source::File(file),
            data_start,
            entries,
            precision,
        })
    }

    /// How `Dense::load` stores the layers of this file.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    fn entry(&self, name: &str) -> Result<&Entry, String> {
        self.entries
            .get(name)
            .ok_or_else(|| format!("model file has no tensor {name}"))
    }

    fn raw(&self, name: &str, entry: &Entry) -> Result<Vec<u8>, String> {
        let (start, end) = (self.data_start + entry.start, self.data_start + entry.end);
        match &self.source {
            Source::File(file) => {
                let mut bytes = vec![0u8; end - start];
                read_at(file, &mut bytes, start as u64)
                    .map_err(|e| format!("Could not read tensor {name}: {e}"))?;
                Ok(bytes)
            }
        }
    }

    /// Calls `each` with the tensor's bytes, 1 MB at a time, so a large
    /// tensor never exists twice in memory (raw and converted).
    fn chunks(&self, name: &str, entry: &Entry, mut each: impl FnMut(&[u8])) -> Result<(), String> {
        const CHUNK: usize = 1 << 20;
        let (start, end) = (self.data_start + entry.start, self.data_start + entry.end);
        let Source::File(file) = &self.source;
        let mut buffer = vec![0u8; CHUNK.min(end - start)];
        let mut at = start;
        while at < end {
            let len = CHUNK.min(end - at);
            read_at(file, &mut buffer[..len], at as u64)
                .map_err(|e| format!("Could not read tensor {name}: {e}"))?;
            each(&buffer[..len]);
            at += len;
        }
        Ok(())
    }

    /// The tensor as f32 values, from F32, F16, or BF16 storage.
    pub fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let entry = self.entry(name)?;
        let width = match entry.dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            other => return Err(format!("tensor {name} has dtype {other}, not a float")),
        };
        let mut values = Vec::with_capacity((entry.end - entry.start) / width);
        let dtype = entry.dtype.as_str();
        self.chunks(name, entry, |bytes| match dtype {
            "F32" => values.extend(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b)),
            ),
            "F16" => values.extend(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f16_to_f32(u16::from_le_bytes(*b))),
            ),
            _ => values.extend(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16)),
            ),
        })?;
        Ok(values)
    }

    /// Checks the element count, so a wrong config cannot index past a tensor.
    pub fn f32_len(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        let values = self.f32(name)?;
        if values.len() != len {
            return Err(format!(
                "tensor {name} has {} values, expected {len}",
                values.len()
            ));
        }
        Ok(values)
    }

    /// The tensor as f16 bits with `len` values. F16 is read as stored;
    /// F32 and BF16 are rounded to the nearest f16 as they are read.
    pub fn f16_bits(&self, name: &str, len: usize) -> Result<Vec<u16>, String> {
        let entry = self.entry(name)?;
        let width = match entry.dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            other => return Err(format!("tensor {name} has dtype {other}, not a float")),
        };
        let count = (entry.end - entry.start) / width;
        if count != len {
            return Err(format!("tensor {name} has {count} values, expected {len}"));
        }
        let mut bits = Vec::with_capacity(len);
        let dtype = entry.dtype.as_str();
        self.chunks(name, entry, |bytes| match dtype {
            "F16" => bits.extend(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| u16::from_le_bytes(*b)),
            ),
            "F32" => bits.extend(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32_to_f16(f32::from_le_bytes(*b))),
            ),
            _ => {
                bits.extend(
                    bytes.as_chunks::<2>().0.iter().map(|b| {
                        f32_to_f16(f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
                    }),
                )
            }
        })?;
        Ok(bits)
    }

    pub fn u8(&self, name: &str) -> Result<Vec<u8>, String> {
        let entry = self.entry(name)?;
        if entry.dtype != "U8" {
            return Err(format!("tensor {name} has dtype {}, not U8", entry.dtype));
        }
        self.raw(name, entry)
    }
}

/// Name to dtype and byte range, checked against the file length.
fn parse_header(
    header: &[u8],
    data_start: usize,
    total: usize,
) -> Result<HashMap<String, Entry>, String> {
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(header)
        .map_err(|e| format!("safetensors header is not JSON: {e}"))?;
    let mut entries = HashMap::new();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = value["dtype"].as_str().unwrap_or_default().to_string();
        let offsets: Vec<usize> = value["data_offsets"]
            .as_array()
            .map(|o| {
                o.iter()
                    .filter_map(serde_json::Value::as_u64)
                    .map(|d| d as usize)
                    .collect()
            })
            .unwrap_or_default();
        let [start, end] = offsets[..] else {
            return Err(format!("tensor {name} has no data_offsets"));
        };
        if start > end || data_start + end > total {
            return Err(format!("tensor {name} points outside the file"));
        }
        entries.insert(name, Entry { dtype, start, end });
    }
    Ok(entries)
}

/// Positional read of the whole buffer; safe to call from several threads.
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            let read = file.seek_read(&mut buf[done..], offset + done as u64)?;
            if read == 0 {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            done += read;
        }
        Ok(())
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
}

/// IEEE 754 half precision to single precision, including subnormals.
/// Rounds to the nearest f16, ties to even; out of range becomes infinity.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        // Infinity stays infinity; NaN stays a quiet NaN.
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        // Subnormal f16, or zero when too small.
        if half_exponent < -10 {
            return sign;
        }
        let full = mantissa | 0x80_0000;
        let shift = (14 - half_exponent) as u32;
        let half = full >> shift;
        let rest = full & ((1 << shift) - 1);
        let midpoint = 1 << (shift - 1);
        let round = rest > midpoint || (rest == midpoint && half & 1 == 1);
        return sign | (half + u32::from(round)) as u16;
    }
    let half = ((half_exponent as u32) << 10) | (mantissa >> 13);
    let rest = mantissa & 0x1fff;
    let round = rest > 0x1000 || (rest == 0x1000 && half & 1 == 1);
    // A carry out of the mantissa raises the exponent, which is the right result.
    sign | (half + u32::from(round)) as u16
}

pub fn f16_to_f32(half: u16) -> f32 {
    let sign = u32::from(half >> 15) << 31;
    let exponent = u32::from((half >> 10) & 0x1f);
    let mantissa = u32::from(half & 0x3ff);
    let bits = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: value is mantissa * 2^-24. Normalize it.
            let mut exp = 127 - 15 + 1;
            let mut m = mantissa;
            while m & 0x400 == 0 {
                m <<= 1;
                exp -= 1;
            }
            sign | (exp << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}
