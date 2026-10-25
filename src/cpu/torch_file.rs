//! PyTorch checkpoints (`torch.save`, `.pth` / `.pt`): a zip archive with a
//! pickle (`data.pkl`) that names each tensor and points to its storage
//! (`data/<key>`). This reads the zip directory and runs the part of the
//! pickle machine that `torch.save` writes for a state dict or a tensor, so
//! nothing is executed: the only calls it knows are the tensor rebuild and
//! `OrderedDict`. The result is the byte range of each tensor in the file,
//! as the safetensors header gives it.

use std::collections::HashMap;
use std::fs::File;

/// One tensor: its dtype (safetensors names), and its bytes in the file.
pub struct Found {
    pub name: String,
    pub dtype: &'static str,
    pub start: usize,
    pub end: usize,
}

/// The tensors of a checkpoint. Nested dicts join their keys with dots;
/// the `module.` of a model saved from `DataParallel` is dropped. A file
/// that holds one tensor gives it the name "".
pub fn tensors(file: &File, total: usize) -> Result<Vec<Found>, String> {
    let zip = Zip::read(file, total)?;
    let pickle = zip
        .names
        .keys()
        .find(|name| name.ends_with("/data.pkl") || *name == "data.pkl")
        .cloned()
        .ok_or("the file is not a PyTorch checkpoint (no data.pkl)")?;
    let root = pickle
        .strip_suffix("data.pkl")
        .unwrap_or_default()
        .to_string();
    if let Some(order) = zip.names.get(&format!("{root}byteorder")) {
        if zip.bytes(file, order)? != b"little" {
            return Err("the checkpoint is big-endian; Trit reads little-endian files".into());
        }
    }
    let program = zip.bytes(file, &zip.names[&pickle])?;
    let value = Machine::default().run(&program)?;
    let mut tensors = Vec::new();
    collect(&value, String::new(), &mut |name, tensor| {
        let entry = zip
            .names
            .get(&format!("{root}data/{}", tensor.key))
            .ok_or_else(|| format!("tensor {name}: storage {} is missing", tensor.key))?;
        if entry.method != 0 {
            return Err(format!("tensor {name}: its storage is compressed"));
        }
        let width = width(tensor.dtype);
        let count: usize = tensor.shape.iter().product();
        let start = entry.data + tensor.offset * width;
        let end = start + count * width;
        if end > entry.data + entry.size {
            return Err(format!("tensor {name} runs past its storage"));
        }
        tensors.push(Found {
            name,
            dtype: tensor.dtype,
            start,
            end,
        });
        Ok(())
    })?;
    Ok(tensors)
}

fn width(dtype: &str) -> usize {
    match dtype {
        "F64" | "I64" => 8,
        "F32" | "I32" => 4,
        "F16" | "BF16" | "I16" => 2,
        _ => 1,
    }
}

/// Calls `found` for every tensor under `value`, with its dotted name.
fn collect(
    value: &Value,
    name: String,
    found: &mut dyn FnMut(String, &Tensor) -> Result<(), String>,
) -> Result<(), String> {
    match value {
        Value::Tensor(tensor) => found(name, tensor),
        Value::Dict(items) => {
            for (key, item) in items {
                let Value::Str(key) = key else { continue };
                let key = key.strip_prefix("module.").unwrap_or(key);
                let key = if key == "module" { "" } else { key };
                let child = match (name.is_empty(), key.is_empty()) {
                    (true, _) => key.to_string(),
                    (false, true) => name.clone(),
                    (false, false) => format!("{name}.{key}"),
                };
                collect(item, child, found)?;
            }
            Ok(())
        }
        // Other values (numbers, version strings) hold no weights.
        _ => Ok(()),
    }
}

/// A stored file of the zip: its compression method, and its data.
struct Member {
    method: u16,
    /// Offset of the data in the file.
    data: usize,
    size: usize,
}

struct Zip {
    names: HashMap<String, Member>,
}

impl Zip {
    fn read(file: &File, total: usize) -> Result<Self, String> {
        let not_zip = || "the file is not a zip archive (a PyTorch checkpoint is one)".to_string();
        // The end record is in the last 64 KB (its comment is at most 65535 bytes).
        let tail_len = total.min(65_557);
        let tail = read(file, total - tail_len, tail_len)?;
        let end = (0..tail.len().saturating_sub(21))
            .rev()
            .find(|i| tail[*i..*i + 4] == [0x50, 0x4b, 0x05, 0x06])
            .ok_or_else(not_zip)?;
        let record = &tail[end..];
        let mut count = u64::from(u16le(record, 10));
        let mut directory = u64::from(u32le(record, 16));
        if count == 0xffff || directory == 0xffff_ffff {
            // Zip64: the locator just before the end record points to the zip64 end record.
            let locator = end.checked_sub(20).ok_or_else(not_zip)?;
            if tail[locator..locator + 4] != [0x50, 0x4b, 0x06, 0x07] {
                return Err(not_zip());
            }
            let at = u64le(&tail, locator + 8) as usize;
            let record64 = read(file, at, 56)?;
            count = u64le(&record64, 32);
            directory = u64le(&record64, 48);
        }
        let mut names = HashMap::new();
        let mut at = directory as usize;
        for _ in 0..count {
            let header = read(file, at, 46)?;
            if header[..4] != [0x50, 0x4b, 0x01, 0x02] {
                return Err(not_zip());
            }
            let method = u16le(&header, 10);
            let mut size = u64::from(u32le(&header, 20));
            let (name_len, extra_len, comment_len) = (
                usize::from(u16le(&header, 28)),
                usize::from(u16le(&header, 30)),
                usize::from(u16le(&header, 32)),
            );
            let mut local = u64::from(u32le(&header, 42));
            let rest = read(file, at + 46, name_len + extra_len)?;
            let name = String::from_utf8_lossy(&rest[..name_len]).into_owned();
            // Zip64 sizes and offset are in the extra field when the 32-bit fields are full.
            let mut extra = &rest[name_len..];
            while extra.len() >= 4 {
                let (id, len) = (u16le(extra, 0), usize::from(u16le(extra, 2)));
                let body = &extra[4..(4 + len).min(extra.len())];
                if id == 1 {
                    let mut fields = body
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|b| u64::from_le_bytes(*b));
                    if u32le(&header, 24) == 0xffff_ffff {
                        fields.next();
                    }
                    if size == 0xffff_ffff {
                        size = fields.next().unwrap_or(size);
                    }
                    if local == 0xffff_ffff {
                        local = fields.next().unwrap_or(local);
                    }
                }
                extra = &extra[(4 + len).min(extra.len())..];
            }
            let local_header = read(file, local as usize, 30)?;
            let data = local as usize
                + 30
                + usize::from(u16le(&local_header, 26))
                + usize::from(u16le(&local_header, 28));
            if data + size as usize > total {
                return Err(format!("zip member {name} runs past the end of the file"));
            }
            names.insert(
                name,
                Member {
                    method,
                    data,
                    size: size as usize,
                },
            );
            at += 46 + name_len + extra_len + comment_len;
        }
        Ok(Self { names })
    }

    /// The bytes of a member, inflated when it is compressed.
    fn bytes(&self, file: &File, member: &Member) -> Result<Vec<u8>, String> {
        let raw = read(file, member.data, member.size)?;
        match member.method {
            0 => Ok(raw),
            8 => miniz_oxide::inflate::decompress_to_vec(&raw)
                .map_err(|e| format!("cannot inflate a zip member: {e:?}")),
            other => Err(format!("zip compression method {other} is not supported")),
        }
    }
}

fn read(file: &File, at: usize, len: usize) -> Result<Vec<u8>, String> {
    let mut buf = vec![0u8; len];
    super::tensors::read_at(file, &mut buf, at as u64)
        .map_err(|e| format!("cannot read the checkpoint: {e}"))?;
    Ok(buf)
}

fn u16le(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn u64le(b: &[u8], at: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(bytes)
}

#[derive(Clone, Debug)]
struct Tensor {
    key: String,
    dtype: &'static str,
    /// In elements, from the start of the storage.
    offset: usize,
    shape: Vec<usize>,
}

#[derive(Clone, Debug)]
enum Value {
    None,
    Int(i64),
    Str(String),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    /// A class or function by `module.name`.
    Global(String),
    /// A storage from a persistent id: its dtype and its key in `data/`.
    Storage(&'static str, String),
    Tensor(Tensor),
    /// A value that holds no weights: a bool, a float, bytes, another object.
    Other,
    Mark,
}

/// The pickle machine: a stack, the marks, and the memo.
#[derive(Default)]
struct Machine {
    stack: Vec<Value>,
    memo: HashMap<u32, Value>,
}

impl Machine {
    fn run(mut self, code: &[u8]) -> Result<Value, String> {
        let mut at = 0usize;
        let bad = |what: &str| format!("the checkpoint's pickle is not one Trit reads ({what})");
        let take = |at: &mut usize, len: usize| -> Result<&[u8], String> {
            let slice = code.get(*at..*at + len).ok_or_else(|| bad("truncated"))?;
            *at += len;
            Ok(slice)
        };
        loop {
            let op = *code.get(at).ok_or_else(|| bad("no STOP"))?;
            at += 1;
            match op {
                0x80 => at += 1,           // PROTO
                0x95 => at += 8,           // FRAME
                b'.' => return self.pop(), // STOP
                b'}' => self.stack.push(Value::Dict(Vec::new())),
                b']' => self.stack.push(Value::List(Vec::new())),
                b')' => self.stack.push(Value::Tuple(Vec::new())),
                b'(' => self.stack.push(Value::Mark),
                b'N' => self.stack.push(Value::None),
                0x88 | 0x89 => self.stack.push(Value::Other),
                b'K' => {
                    let v = take(&mut at, 1)?[0];
                    self.stack.push(Value::Int(i64::from(v)));
                }
                b'M' => {
                    let v = take(&mut at, 2)?;
                    self.stack.push(Value::Int(i64::from(u16le(v, 0))));
                }
                b'J' => {
                    let v = take(&mut at, 4)?;
                    self.stack.push(Value::Int(i64::from(u32le(v, 0) as i32)));
                }
                0x8a => {
                    // LONG1: little-endian two's complement of n bytes.
                    let n = usize::from(take(&mut at, 1)?[0]);
                    let bytes = take(&mut at, n)?;
                    let mut value: i64 = 0;
                    for (i, b) in bytes.iter().enumerate().take(8) {
                        value |= i64::from(*b) << (8 * i);
                    }
                    if n > 0 && n < 8 && bytes[n - 1] & 0x80 != 0 {
                        value -= 1i64 << (8 * n);
                    }
                    self.stack.push(Value::Int(value));
                }
                b'G' => {
                    take(&mut at, 8)?;
                    self.stack.push(Value::Other);
                }
                b'X' | 0x8c | 0x8d | b'T' | b'U' => {
                    let len = match op {
                        0x8c | b'U' => usize::from(take(&mut at, 1)?[0]),
                        0x8d => u64le(take(&mut at, 8)?, 0) as usize,
                        _ => u32le(take(&mut at, 4)?, 0) as usize,
                    };
                    let text = String::from_utf8_lossy(take(&mut at, len)?).into_owned();
                    self.stack.push(Value::Str(text));
                }
                b'B' | b'C' => {
                    let len = if op == b'C' {
                        usize::from(take(&mut at, 1)?[0])
                    } else {
                        u32le(take(&mut at, 4)?, 0) as usize
                    };
                    take(&mut at, len)?;
                    self.stack.push(Value::Other);
                }
                b'c' => {
                    // GLOBAL: "module\nname\n".
                    let mut line = || -> Result<String, String> {
                        let end = code[at..]
                            .iter()
                            .position(|b| *b == b'\n')
                            .ok_or_else(|| bad("GLOBAL"))?;
                        let text = String::from_utf8_lossy(&code[at..at + end]).into_owned();
                        at += end + 1;
                        Ok(text)
                    };
                    let module = line()?;
                    let name = line()?;
                    self.stack.push(Value::Global(format!("{module}.{name}")));
                }
                0x93 => {
                    // STACK_GLOBAL
                    let name = self.pop()?;
                    let module = self.pop()?;
                    match (module, name) {
                        (Value::Str(m), Value::Str(n)) => {
                            self.stack.push(Value::Global(format!("{m}.{n}")))
                        }
                        _ => return Err(bad("STACK_GLOBAL")),
                    }
                }
                b'q' => {
                    let key = u32::from(take(&mut at, 1)?[0]);
                    self.memoize(key)?;
                }
                b'r' => {
                    let key = u32le(take(&mut at, 4)?, 0);
                    self.memoize(key)?;
                }
                0x94 => {
                    let key = self.memo.len() as u32;
                    self.memoize(key)?;
                }
                b'h' | b'j' => {
                    let key = if op == b'h' {
                        u32::from(take(&mut at, 1)?[0])
                    } else {
                        u32le(take(&mut at, 4)?, 0)
                    };
                    let value = self.memo.get(&key).cloned().ok_or_else(|| bad("memo"))?;
                    self.stack.push(value);
                }
                b'0' => {
                    self.pop()?;
                }
                b'1' => {
                    self.items()?;
                }
                b'2' => {
                    let top = self.stack.last().cloned().ok_or_else(|| bad("DUP"))?;
                    self.stack.push(top);
                }
                b't' => {
                    let items = self.items()?;
                    self.stack.push(Value::Tuple(items));
                }
                0x85..=0x87 => {
                    let n = usize::from(op - 0x84);
                    let split = self
                        .stack
                        .len()
                        .checked_sub(n)
                        .ok_or_else(|| bad("TUPLE"))?;
                    let items = self.stack.split_off(split);
                    self.stack.push(Value::Tuple(items));
                }
                b'a' => {
                    let item = self.pop()?;
                    self.extend(vec![item])?;
                }
                b'e' => {
                    let items = self.items()?;
                    self.extend(items)?;
                }
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    self.set(vec![key, value])?;
                }
                b'u' => {
                    let items = self.items()?;
                    self.set(items)?;
                }
                b'Q' => {
                    // BINPERSID: ('storage', type, key, location, size).
                    let Value::Tuple(id) = self.pop()? else {
                        return Err(bad("persistent id"));
                    };
                    let (Some(Value::Global(kind)), Some(Value::Str(key))) = (id.get(1), id.get(2))
                    else {
                        return Err(bad("storage id"));
                    };
                    self.stack
                        .push(Value::Storage(storage_dtype(kind)?, key.clone()));
                }
                b'R' | 0x81 => {
                    // REDUCE and NEWOBJ: a call of a class or function with a tuple.
                    let Value::Tuple(args) = self.pop()? else {
                        return Err(bad("call arguments"));
                    };
                    let Value::Global(callee) = self.pop()? else {
                        return Err(bad("callee"));
                    };
                    let value = call(&callee, args)?;
                    self.stack.push(value);
                }
                b'b' => {
                    // BUILD: the state of an object (OrderedDict metadata); weights never.
                    self.pop()?;
                }
                other => return Err(bad(&format!("opcode 0x{other:02x}"))),
            }
        }
    }

    fn pop(&mut self) -> Result<Value, String> {
        self.stack
            .pop()
            .ok_or_else(|| "the checkpoint's pickle underflows".to_string())
    }

    fn memoize(&mut self, key: u32) -> Result<(), String> {
        let top = self
            .stack
            .last()
            .cloned()
            .ok_or("the checkpoint's pickle memoizes nothing")?;
        self.memo.insert(key, top);
        Ok(())
    }

    /// The values above the last mark, which goes.
    fn items(&mut self) -> Result<Vec<Value>, String> {
        let mark = self
            .stack
            .iter()
            .rposition(|v| matches!(v, Value::Mark))
            .ok_or("the checkpoint's pickle has no mark")?;
        let items = self.stack.split_off(mark + 1);
        self.stack.pop();
        Ok(items)
    }

    fn extend(&mut self, items: Vec<Value>) -> Result<(), String> {
        match self.stack.last_mut() {
            Some(Value::List(list)) => {
                list.extend(items);
                Ok(())
            }
            _ => Err("the checkpoint's pickle appends to a non-list".into()),
        }
    }

    fn set(&mut self, items: Vec<Value>) -> Result<(), String> {
        let Some(Value::Dict(dict)) = self.stack.last_mut() else {
            // An object's state set item by item (rare); no weights there.
            return Ok(());
        };
        let mut items = items.into_iter();
        while let (Some(key), Some(value)) = (items.next(), items.next()) {
            dict.push((key, value));
        }
        Ok(())
    }
}

fn storage_dtype(kind: &str) -> Result<&'static str, String> {
    Ok(match kind.rsplit('.').next().unwrap_or(kind) {
        "FloatStorage" => "F32",
        "HalfStorage" => "F16",
        "BFloat16Storage" => "BF16",
        "DoubleStorage" => "F64",
        "LongStorage" => "I64",
        "IntStorage" => "I32",
        "ShortStorage" => "I16",
        "CharStorage" => "I8",
        "ByteStorage" => "U8",
        "BoolStorage" => "BOOL",
        other => return Err(format!("storage type {other} is not supported")),
    })
}

/// The calls a state dict needs. The tensor must be contiguous: Trit reads
/// its bytes as one row-major block.
fn call(callee: &str, args: Vec<Value>) -> Result<Value, String> {
    match callee {
        "collections.OrderedDict" => Ok(Value::Dict(Vec::new())),
        "torch._utils._rebuild_tensor_v2" | "torch._utils._rebuild_tensor" => {
            let mut args = args.into_iter();
            let (
                Some(Value::Storage(dtype, key)),
                Some(Value::Int(offset)),
                Some(Value::Tuple(size)),
                Some(Value::Tuple(stride)),
            ) = (args.next(), args.next(), args.next(), args.next())
            else {
                return Err("a tensor in the checkpoint has an unknown layout".into());
            };
            let ints = |values: &[Value]| -> Vec<usize> {
                values
                    .iter()
                    .map(|v| {
                        if let Value::Int(i) = v {
                            *i as usize
                        } else {
                            0
                        }
                    })
                    .collect()
            };
            let (shape, stride) = (ints(&size), ints(&stride));
            let mut expected = 1;
            for (dim, step) in shape.iter().zip(&stride).rev() {
                if *dim > 1 && *step != expected {
                    return Err("a tensor in the checkpoint is not contiguous".into());
                }
                expected *= dim;
            }
            Ok(Value::Tensor(Tensor {
                key,
                dtype,
                offset: offset as usize,
                shape,
            }))
        }
        // torch.Size, parameters, and other wrappers: their first argument.
        "torch.nn.parameter.Parameter" | "torch._utils._rebuild_parameter" => {
            Ok(args.into_iter().next().unwrap_or(Value::None))
        }
        _ => Ok(Value::Other),
    }
}
