//! Model weights in a mapped cache file instead of the heap.
//!
//! A load converts the published weights (f16 to int8, base-3 to 2-bit
//! codes, f16 to f32) into the layouts of the kernels. With
//! `LoadOptions::cache`, the first load writes each converted buffer to a
//! file in the model folder (`.trit/`), tagged with a fingerprint of the
//! source model files. Later loads verify that fingerprint and map named
//! weight buffers directly from the cache file without re-reading or
//! reconverting the source weights.
//!
//! A mapped buffer is file memory: Windows and Linux count it as shared,
//! not private, and can drop its pages under memory pressure and read them
//! back, instead of writing them to the page file. Callers can call
//! `Mapping::prefetch` (via `Trit::prefetch_weights`) at the start of a
//! recording to fault evicted pages back sequentially before release.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::ops::Deref;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

/// Values a cache file can hold: no padding, every bit pattern valid.
pub trait Plain: Copy + Send + Sync + 'static {}
impl Plain for f32 {}
impl Plain for u16 {}
impl Plain for i8 {}
impl Plain for u8 {}

/// Smaller unnamed buffers stay on the heap: they are a few MB in all.
const MIN_MAPPED_BYTES: usize = 64 << 10;
/// Minimum alignment accepted when reading a cache file (backwards-compatible
/// with existing `TRITW004` caches).
const ALIGN: u64 = 64;
/// Alignment used when writing new cache buffers: 128 bytes matches Apple
/// Silicon's cache line size and stays a multiple of x86's 64-byte lines.
const WRITE_ALIGN: u64 = 128;
const MAGIC: &[u8; 8] = b"TRITW004";
/// Magic (8), index offset (8), and source fingerprint (32).
const HEADER: u64 = 48;
/// Trailing checksums at EOF: 32-byte payload digest + 32-byte index SHA-256.
const TRAILER_BYTES: usize = 64;
/// Fixed chunk size for parallel full-payload verification.
const PAYLOAD_CHUNK: usize = 4 << 20;
/// A temporary cache file this old has no writer: a load takes seconds.
const OLD_TEMPORARY: Duration = Duration::from_secs(60 * 60);

/// The weights of one layer: owned, or a view of a mapped cache file.
pub struct Store<T: Plain> {
    inner: Inner<T>,
}

enum Inner<T> {
    Owned(Vec<T>),
    Mapped {
        _map: Arc<Mapping>,
        ptr: *const T,
        len: usize,
    },
}

// SAFETY: a mapped store reads a read-only view that its `Arc<Mapping>`
// keeps alive; nothing writes to it.
unsafe impl<T: Plain> Send for Store<T> {}
// SAFETY: as for `Send`; shared reads of read-only memory.
unsafe impl<T: Plain> Sync for Store<T> {}

impl<T: Plain> std::fmt::Debug for Store<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mapped = matches!(self.inner, Inner::Mapped { .. });
        f.debug_struct("Store")
            .field("len", &self.len())
            .field("mapped", &mapped)
            .finish()
    }
}

impl<T: Plain> Store<T> {
    pub fn owned(values: Vec<T>) -> Self {
        Self {
            inner: Inner::Owned(values),
        }
    }

    /// Maps a named weight buffer of `len` elements directly from the active
    /// read cache without converting source tensors. Returns `None` outside a
    /// read cache session or when `tag` is absent or has a different byte size.
    pub fn mapped(tag: &str, len: usize) -> Option<Self> {
        let byte_len = len.checked_mul(std::mem::size_of::<T>())?;
        let read = {
            let guard = SESSION.lock();
            match guard.as_ref() {
                Some(Cache::Read(read)) => Arc::clone(read),
                _ => return None,
            }
        };
        let &(at, found_len) = read.by_tag.get(tag)?;
        if found_len != byte_len {
            return None;
        }
        let slice = read.map.bytes(at, byte_len);
        let ptr = slice.as_ptr().cast::<T>();
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
            read.missed.store(true, Ordering::Relaxed);
            return None;
        }
        did_read(byte_len as u64);
        Some(Self {
            inner: Inner::Mapped {
                _map: Arc::clone(&read.map),
                ptr,
                len,
            },
        })
    }

    /// Stores a named weight buffer in the active cache session. Unlike
    /// `weights`, named buffers of any non-empty size (such as int8 group
    /// scales) are stored so a cached layer can map all of its buffers without
    /// touching the source weight file.
    pub fn named(tag: &str, values: Vec<T>) -> Self {
        if values.is_empty() {
            return Self::owned(values);
        }
        let read = {
            let mut guard = SESSION.lock();
            match guard.as_mut() {
                Some(Cache::Read(read)) => Some(Arc::clone(read)),
                Some(Cache::Write(writer)) => {
                    writer.add(Some(tag), as_bytes(&values));
                    return Self::owned(values);
                }
                None => return Self::owned(values),
            }
        };
        if let Some(read) = read {
            let bytes = as_bytes(&values);
            if let Some(&(at, len)) = read.by_tag.get(tag) {
                if len == bytes.len() && read.map.bytes(at, len) == bytes {
                    let ptr = read.map.bytes(at, len).as_ptr().cast::<T>();
                    if (ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
                        return Self {
                            inner: Inner::Mapped {
                                _map: Arc::clone(&read.map),
                                ptr,
                                len: values.len(),
                            },
                        };
                    }
                }
            }
            read.missed.store(true, Ordering::Relaxed);
        }
        Self::owned(values)
    }

    /// A model weight. During a load with a cache it is the mapped copy
    /// when the file holds the same bytes, and is written to the file when
    /// the load builds one.
    pub fn weights(values: Vec<T>) -> Self {
        if std::mem::size_of_val(values.as_slice()) < MIN_MAPPED_BYTES {
            return Self::owned(values);
        }
        // Release the session lock before hashing and comparing bytes so
        // parallel block loaders do not serialize on a single mutex.
        let read = {
            let mut guard = SESSION.lock();
            match guard.as_mut() {
                Some(Cache::Read(read)) => Some(Arc::clone(read)),
                Some(Cache::Write(writer)) => {
                    writer.add(None, as_bytes(&values));
                    return Self::owned(values);
                }
                None => return Self::owned(values),
            }
        };
        let Some(read) = read else {
            return Self::owned(values);
        };
        let bytes = as_bytes(&values);
        let found = read.by_key.get(&key(bytes)).and_then(|spans| {
            spans
                .iter()
                .find(|&&(at, len)| len == bytes.len() && read.map.bytes(at, len) == bytes)
        });
        match found {
            Some(&(at, _)) => {
                let ptr = read.map.bytes(at, bytes.len()).as_ptr().cast::<T>();
                if !(ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
                    read.missed.store(true, Ordering::Relaxed);
                    return Self::owned(values);
                }
                Self {
                    inner: Inner::Mapped {
                        _map: Arc::clone(&read.map),
                        ptr,
                        len: values.len(),
                    },
                }
            }
            None => {
                read.missed.store(true, Ordering::Relaxed);
                Self::owned(values)
            }
        }
    }
}

impl<T: Plain> Deref for Store<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match &self.inner {
            Inner::Owned(values) => values,
            // SAFETY: `ptr` and `len` lie inside the view that `_map` keeps
            // mapped, aligned to `ALIGN` (the view starts on a page).
            Inner::Mapped { ptr, len, .. } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
        }
    }
}

fn as_bytes<T: Plain>(values: &[T]) -> &[u8] {
    // SAFETY: `Plain` types have no padding; any value is valid as bytes.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

/// A cheap key: the length and a sample of the bytes. A match is then
/// checked byte for byte.
fn key(bytes: &[u8]) -> u64 {
    const SAMPLES: usize = 64;
    const SPAN: usize = 64;
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ bytes.len() as u64;
    let mut mix = |b: &[u8]| {
        for byte in b {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
        }
    };
    let step = (bytes.len() / SAMPLES).max(1);
    for i in 0..SAMPLES {
        let at = (i * step).min(bytes.len());
        mix(&bytes[at..(at + SPAN).min(bytes.len())]);
    }
    mix(&bytes[bytes.len().saturating_sub(SPAN)..]);
    hash
}

/// Fast 4-lane 64-bit mixing hash over every byte of one `PAYLOAD_CHUNK` slice.
/// Processes 32 bytes per loop iteration so 700 MB verifies in ~15-25 ms across
/// parallel threads while covering 100% of payload bytes.
fn chunk_hash64(bytes: &[u8]) -> u64 {
    const K0: u64 = 0x9e37_79b9_7f4a_7c15;
    const K1: u64 = 0xbf58_476d_1ce4_e5b9;
    const K2: u64 = 0x94d0_49bb_1331_11eb;
    const K3: u64 = 0xd6e8_feb8_6659_fd93;
    let mut s0 = K0 ^ bytes.len() as u64;
    let mut s1 = K1;
    let mut s2 = K2;
    let mut s3 = K3;
    let (blocks, tail) = bytes.as_chunks::<32>();
    for b in blocks {
        let w0 = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        let w1 = u64::from_le_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]);
        let w2 = u64::from_le_bytes([b[16], b[17], b[18], b[19], b[20], b[21], b[22], b[23]]);
        let w3 = u64::from_le_bytes([b[24], b[25], b[26], b[27], b[28], b[29], b[30], b[31]]);
        s0 = (s0 ^ w0).wrapping_mul(K1).rotate_left(31);
        s1 = (s1 ^ w1).wrapping_mul(K2).rotate_left(27);
        s2 = (s2 ^ w2).wrapping_mul(K3).rotate_left(33);
        s3 = (s3 ^ w3).wrapping_mul(K0).rotate_left(29);
    }
    let mut rem = s0 ^ s1.rotate_left(13) ^ s2.rotate_left(29) ^ s3.rotate_left(47);
    let (words, bytes_left) = tail.as_chunks::<8>();
    for w in words {
        rem = (rem ^ u64::from_le_bytes(*w))
            .wrapping_mul(K1)
            .rotate_left(31);
    }
    for &byte in bytes_left {
        rem = (rem ^ u64::from(byte)).wrapping_mul(K2).rotate_left(17);
    }
    rem ^= rem >> 30;
    rem = rem.wrapping_mul(K1);
    rem ^= rem >> 27;
    rem = rem.wrapping_mul(K2);
    rem ^ (rem >> 31)
}

/// Streaming hasher matching `verify_payload_digest` over fixed `PAYLOAD_CHUNK`
/// slices of `[HEADER .. index_at)`.
struct ChunkedPayloadHasher {
    sha: Sha256,
    buf: Vec<u8>,
}

impl ChunkedPayloadHasher {
    fn new() -> Self {
        Self {
            sha: Sha256::new(),
            buf: Vec::with_capacity(PAYLOAD_CHUNK),
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            let need = PAYLOAD_CHUNK - self.buf.len();
            if self.buf.is_empty() && data.len() >= PAYLOAD_CHUNK {
                let h = chunk_hash64(&data[..PAYLOAD_CHUNK]);
                self.sha.update(&h.to_le_bytes());
                data = &data[PAYLOAD_CHUNK..];
            } else {
                let take = need.min(data.len());
                self.buf.extend_from_slice(&data[..take]);
                data = &data[take..];
                if self.buf.len() == PAYLOAD_CHUNK {
                    let h = chunk_hash64(&self.buf);
                    self.sha.update(&h.to_le_bytes());
                    self.buf.clear();
                }
            }
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        if !self.buf.is_empty() {
            let h = chunk_hash64(&self.buf);
            self.sha.update(&h.to_le_bytes());
            self.buf.clear();
        }
        self.sha.finalize()
    }
}

/// Computes the exact same chunked payload digest over `payload` using up to 8
/// threads so full-file verification stays fast and also faults mapped pages
/// into the working set.
fn payload_digest_parallel(payload: &[u8]) -> [u8; 32] {
    if payload.len() <= PAYLOAD_CHUNK {
        let mut sha = Sha256::new();
        if !payload.is_empty() {
            sha.update(&chunk_hash64(payload).to_le_bytes());
        }
        return sha.finalize();
    }
    let chunks: Vec<&[u8]> = payload.chunks(PAYLOAD_CHUNK).collect();
    let num_threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(8)
        .min(chunks.len());
    let mut hashes = vec![0u64; chunks.len()];
    let per_thread = chunks.len().div_ceil(num_threads);
    std::thread::scope(|scope| {
        let mut chunk_iter = chunks.chunks(per_thread);
        for out_slice in hashes.chunks_mut(per_thread) {
            if let Some(in_slice) = chunk_iter.next() {
                scope.spawn(move || {
                    for (slot, chunk) in out_slice.iter_mut().zip(in_slice) {
                        *slot = chunk_hash64(chunk);
                    }
                });
            }
        }
    });
    let mut sha = Sha256::new();
    for h in hashes {
        sha.update(&h.to_le_bytes());
    }
    sha.finalize()
}

/// The load in progress. Loads take turns (`LOADS`), so there is one.
static SESSION: Mutex<Option<Cache>> = Mutex::new(None);
/// One load at a time, with a cache or not: the buffers of a load without
/// one must not reach the file of another.
static LOADS: Mutex<()> = Mutex::new(());

struct ReadCache {
    map: Arc<Mapping>,
    /// `key` to (offset, byte length) for anonymous buffers.
    by_key: HashMap<u64, Vec<(usize, usize)>>,
    /// `tag` to (offset, byte length) for named buffers.
    by_tag: HashMap<String, (usize, usize)>,
    /// A buffer of this `Read` load was missing or mismatched.
    missed: AtomicBool,
}

enum Cache {
    Read(Arc<ReadCache>),
    Write(Box<Writer>),
}

struct IndexEntry {
    key: u64,
    at: u64,
    len: u64,
    tag: Option<String>,
}

struct Writer {
    file: std::io::BufWriter<File>,
    at: u64,
    index: Vec<IndexEntry>,
    fingerprint: [u8; 32],
    payload_hasher: ChunkedPayloadHasher,
    /// The first write error; the file is then dropped.
    failed: Option<std::io::Error>,
}

impl Writer {
    fn create(path: &Path, fingerprint: [u8; 32]) -> std::io::Result<Self> {
        let mut file = std::io::BufWriter::with_capacity(4 << 20, File::create(path)?);
        file.write_all(MAGIC)?;
        file.write_all(&0u64.to_le_bytes())?;
        file.write_all(&fingerprint)?;
        Ok(Self {
            file,
            at: HEADER,
            index: Vec::new(),
            fingerprint,
            payload_hasher: ChunkedPayloadHasher::new(),
            failed: None,
        })
    }

    fn add(&mut self, tag: Option<&str>, bytes: &[u8]) {
        if self.failed.is_some() {
            return;
        }
        let start = self.at.next_multiple_of(WRITE_ALIGN);
        let pad_len = (start - self.at) as usize;
        let padding = [0u8; WRITE_ALIGN as usize];
        let result = self
            .file
            .write_all(&padding[..pad_len])
            .and_then(|()| self.file.write_all(bytes));
        match result {
            Ok(()) => {
                if pad_len > 0 {
                    self.payload_hasher.update(&padding[..pad_len]);
                }
                self.payload_hasher.update(bytes);
                self.index.push(IndexEntry {
                    key: if tag.is_none() { key(bytes) } else { 0 },
                    at: start,
                    len: bytes.len() as u64,
                    tag: tag.map(str::to_string),
                });
                self.at = start + bytes.len() as u64;
            }
            Err(error) => self.failed = Some(error),
        }
    }

    fn finish(mut self) -> std::io::Result<()> {
        if let Some(error) = self.failed.take() {
            return Err(error);
        }
        let index_at = self.at;
        let payload_sha = self.payload_hasher.finalize();
        let mut index_bytes = Vec::with_capacity(8 + self.index.len() * 48);
        index_bytes.extend_from_slice(&(self.index.len() as u64).to_le_bytes());
        for entry in &self.index {
            for value in [entry.key, entry.at, entry.len] {
                index_bytes.extend_from_slice(&value.to_le_bytes());
            }
            let tag_bytes = entry.tag.as_deref().map_or(&[][..], str::as_bytes);
            let tag_len = u32::try_from(tag_bytes.len())
                .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidData))?;
            index_bytes.extend_from_slice(&tag_len.to_le_bytes());
            index_bytes.extend_from_slice(tag_bytes);
        }
        let mut hasher = Sha256::new();
        hasher.update(&index_bytes);
        let index_sha = hasher.finalize();

        self.file.write_all(&index_bytes)?;
        self.file.write_all(&payload_sha)?;
        self.file.write_all(&index_sha)?;
        let mut file = self.file.into_inner().map_err(|e| e.into_error())?;
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::Start(MAGIC.len() as u64))?;
        file.write_all(&index_at.to_le_bytes())?;
        file.write_all(&self.fingerprint)?;
        file.sync_all()
    }
}

type CacheIndices = (
    HashMap<u64, Vec<(usize, usize)>>,
    HashMap<String, (usize, usize)>,
);

/// Reads the index of a cache file whose source fingerprint matches `expected`.
/// Verifies both the index SHA-256 and the full chunked payload digest covering
/// every byte in `[HEADER .. index_at)` before returning any mapped spans.
fn read_index(map: &Mapping, expected: &[u8; 32]) -> Option<CacheIndices> {
    let word = |at: usize| -> Option<u64> {
        let bytes = map.get(at, 8)?;
        Some(u64::from_le_bytes(bytes.try_into().ok()?))
    };
    let u32_at = |at: usize| -> Option<u32> {
        let bytes = map.get(at, 4)?;
        Some(u32::from_le_bytes(bytes.try_into().ok()?))
    };
    if map.get(0, MAGIC.len())? != MAGIC {
        return None;
    }
    let index_at = usize::try_from(word(MAGIC.len())?).ok()?;
    let index_end = map.len.checked_sub(TRAILER_BYTES)?;
    if index_at < HEADER as usize || index_at > index_end {
        return None;
    }
    let fp = map.get(MAGIC.len() + 8, 32)?;
    if fp != expected {
        return None;
    }
    // Verify the index SHA-256 before trusting any offsets, lengths, or tags.
    let index_slice = map.get(index_at, index_end - index_at)?;
    let expected_payload_sha = map.get(index_end, 32)?;
    let expected_index_sha = map.get(index_end + 32, 32)?;
    let mut hasher = Sha256::new();
    hasher.update(index_slice);
    if hasher.finalize().as_slice() != expected_index_sha {
        return None;
    }

    let count = usize::try_from(word(index_at)?).ok()?;
    let mut by_key: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
    let mut by_tag: HashMap<String, (usize, usize)> = HashMap::new();
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(count);
    let mut cursor = index_at.checked_add(8)?;
    for _ in 0..count {
        let key = word(cursor)?;
        let at = usize::try_from(word(cursor.checked_add(8)?)?).ok()?;
        let len = usize::try_from(word(cursor.checked_add(16)?)?).ok()?;
        let tag_len = usize::try_from(u32_at(cursor.checked_add(24)?)?).ok()?;
        cursor = cursor.checked_add(28)?;
        if cursor.checked_add(tag_len)? > index_end {
            return None;
        }
        let tag_bytes = map.get(cursor, tag_len)?;
        cursor = cursor.checked_add(tag_len)?;
        // A mapped buffer can be read as `f32`: it must start aligned and stay
        // strictly within the payload region `[HEADER .. index_at)`.
        if !at.is_multiple_of(ALIGN as usize) || at < HEADER as usize {
            return None;
        }
        let end = at.checked_add(len)?;
        if end > index_at {
            return None;
        }
        map.get(at, len)?;
        spans.push((at, end));
        if tag_len > 0 {
            let tag = std::str::from_utf8(tag_bytes).ok()?.to_string();
            // Reject duplicate tags so a damaged index cannot silently shadow weights.
            if by_tag.insert(tag, (at, len)).is_some() {
                return None;
            }
        } else {
            by_key.entry(key).or_default().push((at, len));
        }
    }
    if cursor != index_end {
        return None;
    }
    // Reject overlapping payload spans so corrupted indices cannot alias buffers.
    spans.sort_unstable();
    for pair in spans.windows(2) {
        if pair[0].1 > pair[1].0 {
            return None;
        }
    }
    // Verify 100% of payload bytes in parallel chunks before trusting any direct map.
    let payload_slice = map.get(HEADER as usize, index_at - HEADER as usize)?;
    if payload_digest_parallel(payload_slice).as_slice() != expected_payload_sha {
        return None;
    }
    Some((by_key, by_tag))
}

struct Progress {
    report: Arc<dyn Fn(f32) + Send + Sync>,
    /// The fraction of the whole load that this pass covers: (start, width).
    span: (f32, f32),
    total: AtomicU64,
    done: AtomicU64,
    /// The highest value reported, in thousandths. All passes of one load
    /// share it, so the value never goes back when a pass starts again.
    shown: Arc<AtomicU64>,
}

static PROGRESS: Mutex<Option<Arc<Progress>>> = Mutex::new(None);

pub(crate) fn will_read(bytes: u64) {
    if let Some(progress) = PROGRESS.lock().as_ref() {
        progress.total.fetch_add(bytes, Ordering::Relaxed);
    }
}

pub(crate) fn did_read(bytes: u64) {
    let Some(progress) = PROGRESS.lock().clone() else {
        return;
    };
    let done = progress.done.fetch_add(bytes, Ordering::Relaxed) + bytes;
    let total = progress.total.load(Ordering::Relaxed).max(1);
    let fraction = (done as f32 / total as f32).min(1.0);
    let value = progress.span.0 + progress.span.1 * fraction;
    let thousandths = (value * 1000.0) as u64;
    if progress.shown.fetch_max(thousandths, Ordering::Relaxed) < thousandths {
        (progress.report)(value);
    }
}

/// The caller's progress function, and the highest value it got in this load.
type Report<'a> = (&'a Arc<dyn Fn(f32) + Send + Sync>, &'a Arc<AtomicU64>);

/// One load pass with `cache` as its session. Returns the model and the
/// session, for its writer or its misses.
fn pass<R>(
    cache: Option<Cache>,
    report: Option<Report<'_>>,
    span: (f32, f32),
    load: impl FnOnce() -> R,
) -> (R, Option<Cache>) {
    *SESSION.lock() = cache;
    *PROGRESS.lock() = report.map(|(report, shown)| {
        Arc::new(Progress {
            report: Arc::clone(report),
            span,
            total: AtomicU64::new(0),
            done: AtomicU64::new(0),
            shown: Arc::clone(shown),
        })
    });
    // Also when `load` panics: a session that stays would take the buffers
    // of the next load, and its progress function would get their reads.
    let _closed = Closed;
    let result = load();
    // A statement of its own: the lock must be free again when `_closed`
    // drops and takes it (a guard in the last expression lives longer).
    let session = SESSION.lock().take();
    (result, session)
}

/// Ends the session and the progress of a pass when it drops.
struct Closed;

impl Drop for Closed {
    fn drop(&mut self) {
        *PROGRESS.lock() = None;
        *SESSION.lock() = None;
    }
}

/// Runs a load that keeps no cache, in its turn: its buffers must not reach
/// the file of a cached load that runs at the same time.
pub(crate) fn in_turn<R>(load: impl FnOnce() -> R) -> R {
    let _turn = LOADS.lock();
    load()
}

fn reader(path: &Path, expected: &[u8; 32]) -> Option<Arc<ReadCache>> {
    let map = Arc::new(Mapping::open(path).ok()?);
    let (by_key, by_tag) = read_index(&map, expected)?;
    Some(Arc::new(ReadCache {
        map,
        by_key,
        by_tag,
        missed: AtomicBool::new(false),
    }))
}

/// Result of `store::load`: the loaded model, its mapped weight file (if any),
/// and whether the load hit an existing cache file.
pub(crate) struct Loaded<M> {
    pub(crate) model: M,
    pub(crate) mapping: Option<Arc<Mapping>>,
    pub(crate) cache_hit: bool,
}

/// Loads with `load`, keeping the converted weights in the cache file
/// `dir/.trit/<name>` when `name` is given. `report` gets the progress from
/// 0 to 1. `release` returns freed memory between passes and after the
/// load (the load's temporaries; see `release_memory`). A cache error
/// gives the plain load: the cache only moves memory, it never fails a load.
pub(crate) fn load<M>(
    dir: &Path,
    name: Option<String>,
    report: Option<Arc<dyn Fn(f32) + Send + Sync>>,
    release: impl Fn(),
    load: impl Fn() -> Result<M, String>,
) -> Result<Loaded<M>, String> {
    let _turn = LOADS.lock();
    let shown = Arc::new(AtomicU64::new(0));
    let report = report.as_ref().map(|report| (report, &shown));
    let finished = |model: Result<M, String>, mapping: Option<Arc<Mapping>>, cache_hit: bool| {
        if model.is_ok() {
            release();
        }
        if let (Some((report, _)), Ok(_)) = (report, &model) {
            report(1.0);
        }
        model.map(|model| Loaded {
            model,
            mapping,
            cache_hit,
        })
    };
    // A mapped file that goes away stops the process (see `local_disk`).
    let Some(name) = name.filter(|_| local_disk(dir)) else {
        return finished(pass(None, report, (0.0, 1.0), &load).0, None, false);
    };
    let Some(fp) = dir_fingerprint(dir) else {
        return finished(pass(None, report, (0.0, 1.0), &load).0, None, false);
    };
    let folder = dir.join(".trit");
    let path = folder.join(&name);
    remove_old_temporaries(&folder);
    // A file from an earlier load: map its buffers directly when complete.
    if let Some(read) = reader(&path, &fp) {
        let map = Arc::clone(&read.map);
        let (model, _) = pass(
            Some(Cache::Read(Arc::clone(&read))),
            report,
            (0.0, 1.0),
            &load,
        );
        if model.is_err() || !read.missed.load(Ordering::Relaxed) {
            return finished(model, Some(map), true);
        }
        // The cache missed a buffer: rebuild the file.
        drop(model);
        drop(read);
        drop(map);
        release();
    }
    let temporary = folder.join(format!("{name}.{}.tmp", std::process::id()));
    let Ok(writer) = std::fs::create_dir_all(&folder).and_then(|()| Writer::create(&temporary, fp))
    else {
        return finished(pass(None, report, (0.0, 1.0), &load).0, None, false);
    };
    let (model, session) = pass(
        Some(Cache::Write(Box::new(writer))),
        report,
        (0.0, 0.85),
        &load,
    );
    // Never publish a cache file when the model load failed: close the writer
    // and remove the temporary file before touching the published cache.
    let model = match model {
        Ok(model) => model,
        Err(error) => {
            drop(session);
            let _ = std::fs::remove_file(&temporary);
            return finished(Err(error), None, false);
        }
    };
    let written = match session {
        Some(Cache::Write(writer)) => (*writer).finish(),
        _ => Err(std::io::ErrorKind::Other.into()),
    }
    // Another process may map the old file; it then stays, and this
    // process keeps its own buffers.
    .and_then(|()| std::fs::rename(&temporary, &path));
    if written.is_err() {
        let _ = std::fs::remove_file(&temporary);
        return finished(Ok(model), None, false);
    }
    remove_stale(&folder, &name);
    // Map the freshly written file on the fast named path so this process
    // shares file memory instead of holding heap weights.
    drop(model);
    release();
    if let Some(read) = reader(&path, &fp) {
        let map = Arc::clone(&read.map);
        let (model, _) = pass(
            Some(Cache::Read(Arc::clone(&read))),
            report,
            (0.85, 0.15),
            &load,
        );
        if model.is_err() || !read.missed.load(Ordering::Relaxed) {
            return finished(model, Some(map), false);
        }
        // A newly written cache that still misses fell out of sync; fall back
        // to an uncached load instead of keeping a partially mapped model.
        drop(model);
        drop(read);
        drop(map);
        release();
    }
    finished(pass(None, report, (0.85, 0.15), &load).0, None, false)
}

/// Recursively collects regular source files (following symlinks) under `dir`,
/// skipping `.trit` cache folders and temporary downloads.
fn collect_source_files(
    root: &Path,
    dir: &Path,
    ancestors: &mut Vec<std::path::PathBuf>,
    out: &mut Vec<(String, std::path::PathBuf, std::fs::Metadata)>,
) -> Option<()> {
    let canonical = dir.canonicalize().ok()?;
    if ancestors.contains(&canonical) {
        return Some(());
    }
    ancestors.push(canonical);
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name == ".trit" || name.ends_with(".tmp") || name.ends_with(".download") {
            continue;
        }
        // `fs::metadata` follows symlinks so linked shards are fingerprinted too.
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        if meta.is_dir() {
            collect_source_files(root, &path, ancestors, out)?;
        } else if meta.is_file() {
            let rel = path
                .strip_prefix(root)
                .ok()?
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, path, meta));
        }
    }
    ancestors.pop();
    Some(())
}

/// Computes a fast fingerprint of the model files in `dir` (including nested
/// shards and symlinked files): sorted relative paths, byte sizes,
/// modification timestamps, full headers/small files, and evenly-spaced 4 KB
/// samples, hashed with SHA-256.
fn dir_fingerprint(dir: &Path) -> Option<[u8; 32]> {
    let mut files = Vec::new();
    collect_source_files(dir, dir, &mut Vec::new(), &mut files)?;
    if files.is_empty() {
        return None;
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut hasher = Sha256::new();
    for (name, path, meta) in files {
        let len = meta.len();
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&len.to_le_bytes());
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0u128, |d| d.as_nanos());
        hasher.update(&mtime.to_le_bytes());

        let file = File::open(&path).ok()?;
        const FULL_HASH_MAX: u64 = 1 << 20;
        const HEADER_BYTES: u64 = 64 << 10;
        const SAMPLES: u64 = 64;
        const BLOCK: usize = 4096;
        let mut buf = [0u8; BLOCK];
        if len <= FULL_HASH_MAX {
            let mut at = 0u64;
            while at < len {
                let take = (len - at).min(BLOCK as u64) as usize;
                read_exact_at(&file, &mut buf[..take], at).ok()?;
                hasher.update(&buf[..take]);
                at += take as u64;
            }
        } else {
            // Hash the full leading 64 KB (safetensors/GGUF header + metadata)
            // plus 64 evenly-spaced 4 KB blocks spanning the rest of the file.
            let mut at = 0u64;
            while at < HEADER_BYTES {
                read_exact_at(&file, &mut buf, at).ok()?;
                hasher.update(&buf);
                at += BLOCK as u64;
            }
            let tail_start = HEADER_BYTES;
            let tail_span = len - tail_start - BLOCK as u64;
            let step = tail_span / (SAMPLES - 1);
            for i in 0..SAMPLES {
                let pos = if i + 1 == SAMPLES {
                    len - BLOCK as u64
                } else {
                    tail_start + i * step
                };
                read_exact_at(&file, &mut buf, pos).ok()?;
                hasher.update(&buf);
            }
        }
    }
    Some(hasher.finalize())
}

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
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

/// Minimal FIPS 180-4 SHA-256 so the cache fingerprint adds no crate dependency.
struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

impl Sha256 {
    const INIT: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    fn new() -> Self {
        Self {
            state: Self::INIT,
            buf: [0; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        let (chunks, rest) = data.as_chunks::<64>();
        for chunk in chunks {
            self.compress(chunk);
        }
        if !rest.is_empty() {
            self.buf[..rest.len()].copy_from_slice(rest);
            self.buf_len = rest.len();
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, word) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*word);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for (&k, &word) in Self::K.iter().zip(&w) {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k)
                .wrapping_add(word);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, delta) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(delta);
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        self.update(&[0x80]);
        let pad = if self.buf_len <= 56 {
            56 - self.buf_len
        } else {
            120 - self.buf_len
        };
        let zeros = [0u8; 64];
        if pad > 64 {
            self.update(&zeros);
            self.update(&zeros[..pad - 64]);
        } else {
            self.update(&zeros[..pad]);
        }
        self.update(&bit_len.to_be_bytes());
        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

/// Whether `dir` is on a disk that stays: not a network folder, and not a
/// removable disk. When a mapped file goes away, the next read of its pages
/// is an access fault that no code can catch, so a model on such a disk
/// loads without the cache. Only Windows can tell here; elsewhere this is true.
#[cfg(windows)]
fn local_disk(dir: &Path) -> bool {
    use std::os::windows::ffi::OsStrExt;
    use windows::core::PCWSTR;
    use windows::Win32::Storage::FileSystem::GetDriveTypeW;
    /// DRIVE_REMOVABLE, DRIVE_REMOTE, DRIVE_CDROM.
    const GOES_AWAY: [u32; 3] = [2, 4, 5];
    let Ok(full) = dir.canonicalize() else {
        return true;
    };
    // `\\?\C:\models\x` for a disk, `\\?\UNC\server\share\x` for a network folder.
    let text = full.to_string_lossy();
    let path = text.strip_prefix(r"\\?\").unwrap_or(&text);
    if path.starts_with("UNC\\") || path.starts_with(r"\\") {
        return false;
    }
    let Some(root) = path.get(..3).filter(|root| root.ends_with(":\\")) else {
        return true;
    };
    let wide: Vec<u16> = std::ffi::OsStr::new(root)
        .encode_wide()
        .chain([0])
        .collect();
    // SAFETY: `wide` is a NUL-terminated string that lives through the call.
    let kind = unsafe { GetDriveTypeW(PCWSTR(wide.as_ptr())) };
    !GOES_AWAY.contains(&kind)
}

#[cfg(target_os = "macos")]
fn local_disk(dir: &Path) -> bool {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let Ok(path) = CString::new(dir.as_os_str().as_bytes()) else {
        return true;
    };
    // SAFETY: `libc::statfs` is a plain C struct; zero-initialization is valid.
    let mut stat: libc::statfs = unsafe { std::mem::zeroed() };
    // SAFETY: `path` is NUL-terminated and `stat` is writable for the call.
    if unsafe { libc::statfs(path.as_ptr(), &mut stat) } != 0 {
        return true;
    }
    (stat.f_flags & libc::MNT_LOCAL as u32) != 0
}

#[cfg(not(any(windows, target_os = "macos")))]
fn local_disk(_dir: &Path) -> bool {
    true
}

/// Removes the cache files of other Trit versions and kernel paths. The
/// files of the other precisions of this version stay: a second program can
/// use one of them.
fn remove_stale(folder: &Path, keep: &str) {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return;
    };
    let (version, kernels) = (version_part(), kernels_part());
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let current = name.starts_with(&version) && name.ends_with(&kernels);
        if name != keep && !current && name.starts_with("weights-") && name.ends_with(".bin") {
            // A file that another process maps stays until it closes it.
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Removes the temporary files of loads that stopped before the end: each
/// is as large as the cache. A recent file can be the load of another
/// process, so it stays.
fn remove_old_temporaries(folder: &Path) {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let old = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .is_ok_and(|at| at.elapsed().is_ok_and(|age| age > OLD_TEMPORARY));
        if old && name.starts_with("weights-") && name.ends_with(".tmp") {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// The cache file name for a load: the Trit version, the precision, and the
/// kernel paths, which set the layouts.
pub(crate) fn file_name(precision: &str) -> String {
    format!("{}{precision}{}", version_part(), kernels_part())
}

/// The start of every cache file name of this Trit version.
fn version_part() -> String {
    format!("weights-{}-", env!("CARGO_PKG_VERSION"))
}

/// The end of every cache file name of this computer's kernel paths.
fn kernels_part() -> String {
    let path: String = super::cpu_path()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    format!("-{path}.bin")
}

/// Hardware page size for working-set probing and sequential prefetching:
/// 16 KB on Apple Silicon, 4 KB on x86 and most other hosts.
fn os_page_size() -> usize {
    #[cfg(unix)]
    {
        use std::sync::OnceLock;
        static PAGE: OnceLock<usize> = OnceLock::new();
        *PAGE.get_or_init(|| {
            // SAFETY: `sysconf(_SC_PAGESIZE)` has no preconditions.
            let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            usize::try_from(raw)
                .ok()
                .filter(|&n| n >= 4096 && n.is_power_of_two())
                .unwrap_or(4096)
        })
    }
    #[cfg(not(unix))]
    {
        4096
    }
}

pub(crate) struct Mapping {
    ptr: *const u8,
    len: usize,
    prefetching: AtomicBool,
    prefetched: AtomicBool,
    #[cfg(windows)]
    handle: windows::Win32::Foundation::HANDLE,
}

// SAFETY: the view is read-only and lives until `drop`.
unsafe impl Send for Mapping {}
// SAFETY: as for `Send`.
unsafe impl Sync for Mapping {}

impl Mapping {
    fn bytes(&self, at: usize, len: usize) -> &[u8] {
        // SAFETY: callers pass ranges that `read_index` checked with `get`.
        unsafe { std::slice::from_raw_parts(self.ptr.add(at), len) }
    }

    fn get(&self, at: usize, len: usize) -> Option<&[u8]> {
        let end = at.checked_add(len)?;
        (end <= self.len).then(|| self.bytes(at, len))
    }

    /// Hints to the OS to read mapped weight pages sequentially and faults
    /// their page-table entries into the process working set so the next
    /// inference pass does not stall on demand page faults.
    ///
    /// After the first touch, probes 16 pages on supported systems. This
    /// skips repeated scans while the sampled pages are still resident.
    pub(crate) fn prefetch(&self) {
        if self.len == 0 || self.prefetching.swap(true, Ordering::AcqRel) {
            return;
        }
        struct Reset<'a>(&'a AtomicBool);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        let _reset = Reset(&self.prefetching);

        // File-cache residency does not prove that this mapping has page-table entries.
        if self.prefetched.load(Ordering::Relaxed) && self.is_sample_resident() {
            return;
        }

        #[cfg(windows)]
        {
            use windows::Win32::System::Memory::{PrefetchVirtualMemory, WIN32_MEMORY_RANGE_ENTRY};
            use windows::Win32::System::Threading::GetCurrentProcess;
            let entry = WIN32_MEMORY_RANGE_ENTRY {
                VirtualAddress: self.ptr.cast_mut().cast(),
                NumberOfBytes: self.len,
            };
            // SAFETY: `[ptr .. ptr + len)` is a live read-only mapping owned by `self`.
            let _ = unsafe { PrefetchVirtualMemory(GetCurrentProcess(), &[entry], 0) };
        }
        #[cfg(unix)]
        {
            // SAFETY: `[ptr .. ptr + len)` is a live read-only mapping owned by `self`.
            unsafe {
                let _ = libc::madvise(self.ptr.cast_mut().cast(), self.len, libc::MADV_WILLNEED);
            }
        }
        // `PrefetchVirtualMemory` / `madvise` brings file pages into the OS
        // cache, but does not install PTEs into this process's working set.
        // Touching one byte per OS page (16 KB on Apple Silicon, 4 KB on x86)
        // moves soft faults off the inference critical path while keeping the
        // pages shared file memory.
        let page = os_page_size();
        let mut acc = 0u8;
        let mut offset = 0usize;
        while offset < self.len {
            // SAFETY: `offset < self.len` and `[ptr .. ptr + len)` is valid read-only memory.
            acc = acc.wrapping_add(unsafe { std::ptr::read_volatile(self.ptr.add(offset)) });
            offset += page;
        }
        std::hint::black_box(acc);
        self.prefetched.store(true, Ordering::Relaxed);
    }

    /// Returns true if a 16-page sample spanning the mapping is already valid
    /// in the current process working set.
    #[cfg(windows)]
    fn is_sample_resident(&self) -> bool {
        use windows::Win32::System::ProcessStatus::{
            QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION,
        };
        use windows::Win32::System::Threading::GetCurrentProcess;
        const SAMPLES: usize = 16;
        const PAGE: usize = 4096;
        let last_page = (self.len.saturating_sub(1)) & !(PAGE - 1);
        let mut info = [PSAPI_WORKING_SET_EX_INFORMATION::default(); SAMPLES];
        for (i, slot) in info.iter_mut().enumerate() {
            let offset = if SAMPLES > 1 {
                ((last_page as u128 * i as u128) / (SAMPLES - 1) as u128) as usize & !(PAGE - 1)
            } else {
                0
            };
            // SAFETY: `offset < self.len` and `self.ptr` is page-aligned (`MapViewOfFile`).
            slot.VirtualAddress = unsafe { self.ptr.add(offset) }.cast_mut().cast();
        }
        let cb = std::mem::size_of_val(&info) as u32;
        // SAFETY: `info` is a stack array of `cb` bytes and `GetCurrentProcess` is valid.
        let ok =
            unsafe { QueryWorkingSetEx(GetCurrentProcess(), info.as_mut_ptr().cast(), cb) }.is_ok();
        if !ok {
            return false;
        }
        info.iter().all(|entry| {
            // SAFETY: `Flags` is the raw `usize` representation of `PSAPI_WORKING_SET_EX_BLOCK`
            // where bit 0 is `Valid`.
            unsafe { entry.VirtualAttributes.Flags & 1 != 0 }
        })
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
    fn is_sample_resident(&self) -> bool {
        const SAMPLES: usize = 16;
        let page = os_page_size();
        let last_page = (self.len.saturating_sub(1)) & !(page - 1);
        for i in 0..SAMPLES {
            let offset = if SAMPLES > 1 {
                ((last_page as u128 * i as u128) / (SAMPLES - 1) as u128) as usize & !(page - 1)
            } else {
                0
            };
            let mut vec = 0u8;
            // SAFETY: `self.ptr` is page-aligned (`mmap`), `offset` is a
            // page-aligned offset `< self.len`, and `vec` holds one byte for
            // the single probed page (`MINCORE_INCORE` is bit 0 on macOS/Linux).
            let rc = unsafe {
                libc::mincore(
                    self.ptr.add(offset).cast_mut().cast(),
                    page,
                    std::ptr::from_mut(&mut vec).cast(),
                )
            };
            if rc != 0 || (vec & 1) == 0 {
                return false;
            }
        }
        true
    }

    #[cfg(not(any(
        windows,
        target_os = "macos",
        target_os = "linux",
        target_os = "android"
    )))]
    fn is_sample_resident(&self) -> bool {
        false
    }

    #[cfg(windows)]
    fn open(path: &Path) -> std::io::Result<Self> {
        use std::os::windows::io::AsRawHandle;
        use windows::Win32::Foundation::{CloseHandle, HANDLE};
        use windows::Win32::System::Memory::{
            CreateFileMappingW, MapViewOfFile, FILE_MAP_READ, PAGE_READONLY,
        };
        let file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidData))?;
        if len < HEADER as usize {
            return Err(std::io::ErrorKind::InvalidData.into());
        }
        // SAFETY: `file` is an open handle; the mapping holds its own
        // reference to the file, so `file` may close after this.
        let handle = unsafe {
            CreateFileMappingW(
                HANDLE(file.as_raw_handle()),
                None,
                PAGE_READONLY,
                0,
                0,
                None,
            )
        }
        .map_err(std::io::Error::other)?;
        // SAFETY: `handle` is the mapping created above; length 0 maps all of it.
        let view = unsafe { MapViewOfFile(handle, FILE_MAP_READ, 0, 0, 0) };
        if view.Value.is_null() {
            let error = std::io::Error::last_os_error();
            // SAFETY: `handle` is open and owned here.
            let _ = unsafe { CloseHandle(handle) };
            return Err(error);
        }
        Ok(Self {
            ptr: view.Value.cast::<u8>().cast_const(),
            len,
            prefetching: AtomicBool::new(false),
            prefetched: AtomicBool::new(false),
            handle,
        })
    }

    #[cfg(unix)]
    fn open(path: &Path) -> std::io::Result<Self> {
        use std::os::unix::io::AsRawFd;
        let file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidData))?;
        if len < HEADER as usize {
            return Err(std::io::ErrorKind::InvalidData.into());
        }
        // SAFETY: a shared read-only mapping of an open file; the mapping
        // stays valid after `file` closes.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self {
            ptr: ptr.cast::<u8>().cast_const(),
            len,
            prefetching: AtomicBool::new(false),
            prefetched: AtomicBool::new(false),
        })
    }

    #[cfg(not(any(windows, unix)))]
    fn open(_path: &Path) -> std::io::Result<Self> {
        Err(std::io::ErrorKind::Unsupported.into())
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        #[cfg(windows)]
        {
            use windows::Win32::Foundation::CloseHandle;
            use windows::Win32::System::Memory::{UnmapViewOfFile, MEMORY_MAPPED_VIEW_ADDRESS};
            // SAFETY: the view and the handle were created in `open` and
            // no store refers to them any more (each holds an `Arc`).
            unsafe {
                let _ = UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: self.ptr.cast_mut().cast(),
                });
                let _ = CloseHandle(self.handle);
            }
        }
        #[cfg(unix)]
        // SAFETY: the mapping was created in `open` with this length.
        unsafe {
            libc::munmap(self.ptr.cast_mut().cast(), self.len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn temp_model_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "trit-store-test-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create test dir");
        std::fs::write(dir.join("model.bin"), b"source-model-bytes-v1").expect("write model");
        dir
    }

    #[test]
    fn failed_load_does_not_publish_or_overwrite_cache() {
        let dir = temp_model_dir("failed-load");
        let name = "weights-test-failed.bin".to_string();
        let cache_path = dir.join(".trit").join(&name);

        let result: Result<Loaded<Store<f32>>, String> = load(
            &dir,
            Some(name.clone()),
            None,
            || {},
            || {
                let _partial = Store::named("layer0", vec![1.0f32; 128]);
                Err("injected load failure".to_string())
            },
        );
        assert!(result.is_err());
        assert!(
            !cache_path.exists(),
            "failed load must not publish a cache file"
        );
        let temps: Vec<_> = std::fs::read_dir(dir.join(".trit"))
            .expect("read .trit")
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(temps.is_empty(), "failed load must remove its temp file");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn named_cache_hit_skips_recompute_and_invalidates_on_source_change() {
        let dir = temp_model_dir("named-hit");
        let name = "weights-test-named.bin".to_string();
        let builds = AtomicUsize::new(0);

        let load_model = || {
            if let Some(mapped) = Store::<f32>::mapped("w0", 256) {
                return Ok(mapped);
            }
            builds.fetch_add(1, Ordering::Relaxed);
            Ok(Store::named("w0", (0..256).map(|i| i as f32).collect()))
        };

        let first = load(&dir, Some(name.clone()), None, || {}, load_model)
            .expect("first load builds cache");
        assert!(!first.cache_hit);
        assert!(first.mapping.is_some());
        assert_eq!(builds.load(Ordering::Relaxed), 1);
        assert_eq!(first.model[42], 42.0);
        if let Some(map) = &first.mapping {
            map.prefetch();
        }
        drop(first);

        let second = load(&dir, Some(name.clone()), None, || {}, load_model)
            .expect("second load hits cache");
        assert!(second.cache_hit);
        assert_eq!(
            builds.load(Ordering::Relaxed),
            1,
            "cache hit must map directly without rebuilding"
        );
        assert_eq!(second.model[42], 42.0);
        drop(second);

        // Mutating the source file invalidates the fingerprint and rebuilds.
        std::fs::write(dir.join("model.bin"), b"source-model-bytes-v2").expect("rewrite model");
        let third = load(&dir, Some(name), None, || {}, load_model)
            .expect("third load rebuilds on source change");
        assert!(!third.cache_hit);
        assert_eq!(builds.load(Ordering::Relaxed), 2);
        drop(third);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_index_rejects_duplicate_tags_and_invalid_spans() {
        let dir = temp_model_dir("bad-index");
        let fp = dir_fingerprint(&dir).expect("fingerprint");

        // 1. Duplicate named tag must be rejected.
        let dup_path = dir.join("dup.bin");
        {
            let mut w = Writer::create(&dup_path, fp).expect("create");
            w.add(Some("same_tag"), &[1u8; 64]);
            w.add(Some("same_tag"), &[2u8; 64]);
            w.finish().expect("finish");
        }
        let dup_map = Mapping::open(&dup_path).expect("open dup");
        assert!(
            read_index(&dup_map, &fp).is_none(),
            "duplicate tags must invalidate the cache index"
        );
        drop(dup_map);

        // 2. Payload span overlapping header (`at = 0`) must be rejected even with valid checksums.
        let hdr_path = dir.join("hdr.bin");
        {
            let mut idx = Vec::new();
            idx.extend_from_slice(&1u64.to_le_bytes());
            idx.extend_from_slice(&0u64.to_le_bytes());
            idx.extend_from_slice(&0u64.to_le_bytes()); // at=0 overlaps header
            idx.extend_from_slice(&16u64.to_le_bytes());
            idx.extend_from_slice(&1u32.to_le_bytes());
            idx.extend_from_slice(b"a");
            let payload_sha = payload_digest_parallel(&[]);
            let mut h = Sha256::new();
            h.update(&idx);
            let mut f = File::create(&hdr_path).expect("create");
            f.write_all(MAGIC).expect("magic");
            f.write_all(&HEADER.to_le_bytes()).expect("index_at");
            f.write_all(&fp).expect("fp");
            f.write_all(&idx).expect("idx");
            f.write_all(&payload_sha).expect("payload_sha");
            f.write_all(&h.finalize()).expect("sha");
        }
        let hdr_map = Mapping::open(&hdr_path).expect("open hdr");
        assert!(
            read_index(&hdr_map, &fp).is_none(),
            "payload span overlapping header must be rejected"
        );
        drop(hdr_map);

        // 3. Overlapping payload spans must be rejected even with valid checksums.
        let overlap_path = dir.join("overlap.bin");
        {
            let payload_start = HEADER.next_multiple_of(ALIGN);
            let payload_end = payload_start + 128;
            let payload_bytes = vec![0u8; (payload_end - HEADER) as usize];
            let payload_sha = payload_digest_parallel(&payload_bytes);
            let mut idx = Vec::new();
            idx.extend_from_slice(&2u64.to_le_bytes());
            // First entry: [payload_start .. payload_start + 128)
            idx.extend_from_slice(&0u64.to_le_bytes());
            idx.extend_from_slice(&payload_start.to_le_bytes());
            idx.extend_from_slice(&128u64.to_le_bytes());
            idx.extend_from_slice(&1u32.to_le_bytes());
            idx.extend_from_slice(b"a");
            // Second entry: [payload_start + 64 .. payload_start + 128) overlaps first
            idx.extend_from_slice(&0u64.to_le_bytes());
            idx.extend_from_slice(&(payload_start + 64).to_le_bytes());
            idx.extend_from_slice(&64u64.to_le_bytes());
            idx.extend_from_slice(&1u32.to_le_bytes());
            idx.extend_from_slice(b"b");
            let mut h = Sha256::new();
            h.update(&idx);
            let mut f = File::create(&overlap_path).expect("create");
            f.write_all(MAGIC).expect("magic");
            f.write_all(&payload_end.to_le_bytes()).expect("index_at");
            f.write_all(&fp).expect("fp");
            f.write_all(&payload_bytes).expect("payload");
            f.write_all(&idx).expect("idx");
            f.write_all(&payload_sha).expect("payload_sha");
            f.write_all(&h.finalize()).expect("sha");
        }
        let overlap_map = Mapping::open(&overlap_path).expect("open overlap");
        assert!(
            read_index(&overlap_map, &fp).is_none(),
            "overlapping payload spans must be rejected"
        );
        drop(overlap_map);

        // 4. Corrupted interior byte (byte 300 of a 1,024-byte payload) must force a rebuild.
        let name = "weights-test-corrupt.bin".to_string();
        let cache_file = dir.join(".trit").join(&name);
        let load_w = || {
            if let Some(m) = Store::<f32>::mapped("w0", 256) {
                return Ok(m);
            }
            Ok(Store::named("w0", (0..256).map(|i| i as f32).collect()))
        };
        let built = load(&dir, Some(name.clone()), None, || {}, load_w).expect("build");
        assert_eq!(built.model[75], 75.0);
        drop(built);
        // Flip byte 300 (float index 75, strictly inside the 1,024-byte buffer).
        let mut raw = std::fs::read(&cache_file).expect("read cache");
        let payload_at = HEADER.next_multiple_of(ALIGN) as usize;
        raw[payload_at + 300] ^= 0xFF;
        std::fs::write(&cache_file, &raw).expect("write corrupt cache");
        let repaired = load(&dir, Some(name), None, || {}, load_w).expect("rebuild");
        assert!(
            !repaired.cache_hit,
            "corrupted interior payload byte must invalidate cache hit and rebuild"
        );
        assert_eq!(repaired.model[75], 75.0);
        drop(repaired);

        // 5. Nested shard file change invalidates the directory fingerprint.
        let shard_dir = dir.join("shards");
        std::fs::create_dir_all(&shard_dir).expect("mkdir shards");
        let shard = shard_dir.join("part-0.safetensors");
        std::fs::write(&shard, b"shard-v1").expect("write shard v1");
        let fp1 = dir_fingerprint(&dir).expect("fp1");
        std::fs::write(&shard, b"shard-v2").expect("write shard v2");
        let fp2 = dir_fingerprint(&dir).expect("fp2");
        assert_ne!(
            fp1, fp2,
            "nested shard changes must invalidate the fingerprint"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
