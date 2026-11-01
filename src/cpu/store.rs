//! Model weights in a mapped cache file instead of the heap.
//!
//! A load converts the published weights (f16 to int8, base-3 to 2-bit
//! codes, f16 to f32) into the layouts of the kernels. With
//! `LoadOptions::cache`, the first load also writes each converted buffer to
//! a file in the model folder (`.trit/`), and every later load maps that
//! file. A mapped buffer is file memory: Windows and Linux count it as
//! shared, not private, and can drop its pages under memory pressure and
//! read them back, instead of writing them to the page file.
//!
//! A load still converts every buffer, then keeps the mapped copy only when
//! its bytes are equal to the fresh one. So a stale or damaged file never
//! changes a product; it only costs a rewrite. Keeping the buffers out of the
//! heap also keeps them out of mimalloc, which holds about 120 MB more than
//! the live weights of a Parakeet model when they are allocated on the
//! worker threads.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::ops::Deref;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Values a cache file can hold: no padding, every bit pattern valid.
pub trait Plain: Copy + Send + Sync + 'static {}
impl Plain for f32 {}
impl Plain for u16 {}
impl Plain for i8 {}
impl Plain for u8 {}

/// Smaller buffers stay on the heap: they are a few MB in all.
const MIN_MAPPED_BYTES: usize = 64 << 10;
/// Each buffer starts on a cache line in the file, so mapped values are aligned.
const ALIGN: u64 = 64;
const MAGIC: &[u8; 8] = b"TRITW001";
/// Magic and the offset of the index.
const HEADER: u64 = 16;

/// The weights of one layer: owned, or a view of a mapped cache file.
pub struct Store<T: Plain> {
    inner: Inner<T>,
}

enum Inner<T> {
    Owned(Vec<T>),
    Mapped {
        /// Keeps the view alive.
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
    /// A buffer that stays on the heap (one that grows after the load).
    pub fn owned(values: Vec<T>) -> Self {
        Self {
            inner: Inner::Owned(values),
        }
    }

    /// A model weight. During a load with a cache it is the mapped copy
    /// when the file holds the same bytes, and is written to the file when
    /// the load builds one.
    pub fn weights(values: Vec<T>) -> Self {
        if std::mem::size_of_val(values.as_slice()) < MIN_MAPPED_BYTES {
            return Self::owned(values);
        }
        let mut session = SESSION.lock();
        let Some(session) = session.as_mut() else {
            return Self::owned(values);
        };
        let bytes = as_bytes(&values);
        match &mut session.cache {
            Cache::Read { map, index } => {
                let found = index.get(&key(bytes)).and_then(|spans| {
                    spans
                        .iter()
                        .find(|&&(at, len)| len == bytes.len() && map.bytes(at, len) == bytes)
                });
                match found {
                    Some(&(at, _)) => {
                        let map = Arc::clone(map);
                        let ptr = map.bytes(at, bytes.len()).as_ptr().cast::<T>();
                        let len = values.len();
                        Self {
                            inner: Inner::Mapped {
                                _map: map,
                                ptr,
                                len,
                            },
                        }
                    }
                    None => {
                        session.missed = true;
                        Self::owned(values)
                    }
                }
            }
            Cache::Write(writer) => {
                writer.add(bytes);
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

/// The load in progress. Loads take turns (`LOADS`), so there is one.
static SESSION: Mutex<Option<Session>> = Mutex::new(None);
/// One load at a time, with a cache or not: the buffers of a load without
/// one must not reach the file of another.
static LOADS: Mutex<()> = Mutex::new(());

struct Session {
    cache: Cache,
    /// A buffer of a `Read` load was not in the file.
    missed: bool,
}

enum Cache {
    Read {
        map: Arc<Mapping>,
        /// `key` to (offset, byte length) of the buffers with that key.
        index: HashMap<u64, Vec<(usize, usize)>>,
    },
    Write(Writer),
}

/// Appends buffers to a new cache file; `finish` writes the index.
struct Writer {
    file: std::io::BufWriter<File>,
    at: u64,
    index: Vec<(u64, u64, u64)>,
    /// The first write error; the file is then dropped.
    failed: Option<std::io::Error>,
}

impl Writer {
    fn create(path: &Path) -> std::io::Result<Self> {
        let mut file = std::io::BufWriter::with_capacity(4 << 20, File::create(path)?);
        file.write_all(MAGIC)?;
        file.write_all(&0u64.to_le_bytes())?;
        Ok(Self {
            file,
            at: HEADER,
            index: Vec::new(),
            failed: None,
        })
    }

    fn add(&mut self, bytes: &[u8]) {
        if self.failed.is_some() {
            return;
        }
        let start = self.at.next_multiple_of(ALIGN);
        let padding = [0u8; ALIGN as usize];
        let result = self
            .file
            .write_all(&padding[..(start - self.at) as usize])
            .and_then(|()| self.file.write_all(bytes));
        match result {
            Ok(()) => {
                self.index.push((key(bytes), start, bytes.len() as u64));
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
        self.file
            .write_all(&(self.index.len() as u64).to_le_bytes())?;
        for (key, at, len) in &self.index {
            for value in [key, at, len] {
                self.file.write_all(&value.to_le_bytes())?;
            }
        }
        let mut file = self.file.into_inner().map_err(|e| e.into_error())?;
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::Start(MAGIC.len() as u64))?;
        file.write_all(&index_at.to_le_bytes())?;
        file.sync_all()
    }
}

/// Reads the index of a cache file. `None` for a file that is not one.
fn read_index(map: &Mapping) -> Option<HashMap<u64, Vec<(usize, usize)>>> {
    let word = |at: usize| -> Option<u64> {
        let bytes = map.get(at, 8)?;
        Some(u64::from_le_bytes(bytes.try_into().ok()?))
    };
    if map.get(0, MAGIC.len())? != MAGIC {
        return None;
    }
    let index_at = usize::try_from(word(MAGIC.len())?).ok()?;
    let count = usize::try_from(word(index_at)?).ok()?;
    let mut index: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
    for i in 0..count {
        let entry = index_at + 8 + i * 24;
        let (key, at, len) = (word(entry)?, word(entry + 8)?, word(entry + 16)?);
        let (at, len) = (usize::try_from(at).ok()?, usize::try_from(len).ok()?);
        map.get(at, len)?;
        index.entry(key).or_default().push((at, len));
    }
    Some(index)
}

/// Progress of the load in progress: bytes read from the weight files.
struct Progress {
    report: Arc<dyn Fn(f32) + Send + Sync>,
    /// The fraction of the whole load that this pass covers: (start, width).
    span: (f32, f32),
    total: AtomicU64,
    done: AtomicU64,
    /// The last value reported, in thousandths.
    shown: AtomicU64,
}

static PROGRESS: Mutex<Option<Arc<Progress>>> = Mutex::new(None);

/// A weight file of `bytes` opens for the load in progress.
pub(crate) fn will_read(bytes: u64) {
    if let Some(progress) = PROGRESS.lock().as_ref() {
        progress.total.fetch_add(bytes, Ordering::Relaxed);
    }
}

/// `bytes` of a weight file were read.
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

/// One load pass with `cache` as its session. Returns the model and the
/// session, for its writer or its misses.
fn pass<R>(
    cache: Option<Cache>,
    report: Option<&Arc<dyn Fn(f32) + Send + Sync>>,
    span: (f32, f32),
    load: impl FnOnce() -> R,
) -> (R, Option<Session>) {
    *SESSION.lock() = cache.map(|cache| Session {
        cache,
        missed: false,
    });
    *PROGRESS.lock() = report.map(|report| {
        Arc::new(Progress {
            report: Arc::clone(report),
            span,
            total: AtomicU64::new(0),
            done: AtomicU64::new(0),
            shown: AtomicU64::new(0),
        })
    });
    let result = load();
    *PROGRESS.lock() = None;
    (result, SESSION.lock().take())
}

/// Runs a load that keeps no cache, in its turn: its buffers must not reach
/// the file of a cached load that runs at the same time.
pub(crate) fn in_turn<R>(load: impl FnOnce() -> R) -> R {
    let _turn = LOADS.lock();
    load()
}

/// A `Read` cache of the file at `path`, when it is one.
fn reader(path: &Path) -> Option<Cache> {
    let map = Arc::new(Mapping::open(path).ok()?);
    let index = read_index(&map)?;
    Some(Cache::Read { map, index })
}

/// Loads with `load`, keeping the converted weights in the cache file
/// `dir/.trit/<name>` when `name` is given. `report` gets the progress from
/// 0 to 1. `release` returns freed memory between passes. A cache error
/// gives the plain load: the cache only moves memory, it never fails a load.
pub(crate) fn load<M>(
    dir: &Path,
    name: Option<String>,
    report: Option<Arc<dyn Fn(f32) + Send + Sync>>,
    release: impl Fn(),
    load: impl Fn() -> Result<M, String>,
) -> Result<M, String> {
    let _turn = LOADS.lock();
    let report = report.as_ref();
    let finished = |model: Result<M, String>| {
        if let (Some(report), Ok(_)) = (report, &model) {
            report(1.0);
        }
        model
    };
    let Some(name) = name else {
        return finished(pass(None, report, (0.0, 1.0), &load).0);
    };
    let folder = dir.join(".trit");
    let path = folder.join(&name);
    // A file from an earlier load: keep the buffers it holds.
    if let Some(cache) = reader(&path) {
        let (model, session) = pass(Some(cache), report, (0.0, 1.0), &load);
        if model.is_err() || !session.is_some_and(|s| s.missed) {
            return finished(model);
        }
        // The weights changed: build the file again.
        drop(model);
        release();
    }
    let temporary = folder.join(format!("{name}.{}.tmp", std::process::id()));
    let Ok(writer) = std::fs::create_dir_all(&folder).and_then(|()| Writer::create(&temporary))
    else {
        return finished(pass(None, report, (0.0, 1.0), &load).0);
    };
    let (model, session) = pass(Some(Cache::Write(writer)), report, (0.0, 0.6), &load);
    let written = match session.map(|s| s.cache) {
        Some(Cache::Write(writer)) => writer.finish(),
        _ => Err(std::io::ErrorKind::Other.into()),
    }
    // Another process may map the old file; it then stays, and this
    // process keeps its own buffers.
    .and_then(|()| std::fs::rename(&temporary, &path));
    if written.is_err() || model.is_err() {
        let _ = std::fs::remove_file(&temporary);
        return finished(model);
    }
    remove_stale(&folder, &name);
    // Load once more from the new file, so this process maps it too.
    drop(model);
    release();
    let cache = reader(&path);
    finished(pass(cache, report, (0.6, 0.4), &load).0)
}

/// Removes the cache files of other Trit versions and kernel paths.
fn remove_stale(folder: &Path, keep: &str) {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name != keep && name.starts_with("weights-") && name.ends_with(".bin") {
            // A file that another process maps stays until it closes it.
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// The cache file name for a load: the Trit version, the precision, and the
/// kernel paths, which set the layouts.
pub(crate) fn file_name(precision: &str) -> String {
    let path: String = super::cpu_path()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    format!(
        "weights-{}-{precision}-{path}.bin",
        env!("CARGO_PKG_VERSION")
    )
}

/// A read-only view of a whole file.
struct Mapping {
    ptr: *const u8,
    len: usize,
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
