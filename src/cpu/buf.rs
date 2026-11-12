//! Reused f32 buffers. A forward pass allocates and frees buffers of the
//! same few sizes in every layer. Fresh large allocations come from Windows
//! as untouched pages, and the page faults on first write stall all threads.
//! A buffer goes back to the pool when it is dropped.
//!
//! Most buffers are outputs that a parallel step writes in full. Zeroing them
//! first ran on the calling thread alone, about a tenth of a take, so those
//! take `scratch` buffers, which keep the stale values of their last use.

use parking_lot::Mutex;
use std::ops::{Deref, DerefMut};

/// The buffers and the bytes that the pool may hold.
#[derive(Clone, Copy)]
struct Limit {
    buffers: usize,
    bytes: usize,
}

/// While no call runs. 64 buffers are enough for every live buffer of one
/// forward pass. 16 MB against 64 MB, 60 clips: Whisper base.en and
/// Moonshine base keep 20 to 75 MB less.
const IDLE: Limit = Limit {
    buffers: 64,
    bytes: 16 << 20,
};
/// While a call runs. A pass of 1,500 frames uses buffers of 2 to 9 MB
/// (Whisper tiny), more than the idle limit holds: each one that the pool
/// did not keep came back as a new buffer, filled with zeros on the calling
/// thread (5.6 % of a clip). 64 MB and not more: on 52 minutes of Parakeet
/// the time is the same with 64, 128, and 256 MB (0.94, 0.91, 0.95 of the
/// time with 16 MB), and the peak private memory is 8, 10, and 14 % higher.
const IN_CALL: Limit = Limit {
    buffers: 256,
    bytes: 64 << 20,
};

/// Freed buffers, kept until the next pass needs them.
struct Pool {
    /// The smallest capacity first: a request finds its buffer by a binary
    /// search. A look at every buffer for each request and each drop cost
    /// Moonshine tiny 6 % with 256 buffers in the pool.
    free: Vec<Vec<f32>>,
    /// The bytes of `free`.
    bytes: usize,
    /// Calls in progress (see `in_call`).
    calls: usize,
}

static POOL: Mutex<Pool> = Mutex::new(Pool {
    free: Vec::new(),
    bytes: 0,
    calls: 0,
});

impl Pool {
    /// The smallest buffer that holds `len` values.
    fn take(&mut self, len: usize) -> Option<Vec<f32>> {
        let at = self.free.partition_point(|buffer| buffer.capacity() < len);
        (at < self.free.len()).then(|| {
            let buffer = self.free.remove(at);
            self.bytes -= buffer.capacity() * 4;
            buffer
        })
    }

    /// Keeps `buffer` when the limit of this moment has room for it.
    fn keep(&mut self, buffer: Vec<f32>) {
        let limit = if self.calls == 0 { IDLE } else { IN_CALL };
        let size = buffer.capacity() * 4;
        if self.free.len() < limit.buffers && self.bytes + size <= limit.bytes {
            let at = self
                .free
                .partition_point(|kept| kept.capacity() < buffer.capacity());
            self.free.insert(at, buffer);
            self.bytes += size;
        }
    }

    /// Back under `limit`. The largest go first: the most memory back for
    /// each buffer lost.
    fn trim(&mut self, limit: Limit) {
        while self.free.len() > limit.buffers || self.bytes > limit.bytes {
            self.bytes -= self.free.pop().map_or(0, |buffer| buffer.capacity() * 4);
        }
    }
}

/// A call that runs: the pool keeps the buffers of its passes until it ends.
#[derive(Debug)]
pub struct InCall;

/// From now until the value drops, the pool keeps what a pass frees. When
/// the last call ends, the pool goes back under its idle limits.
pub fn in_call() -> InCall {
    POOL.lock().calls += 1;
    InCall
}

impl Drop for InCall {
    fn drop(&mut self) {
        // The count changes under the pool's lock, and a buffer that drops
        // reads it under the same lock: no buffer enters the pool with the
        // limits of a call after this trim.
        let mut pool = POOL.lock();
        pool.calls -= 1;
        if pool.calls == 0 {
            pool.trim(IDLE);
        }
    }
}

/// `data` keeps every value it ever held, so reuse never fills it again;
/// the buffer is its first `len` values.
pub struct Buf {
    data: Vec<f32>,
    len: usize,
}

impl Buf {
    /// `len` zeros, in memory that a previous buffer already touched when one fits.
    pub fn zeroed(len: usize) -> Self {
        let mut buf = Self::scratch(len);
        buf.fill(0.0);
        buf
    }

    /// `len` values left from an earlier use (zeros where there was none),
    /// for an output that is written in full before it is read.
    pub fn scratch(len: usize) -> Self {
        let reused = POOL.lock().take(len);
        let mut data = reused.unwrap_or_else(|| Vec::with_capacity(stepped(len)));
        if data.len() < len {
            let _span = crate::prof::span(crate::prof::Stage::BufFill);
            data.resize(len, 0.0);
        }
        Buf { data, len }
    }
}

/// `len` rounded up to a step of one eighth of its power of two: at most
/// 12.5 % more. The windows of a long recording have many lengths, and a
/// buffer of exactly one window's size was of no use to a longer window:
/// the pool filled with a set of buffers for each length (peak memory of
/// Parakeet 15 % up on 161 minutes of audio). With steps, windows of nearly
/// the same length ask for the same capacity.
fn stepped(len: usize) -> usize {
    /// Below this a buffer is cheap, and a step would be most of it.
    const SMALL: usize = 1 << 12;
    /// From 16 MB a buffer does not stay in the pool with others of its
    /// size, and a step of it is megabytes: its own size.
    const LARGE: usize = 4 << 20;
    if !(SMALL..LARGE).contains(&len) {
        return len;
    }
    let step = 1usize << (len.ilog2() - 3);
    len.next_multiple_of(step)
}

impl Buf {
    pub fn into_vec(mut self) -> Vec<f32> {
        let mut values = std::mem::take(&mut self.data);
        values.truncate(self.len);
        values
    }
}

pub fn clear_pool() {
    let mut pool = POOL.lock();
    pool.free.clear();
    pool.bytes = 0;
}

impl Drop for Buf {
    fn drop(&mut self) {
        let vec = std::mem::take(&mut self.data);
        if vec.capacity() == 0 {
            return;
        }
        POOL.lock().keep(vec);
    }
}

impl Deref for Buf {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.data[..self.len]
    }
}

impl DerefMut for Buf {
    fn deref_mut(&mut self) -> &mut [f32] {
        &mut self.data[..self.len]
    }
}
