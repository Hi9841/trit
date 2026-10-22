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

/// Freed buffers, kept until the next take needs them.
static POOL: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::new());
/// Enough for every live buffer of one forward pass.
const MAX_POOLED: usize = 64;
/// Bytes the pool may hold. 16 MB against 64 MB, 60 clips: Whisper base.en
/// and Moonshine base keep 20 to 75 MB less, at about 5 % of Whisper's
/// speed; Parakeet Redux runs the same.
const MAX_POOLED_BYTES: usize = 16 << 20;

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
        let reused = {
            let mut pool = POOL.lock();
            // Smallest pooled buffer that fits.
            let best = pool
                .iter()
                .enumerate()
                .filter(|(_, v)| v.capacity() >= len)
                .min_by_key(|(_, v)| v.capacity())
                .map(|(i, _)| i);
            best.map(|i| pool.swap_remove(i))
        };
        let mut data = reused.unwrap_or_else(|| Vec::with_capacity(len));
        if data.len() < len {
            data.resize(len, 0.0);
        }
        Buf { data, len }
    }
}

/// Frees every pooled buffer.
pub fn clear_pool() {
    POOL.lock().clear();
}

impl Drop for Buf {
    fn drop(&mut self) {
        let vec = std::mem::take(&mut self.data);
        if vec.capacity() == 0 {
            return;
        }
        let mut pool = POOL.lock();
        let held: usize = pool.iter().map(|v| v.capacity() * 4).sum();
        if pool.len() < MAX_POOLED && held + vec.capacity() * 4 <= MAX_POOLED_BYTES {
            pool.push(vec);
        }
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
