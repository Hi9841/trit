//! `trit`: the command line of the Trit speech engine, built for agents
//! (https://axi.md): TOON on stdout, errors on stdout with the fix, exit
//! 0 / 1 / 2 for success / error / usage error, no prompts.

mod cli;

#[cfg(not(feature = "profile"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// A profile build counts each allocation of the process (`trit::prof`).
#[cfg(feature = "profile")]
#[global_allocator]
static GLOBAL: Counted = Counted;

#[cfg(feature = "profile")]
struct Counted;

#[cfg(feature = "profile")]
// SAFETY: every call goes to mimalloc as it came; the counters are atomics.
unsafe impl std::alloc::GlobalAlloc for Counted {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        trit::prof::allocated(layout.size());
        // SAFETY: the caller's contract is the contract of mimalloc.
        unsafe { mimalloc::MiMalloc.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        trit::prof::allocated(layout.size());
        // SAFETY: as for `alloc`.
        unsafe { mimalloc::MiMalloc.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: as for `alloc`.
        unsafe { mimalloc::MiMalloc.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        trit::prof::allocated(size);
        // SAFETY: as for `alloc`.
        unsafe { mimalloc::MiMalloc.realloc(pointer, layout, size) }
    }
}

fn main() -> std::process::ExitCode {
    cli::run()
}
