//! Process-global count of GPU shader modules compiled.
//!
//! The render and GPU crates bump this once per shader module they build; the
//! app's startup splash reads it to show a live "compiling shaders…" count while
//! pipelines compile on the boot worker thread. Plain atomics only — terra-core
//! stays free of `wgpu`, so the counter lives here where every GPU crate can
//! reach it without a dependency edge between them.
//!
//! It is a coarse boot diagnostic, not a synchronization primitive: `Relaxed`
//! ordering is sufficient because nothing gates on the exact value.

use std::sync::atomic::{AtomicU32, Ordering};

static COMPILED: AtomicU32 = AtomicU32::new(0);

/// Record that one shader module finished (or began) compiling.
pub fn record_shader_compiled() {
    COMPILED.fetch_add(1, Ordering::Relaxed);
}

/// Shader modules compiled so far this process.
pub fn shaders_compiled() -> u32 {
    COMPILED.load(Ordering::Relaxed)
}

/// Reset the counter — call at the start of a boot so a re-`resumed()` (or a test)
/// starts from zero.
pub fn reset() {
    COMPILED.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_resets() {
        reset();
        assert_eq!(shaders_compiled(), 0);
        record_shader_compiled();
        record_shader_compiled();
        assert_eq!(shaders_compiled(), 2);
        reset();
        assert_eq!(shaders_compiled(), 0);
    }
}
