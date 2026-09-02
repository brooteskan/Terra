//! Cancellable parallel fill.
//!
//! [`try_par_fill`] writes each slot of an output slice from a pure index
//! function, in parallel over row-length (or otherwise chunked) spans, checking
//! the [`CancelToken`] once per chunk. Returning `Err` from a chunk makes rayon
//! stop scheduling the remaining chunks, so cancel latency is bounded by roughly
//! one in-flight chunk per worker rather than the whole fill.

use crate::CancelToken;
use rayon::prelude::*;

/// Fill `out` in parallel, one chunk of `chunk_len` at a time, checking `token`
/// once per chunk.
///
/// `f(idx)` is called for every global index `idx` in `0..out.len()` and must be
/// a pure function of `idx` (it runs across rayon workers in unspecified order).
/// Returns `true` when the whole slice was filled, `false` if `token` reported
/// cancellation before completion — in which case `out` is left partially
/// written and should be discarded.
///
/// Check granularity is one chunk: pass the row length so a heightfield checks
/// once per row, never per element. `chunk_len` is clamped up to 1 so a zero
/// never stalls the split.
#[must_use]
pub fn try_par_fill<T: Send>(
    token: &CancelToken,
    out: &mut [T],
    chunk_len: usize,
    f: impl Fn(usize) -> T + Sync,
) -> bool {
    let chunk_len = chunk_len.max(1);
    out.par_chunks_mut(chunk_len)
        .enumerate()
        .try_for_each(|(chunk_index, chunk)| {
            if token.is_cancelled() {
                return Err(());
            }
            let base = chunk_index * chunk_len;
            for (offset, slot) in chunk.iter_mut().enumerate() {
                *slot = f(base + offset);
            }
            Ok(())
        })
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn uncancelled_fill_writes_every_slot() {
        let mut out = vec![0usize; 10_000];
        let done = try_par_fill(&CancelToken::never(), &mut out, 128, |idx| idx * 2);
        assert!(done);
        for (idx, value) in out.iter().enumerate() {
            assert_eq!(*value, idx * 2);
        }
    }

    #[test]
    fn ragged_last_chunk_is_indexed_correctly() {
        // len not a multiple of chunk_len exercises the short final chunk.
        let mut out = vec![0usize; 1003];
        let done = try_par_fill(&CancelToken::never(), &mut out, 64, |idx| idx + 1);
        assert!(done);
        assert_eq!(out[1002], 1003);
    }

    #[test]
    fn pre_cancelled_token_returns_false_immediately() {
        let (token, flag) = CancelToken::flag();
        flag.cancel();
        let mut out = vec![0usize; 10_000];
        let done = try_par_fill(&token, &mut out, 128, |idx| idx);
        assert!(!done);
    }

    #[test]
    fn mid_fill_cancel_stops_scheduling_remaining_chunks() {
        // Timing-free proof: a cancel tripped partway through must leave many
        // untouched chunks, because at most one in-flight chunk per worker can
        // run past the cancel. With 8192 chunks and a normal worker pool that is
        // a tiny fraction, so counting written chunks well under the total is a
        // generous, deterministic assertion.
        const CHUNKS: usize = 8192;
        const CHUNK_LEN: usize = 64;
        let sentinel = usize::MAX;
        let mut out = vec![sentinel; CHUNKS * CHUNK_LEN];

        let (token, flag) = CancelToken::flag();
        let flag = Arc::new(flag);
        let cancelled = Arc::new(AtomicBool::new(false));
        let visited = Arc::new(AtomicUsize::new(0));

        let f = {
            let flag = Arc::clone(&flag);
            let cancelled = Arc::clone(&cancelled);
            let visited = Arc::clone(&visited);
            move |idx: usize| {
                // Trip the flag once, early, from whichever worker reaches the
                // trigger index first.
                if idx >= 100
                    && cancelled
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                {
                    flag.cancel();
                }
                visited.fetch_add(1, Ordering::Relaxed);
                idx
            }
        };

        let done = try_par_fill(&token, &mut out, CHUNK_LEN, f);
        assert!(!done, "fill should report cancellation");

        let written = out.iter().filter(|v| **v != sentinel).count();
        assert!(
            written < CHUNKS * CHUNK_LEN / 2,
            "cancel should skip most work, wrote {written} of {}",
            CHUNKS * CHUNK_LEN
        );
    }
}
