//! The single cancellation primitive shared across Terra eval jobs.
//!
//! A [`CancelToken`] is a cheap `Clone` handle answering one question —
//! "should the work in flight stop?" — via a single atomic load. It has three
//! shapes:
//!
//! * [`CancelToken::never`] — never cancels (tests, one-off tools).
//! * [`CancelToken::generation`] — cancels when a shared generation counter
//!   moves off an expected value. This is exactly the eval worker's supersede
//!   model: a newer job bumps the counter, and every older job's token trips.
//! * [`CancelToken::flag`] — a one-shot flag flipped by an owned [`CancelFlag`].
//!   Trivial today; the phase-2 export executors need it.
//!
//! Hierarchical child tokens are intentionally absent until an executor needs
//! them — they are not built speculatively.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// A cheap, clonable "should I stop?" handle.
///
/// [`is_cancelled`](CancelToken::is_cancelled) is a single `Acquire` load,
/// matching the ordering the eval context used before this primitive existed.
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Inner);

#[derive(Clone, Debug, Default)]
enum Inner {
    /// Never cancels.
    #[default]
    Never,
    /// One-shot flag flipped by the paired [`CancelFlag`].
    Flag(Arc<AtomicBool>),
    /// Cancels once `shared` no longer equals `expected`.
    Generation {
        shared: Arc<AtomicU64>,
        expected: u64,
    },
}

/// The cancel half of a [`CancelToken::flag`] pair.
///
/// Dropping it does *not* cancel; cancellation is explicit via
/// [`cancel`](CancelFlag::cancel). Clones share the same flag.
#[derive(Clone, Debug)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelToken {
    /// A token that never reports cancellation.
    pub fn never() -> Self {
        Self(Inner::Never)
    }

    /// A token tied to a shared generation counter.
    ///
    /// Reports cancelled once `shared` holds anything other than `expected`.
    /// Mirrors [`crate`]'s worker supersede model, where submitting a newer job
    /// stores a higher generation into the shared counter.
    pub fn generation(shared: Arc<AtomicU64>, expected: u64) -> Self {
        Self(Inner::Generation { shared, expected })
    }

    /// A one-shot token plus the [`CancelFlag`] that trips it.
    pub fn flag() -> (Self, CancelFlag) {
        let flag = Arc::new(AtomicBool::new(false));
        (Self(Inner::Flag(Arc::clone(&flag))), CancelFlag(flag))
    }

    /// Whether the work backed by this token should stop. A single atomic load.
    pub fn is_cancelled(&self) -> bool {
        match &self.0 {
            Inner::Never => false,
            Inner::Flag(flag) => flag.load(Ordering::Acquire),
            Inner::Generation { shared, expected } => shared.load(Ordering::Acquire) != *expected,
        }
    }
}

impl CancelFlag {
    /// Trip the paired token so every holder observes cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_is_never_cancelled() {
        assert!(!CancelToken::never().is_cancelled());
        assert!(!CancelToken::default().is_cancelled());
    }

    #[test]
    fn flag_trips_on_cancel_and_clones_share() {
        let (token, flag) = CancelToken::flag();
        let token2 = token.clone();
        assert!(!token.is_cancelled());
        flag.cancel();
        assert!(token.is_cancelled());
        assert!(token2.is_cancelled());
    }

    #[test]
    fn generation_cancels_once_counter_moves() {
        let shared = Arc::new(AtomicU64::new(7));
        let token = CancelToken::generation(Arc::clone(&shared), 7);
        assert!(!token.is_cancelled());
        shared.store(8, Ordering::Release);
        assert!(token.is_cancelled());
    }
}
