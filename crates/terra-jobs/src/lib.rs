//! Terra job primitives: a single cancellation token and cancellable parallel
//! helpers built on it.
//!
//! Phase 1 (issue #101) is deliberately small — one [`CancelToken`] and
//! [`try_par_fill`]. The executors, handles, and registry that later phases add
//! grow from these two pieces; the crate depends only on `rayon` so it stays a
//! leaf below `terra-core`.

mod cancel;
mod fill;

pub use cancel::{CancelFlag, CancelToken};
pub use fill::try_par_fill;
