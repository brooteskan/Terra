//! Terra job primitives: a cancellation token, cancellable parallel helpers, and
//! one-shot background jobs built on them.
//!
//! Phase 1 (issue #101) added the [`CancelToken`] and [`try_par_fill`]. Phase 2
//! (issue #102) adds [`spawn_one_shot`] and [`JobHandle`] — a named worker thread
//! with panic containment by construction, atomic progress, and cancellation —
//! the job unit the export, project-IO, and boot subsystems migrate onto. The
//! crate depends only on `rayon`, so it stays a leaf below `terra-core`.

mod cancel;
mod fill;
mod job;

pub use cancel::{CancelFlag, CancelToken};
pub use fill::try_par_fill;
pub use job::{spawn_one_shot, JobCtx, JobError, JobHandle};
