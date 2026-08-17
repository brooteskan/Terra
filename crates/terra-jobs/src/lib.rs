//! Terra job primitives: a cancellation token, cancellable parallel helpers, and
//! one-shot background jobs built on them.
//!
//! Phase 1 (issue #101) added the [`CancelToken`] and [`try_par_fill`]. Phase 2
//! (issue #102) adds [`spawn_one_shot`] and [`JobHandle`] — a named worker thread
//! with panic containment by construction, atomic progress, and cancellation —
//! the job unit the export, project-IO, and boot subsystems migrate onto. Phase 3
//! (issue #103) adds [`LatestWins`], a persistent-state, latest-generation-wins
//! executor that terra-core's `EvalWorker` is re-implemented over. Phase 4 (issue
//! #104) adds [`Pool`], a fixed-size worker pool, and [`Debounced`], a coalescing
//! latest-value-wins worker — the tool-thumbnail decoder and the editor-prefs
//! saver migrate onto them. Phase 5 (issue #105) adds [`JobRegistry`], a per-frame
//! poll registry that pumps every registered [`Pollable`] subsystem once and
//! aggregates their wakefulness, collapsing the winit loop's scattered
//! per-subsystem polling into a single tick. The crate depends only on `rayon`, so
//! it stays a leaf below `terra-core`.

mod cancel;
mod debounced;
mod fill;
mod job;
mod latest_wins;
mod pool;
mod registry;

pub use cancel::{CancelFlag, CancelToken};
pub use debounced::Debounced;
pub use fill::try_par_fill;
pub use job::{spawn_one_shot, JobCtx, JobError, JobHandle};
pub use latest_wins::{JobEvent, LatestWins, SubmitError};
pub use pool::Pool;
pub use registry::{JobRegistry, Pending, Pollable, Tick};
