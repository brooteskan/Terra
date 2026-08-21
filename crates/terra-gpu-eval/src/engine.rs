//! GPU layer-stack evaluation façade.
//!
//! Public lifecycle and result types live beside the stateful runtime so
//! callers do not depend on its internal module layout.

mod api;
mod runtime;
mod stats;

#[cfg(feature = "gpu-parity")]
pub use api::GpuSimulationStateReadback;
pub use api::{
    GpuEvalResult, GpuEvaluationIntent, GpuPreviewFreshness, GpuRefinementJob, GpuRefinementPhase,
    GpuRefinementProgress, GpuRefinementStep,
};
pub use runtime::GpuTerrainEngine;
pub use stats::{GpuEvalStats, GpuPlanOperationDisposition, GpuPlanOperationTrace};
