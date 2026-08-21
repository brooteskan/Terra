//! GPU layer-stack evaluation façade.
//!
//! Public lifecycle and result types live beside the stateful runtime so
//! callers do not depend on its internal module layout.

mod runtime;

pub use runtime::{
    GpuEvalResult, GpuEvalStats, GpuEvaluationIntent, GpuPlanOperationDisposition,
    GpuPlanOperationTrace, GpuPreviewFreshness, GpuRefinementJob, GpuRefinementPhase,
    GpuRefinementProgress, GpuRefinementStep, GpuTerrainEngine,
};
