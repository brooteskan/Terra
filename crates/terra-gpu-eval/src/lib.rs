//! Stateful GPU terrain evaluation built on the reusable primitives in
//! [`terra_gpu`].

pub mod engine;
mod evaluation_timing;

pub use engine::{
    GpuEvalResult, GpuEvalStats, GpuEvaluationIntent, GpuPreviewFreshness, GpuRefinementJob,
    GpuRefinementPhase, GpuRefinementProgress, GpuRefinementStep, GpuTerrainEngine,
};
pub use evaluation_timing::{GpuEvaluationTiming, GpuEvaluationTraceContext};
