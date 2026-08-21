//! App-owned lifecycle for optional Medium/Full GPU refinement.

use std::time::Instant;

use terra_core::eval::PreviewQuality;
use terra_gpu_eval::GpuRefinementJob as EngineRefinementJob;

use super::frame_trace::EvaluationTraceId;
use super::logical_frame::{EditGeneration, FrameIdentity};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RefinementPublicationState {
    Preparing,
    SubmissionInFlight,
    ReadyToPublish,
}

pub(crate) struct RefinementJob {
    pub(crate) id: u64,
    pub(crate) origin: FrameIdentity,
    pub(crate) generation: EditGeneration,
    pub(crate) target_quality: PreviewQuality,
    pub(crate) evaluation: EvaluationTraceId,
    pub(crate) engine: EngineRefinementJob,
    pub(crate) publication: RefinementPublicationState,
    pub(crate) started_at: Instant,
}

impl RefinementJob {
    pub(crate) fn is_fresh(&self, generation: u64) -> bool {
        self.generation.get() == generation
    }
}
