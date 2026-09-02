//! Public evaluation results and resumable-refinement lifecycle types.

use std::collections::HashMap;
use std::sync::mpsc;

use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{LayerId, LayerStack};
use terra_core::mask::MaskAsset;
use terra_core::quality::PreviewQuality;
use terra_core::terrain_plan::{
    CompiledTerrainPlan, PlanInvalidation, PlanOpId, PlanStructureRevision,
};
use terra_gpu::compiled_plan::{GpuPlanResourceBuilder, GpuPlanResources};
use terra_gpu::graph::{GpuComputeGraph, GpuFallbackDiagnostic, GpuLayerPlan};
use terra_gpu::output_identity::GpuTerrainOutputIdentity;

pub use terra_gpu::output_identity::GpuEvaluationIntent;

/// Result of a GPU preview evaluation.
pub struct GpuEvalResult {
    pub width: u32,
    pub height: u32,
    pub world_size: (f32, f32),
    pub height_range: (f32, f32),
    pub fully_gpu: bool,
    /// Whether the visible texture is the complete stack or the truthful local
    /// prefix produced while a globally coupled suffix waits for refinement.
    pub freshness: GpuPreviewFreshness,
    pub cpu: Option<Heightfield>,
    /// First flattened layer that must resume on the CPU. When this is `Some(n)`,
    /// `cpu` is the height entering layer `n`; `Some(0)` is a full-CPU restart seed.
    pub resume_cpu_from: Option<usize>,
    /// Structured planner/runtime reason corresponding to `resume_cpu_from`.
    pub cpu_fallback: Option<GpuFallbackDiagnostic>,
    /// True when the evaluate loop ran (filters may have been applied). False on seed failure.
    pub did_eval: bool,
    /// Correlated semantic/resource identity for the engine texture. `None` is
    /// reserved for failures that produced no presentable GPU candidate.
    pub output_identity: Option<GpuTerrainOutputIdentity>,
}

/// Required-GPU numerical diagnostics for simulation scratch state.
///
/// This is feature-gated because interactive production evaluation must not add
/// synchronous readbacks. The C3 simulation guard enables `gpu-parity` and reads
/// these fields only after a submitted evaluation has completed.
#[cfg(feature = "gpu-parity")]
pub struct GpuSimulationStateReadback {
    /// Raw hydraulic invariant violations observed before final stability clamps.
    pub invalid_state_bits: u32,
    pub hardness: Heightfield,
    pub water_a: Heightfield,
    pub water_b: Heightfield,
    pub sediment_a: Heightfield,
    pub sediment_b: Heightfield,
    pub redistribution: Heightfield,
    pub rainfall: Heightfield,
    pub loose_sediment: Heightfield,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPreviewFreshness {
    Current,
    Deferred {
        from_index: usize,
        from_layer: LayerId,
        deferred_layers: usize,
    },
}

impl GpuPreviewFreshness {
    pub fn is_deferred(self) -> bool {
        matches!(self, Self::Deferred { .. })
    }
}

/// Per-evaluation execution intent. This is deliberately not a user-facing
/// policy surface: the editor has one default behavior for bounded local edits.
/// Observable phase of an optional, resumable compiled-plan refinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuRefinementPhase {
    PreparingResources,
    Encoding,
    SubmissionInFlight,
    ReadyToPublish,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuRefinementProgress {
    pub phase: GpuRefinementPhase,
    pub completed_units: usize,
    pub total_units: usize,
    pub submissions_issued: u32,
    pub submissions_completed: u32,
    pub submissions_in_flight: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuRefinementStep {
    Progressed,
    Submitted { ordinal: u32 },
    AwaitingGpu,
    ReadyToPublish,
}

/// Isolated candidate and execution cursor for optional Medium/Full work.
/// Authored inputs are cloned at creation, so later edits can supersede the job
/// by dropping it without changing the last committed plan resources.
pub struct GpuRefinementJob {
    pub(super) stack: LayerStack,
    pub(super) masks: Vec<MaskAsset>,
    pub(super) graph: GpuComputeGraph,
    pub(super) plan: CompiledTerrainPlan,
    pub(super) expected_revision: PlanStructureRevision,
    pub(super) invalidation: PlanInvalidation,
    pub(super) metrics: HeightfieldMetrics,
    pub(super) quality: PreviewQuality,
    pub(super) builder: Option<GpuPlanResourceBuilder>,
    pub(super) candidate: Option<GpuPlanResources>,
    pub(super) selected: Vec<PlanOpId>,
    pub(super) kernels: HashMap<PlanOpId, GpuLayerPlan>,
    pub(super) published_output_slots:
        HashMap<terra_core::ids::OutputId, terra_core::terrain_plan::FieldSlot>,
    pub(super) cursor: usize,
    pub(super) last_height: Option<terra_core::terrain_plan::FieldSlot>,
    pub(super) completion: Option<mpsc::Receiver<()>>,
    pub(super) final_copy_submitted: bool,
    pub(super) final_copy_complete: bool,
    pub(super) submissions_issued: u32,
    pub(super) submissions_completed: u32,
    pub(super) resource_prepare_us: u64,
    pub(super) encode_us: u64,
    pub(super) range_before: (f32, f32),
    pub(super) trace_context: crate::evaluation_timing::GpuEvaluationTraceContext,
    pub(super) final_submission_serial: terra_gpu::output_identity::GpuSubmissionSerial,
}

impl GpuRefinementJob {
    pub fn quality(&self) -> PreviewQuality {
        self.quality
    }

    pub fn progress(&self) -> GpuRefinementProgress {
        let allocations = self.builder.as_ref().map_or_else(
            || self.candidate.as_ref().map_or(0, |_| self.resource_units()),
            |b| b.completed_allocations(),
        );
        let completed = allocations
            .saturating_add(self.cursor)
            .saturating_add(usize::from(self.final_copy_complete));
        let phase = if self.final_copy_complete {
            GpuRefinementPhase::ReadyToPublish
        } else if self.completion.is_some() {
            GpuRefinementPhase::SubmissionInFlight
        } else if self.builder.is_some() {
            GpuRefinementPhase::PreparingResources
        } else {
            GpuRefinementPhase::Encoding
        };
        GpuRefinementProgress {
            phase,
            completed_units: completed,
            total_units: self.resource_units() + self.selected.len() + 1,
            submissions_issued: self.submissions_issued,
            submissions_completed: self.submissions_completed,
            submissions_in_flight: u8::from(self.completion.is_some()),
        }
    }

    fn resource_units(&self) -> usize {
        self.builder
            .as_ref()
            .map(GpuPlanResourceBuilder::total_allocations)
            .or_else(|| {
                self.candidate
                    .as_ref()
                    .map(|candidate| candidate.layout().allocations().len())
            })
            .unwrap_or(0)
    }
}
