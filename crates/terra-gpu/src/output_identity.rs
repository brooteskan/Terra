//! Stable identities carried from GPU evaluation through terrain presentation.
//!
//! These types describe semantic output and resource lineage only. They do not
//! decide whether a candidate may be published; the renderer's transition
//! validator consumes them in diagnostic/shadow mode.

use terra_core::quality::PreviewQuality;
use terra_core::terrain_plan::FieldSlot;
use terra_core::tiling::SampleRect;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct GpuOutputId(pub u64);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct GpuResourceIncarnation(pub u64);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct GpuSubmissionSerial(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuOutputSlot {
    Ping,
    Pong,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSubmissionCompletion {
    Submitted,
    KnownComplete,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GpuEvaluationIntent {
    InteractiveLocal,
    #[default]
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuOutputCoverage {
    WholeField,
    Patch {
        rect: SampleRect,
        expected_base: Option<GpuOutputId>,
    },
}

impl GpuOutputCoverage {
    pub const fn rect(self) -> Option<SampleRect> {
        match self {
            Self::WholeField => None,
            Self::Patch { rect, .. } => Some(rect),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuOutputCompleteness {
    Complete,
    DeferredSuffix,
    HybridPrefix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuInvalidationKind {
    None,
    Regional,
    FullField,
    Cold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuSelectedFieldIdentity {
    pub selected: FieldSlot,
    pub expected_final: FieldSlot,
    pub resource_incarnation: GpuResourceIncarnation,
    pub physical_allocation: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuOutputResourceIdentity {
    pub device_generation: u64,
    pub incarnation: GpuResourceIncarnation,
    pub slot: GpuOutputSlot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuLastWriteIdentity {
    pub serial: GpuSubmissionSerial,
    pub completion: GpuSubmissionCompletion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuTerrainOutputIdentity {
    pub output: GpuOutputId,
    pub frame_id: u64,
    pub generation: u64,
    pub evaluation_id: u64,
    pub plan_revision: u64,
    pub requested_quality: PreviewQuality,
    pub actual_quality: PreviewQuality,
    pub intent: GpuEvaluationIntent,
    pub selected_field: GpuSelectedFieldIdentity,
    pub output_resource: GpuOutputResourceIdentity,
    pub extent: (u32, u32),
    pub coverage: GpuOutputCoverage,
    pub completeness: GpuOutputCompleteness,
    pub invalidation: GpuInvalidationKind,
    pub last_write: GpuLastWriteIdentity,
}

impl GpuTerrainOutputIdentity {
    pub fn is_current_complete_final(self) -> bool {
        self.selected_field.selected == self.selected_field.expected_final
            && matches!(self.completeness, GpuOutputCompleteness::Complete)
    }
}
