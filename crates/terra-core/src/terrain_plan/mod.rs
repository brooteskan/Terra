//! Backend-neutral compiled terrain evaluation plan.
//!
//! [`crate::layer::LayerStack`] remains the persisted, editable authoring model.
//! A `CompiledTerrainPlan` is a runtime-derived projection: it owns only logical
//! dataflow descriptors and authored provenance. The document continues to own
//! layer, mask, Base-raster, and stroke payloads; CPU/GPU backends own physical
//! fields and execution resources.
//!
//! Plan-local [`PlanOpId`] and [`FieldSlot`] values are meaningful only together
//! with the plan's [`PlanStructureRevision`]. Stable cross-plan identity comes
//! from [`crate::deps::NodeRef`], [`crate::ids::LayerId`], and
//! [`crate::ids::OutputId`].

mod analysis;
mod cache;
mod compiler;
mod dirty;
mod ids;
mod impact;
mod ir;
mod provenance;

pub use analysis::{ExpectedFieldKind, PlanAnalysis, PlanFieldLifetime};
pub use cache::{PlanCacheStats, PlanCacheStatsSnapshot, PlanRevisionError, TerrainPlanCache};
pub use compiler::{compile_terrain_plan, TerrainPlanDiagnostic};
pub use dirty::{
    propagate_plan_edits, FullFieldEscalation, FullFieldReason, PlanDirtyOperation,
    PlanInvalidation,
};
pub use ids::{
    FieldSlot, PlanOpId, PlanStructureRevision, PlanStructureSignature, TerrainPlanStamp,
};
pub use impact::{
    PlanDirtyScope, PropagatedDirtyScope, TerrainEditClass, TerrainPlanPatch, TerrainPlanWork,
};
pub use ir::{
    CompiledTerrainPlan, GroupAuxComposite, GroupCompositeMode, LogicalField, LogicalFieldKind,
    PlanBuildError, PlanOrigin, SeedSource, TerrainOp, TerrainOpKind, TerrainPlanBuilder,
};
pub use provenance::{
    OutputProvenance, PlanAuthoredDependency, PlanNodeSelection, PlanOpSpan, PlanProvenance,
};
