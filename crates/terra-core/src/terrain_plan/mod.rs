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

mod ids;
mod impact;
mod ir;
mod provenance;

pub use ids::{FieldSlot, PlanOpId, PlanStructureRevision, TerrainPlanStamp};
pub use impact::{PlanDirtyScope, TerrainEditClass, TerrainPlanWork};
pub use ir::{
    CompiledTerrainPlan, GroupCompositeMode, LogicalField, LogicalFieldKind, PlanBuildError,
    PlanOrigin, SeedSource, TerrainOp, TerrainOpKind, TerrainPlanBuilder,
};
pub use provenance::PlanProvenance;
