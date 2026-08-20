//! Physical wgpu realization and field-addressable operations for
//! [`terra_core::terrain_plan::CompiledTerrainPlan`].
//!
//! This module deliberately does not walk an authored `LayerStack`. Issue #144
//! wires these resources and operations into `GpuTerrainEngine`; keeping the
//! realization independent here prevents a second planning authority.

mod operations;
mod resources;

pub use operations::{GpuGroupCompositeParams, GpuPlanOperationError, GpuPlanOperations};
pub use resources::{
    GpuFieldBinding, GpuFieldResidency, GpuPhysicalFieldId, GpuPlanAllocation,
    GpuPlanResourceCache, GpuPlanResourceCacheStats, GpuPlanResourceError, GpuPlanResourceKey,
    GpuPlanResourceLayout, GpuPlanResources,
};
