//! Evaluation counters and per-operation trace records.

use terra_core::deps::NodeRef;
use terra_core::terrain_plan::{PlanOpId, PropagatedDirtyScope};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuEvalStats {
    pub resolution: u32,
    pub cold_execution: bool,
    pub selected_operations: u32,
    pub dirty_texels: u64,
    pub resource_prepare_us: u64,
    pub capability_preflight_us: u64,
    pub command_encode_us: u64,
    pub queue_submit_us: u64,
    pub used_layer_zero_region: bool,
    pub sculpt_resampled_texels: u64,
    pub upload_bytes: u64,
    pub blend_workgroups: u64,
    pub copy_workgroups: u64,
    pub cache_copy_workgroups: u64,
    pub reused_contributions: u32,
    /// Bytes transferred for authored SculptStrokes runtime payloads.
    pub stroke_header_upload_bytes: u64,
    pub stroke_point_upload_bytes: u64,
    /// Full stroke payload uploads, including cold creation and capacity growth.
    pub stroke_payload_rebuilds: u32,
    /// Warm compiled-plan executions that reused the existing resource realization.
    pub warm_plan_resource_reuses: u32,
    /// Compiled plan operation disposition for this evaluation.
    pub operations_dispatched: u32,
    pub operations_published: u32,
    pub operations_skipped: u32,
    pub operations_reused: u32,
    pub operations_deferred: u32,
    /// One logical 8x8 dispatch footprint per executed plan operation. This is
    /// deliberately separate from multipass kernel-specific counters above.
    pub plan_workgroups: u64,
    /// Dense height bytes copied back to the CPU by this evaluation.
    pub readback_bytes: u64,
    /// Per-evaluation mask scratch texture churn. Warm executions should report
    /// zero allocations and one or more cache reuses when masks are evaluated.
    pub mask_scratch_texture_allocations: u32,
    pub mask_scratch_reuses: u32,
}

impl GpuEvalStats {
    pub const fn total_upload_bytes(self) -> u64 {
        self.upload_bytes
            .saturating_add(self.stroke_header_upload_bytes)
            .saturating_add(self.stroke_point_upload_bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPlanOperationDisposition {
    Dispatched,
    Published,
    Deferred,
    Reused,
    Skipped,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuPlanOperationTrace {
    pub operation: PlanOpId,
    pub owner: Option<NodeRef>,
    pub incoming_scope: PropagatedDirtyScope,
    pub output_scope: PropagatedDirtyScope,
    pub incoming_region: (u32, u32, u32, u32),
    pub output_region: (u32, u32, u32, u32),
    pub disposition: GpuPlanOperationDisposition,
}
