//! GPU layer-stack preview engine: texture caches, ping-pong sims, no interactive readback.
//!
//! This module owns the stateful wgpu runtime (`GpuTerrainEngine`) and its
//! high-level orchestration. Public contracts, resource management, pipeline
//! families, compiled execution, refinement, readback, and tests live in
//! focused sibling modules without changing the façade.

//! Interactive hard rules (WC): no UI-thread height readback, no mesh rebuild,
//! prefer fully GPU stacks, never present an incomplete prefix as finished Draft.

use crate::evaluation_timing::{
    GpuEvaluationTimer, GpuEvaluationTiming, GpuEvaluationTraceContext,
};
use bytemuck::{Pod, Zeroable};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, TryRecvError};
use std::time::{Instant, SystemTime};
use terra_core::analyze::{
    amplify_sim_levels, apply_transport_model, clamp_timestep_cfl, default_sim_levels,
    draft_sim_levels, LevelStepSettings,
};
use terra_core::deps::NodeRef;
use terra_core::fields::FieldId;
use terra_core::heightfield::{Heightfield, HeightfieldMetrics, TileId};
use terra_core::layer::{
    BlendMode, EffectFilterParams, FractalNoiseType, IslandArchetype, IslandParams, Layer, LayerId,
    LayerKind, LayerStack, MultiScaleAmplifyParams, NoiseParams, PathParams, PlateauParams,
    PolygonHeightMode, PolygonHeightParams, ProceduralGenerator, ProceduralShapeParams,
    SculptParams, SculptStroke, SculptStrokeKind, SculptStrokeParams,
};
use terra_core::mask::{Distribution, MaskAsset, MaskSource};
use terra_core::quality::PreviewQuality;
use terra_core::terrain_plan::{
    compile_terrain_plan, propagate_plan_edits, CompiledTerrainPlan, GroupCompositeMode,
    PlanDirtyScope, PlanInvalidation, PlanOpId, PlanOrigin, PlanStructureRevision,
    PropagatedDirtyScope, TerrainEditClass, TerrainOpKind, TerrainPlanStamp,
};
use terra_gpu::compiled_plan::{
    GpuGroupCompositeParams, GpuPlanOperationError, GpuPlanOperations, GpuPlanResourceCache,
    GpuPlanResourceKey, GpuPlanResources,
};
use terra_gpu::effect_filter::{effect_filter_gpu_spec, EffectFilterGpuPasses};
use terra_gpu::graph::{
    compile_gpu_graph, gpu_blend_mode, GpuComputeGraph, GpuFallbackCode, GpuFallbackDiagnostic,
    GpuFallbackReason, GpuKernel, GpuLayerPlan, BLUR_MAX_RADIUS, EFFECT_FILTER_MAX_RADIUS,
    RIVER_CARVE_MAX_RADIUS,
};
use terra_gpu::output_identity::{
    GpuOutputId, GpuOutputSlot, GpuResourceIncarnation, GpuSubmissionSerial,
    GpuTerrainOutputIdentity,
};
use terra_gpu::{readback_f32, GpuError};
use wgpu::util::DeviceExt;

use super::api::{
    GpuEvalResult, GpuEvaluationIntent, GpuPreviewFreshness, GpuRefinementJob, GpuRefinementStep,
};
use super::stats::{GpuEvalStats, GpuPlanOperationDisposition, GpuPlanOperationTrace};

/// Expand a texel rectangle by `pad` samples on every side, clamped to the
/// current field. Sculpt uses this to distinguish its published rectangle from
/// the larger guard domain required by neighborhood reads.
fn expand_sample_region(
    (x, y, width, height): (u32, u32, u32, u32),
    pad: u32,
    field_width: u32,
    field_height: u32,
) -> (u32, u32, u32, u32) {
    if width == 0 || height == 0 || field_width == 0 || field_height == 0 {
        return (x, y, width, height);
    }
    let x0 = x.saturating_sub(pad);
    let y0 = y.saturating_sub(pad);
    let x1 = x.saturating_add(width).saturating_add(pad).min(field_width);
    let y1 = y
        .saturating_add(height)
        .saturating_add(pad)
        .min(field_height);
    (x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
}

fn sculpt_stamp_guard(p: &SculptStrokeParams) -> u32 {
    u32::from(p.reconcile > 0.0)
}

fn sculpt_source_guard(p: &SculptStrokeParams) -> u32 {
    sculpt_stamp_guard(p)
        + u32::from(p.strokes.iter().any(|stroke| {
            stroke.enabled
                && matches!(
                    stroke.kind,
                    SculptStrokeKind::Smooth
                        | SculptStrokeKind::Pinch
                        | SculptStrokeKind::Coastline
                )
        }))
}

#[path = "compiled_plan.rs"]
mod compiled_plan;
#[path = "compiled_plan_support.rs"]
mod compiled_plan_support;
use compiled_plan_support::{cpu_resume_prefix_is_height_only, BridgePrefix};
#[path = "dirty.rs"]
mod dirty_state;
#[path = "pipelines/mod.rs"]
mod pipelines;
#[path = "readback.rs"]
mod readback;
#[path = "refinement.rs"]
mod refinement;
#[path = "tile_domain.rs"]
mod tile_domain;
use tile_domain::CompiledPlanExecutionOverride;
pub use tile_domain::{
    GpuCompiledTileProducer, GpuTileEvaluationError, GpuTileEvaluationJob, GpuTileProducerStats,
};
#[path = "resources.rs"]
mod resources;
use pipelines::{record_copy_views_region, Pipe};
use resources::{
    build_stroke_buffers, make_runtime_storage_buffer, make_storage_buffer, HeightTex, RgbaTex,
    SourceFingerprint, SourceRasterTex, StrokeHeaderGpu, StrokeRuntimeBuffers, TexSlot,
    UniformPool, INITIAL_STROKE_HEADER_CAPACITY, INITIAL_STROKE_POINT_CAPACITY,
};

fn cpu_required(
    code: GpuFallbackCode,
    family: &'static str,
    detail: impl Into<String>,
) -> GpuError {
    GpuError::RequiresCpu(GpuFallbackReason::new(code, family, detail))
}

use terra_core::tiling::{SampleRect, TileScheduler};

/// Small resident texture extent used while no project is active.
/// `ensure_size` restores the next evaluation's document dimensions.
const PROJECT_RESET_TEXTURE_EXTENT: u32 = 8;
static NEXT_DEVICE_GENERATION: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy)]
struct TileSampleWindow {
    origin_x: u32,
    origin_z: u32,
    level_width: u32,
    level_height: u32,
}

/// GPU stack evaluator for interactive preview.
pub struct GpuTerrainEngine {
    plan_operations: GpuPlanOperations,
    plan_resources: GpuPlanResourceCache,
    device_generation: u64,
    output_resource_incarnation: GpuResourceIncarnation,
    next_output_id: u64,
    next_submission_serial: u64,
    last_output_identity: Option<GpuTerrainOutputIdentity>,
    active_plan_revision: Option<PlanStructureRevision>,
    deferred_plan_resume: Option<(PlanStructureRevision, PlanOpId)>,
    fill: Pipe,
    noise: Pipe,
    blend: Pipe,
    copy: Pipe,
    thermal: Pipe,
    thermal_apply: Pipe,
    hydraulic_outflow: Pipe,
    hydraulic: Pipe,
    blur: Pipe,
    terrace: Pipe,
    ramp: Pipe,
    shapes: Pipe,
    island: Pipe,
    plateau: Pipe,
    river_accum: Pipe,
    river_carve: Pipe,
    stream_power: Pipe,
    amplify_downsample: Pipe,
    amplify_upsample_blend: Pipe,
    effect_filter_range: Pipe,
    effect_filter: Pipe,
    sculpt_strokes: Pipe,
    sculpt_strokes_edited: Pipe,
    sculpt_strokes_flatten_reduce: Pipe,
    sculpt_strokes_flatten_resolve: Pipe,
    sculpt_strokes_reconcile: Pipe,
    path_height: Pipe,
    polygon_height: Pipe,
    heightmap_sample: Pipe,
    uniform_pool: UniformPool,
    ping: HeightTex,
    pong: HeightTex,
    layer_tex: HeightTex,
    mask_ones: HeightTex,
    unit_mask: HeightTex,
    stamp_mask: HeightTex,
    hardness: HeightTex,
    water_a: HeightTex,
    water_b: HeightTex,
    delta: HeightTex,
    sed_a: HeightTex,
    sed_b: HeightTex,
    /// Spatial rainfall multiplier (1 = uniform). Stays on GPU across hydraulic iters.
    rainfall: HeightTex,
    /// Loose sediment thickness for layered erodibility (meters).
    loose_sediment: HeightTex,
    outflow: RgbaTex,
    /// Square reusable level textures used by MultiScaleAmplify. CPU SimLevels
    /// are square and keyed from document width, including on non-square fields.
    amplify_a: HeightTex,
    amplify_b: HeightTex,
    /// SculptStrokes preview scratch: the running stamped height (`sculpt_stamp` and
    /// the ping-pong partner `sculpt_stamp_b`) and the per-texel brush coverage
    /// (`sculpt_edited`), read by the reconcile pass (#113, #117). The Flatten
    /// segmentation (#117) chains stamp segments between the two height buffers.
    sculpt_stamp: HeightTex,
    sculpt_stamp_b: HeightTex,
    sculpt_edited: HeightTex,
    /// Ordered-f32 min/max written by `effect_filter_range` and read by remap kernels.
    effect_filter_range_buffer: wgpu::Buffer,
    /// Hydraulic raw-state invariant flags, written before shader stability clamps.
    simulation_invalid_state_buffer: wgpu::Buffer,
    layer_cache: HashMap<LayerId, HeightTex>,
    /// Stamp2d transform weights paired with reusable compiled candidates.
    stamp_mask_cache: HashMap<LayerId, HeightTex>,
    /// Raw normalized source rasters, independent from output-sized contributions.
    source_rasters: HashMap<PathBuf, SourceRasterTex>,
    /// Persistent authored stroke payloads keyed by stable layer identity. Live
    /// gesture samples update only the appended point tail and active header.
    stroke_runtime: HashMap<LayerId, StrokeRuntimeBuffers>,
    #[cfg(test)]
    source_upload_count: usize,
    dirty: HashSet<LayerId>,
    metrics: HeightfieldMetrics,
    tile_sample_window: Option<TileSampleWindow>,
    approx_range: (f32, f32),
    /// Index of texture holding current composed height: 0=ping, 1=pong.
    current: u8,
    /// Spatial dirty tiles for region present / normal recompute (Wave D).
    tile_sched: TileScheduler,
    /// Optional texel-space edit bounds supplied by interactive painting.
    last_dirty_rect: Option<(u32, u32, u32, u32)>,
    last_quality: Option<PreviewQuality>,
    /// Maximum thermal/hydraulic/stream-power iterations submitted in one interactive tick.
    pub max_sim_iters_per_tick: u32,
    /// Last compiled GPU pass graph for the evaluated stack.
    pub last_graph: GpuComputeGraph,
    /// Kernels dispatched by the most recent `evaluate` walk, in order — the
    /// witness that the executor consumes the compiled plan (B1-D6 revert guard).
    #[cfg(test)]
    executed_kernels: Vec<GpuKernel>,
    last_eval_stats: GpuEvalStats,
    last_plan_operation_trace: Vec<GpuPlanOperationTrace>,
    evaluation_timer: Option<GpuEvaluationTimer>,
    pending_evaluation_trace: Option<GpuEvaluationTraceContext>,
    /// Completion fences retained from superseded jobs. They prevent a later
    /// generation from building a second refinement backlog behind an
    /// unavoidably non-cancellable stale submission.
    retired_refinement_completions: Vec<mpsc::Receiver<()>>,
    #[cfg(test)]
    executed_plan_operations: Vec<PlanOpId>,
}

impl GpuTerrainEngine {
    fn allocate_output_id(&mut self) -> GpuOutputId {
        let id = GpuOutputId(self.next_output_id);
        self.next_output_id = self
            .next_output_id
            .checked_add(1)
            .expect("GPU output identity exhausted");
        id
    }

    fn allocate_submission_serial(&mut self) -> GpuSubmissionSerial {
        let serial = GpuSubmissionSerial(self.next_submission_serial);
        self.next_submission_serial = self
            .next_submission_serial
            .checked_add(1)
            .expect("GPU submission serial exhausted");
        serial
    }

    fn output_slot(&self) -> GpuOutputSlot {
        if self.current == 0 {
            GpuOutputSlot::Ping
        } else {
            GpuOutputSlot::Pong
        }
    }

    pub fn last_output_identity(&self) -> Option<GpuTerrainOutputIdentity> {
        self.last_output_identity
    }

    pub fn last_eval_stats(&self) -> GpuEvalStats {
        self.last_eval_stats
    }

    pub fn set_evaluation_trace_context(&mut self, context: GpuEvaluationTraceContext) {
        self.pending_evaluation_trace = Some(context);
    }

    /// Advances timestamp readbacks without waiting for GPU completion.
    pub fn poll_evaluation_timings(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Vec<GpuEvaluationTiming> {
        let Some(timer) = self.evaluation_timer.as_mut() else {
            return Vec::new();
        };
        timer.poll(device, queue.get_timestamp_period());
        timer.take_completed()
    }

    pub fn last_plan_operation_trace(&self) -> &[GpuPlanOperationTrace] {
        &self.last_plan_operation_trace
    }

    /// Cap thermal/hydraulic/stream-power iterations for the current interactive refinement phase.
    /// `None` means uncapped (export / full quality).
    pub fn set_simulation_iteration_cap(&mut self, cap: Option<u32>) {
        self.max_sim_iters_per_tick = cap.unwrap_or(u32::MAX);
    }

    #[allow(clippy::too_many_arguments)]
    /// Evaluate through the compatibility planner.
    ///
    /// `bridge_prefix` is the complete height entering the first layer marked
    /// dirty. It allows a GPU-compatible suffix to resume after a CPU-baked,
    /// height-only prefix; prefixes carrying auxiliary state are rejected.
    pub fn evaluate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        metrics: HeightfieldMetrics,
        quality: PreviewQuality,
        want_cpu: bool,
        bridge_prefix: Option<&Heightfield>,
    ) -> Result<GpuEvalResult, GpuError> {
        self.evaluate_with_intent(
            device,
            queue,
            stack,
            mask_assets,
            metrics,
            quality,
            want_cpu,
            bridge_prefix,
            GpuEvaluationIntent::Complete,
        )
    }

    /// Compatibility entry point that compiles a validated plan before crossing
    /// the GPU boundary. Production callers should retain a `TerrainPlanCache`
    /// and call [`Self::evaluate_compiled_with_intent`] directly.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_with_intent(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        metrics: HeightfieldMetrics,
        quality: PreviewQuality,
        want_cpu: bool,
        bridge_prefix: Option<&Heightfield>,
        intent: GpuEvaluationIntent,
    ) -> Result<GpuEvalResult, GpuError> {
        let revision = PlanStructureRevision::INITIAL;
        let plan = compile_terrain_plan(stack, mask_assets, TerrainPlanStamp::new(revision))
            .map_err(|diagnostics| {
                cpu_required(
                    GpuFallbackCode::UnsupportedOptions,
                    "terrain plan",
                    diagnostics
                        .first()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "terrain plan compilation failed".to_string()),
                )
            })?;
        // Retain the public flat capability snapshot for compatibility diagnostics
        // and existing profiler consumers. It is not consulted by compiled-plan
        // scheduling or dispatch.
        self.last_graph = compile_gpu_graph(stack, mask_assets);
        let dirty_ids: Vec<LayerId> = self.dirty.iter().copied().collect();
        let scope = self
            .last_dirty_rect
            .map_or(PlanDirtyScope::FullField, |(x, y, w, h)| {
                // The compatibility API receives a preview-space sculpt footprint.
                // Include the bilinear resampling fringe before converting to the
                // resolution-independent plan scope.
                let x0 = x.saturating_sub(1);
                let y0 = y.saturating_sub(1);
                let x1 = x.saturating_add(w).saturating_add(1).min(metrics.width);
                let y1 = y.saturating_add(h).saturating_add(1).min(metrics.height);
                PlanDirtyScope::Region(terra_core::tiling::UvRect {
                    min_u: x0 as f32 / metrics.width.max(1) as f32,
                    min_v: y0 as f32 / metrics.height.max(1) as f32,
                    max_u: x1 as f32 / metrics.width.max(1) as f32,
                    max_v: y1 as f32 / metrics.height.max(1) as f32,
                })
            });
        let edits: Vec<TerrainEditClass> = if self.active_plan_revision.is_none() {
            vec![TerrainEditClass::Structure]
        } else {
            dirty_ids
                .iter()
                .map(|layer| TerrainEditClass::Content {
                    owner: NodeRef::Layer(*layer),
                    fields: vec![FieldId::Height],
                    scope,
                })
                .collect()
        };
        let invalidation = propagate_plan_edits(&plan, &edits);
        let bridge_prefix = bridge_prefix
            .map(|height| {
                if height.metrics.width == 0 || height.metrics.height == 0 {
                    return Err(cpu_required(
                        GpuFallbackCode::RuntimeResourceLimit,
                        "bridge prefix",
                        "the bridge heightfield is empty",
                    ));
                }
                let layers = stack.flatten_layers();
                let Some((first_dirty_index, first_dirty_layer)) = layers
                    .iter()
                    .enumerate()
                    .find(|(_, layer)| self.dirty.contains(&layer.id()))
                else {
                    return Err(cpu_required(
                        GpuFallbackCode::UnsupportedOptions,
                        "bridge prefix",
                        "a bridge prefix requires at least one dirty layer",
                    ));
                };
                if stack.requires_tree_evaluation() {
                    return Err(cpu_required(
                        GpuFallbackCode::UnsupportedOptions,
                        "bridge prefix",
                        "bridge prefixes currently require a flat authored stack",
                    ));
                }
                if first_dirty_index == 0
                    || !cpu_resume_prefix_is_height_only(&layers, first_dirty_index)
                {
                    return Err(cpu_required(
                        GpuFallbackCode::AuxiliaryDependency,
                        "bridge prefix",
                        "the prefix before the first dirty layer is not a complete height-only checkpoint",
                    ));
                }
                Ok(BridgePrefix {
                    height,
                    first_dirty_layer: first_dirty_layer.id(),
                })
            })
            .transpose()?;
        let result = self.evaluate_compiled_with_bridge(
            device,
            queue,
            stack,
            mask_assets,
            &plan,
            revision,
            &invalidation,
            metrics,
            quality,
            want_cpu,
            intent,
            bridge_prefix,
            None,
        );
        if result
            .as_ref()
            .is_ok_and(|result| result.did_eval && !result.freshness.is_deferred())
        {
            for layer in dirty_ids {
                self.dirty.remove(&layer);
            }
        }
        result
    }

    fn expand_range(&mut self, lo: f32, hi: f32) {
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        self.approx_range.0 = self.approx_range.0.min(lo);
        self.approx_range.1 = self.approx_range.1.max(hi);
        if self.approx_range.0 > self.approx_range.1 {
            self.approx_range.1 = self.approx_range.0 + 1e-3;
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod smoke_tests;
