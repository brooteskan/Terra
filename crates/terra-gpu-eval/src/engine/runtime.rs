//! GPU layer-stack preview engine: texture caches, ping-pong sims, no interactive readback.
//!
//! This module owns the stateful wgpu runtime (`GpuTerrainEngine`). Public
//! lifecycle/result types, statistics, and tests live in sibling files; pipeline
//! families can be extracted incrementally without changing the façade.

//! Interactive hard rules (WC): no UI-thread height readback, no mesh rebuild,
//! prefer fully GPU stacks, never present an incomplete prefix as finished Draft.

use crate::evaluation_timing::{
    GpuEvaluationTimer, GpuEvaluationTiming, GpuEvaluationTraceContext,
};
use bytemuck::{Pod, Zeroable};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
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
use terra_gpu::{readback_f32, GpuError};
use wgpu::util::DeviceExt;

#[path = "api.rs"]
mod api;
#[path = "stats.rs"]
mod stats;

pub use api::{
    GpuEvalResult, GpuEvaluationIntent, GpuPreviewFreshness, GpuRefinementJob, GpuRefinementPhase,
    GpuRefinementProgress, GpuRefinementStep,
};
pub use stats::{GpuEvalStats, GpuPlanOperationDisposition, GpuPlanOperationTrace};

fn cpu_required(
    code: GpuFallbackCode,
    family: &'static str,
    detail: impl Into<String>,
) -> GpuError {
    GpuError::RequiresCpu(GpuFallbackReason::new(code, family, detail))
}

#[allow(clippy::too_many_arguments)]
fn record_copy_views_region(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipe: &Pipe,
    source: &wgpu::TextureView,
    destination: &wgpu::TextureView,
    width: u32,
    height: u32,
    region: (u32, u32, u32, u32),
) {
    let (region_x, region_y, region_w, region_h) = region;
    if region_w == 0 || region_h == 0 {
        return;
    }
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("compiled-plan-legacy-copy-uniform"),
        contents: bytemuck::bytes_of(&CopyU {
            width,
            height,
            region_x,
            region_y,
            region_w,
            region_h,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compiled-plan-legacy-copy-bind-group"),
        layout: &pipe.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(source),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(destination),
            },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("compiled-plan-legacy-copy"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipe.pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(region_w.div_ceil(8), region_h.div_ceil(8), 1);
}

#[derive(Debug)]
enum CompiledDispatchError {
    Plan(GpuPlanOperationError),
    Gpu(GpuError),
}

impl From<GpuPlanOperationError> for CompiledDispatchError {
    fn from(error: GpuPlanOperationError) -> Self {
        Self::Plan(error)
    }
}

impl From<GpuError> for CompiledDispatchError {
    fn from(error: GpuError) -> Self {
        Self::Gpu(error)
    }
}

fn kernel_runs_in_place(kernel: GpuKernel) -> bool {
    matches!(
        kernel,
        GpuKernel::Blur
            | GpuKernel::EffectFilter
            | GpuKernel::Terrace
            | GpuKernel::Thermal
            | GpuKernel::Hydraulic
            | GpuKernel::RiverCarve
            | GpuKernel::StreamPower
            | GpuKernel::MultiScaleAmplify
    )
}

fn plan_scope_region(
    scope: PropagatedDirtyScope,
    metrics: HeightfieldMetrics,
) -> (u32, u32, u32, u32) {
    match scope.scope {
        PlanDirtyScope::FullField => (0, 0, metrics.width, metrics.height),
        PlanDirtyScope::Region(region) => {
            if metrics.width == 0 || metrics.height == 0 {
                return (0, 0, 0, 0);
            }
            let x0 = ((region.min_u.clamp(0.0, 1.0) * metrics.width as f32).floor() as u32)
                .min(metrics.width - 1)
                .saturating_sub(scope.halo_samples);
            let y0 = ((region.min_v.clamp(0.0, 1.0) * metrics.height as f32).floor() as u32)
                .min(metrics.height - 1)
                .saturating_sub(scope.halo_samples);
            let x1 = ((region.max_u.clamp(0.0, 1.0) * metrics.width as f32).ceil() as u32)
                .max(x0 + 1)
                .saturating_add(scope.halo_samples)
                .min(metrics.width);
            let y1 = ((region.max_v.clamp(0.0, 1.0) * metrics.height as f32).ceil() as u32)
                .max(y0 + 1)
                .saturating_add(scope.halo_samples)
                .min(metrics.height);
            (x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
        }
    }
}

fn plan_distribution(
    stack: &LayerStack,
    origin: PlanOrigin,
) -> Option<&terra_core::mask::Distribution> {
    match origin {
        PlanOrigin::Authored(NodeRef::Layer(layer)) => {
            stack.find(layer).map(|layer| &layer.common.masks)
        }
        PlanOrigin::Authored(NodeRef::Group(group)) => {
            stack.find_group(group).map(|group| &group.masks)
        }
        _ => None,
    }
}

fn owner_layer_id(owner: Option<NodeRef>) -> Option<LayerId> {
    match owner {
        Some(NodeRef::Layer(layer) | NodeRef::Group(layer)) => Some(layer),
        _ => None,
    }
}

fn plan_fallback_diagnostic(
    plan: &CompiledTerrainPlan,
    stack: &LayerStack,
    operation: PlanOpId,
    reason: GpuFallbackReason,
) -> GpuFallbackDiagnostic {
    let owner = plan.provenance().owner_of(operation);
    let layer_id = owner_layer_id(owner).unwrap_or_default();
    let layer_name = match owner {
        Some(NodeRef::Layer(layer)) => stack
            .find(layer)
            .map(|layer| layer.common.name.clone())
            .unwrap_or_else(|| format!("layer {layer:?}")),
        Some(NodeRef::Group(group)) => stack
            .find_group(group)
            .map(|group| group.name.clone())
            .unwrap_or_else(|| format!("group {group:?}")),
        Some(other) => format!("{other:?}"),
        None => "terrain plan root".to_string(),
    };
    GpuFallbackDiagnostic {
        operation: Some(operation),
        owner,
        layer_index: operation.index(),
        layer_id,
        layer_name,
        reason,
    }
}

fn plan_fallback_result(
    metrics: HeightfieldMetrics,
    height_range: (f32, f32),
    diagnostic: GpuFallbackDiagnostic,
) -> GpuEvalResult {
    GpuEvalResult {
        width: metrics.width,
        height: metrics.height,
        world_size: (metrics.world_size_x, metrics.world_size_z),
        height_range,
        fully_gpu: false,
        freshness: GpuPreviewFreshness::Current,
        cpu: None,
        resume_cpu_from: Some(0),
        cpu_fallback: Some(diagnostic),
        did_eval: false,
    }
}

fn plan_operation_fallback(error: CompiledDispatchError) -> GpuFallbackReason {
    match error {
        CompiledDispatchError::Gpu(GpuError::RequiresCpu(reason)) => reason,
        CompiledDispatchError::Gpu(error) => GpuFallbackReason::new(
            GpuFallbackCode::RuntimeResourceLimit,
            "GPU execution",
            error.to_string(),
        ),
        CompiledDispatchError::Plan(GpuPlanOperationError::UnsupportedBlend(blend)) => {
            GpuFallbackReason::new(
                GpuFallbackCode::BlendMode,
                "blend",
                format!("{blend:?} is not supported by the GPU"),
            )
        }
        CompiledDispatchError::Plan(GpuPlanOperationError::UnsupportedMaskNodes) => {
            GpuFallbackReason::new(
                GpuFallbackCode::MaskNodes,
                "mask",
                "distribution nodes or auxiliary mask inputs are not GPU-resident",
            )
        }
        CompiledDispatchError::Plan(GpuPlanOperationError::MissingMaskAsset(mask)) => {
            GpuFallbackReason::new(
                GpuFallbackCode::MissingMaskAsset,
                "mask",
                format!("mask asset {mask:?} is missing"),
            )
        }
        CompiledDispatchError::Plan(GpuPlanOperationError::UnsupportedMaskSource(source)) => {
            GpuFallbackReason::new(GpuFallbackCode::MaskSource, "mask", source)
        }
        CompiledDispatchError::Plan(GpuPlanOperationError::MaskBlurRadius(radius)) => {
            GpuFallbackReason::new(
                GpuFallbackCode::MaskOperations,
                "mask",
                format!("blur radius {radius} exceeds the GPU limit"),
            )
        }
        CompiledDispatchError::Plan(error) => GpuFallbackReason::new(
            GpuFallbackCode::InvalidConfiguration,
            "compiled operation",
            error.to_string(),
        ),
    }
}

use terra_core::tiling::{SampleRect, TileScheduler};

/// Small resident texture extent used while no project is active.
/// `ensure_size` restores the next evaluation's document dimensions.
const PROJECT_RESET_TEXTURE_EXTENT: u32 = 8;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NoiseU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    seed: u32,
    octaves: u32,
    frequency: f32,
    amplitude: f32,
    lacunarity: f32,
    persistence: f32,
    offset_x: f32,
    offset_z: f32,
    remap_min: f32,
    remap_max: f32,
    noise_type: u32,
    mode: u32,
    warp_strength: f32,
    warp_frequency: f32,
    cell_jitter: f32,
    height_per_cell: f32,
}

#[repr(u32)]
#[derive(Clone, Copy)]
enum NoiseKernelMode {
    /// Existing portable value-noise preview. Its output is already public and
    /// parity-ratcheted, so the new CPU-aligned modes must not rewrite it.
    LegacyValue = 0,
    Perlin = 1,
    Fbm = 2,
    Ridged = 3,
    DomainWarp = 4,
    VoronoiRegions = 5,
}

#[derive(Clone, Copy)]
struct NoiseDispatch {
    noise_type: u32,
    mode: NoiseKernelMode,
    warp_strength: f32,
    warp_frequency: f32,
    cell_jitter: f32,
    height_per_cell: f32,
}

impl NoiseDispatch {
    const fn new(noise_type: u32, mode: NoiseKernelMode) -> Self {
        Self {
            noise_type,
            mode,
            warp_strength: 0.0,
            warp_frequency: 0.0,
            cell_jitter: 0.0,
            height_per_cell: 0.0,
        }
    }

    const fn domain_warp(warp_strength: f32, warp_frequency: f32) -> Self {
        Self {
            noise_type: 1,
            mode: NoiseKernelMode::DomainWarp,
            warp_strength,
            warp_frequency,
            cell_jitter: 0.0,
            height_per_cell: 0.0,
        }
    }

    const fn voronoi_regions(cell_jitter: f32, height_per_cell: f32) -> Self {
        Self {
            noise_type: 0,
            mode: NoiseKernelMode::VoronoiRegions,
            warp_strength: 0.0,
            warp_frequency: 0.0,
            cell_jitter,
            height_per_cell,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlendU {
    width: u32,
    height: u32,
    opacity: f32,
    mode: u32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FillU {
    width: u32,
    height: u32,
    value: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ThermalU {
    width: u32,
    height: u32,
    dx: f32,
    talus: f32,
    strength: f32,
    _p2: f32,
    _p3: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HydraulicU {
    width: u32,
    height: u32,
    timestep: f32,
    rainfall: f32,
    evaporation: f32,
    erosion: f32,
    deposition: f32,
    capacity: f32,
    fan_boost: f32,
    floodplain_bias: f32,
    dx: f32,
    incision_bias: f32,
    bedrock_k: f32,
    sediment_k: f32,
    layered: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurU {
    width: u32,
    height: u32,
    radius: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerraceU {
    width: u32,
    height: u32,
    levels: u32,
    sharpness: f32,
    min_h: f32,
    max_h: f32,
    _p0: f32,
    _p1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RampU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    height_min: f32,
    height_max: f32,
    direction: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CopyU {
    width: u32,
    height: u32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SculptStrokesU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    /// Contiguous stroke range this dispatch stamps: `[stroke_lo, stroke_hi)`.
    /// The edited pass shares the layout with `lo = 0`, `hi = stroke_count`.
    stroke_lo: u32,
    stroke_hi: u32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

/// Reduce pass uniform: measures one Flatten stroke's footprint mean. `stroke_index`
/// selects the header; workgroups address their partial slot via `num_workgroups`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SculptReduceU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    stroke_index: u32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

/// Resolve pass uniform: folds the reduce partials into `targets[target_index]`,
/// falling back to `fallback` (the stroke's `target_height`) when the footprint
/// carried no weight — the exact CPU `flatten_target_for` degenerate branch.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SculptResolveU {
    num_partials: u32,
    target_index: u32,
    fallback: f32,
    _p0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SculptReconcileU {
    width: u32,
    height: u32,
    reconcile: f32,
    _p0: f32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PathU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    point_count: u32,
    carve: u32,
    seed: u32,
    _pad0: u32,
    base_width: f32,
    falloff: f32,
    noise_strength: f32,
    noise_scale: f32,
    height_offset: f32,
    profile: f32,
    _pad1: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PolygonHeightU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    point_count: u32,
    mode: u32,
    carve: u32,
    _pad0: u32,
    target_height: f32,
    falloff: f32,
    _pad1: f32,
    _pad2: f32,
}

/// One stroke's GPU header. Layout mirrors `StrokeHeader` in
/// `shaders/sculpt_strokes.wgsl` (48 bytes, 8-byte aligned for the trailing
/// `vec2<f32>` bbox fields); `points` are uploaded separately as `vec4<f32>`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Pod, Zeroable)]
struct StrokeHeaderGpu {
    kind: u32,
    first_point: u32,
    point_count: u32,
    _pad0: u32,
    radius_m: f32,
    strength: f32,
    target_height: f32,
    falloff: f32,
    bbox_min: [f32; 2],
    bbox_max: [f32; 2],
}

struct StrokeRuntimeBuffers {
    headers: wgpu::Buffer,
    points: wgpu::Buffer,
    header_capacity: usize,
    point_capacity: usize,
    uploaded_headers: Vec<StrokeHeaderGpu>,
    uploaded_points: Vec<[f32; 4]>,
}

const INITIAL_STROKE_HEADER_CAPACITY: usize = 8;
const INITIAL_STROKE_POINT_CAPACITY: usize = 64;

/// Alias-collapsed kind id shared with `shaders/sculpt_strokes.wgsl`. Only kinds
/// the planner admits are ever uploaded — the per-sample maps, the base-3x3
/// Smooth/Pinch/Coastline, and the footprint-mean Flatten (#117), whose per-stroke
/// target the reduce/resolve passes precompute into the `targets` buffer.
fn stroke_kind_gpu_id(kind: SculptStrokeKind) -> u32 {
    match kind {
        SculptStrokeKind::Raise => 0,
        SculptStrokeKind::Lower => 1,
        SculptStrokeKind::Ridge | SculptStrokeKind::MountainStamp => 2,
        SculptStrokeKind::Valley | SculptStrokeKind::ValleyStamp | SculptStrokeKind::RiverPath => 3,
        SculptStrokeKind::Terrace => 4,
        SculptStrokeKind::Roughness | SculptStrokeKind::Noise => 5,
        SculptStrokeKind::Inflate => 6,
        SculptStrokeKind::PlateauStamp => 7,
        SculptStrokeKind::CraterStamp => 8,
        SculptStrokeKind::HeightStamp => 9,
        SculptStrokeKind::Erode | SculptStrokeKind::EncourageErosion => 10,
        // Uplift / Hardness / Sediment / Protect contribute only aux on the CPU;
        // their height is unchanged, but they still mark the edit region.
        SculptStrokeKind::Uplift
        | SculptStrokeKind::Hardness
        | SculptStrokeKind::Sediment
        | SculptStrokeKind::Protect => 11,
        // Smooth pulls each sample toward the clamped 3x3 mean of the layer input
        // (`src`); Pinch applies the same pull at a 1.25 overdrive; Coastline lowers
        // the sample and blends it toward that mean under a weight gate. The stamp
        // kernel reads that neighborhood directly (#114, #115, #116).
        SculptStrokeKind::Smooth => 12,
        SculptStrokeKind::Pinch => 13,
        SculptStrokeKind::Coastline => 14,
        // Flatten settles toward the brush-weighted mean of the running field over
        // its footprint; the reduce/resolve passes compute that scalar per stroke
        // into `targets`, and the stamp arm applies `h + (target - h) * w` (#117).
        SculptStrokeKind::Flatten => 15,
    }
}

/// Flatten a stroke set into GPU headers + a shared `vec4` point pool. Each
/// header carries a world-space footprint (point bbox padded by `radius_m`) so
/// the stamp kernel can cull texts outside the brush; correctness still comes
/// from the per-texel weight test. Both vectors are kept non-empty so the storage
/// bindings are valid even for an empty stroke set (`stroke_count` gates reads).
fn build_stroke_buffers(
    strokes: &[&SculptStroke],
    m: &HeightfieldMetrics,
) -> (Vec<StrokeHeaderGpu>, Vec<[f32; 4]>) {
    let sx = m.world_size_x;
    let sz = m.world_size_z;
    let mut headers = Vec::with_capacity(strokes.len());
    let mut points: Vec<[f32; 4]> = Vec::new();
    for stroke in strokes {
        let first_point = points.len() as u32;
        let (mut min_x, mut min_z) = (f32::INFINITY, f32::INFINITY);
        let (mut max_x, mut max_z) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for pt in &stroke.points {
            let wx = pt.u * sx;
            let wz = pt.v * sz;
            min_x = min_x.min(wx);
            max_x = max_x.max(wx);
            min_z = min_z.min(wz);
            max_z = max_z.max(wz);
            points.push([pt.u, pt.v, pt.pressure, 0.0]);
        }
        let r = stroke.radius_m;
        // A stroke with no points has no footprint: an inverted bbox (min > max)
        // culls every texel, exactly as the CPU produces no contribution.
        let (bbox_min, bbox_max) = if stroke.points.is_empty() {
            ([1.0, 1.0], [-1.0, -1.0])
        } else {
            ([min_x - r, min_z - r], [max_x + r, max_z + r])
        };
        headers.push(StrokeHeaderGpu {
            kind: stroke_kind_gpu_id(stroke.kind),
            first_point,
            point_count: stroke.points.len() as u32,
            _pad0: 0,
            radius_m: stroke.radius_m,
            strength: stroke.strength,
            target_height: stroke.target_height,
            falloff: stroke.falloff,
            bbox_min,
            bbox_max,
        });
    }
    if headers.is_empty() {
        // Placeholder so the storage buffer is bindable; never read (count == 0).
        headers.push(StrokeHeaderGpu {
            kind: 11,
            first_point: 0,
            point_count: 0,
            _pad0: 0,
            radius_m: 0.0,
            strength: 0.0,
            target_height: 0.0,
            falloff: 0.0,
            bbox_min: [1.0, 1.0],
            bbox_max: [-1.0, -1.0],
        });
    }
    if points.is_empty() {
        points.push([0.0, 0.0, 0.0, 0.0]);
    }
    (headers, points)
}

fn make_storage_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    bytes: &[u8],
) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (bytes.len() as u64).max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytes);
    buf
}

fn make_runtime_storage_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    element_size: usize,
    capacity: usize,
    bytes: &[u8],
) -> wgpu::Buffer {
    let size = element_size.saturating_mul(capacity.max(1)).max(16) as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        queue.write_buffer(&buffer, 0, bytes);
    }
    buffer
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShapeU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    seed: u32,
    octaves: u32,
    frequency: f32,
    amplitude: f32,
    lacunarity: f32,
    persistence: f32,
    offset_x: f32,
    offset_z: f32,
    ridge_sharpness: f32,
    range_angle: f32,
    range_width: f32,
    wave_frequency: f32,
    asymmetry: f32,
    depth: f32,
    canyon_width: f32,
    meander: f32,
    shape_mode: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlateauU {
    width: u32,
    height: u32,
    low: f32,
    high: f32,
    soft: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IslandU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    seed: u32,
    archetype: u32,
    _pad_u: [u32; 2],
    center_u: f32,
    center_v: f32,
    rotation_deg: f32,
    radius: f32,
    aspect: f32,
    sea_level: f32,
    ocean_floor: f32,
    mountain_height: f32,
    shelf_width: f32,
    shelf_depth: f32,
    beach_width: f32,
    beach_height: f32,
    reef_width: f32,
    reef_depth: f32,
    coastline_warp: f32,
    coastline_frequency: f32,
    mountain_power: f32,
    ridge_strength: f32,
    ridge_frequency: f32,
    lagoon_radius: f32,
    _pad_f: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RiverAccumU {
    width: u32,
    height: u32,
    _p0: f32,
    _p1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RiverCarveU {
    width: u32,
    height: u32,
    threshold: f32,
    depth: f32,
    channel_width: f32,
    bank_smooth: f32,
    max_radius: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StreamPowerU {
    width: u32,
    height: u32,
    k: f32,
    m: f32,
    n: f32,
    dt: f32,
    uplift: f32,
    base_level: f32,
    cell_area: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AmplifyDownsampleU {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    area_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AmplifyBlendU {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    hardness: f32,
    ridge_lock: f32,
    lock_strength: f32,
    detail_boost: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EffectFilterU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    mode: u32,
    radius: u32,
    iterations: u32,
    seed: u32,
    strength: f32,
    amount: f32,
    frequency: f32,
    sea_level: f32,
    beach_width: f32,
    slope_min: f32,
    slope_max: f32,
    rock_hardness: f32,
    terrace_height: f32,
    terrace_offset: f32,
    rotation_deg: f32,
    anisotropy: f32,
    warp_strength: f32,
    warp_frequency: f32,
    dx: f32,
    invert: f32,
    flow_threshold: f32,
    wall_steepness: f32,
    valley_floor: f32,
    talus_mix: f32,
    top_smoothness: f32,
    riser_sharpness: f32,
    lacunarity: f32,
    persistence: f32,
    octaves: u32,
    voronoi_feature: u32,
    tileable: u32,
    _pad_params: u32,
    crater_radius: f32,
    dz: f32,
    _pad_metric0: f32,
    _pad_metric1: f32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EffectRangeU {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HeightmapSampleU {
    width: u32,
    height: u32,
    source_width: u32,
    source_height: u32,
    mode: u32,
    _pad0: u32,
    height_scale: f32,
    height_offset: f32,
    world_x: f32,
    world_z: f32,
    offset_x: f32,
    offset_z: f32,
    inv_scale: f32,
    sin_t: f32,
    cos_t: f32,
    blend_size: f32,
    blend_roundness: f32,
    _pad1: [f32; 3],
}

#[derive(Clone, Copy)]
enum TexSlot {
    Ping,
    Pong,
    Layer,
    MaskOnes,
    UnitMask,
    StampMask,
    Hardness,
    WaterA,
    WaterB,
    SedA,
    SedB,
    Rainfall,
    LooseSediment,
    SculptStamp,
}

struct HeightTex {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl HeightTex {
    fn new(device: &wgpu::Device, label: &str, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            width,
            height,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct SourceFingerprint {
    len: u64,
    modified: Option<SystemTime>,
}

struct SourceRasterTex {
    tex: HeightTex,
    fingerprint: Option<SourceFingerprint>,
}

/// RGBA float texture for hydraulic outflow fluxes (L,R,D,U).
struct RgbaTex {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl RgbaTex {
    fn new(device: &wgpu::Device, label: &str, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
        }
    }
}

struct Pipe {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

fn make_pipe(device: &wgpu::Device, label: &str, wgsl: &str, bgl: wgpu::BindGroupLayout) -> Pipe {
    terra_core::shader_progress::record_shader_compiled();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    Pipe { pipeline, bgl }
}

fn storage_write_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::R32Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn storage_write_rgba_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba32Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn tex_read_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_read_buffer_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_rw_buffer_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Ring of small uniform buffers so many dispatches can share one submit.
struct UniformPool {
    buffers: Vec<wgpu::Buffer>,
    next: usize,
}

impl UniformPool {
    const SLOT_SIZE: u64 = 256;

    fn new(device: &wgpu::Device, capacity: usize) -> Self {
        let mut buffers = Vec::with_capacity(capacity);
        for i in 0..capacity {
            buffers.push(Self::make_slot(device, i));
        }
        Self { buffers, next: 0 }
    }

    fn make_slot(device: &wgpu::Device, index: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("gpu-engine-u-{index}")),
            size: Self::SLOT_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn reset(&mut self) {
        self.next = 0;
    }

    fn write<T: Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &T,
    ) -> wgpu::Buffer {
        debug_assert!(
            std::mem::size_of::<T>() as u64 <= Self::SLOT_SIZE,
            "uniform larger than pool slot"
        );
        if self.next >= self.buffers.len() {
            let start = self.buffers.len();
            let grow = self.buffers.len().max(8);
            for i in start..start + grow {
                self.buffers.push(Self::make_slot(device, i));
            }
        }
        let buf = self.buffers[self.next].clone();
        self.next += 1;
        queue.write_buffer(&buf, 0, bytemuck::bytes_of(data));
        buf
    }
}

/// Generators whose height field does not depend on the composed input below them.
fn layer_input_independent(kind: &LayerKind) -> bool {
    matches!(
        kind,
        LayerKind::Flat(_)
            | LayerKind::Ramp(_)
            | LayerKind::NoiseValue(_)
            | LayerKind::NoisePerlin(_)
            | LayerKind::Fbm(_)
            | LayerKind::Ridged(_)
            | LayerKind::Mountains(_)
            | LayerKind::Dunes(_)
            | LayerKind::Canyons(_)
            | LayerKind::Mesa(_)
            | LayerKind::Volcano(_)
            | LayerKind::Uplift(_)
            | LayerKind::Island(_)
            | LayerKind::VoronoiRegions(_)
            | LayerKind::ProceduralShape(_)
            | LayerKind::ImportHeightmap(_)
            | LayerKind::Stamp2d(_)
    )
}

/// Whether height alone completely represents the enabled prefix before `resume_index`.
///
/// CPU suffix evaluation also observes auxiliary fields and named layer outputs. Until
/// `GpuEvalResult` carries those checkpoints, any prefix that publishes them must restart
/// on the CPU from layer zero rather than borrowing stale state from another generation.
fn cpu_resume_prefix_is_height_only(layers: &[&Layer], resume_index: usize) -> bool {
    layers.iter().take(resume_index).all(|layer| {
        !layer.common.enabled
            || (layer.common.outputs.is_empty()
                && (matches!(&layer.kind, LayerKind::Path(params) if !params.carve)
                    || layer
                        .kind
                        .produced_fields()
                        .into_iter()
                        .all(|field| field == FieldId::Height)))
    })
}

/// GPU stack evaluator for interactive preview.
pub struct GpuTerrainEngine {
    plan_operations: GpuPlanOperations,
    plan_resources: GpuPlanResourceCache,
    device_generation: u64,
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
    layer_cache: HashMap<LayerId, HeightTex>,
    /// Raw normalized source rasters, independent from output-sized contributions.
    source_rasters: HashMap<PathBuf, SourceRasterTex>,
    /// Persistent authored stroke payloads keyed by stable layer identity. Live
    /// gesture samples update only the appended point tail and active header.
    stroke_runtime: HashMap<LayerId, StrokeRuntimeBuffers>,
    #[cfg(test)]
    source_upload_count: usize,
    dirty: HashSet<LayerId>,
    metrics: HeightfieldMetrics,
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
    pub fn new(device: &wgpu::Device, initial: u32) -> Self {
        let w = initial.max(PROJECT_RESET_TEXTURE_EXTENT);
        let fill_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fill-bgl"),
            entries: &[uniform_entry(0), storage_write_entry(1)],
        });
        let noise_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("noise-bgl"),
            entries: &[uniform_entry(0), storage_write_entry(1)],
        });
        let blend_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blend-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                tex_read_entry(2),
                tex_read_entry(3),
                tex_read_entry(4),
                storage_write_entry(5),
            ],
        });
        let copy_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("copy-bgl"),
            entries: &[uniform_entry(0), tex_read_entry(1), storage_write_entry(2)],
        });
        let thermal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("thermal-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                storage_write_entry(2),
                tex_read_entry(3),
            ],
        });
        let thermal_apply_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("thermal-apply-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                tex_read_entry(2),
                storage_write_entry(3),
            ],
        });
        let hydraulic_outflow_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hydraulic-outflow-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1),
                    tex_read_entry(2),
                    storage_write_rgba_entry(3),
                    tex_read_entry(4),
                ],
            });
        let hydraulic_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hydraulic-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                tex_read_entry(2),
                tex_read_entry(3),
                tex_read_entry(4),
                storage_write_entry(5),
                storage_write_entry(6),
                storage_write_entry(7),
                tex_read_entry(8),
                tex_read_entry(9),
                tex_read_entry(10),
            ],
        });
        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur-bgl"),
            entries: &[uniform_entry(0), tex_read_entry(1), storage_write_entry(2)],
        });
        let terrace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrace-bgl"),
            entries: &[uniform_entry(0), tex_read_entry(1), storage_write_entry(2)],
        });
        let ramp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ramp-bgl"),
            entries: &[uniform_entry(0), storage_write_entry(1)],
        });
        let shapes_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shapes-bgl"),
            entries: &[uniform_entry(0), storage_write_entry(1)],
        });
        let island_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("island-bgl"),
            entries: &[uniform_entry(0), storage_write_entry(1)],
        });
        let plateau_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("plateau-bgl"),
            entries: &[uniform_entry(0), tex_read_entry(1), storage_write_entry(2)],
        });
        let path_height_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("path-height-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                storage_read_buffer_entry(2),
                storage_write_entry(3),
            ],
        });
        let polygon_height_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("polygon-height-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1),
                    storage_read_buffer_entry(2),
                    storage_write_entry(3),
                ],
            });
        let heightmap_sample_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("heightmap-sample-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1),
                    storage_write_entry(2),
                    storage_write_entry(3),
                ],
            });
        let river_accum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("river-accum-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                tex_read_entry(2),
                storage_write_entry(3),
            ],
        });
        let river_carve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("river-carve-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                tex_read_entry(2),
                storage_write_entry(3),
            ],
        });
        let stream_power_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stream-power-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                tex_read_entry(2),
                tex_read_entry(3),
                storage_write_entry(4),
            ],
        });
        let amplify_downsample_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("amplify-downsample-bgl"),
                entries: &[uniform_entry(0), tex_read_entry(1), storage_write_entry(2)],
            });
        let amplify_upsample_blend_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("amplify-upsample-blend-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1),
                    tex_read_entry(2),
                    storage_write_entry(3),
                ],
            });

        let fill = make_pipe(
            device,
            "fill",
            include_str!("../shaders/fill.wgsl"),
            fill_bgl,
        );
        let noise = make_pipe(
            device,
            "noise",
            include_str!("../shaders/noise.wgsl"),
            noise_bgl,
        );
        let blend = make_pipe(
            device,
            "blend",
            include_str!("../shaders/blend.wgsl"),
            blend_bgl,
        );
        let copy = make_pipe(
            device,
            "copy",
            include_str!("../shaders/copy.wgsl"),
            copy_bgl,
        );
        let thermal = make_pipe(
            device,
            "thermal",
            include_str!("../shaders/thermal_tex.wgsl"),
            thermal_bgl,
        );
        let thermal_apply = make_pipe(
            device,
            "thermal-apply",
            include_str!("../shaders/thermal_apply.wgsl"),
            thermal_apply_bgl,
        );
        let hydraulic_outflow = make_pipe(
            device,
            "hydraulic-outflow",
            include_str!("../shaders/hydraulic_outflow.wgsl"),
            hydraulic_outflow_bgl,
        );
        let hydraulic = make_pipe(
            device,
            "hydraulic",
            include_str!("../shaders/hydraulic_tex.wgsl"),
            hydraulic_bgl,
        );
        let blur = make_pipe(
            device,
            "blur",
            include_str!("../shaders/blur.wgsl"),
            blur_bgl,
        );
        let terrace = make_pipe(
            device,
            "terrace",
            include_str!("../shaders/terrace.wgsl"),
            terrace_bgl,
        );
        let ramp = make_pipe(
            device,
            "ramp",
            include_str!("../shaders/ramp.wgsl"),
            ramp_bgl,
        );
        let shapes = make_pipe(
            device,
            "shapes",
            include_str!("../shaders/shapes.wgsl"),
            shapes_bgl,
        );
        let island = make_pipe(
            device,
            "island",
            include_str!("../shaders/island.wgsl"),
            island_bgl,
        );
        let plateau = make_pipe(
            device,
            "plateau",
            include_str!("../shaders/plateau.wgsl"),
            plateau_bgl,
        );
        let path_height = make_pipe(
            device,
            "path-height",
            include_str!("../shaders/path_height.wgsl"),
            path_height_bgl,
        );
        let polygon_height = make_pipe(
            device,
            "polygon-height",
            include_str!("../shaders/polygon_height.wgsl"),
            polygon_height_bgl,
        );
        let heightmap_sample = make_pipe(
            device,
            "heightmap-sample",
            include_str!("../shaders/heightmap_sample.wgsl"),
            heightmap_sample_bgl,
        );
        let river_accum = make_pipe(
            device,
            "river-accum",
            include_str!("../shaders/river_accum.wgsl"),
            river_accum_bgl,
        );
        let river_carve = make_pipe(
            device,
            "river-carve",
            include_str!("../shaders/river_carve.wgsl"),
            river_carve_bgl,
        );
        let stream_power = make_pipe(
            device,
            "stream-power-incision",
            include_str!("../shaders/stream_power_incision.wgsl"),
            stream_power_bgl,
        );
        let amplify_downsample = make_pipe(
            device,
            "amplify-downsample",
            include_str!("../shaders/amplify_downsample.wgsl"),
            amplify_downsample_bgl,
        );
        let amplify_upsample_blend = make_pipe(
            device,
            "amplify-upsample-blend",
            include_str!("../shaders/amplify_upsample_blend.wgsl"),
            amplify_upsample_blend_bgl,
        );
        let effect_filter_range_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("effect-filter-range-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1),
                    storage_rw_buffer_entry(2),
                ],
            });
        let effect_filter_range = make_pipe(
            device,
            "effect-filter-range",
            include_str!("../shaders/effect_filter_range.wgsl"),
            effect_filter_range_bgl,
        );
        let effect_filter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("effect-filter-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                storage_write_entry(2),
                storage_read_buffer_entry(3),
            ],
        });
        let effect_filter = make_pipe(
            device,
            "effect-filter",
            include_str!("../shaders/effect_filter.wgsl"),
            effect_filter_bgl,
        );
        let sculpt_strokes_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sculpt-strokes-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1), // src_original (base neighborhood)
                    tex_read_entry(2), // running_in (chained height)
                    storage_read_buffer_entry(3), // headers
                    storage_read_buffer_entry(4), // points
                    storage_read_buffer_entry(5), // targets (Flatten footprint means)
                    storage_write_entry(6), // stamp_out
                ],
            });
        let sculpt_strokes_edited_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sculpt-strokes-edited-bgl"),
                entries: &[
                    uniform_entry(0),
                    storage_read_buffer_entry(1), // headers
                    storage_read_buffer_entry(2), // points
                    storage_write_entry(3),       // edited_out
                ],
            });
        let sculpt_strokes_flatten_reduce_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sculpt-strokes-flatten-reduce-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1), // running field entering the stroke
                    storage_read_buffer_entry(2), // headers
                    storage_read_buffer_entry(3), // points
                    storage_rw_buffer_entry(4), // partials
                ],
            });
        let sculpt_strokes_flatten_resolve_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sculpt-strokes-flatten-resolve-bgl"),
                entries: &[
                    uniform_entry(0),
                    storage_read_buffer_entry(1), // partials
                    storage_rw_buffer_entry(2),   // targets
                ],
            });
        let sculpt_strokes_reconcile_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sculpt-strokes-reconcile-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1),
                    tex_read_entry(2),
                    storage_write_entry(3),
                ],
            });
        let sculpt_strokes = make_pipe(
            device,
            "sculpt-strokes",
            include_str!("../shaders/sculpt_strokes.wgsl"),
            sculpt_strokes_bgl,
        );
        let sculpt_strokes_edited = make_pipe(
            device,
            "sculpt-strokes-edited",
            include_str!("../shaders/sculpt_strokes_edited.wgsl"),
            sculpt_strokes_edited_bgl,
        );
        let sculpt_strokes_flatten_reduce = make_pipe(
            device,
            "sculpt-strokes-flatten-reduce",
            include_str!("../shaders/sculpt_strokes_flatten_reduce.wgsl"),
            sculpt_strokes_flatten_reduce_bgl,
        );
        let sculpt_strokes_flatten_resolve = make_pipe(
            device,
            "sculpt-strokes-flatten-resolve",
            include_str!("../shaders/sculpt_strokes_flatten_resolve.wgsl"),
            sculpt_strokes_flatten_resolve_bgl,
        );
        let sculpt_strokes_reconcile = make_pipe(
            device,
            "sculpt-strokes-reconcile",
            include_str!("../shaders/sculpt_strokes_reconcile.wgsl"),
            sculpt_strokes_reconcile_bgl,
        );
        let ping = HeightTex::new(device, "ping", w, w);
        let pong = HeightTex::new(device, "pong", w, w);
        let layer_tex = HeightTex::new(device, "layer", w, w);
        let mask_ones = HeightTex::new(device, "mask-ones", w, w);
        let unit_mask = HeightTex::new(device, "unit-mask", w, w);
        let stamp_mask = HeightTex::new(device, "stamp-mask", w, w);
        let sim_side = w;
        let hardness = HeightTex::new(device, "hardness", sim_side, sim_side);
        let water_a = HeightTex::new(device, "water-a", sim_side, sim_side);
        let water_b = HeightTex::new(device, "water-b", sim_side, sim_side);
        let delta = HeightTex::new(device, "thermal-delta", sim_side, sim_side);
        let sed_a = HeightTex::new(device, "sed-a", sim_side, sim_side);
        let sed_b = HeightTex::new(device, "sed-b", sim_side, sim_side);
        let rainfall = HeightTex::new(device, "rainfall", sim_side, sim_side);
        let loose_sediment = HeightTex::new(device, "loose-sediment", sim_side, sim_side);
        let outflow = RgbaTex::new(device, "hydraulic-outflow", sim_side, sim_side);
        let amplify_a = HeightTex::new(device, "amplify-a", sim_side, sim_side);
        let amplify_b = HeightTex::new(device, "amplify-b", sim_side, sim_side);
        let sculpt_stamp = HeightTex::new(device, "sculpt-stamp", w, w);
        let sculpt_stamp_b = HeightTex::new(device, "sculpt-stamp-b", w, w);
        let sculpt_edited = HeightTex::new(device, "sculpt-edited", w, w);
        let effect_filter_range_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("effect-filter-range"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            plan_operations: GpuPlanOperations::new(device),
            plan_resources: GpuPlanResourceCache::default(),
            device_generation: 1,
            active_plan_revision: None,
            deferred_plan_resume: None,
            fill,
            noise,
            blend,
            copy,
            thermal,
            thermal_apply,
            hydraulic_outflow,
            hydraulic,
            blur,
            terrace,
            ramp,
            shapes,
            island,
            plateau,
            river_accum,
            river_carve,
            stream_power,
            amplify_downsample,
            amplify_upsample_blend,
            effect_filter_range,
            effect_filter,
            sculpt_strokes,
            sculpt_strokes_edited,
            sculpt_strokes_flatten_reduce,
            sculpt_strokes_flatten_resolve,
            sculpt_strokes_reconcile,
            path_height,
            polygon_height,
            heightmap_sample,
            uniform_pool: UniformPool::new(device, 64),
            ping,
            pong,
            layer_tex,
            mask_ones,
            unit_mask,
            stamp_mask,
            hardness,
            water_a,
            water_b,
            delta,
            sed_a,
            sed_b,
            rainfall,
            loose_sediment,
            outflow,
            amplify_a,
            amplify_b,
            sculpt_stamp,
            sculpt_stamp_b,
            sculpt_edited,
            effect_filter_range_buffer,
            layer_cache: HashMap::new(),
            source_rasters: HashMap::new(),
            stroke_runtime: HashMap::new(),
            #[cfg(test)]
            source_upload_count: 0,
            dirty: HashSet::new(),
            metrics: HeightfieldMetrics {
                width: w,
                height: w,
                world_size_x: 1000.0,
                world_size_z: 1000.0,
                tile_size: w,
                halo: 0,
            },
            approx_range: (0.0, 120.0),
            current: 0,
            tile_sched: TileScheduler::new(),
            last_dirty_rect: None,
            last_quality: None,
            max_sim_iters_per_tick: 8,
            last_graph: GpuComputeGraph::default(),
            #[cfg(test)]
            executed_kernels: Vec::new(),
            last_eval_stats: GpuEvalStats::default(),
            last_plan_operation_trace: Vec::new(),
            evaluation_timer: GpuEvaluationTimer::try_new(device),
            pending_evaluation_trace: None,
            retired_refinement_completions: Vec::new(),
            #[cfg(test)]
            executed_plan_operations: Vec::new(),
        }
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

    /// Bounding sample rect of tiles touched since last clear (padded for normals).
    pub fn dirty_region(&self, pad: u32) -> Option<SampleRect> {
        self.tile_sched.dirty_bounds(&self.metrics, pad)
    }

    /// Snapshot of dirty tile IDs for viewport debug overlay (does not clear).
    pub fn dirty_tiles(&self) -> &[TileId] {
        &self.tile_sched.dirty
    }

    /// Cap thermal/hydraulic/stream-power iterations for the current interactive refinement phase.
    /// `None` means uncapped (export / full quality).
    pub fn set_simulation_iteration_cap(&mut self, cap: Option<u32>) {
        self.max_sim_iters_per_tick = cap.unwrap_or(u32::MAX);
    }

    pub fn take_dirty_region(&mut self, pad: u32) -> Option<SampleRect> {
        let r = self.dirty_region(pad);
        self.tile_sched.clear();
        r
    }

    fn mark_all_tiles_dirty(&mut self) {
        self.tile_sched.clear();
        for tz in 0..self.metrics.tiles_z() {
            for tx in 0..self.metrics.tiles_x() {
                self.tile_sched.mark_tile(TileId { tx, tz });
            }
        }
    }

    pub fn mark_dirty(&mut self, id: LayerId) {
        self.dirty.insert(id);
    }

    /// Set the texel-space bounds of the most recent local terrain edit.
    pub fn set_dirty_rect(&mut self, rect: Option<(u32, u32, u32, u32)>) {
        self.last_dirty_rect = rect;
    }

    fn mark_tiles_overlapping_rect(&mut self, rect: (u32, u32, u32, u32)) {
        let (x, y, w, h) = rect;
        if w == 0 || h == 0 || self.metrics.width == 0 || self.metrics.height == 0 {
            return;
        }
        let max_x = x
            .saturating_add(w)
            .saturating_sub(1)
            .min(self.metrics.width - 1);
        let max_y = y
            .saturating_add(h)
            .saturating_sub(1)
            .min(self.metrics.height - 1);
        let tx0 = x.min(self.metrics.width - 1) / self.metrics.tile_size;
        let tz0 = y.min(self.metrics.height - 1) / self.metrics.tile_size;
        let tx1 = max_x / self.metrics.tile_size;
        let tz1 = max_y / self.metrics.tile_size;
        self.tile_sched.clear();
        for tz in tz0..=tz1 {
            for tx in tx0..=tx1 {
                self.tile_sched.mark_tile(TileId { tx, tz });
            }
        }
    }

    pub fn mark_dirty_from(&mut self, stack: &LayerStack, id: LayerId) {
        let layers = stack.flatten_layers();
        let mut seen = false;
        for layer in layers {
            if layer.id() == id {
                seen = true;
            }
            if seen {
                self.dirty.insert(layer.id());
            }
        }
    }

    /// Upload a CPU heightfield into the layer cache (WC bridge: bake shapes, keep filters live).
    pub fn ingest_height(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        height: &Heightfield,
        height_range: (f32, f32),
    ) {
        if height.metrics.width == 0 || height.metrics.height == 0 {
            return;
        }
        self.ensure_size(device, height.metrics);
        let w = height.metrics.width;
        let h = height.metrics.height;
        let needs_new = self
            .layer_cache
            .get(&id)
            .map(|t| t.width != w || t.height != h)
            .unwrap_or(true);
        if needs_new {
            self.layer_cache
                .insert(id, HeightTex::new(device, "layer-cache", w, h));
        }
        let dense = height.to_dense();
        let cache = self.layer_cache.get(&id).expect("cache just inserted");
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &cache.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&dense),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.approx_range = height_range;
        self.dirty.remove(&id);
    }

    /// Upload a heightfield into the current ping/pong working buffer.
    pub fn mark_all_dirty(&mut self, stack: &LayerStack) {
        for layer in stack.flatten_layers() {
            self.dirty.insert(layer.id());
        }
    }

    /// Drop all project-owned GPU caches and replace project-sized working textures
    /// with the small resident baseline so a new/opened document starts clean.
    pub fn reset_project_state(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.plan_resources = GpuPlanResourceCache::default();
        self.active_plan_revision = None;
        self.deferred_plan_resume = None;
        self.layer_cache.clear();
        self.source_rasters.clear();
        self.stroke_runtime.clear();
        self.dirty.clear();
        self.last_dirty_rect = None;
        self.last_quality = None;
        self.last_graph = terra_gpu::graph::GpuComputeGraph::default();
        self.last_eval_stats = GpuEvalStats::default();
        self.last_plan_operation_trace.clear();
        self.tile_sched = TileScheduler::new();
        self.approx_range = (0.0, 1.0);
        self.current = 0;
        self.uniform_pool.reset();
        let baseline_metrics = HeightfieldMetrics {
            width: PROJECT_RESET_TEXTURE_EXTENT,
            height: PROJECT_RESET_TEXTURE_EXTENT,
            world_size_x: self.metrics.world_size_x,
            world_size_z: self.metrics.world_size_z,
            tile_size: PROJECT_RESET_TEXTURE_EXTENT,
            halo: 0,
        };
        self.ensure_size(device, baseline_metrics);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu-project-reset"),
        });
        self.fill_slot(device, queue, &mut encoder, TexSlot::Ping, 0.0);
        self.fill_slot(device, queue, &mut encoder, TexSlot::Pong, 0.0);
        self.fill_slot(device, queue, &mut encoder, TexSlot::Layer, 0.0);
        self.fill_slot(device, queue, &mut encoder, TexSlot::MaskOnes, 1.0);
        self.fill_slot(device, queue, &mut encoder, TexSlot::UnitMask, 1.0);
        queue.submit(Some(encoder.finish()));
    }

    pub fn output_texture(&self) -> &wgpu::Texture {
        if self.current == 0 {
            &self.ping.texture
        } else {
            &self.pong.texture
        }
    }

    /// Current evaluated height field view (R32Float) — sample directly from the renderer when formats match.
    pub fn output_texture_view(&self) -> &wgpu::TextureView {
        if self.current == 0 {
            &self.ping.view
        } else {
            &self.pong.view
        }
    }

    /// Alias for [`Self::output_texture_view`].
    pub fn height_texture_view(&self) -> &wgpu::TextureView {
        self.output_texture_view()
    }

    fn ensure_size(&mut self, device: &wgpu::Device, metrics: HeightfieldMetrics) {
        let w = metrics.width.max(PROJECT_RESET_TEXTURE_EXTENT);
        let h = metrics.height.max(PROJECT_RESET_TEXTURE_EXTENT);
        if self.ping.width == w
            && self.ping.height == h
            && self.metrics.world_size_x == metrics.world_size_x
        {
            self.metrics = metrics;
            return;
        }
        self.metrics = metrics;
        self.ping = HeightTex::new(device, "ping", w, h);
        self.pong = HeightTex::new(device, "pong", w, h);
        self.layer_tex = HeightTex::new(device, "layer", w, h);
        self.mask_ones = HeightTex::new(device, "mask-ones", w, h);
        self.unit_mask = HeightTex::new(device, "unit-mask", w, h);
        self.stamp_mask = HeightTex::new(device, "stamp-mask", w, h);
        self.hardness = HeightTex::new(device, "hardness", w, h);
        self.water_a = HeightTex::new(device, "water-a", w, h);
        self.water_b = HeightTex::new(device, "water-b", w, h);
        self.delta = HeightTex::new(device, "thermal-delta", w, h);
        self.sed_a = HeightTex::new(device, "sed-a", w, h);
        self.sed_b = HeightTex::new(device, "sed-b", w, h);
        self.rainfall = HeightTex::new(device, "rainfall", w, h);
        self.loose_sediment = HeightTex::new(device, "loose-sediment", w, h);
        self.outflow = RgbaTex::new(device, "hydraulic-outflow", w, h);
        self.amplify_a = HeightTex::new(device, "amplify-a", w, h);
        self.amplify_b = HeightTex::new(device, "amplify-b", w, h);
        self.sculpt_stamp = HeightTex::new(device, "sculpt-stamp", w, h);
        self.sculpt_stamp_b = HeightTex::new(device, "sculpt-stamp-b", w, h);
        self.sculpt_edited = HeightTex::new(device, "sculpt-edited", w, h);
        self.layer_cache.clear();
        self.stroke_runtime.clear();
        self.dirty.clear();
    }

    fn swap_current(&mut self) {
        self.current = 1 - self.current;
    }

    /// Write uniforms into the next pool slot and return that buffer.
    /// Each dispatch must bind its own slot ÔÇö wgpu applies all `queue.write_buffer`
    /// transfers before the command buffer, so a single shared buffer would make
    /// every pass see only the last write.
    fn write_uniform<T: Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &T,
    ) -> wgpu::Buffer {
        self.uniform_pool.write(device, queue, data)
    }

    fn view_of(&self, slot: TexSlot) -> &wgpu::TextureView {
        match slot {
            TexSlot::Ping => &self.ping.view,
            TexSlot::Pong => &self.pong.view,
            TexSlot::Layer => &self.layer_tex.view,
            TexSlot::MaskOnes => &self.mask_ones.view,
            TexSlot::UnitMask => &self.unit_mask.view,
            TexSlot::StampMask => &self.stamp_mask.view,
            TexSlot::Hardness => &self.hardness.view,
            TexSlot::WaterA => &self.water_a.view,
            TexSlot::WaterB => &self.water_b.view,
            TexSlot::SedA => &self.sed_a.view,
            TexSlot::SedB => &self.sed_b.view,
            TexSlot::Rainfall => &self.rainfall.view,
            TexSlot::LooseSediment => &self.loose_sediment.view,
            TexSlot::SculptStamp => &self.sculpt_stamp.view,
        }
    }

    fn fill_slot(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        slot: TexSlot,
        value: f32,
    ) {
        let u = FillU {
            width: self.metrics.width,
            height: self.metrics.height,
            value,
            _pad: 0.0,
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let view = self.view_of(slot);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fill-bg"),
            layout: &self.fill.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fill"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fill.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                self.metrics.width.div_ceil(8),
                self.metrics.height.div_ceil(8),
                1,
            );
        }
    }

    fn fill_view_extent(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        extent: [u32; 2],
        value: f32,
    ) {
        let [width, height] = extent;
        let uniform = FillU {
            width,
            height,
            value,
            _pad: 0.0,
        };
        let buffer = self.write_uniform(device, queue, &uniform);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fill-extent-bg"),
            layout: &self.fill.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fill-extent"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.fill.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    fn copy_slots(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        src: TexSlot,
        dst: TexSlot,
    ) {
        let u = CopyU {
            width: self.metrics.width,
            height: self.metrics.height,
            region_x: 0,
            region_y: 0,
            region_w: self.metrics.width,
            region_h: self.metrics.height,
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("copy-bg"),
            layout: &self.copy.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(self.view_of(src)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(self.view_of(dst)),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("copy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                self.metrics.width.div_ceil(8),
                self.metrics.height.div_ceil(8),
                1,
            );
        }
    }

    fn noise_type_u(t: FractalNoiseType) -> Option<u32> {
        match t {
            FractalNoiseType::Value => Some(0),
            FractalNoiseType::Perlin => Some(1),
            FractalNoiseType::OpenSimplex => None,
        }
    }

    fn gen_noise(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &NoiseParams,
        dispatch: NoiseDispatch,
    ) {
        self.gen_noise_to(device, queue, encoder, p, dispatch, TexSlot::Layer);
    }

    fn gen_noise_to(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &NoiseParams,
        dispatch: NoiseDispatch,
        destination: TexSlot,
    ) {
        let u = NoiseU {
            width: self.metrics.width,
            height: self.metrics.height,
            world_x: self.metrics.world_size_x,
            world_z: self.metrics.world_size_z,
            seed: (p.seed & 0xFFFF_FFFF) as u32,
            octaves: p.octaves.max(1),
            frequency: p.frequency,
            amplitude: p.amplitude,
            lacunarity: p.lacunarity,
            persistence: p.persistence,
            offset_x: p.offset_x,
            offset_z: p.offset_z,
            remap_min: p.remap_min,
            remap_max: p.remap_max,
            noise_type: dispatch.noise_type,
            mode: dispatch.mode as u32,
            warp_strength: dispatch.warp_strength,
            warp_frequency: dispatch.warp_frequency,
            cell_jitter: dispatch.cell_jitter,
            height_per_cell: dispatch.height_per_cell,
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let destination = self.view_of(destination);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("noise-bg"),
            layout: &self.noise.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(destination),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("noise"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.noise.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                self.metrics.width.div_ceil(8),
                self.metrics.height.div_ceil(8),
                1,
            );
        }
    }

    fn gen_shape(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        u: ShapeU,
    ) {
        let u_buf = self.write_uniform(device, queue, &u);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shapes-bg"),
            layout: &self.shapes.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shapes"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shapes.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                self.metrics.width.div_ceil(8),
                self.metrics.height.div_ceil(8),
                1,
            );
        }
    }

    fn gen_island(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &IslandParams,
    ) {
        let archetype = match p.archetype {
            IslandArchetype::VolcanicHighIsland => 0,
            IslandArchetype::Archipelago => 1,
            IslandArchetype::Atoll => 2,
        };
        let u = IslandU {
            width: self.metrics.width,
            height: self.metrics.height,
            world_x: self.metrics.world_size_x,
            world_z: self.metrics.world_size_z,
            seed: p.seed as u32,
            archetype,
            _pad_u: [0; 2],
            center_u: p.center_u,
            center_v: p.center_v,
            rotation_deg: p.rotation_deg,
            radius: p.radius,
            aspect: p.aspect,
            sea_level: p.sea_level,
            ocean_floor: p.ocean_floor,
            mountain_height: p.mountain_height,
            shelf_width: p.shelf_width,
            shelf_depth: p.shelf_depth,
            beach_width: p.beach_width,
            beach_height: p.beach_height,
            reef_width: p.reef_width,
            reef_depth: p.reef_depth,
            coastline_warp: p.coastline_warp,
            coastline_frequency: p.coastline_frequency,
            mountain_power: p.mountain_power,
            ridge_strength: p.ridge_strength,
            ridge_frequency: p.ridge_frequency,
            lagoon_radius: p.lagoon_radius,
            _pad_f: [0.0; 4],
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("island-bg"),
            layout: &self.island.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("island"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.island.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(
            self.metrics.width.div_ceil(8),
            self.metrics.height.div_ceil(8),
            1,
        );
    }

    fn gen_plateau(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &PlateauParams,
    ) {
        let source = if self.current == 0 {
            TexSlot::Ping
        } else {
            TexSlot::Pong
        };
        self.gen_plateau_between(device, queue, encoder, p, source, TexSlot::Layer);
    }

    fn gen_plateau_between(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &PlateauParams,
        source: TexSlot,
        destination: TexSlot,
    ) {
        let u = PlateauU {
            width: self.metrics.width,
            height: self.metrics.height,
            low: p.low,
            high: p.high,
            soft: p.soft,
            _pad: [0.0; 3],
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let src_view = self.view_of(source);
        let dst_view = self.view_of(destination);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("plateau-bg"),
            layout: &self.plateau.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(dst_view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("plateau"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.plateau.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(
            self.metrics.width.div_ceil(8),
            self.metrics.height.div_ceil(8),
            1,
        );
    }

    fn run_path_height(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &PathParams,
    ) {
        let samples = terra_core::generators::path_samples(
            p,
            self.metrics.world_size_x,
            self.metrics.world_size_z,
        );
        let mut points: Vec<[f32; 4]> = samples
            .iter()
            .map(|sample| [sample.x, sample.z, sample.height, sample.width])
            .collect();
        let point_count = points.len() as u32;
        if points.is_empty() {
            points.push([0.0; 4]);
        }
        let point_buffer = make_storage_buffer(
            device,
            queue,
            "path-height-points",
            bytemuck::cast_slice(&points),
        );
        let uniform = PathU {
            width: self.metrics.width,
            height: self.metrics.height,
            world_x: self.metrics.world_size_x,
            world_z: self.metrics.world_size_z,
            point_count,
            carve: u32::from(p.carve),
            seed: p.seed as u32,
            _pad0: 0,
            base_width: p.width,
            falloff: p.falloff,
            noise_strength: p.noise_strength,
            noise_scale: p.noise_scale,
            height_offset: p.height_offset,
            profile: p.profile,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let uniform_buffer = self.write_uniform(device, queue, &uniform);
        let src = if self.current == 0 {
            &self.ping.view
        } else {
            &self.pong.view
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path-height-bg"),
            layout: &self.path_height.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: point_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("path-height"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.path_height.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            self.metrics.width.div_ceil(8),
            self.metrics.height.div_ceil(8),
            1,
        );
    }

    fn run_polygon_height(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &PolygonHeightParams,
    ) {
        let mut points: Vec<[f32; 4]> = p
            .points
            .iter()
            .map(|point| [point[0], point[1], 0.0, 0.0])
            .collect();
        let point_count = points.len() as u32;
        if points.is_empty() {
            points.push([0.0; 4]);
        }
        let point_buffer = make_storage_buffer(
            device,
            queue,
            "polygon-height-points",
            bytemuck::cast_slice(&points),
        );
        let short_axis = self
            .metrics
            .world_size_x
            .min(self.metrics.world_size_z)
            .max(1.0);
        let uniform = PolygonHeightU {
            width: self.metrics.width,
            height: self.metrics.height,
            world_x: self.metrics.world_size_x,
            world_z: self.metrics.world_size_z,
            point_count,
            mode: match p.mode {
                PolygonHeightMode::RaiseBy => 0,
                PolygonHeightMode::SetElevation => 1,
            },
            carve: u32::from(p.carve),
            _pad0: 0,
            target_height: p.height,
            falloff: (p.falloff.clamp(0.0, 0.5) * short_axis).max(1.0e-3),
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let uniform_buffer = self.write_uniform(device, queue, &uniform);
        let src = if self.current == 0 {
            &self.ping.view
        } else {
            &self.pong.view
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("polygon-height-bg"),
            layout: &self.polygon_height.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: point_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("polygon-height"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.polygon_height.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            self.metrics.width.div_ceil(8),
            self.metrics.height.div_ceil(8),
            1,
        );
    }

    fn run_procedural_crater(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &EffectFilterParams,
    ) {
        self.fill_slot(device, queue, encoder, TexSlot::SculptStamp, 80.0);
        let spec = effect_filter_gpu_spec(p)
            .expect("planner admitted only an executable procedural Crater");
        let uniform = self.effect_filter_uniform(p, spec.mode, 1, (0, 0, 0, 0));
        let uniform_buffer = self.write_uniform(device, queue, &uniform);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("procedural-crater-bg"),
            layout: &self.effect_filter.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.sculpt_stamp.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.effect_filter_range_buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("procedural-crater"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.effect_filter.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            self.metrics.width.div_ceil(8),
            self.metrics.height.div_ceil(8),
            1,
        );
    }

    fn run_procedural_shape(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &ProceduralShapeParams,
    ) -> Result<(), GpuError> {
        match p.generator {
            ProceduralGenerator::Mountain => {
                let q = &p.mountain;
                self.gen_shape(
                    device,
                    queue,
                    encoder,
                    ShapeU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        world_x: self.metrics.world_size_x,
                        world_z: self.metrics.world_size_z,
                        seed: q.base.seed as u32,
                        octaves: q.base.octaves.max(1),
                        frequency: q.base.frequency,
                        amplitude: q.base.amplitude,
                        lacunarity: q.base.lacunarity,
                        persistence: q.base.persistence,
                        offset_x: q.base.offset_x,
                        offset_z: q.base.offset_z,
                        ridge_sharpness: q.ridge_sharpness,
                        range_angle: q.range_angle,
                        range_width: q.range_width,
                        wave_frequency: 0.0,
                        asymmetry: 0.0,
                        depth: 0.0,
                        canyon_width: 0.0,
                        meander: q.crest_detail,
                        shape_mode: 0,
                        _pad: 0,
                    },
                );
                self.expand_range(0.0, q.base.amplitude);
            }
            ProceduralGenerator::Hills => {
                let noise_type = Self::noise_type_u(p.hills.noise).ok_or_else(|| {
                    cpu_required(
                        GpuFallbackCode::UnsupportedOptions,
                        "procedural shape",
                        "Hills noise type is outside the compiled GPU plan",
                    )
                })?;
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    &p.hills.base,
                    NoiseDispatch::new(noise_type, NoiseKernelMode::Fbm),
                );
                let amplitude = p.hills.base.amplitude.abs();
                self.expand_range(-amplitude, amplitude);
            }
            ProceduralGenerator::Plateau => {
                let noise_type = Self::noise_type_u(p.hills.noise).ok_or_else(|| {
                    cpu_required(
                        GpuFallbackCode::UnsupportedOptions,
                        "procedural shape",
                        "Plateau noise type is outside the compiled GPU plan",
                    )
                })?;
                self.gen_noise_to(
                    device,
                    queue,
                    encoder,
                    &p.hills.base,
                    NoiseDispatch::new(noise_type, NoiseKernelMode::Fbm),
                    TexSlot::SculptStamp,
                );
                self.gen_plateau_between(
                    device,
                    queue,
                    encoder,
                    &p.plateau,
                    TexSlot::SculptStamp,
                    TexSlot::Layer,
                );
                self.expand_range(p.plateau.low, p.plateau.high);
            }
            ProceduralGenerator::Mesa => {
                let q = &p.mesa;
                self.gen_shape(
                    device,
                    queue,
                    encoder,
                    ShapeU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        world_x: self.metrics.world_size_x,
                        world_z: self.metrics.world_size_z,
                        seed: q.seed as u32,
                        octaves: 3,
                        frequency: 0.001,
                        amplitude: q.height,
                        lacunarity: 2.0,
                        persistence: 0.5,
                        offset_x: q.center_u,
                        offset_z: q.center_v,
                        ridge_sharpness: q.edge_steepness,
                        range_angle: 0.0,
                        range_width: q.radius,
                        wave_frequency: 0.0,
                        asymmetry: 0.0,
                        depth: q.cap_noise,
                        canyon_width: 0.0,
                        meander: q.soft,
                        shape_mode: 5,
                        _pad: 0,
                    },
                );
                self.expand_range(0.0, q.height);
            }
            ProceduralGenerator::Volcano => {
                let q = &p.volcano;
                self.gen_shape(
                    device,
                    queue,
                    encoder,
                    ShapeU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        world_x: self.metrics.world_size_x,
                        world_z: self.metrics.world_size_z,
                        seed: q.seed as u32,
                        octaves: 3,
                        frequency: 0.001,
                        amplitude: q.height,
                        lacunarity: 2.0,
                        persistence: 0.5,
                        offset_x: q.center_u,
                        offset_z: q.center_v,
                        ridge_sharpness: q.flank_power,
                        range_angle: 0.0,
                        range_width: q.radius,
                        wave_frequency: 0.0,
                        asymmetry: 0.0,
                        depth: q.crater_depth,
                        canyon_width: q.crater_radius,
                        meander: q.roughness,
                        shape_mode: 4,
                        _pad: 0,
                    },
                );
                self.expand_range(0.0, q.height);
            }
            ProceduralGenerator::Canyon => {
                let q = &p.canyon;
                self.gen_shape(
                    device,
                    queue,
                    encoder,
                    ShapeU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        world_x: self.metrics.world_size_x,
                        world_z: self.metrics.world_size_z,
                        seed: q.seed as u32,
                        octaves: 1,
                        frequency: 1.0,
                        amplitude: 1.0,
                        lacunarity: 2.0,
                        persistence: 0.5,
                        offset_x: 0.0,
                        offset_z: 0.0,
                        ridge_sharpness: 0.0,
                        range_angle: 0.0,
                        range_width: 0.0,
                        wave_frequency: 0.0,
                        asymmetry: 0.0,
                        depth: q.depth,
                        canyon_width: q.width,
                        meander: q.meander,
                        shape_mode: 2,
                        _pad: 0,
                    },
                );
                self.expand_range(-q.depth, 0.0);
            }
            ProceduralGenerator::Noise => {
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    &p.noise,
                    NoiseDispatch::new(1, NoiseKernelMode::Perlin),
                );
                let amplitude = p.noise.amplitude.abs();
                self.expand_range(-amplitude, amplitude);
            }
            ProceduralGenerator::Crater => {
                self.run_procedural_crater(device, queue, encoder, &p.crater);
                self.expand_range(80.0 - p.crater.amount.abs(), 80.0 + p.crater.amount.abs());
            }
            ProceduralGenerator::Dunes => {
                return Err(cpu_required(
                    GpuFallbackCode::UnsupportedOptions,
                    "procedural shape",
                    "Dunes generator is not parity-covered",
                ));
            }
        }
        Ok(())
    }

    fn river_accumulation_iters(&self, quality: PreviewQuality) -> u32 {
        match quality {
            PreviewQuality::Draft => 12,
            PreviewQuality::Medium => 32,
            PreviewQuality::Full | PreviewQuality::Export => {
                (self.metrics.width.min(self.metrics.height) / 4).clamp(48, 160)
            }
        }
    }

    /// Run the shared iterative D8 accumulation preview against `height_slot` and
    /// return the scratch texture containing the final accumulation field.
    fn run_river_accumulation(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        height_slot: TexSlot,
        quality: PreviewQuality,
    ) -> TexSlot {
        let iters = self.river_accumulation_iters(quality);

        // Seed accumulation with unit rainfall.
        self.fill_slot(device, queue, encoder, TexSlot::WaterA, 1.0);
        self.fill_slot(device, queue, encoder, TexSlot::WaterB, 0.0);

        let accum_u = RiverAccumU {
            width: self.metrics.width,
            height: self.metrics.height,
            _p0: 0.0,
            _p1: 0.0,
        };

        let mut src_a = true;
        for _ in 0..iters {
            let u_buf = self.write_uniform(device, queue, &accum_u);
            let (acc_in, acc_out) = if src_a {
                (&self.water_a.view, &self.water_b.view)
            } else {
                (&self.water_b.view, &self.water_a.view)
            };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("river-accum-bg"),
                layout: &self.river_accum.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(self.view_of(height_slot)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(acc_in),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(acc_out),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("river-accum"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.river_accum.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(
                    self.metrics.width.div_ceil(8),
                    self.metrics.height.div_ceil(8),
                    1,
                );
            }
            src_a = !src_a;
        }

        if src_a {
            TexSlot::WaterA
        } else {
            TexSlot::WaterB
        }
    }

    fn run_river_carve(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &terra_core::layer::RiverCarveParams,
        quality: PreviewQuality,
    ) {
        let height_slot = if self.current == 0 {
            TexSlot::Ping
        } else {
            TexSlot::Pong
        };
        let accum_slot = self.run_river_accumulation(device, queue, encoder, height_slot, quality);

        let carve_u = RiverCarveU {
            width: self.metrics.width,
            height: self.metrics.height,
            threshold: p.accumulation_threshold.max(1.0),
            depth: p.depth,
            channel_width: p.width.max(1.0),
            bank_smooth: p.bank_smooth.max(0.0),
            max_radius: match quality {
                PreviewQuality::Draft => 12,
                PreviewQuality::Medium => 20,
                PreviewQuality::Full | PreviewQuality::Export => RIVER_CARVE_MAX_RADIUS,
            },
            _pad: 0,
        };
        let u_buf = self.write_uniform(device, queue, &carve_u);
        let acc_view = self.view_of(accum_slot);
        let (src, dst) = if self.current == 0 {
            (&self.ping.view, &self.pong.view)
        } else {
            (&self.pong.view, &self.ping.view)
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("river-carve-bg"),
            layout: &self.river_carve.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(acc_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(dst),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("river-carve"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.river_carve.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                self.metrics.width.div_ceil(8),
                self.metrics.height.div_ceil(8),
                1,
            );
        }
        self.swap_current();
        self.expand_range(self.approx_range.0 - p.depth * 2.0, self.approx_range.1);
    }

    fn run_stream_power(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &terra_core::layer::StreamPowerParams,
        quality: PreviewQuality,
    ) {
        let authored_iters = match quality {
            PreviewQuality::Draft => p.iterations.clamp(1, 8),
            PreviewQuality::Medium => p.iterations.clamp(1, 16),
            PreviewQuality::Full | PreviewQuality::Export => p.iterations.max(1),
        };
        // Mirror the CPU processor's default level-step averaging. Non-default
        // authored level controls are rejected by the planner, and document-level
        // schedule variants remain part of the configuration-fallback work in #138.
        let base = match quality {
            PreviewQuality::Draft => draft_sim_levels(self.metrics.width),
            PreviewQuality::Medium | PreviewQuality::Full | PreviewQuality::Export => {
                default_sim_levels(self.metrics.width)
            }
        };
        let levels = LevelStepSettings::default().schedule_for_filter(
            base,
            p.level_count,
            p.start_level,
            p.level_step_strength,
            &p.level_step_curve,
            quality,
        );
        let (iter_scale, effect_scale) = if levels.is_empty() {
            (1.0, 1.0)
        } else {
            let count = levels.len() as f32;
            (
                levels.iter().map(|level| level.iter_scale).sum::<f32>() / count,
                levels.iter().map(|level| level.effect_scale).sum::<f32>() / count,
            )
        };
        let iters = ((authored_iters as f32 * iter_scale).round() as u32)
            .max(1)
            .min(match quality {
                PreviewQuality::Draft => self.max_sim_iters_per_tick.max(1),
                PreviewQuality::Medium | PreviewQuality::Full | PreviewQuality::Export => u32::MAX,
            });
        let drainage_stride = match quality {
            PreviewQuality::Draft => p.drainage_reuse_stride.max(2),
            PreviewQuality::Medium | PreviewQuality::Full | PreviewQuality::Export => {
                p.drainage_reuse_stride.max(1)
            }
        };
        let cell_area = (self.metrics.dx() * self.metrics.dz()).max(1.0e-6);
        let uniform = StreamPowerU {
            width: self.metrics.width,
            height: self.metrics.height,
            k: (p.k * effect_scale).max(0.0),
            m: p.m.max(0.0),
            n: p.n.max(0.0),
            dt: p.dt.max(0.0),
            uplift: p.uplift_rate,
            base_level: p.base_level,
            cell_area,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };

        self.fill_slot(
            device,
            queue,
            encoder,
            TexSlot::Hardness,
            p.hardness.clamp(0.0, 1.0),
        );

        let mut accum_slot = TexSlot::WaterA;
        for iter in 0..iters {
            let height_slot = if self.current == 0 {
                TexSlot::Ping
            } else {
                TexSlot::Pong
            };
            if iter == 0 || iter % drainage_stride == 0 {
                accum_slot =
                    self.run_river_accumulation(device, queue, encoder, height_slot, quality);
            }

            let u_buf = self.write_uniform(device, queue, &uniform);
            let (src, dst) = if self.current == 0 {
                (&self.ping.view, &self.pong.view)
            } else {
                (&self.pong.view, &self.ping.view)
            };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stream-power-bg"),
                layout: &self.stream_power.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(self.view_of(accum_slot)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.hardness.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(dst),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("stream-power-incision"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.stream_power.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(
                    self.metrics.width.div_ceil(8),
                    self.metrics.height.div_ceil(8),
                    1,
                );
            }
            self.swap_current();
        }

        let iter_scale = iters as f32;
        self.expand_range(
            (self.approx_range.0 - 50.0 * iter_scale + p.uplift_rate * iter_scale)
                .max(p.base_level),
            (self.approx_range.1 + p.uplift_rate.max(0.0) * iter_scale).max(p.base_level),
        );
    }

    fn run_amplify_accumulation(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        level_res: u32,
        height_a: bool,
        quality: PreviewQuality,
    ) -> bool {
        let water_a = self.water_a.view.clone();
        let water_b = self.water_b.view.clone();
        self.fill_view_extent(device, queue, encoder, &water_a, [level_res; 2], 1.0);
        self.fill_view_extent(device, queue, encoder, &water_b, [level_res; 2], 0.0);
        let iterations = match quality {
            PreviewQuality::Draft => 12,
            PreviewQuality::Medium => 32,
            PreviewQuality::Full | PreviewQuality::Export => (level_res / 4).clamp(48, 160),
        };
        let uniform = RiverAccumU {
            width: level_res,
            height: level_res,
            _p0: 0.0,
            _p1: 0.0,
        };
        let mut src_a = true;
        for _ in 0..iterations {
            let buffer = self.write_uniform(device, queue, &uniform);
            let height = if height_a {
                &self.amplify_a.view
            } else {
                &self.amplify_b.view
            };
            let (acc_in, acc_out) = if src_a {
                (&self.water_a.view, &self.water_b.view)
            } else {
                (&self.water_b.view, &self.water_a.view)
            };
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("amplify-river-accum-bg"),
                layout: &self.river_accum.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(height),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(acc_in),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(acc_out),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amplify-river-accum"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.river_accum.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(level_res.div_ceil(8), level_res.div_ceil(8), 1);
            drop(pass);
            src_a = !src_a;
        }
        src_a
    }

    fn run_multi_scale_amplify(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        authored: &MultiScaleAmplifyParams,
        quality: PreviewQuality,
    ) {
        let mut params = authored.clone();
        match quality {
            PreviewQuality::Draft => {
                params.thermal_iters = params.thermal_iters.clamp(1, 6);
                params.spe_iters = params.spe_iters.min(2);
                params.level_count = if params.level_count == 0 {
                    2
                } else {
                    params.level_count.min(2)
                };
            }
            PreviewQuality::Medium => {
                params.thermal_iters = params.thermal_iters.clamp(1, 10);
                params.spe_iters = params.spe_iters.min(4);
            }
            PreviewQuality::Full | PreviewQuality::Export => {}
        }
        let levels = match quality {
            PreviewQuality::Draft => {
                amplify_sim_levels(self.metrics.width, params.level_count.clamp(1, 2))
            }
            PreviewQuality::Medium | PreviewQuality::Full | PreviewQuality::Export => {
                amplify_sim_levels(self.metrics.width, params.level_count)
            }
        };
        let required_side = levels
            .iter()
            .map(|level| level.resolution)
            .max()
            .unwrap_or(self.metrics.width)
            .max(1);
        if self.amplify_a.width < required_side || self.amplify_a.height < required_side {
            self.hardness = HeightTex::new(device, "hardness", required_side, required_side);
            self.water_a = HeightTex::new(device, "water-a", required_side, required_side);
            self.water_b = HeightTex::new(device, "water-b", required_side, required_side);
            self.delta = HeightTex::new(device, "thermal-delta", required_side, required_side);
            self.sed_a = HeightTex::new(device, "sed-a", required_side, required_side);
            self.sed_b = HeightTex::new(device, "sed-b", required_side, required_side);
            self.rainfall = HeightTex::new(device, "rainfall", required_side, required_side);
            self.loose_sediment =
                HeightTex::new(device, "loose-sediment", required_side, required_side);
            self.outflow = RgbaTex::new(device, "hydraulic-outflow", required_side, required_side);
            self.amplify_a = HeightTex::new(device, "amplify-a", required_side, required_side);
            self.amplify_b = HeightTex::new(device, "amplify-b", required_side, required_side);
        }
        let hardness = match params.hardness_source {
            MaskSource::Constant(value) => value,
            MaskSource::None => params.hardness,
            _ => unreachable!("planner rejected non-uniform amplify hardness"),
        }
        .clamp(0.0, 1.0);
        let ridge_lock = match params.ridge_lock {
            MaskSource::Constant(value) => value,
            MaskSource::None => 0.0,
            _ => unreachable!("planner rejected non-uniform amplify ridge lock"),
        }
        .clamp(0.0, 1.0);

        let hardness_view = self.hardness.view.clone();
        let rainfall_view = self.rainfall.view.clone();
        let loose_view = self.loose_sediment.view.clone();
        for (index, level) in levels.iter().enumerate() {
            let level_res = level.resolution;
            let source_slot = if self.current == 0 {
                TexSlot::Ping
            } else {
                TexSlot::Pong
            };
            self.copy_slots(device, queue, encoder, source_slot, TexSlot::Layer);

            let downsample = AmplifyDownsampleU {
                src_width: self.metrics.width,
                src_height: self.metrics.height,
                dst_width: level_res,
                dst_height: level_res,
                area_mode: u32::from(
                    level_res < self.metrics.width || level_res < self.metrics.height,
                ),
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let buffer = self.write_uniform(device, queue, &downsample);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("amplify-downsample-bg"),
                layout: &self.amplify_downsample.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.amplify_a.view),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("amplify-downsample"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.amplify_downsample.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(level_res.div_ceil(8), level_res.div_ceil(8), 1);
            }

            self.fill_view_extent(
                device,
                queue,
                encoder,
                &hardness_view,
                [level_res; 2],
                hardness,
            );
            let fine_t = if levels.len() <= 1 {
                1.0
            } else {
                index as f32 / (levels.len() - 1) as f32
            };
            let thermal_w = (1.0 - 0.65 * fine_t) * level.iter_scale;
            let spe_w = (0.25 + 0.75 * fine_t) * level.effect_scale;
            let dep_w = fine_t * level.effect_scale;
            let level_dx = self.metrics.world_size_x / level_res.max(1) as f32;
            let level_dz = self.metrics.world_size_z / level_res.max(1) as f32;

            let thermal_iters = ((params.thermal_iters as f32 * thermal_w).round() as u32).max(1);
            let thermal = ThermalU {
                width: level_res,
                height: level_res,
                dx: level_dx,
                talus: params.talus_angle_deg.to_radians().tan() * level_dx,
                strength: (params.thermal_strength * thermal_w).clamp(0.0, 1.0),
                _p2: 0.0,
                _p3: 0.0,
                _pad: 0.0,
            };
            let mut height_a = true;
            for _ in 0..thermal_iters {
                let uniform = self.write_uniform(device, queue, &thermal);
                let (src, dst) = if height_a {
                    (&self.amplify_a.view, &self.amplify_b.view)
                } else {
                    (&self.amplify_b.view, &self.amplify_a.view)
                };
                let delta_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("amplify-thermal-delta-bg"),
                    layout: &self.thermal.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.delta.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.hardness.view),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("amplify-thermal-delta"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.thermal.pipeline);
                    pass.set_bind_group(0, &delta_group, &[]);
                    pass.dispatch_workgroups(level_res.div_ceil(8), level_res.div_ceil(8), 1);
                }
                let apply_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("amplify-thermal-apply-bg"),
                    layout: &self.thermal_apply.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.delta.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(dst),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("amplify-thermal-apply"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.thermal_apply.pipeline);
                    pass.set_bind_group(0, &apply_group, &[]);
                    pass.dispatch_workgroups(level_res.div_ceil(8), level_res.div_ceil(8), 1);
                }
                height_a = !height_a;
            }

            if params.spe_strength > 1.0e-6 && params.spe_iters > 0 {
                let spe_iters = ((params.spe_iters as f32 * spe_w).round() as u32)
                    .max(if spe_w > 0.15 { 1 } else { 0 });
                let stream_power = StreamPowerU {
                    width: level_res,
                    height: level_res,
                    k: 0.05 * params.spe_strength * spe_w,
                    m: 0.5,
                    n: 1.0,
                    dt: 0.85,
                    uplift: 0.0,
                    base_level: 0.0,
                    cell_area: (level_dx * level_dz).max(1.0e-6),
                    _pad0: 0.0,
                    _pad1: 0.0,
                    _pad2: 0.0,
                };
                for _ in 0..spe_iters {
                    let accum_a = self.run_amplify_accumulation(
                        device, queue, encoder, level_res, height_a, quality,
                    );
                    let uniform = self.write_uniform(device, queue, &stream_power);
                    let (src, dst) = if height_a {
                        (&self.amplify_a.view, &self.amplify_b.view)
                    } else {
                        (&self.amplify_b.view, &self.amplify_a.view)
                    };
                    let accumulation = if accum_a {
                        &self.water_a.view
                    } else {
                        &self.water_b.view
                    };
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("amplify-stream-power-bg"),
                        layout: &self.stream_power.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(accumulation),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&self.hardness.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(dst),
                            },
                        ],
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("amplify-stream-power"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.stream_power.pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(level_res.div_ceil(8), level_res.div_ceil(8), 1);
                    drop(pass);
                    height_a = !height_a;
                }
            }

            if params.deposition_strength > 1.0e-6 && dep_w > 0.2 {
                let hydraulic = terra_core::layer::HydraulicErosionParams {
                    iterations: ((8.0 * dep_w).round() as u32).max(2),
                    rainfall: 0.02 * dep_w,
                    evaporation: 0.015,
                    capacity: 0.12,
                    erosion: 0.12 * dep_w,
                    deposition: (0.55 * params.deposition_strength * dep_w).clamp(0.0, 1.0),
                    timestep: 0.2,
                    hardness: 0.0,
                    hardness_source: MaskSource::None,
                    fan_boost: 0.8 * params.deposition_strength,
                    floodplain_bias: 0.5 * params.deposition_strength,
                    bank_slip: 0.0,
                    sediment_softness: 0.0,
                    ..Default::default()
                };
                let water_a = self.water_a.view.clone();
                let water_b = self.water_b.view.clone();
                let sed_a = self.sed_a.view.clone();
                let sed_b = self.sed_b.view.clone();
                self.fill_view_extent(device, queue, encoder, &water_a, [level_res; 2], 0.0);
                self.fill_view_extent(device, queue, encoder, &water_b, [level_res; 2], 0.0);
                self.fill_view_extent(device, queue, encoder, &sed_a, [level_res; 2], 0.0);
                self.fill_view_extent(device, queue, encoder, &sed_b, [level_res; 2], 0.0);
                self.fill_view_extent(device, queue, encoder, &rainfall_view, [level_res; 2], 1.0);
                self.fill_view_extent(device, queue, encoder, &loose_view, [level_res; 2], 0.0);
                let uniform = HydraulicU {
                    width: level_res,
                    height: level_res,
                    timestep: clamp_timestep_cfl(hydraulic.timestep, level_dx, 4.0),
                    rainfall: hydraulic.rainfall,
                    evaporation: hydraulic.evaporation,
                    erosion: hydraulic.erosion,
                    deposition: hydraulic.deposition,
                    capacity: hydraulic.capacity,
                    fan_boost: hydraulic.fan_boost,
                    floodplain_bias: hydraulic.floodplain_bias,
                    dx: level_dx,
                    incision_bias: hydraulic.incision_bias.max(0.05),
                    bedrock_k: hydraulic.bedrock_hardness.clamp(0.0, 1.0),
                    sediment_k: hydraulic.sediment_hardness.clamp(0.0, 1.0),
                    layered: 0.0,
                    _pad1: 0.0,
                };
                let mut water_flip = false;
                for _ in 0..hydraulic.iterations {
                    let buffer = self.write_uniform(device, queue, &uniform);
                    let (height_src, height_dst) = if height_a {
                        (&self.amplify_a.view, &self.amplify_b.view)
                    } else {
                        (&self.amplify_b.view, &self.amplify_a.view)
                    };
                    let (water_src, water_dst) = if water_flip {
                        (&self.water_b.view, &self.water_a.view)
                    } else {
                        (&self.water_a.view, &self.water_b.view)
                    };
                    let (sed_src, sed_dst) = if water_flip {
                        (&self.sed_b.view, &self.sed_a.view)
                    } else {
                        (&self.sed_a.view, &self.sed_b.view)
                    };
                    let outflow_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("amplify-hydraulic-outflow-bg"),
                        layout: &self.hydraulic_outflow.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(height_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(water_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&self.outflow.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(&self.rainfall.view),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("amplify-hydraulic-outflow"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.hydraulic_outflow.pipeline);
                        pass.set_bind_group(0, &outflow_group, &[]);
                        pass.dispatch_workgroups(level_res.div_ceil(8), level_res.div_ceil(8), 1);
                    }
                    let hydraulic_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("amplify-hydraulic-bg"),
                        layout: &self.hydraulic.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(height_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(water_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(sed_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(&self.outflow.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::TextureView(height_dst),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(water_dst),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: wgpu::BindingResource::TextureView(sed_dst),
                            },
                            wgpu::BindGroupEntry {
                                binding: 8,
                                resource: wgpu::BindingResource::TextureView(&self.hardness.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 9,
                                resource: wgpu::BindingResource::TextureView(&self.rainfall.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 10,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.loose_sediment.view,
                                ),
                            },
                        ],
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("amplify-hydraulic"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.hydraulic.pipeline);
                    pass.set_bind_group(0, &hydraulic_group, &[]);
                    pass.dispatch_workgroups(level_res.div_ceil(8), level_res.div_ceil(8), 1);
                    drop(pass);
                    height_a = !height_a;
                    water_flip = !water_flip;
                }
            }

            let blend = AmplifyBlendU {
                src_width: level_res,
                src_height: level_res,
                dst_width: self.metrics.width,
                dst_height: self.metrics.height,
                hardness,
                ridge_lock,
                lock_strength: params.lock_strength,
                detail_boost: params.detail_boost,
            };
            let buffer = self.write_uniform(device, queue, &blend);
            let processed = if height_a {
                &self.amplify_a.view
            } else {
                &self.amplify_b.view
            };
            let destination = if self.current == 0 {
                &self.pong.view
            } else {
                &self.ping.view
            };
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("amplify-upsample-blend-bg"),
                layout: &self.amplify_upsample_blend.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(processed),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(destination),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amplify-upsample-blend"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.amplify_upsample_blend.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                self.metrics.width.div_ceil(8),
                self.metrics.height.div_ceil(8),
                1,
            );
            drop(pass);
            self.swap_current();
        }
    }

    fn blend_into_current(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        opacity: f32,
        mode: BlendMode,
    ) -> Result<(), GpuError> {
        self.blend_into_current_with_mask(
            device,
            queue,
            encoder,
            opacity,
            mode,
            [TexSlot::MaskOnes, TexSlot::UnitMask],
        )
    }

    fn blend_into_current_with_mask(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        opacity: f32,
        mode: BlendMode,
        masks: [TexSlot; 2],
    ) -> Result<(), GpuError> {
        let u = BlendU {
            width: self.metrics.width,
            height: self.metrics.height,
            opacity,
            mode: gpu_blend_mode(mode).ok_or_else(|| {
                cpu_required(
                    GpuFallbackCode::BlendMode,
                    "blend",
                    format!("{mode:?} is not implemented"),
                )
            })?,
            region_x: 0,
            region_y: 0,
            region_w: self.metrics.width,
            region_h: self.metrics.height,
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let src_ping = self.current == 0;
        let (base, dst) = if src_ping {
            (&self.ping.view, &self.pong.view)
        } else {
            (&self.pong.view, &self.ping.view)
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blend-bg"),
            layout: &self.blend.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(base),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(self.view_of(masks[0])),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(self.view_of(masks[1])),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(dst),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("blend"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blend.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                self.metrics.width.div_ceil(8),
                self.metrics.height.div_ceil(8),
                1,
            );
        }
        self.swap_current();
        Ok(())
    }

    fn scale_iters(quality: PreviewQuality, iters: u32) -> u32 {
        match quality {
            // Draft must still read as a real filter change (WC interactive), not a no-op.
            PreviewQuality::Draft => iters.clamp(2, 8),
            PreviewQuality::Medium => iters.clamp(4, 12),
            PreviewQuality::Full | PreviewQuality::Export => iters.max(1),
        }
    }

    fn effect_filter_iters(quality: PreviewQuality, p: &EffectFilterParams) -> u32 {
        match effect_filter_gpu_spec(p).map(|spec| spec.passes) {
            Some(EffectFilterGpuPasses::LegacyQualityScaled) => {
                Self::scale_iters(quality, p.iterations.max(1)).min(8)
            }
            Some(EffectFilterGpuPasses::Once) | None => 1,
        }
    }

    fn blur_iters(p: &terra_core::layer::BlurParams) -> u32 {
        p.iterations.clamp(1, 8)
    }

    /// Executed iteration count for a layer's kernel — the single source of truth
    /// shared by the kernel dispatch and the dirty-region halo sizing so the two
    /// never disagree about how far a filter reaches.
    fn dirty_dispatch_extent(&self) -> (u32, u32, u32, u32, u32, u32) {
        // Returns (region_x, region_y, region_w, region_h, groups_x, groups_y).
        // `last_dirty_rect` is already expanded by the plan halo in `evaluate`, so
        // no further padding here — this is exactly the region the kernels rewrite.
        if let Some((x, y, w, h)) = self.last_dirty_rect {
            let gx = w.div_ceil(8);
            let gy = h.div_ceil(8);
            (x, y, w, h, gx.max(1), gy.max(1))
        } else {
            let w = self.metrics.width;
            let h = self.metrics.height;
            (0, 0, 0, 0, w.div_ceil(8), h.div_ceil(8))
        }
    }

    fn reduce_effect_filter_range(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // Ordered-f32 encodings of +infinity (min initializer) and -infinity
        // (max initializer). The WGSL transform preserves total numeric ordering
        // across negative and positive finite terrain heights.
        let initial = [0xff80_0000u32, 0x007f_ffffu32];
        queue.write_buffer(
            &self.effect_filter_range_buffer,
            0,
            bytemuck::cast_slice(&initial),
        );
        let uniform = EffectRangeU {
            width: self.metrics.width,
            height: self.metrics.height,
            _pad0: 0,
            _pad1: 0,
        };
        let uniform = self.write_uniform(device, queue, &uniform);
        let src = if self.current == 0 {
            &self.ping.view
        } else {
            &self.pong.view
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("effect-filter-range-bg"),
            layout: &self.effect_filter_range.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.effect_filter_range_buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("effect-filter-range"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.effect_filter_range.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            self.metrics.width.div_ceil(8),
            self.metrics.height.div_ceil(8),
            1,
        );
    }

    fn effect_filter_uniform(
        &self,
        p: &EffectFilterParams,
        mode: u32,
        iterations: u32,
        region: (u32, u32, u32, u32),
    ) -> EffectFilterU {
        let (region_x, region_y, region_w, region_h) = region;
        EffectFilterU {
            width: self.metrics.width,
            height: self.metrics.height,
            world_x: self.metrics.world_size_x,
            world_z: self.metrics.world_size_z,
            mode,
            radius: p.radius.clamp(1, EFFECT_FILTER_MAX_RADIUS),
            iterations,
            seed: (p.seed & 0xFFFF_FFFF) as u32,
            strength: p.strength.clamp(0.0, 1.0),
            amount: p.amount,
            frequency: p.effective_frequency(),
            sea_level: p.sea_level,
            beach_width: p
                .beach_width
                .max(p.crater_radius * self.metrics.world_size_x * 0.5),
            slope_min: p.slope_min,
            slope_max: p.slope_max,
            rock_hardness: p.rock_hardness,
            terrace_height: p.terrace_height,
            terrace_offset: p.terrace_offset,
            rotation_deg: p.rotation_deg,
            anisotropy: p.anisotropy,
            warp_strength: p.warp_strength,
            warp_frequency: p.warp_frequency,
            dx: self.metrics.dx(),
            invert: if p.invert { 1.0 } else { 0.0 },
            flow_threshold: p.flow_threshold,
            wall_steepness: p.wall_steepness,
            valley_floor: p.valley_floor,
            talus_mix: p.talus_mix,
            top_smoothness: p.top_smoothness,
            riser_sharpness: p.riser_sharpness,
            lacunarity: p.lacunarity,
            persistence: p.persistence,
            octaves: p.octaves,
            voronoi_feature: match p.voronoi_feature {
                terra_core::noise::WorleyFeature::F1 => 0,
                terra_core::noise::WorleyFeature::F2 => 1,
                terra_core::noise::WorleyFeature::F2MinusF1 => 2,
            },
            tileable: u32::from(p.tileable),
            _pad_params: 0,
            crater_radius: p.crater_radius,
            dz: self.metrics.dz(),
            _pad_metric0: 0.0,
            _pad_metric1: 0.0,
            region_x,
            region_y,
            region_w,
            region_h,
        }
    }

    fn run_effect_filter(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &EffectFilterParams,
        quality: PreviewQuality,
    ) {
        let spec = effect_filter_gpu_spec(p)
            .expect("compiled EffectFilter plan must retain an executable spec");
        let mode = spec.mode;
        let iters = Self::effect_filter_iters(quality, p);
        if spec.needs_height_range {
            self.reduce_effect_filter_range(device, queue, encoder);
        }
        let (rx, ry, rw, rh, gx, gy) = self.dirty_dispatch_extent();
        for _ in 0..iters {
            let u = self.effect_filter_uniform(p, mode, iters, (rx, ry, rw, rh));
            let u_buf = self.write_uniform(device, queue, &u);
            let src_ping = self.current == 0;
            let (src, dst) = if src_ping {
                (&self.ping.view, &self.pong.view)
            } else {
                (&self.pong.view, &self.ping.view)
            };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("effect-filter-bg"),
                layout: &self.effect_filter.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(dst),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.effect_filter_range_buffer.as_entire_binding(),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("effect-filter"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.effect_filter.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            self.swap_current();
        }
    }

    #[allow(clippy::too_many_arguments)]
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
        _bridge_prefix: Option<&Heightfield>,
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
        let result = self.evaluate_compiled_with_intent(
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

    #[allow(clippy::too_many_arguments)]
    fn record_compiled_operation(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        plan: &CompiledTerrainPlan,
        candidate: &GpuPlanResources,
        operation_id: PlanOpId,
        kernel: Option<GpuLayerPlan>,
        quality: PreviewQuality,
        scope: PropagatedDirtyScope,
        cold: bool,
        invalidation: &PlanInvalidation,
        published_output_slots: &HashMap<
            terra_core::ids::OutputId,
            terra_core::terrain_plan::FieldSlot,
        >,
    ) -> Result<(), CompiledDispatchError> {
        let operation = plan
            .operation(operation_id)
            .expect("selected operation belongs to plan");
        let region = plan_scope_region(scope, self.metrics);
        match &operation.kind {
            TerrainOpKind::Seed { source, output } => {
                self.plan_operations.seed_field_region(
                    device,
                    encoder,
                    candidate,
                    *source,
                    *output,
                    Some(region),
                )?;
            }
            TerrainOpKind::EvaluateMask {
                input_height,
                input_fields: _,
                output_mask,
            } => {
                let result = if let Some(distribution) = plan_distribution(stack, operation.origin)
                {
                    self.plan_operations.evaluate_distribution_resolved_region(
                        device,
                        encoder,
                        candidate,
                        *input_height,
                        *output_mask,
                        distribution,
                        mask_assets,
                        published_output_slots,
                        self.metrics.dx(),
                        self.metrics.dz(),
                        Some(region),
                    )
                } else {
                    Err(GpuPlanOperationError::UnsupportedMaskNodes)
                };
                result?;
            }
            TerrainOpKind::RunLayerKernel {
                layer,
                input_height,
                output_candidate,
                ..
            } => {
                let authored = stack.find(*layer).ok_or_else(|| {
                    CompiledDispatchError::Gpu(cpu_required(
                        GpuFallbackCode::UnsupportedOptions,
                        "terrain plan",
                        "compiled layer owner is missing from the authored document",
                    ))
                })?;
                let kernel = kernel.ok_or_else(|| {
                    CompiledDispatchError::Gpu(cpu_required(
                        GpuFallbackCode::UnsupportedOptions,
                        "GPU capability graph",
                        "layer has no executable GPU kernel choice",
                    ))
                })?;
                record_copy_views_region(
                    device,
                    encoder,
                    &self.copy,
                    candidate
                        .view(*input_height)
                        .map_err(GpuPlanOperationError::from)?,
                    &self.ping.view,
                    self.metrics.width,
                    self.metrics.height,
                    region,
                );
                self.current = 0;
                self.last_dirty_rect = (!scope.is_full()).then_some(region);
                let runs_in_place = kernel_runs_in_place(kernel.kernel);
                if runs_in_place {
                    record_copy_views_region(
                        device,
                        encoder,
                        &self.copy,
                        &self.ping.view,
                        &self.layer_tex.view,
                        self.metrics.width,
                        self.metrics.height,
                        region,
                    );
                }
                if let LayerKind::SculptBase(params) = &authored.kind {
                    let patch_region = (!cold
                        && invalidation.patched_operations.contains(&operation_id)
                        && !scope.is_full())
                    .then_some(region);
                    self.record_sculpt_to_layer(device, encoder, params, patch_region);
                }
                let mut candidate_layer = authored.clone();
                candidate_layer.common.opacity = 1.0;
                candidate_layer.common.blend = BlendMode::Replace;
                candidate_layer.common.masks = Distribution::default();
                self.eval_layer(
                    device,
                    queue,
                    encoder,
                    &candidate_layer,
                    kernel.kernel,
                    quality,
                )?;
                let source = if runs_in_place {
                    if self.current == 0 {
                        &self.ping.view
                    } else {
                        &self.pong.view
                    }
                } else {
                    &self.layer_tex.view
                };
                record_copy_views_region(
                    device,
                    encoder,
                    &self.copy,
                    source,
                    candidate
                        .view(*output_candidate)
                        .map_err(GpuPlanOperationError::from)?,
                    self.metrics.width,
                    self.metrics.height,
                    region,
                );
            }
            TerrainOpKind::CompositeLayer {
                layer,
                base,
                candidate: layer_candidate,
                mask,
                output,
            } => {
                let authored = stack.find(*layer).expect("compiled layer owner");
                self.plan_operations.composite_group_region(
                    device,
                    encoder,
                    candidate,
                    *base,
                    *base,
                    *layer_candidate,
                    *mask,
                    *output,
                    GpuGroupCompositeParams {
                        blend: authored.common.blend,
                        opacity: authored.common.opacity,
                        mode: GroupCompositeMode::Standard,
                    },
                    Some(region),
                )?;
            }
            TerrainOpKind::CompositeGroup {
                group,
                parent,
                private_seed,
                child_output,
                mask,
                output,
                mode,
            } => {
                let authored = stack.find_group(*group).expect("compiled group owner");
                let opacity = if authored.group_kind == terra_core::layer::GroupKind::Biome {
                    authored.opacity * authored.filter_blending
                } else {
                    authored.opacity
                };
                self.plan_operations.composite_group_region(
                    device,
                    encoder,
                    candidate,
                    *parent,
                    *private_seed,
                    *child_output,
                    *mask,
                    *output,
                    GpuGroupCompositeParams {
                        blend: authored.blend,
                        opacity,
                        mode: *mode,
                    },
                    Some(region),
                )?;
            }
            TerrainOpKind::CompositeAuxField {
                group,
                mask,
                composite,
            } => {
                let authored = stack.find_group(*group).expect("compiled group owner");
                let opacity = if authored.group_kind == terra_core::layer::GroupKind::Biome {
                    authored.opacity * authored.filter_blending
                } else {
                    authored.opacity
                };
                self.plan_operations.composite_aux_region(
                    device,
                    encoder,
                    candidate,
                    composite.parent,
                    composite.child,
                    *mask,
                    composite.output,
                    opacity,
                    Some(region),
                )?;
            }
            TerrainOpKind::PublishOutput { .. } => {}
        }
        Ok(())
    }

    /// Start an isolated, resumable Complete-quality evaluation. This API is
    /// intentionally separate from the required interactive evaluator: it
    /// always recomputes into staged resources and never exposes a partial
    /// candidate through `plan_resources.current()`.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_compiled_refinement(
        &mut self,
        device: &wgpu::Device,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        plan: &CompiledTerrainPlan,
        expected_revision: PlanStructureRevision,
        invalidation: &PlanInvalidation,
        metrics: HeightfieldMetrics,
        quality: PreviewQuality,
    ) -> Result<GpuRefinementJob, GpuError> {
        if !plan.matches_structure_revision(expected_revision) {
            return Err(GpuError::StalePlan {
                plan_revision: plan.stamp().structure_revision.get(),
                expected_revision: expected_revision.get(),
            });
        }
        if !matches!(quality, PreviewQuality::Medium | PreviewQuality::Full) {
            return Err(GpuError::Wgpu(
                "resumable refinement only accepts Medium or Full quality".into(),
            ));
        }

        self.ensure_size(device, metrics);
        self.uniform_pool.reset();
        self.plan_operations.begin_evaluation();
        self.last_plan_operation_trace.clear();
        self.last_eval_stats = GpuEvalStats {
            resolution: metrics.width.max(metrics.height),
            cold_execution: true,
            ..GpuEvalStats::default()
        };

        let selected: Vec<_> = plan
            .operations()
            .iter()
            .enumerate()
            .filter_map(|(index, _)| {
                let id = PlanOpId::from_index(index);
                plan.analysis().operation_is_live(id).then_some(id)
            })
            .collect();
        let graph = compile_gpu_graph(stack, mask_assets);
        let flat_layers = stack.flatten_layers();
        let decisions: HashMap<LayerId, (Option<GpuLayerPlan>, Option<GpuFallbackReason>)> =
            flat_layers
                .iter()
                .enumerate()
                .map(|(index, layer)| {
                    (
                        layer.id(),
                        (
                            graph.plans.get(index).copied().flatten(),
                            graph.fallback_reasons.get(index).cloned().flatten(),
                        ),
                    )
                })
                .collect();
        let mut kernels = HashMap::new();
        for operation_id in &selected {
            let operation = plan.operation(*operation_id).expect("live plan operation");
            if let TerrainOpKind::RunLayerKernel {
                layer,
                output_fields,
                ..
            } = &operation.kind
            {
                if output_fields.iter().any(|field| {
                    plan.analysis().field_is_live(*field)
                        && plan.analysis().consumers(*field).iter().any(|consumer| {
                            !plan.operation(*consumer).is_some_and(|operation| {
                                matches!(operation.kind, TerrainOpKind::PublishOutput { .. })
                            })
                        })
                }) {
                    return Err(cpu_required(
                        GpuFallbackCode::AuxiliaryDependency,
                        "auxiliary field",
                        "the GPU kernel does not yet publish a live auxiliary output",
                    ));
                }
                match decisions.get(layer) {
                    Some((Some(kernel), _)) => {
                        kernels.insert(*operation_id, *kernel);
                    }
                    decision => {
                        let reason = decision
                            .and_then(|(_, reason)| reason.clone())
                            .unwrap_or_else(|| {
                                GpuFallbackReason::new(
                                    GpuFallbackCode::UnsupportedOptions,
                                    "GPU capability graph",
                                    "layer has no executable GPU kernel choice",
                                )
                            });
                        return Err(GpuError::RequiresCpu(reason));
                    }
                }
            }
        }
        let key = GpuPlanResourceKey::new(metrics.width, metrics.height, self.device_generation);
        let builder = self
            .plan_resources
            .begin_candidate(device, plan, key)
            .map_err(|error| GpuError::Wgpu(error.to_string()))?;
        let published_output_slots = plan
            .operations()
            .iter()
            .filter_map(|operation| match operation.kind {
                TerrainOpKind::PublishOutput { output, source } => Some((output, source)),
                _ => None,
            })
            .collect();

        Ok(GpuRefinementJob {
            stack: stack.clone(),
            masks: mask_assets.to_vec(),
            graph,
            plan: plan.clone(),
            expected_revision,
            invalidation: invalidation.clone(),
            metrics,
            quality,
            builder: Some(builder),
            candidate: None,
            selected,
            kernels,
            published_output_slots,
            cursor: 0,
            last_height: None,
            completion: None,
            final_copy_submitted: false,
            final_copy_complete: false,
            submissions_issued: 0,
            submissions_completed: 0,
            resource_prepare_us: 0,
            encode_us: 0,
            range_before: self.approx_range,
        })
    }

    /// Advance by one allocation or one GPU submission. A job never has more
    /// than one refinement submission outstanding; required interactive work is
    /// free to submit behind that single unavoidable command buffer.
    pub fn advance_compiled_refinement(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        job: &mut GpuRefinementJob,
    ) -> Result<GpuRefinementStep, GpuError> {
        if job.final_copy_complete {
            return Ok(GpuRefinementStep::ReadyToPublish);
        }
        if let Some(completion) = job.completion.as_ref() {
            let _ = device.poll(wgpu::Maintain::Poll);
            match completion.try_recv() {
                Ok(()) | Err(TryRecvError::Disconnected) => {
                    job.completion = None;
                    job.submissions_completed = job.submissions_completed.saturating_add(1);
                    if job.final_copy_submitted && job.cursor == job.selected.len() {
                        job.final_copy_complete = true;
                        return Ok(GpuRefinementStep::ReadyToPublish);
                    }
                    return Ok(GpuRefinementStep::Progressed);
                }
                Err(TryRecvError::Empty) => return Ok(GpuRefinementStep::AwaitingGpu),
            }
        }

        if let Some(builder) = job.builder.as_mut() {
            let started = Instant::now();
            let complete = builder.advance(device);
            job.resource_prepare_us = job
                .resource_prepare_us
                .saturating_add(started.elapsed().as_micros() as u64);
            if complete {
                let builder = job.builder.take().expect("builder exists");
                job.candidate = Some(
                    builder
                        .finish()
                        .map_err(|error| GpuError::Wgpu(error.to_string()))?,
                );
            }
            return Ok(GpuRefinementStep::Progressed);
        }

        if !self.retired_refinement_completions.is_empty() {
            let _ = device.poll(wgpu::Maintain::Poll);
            self.retired_refinement_completions
                .retain_mut(|completion| matches!(completion.try_recv(), Err(TryRecvError::Empty)));
            if !self.retired_refinement_completions.is_empty() {
                return Ok(GpuRefinementStep::AwaitingGpu);
            }
        }

        let candidate = job
            .candidate
            .as_ref()
            .expect("completed candidate resources");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compiled-terrain-refinement-unit"),
        });
        let encode_started = Instant::now();
        if let Some(operation_id) = job.selected.get(job.cursor).copied() {
            let scope = PropagatedDirtyScope::new(PlanDirtyScope::FullField);
            if let Err(error) = self.record_compiled_operation(
                device,
                queue,
                &mut encoder,
                &job.stack,
                &job.masks,
                &job.plan,
                candidate,
                operation_id,
                job.kernels.get(&operation_id).copied(),
                job.quality,
                scope,
                true,
                &job.invalidation,
                &job.published_output_slots,
            ) {
                let diagnostic = plan_fallback_diagnostic(
                    &job.plan,
                    &job.stack,
                    operation_id,
                    plan_operation_fallback(error),
                );
                return Err(GpuError::RequiresCpu(diagnostic.reason));
            }
            let operation = job
                .plan
                .operation(operation_id)
                .expect("selected operation");
            if matches!(operation.kind, TerrainOpKind::PublishOutput { .. }) {
                self.last_eval_stats.operations_published =
                    self.last_eval_stats.operations_published.saturating_add(1);
            } else {
                self.last_eval_stats.operations_dispatched =
                    self.last_eval_stats.operations_dispatched.saturating_add(1);
                self.last_eval_stats.plan_workgroups =
                    self.last_eval_stats.plan_workgroups.saturating_add(
                        u64::from(job.metrics.width.div_ceil(8))
                            * u64::from(job.metrics.height.div_ceil(8)),
                    );
            }
            for field in job.plan.analysis().outputs(operation_id) {
                if job.plan.field(*field).is_some_and(|field| {
                    matches!(
                        field.kind,
                        terra_core::terrain_plan::LogicalFieldKind::Height
                    )
                }) {
                    job.last_height = Some(*field);
                }
            }
            job.cursor += 1;
        } else {
            let presentation = candidate
                .view(job.plan.final_height())
                .map_err(|error| GpuError::Wgpu(error.to_string()))?;
            record_copy_views_region(
                device,
                &mut encoder,
                &self.copy,
                presentation,
                &self.ping.view,
                job.metrics.width,
                job.metrics.height,
                (0, 0, job.metrics.width, job.metrics.height),
            );
            job.final_copy_submitted = true;
        }
        job.encode_us = job
            .encode_us
            .saturating_add(encode_started.elapsed().as_micros() as u64);
        let submit_started = Instant::now();
        queue.submit(Some(encoder.finish()));
        self.last_eval_stats.queue_submit_us = self
            .last_eval_stats
            .queue_submit_us
            .saturating_add(submit_started.elapsed().as_micros() as u64);
        job.submissions_issued = job.submissions_issued.saturating_add(1);
        let ordinal = job.submissions_issued;
        let (sender, receiver) = mpsc::channel();
        queue.on_submitted_work_done(move || {
            let _ = sender.send(());
        });
        job.completion = Some(receiver);
        Ok(GpuRefinementStep::Submitted { ordinal })
    }

    pub fn publish_compiled_refinement(
        &mut self,
        mut job: GpuRefinementJob,
    ) -> Result<GpuEvalResult, GpuError> {
        if !job.final_copy_complete {
            return Err(GpuError::Wgpu(
                "refinement candidate published before GPU completion".into(),
            ));
        }
        let candidate = job.candidate.take().expect("completed candidate");
        self.plan_resources.commit_candidate(candidate);
        self.last_graph = job.graph;
        self.active_plan_revision = Some(job.expected_revision);
        self.deferred_plan_resume = None;
        self.last_quality = Some(job.quality);
        self.current = 0;
        self.last_dirty_rect = None;
        self.mark_all_tiles_dirty();
        self.last_eval_stats.selected_operations = job.selected.len() as u32;
        self.last_eval_stats.resource_prepare_us = job.resource_prepare_us;
        self.last_eval_stats.command_encode_us = job.encode_us;
        self.last_eval_stats.dirty_texels =
            u64::from(job.metrics.width).saturating_mul(u64::from(job.metrics.height));
        Ok(GpuEvalResult {
            width: job.metrics.width,
            height: job.metrics.height,
            world_size: (job.metrics.world_size_x, job.metrics.world_size_z),
            height_range: self.approx_range,
            fully_gpu: true,
            freshness: GpuPreviewFreshness::Current,
            cpu: None,
            resume_cpu_from: None,
            cpu_fallback: None,
            did_eval: !job.selected.is_empty(),
        })
    }

    pub fn abandon_compiled_refinement(&mut self, job: GpuRefinementJob) {
        if let Some(completion) = job.completion {
            self.retired_refinement_completions.push(completion);
        }
        self.approx_range = job.range_before;
        self.last_dirty_rect = None;
        self.pending_evaluation_trace = None;
    }

    /// Global refinement queue depth, including a fence retained from a
    /// superseded generation. Required interactive submissions are not counted.
    pub fn refinement_submissions_in_flight(&self, job: &GpuRefinementJob) -> usize {
        self.retired_refinement_completions.len() + usize::from(job.completion.is_some())
    }

    /// Execute a validated terrain plan. Operation order, field wiring, dirty
    /// propagation, and provenance come exclusively from `plan`; `stack` is
    /// consulted only to resolve mutable authored payloads by stable id.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_compiled_with_intent(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        plan: &CompiledTerrainPlan,
        expected_revision: PlanStructureRevision,
        invalidation: &PlanInvalidation,
        metrics: HeightfieldMetrics,
        quality: PreviewQuality,
        want_cpu: bool,
        intent: GpuEvaluationIntent,
    ) -> Result<GpuEvalResult, GpuError> {
        profiling::scope!("gpu_compiled_plan_eval");
        if !plan.matches_structure_revision(expected_revision) {
            return Err(GpuError::StalePlan {
                plan_revision: plan.stamp().structure_revision.get(),
                expected_revision: expected_revision.get(),
            });
        }

        self.last_eval_stats = GpuEvalStats::default();
        self.last_eval_stats.resolution = metrics.width.max(metrics.height);
        let resource_prepare_started = std::time::Instant::now();
        self.ensure_size(device, metrics);
        self.uniform_pool.reset();
        self.plan_operations.begin_evaluation();
        self.last_plan_operation_trace.clear();
        let quality_changed = self.last_quality.replace(quality) != Some(quality);
        #[cfg(test)]
        {
            self.executed_plan_operations.clear();
            self.executed_kernels.clear();
        }

        let key = GpuPlanResourceKey::new(metrics.width, metrics.height, self.device_generation);
        let compatible_active = self.plan_resources.current().is_some_and(|resources| {
            resources.key() == key
                && resources.layout().structure_signature() == plan.structure_signature()
        });
        let cold = quality_changed
            || !compatible_active
            || self.active_plan_revision != Some(expected_revision);
        // Cold/structural/resource executions remain transactional candidates.
        // A compatible warm content edit executes against the retained realization
        // after preflight, avoiding per-dab allocation and whole-texture copies.
        let mut staged_candidate = if cold {
            Some(
                self.plan_resources
                    .stage_candidate(device, plan, key)
                    .map_err(|error| GpuError::Wgpu(error.to_string()))?,
            )
        } else {
            None
        };
        self.last_eval_stats.cold_execution = cold;
        self.last_eval_stats.resource_prepare_us =
            resource_prepare_started.elapsed().as_micros() as u64;
        let capability_preflight_started = std::time::Instant::now();

        let mut requested: Vec<PlanOpId> = if cold {
            plan.operations()
                .iter()
                .enumerate()
                .filter_map(|(index, _)| {
                    let id = PlanOpId::from_index(index);
                    plan.analysis().operation_is_live(id).then_some(id)
                })
                .collect()
        } else {
            invalidation
                .operations
                .iter()
                .map(|dirty| dirty.operation)
                .collect()
        };
        let mut reused_plan_candidates = 0u32;
        let mut reused_plan_operations = Vec::new();
        if !cold {
            requested.retain(|operation_id| {
                let Some(operation) = plan.operation(*operation_id) else {
                    return false;
                };
                let TerrainOpKind::RunLayerKernel { layer, .. } = operation.kind else {
                    return true;
                };
                let patched = invalidation.patched_operations.contains(operation_id);
                let input_independent = stack
                    .find(layer)
                    .is_some_and(|layer| layer_input_independent(&layer.kind));
                if !patched && input_independent {
                    reused_plan_candidates += 1;
                    reused_plan_operations.push(*operation_id);
                    false
                } else {
                    true
                }
            });
        }
        self.last_eval_stats.reused_contributions = reused_plan_candidates;
        self.last_eval_stats.operations_reused = reused_plan_candidates;
        let selected = staged_candidate
            .as_ref()
            .map(|candidate| candidate.layout())
            .or_else(|| self.plan_resources.current().map(|active| active.layout()))
            .expect("cold candidate or compatible active plan resources")
            .materialization_operations(plan, &requested);
        let merged_scope = if cold {
            PropagatedDirtyScope::new(PlanDirtyScope::FullField)
        } else {
            invalidation
                .operations
                .iter()
                .map(|dirty| dirty.scope)
                .reduce(PropagatedDirtyScope::merge)
                .unwrap_or_else(|| PropagatedDirtyScope::new(PlanDirtyScope::FullField))
        };
        let scope_for = |operation: PlanOpId| {
            if cold {
                return PropagatedDirtyScope::new(PlanDirtyScope::FullField);
            }
            invalidation
                .operations
                .iter()
                .find_map(|dirty| (dirty.operation == operation).then_some(dirty.scope))
                .unwrap_or(merged_scope)
        };

        let deferred_at = if !cold && intent == GpuEvaluationIntent::InteractiveLocal {
            requested
                .iter()
                .copied()
                .find(|operation| scope_for(*operation).is_full())
        } else {
            None
        };
        if !cold
            && requested.iter().any(|operation| {
                let Some(TerrainOpKind::RunLayerKernel { layer, .. }) =
                    plan.operation(*operation).map(|operation| &operation.kind)
                else {
                    return false;
                };
                stack
                    .flatten_layers()
                    .first()
                    .is_some_and(|first| first.id() == *layer)
                    && !scope_for(*operation).is_full()
            })
        {
            self.last_eval_stats.used_layer_zero_region = true;
        }
        let execution_end = deferred_at.map_or(usize::MAX, PlanOpId::index);
        self.last_eval_stats.operations_deferred = requested
            .iter()
            .filter(|operation| operation.index() >= execution_end)
            .count() as u32;
        let mut selected: Vec<PlanOpId> = selected
            .into_iter()
            .filter(|operation| operation.index() < execution_end)
            .collect();
        if intent == GpuEvaluationIntent::Complete {
            if let Some((revision, resume)) = self.deferred_plan_resume {
                if revision == expected_revision {
                    selected.retain(|operation| operation.index() >= resume.index());
                }
            }
        }

        // Consume the one flat capability compilation as the backend adapter for
        // authored payloads. The terrain plan remains the scheduling authority;
        // this graph supplies only executable kernel choices and rejection detail.
        self.last_graph = compile_gpu_graph(stack, mask_assets);
        let flat_layers = stack.flatten_layers();
        let layer_gpu_decisions: HashMap<
            LayerId,
            (Option<GpuLayerPlan>, Option<GpuFallbackReason>),
        > = flat_layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                (
                    layer.id(),
                    (
                        self.last_graph.plans.get(index).copied().flatten(),
                        self.last_graph
                            .fallback_reasons
                            .get(index)
                            .cloned()
                            .flatten(),
                    ),
                )
            })
            .collect();
        let mut kernels = HashMap::<PlanOpId, GpuLayerPlan>::new();
        let mut planned_fallback = None;
        for operation_id in &selected {
            let operation = plan
                .operation(*operation_id)
                .expect("selected operation belongs to plan");
            if let TerrainOpKind::RunLayerKernel {
                layer,
                output_fields,
                ..
            } = &operation.kind
            {
                let Some(_authored) = stack.find(*layer) else {
                    let diagnostic = plan_fallback_diagnostic(
                        plan,
                        stack,
                        *operation_id,
                        GpuFallbackReason::new(
                            GpuFallbackCode::UnsupportedOptions,
                            "terrain plan",
                            "compiled layer owner is missing from the authored document",
                        ),
                    );
                    planned_fallback = Some(diagnostic);
                    break;
                };
                if output_fields.iter().any(|field| {
                    plan.analysis().field_is_live(*field)
                        && plan.analysis().consumers(*field).iter().any(|consumer| {
                            !plan.operation(*consumer).is_some_and(|operation| {
                                matches!(operation.kind, TerrainOpKind::PublishOutput { .. })
                            })
                        })
                }) {
                    let diagnostic = plan_fallback_diagnostic(
                        plan,
                        stack,
                        *operation_id,
                        GpuFallbackReason::new(
                            GpuFallbackCode::AuxiliaryDependency,
                            "auxiliary field",
                            "the GPU kernel does not yet publish a live auxiliary output",
                        ),
                    );
                    planned_fallback = Some(diagnostic);
                    break;
                }
                match layer_gpu_decisions.get(layer) {
                    Some((Some(kernel), _)) => {
                        kernels.insert(*operation_id, *kernel);
                    }
                    decision => {
                        let reason = decision
                            .and_then(|(_, reason)| reason.clone())
                            .unwrap_or_else(|| {
                                GpuFallbackReason::new(
                                    GpuFallbackCode::UnsupportedOptions,
                                    "GPU capability graph",
                                    "layer has no executable GPU kernel choice",
                                )
                            });
                        let diagnostic =
                            plan_fallback_diagnostic(plan, stack, *operation_id, reason);
                        planned_fallback = Some(diagnostic);
                        break;
                    }
                }
            }
        }
        if let Some(diagnostic) = &planned_fallback {
            if stack.requires_tree_evaluation() {
                return Ok(plan_fallback_result(
                    metrics,
                    self.approx_range,
                    diagnostic.clone(),
                ));
            }
            let boundary = diagnostic.operation.map_or(0, PlanOpId::index);
            selected.retain(|operation| operation.index() < boundary);
        }

        let warm_execution = staged_candidate.is_none();
        let candidate = staged_candidate.take().unwrap_or_else(|| {
            self.last_eval_stats.warm_plan_resource_reuses = self
                .last_eval_stats
                .warm_plan_resource_reuses
                .saturating_add(1);
            self.plan_resources
                .take_current()
                .expect("compatible warm realization checked above")
        });
        self.last_eval_stats.selected_operations = selected.len() as u32;
        self.last_eval_stats.capability_preflight_us =
            capability_preflight_started.elapsed().as_micros() as u64;
        let command_encode_started = std::time::Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compiled-terrain-plan"),
        });
        let evaluation_timing_slot = self
            .evaluation_timer
            .as_mut()
            .and_then(|timer| timer.begin(&mut encoder));

        let mut last_height = None;
        let published_output_slots: HashMap<_, _> = plan
            .operations()
            .iter()
            .filter_map(|operation| match operation.kind {
                TerrainOpKind::PublishOutput { output, source } => Some((output, source)),
                _ => None,
            })
            .collect();
        for operation_id in &selected {
            let operation = plan
                .operation(*operation_id)
                .expect("selected operation belongs to plan");
            let region = plan_scope_region(scope_for(*operation_id), metrics);
            let execution = (|| -> Result<(), CompiledDispatchError> {
                match &operation.kind {
                    TerrainOpKind::Seed { source, output } => {
                        self.plan_operations.seed_field_region(
                            device,
                            &mut encoder,
                            &candidate,
                            *source,
                            *output,
                            Some(region),
                        )?;
                        Ok(())
                    }
                    TerrainOpKind::EvaluateMask {
                        input_height,
                        input_fields: _,
                        output_mask,
                    } => {
                        let result = if let Some(distribution) =
                            plan_distribution(stack, operation.origin)
                        {
                            self.plan_operations.evaluate_distribution_resolved_region(
                                device,
                                &mut encoder,
                                &candidate,
                                *input_height,
                                *output_mask,
                                distribution,
                                mask_assets,
                                &published_output_slots,
                                metrics.dx(),
                                metrics.dz(),
                                Some(region),
                            )
                        } else {
                            Err(GpuPlanOperationError::UnsupportedMaskNodes)
                        };
                        result?;
                        Ok(())
                    }
                    TerrainOpKind::RunLayerKernel {
                        layer,
                        input_height,
                        output_candidate,
                        ..
                    } => {
                        let authored = stack.find(*layer).expect("preflight resolved layer");
                        let kernel = kernels
                            .get(operation_id)
                            .expect("preflight compiled layer kernel");
                        record_copy_views_region(
                            device,
                            &mut encoder,
                            &self.copy,
                            candidate
                                .view(*input_height)
                                .map_err(GpuPlanOperationError::from)?,
                            &self.ping.view,
                            metrics.width,
                            metrics.height,
                            // `ping` is also the last-presented height texture. A
                            // warm local execution must preserve its pixels outside
                            // the propagated scope; the kernel only reads the
                            // region (including its planned halo) below.
                            region,
                        );
                        self.current = 0;
                        self.last_dirty_rect =
                            (!scope_for(*operation_id).is_full()).then_some(region);
                        let runs_in_place = kernel_runs_in_place(kernel.kernel);
                        if runs_in_place {
                            record_copy_views_region(
                                device,
                                &mut encoder,
                                &self.copy,
                                &self.ping.view,
                                &self.layer_tex.view,
                                metrics.width,
                                metrics.height,
                                region,
                            );
                        }
                        if let LayerKind::SculptBase(params) = &authored.kind {
                            let patch_region = (!cold
                                && invalidation.patched_operations.contains(operation_id)
                                && !scope_for(*operation_id).is_full())
                            .then_some(region);
                            self.record_sculpt_to_layer(device, &mut encoder, params, patch_region);
                        }
                        // Legacy kernels historically performed the authored outer
                        // composite themselves. A compiled plan has an explicit
                        // `CompositeLayer` operation, so run the adapter in candidate
                        // mode and leave authored opacity/blend/mask to that operation.
                        let mut candidate_layer = authored.clone();
                        candidate_layer.common.opacity = 1.0;
                        candidate_layer.common.blend = BlendMode::Replace;
                        candidate_layer.common.masks = Distribution::default();
                        self.eval_layer(
                            device,
                            queue,
                            &mut encoder,
                            &candidate_layer,
                            kernel.kernel,
                            quality,
                        )?;
                        let source = if runs_in_place {
                            if self.current == 0 {
                                &self.ping.view
                            } else {
                                &self.pong.view
                            }
                        } else {
                            &self.layer_tex.view
                        };
                        record_copy_views_region(
                            device,
                            &mut encoder,
                            &self.copy,
                            source,
                            candidate
                                .view(*output_candidate)
                                .map_err(GpuPlanOperationError::from)?,
                            metrics.width,
                            metrics.height,
                            region,
                        );
                        Ok(())
                    }
                    TerrainOpKind::CompositeLayer {
                        layer,
                        base,
                        candidate: layer_candidate,
                        mask,
                        output,
                    } => {
                        let authored = stack.find(*layer).expect("compiled layer owner");
                        self.plan_operations.composite_group_region(
                            device,
                            &mut encoder,
                            &candidate,
                            *base,
                            *base,
                            *layer_candidate,
                            *mask,
                            *output,
                            GpuGroupCompositeParams {
                                blend: authored.common.blend,
                                opacity: authored.common.opacity,
                                mode: GroupCompositeMode::Standard,
                            },
                            Some(region),
                        )?;
                        Ok(())
                    }
                    TerrainOpKind::CompositeGroup {
                        group,
                        parent,
                        private_seed,
                        child_output,
                        mask,
                        output,
                        mode,
                    } => {
                        let authored = stack.find_group(*group).expect("compiled group owner");
                        let opacity = if authored.group_kind == terra_core::layer::GroupKind::Biome
                        {
                            authored.opacity * authored.filter_blending
                        } else {
                            authored.opacity
                        };
                        self.plan_operations.composite_group_region(
                            device,
                            &mut encoder,
                            &candidate,
                            *parent,
                            *private_seed,
                            *child_output,
                            *mask,
                            *output,
                            GpuGroupCompositeParams {
                                blend: authored.blend,
                                opacity,
                                mode: *mode,
                            },
                            Some(region),
                        )?;
                        Ok(())
                    }
                    TerrainOpKind::CompositeAuxField {
                        group,
                        mask,
                        composite,
                    } => {
                        let authored = stack.find_group(*group).expect("compiled group owner");
                        let opacity = if authored.group_kind == terra_core::layer::GroupKind::Biome
                        {
                            authored.opacity * authored.filter_blending
                        } else {
                            authored.opacity
                        };
                        self.plan_operations.composite_aux_region(
                            device,
                            &mut encoder,
                            &candidate,
                            composite.parent,
                            composite.child,
                            *mask,
                            composite.output,
                            opacity,
                            Some(region),
                        )?;
                        Ok(())
                    }
                    TerrainOpKind::PublishOutput { .. } => Ok(()),
                }
            })();
            if let Err(error) = execution {
                let diagnostic = plan_fallback_diagnostic(
                    plan,
                    stack,
                    *operation_id,
                    plan_operation_fallback(error),
                );
                if warm_execution {
                    self.plan_resources.restore_current(candidate);
                }
                return Ok(plan_fallback_result(metrics, self.approx_range, diagnostic));
            }
            if matches!(operation.kind, TerrainOpKind::PublishOutput { .. }) {
                self.last_eval_stats.operations_published =
                    self.last_eval_stats.operations_published.saturating_add(1);
            } else {
                self.last_eval_stats.operations_dispatched =
                    self.last_eval_stats.operations_dispatched.saturating_add(1);
                self.last_eval_stats.plan_workgroups =
                    self.last_eval_stats.plan_workgroups.saturating_add(
                        u64::from(region.2.div_ceil(8)) * u64::from(region.3.div_ceil(8)),
                    );
            }
            for field in plan.analysis().outputs(*operation_id) {
                if plan.field(*field).is_some_and(|field| {
                    matches!(
                        field.kind,
                        terra_core::terrain_plan::LogicalFieldKind::Height
                    )
                }) {
                    last_height = Some(*field);
                }
            }
            #[cfg(test)]
            self.executed_plan_operations.push(*operation_id);
        }

        let freshness = deferred_at.map_or(GpuPreviewFreshness::Current, |operation| {
            let owner = plan.provenance().owner_of(operation);
            let from_layer = owner_layer_id(owner).unwrap_or_default();
            let from_index = flat_layers
                .iter()
                .position(|layer| layer.id() == from_layer)
                .unwrap_or(0);
            GpuPreviewFreshness::Deferred {
                from_index,
                from_layer,
                deferred_layers: flat_layers.len().saturating_sub(from_index),
            }
        });
        let presentation_field = if deferred_at.is_some() || planned_fallback.is_some() {
            last_height.unwrap_or(plan.final_height())
        } else {
            plan.final_height()
        };
        let present_scope = if cold {
            PropagatedDirtyScope::new(PlanDirtyScope::FullField)
        } else {
            merged_scope
        };
        let present_region = plan_scope_region(present_scope, metrics);
        let presentation_view = match candidate.view(presentation_field) {
            Ok(view) => view,
            Err(error) => {
                if warm_execution {
                    self.plan_resources.restore_current(candidate);
                }
                return Err(GpuError::Wgpu(error.to_string()));
            }
        };
        record_copy_views_region(
            device,
            &mut encoder,
            &self.copy,
            presentation_view,
            &self.ping.view,
            metrics.width,
            metrics.height,
            present_region,
        );
        let (mask_scratch_texture_allocations, mask_scratch_reuses) =
            self.plan_operations.mask_scratch_stats();
        self.last_eval_stats.mask_scratch_texture_allocations = mask_scratch_texture_allocations;
        self.last_eval_stats.mask_scratch_reuses = mask_scratch_reuses;
        self.last_eval_stats.dirty_texels =
            u64::from(present_region.2).saturating_mul(u64::from(present_region.3));
        if let (Some(timer), Some(slot), Some(context)) = (
            self.evaluation_timer.as_mut(),
            evaluation_timing_slot,
            self.pending_evaluation_trace.take(),
        ) {
            timer.finish(&mut encoder, slot, context);
        }
        self.last_eval_stats.command_encode_us =
            command_encode_started.elapsed().as_micros() as u64;
        let queue_submit_started = std::time::Instant::now();
        queue.submit(Some(encoder.finish()));
        self.last_eval_stats.queue_submit_us = queue_submit_started.elapsed().as_micros() as u64;
        self.current = 0;
        self.last_dirty_rect = None;
        if warm_execution {
            self.plan_resources.restore_current(candidate);
        } else {
            self.plan_resources.commit_candidate(candidate);
        }
        self.active_plan_revision = Some(expected_revision);
        self.deferred_plan_resume = deferred_at.map(|operation| (expected_revision, operation));
        if present_scope.is_full() {
            self.mark_all_tiles_dirty();
        } else {
            self.mark_tiles_overlapping_rect(present_region);
        }

        let fallback_resume = planned_fallback.as_ref().map(|diagnostic| {
            flat_layers
                .iter()
                .position(|layer| layer.id() == diagnostic.layer_id)
                .unwrap_or(0)
        });
        let fallback_resume = match fallback_resume {
            Some(index) if !cpu_resume_prefix_is_height_only(&flat_layers, index) => Some(0),
            other => other,
        };
        let cpu = if want_cpu && fallback_resume == Some(0) {
            Some(Heightfield::zeros(metrics))
        } else if want_cpu {
            self.last_eval_stats.readback_bytes = u64::from(metrics.width)
                .saturating_mul(u64::from(metrics.height))
                .saturating_mul(4);
            Some(self.readback_current(device, queue)?)
        } else {
            None
        };
        let live_operations = plan
            .operations()
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                plan.analysis()
                    .operation_is_live(PlanOpId::from_index(*index))
            })
            .count() as u32;
        self.last_eval_stats.operations_skipped = live_operations.saturating_sub(
            self.last_eval_stats
                .operations_dispatched
                .saturating_add(self.last_eval_stats.operations_published)
                .saturating_add(self.last_eval_stats.operations_deferred)
                .saturating_add(self.last_eval_stats.operations_reused),
        );
        let full_scope = PropagatedDirtyScope::new(PlanDirtyScope::FullField);
        self.last_plan_operation_trace = plan
            .operations()
            .iter()
            .enumerate()
            .filter_map(|(index, _)| {
                let operation = PlanOpId::from_index(index);
                if !plan.analysis().operation_is_live(operation) {
                    return None;
                }
                let dirty = invalidation
                    .operations
                    .iter()
                    .find(|dirty| dirty.operation == operation);
                let incoming_scope = if cold {
                    full_scope
                } else {
                    dirty.map_or(merged_scope, |dirty| dirty.incoming_scope)
                };
                let output_scope = if cold {
                    full_scope
                } else {
                    dirty.map_or(merged_scope, |dirty| dirty.scope)
                };
                let disposition = if selected.contains(&operation)
                    && matches!(
                        plan.operations()[index].kind,
                        TerrainOpKind::PublishOutput { .. }
                    ) {
                    GpuPlanOperationDisposition::Published
                } else if selected.contains(&operation) {
                    GpuPlanOperationDisposition::Dispatched
                } else if operation.index() >= execution_end && requested.contains(&operation) {
                    GpuPlanOperationDisposition::Deferred
                } else if reused_plan_operations.contains(&operation) {
                    GpuPlanOperationDisposition::Reused
                } else {
                    GpuPlanOperationDisposition::Skipped
                };
                Some(GpuPlanOperationTrace {
                    operation,
                    owner: plan.provenance().owner_of(operation),
                    incoming_scope,
                    output_scope,
                    incoming_region: plan_scope_region(incoming_scope, metrics),
                    output_region: plan_scope_region(output_scope, metrics),
                    disposition,
                })
            })
            .collect();
        Ok(GpuEvalResult {
            width: metrics.width,
            height: metrics.height,
            world_size: (metrics.world_size_x, metrics.world_size_z),
            height_range: self.approx_range,
            fully_gpu: deferred_at.is_none() && planned_fallback.is_none(),
            freshness,
            cpu,
            resume_cpu_from: fallback_resume,
            cpu_fallback: planned_fallback,
            did_eval: !selected.is_empty(),
        })
    }

    fn record_sculpt_to_layer(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &SculptParams,
        region: Option<(u32, u32, u32, u32)>,
    ) {
        let full_width = self.metrics.width;
        let full_height = self.metrics.height;
        let (origin_x, origin_y, width, height) = region.unwrap_or((0, 0, full_width, full_height));
        let row_bytes = width.saturating_mul(4);
        let padded_row_bytes = row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let mut upload = vec![0u8; padded_row_bytes as usize * height as usize];
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for local_y in 0..height {
            let row_start = local_y as usize * padded_row_bytes as usize;
            let row = &mut upload[row_start..row_start + row_bytes as usize];
            for local_x in 0..width {
                let x = origin_x + local_x;
                let y = origin_y + local_y;
                let u = (x as f32 + 0.5) / full_width.max(1) as f32;
                let v = (y as f32 + 0.5) / full_height.max(1) as f32;
                let sample = params.sample_bilinear(u, v);
                lo = lo.min(sample);
                hi = hi.max(sample);
                let offset = local_x as usize * 4;
                row[offset..offset + 4].copy_from_slice(&sample.to_ne_bytes());
            }
        }
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("compiled-plan-sculpt-upload"),
            contents: &upload,
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(height),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.layer_tex.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: origin_x,
                    y: origin_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        if lo <= hi {
            self.expand_range(lo, hi);
        }
        let texels = u64::from(width) * u64::from(height);
        self.last_eval_stats.sculpt_resampled_texels = self
            .last_eval_stats
            .sculpt_resampled_texels
            .saturating_add(texels);
        self.last_eval_stats.upload_bytes = self
            .last_eval_stats
            .upload_bytes
            .saturating_add(texels.saturating_mul(4));
    }

    /// Resample and upload only a warm edit's destination footprint. Destination
    /// coordinates remain absolute so differing authoring/preview resolutions use
    /// the same bilinear mapping as the full upload.
    fn ensure_source_raster(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &str,
    ) -> Result<PathBuf, GpuError> {
        let key = if path.is_empty() {
            PathBuf::from("<empty-heightmap>")
        } else {
            std::fs::canonicalize(path).unwrap_or_else(|_| Path::new(path).to_path_buf())
        };
        let fingerprint = if path.is_empty() {
            None
        } else {
            let metadata = std::fs::metadata(path)
                .map_err(|error| GpuError::SourceAsset(format!("{path}: {error}")))?;
            Some(SourceFingerprint {
                len: metadata.len(),
                modified: metadata.modified().ok(),
            })
        };
        if self
            .source_rasters
            .get(&key)
            .is_some_and(|entry| entry.fingerprint == fingerprint)
        {
            return Ok(key);
        }
        let decoded = if path.is_empty() {
            terra_core::generators::DecodedHeightmap {
                width: 1,
                height: 1,
                samples: vec![0.0],
            }
        } else {
            terra_core::generators::load_heightmap(path)
                .map_err(|error| GpuError::SourceAsset(error.to_string()))?
        };
        let limit = device.limits().max_texture_dimension_2d;
        if decoded.width > limit || decoded.height > limit {
            return Err(cpu_required(
                GpuFallbackCode::RuntimeResourceLimit,
                "heightmap",
                format!(
                    "source {}x{} exceeds device texture limit {}",
                    decoded.width, decoded.height, limit
                ),
            ));
        }
        let tex = HeightTex::new(
            device,
            "source-heightmap",
            decoded.width.max(1),
            decoded.height.max(1),
        );
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&decoded.samples),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(decoded.width * 4),
                rows_per_image: Some(decoded.height),
            },
            wgpu::Extent3d {
                width: decoded.width,
                height: decoded.height,
                depth_or_array_layers: 1,
            },
        );
        self.source_rasters
            .insert(key.clone(), SourceRasterTex { tex, fingerprint });
        #[cfg(test)]
        {
            self.source_upload_count += 1;
        }
        Ok(key)
    }

    fn run_heightmap_sample(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        layer: &Layer,
    ) -> Result<(), GpuError> {
        let (params, transform) = match &layer.kind {
            LayerKind::ImportHeightmap(params) => (params, None),
            LayerKind::Stamp2d(params) => {
                (&params.heightmap, layer.common.shape_transform.as_ref())
            }
            _ => {
                return Err(cpu_required(
                    GpuFallbackCode::UnsupportedLayerKind,
                    "heightmap",
                    "heightmap sampler received a non-heightmap layer",
                ));
            }
        };
        let key = self.ensure_source_raster(device, queue, &params.path)?;
        let source = self.source_rasters.get(&key).expect("source just loaded");
        let (mode, offset_x, offset_z, inv_scale, sin_t, cos_t, blend_size, roundness) =
            if let Some(transform) = transform {
                let theta = -transform.rotation_deg.to_radians();
                let (sin_t, cos_t) = theta.sin_cos();
                (
                    1,
                    transform.offset_x,
                    transform.offset_z,
                    1.0 / transform.scale.max(1e-6),
                    sin_t,
                    cos_t,
                    transform.blend_size,
                    transform.blend_roundness,
                )
            } else {
                (0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
            };
        let uniform = HeightmapSampleU {
            width: self.metrics.width,
            height: self.metrics.height,
            source_width: source.tex.width,
            source_height: source.tex.height,
            mode,
            _pad0: 0,
            height_scale: if params.path.is_empty() {
                0.0
            } else {
                params.height_scale
            },
            height_offset: if params.path.is_empty() {
                0.0
            } else {
                params.height_offset
            },
            world_x: self.metrics.world_size_x,
            world_z: self.metrics.world_size_z,
            offset_x,
            offset_z,
            inv_scale,
            sin_t,
            cos_t,
            blend_size,
            blend_roundness: roundness,
            _pad1: [0.0; 3],
        };
        let uniform_buf = self.write_uniform(device, queue, &uniform);
        let source = self.source_rasters.get(&key).expect("source retained");
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("heightmap-sample-bg"),
            layout: &self.heightmap_sample.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&source.tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.stamp_mask.view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("heightmap-sample"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.heightmap_sample.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(
            self.metrics.width.div_ceil(8),
            self.metrics.height.div_ceil(8),
            1,
        );
        Ok(())
    }

    /// Stamp the stroke set into the running height, measure each Flatten target,
    /// then relax into `layer_tex` (the layer contribution the standard blend
    /// consumes). Full-field like every other layer's contribution — `layer_tex` is
    /// shared scratch, so it must be valid everywhere the full-field blend reads it
    /// (#113).
    ///
    /// Most kinds stamp in a single pass over the whole set. Flatten (#117) splits
    /// the set: each Flatten's target is the brush-weighted mean of the *running*
    /// field over its footprint, so the stroke run is cut before every Flatten, the
    /// prior segment is stamped into a ping-pong height buffer, and a reduce/resolve
    /// pair measures that buffer into `targets[f]` before the Flatten (in the next
    /// segment) reads it. With no Flatten present this degenerates to one stamp of
    /// `[0, n)` reading the layer input — the pre-#117 path. `edited` is order- and
    /// target-independent, so a single pass computes it over the whole set.
    fn stroke_runtime_buffers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layer: LayerId,
        headers: Vec<StrokeHeaderGpu>,
        points: Vec<[f32; 4]>,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let header_size = std::mem::size_of::<StrokeHeaderGpu>();
        let point_size = std::mem::size_of::<[f32; 4]>();
        let existing = self.stroke_runtime.remove(&layer);
        let runtime = if let Some(mut runtime) = existing {
            let points_append = runtime.uploaded_points.len() <= points.len()
                && runtime.uploaded_points == points[..runtime.uploaded_points.len()];
            let header_start = if runtime.uploaded_headers.len() == headers.len()
                && !headers.is_empty()
                && runtime.uploaded_headers[..headers.len() - 1] == headers[..headers.len() - 1]
            {
                Some(headers.len() - 1)
            } else if runtime.uploaded_headers.len() < headers.len()
                && runtime.uploaded_headers == headers[..runtime.uploaded_headers.len()]
            {
                Some(runtime.uploaded_headers.len())
            } else if runtime.uploaded_headers == headers {
                Some(headers.len())
            } else {
                None
            };
            let has_capacity =
                headers.len() <= runtime.header_capacity && points.len() <= runtime.point_capacity;

            if let Some(header_start) = header_start.filter(|_| points_append && has_capacity) {
                if header_start < headers.len() {
                    let bytes = bytemuck::cast_slice(&headers[header_start..]);
                    queue.write_buffer(
                        &runtime.headers,
                        (header_start * header_size) as u64,
                        bytes,
                    );
                    self.last_eval_stats.stroke_header_upload_bytes = self
                        .last_eval_stats
                        .stroke_header_upload_bytes
                        .saturating_add(bytes.len() as u64);
                }
                let point_start = runtime.uploaded_points.len();
                if point_start < points.len() {
                    let bytes = bytemuck::cast_slice(&points[point_start..]);
                    queue.write_buffer(&runtime.points, (point_start * point_size) as u64, bytes);
                    self.last_eval_stats.stroke_point_upload_bytes = self
                        .last_eval_stats
                        .stroke_point_upload_bytes
                        .saturating_add(bytes.len() as u64);
                }
                runtime.uploaded_headers = headers;
                runtime.uploaded_points = points;
                runtime
            } else {
                let header_capacity = headers
                    .len()
                    .max(INITIAL_STROKE_HEADER_CAPACITY)
                    .next_power_of_two();
                let point_capacity = points
                    .len()
                    .max(INITIAL_STROKE_POINT_CAPACITY)
                    .next_power_of_two();
                let header_bytes = bytemuck::cast_slice(&headers);
                let point_bytes = bytemuck::cast_slice(&points);
                self.last_eval_stats.stroke_header_upload_bytes = self
                    .last_eval_stats
                    .stroke_header_upload_bytes
                    .saturating_add(header_bytes.len() as u64);
                self.last_eval_stats.stroke_point_upload_bytes = self
                    .last_eval_stats
                    .stroke_point_upload_bytes
                    .saturating_add(point_bytes.len() as u64);
                self.last_eval_stats.stroke_payload_rebuilds = self
                    .last_eval_stats
                    .stroke_payload_rebuilds
                    .saturating_add(1);
                StrokeRuntimeBuffers {
                    headers: make_runtime_storage_buffer(
                        device,
                        queue,
                        "sculpt-stroke-headers",
                        header_size,
                        header_capacity,
                        header_bytes,
                    ),
                    points: make_runtime_storage_buffer(
                        device,
                        queue,
                        "sculpt-stroke-points",
                        point_size,
                        point_capacity,
                        point_bytes,
                    ),
                    header_capacity,
                    point_capacity,
                    uploaded_headers: headers,
                    uploaded_points: points,
                }
            }
        } else {
            let header_capacity = headers
                .len()
                .max(INITIAL_STROKE_HEADER_CAPACITY)
                .next_power_of_two();
            let point_capacity = points
                .len()
                .max(INITIAL_STROKE_POINT_CAPACITY)
                .next_power_of_two();
            let header_bytes = bytemuck::cast_slice(&headers);
            let point_bytes = bytemuck::cast_slice(&points);
            self.last_eval_stats.stroke_header_upload_bytes = self
                .last_eval_stats
                .stroke_header_upload_bytes
                .saturating_add(header_bytes.len() as u64);
            self.last_eval_stats.stroke_point_upload_bytes = self
                .last_eval_stats
                .stroke_point_upload_bytes
                .saturating_add(point_bytes.len() as u64);
            self.last_eval_stats.stroke_payload_rebuilds = self
                .last_eval_stats
                .stroke_payload_rebuilds
                .saturating_add(1);
            StrokeRuntimeBuffers {
                headers: make_runtime_storage_buffer(
                    device,
                    queue,
                    "sculpt-stroke-headers",
                    header_size,
                    header_capacity,
                    header_bytes,
                ),
                points: make_runtime_storage_buffer(
                    device,
                    queue,
                    "sculpt-stroke-points",
                    point_size,
                    point_capacity,
                    point_bytes,
                ),
                header_capacity,
                point_capacity,
                uploaded_headers: headers,
                uploaded_points: points,
            }
        };
        let result = (runtime.headers.clone(), runtime.points.clone());
        self.stroke_runtime.insert(layer, runtime);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn run_sculpt_strokes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        layer: LayerId,
        p: &SculptStrokeParams,
        source: TexSlot,
        region: (u32, u32, u32, u32),
    ) {
        let strokes: Vec<&SculptStroke> = p.strokes.iter().filter(|s| s.enabled).collect();
        let (headers, points) = build_stroke_buffers(&strokes, &self.metrics);
        let (header_buf, point_buf) =
            self.stroke_runtime_buffers(device, queue, layer, headers, points);

        let width = self.metrics.width;
        let height = self.metrics.height;
        let world_x = self.metrics.world_size_x;
        let world_z = self.metrics.world_size_z;
        let n = strokes.len() as u32;
        let (region_x, region_y, region_w, region_h) = region;
        let gx = region_w.div_ceil(8).max(1);
        let gy = region_h.div_ceil(8).max(1);
        let num_partials = gx * gy;

        // Flatten footprint means, indexed by global stroke id; `partials` is the
        // reduce pass's per-workgroup scratch. Both are written by the GPU, so they
        // only need a valid (zeroed) backing until then.
        let targets_buf = make_storage_buffer(
            device,
            queue,
            "sculpt-stroke-flatten-targets",
            bytemuck::cast_slice(&vec![0f32; n.max(1) as usize]),
        );
        let partials_buf = make_storage_buffer(
            device,
            queue,
            "sculpt-stroke-flatten-partials",
            bytemuck::cast_slice(&vec![[0f32; 2]; num_partials.max(1) as usize]),
        );

        // The running field lives in one of three textures: the layer input (`Src`,
        // ping/pong) or the two ping-pong scratch buffers. `Src` is never written.
        #[derive(Clone, Copy)]
        enum RunSlot {
            Src,
            A,
            B,
        }
        fn flip(s: RunSlot) -> RunSlot {
            match s {
                RunSlot::Src | RunSlot::B => RunSlot::A,
                RunSlot::A => RunSlot::B,
            }
        }
        #[derive(Clone, Copy)]
        struct StampOp {
            lo: u32,
            hi: u32,
            in_slot: RunSlot,
            out_slot: RunSlot,
        }
        #[derive(Clone, Copy)]
        struct ReduceOp {
            stroke_index: u32,
            field: RunSlot,
            target_index: u32,
            fallback: f32,
        }
        #[derive(Clone, Copy)]
        enum Op {
            Stamp(StampOp),
            Reduce(ReduceOp),
        }

        // Cut the run before each Flatten. `cur` is the field entering the next
        // segment; a Flatten's reduce measures it, and the segment that finally
        // applies the Flatten reads `targets[f]` the resolve just wrote.
        let mut ops: Vec<Op> = Vec::new();
        let mut cur = RunSlot::Src;
        let mut prev = 0u32;
        for (idx, stroke) in strokes.iter().enumerate() {
            if !matches!(stroke.kind, SculptStrokeKind::Flatten) {
                continue;
            }
            let f = idx as u32;
            if f > prev {
                let out = flip(cur);
                ops.push(Op::Stamp(StampOp {
                    lo: prev,
                    hi: f,
                    in_slot: cur,
                    out_slot: out,
                }));
                cur = out;
            }
            ops.push(Op::Reduce(ReduceOp {
                stroke_index: f,
                field: cur,
                target_index: f,
                fallback: stroke.target_height,
            }));
            prev = f;
        }
        if n > prev {
            let out = flip(cur);
            ops.push(Op::Stamp(StampOp {
                lo: prev,
                hi: n,
                in_slot: cur,
                out_slot: out,
            }));
            cur = out;
        }
        let final_slot = cur;

        // All `&mut self` (uniform-pool) writes happen before any texture-view
        // borrow, matching `blend_into_current`; the ops carry their slots so the
        // dispatch phase needs no further planning.
        enum PassU {
            Stamp(wgpu::Buffer),
            Reduce {
                reduce: wgpu::Buffer,
                resolve: wgpu::Buffer,
            },
        }
        let mut pass_us: Vec<PassU> = Vec::with_capacity(ops.len());
        for op in &ops {
            match *op {
                Op::Stamp(s) => {
                    let u = SculptStrokesU {
                        width,
                        height,
                        world_x,
                        world_z,
                        stroke_lo: s.lo,
                        stroke_hi: s.hi,
                        region_x,
                        region_y,
                        region_w,
                        region_h,
                    };
                    pass_us.push(PassU::Stamp(self.write_uniform(device, queue, &u)));
                }
                Op::Reduce(r) => {
                    let ru = SculptReduceU {
                        width,
                        height,
                        world_x,
                        world_z,
                        stroke_index: r.stroke_index,
                        region_x,
                        region_y,
                        region_w,
                        region_h,
                    };
                    let sv = SculptResolveU {
                        num_partials,
                        target_index: r.target_index,
                        fallback: r.fallback,
                        _p0: 0.0,
                    };
                    let reduce = self.write_uniform(device, queue, &ru);
                    let resolve = self.write_uniform(device, queue, &sv);
                    pass_us.push(PassU::Reduce { reduce, resolve });
                }
            }
        }
        let edited_u = SculptStrokesU {
            width,
            height,
            world_x,
            world_z,
            stroke_lo: 0,
            stroke_hi: n,
            region_x,
            region_y,
            region_w,
            region_h,
        };
        let edited_u_buf = self.write_uniform(device, queue, &edited_u);
        let recon_u = SculptReconcileU {
            width,
            height,
            reconcile: p.reconcile,
            _p0: 0.0,
            region_x,
            region_y,
            region_w,
            region_h,
        };
        let recon_u_buf = self.write_uniform(device, queue, &recon_u);

        // Immutable view borrows only, from here down.
        let src_view = self.view_of(source);
        let stamp_a = &self.sculpt_stamp.view;
        let stamp_b = &self.sculpt_stamp_b.view;
        let slot_view = |slot: RunSlot| match slot {
            RunSlot::Src => src_view,
            RunSlot::A => stamp_a,
            RunSlot::B => stamp_b,
        };

        // Edited coverage: one order-independent pass over the whole set.
        let edited_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sculpt-strokes-edited-bg"),
            layout: &self.sculpt_strokes_edited.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: edited_u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: header_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: point_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.sculpt_edited.view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sculpt-strokes-edited"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sculpt_strokes_edited.pipeline);
            pass.set_bind_group(0, &edited_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Segmented stamp + Flatten reductions, in execution order.
        for (op, pu) in ops.iter().zip(pass_us.iter()) {
            match (op, pu) {
                (Op::Stamp(s), PassU::Stamp(u_buf)) => {
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("sculpt-strokes-bg"),
                        layout: &self.sculpt_strokes.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(src_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(slot_view(s.in_slot)),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: header_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: point_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: targets_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(slot_view(s.out_slot)),
                            },
                        ],
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("sculpt-strokes-stamp"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.sculpt_strokes.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                (Op::Reduce(r), PassU::Reduce { reduce, resolve }) => {
                    let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("sculpt-strokes-flatten-reduce-bg"),
                        layout: &self.sculpt_strokes_flatten_reduce.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: reduce.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(slot_view(r.field)),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: header_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: point_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: partials_buf.as_entire_binding(),
                            },
                        ],
                    });
                    let resolve_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("sculpt-strokes-flatten-resolve-bg"),
                        layout: &self.sculpt_strokes_flatten_resolve.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: resolve.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: partials_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: targets_buf.as_entire_binding(),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("sculpt-strokes-flatten-reduce"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.sculpt_strokes_flatten_reduce.pipeline);
                        pass.set_bind_group(0, &reduce_bg, &[]);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("sculpt-strokes-flatten-resolve"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.sculpt_strokes_flatten_resolve.pipeline);
                        pass.set_bind_group(0, &resolve_bg, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                }
                _ => unreachable!("ops and pass uniforms are built in lockstep"),
            }
        }

        // Reconcile the final running field into the layer contribution.
        let recon_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sculpt-strokes-reconcile-bg"),
            layout: &self.sculpt_strokes_reconcile.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: recon_u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(slot_view(final_slot)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.sculpt_edited.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sculpt-strokes-reconcile"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sculpt_strokes_reconcile.pipeline);
            pass.set_bind_group(0, &recon_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
    }

    fn eval_layer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        layer: &Layer,
        kernel: GpuKernel,
        quality: PreviewQuality,
    ) -> Result<(), GpuError> {
        #[cfg(test)]
        self.executed_kernels.push(kernel);
        match (kernel, &layer.kind) {
            (GpuKernel::HeightmapSample, LayerKind::ImportHeightmap(p)) => {
                self.run_heightmap_sample(device, queue, encoder, layer)?;
                let endpoint = p.height_offset + p.height_scale;
                self.expand_range(p.height_offset.min(endpoint), p.height_offset.max(endpoint));
                self.blend_into_current_with_mask(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                    [TexSlot::MaskOnes, TexSlot::StampMask],
                )?;
            }
            (GpuKernel::HeightmapSample, LayerKind::Stamp2d(p)) => {
                self.run_heightmap_sample(device, queue, encoder, layer)?;
                let endpoint = p.heightmap.height_offset + p.heightmap.height_scale;
                self.expand_range(
                    p.heightmap.height_offset.min(endpoint),
                    p.heightmap.height_offset.max(endpoint),
                );
                self.blend_into_current_with_mask(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                    [TexSlot::MaskOnes, TexSlot::StampMask],
                )?;
            }
            (GpuKernel::Sculpt, LayerKind::SculptBase(p)) => {
                // `layer_tex` was filled by `record_sculpt_to_layer` before this call.
                if self.last_dirty_rect.is_none() {
                    let (lo, hi) = p.sample_range();
                    self.expand_range(lo, hi);
                }
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::SculptStrokes, LayerKind::SculptStrokes(p)) => {
                // Stamp every stroke, then reconcile into `layer_tex`; the standard
                // blend below reproduces the CPU composite for the supported blends.
                let source = if self.current == 0 {
                    TexSlot::Ping
                } else {
                    TexSlot::Pong
                };
                self.run_sculpt_strokes(
                    device,
                    queue,
                    encoder,
                    layer.id(),
                    p,
                    source,
                    self.last_dirty_rect
                        .unwrap_or((0, 0, self.metrics.width, self.metrics.height)),
                );
                // Presentation range: fold in only the *absolute* stamp targets, like
                // every other kernel expands with stable values. The additive kinds
                // are relative to the (already-ranged) input, so widening by their
                // magnitude here would be relative to the accumulating range and
                // compound across incremental dabs — sinking the render's slab base
                // (`min_h - f(span)`) a little further on every drag step. Their exact
                // extent is left to the async CPU refine; height itself is unaffected.
                // Flatten is deliberately absent: its target is a mean of heights
                // already in range and it settles `h` toward that mean, so it cannot
                // exceed the current extent — and its `target_height` is ignored (#117).
                for stroke in &p.strokes {
                    if stroke.enabled
                        && matches!(
                            stroke.kind,
                            SculptStrokeKind::HeightStamp | SculptStrokeKind::PlateauStamp
                        )
                    {
                        self.expand_range(stroke.target_height, stroke.target_height);
                    }
                }
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Path, LayerKind::Path(p)) => {
                self.run_path_height(device, queue, encoder, p);
                let node_height = p
                    .nodes
                    .iter()
                    .map(|node| node.height.abs())
                    .fold(0.0, f32::max);
                let reach = p.height_offset.abs() + node_height + p.noise_strength.abs();
                self.expand_range(self.approx_range.0 - reach, self.approx_range.1 + reach);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::PolygonHeight, LayerKind::PolygonHeight(p)) => {
                self.run_polygon_height(device, queue, encoder, p);
                match p.mode {
                    PolygonHeightMode::RaiseBy => {
                        let reach = p.height.abs();
                        self.expand_range(self.approx_range.0 - reach, self.approx_range.1 + reach);
                    }
                    PolygonHeightMode::SetElevation if p.carve => {
                        self.expand_range(
                            self.approx_range.0 - p.height.abs(),
                            self.approx_range.1,
                        );
                    }
                    PolygonHeightMode::SetElevation => {
                        self.expand_range(p.height, p.height);
                    }
                }
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::ProceduralShape, LayerKind::ProceduralShape(p)) => {
                self.run_procedural_shape(device, queue, encoder, p)?;
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Fill, LayerKind::Flat(p)) => {
                self.fill_slot(device, queue, encoder, TexSlot::Layer, p.height);
                self.expand_range(p.height, p.height);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Ramp, LayerKind::Ramp(p)) => {
                let u = RampU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    height_min: p.height_min,
                    height_max: p.height_max,
                    direction: p.direction,
                    _pad: 0.0,
                };
                let u_buf = self.write_uniform(device, queue, &u);
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ramp-bg"),
                    layout: &self.ramp.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: u_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ramp"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.ramp.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(
                        self.metrics.width.div_ceil(8),
                        self.metrics.height.div_ceil(8),
                        1,
                    );
                }
                self.expand_range(p.height_min, p.height_max);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::NoiseValue(p)) => {
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    p,
                    NoiseDispatch::new(0, NoiseKernelMode::LegacyValue),
                );
                self.expand_range(0.0, p.amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::NoisePerlin(p)) => {
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    p,
                    NoiseDispatch::new(1, NoiseKernelMode::Perlin),
                );
                let amplitude = p.amplitude.abs();
                self.expand_range(-amplitude, amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::Fbm(p)) => {
                let nt = Self::noise_type_u(p.noise).ok_or_else(|| {
                    cpu_required(
                        GpuFallbackCode::UnsupportedOptions,
                        "noise",
                        "fBm noise type is outside the compiled plan",
                    )
                })?;
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    &p.base,
                    NoiseDispatch::new(nt, NoiseKernelMode::Fbm),
                );
                let amplitude = p.base.amplitude.abs();
                self.expand_range(-amplitude, amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::Ridged(p)) => {
                let nt = Self::noise_type_u(p.noise).ok_or_else(|| {
                    cpu_required(
                        GpuFallbackCode::UnsupportedOptions,
                        "noise",
                        "ridged noise type is outside the compiled plan",
                    )
                })?;
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    &p.base,
                    NoiseDispatch::new(nt, NoiseKernelMode::Ridged),
                );
                self.expand_range(p.base.amplitude.min(0.0), p.base.amplitude.max(0.0));
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            // Dedicated range-mask / dune asymmetry / canyon meander kernels.
            (GpuKernel::Shape, LayerKind::Mountains(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.base.seed & 0xFFFF_FFFF) as u32,
                    octaves: p.base.octaves.max(1),
                    frequency: p.base.frequency,
                    amplitude: p.base.amplitude,
                    lacunarity: p.base.lacunarity,
                    persistence: p.base.persistence,
                    offset_x: p.base.offset_x,
                    offset_z: p.base.offset_z,
                    ridge_sharpness: p.ridge_sharpness,
                    range_angle: p.range_angle,
                    range_width: p.range_width,
                    wave_frequency: 0.0,
                    asymmetry: 0.0,
                    depth: 0.0,
                    canyon_width: 0.0,
                    meander: p.crest_detail,
                    shape_mode: 0,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(0.0, p.base.amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Dunes(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.base.seed & 0xFFFF_FFFF) as u32,
                    octaves: p.base.octaves.max(1),
                    frequency: p.base.frequency,
                    amplitude: p.effective_height(),
                    lacunarity: p.base.lacunarity,
                    persistence: p.base.persistence,
                    offset_x: p.base.offset_x,
                    offset_z: p.base.offset_z,
                    ridge_sharpness: p.effective_crest_sharpness(),
                    range_angle: p.direction_deg,
                    range_width: p.linearity,
                    wave_frequency: p.effective_scale(),
                    asymmetry: p.effective_crest_sharpness(),
                    depth: p.trough_depth,
                    canyon_width: p.basin_floor,
                    meander: p.wind_strength,
                    shape_mode: 1,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(0.0, p.effective_height());
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Canyons(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.seed & 0xFFFF_FFFF) as u32,
                    octaves: 1,
                    frequency: 1.0,
                    amplitude: 1.0,
                    lacunarity: 2.0,
                    persistence: 0.5,
                    offset_x: 0.0,
                    offset_z: 0.0,
                    ridge_sharpness: 0.0,
                    range_angle: 0.0,
                    range_width: 0.0,
                    wave_frequency: 0.0,
                    asymmetry: 0.0,
                    depth: p.depth,
                    canyon_width: p.width,
                    meander: p.meander,
                    shape_mode: 2,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(-p.depth, 0.0);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::DomainWarp(p)) => {
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    &p.base,
                    NoiseDispatch::domain_warp(p.warp_strength, p.warp_frequency),
                );
                let amplitude = p.base.amplitude.abs();
                self.expand_range(-amplitude, amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::VoronoiRegions(p)) => {
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    &p.base,
                    NoiseDispatch::voronoi_regions(p.cell_jitter, p.height_per_cell),
                );
                let worley_lo = 0.25 * p.base.amplitude * p.base.remap_min;
                let worley_hi = 0.25 * p.base.amplitude * p.base.remap_max;
                let cell_span = (p.height_per_cell * p.cell_jitter).abs();
                self.expand_range(
                    worley_lo.min(worley_hi) - cell_span,
                    worley_lo.max(worley_hi) + cell_span,
                );
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Thermal, LayerKind::ThermalErosion(p)) => {
                let talus = p.talus_angle_deg.to_radians().tan() * self.metrics.dx();
                let iters = Self::scale_iters(quality, p.iterations).min(match quality {
                    PreviewQuality::Draft => self.max_sim_iters_per_tick.max(1),
                    PreviewQuality::Medium => 24,
                    PreviewQuality::Full | PreviewQuality::Export => u32::MAX,
                });
                self.fill_slot(
                    device,
                    queue,
                    encoder,
                    TexSlot::Hardness,
                    p.hardness.clamp(0.0, 1.0),
                );
                for _ in 0..iters {
                    let strength = p.strength;
                    let talus_v = talus;
                    // Inline thermal step (avoid borrowing self.thermal while mutably borrowing self)
                    let u = ThermalU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        dx: self.metrics.dx(),
                        talus: talus_v,
                        strength,
                        _p2: 0.0,
                        _p3: 0.0,
                        _pad: 0.0,
                    };
                    let u_buf = self.write_uniform(device, queue, &u);
                    let src_ping = self.current == 0;
                    let (src, dst) = if src_ping {
                        (&self.ping.view, &self.pong.view)
                    } else {
                        (&self.pong.view, &self.ping.view)
                    };
                    let delta_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("thermal-delta-bg"),
                        layout: &self.thermal.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&self.delta.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&self.hardness.view),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("thermal-delta"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.thermal.pipeline);
                        pass.set_bind_group(0, &delta_bg, &[]);
                        pass.dispatch_workgroups(
                            self.metrics.width.div_ceil(8),
                            self.metrics.height.div_ceil(8),
                            1,
                        );
                    }
                    let apply_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("thermal-apply-bg"),
                        layout: &self.thermal_apply.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&self.delta.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(dst),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("thermal-apply"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.thermal_apply.pipeline);
                        pass.set_bind_group(0, &apply_bg, &[]);
                        pass.dispatch_workgroups(
                            self.metrics.width.div_ceil(8),
                            self.metrics.height.div_ceil(8),
                            1,
                        );
                    }
                    self.swap_current();
                }
            }
            (GpuKernel::Hydraulic, LayerKind::HydraulicErosion(p)) => {
                let p = apply_transport_model(p, p.transport_model);
                self.fill_slot(device, queue, encoder, TexSlot::WaterA, 0.0);
                self.fill_slot(device, queue, encoder, TexSlot::WaterB, 0.0);
                self.fill_slot(device, queue, encoder, TexSlot::SedA, 0.0);
                self.fill_slot(device, queue, encoder, TexSlot::SedB, 0.0);
                self.fill_slot(device, queue, encoder, TexSlot::Rainfall, 1.0);
                self.fill_slot(
                    device,
                    queue,
                    encoder,
                    TexSlot::LooseSediment,
                    if p.layered_materials {
                        p.initial_sediment_thickness.max(0.0)
                    } else {
                        0.0
                    },
                );
                let eff_k = if p.layered_materials {
                    p.sediment_hardness.clamp(0.0, 1.0)
                } else {
                    p.hardness.clamp(0.0, 1.0)
                };
                self.fill_slot(device, queue, encoder, TexSlot::Hardness, eff_k);
                let iters = Self::scale_iters(quality, p.iterations).min(match quality {
                    PreviewQuality::Draft => self.max_sim_iters_per_tick.max(1),
                    PreviewQuality::Medium => 24,
                    PreviewQuality::Full | PreviewQuality::Export => u32::MAX,
                });
                let timestep = clamp_timestep_cfl(p.timestep, self.metrics.dx(), 4.0);
                let mut water_flip = false;
                for _ in 0..iters {
                    let u = HydraulicU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        timestep,
                        rainfall: p.rainfall,
                        evaporation: p.evaporation,
                        erosion: p.erosion,
                        deposition: p.deposition,
                        capacity: p.capacity,
                        fan_boost: p.fan_boost,
                        floodplain_bias: p.floodplain_bias,
                        dx: self.metrics.dx(),
                        incision_bias: p.incision_bias.max(0.05),
                        bedrock_k: p.bedrock_hardness.clamp(0.0, 1.0),
                        sediment_k: p.sediment_hardness.clamp(0.0, 1.0),
                        layered: if p.layered_materials { 1.0 } else { 0.0 },
                        _pad1: 0.0,
                    };
                    let u_buf = self.write_uniform(device, queue, &u);
                    let src_ping = self.current == 0;
                    let (h_src, h_dst) = if src_ping {
                        (&self.ping.view, &self.pong.view)
                    } else {
                        (&self.pong.view, &self.ping.view)
                    };
                    let (w_src, w_dst) = if water_flip {
                        (&self.water_b.view, &self.water_a.view)
                    } else {
                        (&self.water_a.view, &self.water_b.view)
                    };
                    let (s_src, s_dst) = if water_flip {
                        (&self.sed_b.view, &self.sed_a.view)
                    } else {
                        (&self.sed_a.view, &self.sed_b.view)
                    };
                    let outflow_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hydraulic-outflow-bg"),
                        layout: &self.hydraulic_outflow.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(h_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(w_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&self.outflow.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(&self.rainfall.view),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("hydraulic-outflow"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.hydraulic_outflow.pipeline);
                        pass.set_bind_group(0, &outflow_bg, &[]);
                        pass.dispatch_workgroups(
                            self.metrics.width.div_ceil(8),
                            self.metrics.height.div_ceil(8),
                            1,
                        );
                    }
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hydraulic-bg"),
                        layout: &self.hydraulic.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(h_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(w_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(s_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(&self.outflow.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::TextureView(h_dst),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(w_dst),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: wgpu::BindingResource::TextureView(s_dst),
                            },
                            wgpu::BindGroupEntry {
                                binding: 8,
                                resource: wgpu::BindingResource::TextureView(&self.hardness.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 9,
                                resource: wgpu::BindingResource::TextureView(&self.rainfall.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 10,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.loose_sediment.view,
                                ),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("hydraulic"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.hydraulic.pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(
                            self.metrics.width.div_ceil(8),
                            self.metrics.height.div_ceil(8),
                            1,
                        );
                    }
                    self.swap_current();
                    water_flip = !water_flip;
                }
            }
            (GpuKernel::RiverCarve, LayerKind::RiverCarve(p)) => {
                self.run_river_carve(device, queue, encoder, p, quality);
            }
            (GpuKernel::StreamPower, LayerKind::StreamPowerErosion(p)) => {
                self.run_stream_power(device, queue, encoder, p, quality);
            }
            (GpuKernel::MultiScaleAmplify, LayerKind::MultiScaleAmplify(p)) => {
                self.run_multi_scale_amplify(device, queue, encoder, p, quality);
            }
            (GpuKernel::Blur, LayerKind::Blur(p)) => {
                let iters = Self::blur_iters(p);
                for _ in 0..iters {
                    let u = BlurU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        radius: p.radius.clamp(1, BLUR_MAX_RADIUS),
                        _pad: 0,
                    };
                    let u_buf = self.write_uniform(device, queue, &u);
                    let src_ping = self.current == 0;
                    let (src, dst) = if src_ping {
                        (&self.ping.view, &self.pong.view)
                    } else {
                        (&self.pong.view, &self.ping.view)
                    };
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("blur-bg"),
                        layout: &self.blur.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(dst),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("blur"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.blur.pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(
                            self.metrics.width.div_ceil(8),
                            self.metrics.height.div_ceil(8),
                            1,
                        );
                    }
                    self.swap_current();
                }
            }
            (GpuKernel::EffectFilter, LayerKind::EffectFilter(p)) => {
                self.run_effect_filter(device, queue, encoder, p, quality);
                let amp = p.amount.abs().max(1.0);
                self.expand_range(self.approx_range.0 - amp, self.approx_range.1 + amp);
            }
            (GpuKernel::Shape, LayerKind::Mesa(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.seed & 0xFFFF_FFFF) as u32,
                    octaves: 3,
                    frequency: 0.001,
                    amplitude: p.height,
                    lacunarity: 2.0,
                    persistence: 0.5,
                    offset_x: p.center_u,
                    offset_z: p.center_v,
                    ridge_sharpness: p.edge_steepness,
                    range_angle: 0.0,
                    range_width: p.radius,
                    wave_frequency: 0.0,
                    asymmetry: 0.0,
                    depth: p.cap_noise,
                    canyon_width: 0.0,
                    meander: p.soft,
                    shape_mode: 5,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(0.0, p.height);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Volcano(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.seed & 0xFFFF_FFFF) as u32,
                    octaves: 3,
                    frequency: 0.001,
                    amplitude: p.height,
                    lacunarity: 2.0,
                    persistence: 0.5,
                    offset_x: p.center_u,
                    offset_z: p.center_v,
                    ridge_sharpness: p.flank_power,
                    range_angle: 0.0,
                    range_width: p.radius,
                    wave_frequency: 0.0,
                    asymmetry: 0.0,
                    depth: p.crater_depth,
                    canyon_width: p.crater_radius,
                    meander: p.roughness,
                    shape_mode: 4,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(0.0, p.height);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Uplift(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.seed & 0xFFFF_FFFF) as u32,
                    octaves: p.detail_octaves.max(1),
                    frequency: p.frequency,
                    amplitude: p.amplitude,
                    lacunarity: 2.0,
                    persistence: 0.5,
                    offset_x: 0.0,
                    offset_z: 0.0,
                    ridge_sharpness: p.ridge_power,
                    range_angle: p.range_angle,
                    range_width: p.corridor_width,
                    wave_frequency: p.detail_frequency,
                    asymmetry: p.altitude_fade,
                    depth: p.detail_amplitude,
                    canyon_width: 0.0,
                    meander: p.warp_strength,
                    shape_mode: 3,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(0.0, p.amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Island(p)) => {
                if p.archetype == IslandArchetype::VolcanicHighIsland {
                    // Preserve the already-admitted volcanic compatibility preview.
                    let u = ShapeU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        world_x: self.metrics.world_size_x,
                        world_z: self.metrics.world_size_z,
                        seed: p.seed as u32,
                        octaves: 4,
                        frequency: p.ridge_frequency.max(0.0001),
                        amplitude: p.mountain_height,
                        lacunarity: 2.0,
                        persistence: 0.5,
                        offset_x: p.center_u,
                        offset_z: p.center_v,
                        ridge_sharpness: p.mountain_power,
                        range_angle: p.rotation_deg,
                        range_width: p.radius,
                        wave_frequency: p.coastline_frequency,
                        asymmetry: p.aspect,
                        depth: p.beach_height,
                        canyon_width: p.lagoon_radius,
                        meander: p.coastline_warp,
                        shape_mode: 6,
                        _pad: 0,
                    };
                    self.gen_shape(device, queue, encoder, u);
                } else {
                    self.gen_island(device, queue, encoder, p);
                }
                self.expand_range(p.ocean_floor, p.mountain_height);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Plateau(p)) => {
                self.gen_plateau(device, queue, encoder, p);
                self.expand_range(p.low, p.high);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Terrace, LayerKind::Terrace(p)) => {
                let u = TerraceU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    levels: p.levels,
                    sharpness: p.sharpness,
                    min_h: self.approx_range.0 - 1e-3,
                    max_h: self.approx_range.1 + 1e-3,
                    _p0: 0.0,
                    _p1: 0.0,
                };
                let u_buf = self.write_uniform(device, queue, &u);
                let src_ping = self.current == 0;
                let (src, dst) = if src_ping {
                    (&self.ping.view, &self.pong.view)
                } else {
                    (&self.pong.view, &self.ping.view)
                };
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("terrace-bg"),
                    layout: &self.terrace.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: u_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(dst),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("terrace"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.terrace.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(
                        self.metrics.width.div_ceil(8),
                        self.metrics.height.div_ceil(8),
                        1,
                    );
                }
                self.swap_current();
            }
            (planned, actual) => {
                return Err(GpuError::Wgpu(format!(
                    "GPU support plan {planned:?} does not match layer {actual:?}"
                )));
            }
        }
        Ok(())
    }

    fn expand_range(&mut self, lo: f32, hi: f32) {
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        self.approx_range.0 = self.approx_range.0.min(lo);
        self.approx_range.1 = self.approx_range.1.max(hi);
        if self.approx_range.0 > self.approx_range.1 {
            self.approx_range.1 = self.approx_range.0 + 1e-3;
        }
    }

    pub fn readback_current(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Heightfield, GpuError> {
        let w = self.metrics.width;
        let h = self.metrics.height;
        let unpadded = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-readback-buf"),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu-readback-enc"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: self.output_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        let padded_f32 = readback_f32(device, queue, &buf, (padded * h / 4) as usize)?;
        let mut dense = Vec::with_capacity((w * h) as usize);
        let row_floats = (padded / 4) as usize;
        for y in 0..h as usize {
            let start = y * row_floats;
            dense.extend_from_slice(&padded_f32[start..start + w as usize]);
        }
        Ok(Heightfield::from_dense(self.metrics, &dense))
    }
}

/// `bridge_prefix` is only safe when it is the height *entering* `first_dirty`
/// (GPU/CPU cache of the previous layer). Full-stack last_good is never safe.
#[cfg(test)]
#[path = "tests.rs"]
mod smoke_tests;
