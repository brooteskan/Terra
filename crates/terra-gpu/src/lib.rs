//! GPU compute for Terra (WGSL). CPU references in `terra-core` remain the test oracle.
//!
//! Interactive hard rules: GPU-resident heightfields, no UI-thread readback, no mesh rebuild,
//! dirty-tile compute, incomplete GPU prefixes are never treated as finished Draft.

pub mod compiled_plan;
pub mod derivatives;
pub mod effect_filter;
pub mod graph;
pub mod output_identity;
pub mod parity;
pub mod pyramid;
pub mod tile_cache;

pub use derivatives::{cpu_slope_oracle, run_derivative_gpu, GpuDerivativeMode};
pub use graph::{
    compile_gpu_graph, expand_dirty_rect, layer_gpu_supported, GpuComputeGraph, GpuDirtyPolicy,
    GpuFallbackCode, GpuFallbackDiagnostic, GpuFallbackReason, GpuKernel, GpuLayerPlan,
    BLUR_MAX_RADIUS, EFFECT_FILTER_MAX_RADIUS, RIVER_CARVE_MAX_RADIUS,
};
pub use pyramid::{
    GpuHeightPyramid, GpuHeightPyramidMaterializer, GpuPyramidContentIdentity, GpuPyramidError,
    GpuPyramidErrorReadback, GpuPyramidPlanningMetadata,
};
pub use tile_cache::{
    GpuPageTableEntry, GpuTerrainLevelEntry, GpuTileAtlas, GpuTileCacheError, GpuTileUpload,
    GpuVirtualPageEntry,
};

use thiserror::Error;

type PipelineCaches = std::collections::HashMap<wgpu::Device, wgpu::PipelineCache>;
type RenderPipelineKey = (wgpu::Device, &'static str, wgpu::TextureFormat);
type ComputePipelineKey = (wgpu::Device, &'static str);

static PIPELINE_CACHES: std::sync::OnceLock<std::sync::Mutex<PipelineCaches>> =
    std::sync::OnceLock::new();
static RENDER_PIPELINES: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<RenderPipelineKey, wgpu::RenderPipeline>>,
> = std::sync::OnceLock::new();
static COMPUTE_PIPELINES: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<ComputePipelineKey, wgpu::ComputePipeline>>,
> = std::sync::OnceLock::new();

/// Process-local driver pipeline cache shared by all Terra GPU consumers using
/// the same device.
///
/// Renderer instances retain independent mutable resources while repeated
/// pipeline creation can reuse backend compilation work.
pub fn shared_pipeline_cache(device: &wgpu::Device) -> Option<wgpu::PipelineCache> {
    if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
        return None;
    }
    let caches = PIPELINE_CACHES.get_or_init(|| std::sync::Mutex::new(PipelineCaches::new()));
    let mut caches = caches
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Some(
        caches
            .entry(device.clone())
            .or_insert_with(|| {
                // SAFETY: no externally supplied cache bytes are provided, so there
                // is no data-validity invariant for the caller to uphold.
                unsafe {
                    device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                        label: Some("terra-shared-pipeline-cache"),
                        data: None,
                        fallback: true,
                    })
                }
            })
            .clone(),
    )
}

/// Return one exact render-pipeline handle per device, label, and target format.
/// The creation closure runs at most once for a key in the current process.
pub fn cached_render_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    format: wgpu::TextureFormat,
    create: impl FnOnce() -> wgpu::RenderPipeline,
) -> wgpu::RenderPipeline {
    let pipelines =
        RENDER_PIPELINES.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut pipelines = pipelines
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let key = (device.clone(), label, format);
    pipelines.entry(key).or_insert_with(create).clone()
}

/// Return one exact compute-pipeline handle per device and label.
/// The creation closure runs at most once for a key in the current process.
pub fn cached_compute_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    create: impl FnOnce() -> wgpu::ComputePipeline,
) -> wgpu::ComputePipeline {
    let pipelines =
        COMPUTE_PIPELINES.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut pipelines = pipelines
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let key = (device.clone(), label);
    pipelines.entry(key).or_insert_with(create).clone()
}

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("wgpu: {0}")]
    Wgpu(String),
    /// Stack needs the CPU evaluator for a structured, user-visible reason.
    #[error("cpu evaluation required: {0:?}")]
    RequiresCpu(GpuFallbackReason),
    #[error(
        "compiled terrain plan revision {plan_revision} is stale; expected revision {expected_revision}"
    )]
    StalePlan {
        plan_revision: u64,
        expected_revision: u64,
    },
    #[error("failed to load source asset: {0}")]
    SourceAsset(String),
}

pub fn readback_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    n: usize,
) -> Result<Vec<f32>, GpuError> {
    let size = (n * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback-enc"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| GpuError::Wgpu(e.to_string()))?
        .map_err(|e| GpuError::Wgpu(e.to_string()))?;
    let data = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(out)
}
