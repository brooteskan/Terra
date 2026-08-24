//! GPU compute for Terra (WGSL). CPU references in `terra-core` remain the test oracle.
//!
//! Interactive hard rules: GPU-resident heightfields, no UI-thread readback, no mesh rebuild,
//! dirty-tile compute, incomplete GPU prefixes are never treated as finished Draft.

mod binding_layout;
pub mod compiled_plan;
pub mod derivatives;
pub mod effect_filter;
pub mod graph;
pub mod output_identity;
pub mod parity;
pub mod pyramid;
pub mod tile_cache;

pub use binding_layout::{
    uniform_texture_compute_layout, uniform_texture_compute_layout_entries,
    write_storage_texture_binding,
};
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

/// Pipeline reuse scoped to one owning GPU context.
///
/// The registry deliberately contains no device handle in its keys. Dropping
/// the context drops this registry and all cached pipeline handles with it.
pub struct PipelineCacheRegistry {
    driver: Option<wgpu::PipelineCache>,
    render: std::sync::Mutex<
        std::collections::HashMap<(&'static str, wgpu::TextureFormat), wgpu::RenderPipeline>,
    >,
    compute: std::sync::Mutex<std::collections::HashMap<&'static str, wgpu::ComputePipeline>>,
}

impl PipelineCacheRegistry {
    pub fn new(device: &wgpu::Device) -> Self {
        let driver = device
            .features()
            .contains(wgpu::Features::PIPELINE_CACHE)
            .then(|| {
                // SAFETY: no externally supplied cache bytes are provided, so there
                // is no data-validity invariant for the caller to uphold.
                unsafe {
                    device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                        label: Some("terra-context-pipeline-cache"),
                        data: None,
                        fallback: true,
                    })
                }
            });
        Self {
            driver,
            render: std::sync::Mutex::new(std::collections::HashMap::new()),
            compute: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    pub fn driver_cache(&self) -> Option<&wgpu::PipelineCache> {
        self.driver.as_ref()
    }

    /// Return one exact render-pipeline handle per label and target format.
    pub fn render_pipeline(
        &self,
        label: &'static str,
        format: wgpu::TextureFormat,
        create: impl FnOnce() -> wgpu::RenderPipeline,
    ) -> wgpu::RenderPipeline {
        let mut pipelines = self
            .render
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let key = (label, format);
        pipelines
            .entry(key)
            .or_insert_with(|| {
                terra_telemetry::measure(
                    terra_telemetry::CompilationKind::RenderPipeline,
                    label,
                    create,
                )
            })
            .clone()
    }

    /// Return one exact compute-pipeline handle per label.
    pub fn compute_pipeline(
        &self,
        label: &'static str,
        create: impl FnOnce() -> wgpu::ComputePipeline,
    ) -> wgpu::ComputePipeline {
        let mut pipelines = self
            .compute
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pipelines
            .entry(label)
            .or_insert_with(|| {
                terra_telemetry::measure(
                    terra_telemetry::CompilationKind::ComputePipeline,
                    label,
                    create,
                )
            })
            .clone()
    }
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
