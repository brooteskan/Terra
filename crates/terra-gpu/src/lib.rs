//! GPU compute for Terra (WGSL). CPU references in `terra-core` remain the test oracle.
//!
//! Interactive hard rules: GPU-resident heightfields, no UI-thread readback, no mesh rebuild,
//! dirty-tile compute, incomplete GPU prefixes are never treated as finished Draft.

pub mod derivatives;
pub mod effect_filter;
pub mod engine;
pub mod graph;
pub mod memory;
pub mod parity;
pub mod tile_cache;

pub use derivatives::{cpu_slope_oracle, run_derivative_gpu, GpuDerivativeMode};
pub use engine::{GpuEvalResult, GpuTerrainEngine};
pub use graph::{
    compile_gpu_graph, expand_dirty_rect, layer_gpu_supported, GpuComputeGraph, GpuKernel, GpuPass,
    GpuPassKind,
};
pub use tile_cache::{GpuPageTableEntry, GpuTileAtlas, GpuTileCacheError, GpuTileUpload};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("wgpu: {0}")]
    Wgpu(String),
    /// Stack needs the CPU tree evaluator (scoped groups / unsupported layers).
    #[error("cpu evaluation required")]
    RequiresCpu,
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
