//! Raster lit presentation — high-res full-world and crack-free clipmap.
//!
//! Pipelines live on `TerrainRenderer` until fully extracted; this module owns
//! draw planning and pass parameters.

use glam::Mat4;

use crate::clipmap::{ClipmapConfig, ClipmapPresentInput, ClipmapPresentPlan};

/// Inputs for one RasterLit present pass.
#[derive(Clone, Copy)]
pub struct RasterLitDrawParams<'a> {
    pub view_proj: Mat4,
    pub color_view: &'a wgpu::TextureView,
    pub depth_view: &'a wgpu::TextureView,
    pub clear: [f32; 3],
    pub draw_ocean: bool,
    pub draw_wireframe: bool,
    pub draw_overlays: bool,
}

/// Plan clipmap / single-grid present for the current camera and height density.
pub fn plan_raster_present(
    clipmap: &ClipmapConfig,
    input: ClipmapPresentInput,
) -> ClipmapPresentPlan {
    ClipmapPresentPlan::build(clipmap, input)
}
