//! Raster lit presentation — high-res full-world and crack-free clipmap.
//!
//! Pipelines live on `TerrainRenderer` until fully extracted; this module owns
//! draw planning and pass parameters.

use glam::Mat4;

use crate::clipmap::{ClipmapConfig, ClipmapPresentPlan};

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
    camera_x: f32,
    camera_z: f32,
    world_x: f32,
    world_z: f32,
    height_tex_w: u32,
    height_tex_h: u32,
) -> ClipmapPresentPlan {
    ClipmapPresentPlan::build(
        clipmap,
        camera_x,
        camera_z,
        world_x,
        world_z,
        height_tex_w,
        height_tex_h,
    )
}
