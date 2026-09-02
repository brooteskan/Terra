//! Progressive post: temporal + denoise + composite over typed HDR/GBuffer.
//!
//! No optional `scene_override` — callers always pass explicit frames.

use glam::Mat4;

/// HDR color target for temporal accumulation input.
pub struct HdrFrame<'a> {
    pub color: &'a wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

/// Geometry buffers used by temporal reject / denoise guides.
pub struct GBufferViews<'a> {
    pub depth: &'a wgpu::TextureView,
    pub normal: Option<&'a wgpu::TextureView>,
}

/// Thin façade — delegates to [`crate::progressive::ProgressiveRenderer`] until
/// the post stack is fully moved here.
pub struct ProgressivePostPipeline;

impl ProgressivePostPipeline {
    /// Resolve progressive history from an explicit HDR + depth pair.
    // wgpu resolve pass: the progressive renderer + device/queue/encoder + the
    // HDR/g-buffer/surface handles + view-projection and per-pass tuning, each
    // bound once. Kept flat.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_hdr(
        progressive: &mut crate::progressive::ProgressiveRenderer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        hdr: HdrFrame<'_>,
        gbuffer: GBufferViews<'_>,
        surface: &wgpu::TextureView,
        view_proj: Mat4,
        depth_rel_tol: f32,
        variance_scale: f32,
        filter_passes: u32,
        temporal_ts: Option<wgpu::RenderPassTimestampWrites<'_>>,
        denoise_ts: Option<wgpu::RenderPassTimestampWrites<'_>>,
        debug_viz_mode: u32,
    ) {
        let _ = (hdr.width, hdr.height, gbuffer.normal);
        progressive.resolve_hdr(
            device,
            queue,
            encoder,
            hdr.color,
            gbuffer.depth,
            surface,
            view_proj,
            depth_rel_tol,
            variance_scale,
            filter_passes,
            temporal_ts,
            denoise_ts,
            debug_viz_mode,
        );
    }
}
