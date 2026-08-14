//! First end-to-end offscreen render of the full `TerrainRenderer` (#6).
//!
//! Skips silently when no GPU adapter is available (the terra-test-gpu
//! convention), so the suite stays green on machines without a GPU.
//!
//! # Error-scope constraint
//!
//! `terra_test_gpu::headless()` shares one `wgpu::Device` across every test in
//! this binary, wgpu error scopes are device-global, and the test harness runs
//! tests on parallel threads. All `push`/`pop_error_scope` usage in
//! terra-render must therefore stay inside this single `#[test]`; extend this
//! test rather than adding another that pushes error scopes.

use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_render::{GpuContext, TerrainRenderer, ViewportRendererMode};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const W: u32 = 64;
const H: u32 = 64;

/// A prefill no real frame can reproduce: proves every pixel was written.
const MAGENTA: wgpu::Color = wgpu::Color {
    r: 1.0,
    g: 0.0,
    b: 1.0,
    a: 1.0,
};
const MAGENTA_RGBA: [u8; 4] = [255, 0, 255, 255];

/// The RasterLit backend clears the whole target with the opaque atmosphere
/// colour and then draws opaque terrain over it, so a correct frame leaves no
/// magenta prefill and every pixel fully opaque.
fn assert_fully_overwritten(pixels: &terra_test_gpu::Pixels, label: &str) {
    assert!(
        !pixels.any(MAGENTA_RGBA),
        "{label}: magenta prefill survived — the frame did not cover the target"
    );
    for y in 0..pixels.height() {
        for x in 0..pixels.width() {
            assert_eq!(
                pixels.get(x, y)[3],
                255,
                "{label}: non-opaque alpha at ({x}, {y})"
            );
        }
    }
}

#[test]
fn raster_frame_covers_offscreen_target() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };

    // Same GpuContext path the app takes: hand the renderer the device/queue,
    // never source them back through it. Cloning shares the harness's one device,
    // so the validation error scopes below still target the renderer's device.
    let ctx = GpuContext {
        device: gpu.device.clone(),
        queue: gpu.queue.clone(),
        surface_format: FORMAT,
    };
    let mut renderer = TerrainRenderer::new_headless(&ctx, W, H);
    // Raster is already the default; pin it so a future default change cannot
    // silently turn this into a path-tracer test — only the RasterLit backend
    // clears every pixel with the opaque atmosphere colour.
    renderer.set_renderer_mode(ViewportRendererMode::Raster);

    let target = gpu.target(W, H, FORMAT);

    // Frame 1: no heightfield uploaded — the placeholder world must still cover
    // the whole target.
    gpu.fill(&target, MAGENTA);
    gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    renderer.render_to_view(&target.view, W, H);
    let err = pollster::block_on(gpu.device.pop_error_scope());
    assert!(err.is_none(), "validation error in bare frame: {err:?}");
    assert_fully_overwritten(&gpu.read_rgba8(&target), "bare frame");

    // Frame 2: upload real height data through the staging path, then render
    // again — exercises the height upload, clipmap rebuild, camera framing, and
    // frame-index advance on top of the render itself.
    let heights = Heightfield::filled(HeightfieldMetrics::new(64, 64, 1024.0, 1024.0), 25.0);
    gpu.fill(&target, MAGENTA);
    gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    renderer.upload_heightfield(&heights);
    renderer.render_to_view(&target.view, W, H);
    let err = pollster::block_on(gpu.device.pop_error_scope());
    assert!(
        err.is_none(),
        "validation error after height upload: {err:?}"
    );
    assert_fully_overwritten(&gpu.read_rgba8(&target), "uploaded frame");

    // Frame 3: enable raster cast shadows (dormant until wired to shadow_strength)
    // so the directional depth pass actually runs — guard it against validation
    // errors and confirm the frame still fully covers the target.
    renderer.lighting.shadow_strength = 1.0;
    gpu.fill(&target, MAGENTA);
    gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    renderer.render_to_view(&target.view, W, H);
    let err = pollster::block_on(gpu.device.pop_error_scope());
    assert!(
        err.is_none(),
        "validation error with shadows enabled: {err:?}"
    );
    assert_fully_overwritten(&gpu.read_rgba8(&target), "shadowed frame");

    // Frame 4: switch to the progressive path tracer so the frame graph takes
    // the ProgressivePt branch. This exercises the schedule's pt_dispatch gating
    // and runs the end-of-frame debug_assert that the recorded passes match the
    // plan for a non-raster backend. Pixel coverage is not asserted — only the
    // RasterLit backend clears every pixel with the opaque atmosphere colour —
    // but the frame must still record without validation errors.
    renderer.set_renderer_mode(ViewportRendererMode::ProgressiveRayTraced);
    gpu.fill(&target, MAGENTA);
    gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    renderer.render_to_view(&target.view, W, H);
    let err = pollster::block_on(gpu.device.pop_error_scope());
    assert!(
        err.is_none(),
        "validation error in path-traced frame: {err:?}"
    );
}
