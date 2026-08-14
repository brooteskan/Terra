//! Frame-seam compositing regression (#16, finding A1-G3).
//!
//! The editor renders terrain into the swapchain frame, then runs the GUI pass
//! over the *same* view with `LoadOp::Load` before presenting (see
//! terra-render's crate-level "Frame seam" docs). terra-app is the only crate
//! that depends on both terra-render and terra-gui, so it is the only place the
//! two real renderers can be driven into one target and checked together —
//! terra-gui's `frame_seam.rs` can only test the GUI pass against a plain fill.
//!
//! This drives the seam offscreen and asserts both halves: the GUI lands on the
//! target, and the terrain outside the GUI is left intact. Flipping the GUI
//! pass to `LoadOp::Clear`, or dropping the terrain render, makes it fail.
//!
//! # Error-scope constraint
//!
//! `terra_test_gpu::headless()` shares one `wgpu::Device` across every test in
//! this binary, wgpu error scopes are device-global, and the harness runs tests
//! on parallel threads. All `push`/`pop_error_scope` usage must therefore stay
//! inside this single `#[test]`; extend this test rather than adding another
//! that pushes error scopes.
//!
//! Skips silently when no graphics adapter is available (the terra-test-gpu
//! convention), so the suite stays green on machines without a GPU.

use terra_gui::{Color, GuiContext, GuiInput, GuiRenderer, GuiState, Rect};
use terra_render::{GpuContext, TerrainRenderer, ViewportRendererMode};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const W: u32 = 64;
const H: u32 = 64;

/// A colour no terrain frame produces, used two ways: as a prefill a correct
/// terrain render must fully overwrite, and — because the terrain is then proven
/// magenta-free — as the GUI quad colour, so any magenta in the final frame is
/// provably the GUI's and not leftover terrain.
const MAGENTA: wgpu::Color = wgpu::Color {
    r: 1.0,
    g: 0.0,
    b: 1.0,
    a: 1.0,
};
const MAGENTA_RGBA: [u8; 4] = [255, 0, 255, 255];

/// Side of the opaque GUI quad drawn in the top-left corner (logical px; the
/// frame uses pixels-per-point 1.0, so this is also its size in texels).
const QUAD: f32 = 8.0;
/// Terrain-preservation is checked past a one-texel band around the quad, so a
/// boundary texel cannot trip a spurious failure. The quad hugs the top-left
/// corner, so the band is the top-left `GUARD`×`GUARD` block.
const GUARD: u32 = QUAD as u32 + 1;

/// Terrain renders, then the GUI composites over it without erasing it.
///
/// Revert check (the #1 closure criterion): flip the GUI pass in
/// `terra-gui/src/renderer.rs` to `LoadOp::Clear`, or delete the
/// `render_to_view` call below, and this test fails.
#[test]
fn gui_composites_over_terrain_without_erasing_it() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };

    // Same GpuContext path the app takes: hand the renderer the device/queue.
    // Cloning shares the harness's one device, so the validation error scopes
    // below still target the renderer's device.
    let ctx = GpuContext {
        device: gpu.device.clone(),
        queue: gpu.queue.clone(),
        surface_format: FORMAT,
    };
    let mut renderer = TerrainRenderer::new_headless(&ctx, W, H);
    // Only the RasterLit backend clears every pixel with the opaque atmosphere
    // colour, so it is the one whose frame can be proven to cover the target.
    // Pin it so a future default change cannot silently weaken this test.
    renderer.set_renderer_mode(ViewportRendererMode::Raster);

    let target = gpu.target(W, H, FORMAT);

    // Terrain half. Prefill magenta and require the render to overwrite all of
    // it: this is the tripwire for a dropped terrain pass — without the render
    // the prefill survives and the coverage assert fires. Snapshot the result
    // as the terrain baseline before the GUI touches the target.
    gpu.fill(&target, MAGENTA);
    gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    renderer.render_to_view(&target.view, W, H);
    let err = pollster::block_on(gpu.device.pop_error_scope());
    assert!(err.is_none(), "validation error rendering terrain: {err:?}");
    let terrain = gpu.read_rgba8(&target);
    assert!(
        !terrain.any(MAGENTA_RGBA),
        "terrain did not cover the target — magenta prefill survived, so the \
         preservation check below would be vacuous"
    );

    // GUI half. Draw one opaque magenta quad in the top-left, then run the real
    // GUI pass over the SAME target view — exactly how the app composites the
    // GUI over terrain. The quad forces the pass to run (an empty frame skips it
    // and never reaches the load op).
    let mut gui_renderer = GuiRenderer::new(&gpu.device, &gpu.queue, FORMAT);
    let mut gui_state = GuiState::default();
    let mut gui = GuiContext::begin(W as f32, H as f32, 1.0, GuiInput::default(), &mut gui_state);
    gui.draw.panel(
        Rect::from_min_max(0.0, 0.0, QUAD, QUAD),
        Color::rgb(1.0, 0.0, 1.0),
    );
    gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    gui_renderer.render(&gpu.device, &gpu.queue, &target.view, &mut gui, W, H);
    let err = pollster::block_on(gpu.device.pop_error_scope());
    assert!(err.is_none(), "validation error rendering GUI: {err:?}");
    let composited = gpu.read_rgba8(&target);

    // The GUI landed: an interior quad texel is magenta, which only the GUI can
    // have produced — the terrain frame was proven magenta-free above.
    assert_eq!(
        composited.get(2, 2),
        MAGENTA_RGBA,
        "GUI quad did not composite over the terrain target"
    );

    // The terrain survived: every texel past the quad's guard band matches the
    // terrain baseline byte-for-byte. Only the GUI pass ran between the two
    // readbacks, so a `LoadOp::Clear` in that pass would wipe these and fail.
    for y in 0..H {
        for x in 0..W {
            if x < GUARD && y < GUARD {
                continue;
            }
            assert_eq!(
                composited.get(x, y),
                terrain.get(x, y),
                "terrain texel ({x}, {y}) changed under the GUI pass — it must \
                 LoadOp::Load, not Clear"
            );
        }
    }
}
