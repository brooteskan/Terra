//! The UI pass composites *over* whatever is already in the colour target.
//!
//! The editor draws terrain first, then runs the GUI pass on the same surface
//! view before presenting. That works only because the GUI pass loads the
//! existing attachment instead of clearing it. Nothing in the type system
//! enforces this: flipping the pass's `LoadOp::Load` to `LoadOp::Clear` would
//! blank the viewport with no compile error.
//!
//! # Why these tests draw something
//!
//! `GuiRenderer::render` returns early when a frame produces no vertices, so
//! the render pass is never begun and `LoadOp` is never reached. A frame that
//! draws nothing therefore cannot detect a clear — it leaves the target intact
//! for the unrelated reason that no pass ran. Every test that means to exercise
//! the load behaviour must emit geometry and then sample a region that geometry
//! does not cover.
//!
//! All tests skip when no graphics adapter is available.

use terra_gui::{Color, GuiContext, GuiInput, GuiRenderer, GuiState, Rect};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const W: u32 = 64;
const H: u32 = 32;

/// Terrain stand-in: an unmistakable colour already present in the target.
const BACKDROP: wgpu::Color = wgpu::Color {
    r: 1.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};
const BACKDROP_RGBA: [u8; 4] = [255, 0, 0, 255];
const MARKER_RGBA: [u8; 4] = [0, 0, 255, 255];

/// Side of the opaque marker drawn in the top-left corner.
const MARKER: f32 = 8.0;

/// Render one GUI frame over `target`, with `build` supplying the frame's content.
fn render_gui_frame(
    gpu: &terra_test_gpu::TestGpu,
    target: &terra_test_gpu::TestTarget,
    build: impl FnOnce(&mut GuiContext<'_>),
) {
    let mut renderer = GuiRenderer::new(&gpu.device, &gpu.queue, FORMAT);
    let mut state = GuiState::default();
    let mut ctx = GuiContext::begin(W as f32, H as f32, 1.0, GuiInput::default(), &mut state);
    build(&mut ctx);
    renderer.render(&gpu.device, &gpu.queue, &target.view, &mut ctx, W, H);
}

/// Draw an opaque marker in the top-left corner, leaving the rest untouched.
fn draw_marker(ctx: &mut GuiContext<'_>) {
    ctx.draw.panel(
        Rect::from_min_max(0.0, 0.0, MARKER, MARKER),
        Color::rgb(0.0, 0.0, 1.0),
    );
}

/// A GUI frame must not erase what was already rendered underneath it.
///
/// The marker forces the pass to run; the far corner is sampled because no GUI
/// geometry reaches it, so it still holds the backdrop unless the pass cleared.
///
/// Revert check: set the UI pass in `renderer.rs` to `LoadOp::Clear` and this
/// must fail.
#[test]
fn gui_pass_preserves_backdrop() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let target = gpu.target(W, H, FORMAT);
    gpu.fill(&target, BACKDROP);

    render_gui_frame(gpu, &target, draw_marker);

    let pixels = gpu.read_rgba8(&target);
    assert_eq!(
        pixels.get(W - 1, H - 1),
        BACKDROP_RGBA,
        "the GUI pass cleared the target; it must load the existing attachment"
    );
    assert_eq!(
        pixels.get(W / 2, H / 2),
        BACKDROP_RGBA,
        "backdrop must survive everywhere the GUI did not draw"
    );
}

/// Control: the GUI pass genuinely writes to the target it is handed.
///
/// Without this, [`gui_pass_preserves_backdrop`] could pass simply because the
/// renderer drew nothing at all.
#[test]
fn gui_pass_can_modify_the_target() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let target = gpu.target(W, H, FORMAT);
    gpu.fill(&target, BACKDROP);

    render_gui_frame(gpu, &target, draw_marker);

    let covered = gpu.read_rgba8(&target).get(2, 2);
    assert_ne!(
        covered, BACKDROP_RGBA,
        "GUI geometry never reached the target — the preserve test would be vacuous"
    );
    assert_eq!(
        covered, MARKER_RGBA,
        "marker should composite as opaque blue"
    );
}

/// A frame with no geometry skips the pass entirely.
///
/// Documented so the next reader does not "simplify" the tests above into an
/// empty draw list: that version cannot observe `LoadOp` and silently stops
/// testing anything.
#[test]
fn empty_gui_frame_skips_the_pass() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let target = gpu.target(W, H, FORMAT);
    gpu.fill(&target, BACKDROP);

    render_gui_frame(gpu, &target, |_ctx| {});

    assert!(
        gpu.read_rgba8(&target).all(BACKDROP_RGBA),
        "a frame with no vertices should not touch the attachment at all"
    );
}
