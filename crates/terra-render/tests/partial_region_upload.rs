//! Adversarial (#6 test round): a partial dirty-region height upload must not
//! disturb texels *outside* the dirty rect.
//!
//! The CPU upload path writes dirty rects into the **write** double-buffer slot
//! and then swaps it to display, but never seeds that slot from the current
//! **display** slot first. The GPU-to-GPU path does exactly that seed copy
//! (`copy_from_texture_region_and_swap`, "Seed write slot from last display"),
//! so the two paths disagree. Consequence: everything outside the dirty rect
//! reverts to whatever stale content the write slot happened to hold — zeros on
//! its first reuse.
//!
//! The check is upload-path equivalence: uploading identical data either as one
//! full upload or as a partial rect must present the same frame. A correct
//! implementation yields pixel-identical frames; the missing seed-copy makes the
//! partial frame collapse to (mostly) the stale buffer.
//!
//! Skips silently without a GPU adapter (the terra-test-gpu convention).

use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::tiling::SampleRect;
use terra_render::{GpuContext, TerrainRenderer, ViewportRendererMode};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const W: u32 = 128;
const H: u32 = 128;
/// Matches `HeightGpu`'s initial texture size so neither upload reallocates the
/// double-buffer slots — this isolates the seed-copy defect from resize churn.
const RES: u32 = 256;

/// Count differing pixels and report the first mismatch for a legible failure.
fn diff_pixels(
    a: &terra_test_gpu::Pixels,
    b: &terra_test_gpu::Pixels,
) -> (u32, Option<(u32, u32, [u8; 4], [u8; 4])>) {
    assert_eq!((a.width(), a.height()), (b.width(), b.height()));
    let mut count = 0u32;
    let mut first = None;
    for y in 0..a.height() {
        for x in 0..a.width() {
            let (pa, pb) = (a.get(x, y), b.get(x, y));
            if pa != pb {
                count += 1;
                if first.is_none() {
                    first = Some((x, y, pa, pb));
                }
            }
        }
    }
    (count, first)
}

#[test]
fn partial_upload_preserves_texels_outside_dirty_rect() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };

    // Same GpuContext path the app takes: hand the renderer the device/queue
    // rather than sourcing them back through it.
    let ctx = GpuContext {
        device: gpu.device.clone(),
        queue: gpu.queue.clone(),
        surface_format: FORMAT,
    };
    let mut renderer = TerrainRenderer::new_headless(&ctx, W, H);
    // Pin raster so a future default change can't turn this into a PT test.
    renderer.set_renderer_mode(ViewportRendererMode::Raster);

    let target = gpu.target(W, H, FORMAT);

    // A west→east height ramp: the collapse is visible in both geometry and the
    // height tint. min/max are stable across both uploads, so camera framing
    // (latched on the first present) does not move between the two renders.
    let metrics = HeightfieldMetrics::new(RES, RES, 1024.0, 1024.0);
    let mut field = Heightfield::new(metrics);
    for j in 0..RES {
        for i in 0..RES {
            field.set(i, j, (i as f32 / (RES - 1) as f32) * 300.0);
        }
    }

    // Full upload → reference frame.
    renderer.upload_heightfield(&field);
    renderer.render_to_view(&target.view, W, H);
    let full_frame = gpu.read_rgba8(&target);

    // Re-upload the *same* data, marking only a tall east-side rect dirty. A
    // correct partial path leaves the display identical to the full upload.
    let dirty = [SampleRect {
        x: RES - 32,
        y: RES / 2 - 16,
        w: 32,
        h: 32,
    }];
    renderer.upload_heightfield_regions(&field, Some(&dirty));
    renderer.render_to_view(&target.view, W, H);
    let partial_frame = gpu.read_rgba8(&target);

    let (count, first) = diff_pixels(&full_frame, &partial_frame);
    assert_eq!(
        count, 0,
        "partial upload of identical data changed {count} pixel(s) (first {first:?}); \
         texels outside the dirty rect reverted to a stale write-slot buffer"
    );
}
