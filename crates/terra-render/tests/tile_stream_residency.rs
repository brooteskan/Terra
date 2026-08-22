//! Behavioral residency test: the GPU page table decides whether a streamed
//! page is visible, and its output-revision stamp rejects stale pages (#171).

use terra_core::{FieldId, Heightfield, HeightfieldMetrics, TerrainTileKey, TileId};
use terra_gpu::GpuTileAtlas;
use terra_render::{GpuContext, TerrainRenderer, ViewportRendererMode};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const W: u32 = 96;
const H: u32 = 96;

fn renderer(ctx: &GpuContext, height: &Heightfield) -> TerrainRenderer {
    let mut renderer = TerrainRenderer::new_headless(ctx, W, H);
    renderer.set_renderer_mode(ViewportRendererMode::Raster);
    renderer.upload_heightfield(height);
    renderer
}

fn differing_pixels(left: &terra_test_gpu::Pixels, right: &terra_test_gpu::Pixels) -> usize {
    let mut count = 0;
    for y in 0..left.height() {
        for x in 0..left.width() {
            count += usize::from(left.get(x, y) != right.get(x, y));
        }
    }
    count
}

#[test]
fn stale_page_revision_falls_back_to_monolithic_height() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let ctx = GpuContext {
        device: gpu.device.clone(),
        queue: gpu.queue.clone(),
        surface_format: FORMAT,
    };
    let metrics = HeightfieldMetrics {
        width: 64,
        height: 64,
        world_size_x: 1024.0,
        world_size_z: 1024.0,
        tile_size: 64,
        halo: 2,
    };
    let fallback = Heightfield::filled(metrics, 0.0);
    let streamed = Heightfield::filled(metrics, 80.0);
    let tile = streamed.tiles().first().expect("one streamed tile");
    let key = TerrainTileKey {
        layer: None,
        field: FieldId::Height,
        level: 0,
        tile: TileId { tx: 0, tz: 0 },
    };
    let mut atlas = GpuTileAtlas::new(&gpu.device, metrics.tile_size, metrics.halo, 1)
        .expect("one-page test atlas");
    atlas
        .upload_height_tile(&gpu.queue, key, tile, 7, 7)
        .expect("streamed page upload");

    let mut control = renderer(&ctx, &fallback);
    let mut current = renderer(&ctx, &fallback);
    current.set_tile_stream_resources(
        atlas.create_texture_view(),
        atlas.page_table_buffer_cloned(),
        metrics.tile_size,
        metrics.halo,
        atlas.max_pages(),
        0,
        (metrics.width, metrics.height),
        7,
        true,
    );
    let mut stale = renderer(&ctx, &fallback);
    stale.set_tile_stream_resources(
        atlas.create_texture_view(),
        atlas.page_table_buffer_cloned(),
        metrics.tile_size,
        metrics.halo,
        atlas.max_pages(),
        0,
        (metrics.width, metrics.height),
        8,
        true,
    );

    let control_target = gpu.target(W, H, FORMAT);
    let current_target = gpu.target(W, H, FORMAT);
    let stale_target = gpu.target(W, H, FORMAT);
    control.render_to_view(&control_target.view, W, H);
    current.render_to_view(&current_target.view, W, H);
    stale.render_to_view(&stale_target.view, W, H);
    let control_pixels = gpu.read_rgba8(&control_target);
    let current_pixels = gpu.read_rgba8(&current_target);
    let stale_pixels = gpu.read_rgba8(&stale_target);

    assert!(
        differing_pixels(&control_pixels, &current_pixels) > 0,
        "matching revision must consume the visibly different atlas page"
    );
    assert_eq!(
        differing_pixels(&control_pixels, &stale_pixels),
        0,
        "a mismatched revision must reject the page-table row and render the monolithic fallback"
    );
}
