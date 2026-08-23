//! Behavioral residency test: the GPU page table decides whether a streamed
//! page is visible, and its output-revision stamp rejects stale pages (#171).

use terra_core::{
    Heightfield, HeightfieldMetrics, PyramidConfig, TerrainContentStamp, TerrainPyramid,
    TerrainTileKey, TileId,
};
use terra_gpu::GpuTileAtlas;
use terra_render::{
    GpuContext, TerrainRenderer, TerrainTerminalFallback, TerrainTileStreamResources,
    ViewportRendererMode,
};

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

fn stream_resources(
    atlas: &GpuTileAtlas,
    pyramid: &TerrainPyramid,
    content: TerrainContentStamp,
    terminal_fallback: TerrainTerminalFallback,
) -> TerrainTileStreamResources {
    let target = pyramid.level(pyramid.max_level()).unwrap();
    TerrainTileStreamResources {
        atlas_view: atlas.create_texture_view(),
        physical_page_table: atlas.page_table_buffer_cloned(),
        virtual_page_table: atlas.virtual_page_table_buffer_cloned(),
        level_table: atlas.level_table_buffer_cloned(),
        tile_size: pyramid.config.tile_size,
        halo: pyramid.config.halo,
        max_pages: atlas.max_pages(),
        level_count: atlas.level_count(),
        target_level: target.index,
        target_resolution: target.resolution,
        content,
        transition_frames: 0,
        terminal_fallback,
        enable: true,
    }
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
    let pyramid = TerrainPyramid::new(PyramidConfig {
        target_resolution: metrics.width,
        world_size_x: metrics.world_size_x,
        world_size_z: metrics.world_size_z,
        tile_size: metrics.tile_size,
        halo: metrics.halo,
    });
    let level = pyramid.max_level();
    let key = TerrainTileKey::height(pyramid.address(level, TileId { tx: 0, tz: 0 }).unwrap());
    let mut atlas = GpuTileAtlas::new(&gpu.device, metrics.tile_size, metrics.halo, 1)
        .expect("one-page test atlas");
    atlas.configure_hierarchy(&gpu.device, &gpu.queue, &pyramid);
    let content = TerrainContentStamp {
        document_revision: 3,
        plan_revision: 5,
        output_revision: 7,
        content_revision: 11,
    };
    atlas
        .upload_height_tile_current(&gpu.queue, key, tile, content)
        .expect("streamed page upload");

    let mut control = renderer(&ctx, &fallback);
    let mut current = renderer(&ctx, &fallback);
    let resources = |content| TerrainTileStreamResources {
        atlas_view: atlas.create_texture_view(),
        physical_page_table: atlas.page_table_buffer_cloned(),
        virtual_page_table: atlas.virtual_page_table_buffer_cloned(),
        level_table: atlas.level_table_buffer_cloned(),
        tile_size: metrics.tile_size,
        halo: metrics.halo,
        max_pages: atlas.max_pages(),
        level_count: atlas.level_count(),
        target_level: level,
        target_resolution: metrics.width,
        content,
        transition_frames: 0,
        terminal_fallback: TerrainTerminalFallback::MonolithicMigration,
        enable: true,
    };
    current.set_tile_stream_resources(resources(content));
    let mut stale = renderer(&ctx, &fallback);
    stale.set_tile_stream_resources(resources(TerrainContentStamp {
        output_revision: 8,
        ..content
    }));

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

#[test]
fn resident_child_refines_and_unpublish_returns_to_current_root() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let ctx = GpuContext {
        device: gpu.device.clone(),
        queue: gpu.queue.clone(),
        surface_format: FORMAT,
    };
    let pyramid = TerrainPyramid::new(PyramidConfig {
        target_resolution: 64,
        world_size_x: 1024.0,
        world_size_z: 1024.0,
        tile_size: 32,
        halo: 2,
    });
    let content = TerrainContentStamp {
        document_revision: 13,
        plan_revision: 17,
        output_revision: 19,
        content_revision: 23,
    };
    let tile_id = TileId { tx: 0, tz: 0 };
    let root_key = TerrainTileKey::height(pyramid.address(0, tile_id).unwrap());
    let child_key = TerrainTileKey::height(pyramid.address(pyramid.max_level(), tile_id).unwrap());
    let root = Heightfield::filled(pyramid.level_metrics(0).unwrap(), 20.0);
    let fine = Heightfield::filled(pyramid.level_metrics(pyramid.max_level()).unwrap(), 100.0);
    let monolithic = Heightfield::filled(HeightfieldMetrics::new(64, 64, 1024.0, 1024.0), 0.0);
    let mut atlas = GpuTileAtlas::new(&gpu.device, 32, 2, 2).unwrap();
    atlas.configure_hierarchy(&gpu.device, &gpu.queue, &pyramid);
    atlas
        .upload_height_tile_current(
            &gpu.queue,
            root_key.clone(),
            root.tile(tile_id).unwrap(),
            content,
        )
        .unwrap();
    assert!(atlas.pin(&root_key));

    let mut renderer = renderer(&ctx, &monolithic);
    renderer.set_tile_stream_resources(stream_resources(
        &atlas,
        &pyramid,
        content,
        TerrainTerminalFallback::RootRequired,
    ));
    let coarse_target = gpu.target(W, H, FORMAT);
    renderer.render_to_view(&coarse_target.view, W, H);
    let coarse = gpu.read_rgba8(&coarse_target);

    atlas
        .upload_height_tile_current(
            &gpu.queue,
            child_key.clone(),
            fine.tile(tile_id).unwrap(),
            content,
        )
        .unwrap();
    let child_target = gpu.target(W, H, FORMAT);
    renderer.render_to_view(&child_target.view, W, H);
    let child = gpu.read_rgba8(&child_target);
    assert!(
        differing_pixels(&coarse, &child) > 0,
        "a current fine child must refine its covered region"
    );

    assert!(atlas.unpublish(&gpu.queue, &child_key));
    let restored_target = gpu.target(W, H, FORMAT);
    renderer.render_to_view(&restored_target.view, W, H);
    let restored = gpu.read_rgba8(&restored_target);
    assert_eq!(
        differing_pixels(&coarse, &restored),
        0,
        "removing the child must safely restore resident-root sampling"
    );

    atlas
        .upload_height_tile_current(
            &gpu.queue,
            child_key,
            fine.tile(TileId { tx: 0, tz: 0 }).unwrap(),
            TerrainContentStamp {
                document_revision: content.document_revision + 1,
                ..content
            },
        )
        .unwrap();
    let stale_target = gpu.target(W, H, FORMAT);
    renderer.render_to_view(&stale_target.view, W, H);
    let stale = gpu.read_rgba8(&stale_target);
    assert_eq!(
        differing_pixels(&coarse, &stale),
        0,
        "a child from another document revision must never replace the current root"
    );
}
