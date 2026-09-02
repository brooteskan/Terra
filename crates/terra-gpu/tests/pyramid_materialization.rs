use terra_core::{FieldId, PyramidConfig, TerrainPyramid, TerrainTileKey, TileId};
use terra_gpu::output_identity::GpuOutputId;
use terra_gpu::{
    GpuHeightPyramidMaterializer, GpuPyramidContentIdentity, GpuTileAtlas, GpuTileCacheError,
};

fn source_texture(
    gpu: &terra_test_gpu::TestGpu,
    width: u32,
    height: u32,
    values: &[f32],
) -> wgpu::Texture {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pyramid-test-source"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(values),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    texture
}

fn identity(output_revision: u64, output: u64) -> GpuPyramidContentIdentity {
    GpuPyramidContentIdentity {
        output_revision,
        output: GpuOutputId(output),
        generation: 3,
        plan_revision: 5,
    }
}

fn area_downsample(child: &[f32], child_size: u32, parent_size: u32) -> Vec<f32> {
    let mut result = vec![0.0; (parent_size * parent_size) as usize];
    for py in 0..parent_size {
        for px in 0..parent_size {
            let px0 = px as f64 / parent_size as f64;
            let px1 = (px + 1) as f64 / parent_size as f64;
            let py0 = py as f64 / parent_size as f64;
            let py1 = (py + 1) as f64 / parent_size as f64;
            let cx0 = (px0 * child_size as f64).floor() as u32;
            let cx1 = ((px1 * child_size as f64).ceil() as u32).min(child_size);
            let cy0 = (py0 * child_size as f64).floor() as u32;
            let cy1 = ((py1 * child_size as f64).ceil() as u32).min(child_size);
            let mut total = 0.0f64;
            let mut weights = 0.0f64;
            for cy in cy0..cy1 {
                let sy0 = cy as f64 / child_size as f64;
                let sy1 = (cy + 1) as f64 / child_size as f64;
                let wy = (py1.min(sy1) - py0.max(sy0)).max(0.0);
                for cx in cx0..cx1 {
                    let sx0 = cx as f64 / child_size as f64;
                    let sx1 = (cx + 1) as f64 / child_size as f64;
                    let weight = (px1.min(sx1) - px0.max(sx0)).max(0.0) * wy;
                    total += child[(cy * child_size + cx) as usize] as f64 * weight;
                    weights += weight;
                }
            }
            result[(py * parent_size + px) as usize] = (total / weights) as f32;
        }
    }
    result
}

#[test]
fn materializes_three_levels_without_cpu_evaluator_and_is_deterministic() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let mut config = PyramidConfig::new(9, 900.0, 450.0);
    config.tile_size = 4;
    config.halo = 1;
    let descriptor = TerrainPyramid::new(config);
    assert_eq!(
        descriptor
            .levels
            .iter()
            .map(|level| level.resolution)
            .collect::<Vec<_>>(),
        vec![2, 3, 5, 9]
    );
    let values = (0..81)
        .map(|index| ((index * 17 + 3) % 41) as f32 - 20.0)
        .collect::<Vec<_>>();
    let source = source_texture(gpu, 9, 9, &values);
    let materializer = GpuHeightPyramidMaterializer::new(&gpu.device);
    let first = materializer
        .materialize(
            &gpu.device,
            &gpu.queue,
            &descriptor,
            &source,
            (9, 9),
            identity(7, 11),
        )
        .unwrap();
    let second = materializer
        .materialize(
            &gpu.device,
            &gpu.queue,
            &descriptor,
            &source,
            (9, 9),
            identity(7, 12),
        )
        .unwrap();

    assert_eq!(first.source_level(), 3);
    let fine = first
        .read_level_blocking(&gpu.device, &gpu.queue, 3)
        .unwrap();
    assert_eq!(fine, values);

    let expected_five = area_downsample(&values, 9, 5);
    let actual_five = first
        .read_level_blocking(&gpu.device, &gpu.queue, 2)
        .unwrap();
    for (actual, expected) in actual_five.iter().zip(&expected_five) {
        assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
    }

    for level in 0..=first.source_level() {
        assert_eq!(
            first
                .read_level_blocking(&gpu.device, &gpu.queue, level)
                .unwrap(),
            second
                .read_level_blocking(&gpu.device, &gpu.queue, level)
                .unwrap(),
            "level {level} must be deterministic"
        );
    }
    assert_eq!(
        first.read_error_bits_blocking(&gpu.device, &gpu.queue),
        second.read_error_bits_blocking(&gpu.device, &gpu.queue)
    );
}

#[test]
fn measured_error_is_finite_zero_for_flat_and_increases_for_lost_feature() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let mut config = PyramidConfig::new(9, 900.0, 900.0);
    config.tile_size = 4;
    let descriptor = TerrainPyramid::new(config);
    let materializer = GpuHeightPyramidMaterializer::new(&gpu.device);

    let flat_source = source_texture(gpu, 9, 9, &vec![12.5; 81]);
    let flat = materializer
        .materialize(
            &gpu.device,
            &gpu.queue,
            &descriptor,
            &flat_source,
            (9, 9),
            identity(1, 1),
        )
        .unwrap();
    let flat_errors = flat.read_error_bits_blocking(&gpu.device, &gpu.queue);
    assert!(flat_errors.iter().all(|bits| f32::from_bits(*bits) == 0.0));

    let mut feature = vec![0.0; 81];
    feature[4 * 9 + 4] = 100.0;
    let feature_source = source_texture(gpu, 9, 9, &feature);
    let featured = materializer
        .materialize(
            &gpu.device,
            &gpu.queue,
            &descriptor,
            &feature_source,
            (9, 9),
            identity(2, 2),
        )
        .unwrap();
    let errors = featured.read_error_bits_blocking(&gpu.device, &gpu.queue);
    assert!(errors.iter().all(|bits| {
        let value = f32::from_bits(*bits);
        value.is_finite() && value >= 0.0
    }));
    let key = TerrainTileKey {
        layer: None,
        field: FieldId::Height,
        level: featured.source_level(),
        tile: TileId { tx: 1, tz: 1 },
    };
    let feature_error = f32::from_bits(errors[featured.tile_error_index(&key).unwrap() as usize]);
    assert!(feature_error > 0.0, "lost spike must produce error");
}

#[test]
fn planning_metadata_readback_is_async_compact_and_identity_stamped() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let mut config = PyramidConfig::new(9, 900.0, 900.0);
    config.tile_size = 4;
    let descriptor = TerrainPyramid::new(config);
    let mut values = vec![0.0; 81];
    values[40] = 75.0;
    let source = source_texture(gpu, 9, 9, &values);
    let pyramid = GpuHeightPyramidMaterializer::new(&gpu.device)
        .materialize(
            &gpu.device,
            &gpu.queue,
            &descriptor,
            &source,
            (9, 9),
            identity(31, 41),
        )
        .unwrap();
    let expected = pyramid
        .read_error_bits_blocking(&gpu.device, &gpu.queue)
        .into_iter()
        .map(f32::from_bits)
        .collect::<Vec<_>>();

    let mut readback = pyramid.begin_error_readback(&gpu.device, &gpu.queue);
    assert_eq!(readback.identity(), pyramid.identity());
    gpu.device.poll(wgpu::Maintain::Wait);
    let metadata = readback
        .poll(&gpu.device)
        .expect("metadata mapping")
        .expect("completed after wait");
    assert_eq!(metadata.identity, pyramid.identity());
    assert_eq!(metadata.geometric_errors, expected);
    assert_eq!(
        metadata.geometric_errors.len(),
        descriptor.metadata_len() as usize
    );
    assert!(readback.poll(&gpu.device).unwrap().is_none());
}

#[test]
fn gpu_publication_preserves_neighbor_halos_partial_edges_and_revision_authority() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let mut config = PyramidConfig::new(9, 900.0, 900.0);
    config.tile_size = 4;
    config.halo = 1;
    let descriptor = TerrainPyramid::new(config);
    let values = (0..9)
        .flat_map(|z| (0..9).map(move |x| (z * 100 + x) as f32))
        .collect::<Vec<_>>();
    let source = source_texture(gpu, 9, 9, &values);
    let materializer = GpuHeightPyramidMaterializer::new(&gpu.device);
    let pyramid = materializer
        .materialize(
            &gpu.device,
            &gpu.queue,
            &descriptor,
            &source,
            (9, 9),
            identity(17, 23),
        )
        .unwrap();
    let level = pyramid.source_level();
    let mut atlas = GpuTileAtlas::new(&gpu.device, 4, 1, 8).unwrap();
    let make_key = |tx| TerrainTileKey {
        layer: None,
        field: FieldId::Height,
        level,
        tile: TileId { tx, tz: 0 },
    };
    let live_identity = pyramid.identity();
    let left_key = make_key(0);
    let left = atlas
        .publish_pyramid_tile_current(
            &gpu.device,
            &gpu.queue,
            &pyramid,
            left_key.clone(),
            live_identity.content_stamp(),
        )
        .unwrap();
    assert!(atlas.is_current(&left_key, live_identity.content_stamp()));
    let right = atlas
        .publish_pyramid_tile(&gpu.device, &gpu.queue, &pyramid, make_key(1), 17)
        .unwrap();
    let edge = atlas
        .publish_pyramid_tile(&gpu.device, &gpu.queue, &pyramid, make_key(2), 17)
        .unwrap();

    let page_extent = atlas.page_extent() as usize;
    let left_page = atlas.read_page_blocking(&gpu.device, &gpu.queue, left.handle.slot);
    let right_page = atlas.read_page_blocking(&gpu.device, &gpu.queue, right.handle.slot);
    for row in 1..=4usize {
        assert_eq!(
            left_page[row * page_extent + 5],
            right_page[row * page_extent + 1],
            "left right-halo must equal right first interior"
        );
        assert_eq!(
            right_page[row * page_extent],
            left_page[row * page_extent + 4],
            "right left-halo must equal left final interior"
        );
    }

    let edge_page = atlas.read_page_blocking(&gpu.device, &gpu.queue, edge.handle.slot);
    for row in 1..=4usize {
        assert_eq!(
            edge_page[row * page_extent + 2],
            edge_page[row * page_extent + 1],
            "world-edge halo clamps to the final interior sample"
        );
        assert!(edge_page[row * page_extent + 3..row * page_extent + 6]
            .iter()
            .all(|value| *value == 0.0));
    }

    let rows = atlas.read_page_table_blocking(&gpu.device, &gpu.queue);
    assert_eq!(rows.iter().filter(|row| row.valid != 0).count(), 3);
    assert_eq!(atlas.residency().stats().resident_tiles, 3);
    assert!(rows.iter().filter(|row| row.valid != 0).all(|row| {
        row.level == u32::from(level) && row.output_revision_lo == 17 && row.output_revision_hi == 0
    }));

    let before = atlas.residency().stats();
    let stale = atlas.publish_pyramid_tile(
        &gpu.device,
        &gpu.queue,
        &pyramid,
        TerrainTileKey {
            layer: None,
            field: FieldId::Height,
            level,
            tile: TileId { tx: 0, tz: 1 },
        },
        18,
    );
    assert!(matches!(
        stale,
        Err(GpuTileCacheError::StalePyramid {
            content: 17,
            live: 18
        })
    ));
    assert_eq!(
        atlas.residency().stats().resident_tiles,
        before.resident_tiles
    );
    assert_eq!(
        atlas
            .read_page_table_blocking(&gpu.device, &gpu.queue)
            .iter()
            .filter(|row| row.valid != 0)
            .count(),
        3
    );

    let stale_identity = GpuPyramidContentIdentity {
        plan_revision: live_identity.plan_revision + 1,
        ..live_identity
    };
    let before = atlas.residency().stats().resident_tiles;
    assert!(matches!(
        atlas.publish_pyramid_tile_current(
            &gpu.device,
            &gpu.queue,
            &pyramid,
            make_key(0),
            stale_identity.content_stamp(),
        ),
        Err(GpuTileCacheError::StalePyramidIdentity)
    ));
    assert_eq!(atlas.residency().stats().resident_tiles, before);
}
