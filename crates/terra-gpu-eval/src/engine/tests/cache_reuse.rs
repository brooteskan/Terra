use super::*;

#[test]
fn sculpt_strokes_kernel_is_executed_from_the_plan() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
    let mut stack = LayerStack::new();
    let base = Layer::new("base", LayerKind::Flat(FlatParams { height: 5.0 }));
    let base_id = base.id();
    stack.push(base);
    stack.push(Layer::new(
        "strokes",
        LayerKind::SculptStrokes(raise_strokes(0.5, 0.5, 10.0)),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_dirty(base_id);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("fully-GPU stroke evaluation");

    let planned: Vec<GpuKernel> = engine
        .last_graph
        .plans
        .iter()
        .flatten()
        .map(|plan| plan.kernel)
        .collect();
    assert_eq!(planned, vec![GpuKernel::Fill, GpuKernel::SculptStrokes]);
    assert_eq!(engine.executed_kernels, planned);
}

/// #126 regression: procedural shapes publish a reusable contribution. An
/// upstream sculpt edit must recompute the input-dependent stroke layer but
/// blend the cached Volcano contribution without dispatching Shape again.
#[test]
fn warm_cache_base_edit_reuses_input_independent_volcano_contribution() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
    let mut stack = LayerStack::new();
    let base = Layer::new("base", LayerKind::SculptBase(SculptParams::filled(32, 5.0)));
    let base_id = base.id();
    stack.push(base);
    stack.push(Layer::new(
        "strokes",
        LayerKind::SculptStrokes(raise_strokes(0.5, 0.5, 10.0)),
    ));
    stack.push(Layer::new(
        "volcano",
        LayerKind::Volcano(terra_core::layer::VolcanoParams::default()),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm shape contribution cache");
    assert!(engine.plan_resources.current().is_some());

    let Some(layer) = stack.find_mut(base_id) else {
        panic!("base layer disappeared");
    };
    let LayerKind::SculptBase(params) = &mut layer.kind else {
        panic!("base changed kind");
    };
    params.samples[(16 * 32 + 16) as usize] += 3.0;
    engine.set_dirty_rect(Some((16, 16, 1, 1)));
    engine.mark_dirty(base_id);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("incremental base edit");

    assert_eq!(
        engine.executed_kernels,
        vec![GpuKernel::Sculpt, GpuKernel::SculptStrokes]
    );
    assert!(
        !engine.executed_kernels.contains(&GpuKernel::Shape),
        "cached Volcano contribution must avoid Shape dispatch"
    );
    let stats = engine.last_eval_stats();
    assert!(stats.used_layer_zero_region);
    assert_eq!(stats.reused_contributions, 1);
    assert!(stats.sculpt_resampled_texels < u64::from(metrics.width * metrics.height));
}

/// #107: VoronoiRegions is input-independent, so an upstream bounded edit
/// re-blends its warm contribution without re-running the 3x3 Worley search.
#[test]
fn warm_cache_base_edit_reuses_voronoi_contribution() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 48u32;
    let metrics = HeightfieldMetrics::new(res, res, 240.0, 240.0);
    let rect = (20u32, 20u32, 8u32, 8u32);
    let mut stack = LayerStack::new();
    let base = Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(res, 5.0)),
    );
    let base_id = base.id();
    stack.push(base);
    let mut voronoi = Layer::new(
        "voronoi",
        LayerKind::VoronoiRegions(VoronoiParams::default()),
    );
    voronoi.common.blend = BlendMode::Add;
    stack.push(voronoi);

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm Voronoi contribution cache");

    let LayerKind::SculptBase(params) = &mut stack.find_mut(base_id).expect("base layer").kind
    else {
        panic!("base changed kind");
    };
    for y in rect.1..rect.1 + rect.3 {
        for x in rect.0..rect.0 + rect.2 {
            params.samples[(y * res + x) as usize] += 7.0;
        }
    }
    engine.set_dirty_rect(Some(rect));
    engine.mark_dirty(base_id);
    let incremental = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("bounded edit with cached Voronoi contribution")
        .cpu
        .expect("incremental GPU readback");

    let mut oracle_engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    oracle_engine.mark_all_dirty(&stack);
    let oracle = oracle_engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("fresh Voronoi GPU oracle")
        .cpu
        .expect("oracle GPU readback");

    let error = terra_gpu::parity::max_abs_diff(&incremental.to_dense(), &oracle.to_dense());
    assert!(
        error <= 1.0e-3,
        "cached Voronoi contribution drifted by {error}"
    );
    assert_eq!(engine.executed_kernels, vec![GpuKernel::Sculpt]);
    assert_eq!(engine.last_eval_stats().reused_contributions, 1);
}

/// #136: a warm first-layer SculptBase edit remains bounded through upload,
/// SculptStrokes, cached Volcano re-blend, composite-cache maintenance, and
/// presentation while matching a fresh full-field GPU oracle everywhere.
#[test]
fn warm_layer_zero_sculpt_edit_is_region_complete() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 96u32;
    let mut metrics = HeightfieldMetrics::new(res, res, 960.0, 960.0);
    metrics.tile_size = 16;
    metrics.halo = 2;
    let rect = (40u32, 40u32, 16u32, 16u32);

    let mut stack = LayerStack::new();
    let base = Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(24, 12.0)),
    );
    let base_id = base.id();
    stack.push(base);
    stack.push(Layer::new(
        "strokes",
        LayerKind::SculptStrokes(raise_strokes(0.5, 0.5, 8.0)),
    ));
    stack.push(Layer::new(
        "volcano",
        LayerKind::Volcano(terra_core::layer::VolcanoParams::default()),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, res);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm representative stack");

    let LayerKind::SculptBase(params) = &mut stack.find_mut(base_id).expect("base layer").kind
    else {
        panic!("base changed kind");
    };
    params.stamp_circle(0.5, 0.5, 0.06, 4.0, 0);

    engine.set_dirty_rect(Some(rect));
    engine.mark_dirty(base_id);
    let incremental = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("bounded warm edit")
        .cpu
        .expect("incremental readback");

    let mut oracle_engine = GpuTerrainEngine::new(&gpu.device, res);
    let oracle = oracle_engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("fresh full-field oracle")
        .cpu
        .expect("oracle readback");
    let error = terra_gpu::parity::max_abs_diff(&incremental.to_dense(), &oracle.to_dense());
    let worst = incremental
        .to_dense()
        .iter()
        .zip(oracle.to_dense())
        .enumerate()
        .max_by(|(_, (a0, b0)), (_, (a1, b1))| (*a0 - *b0).abs().total_cmp(&(*a1 - *b1).abs()))
        .map(|(index, (a, b))| (index % res as usize, index / res as usize, *a, b));
    assert!(
        error <= 1.0e-3,
        "bounded layer-zero edit drifted by {error} at {worst:?}"
    );

    let stats = engine.last_eval_stats();
    assert!(stats.used_layer_zero_region);
    assert_eq!(stats.reused_contributions, 1);
    assert_eq!(
        engine.executed_kernels,
        vec![GpuKernel::Sculpt, GpuKernel::SculptStrokes]
    );
    assert!(
        engine.dirty_tiles().len() < (metrics.tiles_x() * metrics.tiles_z()) as usize,
        "a warm compiled-plan stroke suffix must preserve bounded presentation"
    );
    assert!(
        stats.upload_bytes < u64::from(metrics.width * metrics.height * 4),
        "compiled-plan Base upload must scale with the expanded edit region"
    );
}

#[test]
fn layer_zero_region_requires_warm_stable_unmasked_caches() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 48u32;
    let mut metrics = HeightfieldMetrics::new(res, res, 480.0, 480.0);
    metrics.tile_size = 8;
    let rect = (20, 20, 4, 4);
    let build = || {
        let mut stack = LayerStack::new();
        let base = Layer::new(
            "base",
            LayerKind::SculptBase(SculptParams::filled(res, 10.0)),
        );
        let id = base.id();
        stack.push(base);
        stack.push(Layer::new(
            "strokes",
            LayerKind::SculptStrokes(raise_strokes(0.5, 0.5, 5.0)),
        ));
        stack.push(Layer::new(
            "volcano",
            LayerKind::Volcano(terra_core::layer::VolcanoParams::default()),
        ));
        (stack, id)
    };

    let (cold_stack, cold_id) = build();
    let mut cold = GpuTerrainEngine::new(&gpu.device, res);
    cold.set_dirty_rect(Some(rect));
    cold.mark_dirty(cold_id);
    cold.evaluate(
        &gpu.device,
        &gpu.queue,
        &cold_stack,
        &[],
        metrics,
        PreviewQuality::Draft,
        false,
        None,
    )
    .expect("cold fallback");
    assert!(!cold.last_eval_stats().used_layer_zero_region);
    assert_eq!(
        cold.dirty_tiles().len(),
        (metrics.tiles_x() * metrics.tiles_z()) as usize
    );

    let (quality_stack, quality_id) = build();
    let mut quality = GpuTerrainEngine::new(&gpu.device, res);
    quality.mark_all_dirty(&quality_stack);
    quality
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &quality_stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm quality caches");
    quality.set_dirty_rect(Some(rect));
    quality.mark_dirty(quality_id);
    quality
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &quality_stack,
            &[],
            metrics,
            PreviewQuality::Medium,
            false,
            None,
        )
        .expect("quality-change fallback");
    assert!(!quality.last_eval_stats().used_layer_zero_region);

    let mask = MaskAsset::new(MaskId::new(), "constant", MaskSource::Constant(0.5));
    let (mut masked_stack, masked_id) = build();
    masked_stack
        .find_mut(masked_id)
        .expect("masked base")
        .common
        .masks
        .push(MaskRef::new(mask.id));
    let mut masked = GpuTerrainEngine::new(&gpu.device, res);
    masked.mark_all_dirty(&masked_stack);
    masked
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &masked_stack,
            std::slice::from_ref(&mask),
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm masked stack");
    masked.set_dirty_rect(Some(rect));
    masked.mark_dirty(masked_id);
    masked
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &masked_stack,
            std::slice::from_ref(&mask),
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("masked fallback");
    assert!(masked.last_eval_stats().used_layer_zero_region);
}

/// #133 regression: decoded source textures are shared by asset identity,
/// while each layer keeps an output-sized contribution cache. A bounded base
/// edit therefore reblends both raster contributions without decoding,
/// uploading, or dispatching the sampling kernel again.
#[test]
fn warm_cache_base_edit_reuses_heightmap_contributions_and_source_texture() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("terra-gpu-cache-{unique}.png"));
    let fixture = image::ImageBuffer::from_fn(5, 7, |x, y| {
        image::Luma([((x * 8000 + y * 6000) % 65536) as u16])
    });
    fixture.save(&path).expect("write source fixture");

    let metrics = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
    let mut stack = LayerStack::new();
    let base = Layer::new("base", LayerKind::SculptBase(SculptParams::filled(32, 5.0)));
    let base_id = base.id();
    stack.push(base);
    let params = ImportHeightmapParams {
        path: path.to_string_lossy().into_owned(),
        height_scale: 20.0,
        height_offset: 2.0,
    };
    stack.push(Layer::new(
        "import",
        LayerKind::ImportHeightmap(params.clone()),
    ));
    let mut stamp = Layer::new(
        "stamp",
        LayerKind::Stamp2d(Stamp2dParams { heightmap: params }),
    );
    stamp.common.shape_transform = Some(terra_core::biome_paint::ShapeTransform {
        offset_x: 18.0,
        offset_z: -12.0,
        scale: 0.7,
        rotation_deg: 27.0,
        blend_size: 0.2,
        blend_roundness: 0.35,
    });
    stack.push(stamp);

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm raster contributions");
    assert_eq!(engine.source_upload_count, 1, "same asset uploads once");

    let Some(layer) = stack.find_mut(base_id) else {
        panic!("base layer disappeared");
    };
    let LayerKind::SculptBase(params) = &mut layer.kind else {
        panic!("base changed kind");
    };
    params.samples[16 * 32 + 16] += 3.0;
    engine.set_dirty_rect(Some((16, 16, 1, 1)));
    engine.mark_dirty(base_id);
    let warm = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("incremental base edit");

    assert_eq!(engine.executed_kernels, vec![GpuKernel::Sculpt]);
    assert_eq!(engine.source_upload_count, 1);
    let actual = warm.cpu.expect("warm transformed-stamp readback");
    let expected = cpu_oracle(&stack, metrics);
    let max_error = actual
        .to_dense()
        .iter()
        .zip(expected.to_dense())
        .map(|(gpu, cpu)| (gpu - cpu).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error < 0.001, "warm Stamp2d max error {max_error}");
    let _ = std::fs::remove_file(path);
}

/// #132 regression: the picker wrapper has the same input-independent cache
/// semantics as its delegated generator. A base edit reblends the cached
/// contribution without redispatching the procedural kernel.
#[test]
fn warm_cache_base_edit_reuses_procedural_shape_contribution() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(24, 24, 240.0, 240.0);
    let mut stack = LayerStack::new();
    let base = Layer::new("base", LayerKind::SculptBase(SculptParams::filled(24, 5.0)));
    let base_id = base.id();
    stack.push(base);
    stack.push(Layer::new(
        "procedural volcano",
        LayerKind::ProceduralShape(ProceduralShapeParams::with_generator(
            ProceduralGenerator::Volcano,
        )),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm procedural contribution cache");
    assert!(engine.plan_resources.current().is_some());

    let Some(layer) = stack.find_mut(base_id) else {
        panic!("base layer disappeared");
    };
    let LayerKind::SculptBase(params) = &mut layer.kind else {
        panic!("base changed kind");
    };
    params.samples[12 * 24 + 12] += 3.0;
    engine.set_dirty_rect(Some((12, 12, 1, 1)));
    engine.mark_dirty(base_id);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("incremental base edit");

    assert_eq!(engine.executed_kernels, vec![GpuKernel::Sculpt]);
    assert!(
        !engine
            .executed_kernels
            .contains(&GpuKernel::ProceduralShape),
        "cached picker contribution must avoid ProceduralShape dispatch"
    );
}
