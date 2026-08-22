use terra_core::heightfield::HeightfieldMetrics;
use terra_core::layer::{
    BlendMode, BlurParams, FlatParams, Layer, LayerKind, LayerStack, NoiseParams, SculptParams,
};
use terra_core::mask::{MaskAsset, MaskId, MaskRef, MaskSource};
use terra_core::quality::PreviewQuality;
use terra_core::terrain_plan::{
    compile_terrain_plan, propagate_plan_edits, PlanStructureRevision, TerrainEditClass,
    TerrainPlanStamp,
};
use terra_core::{
    FieldId, PyramidConfig, TerrainContentStamp, TerrainEvaluationDomain, TerrainPyramid,
    TerrainTileKey, TileId,
};
use terra_gpu::GpuTileAtlas;
use terra_gpu_eval::{GpuCompiledTileProducer, GpuEvaluationIntent, GpuTerrainEngine};

fn fixture(resolution: u32) -> (LayerStack, Vec<MaskAsset>) {
    let samples = (0..resolution)
        .flat_map(|z| (0..resolution).map(move |x| x as f32 * 0.25 + z as f32 * 0.5))
        .collect();
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "sculpt base",
        LayerKind::SculptBase(SculptParams {
            width: resolution,
            height: resolution,
            samples,
            fill_height: 0.0,
        }),
    ));
    let mut noise = Layer::new(
        "world-space generator",
        LayerKind::NoiseValue(NoiseParams {
            seed: 918_273,
            frequency: 0.017,
            amplitude: 12.0,
            octaves: 3,
            offset_x: 4.25,
            offset_z: -7.5,
            ..NoiseParams::default()
        }),
    );
    noise.common.blend = BlendMode::Add;
    stack.push(noise);
    stack.push(Layer::new(
        "local blur",
        LayerKind::Blur(BlurParams {
            radius: 2,
            iterations: 1,
        }),
    ));
    let mask = MaskAsset::new(MaskId::new(), "constant", MaskSource::Constant(0.5));
    let mut add = Layer::new("masked add", LayerKind::Flat(FlatParams { height: 8.0 }));
    add.common.blend = BlendMode::Add;
    add.common.masks.push(MaskRef::new(mask.id));
    stack.push(add);
    (stack, vec![mask])
}

fn wait_for_tile(
    producer: &mut GpuCompiledTileProducer,
    gpu: &terra_test_gpu::TestGpu,
    job: &mut terra_gpu_eval::GpuTileEvaluationJob,
) {
    let _ = gpu.device.poll(wgpu::Maintain::Wait);
    assert!(producer.poll(&gpu.device, job));
}

#[test]
fn compiled_tile_matches_full_gpu_and_allocates_only_the_domain() {
    let gpu = terra_test_gpu::headless_required();
    let resolution = 64;
    let metrics = HeightfieldMetrics::new(resolution, resolution, 640.0, 320.0);
    let (stack, masks) = fixture(resolution);
    let revision = PlanStructureRevision::new(7);
    let plan = compile_terrain_plan(&stack, &masks, TerrainPlanStamp::new(revision)).unwrap();
    let invalidation = propagate_plan_edits(&plan, &[TerrainEditClass::Structure]);

    let mut full_engine = GpuTerrainEngine::new(&gpu.device, resolution);
    full_engine.mark_all_dirty(&stack);
    let full = full_engine
        .evaluate_compiled_with_intent(
            &gpu.device,
            &gpu.queue,
            &stack,
            &masks,
            &plan,
            revision,
            &invalidation,
            metrics,
            PreviewQuality::Full,
            true,
            GpuEvaluationIntent::Complete,
        )
        .unwrap()
        .cpu
        .unwrap()
        .to_dense();

    let mut config = PyramidConfig::new(resolution, 640.0, 320.0);
    config.tile_size = 24;
    config.halo = 2;
    let pyramid = TerrainPyramid::new(config);
    let slice = GpuCompiledTileProducer::analyze(&stack, &plan).unwrap();
    let stamp = TerrainContentStamp {
        document_revision: 11,
        plan_revision: revision.get(),
        output_revision: 13,
        content_revision: 17,
    };
    let domain = TerrainEvaluationDomain::for_tile(
        &pyramid,
        TerrainTileKey {
            layer: None,
            field: FieldId::Height,
            level: pyramid.max_level(),
            tile: TileId { tx: 1, tz: 1 },
        },
        config.halo,
        slice.operation_halo,
        stamp,
    )
    .unwrap();
    let mut producer = GpuCompiledTileProducer::new();
    let mut job = producer
        .begin(
            &gpu.device,
            &gpu.queue,
            &stack,
            &masks,
            &plan,
            &invalidation,
            PreviewQuality::Full,
            domain.clone(),
        )
        .unwrap();
    assert_eq!(producer.stats().engine_allocations, 1);
    assert!(producer.stats().evaluated_texels < u64::from(resolution * resolution));
    wait_for_tile(&mut producer, gpu, &mut job);
    let local = job
        .readback_height(&gpu.device, &gpu.queue)
        .unwrap()
        .to_dense();
    for z in domain.interior.origin_z..domain.interior.origin_z + domain.interior.height {
        for x in domain.interior.origin_x..domain.interior.origin_x + domain.interior.width {
            let lx = x - domain.evaluation.origin_x;
            let lz = z - domain.evaluation.origin_z;
            let actual = local[(lz * domain.evaluation.width + lx) as usize];
            let expected = full[(z * resolution + x) as usize];
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "tile differs at ({x},{z}): actual={actual}, expected={expected}"
            );
        }
    }

    let mut atlas = GpuTileAtlas::new(&gpu.device, config.tile_size, config.halo, 4).unwrap();
    atlas.configure_hierarchy(&gpu.device, &gpu.queue, &pyramid);
    assert_eq!(atlas.residency().stats().resident_tiles, 0);
    let stale = TerrainContentStamp {
        content_revision: stamp.content_revision + 1,
        ..stamp
    };
    assert!(atlas
        .publish_evaluated_tile_current_at_frame(
            &gpu.device,
            &gpu.queue,
            job.output_texture_view().unwrap(),
            job.domain(),
            stale,
            1,
        )
        .is_err());
    assert_eq!(atlas.residency().stats().resident_tiles, 0);
    atlas
        .publish_evaluated_tile_current_at_frame(
            &gpu.device,
            &gpu.queue,
            job.output_texture_view().unwrap(),
            job.domain(),
            stamp,
            1,
        )
        .unwrap();
    assert_eq!(atlas.residency().stats().resident_tiles, 1);
    producer.recycle(job);
}

#[test]
fn adjacent_domains_share_samples_and_cancelled_resources_recycle_after_fence() {
    let gpu = terra_test_gpu::headless_required();
    let resolution = 48;
    let (stack, masks) = fixture(resolution);
    let revision = PlanStructureRevision::new(23);
    let plan = compile_terrain_plan(&stack, &masks, TerrainPlanStamp::new(revision)).unwrap();
    let invalidation = propagate_plan_edits(&plan, &[TerrainEditClass::Structure]);
    let slice = GpuCompiledTileProducer::analyze(&stack, &plan).unwrap();
    let mut config = PyramidConfig::new(resolution, 480.0, 240.0);
    config.tile_size = 24;
    config.halo = 2;
    let pyramid = TerrainPyramid::new(config);
    let stamp = TerrainContentStamp {
        document_revision: 19,
        plan_revision: revision.get(),
        output_revision: 29,
        content_revision: 31,
    };
    let make_domain = |tx| {
        TerrainEvaluationDomain::for_tile(
            &pyramid,
            TerrainTileKey {
                layer: None,
                field: FieldId::Height,
                level: pyramid.max_level(),
                tile: TileId { tx, tz: 0 },
            },
            config.halo,
            slice.operation_halo,
            stamp,
        )
        .unwrap()
    };
    let mut producer = GpuCompiledTileProducer::new();
    let cancelled = producer
        .begin(
            &gpu.device,
            &gpu.queue,
            &stack,
            &masks,
            &plan,
            &invalidation,
            PreviewQuality::Full,
            make_domain(0),
        )
        .unwrap();
    producer.cancel(cancelled);
    let _ = gpu.device.poll(wgpu::Maintain::Wait);

    let mut left = producer
        .begin(
            &gpu.device,
            &gpu.queue,
            &stack,
            &masks,
            &plan,
            &invalidation,
            PreviewQuality::Full,
            make_domain(0),
        )
        .unwrap();
    assert!(producer.stats().engine_reuses >= 1);
    wait_for_tile(&mut producer, gpu, &mut left);
    let left_domain = left.domain().clone();
    let left_data = left
        .readback_height(&gpu.device, &gpu.queue)
        .unwrap()
        .to_dense();
    producer.recycle(left);

    let mut right = producer
        .begin(
            &gpu.device,
            &gpu.queue,
            &stack,
            &masks,
            &plan,
            &invalidation,
            PreviewQuality::Full,
            make_domain(1),
        )
        .unwrap();
    wait_for_tile(&mut producer, gpu, &mut right);
    let right_domain = right.domain().clone();
    let right_data = right
        .readback_height(&gpu.device, &gpu.queue)
        .unwrap()
        .to_dense();

    for global_x in [23u32, 24u32] {
        for global_z in 2..22u32 {
            let lx = global_x - left_domain.evaluation.origin_x;
            let lz = global_z - left_domain.evaluation.origin_z;
            let rx = global_x - right_domain.evaluation.origin_x;
            let rz = global_z - right_domain.evaluation.origin_z;
            let a = left_data[(lz * left_domain.evaluation.width + lx) as usize];
            let b = right_data[(rz * right_domain.evaluation.width + rx) as usize];
            assert!(
                (a - b).abs() <= 1.0e-5,
                "overlapping boundary sample ({global_x},{global_z}) differs: {a} vs {b}"
            );
        }
    }
    producer.recycle(right);
}
