use terra_core::authoring::{SculptPoint, SculptStroke, SculptStrokeKind, SculptStrokeParams};
use terra_core::deps::NodeRef;
use terra_core::heightfield::HeightfieldMetrics;
use terra_core::layer::{
    BlendMode, BlurParams, FlatParams, Layer, LayerKind, LayerStack, NoiseParams, RiverCarveParams,
    SculptParams,
};
use terra_core::mask::{MaskAsset, MaskId, MaskRef, MaskSource};
use terra_core::quality::PreviewQuality;
use terra_core::terrain_plan::{
    compile_terrain_plan, propagate_plan_edits, PlanDirtyScope, PlanStructureRevision,
    TerrainEditClass, TerrainPlanStamp,
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
    // Finite-width Terrace filters the running quantized target spatially.
    // Keeping it in the shared fixture makes both the full-vs-tile and adjacent
    // domain tests cover its conservative halo and seam contract.
    stack.push(Layer::new(
        "soft terrace",
        LayerKind::SculptStrokes(SculptStrokeParams {
            strokes: vec![SculptStroke {
                kind: SculptStrokeKind::Terrace,
                points: vec![SculptPoint {
                    u: 0.5,
                    v: 0.5,
                    pressure: 1.0,
                }],
                radius_m: 1_000.0,
                strength: 8.0,
                target_height: 0.0,
                riser_width_m: 12.0,
                falloff: 1.5,
                enabled: true,
            }],
            reconcile: 0.15,
        }),
    ));
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
            PreviewQuality::Export,
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
            PreviewQuality::Export,
            domain.clone(),
        )
        .unwrap();
    assert_eq!(producer.stats().engine_allocations, 1);
    assert_eq!(
        producer.stats().evaluated_texels,
        u64::from(domain.evaluation.width) * u64::from(domain.evaluation.height),
        "the producer must allocate exactly the conservative evaluation domain"
    );
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

#[test]
fn basin_checkpoint_tiles_are_seamless_and_match_complete_gpu() {
    let gpu = terra_test_gpu::headless_required();
    let resolution = 48;
    let metrics = HeightfieldMetrics::new(resolution, resolution, 480.0, 480.0);
    let samples = (0..resolution)
        .flat_map(|z| {
            (0..resolution).map(move |x| {
                let ridge = ((x as f32 - 24.0).abs() * 0.35).min(7.0);
                30.0 + ridge + z as f32 * 0.12
            })
        })
        .collect();
    let mut stack = LayerStack::new();
    let base = Layer::new(
        "basin base",
        LayerKind::SculptBase(SculptParams {
            width: resolution,
            height: resolution,
            samples,
            fill_height: 0.0,
        }),
    );
    let base_id = base.id();
    stack.push(base);
    stack.push(Layer::new(
        "basin river",
        LayerKind::RiverCarve(RiverCarveParams {
            accumulation_threshold: 2.0,
            depth: 3.0,
            width: 1.0,
            bank_smooth: 0.0,
            use_dinfinity: false,
            ..RiverCarveParams::default()
        }),
    ));
    stack.push(Layer::new(
        "post-checkpoint blur",
        LayerKind::Blur(BlurParams {
            radius: 1,
            iterations: 1,
        }),
    ));
    let mask = MaskAsset::new(MaskId::new(), "half", MaskSource::Constant(0.5));
    let mut add = Layer::new(
        "post-checkpoint mask",
        LayerKind::Flat(FlatParams { height: 2.0 }),
    );
    add.common.blend = BlendMode::Add;
    add.common.masks.push(MaskRef::new(mask.id));
    stack.push(add);
    let masks = vec![mask];

    let revision = PlanStructureRevision::new(41);
    let plan = compile_terrain_plan(&stack, &masks, TerrainPlanStamp::new(revision)).unwrap();
    let invalidation = propagate_plan_edits(&plan, &[TerrainEditClass::Structure]);
    let slice = GpuCompiledTileProducer::analyze(&stack, &plan).unwrap();
    assert_eq!(slice.operation_halo, 1);

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

    let mut config = PyramidConfig::new(resolution, 480.0, 480.0);
    config.tile_size = 24;
    config.halo = 2;
    let pyramid = TerrainPyramid::new(config);
    let stamp = TerrainContentStamp {
        document_revision: 43,
        plan_revision: revision.get(),
        output_revision: 47,
        content_revision: 53,
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
    let mut tiles = Vec::new();
    for tx in 0..=1 {
        let mut job = producer
            .begin(
                &gpu.device,
                &gpu.queue,
                &stack,
                &masks,
                &plan,
                &invalidation,
                PreviewQuality::Full,
                make_domain(tx),
            )
            .unwrap();
        wait_for_tile(&mut producer, gpu, &mut job);
        let domain = job.domain().clone();
        let values = job
            .readback_height(&gpu.device, &gpu.queue)
            .unwrap()
            .to_dense();
        for z in domain.interior.origin_z..domain.interior.origin_z + domain.interior.height {
            for x in domain.interior.origin_x..domain.interior.origin_x + domain.interior.width {
                let local_x = x - domain.evaluation.origin_x;
                let local_z = z - domain.evaluation.origin_z;
                let actual = values[(local_z * domain.evaluation.width + local_x) as usize];
                let expected = full[(z * resolution + x) as usize];
                assert!(
                    (actual - expected).abs() <= 1.0e-5,
                    "checkpoint tile differs at ({x},{z}): {actual} vs {expected}"
                );
            }
        }
        tiles.push((domain, values));
        producer.recycle(job);
    }
    assert_eq!(producer.stats().checkpoint_builds, 1);
    assert!(producer.stats().checkpoint_reuses >= 1);

    let (left_domain, left) = &tiles[0];
    let (right_domain, right) = &tiles[1];
    for global_z in 2..22 {
        for global_x in [23, 24] {
            let left_index = (global_z - left_domain.evaluation.origin_z)
                * left_domain.evaluation.width
                + global_x
                - left_domain.evaluation.origin_x;
            let right_index = (global_z - right_domain.evaluation.origin_z)
                * right_domain.evaluation.width
                + global_x
                - right_domain.evaluation.origin_x;
            assert!(
                (left[left_index as usize] - right[right_index as usize]).abs() <= 1.0e-5,
                "checkpoint boundary sample ({global_x},{global_z}) is discontinuous"
            );
        }
    }

    // Submit work using the old immutable checkpoint, cancel its publication,
    // then advance the content revision and change an upstream prefix payload.
    // The replacement tile must come entirely from the new checkpoint.
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
    let LayerKind::SculptBase(params) = &mut stack.find_mut(base_id).unwrap().kind else {
        panic!("basin fixture base changed kind");
    };
    for sample in &mut params.samples {
        *sample += 10.0;
    }
    let next_invalidation = propagate_plan_edits(
        &plan,
        &[TerrainEditClass::Content {
            owner: NodeRef::Layer(base_id),
            fields: vec![FieldId::Height],
            scope: PlanDirtyScope::FullField,
        }],
    );
    let next_stamp = TerrainContentStamp {
        document_revision: stamp.document_revision + 1,
        output_revision: stamp.output_revision + 1,
        content_revision: stamp.content_revision + 1,
        ..stamp
    };
    let next_domain = TerrainEvaluationDomain::for_tile(
        &pyramid,
        TerrainTileKey {
            layer: None,
            field: FieldId::Height,
            level: pyramid.max_level(),
            tile: TileId { tx: 0, tz: 0 },
        },
        config.halo,
        slice.operation_halo,
        next_stamp,
    )
    .unwrap();
    let mut replacement = producer
        .begin(
            &gpu.device,
            &gpu.queue,
            &stack,
            &masks,
            &plan,
            &next_invalidation,
            PreviewQuality::Full,
            next_domain,
        )
        .unwrap();
    wait_for_tile(&mut producer, gpu, &mut replacement);
    let replacement_values = replacement
        .readback_height(&gpu.device, &gpu.queue)
        .unwrap()
        .to_dense();
    let old_domain = &tiles[0].0;
    let sample_x = 8;
    let sample_z = 8;
    let old_index = (sample_z - old_domain.evaluation.origin_z) * old_domain.evaluation.width
        + sample_x
        - old_domain.evaluation.origin_x;
    let new_domain = replacement.domain();
    let new_index = (sample_z - new_domain.evaluation.origin_z) * new_domain.evaluation.width
        + sample_x
        - new_domain.evaluation.origin_x;
    assert!(
        (replacement_values[new_index as usize] - tiles[0].1[old_index as usize] - 10.0).abs()
            <= 1.0e-4,
        "replacement tile mixed old and new checkpoint revisions"
    );
    assert_eq!(producer.stats().checkpoint_builds, 2);
    producer.recycle(replacement);
}
