use std::collections::BTreeMap;

use terra_core::layer::{
    BlendMode, BlurParams, FlatParams, Layer, LayerKind, LayerStack, NoiseParams,
};
use terra_core::quality::PreviewQuality;
use terra_core::terrain_plan::{
    compile_terrain_plan, propagate_plan_edits, PlanStructureRevision, TerrainEditClass,
    TerrainPlanStamp,
};
use terra_core::{TerrainContentStamp, TerrainEvaluationDomain, TerrainTileKey};
use terra_gpu_eval::{GpuCompiledTileProducer, GpuPackedHeightTile, GpuTileEvaluationError};
use terra_world::{
    InfiniteTopology, InfiniteTopologyConfig, Lod, TileAddress, TileCoord, WorldPosition,
};

const TILE_SIZE: u32 = 32;
const PUBLICATION_HALO: u32 = 2;

fn topology() -> InfiniteTopology {
    InfiniteTopology::try_new(InfiniteTopologyConfig {
        origin: WorldPosition::try_new(8_000_000.125, -7_000_000.375).unwrap(),
        tile_size: TILE_SIZE,
        finest_spacing_m: 0.5,
        max_lod: Lod::try_new(8).unwrap(),
    })
    .unwrap()
}

fn stack(with_blur: bool) -> LayerStack {
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "absolute perlin",
        LayerKind::NoisePerlin(NoiseParams {
            seed: 91_827,
            frequency: 0.017,
            amplitude: 25.0,
            octaves: 3,
            offset_x: 4.25,
            offset_z: -7.5,
            ..NoiseParams::default()
        }),
    ));
    if with_blur {
        stack.push(Layer::new(
            "bounded blur",
            LayerKind::Blur(BlurParams {
                radius: 2,
                iterations: 2,
            }),
        ));
    }
    let mut add = Layer::new("blend", LayerKind::Flat(FlatParams { height: 3.0 }));
    add.common.blend = BlendMode::Add;
    stack.push(add);
    stack
}

fn compile(stack: &LayerStack, revision: u64) -> terra_core::terrain_plan::CompiledTerrainPlan {
    compile_terrain_plan(
        stack,
        &[],
        TerrainPlanStamp::new(PlanStructureRevision::new(revision)),
    )
    .unwrap()
}

fn stamp(revision: u64) -> TerrainContentStamp {
    TerrainContentStamp {
        document_revision: 13,
        plan_revision: revision,
        output_revision: 17,
        content_revision: 19,
    }
}

fn evaluate(
    gpu: &terra_test_gpu::TestGpu,
    producer: &mut GpuCompiledTileProducer,
    topology: &InfiniteTopology,
    stack: &LayerStack,
    plan: &terra_core::terrain_plan::CompiledTerrainPlan,
    coord: TileCoord,
) -> (TerrainEvaluationDomain, GpuPackedHeightTile) {
    let slice = GpuCompiledTileProducer::analyze_infinite(stack, &[], plan).unwrap();
    let domain = TerrainEvaluationDomain::for_infinite_tile(
        topology,
        TerrainTileKey::height(TileAddress::new(Lod::FINEST, coord)),
        77,
        PUBLICATION_HALO,
        slice.operation_halo,
        stamp(plan.stamp().structure_revision.get()),
    )
    .unwrap();
    let invalidation = propagate_plan_edits(plan, &[TerrainEditClass::Structure]);
    let mut job = producer
        .begin(
            &gpu.device,
            &gpu.queue,
            stack,
            &[],
            plan,
            &invalidation,
            PreviewQuality::Full,
            domain.clone(),
        )
        .unwrap();
    let _ = gpu.device.poll(wgpu::Maintain::Wait);
    assert!(producer.poll(&gpu.device, &mut job));
    let page_extent = TILE_SIZE + PUBLICATION_HALO * 2;
    let mut readback = job
        .begin_packed_readback(&gpu.device, &gpu.queue, page_extent)
        .unwrap();
    let _ = gpu.device.poll(wgpu::Maintain::Wait);
    let packed = producer
        .poll_packed_readback(&gpu.device, &mut readback)
        .unwrap()
        .expect("mapped packed tile");
    (domain, packed)
}

fn absolute_samples(
    domain: &TerrainEvaluationDomain,
    tile: &GpuPackedHeightTile,
) -> BTreeMap<(i64, i64), u32> {
    let origin_x = domain.spatial.interior.samples.origin.x - i64::from(tile.halo);
    let origin_z = domain.spatial.interior.samples.origin.z - i64::from(tile.halo);
    let mut samples = BTreeMap::new();
    let valid_width = tile.interior_width + tile.halo * 2;
    let valid_height = tile.interior_height + tile.halo * 2;
    for z in 0..valid_height {
        for x in 0..valid_width {
            samples.insert(
                (origin_x + i64::from(x), origin_z + i64::from(z)),
                tile.samples[(z * tile.page_extent + x) as usize].to_bits(),
            );
        }
    }
    samples
}

fn assert_overlaps_match(
    left_domain: &TerrainEvaluationDomain,
    left: &GpuPackedHeightTile,
    right_domain: &TerrainEvaluationDomain,
    right: &GpuPackedHeightTile,
) {
    let left = absolute_samples(left_domain, left);
    let right = absolute_samples(right_domain, right);
    let mut overlap = 0;
    for (coord, value) in left {
        if let Some(other) = right.get(&coord) {
            assert_eq!(value, *other, "GPU seam at {coord:?}");
            overlap += 1;
        }
    }
    assert!(overlap > 0);
}

#[test]
fn infinite_gpu_tiles_are_order_independent_regenerable_and_unclamped() {
    let gpu = terra_test_gpu::headless_required();
    let topology = topology();
    let stack = stack(false);
    let plan = compile(&stack, 185);
    let mut producer = GpuCompiledTileProducer::new();
    let (right_domain, right) = evaluate(
        gpu,
        &mut producer,
        &topology,
        &stack,
        &plan,
        TileCoord { x: 0, z: 0 },
    );
    let (left_domain, left) = evaluate(
        gpu,
        &mut producer,
        &topology,
        &stack,
        &plan,
        TileCoord { x: -1, z: 0 },
    );
    let (_, regenerated) = evaluate(
        gpu,
        &mut producer,
        &topology,
        &stack,
        &plan,
        TileCoord { x: -1, z: 0 },
    );
    assert_eq!(left.samples, regenerated.samples);
    assert_overlaps_match(&left_domain, &left, &right_domain, &right);
    assert!(left.samples.windows(2).any(|pair| pair[0] != pair[1]));
}

#[test]
fn infinite_gpu_blur_uses_guard_and_has_no_publication_seam() {
    let gpu = terra_test_gpu::headless_required();
    let topology = topology();
    let stack = stack(true);
    let plan = compile(&stack, 186);
    let slice = GpuCompiledTileProducer::analyze_infinite(&stack, &[], &plan).unwrap();
    assert_eq!(slice.operation_halo, 4);
    let mut producer = GpuCompiledTileProducer::new();
    let (left_domain, left) = evaluate(
        gpu,
        &mut producer,
        &topology,
        &stack,
        &plan,
        TileCoord { x: -1, z: 0 },
    );
    let (right_domain, right) = evaluate(
        gpu,
        &mut producer,
        &topology,
        &stack,
        &plan,
        TileCoord { x: 0, z: 0 },
    );
    assert_overlaps_match(&left_domain, &left, &right_domain, &right);
}

#[test]
fn infinite_gpu_rejects_unsupported_work_before_dispatch() {
    let gpu = terra_test_gpu::headless_required();
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "cpu-only",
        LayerKind::NoiseOpenSimplex(NoiseParams::default()),
    ));
    let plan = compile(&stack, 187);
    let domain = TerrainEvaluationDomain::for_infinite_tile(
        &topology(),
        TerrainTileKey::height(TileAddress::new(Lod::FINEST, TileCoord::ZERO)),
        1,
        PUBLICATION_HALO,
        0,
        stamp(187),
    )
    .unwrap();
    let invalidation = propagate_plan_edits(&plan, &[TerrainEditClass::Structure]);
    let mut producer = GpuCompiledTileProducer::new();
    assert!(matches!(
        producer.begin(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            &plan,
            &invalidation,
            PreviewQuality::Full,
            domain,
        ),
        Err(GpuTileEvaluationError::UnsupportedOperation { .. })
    ));
    assert_eq!(producer.stats().submitted, 0);
    assert_eq!(producer.stats().engine_allocations, 0);
}
