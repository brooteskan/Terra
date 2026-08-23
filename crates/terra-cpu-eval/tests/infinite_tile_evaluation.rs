use std::collections::BTreeMap;

use terra_core::layer::{
    BlendMode, BlurParams, Layer, LayerKind, LayerStack, NoiseParams, WorleyParams,
};
use terra_core::terrain_plan::{compile_terrain_plan, PlanStructureRevision, TerrainPlanStamp};
use terra_core::{TerrainContentStamp, TerrainEvaluationDomain, TerrainTileKey};
use terra_cpu_eval::{CpuPackedHeightTile, CpuTileEvaluationError, InfiniteTileEvaluator};
use terra_world::{
    InfiniteTopology, InfiniteTopologyConfig, Lod, TileAddress, TileCoord, WorldPosition,
};

const TILE_SIZE: u32 = 32;
const PUBLICATION_HALO: u32 = 2;

fn topology(spacing: f64) -> InfiniteTopology {
    InfiniteTopology::try_new(InfiniteTopologyConfig {
        origin: WorldPosition::ORIGIN,
        tile_size: TILE_SIZE,
        finest_spacing_m: spacing,
        max_lod: Lod::try_new(8).unwrap(),
    })
    .unwrap()
}

fn noise_stack(with_blur: bool) -> LayerStack {
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "absolute noise",
        LayerKind::NoisePerlin(NoiseParams {
            seed: 0x1_0000_0123,
            frequency: 0.013,
            amplitude: 40.0,
            octaves: 4,
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
    let mut offset = Layer::new(
        "stable blend",
        LayerKind::Flat(terra_core::layer::FlatParams { height: 3.0 }),
    );
    offset.common.blend = BlendMode::Add;
    stack.push(offset);
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
        document_revision: 5,
        plan_revision: revision,
        output_revision: 7,
        content_revision: 11,
    }
}

fn evaluate(
    topology: &InfiniteTopology,
    stack: &LayerStack,
    plan: &terra_core::terrain_plan::CompiledTerrainPlan,
    coord: TileCoord,
) -> (TerrainEvaluationDomain, CpuPackedHeightTile) {
    let operation_halo = terra_core::terrain_plan::resolve_infinite_plan_domain(
        stack,
        &[],
        plan,
        plan.final_height(),
    )
    .unwrap()
    .operation_halo;
    let domain = TerrainEvaluationDomain::for_infinite_tile(
        topology,
        TerrainTileKey::height(TileAddress::new(Lod::FINEST, coord)),
        77,
        PUBLICATION_HALO,
        operation_halo,
        stamp(plan.stamp().structure_revision.get()),
    )
    .unwrap();
    let tile = InfiniteTileEvaluator::new()
        .evaluate(stack, &[], plan, domain.clone())
        .unwrap();
    (domain, tile)
}

fn absolute_samples(
    domain: &TerrainEvaluationDomain,
    tile: &CpuPackedHeightTile,
) -> BTreeMap<(i64, i64), u32> {
    let origin_x = domain.spatial.interior.samples.origin.x - i64::from(tile.halo);
    let origin_z = domain.spatial.interior.samples.origin.z - i64::from(tile.halo);
    let mut samples = BTreeMap::new();
    for z in 0..tile.height {
        for x in 0..tile.width {
            samples.insert(
                (origin_x + i64::from(x), origin_z + i64::from(z)),
                tile.samples[(z * tile.width + x) as usize].to_bits(),
            );
        }
    }
    samples
}

#[test]
fn negative_positive_tiles_regenerate_and_share_zero_halo_samples() {
    let topology = topology(1.0);
    let stack = noise_stack(false);
    let plan = compile(&stack, 185);
    let proof_domain = TerrainEvaluationDomain::for_infinite_tile(
        &topology,
        TerrainTileKey::height(TileAddress::new(Lod::FINEST, TileCoord { x: -1, z: 0 })),
        0xfeed_beef,
        1,
        0,
        stamp(185),
    )
    .unwrap();
    let tile = InfiniteTileEvaluator::new()
        .evaluate(&stack, &[], &plan, proof_domain)
        .unwrap();
    assert!(!tile.samples.is_empty());
    let (left_domain, left) = evaluate(&topology, &stack, &plan, TileCoord { x: -1, z: 0 });
    let (right_domain, right) = evaluate(&topology, &stack, &plan, TileCoord { x: 0, z: 0 });
    let (_, regenerated) = evaluate(&topology, &stack, &plan, TileCoord { x: -1, z: 0 });
    assert_eq!(left, regenerated);

    let left_samples = absolute_samples(&left_domain, &left);
    let right_samples = absolute_samples(&right_domain, &right);
    let mut overlap = 0;
    for (coord, value) in &left_samples {
        if let Some(other) = right_samples.get(coord) {
            assert_eq!(value, other, "overlap differs at {coord:?}");
            overlap += 1;
        }
    }
    assert!(overlap > 0);
    assert!(left.samples.windows(2).any(|pair| pair[0] != pair[1]));
}

#[test]
fn bounded_blur_uses_declared_guard_and_crops_without_seams() {
    let topology = topology(1.0);
    let stack = noise_stack(true);
    let plan = compile(&stack, 186);
    let slice = terra_core::terrain_plan::resolve_infinite_plan_domain(
        &stack,
        &[],
        &plan,
        plan.final_height(),
    )
    .unwrap();
    assert_eq!(slice.operation_halo, 4);
    let (left_domain, left) = evaluate(&topology, &stack, &plan, TileCoord { x: -1, z: 0 });
    let (right_domain, right) = evaluate(&topology, &stack, &plan, TileCoord { x: 0, z: 0 });
    let left_samples = absolute_samples(&left_domain, &left);
    let right_samples = absolute_samples(&right_domain, &right);
    for (coord, value) in &left_samples {
        if let Some(other) = right_samples.get(coord) {
            assert_eq!(value, other, "blur seam at {coord:?}");
        }
    }
}

#[test]
fn cpu_only_generators_and_large_signed_addresses_are_supported() {
    for kind in [
        LayerKind::NoiseOpenSimplex(NoiseParams::default()),
        LayerKind::NoiseWorley(WorleyParams::default()),
    ] {
        let mut stack = LayerStack::new();
        stack.push(Layer::new("cpu coordinate generator", kind));
        let plan = compile(&stack, 187);
        let (domain, tile) = evaluate(
            &topology(0.25),
            &stack,
            &plan,
            TileCoord {
                x: 1_000_000,
                z: -1_000_000,
            },
        );
        assert_eq!(domain.spatial.interior.samples.origin.x, 32_000_000);
        assert!(tile.samples.iter().all(|value| value.is_finite()));
        let (_, again) = evaluate(
            &topology(0.25),
            &stack,
            &plan,
            TileCoord {
                x: 1_000_000,
                z: -1_000_000,
            },
        );
        assert_eq!(tile, again);
    }
}

#[test]
fn unsupported_graph_rejects_before_evaluation() {
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "basin operation",
        LayerKind::RiverCarve(Default::default()),
    ));
    let plan = compile(&stack, 188);
    let domain = TerrainEvaluationDomain::for_infinite_tile(
        &topology(1.0),
        TerrainTileKey::height(TileAddress::new(Lod::FINEST, TileCoord::ZERO)),
        1,
        PUBLICATION_HALO,
        0,
        stamp(188),
    )
    .unwrap();
    assert!(matches!(
        InfiniteTileEvaluator::new().evaluate(&stack, &[], &plan, domain),
        Err(CpuTileEvaluationError::Domain(_))
    ));
}
