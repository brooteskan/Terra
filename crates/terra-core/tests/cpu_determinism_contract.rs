//! C2-G1 CPU contract for deterministic evaluation, tiled storage, and halo exchange.

use terra_core::eval::{EvalContext, PreviewQuality, StackEvaluator};
use terra_core::generators::{
    path_stamp, path_stamp_tile, polygon_height, polygon_height_tile, sculpt_base, sculpt_base_tile,
};
use terra_core::heightfield::{HeightTile, Heightfield, HeightfieldMetrics, TileId};
use terra_core::layer::{
    BlurParams, FbmParams, FractalNoiseType, Layer, LayerKind, LayerStack, NoiseParams, PathNode,
    PathParams, PolygonHeightMode, PolygonHeightParams, SculptParams,
};
use terra_core::noise::{open_simplex2, perlin2, value_noise2, worley2, WorleyMetric};
use terra_core::tiling::{map_tiles_batched, measure_seams, TileScheduler};

const NOISE_POINTS: [(f32, f32); 4] = [
    (0.125, 0.375),
    (1.25, 3.5),
    (-7.75, 11.125),
    (93.625, -41.875),
];

// Supported-build CPU contract: these are deliberately bit-level fixtures, not a
// claim that arbitrary toolchain/libm combinations produce the same results.
const PINNED_LOW_SEED_BITS: [[u32; 4]; 4] = [
    [0xbe78_d93d, 0xbe8f_5aec, 0x3e9a_4f29, 0x3dcc_065b],
    [0x3f48_1de3, 0xbeff_f900, 0x3db5_43c0, 0xbd3d_881d],
    [0x3e56_80e5, 0xbe75_dcc9, 0xbe7a_4b98, 0xbe89_a90c],
    [0x3f40_84c4, 0x3ee2_8e60, 0x3ef9_b170, 0x3ecf_4780],
];

fn metrics(
    width: u32,
    height: u32,
    world_size_x: f32,
    world_size_z: f32,
    tile_size: u32,
    halo: u32,
) -> HeightfieldMetrics {
    HeightfieldMetrics {
        width,
        height,
        world_size_x,
        world_size_z,
        tile_size,
        halo,
    }
}

fn field_bits(field: &Heightfield) -> Vec<u32> {
    field.to_dense().into_iter().map(f32::to_bits).collect()
}

fn assert_field_bits_eq(
    expected: &[u32],
    actual: &[u32],
    width: u32,
    context: impl std::fmt::Display,
) {
    assert_eq!(expected.len(), actual.len(), "{context}: field length");
    if let Some(index) = expected.iter().zip(actual).position(|(a, b)| a != b) {
        let i = index as u32 % width;
        let j = index as u32 / width;
        panic!(
            "{context}: mismatch at ({i}, {j}); expected {:#010x}, got {:#010x}",
            expected[index], actual[index]
        );
    }
}

fn patterned_dense(width: u32, height: u32) -> Vec<f32> {
    (0..height)
        .flat_map(|j| {
            (0..width).map(move |i| {
                let base = ((i * 17 + j * 31 + (i * j) % 19) % 257) as f32;
                base * 0.25 - 20.0
            })
        })
        .collect()
}

fn patterned_heightfield(metrics: HeightfieldMetrics) -> Heightfield {
    Heightfield::from_dense(metrics, &patterned_dense(metrics.width, metrics.height))
}

fn sculpt_fixture() -> SculptParams {
    let resolution = 17;
    let samples = (0..resolution)
        .flat_map(|j| {
            (0..resolution)
                .map(move |i| 12.0 + i as f32 * 0.75 + j as f32 * 1.25 + ((i * j) % 5) as f32)
        })
        .collect();
    SculptParams {
        resolution,
        samples,
        fill_height: 12.0,
    }
}

fn polygon_fixture() -> PolygonHeightParams {
    PolygonHeightParams {
        points: vec![[0.08, 0.18], [0.76, 0.07], [0.94, 0.78], [0.31, 0.93]],
        height: 17.5,
        falloff: 0.065,
        carve: false,
        mode: PolygonHeightMode::RaiseBy,
    }
}

fn path_fixture(seed: u64) -> PathParams {
    PathParams {
        nodes: vec![
            PathNode {
                u: 0.03,
                v: 0.84,
                height: 2.0,
                width: 0.8,
            },
            PathNode {
                u: 0.48,
                v: 0.42,
                height: 5.0,
                width: 1.2,
            },
            PathNode {
                u: 0.97,
                v: 0.12,
                height: 3.0,
                width: 0.9,
            },
        ],
        width: 24.0,
        falloff: 13.0,
        noise_strength: 1.75,
        noise_scale: 0.037,
        closed: false,
        height_offset: 8.0,
        carve: true,
        seed,
        spline: false,
        profile: 1.0,
    }
}

fn authored_stack(seed: u64) -> LayerStack {
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Foundation",
        LayerKind::SculptBase(sculpt_fixture()),
    ));
    stack.push(Layer::new(
        "Seeded relief",
        LayerKind::Fbm(FbmParams {
            base: NoiseParams {
                seed,
                frequency: 0.0065,
                amplitude: 28.0,
                octaves: 4,
                persistence: 0.53,
                lacunarity: 2.1,
                ..NoiseParams::default()
            },
            noise: FractalNoiseType::Perlin,
        }),
    ));
    stack.push(Layer::new(
        "Cross-tile polygon",
        LayerKind::PolygonHeight(polygon_fixture()),
    ));
    stack.push(Layer::new(
        "Cross-tile path",
        LayerKind::Path(path_fixture(seed.wrapping_add(99))),
    ));
    stack.push(Layer::new(
        "Final blur",
        LayerKind::Blur(BlurParams {
            radius: 1,
            iterations: 2,
        }),
    ));
    stack
}

fn evaluate_stack(stack: &LayerStack, metrics: HeightfieldMetrics) -> Vec<u32> {
    let mut evaluator = StackEvaluator::new();
    let mut context = EvalContext::new(metrics);
    context.quality = PreviewQuality::Full;
    field_bits(
        &evaluator
            .rebuild_all(stack, &mut context)
            .expect("evaluate CPU layer stack"),
    )
}

fn primitive_bits(seed: u64) -> [[u32; 4]; 4] {
    let sample = |f: fn(f32, f32, u64) -> f32| NOISE_POINTS.map(|(x, z)| f(x, z, seed).to_bits());
    [
        sample(value_noise2),
        sample(perlin2),
        sample(open_simplex2),
        NOISE_POINTS.map(|(x, z)| worley2(x, z, seed, WorleyMetric::Manhattan).f1.to_bits()),
    ]
}

#[test]
fn noise_surface_is_repeatable_and_high_seed_bits_are_observable() {
    let low_seed = 7;
    let high_seed = low_seed + (1u64 << 32);
    let low = primitive_bits(low_seed);
    let low_again = primitive_bits(low_seed);
    assert_eq!(
        low, low_again,
        "public CPU primitives must repeat bit-for-bit"
    );
    assert_eq!(
        low, PINNED_LOW_SEED_BITS,
        "established low-seed CPU output changed"
    );

    let high = primitive_bits(high_seed);
    for (name, (low_bits, high_bits)) in ["value", "perlin", "open_simplex", "worley"]
        .into_iter()
        .zip(low.iter().zip(&high))
    {
        assert_ne!(
            low_bits, high_bits,
            "{name} aliases seeds separated by 2^32"
        );
    }

    let stack_metrics = metrics(24, 20, 240.0, 200.0, 16, 2);
    let low_stack = evaluate_stack(&authored_stack(low_seed), stack_metrics);
    let high_stack = evaluate_stack(&authored_stack(high_seed), stack_metrics);
    assert_ne!(
        low_stack, high_stack,
        "StackEvaluator must observe the upper half of authored u64 seeds"
    );
}

#[test]
fn layer_common_has_no_seed_override() {
    let layer = Layer::new(
        "Noise",
        LayerKind::NoiseValue(NoiseParams {
            seed: 7,
            ..NoiseParams::default()
        }),
    );
    let serialized = serde_json::to_value(layer).expect("serialize representative layer");
    let common = serialized["common"]
        .as_object()
        .expect("serialized LayerCommon object");
    assert!(
        !common.contains_key("seed"),
        "per-kind parameters are the only authored seed authority"
    );
}

#[test]
fn authored_stack_is_repeatable_and_tile_count_independent() {
    const WIDTH: u32 = 270;
    const HEIGHT: u32 = 258;
    const FULL_FIELD_TILE: u32 = WIDTH;
    let stack = authored_stack(0xA17C_0042);
    let reference_metrics = metrics(WIDTH, HEIGHT, 2700.0, 2580.0, FULL_FIELD_TILE, 3);
    let reference = evaluate_stack(&stack, reference_metrics);
    let repeated = evaluate_stack(&stack, reference_metrics);
    assert_field_bits_eq(
        &reference,
        &repeated,
        WIDTH,
        "repeated full-field evaluation",
    );

    for tile_size in [16, 32, 256, FULL_FIELD_TILE] {
        let actual = evaluate_stack(&stack, metrics(WIDTH, HEIGHT, 2700.0, 2580.0, tile_size, 3));
        assert_field_bits_eq(
            &reference,
            &actual,
            WIDTH,
            format_args!("StackEvaluator tile_size={tile_size}"),
        );
    }
}

fn dense_clamped_stencil(
    mut source: Vec<f32>,
    width: u32,
    height: u32,
    iterations: u32,
) -> Vec<u32> {
    for _ in 0..iterations {
        let mut output = vec![0.0; source.len()];
        for j in 0..height {
            for i in 0..width {
                let mut sum = 0.0;
                for dj in -1..=1 {
                    for di in -1..=1 {
                        let sample_i = (i as i32 + di).clamp(0, width as i32 - 1) as u32;
                        let sample_j = (j as i32 + dj).clamp(0, height as i32 - 1) as u32;
                        sum += source[(sample_j * width + sample_i) as usize];
                    }
                }
                output[(j * width + i) as usize] = sum / 9.0;
            }
        }
        source = output;
    }
    source.into_iter().map(f32::to_bits).collect()
}

#[test]
fn batched_tiled_stencil_matches_dense_oracle() {
    const WIDTH: u32 = 70;
    const HEIGHT: u32 = 58;
    const ITERATIONS: u32 = 4;
    let input = patterned_dense(WIDTH, HEIGHT);
    let reference = dense_clamped_stencil(input.clone(), WIDTH, HEIGHT, ITERATIONS);

    for tile_size in [16, 32, 70, 256] {
        for iters_per_batch in [1, 2, 4] {
            let field_metrics = metrics(WIDTH, HEIGHT, WIDTH as f32, HEIGHT as f32, tile_size, 2);
            let mut field = Heightfield::from_dense(field_metrics, &input);
            let seam = map_tiles_batched(
                &mut field,
                ITERATIONS,
                iters_per_batch,
                |tile, lx, lz, _, _| {
                    let mut sum = 0.0;
                    for dj in -1..=1 {
                        for di in -1..=1 {
                            sum += tile.get_with_halo(lx as i32 + di, lz as i32 + dj);
                        }
                    }
                    sum / 9.0
                },
            );
            assert_eq!(seam, 0.0, "tile_size={tile_size}, batch={iters_per_batch}");
            assert_eq!(
                measure_seams(&field),
                0.0,
                "tile_size={tile_size}, batch={iters_per_batch}"
            );
            assert_field_bits_eq(
                &reference,
                &field_bits(&field),
                WIDTH,
                format_args!("dense oracle tile_size={tile_size}, batch={iters_per_batch}"),
            );
        }
    }
}

fn assert_tile_matches_region(
    full: &Heightfield,
    tile: &HeightTile,
    context: impl std::fmt::Display,
) {
    let (origin_x, origin_z) = tile.interior_origin(&full.metrics);
    for lz in 0..tile.interior_height {
        for lx in 0..tile.interior_width {
            assert_eq!(
                full.get(origin_x + lx, origin_z + lz).to_bits(),
                tile.get_interior(lx, lz).to_bits(),
                "{context}: tile {:?} local ({lx}, {lz})",
                tile.id
            );
        }
    }
}

#[test]
fn sculpt_tiles_match_full_field_regions() {
    let field_metrics = metrics(70, 58, 700.0, 580.0, 32, 3);
    let params = sculpt_fixture();
    let full = sculpt_base(field_metrics, &params);
    for tile in full.tiles() {
        let per_tile = sculpt_base_tile(field_metrics, &params, tile.id);
        assert_tile_matches_region(&full, &per_tile, "sculpt");
    }
}

#[test]
fn polygon_tiles_match_full_field_regions() {
    let field_metrics = metrics(70, 58, 700.0, 580.0, 32, 3);
    let input = patterned_heightfield(field_metrics);
    let params = polygon_fixture();
    let full = polygon_height(&input, &params);
    for tile in input.tiles() {
        let per_tile = polygon_height_tile(&input, &params, tile.id)
            .expect("valid polygon tile in the input field");
        assert_tile_matches_region(&full, &per_tile, "polygon");
    }
}

#[test]
fn path_tiles_match_full_field_height_and_wetness_regions() {
    let field_metrics = metrics(70, 58, 700.0, 580.0, 32, 3);
    let input = patterned_heightfield(field_metrics);
    let params = path_fixture(0x1_0000_0042);
    let (full_height, full_wetness) = path_stamp(&input, &params);

    for tile in input.tiles() {
        let per_tile =
            path_stamp_tile(&input, &params, tile.id).expect("valid path tile in the input field");
        assert_tile_matches_region(&full_height, &per_tile.height, "path height");
        let (origin_x, origin_z) = per_tile.height.interior_origin(&field_metrics);
        for lz in 0..per_tile.height.interior_height {
            for lx in 0..per_tile.height.interior_width {
                let local = (lz * per_tile.height.interior_width + lx) as usize;
                assert_eq!(
                    full_wetness.get(origin_x + lx, origin_z + lz).to_bits(),
                    per_tile.wetness[local].to_bits(),
                    "path wetness: tile {:?} local ({lx}, {lz})",
                    tile.id
                );
            }
        }
    }
}

fn halo_mismatches(field: &Heightfield) -> Vec<(TileId, i32, i32, u32, u32)> {
    let mut mismatches = Vec::new();
    let halo = field.metrics.halo as i32;
    for tile in field.tiles() {
        let (origin_x, origin_z) = tile.interior_origin(&field.metrics);
        for gz in -halo..tile.interior_height as i32 + halo {
            for gx in -halo..tile.interior_width as i32 + halo {
                let interior = gx >= 0
                    && gz >= 0
                    && gx < tile.interior_width as i32
                    && gz < tile.interior_height as i32;
                if interior {
                    continue;
                }
                let expected = field
                    .get_clamped(origin_x as i32 + gx, origin_z as i32 + gz)
                    .to_bits();
                let actual = tile.get_with_halo(gx, gz).to_bits();
                if expected != actual {
                    mismatches.push((tile.id, gx, gz, expected, actual));
                }
            }
        }
    }
    mismatches
}

fn assert_halos_match_global_clamped_oracle(field: &Heightfield) {
    let mismatches = halo_mismatches(field);
    if let Some((id, gx, gz, expected, actual)) = mismatches.first() {
        panic!(
            "{} halo mismatches; first at tile {id:?} local ({gx}, {gz}): \
             expected {expected:#010x}, got {actual:#010x}",
            mismatches.len()
        );
    }
}

#[test]
fn halo_sync_matches_global_clamped_oracle_and_detects_stale_reverse_copies() {
    let field_metrics = metrics(70, 58, 70.0, 58.0, 32, 3);
    let mut field = patterned_heightfield(field_metrics);
    let partial_id = TileId { tx: 2, tz: 1 };
    let partial = field.tile(partial_id).expect("partial final tile");
    assert_eq!((partial.interior_width, partial.interior_height), (6, 26));
    assert_halos_match_global_clamped_oracle(&field);

    // This source feeds the left, top, and diagonal neighbors' ghost storage.
    let stale_source = (64, 32);
    field.set(stale_source.0, stale_source.1, 123_456.0);
    assert!(
        !halo_mismatches(&field).is_empty(),
        "the deliberately stale ghost copies must be observable before synchronization"
    );
    assert!(
        measure_seams(&field) > 0.0,
        "the production metric must detect stale reverse edge copies"
    );

    let mut scheduler = TileScheduler::new();
    scheduler.mark_tile(partial_id);
    assert_eq!(scheduler.sync_dirty(&mut field), 0.0);
    assert_eq!(measure_seams(&field), 0.0);
    assert_halos_match_global_clamped_oracle(&field);
}
