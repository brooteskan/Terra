use std::collections::HashMap;

use terra_core::biome_paint::ShapeTransform;
use terra_core::eval::{EvalContext, PreviewQuality, StackEvaluator};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{
    BindingSource, BlendMode, BlurParams, CanyonParams, DomainWarpParams, DuneParams,
    EffectFilterKind, EffectFilterParams, FbmParams, FlatParams, FractalNoiseType,
    HydraulicErosionParams, ImportHeightmapParams, IslandParams, LandscapeEvolutionParams, Layer,
    LayerKind, LayerStack, MesaParams, MountainParams, MultiScaleAmplifyParams, NoiseParams,
    ParamBinding, PathNode, PathParams, PlateauParams, PolygonHeightMode, PolygonHeightParams,
    ProceduralGenerator, ProceduralShapeParams, RampParams, RiverCarveParams, SculptParams,
    SculptPoint, SculptStroke, SculptStrokeKind, SculptStrokeParams, Stamp2dParams,
    StreamPowerParams, TerraceParams, ThermalErosionParams, UpliftParams, VolcanoParams,
    VoronoiParams,
};
use terra_core::mask::{bake_mask_assets, MaskAsset, MaskId, MaskRef, MaskSource};
use terra_gpu::parity::{
    assert_field_parity, ADD_SET_FILTER_PREVIEW, ARCHIPELAGO_PREVIEW, ATOLL_PREVIEW, BLUR_PREVIEW,
    CANYONS_PREVIEW, CURVE_FILTER_PREVIEW, CUTOFF_FILTER_PREVIEW, DEFLATE_FILTER_PREVIEW,
    DENOISE_FILTER_PREVIEW, DOMAIN_WARP_PREVIEW, DUNES_PREVIEW, EFFECT_FILTER_EXACT_PREVIEW,
    EFFECT_FILTER_SPATIAL_PREVIEW, EFFECT_FILTER_WARP_PREVIEW, EXACT_HEIGHT, FBM_PERLIN_PREVIEW,
    FBM_VALUE_PREVIEW, HEIGHTMAP_SAMPLE_PREVIEW, HYDRAULIC_PREVIEW, INFLATE_FILTER_PREVIEW,
    MESA_PREVIEW, MOUNTAINS_PREVIEW, MULTI_SCALE_AMPLIFY_PREVIEW, PATH_HEIGHT_PREVIEW,
    PERLIN_NOISE_PREVIEW, PLATEAU_PREVIEW, POLYGON_HEIGHT_PREVIEW, RIDGED_PERLIN_PREVIEW,
    RIDGED_VALUE_PREVIEW, RIVER_CARVE_D8_PREVIEW, RIVER_CARVE_DINFINITY_PREVIEW,
    SCULPT_STROKES_PREVIEW, SIMPLE_MASK, SMOOTH_FILTER_PREVIEW, STREAM_POWER_D8_PREVIEW,
    STREAM_POWER_DINFINITY_PREVIEW, TERRACE_PREVIEW, THERMAL_PREVIEW, UPLIFT_PREVIEW,
    VALUE_NOISE_PREVIEW, VOLCANIC_ISLAND_PREVIEW, VOLCANO_PREVIEW, VORONOI_REGIONS_PREVIEW,
};
use terra_gpu::GpuTerrainEngine;

const QUALITY: PreviewQuality = PreviewQuality::Draft;

struct TempHeightmap(std::path::PathBuf);

impl TempHeightmap {
    fn new() -> Self {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("terra-heightmap-{unique}.png"));
        let image = image::ImageBuffer::from_fn(7, 5, |x, y| {
            image::Luma([((x * 7000 + y * 9000) % 65536) as u16])
        });
        image.save(&path).expect("write heightmap fixture");
        Self(path)
    }

    fn path(&self) -> String {
        self.0.to_string_lossy().into_owned()
    }
}

impl Drop for TempHeightmap {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn cpu_oracle(
    stack: &LayerStack,
    assets: &[MaskAsset],
    metrics: HeightfieldMetrics,
) -> Heightfield {
    let mut evaluator = StackEvaluator::new();
    let mut context = EvalContext::new(metrics);
    context.quality = QUALITY;
    context.mask_assets = assets.to_vec();
    context.masks = bake_mask_assets(
        assets,
        &Heightfield::zeros(metrics),
        metrics,
        &HashMap::new(),
    );
    evaluator
        .rebuild_all(stack, &mut context)
        .expect("CPU parity oracle")
}

fn gpu_eval(stack: &LayerStack, assets: &[MaskAsset], metrics: HeightfieldMetrics) -> Heightfield {
    let gpu = terra_test_gpu::headless_required();
    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(stack);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            stack,
            assets,
            metrics,
            QUALITY,
            true,
            None,
        )
        .expect("GPU parity evaluation");
    assert!(
        result.fully_gpu,
        "fixture unexpectedly selected CPU fallback"
    );
    assert_eq!(result.resume_cpu_from, None);
    result.cpu.expect("GPU-required readback")
}

fn patterned_sculpt(width: u32, height: u32) -> SculptParams {
    let samples = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                let ramp = x as f32 * 0.7 + y as f32 * 0.35;
                let checker = if (x / 3 + y / 2) % 2 == 0 { 8.0 } else { -3.0 };
                20.0 + ramp + checker
            })
        })
        .collect();
    SculptParams {
        width,
        height,
        samples,
        fill_height: 0.0,
    }
}

#[test]
fn gpu_required_authored_stack_matches_cpu_with_named_tolerance() {
    let metrics = HeightfieldMetrics::new(32, 32, 320.0, 80.0);
    let mask = MaskAsset::new(MaskId::new(), "constant", MaskSource::Constant(0.6));
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "base",
        LayerKind::Flat(FlatParams { height: 10.0 }),
    ));

    let mut ramp = Layer::new(
        "ramp",
        LayerKind::Ramp(RampParams {
            height_min: 0.0,
            height_max: 12.0,
            direction: 0.0,
        }),
    );
    ramp.common.blend = BlendMode::Add;
    ramp.common.opacity = 0.4;
    stack.push(ramp);

    let mut masked = Layer::new("masked add", LayerKind::Flat(FlatParams { height: 5.0 }));
    masked.common.blend = BlendMode::Add;
    let mut mask_ref = MaskRef::new(mask.id);
    mask_ref.strength = 0.75;
    mask_ref.invert = true;
    masked.common.masks.push(mask_ref);
    stack.push(masked);

    let cpu = cpu_oracle(&stack, std::slice::from_ref(&mask), metrics);
    let gpu = gpu_eval(&stack, std::slice::from_ref(&mask), metrics);
    assert_field_parity("stack.flat-ramp-constant-mask", &gpu, &cpu, EXACT_HEIGHT);
}

#[test]
fn gpu_required_import_heightmap_matches_cpu_oracle() {
    let source = TempHeightmap::new();
    let metrics = HeightfieldMetrics::new(31, 19, 310.0, 95.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "import",
        LayerKind::ImportHeightmap(ImportHeightmapParams {
            path: source.path(),
            height_scale: 173.0,
            height_offset: -21.5,
        }),
    ));
    let cpu = cpu_oracle(&stack, &[], metrics);
    let gpu = gpu_eval(&stack, &[], metrics);
    assert_field_parity(
        "asset.heightmap-sample.import",
        &gpu,
        &cpu,
        HEIGHTMAP_SAMPLE_PREVIEW,
    );
}

#[test]
fn gpu_required_transformed_stamp2d_matches_cpu_oracle() {
    let source = TempHeightmap::new();
    let metrics = HeightfieldMetrics::new(37, 23, 370.0, 138.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "base",
        LayerKind::Flat(FlatParams { height: 12.0 }),
    ));
    let mut stamp = Layer::new(
        "stamp",
        LayerKind::Stamp2d(Stamp2dParams {
            heightmap: ImportHeightmapParams {
                path: source.path(),
                height_scale: 91.0,
                height_offset: 4.0,
            },
        }),
    );
    stamp.common.shape_transform = Some(ShapeTransform {
        offset_x: 27.0,
        offset_z: -9.0,
        scale: 0.63,
        rotation_deg: 31.0,
        blend_size: 0.28,
        blend_roundness: 0.42,
    });
    stamp.common.opacity = 0.73;
    stack.push(stamp);
    let cpu = cpu_oracle(&stack, &[], metrics);
    let gpu = gpu_eval(&stack, &[], metrics);
    assert_field_parity(
        "asset.heightmap-sample.stamp2d",
        &gpu,
        &cpu,
        HEIGHTMAP_SAMPLE_PREVIEW,
    );
}

#[test]
fn gpu_required_supported_mask_sources_match_cpu() {
    let metrics = HeightfieldMetrics::new(32, 32, 320.0, 80.0);
    for source in [
        MaskSource::Constant(0.35),
        MaskSource::Height {
            min: 20.0,
            max: 45.0,
        },
        MaskSource::Slope {
            min_deg: 2.0,
            max_deg: 24.0,
        },
    ] {
        let asset = MaskAsset::new(MaskId::new(), "mask", source);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "pattern",
            LayerKind::SculptBase(patterned_sculpt(32, 32)),
        ));
        let mut probe = Layer::new("probe", LayerKind::Flat(FlatParams { height: 1.0 }));
        probe.common.blend = BlendMode::Add;
        let mut mask_ref = MaskRef::new(asset.id);
        mask_ref.strength = 0.65;
        mask_ref.invert = true;
        probe.common.masks.push(mask_ref);
        stack.push(probe);

        let cpu = cpu_oracle(&stack, std::slice::from_ref(&asset), metrics);
        let gpu = gpu_eval(&stack, std::slice::from_ref(&asset), metrics);
        assert_field_parity("mask.simple", &gpu, &cpu, SIMPLE_MASK);
    }
}

#[test]
fn gpu_required_local_filter_stack_matches_cpu() {
    let metrics = HeightfieldMetrics::new(32, 32, 96.0, 64.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "pattern",
        LayerKind::SculptBase(patterned_sculpt(32, 32)),
    ));
    stack.push(Layer::new(
        "blur",
        LayerKind::Blur(BlurParams {
            radius: 2,
            iterations: 1,
        }),
    ));
    let cpu = cpu_oracle(&stack, &[], metrics);
    let gpu = gpu_eval(&stack, &[], metrics);
    assert_field_parity("filter.blur", &gpu, &cpu, BLUR_PREVIEW);

    stack.push(Layer::new(
        "terrace",
        LayerKind::Terrace(TerraceParams {
            levels: 7,
            sharpness: 0.7,
        }),
    ));

    let cpu = cpu_oracle(&stack, &[], metrics);
    let gpu = gpu_eval(&stack, &[], metrics);
    assert_field_parity("stack.sculpt-blur-terrace", &gpu, &cpu, TERRACE_PREVIEW);
}

#[test]
fn gpu_required_noise_and_effect_filter_approximations_are_bounded() {
    let metrics = HeightfieldMetrics::new(24, 24, 120.0, 72.0);

    let mut noise_stack = LayerStack::new();
    noise_stack.push(Layer::new(
        "value noise",
        LayerKind::NoiseValue(NoiseParams {
            seed: 17,
            frequency: 0.035,
            amplitude: 20.0,
            octaves: 3,
            ..NoiseParams::default()
        }),
    ));
    let cpu = cpu_oracle(&noise_stack, &[], metrics);
    let gpu = gpu_eval(&noise_stack, &[], metrics);
    assert_field_parity("noise.value", &gpu, &cpu, VALUE_NOISE_PREVIEW);

    for (name, params, tolerance) in [
        (
            "effect.smooth",
            EffectFilterParams {
                kind: EffectFilterKind::Smooth,
                iterations: 1,
                radius: 2,
                ..EffectFilterParams::default()
            },
            SMOOTH_FILTER_PREVIEW,
        ),
        (
            "effect.inflate",
            EffectFilterParams {
                iterations: 1,
                ..EffectFilterParams::inflate()
            },
            INFLATE_FILTER_PREVIEW,
        ),
        (
            "effect.denoise",
            EffectFilterParams {
                kind: EffectFilterKind::Denoise,
                iterations: 1,
                radius: 2,
                ..EffectFilterParams::default()
            },
            DENOISE_FILTER_PREVIEW,
        ),
        (
            "effect.add-set",
            EffectFilterParams::add_set(),
            ADD_SET_FILTER_PREVIEW,
        ),
        (
            "effect.add-set.absolute",
            EffectFilterParams {
                sea_level: 1.0,
                amount: 17.25,
                ..EffectFilterParams::add_set()
            },
            ADD_SET_FILTER_PREVIEW,
        ),
        (
            "effect.deflate",
            EffectFilterParams::deflate(),
            DEFLATE_FILTER_PREVIEW,
        ),
        (
            "effect.curve",
            EffectFilterParams::curve(),
            CURVE_FILTER_PREVIEW,
        ),
        (
            "effect.cutoff",
            EffectFilterParams::cutoff(),
            CUTOFF_FILTER_PREVIEW,
        ),
        (
            "effect.terrace-simple",
            EffectFilterParams::terrace_simple(),
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.shore",
            EffectFilterParams::shore(),
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.blocks",
            EffectFilterParams::blocks(),
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.zero-edge",
            EffectFilterParams::zero_edge(),
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.squeeze",
            EffectFilterParams::squeeze(),
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.directional-blur",
            EffectFilterParams::directional_blur(),
            EFFECT_FILTER_SPATIAL_PREVIEW,
        ),
        (
            "effect.angle-blur",
            EffectFilterParams::angle_blur(),
            EFFECT_FILTER_SPATIAL_PREVIEW,
        ),
        (
            "effect.swirl",
            EffectFilterParams::swirl(),
            EFFECT_FILTER_WARP_PREVIEW,
        ),
        (
            "effect.crater",
            EffectFilterParams::crater(),
            EFFECT_FILTER_SPATIAL_PREVIEW,
        ),
        (
            "effect.distortion",
            EffectFilterParams::distortion(),
            EFFECT_FILTER_WARP_PREVIEW,
        ),
        (
            "effect.balloon",
            EffectFilterParams::balloon(),
            EFFECT_FILTER_SPATIAL_PREVIEW,
        ),
        (
            "effect.noise-perlin",
            EffectFilterParams::noise_perlin(),
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.noise-value",
            EffectFilterParams::noise_value(),
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.noise-white",
            EffectFilterParams::noise_white(),
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.noise-wave",
            EffectFilterParams::noise_wave(),
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.scatter-detail",
            EffectFilterParams::scatter_detail(),
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.spike-removal",
            EffectFilterParams::spike_removal(),
            EFFECT_FILTER_SPATIAL_PREVIEW,
        ),
        (
            "effect.noise-billow",
            EffectFilterParams::noise_billow(),
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.noise-ridged",
            EffectFilterParams::noise_ridged(),
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.ridged",
            EffectFilterParams::ridged(),
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.rugged",
            EffectFilterParams::rugged(),
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.border-blend.absolute",
            EffectFilterParams {
                sea_level: 11.0,
                ..EffectFilterParams::border_blend()
            },
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.flatten.absolute",
            EffectFilterParams {
                sea_level: 11.0,
                ..EffectFilterParams::flatten_filter()
            },
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.hexagons",
            EffectFilterParams::hexagons(),
            EFFECT_FILTER_EXACT_PREVIEW,
        ),
        (
            "effect.terrace-steep",
            EffectFilterParams::terrace_steep(),
            EFFECT_FILTER_SPATIAL_PREVIEW,
        ),
    ] {
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "pattern",
            LayerKind::SculptBase(patterned_sculpt(24, 24)),
        ));
        stack.push(Layer::new(name, LayerKind::EffectFilter(params)));
        let cpu = cpu_oracle(&stack, &[], metrics);
        let gpu = gpu_eval(&stack, &[], metrics);
        assert_field_parity(name, &gpu, &cpu, tolerance);
    }
}

#[test]
fn gpu_required_noise_family_previews_are_bounded() {
    let metrics = HeightfieldMetrics::new(31, 23, 155.0, 92.0);
    let base = NoiseParams {
        seed: 17,
        frequency: 0.035,
        amplitude: 20.0,
        octaves: 4,
        lacunarity: 2.15,
        persistence: 0.43,
        offset_x: 7.25,
        offset_z: -3.75,
        remap_min: -0.72,
        remap_max: 0.81,
    };
    let cases = [
        (
            "noise.perlin",
            LayerKind::NoisePerlin(NoiseParams {
                octaves: 1,
                ..base.clone()
            }),
            PERLIN_NOISE_PREVIEW,
        ),
        (
            "noise.fbm.value",
            LayerKind::Fbm(FbmParams {
                base: base.clone(),
                noise: FractalNoiseType::Value,
            }),
            FBM_VALUE_PREVIEW,
        ),
        (
            "noise.perlin.fractal",
            LayerKind::NoisePerlin(base.clone()),
            PERLIN_NOISE_PREVIEW,
        ),
        (
            "noise.fbm.perlin",
            LayerKind::Fbm(FbmParams {
                base: base.clone(),
                noise: FractalNoiseType::Perlin,
            }),
            FBM_PERLIN_PREVIEW,
        ),
        (
            "noise.ridged.value",
            LayerKind::Ridged(FbmParams {
                base: base.clone(),
                noise: FractalNoiseType::Value,
            }),
            RIDGED_VALUE_PREVIEW,
        ),
        (
            "noise.ridged.perlin",
            LayerKind::Ridged(FbmParams {
                base: base.clone(),
                noise: FractalNoiseType::Perlin,
            }),
            RIDGED_PERLIN_PREVIEW,
        ),
        (
            "noise.domain-warp",
            LayerKind::DomainWarp(DomainWarpParams {
                base: base.clone(),
                warp_strength: 18.0,
                warp_frequency: 0.012,
            }),
            DOMAIN_WARP_PREVIEW,
        ),
        (
            "noise.voronoi-regions",
            LayerKind::VoronoiRegions(VoronoiParams {
                base: base.clone(),
                cell_jitter: 0.63,
                height_per_cell: 37.0,
            }),
            VORONOI_REGIONS_PREVIEW,
        ),
    ];

    for (name, kind, tolerance) in cases {
        let ridged_amplitude = match &kind {
            LayerKind::Ridged(p) => Some(p.base.amplitude),
            _ => None,
        };
        let mut stack = LayerStack::new();
        stack.push(Layer::new(name, kind));
        let cpu = cpu_oracle(&stack, &[], metrics);
        let gpu = gpu_eval(&stack, &[], metrics);
        assert_field_parity(name, &gpu, &cpu, tolerance);
        if let Some(amplitude) = ridged_amplitude {
            assert!(
                gpu.to_dense()
                    .iter()
                    .all(|&height| height >= -1.0e-4 && height <= amplitude + 1.0e-4),
                "{name} escaped [0, amplitude]"
            );
        }
    }

    let domain = DomainWarpParams {
        base,
        warp_strength: 18.0,
        warp_frequency: 0.012,
    };
    let mut warped_stack = LayerStack::new();
    warped_stack.push(Layer::new("warped", LayerKind::DomainWarp(domain.clone())));
    let mut unwarped_stack = LayerStack::new();
    unwarped_stack.push(Layer::new(
        "unwarped",
        LayerKind::DomainWarp(DomainWarpParams {
            warp_strength: 0.0,
            ..domain
        }),
    ));
    let warped = gpu_eval(&warped_stack, &[], metrics).to_dense();
    let unwarped = gpu_eval(&unwarped_stack, &[], metrics).to_dense();
    let warp_effect = terra_gpu::parity::max_abs_diff(&warped, &unwarped);
    assert!(
        warp_effect > 1.0,
        "DomainWarp parameters had no material effect"
    );

    let voronoi = VoronoiParams {
        base: NoiseParams {
            seed: 29,
            frequency: 0.047,
            amplitude: 31.0,
            offset_x: -8.5,
            offset_z: 5.75,
            remap_min: -0.35,
            remap_max: 0.92,
            ..NoiseParams::default()
        },
        cell_jitter: 0.72,
        height_per_cell: 41.0,
    };
    let mut cellular_stack = LayerStack::new();
    cellular_stack.push(Layer::new(
        "cellular",
        LayerKind::VoronoiRegions(voronoi.clone()),
    ));
    let mut worley_only_stack = LayerStack::new();
    worley_only_stack.push(Layer::new(
        "worley only",
        LayerKind::VoronoiRegions(VoronoiParams {
            cell_jitter: 0.0,
            ..voronoi
        }),
    ));
    let cellular = gpu_eval(&cellular_stack, &[], metrics).to_dense();
    let worley_only = gpu_eval(&worley_only_stack, &[], metrics).to_dense();
    let cellular_effect = terra_gpu::parity::max_abs_diff(&cellular, &worley_only);
    assert!(
        cellular_effect > 1.0,
        "VoronoiRegions cell_jitter/height_per_cell term had no material effect"
    );
}

/// Pull the shipped "Shelf Flatten" params straight out of the Tropical Island
/// palette so this GPU test tracks whatever config actually ships — mirroring the
/// CPU-side helper in `terra-core/tests/shelf_flatten_bathymetry.rs`. Fails loudly
/// if the layer is renamed or is no longer an EffectFilter.
fn shipped_shelf_flatten() -> EffectFilterParams {
    let lib = terra_core::BiomeLibrary::tropical_island_palette();
    for def in &lib.definitions {
        for (name, kind) in &def.terrain_layers {
            if name == "Shelf Flatten" {
                match kind {
                    LayerKind::EffectFilter(p) => return p.clone(),
                    _ => panic!("'Shelf Flatten' is no longer an EffectFilter layer"),
                }
            }
        }
    }
    panic!("Tropical Island palette no longer has a 'Shelf Flatten' terrain layer");
}

#[test]
fn gpu_required_denoise_preserves_shelf_basin_discontinuity() {
    // #119: the shipped reef "Shelf Flatten" (Denoise/bilateral) must now preview on
    // the GPU without bleeding the shallow shelf into the adjacent deep basin — the
    // divergence class #95 hardened the CPU export against, guarded here in preview
    // too. The fixture mirrors terra-core's shelf_flatten_bathymetry: a rippled
    // ~-7 m shelf abutting a flat ~-218 m basin across a ~211 m step, filtered at the
    // config pulled live from the shipped palette.
    const RES: u32 = 64;
    const SHELF_LAST_COL: u32 = 31;
    const SHELF_DEPTH: f32 = -7.0;
    const BASIN_DEPTH: f32 = -218.0;
    const RIPPLE: f32 = 2.0;

    let metrics = HeightfieldMetrics::new(RES, RES, RES as f32 * 10.0, RES as f32 * 10.0);
    let samples: Vec<f32> = (0..RES)
        .flat_map(|j| {
            (0..RES).map(move |i| {
                if i <= SHELF_LAST_COL {
                    // Deterministic +/-2 m checkerboard: real high-frequency detail
                    // for the denoise to attenuate on the shelf itself.
                    let ripple = if (i + j) % 2 == 0 { RIPPLE } else { -RIPPLE };
                    SHELF_DEPTH + ripple
                } else {
                    BASIN_DEPTH
                }
            })
        })
        .collect();

    let params = shipped_shelf_flatten();
    // Premise guard: the shipped layer is the bilateral Denoise this test exercises.
    assert_eq!(params.kind, EffectFilterKind::Denoise);

    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams {
            width: RES,
            height: RES,
            samples,
            fill_height: 0.0,
        }),
    ));
    stack.push(Layer::new("Shelf Flatten", LayerKind::EffectFilter(params)));

    let cpu = cpu_oracle(&stack, &[], metrics);
    // gpu_eval asserts fully_gpu: the stack compiles a GPU plan with no CPU fallback
    // from the Denoise layer (acceptance criterion 1).
    let gpu = gpu_eval(&stack, &[], metrics);
    assert_field_parity(
        "effect.denoise.shelf-basin",
        &gpu,
        &cpu,
        DENOISE_FILTER_PREVIEW,
    );

    // Headline, asserted directly on the GPU output so it survives even if the parity
    // tolerance is later loosened: every basin cell stays within 5 m of the true
    // basin depth — the same bound terra-core's CPU shelf test uses. A box blur lifts
    // the first basin columns by tens of metres; the edge-aware bilateral must not.
    let mut worst = 0.0f32;
    let mut worst_at = (0u32, 0u32);
    for j in 0..RES {
        for i in (SHELF_LAST_COL + 1)..RES {
            let dev = (gpu.get(i, j) - BASIN_DEPTH).abs();
            if dev > worst {
                worst = dev;
                worst_at = (i, j);
            }
        }
    }
    assert!(
        worst <= 5.0,
        "GPU Denoise bled the shelf into the basin: {worst:.1} m at {worst_at:?}"
    );
}

#[test]
fn gpu_required_simulation_previews_have_bounded_full_field_error() {
    let metrics = HeightfieldMetrics::new(24, 24, 48.0, 36.0);
    for (name, kind, tolerance) in [
        (
            "thermal",
            LayerKind::ThermalErosion(ThermalErosionParams {
                iterations: 2,
                layered_materials: false,
                weathering_rate: 0.0,
                ..ThermalErosionParams::default()
            }),
            THERMAL_PREVIEW,
        ),
        (
            "hydraulic",
            LayerKind::HydraulicErosion(HydraulicErosionParams {
                iterations: 2,
                particle_density: 0.0,
                layered_materials: false,
                ..HydraulicErosionParams::default()
            }),
            HYDRAULIC_PREVIEW,
        ),
    ] {
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "pattern",
            LayerKind::SculptBase(patterned_sculpt(24, 24)),
        ));
        stack.push(Layer::new(name, kind));
        let cpu = cpu_oracle(&stack, &[], metrics);
        let gpu = gpu_eval(&stack, &[], metrics);
        assert_field_parity(name, &gpu, &cpu, tolerance);
    }
}

#[test]
fn gpu_required_river_carve_d8_and_dinfinity_previews_are_bounded() {
    const RES: u32 = 12;
    let metrics = HeightfieldMetrics::new(RES, RES, 120.0, 120.0);
    let center = (RES as f32 - 1.0) * 0.5;
    let samples: Vec<f32> = (0..RES)
        .flat_map(|y| {
            (0..RES).map(move |x| {
                // An open, monotone V-shaped drainage basin avoids depression-fill
                // ambiguity while exercising channel convergence and overlapping banks.
                180.0 - y as f32 * 3.0 + (x as f32 - center).abs() * 1.5 + x as f32 * 0.01
            })
        })
        .collect();

    for use_dinfinity in [false, true] {
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "drainage basin",
            LayerKind::SculptBase(SculptParams {
                width: RES,
                height: RES,
                samples: samples.clone(),
                fill_height: 0.0,
            }),
        ));
        stack.push(Layer::new(
            "river carve",
            LayerKind::RiverCarve(RiverCarveParams {
                accumulation_threshold: 3.0,
                depth: 2.0,
                width: 1.5,
                bank_smooth: 0.4,
                use_dinfinity,
                ..RiverCarveParams::default()
            }),
        ));
        let cpu = cpu_oracle(&stack, &[], metrics);
        let gpu = gpu_eval(&stack, &[], metrics);
        let (contract, tolerance) = if use_dinfinity {
            (
                "simulation.river-carve.d-infinity",
                RIVER_CARVE_DINFINITY_PREVIEW,
            )
        } else {
            ("simulation.river-carve.d8", RIVER_CARVE_D8_PREVIEW)
        };
        assert_field_parity(contract, &gpu, &cpu, tolerance);
    }
}

#[test]
fn gpu_required_stream_power_d8_and_dinfinity_previews_are_bounded() {
    const RES: u32 = 12;
    let metrics = HeightfieldMetrics::new(RES, RES, 120.0, 120.0);
    let center = (RES as f32 - 1.0) * 0.5;
    let samples: Vec<f32> = (0..RES)
        .flat_map(|y| {
            (0..RES).map(move |x| {
                180.0 - y as f32 * 3.0 + (x as f32 - center).abs() * 1.5 + x as f32 * 0.01
            })
        })
        .collect();

    for use_dinfinity in [false, true] {
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "drainage basin",
            LayerKind::SculptBase(SculptParams {
                width: RES,
                height: RES,
                samples: samples.clone(),
                fill_height: 0.0,
            }),
        ));
        stack.push(Layer::new(
            "stream power",
            LayerKind::StreamPowerErosion(StreamPowerParams {
                iterations: 3,
                k: 0.002,
                m: 0.5,
                n: 1.0,
                dt: 0.75,
                uplift_rate: 0.05,
                base_level: 100.0,
                hardness: 0.2,
                use_dinfinity,
                ..StreamPowerParams::default()
            }),
        ));
        let cpu = cpu_oracle(&stack, &[], metrics);
        let gpu = gpu_eval(&stack, &[], metrics);
        let (contract, tolerance) = if use_dinfinity {
            (
                "simulation.stream-power.d-infinity",
                STREAM_POWER_DINFINITY_PREVIEW,
            )
        } else {
            ("simulation.stream-power.d8", STREAM_POWER_D8_PREVIEW)
        };
        assert_field_parity(contract, &gpu, &cpu, tolerance);
        assert!(
            gpu.to_dense().iter().all(|height| height.is_finite()),
            "{contract} produced a non-finite height"
        );
        assert!(
            gpu.to_dense().iter().all(|height| *height >= 100.0),
            "{contract} crossed the authored base level"
        );
        assert!(
            gpu.to_dense()
                .iter()
                .zip(&samples)
                .any(|(after, before)| after < before),
            "{contract} must incise at least one texel"
        );
    }
}

#[test]
fn gpu_required_multi_scale_amplify_preview_is_bounded_across_two_levels() {
    const RES: u32 = 128;
    let metrics = HeightfieldMetrics::new(RES, RES, 512.0, 384.0);
    let center = (RES as f32 - 1.0) * 0.5;
    let samples: Vec<f32> = (0..RES)
        .flat_map(|y| {
            (0..RES).map(move |x| {
                let basin = 220.0 - y as f32 * 0.45 + (x as f32 - center).abs() * 0.22;
                let ripple = ((x * 7 + y * 3) % 11) as f32 * 0.15;
                basin + ripple
            })
        })
        .collect();
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "open basin",
        LayerKind::SculptBase(SculptParams {
            width: RES,
            height: RES,
            samples,
            fill_height: 0.0,
        }),
    ));
    stack.push(Layer::new(
        "multi scale",
        LayerKind::MultiScaleAmplify(MultiScaleAmplifyParams {
            level_count: 2,
            thermal_iters: 2,
            spe_iters: 1,
            hardness: 0.2,
            ..MultiScaleAmplifyParams::default()
        }),
    ));

    let cpu = cpu_oracle(&stack, &[], metrics);
    let gpu = gpu_eval(&stack, &[], metrics);
    assert_field_parity(
        "simulation.multi-scale-amplify",
        &gpu,
        &cpu,
        MULTI_SCALE_AMPLIFY_PREVIEW,
    );
    assert!(gpu.to_dense().iter().all(|height| height.is_finite()));
}

#[test]
fn gpu_required_volcanic_island_approximation_is_bounded() {
    let metrics = HeightfieldMetrics::new(32, 32, 640.0, 480.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "volcanic island",
        LayerKind::Island(IslandParams::default()),
    ));
    let cpu = cpu_oracle(&stack, &[], metrics);
    let gpu = gpu_eval(&stack, &[], metrics);
    assert_field_parity("island.volcanic-high", &gpu, &cpu, VOLCANIC_ISLAND_PREVIEW);
}

#[test]
fn gpu_required_shape_family_previews_are_bounded() {
    let metrics = HeightfieldMetrics::new(32, 28, 720.0, 510.0);
    for (contract, kind, tolerance) in [
        (
            "shape.mountains",
            LayerKind::Mountains(MountainParams::default()),
            MOUNTAINS_PREVIEW,
        ),
        (
            "shape.dunes",
            LayerKind::Dunes(DuneParams::default()),
            DUNES_PREVIEW,
        ),
        (
            "shape.canyons",
            LayerKind::Canyons(CanyonParams::default()),
            CANYONS_PREVIEW,
        ),
        (
            "shape.mesa",
            LayerKind::Mesa(MesaParams::default()),
            MESA_PREVIEW,
        ),
        (
            "shape.volcano",
            LayerKind::Volcano(VolcanoParams::default()),
            VOLCANO_PREVIEW,
        ),
        (
            "shape.uplift",
            LayerKind::Uplift(UpliftParams::default()),
            UPLIFT_PREVIEW,
        ),
        (
            "island.archipelago",
            LayerKind::Island(IslandParams::archipelago()),
            ARCHIPELAGO_PREVIEW,
        ),
        (
            "island.atoll",
            LayerKind::Island(IslandParams::atoll()),
            ATOLL_PREVIEW,
        ),
    ] {
        let mut stack = LayerStack::new();
        stack.push(Layer::new(contract, kind));
        let cpu = cpu_oracle(&stack, &[], metrics);
        let gpu = gpu_eval(&stack, &[], metrics);
        assert_field_parity(contract, &gpu, &cpu, tolerance);
    }

    let mut plateau_stack = LayerStack::new();
    plateau_stack.push(Layer::new(
        "plateau input",
        LayerKind::SculptBase(patterned_sculpt(metrics.width, metrics.height)),
    ));
    plateau_stack.push(Layer::new(
        "shape.plateau",
        LayerKind::Plateau(PlateauParams::default()),
    ));
    let cpu = cpu_oracle(&plateau_stack, &[], metrics);
    let gpu = gpu_eval(&plateau_stack, &[], metrics);
    assert_field_parity("shape.plateau", &gpu, &cpu, PLATEAU_PREVIEW);
}

fn pt(u: f32, v: f32, pressure: f32) -> SculptPoint {
    SculptPoint { u, v, pressure }
}

/// Stroke set exercising every GPU-supported kind (per-sample maps + distance
/// stamps + an alias + an aux-only kind + the base-neighborhood Smooth, Pinch, and
/// Coastline + the footprint-mean Flatten), with multi-point polylines, varied
/// pressure, and a single-point stroke.
fn supported_stroke_set(reconcile: f32) -> SculptStrokeParams {
    let stroke = |kind, points, radius_m, strength, target_height| SculptStroke {
        kind,
        points,
        radius_m,
        strength,
        target_height,
        falloff: 1.5,
        enabled: true,
    };
    SculptStrokeParams {
        strokes: vec![
            stroke(
                SculptStrokeKind::Raise,
                vec![pt(0.25, 0.3, 1.0), pt(0.45, 0.4, 0.7)],
                70.0,
                10.0,
                0.0,
            ),
            stroke(
                SculptStrokeKind::Lower,
                vec![pt(0.7, 0.65, 1.0)],
                60.0,
                6.0,
                0.0,
            ),
            stroke(
                SculptStrokeKind::Ridge,
                vec![pt(0.2, 0.7, 1.0), pt(0.4, 0.75, 1.0), pt(0.55, 0.6, 0.5)],
                55.0,
                8.0,
                0.0,
            ),
            stroke(
                SculptStrokeKind::Valley,
                vec![pt(0.6, 0.2, 0.9), pt(0.8, 0.35, 1.0)],
                50.0,
                7.0,
                0.0,
            ),
            stroke(
                SculptStrokeKind::Inflate,
                vec![pt(0.5, 0.5, 1.0)],
                65.0,
                5.0,
                0.0,
            ),
            stroke(
                SculptStrokeKind::Terrace,
                vec![pt(0.35, 0.5, 1.0)],
                80.0,
                5.0,
                0.0,
            ),
            stroke(
                SculptStrokeKind::Noise,
                vec![pt(0.5, 0.8, 1.0)],
                60.0,
                4.0,
                0.0,
            ),
            stroke(
                SculptStrokeKind::HeightStamp,
                vec![pt(0.8, 0.8, 1.0)],
                45.0,
                6.0,
                30.0,
            ),
            stroke(
                SculptStrokeKind::PlateauStamp,
                vec![pt(0.2, 0.2, 1.0)],
                50.0,
                6.0,
                25.0,
            ),
            stroke(
                SculptStrokeKind::CraterStamp,
                vec![pt(0.65, 0.5, 1.0)],
                55.0,
                9.0,
                0.0,
            ),
            // Alias of Ridge; aux-only kind exercises the reconcile edit weight.
            stroke(
                SculptStrokeKind::MountainStamp,
                vec![pt(0.15, 0.5, 1.0)],
                45.0,
                7.0,
                0.0,
            ),
            stroke(
                SculptStrokeKind::Uplift,
                vec![pt(0.5, 0.15, 1.0)],
                50.0,
                3.0,
                0.0,
            ),
            // Smooth pulls each sample toward the 3x3 mean of the *layer input*.
            // Placed last and routed across the Raise footprint (0.3, 0.35) so a GPU
            // that wrongly averaged the running (raised) height instead of `base`
            // would diverge here; the (0.05, 0.05) endpoint drives the brush onto the
            // border texels, exercising the clamped-edge taps.
            stroke(
                SculptStrokeKind::Smooth,
                vec![pt(0.05, 0.05, 1.0), pt(0.3, 0.35, 1.0)],
                60.0,
                5.0,
                0.0,
            ),
            // Pinch is Smooth's base-3x3 pull at a 1.25 overdrive. Routed across the
            // Ridge crest (0.4, 0.75), where the running height sits well above `base`,
            // so a GPU that averaged the running (ridged) height instead of `src`, or
            // dropped the 1.25 gain, diverges by metres there — far beyond tolerance.
            // The (0.6, 0.95) endpoint drives the brush onto the bottom border for the
            // clamped-edge taps, an edge the Smooth stroke does not visit.
            stroke(
                SculptStrokeKind::Pinch,
                vec![pt(0.6, 0.95, 1.0), pt(0.4, 0.75, 1.0)],
                60.0,
                5.0,
                0.0,
            ),
            // Coastline lowers the sample by `abs(strength * w) * 0.25`, blends it 0.55
            // toward the clamped 3x3 mean of `src`, then gates the whole thing by `w`
            // (`h * (1 - w)` at the falloff edge). Unlike Smooth/Pinch it uses *both*
            // weights — `s = strength * w` in the lowering term and `w` in the gate — so
            // it catches an s-vs-w swap as well as a running-vs-`src` average. Routed
            // across the CraterStamp bowl (0.65, 0.55), where the running height sits
            // well below `base`, and out to the (0.95, 0.45) right border — a clamped
            // edge neither Smooth (top-left) nor Pinch (bottom) visits.
            stroke(
                SculptStrokeKind::Coastline,
                vec![pt(0.95, 0.45, 1.0), pt(0.65, 0.55, 1.0)],
                60.0,
                6.0,
                0.0,
            ),
            // Flatten settles the footprint toward the brush-weighted mean of the
            // *running* field — the height every prior stroke has already written,
            // not the layer input. Routed across the CraterStamp bowl (0.65, 0.5,
            // running well below base) and the Ridge crest (0.4, 0.75, running well
            // above base): the running mean there differs from the base mean by
            // metres, so a GPU that reduced over `src` instead of the running stamp,
            // or skipped the reduction, diverges far beyond tolerance. `target_height`
            // is deliberately absurd (900) — Flatten ignores it whenever the footprint
            // carries weight, and it must never leak into the presentation range.
            stroke(
                SculptStrokeKind::Flatten,
                vec![pt(0.4, 0.75, 1.0), pt(0.65, 0.5, 1.0)],
                70.0,
                4.0,
                900.0,
            ),
            // A per-sample stroke *after* the first Flatten: exercises the stamp
            // segment that runs once the reduction has settled its target, and its
            // footprint overlaps the flattened band so stroke ordering is observable.
            stroke(
                SculptStrokeKind::Lower,
                vec![pt(0.5, 0.6, 1.0)],
                45.0,
                5.0,
                0.0,
            ),
            // A second Flatten forces multi-segment sequencing: its footprint mean is
            // measured against a running field that already carries the first Flatten
            // *and* the Lower above, so a GPU that reused a stale reduction or reduced
            // against the wrong segment diverges. Spanning (0.5, 0.55) sits over both.
            stroke(
                SculptStrokeKind::Flatten,
                vec![pt(0.45, 0.55, 1.0), pt(0.6, 0.5, 0.8)],
                60.0,
                4.0,
                900.0,
            ),
        ],
        reconcile,
    }
}

fn sculpt_strokes_stack(reconcile: f32) -> (LayerStack, HeightfieldMetrics) {
    let metrics = HeightfieldMetrics::new(48, 48, 480.0, 480.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "base",
        LayerKind::SculptBase(patterned_sculpt(48, 48)),
    ));
    stack.push(Layer::new(
        "strokes",
        LayerKind::SculptStrokes(supported_stroke_set(reconcile)),
    ));
    (stack, metrics)
}

#[test]
fn gpu_required_sculpt_strokes_match_cpu_with_and_without_reconcile() {
    for reconcile in [0.2, 0.0] {
        let (stack, metrics) = sculpt_strokes_stack(reconcile);
        let cpu = cpu_oracle(&stack, &[], metrics);
        let gpu = gpu_eval(&stack, &[], metrics);
        assert_field_parity(
            "authoring.sculpt-strokes",
            &gpu,
            &cpu,
            SCULPT_STROKES_PREVIEW,
        );
    }
}

#[test]
fn gpu_required_sculpt_strokes_match_cpu_under_add_blend_and_mask() {
    // The stamped field flows through the standard blend, so a non-default outer
    // composite (Add + opacity + a Constant mask) must still match the CPU.
    let metrics = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
    let mask = MaskAsset::new(MaskId::new(), "constant", MaskSource::Constant(0.6));
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "base",
        LayerKind::SculptBase(patterned_sculpt(32, 32)),
    ));
    let mut strokes = Layer::new(
        "strokes",
        LayerKind::SculptStrokes(supported_stroke_set(0.15)),
    );
    strokes.common.blend = BlendMode::Add;
    strokes.common.opacity = 0.7;
    let mut mask_ref = MaskRef::new(mask.id);
    mask_ref.strength = 0.8;
    strokes.common.masks.push(mask_ref);
    stack.push(strokes);

    let cpu = cpu_oracle(&stack, std::slice::from_ref(&mask), metrics);
    let gpu = gpu_eval(&stack, std::slice::from_ref(&mask), metrics);
    assert_field_parity(
        "authoring.sculpt-strokes-blended",
        &gpu,
        &cpu,
        SCULPT_STROKES_PREVIEW,
    );
}

#[test]
fn sculpt_strokes_fall_back_when_a_downstream_layer_consumes_aux() {
    // A stroke layer sitting under a layer that consumes its per-texel aux must
    // report a CPU-resume boundary at the stroke layer rather than compiling fully
    // GPU (the GPU preview drops that aux). This holds for a Flatten stroke as much
    // as for the per-sample kinds — Flatten previews on the GPU (#117), but the aux
    // gate still demotes it when a downstream layer reads what the preview dropped.
    let gpu = terra_test_gpu::headless_required();
    let metrics = HeightfieldMetrics::new(24, 24, 240.0, 240.0);

    let mut flatten_stack = LayerStack::new();
    flatten_stack.push(Layer::new(
        "base",
        LayerKind::SculptBase(patterned_sculpt(24, 24)),
    ));
    flatten_stack.push(Layer::new(
        "flatten",
        LayerKind::SculptStrokes(SculptStrokeParams {
            strokes: vec![SculptStroke {
                kind: SculptStrokeKind::Flatten,
                points: vec![pt(0.5, 0.5, 1.0)],
                radius_m: 80.0,
                strength: 4.0,
                target_height: 0.0,
                falloff: 1.5,
                enabled: true,
            }],
            reconcile: 0.15,
        }),
    ));
    flatten_stack.push(Layer::new(
        "evolve",
        LayerKind::LandscapeEvolution(LandscapeEvolutionParams::default()),
    ));

    let mut consumer_stack = LayerStack::new();
    consumer_stack.push(Layer::new(
        "base",
        LayerKind::SculptBase(patterned_sculpt(24, 24)),
    ));
    consumer_stack.push(Layer::new(
        "strokes",
        LayerKind::SculptStrokes(supported_stroke_set(0.15)),
    ));
    consumer_stack.push(Layer::new(
        "evolve",
        LayerKind::LandscapeEvolution(LandscapeEvolutionParams::default()),
    ));

    for (name, stack, resume) in [
        ("flatten", flatten_stack, 1usize),
        ("aux-consumer", consumer_stack, 1usize),
    ] {
        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        engine.mark_all_dirty(&stack);
        let result = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                QUALITY,
                false,
                None,
            )
            .expect("gpu eval");
        assert!(!result.fully_gpu, "{name} must not be fully GPU");
        assert_eq!(result.resume_cpu_from, Some(resume), "{name}");
    }
}

#[test]
fn gpu_required_hybrid_checkpoint_applies_suffix_once() {
    let gpu = terra_test_gpu::headless_required();
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 80.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "base",
        LayerKind::Flat(FlatParams { height: 10.0 }),
    ));
    let mut unsupported = Layer::new(
        "CPU-bound opacity binding",
        LayerKind::Flat(FlatParams { height: 10.0 }),
    );
    unsupported
        .common
        .param_bindings
        .push(ParamBinding::new("opacity", BindingSource::Constant(0.5)));
    stack.push(unsupported);
    let mut downstream = Layer::new("downstream", LayerKind::Flat(FlatParams { height: 2.0 }));
    downstream.common.blend = BlendMode::Add;
    stack.push(downstream);

    let expected = cpu_oracle(&stack, &[], metrics);
    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            QUALITY,
            true,
            None,
        )
        .expect("hybrid checkpoint");
    assert_eq!(result.resume_cpu_from, Some(1));
    let checkpoint = result.cpu.expect("height entering layer one");

    let mut evaluator = StackEvaluator::new();
    let mut context = EvalContext::new(metrics);
    context.quality = QUALITY;
    let completed = evaluator
        .evaluate_suffix(&stack, &mut context, 1, checkpoint)
        .expect("CPU suffix");
    assert_field_parity(
        "hybrid.non-idempotent-suffix",
        &completed,
        &expected,
        EXACT_HEIGHT,
    );
}

#[test]
fn gpu_required_path_and_polygon_height_match_cpu() {
    let metrics = HeightfieldMetrics::new(29, 21, 290.0, 84.0);
    let base = Layer::new(
        "pattern",
        LayerKind::SculptBase(patterned_sculpt(metrics.width, metrics.height)),
    );

    for carve in [false, true] {
        let mut stack = LayerStack::new();
        stack.push(base.clone());
        stack.push(Layer::new(
            if carve { "carved path" } else { "raised path" },
            LayerKind::Path(PathParams {
                nodes: vec![
                    PathNode {
                        u: 0.08,
                        v: 0.2,
                        height: 1.0,
                        width: 0.8,
                    },
                    PathNode {
                        u: 0.35,
                        v: 0.75,
                        height: 4.0,
                        width: 1.2,
                    },
                    PathNode {
                        u: 0.7,
                        v: 0.35,
                        height: -2.0,
                        width: 0.7,
                    },
                    PathNode {
                        u: 0.94,
                        v: 0.8,
                        height: 2.0,
                        width: 1.0,
                    },
                ],
                width: 12.0,
                falloff: 7.0,
                noise_strength: 1.5,
                noise_scale: 0.025,
                height_offset: 6.0,
                carve,
                seed: 19,
                spline: true,
                profile: 1.6,
                closed: false,
            }),
        ));
        let cpu = cpu_oracle(&stack, &[], metrics);
        let gpu = gpu_eval(&stack, &[], metrics);
        assert_field_parity("authoring.path-height", &gpu, &cpu, PATH_HEIGHT_PREVIEW);
    }

    for (mode, carve) in [
        (PolygonHeightMode::RaiseBy, false),
        (PolygonHeightMode::RaiseBy, true),
        (PolygonHeightMode::SetElevation, false),
        (PolygonHeightMode::SetElevation, true),
    ] {
        let mut stack = LayerStack::new();
        stack.push(base.clone());
        stack.push(Layer::new(
            "concave polygon",
            LayerKind::PolygonHeight(PolygonHeightParams {
                points: vec![
                    [0.12, 0.15],
                    [0.86, 0.18],
                    [0.5, 0.48],
                    [0.82, 0.86],
                    [0.16, 0.78],
                ],
                height: 13.0,
                falloff: 0.08,
                carve,
                mode,
            }),
        ));
        let cpu = cpu_oracle(&stack, &[], metrics);
        let gpu = gpu_eval(&stack, &[], metrics);
        assert_field_parity(
            "authoring.polygon-height",
            &gpu,
            &cpu,
            POLYGON_HEIGHT_PREVIEW,
        );
    }
}

#[test]
fn gpu_required_procedural_shape_variants_delegate_to_bounded_kernels() {
    let metrics = HeightfieldMetrics::new(24, 19, 144.0, 76.0);
    for &generator in ProceduralGenerator::ALL {
        if generator == ProceduralGenerator::Dunes {
            continue;
        }
        let params = ProceduralShapeParams::with_generator(generator);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            generator.label(),
            LayerKind::ProceduralShape(params),
        ));
        let tolerance = match generator {
            ProceduralGenerator::Mountain => MOUNTAINS_PREVIEW,
            ProceduralGenerator::Hills => FBM_PERLIN_PREVIEW,
            ProceduralGenerator::Plateau => PLATEAU_PREVIEW,
            ProceduralGenerator::Mesa => MESA_PREVIEW,
            ProceduralGenerator::Volcano => VOLCANO_PREVIEW,
            ProceduralGenerator::Canyon => CANYONS_PREVIEW,
            ProceduralGenerator::Crater => EFFECT_FILTER_SPATIAL_PREVIEW,
            ProceduralGenerator::Noise => PERLIN_NOISE_PREVIEW,
            ProceduralGenerator::Dunes => unreachable!(),
        };
        let cpu = cpu_oracle(&stack, &[], metrics);
        let gpu = gpu_eval(&stack, &[], metrics);
        assert_field_parity("shape.procedural", &gpu, &cpu, tolerance);
    }
}
