use super::*;

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
fn gpu_required_terrace_is_history_independent_for_negative_height_range() {
    let metrics = HeightfieldMetrics::new(19, 17, 171.0, 85.0);
    let samples = (0..metrics.height)
        .flat_map(|y| {
            (0..metrics.width).map(move |x| {
                -84.0
                    + x as f32 * 1.35
                    + y as f32 * 0.62
                    + if (x + 2 * y) % 5 == 0 { 7.5 } else { -2.25 }
            })
        })
        .collect();
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "negative pattern",
        LayerKind::SculptBase(SculptParams {
            width: metrics.width,
            height: metrics.height,
            samples,
            fill_height: -40.0,
        }),
    ));
    stack.push(Layer::new(
        "blur",
        LayerKind::Blur(BlurParams {
            radius: 2,
            iterations: 1,
        }),
    ));
    stack.push(Layer::new(
        "terrace",
        LayerKind::Terrace(TerraceParams {
            levels: 9,
            sharpness: 0.82,
        }),
    ));

    let gpu = terra_test_gpu::headless_required();
    let mut cold_engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let cold = gpu_eval_with_engine(&mut cold_engine, gpu, &stack, &[], metrics);

    let mut pollution = LayerStack::new();
    pollution.push(Layer::new(
        "low pollution",
        LayerKind::Flat(FlatParams { height: -900.0 }),
    ));
    pollution.push(Layer::new(
        "high pollution",
        LayerKind::Flat(FlatParams { height: 1_400.0 }),
    ));
    let mut polluted_engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let _ = gpu_eval_with_engine(&mut polluted_engine, gpu, &pollution, &[], metrics);
    let polluted = gpu_eval_with_engine(&mut polluted_engine, gpu, &stack, &[], metrics);

    assert_field_parity(
        "terrace.history-independence",
        &polluted,
        &cold,
        EXACT_HEIGHT,
    );
    let cpu = cpu_oracle(&stack, &[], metrics);
    assert_field_parity("terrace.negative-range", &cold, &cpu, TERRACE_PREVIEW);
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
