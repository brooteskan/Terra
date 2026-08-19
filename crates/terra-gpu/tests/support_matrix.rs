use terra_core::eval::{EvalContext, StackEvaluator};
use terra_core::heightfield::HeightfieldMetrics;
use terra_core::layer::{
    BiomesParams, BlendMode, CanyonParams, DomainWarpParams, DuneParams, EffectFilterKind,
    EffectFilterParams, FbmParams, FlatParams, FractalNoiseType, IslandParams, Layer, LayerKind,
    LayerStack, LayerTypeRegistry, MesaParams, MountainParams, MultiScaleAmplifyParams,
    NoiseParams, PathNode, PathParams, PlateauParams, PolygonHeightMode, PolygonHeightParams,
    ProceduralGenerator, ProceduralShapeParams, RiverCarveParams, StreamPowerParams, UpliftParams,
    VolcanoParams,
};
use terra_core::mask::MaskSource;
use terra_gpu::{compile_gpu_graph, layer_gpu_supported, GpuDirtyPolicy, GpuKernel};

fn graph_for(layer: Layer) -> terra_gpu::GpuComputeGraph {
    let mut stack = LayerStack::new();
    stack.push(layer);
    compile_gpu_graph(&stack, &[])
}

#[test]
fn every_builtin_default_has_consistent_public_support_graph_and_kernel() {
    let registry = LayerTypeRegistry::builtin();
    for meta in registry.all() {
        let layer = registry.create(meta.type_id).expect("registered factory");
        let supported = layer_gpu_supported(&layer, &[]);
        let graph = graph_for(layer.clone());
        assert_eq!(graph.fully_gpu(), supported, "{}", meta.type_id);
        assert_eq!(
            graph.cpu_from,
            (!supported).then_some(0),
            "{}",
            meta.type_id
        );
        if supported {
            let [Some(plan)] = graph.plans.as_slice() else {
                panic!("{} must compile to exactly one GPU plan", meta.type_id);
            };
            assert!(
                plan.kernel.matches_layer_kind(&layer.kind),
                "{} selected incompatible {:?}",
                meta.type_id,
                plan.kernel
            );
        }
    }
}

#[test]
fn authored_shape_layers_have_explicit_gpu_configuration_boundaries() {
    let path = Layer::new(
        "path",
        LayerKind::Path(PathParams {
            nodes: vec![
                PathNode {
                    u: 0.1,
                    v: 0.2,
                    height: 2.0,
                    width: 1.0,
                },
                PathNode {
                    u: 0.8,
                    v: 0.7,
                    height: 4.0,
                    width: 0.8,
                },
            ],
            ..PathParams::default()
        }),
    );
    let path_graph = graph_for(path);
    assert!(path_graph.fully_gpu());
    assert_eq!(
        path_graph.plans[0].expect("path plan").kernel,
        GpuKernel::Path
    );

    let polygon = Layer::new(
        "polygon",
        LayerKind::PolygonHeight(PolygonHeightParams {
            points: vec![[0.1, 0.1], [0.8, 0.2], [0.5, 0.9]],
            mode: PolygonHeightMode::SetElevation,
            carve: true,
            ..PolygonHeightParams::default()
        }),
    );
    let polygon_graph = graph_for(polygon);
    assert!(polygon_graph.fully_gpu());
    assert_eq!(
        polygon_graph.plans[0].expect("polygon plan").kernel,
        GpuKernel::PolygonHeight
    );

    for &generator in ProceduralGenerator::ALL {
        let layer = Layer::new(
            generator.label(),
            LayerKind::ProceduralShape(ProceduralShapeParams::with_generator(generator)),
        );
        let supported = generator != ProceduralGenerator::Dunes;
        assert_eq!(layer_gpu_supported(&layer, &[]), supported, "{generator:?}");
        let graph = graph_for(layer);
        assert_eq!(graph.fully_gpu(), supported, "{generator:?}");
        if supported {
            assert_eq!(
                graph.plans[0].expect("procedural plan").kernel,
                GpuKernel::ProceduralShape
            );
        }
    }

    let rejected = [
        Layer::new(
            "non-finite path",
            LayerKind::Path(PathParams {
                width: f32::NAN,
                ..PathParams::default()
            }),
        ),
        Layer::new(
            "overflowing path seed",
            LayerKind::Path(PathParams {
                noise_strength: 1.0,
                seed: u64::from(u32::MAX) + 1,
                ..PathParams::default()
            }),
        ),
        Layer::new(
            "non-finite polygon",
            LayerKind::PolygonHeight(PolygonHeightParams {
                height: f32::NAN,
                ..PolygonHeightParams::default()
            }),
        ),
        Layer::new(
            "iterative crater",
            LayerKind::ProceduralShape(ProceduralShapeParams {
                crater: EffectFilterParams {
                    iterations: 2,
                    ..EffectFilterParams::crater()
                },
                generator: ProceduralGenerator::Crater,
                ..ProceduralShapeParams::default()
            }),
        ),
    ];
    for layer in rejected {
        assert!(!layer_gpu_supported(&layer, &[]), "{}", layer.common.name);
        assert_eq!(graph_for(layer).cpu_from, Some(0));
    }
}

#[test]
fn carved_path_is_demoted_only_when_later_wetness_is_observable() {
    let make_path = |carve| {
        Layer::new(
            "path",
            LayerKind::Path(PathParams {
                carve,
                nodes: vec![
                    PathNode {
                        u: 0.1,
                        v: 0.2,
                        height: 2.0,
                        width: 1.0,
                    },
                    PathNode {
                        u: 0.8,
                        v: 0.7,
                        height: 4.0,
                        width: 0.8,
                    },
                ],
                ..PathParams::default()
            }),
        )
    };

    let mut no_consumer = LayerStack::new();
    no_consumer.push(make_path(true));
    no_consumer.push(Layer::new(
        "flat",
        LayerKind::Flat(FlatParams { height: 1.0 }),
    ));
    assert!(compile_gpu_graph(&no_consumer, &[]).plans[0].is_some());

    let mut consumer = LayerStack::new();
    consumer.push(make_path(true));
    consumer.push(Layer::new(
        "biomes",
        LayerKind::Biomes(BiomesParams::default()),
    ));
    let graph = compile_gpu_graph(&consumer, &[]);
    assert!(graph.plans[0].is_none());
    assert_eq!(graph.cpu_from, Some(0));

    let mut raise_only = LayerStack::new();
    raise_only.push(make_path(false));
    raise_only.push(Layer::new(
        "biomes",
        LayerKind::Biomes(BiomesParams::default()),
    ));
    let graph = compile_gpu_graph(&raise_only, &[]);
    assert!(graph.plans[0].is_some());
    assert_eq!(graph.cpu_from, Some(1));
}

#[test]
fn every_effect_filter_variant_has_an_explicit_executable_plan() {
    for &kind in EffectFilterKind::ALL {
        let layer = Layer::new(
            kind.label(),
            LayerKind::EffectFilter(EffectFilterParams {
                kind,
                ..EffectFilterParams::default()
            }),
        );
        let graph = graph_for(layer);
        let supported = matches!(
            kind,
            EffectFilterKind::Smooth
                | EffectFilterKind::Inflate
                | EffectFilterKind::Denoise
                | EffectFilterKind::AddSet
                | EffectFilterKind::Deflate
                | EffectFilterKind::Curve
                | EffectFilterKind::Cutoff
                | EffectFilterKind::TerraceSimple
                | EffectFilterKind::Shore
                | EffectFilterKind::Blocks
                | EffectFilterKind::ZeroEdge
                | EffectFilterKind::Squeeze
                | EffectFilterKind::DirectionalBlur
                | EffectFilterKind::AngleBlur
                | EffectFilterKind::Swirl
                | EffectFilterKind::Crater
                | EffectFilterKind::Distortion
                | EffectFilterKind::Balloon
                | EffectFilterKind::NoisePerlin
                | EffectFilterKind::NoiseValue
                | EffectFilterKind::NoiseWhite
                | EffectFilterKind::NoiseWave
                | EffectFilterKind::ScatterDetail
                | EffectFilterKind::NoiseBillow
                | EffectFilterKind::NoiseRidged
                | EffectFilterKind::Ridged
                | EffectFilterKind::Rugged
                | EffectFilterKind::Hexagons
                | EffectFilterKind::TerraceSteep
        );
        assert_eq!(graph.fully_gpu(), supported, "{}", kind.label());
        if supported {
            let plan = graph.plans[0].expect("supported filter retains a plan");
            assert_eq!(plan.kernel, GpuKernel::EffectFilter);
            let expected_policy = match kind {
                EffectFilterKind::Curve
                | EffectFilterKind::Cutoff
                | EffectFilterKind::TerraceSimple
                | EffectFilterKind::ZeroEdge
                | EffectFilterKind::Squeeze
                | EffectFilterKind::Swirl
                | EffectFilterKind::Distortion
                | EffectFilterKind::Hexagons
                | EffectFilterKind::TerraceSteep => GpuDirtyPolicy::FullField,
                _ => GpuDirtyPolicy::Local,
            };
            assert_eq!(plan.dirty_policy, expected_policy, "{}", kind.label());
        } else {
            assert_eq!(graph.cpu_from, Some(0));
        }
    }

    let spike = Layer::new(
        "spike removal radius one",
        LayerKind::EffectFilter(EffectFilterParams::spike_removal()),
    );
    let graph = graph_for(spike);
    assert!(graph.fully_gpu());
    assert_eq!(
        graph.plans[0].expect("spike plan").dirty_policy,
        GpuDirtyPolicy::Local
    );

    let unsupported_spike = Layer::new(
        "spike removal radius two",
        LayerKind::EffectFilter(EffectFilterParams {
            radius: 2,
            ..EffectFilterParams::spike_removal()
        }),
    );
    assert!(!graph_for(unsupported_spike).fully_gpu());

    let overflowing_noise = Layer::new(
        "overflowing billow seed stream",
        LayerKind::EffectFilter(EffectFilterParams {
            seed: u64::from(u32::MAX) - 100,
            ..EffectFilterParams::noise_billow()
        }),
    );
    assert!(!graph_for(overflowing_noise).fully_gpu());

    let oversized_radius = Layer::new(
        "oversized directional radius",
        LayerKind::EffectFilter(EffectFilterParams {
            radius: terra_gpu::EFFECT_FILTER_MAX_RADIUS + 1,
            ..EffectFilterParams::directional_blur()
        }),
    );
    assert!(!graph_for(oversized_radius).fully_gpu());

    let overflowing_warp_seed = Layer::new(
        "overflowing warp seed",
        LayerKind::EffectFilter(EffectFilterParams {
            seed: u64::from(u32::MAX),
            warp_strength: 1.0,
            ..EffectFilterParams::noise_wave()
        }),
    );
    assert!(!graph_for(overflowing_warp_seed).fully_gpu());

    for params in [
        EffectFilterParams {
            sea_level: 12.0,
            ..EffectFilterParams::border_blend()
        },
        EffectFilterParams {
            sea_level: 12.0,
            ..EffectFilterParams::flatten_filter()
        },
    ] {
        let graph = graph_for(Layer::new(
            "absolute target",
            LayerKind::EffectFilter(params),
        ));
        assert!(graph.fully_gpu());
        assert_eq!(
            graph.plans[0].expect("absolute plan").dirty_policy,
            GpuDirtyPolicy::Local
        );
    }
}

#[test]
fn river_carve_defaults_and_configuration_boundaries_are_explicit() {
    let default = Layer::new("river", LayerKind::RiverCarve(RiverCarveParams::default()));
    let graph = graph_for(default);
    assert!(graph.fully_gpu());
    let plan = graph.plans[0].expect("RiverCarve plan");
    assert_eq!(plan.kernel, GpuKernel::RiverCarve);
    assert_eq!(plan.dirty_policy, GpuDirtyPolicy::FullField);

    let unsupported = [
        RiverCarveParams {
            guide: MaskSource::Wetness,
            ..RiverCarveParams::default()
        },
        RiverCarveParams {
            bank_smooth: 3.0,
            ..RiverCarveParams::default()
        },
        RiverCarveParams {
            accumulation_threshold: 0.0,
            ..RiverCarveParams::default()
        },
    ];
    for params in unsupported {
        let layer = Layer::new("unsupported river", LayerKind::RiverCarve(params));
        assert!(!layer_gpu_supported(&layer, &[]));
        assert_eq!(graph_for(layer).cpu_from, Some(0));
    }
}

#[test]
fn stream_power_defaults_and_configuration_boundaries_are_explicit() {
    let default = Layer::new(
        "stream power",
        LayerKind::StreamPowerErosion(StreamPowerParams::default()),
    );
    let graph = graph_for(default);
    assert!(graph.fully_gpu());
    let plan = graph.plans[0].expect("StreamPower plan");
    assert_eq!(plan.kernel, GpuKernel::StreamPower);
    assert_eq!(plan.dirty_policy, GpuDirtyPolicy::FullField);
    assert_eq!(plan.halo_texels, 0);

    let unsupported = [
        StreamPowerParams {
            hardness_source: MaskSource::Hardness,
            ..StreamPowerParams::default()
        },
        StreamPowerParams {
            dendritic_seed: 0.2,
            ..StreamPowerParams::default()
        },
        StreamPowerParams {
            refill_each_iter: true,
            ..StreamPowerParams::default()
        },
        StreamPowerParams {
            level_count: 1,
            ..StreamPowerParams::default()
        },
        StreamPowerParams {
            dt: f32::NAN,
            ..StreamPowerParams::default()
        },
    ];
    for params in unsupported {
        let layer = Layer::new(
            "unsupported stream power",
            LayerKind::StreamPowerErosion(params),
        );
        assert!(!layer_gpu_supported(&layer, &[]));
        assert_eq!(graph_for(layer).cpu_from, Some(0));
    }
}

#[test]
fn multi_scale_amplify_defaults_and_configuration_boundaries_are_explicit() {
    let default = Layer::new(
        "multi scale",
        LayerKind::MultiScaleAmplify(MultiScaleAmplifyParams::default()),
    );
    let graph = graph_for(default);
    assert!(graph.fully_gpu());
    let plan = graph.plans[0].expect("MultiScaleAmplify plan");
    assert_eq!(plan.kernel, GpuKernel::MultiScaleAmplify);
    assert_eq!(plan.dirty_policy, GpuDirtyPolicy::FullField);
    assert_eq!(plan.halo_texels, 0);

    for params in [
        MultiScaleAmplifyParams {
            hardness_source: MaskSource::Hardness,
            ..MultiScaleAmplifyParams::default()
        },
        MultiScaleAmplifyParams {
            ridge_lock: MaskSource::Wetness,
            ..MultiScaleAmplifyParams::default()
        },
        MultiScaleAmplifyParams {
            thermal_strength: f32::NAN,
            ..MultiScaleAmplifyParams::default()
        },
    ] {
        let layer = Layer::new("unsupported amplify", LayerKind::MultiScaleAmplify(params));
        assert!(!layer_gpu_supported(&layer, &[]));
        assert_eq!(graph_for(layer).cpu_from, Some(0));
    }
}

#[test]
fn fractal_noise_variants_and_blend_modes_are_explicitly_classified() {
    for noise in [
        FractalNoiseType::Value,
        FractalNoiseType::Perlin,
        FractalNoiseType::OpenSimplex,
    ] {
        for make_kind in [LayerKind::Fbm, LayerKind::Ridged] {
            let layer = Layer::new(
                "fractal",
                make_kind(FbmParams {
                    noise,
                    ..FbmParams::default()
                }),
            );
            let supported = matches!(noise, FractalNoiseType::Value | FractalNoiseType::Perlin);
            assert_eq!(layer_gpu_supported(&layer, &[]), supported, "{noise:?}");
        }
    }

    for (blend, supported) in [
        (BlendMode::Normal, true),
        (BlendMode::Replace, true),
        (BlendMode::Interpolate, true),
        (BlendMode::Add, true),
        (BlendMode::Subtract, true),
        (BlendMode::Multiply, true),
        (BlendMode::Min, true),
        (BlendMode::Max, true),
        (BlendMode::Overlay, true),
        (BlendMode::HeightBlend, false),
        (BlendMode::SmoothMaximum, false),
        (BlendMode::SmoothMinimum, false),
        (BlendMode::SmoothUnion, false),
        (BlendMode::SmoothSubtraction, false),
    ] {
        let mut layer = Layer::new("blend", LayerKind::Flat(FlatParams { height: 2.0 }));
        layer.common.blend = blend;
        assert_eq!(layer_gpu_supported(&layer, &[]), supported, "{blend:?}");
    }
}

#[test]
fn noise_family_defaults_and_seed_stream_boundaries_are_explicit() {
    let defaults = [
        Layer::new("perlin", LayerKind::NoisePerlin(NoiseParams::default())),
        Layer::new("fbm", LayerKind::Fbm(FbmParams::default())),
        Layer::new("ridged", LayerKind::Ridged(FbmParams::default())),
        Layer::new(
            "domain warp",
            LayerKind::DomainWarp(DomainWarpParams::default()),
        ),
    ];
    for layer in defaults {
        let graph = graph_for(layer.clone());
        assert!(graph.fully_gpu(), "{}", layer.common.name);
        assert_eq!(graph.plans[0].expect("noise plan").kernel, GpuKernel::Noise);
    }

    let mut unsupported_blend = Layer::new(
        "warped height blend",
        LayerKind::DomainWarp(DomainWarpParams::default()),
    );
    unsupported_blend.common.blend = BlendMode::HeightBlend;
    assert!(!layer_gpu_supported(&unsupported_blend, &[]));
    assert_eq!(graph_for(unsupported_blend).cpu_from, Some(0));

    let near_limit = u64::from(u32::MAX) - 100;
    let rejected = [
        Layer::new(
            "perlin overflow",
            LayerKind::NoisePerlin(NoiseParams {
                seed: near_limit,
                octaves: 2,
                ..NoiseParams::default()
            }),
        ),
        Layer::new(
            "fbm overflow",
            LayerKind::Fbm(FbmParams {
                base: NoiseParams {
                    seed: near_limit,
                    octaves: 2,
                    ..NoiseParams::default()
                },
                ..FbmParams::default()
            }),
        ),
        Layer::new(
            "ridged overflow",
            LayerKind::Ridged(FbmParams {
                base: NoiseParams {
                    seed: near_limit,
                    octaves: 2,
                    ..NoiseParams::default()
                },
                ..FbmParams::default()
            }),
        ),
        Layer::new(
            "warp overflow",
            LayerKind::DomainWarp(DomainWarpParams {
                base: NoiseParams {
                    seed: u64::from(u32::MAX),
                    octaves: 1,
                    ..NoiseParams::default()
                },
                ..DomainWarpParams::default()
            }),
        ),
        Layer::new(
            "too many perlin octaves",
            LayerKind::NoisePerlin(NoiseParams {
                octaves: 13,
                ..NoiseParams::default()
            }),
        ),
    ];
    for layer in rejected {
        assert!(!layer_gpu_supported(&layer, &[]), "{}", layer.common.name);
        assert_eq!(graph_for(layer).cpu_from, Some(0));
    }
}

#[test]
fn shape_family_defaults_and_island_archetypes_are_explicitly_supported() {
    let layers = [
        Layer::new("mountains", LayerKind::Mountains(MountainParams::default())),
        Layer::new("dunes", LayerKind::Dunes(DuneParams::default())),
        Layer::new("canyons", LayerKind::Canyons(CanyonParams::default())),
        Layer::new("mesa", LayerKind::Mesa(MesaParams::default())),
        Layer::new("volcano", LayerKind::Volcano(VolcanoParams::default())),
        Layer::new("uplift", LayerKind::Uplift(UpliftParams::default())),
        Layer::new("plateau", LayerKind::Plateau(PlateauParams::default())),
        Layer::new("volcanic", LayerKind::Island(IslandParams::default())),
        Layer::new(
            "archipelago",
            LayerKind::Island(IslandParams::archipelago()),
        ),
        Layer::new("atoll", LayerKind::Island(IslandParams::atoll())),
    ];
    for layer in layers {
        let graph = graph_for(layer.clone());
        assert!(graph.fully_gpu(), "{}", layer.common.name);
        assert_eq!(graph.plans[0].expect("shape plan").kernel, GpuKernel::Shape);
    }

    let mut custom_transport = DuneParams::default();
    custom_transport.iterations += 1;
    let layer = Layer::new("custom dune transport", LayerKind::Dunes(custom_transport));
    assert!(!layer_gpu_supported(&layer, &[]));
    assert_eq!(graph_for(layer).cpu_from, Some(0));
}

#[test]
fn first_unsupported_configuration_owns_cpu_from() {
    let low_seed = 7;
    let high_seed = NoiseParams {
        seed: low_seed + (1u64 << 32),
        ..NoiseParams::default()
    };
    assert!(layer_gpu_supported(
        &Layer::new(
            "low seed",
            LayerKind::NoiseValue(NoiseParams {
                seed: low_seed,
                ..NoiseParams::default()
            })
        ),
        &[]
    ));
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "prefix",
        LayerKind::Flat(FlatParams { height: 4.0 }),
    ));
    stack.push(Layer::new(
        "canonical high seed",
        LayerKind::NoiseValue(high_seed),
    ));
    let mut suffix = Layer::new("suffix", LayerKind::Flat(FlatParams { height: 2.0 }));
    suffix.common.blend = BlendMode::Add;
    stack.push(suffix);

    let graph = compile_gpu_graph(&stack, &[]);
    assert_eq!(graph.cpu_from, Some(1));
    // Supported layers below and above the CPU boundary keep plans; only the
    // unsupported owner at index 1 is None.
    assert!(graph.plans[0].is_some());
    assert!(graph.plans[1].is_none());
    assert!(graph.plans[2].is_some());
}

#[test]
fn high_seed_fallback_uses_the_repaired_cpu_field() {
    fn cpu_bits(seed: u64) -> Vec<u32> {
        let metrics = HeightfieldMetrics::new(24, 24, 120.0, 120.0);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "value noise",
            LayerKind::NoiseValue(NoiseParams {
                seed,
                frequency: 0.035,
                amplitude: 20.0,
                octaves: 3,
                ..NoiseParams::default()
            }),
        ));
        let mut evaluator = StackEvaluator::new();
        evaluator
            .rebuild_all(&stack, &mut EvalContext::new(metrics))
            .expect("CPU high-seed fallback oracle")
            .to_dense()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    let low = 7;
    let high = low + (1u64 << 32);
    let high_layer = Layer::new(
        "high seed",
        LayerKind::NoiseValue(NoiseParams {
            seed: high,
            ..NoiseParams::default()
        }),
    );
    assert_eq!(graph_for(high_layer).cpu_from, Some(0));
    assert_ne!(cpu_bits(low), cpu_bits(high));
}
