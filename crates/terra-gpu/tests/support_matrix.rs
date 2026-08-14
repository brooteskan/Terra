use terra_core::eval::{EvalContext, StackEvaluator};
use terra_core::heightfield::HeightfieldMetrics;
use terra_core::layer::{
    BlendMode, EffectFilterKind, EffectFilterParams, FbmParams, FlatParams, FractalNoiseType,
    Layer, LayerKind, LayerStack, LayerTypeRegistry, NoiseParams,
};
use terra_gpu::{compile_gpu_graph, layer_gpu_supported, GpuKernel};

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
            let [pass] = graph.passes.as_slice() else {
                panic!("{} must compile to exactly one pass", meta.type_id);
            };
            assert!(
                pass.kernel.matches_layer_kind(&layer.kind),
                "{} selected incompatible {:?}",
                meta.type_id,
                pass.kernel
            );
        }
    }
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
        let supported = matches!(kind, EffectFilterKind::Smooth | EffectFilterKind::Inflate);
        assert_eq!(graph.fully_gpu(), supported, "{}", kind.label());
        if supported {
            assert_eq!(graph.passes[0].kernel, GpuKernel::EffectFilter);
        } else {
            assert_eq!(graph.cpu_from, Some(0));
        }
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
            assert!(!layer_gpu_supported(&layer, &[]), "{noise:?}");
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
    assert_eq!(graph.passes.len(), 1);
    assert_eq!(graph.passes[0].flat_index, 0);
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
