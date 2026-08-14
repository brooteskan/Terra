use std::collections::HashMap;

use terra_core::eval::{EvalContext, PreviewQuality, StackEvaluator};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{
    BlendMode, BlurParams, EffectFilterKind, EffectFilterParams, FlatParams,
    HydraulicErosionParams, IslandParams, Layer, LayerKind, LayerStack, NoiseParams, RampParams,
    SculptParams, TerraceParams, ThermalErosionParams,
};
use terra_core::mask::{bake_mask_assets, MaskAsset, MaskId, MaskRef, MaskSource};
use terra_gpu::parity::{
    assert_field_parity, BLUR_PREVIEW, EXACT_HEIGHT, HYDRAULIC_PREVIEW, INFLATE_FILTER_PREVIEW,
    SIMPLE_MASK, SMOOTH_FILTER_PREVIEW, TERRACE_PREVIEW, THERMAL_PREVIEW, VALUE_NOISE_PREVIEW,
    VOLCANIC_ISLAND_PREVIEW,
};
use terra_gpu::GpuTerrainEngine;

const QUALITY: PreviewQuality = PreviewQuality::Draft;

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
        resolution: width,
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
fn gpu_required_hybrid_checkpoint_applies_suffix_once() {
    let gpu = terra_test_gpu::headless_required();
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 80.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "base",
        LayerKind::Flat(FlatParams { height: 10.0 }),
    ));
    let mut unsupported = Layer::new(
        "half add-set",
        LayerKind::EffectFilter(EffectFilterParams::add_set()),
    );
    unsupported.common.opacity = 0.5;
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
