use std::collections::HashMap;

use terra_core::eval::{EvalContext, PreviewQuality, StackEvaluator};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{
    BlendMode, BlurParams, EffectFilterKind, EffectFilterParams, FlatParams,
    HydraulicErosionParams, IslandParams, Layer, LayerKind, LayerStack, LandscapeEvolutionParams,
    NoiseParams, RampParams, SculptParams, SculptPoint, SculptStroke, SculptStrokeKind,
    SculptStrokeParams, TerraceParams, ThermalErosionParams,
};
use terra_core::mask::{bake_mask_assets, MaskAsset, MaskId, MaskRef, MaskSource};
use terra_gpu::parity::{
    assert_field_parity, BLUR_PREVIEW, DENOISE_FILTER_PREVIEW, EXACT_HEIGHT, HYDRAULIC_PREVIEW,
    INFLATE_FILTER_PREVIEW, SCULPT_STROKES_PREVIEW, SIMPLE_MASK, SMOOTH_FILTER_PREVIEW,
    TERRACE_PREVIEW, THERMAL_PREVIEW, VALUE_NOISE_PREVIEW, VOLCANIC_ISLAND_PREVIEW,
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
            stroke(SculptStrokeKind::Lower, vec![pt(0.7, 0.65, 1.0)], 60.0, 6.0, 0.0),
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
            stroke(SculptStrokeKind::Inflate, vec![pt(0.5, 0.5, 1.0)], 65.0, 5.0, 0.0),
            stroke(SculptStrokeKind::Terrace, vec![pt(0.35, 0.5, 1.0)], 80.0, 5.0, 0.0),
            stroke(SculptStrokeKind::Noise, vec![pt(0.5, 0.8, 1.0)], 60.0, 4.0, 0.0),
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
            stroke(SculptStrokeKind::CraterStamp, vec![pt(0.65, 0.5, 1.0)], 55.0, 9.0, 0.0),
            // Alias of Ridge; aux-only kind exercises the reconcile edit weight.
            stroke(SculptStrokeKind::MountainStamp, vec![pt(0.15, 0.5, 1.0)], 45.0, 7.0, 0.0),
            stroke(SculptStrokeKind::Uplift, vec![pt(0.5, 0.15, 1.0)], 50.0, 3.0, 0.0),
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
    assert_field_parity("authoring.sculpt-strokes-blended", &gpu, &cpu, SCULPT_STROKES_PREVIEW);
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
