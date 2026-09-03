use super::*;

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
/// stamps + an alias + an aux-only kind + gradient Smooth, Pinch, and
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
            // Smooth minimizes weighted curvature on the *running* height. Placed
            // after Raise and routed across its footprint so ordering is observable;
            // the 8-sample spread exercises the adjustable wide stencil, while the
            // (0.05, 0.05) endpoint exercises the no-flux field edge.
            stroke(
                SculptStrokeKind::Smooth,
                vec![pt(0.05, 0.05, 1.0), pt(0.3, 0.35, 1.0)],
                60.0,
                5.0,
                8.0,
            ),
            // Pinch is Smooth's base-3x3 pull with a bounded, strength-weighted
            // 1.25 gain, routed across the Ridge crest (0.4, 0.75), where the
            // running height sits well above `base`,
            // so a GPU that averaged the running (ridged) height instead of `src`, or
            // dropped the bounded gain, diverges by metres there — far beyond tolerance.
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
fn hundred_sample_smooth_spread_is_continuous_and_matches_cpu() {
    let resolution = 257u32;
    let metrics = HeightfieldMetrics::new(resolution, resolution, 256.0, 256.0);
    let mut base = SculptParams::filled(resolution, 0.0);
    for y in 0..resolution {
        for x in resolution / 2..resolution {
            base.samples[(y * resolution + x) as usize] = 10.0;
        }
    }
    let mut stack = LayerStack::new();
    stack.push(Layer::new("step", LayerKind::SculptBase(base)));
    stack.push(Layer::new(
        "wide smooth",
        LayerKind::SculptStrokes(SculptStrokeParams {
            strokes: vec![SculptStroke {
                kind: SculptStrokeKind::Smooth,
                points: vec![pt(0.5, 0.5, 1.0)],
                radius_m: 100.0,
                strength: 1.0,
                target_height: 100.0,
                falloff: 1.5,
                enabled: true,
            }],
            reconcile: 0.15,
        }),
    ));

    let cpu = cpu_oracle(&stack, &[], metrics);
    let gpu = gpu_eval(&stack, &[], metrics);
    assert_field_parity(
        "shape.sculpt_smooth_spread_100",
        &gpu,
        &cpu,
        SCULPT_STROKES_PREVIEW,
    );

    let center = resolution / 2;
    let row: Vec<f32> = (0..resolution).map(|x| gpu.get(x, center)).collect();
    let transition_width = row
        .iter()
        .filter(|&&value| value > 1.0e-3 && value < 10.0 - 1.0e-3)
        .count();
    let reversal_depth = row
        .windows(2)
        .map(|pair| (pair[0] - pair[1]).max(0.0))
        .fold(0.0f32, f32::max);
    let maximum_local_rise = row
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .fold(0.0f32, f32::max);
    assert!(transition_width >= 50, "transition={transition_width}");
    assert!(reversal_depth <= 1.0e-3, "reversal={reversal_depth}");
    assert!(
        maximum_local_rise <= 0.5,
        "wide Smooth retained a {maximum_local_rise}m one-sample terrace riser"
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
