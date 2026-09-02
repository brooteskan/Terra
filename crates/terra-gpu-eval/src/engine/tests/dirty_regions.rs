use super::*;

/// #127 interaction policy: a local edit may update its cheap prefix while a
/// full-field river pass waits for refinement. The deferred suffix must stay
/// dirty so the next non-interactive evaluation cannot reuse a stale top cache.
#[test]
fn local_edit_defers_full_field_river_suffix_until_refinement() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
    let mut stack = LayerStack::new();
    let base = Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(32, 12.0)),
    );
    let base_id = base.id();
    stack.push(base);
    let river = Layer::new(
        "rivers",
        LayerKind::RiverCarve(RiverCarveParams {
            accumulation_threshold: 2.0,
            width: 1.0,
            bank_smooth: 0.0,
            use_dinfinity: false,
            ..RiverCarveParams::default()
        }),
    );
    let river_id = river.id();
    stack.push(river);
    stack.push(Layer::new("blur", LayerKind::Blur(BlurParams::default())));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm full stack");

    let Some(layer) = stack.find_mut(base_id) else {
        panic!("base layer disappeared");
    };
    let LayerKind::SculptBase(params) = &mut layer.kind else {
        panic!("base changed kind");
    };
    params.samples[16 * 32 + 16] += 3.0;
    engine.set_dirty_rect(Some((16, 16, 1, 1)));
    engine.mark_dirty(base_id);
    let interactive = engine
        .evaluate_with_intent(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
            GpuEvaluationIntent::InteractiveLocal,
        )
        .expect("interactive prefix");
    assert_eq!(
        interactive.freshness,
        GpuPreviewFreshness::Deferred {
            from_index: 1,
            from_layer: river_id,
            deferred_layers: 2,
        }
    );
    assert!(!interactive.fully_gpu);
    assert_eq!(interactive.resume_cpu_from, None);
    assert_eq!(engine.executed_kernels, vec![GpuKernel::Sculpt]);
    let stats = engine.last_eval_stats();
    assert!(stats.used_layer_zero_region);
    assert!(
        stats.upload_bytes < u64::from(metrics.width * metrics.height * 4),
        "interactive prefix must upload only the bounded edit"
    );

    let refined = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("refined full-field suffix");
    assert_eq!(refined.freshness, GpuPreviewFreshness::Current);
    assert!(refined.fully_gpu);
    assert_eq!(
        engine.executed_kernels,
        vec![GpuKernel::RiverCarve, GpuKernel::Blur]
    );

    engine.set_dirty_rect(Some((16, 16, 1, 1)));
    engine.mark_dirty(river_id);
    let boundary_edit = engine
        .evaluate_with_intent(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
            GpuEvaluationIntent::InteractiveLocal,
        )
        .expect("edit at full-field boundary");
    assert!(!boundary_edit.did_eval, "there is no exact local prefix");
    assert_eq!(
        boundary_edit.freshness,
        GpuPreviewFreshness::Deferred {
            from_index: 1,
            from_layer: river_id,
            deferred_layers: 2,
        }
    );
    assert!(engine.executed_kernels.is_empty());
}

/// The first enabled FullField pass is a hard boundary: later FullField and
/// local layers all belong to one suffix and execute once during completion.
#[test]
fn local_edit_defers_entire_suffix_from_first_full_field_boundary() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let mut stack = LayerStack::new();
    let base = Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(16, 12.0)),
    );
    let base_id = base.id();
    stack.push(base);
    let river = Layer::new(
        "rivers",
        LayerKind::RiverCarve(RiverCarveParams {
            accumulation_threshold: 2.0,
            width: 1.0,
            bank_smooth: 0.0,
            use_dinfinity: false,
            ..RiverCarveParams::default()
        }),
    );
    let river_id = river.id();
    stack.push(river);
    stack.push(Layer::new(
        "stream power",
        LayerKind::StreamPowerErosion(StreamPowerParams {
            iterations: 1,
            k: 0.002,
            base_level: 0.0,
            ..StreamPowerParams::default()
        }),
    ));
    stack.push(Layer::new("blur", LayerKind::Blur(BlurParams::default())));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm multi-boundary stack");

    let LayerKind::SculptBase(params) = &mut stack.find_mut(base_id).expect("base").kind else {
        panic!("base changed kind");
    };
    params.samples[8 * 16 + 8] += 3.0;
    engine.set_dirty_rect(Some((8, 8, 1, 1)));
    engine.mark_dirty(base_id);
    let interactive = engine
        .evaluate_with_intent(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
            GpuEvaluationIntent::InteractiveLocal,
        )
        .expect("interactive prefix");
    assert_eq!(
        interactive.freshness,
        GpuPreviewFreshness::Deferred {
            from_index: 1,
            from_layer: river_id,
            deferred_layers: 3,
        }
    );
    assert_eq!(engine.executed_kernels, vec![GpuKernel::Sculpt]);

    let complete = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("complete entire suffix");
    assert_eq!(complete.freshness, GpuPreviewFreshness::Current);
    assert!(!complete.fully_gpu);
    assert!(engine.executed_kernels.is_empty());
    let fallback = complete
        .cpu_fallback
        .expect("live river auxiliary dependency must identify its plan boundary");
    assert_eq!(fallback.reason.code, GpuFallbackCode::AuxiliaryDependency);
    assert_eq!(fallback.layer_id, river_id);
    assert!(fallback.operation.is_some());
    assert_eq!(fallback.owner, Some(NodeRef::Layer(river_id)));
}

/// #129 interaction policy: StreamPower shares the generic FullField deferral
/// path, so no obsolete accumulation/incision sequence launches per pointer dab.
#[test]
fn local_edit_defers_full_field_stream_power_suffix_until_refinement() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let mut stack = LayerStack::new();
    let base = Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(16, 12.0)),
    );
    let base_id = base.id();
    stack.push(base);
    stack.push(Layer::new(
        "stream power",
        LayerKind::StreamPowerErosion(StreamPowerParams {
            iterations: 1,
            k: 0.002,
            base_level: 0.0,
            ..StreamPowerParams::default()
        }),
    ));
    stack.push(Layer::new("blur", LayerKind::Blur(BlurParams::default())));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm stream-power stack");

    let Some(layer) = stack.find_mut(base_id) else {
        panic!("base layer disappeared");
    };
    let LayerKind::SculptBase(params) = &mut layer.kind else {
        panic!("base changed kind");
    };
    params.samples[8 * 16 + 8] += 3.0;
    engine.set_dirty_rect(Some((8, 8, 1, 1)));
    engine.mark_dirty(base_id);
    let interactive = engine
        .evaluate_with_intent(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
            GpuEvaluationIntent::InteractiveLocal,
        )
        .expect("interactive stream-power prefix");
    assert!(interactive.freshness.is_deferred());
    assert!(!interactive.fully_gpu);
    assert_eq!(interactive.resume_cpu_from, None);
    assert_eq!(engine.executed_kernels, vec![GpuKernel::Sculpt]);

    let refined = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("refined stream-power suffix");
    assert_eq!(refined.freshness, GpuPreviewFreshness::Current);
    assert!(refined.fully_gpu);
    assert_eq!(
        engine.executed_kernels,
        vec![GpuKernel::StreamPower, GpuKernel::Blur]
    );
}

/// #131 interaction policy: the multi-level solver must use the same generic
/// FullField deferral as the other drainage-coupled GPU previews.
#[test]
fn local_edit_defers_multi_scale_amplify_suffix_until_refinement() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let mut stack = LayerStack::new();
    let base = Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(16, 12.0)),
    );
    let base_id = base.id();
    stack.push(base);
    stack.push(Layer::new(
        "multi scale",
        LayerKind::MultiScaleAmplify(MultiScaleAmplifyParams {
            thermal_iters: 1,
            spe_strength: 0.0,
            spe_iters: 0,
            deposition_strength: 0.0,
            hardness_source: MaskSource::Constant(0.2),
            ridge_lock: MaskSource::Constant(0.15),
            ..MultiScaleAmplifyParams::default()
        }),
    ));
    stack.push(Layer::new("blur", LayerKind::Blur(BlurParams::default())));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm multi-scale stack");

    let Some(layer) = stack.find_mut(base_id) else {
        panic!("base layer disappeared");
    };
    let LayerKind::SculptBase(params) = &mut layer.kind else {
        panic!("base changed kind");
    };
    params.samples[8 * 16 + 8] += 3.0;
    engine.set_dirty_rect(Some((8, 8, 1, 1)));
    engine.mark_dirty(base_id);
    let interactive = engine
        .evaluate_with_intent(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
            GpuEvaluationIntent::InteractiveLocal,
        )
        .expect("interactive multi-scale prefix");
    assert!(interactive.freshness.is_deferred());
    assert!(!interactive.fully_gpu);
    assert_eq!(engine.executed_kernels, vec![GpuKernel::Sculpt]);

    let refined = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("refined multi-scale suffix");
    assert_eq!(refined.freshness, GpuPreviewFreshness::Current);
    assert!(refined.fully_gpu);
    assert_eq!(
        engine.executed_kernels,
        vec![GpuKernel::MultiScaleAmplify, GpuKernel::Blur]
    );
}

/// Two `SculptStrokes` layers in one evaluate walk share the stamp/edited/layer
/// scratch textures. The second must read the first layer's composited result
/// (not a clobbered scratch), so the whole field must still match the CPU, which
/// applies the layers in the same order.
#[test]
fn stacked_sculpt_stroke_layers_apply_in_order_and_match_cpu() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(40, 40, 400.0, 400.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(40, 12.0)),
    ));
    stack.push(Layer::new(
        "s1",
        LayerKind::SculptStrokes(raise_strokes(0.45, 0.5, 10.0)),
    ));
    stack.push(Layer::new(
        "s2",
        LayerKind::SculptStrokes(raise_strokes(0.55, 0.5, 6.0)),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    let gpu_h = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("stacked stroke evaluation")
        .cpu
        .expect("gpu readback");

    let cpu = cpu_oracle(&stack, metrics);
    terra_gpu::parity::assert_field_parity(
        "authoring.sculpt-strokes-stacked",
        &gpu_h,
        &cpu,
        terra_gpu::parity::SCULPT_STROKES_PREVIEW,
    );
}

/// A SculptStrokes layer sits above the base, so a stroke dab is an incremental
/// eval that seeds `approx_range` from cache rather than resetting it. Additive
/// strokes must not widen that carried range, or it drifts on every drag step and
/// the renderer's slab base (`min_h - f(max_h - min_h)`) visibly sinks. Repeated
/// identical dabs must leave the presentation range fixed.
#[test]
fn incremental_sculpt_stroke_dabs_do_not_drift_the_presentation_range() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(48, 48, 480.0, 480.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(48, 30.0)),
    ));
    let strokes_layer = Layer::new(
        "strokes",
        LayerKind::SculptStrokes(raise_strokes(0.5, 0.5, 12.0)),
    );
    let strokes_id = strokes_layer.id();
    stack.push(strokes_layer);

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("full stroke evaluation");
    let range_after_full = engine.approx_range;

    for _ in 0..6 {
        engine.set_dirty_rect(Some((16, 16, 16, 16)));
        engine.mark_dirty(strokes_id);
        engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Draft,
                false,
                None,
            )
            .expect("incremental stroke dab");
    }
    assert_eq!(
        engine.approx_range, range_after_full,
        "incremental stroke dabs drifted the presentation range"
    );
}

/// #145: a continuing drag patches only the changed stroke header and the
/// appended point. It reuses both the compiled-plan realization and spare
/// buffer capacity, while the settled regional result remains identical to
/// a fresh full evaluation.
#[test]
fn warm_stroke_append_uploads_only_the_runtime_tail() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 48u32;
    let metrics = HeightfieldMetrics::new(res, res, 480.0, 480.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(res, 20.0)),
    ));
    let params = raise_strokes(0.46, 0.5, 8.0);
    let strokes = Layer::new("strokes", LayerKind::SculptStrokes(params));
    let strokes_id = strokes.id();
    stack.push(strokes);

    let mut engine = GpuTerrainEngine::new(&gpu.device, res);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("prime stroke runtime with spare geometric capacity");

    let LayerKind::SculptStrokes(params) =
        &mut stack.find_mut(strokes_id).expect("stroke layer").kind
    else {
        panic!("stroke layer changed kind");
    };
    params.strokes[0]
        .points
        .push(terra_core::layer::SculptPoint {
            u: 0.52,
            v: 0.5,
            pressure: 1.0,
        });
    engine.set_dirty_rect(Some((18, 18, 16, 12)));
    engine.mark_dirty(strokes_id);
    let incremental = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("append one warm stroke point")
        .cpu
        .expect("incremental readback");

    let stats = engine.last_eval_stats();
    assert_eq!(stats.stroke_payload_rebuilds, 0);
    assert_eq!(
        stats.stroke_header_upload_bytes,
        std::mem::size_of::<StrokeHeaderGpu>() as u64
    );
    assert_eq!(stats.stroke_point_upload_bytes, 16);
    assert_eq!(stats.warm_plan_resource_reuses, 1);
    assert!(
        stats.blend_workgroups < u64::from(res.div_ceil(8) * res.div_ceil(8)),
        "warm stroke work must remain below a full-field dispatch"
    );

    let mut oracle_engine = GpuTerrainEngine::new(&gpu.device, res);
    let oracle = oracle_engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("fresh stroke oracle")
        .cpu
        .expect("oracle readback");
    let error = terra_gpu::parity::max_abs_diff(&incremental.to_dense(), &oracle.to_dense());
    assert!(error <= 1.0e-3, "warm stroke append drifted by {error}");
}

/// A warm Pinch drag reads the immutable layer input one sample beyond its stamp
/// rectangle, then reconcile reads the stamped field one sample farther. Those
/// guard samples must be refreshed without publishing them; otherwise every
/// bounded dab leaves a rectangular seam until a later full-field evaluation.
#[test]
fn warm_pinch_drag_matches_full_field_before_refinement() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 64u32;
    let metrics = HeightfieldMetrics::new(res, res, 640.0, 640.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "varied base",
        LayerKind::SculptBase(varied_sculpt(res)),
    ));
    let strokes = Layer::new(
        "pinch",
        LayerKind::SculptStrokes(SculptStrokeParams {
            strokes: vec![terra_core::layer::SculptStroke {
                kind: SculptStrokeKind::Pinch,
                points: vec![terra_core::layer::SculptPoint {
                    u: 0.44,
                    v: 0.5,
                    pressure: 1.0,
                }],
                radius_m: 55.0,
                strength: 0.85,
                target_height: 0.0,
                falloff: 0.7,
                enabled: true,
            }],
            reconcile: 0.15,
        }),
    );
    let strokes_id = strokes.id();
    stack.push(strokes);

    let mut engine = GpuTerrainEngine::new(&gpu.device, res);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("prime pinch field");

    for u in [0.47, 0.50, 0.53, 0.56] {
        let LayerKind::SculptStrokes(params) =
            &mut stack.find_mut(strokes_id).expect("pinch layer").kind
        else {
            panic!("pinch layer changed kind");
        };
        params.strokes[0]
            .points
            .push(terra_core::layer::SculptPoint {
                u,
                v: 0.5,
                pressure: 1.0,
            });

        let center_x = (u * res as f32).floor() as u32;
        engine.set_dirty_rect(Some((center_x.saturating_sub(7), 25, 15, 15)));
        engine.mark_dirty(strokes_id);
        let incremental = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Draft,
                true,
                None,
            )
            .expect("bounded pinch update")
            .cpu
            .expect("bounded pinch readback");

        let mut oracle_engine = GpuTerrainEngine::new(&gpu.device, res);
        let oracle = oracle_engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Draft,
                true,
                None,
            )
            .expect("full pinch oracle")
            .cpu
            .expect("full pinch readback");
        let incremental_dense = incremental.to_dense();
        let oracle_dense = oracle.to_dense();
        let (max_index, error) = incremental_dense
            .iter()
            .zip(&oracle_dense)
            .enumerate()
            .map(|(index, (got, want))| (index, (got - want).abs()))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("non-empty heightfield");
        assert!(
            error <= 1.0e-3,
            "bounded Pinch at u={u} left a rectangular seam at ({}, {}) (max error {error})",
            max_index % res as usize,
            max_index / res as usize,
        );
    }
}

/// #125: domain displacement changes only which procedural-noise coordinate is
/// generated. It never samples a displaced texel from the entering height, so a
/// bounded upstream edit still passes through its Add blend without a
/// warp-strength-sized dirty halo.
#[test]
fn dirty_rect_domain_warp_matches_full_field_evaluation() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 64u32;
    let metrics = HeightfieldMetrics::new(res, res, 192.0, 128.0);
    let rect = (24u32, 20u32, 12u32, 10u32);
    let mut stack = LayerStack::new();
    let base = Layer::new("base", LayerKind::Flat(FlatParams { height: 4.0 }));
    let base_id = base.id();
    stack.push(base);
    let sculpt = Layer::new(
        "sculpt",
        LayerKind::SculptBase(SculptParams::filled(res, 20.0)),
    );
    let sculpt_id = sculpt.id();
    stack.push(sculpt);
    stack.push(Layer::new(
        "warp",
        LayerKind::DomainWarp(DomainWarpParams {
            warp_strength: 35.0,
            warp_frequency: 0.018,
            ..DomainWarpParams::default()
        }),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_dirty(base_id);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm domain-warp stack");

    if let LayerKind::SculptBase(params) = &mut stack.flatten_layers_mut()[1].kind {
        for y in rect.1..rect.1 + rect.3 {
            for x in rect.0..rect.0 + rect.2 {
                params.samples[(y * res + x) as usize] = 65.0;
            }
        }
    }
    engine.set_dirty_rect(Some(rect));
    engine.mark_dirty(sculpt_id);
    let incremental = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("incremental domain-warp evaluation")
        .cpu
        .expect("incremental readback");

    let mut oracle_engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let oracle = oracle_engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("full domain-warp evaluation")
        .cpu
        .expect("oracle readback");
    let error = terra_gpu::parity::max_abs_diff(&incremental.to_dense(), &oracle.to_dense());
    let worst = incremental
        .to_dense()
        .iter()
        .zip(oracle.to_dense())
        .enumerate()
        .max_by(|(_, (a0, b0)), (_, (a1, b1))| (*a0 - *b0).abs().total_cmp(&(*a1 - *b1).abs()))
        .map(|(index, (a, b))| (index % res as usize, index / res as usize, *a, b));
    assert!(
        error <= 1.0e-3,
        "incremental DomainWarp drifted by {error} at {worst:?}"
    );
}

/// B1-D6 / C1-C2 — #90's explicit rect-edge-artifact answer. A flat field with
/// a tall bump inside the edit rect: a wide Smooth (radius 16, 2 iters) spreads
/// that bump ~32 texels. A probe in the spread ring — outside the retired
/// 8-texel halo but inside the true reach — must match the full-field oracle.
/// The plan halo covers it; the hardcoded 8-texel halo left the ring unfiltered
/// (the visible artifact). Away from the bump the field is flat, so filtered ==
/// unfiltered there and the probe isolates exactly the halo coverage.
#[test]
fn dirty_rect_effect_filter_recomputes_full_kernel_reach() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 96u32;
    let metrics = HeightfieldMetrics::new(res, res, 192.0, 192.0);
    let rect = (40u32, 40u32, 16u32, 16u32);

    // Flat base at index 0 gives the sculpt a cached prefix (first_dirty > 0 =>
    // incremental rect path).
    let build = || {
        let mut sculpt = SculptParams::filled(res, 20.0);
        for y in rect.1..rect.1 + rect.3 {
            for x in rect.0..rect.0 + rect.2 {
                sculpt.samples[(y * res + x) as usize] = 420.0;
            }
        }
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "base",
            LayerKind::Flat(FlatParams { height: 4.0 }),
        ));
        let sculpt_layer = Layer::new("sculpt", LayerKind::SculptBase(sculpt));
        let sculpt_id = sculpt_layer.id();
        stack.push(sculpt_layer);
        stack.push(Layer::new(
            "smooth",
            LayerKind::EffectFilter(EffectFilterParams {
                radius: 16,
                iterations: 2,
                strength: 1.0,
                ..EffectFilterParams::smooth()
            }),
        ));
        (stack, sculpt_id)
    };

    let (stack, sculpt_id) = build();
    let base_id = stack.flatten_layers()[0].id();

    // Warm the caches with a full-field pass.
    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_dirty(base_id);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm full evaluation");

    // Incremental rect eval: dirty the sculpt, bound the edit to `rect`.
    engine.set_dirty_rect(Some(rect));
    engine.mark_dirty(sculpt_id);
    let incremental = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("incremental rect eval")
        .cpu
        .expect("incremental readback");

    // Full-field oracle from a fresh engine.
    let mut oracle_engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let oracle = oracle_engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("oracle eval")
        .cpu
        .expect("oracle readback");

    // Probe the spread ring: outside rect+8 (the retired halo would leave it
    // unfiltered) but inside rect+32 (the true reach), where the bump has spread.
    let (px, py) = (70u32, 48u32);
    let spread = oracle.get(px, py) - 20.0;
    assert!(
        spread > 2.0,
        "fixture bump did not spread into the probe ring (spread {spread})"
    );
    let err = (incremental.get(px, py) - oracle.get(px, py)).abs();
    assert!(
        err < 0.5,
        "incremental filter left the spread ring stale vs the full oracle by {err}; \
             the recompute halo under-covers the kernel reach (rect-edge artifact)"
    );
}

/// The mask bake must cover the whole field, not just the edit rect: it feeds a
/// full-field blend and a full-field layer cache, so a region-only bake would
/// leave mask = 1.0 (and the wrong blended height) outside the rect. This guards
/// the mask-bake fix that landed with B1-D6 (#90).
#[test]
fn dirty_rect_masked_generator_bakes_full_field() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 64u32;
    let metrics = HeightfieldMetrics::new(res, res, 128.0, 128.0);
    let mask = MaskAsset::new(MaskId::new(), "constant", MaskSource::Constant(0.25));
    let assets = vec![mask.clone()];

    let build = || {
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "base",
            LayerKind::Flat(FlatParams { height: 4.0 }),
        ));
        let sculpt_layer = Layer::new("sculpt", LayerKind::SculptBase(varied_sculpt(res)));
        let sculpt_id = sculpt_layer.id();
        stack.push(sculpt_layer);
        let mut masked = Layer::new("masked add", LayerKind::Flat(FlatParams { height: 50.0 }));
        masked.common.blend = BlendMode::Add;
        masked.common.masks.push(MaskRef::new(mask.id));
        let masked_id = masked.id();
        stack.push(masked);
        (stack, sculpt_id, masked_id)
    };

    // Warm the engine with a full-field pass.
    let (stack, _sculpt_id, masked_id) = build();
    let base_id = stack.flatten_layers()[0].id();
    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_dirty(base_id);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &assets,
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("masked full evaluation");

    // Incremental: re-bake the masked layer under a small dirty rect. No filter is
    // involved, so a correct full-field bake makes the whole readback match the
    // oracle; a region-only bake corrupts everything outside the rect.
    let rect = (24u32, 24u32, 12u32, 12u32);
    engine.set_dirty_rect(Some(rect));
    engine.mark_dirty(masked_id);
    let incremental = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &assets,
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("incremental masked evaluation")
        .cpu
        .expect("incremental readback");

    let mut oracle_engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let oracle = oracle_engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &assets,
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("oracle masked evaluation")
        .cpu
        .expect("oracle readback");

    let mut max_err = 0.0f32;
    for y in 0..res {
        for x in 0..res {
            max_err = max_err.max((incremental.get(x, y) - oracle.get(x, y)).abs());
        }
    }
    assert!(
        max_err < 0.05,
        "masked incremental diverged from the full oracle by {max_err}; the mask bake \
             did not cover the full field outside the dirty rect"
    );
}

/// #148: warm Raise and Pinch gestures on both authored sculpt payloads
/// remain on the bounded compiled tree plan.
#[test]
fn untitled6_tree_warm_brushes_stay_bounded() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 96u32;
    for (target_strokes, brush) in [
        (false, SculptStrokeKind::Raise),
        (false, SculptStrokeKind::Pinch),
        (true, SculptStrokeKind::Raise),
        (true, SculptStrokeKind::Pinch),
    ] {
        let (mut document, ids) = untitled6_document(res, Untitled6Variant::ProductionTopology);
        document.metrics.tile_size = 16;
        document.metrics.halo = 2;
        let target = if target_strokes {
            ids.semantic_sculpt
        } else {
            ids.base
        };
        let mut cache = TerrainPlanCache::new();
        let cold = cache
            .update(
                &document.stack,
                &document.masks,
                &[TerrainEditClass::Structure],
            )
            .expect("compile fixture");
        let mut engine = GpuTerrainEngine::new(&gpu.device, res);
        engine
            .evaluate_compiled_with_intent(
                &gpu.device,
                &gpu.queue,
                &document.stack,
                &document.masks,
                cache.current_plan().unwrap(),
                cache.structure_revision(),
                &cold,
                document.metrics,
                PreviewQuality::Draft,
                false,
                GpuEvaluationIntent::Complete,
            )
            .expect("warm resources");
        let plan_before = cache.stats().snapshot();

        let layer = document.stack.find_mut(target).expect("gesture target");
        for (index, u) in [0.47, 0.50, 0.53].into_iter().enumerate() {
            layer.apply_brush(
                brush,
                BrushDab {
                    u,
                    v: 0.5,
                    radius_uv: 0.035,
                    radius_m: 90.0,
                    strength: 4.0,
                    target_height: 24.0,
                    falloff: 0.55,
                    continuing: index != 0,
                },
            );
        }
        let invalidation = cache
            .update(
                &document.stack,
                &document.masks,
                &[TerrainEditClass::Content {
                    owner: NodeRef::Layer(target),
                    fields: vec![FieldId::Height],
                    scope: PlanDirtyScope::Region(UvRect::from_center_radius(0.5, 0.5, 0.07)),
                }],
            )
            .expect("patch gesture");
        let result = engine
            .evaluate_compiled_with_intent(
                &gpu.device,
                &gpu.queue,
                &document.stack,
                &document.masks,
                cache.current_plan().unwrap(),
                cache.structure_revision(),
                &invalidation,
                document.metrics,
                PreviewQuality::Draft,
                false,
                GpuEvaluationIntent::InteractiveLocal,
            )
            .expect("interactive gesture");
        assert!(result.fully_gpu, "{target_strokes}/{brush:?}");
        assert_eq!(result.freshness, GpuPreviewFreshness::Current);
        assert!(result.cpu_fallback.is_none());

        let plan_after = cache.stats().snapshot();
        assert_eq!(plan_after.plan_compiles, plan_before.plan_compiles);
        assert_eq!(
            plan_after.authored_tree_walks,
            plan_before.authored_tree_walks
        );
        assert_eq!(plan_after.dependency_builds, plan_before.dependency_builds);
        let stats = engine.last_eval_stats();
        assert!(stats.operations_dispatched > 0);
        assert!(
            stats.operations_reused > 0,
            "Volcano contribution should be reused"
        );
        assert_eq!(stats.operations_deferred, 0);
        assert_eq!(stats.readback_bytes, 0);
        assert!(stats.upload_bytes < u64::from(res * res * 4), "{stats:?}");
        assert!(engine.last_plan_operation_trace().iter().any(|operation| {
            operation.disposition == GpuPlanOperationDisposition::Dispatched
                && operation.output_region.2 < res
                && operation.output_region.3 < res
        }));

        for quality in [PreviewQuality::Medium, PreviewQuality::Full] {
            let result = engine
                .evaluate_compiled_with_intent(
                    &gpu.device,
                    &gpu.queue,
                    &document.stack,
                    &document.masks,
                    cache.current_plan().unwrap(),
                    cache.structure_revision(),
                    &PlanInvalidation::default(),
                    document.metrics,
                    quality,
                    false,
                    GpuEvaluationIntent::Complete,
                )
                .unwrap_or_else(|error| panic!("{quality:?} refinement failed: {error}"));
            assert!(result.fully_gpu, "{target_strokes}/{brush:?}/{quality:?}");
            assert_eq!(result.freshness, GpuPreviewFreshness::Current);
            assert!(result.cpu_fallback.is_none(), "{quality:?}");
            let stats = engine.last_eval_stats();
            assert_eq!(stats.operations_deferred, 0, "{quality:?}");
            assert_eq!(stats.readback_bytes, 0, "{quality:?}");
        }

        let settled = engine
            .readback_current(&gpu.device, &gpu.queue)
            .expect("settled preview");
        let oracle = cpu_oracle(&document.stack, document.metrics);
        terra_gpu::parity::assert_field_parity(
            &format!("Untitled6 target_strokes={target_strokes} brush={brush:?}"),
            &settled,
            &oracle,
            terra_gpu::parity::UNTITLED6_INTERACTION,
        );
    }
}
