use super::*;

#[test]
fn warm_soft_terrace_width_edit_matches_full_cpu() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 96u32;
    let metrics = HeightfieldMetrics::new(res, res, 768.0, 768.0);
    let mut base = SculptParams::filled(res, 0.0);
    for y in 0..res {
        for x in 0..res {
            base.samples[(y * res + x) as usize] = x as f32 * 0.4 + y as f32 * 0.05;
        }
    }
    let mut stack = LayerStack::new();
    stack.push(Layer::new("ramp", LayerKind::SculptBase(base)));
    let layer = Layer::new(
        "soft terrace",
        LayerKind::SculptStrokes(SculptStrokeParams {
            strokes: vec![terra_core::layer::SculptStroke {
                kind: SculptStrokeKind::Terrace,
                points: vec![terra_core::layer::SculptPoint {
                    u: 0.5,
                    v: 0.5,
                    pressure: 1.0,
                }],
                radius_m: 100.0,
                strength: 8.0,
                target_height: 0.0,
                riser_width_m: 12.0,
                falloff: 1.5,
                enabled: true,
            }],
            reconcile: 0.15,
        }),
    );
    let layer_id = layer.id();
    stack.push(layer);

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
        .expect("warm soft Terrace");
    let before = engine
        .readback_current(&gpu.device, &gpu.queue)
        .expect("initial Terrace readback");

    let LayerKind::SculptStrokes(params) = &mut stack.find_mut(layer_id).unwrap().kind else {
        unreachable!()
    };
    params.strokes[0].riser_width_m = 40.0;
    engine.set_dirty_rect(Some((31, 31, 34, 34)));
    engine.mark_dirty(layer_id);
    let result = engine
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
        .expect("warm Terrace width edit");
    assert!(result.output_identity.is_some());
    let after = engine
        .readback_current(&gpu.device, &gpu.queue)
        .expect("edited Terrace readback");
    let displacement = terra_gpu::parity::max_abs_diff(&before.to_dense(), &after.to_dense());
    assert!(displacement > 0.05, "width edit moved only {displacement}m");

    let oracle = cpu_oracle(&stack, metrics);
    let error = terra_gpu::parity::max_abs_diff(&after.to_dense(), &oracle.to_dense());
    assert!(
        error <= 1.0e-3,
        "warm soft Terrace differs from CPU by {error}"
    );
}
