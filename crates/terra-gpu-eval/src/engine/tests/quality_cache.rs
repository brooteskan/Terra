use super::*;

fn amplify_stack(resolution: u32) -> LayerStack {
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "base",
        LayerKind::SculptBase(varied_sculpt(resolution)),
    ));
    stack.push(Layer::new(
        "amplify",
        LayerKind::MultiScaleAmplify(MultiScaleAmplifyParams {
            level_count: 2,
            thermal_iters: 14,
            spe_iters: 7,
            ..MultiScaleAmplifyParams::default()
        }),
    ));
    stack
}

/// #192: quality participates in the synchronous compiled-plan cache key even
/// when the dimensions and authored plan are unchanged.
#[test]
fn same_resolution_medium_to_full_is_cold_and_matches_cold_full() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let resolution = 32;
    let metrics = HeightfieldMetrics::new(resolution, resolution, 320.0, 320.0);
    let stack = amplify_stack(resolution);

    let mut transitioned = GpuTerrainEngine::new(&gpu.device, resolution);
    transitioned.mark_all_dirty(&stack);
    let medium = transitioned
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Medium,
            true,
            None,
        )
        .expect("Medium evaluation");
    let full = transitioned
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Full,
            true,
            None,
        )
        .expect("transitioned Full evaluation");
    let stats = transitioned.last_eval_stats();
    assert!(stats.cold_execution);
    assert_eq!(stats.reused_contributions, 0);
    assert_eq!(stats.operations_reused, 0);
    let identity = full.output_identity.expect("Full output identity");
    assert_eq!(identity.requested_quality, PreviewQuality::Full);
    assert_eq!(identity.actual_quality, PreviewQuality::Full);

    let mut cold = GpuTerrainEngine::new(&gpu.device, resolution);
    cold.mark_all_dirty(&stack);
    let cold_full = cold
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Full,
            true,
            None,
        )
        .expect("cold Full evaluation");
    let transitioned_field = full.cpu.expect("transitioned readback").to_dense();
    let cold_field = cold_full.cpu.expect("cold readback").to_dense();
    assert_eq!(
        terra_gpu::parity::max_abs_diff(&transitioned_field, &cold_field),
        0.0
    );
    assert!(
        terra_gpu::parity::max_abs_diff(
            &medium.cpu.expect("Medium readback").to_dense(),
            &cold_field,
        ) > 1e-5,
        "fixture must distinguish Medium from Full"
    );
}

/// A failed quality transition must not poison the next attempt into treating
/// old Medium resources as a warm Full realization.
#[test]
fn failed_quality_transition_keeps_retry_cold() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let resolution = 24;
    let metrics = HeightfieldMetrics::new(resolution, resolution, 240.0, 240.0);
    let stack = amplify_stack(resolution);
    let mut cache = TerrainPlanCache::new();
    let initial = cache
        .update(&stack, &[], &[TerrainEditClass::Structure])
        .expect("compile stack");
    let plan = cache.current_plan().expect("compiled plan");
    let revision = cache.structure_revision();
    let mut engine = GpuTerrainEngine::new(&gpu.device, resolution);
    engine
        .evaluate_compiled_with_intent(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            plan,
            revision,
            &initial,
            metrics,
            PreviewQuality::Medium,
            false,
            GpuEvaluationIntent::Complete,
        )
        .expect("initial Medium evaluation");
    assert_eq!(engine.last_quality, Some(PreviewQuality::Medium));

    let bridge = Heightfield::zeros(metrics);
    let failed = engine.evaluate_compiled_with_bridge(
        &gpu.device,
        &gpu.queue,
        &stack,
        &[],
        plan,
        revision,
        &PlanInvalidation::default(),
        metrics,
        PreviewQuality::Full,
        false,
        GpuEvaluationIntent::Complete,
        Some(BridgePrefix {
            height: &bridge,
            first_dirty_layer: LayerId::new(),
        }),
        None,
    );
    assert!(failed.is_err());
    assert_eq!(engine.last_quality, Some(PreviewQuality::Medium));

    engine
        .evaluate_compiled_with_intent(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            plan,
            revision,
            &PlanInvalidation::default(),
            metrics,
            PreviewQuality::Full,
            false,
            GpuEvaluationIntent::Complete,
        )
        .expect("Full retry");
    assert!(engine.last_eval_stats().cold_execution);
    assert_eq!(engine.last_quality, Some(PreviewQuality::Full));
}

/// The resumable refinement path has the same quality boundary and only
/// publishes Full identity/resources after its final fence completes.
#[test]
fn staged_medium_to_full_publishes_cold_full_result() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let resolution = 24;
    let metrics = HeightfieldMetrics::new(resolution, resolution, 240.0, 240.0);
    let stack = amplify_stack(resolution);
    let mut cache = TerrainPlanCache::new();
    let invalidation = cache
        .update(&stack, &[], &[TerrainEditClass::Structure])
        .expect("compile stack");
    let plan = cache.current_plan().expect("compiled plan");
    let revision = cache.structure_revision();

    let mut staged = GpuTerrainEngine::new(&gpu.device, resolution);
    staged
        .evaluate_compiled_with_intent(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            plan,
            revision,
            &invalidation,
            metrics,
            PreviewQuality::Medium,
            false,
            GpuEvaluationIntent::Complete,
        )
        .expect("initial Medium evaluation");
    let mut job = staged
        .begin_compiled_refinement(
            &gpu.device,
            &stack,
            &[],
            plan,
            revision,
            &PlanInvalidation::default(),
            metrics,
            PreviewQuality::Full,
        )
        .expect("begin Full refinement");
    for _ in 0..10_000 {
        let step = staged
            .advance_compiled_refinement(&gpu.device, &gpu.queue, &mut job)
            .expect("advance Full refinement");
        if matches!(
            step,
            GpuRefinementStep::Submitted { .. } | GpuRefinementStep::AwaitingGpu
        ) {
            let _ = gpu.device.poll(wgpu::Maintain::Wait);
        }
        if matches!(step, GpuRefinementStep::ReadyToPublish) {
            break;
        }
    }
    assert!(job.final_copy_complete, "Full refinement did not finish");
    let result = staged
        .publish_compiled_refinement(job)
        .expect("publish Full refinement");
    let identity = result.output_identity.expect("Full refinement identity");
    assert_eq!(identity.requested_quality, PreviewQuality::Full);
    assert_eq!(identity.actual_quality, PreviewQuality::Full);
    assert!(staged.last_eval_stats().cold_execution);
    assert_eq!(staged.last_eval_stats().reused_contributions, 0);
    assert_eq!(staged.last_eval_stats().operations_reused, 0);

    let staged_field = staged
        .readback_current(&gpu.device, &gpu.queue)
        .expect("staged Full readback")
        .to_dense();
    let mut cold = GpuTerrainEngine::new(&gpu.device, resolution);
    cold.mark_all_dirty(&stack);
    let cold_field = cold
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Full,
            true,
            None,
        )
        .expect("cold Full evaluation")
        .cpu
        .expect("cold Full readback")
        .to_dense();
    assert_eq!(
        terra_gpu::parity::max_abs_diff(&staged_field, &cold_field),
        0.0
    );
}
