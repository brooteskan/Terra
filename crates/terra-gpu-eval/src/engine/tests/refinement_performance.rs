use super::*;

/// #154: optional refinement is split into fenced units, keeps its plan
/// resources private until publication, and produces the same field as the
/// existing complete evaluator.
#[test]
fn resumable_refinement_is_depth_one_transactional_and_matches_complete_eval() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 64u32;
    let (document, _) = untitled6_document(res, Untitled6Variant::ProductionTopology);
    let mut cache = TerrainPlanCache::new();
    let invalidation = cache
        .update(
            &document.stack,
            &document.masks,
            &[TerrainEditClass::Structure],
        )
        .expect("compile fixture");
    let plan = cache.current_plan().expect("compiled plan");
    let revision = cache.structure_revision();

    let mut complete = GpuTerrainEngine::new(&gpu.device, res);
    complete
        .evaluate_compiled_with_intent(
            &gpu.device,
            &gpu.queue,
            &document.stack,
            &document.masks,
            plan,
            revision,
            &invalidation,
            document.metrics,
            PreviewQuality::Medium,
            false,
            GpuEvaluationIntent::Complete,
        )
        .expect("complete Medium evaluation");
    let expected = complete
        .readback_current(&gpu.device, &gpu.queue)
        .expect("complete readback");

    let mut resumable = GpuTerrainEngine::new(&gpu.device, res);
    let committed_before = resumable.plan_resources.stats().committed_candidates;
    let mut job = resumable
        .begin_compiled_refinement(
            &gpu.device,
            &document.stack,
            &document.masks,
            plan,
            revision,
            &invalidation,
            document.metrics,
            PreviewQuality::Medium,
        )
        .expect("begin resumable Medium evaluation");
    assert!(resumable.plan_resources.current().is_none());
    assert!(resumable.last_graph.plans.is_empty());

    let mut submitted = 0u32;
    let mut max_depth = 0u8;
    for _ in 0..10_000 {
        let step = resumable
            .advance_compiled_refinement(&gpu.device, &gpu.queue, &mut job)
            .expect("advance resumable evaluation");
        let progress = job.progress();
        max_depth = max_depth.max(progress.submissions_in_flight);
        if matches!(step, GpuRefinementStep::Submitted { .. }) {
            submitted += 1;
            let _ = gpu.device.poll(wgpu::Maintain::Wait);
        }
        if matches!(step, GpuRefinementStep::AwaitingGpu) {
            let _ = gpu.device.poll(wgpu::Maintain::Wait);
        }
        if matches!(step, GpuRefinementStep::ReadyToPublish) {
            break;
        }
    }
    assert!(
        job.final_copy_complete,
        "refinement did not reach its final fence"
    );
    assert!(submitted > 1, "the plan should be split across submissions");
    assert_eq!(max_depth, 1, "refinement queue depth must remain bounded");
    assert!(resumable.plan_resources.current().is_none());
    assert_eq!(
        resumable.plan_resources.stats().committed_candidates,
        committed_before,
        "candidate resources must not become canonical before publication"
    );

    let result = resumable
        .publish_compiled_refinement(&gpu.device, &gpu.queue, job)
        .expect("publish fenced candidate");
    assert!(result.fully_gpu);
    assert_eq!(result.freshness, GpuPreviewFreshness::Current);
    let identity = result.output_identity.expect("refinement output identity");
    assert!(identity.is_current_complete_final());
    assert_eq!(
        identity.coverage,
        terra_gpu::output_identity::GpuOutputCoverage::WholeField
    );
    assert_eq!(
        identity.last_write.completion,
        terra_gpu::output_identity::GpuSubmissionCompletion::Submitted
    );
    assert!(identity.last_write.serial.0 > 0);
    assert!(resumable.plan_resources.current().is_some());
    assert!(!resumable.last_graph.plans.is_empty());
    assert_eq!(
        resumable.plan_resources.stats().committed_candidates,
        committed_before + 1
    );
    let actual = resumable
        .readback_current(&gpu.device, &gpu.queue)
        .expect("resumable readback");
    terra_gpu::parity::assert_field_parity(
        "resumable Medium vs complete Medium",
        &actual,
        &expected,
        terra_gpu::parity::UNTITLED6_INTERACTION,
    );

    // A stroke arriving after submission abandons the cursor but cannot
    // cancel that submission. Its fence must constrain the next generation.
    let mut stale = resumable
        .begin_compiled_refinement(
            &gpu.device,
            &document.stack,
            &document.masks,
            plan,
            revision,
            &invalidation,
            document.metrics,
            PreviewQuality::Full,
        )
        .expect("begin stale Full generation");
    loop {
        if matches!(
            resumable
                .advance_compiled_refinement(&gpu.device, &gpu.queue, &mut stale)
                .expect("advance stale generation"),
            GpuRefinementStep::Submitted { .. }
        ) {
            break;
        }
    }
    resumable.abandon_compiled_refinement(stale);
    let fresh = resumable
        .begin_compiled_refinement(
            &gpu.device,
            &document.stack,
            &document.masks,
            plan,
            revision,
            &invalidation,
            document.metrics,
            PreviewQuality::Medium,
        )
        .expect("begin replacement generation");
    assert_eq!(
        resumable.refinement_submissions_in_flight(&fresh),
        1,
        "the abandoned submission fence must remain in the global depth bound"
    );
    resumable.abandon_compiled_refinement(fresh);
    let _ = gpu.device.poll(wgpu::Maintain::Wait);
}

/// Manual release probe for the acceptance resolutions. Adapter timing is
/// reported, not asserted, because CI hardware is intentionally variable.
#[test]
#[ignore = "run in release mode to record #150/#152 adapter timings"]
fn untitled6_release_timing_probe_2048_4096() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    for res in [2048u32, 4096] {
        let (mut document, ids) = untitled6_document(res, Untitled6Variant::ProductionTopology);
        let mut cache = TerrainPlanCache::new();
        let cold_invalidation = cache
            .update(
                &document.stack,
                &document.masks,
                &[TerrainEditClass::Structure],
            )
            .unwrap();
        let mut engine = GpuTerrainEngine::new(&gpu.device, res);
        let cold_started = std::time::Instant::now();
        let cold_result = engine
            .evaluate_compiled_with_intent(
                &gpu.device,
                &gpu.queue,
                &document.stack,
                &document.masks,
                cache.current_plan().unwrap(),
                cache.structure_revision(),
                &cold_invalidation,
                document.metrics,
                PreviewQuality::Draft,
                false,
                GpuEvaluationIntent::Complete,
            )
            .unwrap();
        let _ = gpu.device.poll(wgpu::Maintain::Wait);
        let cold_ms = cold_started.elapsed().as_secs_f64() * 1000.0;
        let cold_stats = engine.last_eval_stats();
        let cold_plan_stats = cache.stats().snapshot();
        assert!(cold_result.fully_gpu);
        assert!(cold_result.cpu_fallback.is_none());
        assert_eq!(cold_stats.readback_bytes, 0);

        document.stack.find_mut(ids.base).unwrap().apply_brush(
            SculptStrokeKind::Raise,
            BrushDab {
                u: 0.5,
                v: 0.5,
                radius_uv: 0.01,
                radius_m: 40.0,
                strength: 3.0,
                target_height: 0.0,
                riser_width_m: 0.0,
                falloff: 0.5,
                continuing: false,
            },
        );
        let invalidation = cache
            .update(
                &document.stack,
                &document.masks,
                &[TerrainEditClass::Content {
                    owner: NodeRef::Layer(ids.base),
                    fields: vec![FieldId::Height],
                    scope: PlanDirtyScope::Region(UvRect::from_center_radius(0.5, 0.5, 0.01)),
                }],
            )
            .unwrap();
        let warm_started = std::time::Instant::now();
        let warm_result = engine
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
            .unwrap();
        let _ = gpu.device.poll(wgpu::Maintain::Wait);
        let warm_ms = warm_started.elapsed().as_secs_f64() * 1000.0;
        let warm_stats = engine.last_eval_stats();
        let warm_plan_stats = cache.stats().snapshot();
        assert!(warm_result.fully_gpu);
        assert!(warm_result.cpu_fallback.is_none());
        assert_eq!(warm_stats.readback_bytes, 0);

        let mut transition_full_ms = Vec::with_capacity(10);
        for sample in 0..10 {
            let draft = engine
                .evaluate_compiled_with_intent(
                    &gpu.device,
                    &gpu.queue,
                    &document.stack,
                    &document.masks,
                    cache.current_plan().unwrap(),
                    cache.structure_revision(),
                    &PlanInvalidation::default(),
                    document.metrics,
                    PreviewQuality::Draft,
                    false,
                    GpuEvaluationIntent::Complete,
                )
                .unwrap();
            let _ = gpu.device.poll(wgpu::Maintain::Wait);
            assert!(draft.fully_gpu);
            document.stack.find_mut(ids.base).unwrap().apply_brush(
                SculptStrokeKind::Raise,
                BrushDab {
                    u: 0.40 + sample as f32 * 0.002,
                    v: 0.5,
                    radius_uv: 0.01,
                    radius_m: 40.0,
                    strength: 0.25,
                    target_height: 0.0,
                    riser_width_m: 0.0,
                    falloff: 0.5,
                    continuing: false,
                },
            );
            let transition_invalidation = cache
                .update(
                    &document.stack,
                    &document.masks,
                    &[TerrainEditClass::Content {
                        owner: NodeRef::Layer(ids.base),
                        fields: vec![FieldId::Height],
                        scope: PlanDirtyScope::Region(UvRect::from_center_radius(
                            0.40 + sample as f32 * 0.002,
                            0.5,
                            0.01,
                        )),
                    }],
                )
                .unwrap();
            let started = std::time::Instant::now();
            let full = engine
                .evaluate_compiled_with_intent(
                    &gpu.device,
                    &gpu.queue,
                    &document.stack,
                    &document.masks,
                    cache.current_plan().unwrap(),
                    cache.structure_revision(),
                    &transition_invalidation,
                    document.metrics,
                    PreviewQuality::Full,
                    false,
                    GpuEvaluationIntent::Complete,
                )
                .unwrap();
            let _ = gpu.device.poll(wgpu::Maintain::Wait);
            assert!(full.fully_gpu);
            transition_full_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        transition_full_ms.sort_by(f64::total_cmp);
        let transition_p50 = transition_full_ms[(transition_full_ms.len() * 50).div_ceil(100) - 1];
        let transition_p95 = transition_full_ms[(transition_full_ms.len() * 95).div_ceil(100) - 1];
        let transition_max = *transition_full_ms.last().unwrap();

        let mut full_ms = Vec::with_capacity(20);
        for sample in 0..20 {
            document.stack.find_mut(ids.base).unwrap().apply_brush(
                SculptStrokeKind::Raise,
                BrushDab {
                    u: 0.45 + sample as f32 * 0.002,
                    v: 0.5,
                    radius_uv: 0.01,
                    radius_m: 40.0,
                    strength: 0.25,
                    target_height: 0.0,
                    riser_width_m: 0.0,
                    falloff: 0.5,
                    continuing: false,
                },
            );
            let full_invalidation = cache
                .update(
                    &document.stack,
                    &document.masks,
                    &[TerrainEditClass::Content {
                        owner: NodeRef::Layer(ids.base),
                        fields: vec![FieldId::Height],
                        scope: PlanDirtyScope::Region(UvRect::from_center_radius(
                            0.45 + sample as f32 * 0.002,
                            0.5,
                            0.01,
                        )),
                    }],
                )
                .unwrap();
            let started = std::time::Instant::now();
            let full = engine
                .evaluate_compiled_with_intent(
                    &gpu.device,
                    &gpu.queue,
                    &document.stack,
                    &document.masks,
                    cache.current_plan().unwrap(),
                    cache.structure_revision(),
                    &full_invalidation,
                    document.metrics,
                    PreviewQuality::Full,
                    false,
                    GpuEvaluationIntent::Complete,
                )
                .unwrap();
            let _ = gpu.device.poll(wgpu::Maintain::Wait);
            assert!(full.fully_gpu);
            assert_eq!(engine.last_eval_stats().readback_bytes, 0);
            full_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        full_ms.sort_by(f64::total_cmp);
        let p50 = full_ms[(full_ms.len() * 50).div_ceil(100) - 1];
        let p95 = full_ms[(full_ms.len() * 95).div_ceil(100) - 1];
        let max = *full_ms.last().unwrap();
        println!(
                "issue150_152 resolution={res} adapter={:?} cold_ms={cold_ms:.3} warm_ms={warm_ms:.3} transition_n={} transition_full_p50_ms={transition_p50:.3} transition_full_p95_ms={transition_p95:.3} transition_full_max_ms={transition_max:.3} resident_n={} resident_full_p50_ms={p50:.3} resident_full_p95_ms={p95:.3} resident_full_max_ms={max:.3} cold_stats={cold_stats:?} warm_stats={warm_stats:?} full_stats={:?} cold_plan_stats={cold_plan_stats:?} warm_plan_stats={warm_plan_stats:?}",
                gpu.adapter_info,
                transition_full_ms.len(),
                full_ms.len(),
                engine.last_eval_stats(),
            );
    }
}
