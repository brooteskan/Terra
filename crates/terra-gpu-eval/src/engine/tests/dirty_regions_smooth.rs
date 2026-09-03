use super::*;

/// A wide Smooth stroke changes the configured reach of the already-compiled
/// SculptStrokes kernel. Compare the first interactive publication, before any
/// quality refinement can hide a clipped patch, with a cold realization of the
/// same authored tree.
#[test]
fn untitled6_wide_smooth_first_publication_matches_cold_evaluation() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 256u32;
    let (mut document, ids) = untitled6_document(res, Untitled6Variant::ProductionTopology);
    document.metrics.tile_size = 32;
    document.metrics.halo = 2;

    let mut cache = TerrainPlanCache::new();
    let cold = cache
        .update(
            &document.stack,
            &document.masks,
            &[TerrainEditClass::Structure],
        )
        .expect("compile fixture");
    let mut warm_engine = GpuTerrainEngine::new(&gpu.device, res);
    warm_engine
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
        .expect("prime production topology");

    document
        .stack
        .find_mut(ids.semantic_sculpt)
        .expect("semantic sculpt layer")
        .apply_brush(
            SculptStrokeKind::Smooth,
            BrushDab {
                u: 0.52,
                v: 0.52,
                radius_uv: 0.05,
                radius_m: 0.05 * document.metrics.world_size_x,
                strength: 1.0,
                target_height: 8.0,
                riser_width_m: 0.0,
                falloff: 0.1,
                continuing: false,
            },
        );
    let edit = TerrainEditClass::Content {
        owner: NodeRef::Layer(ids.semantic_sculpt),
        fields: vec![FieldId::Height],
        scope: PlanDirtyScope::Region(UvRect::from_center_radius(0.52, 0.52, 0.05)),
    };
    let invalidation = cache
        .update(&document.stack, &document.masks, &[edit])
        .expect("patch wide Smooth stroke");
    let interactive = warm_engine
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
        .expect("publish first wide Smooth result");
    assert!(interactive.output_identity.is_some());
    let warm = warm_engine
        .readback_current(&gpu.device, &gpu.queue)
        .expect("interactive readback");

    let mut cold_cache = TerrainPlanCache::new();
    let cold_invalidation = cold_cache
        .update(
            &document.stack,
            &document.masks,
            &[TerrainEditClass::Structure],
        )
        .expect("compile cold oracle");
    let mut cold_engine = GpuTerrainEngine::new(&gpu.device, res);
    cold_engine
        .evaluate_compiled_with_intent(
            &gpu.device,
            &gpu.queue,
            &document.stack,
            &document.masks,
            cold_cache.current_plan().unwrap(),
            cold_cache.structure_revision(),
            &cold_invalidation,
            document.metrics,
            PreviewQuality::Draft,
            false,
            GpuEvaluationIntent::Complete,
        )
        .expect("cold wide Smooth result");
    let cold = cold_engine
        .readback_current(&gpu.device, &gpu.queue)
        .expect("cold readback");

    let error = terra_gpu::parity::max_abs_diff(&warm.to_dense(), &cold.to_dense());
    assert!(
        error <= 1.0e-3,
        "first wide Smooth publication left a rectangular seam (max error {error})"
    );
}
