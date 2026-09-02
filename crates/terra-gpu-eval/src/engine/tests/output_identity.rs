use super::*;
use terra_gpu::output_identity::{GpuOutputCompleteness, GpuOutputCoverage};

#[test]
fn warm_outputs_chain_identity_and_cold_realization_changes_incarnation() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(64, 64, 640.0, 640.0);
    let mut stack = LayerStack::new();
    let base = Layer::new("base", LayerKind::SculptBase(SculptParams::filled(64, 5.0)));
    let base_id = base.id();
    stack.push(base);
    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);

    engine.mark_all_dirty(&stack);
    let cold = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Full,
            false,
            None,
        )
        .expect("cold output");
    let cold_identity = cold.output_identity.expect("cold identity");
    assert!(cold_identity.is_current_complete_final());
    assert_eq!(cold_identity.coverage, GpuOutputCoverage::WholeField);
    assert_eq!(cold_identity.completeness, GpuOutputCompleteness::Complete);

    let LayerKind::SculptBase(params) = &mut stack.find_mut(base_id).expect("base").kind else {
        panic!("base changed kind");
    };
    params.samples[(32 * 64 + 32) as usize] += 2.0;
    engine.set_dirty_rect(Some((31, 31, 3, 3)));
    engine.mark_dirty(base_id);
    let warm = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Full,
            false,
            None,
        )
        .expect("warm output");
    let warm_identity = warm.output_identity.expect("warm identity");
    let GpuOutputCoverage::Patch { expected_base, .. } = warm_identity.coverage else {
        panic!("warm edit must be a patch");
    };
    assert_eq!(expected_base, Some(cold_identity.output));
    assert_eq!(
        warm_identity.selected_field.resource_incarnation,
        cold_identity.selected_field.resource_incarnation
    );
    assert_eq!(
        warm_identity.output_resource.incarnation,
        cold_identity.output_resource.incarnation
    );
    assert!(warm_identity.output.0 > cold_identity.output.0);

    engine.reset_project_state(&gpu.device, &gpu.queue);
    engine.mark_all_dirty(&stack);
    let reset = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Full,
            false,
            None,
        )
        .expect("post-reset output");
    let reset_identity = reset.output_identity.expect("reset identity");
    assert!(
        reset_identity.selected_field.resource_incarnation.0
            > warm_identity.selected_field.resource_incarnation.0
    );
    assert_eq!(reset_identity.coverage, GpuOutputCoverage::WholeField);
}
