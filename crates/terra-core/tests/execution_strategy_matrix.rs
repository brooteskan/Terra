//! Checked-in execution-strategy completeness ratchet for issue #177.

use terra_core::layer::{
    EffectFilterKind, EffectFilterParams, Layer, LayerKind, LayerStack, LayerTypeRegistry,
};
use terra_core::terrain_plan::{
    compile_terrain_plan, resolve_plan_execution_strategy, PlanStructureRevision,
    TerrainPlanExecutionStrategy, TerrainPlanStamp,
};

fn strategy_for(kind: LayerKind) -> TerrainPlanExecutionStrategy {
    let mut stack = LayerStack::new();
    stack.push(Layer::new("matrix", kind));
    let plan = compile_terrain_plan(
        &stack,
        &[],
        TerrainPlanStamp::new(PlanStructureRevision::new(177)),
    )
    .expect("every registered operation must compile for strategy classification");
    resolve_plan_execution_strategy(&plan, plan.final_height())
}

#[test]
fn every_registered_layer_has_an_execution_strategy() {
    let registry = LayerTypeRegistry::builtin();
    for metadata in registry.all() {
        let layer = registry
            .create(metadata.type_id)
            .unwrap_or_else(|| panic!("{} has no factory", metadata.type_id));
        let strategy = strategy_for(layer.kind);
        match strategy {
            TerrainPlanExecutionStrategy::Local(slice) => assert!(
                !slice.operations.is_empty(),
                "{} classified local without executable work",
                metadata.type_id
            ),
            TerrainPlanExecutionStrategy::Checkpointed { checkpoint, .. } => assert!(
                !checkpoint.blockers.is_empty(),
                "{} classified checkpointed without a global blocker",
                metadata.type_id
            ),
        }
    }
}

#[test]
fn every_effect_filter_subkind_has_an_execution_strategy() {
    for kind in EffectFilterKind::ALL {
        let strategy = strategy_for(LayerKind::EffectFilter(EffectFilterParams {
            kind: *kind,
            ..EffectFilterParams::default()
        }));
        assert!(
            matches!(
                strategy,
                TerrainPlanExecutionStrategy::Local(_)
                    | TerrainPlanExecutionStrategy::Checkpointed { .. }
            ),
            "unclassified effect filter {kind:?}"
        );
    }
}

#[test]
fn representative_basin_operation_is_checkpointed() {
    assert!(matches!(
        strategy_for(LayerKind::RiverCarve(Default::default())),
        TerrainPlanExecutionStrategy::Checkpointed { .. }
    ));
}
