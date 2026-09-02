use terra_core::deps::NodeRef;
use terra_core::ids::LayerId;
use terra_core::invalidation::{AuxReach, Reach};
use terra_core::terrain_plan::{
    ExpectedFieldKind, LogicalFieldKind, PlanBuildError, PlanOrigin, PlanStructureRevision,
    SeedSource, TerrainOp, TerrainOpKind, TerrainPlanBuilder, TerrainPlanStamp,
};

fn stamp() -> TerrainPlanStamp {
    TerrainPlanStamp::new(PlanStructureRevision::new(4))
}

#[test]
fn validation_rejects_use_before_produce_with_operation_context() {
    let owner = PlanOrigin::Authored(NodeRef::Layer(LayerId::from_u128(1)));
    let mut builder = TerrainPlanBuilder::new(stamp());
    let first = builder.add_field(LogicalFieldKind::Height, owner);
    let future = builder.add_field(LogicalFieldKind::Height, owner);
    builder.add_operation(TerrainOp {
        origin: owner,
        reach: Reach::LOCAL,
        aux_reach: AuxReach::HeightOnly,
        kind: TerrainOpKind::Seed {
            source: SeedSource::Copy(future),
            output: first,
        },
    });
    builder.add_operation(TerrainOp {
        origin: owner,
        reach: Reach::LOCAL,
        aux_reach: AuxReach::HeightOnly,
        kind: TerrainOpKind::Seed {
            source: SeedSource::Zero,
            output: future,
        },
    });

    assert!(matches!(
        builder.finish(first),
        Err(PlanBuildError::UseBeforeProduce {
            field,
            producer,
            consumer,
            owner: Some(NodeRef::Layer(_)),
        }) if field == future && producer.index() == 1 && consumer.index() == 0
    ));
}

#[test]
fn validation_rejects_unproduced_fields_and_kind_mismatches() {
    let mut unproduced = TerrainPlanBuilder::new(stamp());
    let root = unproduced.add_field(LogicalFieldKind::Height, PlanOrigin::Root);
    let orphan = unproduced.add_field(LogicalFieldKind::Mask, PlanOrigin::Root);
    unproduced.add_operation(TerrainOp {
        origin: PlanOrigin::Root,
        reach: Reach::LOCAL,
        aux_reach: AuxReach::HeightOnly,
        kind: TerrainOpKind::Seed {
            source: SeedSource::Zero,
            output: root,
        },
    });
    assert!(matches!(
        unproduced.finish(root),
        Err(PlanBuildError::UnproducedField { slot, .. }) if slot == orphan.index()
    ));

    let mut wrong_kind = TerrainPlanBuilder::new(stamp());
    let mask = wrong_kind.add_field(LogicalFieldKind::Mask, PlanOrigin::Root);
    wrong_kind.add_operation(TerrainOp {
        origin: PlanOrigin::Root,
        reach: Reach::LOCAL,
        aux_reach: AuxReach::HeightOnly,
        kind: TerrainOpKind::Seed {
            source: SeedSource::Zero,
            output: mask,
        },
    });
    assert!(matches!(
        wrong_kind.finish(mask),
        Err(PlanBuildError::InvalidFinalHeightKind { .. })
            | Err(PlanBuildError::FieldKindMismatch {
                expected: ExpectedFieldKind::Height,
                ..
            })
    ));
}

#[test]
fn validated_analysis_reports_consumers_liveness_and_lifetimes() {
    let mut builder = TerrainPlanBuilder::new(stamp());
    let root = builder.add_field(LogicalFieldKind::Height, PlanOrigin::Root);
    let output = builder.add_field(LogicalFieldKind::Height, PlanOrigin::Root);
    builder.add_operation(TerrainOp {
        origin: PlanOrigin::Root,
        reach: Reach::LOCAL,
        aux_reach: AuxReach::HeightOnly,
        kind: TerrainOpKind::Seed {
            source: SeedSource::Zero,
            output: root,
        },
    });
    builder.add_operation(TerrainOp {
        origin: PlanOrigin::Root,
        reach: Reach::LOCAL,
        aux_reach: AuxReach::HeightOnly,
        kind: TerrainOpKind::Seed {
            source: SeedSource::Copy(root),
            output,
        },
    });
    let plan = builder.finish(output).expect("valid plan");
    assert_eq!(plan.analysis().consumers(root)[0].index(), 1);
    assert!(plan.analysis().field_is_live(root));
    assert_eq!(
        plan.analysis()
            .lifetime(root)
            .unwrap()
            .last_operation
            .index(),
        1
    );
}
