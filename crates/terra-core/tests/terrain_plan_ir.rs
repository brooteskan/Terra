use terra_core::deps::NodeRef;
use terra_core::field_data::FieldId;
use terra_core::ids::{LayerId, OutputId};
use terra_core::invalidation::Reach;
use terra_core::terrain_plan::{
    CompiledTerrainPlan, GroupCompositeMode, LogicalFieldKind, PlanDirtyScope, PlanOrigin,
    PlanStructureRevision, SeedSource, TerrainEditClass, TerrainOp, TerrainOpKind,
    TerrainPlanBuilder, TerrainPlanStamp,
};
use terra_core::tiling::UvRect;

fn owner(id: LayerId) -> PlanOrigin {
    PlanOrigin::Authored(NodeRef::Layer(id))
}

fn group_owner(id: LayerId) -> PlanOrigin {
    PlanOrigin::Authored(NodeRef::Group(id))
}

fn flat_plan(revision: u64) -> (CompiledTerrainPlan, LayerId) {
    let base = LayerId::from_u128(1);
    let mut builder =
        TerrainPlanBuilder::new(TerrainPlanStamp::new(PlanStructureRevision::new(revision)));
    let root = builder.add_field(LogicalFieldKind::Height, PlanOrigin::Root);
    let candidate = builder.add_field(LogicalFieldKind::Height, owner(base));
    let mask = builder.add_field(LogicalFieldKind::Mask, owner(base));
    let composed = builder.add_field(LogicalFieldKind::Height, owner(base));

    builder.add_operation(TerrainOp {
        origin: PlanOrigin::Root,
        reach: Reach::LOCAL,
        aux_reach: terra_core::invalidation::AuxReach::HeightOnly,
        kind: TerrainOpKind::Seed {
            source: SeedSource::Zero,
            output: root,
        },
    });
    builder.add_operation(TerrainOp {
        origin: owner(base),
        reach: Reach::LOCAL,
        aux_reach: terra_core::invalidation::AuxReach::HeightOnly,
        kind: TerrainOpKind::RunLayerKernel {
            layer: base,
            type_id: "sculpt_base".into(),
            input_height: root,
            input_fields: Vec::new(),
            output_candidate: candidate,
            output_fields: Vec::new(),
        },
    });
    builder.add_operation(TerrainOp {
        origin: owner(base),
        reach: Reach::LOCAL,
        aux_reach: terra_core::invalidation::AuxReach::HeightOnly,
        kind: TerrainOpKind::EvaluateMask {
            input_height: root,
            input_fields: Vec::new(),
            output_mask: mask,
        },
    });
    builder.add_operation(TerrainOp {
        origin: owner(base),
        reach: Reach::LOCAL,
        aux_reach: terra_core::invalidation::AuxReach::HeightOnly,
        kind: TerrainOpKind::CompositeLayer {
            layer: base,
            base: root,
            candidate,
            mask,
            output: composed,
        },
    });

    (builder.finish(composed).expect("valid flat plan"), base)
}

#[test]
fn flat_plan_has_ordered_dataflow_and_bidirectional_provenance() {
    let (plan, base) = flat_plan(7);
    assert_eq!(plan.fields().len(), 4);
    assert_eq!(plan.operations().len(), 4);
    assert_eq!(plan.final_height().index(), 3);
    let base_operation_indices: Vec<_> = plan
        .provenance()
        .operations_for(NodeRef::Layer(base))
        .iter()
        .map(|id| id.index())
        .collect();
    assert_eq!(base_operation_indices, vec![1, 2, 3]);
    let producer = plan
        .provenance()
        .producer_of(plan.final_height())
        .expect("final height producer");
    assert_eq!(producer.index(), 3);
    assert_eq!(
        plan.provenance().owner_of(producer),
        Some(NodeRef::Layer(base))
    );
}

#[test]
fn isolated_copy_input_plan_keeps_private_and_parent_fields_distinct() {
    let group = LayerId::from_u128(10);
    let child = LayerId::from_u128(11);
    let output_id = OutputId::new();
    let mut builder = TerrainPlanBuilder::new(TerrainPlanStamp::new(PlanStructureRevision::new(3)));
    let parent = builder.add_field(LogicalFieldKind::Height, PlanOrigin::Root);
    let private_seed = builder.add_field(LogicalFieldKind::Height, group_owner(group));
    let child_candidate = builder.add_field(LogicalFieldKind::Height, owner(child));
    let child_output = builder.add_field(LogicalFieldKind::Height, owner(child));
    let child_mask = builder.add_field(LogicalFieldKind::Mask, owner(child));
    let group_mask = builder.add_field(LogicalFieldKind::Mask, group_owner(group));
    let group_output = builder.add_field(LogicalFieldKind::Height, group_owner(group));

    builder.add_operation(TerrainOp {
        origin: PlanOrigin::Root,
        reach: Reach::LOCAL,
        aux_reach: terra_core::invalidation::AuxReach::HeightOnly,
        kind: TerrainOpKind::Seed {
            source: SeedSource::Zero,
            output: parent,
        },
    });
    builder.add_operation(TerrainOp {
        origin: group_owner(group),
        reach: Reach::LOCAL,
        aux_reach: terra_core::invalidation::AuxReach::HeightOnly,
        kind: TerrainOpKind::Seed {
            source: SeedSource::Copy(parent),
            output: private_seed,
        },
    });
    builder.add_operation(TerrainOp {
        origin: owner(child),
        reach: Reach::LOCAL,
        aux_reach: terra_core::invalidation::AuxReach::HeightOnly,
        kind: TerrainOpKind::RunLayerKernel {
            layer: child,
            type_id: "volcano".into(),
            input_height: private_seed,
            input_fields: Vec::new(),
            output_candidate: child_candidate,
            output_fields: Vec::new(),
        },
    });
    builder.add_operation(TerrainOp {
        origin: owner(child),
        reach: Reach::LOCAL,
        aux_reach: terra_core::invalidation::AuxReach::HeightOnly,
        kind: TerrainOpKind::EvaluateMask {
            input_height: private_seed,
            input_fields: Vec::new(),
            output_mask: child_mask,
        },
    });
    builder.add_operation(TerrainOp {
        origin: owner(child),
        reach: Reach::LOCAL,
        aux_reach: terra_core::invalidation::AuxReach::HeightOnly,
        kind: TerrainOpKind::CompositeLayer {
            layer: child,
            base: private_seed,
            candidate: child_candidate,
            mask: child_mask,
            output: child_output,
        },
    });
    builder.add_operation(TerrainOp {
        origin: group_owner(group),
        reach: Reach::LOCAL,
        aux_reach: terra_core::invalidation::AuxReach::HeightOnly,
        kind: TerrainOpKind::EvaluateMask {
            input_height: parent,
            input_fields: Vec::new(),
            output_mask: group_mask,
        },
    });
    builder.add_operation(TerrainOp {
        origin: group_owner(group),
        reach: Reach::LOCAL,
        aux_reach: terra_core::invalidation::AuxReach::HeightOnly,
        kind: TerrainOpKind::CompositeGroup {
            group,
            parent,
            private_seed,
            child_output,
            mask: group_mask,
            output: group_output,
            mode: GroupCompositeMode::BiomeHeightDelta,
            aux: Vec::new(),
        },
    });
    builder.add_operation(TerrainOp {
        origin: PlanOrigin::Authored(NodeRef::Output(output_id)),
        reach: Reach::LOCAL,
        aux_reach: terra_core::invalidation::AuxReach::HeightOnly,
        kind: TerrainOpKind::PublishOutput {
            output: output_id,
            source: group_output,
        },
    });

    let plan = builder.finish(group_output).expect("valid isolated plan");
    assert_ne!(parent, private_seed);
    assert_eq!(
        plan.provenance().field_for_output(output_id),
        Some(group_output)
    );
    let group_ops = plan.provenance().operations_for(NodeRef::Group(group));
    assert_eq!(group_ops.len(), 3);
    assert!(matches!(
        &plan.operations()[6].kind,
        TerrainOpKind::CompositeGroup {
            mode: GroupCompositeMode::BiomeHeightDelta,
            ..
        }
    ));
}

#[test]
fn structural_revision_is_separate_from_nonstructural_edit_work() {
    let (plan, base) = flat_plan(9);
    assert!(plan.matches_structure_revision(PlanStructureRevision::new(9)));
    assert!(!plan.matches_structure_revision(PlanStructureRevision::new(10)));

    let content = TerrainEditClass::Content {
        owner: NodeRef::Layer(base),
        fields: vec![FieldId::Height],
        scope: PlanDirtyScope::Region(UvRect::from_center_radius(0.5, 0.5, 0.05)),
    }
    .required_work();
    assert!(content.patch_content);
    assert!(!content.compile_structure);

    let parameters = TerrainEditClass::Parameters {
        owner: NodeRef::Layer(base),
    }
    .required_work();
    assert!(parameters.patch_parameters);
    assert!(!parameters.compile_structure);

    let structural = TerrainEditClass::Structure.required_work();
    assert!(structural.compile_structure);
    assert!(!structural.patch_content);

    let resources = TerrainEditClass::Resources.required_work();
    assert!(resources.realize_resources);
    assert!(!resources.compile_structure);

    assert!(TerrainEditClass::ViewOnly.required_work().is_empty());
    let merged = content.merge(resources);
    assert!(merged.patch_content && merged.realize_resources);
}
