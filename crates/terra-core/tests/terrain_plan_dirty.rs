use terra_core::deps::NodeRef;
use terra_core::field_data::FieldId;
use terra_core::ids::{LayerId, OutputId};
use terra_core::invalidation::{AuxReach, Reach};
use terra_core::layer::{
    Layer, LayerKind, LayerStack, MaterialsParams, StreamPowerParams, VegetationParams,
};
use terra_core::terrain_plan::{
    compile_terrain_plan, propagate_plan_edits, FullFieldReason, LogicalFieldKind, PlanDirtyScope,
    PlanOrigin, PlanStructureRevision, SeedSource, TerrainEditClass, TerrainOp, TerrainOpKind,
    TerrainPlanBuilder, TerrainPlanStamp,
};
use terra_core::tiling::UvRect;

fn owner(id: LayerId) -> PlanOrigin {
    PlanOrigin::Authored(NodeRef::Layer(id))
}

fn edit(id: LayerId) -> TerrainEditClass {
    TerrainEditClass::Content {
        owner: NodeRef::Layer(id),
        fields: vec![FieldId::Height],
        scope: PlanDirtyScope::Region(UvRect::from_center_radius(0.5, 0.5, 0.01)),
    }
}

fn aux_plan(publish_aux: bool) -> (terra_core::terrain_plan::CompiledTerrainPlan, LayerId) {
    let id = LayerId::from_u128(11);
    let mut builder = TerrainPlanBuilder::new(TerrainPlanStamp::new(PlanStructureRevision::new(1)));
    let root = builder.add_field(LogicalFieldKind::Height, PlanOrigin::Root);
    let candidate = builder.add_field(LogicalFieldKind::Height, owner(id));
    let aux = builder.add_field(LogicalFieldKind::Auxiliary(FieldId::Hardness), owner(id));
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
        origin: owner(id),
        reach: Reach::Localized { halo_samples: 3 },
        aux_reach: AuxReach::Global,
        kind: TerrainOpKind::RunLayerKernel {
            layer: id,
            type_id: "fixture".into(),
            input_height: root,
            input_fields: Vec::new(),
            output_candidate: candidate,
            output_fields: vec![aux],
        },
    });
    if publish_aux {
        builder.add_operation(TerrainOp {
            origin: PlanOrigin::Authored(NodeRef::Output(OutputId::new())),
            reach: Reach::LOCAL,
            aux_reach: AuxReach::HeightOnly,
            kind: TerrainOpKind::PublishOutput {
                output: OutputId::new(),
                source: aux,
            },
        });
    }
    (builder.finish(candidate).expect("valid aux plan"), id)
}

#[test]
fn localized_reach_accumulates_without_globalizing_unobserved_aux() {
    let (plan, id) = aux_plan(false);
    let invalidation = propagate_plan_edits(&plan, &[edit(id)]);
    assert!(invalidation.first_full_field_escalation.is_none());
    let candidate = invalidation
        .dirty_fields
        .iter()
        .find(|(field, _)| matches!(plan.field(*field).unwrap().kind, LogicalFieldKind::Height))
        .expect("dirty height");
    assert_eq!(candidate.1.halo_samples, 3);
}

#[test]
fn observed_global_aux_escalates_at_its_producer() {
    let (plan, id) = aux_plan(true);
    let invalidation = propagate_plan_edits(&plan, &[edit(id)]);
    let escalation = invalidation
        .first_full_field_escalation
        .expect("observed global aux must escalate");
    assert_eq!(escalation.operation.index(), 1);
    assert_eq!(escalation.reason, FullFieldReason::ObservedGlobalAuxiliary);
    assert!(escalation.field.is_some());
}

#[test]
fn structural_invalidation_reaches_the_live_plan_once_in_order() {
    let (plan, _) = aux_plan(false);
    let invalidation = propagate_plan_edits(&plan, &[TerrainEditClass::Structure]);
    let indices: Vec<_> = invalidation
        .operations
        .iter()
        .map(|entry| entry.operation.index())
        .collect();
    assert_eq!(indices, vec![0, 1]);
    assert_eq!(
        invalidation.first_full_field_escalation.unwrap().reason,
        FullFieldReason::StructuralChange
    );
}

#[test]
fn vegetation_hardness_edit_invalidates_downstream_stream_power() {
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Materials",
        LayerKind::Materials(MaterialsParams::default()),
    ));
    let vegetation = Layer::new(
        "Vegetation",
        LayerKind::Vegetation(VegetationParams {
            root_cohesion: 0.5,
            ..VegetationParams::default()
        }),
    );
    let vegetation_id = vegetation.id();
    stack.push(vegetation);
    let stream_power = Layer::new(
        "Stream Power",
        LayerKind::StreamPowerErosion(StreamPowerParams::default()),
    );
    let stream_power_id = stream_power.id();
    stack.push(stream_power);

    let plan = compile_terrain_plan(
        &stack,
        &[],
        TerrainPlanStamp::new(PlanStructureRevision::new(9)),
    )
    .expect("vegetation hardness plan");
    let stream_power_inputs = plan
        .operations()
        .iter()
        .find_map(|operation| match &operation.kind {
            TerrainOpKind::RunLayerKernel {
                layer,
                input_fields,
                ..
            } if *layer == stream_power_id => Some(input_fields),
            _ => None,
        })
        .expect("stream-power kernel");
    let vegetation_hardness = stream_power_inputs
        .iter()
        .copied()
        .find(|slot| {
            plan.field(*slot).is_some_and(|field| {
                matches!(&field.kind, LogicalFieldKind::Auxiliary(FieldId::Hardness))
                    && field.origin.authored() == Some(NodeRef::Layer(vegetation_id))
            })
        })
        .expect("stream power consumes vegetation hardness");
    assert!(plan.analysis().field_is_live(vegetation_hardness));

    let invalidation = propagate_plan_edits(
        &plan,
        &[TerrainEditClass::Parameters {
            owner: NodeRef::Layer(vegetation_id),
        }],
    );
    assert!(invalidation.operations.iter().any(|dirty| {
        plan.operation(dirty.operation).is_some_and(|operation| {
            matches!(
                operation.kind,
                TerrainOpKind::RunLayerKernel { layer, .. } if layer == stream_power_id
            )
        })
    }));
}
