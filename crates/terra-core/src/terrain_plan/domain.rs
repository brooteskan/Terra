use std::collections::{HashMap, HashSet, VecDeque};

use crate::deps::NodeRef;
use crate::invalidation::{AuxReach, Reach};

use super::{CompiledTerrainPlan, FieldSlot, LogicalFieldKind, PlanOpId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainPlanDomainRejectReason {
    FullReach,
    GlobalAuxiliary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerrainPlanDomainRejection {
    pub operation: PlanOpId,
    pub owner: Option<NodeRef>,
    pub field: FieldSlot,
    pub reason: TerrainPlanDomainRejectReason,
}

/// Ordered live plan slice and guard radius required to produce one field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerrainPlanDomainSlice {
    pub output: FieldSlot,
    pub operations: Vec<PlanOpId>,
    pub operation_halo: u32,
}

/// Resolve the local plan slice required for `output`.
/// Sequential reaches add while converging dependency branches take their maximum.
pub fn resolve_plan_domain(
    plan: &CompiledTerrainPlan,
    output: FieldSlot,
) -> Result<TerrainPlanDomainSlice, TerrainPlanDomainRejection> {
    let mut required = HashMap::<FieldSlot, u32>::new();
    let mut pending = VecDeque::from([(output, 0u32)]);
    let mut selected = HashSet::new();
    let mut maximum = 0u32;

    while let Some((field, downstream_halo)) = pending.pop_front() {
        if required
            .get(&field)
            .is_some_and(|known| *known >= downstream_halo)
        {
            continue;
        }
        required.insert(field, downstream_halo);
        maximum = maximum.max(downstream_halo);
        let producer = plan
            .provenance()
            .producer_of(field)
            .expect("validated plan fields have producers");
        let operation = plan.operation(producer).expect("validated plan operation");
        if matches!(
            plan.field(field).map(|field| &field.kind),
            Some(LogicalFieldKind::Auxiliary(_))
        ) && operation.aux_reach == AuxReach::Global
        {
            return Err(TerrainPlanDomainRejection {
                operation: producer,
                owner: plan.provenance().owner_of(producer),
                field,
                reason: TerrainPlanDomainRejectReason::GlobalAuxiliary,
            });
        }
        let input_halo = match operation.reach {
            Reach::Full => {
                return Err(TerrainPlanDomainRejection {
                    operation: producer,
                    owner: plan.provenance().owner_of(producer),
                    field,
                    reason: TerrainPlanDomainRejectReason::FullReach,
                });
            }
            Reach::Localized { halo_samples } => downstream_halo.saturating_add(halo_samples),
        };
        selected.insert(producer);
        for input in plan.analysis().inputs(producer) {
            pending.push_back((*input, input_halo));
        }
        maximum = maximum.max(input_halo);
    }

    let mut operations: Vec<_> = selected.into_iter().collect();
    operations.sort_by_key(|operation| operation.index());
    Ok(TerrainPlanDomainSlice {
        output,
        operations,
        operation_halo: maximum,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_data::FieldId;
    use crate::ids::LayerId;
    use crate::terrain_plan::{
        LogicalFieldKind, PlanOrigin, PlanStructureRevision, SeedSource, TerrainOp, TerrainOpKind,
        TerrainPlanBuilder, TerrainPlanStamp,
    };

    fn local_chain(second: Reach, aux_reach: AuxReach, consume_aux: bool) -> CompiledTerrainPlan {
        let first_owner = LayerId::from_u128(1);
        let second_owner = LayerId::from_u128(2);
        let mut builder =
            TerrainPlanBuilder::new(TerrainPlanStamp::new(PlanStructureRevision::new(9)));
        let root = builder.add_field(LogicalFieldKind::Height, PlanOrigin::Root);
        let first = builder.add_field(
            LogicalFieldKind::Height,
            PlanOrigin::Authored(NodeRef::Layer(first_owner)),
        );
        let aux = builder.add_field(
            LogicalFieldKind::Auxiliary(FieldId::Hardness),
            PlanOrigin::Authored(NodeRef::Layer(first_owner)),
        );
        let output = builder.add_field(
            LogicalFieldKind::Height,
            PlanOrigin::Authored(NodeRef::Layer(second_owner)),
        );
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
            origin: PlanOrigin::Authored(NodeRef::Layer(first_owner)),
            reach: Reach::Localized { halo_samples: 3 },
            aux_reach,
            kind: TerrainOpKind::RunLayerKernel {
                layer: first_owner,
                type_id: "first".into(),
                input_height: root,
                input_fields: Vec::new(),
                output_candidate: first,
                output_fields: vec![aux],
            },
        });
        builder.add_operation(TerrainOp {
            origin: PlanOrigin::Authored(NodeRef::Layer(second_owner)),
            reach: second,
            aux_reach: AuxReach::HeightOnly,
            kind: TerrainOpKind::RunLayerKernel {
                layer: second_owner,
                type_id: "second".into(),
                input_height: first,
                input_fields: consume_aux.then_some(aux).into_iter().collect(),
                output_candidate: output,
                output_fields: Vec::new(),
            },
        });
        builder.finish(output).unwrap()
    }

    #[test]
    fn sequential_local_reach_adds() {
        let plan = local_chain(
            Reach::Localized { halo_samples: 5 },
            AuxReach::PerTexel,
            false,
        );
        let slice = resolve_plan_domain(&plan, plan.final_height()).unwrap();
        assert_eq!(slice.operation_halo, 8);
        assert_eq!(slice.operations.len(), 3);
    }

    #[test]
    fn full_reach_rejects_at_owning_operation() {
        let plan = local_chain(Reach::Full, AuxReach::PerTexel, false);
        let rejection = resolve_plan_domain(&plan, plan.final_height()).unwrap_err();
        assert_eq!(rejection.operation.index(), 2);
        assert_eq!(rejection.reason, TerrainPlanDomainRejectReason::FullReach);
    }

    #[test]
    fn consumed_global_auxiliary_rejects() {
        let plan = local_chain(Reach::LOCAL, AuxReach::Global, true);
        let rejection = resolve_plan_domain(&plan, plan.final_height()).unwrap_err();
        assert_eq!(rejection.operation.index(), 1);
        assert_eq!(
            rejection.reason,
            TerrainPlanDomainRejectReason::GlobalAuxiliary
        );
    }
}
