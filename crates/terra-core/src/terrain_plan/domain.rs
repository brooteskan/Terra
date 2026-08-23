use std::collections::{HashMap, HashSet, VecDeque};

use crate::deps::NodeRef;
use crate::invalidation::{AuxReach, Reach, SpatialRejectReason};
use crate::layer::LayerStack;
use crate::mask::MaskAsset;

use super::{CompiledTerrainPlan, FieldSlot, LogicalFieldKind, PlanInvalidation, PlanOpId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainPlanDomainRejectReason {
    FullReach,
    GlobalAuxiliary,
    Infinite(SpatialRejectReason),
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

/// Immutable full-field boundary shared by all tile evaluations of a revision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerrainPlanCheckpoint {
    /// Half-open operation cut: prefix operations have indices below this value.
    pub boundary: usize,
    pub prefix_operations: Vec<PlanOpId>,
    /// Values crossing from the complete-field prefix into the local suffix.
    pub frontier_fields: Vec<FieldSlot>,
    /// Full/global operations which require this checkpoint.
    pub blockers: Vec<TerrainPlanDomainRejection>,
}

impl TerrainPlanCheckpoint {
    /// Whether authored work for a new content revision intersects this prefix.
    /// Schedulers use this before publishing a checkpoint under the new stamp.
    pub fn is_invalidated_by(&self, invalidation: &PlanInvalidation) -> bool {
        invalidation
            .patched_operations
            .iter()
            .any(|operation| operation.index() < self.boundary)
            || invalidation
                .operations
                .iter()
                .any(|dirty| dirty.operation.index() < self.boundary)
    }
}

/// Backend-neutral execution decision for one requested output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerrainPlanExecutionStrategy {
    Local(TerrainPlanDomainSlice),
    Checkpointed {
        checkpoint: TerrainPlanCheckpoint,
        suffix: TerrainPlanDomainSlice,
    },
}

/// Resolve a complete execution strategy for `output`.
///
/// Any live full-reach or consumed global-auxiliary operation is placed in one
/// complete-field prefix. The cut advances through the owning authored span so
/// a layer's kernel and composite are never split across execution modes.
pub fn resolve_plan_execution_strategy(
    plan: &CompiledTerrainPlan,
    output: FieldSlot,
) -> TerrainPlanExecutionStrategy {
    let live = dependency_operations(plan, output);
    let mut blockers = live
        .iter()
        .copied()
        .filter_map(|operation| checkpoint_blocker(plan, operation))
        .collect::<Vec<_>>();
    blockers.sort_by_key(|blocker| blocker.operation.index());

    if blockers.is_empty() {
        return TerrainPlanExecutionStrategy::Local(
            resolve_plan_domain(plan, output).expect("local strategy passed domain analysis"),
        );
    }

    let boundary = blockers.iter().fold(0usize, |current, blocker| {
        let operation_end = blocker.operation.index().saturating_add(1);
        let owner_end = blocker.owner.map_or(operation_end, |owner| {
            plan.provenance()
                .spans_for(owner)
                .iter()
                .filter(|span| {
                    span.start.index() <= blocker.operation.index()
                        && blocker.operation.index() < span.end_exclusive
                })
                .map(|span| span.end_exclusive)
                .max()
                .unwrap_or(operation_end)
        });
        current.max(owner_end)
    });

    let prefix_operations = sorted_operations(
        live.iter()
            .copied()
            .filter(|operation| operation.index() < boundary),
    );
    let suffix_set = live
        .iter()
        .copied()
        .filter(|operation| operation.index() >= boundary)
        .collect::<HashSet<_>>();

    let mut frontier = HashSet::new();
    for operation in &suffix_set {
        for input in plan.analysis().inputs(*operation) {
            let producer = plan
                .provenance()
                .producer_of(*input)
                .expect("validated plan fields have producers");
            if producer.index() < boundary {
                frontier.insert(*input);
            }
        }
    }
    let output_producer = plan
        .provenance()
        .producer_of(output)
        .expect("validated plan output has a producer");
    if output_producer.index() < boundary {
        frontier.insert(output);
    }
    let mut frontier_fields = frontier.into_iter().collect::<Vec<_>>();
    frontier_fields.sort_by_key(|field| field.index());

    let suffix = resolve_local_suffix(plan, output, &suffix_set, boundary);
    TerrainPlanExecutionStrategy::Checkpointed {
        checkpoint: TerrainPlanCheckpoint {
            boundary,
            prefix_operations,
            frontier_fields,
            blockers,
        },
        suffix,
    }
}

fn checkpoint_blocker(
    plan: &CompiledTerrainPlan,
    operation_id: PlanOpId,
) -> Option<TerrainPlanDomainRejection> {
    let operation = plan.operation(operation_id)?;
    if operation.reach == Reach::Full {
        return Some(TerrainPlanDomainRejection {
            operation: operation_id,
            owner: plan.provenance().owner_of(operation_id),
            field: *plan.analysis().outputs(operation_id).first()?,
            reason: TerrainPlanDomainRejectReason::FullReach,
        });
    }
    if operation.aux_reach != AuxReach::Global {
        return None;
    }
    let field = plan
        .analysis()
        .outputs(operation_id)
        .iter()
        .copied()
        .find(|field| {
            plan.analysis().field_is_live(*field)
                && matches!(
                    plan.field(*field).map(|field| &field.kind),
                    Some(LogicalFieldKind::Auxiliary(_))
                )
        })?;
    Some(TerrainPlanDomainRejection {
        operation: operation_id,
        owner: plan.provenance().owner_of(operation_id),
        field,
        reason: TerrainPlanDomainRejectReason::GlobalAuxiliary,
    })
}

fn dependency_operations(plan: &CompiledTerrainPlan, output: FieldSlot) -> HashSet<PlanOpId> {
    let mut fields = HashSet::new();
    let mut operations = HashSet::new();
    let mut pending = VecDeque::from([output]);
    while let Some(field) = pending.pop_front() {
        if !fields.insert(field) {
            continue;
        }
        let producer = plan
            .provenance()
            .producer_of(field)
            .expect("validated plan fields have producers");
        if operations.insert(producer) {
            pending.extend(plan.analysis().inputs(producer).iter().copied());
        }
    }
    operations
}

fn resolve_local_suffix(
    plan: &CompiledTerrainPlan,
    output: FieldSlot,
    selected: &HashSet<PlanOpId>,
    boundary: usize,
) -> TerrainPlanDomainSlice {
    let mut required = HashMap::<FieldSlot, u32>::new();
    let mut pending = VecDeque::from([(output, 0u32)]);
    let mut maximum = 0u32;
    while let Some((field, downstream_halo)) = pending.pop_front() {
        if required
            .get(&field)
            .is_some_and(|known| *known >= downstream_halo)
        {
            continue;
        }
        required.insert(field, downstream_halo);
        let producer = plan
            .provenance()
            .producer_of(field)
            .expect("validated plan fields have producers");
        if producer.index() < boundary || !selected.contains(&producer) {
            continue;
        }
        let operation = plan.operation(producer).expect("validated plan operation");
        debug_assert_ne!(operation.reach, Reach::Full);
        let input_halo = match operation.reach {
            Reach::Full => downstream_halo,
            Reach::Localized { halo_samples } => downstream_halo.saturating_add(halo_samples),
        };
        maximum = maximum.max(input_halo);
        pending.extend(
            plan.analysis()
                .inputs(producer)
                .iter()
                .copied()
                .map(|input| (input, input_halo)),
        );
    }
    TerrainPlanDomainSlice {
        output,
        operations: sorted_operations(selected.iter().copied()),
        operation_halo: maximum,
    }
}

fn sorted_operations(operations: impl Iterator<Item = PlanOpId>) -> Vec<PlanOpId> {
    let mut operations = operations.collect::<Vec<_>>();
    operations.sort_by_key(|operation| operation.index());
    operations
}

/// Resolve the local plan slice required for `output`.
/// Sequential reaches add while converging dependency branches take their maximum.
pub fn resolve_plan_domain(
    plan: &CompiledTerrainPlan,
    output: FieldSlot,
) -> Result<TerrainPlanDomainSlice, TerrainPlanDomainRejection> {
    resolve_plan_domain_with(plan, output, |plan, producer, field| {
        let operation = plan.operation(producer).expect("validated plan operation");
        if matches!(
            plan.field(field).map(|field| &field.kind),
            Some(LogicalFieldKind::Auxiliary(_))
        ) && operation.aux_reach == AuxReach::Global
        {
            Some(TerrainPlanDomainRejectReason::GlobalAuxiliary)
        } else if operation.reach == Reach::Full {
            Some(TerrainPlanDomainRejectReason::FullReach)
        } else {
            None
        }
    })
}

/// Resolve the direct sparse-tile slice required for an Infinite project.
///
/// Unlike bounded execution, this never creates a complete-field checkpoint:
/// every live dependency must declare a direct Infinite capability and a finite
/// [`Reach`].
pub fn resolve_infinite_plan_domain(
    stack: &LayerStack,
    mask_assets: &[MaskAsset],
    plan: &CompiledTerrainPlan,
    output: FieldSlot,
) -> Result<TerrainPlanDomainSlice, TerrainPlanDomainRejection> {
    resolve_plan_domain_with(plan, output, |plan, producer, field| {
        let operation = plan.operation(producer).expect("validated plan operation");
        if matches!(
            plan.field(field).map(|field| &field.kind),
            Some(LogicalFieldKind::Auxiliary(_))
        ) && operation.aux_reach == AuxReach::Global
        {
            return Some(TerrainPlanDomainRejectReason::Infinite(
                SpatialRejectReason::GlobalAuxiliary,
            ));
        }
        let Some(contract) = super::operation_spatial_contract(stack, mask_assets, plan, producer)
        else {
            return Some(TerrainPlanDomainRejectReason::Infinite(
                SpatialRejectReason::UnclassifiedOperation,
            ));
        };
        if let Some(reason) = contract.infinite.rejection() {
            return Some(TerrainPlanDomainRejectReason::Infinite(reason));
        }
        if contract.reach == Reach::Full {
            return Some(TerrainPlanDomainRejectReason::Infinite(
                SpatialRejectReason::RequiresCompleteField,
            ));
        }
        None
    })
}

fn resolve_plan_domain_with(
    plan: &CompiledTerrainPlan,
    output: FieldSlot,
    reject: impl Fn(&CompiledTerrainPlan, PlanOpId, FieldSlot) -> Option<TerrainPlanDomainRejectReason>,
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
        if let Some(reason) = reject(plan, producer, field) {
            return Err(TerrainPlanDomainRejection {
                operation: producer,
                owner: plan.provenance().owner_of(producer),
                field,
                reason,
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

    fn infinite_fixture_stack() -> crate::layer::LayerStack {
        let mut first =
            crate::layer::Layer::new("first", crate::layer::LayerKind::Flat(Default::default()));
        first.common.id = LayerId::from_u128(1);
        let mut second =
            crate::layer::Layer::new("second", crate::layer::LayerKind::Flat(Default::default()));
        second.common.id = LayerId::from_u128(2);
        let mut stack = crate::layer::LayerStack::new();
        stack.push(first);
        stack.push(second);
        stack
    }

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

    #[test]
    fn infinite_consumed_global_auxiliary_has_project_reason() {
        let plan = local_chain(Reach::LOCAL, AuxReach::Global, true);
        let rejection = resolve_infinite_plan_domain(
            &infinite_fixture_stack(),
            &[],
            &plan,
            plan.final_height(),
        )
        .unwrap_err();
        assert_eq!(
            rejection.reason,
            TerrainPlanDomainRejectReason::Infinite(SpatialRejectReason::GlobalAuxiliary)
        );
    }

    #[test]
    fn full_reach_is_never_classified_as_an_ordinary_local_tile() {
        let plan = local_chain(Reach::Full, AuxReach::PerTexel, false);
        let TerrainPlanExecutionStrategy::Checkpointed { checkpoint, suffix } =
            resolve_plan_execution_strategy(&plan, plan.final_height())
        else {
            panic!("full-reach plan must be checkpointed");
        };
        assert_eq!(checkpoint.blockers.len(), 1);
        assert_eq!(checkpoint.blockers[0].operation.index(), 2);
        assert_eq!(
            checkpoint.blockers[0].reason,
            TerrainPlanDomainRejectReason::FullReach
        );
        assert!(suffix.operations.is_empty());
        assert_eq!(checkpoint.frontier_fields, vec![plan.final_height()]);
    }

    #[test]
    fn consumed_global_auxiliary_creates_a_frontier_for_the_local_suffix() {
        let plan = local_chain(Reach::LOCAL, AuxReach::Global, true);
        let TerrainPlanExecutionStrategy::Checkpointed { checkpoint, suffix } =
            resolve_plan_execution_strategy(&plan, plan.final_height())
        else {
            panic!("global auxiliary plan must be checkpointed");
        };
        assert_eq!(checkpoint.blockers[0].operation.index(), 1);
        assert_eq!(suffix.operations, vec![PlanOpId::from_index(2)]);
        let auxiliary = plan
            .fields()
            .iter()
            .find(|field| matches!(field.kind, LogicalFieldKind::Auxiliary(_)))
            .expect("fixture auxiliary field")
            .slot;
        assert!(checkpoint.frontier_fields.contains(&auxiliary));
    }

    #[test]
    fn upstream_edit_invalidates_the_checkpoint_prefix() {
        let plan = local_chain(Reach::Full, AuxReach::PerTexel, false);
        let TerrainPlanExecutionStrategy::Checkpointed { checkpoint, .. } =
            resolve_plan_execution_strategy(&plan, plan.final_height())
        else {
            panic!("fixture must be checkpointed");
        };
        let first_owner = NodeRef::Layer(LayerId::from_u128(1));
        let invalidation = crate::terrain_plan::propagate_plan_edits(
            &plan,
            &[crate::terrain_plan::TerrainEditClass::Parameters { owner: first_owner }],
        );
        assert!(checkpoint.is_invalidated_by(&invalidation));
    }
}
