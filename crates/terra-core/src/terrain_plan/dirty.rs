//! Dependency-aware plan invalidation and spatial dirty propagation.

use crate::deps::NodeRef;
use crate::field_data::FieldId;
use crate::invalidation::AuxReach;

use super::{
    CompiledTerrainPlan, FieldSlot, LogicalFieldKind, PlanDirtyScope, PlanOpId,
    PropagatedDirtyScope, TerrainEditClass,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullFieldReason {
    OperationReach,
    ObservedGlobalAuxiliary,
    StructuralChange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullFieldEscalation {
    pub operation: PlanOpId,
    pub owner: Option<NodeRef>,
    pub field: Option<FieldSlot>,
    pub reason: FullFieldReason,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanDirtyOperation {
    pub operation: PlanOpId,
    /// Scope arriving at this operation before applying its own reach contract.
    pub incoming_scope: PropagatedDirtyScope,
    /// Scope emitted after applying operation and observed auxiliary reach.
    pub scope: PropagatedDirtyScope,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PlanInvalidation {
    pub patched_operations: Vec<PlanOpId>,
    pub operations: Vec<PlanDirtyOperation>,
    pub dirty_fields: Vec<(FieldSlot, PropagatedDirtyScope)>,
    pub first_full_field_escalation: Option<FullFieldEscalation>,
}

/// Translate authored edits into an ordered, live dirty subgraph.
pub fn propagate_plan_edits(
    plan: &CompiledTerrainPlan,
    edits: &[TerrainEditClass],
) -> PlanInvalidation {
    let operation_count = plan.operations().len();
    let field_count = plan.fields().len();
    let mut seeds: Vec<Option<PropagatedDirtyScope>> = vec![None; operation_count];
    let mut patched = vec![false; operation_count];
    let mut structural = false;

    for edit in edits {
        match edit {
            TerrainEditClass::ViewOnly | TerrainEditClass::Resources => {}
            TerrainEditClass::Structure => {
                structural = true;
                for (index, _) in plan.operations().iter().enumerate() {
                    if plan
                        .analysis()
                        .operation_is_live(PlanOpId::from_index(index))
                    {
                        merge_scope(
                            &mut seeds[index],
                            PropagatedDirtyScope::new(PlanDirtyScope::FullField),
                        );
                    }
                }
            }
            TerrainEditClass::Parameters { owner } => {
                seed_owner(
                    plan,
                    *owner,
                    PropagatedDirtyScope::new(PlanDirtyScope::FullField),
                    &mut seeds,
                    &mut patched,
                );
            }
            TerrainEditClass::Content {
                owner,
                fields,
                scope,
            } => {
                let propagated = PropagatedDirtyScope::new(*scope);
                seed_owner_fields(plan, *owner, fields, propagated, &mut seeds, &mut patched);
            }
        }
    }

    let mut field_scopes: Vec<Option<PropagatedDirtyScope>> = vec![None; field_count];
    let mut operations = Vec::new();
    let mut first_full_field_escalation = None;

    for (index, operation) in plan.operations().iter().enumerate() {
        let operation_id = PlanOpId::from_index(index);
        if !plan.analysis().operation_is_live(operation_id) {
            continue;
        }
        let mut incoming = seeds[index];
        for input in plan.analysis().inputs(operation_id) {
            if let Some(scope) = field_scopes[input.index()] {
                merge_scope(&mut incoming, scope);
            }
        }
        let Some(incoming) = incoming else {
            continue;
        };

        if structural && incoming.is_full() && first_full_field_escalation.is_none() {
            first_full_field_escalation = Some(FullFieldEscalation {
                operation: operation_id,
                owner: plan.provenance().owner_of(operation_id),
                field: None,
                reason: FullFieldReason::StructuralChange,
            });
        }

        let height_scope = incoming.expand(operation.reach);
        if !incoming.is_full() && height_scope.is_full() && first_full_field_escalation.is_none() {
            first_full_field_escalation = Some(FullFieldEscalation {
                operation: operation_id,
                owner: plan.provenance().owner_of(operation_id),
                field: None,
                reason: if structural {
                    FullFieldReason::StructuralChange
                } else {
                    FullFieldReason::OperationReach
                },
            });
        }

        let mut operation_scope = height_scope;
        for output in plan.analysis().outputs(operation_id) {
            if !plan.analysis().field_is_live(*output) {
                continue;
            }
            let field = plan.field(*output).expect("validated analysis field");
            let output_scope = match field.kind {
                LogicalFieldKind::Auxiliary(_) => match operation.aux_reach {
                    AuxReach::Global => {
                        let full = PropagatedDirtyScope::new(PlanDirtyScope::FullField);
                        if !height_scope.is_full() && first_full_field_escalation.is_none() {
                            first_full_field_escalation = Some(FullFieldEscalation {
                                operation: operation_id,
                                owner: plan.provenance().owner_of(operation_id),
                                field: Some(*output),
                                reason: FullFieldReason::ObservedGlobalAuxiliary,
                            });
                        }
                        full
                    }
                    AuxReach::HeightOnly | AuxReach::PerTexel => height_scope,
                },
                LogicalFieldKind::Height | LogicalFieldKind::Mask => height_scope,
            };
            merge_scope(&mut field_scopes[output.index()], output_scope);
            operation_scope = operation_scope.merge(output_scope);
        }
        operations.push(PlanDirtyOperation {
            operation: operation_id,
            incoming_scope: incoming,
            scope: operation_scope,
        });
    }

    let patched_operations = patched
        .into_iter()
        .enumerate()
        .filter(|(_, patched)| *patched)
        .map(|(index, _)| PlanOpId::from_index(index))
        .collect();
    let dirty_fields = field_scopes
        .into_iter()
        .enumerate()
        .filter_map(|(index, scope)| scope.map(|scope| (FieldSlot::from_index(index), scope)))
        .collect();

    PlanInvalidation {
        patched_operations,
        operations,
        dirty_fields,
        first_full_field_escalation,
    }
}

fn seed_owner_fields(
    plan: &CompiledTerrainPlan,
    owner: NodeRef,
    fields: &[FieldId],
    scope: PropagatedDirtyScope,
    seeds: &mut [Option<PropagatedDirtyScope>],
    patched: &mut [bool],
) {
    // Content changes patch the owner's kernel payload itself. Height fields
    // recorded in provenance are observable post-composite fields, so seeding
    // only their producer would otherwise miss the `RunLayerKernel` operation.
    for operation in plan.provenance().operations_for(owner) {
        if plan.operation(*operation).is_some_and(|operation| {
            matches!(operation.kind, super::TerrainOpKind::RunLayerKernel { .. })
        }) {
            merge_scope(&mut seeds[operation.index()], scope);
            patched[operation.index()] = true;
        }
    }
    let mut matched_field = false;
    for field in plan.provenance().fields_for(owner) {
        let matches = plan
            .field(*field)
            .is_some_and(|logical| match &logical.kind {
                LogicalFieldKind::Height => fields.contains(&FieldId::Height),
                LogicalFieldKind::Auxiliary(field) => fields.contains(field),
                LogicalFieldKind::Mask => false,
            });
        if matches {
            matched_field = true;
            if let Some(producer) = plan.provenance().producer_of(*field) {
                merge_scope(&mut seeds[producer.index()], scope);
                patched[producer.index()] = true;
            }
        }
    }
    if !matched_field {
        seed_owner(plan, owner, scope, seeds, patched);
    } else {
        for consumer in plan.provenance().consumers_for(owner) {
            merge_scope(&mut seeds[consumer.index()], scope);
        }
    }
}

fn seed_owner(
    plan: &CompiledTerrainPlan,
    owner: NodeRef,
    scope: PropagatedDirtyScope,
    seeds: &mut [Option<PropagatedDirtyScope>],
    patched: &mut [bool],
) {
    for operation in plan.provenance().operations_for(owner) {
        merge_scope(&mut seeds[operation.index()], scope);
        patched[operation.index()] = true;
    }
    for operation in plan.provenance().consumers_for(owner) {
        merge_scope(&mut seeds[operation.index()], scope);
    }
}

fn merge_scope(target: &mut Option<PropagatedDirtyScope>, incoming: PropagatedDirtyScope) {
    *target = Some(match *target {
        Some(existing) => existing.merge(incoming),
        None => incoming,
    });
}
