//! Authored-to-plan and plan-to-authored provenance.

use std::collections::HashMap;

use crate::deps::NodeRef;
use crate::ids::OutputId;

use super::{FieldSlot, PlanOpId};

/// Stable authored provenance for plan-local operations and logical fields.
#[derive(Debug, Clone, Default)]
pub struct PlanProvenance {
    authored_to_ops: HashMap<NodeRef, Vec<PlanOpId>>,
    op_to_authored: Vec<Option<NodeRef>>,
    output_fields: HashMap<OutputId, FieldSlot>,
    field_producers: Vec<Option<PlanOpId>>,
}

impl PlanProvenance {
    pub fn operations_for(&self, owner: NodeRef) -> &[PlanOpId] {
        self.authored_to_ops
            .get(&owner)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn owner_of(&self, operation: PlanOpId) -> Option<NodeRef> {
        self.op_to_authored
            .get(operation.index())
            .copied()
            .flatten()
    }

    pub fn field_for_output(&self, output: OutputId) -> Option<FieldSlot> {
        self.output_fields.get(&output).copied()
    }

    pub fn producer_of(&self, field: FieldSlot) -> Option<PlanOpId> {
        self.field_producers.get(field.index()).copied().flatten()
    }

    pub(crate) fn with_capacities(operation_count: usize, field_count: usize) -> Self {
        Self {
            authored_to_ops: HashMap::new(),
            op_to_authored: Vec::with_capacity(operation_count),
            output_fields: HashMap::new(),
            field_producers: vec![None; field_count],
        }
    }

    pub(crate) fn record_operation(&mut self, operation: PlanOpId, owner: Option<NodeRef>) {
        debug_assert_eq!(operation.index(), self.op_to_authored.len());
        self.op_to_authored.push(owner);
        if let Some(owner) = owner {
            self.authored_to_ops
                .entry(owner)
                .or_default()
                .push(operation);
        }
    }

    pub(crate) fn record_producer(&mut self, field: FieldSlot, operation: PlanOpId) {
        self.field_producers[field.index()] = Some(operation);
    }

    pub(crate) fn record_output(&mut self, output: OutputId, field: FieldSlot) -> bool {
        self.output_fields.insert(output, field).is_none()
    }
}
