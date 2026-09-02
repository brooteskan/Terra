//! Authored-to-plan and plan-to-authored provenance.

use std::collections::HashMap;

use crate::deps::DepKind;
use crate::deps::NodeRef;
use crate::ids::OutputId;

use super::{FieldSlot, PlanOpId};

/// How an authored node participates in the compiled solo projection.
///
/// This is separate from authored enabled state: a disabled solo node still
/// participates in selection even though it emits no operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanNodeSelection {
    /// No solo exists at this sibling level, so the node is included normally.
    Unfiltered,
    /// A solo exists at this sibling level and this node is on a participating path.
    IncludedBySolo,
    /// A solo exists at this or an ancestor sibling level and excludes this node.
    ExcludedBySolo,
}

impl PlanNodeSelection {
    pub const fn participates(self) -> bool {
        !matches!(self, Self::ExcludedBySolo)
    }
}

/// Half-open lexical operation span belonging to an authored layer/group.
///
/// For a group this includes nested child work. Exact operations directly
/// owned by the group remain available through [`PlanProvenance::operations_for`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanOpSpan {
    pub start: PlanOpId,
    pub end_exclusive: usize,
}

/// Resolved provenance of one stable named output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputProvenance {
    pub field: FieldSlot,
    pub publisher: PlanOpId,
    pub owner: Option<NodeRef>,
}

/// An authored dependency (mask, output, binding, or group input) consumed by
/// a plan operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanAuthoredDependency {
    pub source: NodeRef,
    pub consumer: PlanOpId,
    pub kind: DepKind,
}

/// Stable authored provenance for plan-local operations and logical fields.
#[derive(Debug, Clone, Default)]
pub struct PlanProvenance {
    node_selection: HashMap<NodeRef, PlanNodeSelection>,
    authored_to_ops: HashMap<NodeRef, Vec<PlanOpId>>,
    op_to_authored: Vec<Option<NodeRef>>,
    authored_to_fields: HashMap<NodeRef, Vec<FieldSlot>>,
    field_to_authored: Vec<Option<NodeRef>>,
    owner_spans: HashMap<NodeRef, Vec<PlanOpSpan>>,
    outputs: HashMap<OutputId, OutputProvenance>,
    field_producers: Vec<Option<PlanOpId>>,
    authored_consumers: HashMap<NodeRef, Vec<PlanOpId>>,
    dependencies: Vec<PlanAuthoredDependency>,
}

impl PlanProvenance {
    pub fn selection_for(&self, owner: NodeRef) -> Option<PlanNodeSelection> {
        self.node_selection.get(&owner).copied()
    }

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
        self.outputs.get(&output).map(|entry| entry.field)
    }

    pub fn output(&self, output: OutputId) -> Option<OutputProvenance> {
        self.outputs.get(&output).copied()
    }

    pub fn fields_for(&self, owner: NodeRef) -> &[FieldSlot] {
        self.authored_to_fields
            .get(&owner)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn owner_of_field(&self, field: FieldSlot) -> Option<NodeRef> {
        self.field_to_authored.get(field.index()).copied().flatten()
    }

    pub fn spans_for(&self, owner: NodeRef) -> &[PlanOpSpan] {
        self.owner_spans
            .get(&owner)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn consumers_for(&self, source: NodeRef) -> &[PlanOpId] {
        self.authored_consumers
            .get(&source)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn dependencies(&self) -> &[PlanAuthoredDependency] {
        &self.dependencies
    }

    pub fn producer_of(&self, field: FieldSlot) -> Option<PlanOpId> {
        self.field_producers.get(field.index()).copied().flatten()
    }

    pub(crate) fn with_capacities(operation_count: usize, field_count: usize) -> Self {
        Self {
            node_selection: HashMap::new(),
            authored_to_ops: HashMap::new(),
            op_to_authored: Vec::with_capacity(operation_count),
            authored_to_fields: HashMap::new(),
            field_to_authored: vec![None; field_count],
            owner_spans: HashMap::new(),
            outputs: HashMap::new(),
            field_producers: vec![None; field_count],
            authored_consumers: HashMap::new(),
            dependencies: Vec::new(),
        }
    }

    pub(crate) fn record_node_selection(&mut self, owner: NodeRef, selection: PlanNodeSelection) {
        self.node_selection.insert(owner, selection);
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

    pub(crate) fn record_field(&mut self, field: FieldSlot, owner: Option<NodeRef>) {
        self.field_to_authored[field.index()] = owner;
        if let Some(owner) = owner {
            self.authored_to_fields
                .entry(owner)
                .or_default()
                .push(field);
        }
    }

    pub(crate) fn record_span(&mut self, owner: NodeRef, span: PlanOpSpan) {
        self.owner_spans.entry(owner).or_default().push(span);
    }

    pub(crate) fn record_owner_field(&mut self, owner: NodeRef, field: FieldSlot) {
        let fields = self.authored_to_fields.entry(owner).or_default();
        if !fields.contains(&field) {
            fields.push(field);
        }
    }

    pub(crate) fn record_dependency(&mut self, dependency: PlanAuthoredDependency) {
        let consumers = self
            .authored_consumers
            .entry(dependency.source)
            .or_default();
        if !consumers.contains(&dependency.consumer) {
            consumers.push(dependency.consumer);
        }
        self.dependencies.push(dependency);
    }

    pub(crate) fn record_output(
        &mut self,
        output: OutputId,
        field: FieldSlot,
        publisher: PlanOpId,
        owner: Option<NodeRef>,
    ) -> bool {
        self.outputs
            .insert(
                output,
                OutputProvenance {
                    field,
                    publisher,
                    owner,
                },
            )
            .is_none()
    }
}
