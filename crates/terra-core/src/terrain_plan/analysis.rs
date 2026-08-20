//! Validated field def-use, liveness, and lifetime metadata.

use std::collections::VecDeque;

use super::{
    CompiledTerrainPlan, FieldSlot, LogicalField, LogicalFieldKind, PlanBuildError, PlanOpId,
    PlanProvenance, TerrainOp, TerrainOpKind,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedFieldKind {
    Height,
    Auxiliary,
    Mask,
    HeightOrAuxiliary,
}

impl ExpectedFieldKind {
    fn accepts(self, actual: &LogicalFieldKind) -> bool {
        match self {
            Self::Height => matches!(actual, LogicalFieldKind::Height),
            Self::Auxiliary => matches!(actual, LogicalFieldKind::Auxiliary(_)),
            Self::Mask => matches!(actual, LogicalFieldKind::Mask),
            Self::HeightOrAuxiliary => !matches!(actual, LogicalFieldKind::Mask),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanFieldLifetime {
    pub first_operation: PlanOpId,
    pub last_operation: PlanOpId,
}

/// Immutable analysis derived from one validated plan structure.
#[derive(Debug, Clone)]
pub struct PlanAnalysis {
    inputs: Vec<Vec<FieldSlot>>,
    outputs: Vec<Vec<FieldSlot>>,
    consumers: Vec<Vec<PlanOpId>>,
    live_fields: Vec<bool>,
    live_operations: Vec<bool>,
    lifetimes: Vec<PlanFieldLifetime>,
}

impl PlanAnalysis {
    pub fn inputs(&self, operation: PlanOpId) -> &[FieldSlot] {
        self.inputs
            .get(operation.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn outputs(&self, operation: PlanOpId) -> &[FieldSlot] {
        self.outputs
            .get(operation.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn consumers(&self, field: FieldSlot) -> &[PlanOpId] {
        self.consumers
            .get(field.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn field_is_live(&self, field: FieldSlot) -> bool {
        self.live_fields
            .get(field.index())
            .copied()
            .unwrap_or(false)
    }

    pub fn operation_is_live(&self, operation: PlanOpId) -> bool {
        self.live_operations
            .get(operation.index())
            .copied()
            .unwrap_or(false)
    }

    pub fn lifetime(&self, field: FieldSlot) -> Option<PlanFieldLifetime> {
        self.lifetimes.get(field.index()).copied()
    }
}

pub(crate) fn validate_and_analyze(
    fields: &[LogicalField],
    operations: &[TerrainOp],
    provenance: &PlanProvenance,
    final_height: FieldSlot,
) -> Result<PlanAnalysis, PlanBuildError> {
    let field_count = fields.len();
    let operation_count = operations.len();
    let final_field = fields
        .get(final_height.index())
        .ok_or(PlanBuildError::InvalidFieldSlot {
            slot: final_height.index(),
            field_count,
        })?;
    if !matches!(final_field.kind, LogicalFieldKind::Height) {
        return Err(PlanBuildError::InvalidFinalHeightKind {
            slot: final_height.index(),
            actual: final_field.kind.clone(),
        });
    }

    let mut inputs = Vec::with_capacity(operation_count);
    let mut outputs = Vec::with_capacity(operation_count);
    let mut consumers = vec![Vec::new(); field_count];
    let mut adjacency = vec![Vec::new(); operation_count];
    let mut indegree = vec![0_usize; operation_count];

    for (index, operation) in operations.iter().enumerate() {
        let operation_id = PlanOpId::from_index(index);
        validate_operation_kinds(operation_id, operation, fields)?;
        let operation_inputs = operation.kind.input_slots();
        let operation_outputs = operation.kind.output_slots();
        for input in &operation_inputs {
            let producer =
                provenance
                    .producer_of(*input)
                    .ok_or(PlanBuildError::UnproducedFieldInput {
                        operation: operation_id,
                        owner: operation.origin.authored(),
                        slot: input.index(),
                    })?;
            consumers[input.index()].push(operation_id);
            if !adjacency[producer.index()].contains(&operation_id) {
                adjacency[producer.index()].push(operation_id);
                indegree[operation_id.index()] += 1;
            }
        }
        inputs.push(operation_inputs);
        outputs.push(operation_outputs);
    }

    for field in fields {
        if provenance.producer_of(field.slot).is_none() {
            return Err(PlanBuildError::UnproducedField {
                slot: field.slot.index(),
                owner: field.origin.authored(),
            });
        }
    }

    let mut queue = VecDeque::new();
    for (index, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            queue.push_back(PlanOpId::from_index(index));
        }
    }
    let mut visited = 0_usize;
    while let Some(operation) = queue.pop_front() {
        visited += 1;
        for consumer in &adjacency[operation.index()] {
            let degree = &mut indegree[consumer.index()];
            *degree = degree.saturating_sub(1);
            if *degree == 0 {
                queue.push_back(*consumer);
            }
        }
    }
    if visited != operation_count {
        let cycle_operations: Vec<_> = indegree
            .iter()
            .enumerate()
            .filter(|(_, degree)| **degree > 0)
            .map(|(index, _)| PlanOpId::from_index(index))
            .collect();
        let mut owners = Vec::new();
        for operation in &cycle_operations {
            if let Some(owner) = provenance.owner_of(*operation) {
                if !owners.contains(&owner) {
                    owners.push(owner);
                }
            }
        }
        return Err(PlanBuildError::DependencyCycle {
            operations: cycle_operations,
            owners,
        });
    }

    for (consumer_index, operation_inputs) in inputs.iter().enumerate() {
        let consumer = PlanOpId::from_index(consumer_index);
        for field in operation_inputs {
            let producer = provenance
                .producer_of(*field)
                .expect("producer checked above");
            if producer.index() >= consumer.index() {
                return Err(PlanBuildError::UseBeforeProduce {
                    field: *field,
                    producer,
                    consumer,
                    owner: provenance.owner_of(consumer),
                });
            }
        }
    }

    let mut live_fields = vec![false; field_count];
    let mut live_operations = vec![false; operation_count];
    let mut stack = vec![final_height];
    for operation in operations {
        if let TerrainOpKind::PublishOutput { source, .. } = operation.kind {
            stack.push(source);
        }
    }
    while let Some(field) = stack.pop() {
        if live_fields[field.index()] {
            continue;
        }
        live_fields[field.index()] = true;
        let producer = provenance
            .producer_of(field)
            .expect("all fields have producers after validation");
        if live_operations[producer.index()] {
            continue;
        }
        live_operations[producer.index()] = true;
        stack.extend(inputs[producer.index()].iter().copied());
    }
    for (index, operation) in operations.iter().enumerate() {
        if matches!(operation.kind, TerrainOpKind::PublishOutput { .. }) {
            live_operations[index] = true;
        }
    }

    let mut lifetimes = Vec::with_capacity(field_count);
    for field in fields {
        let producer = provenance
            .producer_of(field.slot)
            .expect("all fields have producers after validation");
        let last = consumers[field.slot.index()]
            .iter()
            .copied()
            .max_by_key(|operation| operation.index())
            .unwrap_or(producer);
        let last = if field.slot == final_height && last.index() + 1 < operation_count {
            PlanOpId::from_index(operation_count - 1)
        } else {
            last
        };
        lifetimes.push(PlanFieldLifetime {
            first_operation: producer,
            last_operation: last,
        });
    }

    Ok(PlanAnalysis {
        inputs,
        outputs,
        consumers,
        live_fields,
        live_operations,
        lifetimes,
    })
}

fn validate_operation_kinds(
    operation_id: PlanOpId,
    operation: &TerrainOp,
    fields: &[LogicalField],
) -> Result<(), PlanBuildError> {
    let owner = operation.origin.authored();
    let expect = |slot: FieldSlot, expected: ExpectedFieldKind| {
        let field = fields
            .get(slot.index())
            .ok_or(PlanBuildError::InvalidFieldSlot {
                slot: slot.index(),
                field_count: fields.len(),
            })?;
        if expected.accepts(&field.kind) {
            Ok(())
        } else {
            Err(PlanBuildError::FieldKindMismatch {
                operation: operation_id,
                owner,
                field: slot,
                expected,
                actual: field.kind.clone(),
            })
        }
    };

    match &operation.kind {
        TerrainOpKind::Seed { source, output } => {
            expect(*output, ExpectedFieldKind::Height)?;
            match source {
                super::SeedSource::Zero => {}
                super::SeedSource::Copy(field) | super::SeedSource::Selected(field) => {
                    expect(*field, ExpectedFieldKind::HeightOrAuxiliary)?;
                }
            }
        }
        TerrainOpKind::EvaluateMask {
            input_height,
            input_fields,
            output_mask,
        } => {
            expect(*input_height, ExpectedFieldKind::Height)?;
            for field in input_fields {
                expect(*field, ExpectedFieldKind::HeightOrAuxiliary)?;
            }
            expect(*output_mask, ExpectedFieldKind::Mask)?;
        }
        TerrainOpKind::RunLayerKernel {
            input_height,
            input_fields,
            output_candidate,
            output_fields,
            ..
        } => {
            expect(*input_height, ExpectedFieldKind::Height)?;
            for field in input_fields {
                expect(*field, ExpectedFieldKind::HeightOrAuxiliary)?;
            }
            expect(*output_candidate, ExpectedFieldKind::Height)?;
            for field in output_fields {
                expect(*field, ExpectedFieldKind::Auxiliary)?;
            }
        }
        TerrainOpKind::CompositeLayer {
            base,
            candidate,
            mask,
            output,
            ..
        } => {
            expect(*base, ExpectedFieldKind::Height)?;
            expect(*candidate, ExpectedFieldKind::Height)?;
            expect(*mask, ExpectedFieldKind::Mask)?;
            expect(*output, ExpectedFieldKind::Height)?;
        }
        TerrainOpKind::CompositeGroup {
            parent,
            private_seed,
            child_output,
            mask,
            output,
            ..
        } => {
            for field in [parent, private_seed, child_output, output] {
                expect(*field, ExpectedFieldKind::Height)?;
            }
            expect(*mask, ExpectedFieldKind::Mask)?;
        }
        TerrainOpKind::CompositeAuxField {
            mask, composite, ..
        } => {
            expect(*mask, ExpectedFieldKind::Mask)?;
            if let Some(parent) = composite.parent {
                expect(parent, ExpectedFieldKind::Auxiliary)?;
            }
            expect(composite.child, ExpectedFieldKind::Auxiliary)?;
            expect(composite.output, ExpectedFieldKind::Auxiliary)?;
        }
        TerrainOpKind::PublishOutput { source, .. } => {
            expect(*source, ExpectedFieldKind::HeightOrAuxiliary)?;
        }
    }
    Ok(())
}

impl CompiledTerrainPlan {
    pub const fn analysis(&self) -> &PlanAnalysis {
        &self.analysis
    }
}
