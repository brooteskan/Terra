//! Logical fields, operations, and construction of a compiled terrain plan.

use super::{FieldSlot, PlanOpId, PlanProvenance, PlanStructureRevision, TerrainPlanStamp};
use crate::deps::NodeRef;
use crate::field_data::FieldId;
use crate::ids::{LayerId, OutputId};
use crate::invalidation::Reach;

/// Semantic kind of one logical field. Physical representation belongs to a backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LogicalFieldKind {
    Height,
    Auxiliary(FieldId),
    Mask,
}

/// Provenance origin for a plan descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanOrigin {
    Root,
    Authored(NodeRef),
}

impl PlanOrigin {
    pub const fn authored(self) -> Option<NodeRef> {
        match self {
            Self::Root => None,
            Self::Authored(owner) => Some(owner),
        }
    }
}

/// Description of one logical field owned by the plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalField {
    pub slot: FieldSlot,
    pub kind: LogicalFieldKind,
    pub origin: PlanOrigin,
}

/// Source used to initialize a working heightfield.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedSource {
    Zero,
    Copy(FieldSlot),
    Selected(FieldSlot),
}

/// Height-composition equation selected by authored group semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupCompositeMode {
    Standard,
    BiomeHeightDelta,
}

/// One operation in the backend-neutral ordered execution plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerrainOpKind {
    Seed {
        source: SeedSource,
        output: FieldSlot,
    },
    EvaluateMask {
        input_height: FieldSlot,
        output_mask: FieldSlot,
    },
    RunLayerKernel {
        layer: LayerId,
        type_id: String,
        input_height: FieldSlot,
        input_fields: Vec<FieldSlot>,
        output_candidate: FieldSlot,
        output_fields: Vec<FieldSlot>,
    },
    CompositeLayer {
        layer: LayerId,
        base: FieldSlot,
        candidate: FieldSlot,
        mask: FieldSlot,
        output: FieldSlot,
    },
    CompositeGroup {
        group: LayerId,
        parent: FieldSlot,
        private_seed: FieldSlot,
        child_output: FieldSlot,
        mask: FieldSlot,
        output: FieldSlot,
        mode: GroupCompositeMode,
    },
    PublishOutput {
        output: OutputId,
        source: FieldSlot,
    },
}

impl TerrainOpKind {
    fn input_slots(&self) -> Vec<FieldSlot> {
        match self {
            Self::Seed { source, .. } => match source {
                SeedSource::Zero => Vec::new(),
                SeedSource::Copy(slot) | SeedSource::Selected(slot) => vec![*slot],
            },
            Self::EvaluateMask { input_height, .. } => vec![*input_height],
            Self::RunLayerKernel {
                input_height,
                input_fields,
                ..
            } => {
                let mut slots = Vec::with_capacity(1 + input_fields.len());
                slots.push(*input_height);
                slots.extend(input_fields.iter().copied());
                slots
            }
            Self::CompositeLayer {
                base,
                candidate,
                mask,
                ..
            } => vec![*base, *candidate, *mask],
            Self::CompositeGroup {
                parent,
                private_seed,
                child_output,
                mask,
                ..
            } => vec![*parent, *private_seed, *child_output, *mask],
            Self::PublishOutput { source, .. } => vec![*source],
        }
    }

    fn output_slots(&self) -> Vec<FieldSlot> {
        match self {
            Self::Seed { output, .. }
            | Self::CompositeLayer { output, .. }
            | Self::CompositeGroup { output, .. } => vec![*output],
            Self::EvaluateMask { output_mask, .. } => vec![*output_mask],
            Self::RunLayerKernel {
                output_candidate,
                output_fields,
                ..
            } => {
                let mut slots = Vec::with_capacity(1 + output_fields.len());
                slots.push(*output_candidate);
                slots.extend(output_fields.iter().copied());
                slots
            }
            Self::PublishOutput { .. } => Vec::new(),
        }
    }
}

/// Operation plus authored origin and resolved spatial reach.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerrainOp {
    pub origin: PlanOrigin,
    pub reach: Reach,
    pub kind: TerrainOpKind,
}

/// Runtime-derived, backend-neutral terrain evaluation plan.
#[derive(Debug, Clone)]
pub struct CompiledTerrainPlan {
    stamp: TerrainPlanStamp,
    fields: Vec<LogicalField>,
    operations: Vec<TerrainOp>,
    provenance: PlanProvenance,
    final_height: FieldSlot,
}

impl CompiledTerrainPlan {
    pub const fn stamp(&self) -> TerrainPlanStamp {
        self.stamp
    }

    pub fn matches_structure_revision(&self, revision: PlanStructureRevision) -> bool {
        self.stamp.structure_revision == revision
    }

    pub fn fields(&self) -> &[LogicalField] {
        &self.fields
    }

    pub fn operations(&self) -> &[TerrainOp] {
        &self.operations
    }

    pub fn field(&self, slot: FieldSlot) -> Option<&LogicalField> {
        self.fields.get(slot.index())
    }

    pub fn operation(&self, id: PlanOpId) -> Option<&TerrainOp> {
        self.operations.get(id.index())
    }

    pub const fn final_height(&self) -> FieldSlot {
        self.final_height
    }

    pub const fn provenance(&self) -> &PlanProvenance {
        &self.provenance
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PlanBuildError {
    #[error("field slot {slot} is outside the plan's {field_count} fields")]
    InvalidFieldSlot { slot: usize, field_count: usize },
    #[error("logical field slot {slot} has more than one producer")]
    MultipleFieldProducers { slot: usize },
    #[error("published output {output:?} is declared more than once")]
    DuplicatePublishedOutput { output: OutputId },
}

/// Construction helper used by the tree compiler and semantic fixtures.
///
/// It establishes contiguous IDs and basic slot/provenance integrity. Complete
/// dependency and lifetime validation belongs to the later plan-validation phase.
#[derive(Debug)]
pub struct TerrainPlanBuilder {
    stamp: TerrainPlanStamp,
    fields: Vec<LogicalField>,
    operations: Vec<TerrainOp>,
}

impl TerrainPlanBuilder {
    pub fn new(stamp: TerrainPlanStamp) -> Self {
        Self {
            stamp,
            fields: Vec::new(),
            operations: Vec::new(),
        }
    }

    pub fn add_field(&mut self, kind: LogicalFieldKind, origin: PlanOrigin) -> FieldSlot {
        let slot = FieldSlot::from_index(self.fields.len());
        self.fields.push(LogicalField { slot, kind, origin });
        slot
    }

    pub fn add_operation(&mut self, operation: TerrainOp) -> PlanOpId {
        let id = PlanOpId::from_index(self.operations.len());
        self.operations.push(operation);
        id
    }

    pub fn finish(self, final_height: FieldSlot) -> Result<CompiledTerrainPlan, PlanBuildError> {
        let field_count = self.fields.len();
        check_slot(final_height, field_count)?;

        let mut provenance = PlanProvenance::with_capacities(self.operations.len(), field_count);
        for (index, operation) in self.operations.iter().enumerate() {
            let operation_id = PlanOpId::from_index(index);
            provenance.record_operation(operation_id, operation.origin.authored());

            for input in operation.kind.input_slots() {
                check_slot(input, field_count)?;
            }
            for output in operation.kind.output_slots() {
                check_slot(output, field_count)?;
                if provenance.producer_of(output).is_some() {
                    return Err(PlanBuildError::MultipleFieldProducers {
                        slot: output.index(),
                    });
                }
                provenance.record_producer(output, operation_id);
            }
            if let TerrainOpKind::PublishOutput { output, source } = &operation.kind {
                if !provenance.record_output(*output, *source) {
                    return Err(PlanBuildError::DuplicatePublishedOutput { output: *output });
                }
            }
        }

        Ok(CompiledTerrainPlan {
            stamp: self.stamp,
            fields: self.fields,
            operations: self.operations,
            provenance,
            final_height,
        })
    }
}

fn check_slot(slot: FieldSlot, field_count: usize) -> Result<(), PlanBuildError> {
    if slot.index() < field_count {
        Ok(())
    } else {
        Err(PlanBuildError::InvalidFieldSlot {
            slot: slot.index(),
            field_count,
        })
    }
}
