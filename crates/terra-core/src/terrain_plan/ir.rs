//! Logical fields, operations, and construction of a compiled terrain plan.

use std::hash::{Hash, Hasher};

use super::{
    FieldSlot, PlanOpId, PlanProvenance, PlanStructureRevision, PlanStructureSignature,
    TerrainPlanStamp,
};
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SeedSource {
    Zero,
    Copy(FieldSlot),
    Selected(FieldSlot),
}

/// Height-composition equation selected by authored group semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupCompositeMode {
    Standard,
    BiomeHeightDelta,
}

/// One auxiliary field crossing an isolated-group boundary.
///
/// `parent` is absent when the field is first produced inside the group; backends
/// treat that case as a zero-valued parent field before applying the group mask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupAuxComposite {
    pub field: FieldId,
    pub parent: Option<FieldSlot>,
    pub child: FieldSlot,
    pub output: FieldSlot,
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
        input_fields: Vec<FieldSlot>,
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
        aux: Vec<GroupAuxComposite>,
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
            Self::EvaluateMask {
                input_height,
                input_fields,
                ..
            } => {
                let mut slots = Vec::with_capacity(1 + input_fields.len());
                slots.push(*input_height);
                slots.extend(input_fields.iter().copied());
                slots
            }
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
                aux,
                ..
            } => {
                let mut slots = Vec::with_capacity(4 + aux.len() * 2);
                slots.extend([*parent, *private_seed, *child_output, *mask]);
                for field in aux {
                    if let Some(parent) = field.parent {
                        slots.push(parent);
                    }
                    slots.push(field.child);
                }
                slots
            }
            Self::PublishOutput { source, .. } => vec![*source],
        }
    }

    fn output_slots(&self) -> Vec<FieldSlot> {
        match self {
            Self::Seed { output, .. } | Self::CompositeLayer { output, .. } => vec![*output],
            Self::CompositeGroup { output, aux, .. } => {
                let mut slots = Vec::with_capacity(1 + aux.len());
                slots.push(*output);
                slots.extend(aux.iter().map(|field| field.output));
                slots
            }
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
    structure_signature: PlanStructureSignature,
    fields: Vec<LogicalField>,
    operations: Vec<TerrainOp>,
    provenance: PlanProvenance,
    final_height: FieldSlot,
}

impl CompiledTerrainPlan {
    pub const fn stamp(&self) -> TerrainPlanStamp {
        self.stamp
    }

    pub const fn structure_signature(&self) -> PlanStructureSignature {
        self.structure_signature
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

        let structure_signature = structure_signature(&self.fields, &self.operations, final_height);
        Ok(CompiledTerrainPlan {
            stamp: self.stamp,
            structure_signature,
            fields: self.fields,
            operations: self.operations,
            provenance,
            final_height,
        })
    }
}

fn structure_signature(
    fields: &[LogicalField],
    operations: &[TerrainOp],
    final_height: FieldSlot,
) -> PlanStructureSignature {
    let mut hasher = StablePlanHasher::default();
    fields.len().hash(&mut hasher);
    for field in fields {
        field.slot.hash(&mut hasher);
        field.kind.hash(&mut hasher);
        field.origin.hash(&mut hasher);
    }
    operations.len().hash(&mut hasher);
    for operation in operations {
        operation.origin.hash(&mut hasher);
        hash_operation_kind(&operation.kind, &mut hasher);
    }
    final_height.hash(&mut hasher);
    PlanStructureSignature::from_hash(hasher.finish())
}

fn hash_operation_kind(kind: &TerrainOpKind, hasher: &mut impl Hasher) {
    match kind {
        TerrainOpKind::Seed { source, output } => {
            0_u8.hash(hasher);
            source.hash(hasher);
            output.hash(hasher);
        }
        TerrainOpKind::EvaluateMask {
            input_height,
            input_fields,
            output_mask,
        } => {
            1_u8.hash(hasher);
            input_height.hash(hasher);
            input_fields.hash(hasher);
            output_mask.hash(hasher);
        }
        TerrainOpKind::RunLayerKernel {
            layer,
            type_id,
            input_height,
            input_fields,
            output_candidate,
            output_fields,
        } => {
            2_u8.hash(hasher);
            layer.hash(hasher);
            type_id.hash(hasher);
            input_height.hash(hasher);
            input_fields.hash(hasher);
            output_candidate.hash(hasher);
            output_fields.hash(hasher);
        }
        TerrainOpKind::CompositeLayer {
            layer,
            base,
            candidate,
            mask,
            output,
        } => {
            3_u8.hash(hasher);
            layer.hash(hasher);
            base.hash(hasher);
            candidate.hash(hasher);
            mask.hash(hasher);
            output.hash(hasher);
        }
        TerrainOpKind::CompositeGroup {
            group,
            parent,
            private_seed,
            child_output,
            mask,
            output,
            mode,
            aux,
        } => {
            4_u8.hash(hasher);
            group.hash(hasher);
            parent.hash(hasher);
            private_seed.hash(hasher);
            child_output.hash(hasher);
            mask.hash(hasher);
            output.hash(hasher);
            mode.hash(hasher);
            for field in aux {
                field.field.hash(hasher);
                field.parent.hash(hasher);
                field.child.hash(hasher);
                field.output.hash(hasher);
            }
        }
        TerrainOpKind::PublishOutput { output, source } => {
            5_u8.hash(hasher);
            output.hash(hasher);
            source.hash(hasher);
        }
    }
}

/// Fixed FNV-1a rather than `DefaultHasher`, whose algorithm is not a stability contract.
struct StablePlanHasher(u64);

impl Default for StablePlanHasher {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for StablePlanHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
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
