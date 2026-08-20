//! Plan-local identity and structural compatibility stamps.

/// Monotonic identity of the authored structure from which a plan was built.
///
/// This is deliberately distinct from terrain output revisions: content edits
/// advance output freshness without invalidating plan-local operation or field
/// identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlanStructureRevision(u64);

impl PlanStructureRevision {
    pub const INITIAL: Self = Self(0);

    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Default for PlanStructureRevision {
    fn default() -> Self {
        Self::INITIAL
    }
}

/// Source-compatibility stamp carried by a compiled plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TerrainPlanStamp {
    pub structure_revision: PlanStructureRevision,
}

impl TerrainPlanStamp {
    pub const fn new(structure_revision: PlanStructureRevision) -> Self {
        Self { structure_revision }
    }
}

/// Index of one operation within a particular compiled plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlanOpId(u32);

impl PlanOpId {
    pub(crate) fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("terrain plan operation count exceeds u32"))
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Index of one logical field within a particular compiled plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FieldSlot(u32);

impl FieldSlot {
    pub(crate) fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("terrain plan field count exceeds u32"))
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}
