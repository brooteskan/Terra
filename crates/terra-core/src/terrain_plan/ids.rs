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

    /// Advance to a new authored-structure generation.
    ///
    /// Revision zero is not special after construction, so wrapping is both
    /// deterministic and consistent with the output-revision counters used by
    /// the terrain runtime.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
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

/// Deterministic signature of a plan's logical fields and ordered operations.
///
/// Numeric authored parameters and the source revision are intentionally absent:
/// they can change while the compiled operation/resource shape remains reusable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PlanStructureSignature(u64);

impl PlanStructureSignature {
    pub(crate) const fn from_hash(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Index of one operation within a particular compiled plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlanOpId(u32);

impl PlanOpId {
    /// Construct a plan-local operation id while iterating a validated plan.
    ///
    /// Backends use this to address immutable analysis tables in the same plan;
    /// callers must not carry the id across structure revisions.
    pub fn from_index(index: usize) -> Self {
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
