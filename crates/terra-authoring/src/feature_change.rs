use crate::AuthoredFeatureId;
use terra_world::WorldBounds;

/// Active world-space coverage before and after one authored-feature mutation.
///
/// Disabled, empty, and removed records have no active bounds. Carrying both
/// sides lets callers update runtime indexes and later invalidation systems
/// without inspecting or re-deriving the mutated record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureChange {
    pub id: AuthoredFeatureId,
    pub previous_bounds: Option<WorldBounds>,
    pub replacement_bounds: Option<WorldBounds>,
}
