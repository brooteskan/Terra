//! Change classification at the authoring-to-plan boundary.

use crate::deps::NodeRef;
use crate::field_data::FieldId;
use crate::tiling::UvRect;

use super::PlanOpId;

/// Resolution-independent spatial scope of a content edit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlanDirtyScope {
    Region(UvRect),
    FullField,
}

impl PlanDirtyScope {
    /// Conservatively combine dirty scopes. Whole-field dirtiness is absorbing.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::FullField, _) | (_, Self::FullField) => Self::FullField,
            (Self::Region(a), Self::Region(b)) => Self::Region(a.union(b)),
        }
    }
}

/// Resolution-independent dirty scope after one or more plan operations.
///
/// The authored footprint remains in normalized UV while localized operation
/// reach accumulates in samples. A backend converts both only after choosing
/// its concrete preview/export resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropagatedDirtyScope {
    pub scope: PlanDirtyScope,
    pub halo_samples: u32,
}

impl PropagatedDirtyScope {
    pub const fn new(scope: PlanDirtyScope) -> Self {
        Self {
            scope,
            halo_samples: 0,
        }
    }

    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            scope: self.scope.merge(other.scope),
            halo_samples: self.halo_samples.max(other.halo_samples),
        }
    }

    #[must_use]
    pub fn expand(self, reach: crate::invalidation::Reach) -> Self {
        match reach {
            crate::invalidation::Reach::Full => Self {
                scope: PlanDirtyScope::FullField,
                halo_samples: 0,
            },
            crate::invalidation::Reach::Localized { halo_samples } => Self {
                halo_samples: self.halo_samples.saturating_add(halo_samples),
                ..self
            },
        }
    }

    pub const fn is_full(self) -> bool {
        matches!(self.scope, PlanDirtyScope::FullField)
    }
}

/// Semantic class of an authored edit.
///
/// Callers may attach more detailed command data, but they must reduce it to one
/// of these classes before deciding whether the plan shape is still reusable.
#[derive(Debug, Clone, PartialEq)]
pub enum TerrainEditClass {
    /// Camera, selection, rename, collapse, colour, and similar presentation-only edits.
    ViewOnly,
    /// Base/stroke/mask pixels or another field payload changed.
    Content {
        owner: NodeRef,
        fields: Vec<FieldId>,
        scope: PlanDirtyScope,
    },
    /// Numeric/configuration data changed without changing operation/dependency shape.
    Parameters { owner: NodeRef },
    /// Preview metrics, device, format, or quality require backend re-realization.
    Resources,
    /// Topology, operation kind, group semantics, solo selection, or dependency edges changed.
    Structure,
}

/// Independent work axes resulting from one or more edits.
///
/// This is intentionally not a severity enum: a batch can require both payload
/// patching and resource realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TerrainPlanWork {
    pub compile_structure: bool,
    pub patch_parameters: bool,
    pub patch_content: bool,
    pub realize_resources: bool,
}

/// Operations whose authored/runtime payload must be refreshed without
/// rebuilding the structural plan.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TerrainPlanPatch {
    pub operations: Vec<PlanOpId>,
}

impl TerrainPlanWork {
    pub const NONE: Self = Self {
        compile_structure: false,
        patch_parameters: false,
        patch_content: false,
        realize_resources: false,
    };

    #[must_use]
    pub const fn merge(self, other: Self) -> Self {
        Self {
            compile_structure: self.compile_structure || other.compile_structure,
            patch_parameters: self.patch_parameters || other.patch_parameters,
            patch_content: self.patch_content || other.patch_content,
            realize_resources: self.realize_resources || other.realize_resources,
        }
    }

    pub const fn is_empty(self) -> bool {
        !self.compile_structure
            && !self.patch_parameters
            && !self.patch_content
            && !self.realize_resources
    }
}

impl TerrainEditClass {
    pub const fn required_work(&self) -> TerrainPlanWork {
        match self {
            Self::ViewOnly => TerrainPlanWork::NONE,
            Self::Content { .. } => TerrainPlanWork {
                patch_content: true,
                ..TerrainPlanWork::NONE
            },
            Self::Parameters { .. } => TerrainPlanWork {
                patch_parameters: true,
                ..TerrainPlanWork::NONE
            },
            Self::Resources => TerrainPlanWork {
                realize_resources: true,
                ..TerrainPlanWork::NONE
            },
            Self::Structure => TerrainPlanWork {
                compile_structure: true,
                ..TerrainPlanWork::NONE
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_field_scope_absorbs_regions() {
        let region = PlanDirtyScope::Region(UvRect::from_center_radius(0.5, 0.5, 0.1));
        assert_eq!(
            region.merge(PlanDirtyScope::FullField),
            PlanDirtyScope::FullField
        );
        assert_eq!(
            PlanDirtyScope::FullField.merge(region),
            PlanDirtyScope::FullField
        );
    }

    #[test]
    fn work_axes_merge_without_losing_independent_requirements() {
        let work = TerrainEditClass::Parameters {
            owner: NodeRef::Layer(crate::ids::LayerId::from_u128(1)),
        }
        .required_work()
        .merge(TerrainEditClass::Resources.required_work());
        assert!(work.patch_parameters);
        assert!(work.realize_resources);
        assert!(!work.compile_structure);
        assert!(!work.patch_content);
    }
}
