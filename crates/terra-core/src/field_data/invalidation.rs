//! Phase 11 Rule 4 — recompute dependencies when shared physical state changes.
//!
//! When height (or another producer field) changes, invalidate derived analysis
//! and hydro dependents. Iterative algorithms that own temporary internal copies
//! of slope/flow/etc. should call [`IterativeFieldGuard`] so the shared cache is
//! not thrashed mid-iteration.

use super::FieldId;

/// Fields that must be recomputed after `changed` is written (excluding `changed`).
///
/// Order is a suggested refresh priority (cheap local derivatives first).
pub fn fields_invalidated_by(changed: &FieldId) -> Vec<FieldId> {
    match changed {
        FieldId::Height | FieldId::BaseElevation | FieldId::BedrockHeight => {
            height_dependents().to_vec()
        }
        FieldId::SedimentThickness
        | FieldId::SoilDepth
        | FieldId::SoilThickness
        | FieldId::SandDepth
        | FieldId::DebrisDepth
        | FieldId::SnowDepth => {
            // Surface composition changes slope of the *exposed* DEM only when
            // height was already updated; still refresh stability / wetness cues.
            vec![
                FieldId::TalusStability,
                FieldId::Instability,
                FieldId::SoilMoisture,
            ]
        }
        FieldId::Hardness
        | FieldId::Lithology
        | FieldId::Erodibility
        | FieldId::StrataMaterialId => {
            vec![
                FieldId::ErosionRate,
                FieldId::DepositionRate,
                FieldId::TalusStability,
            ]
        }
        FieldId::Rainfall | FieldId::Water | FieldId::WaterDepth => {
            vec![
                FieldId::Wetness,
                FieldId::WaterDischarge,
                FieldId::FlowAccumulation,
                FieldId::SoilMoisture,
            ]
        }
        FieldId::FlowDirection | FieldId::FlowAccumulation => {
            vec![
                FieldId::StreamOrder,
                FieldId::ChannelMask,
                FieldId::WatershedId,
                FieldId::WatershedBoundary,
                FieldId::DrainageDensity,
                FieldId::DistanceToChannel,
                FieldId::WaterDischarge,
            ]
        }
        _ => Vec::new(),
    }
}

/// Canonical dependents of a DEM height write (Rule 4 example set).
pub fn height_dependents() -> &'static [FieldId] {
    &[
        FieldId::Gradient,
        FieldId::Slope,
        FieldId::Aspect,
        FieldId::Curvature,
        FieldId::MeanCurvature,
        FieldId::ProfileCurvature,
        FieldId::PlanCurvature,
        FieldId::GaussianCurvature,
        FieldId::Laplacian,
        FieldId::Convexity,
        FieldId::Concavity,
        FieldId::Cavity,
        FieldId::Roughness,
        FieldId::RidgeLikelihood,
        FieldId::ValleyLikelihood,
        FieldId::Normals,
        // Hydro / drainage rebuild
        FieldId::FlowDirection,
        FieldId::FlowAccumulation,
        FieldId::WaterDischarge,
        FieldId::WatershedId,
        FieldId::WatershedBoundary,
        FieldId::ChannelMask,
        FieldId::StreamOrder,
        FieldId::DrainageDensity,
        FieldId::DistanceToChannel,
        FieldId::DistanceToRidge,
        FieldId::DistanceToValley,
        FieldId::ChannelWidth,
        FieldId::Wetness,
    ]
}

/// Shared physical geology fields that process ops should prefer over ad-hoc height offsets.
pub fn shared_physical_fields() -> &'static [FieldId] {
    &[
        FieldId::Height,
        FieldId::BedrockHeight,
        FieldId::SedimentThickness,
        FieldId::SoilDepth,
        FieldId::Water,
        FieldId::WaterDepth,
        FieldId::FlowDirection,
        FieldId::FlowAccumulation,
        FieldId::Hardness,
        FieldId::Lithology,
    ]
}

/// Suppresses shared derived-field invalidation while an iterative solver owns
/// temporary internal slope / flow / sediment scratch.
#[derive(Debug, Default)]
pub struct IterativeFieldGuard {
    depth: u32,
}

impl IterativeFieldGuard {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enter(&mut self) {
        self.depth = self.depth.saturating_add(1);
    }

    pub fn exit(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    pub fn is_active(&self) -> bool {
        self.depth > 0
    }

    /// Returns dependents to invalidate, or empty while a temporary internal
    /// field set is owned by an iterative algorithm.
    pub fn invalidate_for(&self, changed: &FieldId) -> Vec<FieldId> {
        if self.is_active() {
            Vec::new()
        } else {
            fields_invalidated_by(changed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn height_invalidates_slope_and_flow() {
        let deps = fields_invalidated_by(&FieldId::Height);
        assert!(deps.contains(&FieldId::Slope));
        assert!(deps.contains(&FieldId::Curvature));
        assert!(deps.contains(&FieldId::FlowDirection));
        assert!(deps.contains(&FieldId::FlowAccumulation));
        assert!(deps.contains(&FieldId::DrainageDensity));
    }

    #[test]
    fn iterative_guard_defers_invalidation() {
        let mut g = IterativeFieldGuard::new();
        g.enter();
        assert!(g.invalidate_for(&FieldId::Height).is_empty());
        g.exit();
        assert!(!g.invalidate_for(&FieldId::Height).is_empty());
    }

    #[test]
    fn shared_physical_state_lists_lithology() {
        assert!(shared_physical_fields().contains(&FieldId::Lithology));
        assert!(shared_physical_fields().contains(&FieldId::BedrockHeight));
    }
}
