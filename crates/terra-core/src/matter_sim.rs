//! Behavior for physical matter simulations — diagnostics, downstream
//! consumers, and scenario-output sync.
//!
//! Matter types — Water/Rivers, Snow, Sand, Debris — sit on top of existing
//! [`crate::simulation_scenario::SimulationScenario`] and Simulation Layers.
//! The persisted config vocabulary (`MatterSimConfig` and the artist/param
//! structs it carries) therefore lives in `simulation_scenario` as document
//! state — `SimulationScenario` embeds it — and is re-exported here so callers
//! keep their `matter_sim::` paths. This module owns the *behavior*:
//! diagnostics, selective apply, and the consumer/output surface. Solvers are
//! not rewritten.

use crate::domain::SoftDiagnostic;
use crate::fields::FieldId;
use crate::simulation_scenario::SimulationScenario;

pub use crate::simulation_scenario::{
    MatterAdvancedParams, MatterArtistControls, MatterArtistSource, MatterSimConfig, MatterType,
};

impl MatterSimConfig {
    /// Soft diagnostics for unstable / excessive Advanced settings.
    pub fn diagnostics(&self) -> Vec<SoftDiagnostic> {
        diagnose_matter_sim(self)
    }
}

/// Non-blocking diagnostics for unstable or excessive settings.
pub fn diagnose_matter_sim(cfg: &MatterSimConfig) -> Vec<SoftDiagnostic> {
    let mut out = Vec::new();
    let name = cfg.matter.label();
    if cfg.sources.is_empty() {
        out.push(SoftDiagnostic::new(
            "matter_no_sources",
            format!("‘{name}’ has no sources — paint or enable rainfall / snowfall."),
        ));
    }
    if cfg.advanced.iterations > 256 {
        out.push(SoftDiagnostic::new(
            "matter_excessive_iterations",
            format!(
                "‘{name}’ iterations ({}) are very high — may be slow or unstable.",
                cfg.advanced.iterations
            ),
        ));
    }
    if cfg.artist.strength > 1.5 {
        out.push(SoftDiagnostic::new(
            "matter_excessive_strength",
            format!("‘{name}’ strength is unusually high — results may overshoot."),
        ));
    }
    if cfg.artist.repose_angle_deg < 5.0 || cfg.artist.repose_angle_deg > 70.0 {
        out.push(SoftDiagnostic::new(
            "matter_unstable_angle",
            format!(
                "‘{name}’ repose angle {:.0}° is extreme — check Advanced / artist angle.",
                cfg.artist.repose_angle_deg
            ),
        ));
    }
    if cfg.advanced.spawn_amount > 20.0 {
        out.push(SoftDiagnostic::new(
            "matter_excessive_spawn",
            format!("‘{name}’ spawn amount is excessive — reduce under Advanced."),
        ));
    }
    if cfg.advanced.allow_height_delta
        && cfg
            .apply_selected
            .iter()
            .any(|f| matches!(f, FieldId::Height))
    {
        out.push(SoftDiagnostic::new(
            "matter_height_apply_enabled",
            format!(
                "‘{name}’ will apply height deltas when Apply is used — non-destructive until then."
            ),
        ));
    }
    if cfg.artist.paint_sources && !cfg.sources.iter().any(|s| s.is_paintable()) {
        out.push(SoftDiagnostic::new(
            "matter_paint_no_target",
            format!("Source painting is on but ‘{name}’ has no paintable source."),
        ));
    }
    out
}

/// Attach matter config metadata onto a scenario via output application fields.
pub fn sync_scenario_outputs_from_matter(scenario: &mut SimulationScenario, cfg: &MatterSimConfig) {
    scenario.output_application.selected_fields = cfg.apply_selected.clone();
    scenario.output_application.apply_height = cfg.advanced.allow_height_delta;
    scenario.output_application.apply_aux = true;
}

/// Downstream consumers that can read matter output maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatterOutputConsumer {
    WorldRules,
    Materials,
    BiomePlacement,
    Scatter,
    OtherSimulation,
}

impl MatterOutputConsumer {
    pub fn can_consume(self, field: &FieldId) -> bool {
        match self {
            Self::WorldRules => !matches!(field, FieldId::Height),
            Self::Materials => matches!(
                field,
                FieldId::Wetness
                    | FieldId::Snow
                    | FieldId::SnowMaterialMask
                    | FieldId::SandMaterialMask
                    | FieldId::Materials
                    | FieldId::Floodplain
                    | FieldId::WaterDepth
            ),
            Self::BiomePlacement => matches!(
                field,
                FieldId::Wetness
                    | FieldId::Snow
                    | FieldId::Temperature
                    | FieldId::SandDepth
                    | FieldId::WaterDepth
            ),
            Self::Scatter => matches!(
                field,
                FieldId::ScatterCandidates
                    | FieldId::Instability
                    | FieldId::SlidePath
                    | FieldId::SandDepth
                    | FieldId::DebrisDepth
            ),
            Self::OtherSimulation => true,
        }
    }
}

/// Fields from a matter config available to a given consumer.
pub fn outputs_for_consumer(cfg: &MatterSimConfig, consumer: MatterOutputConsumer) -> Vec<FieldId> {
    cfg.outputs
        .iter()
        .filter(|f| consumer.can_consume(f))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation_scenario::ScenarioPassKind;

    #[test]
    fn water_declares_expected_io() {
        let w = MatterSimConfig::water_rivers();
        assert!(w.outputs.contains(&FieldId::RiverChannel));
        assert!(w.outputs.contains(&FieldId::WaterDepth));
        assert!(w.outputs.contains(&FieldId::Floodplain));
        assert!(w.inputs.contains(&FieldId::Height));
    }

    #[test]
    fn sand_build_scenario_uses_sand_pass() {
        let s = MatterSimConfig::sand().build_scenario();
        assert!(s.passes.iter().any(|p| p.kind == ScenarioPassKind::Sand));
    }
}
