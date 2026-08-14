//! Authoring chronology and stack evaluation.
//!
//! Artists may create and edit project content in any order (biomes before
//! shapes, materials before simulations, snow rules before mountains, etc.).
//! Authoring chronology does not create a separate stage schedule. Evaluation
//! walks the resulting [`LayerStack`] tree bottom-to-top, preserving its group
//! and solo semantics. Chronology matters only when authoring operations leave
//! the stack in a different order.
//!
//! Incomplete but valid projects produce empty outputs and soft diagnostics —
//! never fatal errors solely because coverage is zero or a stage has no sources.

use crate::biome_definition::{BiomeDefinition, BiomeLibrary};
use crate::biome_paint::BiomeLayer;
use crate::landscape_blueprint::EvalStage;
use crate::layer::{LayerKind, LayerStack, WorkflowStage};

/// Non-blocking diagnostic for an incomplete-but-valid project state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoftDiagnostic {
    pub code: String,
    pub message: String,
}

impl SoftDiagnostic {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

/// Content may be authored in any chronology; evaluation follows the resulting stack tree.
pub fn authoring_order_is_arbitrary() -> bool {
    true
}

/// Metadata order used for invalidation and optional explicit authoring sorts.
///
/// This does not reorder [`crate::eval::StackEvaluator`] execution.
pub fn evaluation_eval_stage_order() -> &'static [EvalStage] {
    &[
        EvalStage::Blueprint,
        EvalStage::PreBiomeFields,
        EvalStage::Climate,
        EvalStage::BiomePlacement,
        EvalStage::FieldBlend,
        EvalStage::SharedHydro,
        EvalStage::Materials,
        EvalStage::Vegetation,
        EvalStage::FineDetail,
        EvalStage::Export,
    ]
}

/// Artist workflow metadata stages — presentation hints only, not eval order.
pub fn workflow_stage_metadata_order() -> &'static [WorkflowStage] {
    &WorkflowStage::ALL
}

/// Actual height evaluation entry outline for a single terrain stack.
pub fn world_eval_outline() -> Vec<&'static str> {
    vec!["terrain_stack"]
}

/// Collect soft diagnostics for incomplete-but-valid project content.
///
/// These never block evaluation or serialization.
pub fn incomplete_project_diagnostics(
    stack: &LayerStack,
    biome_library: &BiomeLibrary,
    biome_layers: &[BiomeLayer],
) -> Vec<SoftDiagnostic> {
    let mut out = Vec::new();

    if stack.flatten_layers().is_empty() {
        out.push(SoftDiagnostic::new(
            "stack_without_shape_layers",
            "Terrain stack has no layers yet — evaluates as identity height.",
        ));
    }

    for def in &biome_library.definitions {
        diagnose_biome_definition(def, &mut out);
    }

    for layer in stack.flatten_layers() {
        diagnose_layer(layer, &mut out);
    }

    for bl in biome_layers {
        if bl.channels.is_empty() {
            out.push(SoftDiagnostic::new(
                "empty_biome_coverage",
                format!(
                    "Placement layer '{}' has no painted channels — coverage is empty (valid).",
                    bl.name
                ),
            ));
        }
    }

    out.sort_by(|a, b| (&a.code, &a.message).cmp(&(&b.code, &b.message)));
    out.dedup();

    out
}

fn diagnose_layer(layer: &crate::layer::Layer, out: &mut Vec<SoftDiagnostic>) {
    match &layer.kind {
        LayerKind::Materials(p) if p.rules.is_empty() && p.strata.is_empty() => {
            out.push(SoftDiagnostic::new(
                "materials_without_rules",
                format!(
                    "Materials layer '{}' has no rules yet — height passes through.",
                    layer.common.name
                ),
            ));
        }
        LayerKind::HydraulicErosion(_)
        | LayerKind::ThermalErosion(_)
        | LayerKind::StreamPowerErosion(_) => {
            if !layer.common.enabled {
                out.push(SoftDiagnostic::new(
                    "simulation_disabled",
                    format!(
                        "Simulation '{}' is present but disabled.",
                        layer.common.name
                    ),
                ));
            }
        }
        _ => {}
    }
}

fn diagnose_biome_definition(def: &BiomeDefinition, out: &mut Vec<SoftDiagnostic>) {
    if def.group_id.is_none() {
        out.push(SoftDiagnostic::new(
            "biome_unlinked",
            format!(
                "Biome '{}' has no linked implementation group yet.",
                def.name
            ),
        ));
    }
    if def.placement.rules.is_none() {
        out.push(SoftDiagnostic::new(
            "biome_empty_placement_rules",
            format!(
                "Biome '{}' has no placement rules — may produce zero coverage (valid).",
                def.name
            ),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authoring_chronology_has_no_separate_stage_contract() {
        assert!(authoring_order_is_arbitrary());
    }
}
