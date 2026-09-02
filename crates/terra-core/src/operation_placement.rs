//! Per-operation placement within a Biome (Develop workspace).
//!
//! Artist rule:
//!   effective operation placement = Biome Effective Placement × operation PlacementDefinition
//!
//! Default Apply Where = Entire Biome (identity local placement). Artists never
//! manually assign a Biome Mask — inheritance is automatic from the enclosing biome.

pub use crate::layer::{ApplyWhere, DevelopCategory, OperationPlacement};
use crate::mask::{
    CompareOp, Condition, ConditionChannel, Distribution, PlacementDefinition, PlacementSource,
    RuleGroup, RuleGroupMode, RuleNode,
};

impl ApplyWhere {
    pub fn all() -> &'static [ApplyWhere] {
        &[
            Self::EntireBiome,
            Self::PaintedRestriction,
            Self::HeightRange,
            Self::SlopeRange,
            Self::NearWater,
            Self::NearRivers,
            Self::FlowRange,
            Self::Curvature,
            Self::CustomConditions,
            Self::AdvancedMask,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::EntireBiome => "Entire Biome",
            Self::PaintedRestriction => "Painted Restriction",
            Self::HeightRange => "Height Range",
            Self::SlopeRange => "Slope Range",
            Self::NearWater => "Near Water",
            Self::NearRivers => "Near Rivers",
            Self::FlowRange => "Flow Range",
            Self::Curvature => "Curvature",
            Self::CustomConditions => "Custom Conditions",
            Self::AdvancedMask => "Advanced Mask",
        }
    }
}

impl OperationPlacement {
    /// New ops default to Entire Biome.
    pub fn entire_biome() -> Self {
        Self::default()
    }

    /// True when local placement is identity (full coverage inside biome).
    pub fn is_entire_biome(&self) -> bool {
        matches!(self.apply_where, ApplyWhere::EntireBiome)
            && self.definition.root.children.is_empty()
            && self.definition.coverage.is_empty()
            && !matches!(self.definition.source, PlacementSource::Custom)
    }

    /// Rebuild local [`PlacementDefinition`] from Apply Where presets (unless Advanced).
    pub fn sync_definition_from_apply_where(&mut self) {
        if matches!(self.apply_where, ApplyWhere::AdvancedMask) {
            return;
        }
        if matches!(self.apply_where, ApplyWhere::EntireBiome) {
            self.definition = PlacementDefinition::default();
            return;
        }
        if matches!(self.definition.source, PlacementSource::Custom)
            && matches!(self.apply_where, ApplyWhere::CustomConditions)
        {
            // Keep custom conditions artist edits.
            return;
        }
        let mut def = PlacementDefinition::default();
        def.source = PlacementSource::Rules;
        let mut children = Vec::new();
        match self.apply_where {
            ApplyWhere::EntireBiome | ApplyWhere::AdvancedMask | ApplyWhere::CustomConditions => {}
            ApplyWhere::PaintedRestriction => {
                // Paint entry is bound later via paint_mask; leave rules empty → paint-only when set.
            }
            ApplyWhere::HeightRange => {
                children.push(RuleNode::Condition(Condition {
                    channel: ConditionChannel::Height,
                    op: CompareOp::Between,
                    a: self.height_min,
                    b: self.height_max,
                    falloff: 8.0,
                    seed: 0,
                    frequency: 0.02,
                }));
            }
            ApplyWhere::SlopeRange => {
                children.push(RuleNode::Condition(Condition {
                    channel: ConditionChannel::Slope,
                    op: CompareOp::Between,
                    a: self.slope_min,
                    b: self.slope_max,
                    falloff: 2.0,
                    seed: 0,
                    frequency: 0.02,
                }));
            }
            ApplyWhere::NearWater => {
                children.push(RuleNode::Condition(Condition {
                    channel: ConditionChannel::WaterDistance,
                    op: CompareOp::Below,
                    a: self.near_distance_m,
                    b: 0.0,
                    falloff: self.near_distance_m * 0.25,
                    seed: 0,
                    frequency: 0.02,
                }));
            }
            ApplyWhere::NearRivers | ApplyWhere::FlowRange => {
                children.push(RuleNode::Condition(Condition {
                    channel: ConditionChannel::Flow,
                    op: CompareOp::Above,
                    a: self.flow_min,
                    b: 0.0,
                    falloff: 0.05,
                    seed: 0,
                    frequency: 0.02,
                }));
            }
            ApplyWhere::Curvature => {
                children.push(RuleNode::Condition(Condition {
                    channel: ConditionChannel::Curvature,
                    op: CompareOp::Above,
                    a: 0.02,
                    b: 0.0,
                    falloff: 0.01,
                    seed: 0,
                    frequency: 0.02,
                }));
            }
        }
        def.root = RuleGroup {
            mode: RuleGroupMode::All,
            invert: false,
            children,
        };
        def.recompute_hash();
        self.definition = def;
    }

    /// Compile local distribution for layer.common.masks (identity when Entire Biome).
    pub fn compile_local_distribution(&self) -> Distribution {
        if self.is_entire_biome() {
            return Distribution::new();
        }
        match self.definition.source {
            PlacementSource::Custom => self
                .definition
                .custom_stack
                .clone()
                .unwrap_or_else(|| self.definition.compile()),
            PlacementSource::Rules => self.definition.compile(),
        }
    }

    /// Apply local compile onto a layer's mask stack (does not write biome mask).
    pub fn write_to_layer_masks(&self, masks: &mut Distribution) {
        *masks = self.compile_local_distribution();
    }

    /// Readable summary without raw mask IDs.
    pub fn summary_lines(&self, biome_name: &str) -> Vec<String> {
        let mut lines = vec![format!("Applies within {biome_name}")];
        match self.apply_where {
            ApplyWhere::EntireBiome => {
                lines.push("Where: Entire Biome".into());
            }
            ApplyWhere::PaintedRestriction => {
                lines.push("Where: painted restriction inside biome".into());
            }
            ApplyWhere::HeightRange => {
                lines.push(format!(
                    "Where height is between {:.0} m and {:.0} m",
                    self.height_min, self.height_max
                ));
            }
            ApplyWhere::SlopeRange => {
                lines.push(format!(
                    "Where slope is between {:.0}° and {:.0}°",
                    self.slope_min, self.slope_max
                ));
            }
            ApplyWhere::NearWater => {
                lines.push(format!(
                    "Where within {:.0} m of water",
                    self.near_distance_m
                ));
            }
            ApplyWhere::NearRivers => {
                lines.push(format!(
                    "Where near rivers (flow above {:.2})",
                    self.flow_min
                ));
            }
            ApplyWhere::FlowRange => {
                lines.push(format!("Where flow is above {:.2}", self.flow_min));
            }
            ApplyWhere::Curvature => {
                lines.push("Where curvature is high".into());
            }
            ApplyWhere::CustomConditions => {
                lines.push("Where custom conditions apply".into());
                for line in condition_summaries(&self.definition) {
                    lines.push(format!("And {line}"));
                }
            }
            ApplyWhere::AdvancedMask => {
                lines.push("Where: Advanced Mask Stack (custom)".into());
            }
        }
        // Extra AND lines for height+slope style summaries when both present in rules.
        if matches!(self.apply_where, ApplyWhere::HeightRange) {
            // already covered
        }
        lines
    }

    /// Single paragraph for compact UI.
    pub fn summary_paragraph(&self, biome_name: &str) -> String {
        self.summary_lines(biome_name).join("\n")
    }
}

fn condition_summaries(def: &PlacementDefinition) -> Vec<String> {
    let mut out = Vec::new();
    collect_conditions(&def.root, &mut out);
    out
}

fn collect_conditions(group: &RuleGroup, out: &mut Vec<String>) {
    for child in &group.children {
        match child {
            RuleNode::Condition(c) => out.push(condition_phrase(c)),
            RuleNode::Group(g) => collect_conditions(g, out),
            RuleNode::CoverageRef { .. } => out.push("coverage term".into()),
        }
    }
}

fn condition_phrase(c: &Condition) -> String {
    match (c.channel, c.op) {
        (ConditionChannel::Height, CompareOp::Above) => {
            format!("height is above {:.0} m", c.a)
        }
        (ConditionChannel::Height, CompareOp::Below) => {
            format!("height is below {:.0} m", c.a)
        }
        (ConditionChannel::Height, CompareOp::Between) => {
            format!("height is between {:.0} m and {:.0} m", c.a, c.b)
        }
        (ConditionChannel::Slope, CompareOp::Above) => format!("slope is above {:.0}°", c.a),
        (ConditionChannel::Slope, CompareOp::Below) => format!("slope is below {:.0}°", c.a),
        (ConditionChannel::Slope, CompareOp::Between) => {
            format!("slope is between {:.0}° and {:.0}°", c.a, c.b)
        }
        (ConditionChannel::Flow, CompareOp::Above) => format!("flow is above {:.2}", c.a),
        (ConditionChannel::WaterDistance, CompareOp::Below) => {
            format!("within {:.0} m of water", c.a)
        }
        (ConditionChannel::Curvature, _) => "curvature matches".into(),
        _ => format!("{:?} {:?}", c.channel, c.op),
    }
}

impl DevelopCategory {
    pub fn all() -> &'static [DevelopCategory] {
        &[
            Self::Terrain,
            Self::Materials,
            Self::Simulation,
            Self::Vegetation,
            Self::Objects,
            Self::Placement,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Terrain => "Terrain",
            Self::Materials => "Materials",
            Self::Simulation => "Simulation",
            Self::Vegetation => "Vegetation",
            Self::Objects => "Objects",
            Self::Placement => "Placement",
        }
    }

    pub fn add_action_label(self) -> &'static str {
        match self {
            Self::Terrain => "Add Terrain Effect",
            Self::Materials => "Add Material",
            Self::Simulation => "Add Simulation",
            Self::Vegetation => "Add Vegetation",
            Self::Objects => "Add Object",
            Self::Placement => "Edit Placement",
        }
    }

    pub fn biome_section(self) -> Option<crate::layer::BiomeSection> {
        use crate::layer::BiomeSection;
        match self {
            Self::Terrain => Some(BiomeSection::Filters),
            Self::Materials => Some(BiomeSection::Materials),
            Self::Simulation => Some(BiomeSection::LocalSims),
            Self::Vegetation | Self::Objects => Some(BiomeSection::Objects),
            Self::Placement => None,
        }
    }

    /// Default layer kind for contextual Add actions.
    pub fn default_layer_kind(self) -> crate::layer::LayerKind {
        use crate::layer::{
            EffectFilterParams, LayerKind, MaterialsParams, SandSimParams, VegetationParams,
        };
        match self {
            Self::Terrain => LayerKind::EffectFilter(EffectFilterParams::default()),
            Self::Materials => LayerKind::Materials(MaterialsParams::default()),
            Self::Simulation => LayerKind::SandSimulation(SandSimParams::default()),
            Self::Vegetation => LayerKind::Vegetation(VegetationParams::default()),
            Self::Objects => LayerKind::Vegetation(VegetationParams {
                density: 0.35,
                ..Default::default()
            }),
            Self::Placement => LayerKind::EffectFilter(EffectFilterParams::default()),
        }
    }
}

/// Create a Develop operation with Entire Biome placement (default).
pub fn create_develop_operation(
    category: DevelopCategory,
    name: impl Into<String>,
) -> crate::layer::Layer {
    let mut layer = crate::layer::Layer::new(name, category.default_layer_kind());
    layer.common.operation_placement = OperationPlacement::entire_biome();
    layer.common.develop_category = Some(category);
    layer.sync_operation_placement_masks();
    layer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entire_biome_default_is_identity() {
        let p = OperationPlacement::entire_biome();
        assert!(p.is_entire_biome());
        assert!(p.compile_local_distribution().is_empty());
        let lines = p.summary_lines("Alpine Mountains");
        assert!(lines[0].contains("Alpine Mountains"));
        assert!(lines.iter().any(|l| l.contains("Entire Biome")));
    }

    #[test]
    fn height_range_summary_readable() {
        let mut p = OperationPlacement::default();
        p.apply_where = ApplyWhere::HeightRange;
        p.height_min = 1200.0;
        p.height_max = 5000.0;
        p.sync_definition_from_apply_where();
        let text = p.summary_paragraph("Alpine Mountains");
        assert!(text.contains("Alpine Mountains"));
        assert!(text.contains("1200"));
        assert!(!text.contains("MaskId"));
        assert!(!p.compile_local_distribution().is_empty());
    }

    #[test]
    fn slope_below_summary() {
        let mut p = OperationPlacement::default();
        p.apply_where = ApplyWhere::SlopeRange;
        p.slope_min = 0.0;
        p.slope_max = 50.0;
        p.sync_definition_from_apply_where();
        let text = p.summary_paragraph("Alpine Mountains");
        assert!(text.contains("50"));
    }
}
