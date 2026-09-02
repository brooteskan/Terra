//! World Rules — first-class condition-driven operations across Regions and Biomes.
//!
//! Artist intentions such as “snow above 1 200 m” or “beaches near water on shallow
//! slopes” are authored as project entities (not hidden generated masks). Placement
//! compiles into the existing [`PlacementDefinition`] → [`Distribution`] bake path.
//!
//! Multiple effects share one rule placement; each effect can be toggled independently.
//! Rules may exist before they match any terrain — zero coverage is not an error.

use crate::biome_definition::BiomeDefinitionId;
use crate::domain::SoftDiagnostic;
use crate::heightfield::HeightfieldMetrics;
use crate::landscape_blueprint::EvalStage;
use crate::layer::{BlendMode, BuildStatus, CachePolicy};
use crate::mask::{
    coverage_estimate, CompareOp, Condition, ConditionChannel, DistBakeContext, Distribution,
    PlacementDefinition, PlacementSource, RuleGroup, RuleGroupMode, RuleNode,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable identity for a World Rule project entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorldRuleId(pub Uuid);

impl WorldRuleId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for WorldRuleId {
    fn default() -> Self {
        Self::new()
    }
}

/// Where a World Rule applies geographically.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum WorldRuleScope {
    /// Every Biome in the project.
    #[default]
    EntireWorld,
    /// Only the listed Biome definitions (and their linked groups).
    SelectedBiomes(Vec<BiomeDefinitionId>),
    /// World minus listed Biomes.
    Excluding {
        #[serde(default)]
        biomes: Vec<BiomeDefinitionId>,
    },
    /// Additional painted restriction (world/surface paint) × placement.
    PaintedRestriction {
        #[serde(default)]
        paint_mask: Option<crate::mask::MaskId>,
    },
}

impl WorldRuleScope {
    pub fn label(&self) -> &'static str {
        match self {
            Self::EntireWorld => "Entire World",
            Self::SelectedBiomes(_) => "Selected Biomes",
            Self::Excluding { .. } => "Excluding Selection",
            Self::PaintedRestriction { .. } => "Painted Restriction",
        }
    }

    pub fn includes_biome(&self, id: BiomeDefinitionId) -> bool {
        match self {
            Self::EntireWorld | Self::PaintedRestriction { .. } => true,
            Self::SelectedBiomes(ids) => ids.contains(&id),
            Self::Excluding { biomes, .. } => !biomes.contains(&id),
        }
    }
}

/// Artist-facing execution phase for World Rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WorldRulePhase {
    BeforePhysics,
    PhysicsInput,
    AfterPhysics,
    #[default]
    Materials,
    Scatter,
    Objects,
}

impl WorldRulePhase {
    pub fn all() -> &'static [WorldRulePhase] {
        &[
            Self::BeforePhysics,
            Self::PhysicsInput,
            Self::AfterPhysics,
            Self::Materials,
            Self::Scatter,
            Self::Objects,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::BeforePhysics => "Before Physics",
            Self::PhysicsInput => "Physics Input",
            Self::AfterPhysics => "After Physics",
            Self::Materials => "Materials",
            Self::Scatter => "Scatter",
            Self::Objects => "Objects",
        }
    }

    /// Map to eval-stage ordinal for selective invalidation.
    pub fn eval_stage(self) -> EvalStage {
        match self {
            Self::BeforePhysics => EvalStage::PreBiomeFields,
            Self::PhysicsInput => EvalStage::SharedHydro,
            Self::AfterPhysics => EvalStage::SharedHydro,
            Self::Materials => EvalStage::Materials,
            Self::Scatter => EvalStage::Vegetation,
            Self::Objects => EvalStage::FineDetail,
        }
    }
}

/// Kind of effect applied inside the shared rule placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldRuleEffectKind {
    TerrainEffect,
    Material,
    SimulationInput,
    Scatter,
    ObjectExclusion,
    BiomeInfluence,
    GeneratedOutputMap,
}

impl WorldRuleEffectKind {
    pub fn all() -> &'static [WorldRuleEffectKind] {
        &[
            Self::TerrainEffect,
            Self::Material,
            Self::SimulationInput,
            Self::Scatter,
            Self::ObjectExclusion,
            Self::BiomeInfluence,
            Self::GeneratedOutputMap,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::TerrainEffect => "Terrain Effect",
            Self::Material => "Material",
            Self::SimulationInput => "Simulation Input",
            Self::Scatter => "Scatter",
            Self::ObjectExclusion => "Object Exclusion",
            Self::BiomeInfluence => "Biome Influence",
            Self::GeneratedOutputMap => "Generated Output Map",
        }
    }

    /// Infer execution phase from effect type.
    pub fn inferred_phase(self) -> WorldRulePhase {
        match self {
            Self::TerrainEffect => WorldRulePhase::BeforePhysics,
            Self::SimulationInput => WorldRulePhase::PhysicsInput,
            Self::Material | Self::BiomeInfluence | Self::GeneratedOutputMap => {
                WorldRulePhase::Materials
            }
            Self::Scatter => WorldRulePhase::Scatter,
            Self::ObjectExclusion => WorldRulePhase::Objects,
        }
    }
}

/// One effect under a World Rule (shares the rule’s placement).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldRuleEffect {
    pub id: Uuid,
    pub name: String,
    pub enabled: bool,
    pub kind: WorldRuleEffectKind,
    /// Optional strength / opacity \[0,1\].
    #[serde(default = "default_strength")]
    pub strength: f32,
    /// Free-form params (material id, filter type, …) — specialised per kind.
    #[serde(default)]
    pub params: serde_json::Value,
}

fn default_strength() -> f32 {
    1.0
}

impl WorldRuleEffect {
    pub fn new(name: impl Into<String>, kind: WorldRuleEffectKind) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            enabled: true,
            kind,
            strength: 1.0,
            params: serde_json::Value::Null,
        }
    }
}

/// First-class World Rule project entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldRule {
    pub id: WorldRuleId,
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub scope: WorldRuleScope,
    /// Shared placement for all effects (same IR as Develop Apply Where).
    #[serde(default)]
    pub placement: PlacementDefinition,
    /// Explicit phase; when `None`, inferred from enabled effects.
    #[serde(default)]
    pub phase_override: Option<WorldRulePhase>,
    #[serde(default)]
    pub effects: Vec<WorldRuleEffect>,
    /// Higher priority wins on overlap (stable sort: priority desc, then insertion).
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub blend: BlendMode,
    #[serde(default)]
    pub build_status: BuildStatus,
    #[serde(default)]
    pub cache_policy: CachePolicy,
    /// Last coverage estimate \[0,1\] (session hint; may be stale).
    #[serde(default)]
    pub coverage_estimate: f32,
}

fn default_true() -> bool {
    true
}

impl WorldRule {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: WorldRuleId::new(),
            name: name.into(),
            enabled: true,
            scope: WorldRuleScope::EntireWorld,
            placement: PlacementDefinition::default(),
            phase_override: None,
            effects: Vec::new(),
            priority: 0,
            blend: BlendMode::Normal,
            build_status: BuildStatus::Idle,
            cache_policy: CachePolicy::Live,
            coverage_estimate: 0.0,
        }
    }

    /// Resolved execution phase — override or inferred from effects.
    pub fn resolved_phase(&self) -> WorldRulePhase {
        if let Some(p) = self.phase_override {
            return p;
        }
        let enabled: Vec<_> = self.effects.iter().filter(|e| e.enabled).collect();
        if enabled.is_empty() {
            return WorldRulePhase::Materials;
        }
        // If all agree, use that; else Materials as safe default (Advanced may set override).
        let first = enabled[0].kind.inferred_phase();
        if enabled.iter().all(|e| e.kind.inferred_phase() == first) {
            first
        } else {
            // Ambiguity → prefer earliest (most conservative rebuild) among enabled.
            enabled
                .iter()
                .map(|e| e.kind.inferred_phase())
                .min_by_key(|p| p.eval_stage().order())
                .unwrap_or(WorldRulePhase::Materials)
        }
    }

    /// True when phase was auto-inferred (hide under Advanced unless ambiguous).
    pub fn phase_needs_user_input(&self) -> bool {
        if self.phase_override.is_some() {
            return false;
        }
        let enabled: Vec<_> = self.effects.iter().filter(|e| e.enabled).collect();
        if enabled.len() < 2 {
            return false;
        }
        let first = enabled[0].kind.inferred_phase();
        !enabled.iter().all(|e| e.kind.inferred_phase() == first)
    }

    /// Compile placement into mask infrastructure.
    pub fn compile_placement(&self) -> Distribution {
        match self.placement.source {
            PlacementSource::Custom => self
                .placement
                .custom_stack
                .clone()
                .unwrap_or_else(|| self.placement.compile()),
            PlacementSource::Rules => self.placement.compile(),
        }
    }

    /// Refresh coverage estimate from bake context. Zero is valid (not an error).
    pub fn refresh_coverage(
        &mut self,
        metrics: HeightfieldMetrics,
        ctx: &DistBakeContext<'_>,
    ) -> f32 {
        let cov = coverage_estimate(&self.placement, metrics, ctx);
        self.coverage_estimate = cov;
        if cov < 1e-6 {
            self.build_status = BuildStatus::Ready; // exists, just empty
        } else {
            self.build_status = BuildStatus::Ready;
        }
        cov
    }

    /// Non-blocking diagnostics for contradictory / empty conditions.
    pub fn diagnostics(&self) -> Vec<SoftDiagnostic> {
        diagnose_world_rule(self)
    }

    /// Eval stage to dirty when this rule changes.
    pub fn invalidation_stage(&self) -> EvalStage {
        self.resolved_phase().eval_stage()
    }

    pub fn enabled_effects(&self) -> impl Iterator<Item = &WorldRuleEffect> {
        self.effects.iter().filter(|e| e.enabled)
    }

    pub fn summary_lines(&self) -> Vec<String> {
        let mut lines = vec![self.name.clone()];
        lines.push(format!("Scope: {}", self.scope.label()));
        lines.push(format!("Phase: {}", self.resolved_phase().label()));
        let where_phrases = collect_where_phrases(&self.placement.root);
        if where_phrases.is_empty() {
            lines.push("Where: Entire scope".into());
        } else {
            lines.push(format!("Where: {}", where_phrases.join(" · ")));
        }
        lines.push(format!("Coverage: ~{:.0}%", self.coverage_estimate * 100.0));
        for e in self.enabled_effects() {
            lines.push(format!("Apply: {} ({})", e.name, e.kind.label()));
        }
        lines
    }
}

fn collect_where_phrases(group: &crate::mask::RuleGroup) -> Vec<String> {
    use crate::mask::RuleNode;
    let mut out = Vec::new();
    for child in &group.children {
        match child {
            RuleNode::Condition(c) => out.push(c.where_phrase()),
            RuleNode::Group(g) => out.extend(collect_where_phrases(g)),
            RuleNode::CoverageRef { .. } => out.push("Painted coverage".into()),
        }
    }
    if out.len() > 4 {
        out.truncate(4);
        out.push("…".into());
    }
    out
}

/// Project-level World Rule library (ordered by priority for display; sort helpers below).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorldRuleLibrary {
    pub rules: Vec<WorldRule>,
    #[serde(default)]
    pub selected: Option<WorldRuleId>,
}

impl WorldRuleLibrary {
    pub fn get(&self, id: WorldRuleId) -> Option<&WorldRule> {
        self.rules.iter().find(|r| r.id == id)
    }

    pub fn get_mut(&mut self, id: WorldRuleId) -> Option<&mut WorldRule> {
        self.rules.iter_mut().find(|r| r.id == id)
    }

    pub fn push(&mut self, rule: WorldRule) -> WorldRuleId {
        let id = rule.id;
        self.rules.push(rule);
        self.selected = Some(id);
        id
    }

    pub fn remove(&mut self, id: WorldRuleId) -> Option<WorldRule> {
        let idx = self.rules.iter().position(|r| r.id == id)?;
        let removed = self.rules.remove(idx);
        if self.selected == Some(id) {
            self.selected = self.rules.first().map(|r| r.id);
        }
        Some(removed)
    }

    /// Stable priority order: higher priority first, then insertion order.
    pub fn sorted_by_priority(&self) -> Vec<&WorldRule> {
        let mut indexed: Vec<(usize, &WorldRule)> = self.rules.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.priority.cmp(&a.1.priority).then_with(|| a.0.cmp(&b.0)));
        indexed.into_iter().map(|(_, r)| r).collect()
    }

    /// Reorder by moving `id` before `before` (or to end when `before` is None).
    pub fn reorder(&mut self, id: WorldRuleId, before: Option<WorldRuleId>) -> bool {
        let Some(from) = self.rules.iter().position(|r| r.id == id) else {
            return false;
        };
        let rule = self.rules.remove(from);
        let to = before
            .and_then(|b| self.rules.iter().position(|r| r.id == b))
            .unwrap_or(self.rules.len());
        self.rules.insert(to, rule);
        // Re-assign priorities from list order (top = highest).
        let n = self.rules.len() as i32;
        for (i, r) in self.rules.iter_mut().enumerate() {
            r.priority = n - i as i32;
        }
        true
    }

    pub fn set_priority(&mut self, id: WorldRuleId, priority: i32) -> bool {
        if let Some(r) = self.get_mut(id) {
            r.priority = priority;
            true
        } else {
            false
        }
    }

    /// Rules that run in a given phase (enabled only), priority-sorted.
    pub fn rules_for_phase(&self, phase: WorldRulePhase) -> Vec<&WorldRule> {
        self.sorted_by_priority()
            .into_iter()
            .filter(|r| r.enabled && r.resolved_phase() == phase)
            .collect()
    }

    /// Earliest invalidation stage among dirty rule ids.
    pub fn invalidation_stage_for(&self, ids: &[WorldRuleId]) -> Option<EvalStage> {
        ids.iter()
            .filter_map(|id| self.get(*id))
            .map(|r| r.invalidation_stage())
            .min_by_key(|s| s.order())
    }
}

/// Undoable World Rule commands (document-level).
// Variants intentionally carry full before/after `WorldRule` snapshots so undo/redo is
// self-contained; short-lived on the undo stack, not hot-path allocations.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorldRuleCommand {
    Add {
        rule: WorldRule,
        index: usize,
    },
    Remove {
        rule: WorldRule,
        index: usize,
    },
    Replace {
        id: WorldRuleId,
        before: WorldRule,
        after: WorldRule,
    },
    Reorder {
        id: WorldRuleId,
        from: usize,
        to: usize,
        /// Priorities before reorder (for undo).
        previous_priorities: Vec<(WorldRuleId, i32)>,
    },
    SetEnabled {
        id: WorldRuleId,
        enabled: bool,
        previous: bool,
    },
}

impl WorldRuleCommand {
    pub fn apply(&self, lib: &mut WorldRuleLibrary) {
        match self {
            Self::Add { rule, index } => {
                let idx = (*index).min(lib.rules.len());
                lib.rules.insert(idx, rule.clone());
                lib.selected = Some(rule.id);
            }
            Self::Remove { rule, .. } => {
                let _ = lib.remove(rule.id);
            }
            Self::Replace { id, after, .. } => {
                if let Some(r) = lib.get_mut(*id) {
                    *r = after.clone();
                }
            }
            Self::Reorder { id, to, .. } => {
                if let Some(from) = lib.rules.iter().position(|r| r.id == *id) {
                    let rule = lib.rules.remove(from);
                    let to = (*to).min(lib.rules.len());
                    lib.rules.insert(to, rule);
                    let n = lib.rules.len() as i32;
                    for (i, r) in lib.rules.iter_mut().enumerate() {
                        r.priority = n - i as i32;
                    }
                }
            }
            Self::SetEnabled { id, enabled, .. } => {
                if let Some(r) = lib.get_mut(*id) {
                    r.enabled = *enabled;
                }
            }
        }
    }

    pub fn invert(&self, lib: &mut WorldRuleLibrary) {
        match self {
            Self::Add { rule, .. } => {
                let _ = lib.remove(rule.id);
            }
            Self::Remove { rule, index } => {
                let idx = (*index).min(lib.rules.len());
                lib.rules.insert(idx, rule.clone());
                lib.selected = Some(rule.id);
            }
            Self::Replace { id, before, .. } => {
                if let Some(r) = lib.get_mut(*id) {
                    *r = before.clone();
                }
            }
            Self::Reorder {
                previous_priorities,
                ..
            } => {
                // Restore priorities then sort into that order.
                for (id, p) in previous_priorities {
                    if let Some(r) = lib.get_mut(*id) {
                        r.priority = *p;
                    }
                }
                lib.rules.sort_by(|a, b| {
                    b.priority
                        .cmp(&a.priority)
                        .then_with(|| a.name.cmp(&b.name))
                });
            }
            Self::SetEnabled { id, previous, .. } => {
                if let Some(r) = lib.get_mut(*id) {
                    r.enabled = *previous;
                }
            }
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Add { rule, .. } => format!("Added World Rule {}", rule.name),
            Self::Remove { rule, .. } => format!("Removed World Rule {}", rule.name),
            Self::Replace { after, .. } => format!("Edited World Rule {}", after.name),
            Self::Reorder { .. } => "Reordered World Rules".into(),
            Self::SetEnabled { enabled, .. } => {
                if *enabled {
                    "Enabled World Rule".into()
                } else {
                    "Disabled World Rule".into()
                }
            }
        }
    }
}

/// Non-blocking diagnostics (contradictory / empty conditions).
pub fn diagnose_world_rule(rule: &WorldRule) -> Vec<SoftDiagnostic> {
    let mut out = Vec::new();
    if rule.effects.is_empty() {
        out.push(SoftDiagnostic::new(
            "world_rule_no_effects",
            format!(
                "‘{}’ has no effects yet — add Material, Terrain, or Scatter.",
                rule.name
            ),
        ));
    }
    if !rule.effects.is_empty() && rule.effects.iter().all(|e| !e.enabled) {
        out.push(SoftDiagnostic::new(
            "world_rule_effects_disabled",
            format!("All effects on ‘{}’ are disabled.", rule.name),
        ));
    }
    // Contradictory height Between where min > max.
    collect_condition_diags(&rule.placement.root, &rule.name, &mut out);
    if rule.placement.root.children.is_empty()
        && rule.placement.coverage.is_empty()
        && !matches!(rule.placement.source, PlacementSource::Custom)
    {
        out.push(SoftDiagnostic::new(
            "world_rule_empty_placement",
            format!(
                "‘{}’ has no placement conditions — it will cover the entire scope.",
                rule.name
            ),
        ));
    }
    if rule.coverage_estimate < 1e-6
        && (!rule.placement.root.children.is_empty() || !rule.placement.coverage.is_empty())
    {
        out.push(SoftDiagnostic::new(
            "world_rule_zero_coverage",
            format!(
                "‘{}’ currently matches ~0% of terrain — allowed, not an error.",
                rule.name
            ),
        ));
    }
    if rule.phase_needs_user_input() {
        out.push(SoftDiagnostic::new(
            "world_rule_phase_ambiguous",
            format!(
                "‘{}’ effects imply different phases — set phase under Advanced.",
                rule.name
            ),
        ));
    }
    out
}

fn collect_condition_diags(group: &RuleGroup, name: &str, out: &mut Vec<SoftDiagnostic>) {
    for child in &group.children {
        match child {
            RuleNode::Condition(c) => {
                if matches!(c.op, CompareOp::Between) && c.a > c.b {
                    out.push(SoftDiagnostic::new(
                        "world_rule_contradictory_range",
                        format!(
                            "‘{name}’ has a contradictory range ({:.0} … {:.0}).",
                            c.a, c.b
                        ),
                    ));
                }
            }
            RuleNode::Group(g) => collect_condition_diags(g, name, out),
            RuleNode::CoverageRef { .. } => {}
        }
    }
}

/// Build a PlacementDefinition from common artist conditions (AND).
pub fn placement_from_conditions(conditions: Vec<Condition>) -> PlacementDefinition {
    let mut def = PlacementDefinition::default();
    def.source = PlacementSource::Rules;
    def.root = RuleGroup {
        mode: RuleGroupMode::All,
        invert: false,
        children: conditions.into_iter().map(RuleNode::Condition).collect(),
    };
    def.recompute_hash();
    def
}

fn cond(channel: ConditionChannel, op: CompareOp, a: f32, b: f32, falloff: f32) -> Condition {
    Condition {
        channel,
        op,
        a,
        b,
        falloff,
        seed: 0,
        frequency: 0.02,
    }
}

/// Built-in World Rule presets.
pub fn builtin_world_rule_presets() -> Vec<WorldRule> {
    vec![
        snowline_preset(),
        beach_preset(),
        cliff_preset(),
        coastal_wetness_preset(),
        riverbank_preset(),
        underwater_sand_preset(),
        high_altitude_rock_preset(),
    ]
}

pub fn snowline_preset() -> WorldRule {
    let mut r = WorldRule::new("Snowline");
    r.placement = placement_from_conditions(vec![cond(
        ConditionChannel::Height,
        CompareOp::Above,
        1200.0,
        0.0,
        40.0,
    )]);
    r.effects.push(WorldRuleEffect::new(
        "Snow Material",
        WorldRuleEffectKind::Material,
    ));
    r.priority = 80;
    r
}

pub fn beach_preset() -> WorldRule {
    let mut r = WorldRule::new("Beaches");
    r.placement = placement_from_conditions(vec![
        cond(ConditionChannel::Height, CompareOp::Between, -2.0, 8.0, 1.0),
        cond(
            ConditionChannel::WaterDistance,
            CompareOp::Below,
            120.0,
            0.0,
            20.0,
        ),
        cond(ConditionChannel::Slope, CompareOp::Below, 18.0, 0.0, 2.0),
    ]);
    r.effects.push(WorldRuleEffect::new(
        "Sand Material",
        WorldRuleEffectKind::Material,
    ));
    r.effects.push(WorldRuleEffect::new(
        "Slight Terrain Smoothing",
        WorldRuleEffectKind::TerrainEffect,
    ));
    r.effects.push(WorldRuleEffect::new(
        "Beach Rock Scatter",
        WorldRuleEffectKind::Scatter,
    ));
    r.priority = 70;
    r
}

pub fn cliff_preset() -> WorldRule {
    let mut r = WorldRule::new("Cliff");
    r.placement = placement_from_conditions(vec![cond(
        ConditionChannel::Slope,
        CompareOp::Above,
        50.0,
        0.0,
        4.0,
    )]);
    r.effects.push(WorldRuleEffect::new(
        "Cliff Rock Material",
        WorldRuleEffectKind::Material,
    ));
    r.effects.push(WorldRuleEffect::new(
        "Remove Steep Grass",
        WorldRuleEffectKind::ObjectExclusion,
    ));
    r.priority = 75;
    r
}

pub fn coastal_wetness_preset() -> WorldRule {
    let mut r = WorldRule::new("Coastal Wetness");
    r.placement = placement_from_conditions(vec![cond(
        ConditionChannel::WaterDistance,
        CompareOp::Below,
        80.0,
        0.0,
        15.0,
    )]);
    r.effects.push(WorldRuleEffect::new(
        "Wet Materials",
        WorldRuleEffectKind::Material,
    ));
    r.priority = 60;
    r
}

pub fn riverbank_preset() -> WorldRule {
    let mut r = WorldRule::new("Riverbank");
    r.placement = placement_from_conditions(vec![cond(
        ConditionChannel::Flow,
        CompareOp::Above,
        0.12,
        0.0,
        0.04,
    )]);
    r.effects.push(WorldRuleEffect::new(
        "Riverbank Sediment",
        WorldRuleEffectKind::Material,
    ));
    r.priority = 65;
    r
}

pub fn underwater_sand_preset() -> WorldRule {
    let mut r = WorldRule::new("Underwater Sand");
    r.placement = placement_from_conditions(vec![cond(
        ConditionChannel::Height,
        CompareOp::Below,
        0.0,
        0.0,
        2.0,
    )]);
    r.effects.push(WorldRuleEffect::new(
        "Sand Material",
        WorldRuleEffectKind::Material,
    ));
    r.priority = 55;
    r
}

pub fn high_altitude_rock_preset() -> WorldRule {
    let mut r = WorldRule::new("High Altitude Rock");
    r.placement = placement_from_conditions(vec![
        cond(
            ConditionChannel::Height,
            CompareOp::Above,
            1800.0,
            0.0,
            50.0,
        ),
        cond(ConditionChannel::Slope, CompareOp::Above, 25.0, 0.0, 3.0),
    ]);
    r.effects.push(WorldRuleEffect::new(
        "Alpine Rock",
        WorldRuleEffectKind::Material,
    ));
    r.priority = 78;
    r
}

/// Instantiate a preset by name (case-insensitive).
pub fn world_rule_preset_by_name(name: &str) -> Option<WorldRule> {
    let key = name.trim().to_ascii_lowercase();
    builtin_world_rule_presets()
        .into_iter()
        .find(|r| r.name.to_ascii_lowercase() == key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beach_preset_has_multiple_effects() {
        let b = beach_preset();
        assert_eq!(b.effects.len(), 3);
        assert!(!b.compile_placement().nodes.is_empty() || !b.placement.root.children.is_empty());
    }

    #[test]
    fn phase_inferred_from_material() {
        let r = snowline_preset();
        assert_eq!(r.resolved_phase(), WorldRulePhase::Materials);
        assert!(!r.phase_needs_user_input());
    }

    #[test]
    fn beach_phase_ambiguous_without_override() {
        let b = beach_preset();
        // Terrain + Material + Scatter → ambiguous
        assert!(b.phase_needs_user_input());
    }
}
