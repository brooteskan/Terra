//! Artist-facing hierarchy presentation over `doc.stack`.
//!
//! Does **not** fork project state: one ownership tree, workspace-adapted emphasis.
//! Technical DistNode / mask-stack chrome lives under Advanced Placement.

use terra_core::biome_definition::BiomeLibrary;
use terra_core::document::TerrainDocument;
use terra_core::domain::classify_layer_kind;
use terra_core::domain::{classify_in_context, DomainRole};
use terra_core::layer::{
    BiomeSection, BuildStatus, GroupKind, Layer, LayerGroup, LayerId, LayerKind, StackCategory,
    StackNode,
};
use terra_gui::Icon;

use crate::ui::workspace::{workspace_definition, WorkspaceId};

/// Stable presentation sentinels (not project entities).
pub const WORLD_SECTION: u128 = 0x0000_0000_5445_5252_0000_0000_0000_0010;
/// WC Terrain root row (presentation only).
pub const TERRAIN_ROOT: u128 = 0x0000_0000_5445_5252_0000_0000_0000_0012;
/// Single-stack scope key for concept folder presentation IDs.
pub const TERRAIN_SCOPE_KEY: u64 = 0;
const CONCEPT_BASE: u128 = 0x0000_0000_5445_5252_C0C3_0000_0000_0000;
const ADVANCED_BASE: u128 = 0x0000_0000_5445_5252_AD70_0000_0000_0000;
const MASK_STACK_BASE: u128 = 0x0000_0000_5445_5252_4D53_4B31_0000_0000;

/// High-level artist concept folders (presentation only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ArtistConcept {
    Shape = 1,
    Biomes = 2,
    LocalSimulations = 3,
    Objects = 5,
    WorldRules = 6,
    WorldSimulations = 7,
    GlobalMaterials = 8,
    GlobalScatter = 9,
    AdvancedPlacement = 10,
    MaskStack = 11,
    /// Mask Layers folder (WC).
    Masks = 12,
    /// Biome Layers — painted weight / placement layers (WC).
    BiomeLayers = 13,
}

impl ArtistConcept {
    pub fn label(self) -> &'static str {
        match self {
            Self::Shape => "Shape Layers",
            Self::Biomes => "Biomes",
            Self::BiomeLayers => "Biome Layers",
            Self::LocalSimulations => "Local Simulations",
            Self::Objects => "Objects",
            Self::WorldRules => "World Rules",
            Self::WorldSimulations => "Simulation Layers",
            Self::GlobalMaterials => "Materials",
            Self::GlobalScatter => "Scatter",
            Self::AdvancedPlacement => "Distribution",
            Self::MaskStack => "Mask Stack",
            Self::Masks => "Mask Layers",
        }
    }

    /// World Creator terrain stack order (under Terrain root).

    /// WC: Biomes → Biome Layers → Shape Layers → Mask Layers → Simulation Layers.
    pub fn terrain_order() -> &'static [ArtistConcept] {
        &[
            Self::Biomes,
            Self::BiomeLayers,
            Self::Shape,
            Self::Masks,
            Self::WorldSimulations,
        ]
    }

    pub fn world_order() -> &'static [ArtistConcept] {
        Self::terrain_order()
    }

    /// Biome child section order (WC: Filters → Materials → Objects; Local Sims last).
    pub fn biome_section_order() -> &'static [terra_core::layer::BiomeSection] {
        &[
            terra_core::layer::BiomeSection::Filters,
            terra_core::layer::BiomeSection::Materials,
            terra_core::layer::BiomeSection::Objects,
            terra_core::layer::BiomeSection::LocalSims,
        ]
    }

    pub fn icon(self) -> Icon {
        match self {
            Self::Shape => Icon::Mountain,
            Self::Biomes => Icon::Sparkles,
            Self::BiomeLayers => Icon::Layers,
            Self::LocalSimulations | Self::WorldSimulations => Icon::Activity,
            Self::GlobalMaterials => Icon::Grid3x3,
            Self::Objects | Self::GlobalScatter => Icon::Package,
            Self::WorldRules | Self::AdvancedPlacement | Self::MaskStack => Icon::CircleDot,
            Self::Masks => Icon::Paintbrush,
        }
    }

    pub fn preferred_role(self) -> Option<DomainRole> {
        match self {
            Self::Shape => Some(DomainRole::ShapeLayer),
            Self::Biomes | Self::BiomeLayers => Some(DomainRole::TerrainFilter),
            Self::LocalSimulations | Self::WorldSimulations => Some(DomainRole::SimulationLayer),
            Self::GlobalMaterials => Some(DomainRole::MaterialLayer),
            Self::Objects | Self::GlobalScatter => Some(DomainRole::ScatterLayer),
            Self::WorldRules | Self::AdvancedPlacement | Self::MaskStack | Self::Masks => {
                Some(DomainRole::MaskLayer)
            }
        }
    }

    /// Stack category used when routing new layers into WC folders.
    pub fn stack_category(self) -> Option<StackCategory> {
        match self {
            Self::Shape => Some(StackCategory::Shape),
            Self::Biomes => Some(StackCategory::Surface),
            Self::Masks | Self::MaskStack | Self::WorldRules => Some(StackCategory::Mask),
            Self::WorldSimulations | Self::LocalSimulations => Some(StackCategory::Simulation),
            Self::GlobalMaterials => Some(StackCategory::Surface),
            Self::Objects | Self::GlobalScatter => Some(StackCategory::Surface),
            Self::AdvancedPlacement | Self::BiomeLayers => None,
        }
    }

    /// Title for the concept-scoped Quick Add modal.
    pub fn quick_add_title(self) -> &'static str {
        match self {
            Self::Shape => "Add Shape Layer",
            Self::Biomes => "Add Biome",
            Self::BiomeLayers => "Add Biome Layer",
            Self::Masks | Self::MaskStack => "Add Mask",
            Self::WorldSimulations | Self::LocalSimulations => "Add Simulation",
            Self::GlobalMaterials => "Add Material",
            Self::Objects | Self::GlobalScatter => "Add Object",
            Self::WorldRules => "Add World Rule",
            Self::AdvancedPlacement => "Add Distribution",
        }
    }
}

/// Artist-facing biome section titles.
pub fn biome_section_artist_label(section: BiomeSection) -> &'static str {
    match section {
        BiomeSection::Filters => "Filters",
        BiomeSection::Materials => "Materials",
        BiomeSection::Objects => "Objects",
        BiomeSection::LocalSims => "Local Simulations",
    }
}

pub fn concept_row_id(scope_key: u64, concept: ArtistConcept) -> LayerId {
    LayerId::from_u128(CONCEPT_BASE | ((scope_key as u128) << 8) | u128::from(concept as u8))
}

pub fn advanced_placement_id(biome_id: LayerId) -> LayerId {
    LayerId::from_u128(ADVANCED_BASE ^ (biome_id.0.as_u128() & 0xFFFF_FFFF_FFFF_FFFF))
}

pub fn mask_stack_row_id(biome_id: LayerId) -> LayerId {
    LayerId::from_u128(MASK_STACK_BASE ^ (biome_id.0.as_u128() & 0xFFFF_FFFF_FFFF_FFFF))
}

/// Biome summary line: coverage · enabled · priority · health.
pub fn biome_summary_meta(
    g: &LayerGroup,
    library: &BiomeLibrary,
    active: Option<LayerId>,
) -> String {
    let def = library.by_group(g.id);
    let priority = def.map(|d| d.placement.priority).unwrap_or(0);
    let coverage = if g.masks.is_empty() {
        "Coverage —"
    } else {
        "Coverage rules"
    };
    let enabled = if g.enabled { "On" } else { "Off" };
    let mut parts = vec![
        coverage.to_string(),
        enabled.to_string(),
        format!("P{priority}"),
    ];
    if active == Some(g.id) {
        parts.push("Active".into());
    }
    if def.map(|d| d.placement.rules.is_none()).unwrap_or(true) && g.masks.is_empty() {
        parts.push("Warning".into());
    }
    parts.join(" · ")
}

/// Simulation build badge for hierarchy rows.
pub fn sim_status_label(status: BuildStatus, enabled: bool) -> &'static str {
    if !enabled {
        return "Frozen";
    }
    match status {
        BuildStatus::Ready => "Ready",
        BuildStatus::Pending | BuildStatus::Idle | BuildStatus::Outdated => "Outdated",
        BuildStatus::Computing => "Running",
        BuildStatus::Error => "Failed",
    }
}

pub fn is_simulation_kind(kind: &LayerKind) -> bool {
    matches!(classify_layer_kind(kind), DomainRole::SimulationLayer)
}

pub fn layer_build_status(layer: &Layer) -> BuildStatus {
    if !layer.common.enabled {
        return BuildStatus::Idle;
    }
    if layer.common.cached {
        BuildStatus::Ready
    } else {
        BuildStatus::Pending
    }
}

pub fn layer_build_status_with_outdated(layer: &Layer, outdated: &[LayerId]) -> BuildStatus {
    if outdated.contains(&layer.id()) {
        return BuildStatus::Outdated;
    }
    layer_build_status(layer)
}

/// Compact meta line for a simulation scenario row.
pub fn simulation_scenario_meta(
    scenario: &terra_core::simulation_scenario::SimulationScenario,
) -> String {
    let passes = scenario.enabled_passes().count();
    format!(
        "{} · {} pass{}",
        if scenario.enabled { "On" } else { "Off" },
        passes,
        if passes == 1 { "" } else { "es" }
    )
}

/// Whether a presentation row should be de-emphasized for the active workspace.
pub fn emphasis_dims(
    workspace: WorkspaceId,
    role: DomainRole,
    concept: Option<ArtistConcept>,
) -> bool {
    let h = workspace_definition(workspace).hierarchy;
    if !h.dim_unrelated {
        return false;
    }
    if let Some(c) = concept {
        if let Some(prefer) = c.preferred_role() {
            return h.should_dim_role(prefer);
        }
    }
    h.should_dim_role(role)
}

/// World Rules summary.
pub fn world_rules_meta(enabled: bool, has_match: bool, phase: &str) -> String {
    format!(
        "{} · {} · {} · {}",
        if enabled { "On" } else { "Off" },
        if has_match {
            "Matches terrain"
        } else {
            "No match yet"
        },
        phase,
        "Scope World"
    )
}

/// Meta line from a first-class [`terra_core::world_rules::WorldRule`].
pub fn world_rule_entity_meta(rule: &terra_core::world_rules::WorldRule) -> String {
    let match_label = if rule.coverage_estimate > 0.05 {
        format!("~{:.0}% coverage", rule.coverage_estimate * 100.0)
    } else {
        "No match yet".into()
    };
    format!(
        "{} · {} · {} · {}",
        if rule.enabled { "On" } else { "Off" },
        match_label,
        rule.resolved_phase().label(),
        rule.scope.label()
    )
}

pub fn classify_stack_node(
    node: &StackNode,
    category: Option<StackCategory>,
    section: Option<BiomeSection>,
) -> DomainRole {
    match node {
        StackNode::Layer(l) => classify_in_context(&l.kind, category, section),
        StackNode::Group(g) => match g.group_kind {
            GroupKind::Biome => DomainRole::TerrainFilter,
            GroupKind::BiomeSection(s) => match s {
                BiomeSection::Filters => DomainRole::TerrainFilter,
                BiomeSection::Materials => DomainRole::MaterialLayer,
                BiomeSection::Objects => DomainRole::ObjectLayer,
                BiomeSection::LocalSims => DomainRole::SimulationLayer,
            },
            GroupKind::CategoryFolder => match g.category {
                Some(StackCategory::Shape) | Some(StackCategory::Foundation) => {
                    DomainRole::ShapeLayer
                }
                Some(StackCategory::Simulation) => DomainRole::SimulationLayer,
                Some(StackCategory::Mask) => DomainRole::MaskLayer,
                Some(StackCategory::Surface) => DomainRole::TerrainFilter,
                _ => DomainRole::CompatibilityLegacy,
            },
            GroupKind::Generic => DomainRole::CompatibilityLegacy,
        },
    }
}

/// Bucket a top-level stack node into a WC terrain concept folder.
pub fn concept_for_top_level(node: &StackNode, _global: bool) -> ArtistConcept {
    let raw = match node {
        StackNode::Group(g) if matches!(g.group_kind, GroupKind::Biome) => ArtistConcept::Biomes,
        StackNode::Group(g)
            if matches!(g.group_kind, GroupKind::CategoryFolder) || g.category.is_some() =>
        {
            match g.category {
                Some(StackCategory::Shape) | Some(StackCategory::Foundation) => {
                    ArtistConcept::Shape
                }
                Some(StackCategory::Simulation) => ArtistConcept::WorldSimulations,
                Some(StackCategory::Mask) => ArtistConcept::Masks,
                Some(StackCategory::Surface) => ArtistConcept::Biomes,
                _ => ArtistConcept::Shape,
            }
        }
        StackNode::Layer(l) if matches!(l.kind, LayerKind::Biomes(_)) => ArtistConcept::Biomes,
        other => {
            let role = classify_stack_node(other, None, None);
            match role {
                DomainRole::ShapeLayer => ArtistConcept::Shape,
                DomainRole::SimulationLayer => ArtistConcept::WorldSimulations,
                // Orphan materials / objects live under Biomes in WC (no top-level Materials folder).
                DomainRole::MaterialLayer
                | DomainRole::ScatterLayer
                | DomainRole::ObjectLayer
                | DomainRole::TerrainFilter => ArtistConcept::Biomes,
                DomainRole::MaskLayer => ArtistConcept::Masks,
                // Climate LUT / uplift leftovers — never dump climate into Shape.
                DomainRole::CompatibilityLegacy => match other {
                    StackNode::Layer(l) if matches!(l.kind, LayerKind::Biomes(_)) => {
                        ArtistConcept::Biomes
                    }
                    _ => ArtistConcept::Shape,
                },
            }
        }
    };
    normalize_terrain_concept(raw)
}

/// Concept folder that owns children of a WC category folder.
pub fn concept_for_category(category: StackCategory) -> ArtistConcept {
    normalize_terrain_concept(match category {
        StackCategory::Shape | StackCategory::Foundation => ArtistConcept::Shape,
        StackCategory::Surface => ArtistConcept::Biomes,
        StackCategory::Mask => ArtistConcept::Masks,
        StackCategory::Simulation => ArtistConcept::WorldSimulations,
    })
}

/// Map legacy concept ids onto the five WC terrain folders.
pub fn normalize_terrain_concept(concept: ArtistConcept) -> ArtistConcept {
    match concept {
        ArtistConcept::LocalSimulations => ArtistConcept::WorldSimulations,
        ArtistConcept::GlobalMaterials | ArtistConcept::Objects | ArtistConcept::GlobalScatter => {
            ArtistConcept::Biomes
        }
        ArtistConcept::WorldRules => ArtistConcept::Masks,
        other => other,
    }
}

/// Hierarchy identity fingerprint for tests (stable across workspace switches).
pub fn hierarchy_identity_ids(doc: &TerrainDocument) -> Vec<LayerId> {
    let mut ids = Vec::new();
    collect_stack_ids(&doc.stack.nodes, &mut ids);
    ids
}

fn collect_stack_ids(nodes: &[StackNode], ids: &mut Vec<LayerId>) {
    for node in nodes {
        match node {
            StackNode::Layer(l) => ids.push(l.id()),
            StackNode::Group(g) => {
                ids.push(g.id);
                collect_stack_ids(&g.children, ids);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use terra_core::document::TerrainDocument;

    #[test]
    fn identity_ids_stable_across_calls() {
        let doc = TerrainDocument::new_default();
        let a = hierarchy_identity_ids(&doc);
        let b = hierarchy_identity_ids(&doc);
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn mask_stack_id_differs_from_biome() {
        let biome = LayerId::new();
        assert_ne!(mask_stack_row_id(biome), biome);
        assert_ne!(advanced_placement_id(biome), mask_stack_row_id(biome));
    }

    #[test]
    fn artist_section_labels_are_non_technical() {
        assert_eq!(biome_section_artist_label(BiomeSection::Filters), "Filters");
        assert_eq!(
            biome_section_artist_label(BiomeSection::LocalSims),
            "Local Simulations"
        );
    }
}
