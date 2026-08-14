//! Reusable biome definitions (WHAT) separate from placement (WHERE).

use crate::layer::{
    EffectFilterKind, EffectFilterParams, Layer, LayerId, LayerKind, MaterialRule, MaterialsParams,
};
use crate::mask::{DistNode, DistNodeKind, MaskCombine, MaskSource};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable identity for a reusable biome definition asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BiomeDefinitionId(pub Uuid);

impl BiomeDefinitionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for BiomeDefinitionId {
    fn default() -> Self {
        Self::new()
    }
}

/// How manual paint combines with procedural placement rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PlacementCombineMode {
    #[default]
    PaintOnly,
    RulesOnly,
    PaintMulRules,
    PaintAddRules,
    PaintOverridesRules,
    RulesOutsidePaint,
}

impl PlacementCombineMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::PaintOnly => "Paint Only",
            Self::RulesOnly => "Rules Only",
            Self::PaintMulRules => "Paint x Rules",
            Self::PaintAddRules => "Paint + Rules",
            Self::PaintOverridesRules => "Paint Overrides Rules",
            Self::RulesOutsidePaint => "Rules Outside Painted Area",
        }
    }

    /// Plain-language ownership control for artists.
    pub fn artist_label(self) -> &'static str {
        match self {
            Self::PaintOverridesRules | Self::RulesOutsidePaint => "Paint owns · rules fill gaps",
            Self::PaintMulRules => "Guided by rules",
            Self::PaintAddRules => "Paint + rules",
            Self::PaintOnly => "Paint only",
            Self::RulesOnly => "Rules only",
        }
    }

    /// Toggle between ownership paint (default) and guided multiply.
    pub fn cycle_artist(self) -> Self {
        match self {
            Self::PaintOverridesRules | Self::RulesOutsidePaint | Self::PaintOnly => {
                Self::PaintMulRules
            }
            _ => Self::PaintOverridesRules,
        }
    }

    pub fn combine(self, manual: f32, procedural: f32) -> f32 {
        let m = manual.clamp(0.0, 1.0);
        let p = procedural.clamp(0.0, 1.0);
        match self {
            Self::PaintOnly => m,
            Self::RulesOnly => p,
            Self::PaintMulRules => m * p,
            Self::PaintAddRules => (m + p).clamp(0.0, 1.0),
            Self::PaintOverridesRules | Self::RulesOutsidePaint => {
                if m > 1e-4 {
                    m
                } else {
                    p
                }
            }
        }
    }

    /// Map to DistNode / mask stack combine when rules are DistNodes and paint is a mask ref.
    /// Paint is applied *after* DistNodes in [`bake_distribution_with_context`].
    pub fn mask_combine(self) -> MaskCombine {
        match self {
            Self::PaintMulRules => MaskCombine::Multiply,
            Self::PaintAddRules => MaskCombine::Add,
            Self::PaintOnly => MaskCombine::Replace,
            Self::PaintOverridesRules | Self::RulesOutsidePaint => MaskCombine::PaintOverride,
            // Paint entry should be omitted for RulesOnly; Multiply is a harmless fallback.
            Self::RulesOnly => MaskCombine::Multiply,
        }
    }

    /// Whether manual paint should be attached to the biome distribution.
    pub fn uses_manual_paint(self) -> bool {
        !matches!(self, Self::RulesOnly)
    }
}

/// Output-domain overlap policy for biome contributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BiomeOverlapPolicy {
    #[default]
    HeightDelta,
    PriorityOverride,
    NormalizedBlend,
}

/// Default placement rules packaged with a definition.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BiomePlacementRules {
    #[serde(default)]
    pub combine: PlacementCombineMode,
    /// Legacy single DistNode tree (migrated into `definition` when present).
    #[serde(default)]
    pub rules: Option<DistNode>,
    /// Artist PlacementDefinition (compiles into Mask Layer DistNode stack).
    #[serde(default)]
    pub definition: Option<crate::mask::PlacementDefinition>,
    #[serde(default)]
    pub blur_m: f32,
    #[serde(default)]
    pub breakup: f32,
    #[serde(default)]
    pub priority: i32,
}

impl BiomePlacementRules {
    /// Resolve the distribution written onto a biome group.
    pub fn compiled_distribution(&self) -> crate::mask::Distribution {
        if let Some(def) = &self.definition {
            return def.active_distribution();
        }
        if let Some(rules) = &self.rules {
            return crate::mask::Distribution::from_nodes(vec![rules.clone()]);
        }
        crate::mask::Distribution::new()
    }

    /// Ensure `definition` exists, wrapping legacy `rules` when needed.
    pub fn ensure_definition(&mut self) -> &mut crate::mask::PlacementDefinition {
        if self.definition.is_none() {
            self.definition = Some(match &self.rules {
                Some(node) => crate::mask::PlacementDefinition::from_legacy_dist_node(node.clone()),
                None => crate::mask::PlacementDefinition::default(),
            });
        }
        self.definition.as_mut().expect("just inserted")
    }

    /// Mark Mask Stack as Custom after a manual DistNode edit.
    pub fn mark_mask_stack_custom(&mut self, stack: crate::mask::Distribution) {
        self.ensure_definition().mark_custom(stack);
    }

    /// Reset Custom stack by recompiling from PlacementDefinition rules.
    pub fn reset_placement_to_rules(&mut self) -> crate::mask::Distribution {
        self.ensure_definition().reset_to_rules()
    }
}

/// Reusable biome definition asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomeDefinition {
    pub id: BiomeDefinitionId,
    pub name: String,
    pub description: String,
    pub color: [f32; 3],
    #[serde(default)]
    pub group_id: Option<LayerId>,
    #[serde(default)]
    pub terrain_layers: Vec<(String, LayerKind)>,
    #[serde(default)]
    pub material_layers: Vec<(String, LayerKind)>,
    #[serde(default)]
    pub object_layers: Vec<(String, LayerKind)>,
    #[serde(default)]
    pub local_sim_layers: Vec<(String, LayerKind)>,
    #[serde(default)]
    pub placement: BiomePlacementRules,
    #[serde(default)]
    pub terrain_overlap: BiomeOverlapPolicy,
    #[serde(default)]
    pub version: u32,
}

impl BiomeDefinition {
    pub fn new(name: impl Into<String>, color: [f32; 3]) -> Self {
        Self {
            id: BiomeDefinitionId::new(),
            name: name.into(),
            description: String::new(),
            color,
            group_id: None,
            terrain_layers: Vec::new(),
            material_layers: Vec::new(),
            object_layers: Vec::new(),
            local_sim_layers: Vec::new(),
            placement: BiomePlacementRules::default(),
            terrain_overlap: BiomeOverlapPolicy::HeightDelta,
            version: 1,
        }
    }

    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }

    pub fn duplicate_asset(&self) -> Self {
        let mut d = self.clone();
        d.id = BiomeDefinitionId::new();
        d.name = format!("{} Copy", self.name);
        d.group_id = None;
        d.version = 1;
        d
    }

    /// Instantiate into a WC biome group with sections + DistNode placement rules.
    pub fn to_group(&self) -> crate::layer::LayerGroup {
        use crate::layer::{GroupInputMode, LayerGroup};
        let mut g = LayerGroup::biome(self.name.clone());
        g.preview_color = self.color;
        g.input_mode = GroupInputMode::CopyInput;
        g.filter_blending = 1.0;
        for (name, kind) in &self.terrain_layers {
            g.push_into_section(Layer::new(name.clone(), kind.clone()));
        }
        for (name, kind) in &self.material_layers {
            g.push_into_section(Layer::new(name.clone(), kind.clone()));
        }
        for (name, kind) in &self.object_layers {
            g.push_into_section(Layer::new(name.clone(), kind.clone()));
        }
        for (name, kind) in &self.local_sim_layers {
            g.push_into_section(Layer::new(name.clone(), kind.clone()));
        }
        let dist = self.placement.compiled_distribution();
        if !dist.nodes.is_empty() || !dist.entries.is_empty() {
            g.masks = dist;
        }
        g
    }
}

/// Document-level biome library.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BiomeLibrary {
    pub definitions: Vec<BiomeDefinition>,
    #[serde(default)]
    pub selected: Option<BiomeDefinitionId>,
}

impl BiomeLibrary {
    pub fn get(&self, id: BiomeDefinitionId) -> Option<&BiomeDefinition> {
        self.definitions.iter().find(|d| d.id == id)
    }

    pub fn get_mut(&mut self, id: BiomeDefinitionId) -> Option<&mut BiomeDefinition> {
        self.definitions.iter_mut().find(|d| d.id == id)
    }

    pub fn by_group(&self, group_id: LayerId) -> Option<&BiomeDefinition> {
        self.definitions
            .iter()
            .find(|d| d.group_id == Some(group_id))
    }

    pub fn by_group_mut(&mut self, group_id: LayerId) -> Option<&mut BiomeDefinition> {
        self.definitions
            .iter_mut()
            .find(|d| d.group_id == Some(group_id))
    }

    pub fn push(&mut self, def: BiomeDefinition) -> BiomeDefinitionId {
        let id = def.id;
        self.definitions.push(def);
        id
    }

    pub fn default_world_palette() -> Self {
        let mut lib = Self::default();
        for (name, color) in [
            ("Default", [0.45, 0.55, 0.35]),
            ("Water", [0.15, 0.35, 0.65]),
            ("Beach", [0.86, 0.78, 0.55]),
            ("Grassland", [0.45, 0.72, 0.28]),
            ("Rock", [0.55, 0.52, 0.48]),
        ] {
            lib.push(BiomeDefinition::new(name, color));
        }
        lib.selected = lib.definitions.first().map(|d| d.id);
        lib
    }

    /// Usable Tropical Island biomes with real terrain response + placement rules.
    pub fn tropical_island_palette() -> Self {
        let mut lib = Self::default();

        // —— Beach ——
        let mut beach = BiomeDefinition::new("Beach", [0.90, 0.82, 0.55])
            .with_description("Sandy shore above sea level");
        beach.placement.combine = PlacementCombineMode::PaintOverridesRules;
        beach.placement.priority = 10;
        beach.placement.blur_m = 12.0;
        beach.placement.rules = Some(beach_rules());
        beach.terrain_layers = vec![(
            "Shore Soften".into(),
            LayerKind::EffectFilter(EffectFilterParams {
                kind: EffectFilterKind::Shore,
                strength: 0.85,
                ..EffectFilterParams::shore()
            }),
        )];
        beach.material_layers = vec![("Beach Sand".into(), LayerKind::Materials(sand_materials()))];

        // —— Tropical Forest ——
        let mut forest = BiomeDefinition::new("Tropical Forest", [0.12, 0.48, 0.18])
            .with_description("Warm wet canopy interior");
        forest.placement.combine = PlacementCombineMode::PaintOverridesRules;
        forest.placement.blur_m = 20.0;
        forest.placement.breakup = 0.15;
        forest.placement.priority = 5;
        forest.placement.rules = Some(forest_rules());
        forest.terrain_layers = vec![
            (
                "Chemical Weathering".into(),
                LayerKind::EffectFilter(EffectFilterParams {
                    kind: EffectFilterKind::MudSettle,
                    strength: 0.55,
                    ..EffectFilterParams::mud_settle()
                }),
            ),
            (
                "Soft Flows".into(),
                LayerKind::EffectFilter(EffectFilterParams {
                    kind: EffectFilterKind::SoftFlows,
                    strength: 0.4,
                    ..EffectFilterParams::soft_flows()
                }),
            ),
        ];
        forest.material_layers = vec![(
            "Jungle Floor".into(),
            LayerKind::Materials(forest_materials()),
        )];
        forest.object_layers = vec![(
            "Canopy".into(),
            LayerKind::Vegetation(crate::layer::VegetationParams {
                density: 0.55,
                min_distance: 2.5,
                max_slope_deg: 32.0,
                root_cohesion: 0.35,
                ..crate::layer::VegetationParams::default()
            }),
        )];

        // —— Rocky Peaks ——
        let mut rock = BiomeDefinition::new("Rocky Peaks", [0.55, 0.52, 0.50])
            .with_description("Exposed volcanic rock");
        rock.placement.combine = PlacementCombineMode::PaintOverridesRules;
        rock.placement.priority = 20;
        rock.placement.rules = Some(rock_rules());
        rock.terrain_layers = vec![
            (
                "Rocky Sharp".into(),
                LayerKind::EffectFilter(EffectFilterParams {
                    kind: EffectFilterKind::RockySharp,
                    strength: 0.7,
                    ..EffectFilterParams::rocky_sharp()
                }),
            ),
            (
                "Cliff Reinforce".into(),
                LayerKind::EffectFilter(EffectFilterParams {
                    kind: EffectFilterKind::CliffReinforce,
                    strength: 0.45,
                    ..EffectFilterParams::cliff_reinforce()
                }),
            ),
        ];
        rock.material_layers = vec![("Basalt".into(), LayerKind::Materials(rock_materials()))];

        // —— Wetland ——
        let mut wetland = BiomeDefinition::new("Wetland", [0.20, 0.55, 0.48])
            .with_description("Low coastal wetlands");
        wetland.placement.combine = PlacementCombineMode::PaintOverridesRules;
        wetland.placement.priority = 8;
        wetland.placement.rules = Some(wetland_rules());
        wetland.terrain_layers = vec![(
            "Sediment Settle".into(),
            LayerKind::EffectFilter(EffectFilterParams {
                kind: EffectFilterKind::SedimentFillSoft,
                strength: 0.65,
                ..EffectFilterParams::sediment_fill_soft()
            }),
        )];
        wetland.material_layers =
            vec![("Wet Soil".into(), LayerKind::Materials(wetland_materials()))];

        // —— Shallow Water / Reef ——
        let mut reef = BiomeDefinition::new("Shallow Water / Reef", [0.15, 0.55, 0.70])
            .with_description("Shallow shelf and reef");
        reef.placement.combine = PlacementCombineMode::PaintOverridesRules;
        reef.placement.priority = 15;
        reef.placement.rules = Some(reef_rules());
        reef.terrain_layers = vec![(
            "Shelf Flatten".into(),
            LayerKind::EffectFilter(EffectFilterParams {
                kind: EffectFilterKind::Smooth,
                strength: 0.5,
                ..EffectFilterParams::smooth()
            }),
        )];
        reef.material_layers = vec![("Reef".into(), LayerKind::Materials(reef_materials()))];

        for d in [beach, forest, rock, wetland, reef] {
            lib.push(d);
        }
        lib.selected = lib.definitions.first().map(|d| d.id);
        lib
    }

    /// Alpine range biomes (peaks, meadow, conifer, scree).
    pub fn alpine_palette() -> Self {
        let mut lib = Self::default();
        let mut peaks = BiomeDefinition::new("Alpine Peaks", [0.72, 0.74, 0.78])
            .with_description("High rock and snow");
        peaks.placement.combine = PlacementCombineMode::PaintOverridesRules;
        peaks.placement.priority = 20;
        peaks.placement.rules = Some(rock_rules());
        peaks.terrain_layers = vec![(
            "Rocky Sharp".into(),
            LayerKind::EffectFilter(EffectFilterParams {
                kind: EffectFilterKind::RockySharp,
                strength: 0.65,
                ..EffectFilterParams::rocky_sharp()
            }),
        )];
        peaks.material_layers =
            vec![("Alpine Rock".into(), LayerKind::Materials(rock_materials()))];

        let mut meadow = BiomeDefinition::new("Alpine Meadow", [0.42, 0.62, 0.32])
            .with_description("High grassy slopes");
        meadow.placement.combine = PlacementCombineMode::PaintOverridesRules;
        meadow.placement.priority = 8;
        meadow.placement.rules = Some(forest_rules());
        meadow.material_layers = vec![("Meadow".into(), LayerKind::Materials(forest_materials()))];
        meadow.object_layers = vec![(
            "Alpine Grass".into(),
            LayerKind::Vegetation(crate::layer::VegetationParams {
                density: 0.45,
                min_distance: 2.0,
                max_slope_deg: 30.0,
                ..crate::layer::VegetationParams::default()
            }),
        )];

        let mut forest = BiomeDefinition::new("Conifer Belt", [0.18, 0.38, 0.22])
            .with_description("Montane forest");
        forest.placement.combine = PlacementCombineMode::PaintOverridesRules;
        forest.placement.priority = 5;
        forest.placement.rules = Some(forest_rules());
        forest.terrain_layers = vec![(
            "Soft Flows".into(),
            LayerKind::EffectFilter(EffectFilterParams {
                kind: EffectFilterKind::SoftFlows,
                strength: 0.35,
                ..EffectFilterParams::soft_flows()
            }),
        )];
        forest.material_layers = vec![(
            "Forest Floor".into(),
            LayerKind::Materials(forest_materials()),
        )];
        forest.object_layers = vec![(
            "Conifers".into(),
            LayerKind::Vegetation(crate::layer::VegetationParams {
                density: 0.5,
                min_distance: 3.5,
                max_slope_deg: 28.0,
                root_cohesion: 0.4,
                ..crate::layer::VegetationParams::default()
            }),
        )];

        let mut scree =
            BiomeDefinition::new("Scree", [0.58, 0.52, 0.45]).with_description("Talus and debris");
        scree.placement.combine = PlacementCombineMode::PaintOverridesRules;
        scree.placement.priority = 12;
        scree.material_layers = vec![("Talus".into(), LayerKind::Materials(rock_materials()))];

        for d in [peaks, meadow, forest, scree] {
            lib.push(d);
        }
        lib.selected = lib.definitions.first().map(|d| d.id);
        lib
    }

    /// Arid desert biomes (dunes, mesa, canyon, scrub).
    pub fn desert_palette() -> Self {
        let mut lib = Self::default();
        let mut dunes = BiomeDefinition::new("Dunes", [0.86, 0.72, 0.42])
            .with_description("Wind-blown sand seas");
        dunes.placement.combine = PlacementCombineMode::PaintOverridesRules;
        dunes.placement.priority = 6;
        dunes.terrain_layers = vec![(
            "Dune Soften".into(),
            LayerKind::EffectFilter(EffectFilterParams {
                kind: EffectFilterKind::SoftFlows,
                strength: 0.5,
                ..EffectFilterParams::soft_flows()
            }),
        )];
        dunes.material_layers = vec![("Sand".into(), LayerKind::Materials(sand_materials()))];

        let mut mesa = BiomeDefinition::new("Mesa Cap", [0.72, 0.58, 0.42])
            .with_description("Hard caprock flats");
        mesa.placement.combine = PlacementCombineMode::PaintOverridesRules;
        mesa.placement.priority = 15;
        mesa.material_layers = vec![("Caprock".into(), LayerKind::Materials(rock_materials()))];

        let mut canyon = BiomeDefinition::new("Canyon Floor", [0.55, 0.38, 0.28])
            .with_description("Arroyo and wash");
        canyon.placement.combine = PlacementCombineMode::PaintOverridesRules;
        canyon.placement.priority = 10;
        canyon.material_layers = vec![("Alluvium".into(), LayerKind::Materials(sand_materials()))];

        let mut scrub = BiomeDefinition::new("Desert Scrub", [0.55, 0.52, 0.28])
            .with_description("Sparse arid vegetation");
        scrub.placement.combine = PlacementCombineMode::PaintOverridesRules;
        scrub.placement.priority = 4;
        scrub.object_layers = vec![(
            "Scrub".into(),
            LayerKind::Vegetation(crate::layer::VegetationParams {
                density: 0.15,
                min_distance: 6.0,
                max_slope_deg: 25.0,
                ..crate::layer::VegetationParams::default()
            }),
        )];
        scrub.material_layers = vec![("Soil".into(), LayerKind::Materials(sand_materials()))];

        for d in [dunes, mesa, canyon, scrub] {
            lib.push(d);
        }
        lib.selected = lib.definitions.first().map(|d| d.id);
        lib
    }

    /// River valley biomes (floodplain, terrace, hills, channel).
    pub fn river_valley_palette() -> Self {
        let mut lib = Self::default();
        let mut flood = BiomeDefinition::new("Floodplain", [0.40, 0.58, 0.32])
            .with_description("Wet valley floor");
        flood.placement.combine = PlacementCombineMode::PaintOverridesRules;
        flood.placement.priority = 8;
        flood.placement.rules = Some(wetland_rules());
        flood.terrain_layers = vec![(
            "Sediment Settle".into(),
            LayerKind::EffectFilter(EffectFilterParams {
                kind: EffectFilterKind::SedimentFillSoft,
                strength: 0.6,
                ..EffectFilterParams::sediment_fill_soft()
            }),
        )];
        flood.material_layers = vec![("Silt".into(), LayerKind::Materials(wetland_materials()))];
        flood.object_layers = vec![(
            "Riparian".into(),
            LayerKind::Vegetation(crate::layer::VegetationParams {
                density: 0.5,
                min_distance: 2.2,
                max_slope_deg: 18.0,
                ..crate::layer::VegetationParams::default()
            }),
        )];

        let mut terrace = BiomeDefinition::new("River Terrace", [0.48, 0.55, 0.35])
            .with_description("Raised valley benches");
        terrace.placement.combine = PlacementCombineMode::PaintOverridesRules;
        terrace.placement.priority = 6;
        terrace.material_layers = vec![("Loam".into(), LayerKind::Materials(forest_materials()))];

        let mut hills = BiomeDefinition::new("Valley Hills", [0.35, 0.48, 0.28])
            .with_description("Rolling valley sides");
        hills.placement.combine = PlacementCombineMode::PaintOverridesRules;
        hills.placement.priority = 4;
        hills.object_layers = vec![(
            "Hardwood".into(),
            LayerKind::Vegetation(crate::layer::VegetationParams {
                density: 0.4,
                min_distance: 3.0,
                max_slope_deg: 30.0,
                root_cohesion: 0.3,
                ..crate::layer::VegetationParams::default()
            }),
        )];
        hills.material_layers = vec![("Soil".into(), LayerKind::Materials(forest_materials()))];

        let mut channel = BiomeDefinition::new("Active Channel", [0.35, 0.45, 0.55])
            .with_description("River corridor");
        channel.placement.combine = PlacementCombineMode::PaintOverridesRules;
        channel.placement.priority = 18;
        channel.material_layers = vec![("Gravel".into(), LayerKind::Materials(sand_materials()))];

        for d in [flood, terrace, hills, channel] {
            lib.push(d);
        }
        lib.selected = lib.definitions.first().map(|d| d.id);
        lib
    }
}

fn dist_with_blur(base: DistNode, blur_radius: u32) -> DistNode {
    let mut node = base;
    if blur_radius > 0 {
        node.children.push(DistNode::new(DistNodeKind::EffectBlur {
            radius: blur_radius,
        }));
    }
    node
}

fn beach_rules() -> DistNode {
    // Above sea level, low height band, gentle slopes.
    let mut h = DistNode::height(0.5, 18.0);
    h.combine = MaskCombine::Multiply;
    let mut s = DistNode::slope(0.0, 28.0);
    s.combine = MaskCombine::Multiply;
    h.children.push(s);
    dist_with_blur(h, 2)
}

fn forest_rules() -> DistNode {
    let mut h = DistNode::height(8.0, 420.0);
    h.combine = MaskCombine::Multiply;
    let mut s = DistNode::slope(0.0, 42.0);
    s.combine = MaskCombine::Multiply;
    h.children.push(s);
    let mut noise = DistNode::new(DistNodeKind::Noise {
        seed: 77,
        frequency: 0.004,
    });
    noise.opacity = 0.35;
    noise.combine = MaskCombine::Multiply;
    h.children.push(noise);
    dist_with_blur(h, 3)
}

fn rock_rules() -> DistNode {
    let mut h = DistNode::height(180.0, 2000.0);
    h.combine = MaskCombine::Multiply;
    let mut s = DistNode::slope(28.0, 90.0);
    s.combine = MaskCombine::Max;
    h.children.push(s);
    dist_with_blur(h, 1)
}

fn wetland_rules() -> DistNode {
    let mut h = DistNode::height(0.0, 12.0);
    h.combine = MaskCombine::Multiply;
    let mut s = DistNode::slope(0.0, 12.0);
    s.combine = MaskCombine::Multiply;
    h.children.push(s);
    dist_with_blur(h, 3)
}

fn reef_rules() -> DistNode {
    // Near and below sea level.
    let mut sea = DistNode::new(DistNodeKind::SeaLevel {
        level: 0.0,
        width: 14.0,
    });
    sea.combine = MaskCombine::Multiply;
    let mut h = DistNode::height(-40.0, 4.0);
    h.combine = MaskCombine::Multiply;
    sea.children.push(h);
    dist_with_blur(sea, 2)
}

fn material_rule(name: &str, id: u32, tint: [f32; 3], hardness: f32) -> MaterialRule {
    MaterialRule {
        name: name.into(),
        id,
        min_slope_deg: 0.0,
        max_slope_deg: 90.0,
        // Finite bounds — serde_json cannot roundtrip ±Infinity as f32.
        min_height: crate::layer::OPEN_HEIGHT_MIN,
        max_height: crate::layer::OPEN_HEIGHT_MAX,
        mask: MaskSource::None,
        hardness,
        tint,
        roughness: 0.8,
        metalness: 0.02,
        albedo_path: None,
    }
}

fn sand_materials() -> MaterialsParams {
    MaterialsParams {
        default_hardness: 0.25,
        rules: vec![material_rule("Sand", 10, [0.90, 0.82, 0.55], 0.22)],
        ..MaterialsParams::default()
    }
}

fn forest_materials() -> MaterialsParams {
    MaterialsParams {
        default_hardness: 0.35,
        rules: vec![
            material_rule("Humus", 20, [0.22, 0.38, 0.14], 0.3),
            material_rule("Moss", 21, [0.18, 0.42, 0.22], 0.28),
        ],
        ..MaterialsParams::default()
    }
}

fn rock_materials() -> MaterialsParams {
    MaterialsParams {
        default_hardness: 0.85,
        rules: vec![material_rule("Basalt", 30, [0.42, 0.40, 0.38], 0.9)],
        ..MaterialsParams::default()
    }
}

fn wetland_materials() -> MaterialsParams {
    MaterialsParams {
        default_hardness: 0.2,
        rules: vec![material_rule("Mud", 40, [0.28, 0.36, 0.28], 0.18)],
        ..MaterialsParams::default()
    }
}

fn reef_materials() -> MaterialsParams {
    MaterialsParams {
        default_hardness: 0.55,
        rules: vec![material_rule("Coral", 50, [0.35, 0.62, 0.68], 0.5)],
        ..MaterialsParams::default()
    }
}

/// Normalize a set of raw weights in-place; returns the sum before normalize.
pub fn normalize_weights(weights: &mut [f32]) -> f32 {
    let sum: f32 = weights.iter().copied().sum();
    if sum > 1e-6 {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
    sum
}

/// Combine height-delta biome contributions.
pub fn blend_height_deltas(shared_input: f32, deltas: &[(f32, f32)]) -> f32 {
    let mut h = shared_input;
    for &(w, d) in deltas {
        h += w * d;
    }
    h
}

pub fn layers_from_kinds(items: &[(String, LayerKind)]) -> Vec<Layer> {
    items
        .iter()
        .map(|(name, kind)| Layer::new(name.clone(), kind.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_modes() {
        assert!((PlacementCombineMode::PaintMulRules.combine(0.8, 0.5) - 0.4).abs() < 1e-5);
        assert!((PlacementCombineMode::PaintOverridesRules.combine(0.8, 0.5) - 0.8).abs() < 1e-5);
        assert!((PlacementCombineMode::PaintAddRules.combine(0.6, 0.6) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn tropical_palette_has_content() {
        let lib = BiomeLibrary::tropical_island_palette();
        assert_eq!(lib.definitions.len(), 5);
        for d in &lib.definitions {
            assert!(!d.terrain_layers.is_empty(), "{} missing terrain", d.name);
            assert!(d.placement.rules.is_some(), "{} missing rules", d.name);
            let g = d.to_group();
            assert!(g.is_biome());
            assert!(!g.masks.nodes.is_empty() || !d.terrain_layers.is_empty());
        }
    }

    #[test]
    fn normalize_and_height_delta() {
        let mut w = [0.5_f32, 0.5, 0.0];
        assert!((normalize_weights(&mut w) - 1.0).abs() < 1e-5);
        let h = blend_height_deltas(100.0, &[(0.5, 20.0), (0.5, -10.0)]);
        assert!((h - 105.0).abs() < 1e-4);
    }
}
