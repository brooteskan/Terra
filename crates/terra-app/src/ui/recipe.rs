//! Terrain recipe presets and world instances (reusable biome templates).

use serde::{Deserialize, Serialize};
use terra_core::layer::{
    BiomeSection, CoastalParams, GroupEvalMode, GroupInputMode, GroupKind, HydraulicErosionParams,
    Layer, LayerGroup, LayerId, LayerKind, MaterialsParams, MountainParams, PathNode, PathParams,
    RiverCarveParams, RiverNetworkParams, RiverNode, StackNode, StreamPowerParams,
    ThermalErosionParams, UpliftParams,
};

/// Stable identity for a reusable recipe asset (not a world instance).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RecipeId(pub LayerId);

impl RecipeId {
    pub fn new() -> Self {
        Self(LayerId::new())
    }
}

impl Default for RecipeId {
    fn default() -> Self {
        Self::new()
    }
}

/// A reusable ordered collection of layers (terrain recipe / biome template).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupRecipe {
    pub id: RecipeId,
    pub name: String,
    pub description: String,
    pub layers: Vec<(String, LayerKind)>,
    #[serde(default)]
    pub eval_mode: GroupEvalMode,
    #[serde(default)]
    pub input_mode: GroupInputMode,
    /// When true (default), instantiate as a WC biome container with sections.
    #[serde(default = "default_true")]
    pub as_biome: bool,
}

fn default_true() -> bool {
    true
}

pub(crate) const VALLEY_WITH_CREEKS_NAME: &str = "Valley with Creeks";
pub(crate) const VALLEY_WITH_CREEKS_DESCRIPTION: &str =
    "Uplift valley, guide paths, wetness-biased river carve, and ridge-spring creeks.";

/// Canonical product content shared by the flat preset and grouped recipe.
pub(crate) fn valley_with_creeks_layers() -> Vec<(String, LayerKind)> {
    vec![
        (
            "Uplift".into(),
            LayerKind::Uplift(UpliftParams {
                amplitude: 360.0,
                corridor_width: 0.4,
                ..UpliftParams::default()
            }),
        ),
        (
            "Stream Power".into(),
            LayerKind::StreamPowerErosion(StreamPowerParams {
                iterations: 28,
                k: 0.08,
                dendritic_seed: 0.55,
                level_step_strength: 1.1,
                ..StreamPowerParams::default()
            }),
        ),
        (
            "Valley Fill".into(),
            LayerKind::HydraulicErosion(HydraulicErosionParams::depositional()),
        ),
        (
            "Guide Paths".into(),
            LayerKind::Path(PathParams {
                nodes: vec![
                    PathNode {
                        u: 0.22,
                        v: 0.12,
                        height: 0.0,
                        width: 0.7,
                    },
                    PathNode {
                        u: 0.42,
                        v: 0.38,
                        height: 0.0,
                        width: 1.0,
                    },
                    PathNode {
                        u: 0.50,
                        v: 0.52,
                        height: 0.0,
                        width: 1.2,
                    },
                ],
                width: 28.0,
                falloff: 40.0,
                height_offset: -14.0,
                carve: true,
                ..PathParams::default()
            }),
        ),
        (
            "River Carve".into(),
            LayerKind::RiverCarve(RiverCarveParams {
                accumulation_threshold: 28.0,
                depth: 36.0,
                width: 7.0,
                guide: terra_core::mask::MaskSource::Wetness,
                guide_boost: 3.5,
                ..RiverCarveParams::default()
            }),
        ),
        (
            "Creek Network".into(),
            LayerKind::RiverNetwork(RiverNetworkParams {
                springs: vec![
                    RiverNode {
                        u: 0.22,
                        v: 0.18,
                        flow: 1.2,
                        width: 1.0,
                    },
                    RiverNode {
                        u: 0.78,
                        v: 0.20,
                        flow: 1.1,
                        width: 0.95,
                    },
                    RiverNode {
                        u: 0.35,
                        v: 0.12,
                        flow: 1.0,
                        width: 0.85,
                    },
                    RiverNode {
                        u: 0.62,
                        v: 0.14,
                        flow: 1.0,
                        width: 0.9,
                    },
                ],
                auto_generate: true,
                max_length: 512,
                carve_depth: 28.0,
                valley_width: 100.0,
                seed: 7,
            }),
        ),
        (
            "Coastal Edge".into(),
            LayerKind::Coastal(CoastalParams {
                sea_level: 30.0,
                ..CoastalParams::default()
            }),
        ),
        (
            "Materials".into(),
            LayerKind::Materials(MaterialsParams::default()),
        ),
    ]
}

impl GroupRecipe {
    pub fn to_group(&self) -> LayerGroup {
        if self.as_biome {
            let mut g = LayerGroup::biome(self.name.clone());
            g.input_mode = self.input_mode.clone();
            for (name, kind) in &self.layers {
                g.push_into_section(Layer::new(name.clone(), kind.clone()));
            }
            g
        } else {
            let mut g = LayerGroup::new(self.name.clone());
            g.eval_mode = self.eval_mode;
            g.input_mode = self.input_mode.clone();
            for (name, kind) in &self.layers {
                g.children
                    .push(StackNode::Layer(Layer::new(name.clone(), kind.clone())));
            }
            g
        }
    }
}

/// World-specific instance of a recipe with local overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeInstance {
    pub id: LayerId,
    pub name: String,
    pub recipe_id: RecipeId,
    pub recipe_name: String,
    #[serde(default)]
    pub seed_override: Option<u64>,
    #[serde(default)]
    pub height_scale: Option<f32>,
    #[serde(default)]
    pub erosion_scale: Option<f32>,
}

/// Built-in terrain recipes (code-defined; expanded into biomes on apply).
pub fn builtin_recipes() -> Vec<GroupRecipe> {
    vec![
        GroupRecipe {
            id: RecipeId::new(),
            name: "Young Alpine Mountains".into(),
            description: "Mountain range + hydraulic + thermal erosion as a biome container."
                .into(),
            layers: vec![
                (
                    "Mountain Range".into(),
                    LayerKind::Mountains(MountainParams::default()),
                ),
                (
                    "Hydraulic Erosion".into(),
                    LayerKind::HydraulicErosion(HydraulicErosionParams::default()),
                ),
                (
                    "Thermal Erosion".into(),
                    LayerKind::ThermalErosion(ThermalErosionParams::default()),
                ),
            ],
            eval_mode: GroupEvalMode::IsolatedComposite,
            input_mode: GroupInputMode::CopyInput,
            as_biome: true,
        },
        GroupRecipe {
            id: RecipeId::new(),
            name: "Desert Plateau".into(),
            description: "Plateau + canyon + thermal weathering biome.".into(),
            layers: vec![
                ("Plateau".into(), LayerKind::Plateau(Default::default())),
                ("Canyons".into(), LayerKind::Canyons(Default::default())),
                (
                    "Thermal Erosion".into(),
                    LayerKind::ThermalErosion(ThermalErosionParams::default()),
                ),
            ],
            eval_mode: GroupEvalMode::IsolatedComposite,
            input_mode: GroupInputMode::EmptyHeight,
            as_biome: true,
        },
        GroupRecipe {
            id: RecipeId::new(),
            name: VALLEY_WITH_CREEKS_NAME.into(),
            description: VALLEY_WITH_CREEKS_DESCRIPTION.into(),
            layers: valley_with_creeks_layers(),
            eval_mode: GroupEvalMode::IsolatedComposite,
            input_mode: GroupInputMode::CopyInput,
            as_biome: true,
        },
    ]
}

/// Apply a recipe as a biome (or isolated group) onto a stack (returns the new group).
pub fn instantiate_recipe(recipe: &GroupRecipe, instance_name: &str) -> LayerGroup {
    let mut g = recipe.to_group();
    g.name = instance_name.to_string();
    if g.is_biome() {
        g.group_kind = GroupKind::Biome;
        // Generators in biome templates land in Filters; keep LocalSims for pure sims.
        // push_into_section already routed Mountains→Filters and Hydraulic→LocalSims.
        let _ = BiomeSection::Filters;
    }
    g
}
