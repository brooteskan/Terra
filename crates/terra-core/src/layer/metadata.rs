//! Shared layer type / instance metadata (domain-facing; UI-agnostic).

use super::cache_policy::{CachePolicy, CacheState};
use super::operation::{FieldContract, OperationCategory};
use super::workflow::WorkflowStage;
use super::{Layer, LayerGroup, LayerId, LayerKind};
use crate::fields::FieldId;
use serde::{Deserialize, Serialize};

/// Visual accent bucket for lists / badges (not a hardcoded UI colour).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AccentCategory {
    #[default]
    Neutral,
    Foundation,
    Generator,
    Simulation,
    Mask,
    Biome,
    Material,
    Scatter,
    Object,
    Output,
}

impl AccentCategory {
    pub fn label(self) -> &'static str {
        match self {
            AccentCategory::Neutral => "Neutral",
            AccentCategory::Foundation => "Foundation",
            // Maps from WorkflowStage::Generation — artist language is Shape, not Generator.
            AccentCategory::Generator => "Shape",
            AccentCategory::Simulation => "Simulation",
            AccentCategory::Mask => "Mask",
            AccentCategory::Biome => "Biome",
            AccentCategory::Material => "Material",
            AccentCategory::Scatter => "Scatter",
            AccentCategory::Object => "Object",
            AccentCategory::Output => "Output",
        }
    }

    pub fn from_workflow(stage: WorkflowStage) -> Self {
        match stage {
            WorkflowStage::Foundation => AccentCategory::Foundation,
            WorkflowStage::Generation => AccentCategory::Generator,
            WorkflowStage::Simulation => AccentCategory::Simulation,
            WorkflowStage::Masks => AccentCategory::Mask,
            WorkflowStage::BiomePlacement => AccentCategory::Biome,
            WorkflowStage::Materials => AccentCategory::Material,
            WorkflowStage::Scatter => AccentCategory::Scatter,
            WorkflowStage::Objects => AccentCategory::Object,
            WorkflowStage::Output => AccentCategory::Output,
        }
    }
}

/// How a layer type relates to masks / distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum MaskCompatibility {
    /// Layer contribution is scaled by its distribution masks.
    #[default]
    SupportsMasks,
    /// Masks are ignored (rare; mostly organisational).
    IgnoresMasks,
    /// Layer is primarily a mask / weight producer.
    ProducesMask,
}

/// Static capabilities for a layer *type* (not an instance).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerCapabilities {
    pub supports_thumbnails: bool,
    pub supports_duplication: bool,
    pub can_reorder: bool,
    pub can_contain_children: bool,
    /// Appears in Add Layer / registry create paths.
    pub user_creatable: bool,
}

impl Default for LayerCapabilities {
    fn default() -> Self {
        Self {
            supports_thumbnails: true,
            supports_duplication: true,
            can_reorder: true,
            can_contain_children: false,
            user_creatable: true,
        }
    }
}

/// Declared input / output channel summary for UI and validation hints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChannelRequirements {
    pub required: Vec<FieldId>,
    pub optional: Vec<FieldId>,
    pub produced: Vec<FieldId>,
}

impl ChannelRequirements {
    pub fn from_contract(c: &FieldContract) -> Self {
        Self {
            required: c.required.clone(),
            optional: c.optional.clone(),
            produced: c.produced.clone(),
        }
    }
}

/// Stable description of a creatable / known layer *type* (not an instance).
///
/// Not project-serialized — lives in the in-process [`super::LayerTypeRegistry`].
#[derive(Debug, Clone)]
pub struct LayerTypeMeta {
    /// Stable machine id, e.g. `"mountain"`, `"hydraulic_erosion"`.
    pub type_id: &'static str,
    pub display_name: &'static str,
    pub workflow_stage: WorkflowStage,
    /// Icon key for UI mapping (terra-gui / asset thumbs); not a GPU handle.
    pub icon_key: &'static str,
    pub accent: AccentCategory,
    pub description: &'static str,
    pub operation_category: OperationCategory,
    pub channels: ChannelRequirements,
    pub mask_compatibility: MaskCompatibility,
    pub capabilities: LayerCapabilities,
    /// Inspector section ids the UI may open (presentation interprets these).
    pub inspector_sections: &'static [&'static str],
    /// Suggested follow-on type ids (empty if none).
    pub suggested_next: &'static [&'static str],
}

/// Runtime build / evaluation badge for an instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BuildStatus {
    #[default]
    Idle,
    Pending,
    Computing,
    Ready,
    /// Upstream Shape / geometry changed; not rebuilt yet (sims stay frozen).
    Outdated,
    Error,
}

/// Derived metadata for a live layer or group instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInstanceMeta {
    pub id: LayerId,
    pub display_name: String,
    /// Registry type id when this is a leaf layer; `None` for groups.
    pub type_id: Option<String>,
    pub workflow_stage: Option<WorkflowStage>,
    pub icon_key: Option<String>,
    pub accent: AccentCategory,
    pub description: Option<String>,
    pub enabled: bool,
    /// Opacity / influence in \[0, 1\].
    pub opacity: f32,
    pub build_status: BuildStatus,
    pub cache_status: CacheState,
    pub cache_policy: CachePolicy,
    pub channels: ChannelRequirements,
    pub mask_compatibility: MaskCompatibility,
    pub capabilities: LayerCapabilities,
    pub inspector_sections: Vec<String>,
    pub suggested_next: Vec<String>,
}

impl LayerKind {
    /// Core identity shared by direct kind consumers and the type registry.
    ///
    /// Keep this match exhaustive so a new kind cannot be added without also
    /// defining its stable id, default display name, and authoring stage.
    fn type_identity(&self) -> (&'static str, &'static str, WorkflowStage) {
        match self {
            LayerKind::SculptBase(_) => ("sculpt_base", "Base", WorkflowStage::Foundation),
            LayerKind::SculptStrokes(_) => (
                "sculpt_strokes",
                "Semantic Sculpt",
                WorkflowStage::Foundation,
            ),
            LayerKind::TerrainConstraints(_) => (
                "terrain_constraints",
                "Terrain Constraints",
                WorkflowStage::Foundation,
            ),
            LayerKind::GradientReconstruct(_) => (
                "gradient_reconstruct",
                "Gradient Reconstruct",
                WorkflowStage::Generation,
            ),
            LayerKind::LandscapeEvolution(_) => (
                "landscape_evolution",
                "Landscape Evolution",
                WorkflowStage::Simulation,
            ),
            LayerKind::HydrologyRepair(_) => (
                "hydrology_repair",
                "Hydrology Repair",
                WorkflowStage::Simulation,
            ),
            LayerKind::GeomorphicDetail(_) => (
                "geomorphic_detail",
                "Geomorphic Detail",
                WorkflowStage::Generation,
            ),
            LayerKind::EcosystemFeedback(_) => (
                "ecosystem_feedback",
                "Ecosystem Feedback",
                WorkflowStage::Simulation,
            ),
            LayerKind::Flat(_) => ("flat", "Flat", WorkflowStage::Generation),
            LayerKind::Ramp(_) => ("ramp", "Ramp", WorkflowStage::Generation),
            LayerKind::NoiseValue(_) => ("noise_value", "Value Noise", WorkflowStage::Generation),
            LayerKind::NoisePerlin(_) => {
                ("noise_perlin", "Perlin Noise", WorkflowStage::Generation)
            }
            LayerKind::NoiseOpenSimplex(_) => (
                "noise_open_simplex",
                "OpenSimplex Noise",
                WorkflowStage::Generation,
            ),
            LayerKind::NoiseWorley(_) => {
                ("noise_worley", "Worley Noise", WorkflowStage::Generation)
            }
            LayerKind::Fbm(_) => ("fbm", "FBM", WorkflowStage::Generation),
            LayerKind::Ridged(_) => ("ridged", "Ridged", WorkflowStage::Generation),
            LayerKind::DomainWarp(_) => ("domain_warp", "Domain Warp", WorkflowStage::Generation),
            LayerKind::Terrace(_) => ("terrace", "Terrace", WorkflowStage::Generation),
            LayerKind::Plateau(_) => ("plateau", "Plateau", WorkflowStage::Generation),
            LayerKind::Mesa(_) => ("mesa", "Mesa", WorkflowStage::Generation),
            LayerKind::Island(_) => ("island", "Island", WorkflowStage::Generation),
            LayerKind::Mountains(_) => ("mountain", "Mountains", WorkflowStage::Generation),
            LayerKind::Volcano(_) => ("volcano", "Volcano", WorkflowStage::Generation),
            LayerKind::Uplift(_) => ("uplift", "Uplift", WorkflowStage::Generation),
            LayerKind::Dunes(_) => ("dunes", "Dunes", WorkflowStage::Generation),
            LayerKind::Canyons(_) => ("canyon", "Canyons", WorkflowStage::Generation),
            LayerKind::VoronoiRegions(_) => {
                ("voronoi", "Voronoi Regions", WorkflowStage::Generation)
            }
            LayerKind::ImportHeightmap(_) => (
                "import_heightmap",
                "Import Heightmap",
                WorkflowStage::Foundation,
            ),
            LayerKind::ThermalErosion(_) => (
                "thermal_erosion",
                "Thermal Erosion",
                WorkflowStage::Simulation,
            ),
            LayerKind::HydraulicErosion(_) => (
                "hydraulic_erosion",
                "Hydraulic Erosion",
                WorkflowStage::Simulation,
            ),
            LayerKind::DebrisFlow(_) => ("debris_flow", "Debris Flow", WorkflowStage::Simulation),
            LayerKind::StreamPowerErosion(_) => {
                ("stream_power", "Stream Power", WorkflowStage::Simulation)
            }
            LayerKind::MultiScaleAmplify(_) => (
                "multi_scale_amplify",
                "Multi-Scale Amplify",
                WorkflowStage::Simulation,
            ),
            LayerKind::RiverCarve(_) => ("river_carve", "River Carve", WorkflowStage::Simulation),
            LayerKind::Blur(_) => ("blur", "Blur", WorkflowStage::Generation),
            LayerKind::Coastal(_) => ("coastal", "Coastal", WorkflowStage::Simulation),
            LayerKind::EffectFilter(_) => {
                ("effect_filter", "Effect Filter", WorkflowStage::Generation)
            }
            LayerKind::Materials(_) => ("materials", "Materials", WorkflowStage::Materials),
            LayerKind::Biomes(_) => (
                "climate_biomes",
                "Climate Classification",
                WorkflowStage::BiomePlacement,
            ),
            LayerKind::Vegetation(_) => ("vegetation", "Vegetation", WorkflowStage::Scatter),
            LayerKind::OverhangStamp(_) => (
                "overhang_stamp",
                "Overhang Stamp",
                WorkflowStage::Generation,
            ),
            LayerKind::LocalSdf(_) => ("local_sdf", "Local SDF", WorkflowStage::Generation),
            LayerKind::Path(_) => ("path", "Path", WorkflowStage::Generation),
            LayerKind::RiverNetwork(_) => {
                ("river_network", "River Network", WorkflowStage::Simulation)
            }
            LayerKind::SandSimulation(_) => {
                ("sand_simulation", "Sand Simulation", WorkflowStage::Objects)
            }
            LayerKind::FluidSimulation(_) => (
                "fluid_simulation",
                "Fluid Simulation",
                WorkflowStage::Objects,
            ),
            LayerKind::ProceduralShape(_) => (
                "procedural_shape",
                "Procedural Shape",
                WorkflowStage::Generation,
            ),
            LayerKind::Stamp2d(_) => ("stamp_2d", "2D Stamp", WorkflowStage::Foundation),
            LayerKind::Stamp3d(_) => ("stamp_3d", "3D Stamp", WorkflowStage::Foundation),
            LayerKind::PolygonHeight(_) => ("polygon_height", "Polygon", WorkflowStage::Generation),
        }
    }

    /// Stable type id for registry / metadata (serde variant name style).
    pub fn type_id(&self) -> &'static str {
        self.type_identity().0
    }

    /// Default human label for this kind (before instance rename).
    pub fn type_display_name(&self) -> &'static str {
        self.type_identity().1
    }

    /// Artist workflow stage for this kind (metadata).
    pub fn workflow_stage(&self) -> WorkflowStage {
        self.type_identity().2
    }
}

impl Layer {
    /// Build instance metadata from a live layer (optional runtime statuses).
    pub fn instance_meta(
        &self,
        type_meta: Option<&LayerTypeMeta>,
        build_status: BuildStatus,
        cache_status: CacheState,
    ) -> LayerInstanceMeta {
        let contract = FieldContract::from_kind(&self.kind);
        let caps = type_meta
            .map(|m| m.capabilities)
            .unwrap_or(LayerCapabilities {
                supports_thumbnails: true,
                supports_duplication: !self.kind.is_sculpt_base(),
                can_reorder: !self.kind.is_sculpt_base(),
                can_contain_children: false,
                user_creatable: !self.kind.is_sculpt_base(),
            });
        let stage = type_meta
            .map(|m| m.workflow_stage)
            .unwrap_or_else(|| self.kind.workflow_stage());
        LayerInstanceMeta {
            id: self.id(),
            display_name: self.common.name.clone(),
            type_id: Some(self.kind.type_id().into()),
            workflow_stage: Some(stage),
            icon_key: type_meta.map(|m| m.icon_key.to_string()),
            accent: type_meta
                .map(|m| m.accent)
                .unwrap_or_else(|| AccentCategory::from_workflow(stage)),
            description: type_meta.map(|m| m.description.to_string()),
            enabled: self.common.enabled,
            opacity: self.common.opacity,
            build_status,
            cache_status,
            cache_policy: self.common.resolved_cache_policy(),
            channels: ChannelRequirements::from_contract(&contract),
            mask_compatibility: type_meta
                .map(|m| m.mask_compatibility)
                .unwrap_or(MaskCompatibility::SupportsMasks),
            capabilities: caps,
            inspector_sections: type_meta
                .map(|m| {
                    m.inspector_sections
                        .iter()
                        .map(|s| (*s).to_string())
                        .collect()
                })
                .unwrap_or_default(),
            suggested_next: type_meta
                .map(|m| m.suggested_next.iter().map(|s| (*s).to_string()).collect())
                .unwrap_or_default(),
        }
    }
}

impl LayerGroup {
    /// Instance metadata for a group (organisational / biome container).
    pub fn instance_meta(&self, cache_status: CacheState) -> LayerInstanceMeta {
        let (stage, accent, icon, caps) = if self.is_biome() {
            (
                Some(WorkflowStage::BiomePlacement),
                AccentCategory::Biome,
                Some("layers"),
                LayerCapabilities {
                    supports_thumbnails: true,
                    supports_duplication: true,
                    can_reorder: true,
                    can_contain_children: true,
                    user_creatable: true,
                },
            )
        } else if self.is_category_folder() {
            let stage = self
                .category
                .map(WorkflowStage::from_stack_category)
                .unwrap_or(WorkflowStage::Generation);
            (
                Some(stage),
                AccentCategory::from_workflow(stage),
                Some("folder"),
                LayerCapabilities {
                    supports_thumbnails: false,
                    supports_duplication: false,
                    can_reorder: false,
                    can_contain_children: true,
                    user_creatable: false,
                },
            )
        } else {
            (
                Some(WorkflowStage::Generation),
                AccentCategory::Neutral,
                Some("folder"),
                LayerCapabilities {
                    supports_thumbnails: false,
                    supports_duplication: true,
                    can_reorder: true,
                    can_contain_children: true,
                    user_creatable: true,
                },
            )
        };
        LayerInstanceMeta {
            id: self.id,
            display_name: self.name.clone(),
            type_id: None,
            workflow_stage: stage,
            icon_key: icon.map(|s| s.to_string()),
            accent,
            description: None,
            enabled: self.enabled,
            opacity: self.opacity,
            build_status: BuildStatus::Idle,
            cache_status,
            cache_policy: self.cache_policy,
            channels: ChannelRequirements::default(),
            mask_compatibility: MaskCompatibility::SupportsMasks,
            capabilities: caps,
            inspector_sections: vec!["group".into()],
            suggested_next: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{FlatParams, HydraulicErosionParams, MountainParams};

    #[test]
    fn type_ids_stable() {
        assert_eq!(
            LayerKind::Mountains(MountainParams::default()).type_id(),
            "mountain"
        );
        assert_eq!(
            LayerKind::HydraulicErosion(HydraulicErosionParams::default()).type_id(),
            "hydraulic_erosion"
        );
    }

    #[test]
    fn instance_meta_from_layer() {
        let layer = Layer::new("Hills", LayerKind::Flat(FlatParams::default()));
        let meta = layer.instance_meta(None, BuildStatus::Ready, CacheState::Fresh);
        assert_eq!(meta.display_name, "Hills");
        assert_eq!(meta.type_id.as_deref(), Some("flat"));
        assert!(meta.enabled);
    }
}
