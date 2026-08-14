//! Operation categories and field contracts for terrain layers.

use super::LayerKind;
use crate::fields::FieldId;
use crate::tiling::DirtyClass;
use serde::{Deserialize, Serialize};

use super::ScaleBand;

/// Internal height-op classification (create vs transform vs sim / surface).
///
/// **Not** the World Creator artist taxonomy. Artist-facing folders are
/// Shape Layers / Biome Filters / Simulation — see [`StackCategory`](super::group_mode::StackCategory)
/// and [`biome_destination_section`](super::stack::biome_destination_section). Do not surface
/// "Generator" / "Modifier" as Shape workflow labels in the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationCategory {
    /// Creates height contribution (internal; often a Shape Layer or Filter by kind).
    Generator,
    /// Transforms existing height (internal; Path/Polygon stay Shape; Terrace/filters → Biome Filters).
    Modifier,
    Simulation,
    Analysis,
    Surface,
    Group,
    ImportedData,
    Organisation,
}

impl OperationCategory {
    /// Internal / debug label — prefer WC folder names in artist UI.
    pub fn label(self) -> &'static str {
        match self {
            OperationCategory::Generator => "Height source",
            OperationCategory::Modifier => "Height transform",
            OperationCategory::Simulation => "Simulation",
            OperationCategory::Analysis => "Analysis",
            OperationCategory::Surface => "Surface",
            OperationCategory::Group => "Group",
            OperationCategory::ImportedData => "Imported",
            OperationCategory::Organisation => "Organisation",
        }
    }

    /// Internal / debug badge — not used for WC Shape Layers UI.
    pub fn short_badge(self) -> &'static str {
        match self {
            OperationCategory::Generator => "HSRC",
            OperationCategory::Modifier => "HXFM",
            OperationCategory::Simulation => "SIM",
            OperationCategory::Analysis => "ANL",
            OperationCategory::Surface => "SRF",
            OperationCategory::Group => "GRP",
            OperationCategory::ImportedData => "IMP",
            OperationCategory::Organisation => "ORG",
        }
    }
}

impl LayerKind {
    pub fn category(&self) -> OperationCategory {
        match self {
            LayerKind::SculptBase(_)
            | LayerKind::SculptStrokes(_)
            | LayerKind::TerrainConstraints(_)
            | LayerKind::Flat(_)
            | LayerKind::Ramp(_)
            | LayerKind::NoiseValue(_)
            | LayerKind::NoisePerlin(_)
            | LayerKind::NoiseOpenSimplex(_)
            | LayerKind::NoiseWorley(_)
            | LayerKind::Fbm(_)
            | LayerKind::Ridged(_)
            | LayerKind::DomainWarp(_)
            | LayerKind::Mesa(_)
            | LayerKind::Island(_)
            | LayerKind::Mountains(_)
            | LayerKind::Volcano(_)
            | LayerKind::Uplift(_)
            | LayerKind::Dunes(_)
            | LayerKind::Canyons(_)
            | LayerKind::VoronoiRegions(_)
            | LayerKind::ProceduralShape(_)
            | LayerKind::Stamp2d(_)
            | LayerKind::Stamp3d(_) => OperationCategory::Generator,

            LayerKind::ImportHeightmap(_) => OperationCategory::ImportedData,

            LayerKind::Terrace(_)
            | LayerKind::GradientReconstruct(_)
            | LayerKind::GeomorphicDetail(_)
            | LayerKind::Plateau(_)
            | LayerKind::Blur(_)
            | LayerKind::Coastal(_)
            | LayerKind::EffectFilter(_)
            | LayerKind::Path(_)
            | LayerKind::PolygonHeight(_)
            | LayerKind::OverhangStamp(_)
            | LayerKind::LocalSdf(_) => OperationCategory::Modifier,

            LayerKind::ThermalErosion(_)
            | LayerKind::LandscapeEvolution(_)
            | LayerKind::HydrologyRepair(_)
            | LayerKind::EcosystemFeedback(_)
            | LayerKind::HydraulicErosion(_)
            | LayerKind::DebrisFlow(_)
            | LayerKind::StreamPowerErosion(_)
            | LayerKind::MultiScaleAmplify(_)
            | LayerKind::RiverCarve(_)
            | LayerKind::RiverNetwork(_)
            | LayerKind::SandSimulation(_)
            | LayerKind::FluidSimulation(_) => OperationCategory::Simulation,

            LayerKind::Materials(_) | LayerKind::Biomes(_) | LayerKind::Vegetation(_) => {
                OperationCategory::Surface
            }
        }
    }

    /// Fields this operation requires to be present (beyond height for modifiers/sims).
    pub fn required_fields(&self) -> Vec<FieldId> {
        match self {
            LayerKind::ThermalErosion(_)
            | LayerKind::HydraulicErosion(_)
            | LayerKind::DebrisFlow(_)
            | LayerKind::StreamPowerErosion(_)
            | LayerKind::SculptStrokes(_)
            | LayerKind::TerrainConstraints(_)
            | LayerKind::GradientReconstruct(_)
            | LayerKind::LandscapeEvolution(_)
            | LayerKind::HydrologyRepair(_)
            | LayerKind::GeomorphicDetail(_)
            | LayerKind::EcosystemFeedback(_)
            | LayerKind::MultiScaleAmplify(_)
            | LayerKind::RiverCarve(_)
            | LayerKind::RiverNetwork(_)
            | LayerKind::SandSimulation(_)
            | LayerKind::FluidSimulation(_)
            | LayerKind::Terrace(_)
            | LayerKind::Plateau(_)
            | LayerKind::Blur(_)
            | LayerKind::Coastal(_)
            | LayerKind::EffectFilter(_)
            | LayerKind::Path(_)
            | LayerKind::PolygonHeight(_) => vec![FieldId::Height],
            LayerKind::Biomes(_) => vec![FieldId::Height],
            LayerKind::Vegetation(_) => vec![FieldId::Height],
            LayerKind::Materials(_) => vec![FieldId::Height],
            _ => Vec::new(),
        }
    }

    pub fn optional_fields(&self) -> Vec<FieldId> {
        match self {
            LayerKind::SandSimulation(_) => vec![
                FieldId::WindExposure,
                FieldId::Vegetation,
                FieldId::SoilMoisture,
            ],
            LayerKind::HydraulicErosion(_)
            | LayerKind::ThermalErosion(_)
            | LayerKind::DebrisFlow(_) => {
                vec![
                    FieldId::Hardness,
                    FieldId::Rainfall,
                    FieldId::SoilDepth,
                    FieldId::Vegetation,
                    FieldId::BedrockHeight,
                    FieldId::SedimentThickness,
                    FieldId::DebrisDepth,
                ]
            }
            LayerKind::StreamPowerErosion(_) => {
                vec![FieldId::Hardness, FieldId::FlowAccumulation]
            }
            LayerKind::Biomes(_) => vec![
                FieldId::Temperature,
                FieldId::Rainfall,
                FieldId::Wetness,
                FieldId::Slope,
            ],
            LayerKind::Vegetation(_) => {
                vec![
                    FieldId::Biomes,
                    FieldId::Wetness,
                    FieldId::Slope,
                    FieldId::Snow,
                ]
            }
            LayerKind::Materials(_) => {
                vec![
                    FieldId::Slope,
                    FieldId::Curvature,
                    FieldId::Wetness,
                    FieldId::Erosion,
                ]
            }
            _ => Vec::new(),
        }
    }

    pub fn produced_fields(&self) -> Vec<FieldId> {
        match self {
            LayerKind::SculptStrokes(_) => vec![
                FieldId::Height,
                FieldId::Hardness,
                FieldId::Named(crate::fields::keys::SCULPT_PROTECTION.into()),
                FieldId::Named(crate::fields::keys::UPLIFT_RATE.into()),
                FieldId::SedimentThickness,
                FieldId::Named(crate::fields::keys::EDIT_REGION.into()),
            ],
            LayerKind::TerrainConstraints(_) => vec![
                FieldId::Height,
                FieldId::Named(crate::fields::keys::CONSTRAINT_TARGET.into()),
                FieldId::Named(crate::fields::keys::CONSTRAINT_WEIGHT.into()),
                FieldId::Named(crate::fields::keys::SCULPT_PROTECTION.into()),
                FieldId::Named(crate::fields::keys::UPLIFT_RATE.into()),
                FieldId::Named(crate::fields::keys::EDIT_REGION.into()),
            ],
            LayerKind::GradientReconstruct(_) => vec![
                FieldId::Height,
                FieldId::Named(crate::fields::keys::CONSTRAINT_ERROR.into()),
            ],
            LayerKind::LandscapeEvolution(_) => vec![
                FieldId::Height,
                FieldId::FlowDirection,
                FieldId::FlowAccumulation,
                FieldId::StreamOrder,
                FieldId::SpeIncision,
                FieldId::Erosion,
                FieldId::Deposition,
                FieldId::WaterDischarge,
                FieldId::SedimentThickness,
                FieldId::Named(crate::fields::keys::UPLIFT_RATE.into()),
                FieldId::Named(crate::fields::keys::TECTONIC_BASE.into()),
            ],
            LayerKind::HydrologyRepair(_) => vec![
                FieldId::Height,
                FieldId::FlowDirection,
                FieldId::FlowAccumulation,
                FieldId::StreamOrder,
                FieldId::SpeIncision,
                FieldId::Named(crate::fields::keys::REPAIR_REGION.into()),
            ],
            LayerKind::GeomorphicDetail(_) => vec![
                FieldId::Height,
                FieldId::Named(crate::fields::keys::DETAIL_MASK.into()),
                FieldId::FineFlow,
                FieldId::MicroChannel,
                FieldId::RidgeBreakup,
                FieldId::FineErosion,
            ],
            LayerKind::EcosystemFeedback(_) => vec![
                FieldId::Height,
                FieldId::Hardness,
                FieldId::Deposition,
                FieldId::Named(crate::fields::keys::ROOT_COHESION.into()),
            ],
            LayerKind::Island(_) => vec![
                FieldId::Height,
                FieldId::Named(crate::fields::keys::LAND_MASK.into()),
                FieldId::Named(crate::fields::keys::SHORE_DISTANCE.into()),
                FieldId::Named(crate::fields::keys::BATHYMETRY.into()),
                FieldId::Named(crate::fields::keys::SHELF.into()),
                FieldId::Named(crate::fields::keys::BEACH.into()),
                FieldId::Named(crate::fields::keys::REEF.into()),
                FieldId::Named(crate::fields::keys::MOUNTAIN_MASK.into()),
            ],
            LayerKind::HydraulicErosion(_) => vec![
                FieldId::Height,
                FieldId::Wetness,
                FieldId::Sediment,
                FieldId::Erosion,
                FieldId::Deposition,
                FieldId::Water,
                FieldId::WaterVelocity,
                FieldId::FlowAccumulation,
                FieldId::ChannelMask,
                FieldId::BedrockHeight,
                FieldId::SedimentThickness,
                FieldId::Rainfall,
            ],
            LayerKind::SandSimulation(_) | LayerKind::Dunes(_) => vec![
                FieldId::Height,
                FieldId::SandDepth,
                FieldId::BedrockHeight,
                FieldId::WindDirection,
                FieldId::WindSpeed,
                FieldId::SandFlux,
                FieldId::Erosion,
                FieldId::Deposition,
                FieldId::Sheltering,
                FieldId::DuneCrest,
                FieldId::SandMaterialMask,
                FieldId::FlowDirection,
            ],
            LayerKind::ThermalErosion(_) => {
                vec![
                    FieldId::Height,
                    FieldId::Erosion,
                    FieldId::Deposition,
                    FieldId::Hardness,
                    FieldId::BedrockHeight,
                    FieldId::DebrisDepth,
                    FieldId::SedimentThickness,
                    FieldId::TalusStability,
                    FieldId::Instability,
                ]
            }
            LayerKind::DebrisFlow(_) => {
                vec![
                    FieldId::Height,
                    FieldId::Erosion,
                    FieldId::Deposition,
                    FieldId::BedrockHeight,
                    FieldId::DebrisDepth,
                    FieldId::SedimentThickness,
                    FieldId::SlidePath,
                    FieldId::Instability,
                    FieldId::FlowAccumulation,
                ]
            }
            LayerKind::StreamPowerErosion(_) | LayerKind::RiverCarve(_) => vec![
                FieldId::Height,
                FieldId::FlowDirection,
                FieldId::FlowAccumulation,
                FieldId::StreamOrder,
                FieldId::SpeIncision,
            ],
            LayerKind::Materials(_) => {
                vec![
                    FieldId::Materials,
                    FieldId::Hardness,
                    FieldId::StrataReference,
                ]
            }
            LayerKind::Biomes(_) => vec![
                FieldId::Biomes,
                FieldId::Temperature,
                FieldId::Rainfall,
                FieldId::Humidity,
                FieldId::Aridity,
                FieldId::Snow,
                FieldId::SoilMoisture,
                FieldId::WindExposure,
            ],
            LayerKind::Vegetation(_) => vec![FieldId::Vegetation],
            LayerKind::OverhangStamp(_) | LayerKind::LocalSdf(_) => {
                vec![
                    FieldId::Height,
                    FieldId::OverhangCeiling,
                    FieldId::OverhangMask,
                ]
            }
            _ if matches!(self.category(), OperationCategory::Generator) => {
                vec![FieldId::Height]
            }
            _ => vec![FieldId::Height],
        }
    }

    pub fn modified_fields(&self) -> Vec<FieldId> {
        self.produced_fields()
            .into_iter()
            .filter(|f| {
                *f == FieldId::Height
                    || matches!(
                        self.category(),
                        OperationCategory::Simulation | OperationCategory::Modifier
                    )
            })
            .collect()
    }

    pub fn spatial_dependency(&self) -> DirtyClass {
        crate::tiling::dirty_class_for(self)
    }

    /// Phase 11 Rule 3 — scale ownership for this operator family.
    ///
    /// Micro / MultiScale operators must not replace the macro silhouette.
    pub fn scale_band(&self) -> ScaleBand {
        match self {
            // --- MACRO: landmass / tectonic silhouette ---
            LayerKind::Flat(_)
            | LayerKind::Ramp(_)
            | LayerKind::SculptBase(_)
            | LayerKind::ImportHeightmap(_)
            | LayerKind::Island(_)
            | LayerKind::Mountains(_)
            | LayerKind::Mesa(_)
            | LayerKind::Volcano(_)
            | LayerKind::Uplift(_)
            | LayerKind::VoronoiRegions(_)
            | LayerKind::ProceduralShape(_)
            | LayerKind::LandscapeEvolution(_) => ScaleBand::Macro,

            // --- MULTI-SCALE: cascade while locking longer wavelengths ---
            LayerKind::MultiScaleAmplify(_)
            | LayerKind::GeomorphicDetail(_)
            | LayerKind::HydrologyRepair(_) => ScaleBand::MultiScale,

            // --- MICRO: fine surface / decorative detail ---
            LayerKind::Blur(_)
            | LayerKind::EffectFilter(_)
            | LayerKind::OverhangStamp(_)
            | LayerKind::LocalSdf(_)
            | LayerKind::SandSimulation(_)
            | LayerKind::Dunes(_) => ScaleBand::Micro,

            // --- MESO: ridges, valleys, drainage, primary erosion ---
            LayerKind::NoiseValue(_)
            | LayerKind::NoisePerlin(_)
            | LayerKind::NoiseOpenSimplex(_)
            | LayerKind::NoiseWorley(_)
            | LayerKind::Fbm(_)
            | LayerKind::Ridged(_)
            | LayerKind::DomainWarp(_)
            | LayerKind::Terrace(_)
            | LayerKind::Plateau(_)
            | LayerKind::Canyons(_)
            | LayerKind::Path(_)
            | LayerKind::PolygonHeight(_)
            | LayerKind::Stamp2d(_)
            | LayerKind::Stamp3d(_)
            | LayerKind::SculptStrokes(_)
            | LayerKind::TerrainConstraints(_)
            | LayerKind::GradientReconstruct(_)
            | LayerKind::ThermalErosion(_)
            | LayerKind::HydraulicErosion(_)
            | LayerKind::DebrisFlow(_)
            | LayerKind::StreamPowerErosion(_)
            | LayerKind::RiverCarve(_)
            | LayerKind::RiverNetwork(_)
            | LayerKind::Coastal(_)
            | LayerKind::FluidSimulation(_)
            | LayerKind::EcosystemFeedback(_)
            | LayerKind::Materials(_)
            | LayerKind::Biomes(_)
            | LayerKind::Vegetation(_) => ScaleBand::Meso,
        }
    }
}

/// Declared fields for an operation (used by evaluator validation / UI).
#[derive(Debug, Clone)]
pub struct FieldContract {
    pub required: Vec<FieldId>,
    pub optional: Vec<FieldId>,
    pub produced: Vec<FieldId>,
    pub modified: Vec<FieldId>,
    pub spatial: DirtyClass,
}

impl Default for FieldContract {
    fn default() -> Self {
        Self {
            required: Vec::new(),
            optional: Vec::new(),
            produced: Vec::new(),
            modified: Vec::new(),
            spatial: DirtyClass::Local,
        }
    }
}

impl FieldContract {
    pub fn from_kind(kind: &LayerKind) -> Self {
        Self {
            required: kind.required_fields(),
            optional: kind.optional_fields(),
            produced: kind.produced_fields(),
            modified: kind.modified_fields(),
            spatial: kind.spatial_dependency(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{
        EffectFilterParams, HydraulicErosionParams, MultiScaleAmplifyParams, ScaleBand,
        UpliftParams,
    };

    #[test]
    fn hydraulic_is_simulation() {
        let k = LayerKind::HydraulicErosion(HydraulicErosionParams::default());
        assert_eq!(k.category(), OperationCategory::Simulation);
        assert!(k.produced_fields().contains(&FieldId::Wetness));
    }

    #[test]
    fn erosion_family_declares_distinct_debris_and_sediment_fields() {
        for kind in [
            LayerKind::ThermalErosion(Default::default()),
            LayerKind::DebrisFlow(Default::default()),
        ] {
            assert!(kind.optional_fields().contains(&FieldId::DebrisDepth));
            assert!(kind.optional_fields().contains(&FieldId::SedimentThickness));
            assert!(kind.produced_fields().contains(&FieldId::DebrisDepth));
            assert!(kind.produced_fields().contains(&FieldId::SedimentThickness));
        }
    }

    #[test]
    fn scale_band_separates_macro_from_micro() {
        assert_eq!(
            LayerKind::Uplift(UpliftParams::default()).scale_band(),
            ScaleBand::Macro
        );
        assert_eq!(
            LayerKind::HydraulicErosion(HydraulicErosionParams::default()).scale_band(),
            ScaleBand::Meso
        );
        assert_eq!(
            LayerKind::EffectFilter(EffectFilterParams::default()).scale_band(),
            ScaleBand::Micro
        );
        assert_eq!(
            LayerKind::MultiScaleAmplify(MultiScaleAmplifyParams::default()).scale_band(),
            ScaleBand::MultiScale
        );
        assert!(ScaleBand::Micro.respects_macro_silhouette());
    }
}
