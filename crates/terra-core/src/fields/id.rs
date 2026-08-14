//! Stable terrain field identifiers.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::keys;

/// Strong field identity for required/produced/modified declarations and IO.
///
/// Display names remain separate; this enum (and [`FieldId::Custom`]) is what
/// references and serialisation use.
///
/// Variants are additive only — never rename serde tags used by documents.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FieldId {
    // --- Geometry ---
    Height,
    /// Alias-friendly base elevation before local detail (often equals height).
    BaseElevation,
    BedrockHeight,
    SedimentThickness,
    SoilDepth,
    SoilThickness,
    // --- Hydrology ---
    Hardness,
    Water,
    WaterVelocity,
    Sediment,
    DepositedSediment,
    FlowDirection,
    FlowAccumulation,
    WaterDischarge,
    WatershedId,
    ChannelMask,
    Wetness,
    // --- Erosion / geology ---
    Erodibility,
    Erosion,
    Deposition,
    ErosionRate,
    DepositionRate,
    SedimentFlux,
    SedimentCapacity,
    TalusStability,
    StrataMaterialId,
    /// Discrete lithology / rock-unit ID map (encoded like materials, \[0,1\]).
    Lithology,
    StreamOrder,
    SpeIncision,
    // --- Analysis (often derived / transient) ---
    Gradient,
    Slope,
    Aspect,
    Curvature,
    MeanCurvature,
    ProfileCurvature,
    PlanCurvature,
    GaussianCurvature,
    Laplacian,
    Convexity,
    Concavity,
    Cavity,
    Roughness,
    RidgeLikelihood,
    ValleyLikelihood,
    DistanceToChannel,
    DistanceToRidge,
    DistanceToValley,
    DrainageDensity,
    WatershedBoundary,
    ChannelWidth,
    Normals,
    // --- Climate / surface ---
    Temperature,
    Rainfall,
    Humidity,
    Aridity,
    Snow,
    SoilMoisture,
    WindExposure,
    /// Local wind direction (radians).
    WindDirection,
    /// Relative wind speed.
    WindSpeed,
    /// Aeolian sand flux.
    SandFlux,
    /// Wind-shadow / sheltering fraction.
    Sheltering,
    Materials,
    Biomes,
    Vegetation,
    StrataReference,
    OverhangCeiling,
    OverhangMask,
    // Matter simulation outputs
    WaterDepth,
    SnowDepth,
    SandDepth,
    DebrisDepth,
    Floodplain,
    DuneCrest,
    SlidePath,
    Instability,
    Meltwater,
    Drift,
    RiverChannel,
    SandMaterialMask,
    SnowMaterialMask,
    ScatterCandidates,
    /// Fine-scale flow organisation from multi-scale amplification.
    FineFlow,
    /// Nested micro-channel / gully intensity.
    MicroChannel,
    /// Ridge-conditioned surface breakup.
    RidgeBreakup,
    /// Fine erosion / incision intensity from amplification.
    FineErosion,
    /// User-published or free-form field keyed by stable UUID.
    Custom(Uuid),
    /// Legacy string key (migration / extras adapter). Prefer [`FieldId::Custom`].
    Named(String),
}

impl FieldId {
    /// Well-known string key used by AuxMaps / disk cache / mask baking.
    pub fn cache_key(&self) -> String {
        match self {
            FieldId::Height => "height".into(),
            FieldId::BaseElevation => "base_elevation".into(),
            FieldId::BedrockHeight => keys::BEDROCK_HEIGHT.into(),
            FieldId::SedimentThickness => keys::SEDIMENT_THICKNESS.into(),
            FieldId::SoilDepth => keys::SOIL_DEPTH.into(),
            FieldId::SoilThickness => "soil_thickness".into(),
            FieldId::Hardness => keys::HARDNESS.into(),
            FieldId::Water => "water".into(),
            FieldId::WaterVelocity => "water_velocity".into(),
            FieldId::Sediment => keys::SEDIMENT.into(),
            FieldId::DepositedSediment => "deposited_sediment".into(),
            FieldId::FlowDirection => keys::FLOW_DIRECTION.into(),
            FieldId::FlowAccumulation => keys::FLOW_ACCUMULATION.into(),
            FieldId::WaterDischarge => "water_discharge".into(),
            FieldId::WatershedId => "watershed_id".into(),
            FieldId::ChannelMask => "channel_mask".into(),
            FieldId::Wetness => keys::WETNESS.into(),
            FieldId::Erodibility => "erodibility".into(),
            FieldId::Erosion => keys::EROSION.into(),
            FieldId::Deposition => keys::DEPOSITION.into(),
            FieldId::ErosionRate => "erosion_rate".into(),
            FieldId::DepositionRate => "deposition_rate".into(),
            FieldId::SedimentFlux => "sediment_flux".into(),
            FieldId::SedimentCapacity => "sediment_capacity".into(),
            FieldId::TalusStability => keys::TALUS_STABILITY.into(),
            FieldId::StrataMaterialId => "strata_material_id".into(),
            FieldId::Lithology => keys::LITHOLOGY.into(),
            FieldId::StreamOrder => keys::STREAM_ORDER.into(),
            FieldId::SpeIncision => keys::SPE_INCISION.into(),
            FieldId::Gradient => "gradient".into(),
            FieldId::Slope => keys::SLOPE.into(),
            FieldId::Aspect => "aspect".into(),
            FieldId::Curvature => keys::CURVATURE.into(),
            FieldId::MeanCurvature => "mean_curvature".into(),
            FieldId::ProfileCurvature => "profile_curvature".into(),
            FieldId::PlanCurvature => "plan_curvature".into(),
            FieldId::GaussianCurvature => "gaussian_curvature".into(),
            FieldId::Laplacian => "laplacian".into(),
            FieldId::Convexity => "convexity".into(),
            FieldId::Concavity => "concavity".into(),
            FieldId::Cavity => "cavity".into(),
            FieldId::Roughness => "roughness".into(),
            FieldId::RidgeLikelihood => "ridge_likelihood".into(),
            FieldId::ValleyLikelihood => "valley_likelihood".into(),
            FieldId::DistanceToChannel => "distance_to_channel".into(),
            FieldId::DistanceToRidge => "distance_to_ridge".into(),
            FieldId::DistanceToValley => "distance_to_valley".into(),
            FieldId::DrainageDensity => "drainage_density".into(),
            FieldId::WatershedBoundary => "watershed_boundary".into(),
            FieldId::ChannelWidth => "channel_width".into(),
            FieldId::Normals => "normals".into(),
            FieldId::Temperature => keys::TEMPERATURE.into(),
            FieldId::Rainfall => keys::RAINFALL.into(),
            FieldId::Humidity => keys::HUMIDITY.into(),
            FieldId::Aridity => keys::ARIDITY.into(),
            FieldId::Snow => keys::SNOW.into(),
            FieldId::SoilMoisture => keys::SOIL_MOISTURE.into(),
            FieldId::WindExposure => keys::WIND_EXPOSURE.into(),
            FieldId::WindDirection => keys::WIND_DIRECTION.into(),
            FieldId::WindSpeed => keys::WIND_SPEED.into(),
            FieldId::SandFlux => keys::SAND_FLUX.into(),
            FieldId::Sheltering => keys::SHELTERING.into(),
            FieldId::Materials => keys::MATERIALS.into(),
            FieldId::Biomes => keys::BIOMES.into(),
            FieldId::Vegetation => keys::VEGETATION.into(),
            FieldId::StrataReference => keys::STRATA_REFERENCE.into(),
            FieldId::OverhangCeiling => keys::OVERHANG_CEILING.into(),
            FieldId::OverhangMask => keys::OVERHANG_MASK.into(),
            FieldId::WaterDepth => keys::WATER_DEPTH.into(),
            FieldId::SnowDepth => keys::SNOW_DEPTH.into(),
            FieldId::SandDepth => keys::SAND_DEPTH.into(),
            FieldId::DebrisDepth => keys::DEBRIS_DEPTH.into(),
            FieldId::Floodplain => keys::FLOODPLAIN.into(),
            FieldId::DuneCrest => keys::DUNE_CREST.into(),
            FieldId::SlidePath => keys::SLIDE_PATH.into(),
            FieldId::Instability => keys::INSTABILITY.into(),
            FieldId::Meltwater => keys::MELTWATER.into(),
            FieldId::Drift => keys::DRIFT.into(),
            FieldId::RiverChannel => keys::RIVER_CHANNEL.into(),
            FieldId::SandMaterialMask => keys::SAND_MATERIAL_MASK.into(),
            FieldId::SnowMaterialMask => keys::SNOW_MATERIAL_MASK.into(),
            FieldId::ScatterCandidates => keys::SCATTER_CANDIDATES.into(),
            FieldId::FineFlow => keys::FINE_FLOW.into(),
            FieldId::MicroChannel => keys::MICRO_CHANNEL.into(),
            FieldId::RidgeBreakup => keys::RIDGE_BREAKUP.into(),
            FieldId::FineErosion => keys::FINE_EROSION.into(),
            FieldId::Custom(id) => format!("custom:{id}"),
            FieldId::Named(s) => s.clone(),
        }
    }

    pub fn from_cache_key(key: &str) -> Self {
        match key {
            "height" => FieldId::Height,
            "base_elevation" => FieldId::BaseElevation,
            keys::BEDROCK_HEIGHT => FieldId::BedrockHeight,
            keys::SEDIMENT_THICKNESS | keys::SEDIMENT_DEPTH | keys::LOOSE_SEDIMENT => {
                FieldId::SedimentThickness
            }
            keys::SOIL_DEPTH => FieldId::SoilDepth,
            "soil_thickness" => FieldId::SoilThickness,
            keys::HARDNESS => FieldId::Hardness,
            "water" => FieldId::Water,
            "water_velocity" => FieldId::WaterVelocity,
            keys::SEDIMENT => FieldId::Sediment,
            "deposited_sediment" => FieldId::DepositedSediment,
            keys::FLOW_DIRECTION => FieldId::FlowDirection,
            keys::FLOW_ACCUMULATION => FieldId::FlowAccumulation,
            "water_discharge" => FieldId::WaterDischarge,
            "watershed_id" => FieldId::WatershedId,
            "channel_mask" => FieldId::ChannelMask,
            keys::WETNESS => FieldId::Wetness,
            "erodibility" => FieldId::Erodibility,
            keys::EROSION => FieldId::Erosion,
            keys::DEPOSITION => FieldId::Deposition,
            "erosion_rate" => FieldId::ErosionRate,
            "deposition_rate" => FieldId::DepositionRate,
            "sediment_flux" => FieldId::SedimentFlux,
            "sediment_capacity" => FieldId::SedimentCapacity,
            keys::TALUS_STABILITY => FieldId::TalusStability,
            "strata_material_id" => FieldId::StrataMaterialId,
            keys::LITHOLOGY => FieldId::Lithology,
            keys::STREAM_ORDER => FieldId::StreamOrder,
            keys::SPE_INCISION => FieldId::SpeIncision,
            "gradient" => FieldId::Gradient,
            keys::SLOPE => FieldId::Slope,
            "aspect" => FieldId::Aspect,
            keys::CURVATURE => FieldId::Curvature,
            "mean_curvature" => FieldId::MeanCurvature,
            "profile_curvature" => FieldId::ProfileCurvature,
            "plan_curvature" => FieldId::PlanCurvature,
            "gaussian_curvature" => FieldId::GaussianCurvature,
            "laplacian" => FieldId::Laplacian,
            "convexity" => FieldId::Convexity,
            "concavity" => FieldId::Concavity,
            "cavity" => FieldId::Cavity,
            "roughness" => FieldId::Roughness,
            "ridge_likelihood" => FieldId::RidgeLikelihood,
            "valley_likelihood" => FieldId::ValleyLikelihood,
            "distance_to_channel" => FieldId::DistanceToChannel,
            "distance_to_ridge" => FieldId::DistanceToRidge,
            "distance_to_valley" => FieldId::DistanceToValley,
            "drainage_density" => FieldId::DrainageDensity,
            "watershed_boundary" => FieldId::WatershedBoundary,
            "channel_width" => FieldId::ChannelWidth,
            "normals" => FieldId::Normals,
            keys::TEMPERATURE => FieldId::Temperature,
            keys::RAINFALL => FieldId::Rainfall,
            keys::HUMIDITY => FieldId::Humidity,
            keys::ARIDITY => FieldId::Aridity,
            keys::SNOW => FieldId::Snow,
            keys::SOIL_MOISTURE => FieldId::SoilMoisture,
            keys::WIND_EXPOSURE => FieldId::WindExposure,
            keys::WIND_DIRECTION => FieldId::WindDirection,
            keys::WIND_SPEED => FieldId::WindSpeed,
            keys::SAND_FLUX => FieldId::SandFlux,
            keys::SHELTERING => FieldId::Sheltering,
            keys::MATERIALS => FieldId::Materials,
            keys::BIOMES => FieldId::Biomes,
            keys::VEGETATION => FieldId::Vegetation,
            keys::STRATA_REFERENCE => FieldId::StrataReference,
            keys::OVERHANG_CEILING => FieldId::OverhangCeiling,
            keys::OVERHANG_MASK => FieldId::OverhangMask,
            keys::WATER_DEPTH => FieldId::WaterDepth,
            keys::SNOW_DEPTH => FieldId::SnowDepth,
            keys::SAND_DEPTH => FieldId::SandDepth,
            keys::DEBRIS_DEPTH => FieldId::DebrisDepth,
            keys::FLOODPLAIN => FieldId::Floodplain,
            keys::DUNE_CREST => FieldId::DuneCrest,
            keys::SLIDE_PATH => FieldId::SlidePath,
            keys::INSTABILITY => FieldId::Instability,
            keys::MELTWATER => FieldId::Meltwater,
            keys::DRIFT => FieldId::Drift,
            keys::RIVER_CHANNEL => FieldId::RiverChannel,
            keys::SAND_MATERIAL_MASK => FieldId::SandMaterialMask,
            keys::SNOW_MATERIAL_MASK => FieldId::SnowMaterialMask,
            keys::SCATTER_CANDIDATES => FieldId::ScatterCandidates,
            keys::FINE_FLOW => FieldId::FineFlow,
            keys::MICRO_CHANNEL => FieldId::MicroChannel,
            keys::RIDGE_BREAKUP => FieldId::RidgeBreakup,
            keys::FINE_EROSION => FieldId::FineErosion,
            other if other.starts_with("custom:") => {
                if let Ok(u) = Uuid::parse_str(&other["custom:".len()..]) {
                    FieldId::Custom(u)
                } else {
                    FieldId::Named(other.to_string())
                }
            }
            other => FieldId::Named(other.to_string()),
        }
    }

    pub fn display_name(&self) -> String {
        match self {
            FieldId::Height => "Height".into(),
            FieldId::BaseElevation => "Base Elevation".into(),
            FieldId::BedrockHeight => "Bedrock Elevation".into(),
            FieldId::SedimentThickness => "Sediment Thickness".into(),
            FieldId::SoilDepth => "Soil Depth".into(),
            FieldId::SoilThickness => "Soil Thickness".into(),
            FieldId::Hardness => "Hardness".into(),
            FieldId::Water => "Water".into(),
            FieldId::WaterVelocity => "Water Velocity".into(),
            FieldId::Sediment => "Sediment".into(),
            FieldId::DepositedSediment => "Deposited Sediment".into(),
            FieldId::FlowDirection => "Flow Direction".into(),
            FieldId::FlowAccumulation => "Flow Accumulation".into(),
            FieldId::WaterDischarge => "Water Discharge".into(),
            FieldId::WatershedId => "Watershed ID".into(),
            FieldId::ChannelMask => "Channel Mask".into(),
            FieldId::Wetness => "Wetness".into(),
            FieldId::Erodibility => "Erodibility".into(),
            FieldId::Erosion => "Erosion".into(),
            FieldId::Deposition => "Deposition".into(),
            FieldId::ErosionRate => "Erosion Rate".into(),
            FieldId::DepositionRate => "Deposition Rate".into(),
            FieldId::SedimentFlux => "Sediment Flux".into(),
            FieldId::SedimentCapacity => "Sediment Capacity".into(),
            FieldId::TalusStability => "Talus / Slope Stability".into(),
            FieldId::StrataMaterialId => "Strata Material ID".into(),
            FieldId::Lithology => "Lithology".into(),
            FieldId::StreamOrder => "Stream Order".into(),
            FieldId::SpeIncision => "SPE Incision".into(),
            FieldId::Gradient => "Gradient".into(),
            FieldId::Slope => "Slope".into(),
            FieldId::Aspect => "Aspect".into(),
            FieldId::Curvature => "Curvature".into(),
            FieldId::MeanCurvature => "Mean Curvature".into(),
            FieldId::ProfileCurvature => "Profile Curvature".into(),
            FieldId::PlanCurvature => "Plan Curvature".into(),
            FieldId::GaussianCurvature => "Gaussian Curvature".into(),
            FieldId::Laplacian => "Laplacian".into(),
            FieldId::Convexity => "Convexity".into(),
            FieldId::Concavity => "Concavity".into(),
            FieldId::Cavity => "Cavity".into(),
            FieldId::Roughness => "Roughness".into(),
            FieldId::RidgeLikelihood => "Ridge Likelihood".into(),
            FieldId::ValleyLikelihood => "Valley Likelihood".into(),
            FieldId::DistanceToChannel => "Distance to Channel".into(),
            FieldId::DistanceToRidge => "Distance to Ridge".into(),
            FieldId::DistanceToValley => "Distance to Valley".into(),
            FieldId::DrainageDensity => "Drainage Density".into(),
            FieldId::WatershedBoundary => "Watershed Boundary".into(),
            FieldId::ChannelWidth => "Channel Width".into(),
            FieldId::Normals => "Normals".into(),
            FieldId::Temperature => "Temperature".into(),
            FieldId::Rainfall => "Rainfall".into(),
            FieldId::Humidity => "Humidity".into(),
            FieldId::Aridity => "Aridity".into(),
            FieldId::Snow => "Snow".into(),
            FieldId::SoilMoisture => "Soil Moisture".into(),
            FieldId::WindExposure => "Wind Exposure".into(),
            FieldId::WindDirection => "Wind Direction".into(),
            FieldId::WindSpeed => "Wind Speed".into(),
            FieldId::SandFlux => "Sand Flux".into(),
            FieldId::Sheltering => "Sheltering".into(),
            FieldId::Materials => "Materials".into(),
            FieldId::Biomes => "Biomes".into(),
            FieldId::Vegetation => "Vegetation".into(),
            FieldId::StrataReference => "Strata Reference".into(),
            FieldId::OverhangCeiling => "Overhang Ceiling".into(),
            FieldId::OverhangMask => "Overhang Mask".into(),
            FieldId::WaterDepth => "Water Depth".into(),
            FieldId::SnowDepth => "Snow Depth".into(),
            FieldId::SandDepth => "Sand Depth".into(),
            FieldId::DebrisDepth => "Debris Depth".into(),
            FieldId::Floodplain => "Floodplain".into(),
            FieldId::DuneCrest => "Dune Crest".into(),
            FieldId::SlidePath => "Slide Path".into(),
            FieldId::Instability => "Instability".into(),
            FieldId::Meltwater => "Meltwater".into(),
            FieldId::Drift => "Drift".into(),
            FieldId::RiverChannel => "River Channel".into(),
            FieldId::SandMaterialMask => "Sand Material Mask".into(),
            FieldId::SnowMaterialMask => "Snow Material Mask".into(),
            FieldId::ScatterCandidates => "Scatter Candidates".into(),
            FieldId::FineFlow => "Fine Flow".into(),
            FieldId::MicroChannel => "Micro Channel".into(),
            FieldId::RidgeBreakup => "Ridge Breakup".into(),
            FieldId::FineErosion => "Fine Erosion".into(),
            FieldId::Custom(id) => format!("Custom ({id})"),
            FieldId::Named(s) => s.clone(),
        }
    }

    /// Analysis fields that are typically derived lazily rather than authored.
    pub fn is_derived_analysis(&self) -> bool {
        matches!(
            self,
            FieldId::Gradient
                | FieldId::Slope
                | FieldId::Aspect
                | FieldId::Curvature
                | FieldId::MeanCurvature
                | FieldId::ProfileCurvature
                | FieldId::PlanCurvature
                | FieldId::GaussianCurvature
                | FieldId::Laplacian
                | FieldId::Convexity
                | FieldId::Concavity
                | FieldId::Cavity
                | FieldId::Roughness
                | FieldId::RidgeLikelihood
                | FieldId::ValleyLikelihood
                | FieldId::DistanceToChannel
                | FieldId::DistanceToRidge
                | FieldId::DistanceToValley
                | FieldId::DrainageDensity
                | FieldId::WatershedBoundary
                | FieldId::ChannelWidth
                | FieldId::Normals
        )
    }

    /// Whether this field is appropriate for normalised weight blending.
    pub fn is_weight_field(&self) -> bool {
        matches!(
            self,
            FieldId::Materials | FieldId::Biomes | FieldId::Vegetation
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_roundtrip() {
        for id in [
            FieldId::Height,
            FieldId::Hardness,
            FieldId::Wetness,
            FieldId::Custom(Uuid::nil()),
            FieldId::Named("extra".into()),
        ] {
            let key = id.cache_key();
            assert_eq!(FieldId::from_cache_key(&key).cache_key(), key);
        }
    }

    #[test]
    fn legacy_sediment_keys_resolve_to_the_canonical_field() {
        for key in [
            keys::SEDIMENT_THICKNESS,
            keys::SEDIMENT_DEPTH,
            keys::LOOSE_SEDIMENT,
        ] {
            assert_eq!(FieldId::from_cache_key(key), FieldId::SedimentThickness);
            assert_eq!(
                FieldId::from_cache_key(key).cache_key(),
                keys::SEDIMENT_THICKNESS
            );
        }
    }
}
