//! Persisted per-operation placement values.

use crate::mask::PlacementDefinition;
use serde::{Deserialize, Serialize};

/// Simple Apply Where control (artist-facing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ApplyWhere {
    #[default]
    EntireBiome,
    PaintedRestriction,
    HeightRange,
    SlopeRange,
    NearWater,
    NearRivers,
    FlowRange,
    Curvature,
    CustomConditions,
    AdvancedMask,
}

/// Operation placement authored in Develop. Serializes with the layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationPlacement {
    #[serde(default)]
    pub apply_where: ApplyWhere,
    #[serde(default)]
    pub definition: PlacementDefinition,
    #[serde(default = "default_height_min")]
    pub height_min: f32,
    #[serde(default = "default_height_max")]
    pub height_max: f32,
    #[serde(default)]
    pub slope_min: f32,
    #[serde(default = "default_slope_max")]
    pub slope_max: f32,
    #[serde(default = "default_flow_min")]
    pub flow_min: f32,
    #[serde(default = "default_near_m")]
    pub near_distance_m: f32,
}

fn default_height_min() -> f32 {
    0.0
}
fn default_height_max() -> f32 {
    2000.0
}
fn default_slope_max() -> f32 {
    50.0
}
fn default_flow_min() -> f32 {
    0.15
}
fn default_near_m() -> f32 {
    80.0
}

impl Default for OperationPlacement {
    fn default() -> Self {
        Self {
            apply_where: ApplyWhere::EntireBiome,
            definition: PlacementDefinition::default(),
            height_min: default_height_min(),
            height_max: default_height_max(),
            slope_min: 0.0,
            slope_max: default_slope_max(),
            flow_min: default_flow_min(),
            near_distance_m: default_near_m(),
        }
    }
}

/// Develop category for contextual creation under a biome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DevelopCategory {
    Terrain,
    Materials,
    Simulation,
    Vegetation,
    Objects,
    Placement,
}
