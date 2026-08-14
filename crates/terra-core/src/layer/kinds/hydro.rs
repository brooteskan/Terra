//! Layer parameter kinds (split by family).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiverCarveParams {
    pub accumulation_threshold: f32,
    pub depth: f32,
    pub width: f32,
    pub bank_smooth: f32,
    pub use_dinfinity: bool,
    /// Optional guide mask: raises effective accumulation so channels prefer the guide.
    #[serde(default)]
    pub guide: crate::mask::MaskSource,
    /// How strongly `guide` pulls channels (0 = off, ~4 = strong bias).
    #[serde(default = "default_river_guide_boost")]
    pub guide_boost: f32,
}

fn default_river_guide_boost() -> f32 {
    3.0
}

impl Default for RiverCarveParams {
    fn default() -> Self {
        Self {
            accumulation_threshold: 50.0,
            depth: 25.0,
            width: 4.0,
            bank_smooth: 1.5,
            use_dinfinity: true,
            guide: crate::mask::MaskSource::None,
            guide_boost: 3.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoastalParams {
    pub sea_level: f32,
    pub beach_width: f32,
    pub flatten_below: bool,
    /// Maximum depth of the softened underwater shelf; zero preserves flat sea.
    #[serde(default)]
    pub shelf_depth: f32,
}

impl Default for CoastalParams {
    fn default() -> Self {
        Self {
            sea_level: 0.0,
            beach_width: 20.0,
            flatten_below: true,
            shelf_depth: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiverNode {
    pub u: f32,
    pub v: f32,
    pub flow: f32,
    pub width: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiverNetworkParams {
    pub springs: Vec<RiverNode>,
    /// When true, rebuild channels from springs using steepest descent on input.
    pub auto_generate: bool,
    pub max_length: u32,
    pub carve_depth: f32,
    pub valley_width: f32,
    pub seed: u64,
}

impl Default for RiverNetworkParams {
    fn default() -> Self {
        Self {
            springs: vec![RiverNode {
                u: 0.5,
                v: 0.15,
                flow: 1.0,
                width: 1.0,
            }],
            auto_generate: true,
            max_length: 512,
            // World-meter defaults sized for multi-km terrains (cell-aware clamp at eval).
            carve_depth: 18.0,
            valley_width: 80.0,
            seed: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandSimParams {
    /// Total evolution steps (Full quality may run thousands in the eval worker).
    pub iterations: u32,
    /// Avalanche angle of repose (degrees). Legacy name kept for documents.
    pub slope_angle_deg: f32,
    /// Initial sand cover thickness (meters). Also drives sand supply.
    pub spawn_amount: f32,
    /// Mix of evolved height vs input \[0, 1\].
    pub strength: f32,
    /// Prevailing wind direction in degrees (0 = +X).
    #[serde(default)]
    pub wind_direction_deg: f32,
    /// Relative wind speed / transport energy \[0, 2\].
    #[serde(default = "default_sand_wind_speed")]
    pub wind_speed: f32,
    /// Saltation hop length in cells.
    #[serde(default = "default_sand_transport_length")]
    pub transport_length: f32,
    /// Avalanche sweeps per transport step.
    #[serde(default = "default_sand_avalanche_iters")]
    pub avalanche_iters: u32,
    /// Ridge coherence \[0, 1\] (high → linear/transverse fields).
    #[serde(default = "default_sand_linearity")]
    pub linearity: f32,
    /// Bedrock abrasion on bounce \[0, 1\].
    #[serde(default = "default_sand_abrasion")]
    pub abrasion: f32,
    /// Reptation strength \[0, 1\].
    #[serde(default = "default_sand_reptation")]
    pub reptation: f32,
    /// Optional phasor seed before evolution (0 = use uniform spawn cover only).
    #[serde(default)]
    pub seed_amount: f32,
    /// Seed dune scale when `seed_amount` > 0.
    #[serde(default = "default_sand_seed_scale")]
    pub seed_scale: f32,
}

fn default_sand_wind_speed() -> f32 {
    1.0
}
fn default_sand_transport_length() -> f32 {
    5.0
}
fn default_sand_avalanche_iters() -> u32 {
    10
}
fn default_sand_linearity() -> f32 {
    0.65
}
fn default_sand_abrasion() -> f32 {
    0.03
}
fn default_sand_reptation() -> f32 {
    0.2
}
fn default_sand_seed_scale() -> f32 {
    0.012
}

impl Default for SandSimParams {
    fn default() -> Self {
        Self {
            iterations: 256,
            slope_angle_deg: 33.0,
            spawn_amount: 2.5,
            strength: 1.0,
            wind_direction_deg: 0.0,
            wind_speed: default_sand_wind_speed(),
            transport_length: default_sand_transport_length(),
            avalanche_iters: default_sand_avalanche_iters(),
            linearity: default_sand_linearity(),
            abrasion: default_sand_abrasion(),
            reptation: default_sand_reptation(),
            seed_amount: 0.35,
            seed_scale: default_sand_seed_scale(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluidSimParams {
    pub iterations: u32,
    pub viscosity: f32,
    pub spawn_amount: f32,
    pub strength: f32,
}

impl Default for FluidSimParams {
    fn default() -> Self {
        Self {
            iterations: 48,
            viscosity: 0.35,
            spawn_amount: 2.0,
            strength: 1.0,
        }
    }
}
