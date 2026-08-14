//! Layer parameter kinds (split by family).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VegetationParams {
    pub seed: u64,
    pub density: f32,
    pub min_distance: f32,
    pub min_slope_deg: f32,
    pub max_slope_deg: f32,
    pub biome_id: Option<u32>,
    /// Optional one-way root cohesion: boosts hardness where vegetation is dense \[0,1\].
    #[serde(default)]
    pub root_cohesion: f32,
    /// Optional nested coverage distribution (WC-style where veg lands).
    #[serde(default)]
    pub coverage: crate::mask::Distribution,
    /// Per-instance scale lower bound (multiplies base plant size).
    #[serde(default = "default_scale_min")]
    pub scale_min: f32,
    /// Per-instance scale upper bound (multiplies base plant size).
    #[serde(default = "default_scale_max")]
    pub scale_max: f32,
    /// Max yaw variation in degrees (Â±); 180 â‰ˆ full random yaw.
    #[serde(default = "default_yaw_variation_deg")]
    pub yaw_variation_deg: f32,
}

fn default_scale_min() -> f32 {
    0.5
}
fn default_scale_max() -> f32 {
    1.5
}
fn default_yaw_variation_deg() -> f32 {
    180.0
}

impl Default for VegetationParams {
    fn default() -> Self {
        Self {
            seed: 42,
            density: 0.35,
            min_distance: 3.0,
            min_slope_deg: 0.0,
            max_slope_deg: 35.0,
            // None = grow wherever slope allows. Set biome_id to gate on a Biomes layer.
            biome_id: None,
            root_cohesion: 0.0,
            coverage: crate::mask::Distribution::new(),
            scale_min: default_scale_min(),
            scale_max: default_scale_max(),
            yaw_variation_deg: default_yaw_variation_deg(),
        }
    }
}
