//! Persisted parameters for hydrology algorithms.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiverCarveParams {
    pub accumulation_threshold: f32,
    pub depth: f32,
    pub width: f32,
    pub bank_smooth: f32,
    pub use_dinfinity: bool,
    #[serde(default)]
    pub guide: crate::mask_types::MaskSource,
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
            guide: crate::mask_types::MaskSource::None,
            guide_boost: 3.0,
        }
    }
}
