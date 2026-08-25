use crate::AuthoredFeatureId;
use serde::{Deserialize, Serialize};
use terra_world::WorldPosition;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SculptStrokeKind {
    Raise,
    Lower,
    Smooth,
    Flatten,
    Ridge,
    Valley,
    Terrace,
    Roughness,
    Uplift,
    Hardness,
    Sediment,
    Protect,
    EncourageErosion,
    Pinch,
    Inflate,
    Erode,
    Noise,
    MountainStamp,
    ValleyStamp,
    PlateauStamp,
    CraterStamp,
    Coastline,
    RiverPath,
    HeightStamp,
}

impl SculptStrokeKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Raise => "Raise",
            Self::Lower => "Lower",
            Self::Smooth => "Smooth",
            Self::Flatten => "Flatten",
            Self::Ridge => "Ridge",
            Self::Valley => "Valley",
            Self::Terrace => "Terrace",
            Self::Roughness => "Roughness",
            Self::Uplift => "Uplift",
            Self::Hardness => "Hardness",
            Self::Sediment => "Sediment",
            Self::Protect => "Protect",
            Self::EncourageErosion => "Encourage Erosion",
            Self::Pinch => "Pinch",
            Self::Inflate => "Inflate",
            Self::Erode => "Erode",
            Self::Noise => "Noise",
            Self::MountainStamp => "Mountain Stamp",
            Self::ValleyStamp => "Valley Stamp",
            Self::PlateauStamp => "Plateau Stamp",
            Self::CraterStamp => "Crater Stamp",
            Self::Coastline => "Coastline",
            Self::RiverPath => "River Path",
            Self::HeightStamp => "Height Stamp",
        }
    }

    /// Legacy foundation-raster mode, where one exists for this semantic brush.
    pub fn foundation_mode(self) -> Option<u8> {
        match self {
            Self::Raise => Some(0),
            Self::Lower | Self::Erode => Some(1),
            Self::Smooth | Self::Pinch => Some(2),
            Self::Flatten => Some(3),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SculptPoint {
    pub u: f32,
    pub v: f32,
    #[serde(default = "one")]
    pub pressure: f32,
}

impl Default for SculptPoint {
    fn default() -> Self {
        Self {
            u: 0.5,
            v: 0.5,
            pressure: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SculptStroke {
    pub kind: SculptStrokeKind,
    #[serde(default)]
    pub points: Vec<SculptPoint>,
    #[serde(default = "sculpt_radius")]
    pub radius_m: f32,
    #[serde(default = "sculpt_strength")]
    pub strength: f32,
    #[serde(default)]
    pub target_height: f32,
    #[serde(default = "sculpt_falloff")]
    pub falloff: f32,
    #[serde(default = "enabled_default")]
    pub enabled: bool,
}

impl Default for SculptStroke {
    fn default() -> Self {
        Self {
            kind: SculptStrokeKind::Raise,
            points: vec![SculptPoint::default()],
            radius_m: sculpt_radius(),
            strength: sculpt_strength(),
            target_height: 0.0,
            falloff: sculpt_falloff(),
            enabled: true,
        }
    }
}

/// Fixed-origin point stored by `WorldMetresV1` histories.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorldSculptPoint {
    pub position: WorldPosition,
    #[serde(default = "one")]
    pub pressure: f32,
}

/// One finite world-metre sculpt feature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldSculptStroke {
    #[serde(default)]
    pub id: AuthoredFeatureId,
    pub kind: SculptStrokeKind,
    #[serde(default)]
    pub points: Vec<WorldSculptPoint>,
    #[serde(default = "sculpt_radius")]
    pub radius_m: f32,
    #[serde(default = "sculpt_strength")]
    pub strength: f32,
    #[serde(default)]
    pub target_height: f32,
    #[serde(default = "sculpt_falloff")]
    pub falloff: f32,
    #[serde(default = "enabled_default")]
    pub enabled: bool,
}

impl Default for WorldSculptStroke {
    fn default() -> Self {
        Self {
            id: AuthoredFeatureId::new(),
            kind: SculptStrokeKind::Raise,
            points: vec![WorldSculptPoint {
                position: WorldPosition::ORIGIN,
                pressure: 1.0,
            }],
            radius_m: sculpt_radius(),
            strength: sculpt_strength(),
            target_height: 0.0,
            falloff: sculpt_falloff(),
            enabled: true,
        }
    }
}

fn one() -> f32 {
    1.0
}

fn sculpt_radius() -> f32 {
    80.0
}

fn sculpt_strength() -> f32 {
    12.0
}

fn sculpt_falloff() -> f32 {
    1.5
}

fn enabled_default() -> bool {
    true
}
