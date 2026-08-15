//! Layer parameter kinds (split by family).

use super::filter_params::EffectFilterParams;
use super::terrain_params::{
    CanyonParams, DuneParams, FbmParams, ImportHeightmapParams, MesaParams, MountainParams,
    PlateauParams, VolcanoParams,
};
use crate::noise::NoiseParams;
use serde::{Deserialize, Serialize};

/// Spline control point in normalized UV with a relative elevation/depth profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathNode {
    pub u: f32,
    pub v: f32,
    pub height: f32,
    pub width: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathParams {
    pub nodes: Vec<PathNode>,
    pub width: f32,
    pub falloff: f32,
    pub noise_strength: f32,
    pub noise_scale: f32,
    pub closed: bool,
    pub height_offset: f32,
    /// When true, carve below surrounding terrain; else raise/add.
    pub carve: bool,
    pub seed: u64,
    /// Interpolate control points with a Catmull-Rom spline.
    #[serde(default = "path_spline_default")]
    pub spline: bool,
    /// Cross-section shaping. 1 = linear shoulder, >1 = flatter centre / sharper banks.
    #[serde(default = "path_profile_default")]
    pub profile: f32,
}

fn path_spline_default() -> bool {
    true
}

fn path_profile_default() -> f32 {
    1.0
}

impl Default for PathParams {
    fn default() -> Self {
        Self {
            // New path layers enter viewport drawing mode; presets provide nodes explicitly.
            nodes: Vec::new(),
            width: 80.0,
            falloff: 40.0,
            noise_strength: 0.0,
            noise_scale: 0.05,
            closed: false,
            height_offset: 25.0,
            carve: false,
            seed: 1,
            spline: true,
            profile: path_profile_default(),
        }
    }
}

// —— WC-style Shape Layer params ————————————————————————————————————

/// Generator picker for [`LayerKind::ProceduralShape`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProceduralGenerator {
    #[default]
    Mountain,
    Hills,
    Plateau,
    Mesa,
    Volcano,
    Dunes,
    Canyon,
    Crater,
    Noise,
}

impl ProceduralGenerator {
    pub const ALL: &'static [ProceduralGenerator] = &[
        ProceduralGenerator::Mountain,
        ProceduralGenerator::Hills,
        ProceduralGenerator::Plateau,
        ProceduralGenerator::Mesa,
        ProceduralGenerator::Volcano,
        ProceduralGenerator::Dunes,
        ProceduralGenerator::Canyon,
        ProceduralGenerator::Crater,
        ProceduralGenerator::Noise,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Mountain => "Mountain",
            Self::Hills => "Hills",
            Self::Plateau => "Plateau",
            Self::Mesa => "Mesa",
            Self::Volcano => "Volcano",
            Self::Dunes => "Dunes",
            Self::Canyon => "Canyon",
            Self::Crater => "Crater",
            Self::Noise => "Noise",
        }
    }

    pub fn cycle(self) -> Self {
        let idx = Self::ALL.iter().position(|&k| k == self).unwrap_or(0);
        Self::ALL[(idx + 1) % Self::ALL.len()]
    }
}

/// Procedural landscape shape â€” one layer type, many generators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralShapeParams {
    pub generator: ProceduralGenerator,
    pub mountain: MountainParams,
    pub hills: FbmParams,
    pub plateau: PlateauParams,
    pub mesa: MesaParams,
    pub volcano: VolcanoParams,
    pub dunes: DuneParams,
    pub canyon: CanyonParams,
    pub crater: EffectFilterParams,
    pub noise: NoiseParams,
}

impl Default for ProceduralShapeParams {
    fn default() -> Self {
        Self {
            generator: ProceduralGenerator::Mountain,
            mountain: MountainParams::default(),
            hills: FbmParams {
                base: NoiseParams {
                    amplitude: 40.0,
                    frequency: 0.008,
                    octaves: 5,
                    ..NoiseParams::default()
                },
                ..FbmParams::default()
            },
            plateau: PlateauParams::default(),
            mesa: MesaParams::default(),
            volcano: VolcanoParams::default(),
            dunes: DuneParams::default(),
            canyon: CanyonParams::default(),
            crater: EffectFilterParams::crater(),
            noise: NoiseParams {
                amplitude: 30.0,
                frequency: 0.01,
                octaves: 4,
                ..NoiseParams::default()
            },
        }
    }
}

impl ProceduralShapeParams {
    pub fn with_generator(generator: ProceduralGenerator) -> Self {
        Self {
            generator,
            ..Self::default()
        }
    }
}

/// 2D heightmap stamp (positioned via layer area / shape transform).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Stamp2dParams {
    pub heightmap: ImportHeightmapParams,
}

/// 3D mesh / image stamp projected onto the heightfield (OBJ or heightmap path).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stamp3dParams {
    pub path: String,
    pub height_scale: f32,
    pub height_offset: f32,
}

impl Default for Stamp3dParams {
    fn default() -> Self {
        Self {
            path: String::new(),
            height_scale: 40.0,
            height_offset: 0.0,
        }
    }
}

/// Closed polygon raise / carve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolygonHeightMode {
    /// Add a signed elevation delta to the existing terrain.
    RaiseBy,
    /// Blend toward an absolute world elevation.
    SetElevation,
}

impl Default for PolygonHeightMode {
    fn default() -> Self {
        Self::RaiseBy
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolygonHeightParams {
    /// Normalized UV vertices (0â€“1). Need â‰¥ 3 for a fill.
    pub points: Vec<[f32; 2]>,
    /// Absolute target height (meters) when raising; carve depth when `carve`.
    pub height: f32,
    /// Soft edge width as fraction of the shorter world axis.
    pub falloff: f32,
    /// When true, lower terrain inside the polygon instead of raising.
    pub carve: bool,
    /// Relative displacement is predictable on existing relief; absolute mode is useful for pads.
    #[serde(default)]
    pub mode: PolygonHeightMode,
}

impl Default for PolygonHeightParams {
    fn default() -> Self {
        Self {
            // New polygon layers enter viewport drawing mode; no invisible canned square.
            points: Vec::new(),
            height: 40.0,
            falloff: 0.04,
            carve: false,
            mode: PolygonHeightMode::RaiseBy,
        }
    }
}
