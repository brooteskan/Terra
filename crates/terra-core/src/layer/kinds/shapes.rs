//! Layer parameter kinds (split by family).

use super::filters::EffectFilterParams;
use super::noise::{
    CanyonParams, DuneParams, FbmParams, ImportHeightmapParams, MesaParams, MountainParams,
    NoiseParams, PlateauParams, VolcanoParams,
};
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

/// Dual-height cliff undercut / shelf stamp (Phase J opt-in).
///
/// Carves a cavity floor into the DEM inside a UV disk while recording a ceiling aux map so
/// the viewport can draw an overhang silhouette. Layer masks further limit the region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverhangStampParams {
    /// UV center U âˆˆ \[0,1\].
    #[serde(default = "default_overhang_u")]
    pub u: f32,
    /// UV center V âˆˆ \[0,1\].
    #[serde(default = "default_overhang_v")]
    pub v: f32,
    /// Stamp radius in UV space.
    #[serde(default = "default_overhang_radius")]
    pub radius_uv: f32,
    /// Maximum undercut depth in meters (floor drop).
    #[serde(default = "default_overhang_depth")]
    pub depth: f32,
    /// Optional brow / lip raised on the back rim (meters).
    #[serde(default = "default_overhang_lip")]
    pub lip_height: f32,
    /// Azimuth of the open entrance in degrees (0 = +X / east).
    #[serde(default = "default_overhang_entrance")]
    pub entrance_angle_deg: f32,
    /// Soft edge width as a fraction of radius (smaller = harder edge).
    #[serde(default = "default_overhang_falloff")]
    pub falloff: f32,
    #[serde(default = "default_overhang_seed")]
    pub seed: u64,
    /// Wall irregularity \[0,1\].
    #[serde(default = "default_overhang_noise")]
    pub noise_amplitude: f32,
}

fn default_overhang_u() -> f32 {
    0.5
}
fn default_overhang_v() -> f32 {
    0.5
}
fn default_overhang_radius() -> f32 {
    0.08
}
fn default_overhang_depth() -> f32 {
    18.0
}
fn default_overhang_lip() -> f32 {
    2.0
}
fn default_overhang_entrance() -> f32 {
    180.0
}
fn default_overhang_falloff() -> f32 {
    0.35
}
fn default_overhang_seed() -> u64 {
    11
}
fn default_overhang_noise() -> f32 {
    0.25
}

impl Default for OverhangStampParams {
    fn default() -> Self {
        Self {
            u: default_overhang_u(),
            v: default_overhang_v(),
            radius_uv: default_overhang_radius(),
            depth: default_overhang_depth(),
            lip_height: default_overhang_lip(),
            entrance_angle_deg: default_overhang_entrance(),
            falloff: default_overhang_falloff(),
            seed: default_overhang_seed(),
            noise_amplitude: default_overhang_noise(),
        }
    }
}

impl OverhangStampParams {
    /// Preset aimed at a mid-world cliff face opening westward.
    pub fn cliff_overhang() -> Self {
        Self {
            u: 0.52,
            v: 0.5,
            radius_uv: 0.1,
            depth: 22.0,
            lip_height: 3.0,
            entrance_angle_deg: 180.0,
            falloff: 0.3,
            seed: 19,
            noise_amplitude: 0.3,
        }
    }
}

/// Local analytic SDF cave pocket (Phase J opt-in).
///
/// Evaluates a small ellipsoid + entrance tunnel SDF and projects void intervals onto
/// dual-height (carved floor DEM + ceiling aux). Not a full volumetric world store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalSdfParams {
    #[serde(default = "default_sdf_u")]
    pub u: f32,
    #[serde(default = "default_sdf_v")]
    pub v: f32,
    /// Chamber width (X) in meters.
    #[serde(default = "default_sdf_size_x")]
    pub size_x: f32,
    /// Chamber height (Y) in meters.
    #[serde(default = "default_sdf_size_y")]
    pub size_y: f32,
    /// Chamber depth (Z) in meters.
    #[serde(default = "default_sdf_size_z")]
    pub size_z: f32,
    /// How far below the local surface the chamber sits (meters).
    #[serde(default = "default_sdf_depth")]
    pub depth: f32,
    /// Entrance tunnel radius (meters).
    #[serde(default = "default_sdf_entrance_r")]
    pub entrance_radius: f32,
    /// Entrance azimuth in degrees (0 = +X).
    #[serde(default = "default_sdf_entrance_ang")]
    pub entrance_angle_deg: f32,
    #[serde(default)]
    pub lip_height: f32,
    #[serde(default = "default_sdf_seed")]
    pub seed: u64,
    #[serde(default = "default_sdf_noise")]
    pub noise_amplitude: f32,
    /// Vertical SDF samples per column (CPU oracle quality).
    #[serde(default = "default_sdf_samples")]
    pub vertical_samples: u32,
}

fn default_sdf_u() -> f32 {
    0.55
}
fn default_sdf_v() -> f32 {
    0.5
}
fn default_sdf_size_x() -> f32 {
    28.0
}
fn default_sdf_size_y() -> f32 {
    14.0
}
fn default_sdf_size_z() -> f32 {
    22.0
}
fn default_sdf_depth() -> f32 {
    20.0
}
fn default_sdf_entrance_r() -> f32 {
    5.0
}
fn default_sdf_entrance_ang() -> f32 {
    180.0
}
fn default_sdf_seed() -> u64 {
    23
}
fn default_sdf_noise() -> f32 {
    0.2
}
fn default_sdf_samples() -> u32 {
    24
}

impl Default for LocalSdfParams {
    fn default() -> Self {
        Self {
            u: default_sdf_u(),
            v: default_sdf_v(),
            size_x: default_sdf_size_x(),
            size_y: default_sdf_size_y(),
            size_z: default_sdf_size_z(),
            depth: default_sdf_depth(),
            entrance_radius: default_sdf_entrance_r(),
            entrance_angle_deg: default_sdf_entrance_ang(),
            lip_height: 0.5,
            seed: default_sdf_seed(),
            noise_amplitude: default_sdf_noise(),
            vertical_samples: default_sdf_samples(),
        }
    }
}

impl LocalSdfParams {
    /// Compact karst / pocket cave preset.
    pub fn karst_pocket() -> Self {
        Self {
            u: 0.58,
            v: 0.48,
            size_x: 24.0,
            size_y: 12.0,
            size_z: 18.0,
            depth: 18.0,
            entrance_radius: 4.5,
            entrance_angle_deg: 200.0,
            lip_height: 1.0,
            seed: 31,
            noise_amplitude: 0.28,
            vertical_samples: 28,
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
