//! Layer parameter kinds (split by family).

use serde::{Deserialize, Serialize};

pub use crate::noise::{FractalNoiseType, NoiseParams, WorleyFeature, WorleyMetric, WorleyParams};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FbmParams {
    pub base: NoiseParams,
    pub noise: FractalNoiseType,
}

impl Default for FbmParams {
    fn default() -> Self {
        Self {
            base: NoiseParams {
                octaves: 6,
                ..NoiseParams::default()
            },
            noise: FractalNoiseType::Perlin,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainWarpParams {
    pub base: NoiseParams,
    pub warp_strength: f32,
    pub warp_frequency: f32,
}

impl Default for DomainWarpParams {
    fn default() -> Self {
        Self {
            base: NoiseParams {
                octaves: 5,
                ..NoiseParams::default()
            },
            warp_strength: 80.0,
            warp_frequency: 0.0015,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerraceParams {
    pub levels: u32,
    pub sharpness: f32,
}

impl Default for TerraceParams {
    fn default() -> Self {
        Self {
            levels: 8,
            sharpness: 0.85,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlateauParams {
    pub low: f32,
    pub high: f32,
    pub soft: f32,
}

impl Default for PlateauParams {
    fn default() -> Self {
        Self {
            low: 40.0,
            high: 120.0,
            soft: 8.0,
        }
    }
}

/// Flat-topped mesa / butte: hard cap, steep walls, soft talus skirt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MesaParams {
    /// Normalized UV center (0â€“1).
    pub center_u: f32,
    pub center_v: f32,
    /// Outer radius as a fraction of the shorter world axis (0â€“1).
    pub radius: f32,
    /// Cap height above surroundings (meters).
    pub height: f32,
    /// Soft noise modulation on the flat cap (meters).
    pub cap_noise: f32,
    /// Wall steepness (>1 = steeper cliffs).
    pub edge_steepness: f32,
    /// Talus skirt as a fraction of radius beyond the cliff edge.
    pub soft: f32,
    pub seed: u64,
}

impl Default for MesaParams {
    fn default() -> Self {
        Self {
            center_u: 0.5,
            center_v: 0.5,
            radius: 0.22,
            height: 280.0,
            cap_noise: 6.0,
            edge_steepness: 3.2,
            soft: 0.18,
            seed: 11,
        }
    }
}

impl MesaParams {
    /// Smaller footprint, taller walls â€” isolated butte.
    pub fn butte() -> Self {
        Self {
            radius: 0.10,
            height: 340.0,
            cap_noise: 4.0,
            edge_steepness: 4.5,
            soft: 0.28,
            ..Self::default()
        }
    }
}

/// Geological archetype used by the procedural island landscape generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IslandArchetype {
    /// One asymmetric volcanic massif surrounded by a shelf and reef apron.
    #[default]
    VolcanicHighIsland,
    /// Several overlapping eroded volcanic lobes.
    Archipelago,
    /// Low coral ring surrounding a submerged lagoon.
    Atoll,
}

/// Coast-aware island landscape parameters.
///
/// Unlike an unrestricted noise layer, this shape guarantees a submerged outer
/// boundary and produces meaningful coastal zones for later erosion, materials,
/// biomes and ocean rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IslandParams {
    pub seed: u64,
    pub archetype: IslandArchetype,
    /// Normalized placement and rotation of the island footprint.
    pub center_u: f32,
    pub center_v: f32,
    pub rotation_deg: f32,
    /// Radius relative to half the shorter world axis and X/Z anisotropy.
    pub radius: f32,
    pub aspect: f32,
    /// Absolute world heights in metres.
    pub sea_level: f32,
    pub ocean_floor: f32,
    pub mountain_height: f32,
    /// Signed-distance profile zones in metres.
    pub shelf_width: f32,
    pub shelf_depth: f32,
    pub beach_width: f32,
    pub beach_height: f32,
    pub reef_width: f32,
    pub reef_depth: f32,
    /// Low-frequency coastline deformation without losing the closed landmass.
    pub coastline_warp: f32,
    pub coastline_frequency: f32,
    /// Upland profile and asymmetric ridge detail.
    pub mountain_power: f32,
    pub ridge_strength: f32,
    pub ridge_frequency: f32,
    /// Atoll lagoon radius as a fraction of the island radius.
    pub lagoon_radius: f32,
}

impl Default for IslandParams {
    fn default() -> Self {
        Self {
            seed: 63,
            archetype: IslandArchetype::VolcanicHighIsland,
            center_u: 0.5,
            center_v: 0.5,
            rotation_deg: 18.0,
            radius: 0.72,
            aspect: 1.18,
            sea_level: 0.0,
            ocean_floor: -180.0,
            mountain_height: 520.0,
            shelf_width: 260.0,
            shelf_depth: 42.0,
            beach_width: 70.0,
            beach_height: 8.0,
            reef_width: 150.0,
            reef_depth: 4.0,
            coastline_warp: 0.18,
            coastline_frequency: 0.0012,
            mountain_power: 1.55,
            ridge_strength: 0.42,
            ridge_frequency: 0.0024,
            lagoon_radius: 0.42,
        }
    }
}

impl IslandParams {
    pub fn atoll() -> Self {
        Self {
            archetype: IslandArchetype::Atoll,
            radius: 0.78,
            aspect: 1.08,
            mountain_height: 12.0,
            beach_width: 45.0,
            beach_height: 3.0,
            reef_width: 210.0,
            reef_depth: 2.5,
            lagoon_radius: 0.52,
            coastline_warp: 0.12,
            ..Self::default()
        }
    }

    pub fn archipelago() -> Self {
        Self {
            archetype: IslandArchetype::Archipelago,
            radius: 0.68,
            aspect: 1.3,
            mountain_height: 410.0,
            coastline_warp: 0.24,
            ridge_strength: 0.5,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MountainParams {
    pub base: NoiseParams,
    pub ridge_sharpness: f32,
    pub range_angle: f32,
    pub range_width: f32,
    /// High-frequency ridged detail on crests (meters). Seeds knife-edge / couloir structure.
    #[serde(default = "default_mountain_crest_detail")]
    pub crest_detail: f32,
}

fn default_mountain_crest_detail() -> f32 {
    45.0
}

impl Default for MountainParams {
    fn default() -> Self {
        Self {
            base: NoiseParams {
                octaves: 5,
                frequency: 0.0015,
                amplitude: 400.0,
                ..NoiseParams::default()
            },
            ridge_sharpness: 1.8,
            range_angle: 0.4,
            range_width: 0.35,
            crest_detail: default_mountain_crest_detail(),
        }
    }
}

/// Radial volcanic cone: smooth falloff from a center with an optional crater bowl.
///
/// Matches World Creator Landscape Volcano intent (peak, crater radius/depth) as a
/// height authoring primitive â€” not a magmatic simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolcanoParams {
    /// Normalized UV center (0â€“1).
    pub center_u: f32,
    pub center_v: f32,
    /// Outer radius as a fraction of the shorter world axis (0â€“1).
    pub radius: f32,
    /// Peak height above surroundings (meters).
    pub height: f32,
    /// Profile exponent: 1 = linear cone, >1 = steeper flanks / flatter top approach.
    pub flank_power: f32,
    /// Crater radius as a fraction of [`Self::radius`] (0 = no crater).
    pub crater_radius: f32,
    /// Crater depth below the rim (meters).
    pub crater_depth: f32,
    /// Soft noise modulation on the flanks (meters).
    pub roughness: f32,
    pub seed: u64,
}

impl Default for VolcanoParams {
    fn default() -> Self {
        Self {
            center_u: 0.5,
            center_v: 0.5,
            radius: 0.28,
            height: 520.0,
            flank_power: 1.35,
            crater_radius: 0.18,
            crater_depth: 85.0,
            roughness: 18.0,
            seed: 42,
        }
    }
}

/// Artist-directed large landforms: coherent ridge corridors before detail noise.
///
/// Conceptual influence: Schott et al. 2023 (author ridges as uplift). This is a
/// height authoring primitive, **not** a stream-power / SPE solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpliftParams {
    pub seed: u64,
    /// Primary corridor / ridge frequency (world unitsâ»Â¹).
    pub frequency: f32,
    /// Peak uplift height in meters.
    pub amplitude: f32,
    /// Corridor orientation (radians).
    pub range_angle: f32,
    /// Normalized corridor half-width âˆˆ (0,1].
    pub corridor_width: f32,
    /// Sharpens the primary ridge crest (>1 = sharper).
    pub ridge_power: f32,
    /// Domain warp strength for corridor meander (meters scale via frequency).
    pub warp_strength: f32,
    /// Secondary detail amplitude (meters); kept small so structure dominates.
    pub detail_amplitude: f32,
    pub detail_octaves: u32,
    pub detail_frequency: f32,
    /// Musgrave-inspired altitude fade of detail: 0 = uniform detail, 1 = detail
    /// only near uplift crests (valleys stay coherent for drainage).
    pub altitude_fade: f32,
}

impl Default for UpliftParams {
    fn default() -> Self {
        Self {
            seed: 42,
            frequency: 0.0009,
            amplitude: 380.0,
            range_angle: 0.55,
            corridor_width: 0.42,
            ridge_power: 1.6,
            warp_strength: 0.35,
            detail_amplitude: 45.0,
            detail_octaves: 3,
            detail_frequency: 0.004,
            altitude_fade: 0.75,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuneParams {
    pub base: NoiseParams,
    /// Legacy dune wave frequency; prefer [`Self::dune_scale`].
    pub wave_frequency: f32,
    /// Legacy crest asymmetry; prefer [`Self::crest_sharpness`].
    pub asymmetry: f32,
    /// Extra depth carved in interdune troughs (meters).
    #[serde(default)]
    pub trough_depth: f32,
    /// Soft basin floor as a fraction of dune amplitude (0–1).
    #[serde(default)]
    pub basin_floor: f32,
    /// Wind / dune orientation in degrees (0 = +X).
    #[serde(default)]
    pub direction_deg: f32,
    /// Relative wind energy for the limited transport pass \[0, 2\].
    #[serde(default = "default_dune_wind_strength")]
    pub wind_strength: f32,
    /// Sand availability for the seed slab \[0, 2\].
    #[serde(default = "default_dune_sand_supply")]
    pub sand_supply: f32,
    /// Spatial frequency of dune crests (world 1/m). 0 → use `wave_frequency`.
    #[serde(default)]
    pub dune_scale: f32,
    /// Peak dune height in meters. 0 → use `base.amplitude`.
    #[serde(default)]
    pub dune_height: f32,
    /// Slip-face / crest contrast \[0, 1\]. 0 → use `asymmetry`.
    #[serde(default)]
    pub crest_sharpness: f32,
    /// Ridge coherence: high → linear/transverse; low → crescent/star-like \[0, 1\].
    #[serde(default = "default_dune_linearity")]
    pub linearity: f32,
    /// Saltation hop length in cells for the relaxation pass.
    #[serde(default = "default_dune_transport_length")]
    pub transport_length: f32,
    /// Avalanche angle of repose (degrees).
    #[serde(default = "default_dune_avalanche_angle")]
    pub avalanche_angle: f32,
    /// Limited sand-transport / relaxation iterations (interactive filter).
    #[serde(default = "default_dune_iterations")]
    pub iterations: u32,
}

fn default_dune_wind_strength() -> f32 {
    0.85
}
fn default_dune_sand_supply() -> f32 {
    0.75
}
fn default_dune_linearity() -> f32 {
    0.72
}
fn default_dune_transport_length() -> f32 {
    4.5
}
fn default_dune_avalanche_angle() -> f32 {
    33.0
}
fn default_dune_iterations() -> u32 {
    10
}

impl DuneParams {
    /// Effective crest frequency.
    pub fn effective_scale(&self) -> f32 {
        if self.dune_scale > 1e-8 {
            self.dune_scale
        } else {
            self.wave_frequency.max(1e-5)
        }
    }

    /// Effective dune amplitude in meters.
    pub fn effective_height(&self) -> f32 {
        if self.dune_height > 1e-5 {
            self.dune_height
        } else {
            self.base.amplitude.max(0.0)
        }
    }

    /// Effective crest sharpness / slip-face contrast.
    pub fn effective_crest_sharpness(&self) -> f32 {
        if self.crest_sharpness > 1e-5 {
            self.crest_sharpness.clamp(0.0, 1.0)
        } else {
            self.asymmetry.clamp(0.0, 1.0)
        }
    }
}

impl Default for DuneParams {
    fn default() -> Self {
        Self {
            base: NoiseParams {
                octaves: 3,
                frequency: 0.003,
                amplitude: 40.0,
                ..NoiseParams::default()
            },
            wave_frequency: 0.01,
            asymmetry: 0.65,
            trough_depth: 8.0,
            basin_floor: 0.05,
            direction_deg: 0.0,
            wind_strength: default_dune_wind_strength(),
            sand_supply: default_dune_sand_supply(),
            dune_scale: 0.0,
            dune_height: 0.0,
            crest_sharpness: 0.0,
            linearity: default_dune_linearity(),
            transport_length: default_dune_transport_length(),
            avalanche_angle: default_dune_avalanche_angle(),
            iterations: default_dune_iterations(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanyonParams {
    pub depth: f32,
    pub width: f32,
    pub meander: f32,
    pub seed: u64,
}

impl Default for CanyonParams {
    fn default() -> Self {
        Self {
            depth: 180.0,
            width: 120.0,
            meander: 0.4,
            seed: 7,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoronoiParams {
    pub base: NoiseParams,
    pub cell_jitter: f32,
    pub height_per_cell: f32,
}

impl Default for VoronoiParams {
    fn default() -> Self {
        Self {
            base: NoiseParams {
                frequency: 0.003,
                amplitude: 80.0,
                ..NoiseParams::default()
            },
            cell_jitter: 0.75,
            height_per_cell: 60.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportHeightmapParams {
    pub path: String,
    pub height_scale: f32,
    pub height_offset: f32,
}

impl Default for ImportHeightmapParams {
    fn default() -> Self {
        Self {
            path: String::new(),
            height_scale: 500.0,
            height_offset: 0.0,
        }
    }
}
