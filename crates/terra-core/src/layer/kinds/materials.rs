//! Layer parameter kinds (split by family).

use crate::mask::MaskSource;
use serde::{Deserialize, Serialize};

/// JSON-safe lower sentinel for an effectively open material/biome height range.
pub const OPEN_HEIGHT_MIN: f32 = -1_000_000.0;
/// JSON-safe upper sentinel for an effectively open material/biome height range.
pub const OPEN_HEIGHT_MAX: f32 = 1_000_000.0;

fn deserialize_open_height_min<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<f32>::deserialize(deserializer)?.unwrap_or(OPEN_HEIGHT_MIN))
}

fn deserialize_open_height_max<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<f32>::deserialize(deserializer)?.unwrap_or(OPEN_HEIGHT_MAX))
}

/// Lithology class for a geological stratum — drives thermal stability defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StratumMaterial {
    #[default]
    Sedimentary,
    Igneous,
    Metamorphic,
    Unconsolidated,
    Soil,
    Ice,
}

impl StratumMaterial {
    /// Material stability ∈ \[0,1\] for thermal / mass-wasting (1 = holds steep faces).
    pub fn stability(self) -> f32 {
        match self {
            StratumMaterial::Igneous => 0.92,
            StratumMaterial::Metamorphic => 0.88,
            StratumMaterial::Sedimentary => 0.55,
            StratumMaterial::Unconsolidated => 0.18,
            StratumMaterial::Soil => 0.22,
            StratumMaterial::Ice => 0.12,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            StratumMaterial::Sedimentary => "Sedimentary",
            StratumMaterial::Igneous => "Igneous",
            StratumMaterial::Metamorphic => "Metamorphic",
            StratumMaterial::Unconsolidated => "Unconsolidated",
            StratumMaterial::Soil => "Soil",
            StratumMaterial::Ice => "Ice",
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            StratumMaterial::Sedimentary => StratumMaterial::Igneous,
            StratumMaterial::Igneous => StratumMaterial::Metamorphic,
            StratumMaterial::Metamorphic => StratumMaterial::Unconsolidated,
            StratumMaterial::Unconsolidated => StratumMaterial::Soil,
            StratumMaterial::Soil => StratumMaterial::Ice,
            StratumMaterial::Ice => StratumMaterial::Sedimentary,
        }
    }
}

/// Attitude of the geological bed stack (independent of the free surface).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BedGeometry {
    #[default]
    Horizontal,
    /// Planar dip. `dip_deg` from horizontal; `azimuth_deg` is dip direction.
    Tilted { dip_deg: f32, azimuth_deg: f32 },
    /// Sinusoidal fold + mild coherent noise.
    Folded {
        amplitude_m: f32,
        wavelength_m: f32,
        seed: u64,
    },
    /// Low-frequency coherent warp of stratigraphic depth.
    Warped {
        frequency: f32,
        amplitude_m: f32,
        seed: u64,
    },
}

impl BedGeometry {
    /// Stratigraphic depth warp (meters) at world (x, z). Added to topographic depth.
    pub fn depth_warp(self, x: f32, z: f32) -> f32 {
        match self {
            BedGeometry::Horizontal => 0.0,
            BedGeometry::Tilted {
                dip_deg,
                azimuth_deg,
            } => {
                let dip = dip_deg.to_radians();
                let az = azimuth_deg.to_radians();
                (x * az.cos() + z * az.sin()) * dip.sin()
            }
            BedGeometry::Folded {
                amplitude_m,
                wavelength_m,
                seed,
            } => {
                let wl = wavelength_m.max(1.0);
                let tau = std::f32::consts::TAU;
                let fold = (x / wl * tau).sin() * amplitude_m;
                let n = crate::noise::sample_noise(
                    crate::layer::FractalNoiseType::Perlin,
                    x / wl,
                    z / wl,
                    seed,
                );
                fold + n * amplitude_m * 0.35
            }
            BedGeometry::Warped {
                frequency,
                amplitude_m,
                seed,
            } => {
                let f = frequency.max(1e-5);
                crate::noise::sample_noise(
                    crate::layer::FractalNoiseType::Perlin,
                    x * f,
                    z * f,
                    seed,
                ) * amplitude_m
            }
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            BedGeometry::Horizontal => "Horizontal",
            BedGeometry::Tilted { .. } => "Tilted",
            BedGeometry::Folded { .. } => "Folded",
            BedGeometry::Warped { .. } => "Warped",
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            BedGeometry::Horizontal => BedGeometry::Tilted {
                dip_deg: 12.0,
                azimuth_deg: 45.0,
            },
            BedGeometry::Tilted { .. } => BedGeometry::Folded {
                amplitude_m: 25.0,
                wavelength_m: 180.0,
                seed: 11,
            },
            BedGeometry::Folded { .. } => BedGeometry::Warped {
                frequency: 0.012,
                amplitude_m: 22.0,
                seed: 11,
            },
            BedGeometry::Warped { .. } => BedGeometry::Horizontal,
        }
    }
}

/// One layer in a practical (non-volumetric) material stack.
///
/// Strata are stacked from the **surface downward**. Soft-over-hard differential
/// erosion peels the soft cap first; hardness / erodibility at depth \(d\) walk
/// the stack (Šťava-inspired multi-material terrain, simplified to scalar fields).

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stratum {
    pub name: String,
    /// Surface material ID (same encoding as [`MaterialRule::id`], /16 in aux maps).
    pub id: u32,
    /// Bedrock hardness \(K \in [0,1]\).
    #[serde(default = "default_material_hardness")]
    pub hardness: f32,
    /// Thickness in meters. Use a large value for the basal stratum.
    #[serde(default = "default_stratum_thickness")]
    pub thickness: f32,
    /// Hydraulic erodibility ∈ \[0,1\]. Defaults to `1 - hardness`.
    #[serde(default = "default_stratum_erodibility_sentinel")]
    pub erodibility: f32,
    /// Lithology class (thermal stability + rock-filter reuse).
    #[serde(default)]
    pub material_type: StratumMaterial,
}

fn default_stratum_thickness() -> f32 {
    1.0e6
}

/// Sentinel: negative means "derive from hardness" on first read.
fn default_stratum_erodibility_sentinel() -> f32 {
    -1.0
}

impl Stratum {
    pub fn soft_cap(thickness: f32) -> Self {
        Self {
            name: "Soft Cap".into(),
            id: 3,
            hardness: 0.08,
            thickness,
            erodibility: 0.92,
            material_type: StratumMaterial::Unconsolidated,
        }
    }

    pub fn hard_base() -> Self {
        Self {
            name: "Hard Base".into(),
            id: 1,
            hardness: 0.92,
            thickness: default_stratum_thickness(),
            erodibility: 0.08,
            material_type: StratumMaterial::Igneous,
        }
    }

    /// Effective hydraulic erodibility (legacy strata without the field → `1 - K`).
    pub fn effective_erodibility(&self) -> f32 {
        if self.erodibility < 0.0 {
            (1.0 - self.hardness).clamp(0.0, 1.0)
        } else {
            self.erodibility.clamp(0.0, 1.0)
        }
    }

    /// Thermal / cliff stability ∈ \[0,1\] from lithology × hardness.
    pub fn material_stability(&self) -> f32 {
        let base = self.material_type.stability();
        (base * (0.35 + 0.65 * self.hardness.clamp(0.0, 1.0))).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialsParams {
    /// Slope/height/mask classification rules (surface IDs + per-rule hardness).
    pub rules: Vec<MaterialRule>,
    /// Optional vertical stack from surface → bedrock. When non-empty, drives
    /// depth-aware hardness for soft-over-hard stripping under erosion.
    #[serde(default)]
    pub strata: Vec<Stratum>,
    /// Fallback \(K\) when no rule/stratum matches.
    #[serde(default = "default_material_hardness")]
    pub default_hardness: f32,
    /// Optional nested coverage distribution (WC-style where materials land).
    #[serde(default)]
    pub coverage: crate::mask::Distribution,
    /// Bed attitude for the stratum stack (horizontal / tilted / folded / warped).
    #[serde(default)]
    pub bed_geometry: BedGeometry,
}

impl Default for MaterialsParams {
    fn default() -> Self {
        Self {
            rules: vec![
                MaterialRule {
                    name: "Rock".into(),
                    id: 1,
                    min_slope_deg: 35.0,
                    max_slope_deg: 90.0,
                    min_height: OPEN_HEIGHT_MIN,
                    max_height: OPEN_HEIGHT_MAX,
                    mask: MaskSource::None,
                    hardness: 0.85,
                    tint: [0.45, 0.42, 0.38],
                    roughness: 0.85,
                    metalness: 0.05,
                    albedo_path: None,
                },
                MaterialRule {
                    name: "Grass".into(),
                    id: 2,
                    min_slope_deg: 0.0,
                    max_slope_deg: 35.0,
                    min_height: 5.0,
                    max_height: OPEN_HEIGHT_MAX,
                    mask: MaskSource::None,
                    hardness: 0.2,
                    tint: [0.28, 0.48, 0.22],
                    roughness: 0.9,
                    metalness: 0.0,
                    albedo_path: None,
                },
            ],
            strata: Vec::new(),
            default_hardness: 0.5,
            coverage: crate::mask::Distribution::new(),
            bed_geometry: BedGeometry::Horizontal,
        }
    }
}

impl MaterialsParams {
    /// Soft sediment cap over hard rock — differential erosion preset helper.
    pub fn soft_over_hard(cap_thickness: f32) -> Self {
        Self {
            rules: Vec::new(),
            strata: vec![Stratum::soft_cap(cap_thickness), Stratum::hard_base()],
            default_hardness: 0.5,
            coverage: crate::mask::Distribution::new(),
            bed_geometry: BedGeometry::Horizontal,
        }
    }

    /// Alpine peak materials: hard steep rock, softer mid-slope scree, high snow band.
    ///
    /// Matches reference mountain looks (bare knife faces + snow on ledges / couloirs)
    /// when paired with climate biomes and hardness-aware erosion.
    pub fn alpine_peak() -> Self {
        Self {
            rules: vec![
                MaterialRule {
                    name: "Cliff Rock".into(),
                    id: 1,
                    min_slope_deg: 42.0,
                    max_slope_deg: 90.0,
                    min_height: OPEN_HEIGHT_MIN,
                    max_height: OPEN_HEIGHT_MAX,
                    mask: MaskSource::None,
                    hardness: 0.92,
                    tint: [0.31, 0.29, 0.27],
                    roughness: 0.78,
                    metalness: 0.0,
                    albedo_path: None,
                },
                MaterialRule {
                    name: "Snow".into(),
                    id: 4,
                    min_slope_deg: 0.0,
                    max_slope_deg: 48.0,
                    min_height: 220.0,
                    max_height: OPEN_HEIGHT_MAX,
                    mask: MaskSource::None,
                    hardness: 0.12,
                    tint: [0.86, 0.89, 0.93],
                    roughness: 0.36,
                    metalness: 0.0,
                    albedo_path: None,
                },
                MaterialRule {
                    name: "Scree".into(),
                    id: 3,
                    min_slope_deg: 18.0,
                    max_slope_deg: 42.0,
                    min_height: OPEN_HEIGHT_MIN,
                    max_height: OPEN_HEIGHT_MAX,
                    mask: MaskSource::None,
                    hardness: 0.28,
                    tint: [0.43, 0.38, 0.32],
                    roughness: 0.94,
                    metalness: 0.0,
                    albedo_path: None,
                },
                MaterialRule {
                    name: "Alpine Meadow".into(),
                    id: 2,
                    min_slope_deg: 0.0,
                    max_slope_deg: 22.0,
                    min_height: 5.0,
                    max_height: 220.0,
                    mask: MaskSource::None,
                    hardness: 0.18,
                    tint: [0.16, 0.31, 0.12],
                    roughness: 0.91,
                    metalness: 0.0,
                    albedo_path: None,
                },
            ],
            strata: vec![
                Stratum::soft_cap(10.0),
                Stratum {
                    name: "Hard Peak".into(),
                    id: 1,
                    hardness: 0.94,
                    thickness: default_stratum_thickness(),
                    erodibility: 0.06,
                    material_type: StratumMaterial::Igneous,
                },
            ],
            default_hardness: 0.45,
            coverage: crate::mask::Distribution::new(),
            bed_geometry: BedGeometry::Horizontal,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialRule {
    pub name: String,
    pub id: u32,
    pub min_slope_deg: f32,
    pub max_slope_deg: f32,
    #[serde(deserialize_with = "deserialize_open_height_min")]
    pub min_height: f32,
    #[serde(deserialize_with = "deserialize_open_height_max")]
    pub max_height: f32,
    /// Optional painted / procedural mask; cells above 0.5 force this rule's ID.
    pub mask: MaskSource,
    /// Bedrock hardness K âˆˆ \[0,1\] used when baking materials â†’ hardness.
    #[serde(default = "default_material_hardness")]
    pub hardness: f32,
    /// Viewport / export albedo tint (linear RGB).
    #[serde(default = "default_material_tint")]
    pub tint: [f32; 3],
    #[serde(default = "default_material_roughness")]
    pub roughness: f32,
    #[serde(default)]
    pub metalness: f32,
    /// Optional path to an albedo texture (PNG); empty = tint only.
    #[serde(default)]
    pub albedo_path: Option<String>,
}

fn default_material_hardness() -> f32 {
    0.5
}

fn default_material_tint() -> [f32; 3] {
    [0.45, 0.42, 0.38]
}

fn default_material_roughness() -> f32 {
    0.75
}

/// Artist climate controls for biome classification (Phase H).
///
/// Values are normalized artist knobs unless noted (temperatures ≈ [0,1] warm↔cold
/// scale, precip [0,1], heights in meters).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateParams {
    /// Base temperature at sea level (warm â‰ˆ 1).
    #[serde(default = "default_sea_level_temp")]
    pub sea_level_temp: f32,
    /// Temperature drop per meter of elevation.
    #[serde(default = "default_lapse_rate")]
    pub lapse_rate: f32,
    /// Absolute latitude bias (0 equator â†’ 1 polar cool).
    #[serde(default = "default_latitude")]
    pub latitude: f32,
    /// Northâ€“south temperature gradient strength along Z.
    #[serde(default = "default_temp_gradient")]
    pub temp_gradient: f32,
    /// Prevailing wind direction in degrees (0 = +Z / north, 90 = +X / east).
    #[serde(default = "default_wind_dir")]
    pub wind_dir_deg: f32,
    /// Base precipitation scale \[0,1\].
    #[serde(default = "default_base_precip")]
    pub base_precip: f32,
    /// Ambient humidity without ocean/wetness \[0,1\].
    #[serde(default = "default_base_humidity")]
    pub base_humidity: f32,
    /// Windward orographic rainfall boost.
    #[serde(default = "default_orographic")]
    pub orographic_strength: f32,
    /// Leeward rain-shadow dryness.
    #[serde(default = "default_rain_shadow")]
    pub rain_shadow_strength: f32,
    /// How strongly existing wetness aux feeds moisture.
    #[serde(default = "default_moisture_wetness")]
    pub moisture_from_wetness: f32,
    /// Sea / water elevation (meters) for moisture proximity.
    #[serde(default = "default_climate_sea_level")]
    pub sea_level: f32,
    /// Distance scale (meters) for ocean moisture falloff.
    #[serde(default = "default_water_influence")]
    pub water_influence: f32,
    /// Temperature below which snow accumulates (normalized).
    #[serde(default = "default_snow_temp")]
    pub snow_temp: f32,
    /// Elevation (m) above which snow line strengthens.
    #[serde(default = "default_snow_line")]
    pub snow_line_height: f32,
}

fn default_sea_level_temp() -> f32 {
    0.72
}
fn default_lapse_rate() -> f32 {
    0.0012
}
fn default_latitude() -> f32 {
    0.35
}
fn default_temp_gradient() -> f32 {
    0.18
}
fn default_wind_dir() -> f32 {
    90.0
}
fn default_base_precip() -> f32 {
    0.55
}
fn default_base_humidity() -> f32 {
    0.4
}
fn default_orographic() -> f32 {
    0.85
}
fn default_rain_shadow() -> f32 {
    0.7
}
fn default_moisture_wetness() -> f32 {
    0.35
}
fn default_climate_sea_level() -> f32 {
    15.0
}
fn default_water_influence() -> f32 {
    120.0
}
fn default_snow_temp() -> f32 {
    0.28
}
fn default_snow_line() -> f32 {
    180.0
}

impl Default for ClimateParams {
    fn default() -> Self {
        Self {
            sea_level_temp: default_sea_level_temp(),
            lapse_rate: default_lapse_rate(),
            latitude: default_latitude(),
            temp_gradient: default_temp_gradient(),
            wind_dir_deg: default_wind_dir(),
            base_precip: default_base_precip(),
            base_humidity: default_base_humidity(),
            orographic_strength: default_orographic(),
            rain_shadow_strength: default_rain_shadow(),
            moisture_from_wetness: default_moisture_wetness(),
            sea_level: default_climate_sea_level(),
            water_influence: default_water_influence(),
            snow_temp: default_snow_temp(),
            snow_line_height: default_snow_line(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomesParams {
    pub bands: Vec<BiomeBand>,
    /// When true, classify from climate fields (temp/precip/snow/soil).
    #[serde(default = "default_use_climate")]
    pub use_climate: bool,
    #[serde(default)]
    pub climate: ClimateParams,
}

fn default_use_climate() -> bool {
    true
}

impl Default for BiomesParams {
    fn default() -> Self {
        Self {
            bands: crate::climate::default_climate_bands(),
            use_climate: true,
            climate: ClimateParams::default(),
        }
    }
}

impl BiomesParams {
    /// Legacy height/wetness-only bands (preâ€“Phase H).
    pub fn height_bands() -> Self {
        Self {
            use_climate: false,
            climate: ClimateParams::default(),
            bands: vec![
                BiomeBand {
                    name: "Alpine".into(),
                    id: 1,
                    min_height: 200.0,
                    max_height: OPEN_HEIGHT_MAX,
                    min_wetness: 0.0,
                    max_wetness: 1.0,
                    ..BiomeBand::all_climate()
                },
                BiomeBand {
                    name: "Temperate".into(),
                    id: 2,
                    min_height: 20.0,
                    max_height: 200.0,
                    min_wetness: 0.0,
                    max_wetness: 1.0,
                    ..BiomeBand::all_climate()
                },
                BiomeBand {
                    name: "Coast".into(),
                    id: 3,
                    min_height: OPEN_HEIGHT_MIN,
                    max_height: 20.0,
                    min_wetness: 0.0,
                    max_wetness: 1.0,
                    ..BiomeBand::all_climate()
                },
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomeBand {
    pub name: String,
    pub id: u32,
    #[serde(deserialize_with = "deserialize_open_height_min")]
    pub min_height: f32,
    #[serde(deserialize_with = "deserialize_open_height_max")]
    pub max_height: f32,
    pub min_wetness: f32,
    pub max_wetness: f32,
    #[serde(default = "default_band_min_temp")]
    pub min_temp: f32,
    #[serde(default = "default_band_max_temp")]
    pub max_temp: f32,
    #[serde(default)]
    pub min_precip: f32,
    #[serde(default = "default_band_max_one")]
    pub max_precip: f32,
    #[serde(default)]
    pub min_snow: f32,
    #[serde(default = "default_band_max_one")]
    pub max_snow: f32,
    #[serde(default)]
    pub min_soil_moisture: f32,
    #[serde(default = "default_band_max_one")]
    pub max_soil_moisture: f32,
}

fn default_band_min_temp() -> f32 {
    0.0
}
fn default_band_max_temp() -> f32 {
    1.0
}
fn default_band_max_one() -> f32 {
    1.0
}

impl BiomeBand {
    /// Climate ranges that accept any value (legacy height/wetness filters only).
    pub fn all_climate() -> Self {
        Self {
            name: String::new(),
            id: 0,
            min_height: OPEN_HEIGHT_MIN,
            max_height: OPEN_HEIGHT_MAX,
            min_wetness: 0.0,
            max_wetness: 1.0,
            min_temp: 0.0,
            max_temp: 1.0,
            min_precip: 0.0,
            max_precip: 1.0,
            min_snow: 0.0,
            max_snow: 1.0,
            min_soil_moisture: 0.0,
            max_soil_moisture: 1.0,
        }
    }
}
