//! Layer parameter kinds (split by family).

pub use crate::geology::{BedGeometry, Stratum, StratumMaterial};
use crate::mask_types::MaskSource;
use serde::{Deserialize, Serialize};

/// JSON-safe lower sentinel for an effectively open material/biome height range.
pub const OPEN_HEIGHT_MIN: f32 = -1_000_000.0;
/// JSON-safe upper sentinel for an effectively open material/biome height range.
pub const OPEN_HEIGHT_MAX: f32 = 1_000_000.0;

/// Stable default biome classification bands used by persisted biome parameters.
pub fn default_climate_bands() -> Vec<BiomeBand> {
    vec![
        BiomeBand {
            name: "Desert".into(),
            id: 1,
            min_height: OPEN_HEIGHT_MIN,
            max_height: OPEN_HEIGHT_MAX,
            min_wetness: 0.0,
            max_wetness: 1.0,
            min_temp: 0.45,
            max_temp: 1.0,
            min_precip: 0.0,
            max_precip: 0.28,
            min_snow: 0.0,
            max_snow: 0.15,
            min_soil_moisture: 0.0,
            max_soil_moisture: 0.35,
        },
        BiomeBand {
            name: "Grassland".into(),
            id: 2,
            min_height: OPEN_HEIGHT_MIN,
            max_height: OPEN_HEIGHT_MAX,
            min_wetness: 0.0,
            max_wetness: 1.0,
            min_temp: 0.35,
            max_temp: 0.85,
            min_precip: 0.22,
            max_precip: 0.55,
            min_snow: 0.0,
            max_snow: 0.25,
            min_soil_moisture: 0.0,
            max_soil_moisture: 1.0,
        },
        BiomeBand {
            name: "Temperate Forest".into(),
            id: 3,
            min_height: OPEN_HEIGHT_MIN,
            max_height: OPEN_HEIGHT_MAX,
            min_wetness: 0.0,
            max_wetness: 1.0,
            min_temp: 0.28,
            max_temp: 0.75,
            min_precip: 0.45,
            max_precip: 1.0,
            min_snow: 0.0,
            max_snow: 0.35,
            min_soil_moisture: 0.15,
            max_soil_moisture: 1.0,
        },
        BiomeBand {
            name: "Wetland".into(),
            id: 4,
            min_height: OPEN_HEIGHT_MIN,
            max_height: OPEN_HEIGHT_MAX,
            min_wetness: 0.0,
            max_wetness: 1.0,
            min_temp: 0.2,
            max_temp: 0.9,
            min_precip: 0.35,
            max_precip: 1.0,
            min_snow: 0.0,
            max_snow: 0.2,
            min_soil_moisture: 0.65,
            max_soil_moisture: 1.0,
        },
        BiomeBand {
            name: "Boreal".into(),
            id: 5,
            min_height: OPEN_HEIGHT_MIN,
            max_height: OPEN_HEIGHT_MAX,
            min_wetness: 0.0,
            max_wetness: 1.0,
            min_temp: 0.12,
            max_temp: 0.4,
            min_precip: 0.25,
            max_precip: 1.0,
            min_snow: 0.0,
            max_snow: 0.7,
            min_soil_moisture: 0.0,
            max_soil_moisture: 1.0,
        },
        BiomeBand {
            name: "Alpine".into(),
            id: 6,
            min_height: OPEN_HEIGHT_MIN,
            max_height: OPEN_HEIGHT_MAX,
            min_wetness: 0.0,
            max_wetness: 1.0,
            min_temp: 0.0,
            max_temp: 0.28,
            min_precip: 0.0,
            max_precip: 1.0,
            min_snow: 0.35,
            max_snow: 1.0,
            min_soil_moisture: 0.0,
            max_soil_moisture: 1.0,
        },
        BiomeBand {
            name: "Coast".into(),
            id: 7,
            min_height: OPEN_HEIGHT_MIN,
            max_height: 25.0,
            min_wetness: 0.0,
            max_wetness: 1.0,
            min_temp: 0.25,
            max_temp: 1.0,
            min_precip: 0.2,
            max_precip: 1.0,
            min_snow: 0.0,
            max_snow: 0.2,
            min_soil_moisture: 0.0,
            max_soil_moisture: 1.0,
        },
    ]
}

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
                    thickness: 1.0e6,
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
            bands: default_climate_bands(),
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
