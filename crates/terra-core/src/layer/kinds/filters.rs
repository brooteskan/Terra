//! Layer parameter kinds (split by family).

use serde::{Deserialize, Serialize};

use super::noise::WorleyFeature;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlurParams {
    pub radius: u32,
    pub iterations: u32,
}

impl Default for BlurParams {
    fn default() -> Self {
        Self {
            radius: 2,
            iterations: 1,
        }
    }
}

/// WC-inspired effect / arid / erosion filter kinds.
///
/// Serde uses `snake_case` with PascalCase aliases so existing projects keep loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EffectFilterKind {
    #[default]
    #[serde(alias = "Smooth")]
    Smooth,
    #[serde(alias = "Distortion")]
    Distortion,
    #[serde(alias = "SpikeRemoval")]
    SpikeRemoval,
    #[serde(alias = "Shore")]
    Shore,
    #[serde(alias = "Strata")]
    Strata,
    #[serde(alias = "Crater")]
    Crater,
    #[serde(alias = "Denoise")]
    Denoise,
    // Wave A â€” rocky / cliff morphology
    #[serde(alias = "RockySharp")]
    RockySharp,
    #[serde(alias = "RockyWide")]
    RockyWide,
    #[serde(alias = "RockyLayers")]
    RockyLayers,
    #[serde(alias = "CliffReinforce")]
    CliffReinforce,
    // Wave B â€” flow morphology
    #[serde(alias = "SoftFlows")]
    SoftFlows,
    #[serde(alias = "ThinFlows")]
    ThinFlows,
    #[serde(alias = "RidgedFlows")]
    RidgedFlows,
    #[serde(alias = "WideFlows")]
    WideFlows,
    // Wave C â€” sediment fills
    #[serde(alias = "TalusFill")]
    TalusFill,
    #[serde(alias = "SedimentFillSoft")]
    SedimentFillSoft,
    #[serde(alias = "MudSettle")]
    MudSettle,
    #[serde(alias = "HydraulicSediment")]
    HydraulicSediment,
    // Wave D â€” arid / advanced erosion / drift
    RockyPlateaus,
    RockyCliffs,
    RockyHard,
    Canyon,
    Chipped,
    Cliffs,
    /// Basic-erosion â€œRockyâ€ (particle-flow-like ridged carve).
    Rocky,
    SedimentFlows,
    AngleBreak,
    WindCarve,
    // Wave E â€” effect / general
    Inflate,
    Deflate,
    Balloon,
    Blocks,
    Ridged,
    Rugged,
    SmoothRidges,
    AngleBlur,
    /// Fixed-direction anisotropic convolution (vs gradient-aligned Angle Blur).
    DirectionalBlur,
    Squeeze,
    Swirl,
    WashedOff,
    Hexagons,
    ScatterDetail,
    FlattenFilter,
    ZeroEdge,
    BorderBlend,
    /// Design: contrast-like height remap (WC Curve approximation).
    Curve,
    /// Design: hard height clamp / shelf (WC Cutoff approximation).
    Cutoff,
    /// True Kuwahara edge-preserving smoothing.
    Kuwahara,
    /// Terrace Simple — uniform height steps with smooth risers.
    TerraceSimple,
    /// Terrace Irregular — world-space perturbed elevation / spacing / riser phase.
    TerraceIrregular,
    /// Terrace Steep — slope-gated terracing oriented to terrain gradient.
    TerraceSteep,
    /// General: add a constant height, or set absolute height (`sea_level` â‰¥ 0.5 â‡’ Set).
    AddSet,
    /// Design: Voronoi cell imprint.
    DesignVoronoi,
    // Noise family (additive height modulation)
    NoiseBillow,
    NoiseGabor,
    NoisePerlin,
    NoisePhasor,
    NoiseRidged,
    NoiseSimplex,
    NoiseValue,
    NoiseVoronoi,
    NoiseWave,
    NoiseWhite,
}

impl EffectFilterKind {
    pub const ALL: &'static [EffectFilterKind] = &[
        EffectFilterKind::Smooth,
        EffectFilterKind::Distortion,
        EffectFilterKind::SpikeRemoval,
        EffectFilterKind::Shore,
        EffectFilterKind::Strata,
        EffectFilterKind::Crater,
        EffectFilterKind::Denoise,
        EffectFilterKind::RockySharp,
        EffectFilterKind::RockyWide,
        EffectFilterKind::RockyLayers,
        EffectFilterKind::CliffReinforce,
        EffectFilterKind::SoftFlows,
        EffectFilterKind::ThinFlows,
        EffectFilterKind::RidgedFlows,
        EffectFilterKind::WideFlows,
        EffectFilterKind::TalusFill,
        EffectFilterKind::SedimentFillSoft,
        EffectFilterKind::MudSettle,
        EffectFilterKind::HydraulicSediment,
        EffectFilterKind::RockyPlateaus,
        EffectFilterKind::RockyCliffs,
        EffectFilterKind::RockyHard,
        EffectFilterKind::Canyon,
        EffectFilterKind::Chipped,
        EffectFilterKind::Cliffs,
        EffectFilterKind::Rocky,
        EffectFilterKind::SedimentFlows,
        EffectFilterKind::AngleBreak,
        EffectFilterKind::WindCarve,
        EffectFilterKind::Inflate,
        EffectFilterKind::Deflate,
        EffectFilterKind::Balloon,
        EffectFilterKind::Blocks,
        EffectFilterKind::Ridged,
        EffectFilterKind::Rugged,
        EffectFilterKind::SmoothRidges,
        EffectFilterKind::AngleBlur,
        EffectFilterKind::DirectionalBlur,
        EffectFilterKind::Squeeze,
        EffectFilterKind::Swirl,
        EffectFilterKind::WashedOff,
        EffectFilterKind::Hexagons,
        EffectFilterKind::ScatterDetail,
        EffectFilterKind::FlattenFilter,
        EffectFilterKind::ZeroEdge,
        EffectFilterKind::BorderBlend,
        EffectFilterKind::Curve,
        EffectFilterKind::Cutoff,
        EffectFilterKind::Kuwahara,
        EffectFilterKind::TerraceSimple,
        EffectFilterKind::TerraceIrregular,
        EffectFilterKind::TerraceSteep,
        EffectFilterKind::AddSet,
        EffectFilterKind::DesignVoronoi,
        EffectFilterKind::NoiseBillow,
        EffectFilterKind::NoiseGabor,
        EffectFilterKind::NoisePerlin,
        EffectFilterKind::NoisePhasor,
        EffectFilterKind::NoiseRidged,
        EffectFilterKind::NoiseSimplex,
        EffectFilterKind::NoiseValue,
        EffectFilterKind::NoiseVoronoi,
        EffectFilterKind::NoiseWave,
        EffectFilterKind::NoiseWhite,
    ];

    pub fn label(self) -> &'static str {
        match self {
            EffectFilterKind::Smooth => "Smooth",
            EffectFilterKind::Distortion => "Distortion",
            EffectFilterKind::SpikeRemoval => "Spike Removal",
            EffectFilterKind::Shore => "Shore",
            EffectFilterKind::Strata => "Strata",
            EffectFilterKind::Crater => "Crater",
            EffectFilterKind::Denoise => "Denoise",
            EffectFilterKind::RockySharp => "Rocky Sharp",
            EffectFilterKind::RockyWide => "Rocky Wide",
            EffectFilterKind::RockyLayers => "Rocky Layers",
            EffectFilterKind::CliffReinforce => "Cliff Reinforce",
            EffectFilterKind::SoftFlows => "Soft Flows",
            EffectFilterKind::ThinFlows => "Thin Flows",
            EffectFilterKind::RidgedFlows => "Ridged Flows",
            EffectFilterKind::WideFlows => "Wide Flows",
            EffectFilterKind::TalusFill => "Talus Fill",
            EffectFilterKind::SedimentFillSoft => "Sediment Fill Soft",
            EffectFilterKind::MudSettle => "Mud Settle",
            EffectFilterKind::HydraulicSediment => "Hydraulic Sediment",
            EffectFilterKind::RockyPlateaus => "Rocky Plateaus",
            EffectFilterKind::RockyCliffs => "Rocky Cliffs",
            EffectFilterKind::RockyHard => "Rocky Hard",
            EffectFilterKind::Canyon => "Canyon",
            EffectFilterKind::Chipped => "Chipped",
            EffectFilterKind::Cliffs => "Cliffs",
            EffectFilterKind::Rocky => "Rocky",
            EffectFilterKind::SedimentFlows => "Sediment Flows",
            EffectFilterKind::AngleBreak => "Angle Break",
            EffectFilterKind::WindCarve => "Wind Carve",
            EffectFilterKind::Inflate => "Inflate",
            EffectFilterKind::Deflate => "Deflate",
            EffectFilterKind::Balloon => "Balloon",
            EffectFilterKind::Blocks => "Blocks",
            EffectFilterKind::Ridged => "Ridged",
            EffectFilterKind::Rugged => "Rugged",
            EffectFilterKind::SmoothRidges => "Smooth Ridges",
            EffectFilterKind::AngleBlur => "Angle Blur",
            EffectFilterKind::DirectionalBlur => "Directional Blur",
            EffectFilterKind::Squeeze => "Squeeze",
            EffectFilterKind::Swirl => "Swirl",
            EffectFilterKind::WashedOff => "Washed Off",
            EffectFilterKind::Hexagons => "Hexagons",
            EffectFilterKind::ScatterDetail => "Scatter",
            EffectFilterKind::FlattenFilter => "Flatten",
            EffectFilterKind::ZeroEdge => "Zero-Edge",
            EffectFilterKind::BorderBlend => "Border Blend",
            EffectFilterKind::Curve => "Curve",
            EffectFilterKind::Cutoff => "Cutoff",
            EffectFilterKind::Kuwahara => "Kuwahara",
            EffectFilterKind::TerraceSimple => "Terrace Simple",
            EffectFilterKind::TerraceIrregular => "Terrace Irregular",
            EffectFilterKind::TerraceSteep => "Terrace Steep",
            EffectFilterKind::AddSet => "Add/Set",
            EffectFilterKind::DesignVoronoi => "Voronoi",
            EffectFilterKind::NoiseBillow => "Billow",
            EffectFilterKind::NoiseGabor => "Gabor",
            EffectFilterKind::NoisePerlin => "Perlin",
            EffectFilterKind::NoisePhasor => "Phasor",
            EffectFilterKind::NoiseRidged => "Ridged Noise",
            EffectFilterKind::NoiseSimplex => "Simplex",
            EffectFilterKind::NoiseValue => "Value",
            EffectFilterKind::NoiseVoronoi => "Voronoi Noise",
            EffectFilterKind::NoiseWave => "Wave",
            EffectFilterKind::NoiseWhite => "White",
        }
    }

    pub fn cycle(self) -> Self {
        let idx = Self::ALL.iter().position(|&k| k == self).unwrap_or(0);
        Self::ALL[(idx + 1) % Self::ALL.len()]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectFilterParams {
    pub kind: EffectFilterKind,
    /// Mix strength \[0,1\] toward the filtered result.
    pub strength: f32,
    pub invert: bool,
    pub radius: u32,
    pub iterations: u32,
    pub seed: u64,
    pub frequency: f32,
    pub amount: f32,
    pub sea_level: f32,
    pub beach_width: f32,
    /// Crater radius as fraction of world size \[0, 0.5\].
    pub crater_radius: f32,
    /// Slope gate min (degrees) for morphology filters.
    #[serde(default)]
    pub slope_min: f32,
    /// Slope gate max (degrees) for morphology filters.
    #[serde(default = "default_slope_max")]
    pub slope_max: f32,
    /// Flow accumulation threshold (normalized) for flow filters.
    #[serde(default = "default_flow_threshold")]
    pub flow_threshold: f32,
    /// Rock hardness / lithology \[0 soft … 1 hard\] for arid/rock filters.
    #[serde(default = "default_rock_hardness")]
    pub rock_hardness: f32,
    /// Canyon wall / cliff face steepness \[0 soft … 1 sheer\].
    #[serde(default = "default_wall_steepness")]
    pub wall_steepness: f32,
    /// Valley floor / plateau flatness bias \[0 none … 1 wide floors\].
    #[serde(default = "default_valley_floor")]
    pub valley_floor: f32,
    /// Talus / scree apron mix for cliff & canyon filters \[0 none … 1 full\].
    #[serde(default = "default_talus_mix")]
    pub talus_mix: f32,
    /// Terrace step height in meters. When `<= 0`, levels are derived from `amount`.
    #[serde(default)]
    pub terrace_height: f32,
    /// Terrace phase offset \[0, 1).
    #[serde(default)]
    pub terrace_offset: f32,
    /// Round tread tops toward source elevation \[0, 1\].
    #[serde(default = "default_top_smoothness")]
    pub top_smoothness: f32,
    /// Riser sharpness \[0 soft … 1 sheer\].
    #[serde(default = "default_riser_sharpness")]
    pub riser_sharpness: f32,
    /// Fractal octave count for noise-family filters.
    #[serde(default = "default_octaves")]
    pub octaves: u32,
    /// Frequency multiplier between octaves.
    #[serde(default = "default_lacunarity")]
    pub lacunarity: f32,
    /// Amplitude gain / persistence between octaves.
    #[serde(default = "default_persistence")]
    pub persistence: f32,
    /// World-space feature scale in metres. When `> 0`, overrides `frequency` as `1/scale_m`.
    #[serde(default)]
    pub scale_m: f32,
    /// Domain rotation in degrees (noise / directional filters).
    #[serde(default)]
    pub rotation_deg: f32,
    /// Stretch along the rotated X axis (1 = isotropic; >1 elongates).
    #[serde(default = "default_anisotropy")]
    pub anisotropy: f32,
    /// Domain-warp displacement amplitude in metres (0 = off).
    #[serde(default)]
    pub warp_strength: f32,
    /// Domain-warp frequency (cycles per metre).
    #[serde(default = "default_warp_frequency")]
    pub warp_frequency: f32,
    /// When true, wrap noise coordinates into a periodic domain sized by `scale_m` / frequency.
    #[serde(default)]
    pub tileable: bool,
    /// Worley feature for Design / Noise Voronoi (F1, F2, F2−F1).
    #[serde(default = "default_voronoi_feature")]
    pub voronoi_feature: WorleyFeature,
}

fn default_slope_max() -> f32 {
    90.0
}

fn default_flow_threshold() -> f32 {
    0.15
}

fn default_rock_hardness() -> f32 {
    0.55
}

fn default_wall_steepness() -> f32 {
    0.65
}

fn default_valley_floor() -> f32 {
    0.25
}

fn default_talus_mix() -> f32 {
    0.35
}

fn default_top_smoothness() -> f32 {
    0.15
}

fn default_riser_sharpness() -> f32 {
    0.75
}

fn default_octaves() -> u32 {
    5
}

fn default_lacunarity() -> f32 {
    2.0
}

fn default_persistence() -> f32 {
    0.5
}

fn default_anisotropy() -> f32 {
    1.0
}

fn default_warp_frequency() -> f32 {
    0.01
}

fn default_voronoi_feature() -> WorleyFeature {
    WorleyFeature::F2MinusF1
}

impl Default for EffectFilterParams {
    fn default() -> Self {
        Self {
            kind: EffectFilterKind::Smooth,
            strength: 0.55,
            invert: false,
            radius: 2,
            iterations: 1,
            seed: 1,
            frequency: 0.02,
            amount: 8.0,
            sea_level: 0.0,
            beach_width: 24.0,
            crater_radius: 0.18,
            slope_min: 0.0,
            slope_max: 90.0,
            flow_threshold: 0.15,
            rock_hardness: 0.55,
            wall_steepness: 0.65,
            valley_floor: 0.25,
            talus_mix: 0.35,
            terrace_height: 0.0,
            terrace_offset: 0.0,
            top_smoothness: 0.15,
            riser_sharpness: 0.75,
            octaves: 5,
            lacunarity: 2.0,
            persistence: 0.5,
            scale_m: 0.0,
            rotation_deg: 0.0,
            anisotropy: 1.0,
            warp_strength: 0.0,
            warp_frequency: 0.01,
            tileable: false,
            voronoi_feature: WorleyFeature::F2MinusF1,
        }
    }
}

impl EffectFilterParams {
    /// Effective spatial frequency in cycles per metre.
    /// Prefer `scale_m` (wavelength) when set; otherwise use `frequency`.
    pub fn effective_frequency(&self) -> f32 {
        if self.scale_m > 1e-5 {
            1.0 / self.scale_m
        } else {
            self.frequency.max(1e-5)
        }
    }

    /// True for additive noise-family EffectFilters.
    pub fn is_noise_family(&self) -> bool {
        matches!(
            self.kind,
            EffectFilterKind::NoiseBillow
                | EffectFilterKind::NoiseGabor
                | EffectFilterKind::NoisePerlin
                | EffectFilterKind::NoisePhasor
                | EffectFilterKind::NoiseRidged
                | EffectFilterKind::NoiseSimplex
                | EffectFilterKind::NoiseValue
                | EffectFilterKind::NoiseVoronoi
                | EffectFilterKind::NoiseWave
                | EffectFilterKind::NoiseWhite
        )
    }

    pub fn smooth() -> Self {
        Self {
            kind: EffectFilterKind::Smooth,
            ..Self::default()
        }
    }
    pub fn distortion() -> Self {
        Self {
            kind: EffectFilterKind::Distortion,
            strength: 0.65,
            amount: 12.0,
            ..Self::default()
        }
    }
    pub fn spike_removal() -> Self {
        Self {
            kind: EffectFilterKind::SpikeRemoval,
            strength: 0.8,
            radius: 1,
            ..Self::default()
        }
    }
    pub fn shore() -> Self {
        Self {
            kind: EffectFilterKind::Shore,
            strength: 0.9,
            ..Self::default()
        }
    }
    pub fn strata() -> Self {
        Self {
            kind: EffectFilterKind::Strata,
            strength: 0.45,
            frequency: 0.08,
            amount: 4.0,
            ..Self::default()
        }
    }
    pub fn crater() -> Self {
        Self {
            kind: EffectFilterKind::Crater,
            strength: 1.0,
            amount: 35.0,
            crater_radius: 0.2,
            ..Self::default()
        }
    }
    pub fn denoise() -> Self {
        Self {
            kind: EffectFilterKind::Denoise,
            strength: 0.6,
            radius: 2,
            iterations: 2,
            ..Self::default()
        }
    }
    pub fn rocky_sharp() -> Self {
        Self {
            kind: EffectFilterKind::RockySharp,
            strength: 0.55,
            frequency: 0.04,
            amount: 6.0,
            slope_min: 25.0,
            slope_max: 90.0,
            rock_hardness: 0.6,
            talus_mix: 0.0,
            ..Self::default()
        }
    }
    pub fn rocky_wide() -> Self {
        Self {
            kind: EffectFilterKind::RockyWide,
            strength: 0.5,
            frequency: 0.012,
            amount: 8.0,
            slope_min: 10.0,
            slope_max: 55.0,
            rock_hardness: 0.5,
            talus_mix: 0.0,
            ..Self::default()
        }
    }
    pub fn rocky_layers() -> Self {
        Self {
            kind: EffectFilterKind::RockyLayers,
            strength: 0.5,
            frequency: 0.06,
            amount: 5.0,
            slope_min: 20.0,
            slope_max: 90.0,
            rock_hardness: 0.65,
            radius: 2,
            talus_mix: 0.0,
            ..Self::default()
        }
    }
    pub fn cliff_reinforce() -> Self {
        Self {
            kind: EffectFilterKind::CliffReinforce,
            strength: 0.45,
            amount: 4.0,
            radius: 2,
            slope_min: 30.0,
            slope_max: 75.0,
            wall_steepness: 0.7,
            talus_mix: 0.0,
            ..Self::default()
        }
    }
    pub fn soft_flows() -> Self {
        Self {
            kind: EffectFilterKind::SoftFlows,
            strength: 0.5,
            amount: 3.0,
            flow_threshold: 0.08,
            ..Self::default()
        }
    }
    pub fn thin_flows() -> Self {
        Self {
            kind: EffectFilterKind::ThinFlows,
            strength: 0.55,
            amount: 5.0,
            flow_threshold: 0.2,
            ..Self::default()
        }
    }
    pub fn ridged_flows() -> Self {
        Self {
            kind: EffectFilterKind::RidgedFlows,
            strength: 0.5,
            amount: 4.0,
            flow_threshold: 0.12,
            ..Self::default()
        }
    }
    pub fn wide_flows() -> Self {
        Self {
            kind: EffectFilterKind::WideFlows,
            strength: 0.45,
            amount: 3.5,
            radius: 3,
            flow_threshold: 0.1,
            slope_max: 25.0,
            ..Self::default()
        }
    }
    pub fn talus_fill() -> Self {
        Self {
            kind: EffectFilterKind::TalusFill,
            strength: 0.5,
            amount: 4.0,
            slope_min: 35.0,
            ..Self::default()
        }
    }
    pub fn sediment_fill_soft() -> Self {
        Self {
            kind: EffectFilterKind::SedimentFillSoft,
            strength: 0.55,
            amount: 3.0,
            radius: 3,
            slope_max: 18.0,
            ..Self::default()
        }
    }
    pub fn mud_settle() -> Self {
        Self {
            kind: EffectFilterKind::MudSettle,
            strength: 0.7,
            amount: 5.0,
            radius: 2,
            ..Self::default()
        }
    }
    pub fn hydraulic_sediment() -> Self {
        Self {
            kind: EffectFilterKind::HydraulicSediment,
            strength: 0.5,
            amount: 4.0,
            flow_threshold: 0.15,
            slope_max: 22.0,
            ..Self::default()
        }
    }
    pub fn rocky_plateaus() -> Self {
        Self {
            kind: EffectFilterKind::RockyPlateaus,
            strength: 0.55,
            frequency: 0.008,
            amount: 12.0,
            slope_min: 15.0,
            slope_max: 90.0,
            radius: 3,
            rock_hardness: 0.6,
            valley_floor: 0.45,
            talus_mix: 0.4,
            flow_threshold: 0.18,
            iterations: 2,
            ..Self::default()
        }
    }
    pub fn rocky_cliffs() -> Self {
        Self {
            kind: EffectFilterKind::RockyCliffs,
            strength: 0.6,
            frequency: 0.05,
            amount: 8.0,
            slope_min: 35.0,
            slope_max: 90.0,
            rock_hardness: 0.65,
            talus_mix: 0.45,
            radius: 2,
            flow_threshold: 0.12,
            ..Self::default()
        }
    }
    pub fn rocky_hard() -> Self {
        Self {
            kind: EffectFilterKind::RockyHard,
            strength: 0.65,
            frequency: 0.06,
            amount: 9.0,
            slope_min: 40.0,
            slope_max: 90.0,
            rock_hardness: 0.85,
            talus_mix: 0.3,
            iterations: 3,
            ..Self::default()
        }
    }
    pub fn canyon() -> Self {
        Self {
            kind: EffectFilterKind::Canyon,
            strength: 0.6,
            amount: 10.0,
            radius: 4,
            flow_threshold: 0.35,
            rock_hardness: 0.55,
            wall_steepness: 0.7,
            valley_floor: 0.3,
            talus_mix: 0.4,
            iterations: 3,
            beach_width: 8.0,
            ..Self::default()
        }
    }
    pub fn chipped() -> Self {
        Self {
            kind: EffectFilterKind::Chipped,
            strength: 0.5,
            frequency: 0.12,
            amount: 4.0,
            slope_min: 30.0,
            slope_max: 90.0,
            rock_hardness: 0.5,
            radius: 2,
            flow_threshold: 0.12,
            talus_mix: 0.0,
            ..Self::default()
        }
    }
    pub fn cliffs() -> Self {
        Self {
            kind: EffectFilterKind::Cliffs,
            strength: 0.5,
            amount: 5.0,
            radius: 3,
            slope_min: 28.0,
            slope_max: 80.0,
            flow_threshold: 0.15,
            rock_hardness: 0.6,
            wall_steepness: 0.65,
            talus_mix: 0.5,
            iterations: 2,
            ..Self::default()
        }
    }
    pub fn rocky() -> Self {
        Self {
            kind: EffectFilterKind::Rocky,
            strength: 0.5,
            frequency: 0.03,
            amount: 5.0,
            flow_threshold: 0.1,
            slope_min: 15.0,
            rock_hardness: 0.55,
            talus_mix: 0.0,
            ..Self::default()
        }
    }
    pub fn sediment_flows() -> Self {
        Self {
            kind: EffectFilterKind::SedimentFlows,
            strength: 0.55,
            amount: 4.0,
            flow_threshold: 0.1,
            slope_max: 20.0,
            ..Self::default()
        }
    }
    pub fn angle_break() -> Self {
        Self {
            kind: EffectFilterKind::AngleBreak,
            strength: 0.5,
            frequency: 0.04,
            amount: 6.0,
            slope_min: 10.0,
            slope_max: 70.0,
            rotation_deg: 40.0,
            ..Self::default()
        }
    }
    pub fn wind_carve() -> Self {
        Self {
            kind: EffectFilterKind::WindCarve,
            strength: 0.55,
            amount: 5.0,
            radius: 3,
            frequency: 0.02,
            rotation_deg: 25.0,
            iterations: 8,
            ..Self::default()
        }
    }
    pub fn inflate() -> Self {
        Self {
            kind: EffectFilterKind::Inflate,
            strength: 0.5,
            amount: 5.0,
            radius: 2,
            ..Self::default()
        }
    }
    pub fn deflate() -> Self {
        Self {
            kind: EffectFilterKind::Deflate,
            strength: 0.5,
            amount: 5.0,
            radius: 2,
            ..Self::default()
        }
    }
    pub fn balloon() -> Self {
        Self {
            kind: EffectFilterKind::Balloon,
            strength: 0.55,
            amount: 6.0,
            radius: 3,
            slope_min: 5.0,
            slope_max: 60.0,
            ..Self::default()
        }
    }
    pub fn blocks() -> Self {
        Self {
            kind: EffectFilterKind::Blocks,
            strength: 0.6,
            amount: 8.0,
            frequency: 0.02,
            ..Self::default()
        }
    }
    pub fn ridged() -> Self {
        Self {
            kind: EffectFilterKind::Ridged,
            strength: 0.5,
            frequency: 0.035,
            amount: 6.0,
            ..Self::default()
        }
    }
    pub fn rugged() -> Self {
        Self {
            kind: EffectFilterKind::Rugged,
            strength: 0.55,
            frequency: 0.05,
            amount: 7.0,
            ..Self::default()
        }
    }
    pub fn smooth_ridges() -> Self {
        Self {
            kind: EffectFilterKind::SmoothRidges,
            strength: 0.5,
            frequency: 0.025,
            amount: 5.0,
            radius: 2,
            ..Self::default()
        }
    }
    pub fn angle_blur() -> Self {
        Self {
            kind: EffectFilterKind::AngleBlur,
            strength: 0.6,
            amount: 4.0,
            radius: 3,
            ..Self::default()
        }
    }
    pub fn directional_blur() -> Self {
        Self {
            kind: EffectFilterKind::DirectionalBlur,
            strength: 0.65,
            amount: 4.0,
            radius: 4,
            rotation_deg: 35.0,
            ..Self::default()
        }
    }
    pub fn squeeze() -> Self {
        Self {
            kind: EffectFilterKind::Squeeze,
            strength: 0.7,
            amount: 0.55,
            ..Self::default()
        }
    }
    pub fn swirl() -> Self {
        Self {
            kind: EffectFilterKind::Swirl,
            strength: 0.65,
            amount: 0.8,
            frequency: 0.02,
            ..Self::default()
        }
    }
    pub fn washed_off() -> Self {
        Self {
            kind: EffectFilterKind::WashedOff,
            strength: 0.55,
            amount: 4.0,
            radius: 2,
            slope_min: 25.0,
            ..Self::default()
        }
    }
    pub fn hexagons() -> Self {
        Self {
            kind: EffectFilterKind::Hexagons,
            strength: 0.6,
            amount: 6.0,
            frequency: 0.04,
            ..Self::default()
        }
    }
    pub fn scatter_detail() -> Self {
        Self {
            kind: EffectFilterKind::ScatterDetail,
            strength: 0.55,
            frequency: 0.08,
            amount: 8.0,
            ..Self::default()
        }
    }
    pub fn flatten_filter() -> Self {
        Self {
            kind: EffectFilterKind::FlattenFilter,
            strength: 0.7,
            amount: 10.0,
            radius: 4,
            ..Self::default()
        }
    }
    pub fn zero_edge() -> Self {
        Self {
            kind: EffectFilterKind::ZeroEdge,
            strength: 1.0,
            amount: 0.15,
            ..Self::default()
        }
    }
    pub fn border_blend() -> Self {
        Self {
            kind: EffectFilterKind::BorderBlend,
            strength: 0.7,
            amount: 0.2,
            radius: 4,
            ..Self::default()
        }
    }
    pub fn curve() -> Self {
        Self {
            kind: EffectFilterKind::Curve,
            strength: 0.85,
            amount: 0.55,
            ..Self::default()
        }
    }
    pub fn cutoff() -> Self {
        Self {
            kind: EffectFilterKind::Cutoff,
            strength: 1.0,
            amount: 0.35,
            sea_level: 0.45,
            ..Self::default()
        }
    }
    pub fn kuwahara() -> Self {
        Self {
            kind: EffectFilterKind::Kuwahara,
            strength: 0.85,
            radius: 3,
            iterations: 1,
            ..Self::default()
        }
    }
    pub fn terrace_simple() -> Self {
        Self {
            kind: EffectFilterKind::TerraceSimple,
            strength: 0.9,
            amount: 8.0,
            terrace_height: 0.0,
            terrace_offset: 0.0,
            top_smoothness: 0.2,
            riser_sharpness: 0.75,
            ..Self::default()
        }
    }
    pub fn terrace_irregular() -> Self {
        Self {
            kind: EffectFilterKind::TerraceIrregular,
            strength: 0.85,
            amount: 8.0,
            frequency: 0.04,
            seed: 42,
            terrace_height: 0.0,
            terrace_offset: 0.0,
            top_smoothness: 0.25,
            riser_sharpness: 0.55,
            ..Self::default()
        }
    }
    pub fn terrace_steep() -> Self {
        Self {
            kind: EffectFilterKind::TerraceSteep,
            strength: 0.95,
            amount: 6.0,
            slope_min: 10.0,
            slope_max: 60.0,
            terrace_height: 0.0,
            terrace_offset: 0.0,
            top_smoothness: 0.08,
            riser_sharpness: 0.95,
            ..Self::default()
        }
    }
    pub fn add_set() -> Self {
        Self {
            kind: EffectFilterKind::AddSet,
            strength: 1.0,
            amount: 10.0,
            sea_level: 0.0, // < 0.5 â‡’ Add; â‰¥ 0.5 â‡’ Set
            ..Self::default()
        }
    }
    pub fn design_voronoi() -> Self {
        Self {
            kind: EffectFilterKind::DesignVoronoi,
            strength: 0.75,
            amount: 12.0,
            frequency: 0.02,
            seed: 7,
            voronoi_feature: WorleyFeature::F2MinusF1,
            ..Self::default()
        }
    }
    pub fn noise_billow() -> Self {
        Self {
            kind: EffectFilterKind::NoiseBillow,
            strength: 0.7,
            amount: 10.0,
            frequency: 0.02,
            scale_m: 50.0,
            octaves: 5,
            lacunarity: 2.0,
            persistence: 0.5,
            ..Self::default()
        }
    }
    pub fn noise_gabor() -> Self {
        Self {
            kind: EffectFilterKind::NoiseGabor,
            strength: 0.65,
            amount: 8.0,
            frequency: 0.03,
            scale_m: 35.0,
            octaves: 3,
            anisotropy: 1.4,
            ..Self::default()
        }
    }
    pub fn noise_perlin() -> Self {
        Self {
            kind: EffectFilterKind::NoisePerlin,
            strength: 0.7,
            amount: 10.0,
            frequency: 0.02,
            scale_m: 50.0,
            octaves: 5,
            lacunarity: 2.0,
            persistence: 0.5,
            ..Self::default()
        }
    }
    pub fn noise_phasor() -> Self {
        Self {
            kind: EffectFilterKind::NoisePhasor,
            strength: 0.65,
            amount: 8.0,
            frequency: 0.025,
            scale_m: 40.0,
            octaves: 3,
            anisotropy: 1.25,
            ..Self::default()
        }
    }
    pub fn noise_ridged() -> Self {
        Self {
            kind: EffectFilterKind::NoiseRidged,
            strength: 0.7,
            amount: 10.0,
            frequency: 0.02,
            scale_m: 50.0,
            octaves: 5,
            lacunarity: 2.2,
            persistence: 0.5,
            ..Self::default()
        }
    }
    pub fn noise_simplex() -> Self {
        Self {
            kind: EffectFilterKind::NoiseSimplex,
            strength: 0.7,
            amount: 10.0,
            frequency: 0.02,
            scale_m: 50.0,
            octaves: 5,
            ..Self::default()
        }
    }
    pub fn noise_value() -> Self {
        Self {
            kind: EffectFilterKind::NoiseValue,
            strength: 0.7,
            amount: 10.0,
            frequency: 0.025,
            scale_m: 40.0,
            octaves: 4,
            ..Self::default()
        }
    }
    pub fn noise_voronoi() -> Self {
        Self {
            kind: EffectFilterKind::NoiseVoronoi,
            strength: 0.7,
            amount: 10.0,
            frequency: 0.02,
            scale_m: 50.0,
            voronoi_feature: WorleyFeature::F2MinusF1,
            ..Self::default()
        }
    }
    pub fn noise_wave() -> Self {
        Self {
            kind: EffectFilterKind::NoiseWave,
            strength: 0.65,
            amount: 8.0,
            frequency: 0.04,
            scale_m: 25.0,
            rotation_deg: 15.0,
            anisotropy: 1.6,
            ..Self::default()
        }
    }
    pub fn noise_white() -> Self {
        Self {
            kind: EffectFilterKind::NoiseWhite,
            strength: 0.4,
            amount: 4.0,
            frequency: 1.0,
            ..Self::default()
        }
    }
}
