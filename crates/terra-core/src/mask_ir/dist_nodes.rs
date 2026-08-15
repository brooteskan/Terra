//! Persisted distribution-node schema.

use super::MaskCombine;
use crate::mask_types::MaskRef;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DistNodeId(pub Uuid);

impl DistNodeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for DistNodeId {
    fn default() -> Self {
        Self::new()
    }
}

/// One node in a nested distribution stack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistNode {
    pub id: DistNodeId,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    #[serde(default)]
    pub combine: MaskCombine,
    pub kind: DistNodeKind,
    /// Nested effect / child nodes applied after this node's base value.
    #[serde(default)]
    pub children: Vec<DistNode>,
}

fn default_opacity() -> f32 {
    1.0
}

fn default_true() -> bool {
    true
}

impl Default for DistNode {
    fn default() -> Self {
        Self::fill(1.0)
    }
}

impl DistNode {
    pub fn new(kind: DistNodeKind) -> Self {
        Self {
            id: DistNodeId::new(),
            enabled: true,
            opacity: 1.0,
            combine: MaskCombine::Multiply,
            kind,
            children: Vec::new(),
        }
    }

    pub fn fill(value: f32) -> Self {
        Self::new(DistNodeKind::Fill { value })
    }

    pub fn mask_ref(mask: MaskRef) -> Self {
        Self::new(DistNodeKind::MaskAsset { mask })
    }

    pub fn slope(min_deg: f32, max_deg: f32) -> Self {
        Self::new(DistNodeKind::Slope { min_deg, max_deg })
    }

    pub fn height(min: f32, max: f32) -> Self {
        Self::new(DistNodeKind::Height { min, max })
    }

    pub fn label(&self) -> &'static str {
        self.kind.label()
    }
}

/// Procedural / terrain / layer / effect kinds for distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistNodeKind {
    /// Constant fill \[0,1\].
    Fill {
        value: f32,
    },
    /// Reference a project mask asset.
    MaskAsset {
        mask: MaskRef,
    },
    /// Value noise in \[0,1\].
    Noise {
        seed: u64,
        frequency: f32,
    },
    /// Perlin-style gradient noise (fBm), normalized to \[0,1\].
    NoisePerlin {
        seed: u64,
        frequency: f32,
        octaves: u32,
    },
    /// Ridged multifractal noise, normalized to \[0,1\].
    NoiseRidged {
        seed: u64,
        frequency: f32,
        octaves: u32,
    },
    /// Worley / cellular noise (F1 distance), normalized to \[0,1\].
    NoiseWorley {
        seed: u64,
        frequency: f32,
    },
    /// Billowy (absolute-value) fBm noise, normalized to \[0,1\].
    NoiseBillow {
        seed: u64,
        frequency: f32,
        octaves: u32,
    },
    Height {
        min: f32,
        max: f32,
    },
    Slope {
        min_deg: f32,
        max_deg: f32,
    },
    Curvature {
        min: f32,
        max: f32,
    },
    Cavity {
        strength: f32,
    },
    Flow {
        min: f32,
        max: f32,
    },
    SeaLevel {
        level: f32,
        width: f32,
    },
    Occlusion {
        radius: u32,
        strength: f32,
    },
    /// Steepness of the terrain, in degrees (same underlying calc as `Slope`).
    Steepness {
        min_deg: f32,
        max_deg: f32,
    },
    /// Slope-facing direction (aspect) in degrees, with an angular tolerance.
    Angle {
        degrees: f32,
        spread: f32,
    },
    /// Local height variance within `radius` samples.
    Roughness {
        radius: u32,
        strength: f32,
    },
    /// High-frequency noise gated by steep slope, for scattering rock detail.
    Rocks {
        density: f32,
        threshold: f32,
    },
    /// Rim / edge highlighting around steep terrain.
    RockyEdges {
        width: f32,
        strength: f32,
    },
    /// Effect: invert the parent accumulator (used as child effect).
    EffectInvert,
    /// Effect: box blur.
    EffectBlur {
        radius: u32,
    },
    /// Effect: levels (in-black / in-white / gamma) — Region Mask Editor Levels op.
    EffectLevels {
        in_black: f32,
        in_white: f32,
        gamma: f32,
    },
    /// Effect: remap input range to \[0,1\].
    EffectRemap {
        in_min: f32,
        in_max: f32,
    },
    /// Effect: contrast around 0.5.
    EffectContrast {
        amount: f32,
    },
    /// Effect: soft clamp.
    EffectClamp {
        min: f32,
        max: f32,
    },
    /// Effect: soft S-curve contrast (steeper than `EffectContrast` near 0.5).
    EffectCurve {
        contrast: f32,
    },
    /// Effect: domain-warp sample of the input via noise-driven offsets.
    EffectDistortion {
        seed: u64,
        frequency: f32,
        amount: f32,
    },
    /// Effect: sobel-ish edge magnitude of the input.
    EffectEdge {
        strength: f32,
    },
    /// Effect: classic smoothstep remap between two edges.
    EffectSmoothstep {
        edge0: f32,
        edge1: f32,
    },
    /// Effect: cheap iterative "flow" smear (spreads high values into neighbours).
    EffectSimpleFlow {
        iterations: u32,
        strength: f32,
    },
    /// Painted mask asset (alias of [`Self::MaskAsset`] for region-mask catalogs).
    Paint {
        mask: MaskRef,
    },
    /// Soft polygon in UV \[0,1\] (point-in-polygon with optional edge soft width).
    Polygon {
        /// Closed ring of UV points `(u, v)` in \[0,1\].
        points: Vec<[f32; 2]>,
        /// Soft edge width in UV units (0 = hard).
        soft: f32,
    },
    /// Distance-to-polyline ribbon in UV space.
    Spline {
        points: Vec<[f32; 2]>,
        /// Half-width in UV units.
        width: f32,
    },
    /// Distance field from a thresholded mask asset.
    Distance {
        mask: MaskRef,
        /// Distance in samples mapping to 0 outside the core.
        max_distance: f32,
    },
    /// Climate aux channel (real when aux is present; otherwise ones — full coverage).
    Climate {
        channel: ClimateMaskChannel,
    },
    /// Voronoi / Worley cell field (real evaluation via worley noise).
    Voronoi {
        seed: u64,
        frequency: f32,
        /// 0 = F1 fill, 1 = edge emphasis (1 - smoothstep of F1).
        edge_weight: f32,
    },
    /// Imported / project mask asset (alias of [`Self::MaskAsset`]).
    ImportedMask {
        mask: MaskRef,
    },
    /// Fold children with Multiply (accumulator starts at ones). Placement compile.
    GroupAll,
    /// Fold children with Max (accumulator starts at zeros). Placement compile.
    GroupAny,
    /// Morphological expand (dilate) — radius in meters, converted via cell size at bake.
    EffectDilate {
        radius_m: f32,
    },
    /// Morphological contract (erode) — radius in meters.
    EffectErode {
        radius_m: f32,
    },
}

/// Climate channel selector for [`DistNodeKind::Climate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ClimateMaskChannel {
    #[default]
    Temperature,
    Rainfall,
    Humidity,
    Snow,
    SoilMoisture,
    WindExposure,
}

impl ClimateMaskChannel {
    pub fn aux_key(self) -> &'static str {
        match self {
            ClimateMaskChannel::Temperature => "temperature",
            ClimateMaskChannel::Rainfall => "rainfall",
            ClimateMaskChannel::Humidity => "humidity",
            ClimateMaskChannel::Snow => "snow",
            ClimateMaskChannel::SoilMoisture => "soil_moisture",
            ClimateMaskChannel::WindExposure => "wind_exposure",
        }
    }
}

impl DistNodeKind {
    pub fn label(&self) -> &'static str {
        match self {
            DistNodeKind::Fill { .. } => "Fill",
            DistNodeKind::MaskAsset { .. } => "Mask",
            DistNodeKind::Noise { .. } => "Noise",
            DistNodeKind::NoisePerlin { .. } => "Noise (Perlin)",
            DistNodeKind::NoiseRidged { .. } => "Noise (Ridged)",
            DistNodeKind::NoiseWorley { .. } => "Noise (Worley)",
            DistNodeKind::NoiseBillow { .. } => "Noise (Billow)",
            DistNodeKind::Height { .. } => "Height",
            DistNodeKind::Slope { .. } => "Slope",
            DistNodeKind::Curvature { .. } => "Curvature",
            DistNodeKind::Cavity { .. } => "Cavity",
            DistNodeKind::Flow { .. } => "Flow",
            DistNodeKind::SeaLevel { .. } => "Sea Level",
            DistNodeKind::Occlusion { .. } => "Occlusion",
            DistNodeKind::Steepness { .. } => "Steepness",
            DistNodeKind::Angle { .. } => "Angle",
            DistNodeKind::Roughness { .. } => "Roughness",
            DistNodeKind::Rocks { .. } => "Rocks",
            DistNodeKind::RockyEdges { .. } => "Rocky Edges",
            DistNodeKind::EffectInvert => "Invert",
            DistNodeKind::EffectBlur { .. } => "Blur",
            DistNodeKind::EffectLevels { .. } => "Levels",
            DistNodeKind::EffectRemap { .. } => "Remap",
            DistNodeKind::EffectContrast { .. } => "Contrast",
            DistNodeKind::EffectClamp { .. } => "Clamp",
            DistNodeKind::EffectCurve { .. } => "Curve",
            DistNodeKind::EffectDistortion { .. } => "Distortion",
            DistNodeKind::EffectEdge { .. } => "Edge",
            DistNodeKind::EffectSmoothstep { .. } => "Smoothstep",
            DistNodeKind::EffectSimpleFlow { .. } => "Simple Flow",
            DistNodeKind::Paint { .. } => "Paint",
            DistNodeKind::Polygon { .. } => "Polygon",
            DistNodeKind::Spline { .. } => "Spline",
            DistNodeKind::Distance { .. } => "Distance",
            DistNodeKind::Climate { .. } => "Climate",
            DistNodeKind::Voronoi { .. } => "Voronoi",
            DistNodeKind::ImportedMask { .. } => "Imported Mask",
            DistNodeKind::GroupAll => "All",
            DistNodeKind::GroupAny => "Any",
            DistNodeKind::EffectDilate { .. } => "Expand",
            DistNodeKind::EffectErode { .. } => "Contract",
        }
    }

    pub fn is_effect(&self) -> bool {
        matches!(
            self,
            DistNodeKind::EffectInvert
                | DistNodeKind::EffectBlur { .. }
                | DistNodeKind::EffectLevels { .. }
                | DistNodeKind::EffectRemap { .. }
                | DistNodeKind::EffectContrast { .. }
                | DistNodeKind::EffectClamp { .. }
                | DistNodeKind::EffectCurve { .. }
                | DistNodeKind::EffectDistortion { .. }
                | DistNodeKind::EffectEdge { .. }
                | DistNodeKind::EffectSmoothstep { .. }
                | DistNodeKind::EffectSimpleFlow { .. }
                | DistNodeKind::EffectDilate { .. }
                | DistNodeKind::EffectErode { .. }
        )
    }

    pub fn is_placement_group(&self) -> bool {
        matches!(self, DistNodeKind::GroupAll | DistNodeKind::GroupAny)
    }
}
