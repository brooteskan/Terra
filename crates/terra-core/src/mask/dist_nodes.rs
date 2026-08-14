//! World Creator–style distribution nodes (procedural mask graph).

use super::geom::{dist_point_segment, point_in_polygon};
use super::{MaskCombine, MaskField, MaskId, MaskRef};
use crate::heightfield::{Heightfield, HeightfieldMetrics};
use crate::noise::WorleyMetric;
use crate::noise::{perlin2, worley2};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable id for a distribution node in the stack.
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

/// Optional height / aux context for terrain-feature distributions.
pub struct DistBakeContext<'a> {
    pub height: Option<&'a Heightfield>,
    pub slope_deg: Option<&'a [f32]>,
    pub curvature: Option<&'a [f32]>,
    pub flow: Option<&'a [f32]>,
    pub masks: &'a std::collections::HashMap<MaskId, MaskField>,
    /// Optional named aux maps (climate, etc.) for [`DistNodeKind::Climate`].
    pub aux: Option<&'a std::collections::HashMap<String, MaskField>>,
}

impl<'a> DistBakeContext<'a> {
    pub fn masks_only(masks: &'a std::collections::HashMap<MaskId, MaskField>) -> Self {
        Self {
            height: None,
            slope_deg: None,
            curvature: None,
            flow: None,
            masks,
            aux: None,
        }
    }
}

/// Evaluate a DistNode tree into a mask field.
pub fn bake_dist_nodes(
    nodes: &[DistNode],
    metrics: HeightfieldMetrics,
    ctx: &DistBakeContext<'_>,
) -> MaskField {
    if nodes.is_empty() {
        return MaskField::ones(metrics);
    }
    let mut acc = MaskField::ones(metrics);
    for node in nodes {
        if !node.enabled {
            continue;
        }
        let mut field = eval_node_base(node, metrics, ctx);
        // Placement groups fold children inside eval_node_base.
        if !node.children.is_empty() && !node.kind.is_placement_group() {
            let child_baked = bake_dist_nodes(&node.children, metrics, ctx);
            // Children that are pure effects operate on `field`; otherwise multiply-combine.
            let all_effects = node.children.iter().all(|c| c.kind.is_effect());
            if all_effects {
                for c in &node.children {
                    if c.enabled {
                        field = apply_effect(&c.kind, &field, c.opacity);
                    }
                }
            } else {
                for j in 0..metrics.height {
                    for i in 0..metrics.width {
                        let v = field.get(i, j) * child_baked.get(i, j);
                        field.set(i, j, v.clamp(0.0, 1.0));
                    }
                }
            }
        }
        for j in 0..metrics.height {
            for i in 0..metrics.width {
                let v = (field.get(i, j) * node.opacity).clamp(0.0, 1.0);
                let combined = node.combine.apply(acc.get(i, j), v);
                acc.set(i, j, combined);
            }
        }
    }
    acc
}

fn eval_node_base(
    node: &DistNode,
    metrics: HeightfieldMetrics,
    ctx: &DistBakeContext<'_>,
) -> MaskField {
    let w = metrics.width;
    let h = metrics.height;
    match &node.kind {
        DistNodeKind::Fill { value } => MaskField::filled(metrics, value.clamp(0.0, 1.0)),
        DistNodeKind::MaskAsset { mask } => eval_mask_ref(mask, metrics, ctx),
        DistNodeKind::Noise { seed, frequency } => {
            let mut out = MaskField::zeros(metrics);
            let freq = frequency.max(1e-6);
            for j in 0..h {
                for i in 0..w {
                    let x = i as f32 * freq;
                    let y = j as f32 * freq;
                    let n = hash_noise(*seed, x, y);
                    out.set(i, j, n);
                }
            }
            out
        }
        DistNodeKind::NoisePerlin {
            seed,
            frequency,
            octaves,
        } => {
            let mut out = MaskField::zeros(metrics);
            for j in 0..h {
                for i in 0..w {
                    let n = perlin_fbm01(*seed, *frequency, *octaves, i as f32, j as f32);
                    out.set(i, j, n);
                }
            }
            out
        }
        DistNodeKind::NoiseRidged {
            seed,
            frequency,
            octaves,
        } => {
            let mut out = MaskField::zeros(metrics);
            for j in 0..h {
                for i in 0..w {
                    let n = ridged_fbm01(*seed, *frequency, *octaves, i as f32, j as f32);
                    out.set(i, j, n);
                }
            }
            out
        }
        DistNodeKind::NoiseWorley { seed, frequency } => {
            let mut out = MaskField::zeros(metrics);
            for j in 0..h {
                for i in 0..w {
                    let n = worley01(*seed, *frequency, i as f32, j as f32);
                    out.set(i, j, n);
                }
            }
            out
        }
        DistNodeKind::NoiseBillow {
            seed,
            frequency,
            octaves,
        } => {
            let mut out = MaskField::zeros(metrics);
            for j in 0..h {
                for i in 0..w {
                    let n = billow_fbm01(*seed, *frequency, *octaves, i as f32, j as f32);
                    out.set(i, j, n);
                }
            }
            out
        }
        DistNodeKind::Height { min, max } => {
            let Some(hf) = ctx.height else {
                return MaskField::ones(metrics);
            };
            let mut out = MaskField::zeros(metrics);
            let span = (max - min).max(1e-5);
            for j in 0..h {
                for i in 0..w {
                    let z = hf.get(i, j);
                    let t = ((z - min) / span).clamp(0.0, 1.0);
                    out.set(i, j, t);
                }
            }
            out
        }
        DistNodeKind::Slope { min_deg, max_deg } => {
            let Some(slope) = slope_deg_field(metrics, ctx) else {
                return MaskField::ones(metrics);
            };
            map_span_field(metrics, &slope, *min_deg, *max_deg)
        }
        // Same underlying slope calculation as `Slope`; kept as a distinct,
        // more descriptively named node for UI purposes.
        DistNodeKind::Steepness { min_deg, max_deg } => {
            let Some(slope) = slope_deg_field(metrics, ctx) else {
                return MaskField::ones(metrics);
            };
            map_span_field(metrics, &slope, *min_deg, *max_deg)
        }
        DistNodeKind::Angle { degrees, spread } => {
            let Some(hf) = ctx.height else {
                return MaskField::ones(metrics);
            };
            let aspect = aspect_deg_field(metrics, hf);
            let mut out = MaskField::zeros(metrics);
            let spread = spread.max(1e-3);
            for j in 0..h {
                for i in 0..w {
                    let idx = (j * w + i) as usize;
                    let a = aspect[idx];
                    let mut d = (a - degrees).abs() % 360.0;
                    if d > 180.0 {
                        d = 360.0 - d;
                    }
                    let t = (1.0 - d / spread).clamp(0.0, 1.0);
                    out.set(i, j, t);
                }
            }
            out
        }
        DistNodeKind::Roughness { radius, strength } => {
            let Some(hf) = ctx.height else {
                return MaskField::ones(metrics);
            };
            let r = (*radius).max(1) as i32;
            let mut out = MaskField::zeros(metrics);
            for j in 0..h as i32 {
                for i in 0..w as i32 {
                    let mut sum = 0.0f32;
                    let mut sum_sq = 0.0f32;
                    let mut count = 0.0f32;
                    for dj in -r..=r {
                        for di in -r..=r {
                            let x = (i + di).clamp(0, w as i32 - 1) as u32;
                            let y = (j + dj).clamp(0, h as i32 - 1) as u32;
                            let z = hf.get(x, y);
                            sum += z;
                            sum_sq += z * z;
                            count += 1.0;
                        }
                    }
                    let mean = sum / count;
                    let variance = (sum_sq / count - mean * mean).max(0.0);
                    let v = (variance.sqrt() * strength).clamp(0.0, 1.0);
                    out.set(i as u32, j as u32, v);
                }
            }
            out
        }
        DistNodeKind::Rocks { density, threshold } => {
            let Some(slope) = slope_deg_field(metrics, ctx) else {
                return MaskField::ones(metrics);
            };
            let mut out = MaskField::zeros(metrics);
            let freq = density.max(1e-4);
            let thresh = threshold.clamp(0.0, 0.999);
            for j in 0..h {
                for i in 0..w {
                    let idx = (j * w + i) as usize;
                    let slope_mask = (slope[idx] / 90.0).clamp(0.0, 1.0);
                    let n = hash_noise(1337, i as f32 * freq, j as f32 * freq);
                    let gated = if n > thresh {
                        (n - thresh) / (1.0 - thresh)
                    } else {
                        0.0
                    };
                    out.set(i, j, (gated * slope_mask).clamp(0.0, 1.0));
                }
            }
            out
        }
        DistNodeKind::RockyEdges { width, strength } => {
            let Some(slope) = slope_deg_field(metrics, ctx) else {
                return MaskField::ones(metrics);
            };
            let wdt = width.max(1e-3);
            let threshold = (90.0 - wdt).max(0.0);
            let steep: Vec<f32> = slope
                .iter()
                .map(|s| ((s - threshold) / wdt).clamp(0.0, 1.0))
                .collect();
            let edges = sobel_edge_magnitude(&steep, w, h);
            let mut out = MaskField::zeros(metrics);
            for j in 0..h {
                for i in 0..w {
                    let idx = (j * w + i) as usize;
                    out.set(i, j, (edges[idx] * strength).clamp(0.0, 1.0));
                }
            }
            out
        }
        DistNodeKind::Curvature { min, max } => {
            let Some(curv) = ctx.curvature else {
                return MaskField::ones(metrics);
            };
            let mut out = MaskField::zeros(metrics);
            let span = (max - min).max(1e-5);
            for j in 0..h {
                for i in 0..w {
                    let idx = (j as usize) * (w as usize) + (i as usize);
                    let c = curv.get(idx).copied().unwrap_or(0.0);
                    out.set(i, j, ((c - min) / span).clamp(0.0, 1.0));
                }
            }
            out
        }
        DistNodeKind::Cavity { strength } => {
            // Approximate cavity as inverted local curvature when available.
            if let Some(curv) = ctx.curvature {
                let mut out = MaskField::zeros(metrics);
                for j in 0..h {
                    for i in 0..w {
                        let idx = (j as usize) * (w as usize) + (i as usize);
                        let c = curv.get(idx).copied().unwrap_or(0.0);
                        let v = ((-c) * strength).clamp(0.0, 1.0);
                        out.set(i, j, v);
                    }
                }
                out
            } else {
                MaskField::ones(metrics)
            }
        }
        DistNodeKind::Flow { min, max } => {
            let Some(flow) = ctx.flow else {
                return MaskField::ones(metrics);
            };
            let mut out = MaskField::zeros(metrics);
            let span = (max - min).max(1e-5);
            for j in 0..h {
                for i in 0..w {
                    let idx = (j as usize) * (w as usize) + (i as usize);
                    let f = flow.get(idx).copied().unwrap_or(0.0);
                    out.set(i, j, ((f - min) / span).clamp(0.0, 1.0));
                }
            }
            out
        }
        DistNodeKind::SeaLevel { level, width } => {
            let Some(hf) = ctx.height else {
                return MaskField::ones(metrics);
            };
            let mut out = MaskField::zeros(metrics);
            let wdt = width.max(1e-5);
            for j in 0..h {
                for i in 0..w {
                    let z = hf.get(i, j);
                    let t = (1.0 - ((z - level).abs() / wdt)).clamp(0.0, 1.0);
                    out.set(i, j, t);
                }
            }
            out
        }
        DistNodeKind::Occlusion { radius, strength } => {
            let Some(hf) = ctx.height else {
                return MaskField::ones(metrics);
            };
            crate::analyze::ambient_occlusion(hf, *radius, *strength)
        }
        DistNodeKind::Paint { mask } | DistNodeKind::ImportedMask { mask } => {
            eval_mask_ref(mask, metrics, ctx)
        }
        DistNodeKind::Polygon { points, soft } => eval_polygon(metrics, points, *soft),
        DistNodeKind::Spline { points, width } => eval_spline(metrics, points, *width),
        DistNodeKind::Distance { mask, max_distance } => {
            eval_distance(metrics, mask, *max_distance, ctx)
        }
        DistNodeKind::Climate { channel } => {
            if let Some(aux) = ctx.aux {
                if let Some(field) = aux.get(channel.aux_key()) {
                    return field.clone();
                }
            }
            // No climate data yet: full coverage so region masks stay identity-safe.
            MaskField::ones(metrics)
        }
        DistNodeKind::Voronoi {
            seed,
            frequency,
            edge_weight,
        } => {
            let mut out = MaskField::zeros(metrics);
            let edge = edge_weight.clamp(0.0, 1.0);
            for j in 0..h {
                for i in 0..w {
                    let f1 = worley01(*seed, *frequency, i as f32, j as f32);
                    let fill = 1.0 - f1;
                    let edges = (1.0 - (f1 * 4.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);
                    out.set(i, j, (fill * (1.0 - edge) + edges * edge).clamp(0.0, 1.0));
                }
            }
            out
        }
        // Effects as top-level nodes: treat as identity base (children unused).
        // Region Mask Editor uses `evaluate_op_stack` so top-level effects apply
        // to the accumulator instead of this identity stub.
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
        | DistNodeKind::EffectErode { .. } => MaskField::ones(metrics),
        DistNodeKind::GroupAll => fold_placement_group(node, metrics, ctx, true),
        DistNodeKind::GroupAny => fold_placement_group(node, metrics, ctx, false),
    }
}

fn fold_placement_group(
    node: &DistNode,
    metrics: HeightfieldMetrics,
    ctx: &DistBakeContext<'_>,
    all: bool,
) -> MaskField {
    let mut acc = if all {
        MaskField::ones(metrics)
    } else {
        MaskField::zeros(metrics)
    };
    for child in &node.children {
        if !child.enabled {
            continue;
        }
        let field = eval_child_field(child, metrics, ctx);
        for j in 0..metrics.height {
            for i in 0..metrics.width {
                let a = acc.get(i, j);
                let b = (field.get(i, j) * child.opacity).clamp(0.0, 1.0);
                let v = if all {
                    MaskCombine::Multiply.apply(a, b)
                } else {
                    MaskCombine::Max.apply(a, b)
                };
                acc.set(i, j, v);
            }
        }
    }
    acc
}

/// Evaluate a child node (base + its nested children) without sibling combine-from-ones.
fn eval_child_field(
    node: &DistNode,
    metrics: HeightfieldMetrics,
    ctx: &DistBakeContext<'_>,
) -> MaskField {
    let mut field = eval_node_base(node, metrics, ctx);
    if node.children.is_empty() {
        return field;
    }
    if node.kind.is_placement_group() {
        return field;
    }
    let all_effects = node.children.iter().all(|c| c.kind.is_effect());
    if all_effects {
        for c in &node.children {
            if c.enabled {
                field = apply_effect(&c.kind, &field, c.opacity);
            }
        }
    } else {
        let child_baked = bake_dist_nodes(&node.children, metrics, ctx);
        for j in 0..metrics.height {
            for i in 0..metrics.width {
                let v = field.get(i, j) * child_baked.get(i, j);
                field.set(i, j, v.clamp(0.0, 1.0));
            }
        }
    }
    field
}

/// Public wrapper for region-mask sequential op evaluation.
pub fn bake_dist_node_base(
    node: &DistNode,
    metrics: HeightfieldMetrics,
    ctx: &DistBakeContext<'_>,
) -> MaskField {
    eval_node_base(node, metrics, ctx)
}

/// Public wrapper so region mask op stacks can apply effects to an accumulator.
pub fn apply_effect_public(kind: &DistNodeKind, input: &MaskField, opacity: f32) -> MaskField {
    apply_effect(kind, input, opacity)
}

fn eval_mask_ref(
    mask: &MaskRef,
    metrics: HeightfieldMetrics,
    ctx: &DistBakeContext<'_>,
) -> MaskField {
    let field = ctx
        .masks
        .get(&mask.id)
        .cloned()
        .unwrap_or_else(|| MaskField::ones(metrics));
    let mut out = MaskField::zeros(metrics);
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            let mut v = field.get(i, j) * mask.strength;
            if mask.invert {
                v = 1.0 - v;
            }
            out.set(i, j, v.clamp(0.0, 1.0));
        }
    }
    out
}

fn eval_polygon(metrics: HeightfieldMetrics, points: &[[f32; 2]], soft: f32) -> MaskField {
    let mut out = MaskField::zeros(metrics);
    if points.len() < 3 {
        return MaskField::ones(metrics);
    }
    let soft = soft.max(0.0);
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            let u = (i as f32 + 0.5) / metrics.width.max(1) as f32;
            let v = (j as f32 + 0.5) / metrics.height.max(1) as f32;
            let inside = point_in_polygon(u, v, points);
            if soft <= 1e-6 {
                out.set(i, j, if inside { 1.0 } else { 0.0 });
            } else {
                let d = distance_to_polygon_edge(u, v, points);
                let t = if inside {
                    1.0
                } else {
                    (1.0 - d / soft).clamp(0.0, 1.0)
                };
                out.set(i, j, t);
            }
        }
    }
    out
}

fn eval_spline(metrics: HeightfieldMetrics, points: &[[f32; 2]], width: f32) -> MaskField {
    let mut out = MaskField::zeros(metrics);
    if points.len() < 2 {
        return out;
    }
    let half = width.max(1e-5);
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            let u = (i as f32 + 0.5) / metrics.width.max(1) as f32;
            let v = (j as f32 + 0.5) / metrics.height.max(1) as f32;
            let d = distance_to_polyline(u, v, points);
            out.set(i, j, (1.0 - d / half).clamp(0.0, 1.0));
        }
    }
    out
}

fn eval_distance(
    metrics: HeightfieldMetrics,
    mask: &MaskRef,
    max_distance: f32,
    ctx: &DistBakeContext<'_>,
) -> MaskField {
    let core = eval_mask_ref(mask, metrics, ctx);
    let max_d = max_distance.max(1.0);
    let w = metrics.width as i32;
    let h = metrics.height as i32;
    let r = max_d.ceil() as i32;
    let mut out = MaskField::zeros(metrics);
    for j in 0..h {
        for i in 0..w {
            if core.get(i as u32, j as u32) > 0.5 {
                out.set(i as u32, j as u32, 1.0);
                continue;
            }
            let mut best = max_d;
            for dj in -r..=r {
                for di in -r..=r {
                    let x = i + di;
                    let y = j + dj;
                    if x < 0 || y < 0 || x >= w || y >= h {
                        continue;
                    }
                    if core.get(x as u32, y as u32) > 0.5 {
                        let dist = ((di * di + dj * dj) as f32).sqrt();
                        if dist < best {
                            best = dist;
                        }
                    }
                }
            }
            out.set(i as u32, j as u32, (1.0 - best / max_d).clamp(0.0, 1.0));
        }
    }
    out
}

fn distance_to_polygon_edge(x: f32, y: f32, pts: &[[f32; 2]]) -> f32 {
    let mut best = f32::MAX;
    let n = pts.len();
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        best = best.min(dist_point_segment(x, y, a[0], a[1], b[0], b[1]));
    }
    best
}

fn distance_to_polyline(x: f32, y: f32, pts: &[[f32; 2]]) -> f32 {
    let mut best = f32::MAX;
    for i in 0..pts.len().saturating_sub(1) {
        let a = pts[i];
        let b = pts[i + 1];
        best = best.min(dist_point_segment(x, y, a[0], a[1], b[0], b[1]));
    }
    best
}

fn apply_effect(kind: &DistNodeKind, input: &MaskField, opacity: f32) -> MaskField {
    let metrics = input.metrics;
    let w = metrics.width;
    let h = metrics.height;
    let mut out = input.clone();
    match kind {
        DistNodeKind::EffectInvert => {
            for j in 0..h {
                for i in 0..w {
                    let v = 1.0 - input.get(i, j);
                    let blended = input.get(i, j) * (1.0 - opacity) + v * opacity;
                    out.set(i, j, blended.clamp(0.0, 1.0));
                }
            }
        }
        DistNodeKind::EffectLevels {
            in_black,
            in_white,
            gamma,
        } => {
            let black = *in_black;
            let white = (*in_white).max(black + 1e-6);
            let g = (*gamma).max(1e-6);
            for j in 0..h {
                for i in 0..w {
                    let x = input.get(i, j);
                    let t = ((x - black) / (white - black)).clamp(0.0, 1.0);
                    let leveled = t.powf(1.0 / g);
                    let v = x * (1.0 - opacity) + leveled * opacity;
                    out.set(i, j, v.clamp(0.0, 1.0));
                }
            }
        }
        DistNodeKind::EffectBlur { radius } => {
            let r = (*radius).max(1);
            let tmp = box_blur(input, r);
            for j in 0..h {
                for i in 0..w {
                    let v = input.get(i, j) * (1.0 - opacity) + tmp.get(i, j) * opacity;
                    out.set(i, j, v.clamp(0.0, 1.0));
                }
            }
        }
        DistNodeKind::EffectDilate { radius_m } => {
            let r = meters_to_radius_samples(input.metrics, *radius_m);
            let tmp = morph_dilate(input, r);
            for j in 0..h {
                for i in 0..w {
                    let v = input.get(i, j) * (1.0 - opacity) + tmp.get(i, j) * opacity;
                    out.set(i, j, v.clamp(0.0, 1.0));
                }
            }
        }
        DistNodeKind::EffectErode { radius_m } => {
            let r = meters_to_radius_samples(input.metrics, *radius_m);
            let tmp = morph_erode(input, r);
            for j in 0..h {
                for i in 0..w {
                    let v = input.get(i, j) * (1.0 - opacity) + tmp.get(i, j) * opacity;
                    out.set(i, j, v.clamp(0.0, 1.0));
                }
            }
        }
        DistNodeKind::EffectRemap { in_min, in_max } => {
            let span = (in_max - in_min).max(1e-5);
            for j in 0..h {
                for i in 0..w {
                    let t = ((input.get(i, j) - in_min) / span).clamp(0.0, 1.0);
                    let v = input.get(i, j) * (1.0 - opacity) + t * opacity;
                    out.set(i, j, v.clamp(0.0, 1.0));
                }
            }
        }
        DistNodeKind::EffectContrast { amount } => {
            let a = amount.max(0.0);
            for j in 0..h {
                for i in 0..w {
                    let x = input.get(i, j);
                    let t = ((x - 0.5) * a + 0.5).clamp(0.0, 1.0);
                    let v = x * (1.0 - opacity) + t * opacity;
                    out.set(i, j, v.clamp(0.0, 1.0));
                }
            }
        }
        DistNodeKind::EffectClamp { min, max } => {
            for j in 0..h {
                for i in 0..w {
                    let t = input.get(i, j).clamp(*min, *max);
                    let v = input.get(i, j) * (1.0 - opacity) + t * opacity;
                    out.set(i, j, v.clamp(0.0, 1.0));
                }
            }
        }
        DistNodeKind::EffectCurve { contrast } => {
            for j in 0..h {
                for i in 0..w {
                    let x = input.get(i, j);
                    let t = soft_s_curve(x, *contrast);
                    let v = x * (1.0 - opacity) + t * opacity;
                    out.set(i, j, v.clamp(0.0, 1.0));
                }
            }
        }
        DistNodeKind::EffectDistortion {
            seed,
            frequency,
            amount,
        } => {
            let freq = frequency.max(1e-6);
            for j in 0..h {
                for i in 0..w {
                    let nx = hash_noise(*seed, i as f32 * freq, j as f32 * freq) * 2.0 - 1.0;
                    let ny = hash_noise(seed.wrapping_add(1), i as f32 * freq, j as f32 * freq)
                        * 2.0
                        - 1.0;
                    let si = (i as f32 + nx * amount).round().clamp(0.0, (w - 1) as f32) as u32;
                    let sj = (j as f32 + ny * amount).round().clamp(0.0, (h - 1) as f32) as u32;
                    let t = input.get(si, sj);
                    let v = input.get(i, j) * (1.0 - opacity) + t * opacity;
                    out.set(i, j, v.clamp(0.0, 1.0));
                }
            }
        }
        DistNodeKind::EffectEdge { strength } => {
            let edges = sobel_edge_magnitude(input.data(), w, h);
            for j in 0..h {
                for i in 0..w {
                    let idx = (j * w + i) as usize;
                    let t = (edges[idx] * strength).clamp(0.0, 1.0);
                    let v = input.get(i, j) * (1.0 - opacity) + t * opacity;
                    out.set(i, j, v.clamp(0.0, 1.0));
                }
            }
        }
        DistNodeKind::EffectSmoothstep { edge0, edge1 } => {
            let span = (edge1 - edge0).max(1e-5);
            for j in 0..h {
                for i in 0..w {
                    let x = input.get(i, j);
                    let t = ((x - edge0) / span).clamp(0.0, 1.0);
                    let s = t * t * (3.0 - 2.0 * t);
                    let v = x * (1.0 - opacity) + s * opacity;
                    out.set(i, j, v.clamp(0.0, 1.0));
                }
            }
        }
        DistNodeKind::EffectSimpleFlow {
            iterations,
            strength,
        } => {
            let iters = (*iterations).max(1);
            let amt = strength.clamp(0.0, 1.0);
            let mut cur = input.clone();
            for _ in 0..iters {
                let mut next = cur.clone();
                for j in 0..h as i32 {
                    for i in 0..w as i32 {
                        let mut m = cur.get(i as u32, j as u32);
                        for (di, dj) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                            let x = (i + di).clamp(0, w as i32 - 1) as u32;
                            let y = (j + dj).clamp(0, h as i32 - 1) as u32;
                            m = m.max(cur.get(x, y));
                        }
                        let v = cur.get(i as u32, j as u32) * (1.0 - amt) + m * amt;
                        next.set(i as u32, j as u32, v.clamp(0.0, 1.0));
                    }
                }
                cur = next;
            }
            for j in 0..h {
                for i in 0..w {
                    let v = input.get(i, j) * (1.0 - opacity) + cur.get(i, j) * opacity;
                    out.set(i, j, v.clamp(0.0, 1.0));
                }
            }
        }
        _ => {}
    }
    out
}

fn box_blur(input: &MaskField, radius: u32) -> MaskField {
    let metrics = input.metrics;
    let w = metrics.width as i32;
    let h = metrics.height as i32;
    let r = radius as i32;
    let mut out = MaskField::zeros(metrics);
    for j in 0..h {
        for i in 0..w {
            let mut sum = 0.0f32;
            let mut count = 0u32;
            for dj in -r..=r {
                for di in -r..=r {
                    let x = (i + di).clamp(0, w - 1) as u32;
                    let y = (j + dj).clamp(0, h - 1) as u32;
                    sum += input.get(x, y);
                    count += 1;
                }
            }
            out.set(i as u32, j as u32, sum / count.max(1) as f32);
        }
    }
    out
}

fn meters_to_radius_samples(metrics: HeightfieldMetrics, radius_m: f32) -> u32 {
    let cell = metrics.dx().max(1e-3);
    ((radius_m.max(0.0) / cell).round() as u32).max(1)
}

fn morph_dilate(input: &MaskField, radius: u32) -> MaskField {
    let metrics = input.metrics;
    let w = metrics.width as i32;
    let h = metrics.height as i32;
    let r = radius as i32;
    let mut out = MaskField::zeros(metrics);
    for j in 0..h {
        for i in 0..w {
            let mut best = 0.0f32;
            for dj in -r..=r {
                for di in -r..=r {
                    if di * di + dj * dj > r * r {
                        continue;
                    }
                    let x = (i + di).clamp(0, w - 1) as u32;
                    let y = (j + dj).clamp(0, h - 1) as u32;
                    best = best.max(input.get(x, y));
                }
            }
            out.set(i as u32, j as u32, best);
        }
    }
    out
}

fn morph_erode(input: &MaskField, radius: u32) -> MaskField {
    let metrics = input.metrics;
    let w = metrics.width as i32;
    let h = metrics.height as i32;
    let r = radius as i32;
    let mut out = MaskField::zeros(metrics);
    for j in 0..h {
        for i in 0..w {
            let mut best = 1.0f32;
            for dj in -r..=r {
                for di in -r..=r {
                    if di * di + dj * dj > r * r {
                        continue;
                    }
                    let x = (i + di).clamp(0, w - 1) as u32;
                    let y = (j + dj).clamp(0, h - 1) as u32;
                    best = best.min(input.get(x, y));
                }
            }
            out.set(i as u32, j as u32, best);
        }
    }
    out
}

/// Slope in degrees per pixel, from `ctx.slope_deg` if provided, else
/// approximated from height neighbours. Shared by `Slope`, `Steepness`,
/// `Rocks` and `RockyEdges`.
fn slope_deg_field(metrics: HeightfieldMetrics, ctx: &DistBakeContext<'_>) -> Option<Vec<f32>> {
    let w = metrics.width;
    let h = metrics.height;
    if let Some(slope) = ctx.slope_deg {
        return Some(slope.to_vec());
    }
    let hf = ctx.height?;
    let cell = metrics.dx().max(1e-5);
    let mut out = vec![0.0f32; (w * h) as usize];
    for j in 0..h {
        for i in 0..w {
            let z = hf.get(i, j);
            let zx = if i + 1 < w { hf.get(i + 1, j) } else { z };
            let zy = if j + 1 < h { hf.get(i, j + 1) } else { z };
            let gx = (zx - z) / cell;
            let gy = (zy - z) / cell;
            let deg = (gx * gx + gy * gy).sqrt().atan().to_degrees();
            out[(j * w + i) as usize] = deg;
        }
    }
    Some(out)
}

/// Aspect (slope-facing direction) in degrees \[0, 360), from height neighbours.
fn aspect_deg_field(metrics: HeightfieldMetrics, hf: &Heightfield) -> Vec<f32> {
    let w = metrics.width;
    let h = metrics.height;
    let cell = metrics.dx().max(1e-5);
    let mut out = vec![0.0f32; (w * h) as usize];
    for j in 0..h {
        for i in 0..w {
            let zx0 = if i > 0 {
                hf.get(i - 1, j)
            } else {
                hf.get(i, j)
            };
            let zx1 = if i + 1 < w {
                hf.get(i + 1, j)
            } else {
                hf.get(i, j)
            };
            let zy0 = if j > 0 {
                hf.get(i, j - 1)
            } else {
                hf.get(i, j)
            };
            let zy1 = if j + 1 < h {
                hf.get(i, j + 1)
            } else {
                hf.get(i, j)
            };
            let gx = (zx1 - zx0) / (2.0 * cell);
            let gy = (zy1 - zy0) / (2.0 * cell);
            let mut deg = gy.atan2(gx).to_degrees();
            if deg < 0.0 {
                deg += 360.0;
            }
            out[(j * w + i) as usize] = deg;
        }
    }
    out
}

/// Linearly maps a per-pixel value field from `[min, max]` into a `[0,1]` mask.
fn map_span_field(metrics: HeightfieldMetrics, values: &[f32], min: f32, max: f32) -> MaskField {
    let w = metrics.width;
    let h = metrics.height;
    let mut out = MaskField::zeros(metrics);
    let span = (max - min).max(1e-5);
    for j in 0..h {
        for i in 0..w {
            let idx = (j * w + i) as usize;
            let v = values.get(idx).copied().unwrap_or(0.0);
            out.set(i, j, ((v - min) / span).clamp(0.0, 1.0));
        }
    }
    out
}

/// Sobel-ish gradient magnitude of a dense `w*h` field, clamped-edge sampled.
fn sobel_edge_magnitude(values: &[f32], w: u32, h: u32) -> Vec<f32> {
    let wi = w as i32;
    let hi = h as i32;
    let sample = |i: i32, j: i32| -> f32 {
        let x = i.clamp(0, wi - 1) as u32;
        let y = j.clamp(0, hi - 1) as u32;
        values[(y * w + x) as usize]
    };
    let mut out = vec![0.0f32; (w * h) as usize];
    for j in 0..hi {
        for i in 0..wi {
            let gx = sample(i + 1, j) - sample(i - 1, j);
            let gy = sample(i, j + 1) - sample(i, j - 1);
            out[(j as u32 * w + i as u32) as usize] = (gx * gx + gy * gy).sqrt();
        }
    }
    out
}

/// Soft S-curve: blends `x` toward a smoothstep of itself by `contrast`
/// (0 = linear/no change, larger values push harder toward 0/1).
fn soft_s_curve(x: f32, contrast: f32) -> f32 {
    let c = contrast.max(0.0);
    let t = x.clamp(0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    t + (s - t) * (c / (1.0 + c))
}

/// Normalized (\[0,1\]) Perlin fBm.
fn perlin_fbm01(seed: u64, frequency: f32, octaves: u32, x: f32, y: f32) -> f32 {
    let octaves = octaves.max(1);
    let mut amp = 1.0f32;
    let mut freq = frequency.max(1e-6);
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    for o in 0..octaves {
        let n = perlin2(x * freq, y * freq, seed.wrapping_add(o as u64 * 1013));
        sum += n * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    let v = if norm > 0.0 { sum / norm } else { 0.0 };
    (v * 0.5 + 0.5).clamp(0.0, 1.0)
}

/// Normalized (\[0,1\]) ridged multifractal noise.
fn ridged_fbm01(seed: u64, frequency: f32, octaves: u32, x: f32, y: f32) -> f32 {
    let octaves = octaves.max(1);
    let mut amp = 1.0f32;
    let mut freq = frequency.max(1e-6);
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    let mut weight = 1.0f32;
    for o in 0..octaves {
        let mut n = perlin2(x * freq, y * freq, seed.wrapping_add(o as u64 * 9173));
        n = 1.0 - n.abs();
        n *= n;
        n *= weight;
        weight = (n * 2.0).clamp(0.0, 1.0);
        sum += n * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    if norm > 0.0 {
        (sum / norm).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Normalized (\[0,1\]) billowy (absolute-value) fBm.
fn billow_fbm01(seed: u64, frequency: f32, octaves: u32, x: f32, y: f32) -> f32 {
    let octaves = octaves.max(1);
    let mut amp = 1.0f32;
    let mut freq = frequency.max(1e-6);
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    for o in 0..octaves {
        let n = perlin2(x * freq, y * freq, seed.wrapping_add(o as u64 * 6151)).abs();
        sum += n * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    if norm > 0.0 {
        (sum / norm).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Normalized (\[0,1\]) Worley/cellular F1 distance.
fn worley01(seed: u64, frequency: f32, x: f32, y: f32) -> f32 {
    let freq = frequency.max(1e-6);
    let r = worley2(x * freq, y * freq, seed, WorleyMetric::Euclidean);
    (r.f1 / 1.2).clamp(0.0, 1.0)
}

fn hash_noise(seed: u64, x: f32, y: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;
    let a = hash2(seed, ix, iy);
    let b = hash2(seed, ix + 1, iy);
    let c = hash2(seed, ix, iy + 1);
    let d = hash2(seed, ix + 1, iy + 1);
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let ab = a + (b - a) * ux;
    let cd = c + (d - c) * ux;
    ab + (cd - ab) * uy
}

fn hash2(seed: u64, x: i32, y: i32) -> f32 {
    let mut n = seed
        .wrapping_add(x as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(y as u64)
        .wrapping_mul(0xBF58476D1CE4E5B9);
    n ^= n >> 30;
    n = n.wrapping_mul(0x94D049BB133111EB);
    n ^= n >> 31;
    (n as f32) / (u64::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn slope_times_height_with_blur() {
        let metrics = HeightfieldMetrics::new(16, 16, 100.0, 100.0);
        let mut hf = Heightfield::zeros(metrics);
        for j in 0..16u32 {
            for i in 0..16u32 {
                hf.set(i, j, i as f32 * 10.0 + j as f32);
            }
        }
        let masks = HashMap::new();
        let ctx = DistBakeContext {
            height: Some(&hf),
            slope_deg: None,
            curvature: None,
            flow: None,
            masks: &masks,
            aux: None,
        };
        let mut slope = DistNode::slope(0.0, 45.0);
        slope
            .children
            .push(DistNode::new(DistNodeKind::EffectBlur { radius: 1 }));
        let height = DistNode::height(0.0, 200.0);
        height_combine_test(metrics, &ctx, slope, height);
    }

    fn height_combine_test(
        metrics: HeightfieldMetrics,
        ctx: &DistBakeContext<'_>,
        mut slope: DistNode,
        height: DistNode,
    ) {
        slope.combine = MaskCombine::Multiply;
        let nodes = vec![slope, height];
        let baked = bake_dist_nodes(&nodes, metrics, ctx);
        assert!(baked.get(8, 8) >= 0.0 && baked.get(8, 8) <= 1.0);
    }

    fn bumpy_heightfield(metrics: HeightfieldMetrics) -> Heightfield {
        let mut hf = Heightfield::zeros(metrics);
        for j in 0..metrics.height {
            for i in 0..metrics.width {
                let x = i as f32;
                let y = j as f32;
                hf.set(
                    i,
                    j,
                    (x * 0.4).sin() * 10.0 + (y * 0.3).cos() * 6.0 + x * 0.5,
                );
            }
        }
        hf
    }

    #[test]
    fn occlusion_uses_ambient_occlusion_when_height_present() {
        let metrics = HeightfieldMetrics::new(16, 16, 100.0, 100.0);
        let hf = bumpy_heightfield(metrics);
        let masks = HashMap::new();
        let ctx = DistBakeContext {
            height: Some(&hf),
            slope_deg: None,
            curvature: None,
            flow: None,
            masks: &masks,
            aux: None,
        };
        let node = DistNode::new(DistNodeKind::Occlusion {
            radius: 2,
            strength: 1.0,
        });
        let field = eval_node_base(&node, metrics, &ctx);
        let expected = crate::analyze::ambient_occlusion(&hf, 2, 1.0);
        for j in 0..metrics.height {
            for i in 0..metrics.width {
                assert_eq!(field.get(i, j), expected.get(i, j));
            }
        }
    }

    #[test]
    fn occlusion_without_height_is_ones() {
        let metrics = HeightfieldMetrics::new(8, 8, 50.0, 50.0);
        let masks = HashMap::new();
        let ctx = DistBakeContext::masks_only(&masks);
        let node = DistNode::new(DistNodeKind::Occlusion {
            radius: 3,
            strength: 1.0,
        });
        let field = eval_node_base(&node, metrics, &ctx);
        for j in 0..metrics.height {
            for i in 0..metrics.width {
                assert_eq!(field.get(i, j), 1.0);
            }
        }
    }

    #[test]
    fn noise_kinds_produce_finite_values_in_range() {
        let metrics = HeightfieldMetrics::new(20, 20, 100.0, 100.0);
        let masks = HashMap::new();
        let ctx = DistBakeContext::masks_only(&masks);
        let kinds = [
            DistNodeKind::NoisePerlin {
                seed: 7,
                frequency: 0.1,
                octaves: 4,
            },
            DistNodeKind::NoiseRidged {
                seed: 11,
                frequency: 0.08,
                octaves: 3,
            },
            DistNodeKind::NoiseWorley {
                seed: 3,
                frequency: 0.15,
            },
            DistNodeKind::NoiseBillow {
                seed: 42,
                frequency: 0.12,
                octaves: 3,
            },
        ];
        for kind in kinds {
            let node = DistNode::new(kind);
            let field = eval_node_base(&node, metrics, &ctx);
            for j in 0..metrics.height {
                for i in 0..metrics.width {
                    let v = field.get(i, j);
                    assert!(v.is_finite(), "value at ({i},{j}) was not finite");
                    assert!(
                        (0.0..=1.0).contains(&v),
                        "value {v} at ({i},{j}) out of range"
                    );
                }
            }
        }
    }

    #[test]
    fn noise_kinds_are_deterministic_for_same_seed() {
        let metrics = HeightfieldMetrics::new(12, 12, 60.0, 60.0);
        let masks = HashMap::new();
        let ctx = DistBakeContext::masks_only(&masks);
        let node = DistNode::new(DistNodeKind::NoiseWorley {
            seed: 99,
            frequency: 0.2,
        });
        let a = eval_node_base(&node, metrics, &ctx);
        let b = eval_node_base(&node, metrics, &ctx);
        assert_eq!(a.data(), b.data());
    }

    #[test]
    fn new_terrain_and_effect_nodes_bake_finite_masks() {
        let metrics = HeightfieldMetrics::new(16, 16, 100.0, 100.0);
        let hf = bumpy_heightfield(metrics);
        let masks = HashMap::new();
        let ctx = DistBakeContext {
            height: Some(&hf),
            slope_deg: None,
            curvature: None,
            flow: None,
            masks: &masks,
            aux: None,
        };
        let mut steepness = DistNode::new(DistNodeKind::Steepness {
            min_deg: 0.0,
            max_deg: 45.0,
        });
        steepness
            .children
            .push(DistNode::new(DistNodeKind::EffectCurve { contrast: 2.0 }));
        steepness
            .children
            .push(DistNode::new(DistNodeKind::EffectSmoothstep {
                edge0: 0.1,
                edge1: 0.9,
            }));

        let mut rocks = DistNode::new(DistNodeKind::Rocks {
            density: 0.5,
            threshold: 0.6,
        });
        rocks
            .children
            .push(DistNode::new(DistNodeKind::EffectDistortion {
                seed: 5,
                frequency: 0.1,
                amount: 2.0,
            }));
        rocks
            .children
            .push(DistNode::new(DistNodeKind::EffectEdge { strength: 1.0 }));
        rocks
            .children
            .push(DistNode::new(DistNodeKind::EffectSimpleFlow {
                iterations: 3,
                strength: 0.5,
            }));

        let nodes = vec![
            steepness,
            DistNode::new(DistNodeKind::Angle {
                degrees: 90.0,
                spread: 60.0,
            }),
            DistNode::new(DistNodeKind::Roughness {
                radius: 1,
                strength: 0.1,
            }),
            rocks,
            DistNode::new(DistNodeKind::RockyEdges {
                width: 10.0,
                strength: 1.0,
            }),
        ];
        let baked = bake_dist_nodes(&nodes, metrics, &ctx);
        for j in 0..metrics.height {
            for i in 0..metrics.width {
                let v = baked.get(i, j);
                assert!(v.is_finite());
                assert!((0.0..=1.0).contains(&v));
            }
        }
    }
}
