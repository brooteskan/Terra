//! Research-backed, non-destructive terrain authoring operations.
//!
//! This module deliberately composes Terra's existing export-oracle hydrology
//! with semantic strokes, constraints and reconstruction. Authoring data stays
//! resolution independent; only evaluation rasterizes it.

use crate::field_data::keys;
use crate::heightfield::{Heightfield, HeightfieldMetrics, TileId};
use crate::hydro::{self, StreamPowerParams};
use crate::mask::{MaskField, MaskSource};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    /// Radial pinch toward stroke centre.
    Pinch,
    /// Radial inflate / bulge away from centre.
    Inflate,
    /// Soft erode (lower + encourage erosion field).
    Erode,
    /// Procedural noise under the brush.
    Noise,
    /// One-shot mountain-like ridge stamp.
    MountainStamp,
    /// One-shot valley stamp.
    ValleyStamp,
    /// Flattened plateau disk.
    PlateauStamp,
    /// Crater bowl (rim + depression).
    CraterStamp,
    /// Soft coastal lower / smooth.
    Coastline,
    /// River valley path brush.
    RiverPath,
    /// Absolute height stamp (uses `target_height`).
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SculptStrokeParams {
    #[serde(default)]
    pub strokes: Vec<SculptStroke>,
    #[serde(default = "sculpt_reconcile")]
    pub reconcile: f32,
}

fn sculpt_reconcile() -> f32 {
    0.15
}

impl Default for SculptStrokeParams {
    fn default() -> Self {
        Self {
            strokes: Vec::new(),
            reconcile: sculpt_reconcile(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerrainConstraintKind {
    Elevation,
    MinElevation,
    MaxElevation,
    Ridge,
    Valley,
    River,
    Coastline,
    Plateau,
    Cliff,
    PreferredSlope,
    Roughness,
    Outlet,
    Divide,
    Protect,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainConstraint {
    pub kind: TerrainConstraintKind,
    #[serde(default)]
    pub points: Vec<SculptPoint>,
    #[serde(default = "constraint_width")]
    pub width_m: f32,
    #[serde(default)]
    pub value: f32,
    #[serde(default = "one")]
    pub strength: f32,
}

fn constraint_width() -> f32 {
    120.0
}

impl Default for TerrainConstraint {
    fn default() -> Self {
        Self {
            kind: TerrainConstraintKind::Elevation,
            points: vec![SculptPoint::default()],
            width_m: constraint_width(),
            value: 50.0,
            strength: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainConstraintParams {
    #[serde(default)]
    pub constraints: Vec<TerrainConstraint>,
    #[serde(default = "constraint_preview")]
    pub preview_strength: f32,
}

fn constraint_preview() -> f32 {
    0.65
}

impl Default for TerrainConstraintParams {
    fn default() -> Self {
        Self {
            constraints: Vec::new(),
            preview_strength: constraint_preview(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientReconstructParams {
    #[serde(default = "poisson_iterations")]
    pub iterations: u32,
    #[serde(default = "poisson_screening")]
    pub screening: f32,
    #[serde(default = "poisson_constraints")]
    pub constraint_strength: f32,
    #[serde(default = "gradient_smoothing")]
    pub gradient_smoothing: f32,
}

fn poisson_iterations() -> u32 {
    80
}
fn poisson_screening() -> f32 {
    0.08
}
fn poisson_constraints() -> f32 {
    6.0
}
fn gradient_smoothing() -> f32 {
    0.2
}

impl Default for GradientReconstructParams {
    fn default() -> Self {
        Self {
            iterations: poisson_iterations(),
            screening: poisson_screening(),
            constraint_strength: poisson_constraints(),
            gradient_smoothing: gradient_smoothing(),
        }
    }
}

pub use crate::landscape_evolution::{
    BoundaryMode, EvolutionSolverMode, LandscapeEvolutionParams, UpliftMode,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydrologyRepairParams {
    #[serde(default = "repair_iterations")]
    pub iterations: u32,
    #[serde(default = "repair_incision")]
    pub incision: f32,
    #[serde(default = "repair_radius")]
    pub repair_radius_m: f32,
    #[serde(default = "one")]
    pub constraint_preservation: f32,
    #[serde(default = "stream_threshold")]
    pub stream_threshold: f32,
}

fn repair_iterations() -> u32 {
    8
}
fn repair_incision() -> f32 {
    0.018
}
fn repair_radius() -> f32 {
    300.0
}
fn stream_threshold() -> f32 {
    40.0
}

impl Default for HydrologyRepairParams {
    fn default() -> Self {
        Self {
            iterations: repair_iterations(),
            incision: repair_incision(),
            repair_radius_m: repair_radius(),
            constraint_preservation: 1.0,
            stream_threshold: stream_threshold(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeomorphicDetailParams {
    /// Peak meso amplitude in metres (legacy `amplitude` maps here when unset).
    #[serde(default = "detail_amplitude")]
    pub amplitude: f32,
    /// Legacy characteristic wavelength hint (metres); used when band overrides absent.
    #[serde(default = "detail_scale")]
    pub scale_m: f32,
    /// Cascade depth for structured patterns (narrower nests in broader).
    #[serde(default = "detail_octaves")]
    pub octaves: u32,
    #[serde(default = "flow_alignment")]
    pub flow_alignment: f32,
    /// Minimum normalised slope \[0,1\] (~degrees/90) before detail activates.
    #[serde(default = "slope_gate")]
    pub slope_gate: f32,
    #[serde(default = "detail_seed")]
    pub seed: u64,
    #[serde(default = "drainage_preservation")]
    pub preserve_drainage: f32,
    /// Micro-band amplitude in metres (gullies / breakup). 0 → derive from amplitude.
    #[serde(default)]
    pub micro_amplitude_m: Option<f32>,
    /// Macro silhouette lock (high-pass amplify delta).
    #[serde(default = "detail_silhouette")]
    pub silhouette_lock: f32,
    #[serde(default = "detail_ridge_breakup")]
    pub ridge_breakup: f32,
    #[serde(default = "detail_gully")]
    pub gully_strength: f32,
    #[serde(default = "detail_rock")]
    pub rock_roughness: f32,
}

fn detail_amplitude() -> f32 {
    14.0
}
fn detail_scale() -> f32 {
    72.0
}
fn detail_octaves() -> u32 {
    5
}
fn flow_alignment() -> f32 {
    0.9
}
fn slope_gate() -> f32 {
    0.08
}
fn detail_seed() -> u64 {
    73
}
fn drainage_preservation() -> f32 {
    0.65
}
fn detail_silhouette() -> f32 {
    0.9
}
fn detail_ridge_breakup() -> f32 {
    0.9
}
fn detail_gully() -> f32 {
    1.2
}
fn detail_rock() -> f32 {
    0.55
}

impl Default for GeomorphicDetailParams {
    fn default() -> Self {
        Self {
            amplitude: detail_amplitude(),
            scale_m: detail_scale(),
            octaves: detail_octaves(),
            flow_alignment: flow_alignment(),
            slope_gate: slope_gate(),
            seed: detail_seed(),
            preserve_drainage: drainage_preservation(),
            micro_amplitude_m: None,
            silhouette_lock: detail_silhouette(),
            ridge_breakup: detail_ridge_breakup(),
            gully_strength: detail_gully(),
            rock_roughness: detail_rock(),
        }
    }
}

impl GeomorphicDetailParams {
    /// Map authoring params onto the drainage-conditioned amplifier.
    pub fn to_amplification(&self) -> crate::analyze::TerrainAmplificationParams {
        let meso = self.amplitude.max(0.0);
        let micro = self
            .micro_amplitude_m
            .unwrap_or_else(|| (meso * 0.38).max(1.5));
        crate::analyze::TerrainAmplificationParams {
            meso_amplitude_m: meso,
            micro_amplitude_m: micro,
            cascade_levels: self.octaves.clamp(2, 6),
            flow_alignment: self.flow_alignment,
            slope_gate: self.slope_gate,
            preserve_drainage: self.preserve_drainage,
            silhouette_lock: self.silhouette_lock,
            ridge_breakup: self.ridge_breakup,
            gully_strength: self.gully_strength,
            rock_roughness: self.rock_roughness,
            bands: None,
            seed: self.seed,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcosystemFeedbackParams {
    #[serde(default = "feedback_passes")]
    pub passes: u32,
    #[serde(default = "root_cohesion")]
    pub root_cohesion: f32,
    #[serde(default = "interception")]
    pub rainfall_interception: f32,
    #[serde(default = "weathering")]
    pub weathering: f32,
    #[serde(default = "sediment_capture")]
    pub sediment_capture: f32,
    #[serde(default = "feedback_strength")]
    pub strength: f32,
}

fn feedback_passes() -> u32 {
    3
}
fn root_cohesion() -> f32 {
    0.55
}
fn interception() -> f32 {
    0.25
}
fn weathering() -> f32 {
    0.08
}
fn sediment_capture() -> f32 {
    0.3
}
fn feedback_strength() -> f32 {
    0.35
}

impl Default for EcosystemFeedbackParams {
    fn default() -> Self {
        Self {
            passes: feedback_passes(),
            root_cohesion: root_cohesion(),
            rainfall_interception: interception(),
            weathering: weathering(),
            sediment_capture: sediment_capture(),
            strength: feedback_strength(),
        }
    }
}

pub struct AuthoringResult {
    pub height: Heightfield,
    pub fields: HashMap<&'static str, MaskField>,
}

impl AuthoringResult {
    fn new(height: Heightfield) -> Self {
        Self {
            height,
            fields: HashMap::new(),
        }
    }
    fn field(mut self, key: &'static str, value: MaskField) -> Self {
        self.fields.insert(key, value);
        self
    }
}

fn smoothstep_weight(distance: f32, radius: f32, falloff: f32) -> f32 {
    let t = (1.0 - distance / radius.max(1e-4)).clamp(0.0, 1.0);
    (t * t * (3.0 - 2.0 * t)).powf(falloff.max(0.1))
}

fn distance_to_polyline(x: f32, z: f32, points: &[SculptPoint], sx: f32, sz: f32) -> (f32, f32) {
    if points.is_empty() {
        return (f32::INFINITY, 0.0);
    }
    if points.len() == 1 {
        let p = points[0];
        return (
            ((x - p.u * sx).hypot(z - p.v * sz)),
            p.pressure.clamp(0.0, 1.0),
        );
    }
    let mut best = f32::INFINITY;
    let mut pressure = 0.0;
    for pair in points.windows(2) {
        let ax = pair[0].u * sx;
        let az = pair[0].v * sz;
        let bx = pair[1].u * sx;
        let bz = pair[1].v * sz;
        let vx = bx - ax;
        let vz = bz - az;
        let t = (((x - ax) * vx + (z - az) * vz) / (vx * vx + vz * vz).max(1e-8)).clamp(0.0, 1.0);
        let d = (x - (ax + vx * t)).hypot(z - (az + vz * t));
        if d < best {
            best = d;
            pressure =
                (pair[0].pressure + (pair[1].pressure - pair[0].pressure) * t).clamp(0.0, 1.0);
        }
    }
    (best, pressure)
}

fn neighborhood_average(h: &Heightfield, i: u32, j: u32) -> f32 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for dj in -1..=1 {
        for di in -1..=1 {
            sum += h.get_clamped(i as i32 + di, j as i32 + dj);
            n += 1.0;
        }
    }
    sum / n
}

fn hash_noise(x: i32, y: i32, seed: u64) -> f32 {
    let mut n = (x as u64).wrapping_mul(0x9E3779B185EBCA87)
        ^ (y as u64).wrapping_mul(0xC2B2AE3D27D4EB4F)
        ^ seed;
    n ^= n >> 30;
    n = n.wrapping_mul(0xBF58476D1CE4E5B9);
    n ^= n >> 27;
    n = n.wrapping_mul(0x94D049BB133111EB);
    ((n ^ (n >> 31)) as u32 as f32) / u32::MAX as f32 * 2.0 - 1.0
}

/// A resolution-free normalized-UV rectangle `[min_u, max_u] × [min_v, max_v]`.
///
/// Structurally mirrors [`crate::tiling::UvRect`], but is defined here so the
/// low-level `authoring` module stays independent of the higher-level `tiling`
/// module (`tiling` depends on `layer`, which depends on `authoring`; an
/// `authoring → tiling` edge would close a cycle). The app / eval layers lift a
/// stroke-edit footprint into a `UvRect` when handing it to the worker scope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvBounds {
    pub min_u: f32,
    pub min_v: f32,
    pub max_u: f32,
    pub max_v: f32,
}

impl UvBounds {
    /// The smallest box covering both.
    pub fn union(self, other: UvBounds) -> UvBounds {
        UvBounds {
            min_u: self.min_u.min(other.min_u),
            min_v: self.min_v.min(other.min_v),
            max_u: self.max_u.max(other.max_u),
            max_v: self.max_v.max(other.max_v),
        }
    }

    /// Axis-aligned overlap (inclusive edges — touching counts, the safe direction
    /// for a dirty-region union).
    pub fn intersects(self, other: UvBounds) -> bool {
        self.min_u <= other.max_u
            && other.min_u <= self.max_u
            && self.min_v <= other.max_v
            && other.min_v <= self.max_v
    }

    /// Grow by `pad` on every side, clamped to `[0,1]²`.
    pub fn padded(self, pad: f32) -> UvBounds {
        UvBounds {
            min_u: (self.min_u - pad).clamp(0.0, 1.0),
            min_v: (self.min_v - pad).clamp(0.0, 1.0),
            max_u: (self.max_u + pad).clamp(0.0, 1.0),
            max_v: (self.max_v + pad).clamp(0.0, 1.0),
        }
    }
}

/// The stroke's padded footprint in normalized UV: the points' UV bounding box
/// grown by `radius_m` on each axis, clamped to `[0,1]²`. `None` when the stroke
/// has no points (no footprint).
///
/// `smoothstep_weight` is exactly zero at `distance >= radius_m`, and
/// `distance_to_polyline` measures metres with per-axis world scaling, so the pad
/// is anisotropic (`radius_m / world_size_{x,z}`) and this rect is *exactly* the
/// support of the stroke — the region an edit to it can change. This is the
/// resolution-free form fed to the CPU worker scope (#121); the sample-rect
/// [`stroke_footprint_rect`] the #110 fixpoint uses is derived from it.
pub fn stroke_footprint_uv(stroke: &SculptStroke, m: &HeightfieldMetrics) -> Option<UvBounds> {
    if stroke.points.is_empty() {
        return None;
    }
    let ru = stroke.radius_m / m.world_size_x.max(1e-3);
    let rv = stroke.radius_m / m.world_size_z.max(1e-3);
    let (mut u0, mut u1, mut v0, mut v1) = (1.0f32, 0.0f32, 1.0f32, 0.0f32);
    for pt in &stroke.points {
        u0 = u0.min(pt.u);
        u1 = u1.max(pt.u);
        v0 = v0.min(pt.v);
        v1 = v1.max(pt.v);
    }
    Some(UvBounds {
        min_u: (u0 - ru).clamp(0.0, 1.0),
        min_v: (v0 - rv).clamp(0.0, 1.0),
        max_u: (u1 + ru).clamp(0.0, 1.0),
        max_v: (v1 + rv).clamp(0.0, 1.0),
    })
}

/// The stroke's padded footprint as an inclusive sample rectangle
/// `(i0, i1, j0, j1)`, or `None` when the stroke has no points (no footprint).
///
/// This is exactly the region where `smoothstep_weight` can be non-zero, so the
/// stamp and flatten-mean scans are bounded to it. #110's scoped apply grows its
/// working rectangle by this footprint so a Flatten straddling the scope edge
/// still reads a fully-stamped field when it computes its mean. Sample-space form
/// of [`stroke_footprint_uv`] with the floor/ceil rounding that fixpoint depends on.
fn stroke_footprint_rect(
    stroke: &SculptStroke,
    m: &HeightfieldMetrics,
) -> Option<(u32, u32, u32, u32)> {
    let uv = stroke_footprint_uv(stroke, m)?;
    let i0 = (uv.min_u * (m.width - 1) as f32) as u32;
    let i1 = ((uv.max_u * (m.width - 1) as f32).ceil() as u32).min(m.width.saturating_sub(1));
    let j0 = (uv.min_v * (m.height - 1) as f32) as u32;
    let j1 = ((uv.max_v * (m.height - 1) as f32).ceil() as u32).min(m.height.saturating_sub(1));
    Some((i0, i1, j0, j1))
}

/// The normalized-UV region an edit from `prev` to `next` stroke params can
/// change, or `None` when the edit has no bounded footprint and the caller must
/// recompute whole-field.
///
/// `None` when the layer-wide `reconcile` slider changed: it re-weights every
/// stroke's reconcile pass across the whole field, so no per-stroke box bounds it
/// (out of scope for #121).
///
/// Otherwise the region is the union of [`stroke_footprint_uv`] over every stroke
/// that differs between the two lists — found by trimming the common prefix and
/// suffix, so a single slider edit, an enable toggle, a delete, an insert, or a
/// reorder each yield a tight changed-set from one code path (a radius grow/shrink
/// unions old+new extents automatically, since both sides contribute their box) —
/// then a Flatten fixpoint, then padded by `pad_uv` on every side.
///
/// **Flatten coupling.** A Flatten stroke settles toward the brush-weighted mean
/// of the *running field* over its own footprint ([`flatten_target_for`]).
/// Editing an earlier stroke changes that running field, so any later enabled
/// Flatten whose footprint overlaps the edited region shifts across its *entire*
/// footprint, not just the overlap — and the scoped evaluator only *publishes*
/// the tiles this region names, so the region must name them. The fixpoint unions
/// in every enabled Flatten (from either list) whose footprint intersects the
/// accumulated region, until stable. No other kind couples across samples:
/// Smooth/Pinch/Coastline read the *layer input* 3×3 (unchanged by a stroke
/// edit), and every other kind accumulates per-sample.
///
/// **Reconcile halo.** The reconcile pass reads a 3×3, so a changed sample at the
/// very edge of a stroke's support can shift the reconciled value one sample into
/// an overlapping stroke's support. `pad_uv` (pass one Full-res texel,
/// `1.0 / preview_resolution`) covers that one-sample dilation so a
/// boundary-hugging edit cannot leave a stale tile across a tile seam.
pub fn sculpt_edit_footprint(
    prev: &SculptStrokeParams,
    next: &SculptStrokeParams,
    m: &HeightfieldMetrics,
    pad_uv: f32,
) -> Option<UvBounds> {
    if prev.reconcile != next.reconcile {
        return None;
    }
    let a = &prev.strokes;
    let b = &next.strokes;

    // Trim the common prefix/suffix; only the changed middle contributes.
    let mut lo = 0usize;
    while lo < a.len() && lo < b.len() && a[lo] == b[lo] {
        lo += 1;
    }
    let (mut hi_a, mut hi_b) = (a.len(), b.len());
    while hi_a > lo && hi_b > lo && a[hi_a - 1] == b[hi_b - 1] {
        hi_a -= 1;
        hi_b -= 1;
    }

    fn union_into(acc: &mut Option<UvBounds>, rect: Option<UvBounds>) {
        if let Some(rect) = rect {
            *acc = Some(match *acc {
                Some(existing) => existing.union(rect),
                None => rect,
            });
        }
    }

    let mut region: Option<UvBounds> = None;
    for stroke in a[lo..hi_a].iter().chain(b[lo..hi_b].iter()) {
        union_into(&mut region, stroke_footprint_uv(stroke, m));
    }

    // Flatten coupling fixpoint (see doc comment). Only runs when something
    // changed; a no-op for paint-shaped edits that append the last stroke.
    if region.is_some() {
        let flattens: Vec<UvBounds> = a
            .iter()
            .chain(b.iter())
            .filter(|s| s.enabled && matches!(s.kind, SculptStrokeKind::Flatten))
            .filter_map(|s| stroke_footprint_uv(s, m))
            .collect();
        // Monotone growth over a finite set: converges in at most one pass per
        // Flatten (a Flatten reachable only through another is absorbed the pass
        // after that other joins). The bound also caps any degenerate case.
        for _ in 0..=flattens.len() {
            let current = region.expect("region is Some in this branch");
            let mut grown = current;
            for &f in &flattens {
                if current.intersects(f) {
                    grown = grown.union(f);
                }
            }
            region = Some(grown);
            if grown == current {
                break;
            }
        }
    }

    // Identical params (no changed strokes): a degenerate box at the first stroke
    // so the edit stays bounded to ~one tile rather than escalating whole-field.
    // The inspector only emits on a real change, so this is belt-and-braces.
    let region = region.unwrap_or_else(|| {
        a.first()
            .or_else(|| b.first())
            .and_then(|s| stroke_footprint_uv(s, m))
            .unwrap_or(UvBounds {
                min_u: 0.0,
                min_v: 0.0,
                max_u: 0.0,
                max_v: 0.0,
            })
    });
    Some(region.padded(pad_uv))
}

/// Flatten settles the footprint toward the brush-weighted mean of the terrain
/// it is editing (the accumulated `out`), computed once per stroke over the
/// stroke's padded footprint. Sampling the same buffer keeps flatten bounded and
/// idempotent; the app used to pass a target sampled from the composite height,
/// which drifted above the layer's own terrain and let every stroke stack a
/// taller spike. Returns `stroke.target_height` for non-Flatten kinds
/// (HeightStamp keeps its explicit target).
fn flatten_target_for(stroke: &SculptStroke, out: &Heightfield, m: &HeightfieldMetrics) -> f32 {
    if !matches!(stroke.kind, SculptStrokeKind::Flatten) {
        return stroke.target_height;
    }
    let Some((i0, i1, j0, j1)) = stroke_footprint_rect(stroke, m) else {
        return stroke.target_height;
    };
    let mut hsum = 0.0f64;
    let mut wsum = 0.0f64;
    for j in j0..=j1 {
        for i in i0..=i1 {
            let x = m.world_x(i);
            let z = m.world_z(j);
            let (distance, pressure) =
                distance_to_polyline(x, z, &stroke.points, m.world_size_x, m.world_size_z);
            let w = smoothstep_weight(distance, stroke.radius_m, stroke.falloff) * pressure;
            if w > 0.0 {
                hsum += out.get(i, j) as f64 * w as f64;
                wsum += w as f64;
            }
        }
    }
    if wsum > 0.0 {
        (hsum / wsum) as f32
    } else {
        stroke.target_height
    }
}

/// Apply one stroke's height + semantic-aux contribution at a single sample.
/// Shared by the whole-field and scoped stamp loops (#110) so they cannot drift.
/// The four aux channels are max-accumulated in place; `edited` is accumulated by
/// the caller (identical for every kind). Returns the new height at `(i, j)`.
#[allow(clippy::too_many_arguments)]
fn apply_stroke_sample(
    stroke: &SculptStroke,
    base: &Heightfield,
    i: u32,
    j: u32,
    h: f32,
    distance: f32,
    w: f32,
    flatten_target: f32,
    protect: &mut f32,
    uplift: &mut f32,
    hardness: &mut f32,
    sediment: &mut f32,
) -> f32 {
    let s = stroke.strength * w;
    match stroke.kind {
        SculptStrokeKind::Raise => h + s,
        SculptStrokeKind::Lower => h - s,
        SculptStrokeKind::Smooth => h + (neighborhood_average(base, i, j) - h) * w,
        SculptStrokeKind::Flatten | SculptStrokeKind::HeightStamp => {
            // Flatten uses the footprint mean (computed by flatten_target_for);
            // HeightStamp keeps the explicit stroke target.
            h + (flatten_target - h) * w
        }
        SculptStrokeKind::Ridge | SculptStrokeKind::MountainStamp => {
            let t = (1.0 - distance / stroke.radius_m.max(1.0)).max(0.0);
            h + s * t * t
        }
        SculptStrokeKind::Valley | SculptStrokeKind::ValleyStamp | SculptStrokeKind::RiverPath => {
            h - s.abs() * (1.0 - distance / stroke.radius_m.max(1.0)).max(0.0)
        }
        SculptStrokeKind::Terrace => {
            let step = stroke.strength.abs().max(0.1);
            h + ((h / step).round() * step - h) * w
        }
        SculptStrokeKind::Roughness | SculptStrokeKind::Noise => {
            h + hash_noise(i as i32, j as i32, 91) * s
        }
        SculptStrokeKind::Uplift => {
            *uplift = (*uplift).max(s.max(0.0));
            h
        }
        SculptStrokeKind::Hardness => {
            *hardness = (*hardness).max(s.clamp(0.0, 1.0));
            h
        }
        SculptStrokeKind::Sediment => {
            *sediment = (*sediment).max(s.max(0.0));
            h
        }
        SculptStrokeKind::Protect => {
            *protect = (*protect).max(s.clamp(0.0, 1.0));
            h
        }
        SculptStrokeKind::EncourageErosion | SculptStrokeKind::Erode => {
            *protect = (*protect - s.abs().min(1.0)).max(0.0);
            h - s.abs() * 0.35
        }
        SculptStrokeKind::Pinch => {
            // Pull heights toward neighbourhood mean (contract detail).
            let avg = neighborhood_average(base, i, j);
            h + (avg - h) * w * 1.25
        }
        SculptStrokeKind::Inflate => {
            let t = (1.0 - distance / stroke.radius_m.max(1.0)).max(0.0);
            h + s * t
        }
        SculptStrokeKind::PlateauStamp => {
            let t = (1.0 - distance / stroke.radius_m.max(1.0)).max(0.0);
            let plateau = stroke.target_height.max(h + s.abs());
            h + (plateau - h) * w * t
        }
        SculptStrokeKind::CraterStamp => {
            let t = distance / stroke.radius_m.max(1.0);
            if t >= 1.0 {
                h
            } else if t < 0.55 {
                // Bowl.
                h - s.abs() * (1.0 - t / 0.55) * w
            } else {
                // Rim.
                let rim = ((t - 0.55) / 0.45).clamp(0.0, 1.0);
                let bump = (1.0 - (rim - 0.5).abs() * 2.0).max(0.0);
                h + s.abs() * 0.45 * bump * w
            }
        }
        SculptStrokeKind::Coastline => {
            let avg = neighborhood_average(base, i, j);
            let lowered = h - s.abs() * 0.25;
            (lowered + (avg - lowered) * 0.55) * w + h * (1.0 - w)
        }
    }
}

pub fn apply_sculpt_strokes(input: &Heightfield, p: &SculptStrokeParams) -> AuthoringResult {
    let m = input.metrics;
    let mut out = input.clone();
    let n = (m.width * m.height) as usize;
    let mut protect: Vec<f32> = vec![0.0; n];
    let mut uplift: Vec<f32> = vec![0.0; n];
    let mut hardness: Vec<f32> = vec![0.0; n];
    let mut sediment: Vec<f32> = vec![0.0; n];
    let mut edited: Vec<f32> = vec![0.0; n];
    let base = input.clone();
    for stroke in &p.strokes {
        if !stroke.enabled {
            continue;
        }
        let flatten_target = flatten_target_for(stroke, &out, &m);
        for j in 0..m.height {
            for i in 0..m.width {
                let x = m.world_x(i);
                let z = m.world_z(j);
                let (distance, pressure) =
                    distance_to_polyline(x, z, &stroke.points, m.world_size_x, m.world_size_z);
                let w = smoothstep_weight(distance, stroke.radius_m, stroke.falloff) * pressure;
                if w <= 0.0 {
                    continue;
                }
                let idx = (j * m.width + i) as usize;
                edited[idx] = edited[idx].max(w);
                let h = out.get(i, j);
                let next = apply_stroke_sample(
                    stroke,
                    &base,
                    i,
                    j,
                    h,
                    distance,
                    w,
                    flatten_target,
                    &mut protect[idx],
                    &mut uplift[idx],
                    &mut hardness[idx],
                    &mut sediment[idx],
                );
                out.set(i, j, next);
            }
        }
    }
    if p.reconcile > 0.0 {
        let src = out.clone();
        for j in 0..m.height {
            for i in 0..m.width {
                let idx = (j * m.width + i) as usize;
                let a = p.reconcile.clamp(0.0, 1.0) * edited[idx] * 0.35;
                out.set(
                    i,
                    j,
                    src.get(i, j) + (neighborhood_average(&src, i, j) - src.get(i, j)) * a,
                );
            }
        }
    }
    AuthoringResult::new(out)
        .field(keys::SCULPT_PROTECTION, MaskField::from_raw(m, &protect))
        .field(keys::UPLIFT_RATE, MaskField::from_raw(m, &uplift))
        .field(keys::HARDNESS, MaskField::from_raw(m, &hardness))
        .field(keys::SEDIMENT_THICKNESS, MaskField::from_raw(m, &sediment))
        .field(keys::EDIT_REGION, MaskField::from_raw(m, &edited))
}

/// The scope-tile interiors dilated one sample and clamped to the field, as an
/// inclusive sample rectangle. `None` when no scope tile is in range. The one
/// sample of padding lets the reconcile 3x3 on every scope sample read stamped
/// neighbours (#108's `halo_samples: 1`).
fn scope_rect_dilated(input: &Heightfield, scope: &[TileId]) -> Option<(u32, u32, u32, u32)> {
    let m = input.metrics;
    let mut acc: Option<(u32, u32, u32, u32)> = None;
    for &id in scope {
        let Some(tile) = input.tile(id) else {
            continue;
        };
        let (ox, oz) = tile.interior_origin(&m);
        let (ti1, tj1) = (ox + tile.interior_width - 1, oz + tile.interior_height - 1);
        acc = Some(match acc {
            None => (ox, ti1, oz, tj1),
            Some((i0, i1, j0, j1)) => (i0.min(ox), i1.max(ti1), j0.min(oz), j1.max(tj1)),
        });
    }
    let (i0, i1, j0, j1) = acc?;
    Some((
        i0.saturating_sub(1),
        (i1 + 1).min(m.width.saturating_sub(1)),
        j0.saturating_sub(1),
        (j1 + 1).min(m.height.saturating_sub(1)),
    ))
}

/// Region-scoped [`apply_sculpt_strokes`] (#110).
///
/// Produces a height + aux result whose `scope` tiles are bit-identical to the
/// whole-field pass; samples outside the working rectangle are left at `input`
/// and are never read by the caller's scoped blend.
///
/// The working rectangle `S` starts at the scope interiors dilated one sample,
/// then grows to the full footprint of every stroke it intersects — a fixpoint.
/// That guarantees any Flatten whose footprint reaches `S` has that whole
/// footprint stamped by prior strokes before it computes its mean, so the mean
/// (and therefore every scope sample) matches the whole-field result exactly.
/// Strokes that miss `S` are skipped. `S` is bounded by the field, so a giant
/// footprint degrades to whole-field cost rather than being wrong.
pub fn apply_sculpt_strokes_scoped(
    input: &Heightfield,
    p: &SculptStrokeParams,
    scope: &[TileId],
) -> AuthoringResult {
    let m = input.metrics;
    let Some((mut i0, mut i1, mut j0, mut j1)) = scope_rect_dilated(input, scope) else {
        // Empty scope: nothing to recompute. The caller blends no tiles.
        return AuthoringResult::new(input.clone());
    };
    // Fixpoint: absorb the full footprint of every stroke intersecting S.
    loop {
        let mut grew = false;
        for stroke in &p.strokes {
            if !stroke.enabled {
                continue;
            }
            let Some((si0, si1, sj0, sj1)) = stroke_footprint_rect(stroke, &m) else {
                continue;
            };
            let intersects = si0 <= i1 && si1 >= i0 && sj0 <= j1 && sj1 >= j0;
            if intersects {
                let grown = (i0.min(si0), i1.max(si1), j0.min(sj0), j1.max(sj1));
                if grown != (i0, i1, j0, j1) {
                    (i0, i1, j0, j1) = grown;
                    grew = true;
                }
            }
        }
        if !grew {
            break;
        }
    }

    let mut out = input.clone();
    let n = (m.width * m.height) as usize;
    let mut protect: Vec<f32> = vec![0.0; n];
    let mut uplift: Vec<f32> = vec![0.0; n];
    let mut hardness: Vec<f32> = vec![0.0; n];
    let mut sediment: Vec<f32> = vec![0.0; n];
    let mut edited: Vec<f32> = vec![0.0; n];
    let base = input.clone();
    for stroke in &p.strokes {
        if !stroke.enabled {
            continue;
        }
        // Skip strokes whose footprint does not reach S (no effect inside it).
        match stroke_footprint_rect(stroke, &m) {
            Some((si0, si1, sj0, sj1)) if si0 <= i1 && si1 >= i0 && sj0 <= j1 && sj1 >= j0 => {}
            _ => continue,
        }
        let flatten_target = flatten_target_for(stroke, &out, &m);
        for j in j0..=j1 {
            for i in i0..=i1 {
                let x = m.world_x(i);
                let z = m.world_z(j);
                let (distance, pressure) =
                    distance_to_polyline(x, z, &stroke.points, m.world_size_x, m.world_size_z);
                let w = smoothstep_weight(distance, stroke.radius_m, stroke.falloff) * pressure;
                if w <= 0.0 {
                    continue;
                }
                let idx = (j * m.width + i) as usize;
                edited[idx] = edited[idx].max(w);
                let h = out.get(i, j);
                let next = apply_stroke_sample(
                    stroke,
                    &base,
                    i,
                    j,
                    h,
                    distance,
                    w,
                    flatten_target,
                    &mut protect[idx],
                    &mut uplift[idx],
                    &mut hardness[idx],
                    &mut sediment[idx],
                );
                out.set(i, j, next);
            }
        }
    }
    if p.reconcile > 0.0 {
        let src = out.clone();
        for j in j0..=j1 {
            for i in i0..=i1 {
                let idx = (j * m.width + i) as usize;
                let a = p.reconcile.clamp(0.0, 1.0) * edited[idx] * 0.35;
                out.set(
                    i,
                    j,
                    src.get(i, j) + (neighborhood_average(&src, i, j) - src.get(i, j)) * a,
                );
            }
        }
    }
    AuthoringResult::new(out)
        .field(keys::SCULPT_PROTECTION, MaskField::from_raw(m, &protect))
        .field(keys::UPLIFT_RATE, MaskField::from_raw(m, &uplift))
        .field(keys::HARDNESS, MaskField::from_raw(m, &hardness))
        .field(keys::SEDIMENT_THICKNESS, MaskField::from_raw(m, &sediment))
        .field(keys::EDIT_REGION, MaskField::from_raw(m, &edited))
}

pub fn apply_constraints(input: &Heightfield, p: &TerrainConstraintParams) -> AuthoringResult {
    let m = input.metrics;
    let mut out = input.clone();
    let n = (m.width * m.height) as usize;
    let mut target = input.to_dense();
    let mut weight: Vec<f32> = vec![0.0; n];
    let mut protect: Vec<f32> = vec![0.0; n];
    let mut uplift: Vec<f32> = vec![0.0; n];
    for c in &p.constraints {
        for j in 0..m.height {
            for i in 0..m.width {
                let idx = (j * m.width + i) as usize;
                let (d, pressure) = distance_to_polyline(
                    m.world_x(i),
                    m.world_z(j),
                    &c.points,
                    m.world_size_x,
                    m.world_size_z,
                );
                let w =
                    smoothstep_weight(d, c.width_m, 1.25) * pressure * c.strength.clamp(0.0, 1.0);
                if w <= 0.0 {
                    continue;
                }
                let h = input.get(i, j);
                let desired = match c.kind {
                    TerrainConstraintKind::Elevation
                    | TerrainConstraintKind::Coastline
                    | TerrainConstraintKind::Plateau
                    | TerrainConstraintKind::Outlet => c.value,
                    TerrainConstraintKind::MinElevation => h.max(c.value),
                    TerrainConstraintKind::MaxElevation => h.min(c.value),
                    TerrainConstraintKind::Ridge | TerrainConstraintKind::Divide => {
                        h + c.value.abs() * w
                    }
                    TerrainConstraintKind::Valley | TerrainConstraintKind::River => {
                        h - c.value.abs() * w
                    }
                    TerrainConstraintKind::Cliff => {
                        h + c.value * (0.5 - d / c.width_m.max(1.0)).signum() * w
                    }
                    TerrainConstraintKind::PreferredSlope => h + c.value.to_radians().tan() * d * w,
                    TerrainConstraintKind::Roughness => {
                        h + hash_noise(i as i32, j as i32, 313) * c.value * w
                    }
                    TerrainConstraintKind::Protect => {
                        protect[idx] = protect[idx].max(w);
                        h
                    }
                };
                if matches!(
                    c.kind,
                    TerrainConstraintKind::Ridge | TerrainConstraintKind::Divide
                ) {
                    uplift[idx] = uplift[idx].max(c.value.abs() * w);
                }
                if !matches!(c.kind, TerrainConstraintKind::Protect) {
                    target[idx] = target[idx] + (desired - target[idx]) * w;
                    weight[idx] = weight[idx].max(w);
                }
            }
        }
    }
    for j in 0..m.height {
        for i in 0..m.width {
            let idx = (j * m.width + i) as usize;
            let a = weight[idx] * p.preview_strength.clamp(0.0, 1.0);
            out.set(i, j, input.get(i, j) + (target[idx] - input.get(i, j)) * a);
        }
    }
    AuthoringResult::new(out)
        .field(keys::CONSTRAINT_TARGET, MaskField::from_raw(m, &target))
        .field(keys::CONSTRAINT_WEIGHT, MaskField::from_raw(m, &weight))
        .field(keys::SCULPT_PROTECTION, MaskField::from_raw(m, &protect))
        .field(keys::UPLIFT_RATE, MaskField::from_raw(m, &uplift))
        .field(keys::EDIT_REGION, MaskField::from_raw(m, &weight))
}

pub fn gradient_reconstruct(
    input: &Heightfield,
    p: &GradientReconstructParams,
    target: Option<&MaskField>,
    weight: Option<&MaskField>,
) -> AuthoringResult {
    let m = input.metrics;
    let w = m.width as usize;
    let h = m.height as usize;
    if w < 2 || h < 2 {
        return AuthoringResult::new(input.clone());
    }
    let original = input.to_dense();
    let mut current = original.clone();
    let mut next = current.clone();
    let dx2 = m.dx().max(1e-3).powi(2);
    let dz2 = m.dz().max(1e-3).powi(2);
    let screen = p.screening.max(0.0);
    let constraint = p.constraint_strength.max(0.0);
    for _ in 0..p.iterations.max(1) {
        for j in 1..h - 1 {
            for i in 1..w - 1 {
                let idx = j * w + i;
                let cw = weight.map(|f| f.data()[idx]).unwrap_or(0.0).clamp(0.0, 1.0);
                let lambda = screen + constraint * cw;
                let t = target.map(|f| f.data()[idx]).unwrap_or(original[idx]);
                let neighbor = (current[idx - 1] + current[idx + 1]) / dx2
                    + (current[idx - w] + current[idx + w]) / dz2;
                let lap_original = (original[idx - 1] - 2.0 * original[idx] + original[idx + 1])
                    / dx2
                    + (original[idx - w] - 2.0 * original[idx] + original[idx + w]) / dz2;
                let smooth = p.gradient_smoothing.clamp(0.0, 1.0);
                let divergence = lap_original * (1.0 - smooth);
                let denom = 2.0 / dx2 + 2.0 / dz2 + lambda;
                next[idx] =
                    (neighbor - divergence + screen * original[idx] + constraint * cw * t) / denom;
            }
        }
        std::mem::swap(&mut current, &mut next);
    }
    let error: Vec<f32> = current
        .iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();
    AuthoringResult::new(Heightfield::from_dense(m, &current))
        .field(keys::CONSTRAINT_ERROR, MaskField::from_raw(m, &error))
}

pub fn landscape_evolution(
    input: &Heightfield,
    p: &LandscapeEvolutionParams,
    hardness: Option<&MaskField>,
    uplift: Option<&MaskField>,
    protection: Option<&MaskField>,
) -> AuthoringResult {
    let (height, fields) = crate::landscape_evolution::evaluate_landscape_evolution(
        input, p, hardness, uplift, protection, None, None,
    );
    let mut result = AuthoringResult::new(height);
    for (key, value) in fields {
        result = result.field(key, value);
    }
    result
}

pub fn repair_hydrology(
    input: &Heightfield,
    p: &HydrologyRepairParams,
    edit_region: Option<&MaskField>,
    hardness: Option<&MaskField>,
    protection: Option<&MaskField>,
) -> AuthoringResult {
    let m = input.metrics;
    let hard = hardness.cloned().unwrap_or_else(|| MaskField::zeros(m));
    let spe = StreamPowerParams {
        iterations: p.iterations.max(1),
        k: p.incision.max(0.0),
        m: 0.5,
        n: 1.0,
        uplift_rate: 0.0,
        base_level: f32::NEG_INFINITY,
        dt: 1.0,
        use_dinfinity: true,
        refill_each_iter: true,
        drainage_reuse_stride: 1,
        hardness: 0.0,
        hardness_source: MaskSource::None,
        dendritic_seed: 0.0,
        stream_threshold: p.stream_threshold,
        level_count: 0,
        start_level: 0,
        level_step_strength: 1.0,
        level_step_curve: Vec::new(),
    };
    let repaired = hydro::stream_power_erode(input, &spe, &hard);
    let mut region = edit_region.cloned().unwrap_or_else(|| MaskField::ones(m));
    let dilation = (p.repair_radius_m / m.dx().max(m.dz()).max(1.0))
        .ceil()
        .min(48.0) as u32;
    for _ in 0..dilation {
        let src = region.clone();
        for j in 0..m.height {
            for i in 0..m.width {
                let mut v = src.get(i, j);
                for dj in -1..=1 {
                    for di in -1..=1 {
                        v = v.max(src.get(
                            (i as i32 + di).clamp(0, m.width as i32 - 1) as u32,
                            (j as i32 + dj).clamp(0, m.height as i32 - 1) as u32,
                        ));
                    }
                }
                region.set(i, j, v);
            }
        }
    }
    let mut out = input.clone();
    for j in 0..m.height {
        for i in 0..m.width {
            let protected = protection.map(|f| f.get(i, j)).unwrap_or(0.0)
                * p.constraint_preservation.clamp(0.0, 1.0);
            let a = region.get(i, j) * (1.0 - protected);
            out.set(
                i,
                j,
                input.get(i, j) + (repaired.height.get(i, j) - input.get(i, j)) * a,
            );
        }
    }
    AuthoringResult::new(out)
        .field(keys::FLOW_DIRECTION, repaired.flow_direction)
        .field(keys::FLOW_ACCUMULATION, repaired.flow_accumulation)
        .field(keys::STREAM_ORDER, repaired.stream_order)
        .field(keys::SPE_INCISION, repaired.spe_incision)
        .field(keys::REPAIR_REGION, region)
}

pub fn geomorphic_detail(
    input: &Heightfield,
    p: &GeomorphicDetailParams,
    flow_accumulation: Option<&MaskField>,
    protection: Option<&MaskField>,
) -> AuthoringResult {
    geomorphic_detail_with_hardness(input, p, flow_accumulation, None, protection)
}

/// Drainage-conditioned multi-scale amplification (Grenier/Schott-inspired).
///
/// Produces enhanced elevation plus fine flow, micro-channel, ridge-breakup,
/// and fine-erosion maps. Never applies isotropic `height += noise * amount`.
pub fn geomorphic_detail_with_hardness(
    input: &Heightfield,
    p: &GeomorphicDetailParams,
    flow_accumulation: Option<&MaskField>,
    hardness: Option<&MaskField>,
    protection: Option<&MaskField>,
) -> AuthoringResult {
    let amp = crate::analyze::amplify_terrain(
        input,
        &p.to_amplification(),
        flow_accumulation,
        hardness,
        protection,
    );
    AuthoringResult::new(amp.height)
        .field(keys::DETAIL_MASK, amp.detail_mask)
        .field(keys::FINE_FLOW, amp.fine_flow)
        .field(keys::MICRO_CHANNEL, amp.micro_channel)
        .field(keys::RIDGE_BREAKUP, amp.ridge_breakup)
        .field(keys::FINE_EROSION, amp.fine_erosion)
}

pub fn ecosystem_feedback(
    input: &Heightfield,
    p: &EcosystemFeedbackParams,
    vegetation: Option<&MaskField>,
    moisture: Option<&MaskField>,
    hardness: Option<&MaskField>,
    sediment: Option<&MaskField>,
) -> AuthoringResult {
    let m = input.metrics;
    let mut out = input.clone();
    let mut roots = MaskField::zeros(m);
    let mut hard = hardness.cloned().unwrap_or_else(|| MaskField::zeros(m));
    let mut deposit = MaskField::zeros(m);
    for _ in 0..p.passes.clamp(1, 12) {
        let src = out.clone();
        for j in 0..m.height {
            for i in 0..m.width {
                let gx = (src.get_clamped(i as i32 + 1, j as i32)
                    - src.get_clamped(i as i32 - 1, j as i32))
                    / (2.0 * m.dx().max(1e-3));
                let gz = (src.get_clamped(i as i32, j as i32 + 1)
                    - src.get_clamped(i as i32, j as i32 - 1))
                    / (2.0 * m.dz().max(1e-3));
                let slope = gx.hypot(gz);
                let wet = moisture.map(|f| f.get(i, j)).unwrap_or(0.55);
                let veg = vegetation
                    .map(|f| f.get(i, j))
                    .unwrap_or_else(|| (wet * (1.0 - slope / 1.2)).clamp(0.0, 1.0));
                let root = (veg * p.root_cohesion).clamp(0.0, 1.0);
                roots.set(i, j, root);
                hard.set(
                    i,
                    j,
                    (hard.get(i, j) + root * (1.0 - hard.get(i, j))).clamp(0.0, 1.0),
                );
                let bare = (1.0 - root) * (1.0 - p.rainfall_interception * veg).clamp(0.0, 1.0);
                let weather = p.weathering * p.strength * bare * slope.min(1.0) * 0.05;
                let captured = sediment.map(|f| f.get(i, j)).unwrap_or(0.0)
                    * p.sediment_capture
                    * veg
                    * p.strength
                    * 0.02;
                out.set(i, j, src.get(i, j) - weather + captured);
                deposit.set(i, j, (deposit.get(i, j) + captured).clamp(0.0, 1.0));
            }
        }
    }
    AuthoringResult::new(out)
        .field(keys::ROOT_COHESION, roots)
        .field(keys::HARDNESS, hard)
        .field(keys::DEPOSITION, deposit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

    fn plane() -> Heightfield {
        let m = HeightfieldMetrics::new(32, 32, 1000.0, 1000.0);
        let mut h = Heightfield::zeros(m);
        for j in 0..32 {
            for i in 0..32 {
                h.set(i, j, i as f32 + j as f32 * 0.25);
            }
        }
        h
    }

    #[test]
    fn flatten_stays_within_input_range_and_ignores_passed_target() {
        // Sloped terrain. A flatten stroke must settle toward the footprint mean
        // and never exceed the input's own min/max — even if handed an absurd
        // target_height (which is what the composite-sampled app value used to do,
        // producing a runaway spike).
        let h = plane();
        let (lo, hi) = h.min_max();
        let p = SculptStrokeParams {
            strokes: vec![SculptStroke {
                kind: SculptStrokeKind::Flatten,
                points: vec![SculptPoint {
                    u: 0.5,
                    v: 0.5,
                    pressure: 1.0,
                }],
                radius_m: 300.0,
                strength: 4.0,
                target_height: 1_000_000.0, // absurd — must be ignored by Flatten
                falloff: 1.5,
                enabled: true,
            }],
            ..SculptStrokeParams::default()
        };
        let r = apply_sculpt_strokes(&h, &p);
        let (out_lo, out_hi) = r.height.min_max();
        assert!(
            out_lo >= lo - 1e-3 && out_hi <= hi + 1e-3,
            "flatten escaped input range: in [{lo}, {hi}] out [{out_lo}, {out_hi}]"
        );
        assert!(r.height.to_dense().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn flatten_is_idempotent_no_runaway() {
        // Applying the same flatten stroke twice must not keep raising the peak.
        let h = plane();
        let stroke = SculptStroke {
            kind: SculptStrokeKind::Flatten,
            points: vec![SculptPoint {
                u: 0.5,
                v: 0.5,
                pressure: 1.0,
            }],
            radius_m: 300.0,
            strength: 4.0,
            target_height: 0.0,
            falloff: 1.5,
            enabled: true,
        };
        let once = apply_sculpt_strokes(
            &h,
            &SculptStrokeParams {
                strokes: vec![stroke.clone()],
                ..SculptStrokeParams::default()
            },
        )
        .height;
        let twice = apply_sculpt_strokes(
            &once,
            &SculptStrokeParams {
                strokes: vec![stroke],
                ..SculptStrokeParams::default()
            },
        )
        .height;
        let peak_once = once.min_max().1;
        let peak_twice = twice.min_max().1;
        assert!(
            peak_twice <= peak_once + 1e-3,
            "flatten grew on re-apply: {peak_once} -> {peak_twice}"
        );
    }

    #[test]
    fn semantic_stroke_is_resolution_independent_and_local() {
        let h = plane();
        let p = SculptStrokeParams {
            strokes: vec![SculptStroke {
                points: vec![SculptPoint {
                    u: 0.5,
                    v: 0.5,
                    pressure: 1.0,
                }],
                ..SculptStroke::default()
            }],
            ..SculptStrokeParams::default()
        };
        let r = apply_sculpt_strokes(&h, &p);
        assert!(r.height.get(16, 16) > h.get(16, 16));
        assert!((r.height.get(0, 0) - h.get(0, 0)).abs() < 1e-5);
    }

    #[test]
    fn scoped_sculpt_matches_whole_field_on_scope_tiles() {
        // A multi-tile field (tile_size 16 -> 4x4 tile grid on 64^2).
        let m = HeightfieldMetrics {
            width: 64,
            height: 64,
            world_size_x: 1000.0,
            world_size_z: 1000.0,
            tile_size: 16,
            halo: 2,
        };
        let mut h = Heightfield::zeros(m);
        for j in 0..64 {
            for i in 0..64 {
                h.set(
                    i,
                    j,
                    (i as f32) * 0.7 + (j as f32) * 0.3 + ((i * 5 + j * 11) % 17) as f32,
                );
            }
        }
        // Raise + Smooth (reads the 3x3 base) stamp inside a large Flatten's
        // footprint; the Flatten (radius 400 on a 1000 m world) straddles the
        // scope edge, so its footprint mean is only correct if the fixpoint has
        // grown S to cover the whole footprint and stamped the priors there.
        let pt = |u, v| SculptPoint {
            u,
            v,
            pressure: 1.0,
        };
        let p = SculptStrokeParams {
            strokes: vec![
                SculptStroke {
                    kind: SculptStrokeKind::Raise,
                    points: vec![pt(0.4, 0.4)],
                    radius_m: 120.0,
                    strength: 8.0,
                    target_height: 0.0,
                    falloff: 1.5,
                    enabled: true,
                },
                SculptStroke {
                    kind: SculptStrokeKind::Smooth,
                    points: vec![pt(0.6, 0.5)],
                    radius_m: 90.0,
                    strength: 5.0,
                    target_height: 0.0,
                    falloff: 1.2,
                    enabled: true,
                },
                SculptStroke {
                    kind: SculptStrokeKind::Flatten,
                    points: vec![pt(0.5, 0.5)],
                    radius_m: 400.0,
                    strength: 4.0,
                    target_height: 0.0,
                    falloff: 1.5,
                    enabled: true,
                },
                SculptStroke {
                    kind: SculptStrokeKind::Uplift,
                    points: vec![pt(0.5, 0.5)],
                    radius_m: 100.0,
                    strength: 3.0,
                    target_height: 0.0,
                    falloff: 1.5,
                    enabled: true,
                },
            ],
            reconcile: 0.2,
        };
        let whole = apply_sculpt_strokes(&h, &p);
        // Interior tiles plus a partial edge tile.
        let scope = vec![
            TileId { tx: 1, tz: 1 },
            TileId { tx: 2, tz: 1 },
            TileId { tx: 3, tz: 3 },
        ];
        let scoped = apply_sculpt_strokes_scoped(&h, &p, &scope);
        for &id in &scope {
            let tile = h.tile(id).expect("scope tile exists");
            let (ox, oz) = tile.interior_origin(&m);
            for lz in 0..tile.interior_height {
                for lx in 0..tile.interior_width {
                    let (i, j) = (ox + lx, oz + lz);
                    assert_eq!(
                        scoped.height.get(i, j).to_bits(),
                        whole.height.get(i, j).to_bits(),
                        "height mismatch at ({i},{j}) in tile {id:?}"
                    );
                    for key in [
                        keys::SCULPT_PROTECTION,
                        keys::UPLIFT_RATE,
                        keys::HARDNESS,
                        keys::SEDIMENT_THICKNESS,
                        keys::EDIT_REGION,
                    ] {
                        let wf = whole.fields.get(key).expect("whole aux").get(i, j);
                        let sf = scoped.fields.get(key).expect("scoped aux").get(i, j);
                        assert_eq!(
                            sf.to_bits(),
                            wf.to_bits(),
                            "aux {key} mismatch at ({i},{j}) in tile {id:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn constrained_poisson_moves_toward_target_without_nan() {
        let h = plane();
        let m = h.metrics;
        let mut target = MaskField::from_raw(m, &h.to_dense());
        let mut weight = MaskField::zeros(m);
        target.data_mut()[16 * 32 + 16] = 200.0;
        weight.set(16, 16, 1.0);
        let r = gradient_reconstruct(
            &h,
            &GradientReconstructParams::default(),
            Some(&target),
            Some(&weight),
        );
        assert!(r.height.get(16, 16) > h.get(16, 16));
        assert!(r.height.to_dense().iter().all(|v| v.is_finite()));
    }

    // ---- #121: per-stroke edit footprint (scoped invalidation) ----

    fn fp_metrics() -> HeightfieldMetrics {
        HeightfieldMetrics {
            width: 64,
            height: 64,
            world_size_x: 1000.0,
            world_size_z: 1000.0,
            tile_size: 16,
            halo: 2,
        }
    }

    fn fp_field(m: HeightfieldMetrics) -> Heightfield {
        let mut h = Heightfield::zeros(m);
        for j in 0..m.height {
            for i in 0..m.width {
                h.set(
                    i,
                    j,
                    (i as f32) * 0.7 + (j as f32) * 0.3 + ((i * 5 + j * 11) % 17) as f32,
                );
            }
        }
        h
    }

    fn mk_stroke(kind: SculptStrokeKind, u: f32, v: f32, radius_m: f32) -> SculptStroke {
        SculptStroke {
            kind,
            points: vec![SculptPoint {
                u,
                v,
                pressure: 1.0,
            }],
            radius_m,
            strength: 8.0,
            target_height: 0.0,
            falloff: 1.5,
            enabled: true,
        }
    }

    fn mk_params(strokes: Vec<SculptStroke>) -> SculptStrokeParams {
        SculptStrokeParams {
            strokes,
            reconcile: 0.2,
        }
    }

    fn uv_covers(outer: UvBounds, inner: UvBounds) -> bool {
        outer.min_u <= inner.min_u
            && outer.min_v <= inner.min_v
            && outer.max_u >= inner.max_u
            && outer.max_v >= inner.max_v
    }

    /// Lift an authoring `UvBounds` into the `tiling::UvRect` the tile mapper takes
    /// (the same trivial conversion the app/eval layers do).
    fn uv_rect(b: UvBounds) -> crate::tiling::UvRect {
        crate::tiling::UvRect {
            min_u: b.min_u,
            min_v: b.min_v,
            max_u: b.max_u,
            max_v: b.max_v,
        }
    }

    #[test]
    fn stroke_footprint_uv_pads_per_axis_and_none_for_empty() {
        // Non-square world: 2000 m wide, 500 m tall. radius 100 m => UV pad
        // 100/2000 = 0.05 in u, 100/500 = 0.2 in v.
        let m = HeightfieldMetrics {
            width: 64,
            height: 64,
            world_size_x: 2000.0,
            world_size_z: 500.0,
            tile_size: 16,
            halo: 2,
        };
        let s = mk_stroke(SculptStrokeKind::Raise, 0.5, 0.5, 100.0);
        let uv = stroke_footprint_uv(&s, &m).expect("has points");
        assert!((uv.min_u - 0.45).abs() < 1e-6 && (uv.max_u - 0.55).abs() < 1e-6);
        assert!((uv.min_v - 0.30).abs() < 1e-6 && (uv.max_v - 0.70).abs() < 1e-6);

        let mut empty = s.clone();
        empty.points.clear();
        assert!(stroke_footprint_uv(&empty, &m).is_none());
    }

    #[test]
    fn footprint_rect_still_matches_the_uv_form() {
        // The refactor must not perturb the sample-rect rounding #110 relies on.
        let m = fp_metrics();
        for &(u, v, r) in &[(0.5, 0.5, 120.0), (0.1, 0.9, 60.0), (0.99, 0.01, 300.0)] {
            let s = mk_stroke(SculptStrokeKind::Raise, u, v, r);
            let uv = stroke_footprint_uv(&s, &m).unwrap();
            let (i0, i1, j0, j1) = stroke_footprint_rect(&s, &m).unwrap();
            assert_eq!(i0, (uv.min_u * (m.width - 1) as f32) as u32);
            assert_eq!(
                i1,
                ((uv.max_u * (m.width - 1) as f32).ceil() as u32).min(m.width - 1)
            );
            assert_eq!(j0, (uv.min_v * (m.height - 1) as f32) as u32);
            assert_eq!(
                j1,
                ((uv.max_v * (m.height - 1) as f32).ceil() as u32).min(m.height - 1)
            );
        }
    }

    #[test]
    fn edit_footprint_reconcile_change_is_whole_field() {
        let m = fp_metrics();
        let a = mk_params(vec![mk_stroke(SculptStrokeKind::Raise, 0.5, 0.5, 100.0)]);
        let mut b = a.clone();
        b.reconcile += 0.1;
        assert!(
            sculpt_edit_footprint(&a, &b, &m, 1.0 / 64.0).is_none(),
            "the layer-wide reconcile slider stays whole-field"
        );
    }

    #[test]
    fn edit_footprint_strength_only_is_the_stroke_box() {
        let m = fp_metrics();
        let s = mk_stroke(SculptStrokeKind::Raise, 0.3, 0.3, 100.0);
        let a = mk_params(vec![s.clone()]);
        let mut s2 = s.clone();
        s2.strength += 5.0;
        let b = mk_params(vec![s2]);
        let fp = sculpt_edit_footprint(&a, &b, &m, 0.0).expect("bounded");
        assert_eq!(fp, stroke_footprint_uv(&s, &m).unwrap());
    }

    #[test]
    fn edit_footprint_radius_change_unions_both_extents() {
        let m = fp_metrics();
        let small = mk_stroke(SculptStrokeKind::Raise, 0.5, 0.5, 50.0);
        let mut big = small.clone();
        big.radius_m = 200.0;
        let expect = stroke_footprint_uv(&small, &m)
            .unwrap()
            .union(stroke_footprint_uv(&big, &m).unwrap());
        let grow = sculpt_edit_footprint(&mk_params(vec![small.clone()]), &mk_params(vec![big.clone()]), &m, 0.0).unwrap();
        let shrink = sculpt_edit_footprint(&mk_params(vec![big]), &mk_params(vec![small]), &m, 0.0).unwrap();
        assert_eq!(grow, expect, "grow unions old+new");
        assert_eq!(shrink, expect, "shrink is symmetric");
    }

    #[test]
    fn edit_footprint_toggle_and_delete_isolate_the_affected_stroke() {
        let m = fp_metrics();
        let keep = mk_stroke(SculptStrokeKind::Raise, 0.2, 0.2, 60.0);
        let target = mk_stroke(SculptStrokeKind::Raise, 0.7, 0.7, 60.0);
        let a = mk_params(vec![keep.clone(), target.clone()]);
        let expect = stroke_footprint_uv(&target, &m).unwrap();

        let mut off = target.clone();
        off.enabled = false;
        let toggled = sculpt_edit_footprint(&a, &mk_params(vec![keep.clone(), off]), &m, 0.0).unwrap();
        assert_eq!(toggled, expect, "toggle isolates the toggled stroke's box");

        let deleted = sculpt_edit_footprint(&a, &mk_params(vec![keep]), &m, 0.0).unwrap();
        assert_eq!(
            deleted, expect,
            "delete's footprint is the removed stroke's box — no index-shift over-dirty"
        );
    }

    #[test]
    fn edit_footprint_identical_params_stay_bounded() {
        // Belt-and-braces: identical params (no changed stroke) must not escalate to
        // whole-field — they collapse to the first stroke's box, a ~one-tile recompute.
        let m = fp_metrics();
        let s = mk_stroke(SculptStrokeKind::Raise, 0.4, 0.4, 80.0);
        let a = mk_params(vec![s.clone()]);
        let fp = sculpt_edit_footprint(&a, &a, &m, 0.0).expect("identical params stay bounded");
        assert_eq!(fp, stroke_footprint_uv(&s, &m).unwrap());
    }

    #[test]
    fn edit_footprint_expands_over_overlapping_flatten_only() {
        let m = fp_metrics();
        let raise = mk_stroke(SculptStrokeKind::Raise, 0.3, 0.3, 80.0); // u∈[0.22,0.38]
        let near = mk_stroke(SculptStrokeKind::Flatten, 0.4, 0.4, 200.0); // u∈[0.2,0.6]
        let far = mk_stroke(SculptStrokeKind::Flatten, 0.9, 0.9, 40.0); // u∈[0.86,0.94]
        let a = mk_params(vec![raise.clone(), near.clone(), far.clone()]);
        let mut raise2 = raise.clone();
        raise2.strength += 5.0;
        let b = mk_params(vec![raise2, near.clone(), far.clone()]);

        let fp = sculpt_edit_footprint(&a, &b, &m, 0.0).unwrap();
        assert!(
            uv_covers(fp, stroke_footprint_uv(&near, &m).unwrap()),
            "an overlapping enabled Flatten must be unioned in"
        );
        assert!(
            !fp.intersects(stroke_footprint_uv(&far, &m).unwrap()),
            "a disjoint Flatten must stay out (tightness)"
        );
    }

    #[test]
    fn edit_footprint_flatten_chain_is_transitive() {
        let m = fp_metrics();
        let raise = mk_stroke(SculptStrokeKind::Raise, 0.2, 0.5, 100.0); // u∈[0.1,0.3]
        let flat_a = mk_stroke(SculptStrokeKind::Flatten, 0.4, 0.5, 150.0); // u∈[0.25,0.55]
        let flat_b = mk_stroke(SculptStrokeKind::Flatten, 0.65, 0.5, 150.0); // u∈[0.5,0.8]
        let a = mk_params(vec![raise.clone(), flat_a.clone(), flat_b.clone()]);
        let mut raise2 = raise.clone();
        raise2.strength += 5.0;
        let b = mk_params(vec![raise2, flat_a.clone(), flat_b.clone()]);

        // raise∩flat_a and flat_a∩flat_b overlap, but raise∩flat_b is disjoint — so
        // flat_b can only be pulled in transitively through flat_a.
        let fp = sculpt_edit_footprint(&a, &b, &m, 0.0).unwrap();
        assert!(uv_covers(fp, stroke_footprint_uv(&flat_a, &m).unwrap()));
        assert!(
            uv_covers(fp, stroke_footprint_uv(&flat_b, &m).unwrap()),
            "a Flatten reachable only through another Flatten must still be absorbed"
        );
    }

    fn assert_edit_contained(
        m: HeightfieldMetrics,
        h: &Heightfield,
        prev: &SculptStrokeParams,
        next: &SculptStrokeParams,
        label: &str,
    ) {
        // One texel of pad on the shorter axis guarantees ≥1 sample both ways.
        let pad = 1.0 / m.width.min(m.height) as f32;
        let fp = sculpt_edit_footprint(prev, next, &m, pad)
            .unwrap_or_else(|| panic!("{label}: expected a bounded footprint"));
        let tiles = crate::tiling::tiles_for_uv_rect(&m, uv_rect(fp));
        let mut covered = vec![false; (m.width * m.height) as usize];
        for id in &tiles {
            let x0 = id.tx * m.tile_size;
            let y0 = id.tz * m.tile_size;
            let x1 = (x0 + m.tile_size).min(m.width);
            let y1 = (y0 + m.tile_size).min(m.height);
            for j in y0..y1 {
                for i in x0..x1 {
                    covered[(j * m.width + i) as usize] = true;
                }
            }
        }
        let wp = apply_sculpt_strokes(h, prev);
        let wn = apply_sculpt_strokes(h, next);
        let aux_keys = [
            keys::SCULPT_PROTECTION,
            keys::UPLIFT_RATE,
            keys::HARDNESS,
            keys::SEDIMENT_THICKNESS,
            keys::EDIT_REGION,
        ];
        for j in 0..m.height {
            for i in 0..m.width {
                if covered[(j * m.width + i) as usize] {
                    continue;
                }
                assert_eq!(
                    wp.height.get(i, j).to_bits(),
                    wn.height.get(i, j).to_bits(),
                    "{label}: height changed OUTSIDE the reported footprint at ({i},{j})"
                );
                for key in aux_keys {
                    let a = wp.fields.get(key).unwrap().get(i, j);
                    let b = wn.fields.get(key).unwrap().get(i, j);
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{label}: aux {key} changed OUTSIDE the reported footprint at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn edit_footprint_contains_every_changed_sample() {
        // The invariant the whole fix rests on: whole-field prev vs next differ ONLY
        // inside the tiles the reported footprint names — height and every aux field.
        // The stack puts a large Flatten *after* a Raise and a Smooth, so editing an
        // upstream stroke shifts the Flatten across its whole footprint; a footprint
        // missing the Flatten fixpoint would leave changed samples uncovered here.
        let m = fp_metrics();
        let h = fp_field(m);
        let base = vec![
            mk_stroke(SculptStrokeKind::Raise, 0.35, 0.4, 120.0),
            mk_stroke(SculptStrokeKind::Smooth, 0.6, 0.55, 90.0),
            mk_stroke(SculptStrokeKind::Flatten, 0.5, 0.5, 300.0),
            mk_stroke(SculptStrokeKind::Uplift, 0.5, 0.5, 100.0),
        ];
        let prev = mk_params(base.clone());
        let edited = |f: &dyn Fn(&mut Vec<SculptStroke>)| {
            let mut s = base.clone();
            f(&mut s);
            mk_params(s)
        };

        assert_edit_contained(m, &h, &prev, &edited(&|s| s[0].strength += 6.0), "strength@0");
        assert_edit_contained(m, &h, &prev, &edited(&|s| s[0].radius_m = 80.0), "radius-shrink@0");
        assert_edit_contained(m, &h, &prev, &edited(&|s| s[0].radius_m = 180.0), "radius-grow@0");
        assert_edit_contained(m, &h, &prev, &edited(&|s| s[1].enabled = false), "toggle-smooth@1");
        assert_edit_contained(m, &h, &prev, &edited(&|s| { s.remove(1); }), "delete-smooth@1");
        assert_edit_contained(m, &h, &prev, &edited(&|s| s[2].target_height = 40.0), "flatten-target@2");
        assert_edit_contained(m, &h, &prev, &edited(&|s| s[2].strength += 3.0), "flatten-strength@2");
    }
}
