//! Research-backed, non-destructive terrain authoring operations.
//!
//! This module deliberately composes Terra's existing export-oracle hydrology
//! with semantic strokes, constraints and reconstruction. Authoring data stays
//! resolution independent; only evaluation rasterizes it.

use crate::fields::keys;
use crate::heightfield::Heightfield;
use crate::hydro;
use crate::layer::StreamPowerParams;
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Default for SculptStroke {
    fn default() -> Self {
        Self {
            kind: SculptStrokeKind::Raise,
            points: vec![SculptPoint::default()],
            radius_m: sculpt_radius(),
            strength: sculpt_strength(),
            target_height: 0.0,
            falloff: sculpt_falloff(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
                let s = stroke.strength * w;
                let next = match stroke.kind {
                    SculptStrokeKind::Raise => h + s,
                    SculptStrokeKind::Lower => h - s,
                    SculptStrokeKind::Smooth => h + (neighborhood_average(&base, i, j) - h) * w,
                    SculptStrokeKind::Flatten | SculptStrokeKind::HeightStamp => {
                        h + (stroke.target_height - h) * w
                    }
                    SculptStrokeKind::Ridge | SculptStrokeKind::MountainStamp => {
                        let t = (1.0 - distance / stroke.radius_m.max(1.0)).max(0.0);
                        h + s * t * t
                    }
                    SculptStrokeKind::Valley
                    | SculptStrokeKind::ValleyStamp
                    | SculptStrokeKind::RiverPath => {
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
                        uplift[idx] = uplift[idx].max(s.max(0.0));
                        h
                    }
                    SculptStrokeKind::Hardness => {
                        hardness[idx] = hardness[idx].max(s.clamp(0.0, 1.0));
                        h
                    }
                    SculptStrokeKind::Sediment => {
                        sediment[idx] = sediment[idx].max(s.max(0.0));
                        h
                    }
                    SculptStrokeKind::Protect => {
                        protect[idx] = protect[idx].max(s.clamp(0.0, 1.0));
                        h
                    }
                    SculptStrokeKind::EncourageErosion | SculptStrokeKind::Erode => {
                        protect[idx] = (protect[idx] - s.abs().min(1.0)).max(0.0);
                        h - s.abs() * 0.35
                    }
                    SculptStrokeKind::Pinch => {
                        // Pull heights toward neighbourhood mean (contract detail).
                        let avg = neighborhood_average(&base, i, j);
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
                        let avg = neighborhood_average(&base, i, j);
                        let lowered = h - s.abs() * 0.25;
                        (lowered + (avg - lowered) * 0.55) * w + h * (1.0 - w)
                    }
                };
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
}
