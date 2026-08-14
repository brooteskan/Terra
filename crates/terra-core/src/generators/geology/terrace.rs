//! Geological terrace operators — quantise elevation into bands with smooth risers.
//!
//! Not a posterise: tread/riser transitions use smooth nonlinear blends. Irregular
//! mode perturbs elevation / spacing / riser phase in world space *before*
//! quantisation; it never adds post-terrace vertical noise.

use crate::heightfield::Heightfield;
use crate::noise::{self, worley2};
use crate::noise::{FractalNoiseType, WorleyMetric};

use super::super::filter_kernels::{gradient, phasor_noise, slope_deg};

/// Artist controls for the terrace family.
#[derive(Debug, Clone, Copy)]
pub struct TerraceControls {
    /// Vertical terrace interval in meters. When `<= 0`, derive from `levels`.
    pub height_m: f32,
    /// Level count used when `height_m <= 0` (clamped ≥ 2).
    pub levels: u32,
    /// Phase offset in \[0, 1) of the terrace interval.
    pub offset: f32,
    /// Round tread edges toward the source height \[0, 1\].
    pub top_smoothness: f32,
    /// Riser sharpness \[0 soft … 1 sheer\].
    pub riser_sharpness: f32,
    /// World-space perturbation frequency (irregular / steep orientation).
    pub frequency: f32,
    pub seed: u64,
    /// Steep mode: slopes below this (degrees) receive little terracing.
    pub slope_min: f32,
    /// Steep mode: full terracing by this slope (degrees).
    pub slope_max: f32,
}

impl Default for TerraceControls {
    fn default() -> Self {
        Self {
            height_m: 0.0,
            levels: 8,
            offset: 0.0,
            top_smoothness: 0.15,
            riser_sharpness: 0.75,
            frequency: 0.04,
            seed: 1,
            slope_min: 8.0,
            slope_max: 55.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerraceMode {
    Simple,
    Irregular,
    Steep,
}

#[inline]
fn smootherstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn field_range(input: &Heightfield) -> (f32, f32) {
    let mut min_h = f32::INFINITY;
    let mut max_h = f32::NEG_INFINITY;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let v = input.get(i, j);
            min_h = min_h.min(v);
            max_h = max_h.max(v);
        }
    }
    (min_h, max_h)
}

/// World-space perturbation of (elevation band, spacing, riser phase).
/// Returns multipliers/offsets applied in the quantisation domain only.
fn irregular_perturb(x: f32, z: f32, freq: f32, seed: u64) -> (f32, f32, f32) {
    let f = freq.max(1e-5);
    // Low-frequency coherent elevation shift (fraction of local interval later).
    let elev = noise::sample_noise(FractalNoiseType::Perlin, x * f, z * f, seed);
    // Voronoi cells → local spacing variation (procedural noise cells).
    let cell = worley2(
        x * f * 0.85,
        z * f * 0.85,
        seed ^ 0x51CEu64,
        WorleyMetric::Euclidean,
    );
    let spacing = ((cell.f2 - cell.f1) * 2.0 - 1.0).clamp(-1.0, 1.0);
    // Phasor → riser phase wander without high-frequency chatter.
    let phase = phasor_noise(x, z, f * 0.7, seed ^ 0xA11Au64);
    (elev, spacing * 0.35, phase * 0.5)
}

/// Quantise `h` into terrace bands with smooth risers.
///
/// `elev_pert` / `spacing_pert` / `phase_pert` warp the quantisation domain only.
fn quantize_band(
    h: f32,
    min_h: f32,
    range: f32,
    c: &TerraceControls,
    elev_pert: f32,
    spacing_pert: f32,
    phase_pert: f32,
) -> f32 {
    let levels = c.levels.max(2) as f32;
    let base_interval = if c.height_m > 1e-4 {
        c.height_m.max(1e-4)
    } else {
        (range / levels).max(1e-4)
    };
    let interval = (base_interval * (1.0 + spacing_pert * 0.45)).max(1e-4);
    let phase = (c.offset + phase_pert).rem_euclid(1.0);
    // Perturb which band the sample falls into — not the output height after stepping.
    let h_q = h + elev_pert * interval * 0.55;
    let t = (h_q - min_h) / interval + phase;
    let band = t.floor();
    let frac = t - band;

    let sharp = c.riser_sharpness.clamp(0.0, 1.0);
    // Sheer risers → narrow transition; soft → wide nonlinear blend.
    let riser_w = (1.0 - sharp).clamp(0.04, 0.92);
    let tread_end = 1.0 - riser_w;

    let stepped = if frac <= tread_end {
        band
    } else {
        let u = (frac - tread_end) / riser_w.max(1e-4);
        band + smootherstep(u)
    };

    let mut out = min_h + (stepped - phase) * interval;

    // Top smoothness: ease tread corners back toward the source elevation.
    let top = c.top_smoothness.clamp(0.0, 1.0);
    if top > 1e-4 {
        let edge = if frac <= tread_end {
            // Distance to riser start (0 at riser, 1 deep on tread).
            ((tread_end - frac) / tread_end.max(1e-4)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let soften = (1.0 - smootherstep(edge)) * top;
        out = out * (1.0 - soften * 0.65) + h * (soften * 0.65);
    }

    out.clamp(min_h, min_h + range)
}

fn apply_terrace(input: &Heightfield, c: &TerraceControls, mode: TerraceMode) -> Heightfield {
    let mut out = input.clone();
    let (min_h, max_h) = field_range(input);
    let range = (max_h - min_h).max(1e-5);
    let freq = c.frequency.max(1e-5);

    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let h0 = input.get(i, j);
            let x = input.metrics.world_x(i);
            let z = input.metrics.world_z(j);

            let (elev_p, spacing_p, mut phase_p) = match mode {
                TerraceMode::Simple => (0.0, 0.0, 0.0),
                TerraceMode::Irregular => irregular_perturb(x, z, freq, c.seed),
                TerraceMode::Steep => {
                    // Mild coherent phase only — orientation comes from slope/normal.
                    let p = noise::sample_noise(
                        FractalNoiseType::Perlin,
                        x * freq * 0.5,
                        z * freq * 0.5,
                        c.seed ^ 0x57EEu64,
                    ) * 0.15;
                    (0.0, 0.0, p)
                }
            };

            // Steep: terrace orientation follows terrain — phase advances along gradient.
            let mut weight = 1.0f32;
            if mode == TerraceMode::Steep {
                let slope = slope_deg(input, i, j);
                let lo = c.slope_min.min(c.slope_max);
                let hi = c.slope_min.max(c.slope_max).max(lo + 1e-3);
                weight = smootherstep(((slope - lo) / (hi - lo)).clamp(0.0, 1.0));
                let (gx, gz, mag) = gradient(input, i, j);
                if mag > 1e-6 {
                    // Riser faces downhill: phase from projection onto slope direction.
                    phase_p += (x * gx + z * gz) * freq * 0.08;
                }
            }

            let q = quantize_band(h0, min_h, range, c, elev_p, spacing_p, phase_p);
            let h_out = if weight >= 1.0 - 1e-5 {
                q
            } else if weight <= 1e-5 {
                h0
            } else {
                h0 * (1.0 - weight) + q * weight
            };
            out.set(i, j, h_out);
        }
    }
    out
}

pub fn terrace_simple(input: &Heightfield, c: &TerraceControls) -> Heightfield {
    apply_terrace(input, c, TerraceMode::Simple)
}

pub fn terrace_irregular(input: &Heightfield, c: &TerraceControls) -> Heightfield {
    apply_terrace(input, c, TerraceMode::Irregular)
}

pub fn terrace_steep(input: &Heightfield, c: &TerraceControls) -> Heightfield {
    apply_terrace(input, c, TerraceMode::Steep)
}
