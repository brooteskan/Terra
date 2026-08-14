//! Tectonic uplift field synthesis (geological-scale smooth).

use crate::heightfield::HeightfieldMetrics;
use crate::mask::MaskField;
use crate::noise::{open_simplex2, value_noise2};

use super::params::{LandscapeEvolutionParams, UpliftMode};

/// Build a normalised uplift rate field in `[0, peak]` metres / year.
///
/// This synthesises the geological field only. The landscape-evolution
/// operator applies its elevation-lock policy afterwards.
pub fn synthesise_uplift(
    metrics: HeightfieldMetrics,
    p: &LandscapeEvolutionParams,
    painted: Option<&MaskField>,
) -> MaskField {
    let w = metrics.width;
    let h = metrics.height;
    let peak = p.peak_uplift_rate();
    let mut field = MaskField::zeros(metrics);

    match p.uplift_mode {
        UpliftMode::Painted => {
            if let Some(src) = painted {
                for j in 0..h {
                    for i in 0..w {
                        field.set(i, j, src.get(i, j).max(0.0) * peak);
                    }
                }
            } else {
                fill_uniform(&mut field, peak);
            }
        }
        UpliftMode::ShapeDerived => {
            if let Some(src) = painted {
                let mut any = false;
                for j in 0..h {
                    for i in 0..w {
                        let v = src.get(i, j).max(0.0);
                        if v > 1e-6 {
                            any = true;
                        }
                        field.set(i, j, v * peak);
                    }
                }
                if !any {
                    fill_linear_belt(&mut field, p, peak);
                }
            } else {
                fill_linear_belt(&mut field, p, peak);
            }
        }
        UpliftMode::Uniform => fill_uniform(&mut field, peak),
        UpliftMode::Radial => fill_radial(&mut field, p, peak),
        UpliftMode::LinearBelt => fill_linear_belt(&mut field, p, peak),
        UpliftMode::Procedural => fill_procedural(&mut field, p, peak),
    }

    // Soft geological-scale noise (never high-frequency scratches).
    let noise_amp = p.uplift_noise.clamp(0.0, 0.5);
    if noise_amp > 1e-6 {
        let seed = p.uplift_seed;
        for j in 0..h {
            for i in 0..w {
                let u = i as f32 / w.max(1) as f32;
                let v = j as f32 / h.max(1) as f32;
                let n = open_simplex2(u * 2.5, v * 2.5, seed) * 0.6
                    + open_simplex2(u * 5.0 + 19.0, v * 5.0 - 7.0, seed.wrapping_add(91)) * 0.4;
                let mul = (1.0 + n * noise_amp).max(0.0);
                field.set(i, j, field.get(i, j) * mul);
            }
        }
    }

    field
}

fn fill_uniform(field: &mut MaskField, peak: f32) {
    let m = field.metrics;
    for j in 0..m.height {
        for i in 0..m.width {
            field.set(i, j, peak);
        }
    }
}

fn fill_radial(field: &mut MaskField, p: &LandscapeEvolutionParams, peak: f32) {
    let m = field.metrics;
    let cu = p.uplift_center_u.clamp(0.0, 1.0);
    let cv = p.uplift_center_v.clamp(0.0, 1.0);
    let fall = p.uplift_falloff.clamp(0.05, 0.9);
    for j in 0..m.height {
        for i in 0..m.width {
            let u = i as f32 / m.width.max(1) as f32;
            let v = j as f32 / m.height.max(1) as f32;
            let d = ((u - cu) / fall).hypot((v - cv) / fall);
            let w = (1.0 - d).clamp(0.0, 1.0);
            let smooth = w * w * (3.0 - 2.0 * w);
            field.set(i, j, peak * smooth);
        }
    }
}

fn fill_linear_belt(field: &mut MaskField, p: &LandscapeEvolutionParams, peak: f32) {
    let m = field.metrics;
    let cu = p.uplift_center_u.clamp(0.0, 1.0);
    let cv = p.uplift_center_v.clamp(0.0, 1.0);
    let half_w = p.uplift_falloff.clamp(0.04, 0.6);
    let ang = p.uplift_angle_rad;
    let (ca, sa) = (ang.cos(), ang.sin());
    for j in 0..m.height {
        for i in 0..m.width {
            let u = i as f32 / m.width.max(1) as f32 - cu;
            let v = j as f32 / m.height.max(1) as f32 - cv;
            // Distance to belt axis (perpendicular).
            let across = (-sa * u + ca * v).abs() / half_w;
            let w = (1.0 - across).clamp(0.0, 1.0);
            let smooth = w * w * (3.0 - 2.0 * w);
            // Mild along-axis taper so ends fade (asymmetric mountain belts).
            let along = (ca * u + sa * v).abs();
            let taper = (1.0 - (along * 0.55).min(1.0)).clamp(0.15, 1.0);
            field.set(i, j, peak * smooth * taper);
        }
    }
}

fn fill_procedural(field: &mut MaskField, p: &LandscapeEvolutionParams, peak: f32) {
    let m = field.metrics;
    let seed = p.uplift_seed;
    for j in 0..m.height {
        for i in 0..m.width {
            let u = i as f32 / m.width.max(1) as f32;
            let v = j as f32 / m.height.max(1) as f32;
            // Very low frequency — geological wavelength only.
            let n0 = value_noise2(u * 1.8, v * 1.8, seed);
            let n1 = open_simplex2(u * 3.2 + 3.0, v * 3.2 - 1.5, seed.wrapping_add(17));
            let n = (n0 * 0.65 + n1 * 0.35).clamp(-1.0, 1.0);
            let w = ((n + 1.0) * 0.5).powf(1.35);
            field.set(i, j, peak * w);
        }
    }
}

/// Asymmetric uplift fixture (stronger on one side of a belt).
pub fn asymmetric_belt(
    metrics: HeightfieldMetrics,
    peak: f32,
    angle_rad: f32,
    bias: f32,
) -> MaskField {
    let mut p = LandscapeEvolutionParams::default();
    p.uplift_mode = UpliftMode::LinearBelt;
    p.uplift_angle_rad = angle_rad;
    p.uplift_falloff = 0.28;
    p.uplift_noise = 0.05;
    let mut field = synthesise_uplift(metrics, &p, None);
    let m = metrics;
    let (ca, sa) = (angle_rad.cos(), angle_rad.sin());
    for j in 0..m.height {
        for i in 0..m.width {
            let u = i as f32 / m.width.max(1) as f32 - 0.5;
            let v = j as f32 / m.height.max(1) as f32 - 0.5;
            let across = -sa * u + ca * v;
            let side = (0.5 + 0.5 * across.signum() * bias.clamp(0.0, 1.0)).clamp(0.2, 1.5);
            field.set(
                i,
                j,
                field.get(i, j) / p.peak_uplift_rate().max(1e-12) * peak * side,
            );
        }
    }
    field
}
