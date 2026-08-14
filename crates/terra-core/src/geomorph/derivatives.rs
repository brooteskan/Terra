//! Multi-scale terrain derivatives in world units.
//!
//! Finite differences use sampling radius expressed in metres (converted to
//! texels via [`crate::heightfield::world_radius_texels`]). Profile / plan /
//! mean / Gaussian curvatures follow Zevenbergen–Thorne / Evans forms.

use crate::heightfield::{world_radius_texels, Heightfield};
use crate::mask::MaskField;

/// Sampling controls for derivative kernels.
#[derive(Debug, Clone)]
pub struct DerivativeOptions {
    /// Primary analysis radius in world metres (0 → 1 texel).
    pub radius_m: f32,
    /// Additional radii (metres) for multi-scale roughness / openness.
    pub roughness_radii_m: Vec<f32>,
    /// Openness sector count (Yokoyama-style cavity approximation).
    pub openness_sectors: u32,
}

impl Default for DerivativeOptions {
    fn default() -> Self {
        Self {
            radius_m: 0.0,
            roughness_radii_m: vec![8.0, 32.0, 128.0],
            openness_sectors: 8,
        }
    }
}

/// Bundle of commonly requested morphometric fields (mask-normalised where noted).
#[derive(Debug, Clone)]
pub struct DerivativeSet {
    pub gradient_x: MaskField,
    pub gradient_z: MaskField,
    pub gradient_mag: MaskField,
    pub slope: MaskField,
    pub aspect: MaskField,
    pub laplacian: MaskField,
    pub profile_curvature: MaskField,
    pub plan_curvature: MaskField,
    pub mean_curvature: MaskField,
    pub gaussian_curvature: MaskField,
    pub convexity: MaskField,
    pub concavity: MaskField,
    pub roughness: MaskField,
    pub cavity: MaskField,
    pub ridge_likelihood: MaskField,
    pub valley_likelihood: MaskField,
}

impl DerivativeSet {
    pub fn compute(hf: &Heightfield, opts: &DerivativeOptions) -> Self {
        let (gx, gz, gmag) = gradient_components(hf, opts.radius_m);
        let slope = slope_magnitude(hf, opts.radius_m);
        let aspect = aspect_radians(hf, opts.radius_m);
        let lap = laplacian(hf, opts.radius_m);
        let profile = profile_curvature(hf, opts.radius_m);
        let plan = plan_curvature(hf, opts.radius_m);
        let mean = mean_curvature(hf, opts.radius_m);
        let gauss = gaussian_curvature(hf, opts.radius_m);
        let (convex, concav) = convexity_concavity(hf, opts.radius_m);
        let roughness = multi_radius_roughness(hf, &opts.roughness_radii_m);
        let cavity = cavity_openness(hf, opts.radius_m.max(16.0), opts.openness_sectors);
        let (ridge, valley) = ridge_valley_likelihood(hf, opts.radius_m);

        Self {
            gradient_x: gx,
            gradient_z: gz,
            gradient_mag: gmag,
            slope,
            aspect,
            laplacian: lap,
            profile_curvature: profile,
            plan_curvature: plan,
            mean_curvature: mean,
            gaussian_curvature: gauss,
            convexity: convex,
            concavity: concav,
            roughness,
            cavity,
            ridge_likelihood: ridge,
            valley_likelihood: valley,
        }
    }
}

#[inline]
fn sample_radius_texels(hf: &Heightfield, radius_m: f32) -> i32 {
    let t = world_radius_texels(radius_m, hf.metrics);
    t.round().max(1.0) as i32
}

#[inline]
fn sample_h(hf: &Heightfield, i: i32, j: i32) -> f32 {
    let w = hf.metrics.width as i32;
    let h = hf.metrics.height as i32;
    let ii = i.clamp(0, w - 1) as u32;
    let jj = j.clamp(0, h - 1) as u32;
    hf.get(ii, jj)
}

/// First derivatives and magnitude (world units: ∂h/∂x, ∂h/∂z, |∇h|).
///
/// Magnitude is stored raw (not mask-normalised) so consumers can convert.
pub fn gradient_components(hf: &Heightfield, radius_m: f32) -> (MaskField, MaskField, MaskField) {
    use rayon::prelude::*;

    let m = hf.metrics;
    let r = sample_radius_texels(hf, radius_m);
    let dx = m.dx() * r as f32;
    let dz = m.dz() * r as f32;
    let w = m.width as usize;
    let n = (m.width * m.height) as usize;
    let computed: Vec<(f32, f32, f32)> = (0..n)
        .into_par_iter()
        .map(|idx| {
            let i = (idx % w) as i32;
            let j = (idx / w) as i32;
            let px = (sample_h(hf, i + r, j) - sample_h(hf, i - r, j)) / (2.0 * dx.max(1e-6));
            let pz = (sample_h(hf, i, j + r) - sample_h(hf, i, j - r)) / (2.0 * dz.max(1e-6));
            (px, pz, (px * px + pz * pz).sqrt())
        })
        .collect();
    let mut gx_data = Vec::with_capacity(n);
    let mut gz_data = Vec::with_capacity(n);
    let mut mag_data = Vec::with_capacity(n);
    for (px, pz, mag) in computed {
        gx_data.push(px);
        gz_data.push(pz);
        mag_data.push(mag);
    }
    (
        MaskField::from_raw(m, &gx_data),
        MaskField::from_raw(m, &gz_data),
        MaskField::from_raw(m, &mag_data),
    )
}

/// Slope as |∇h| mapped to \[0,1\] via atan → degrees / 90 (artist-friendly).
pub fn slope_magnitude(hf: &Heightfield, radius_m: f32) -> MaskField {
    let (_, _, mag) = gradient_components(hf, radius_m);
    let m = hf.metrics;
    let mut out = MaskField::zeros(m);
    for j in 0..m.height {
        for i in 0..m.width {
            let s = mag.get(i, j).atan().to_degrees() / 90.0;
            out.set(i, j, s.clamp(0.0, 1.0));
        }
    }
    out
}

/// Aspect in radians \[0, 2π), stored as value / 2π in \[0,1\].
pub fn aspect_radians(hf: &Heightfield, radius_m: f32) -> MaskField {
    let (gx, gz, _) = gradient_components(hf, radius_m);
    let m = hf.metrics;
    let mut out = MaskField::zeros(m);
    let two_pi = std::f32::consts::TAU;
    for j in 0..m.height {
        for i in 0..m.width {
            let aspect = gz.get(i, j).atan2(-gx.get(i, j)).rem_euclid(two_pi);
            out.set(i, j, (aspect / two_pi).clamp(0.0, 1.0));
        }
    }
    out
}

fn second_derivatives(
    hf: &Heightfield,
    radius_m: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let m = hf.metrics;
    let w = m.width as usize;
    let h = m.height as usize;
    let r = sample_radius_texels(hf, radius_m);
    let dx = m.dx() * r as f32;
    let dz = m.dz() * r as f32;
    let dx2 = (dx * dx).max(1e-12);
    let dz2 = (dz * dz).max(1e-12);
    let dxdz = (dx * dz).max(1e-12);

    let mut p = vec![0.0f32; w * h];
    let mut q = vec![0.0f32; w * h];
    let mut rr = vec![0.0f32; w * h];
    let mut tt = vec![0.0f32; w * h];
    let mut ss = vec![0.0f32; w * h];

    for j in 0..h as i32 {
        for i in 0..w as i32 {
            let idx = j as usize * w + i as usize;
            let z = sample_h(hf, i, j);
            let zp = sample_h(hf, i + r, j);
            let zm = sample_h(hf, i - r, j);
            let zq = sample_h(hf, i, j + r);
            let zn = sample_h(hf, i, j - r);
            let zpp = sample_h(hf, i + r, j + r);
            let zpm = sample_h(hf, i + r, j - r);
            let zmp = sample_h(hf, i - r, j + r);
            let zmm = sample_h(hf, i - r, j - r);

            p[idx] = (zp - zm) / (2.0 * dx.max(1e-6));
            q[idx] = (zq - zn) / (2.0 * dz.max(1e-6));
            rr[idx] = (zp - 2.0 * z + zm) / dx2;
            tt[idx] = (zq - 2.0 * z + zn) / dz2;
            ss[idx] = (zpp - zpm - zmp + zmm) / (4.0 * dxdz);
        }
    }
    (p, q, rr, tt, ss)
}

fn map_signed_curvature(raw: &[f32], m: crate::heightfield::HeightfieldMetrics) -> MaskField {
    let mut out = MaskField::zeros(m);
    let max_abs = raw
        .iter()
        .copied()
        .fold(0.0f32, |a, v| a.max(v.abs()))
        .max(1e-6);
    for j in 0..m.height {
        for i in 0..m.width {
            let v = raw[(j * m.width + i) as usize];
            // Map signed curvature to [0,1] with 0.5 = flat.
            out.set(
                i,
                j,
                (0.5 + 0.5 * (v / max_abs).clamp(-1.0, 1.0)).clamp(0.0, 1.0),
            );
        }
    }
    out
}

/// Discrete Laplacian ∇²h, mapped to \[0,1\] around 0.5.
pub fn laplacian(hf: &Heightfield, radius_m: f32) -> MaskField {
    let (_, _, r, t, _) = second_derivatives(hf, radius_m);
    let mut raw = vec![0.0f32; r.len()];
    for i in 0..raw.len() {
        raw[i] = r[i] + t[i];
    }
    map_signed_curvature(&raw, hf.metrics)
}

/// Profile (vertical) curvature.
pub fn profile_curvature(hf: &Heightfield, radius_m: f32) -> MaskField {
    let (p, q, r, t, s) = second_derivatives(hf, radius_m);
    let mut raw = vec![0.0f32; p.len()];
    for i in 0..raw.len() {
        let p2 = p[i] * p[i];
        let q2 = q[i] * q[i];
        let pq = p2 + q2;
        if pq < 1e-12 {
            raw[i] = 0.0;
            continue;
        }
        let num = -(r[i] * p2 + 2.0 * s[i] * p[i] * q[i] + t[i] * q2);
        let den = pq * (1.0 + pq).powf(1.5);
        raw[i] = num / den.max(1e-12);
    }
    map_signed_curvature(&raw, hf.metrics)
}

/// Plan (contour) curvature.
pub fn plan_curvature(hf: &Heightfield, radius_m: f32) -> MaskField {
    let (p, q, r, t, s) = second_derivatives(hf, radius_m);
    let mut raw = vec![0.0f32; p.len()];
    for i in 0..raw.len() {
        let p2 = p[i] * p[i];
        let q2 = q[i] * q[i];
        let pq = p2 + q2;
        if pq < 1e-12 {
            raw[i] = 0.0;
            continue;
        }
        let num = -(t[i] * p2 - 2.0 * s[i] * p[i] * q[i] + r[i] * q2);
        let den = pq.powf(1.5);
        raw[i] = num / den.max(1e-12);
    }
    map_signed_curvature(&raw, hf.metrics)
}

/// Mean curvature of the height surface.
pub fn mean_curvature(hf: &Heightfield, radius_m: f32) -> MaskField {
    let (p, q, r, t, s) = second_derivatives(hf, radius_m);
    let mut raw = vec![0.0f32; p.len()];
    for i in 0..raw.len() {
        let p2 = p[i] * p[i];
        let q2 = q[i] * q[i];
        let num = (1.0 + q2) * r[i] - 2.0 * p[i] * q[i] * s[i] + (1.0 + p2) * t[i];
        let den = 2.0 * (1.0 + p2 + q2).powf(1.5);
        raw[i] = num / den.max(1e-12);
    }
    map_signed_curvature(&raw, hf.metrics)
}

/// Gaussian curvature K = (rt − s²) / (1+p²+q²)².
pub fn gaussian_curvature(hf: &Heightfield, radius_m: f32) -> MaskField {
    let (p, q, r, t, s) = second_derivatives(hf, radius_m);
    let mut raw = vec![0.0f32; p.len()];
    for i in 0..raw.len() {
        let den = (1.0 + p[i] * p[i] + q[i] * q[i]).powi(2);
        raw[i] = (r[i] * t[i] - s[i] * s[i]) / den.max(1e-12);
    }
    map_signed_curvature(&raw, hf.metrics)
}

/// Convexity / concavity from mean curvature (positive / negative lobes).
pub fn convexity_concavity(hf: &Heightfield, radius_m: f32) -> (MaskField, MaskField) {
    let mean = mean_curvature(hf, radius_m);
    let m = hf.metrics;
    let mut convex = MaskField::zeros(m);
    let mut concav = MaskField::zeros(m);
    for j in 0..m.height {
        for i in 0..m.width {
            let v = mean.get(i, j);
            convex.set(i, j, ((v - 0.5).max(0.0) * 2.0).clamp(0.0, 1.0));
            concav.set(i, j, ((0.5 - v).max(0.0) * 2.0).clamp(0.0, 1.0));
        }
    }
    (convex, concav)
}

/// Multi-radius height-std roughness, averaged across requested radii.
pub fn multi_radius_roughness(hf: &Heightfield, radii_m: &[f32]) -> MaskField {
    let m = hf.metrics;
    let mut out = MaskField::zeros(m);
    if radii_m.is_empty() {
        return out;
    }
    let mut acc = vec![0.0f32; (m.width * m.height) as usize];
    let mut weight = 0.0f32;
    for &rm in radii_m {
        let r = sample_radius_texels(hf, rm);
        weight += 1.0;
        for j in 0..m.height as i32 {
            for i in 0..m.width as i32 {
                let mut sum = 0.0f32;
                let mut sum2 = 0.0f32;
                let mut n = 0.0f32;
                for dj in -r..=r {
                    for di in -r..=r {
                        if di * di + dj * dj > r * r {
                            continue;
                        }
                        let z = sample_h(hf, i + di, j + dj);
                        sum += z;
                        sum2 += z * z;
                        n += 1.0;
                    }
                }
                let n = n.max(1.0);
                let mean = sum / n;
                let var = (sum2 / n - mean * mean).max(0.0);
                acc[(j * m.width as i32 + i) as usize] += var.sqrt();
            }
        }
    }
    let max_v = acc.iter().copied().fold(0.0f32, f32::max).max(1e-6);
    for j in 0..m.height {
        for i in 0..m.width {
            let v = acc[(j * m.width + i) as usize] / weight.max(1.0);
            out.set(i, j, (v / max_v).clamp(0.0, 1.0));
        }
    }
    out
}

/// Yokoyama-style openness / cavity approximation.
///
/// Positive openness (sky visibility) is inverted so cavities → 1.
pub fn cavity_openness(hf: &Heightfield, radius_m: f32, sectors: u32) -> MaskField {
    let m = hf.metrics;
    let mut out = MaskField::zeros(m);
    let r_max = sample_radius_texels(hf, radius_m).max(2);
    let sectors = sectors.max(4);
    let dx = m.dx();
    let dz = m.dz();

    for j in 0..m.height as i32 {
        for i in 0..m.width as i32 {
            let h0 = sample_h(hf, i, j);
            let mut openness = 0.0f32;
            for s in 0..sectors {
                let ang = (s as f32 / sectors as f32) * std::f32::consts::TAU;
                let dir_x = ang.cos();
                let dir_z = ang.sin();
                let mut max_elev = -std::f32::consts::FRAC_PI_2;
                for step in 1..=r_max {
                    let ni = i + (dir_x * step as f32).round() as i32;
                    let nj = j + (dir_z * step as f32).round() as i32;
                    if ni < 0 || nj < 0 || ni >= m.width as i32 || nj >= m.height as i32 {
                        break;
                    }
                    let dist =
                        ((step as f32 * dir_x * dx).hypot(step as f32 * dir_z * dz)).max(1e-6);
                    let elev = ((sample_h(hf, ni, nj) - h0) / dist).atan();
                    if elev > max_elev {
                        max_elev = elev;
                    }
                }
                // Positive openness ≈ π/2 − max elevation angle.
                openness += std::f32::consts::FRAC_PI_2 - max_elev;
            }
            let mean_open = openness / sectors as f32;
            // Cavity: low openness.
            let cavity =
                (1.0 - (mean_open / std::f32::consts::FRAC_PI_2).clamp(0.0, 1.0)).clamp(0.0, 1.0);
            out.set(i as u32, j as u32, cavity);
        }
    }
    out
}

/// Ridge / valley likelihood from plan curvature and local relief.
pub fn ridge_valley_likelihood(hf: &Heightfield, radius_m: f32) -> (MaskField, MaskField) {
    let plan = plan_curvature(hf, radius_m);
    let slope = slope_magnitude(hf, radius_m);
    let m = hf.metrics;
    let mut ridge = MaskField::zeros(m);
    let mut valley = MaskField::zeros(m);
    for j in 0..m.height {
        for i in 0..m.width {
            let c = plan.get(i, j);
            let s = slope.get(i, j);
            // Plan > 0.5 → divergent (ridge-ish); < 0.5 → convergent (valley-ish).
            let rid = ((c - 0.5).max(0.0) * 2.0) * (0.35 + 0.65 * s);
            let val = ((0.5 - c).max(0.0) * 2.0) * (1.0 - 0.4 * s);
            ridge.set(i, j, rid.clamp(0.0, 1.0));
            valley.set(i, j, val.clamp(0.0, 1.0));
        }
    }
    (ridge, valley)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

    #[test]
    fn plane_has_constant_slope() {
        let m = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..32 {
            for i in 0..32 {
                // 45° in X: rise/run = 1 (dx=10m → rise 10m per cell).
                hf.set(i, j, i as f32 * 10.0);
            }
        }
        let s = slope_magnitude(&hf, 0.0);
        let v = s.get(16, 16);
        assert!((v - 0.5).abs() < 0.05, "slope={v}");
    }

    #[test]
    fn larger_radius_reduces_high_frequency_response() {
        let m = HeightfieldMetrics::new(48, 48, 480.0, 480.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..48 {
            for i in 0..48 {
                let n = (((i * 17 + j * 31) % 7) as f32 - 3.0) * 0.5;
                hf.set(i, j, n);
            }
        }
        let (_, _, g_fine) = gradient_components(&hf, 10.0);
        let (_, _, g_coarse) = gradient_components(&hf, 80.0);
        let mean = |f: &MaskField| {
            let d = f.data();
            d.iter().sum::<f32>() / d.len() as f32
        };
        // Pure high-frequency noise: coarse finite differences attenuate.
        assert!(
            mean(&g_coarse) < mean(&g_fine),
            "coarse={} fine={}",
            mean(&g_coarse),
            mean(&g_fine)
        );
    }
}
