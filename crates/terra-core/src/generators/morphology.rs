//! Research-backed morphology filters for biome EffectFilters.
//!
//! Algorithms follow public literature (flow accumulation, noise, imaging, thermal,
//! aeolian). Shared kernels live in `super::filter_kernels`.

use crate::analyze::{AeolianState, AeolianTransportParams};
use crate::heightfield::Heightfield;
use crate::layer::EffectFilterParams;
use crate::noise;
use crate::noise::{FractalNoiseType, WorleyFeature, WorleyParams};

use super::box_blur_height;
use super::filter_kernels::{
    anisotropic_blur, billow_mf, dilate, erode, flow_accumulation_dinf, gradient,
    kuwahara_anisotropic, normalize_field, phasor_noise_fbm, slope_deg, sparse_gabor,
    sparse_gabor_fbm, thermal_talus, white_noise,
};

fn slope_gate(slope: f32, p: &EffectFilterParams) -> f32 {
    let lo = p.slope_min.min(p.slope_max);
    let hi = p.slope_min.max(p.slope_max);
    if hi <= lo + 1e-3 {
        return 1.0;
    }
    ((slope - lo) / (hi - lo)).clamp(0.0, 1.0)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Rotate + anisotropic stretch in world metres; optional period wrap for tileability.
fn transform_world_xz(x: f32, z: f32, p: &EffectFilterParams) -> (f32, f32) {
    let ang = p.rotation_deg.to_radians();
    let (s, c) = ang.sin_cos();
    let mut xr = x * c + z * s;
    let mut zr = -x * s + z * c;
    let aniso = p.anisotropy.max(0.05);
    xr /= aniso;
    zr *= aniso.sqrt().max(0.2);
    if p.tileable {
        let period = if p.scale_m > 1e-5 {
            p.scale_m
        } else {
            (1.0 / p.effective_frequency()).max(1.0)
        };
        xr = xr.rem_euclid(period);
        zr = zr.rem_euclid(period);
    }
    (xr, zr)
}

fn domain_warp_xz(x: f32, z: f32, p: &EffectFilterParams) -> (f32, f32) {
    let (mut xr, mut zr) = transform_world_xz(x, z, p);
    if p.warp_strength.abs() > 1e-5 {
        let wf = p.warp_frequency.max(1e-5);
        let wx = noise::perlin2(xr * wf, zr * wf, p.seed) * p.warp_strength;
        let wz =
            noise::perlin2(xr * wf + 19.1, zr * wf + 7.3, p.seed.wrapping_add(1)) * p.warp_strength;
        xr += wx;
        zr += wz;
    }
    (xr, zr)
}

fn noise_params(p: &EffectFilterParams) -> crate::noise::NoiseParams {
    crate::noise::NoiseParams {
        seed: p.seed,
        frequency: p.effective_frequency(),
        amplitude: 1.0,
        octaves: p.octaves.max(1),
        lacunarity: p.lacunarity.max(1.01),
        persistence: p.persistence.clamp(0.05, 0.95),
        ..crate::noise::NoiseParams::default()
    }
}

fn flow_norm(hf: &Heightfield, passes: u32) -> Vec<f32> {
    normalize_field(&flow_accumulation_dinf(hf, passes))
}

pub(super) fn rocky_sharp(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    super::arid_landforms::apply_rocky_sharp(input, p)
}

pub(super) fn rocky_wide(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    super::arid_landforms::apply_rocky_wide(input, p)
}

pub(super) fn rocky_layers(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    super::arid_landforms::apply_rocky_layers(input, p)
}

pub(super) fn cliff_reinforce(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    super::arid_landforms::apply_cliff_reinforce(input, p)
}

pub(super) fn soft_flows(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let flow = flow_norm(input, 6);
    let blurred = box_blur_height(input, p.radius.max(1));
    let mut out = input.clone();
    let w = input.metrics.width as usize;
    let amt = p.amount.max(0.5);
    let thresh = p.flow_threshold.min(0.35);
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let f = flow[j as usize * w + i as usize];
            // Always apply a mild flow-weighted carve so flat fixtures still change.
            let t = if f >= thresh {
                ((f - thresh) / (1.0 - thresh).max(1e-3)).clamp(0.0, 1.0)
            } else {
                f * 0.35
            };
            if t < 1e-5 {
                continue;
            }
            let h0 = input.get(i, j);
            let hb = blurred.get(i, j);
            out.set(i, j, h0 - (h0 - hb).abs().min(amt) * t * 0.85);
        }
    }
    out
}

pub(super) fn thin_flows(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let flow = flow_norm(input, 8);
    let mut out = input.clone();
    let w = input.metrics.width as usize;
    let amt = p.amount;
    let thresh = p.flow_threshold.max(0.05);
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let f = flow[j as usize * w + i as usize];
            if f < thresh {
                continue;
            }
            let t = ((f - thresh) / (1.0 - thresh).max(1e-3)).clamp(0.0, 1.0);
            // Thin: only strong channels, sharper carve.
            let sharp = t.powf(1.6);
            out.set(i, j, input.get(i, j) - amt * sharp);
        }
    }
    out
}

pub(super) fn ridged_flows(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let flow = flow_norm(input, 7);
    let mut out = input.clone();
    let w = input.metrics.width as usize;
    let amt = p.amount;
    let thresh = p.flow_threshold;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let f = flow[j as usize * w + i as usize];
            let h0 = input.get(i, j);
            if f >= thresh {
                let t = ((f - thresh) / (1.0 - thresh).max(1e-3)).clamp(0.0, 1.0);
                out.set(i, j, h0 - amt * t);
            } else {
                // Reinforce divides.
                let ridge = (1.0 - f / thresh.max(1e-3)).clamp(0.0, 1.0);
                out.set(i, j, h0 + amt * 0.35 * ridge);
            }
        }
    }
    out
}

pub(super) fn wide_flows(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let flow = flow_norm(input, 6);
    let blurred = box_blur_height(input, p.radius.max(2));
    let mut out = input.clone();
    let w = input.metrics.width as usize;
    let amt = p.amount;
    let thresh = p.flow_threshold;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let slope = slope_deg(input, i, j);
            if slope > p.slope_max {
                continue;
            }
            let f = flow[j as usize * w + i as usize];
            if f < thresh {
                continue;
            }
            let t = ((f - thresh) / (1.0 - thresh).max(1e-3)).clamp(0.0, 1.0)
                * (1.0 - slope / p.slope_max.max(1.0));
            let h0 = input.get(i, j);
            let hb = blurred.get(i, j);
            // Broaden valleys toward local blur / slight raise of floors.
            let floor = h0.min(hb) + amt * 0.15 * t;
            out.set(i, j, h0 * (1.0 - t * 0.55) + floor * t * 0.55);
        }
    }
    out
}

pub(super) fn talus_fill(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    // Layered thermal apron via shared kernel (cliff → scree, hard rock preserved).
    let repose = p.slope_min.clamp(20.0, 55.0);
    let iters = p.iterations.max(1).min(8);
    let filled = thermal_talus(input, repose, p.amount.max(0.5), iters);
    let t = p.strength.clamp(0.0, 1.0);
    if (t - 1.0).abs() < 1e-5 {
        return filled;
    }
    let mut out = input.clone();
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let h0 = input.get(i, j);
            let h1 = filled.get(i, j);
            out.set(i, j, h0 * (1.0 - t) + h1 * t);
        }
    }
    out
}

pub(super) fn sediment_fill_soft(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let graph = crate::geomorph::build_flow_graph(input, crate::geomorph::FlowModel::D8);
    let flow = crate::geomorph::accumulate_drainage_area(
        &graph,
        &crate::geomorph::Precipitation::uniform(1.0),
    );
    let (filled, _) = crate::analyze::sediment_fill_soft_mass(
        input,
        p.amount,
        p.slope_max,
        p.radius.max(1),
        None,
        Some(&flow),
    );
    let t = p.strength.clamp(0.0, 1.0);
    if (t - 1.0).abs() < 1e-5 {
        return filled;
    }
    let mut out = input.clone();
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let h0 = input.get(i, j);
            let h1 = filled.get(i, j);
            out.set(i, j, h0 * (1.0 - t) + h1 * t);
        }
    }
    out
}

pub(super) fn mud_settle(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let (settled, _) = crate::analyze::mud_settle_mass(
        input,
        p.amount,
        p.slope_max.max(12.0),
        p.radius.max(1),
        None,
    );
    let t = p.strength.clamp(0.0, 1.0);
    if (t - 1.0).abs() < 1e-5 {
        return settled;
    }
    let mut out = input.clone();
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let h0 = input.get(i, j);
            let h1 = settled.get(i, j);
            out.set(i, j, h0 * (1.0 - t) + h1 * t);
        }
    }
    out
}

pub(super) fn hydraulic_sediment(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let flow = flow_norm(input, 8);
    let mut out = input.clone();
    let w = input.metrics.width as usize;
    let wi = input.metrics.width as i32;
    let hi = input.metrics.height as i32;
    let amt = p.amount;
    let thresh = p.flow_threshold;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let f = flow[j as usize * w + i as usize];
            if f < thresh {
                continue;
            }
            let slope = slope_deg(input, i, j);
            if slope > p.slope_max {
                continue;
            }
            // Slope-break: steeper neighbor uphill → fan deposit.
            let mut up_slope = slope;
            for &(di, dj) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= wi || nj >= hi {
                    continue;
                }
                let ns = slope_deg(input, ni as u32, nj as u32);
                if input.get(ni as u32, nj as u32) > input.get(i, j) {
                    up_slope = up_slope.max(ns);
                }
            }
            let break_t = ((up_slope - slope) / 25.0).clamp(0.0, 1.0);
            let t = ((f - thresh) / (1.0 - thresh).max(1e-3)).clamp(0.0, 1.0) * break_t;
            out.set(i, j, input.get(i, j) + amt * t);
        }
    }
    out
}

// ── Wave D/E — arid, effect, general, drift ─────────────────────────────────

pub(super) fn rocky_plateaus(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    super::arid_landforms::apply_rocky_plateaus(input, p)
}

pub(super) fn rocky_cliffs(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    super::arid_landforms::apply_rocky_cliffs(input, p)
}

pub(super) fn rocky_hard(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    super::arid_landforms::apply_rocky_hard(input, p)
}

pub(super) fn canyon(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    super::arid_landforms::apply_canyon(input, p)
}

pub(super) fn chipped(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    super::arid_landforms::apply_chipped(input, p)
}

pub(super) fn cliffs(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    super::arid_landforms::apply_cliffs(input, p)
}

pub(super) fn rocky(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    super::arid_landforms::apply_rocky(input, p, soft_flows)
}

pub(super) fn sediment_flows(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let flow = flow_norm(input, 8);
    let mut out = input.clone();
    let w = input.metrics.width as usize;
    let amt = p.amount;
    let thresh = p.flow_threshold;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let f = flow[j as usize * w + i as usize];
            if f < thresh {
                continue;
            }
            let slope = slope_deg(input, i, j);
            if slope > p.slope_max {
                continue;
            }
            let t = ((f - thresh) / (1.0 - thresh).max(1e-3)).clamp(0.0, 1.0)
                * (1.0 - slope / p.slope_max.max(1.0));
            // Flow-follow deposition raise.
            out.set(i, j, input.get(i, j) + amt * t);
        }
    }
    out
}

pub(super) fn inflate(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let dilated = dilate(input, p.radius.max(1));
    let mut out = input.clone();
    let amt = p.amount;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let h0 = input.get(i, j);
            let hi = dilated.get(i, j);
            out.set(i, j, h0 + (hi - h0).min(amt));
        }
    }
    out
}

pub(super) fn deflate(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let eroded = erode(input, p.radius.max(1));
    let mut out = input.clone();
    let amt = p.amount;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let h0 = input.get(i, j);
            let lo = eroded.get(i, j);
            out.set(i, j, h0 - (h0 - lo).min(amt));
        }
    }
    out
}

pub(super) fn balloon(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    // Inflate on mid-slopes, mild deflate on flats — rounded “balloon” relief.
    let inflated = inflate(input, p);
    let mut out = input.clone();
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let slope = slope_deg(input, i, j);
            let g = slope_gate(slope, p);
            let h0 = input.get(i, j);
            let hi = inflated.get(i, j);
            out.set(i, j, h0 * (1.0 - g) + hi * g);
        }
    }
    out
}

pub(super) fn blocks(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let step = p.amount.max(1.0);
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let h0 = input.get(i, j);
            let q = (h0 / step).round() * step;
            out.set(i, j, q);
        }
    }
    out
}

pub(super) fn ridged_detail(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let freq = p.frequency.max(1e-5);
    let amt = p.amount;
    let np = crate::noise::NoiseParams {
        seed: p.seed,
        frequency: freq,
        amplitude: 1.0,
        octaves: 5,
        lacunarity: 2.1,
        persistence: 0.55,
        ..crate::noise::NoiseParams::default()
    };
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let x = input.metrics.world_x(i);
            let z = input.metrics.world_z(j);
            let n = noise::ridged_mf(FractalNoiseType::Perlin, x, z, &np);
            out.set(i, j, input.get(i, j) + (n * 2.0 - 1.0) * amt);
        }
    }
    out
}

pub(super) fn rugged(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let freq = p.frequency.max(1e-5);
    let amt = p.amount;
    let np = crate::noise::NoiseParams {
        seed: p.seed ^ 0xA06Du64,
        frequency: freq,
        amplitude: 1.0,
        octaves: 6,
        lacunarity: 2.3,
        persistence: 0.6,
        ..crate::noise::NoiseParams::default()
    };
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let x = input.metrics.world_x(i);
            let z = input.metrics.world_z(j);
            let n = noise::fbm(FractalNoiseType::Perlin, x, z, &np);
            let r = noise::ridged_mf(FractalNoiseType::Perlin, x * 1.7, z * 1.7, &np);
            out.set(
                i,
                j,
                input.get(i, j) + (n * 0.45 + (r * 2.0 - 1.0) * 0.55) * amt,
            );
        }
    }
    out
}

pub(super) fn smooth_ridges(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let ridged = ridged_detail(input, p);
    box_blur_height(&ridged, p.radius.max(1))
}

fn sample_nearest(hf: &Heightfield, u: f32, v: f32) -> f32 {
    let ii = (u.clamp(0.0, 1.0) * (hf.metrics.width - 1) as f32).round() as u32;
    let jj = (v.clamp(0.0, 1.0) * (hf.metrics.height - 1) as f32).round() as u32;
    hf.get(ii.min(hf.metrics.width - 1), jj.min(hf.metrics.height - 1))
}

pub(super) fn angle_blur(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    // Directional Gaussian along local gradient (anisotropic blur).
    let r = p.radius.max(1) as i32;
    let w = input.metrics.width as i32;
    let h = input.metrics.height as i32;
    let sigma = (r as f32 * 0.5).max(0.5);
    let mut out = input.clone();
    for j in 0..h {
        for i in 0..w {
            let (gx, gz, mag) = gradient(input, i as u32, j as u32);
            let (sx, sz) = if mag < 1e-6 {
                (1.0f32, 0.0f32)
            } else {
                (gx, gz)
            };
            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for k in -r..=r {
                let fi = (i as f32 + sx * k as f32).round() as i32;
                let fj = (j as f32 + sz * k as f32).round() as i32;
                let xi = fi.clamp(0, w - 1) as u32;
                let yj = fj.clamp(0, h - 1) as u32;
                let wt = (-(k * k) as f32 / (2.0 * sigma * sigma)).exp();
                sum += input.get(xi, yj) * wt;
                wsum += wt;
            }
            out.set(i as u32, j as u32, sum / wsum.max(1e-5));
        }
    }
    out
}

/// Fixed-direction anisotropic convolution (artist-chosen angle via `rotation_deg`).
pub(super) fn directional_blur(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let ang = p.rotation_deg.to_radians();
    let (s, c) = ang.sin_cos();
    anisotropic_blur(input, p.radius.max(1), c, s)
}

pub(super) fn squeeze(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    // Nonlinear height-range remapping toward midtones (Phase 10 Squeeze).
    let mut out = input.clone();
    let w = input.metrics.width;
    let h = input.metrics.height;
    let mut min_h = f32::INFINITY;
    let mut max_h = f32::NEG_INFINITY;
    for j in 0..h {
        for i in 0..w {
            let v = input.get(i, j);
            min_h = min_h.min(v);
            max_h = max_h.max(v);
        }
    }
    let range = (max_h - min_h).max(1e-5);
    // amount ∈ [0,1]: 0 = identity, 1 = strong midtone compression.
    let t = if p.amount > 1.0 {
        (p.amount / (p.amount + 8.0)).clamp(0.0, 1.0)
    } else {
        p.amount.clamp(0.0, 1.0)
    };
    let gamma = 1.0 + t * 2.5;
    for j in 0..h {
        for i in 0..w {
            let h0 = input.get(i, j);
            let n = ((h0 - min_h) / range).clamp(0.0, 1.0);
            let centered = (n - 0.5) * 2.0;
            let squeezed = 0.5 + 0.5 * centered.signum() * centered.abs().powf(gamma);
            let h1 = min_h + squeezed * range;
            out.set(i, j, h0 * (1.0 - t) + h1 * t);
        }
    }
    out
}

pub(super) fn swirl(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let angle_scale = p.amount.max(0.1);
    let freq = p.frequency.max(1e-5);
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let u = (i as f32 + 0.5) / input.metrics.width as f32;
            let v = (j as f32 + 0.5) / input.metrics.height as f32;
            let cu = u - 0.5;
            let cv = v - 0.5;
            let r = (cu * cu + cv * cv).sqrt();
            let x = input.metrics.world_x(i);
            let z = input.metrics.world_z(j);
            let n = noise::sample_noise(FractalNoiseType::Perlin, x * freq, z * freq, p.seed);
            let ang = (1.0 - r) * angle_scale * (0.5 + n);
            let (sa, ca) = ang.sin_cos();
            let nu = (0.5 + cu * ca - cv * sa).clamp(0.0, 1.0);
            let nv = (0.5 + cu * sa + cv * ca).clamp(0.0, 1.0);
            out.set(i, j, sample_nearest(input, nu, nv));
        }
    }
    out
}

pub(super) fn washed_off(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    // Preferentially suppress exposed / convex fine relief (flow + slope aware).
    let blurred = box_blur_height(input, p.radius.max(1));
    let flow = flow_norm(input, 4);
    let mut out = input.clone();
    let amt = p.amount;
    let w = input.metrics.width as usize;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let slope = slope_deg(input, i, j);
            let steep = ((slope - p.slope_min) / (90.0 - p.slope_min).max(1.0)).clamp(0.0, 1.0);
            let f = flow[j as usize * w + i as usize];
            // Convex / exposed: high slope, low accumulation.
            let exposed = (steep * (1.0 - f)).clamp(0.0, 1.0);
            let h0 = input.get(i, j);
            let hb = blurred.get(i, j);
            let smoothed = h0 * (1.0 - exposed * 0.7) + hb * exposed * 0.7;
            out.set(i, j, smoothed - amt * exposed * 0.4);
        }
    }
    out
}

pub(super) fn hexagons(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let cell = (1.0 / p.frequency.max(1e-5)).max(2.0);
    let amt = p.amount;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let x = input.metrics.world_x(i);
            let z = input.metrics.world_z(j);
            // Axial hex snap.
            let q = (x * (2.0 / 3.0)) / cell;
            let r = ((-x / 3.0) + (3.0f32.sqrt() / 3.0) * z) / cell;
            let qi = q.round();
            let ri = r.round();
            let cx = cell * (3.0 / 2.0) * qi;
            let cz = cell * (3.0f32.sqrt() / 2.0 * qi + 3.0f32.sqrt() * ri);
            let u = (cx / input.metrics.world_size_x).clamp(0.0, 1.0);
            let v = (cz / input.metrics.world_size_z).clamp(0.0, 1.0);
            let cell_h = sample_nearest(input, u, v);
            let h0 = input.get(i, j);
            let t = (amt / (amt + 4.0)).clamp(0.2, 0.9);
            out.set(i, j, h0 * (1.0 - t) + cell_h * t);
        }
    }
    out
}

pub(super) fn scatter_detail(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    // Deterministic sparse impulses (hash-gated), not continuous noise.
    let mut out = input.clone();
    let freq = p.effective_frequency();
    let amt = p.amount;
    let cell = (1.0 / freq).max(2.0);
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let x = input.metrics.world_x(i);
            let z = input.metrics.world_z(j);
            let (wx, wz) = domain_warp_xz(x, z, p);
            let cx = (wx / cell).floor() as i32;
            let cz = (wz / cell).floor() as i32;
            let h = crate::noise::hash2(cx, cz, crate::noise::canonical_seed32(p.seed));
            // Sparse: only ~12% of cells fire an impulse.
            if (h % 1000) > 120 {
                continue;
            }
            let jx = ((h >> 8) & 0xFF) as f32 / 255.0;
            let jz = ((h >> 16) & 0xFF) as f32 / 255.0;
            let ox = (cx as f32 + jx) * cell;
            let oz = (cz as f32 + jz) * cell;
            let dx = wx - ox;
            let dz = wz - oz;
            let sigma = cell * 0.28;
            let env = (-(dx * dx + dz * dz) / (2.0 * sigma * sigma)).exp();
            if env < 1e-3 {
                continue;
            }
            let amp = (((h >> 24) & 0xFF) as f32 / 255.0 * 2.0 - 1.0) * amt;
            out.set(i, j, input.get(i, j) + amp * env);
        }
    }
    out
}

pub(super) fn flatten_filter(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    // Weighted interpolation toward a reference elevation (sea_level) or local median/mean.
    let r = p.radius.max(1) as i32;
    let w = input.metrics.width as i32;
    let h = input.metrics.height as i32;
    let mut out = input.clone();
    let pull = if p.amount > 1.0 {
        (p.amount / (p.amount + 8.0)).clamp(0.15, 0.95)
    } else {
        p.amount.clamp(0.15, 0.95)
    };
    let use_absolute = p.sea_level.abs() > 1e-4 || p.sea_level != 0.0;
    // Prefer absolute target when sea_level was authored away from default 0 on Flatten.
    // For Flatten presets sea_level is 0 — use local reference. Artist can set sea_level as target.
    let absolute_target = p.sea_level;
    for j in 0..h {
        for i in 0..w {
            let mut vals = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
            let mut wsum = 0.0f32;
            let mut wmean = 0.0f32;
            for dj in -r..=r {
                for di in -r..=r {
                    let x = (i + di).clamp(0, w - 1) as u32;
                    let y = (j + dj).clamp(0, h - 1) as u32;
                    let d2 = (di * di + dj * dj) as f32;
                    let wt = (-d2 / ((r * r) as f32 * 0.5 + 1e-3)).exp();
                    let v = input.get(x, y);
                    vals.push(v);
                    wmean += v * wt;
                    wsum += wt;
                }
            }
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = vals[vals.len() / 2];
            let mean = wmean / wsum.max(1e-5);
            let local = mid * 0.55 + mean * 0.45;
            let target = if use_absolute && absolute_target.abs() > 1e-3 {
                absolute_target
            } else {
                local
            };
            let h0 = input.get(i as u32, j as u32);
            out.set(i as u32, j as u32, h0 * (1.0 - pull) + target * pull);
        }
    }
    out
}

pub(super) fn zero_edge(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    // World-space distance-to-edge falloff toward base elevation (field min or sea_level).
    let mut out = input.clone();
    let w = input.metrics.width;
    let h = input.metrics.height;
    let dx = input.metrics.dx().max(1e-5);
    let dz = input.metrics.dz().max(1e-5);
    let margin_m = if p.amount > 1.0 {
        p.amount
    } else {
        (p.amount.clamp(0.02, 0.45) * input.metrics.world_size_x.min(input.metrics.world_size_z))
            .max(dx)
    };
    let mut min_h = f32::INFINITY;
    for j in 0..h {
        for i in 0..w {
            min_h = min_h.min(input.get(i, j));
        }
    }
    let base = if p.sea_level.abs() > 1e-4 {
        p.sea_level
    } else {
        min_h
    };
    for j in 0..h {
        for i in 0..w {
            let dist_m = (i as f32 * dx)
                .min((w - 1 - i) as f32 * dx)
                .min(j as f32 * dz)
                .min((h - 1 - j) as f32 * dz);
            let t = smoothstep(0.0, margin_m, dist_m);
            let h0 = input.get(i, j);
            out.set(i, j, base * (1.0 - t) + h0 * t);
        }
    }
    out
}

pub(super) fn border_blend(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    // World-space signed distance to boundary → smoothstep toward target elevation.
    let mut out = input.clone();
    let w = input.metrics.width;
    let h = input.metrics.height;
    let dx = input.metrics.dx().max(1e-5);
    let dz = input.metrics.dz().max(1e-5);
    let margin_m = if p.amount > 1.0 {
        p.amount
    } else {
        (p.amount.clamp(0.02, 0.45) * input.metrics.world_size_x.min(input.metrics.world_size_z))
            .max(dx)
    };
    let mut sum = 0.0f32;
    let mut count = 0u32;
    let rim = p.radius.max(1);
    for j in 0..h {
        for i in 0..w {
            if i < rim || j < rim || i + rim >= w || j + rim >= h {
                sum += input.get(i, j);
                count += 1;
            }
        }
    }
    let border_avg = if count > 0 { sum / count as f32 } else { 0.0 };
    let target = if p.sea_level.abs() > 1e-3 {
        p.sea_level
    } else {
        border_avg
    };
    for j in 0..h {
        for i in 0..w {
            // Positive inside; zero on boundary (Chebyshev-ish world distance).
            let dist_m = (i as f32 * dx)
                .min((w - 1 - i) as f32 * dx)
                .min(j as f32 * dz)
                .min((h - 1 - j) as f32 * dz);
            let blend = 1.0 - smoothstep(0.0, margin_m, dist_m);
            let h0 = input.get(i, j);
            out.set(i, j, h0 * (1.0 - blend) + target * blend);
        }
    }
    out
}

/// Contrast-like height remapping (WC Curve approximation).
/// `amount` in \[0,1\]: mid = linear; higher = S-curve contrast; lower = flatten midtones.
pub(super) fn curve_remap(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let w = input.metrics.width;
    let h = input.metrics.height;
    let mut min_h = f32::INFINITY;
    let mut max_h = f32::NEG_INFINITY;
    for j in 0..h {
        for i in 0..w {
            let v = input.get(i, j);
            min_h = min_h.min(v);
            max_h = max_h.max(v);
        }
    }
    let range = (max_h - min_h).max(1e-5);
    // amount 0 → gamma ~2 (compress), 0.5 → linear, 1 → strong S-curve
    let a = p.amount.clamp(0.0, 1.0);
    let gamma = if a < 0.5 {
        1.0 + (0.5 - a) * 2.0 // 1..2
    } else {
        1.0 / (1.0 + (a - 0.5) * 1.5).max(0.35) // ~1..0.4
    };
    for j in 0..h {
        for i in 0..w {
            let n = ((input.get(i, j) - min_h) / range).clamp(0.0, 1.0);
            let shaped = if a >= 0.5 {
                // Smoothstep-ish S-curve toward extremes
                let t = n.powf(gamma);
                let s = t * t * (3.0 - 2.0 * t);
                n * (1.0 - (a - 0.5) * 2.0) + s * ((a - 0.5) * 2.0)
            } else {
                n.powf(gamma)
            };
            out.set(i, j, min_h + shaped.clamp(0.0, 1.0) * range);
        }
    }
    out
}

/// Hard height shelf / clamp (WC Cutoff approximation).
/// `sea_level` is normalized \[0,1\] cutoff within the field range;
/// `amount` softens the transition (0 = hard, 1 = softer ramp).
pub(super) fn cutoff(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let w = input.metrics.width;
    let h = input.metrics.height;
    let mut min_h = f32::INFINITY;
    let mut max_h = f32::NEG_INFINITY;
    for j in 0..h {
        for i in 0..w {
            let v = input.get(i, j);
            min_h = min_h.min(v);
            max_h = max_h.max(v);
        }
    }
    let range = (max_h - min_h).max(1e-5);
    let cut = min_h + p.sea_level.clamp(0.0, 1.0) * range;
    let soft = (p.amount.clamp(0.0, 1.0) * range * 0.15).max(1e-4);
    for j in 0..h {
        for i in 0..w {
            let h0 = input.get(i, j);
            let t = ((h0 - (cut - soft)) / (2.0 * soft)).clamp(0.0, 1.0);
            // Below cutoff → pull toward cut; above → keep (or soft blend near edge)
            let below = cut;
            out.set(i, j, below * (1.0 - t) + h0 * t);
        }
    }
    out
}

pub(super) fn angle_break(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    // Directional slope analysis: break / deposit along critical orientation.
    let mut out = input.clone();
    let freq = p.effective_frequency();
    let amt = p.amount;
    let step = (amt * 0.45).max(0.35);
    let ang = p.rotation_deg.to_radians();
    let (ws, wc) = ang.sin_cos();
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let slope = slope_deg(input, i, j);
            let g = slope_gate(slope, p);
            if g < 1e-4 {
                continue;
            }
            let (gx, gz, mag) = gradient(input, i, j);
            // Aspect alignment with critical wind / break direction.
            let aspect = if mag > 1e-6 {
                (gx * wc + gz * ws).abs() / mag.max(1e-6)
            } else {
                0.0
            };
            let aligned = aspect.clamp(0.0, 1.0);
            let x = input.metrics.world_x(i);
            let z = input.metrics.world_z(j);
            let (wx, wz) = domain_warp_xz(x, z, p);
            let n = sparse_gabor(wx, wz, freq, p.seed ^ 0xA18Eu64);
            let h0 = input.get(i, j);
            // Break (quantize down) on windward-aligned faces; mild deposit leeward.
            let broken = (h0 / step + n * 0.35).floor() * step;
            let deposit = n.max(0.0) * amt * 0.15 * (1.0 - aligned);
            let mix = (g * (0.35 + 0.65 * aligned)).clamp(0.0, 1.0);
            out.set(
                i,
                j,
                h0 * (1.0 - mix * 0.8) + broken * mix * 0.8 + deposit * g,
            );
        }
    }
    out
}

pub(super) fn wind_carve(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    // Prefer Sand/Aeolian subsystem for physical transport when iterations request it.
    let steps = p.iterations.max(1);
    if steps >= 2 {
        return wind_carve_aeolian(input, p, steps);
    }
    wind_carve_fast(input, p)
}

fn wind_carve_aeolian(input: &Heightfield, p: &EffectFilterParams, steps: u32) -> Heightfield {
    let sand_cover = (p.amount * 0.35).clamp(0.15, 8.0);
    let mut state = AeolianState::from_height_and_sand(input, sand_cover, None);
    let transport = AeolianTransportParams {
        wind_direction_deg: p.rotation_deg,
        wind_speed: (p.amount / 8.0).clamp(0.35, 2.0),
        transport_length: (p.radius.max(1) as f32 + 2.0).clamp(2.0, 16.0),
        repose_angle_deg: 32.0,
        slab_size: (p.amount * 0.04).clamp(0.05, 0.8),
        avalanche_iters: 4,
        abrasion: (1.0 - p.rock_hardness).clamp(0.0, 0.2) * 0.15,
        reptation: 0.12,
        wind_warp: 0.3,
        linearity: p.anisotropy.clamp(0.2, 1.0),
    };
    let budget = steps.min(24);
    state.evolve(&transport, budget);
    let evolved = state.sync_height();
    // Mix toward evolved surface by strength proxy (amount-normalised).
    let t = (p.amount / (p.amount + 6.0)).clamp(0.25, 1.0);
    let mut out = input.clone();
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let h0 = input.get(i, j);
            let h1 = evolved.get(i, j);
            out.set(i, j, h0 * (1.0 - t) + h1 * t);
        }
    }
    out
}

fn wind_carve_fast(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    // Interactive Werner-class approximation (single pass).
    let mut out = input.clone();
    let r = p.radius.max(1) as i32;
    let w = input.metrics.width as i32;
    let h = input.metrics.height as i32;
    let amt = p.amount;
    let ang = p.rotation_deg.to_radians();
    let (wind_z, wind_x) = ang.sin_cos(); // 0° → +X
    let freq = p.effective_frequency();
    for j in 0..h {
        for i in 0..w {
            let x = input.metrics.world_x(i as u32);
            let z = input.metrics.world_z(j as u32);
            let (wx, wz) = domain_warp_xz(x, z, p);
            let n = noise::sample_noise(
                FractalNoiseType::Perlin,
                wx * freq,
                wz * freq,
                p.seed ^ 0xA1ADu64,
            );
            let mut windward = 0.0f32;
            let mut leeward = 0.0f32;
            let mut ww = 0.0f32;
            let mut lw = 0.0f32;
            for k in 1..=r {
                let t = k as f32;
                let wi = (i as f32 - wind_x * t).round() as i32;
                let wj = (j as f32 - wind_z * t).round() as i32;
                let li = (i as f32 + wind_x * t).round() as i32;
                let lj = (j as f32 + wind_z * t).round() as i32;
                let wt = 1.0 - t / (r as f32 + 1.0);
                if wi >= 0 && wj >= 0 && wi < w && wj < h {
                    windward += input.get(wi as u32, wj as u32) * wt;
                    ww += wt;
                }
                if li >= 0 && lj >= 0 && li < w && lj < h {
                    leeward += input.get(li as u32, lj as u32) * wt;
                    lw += wt;
                }
            }
            let h0 = input.get(i as u32, j as u32);
            let wind_h = if ww > 1e-5 { windward / ww } else { h0 };
            let lee_h = if lw > 1e-5 { leeward / lw } else { h0 };
            let erode = ((wind_h - h0).max(0.0) * 0.55).min(amt);
            let deposit = ((h0 - lee_h).max(0.0) * 0.25).min(amt * 0.5);
            let delta = (-erode + deposit) * (0.45 + n.abs() * 0.55);
            out.set(i as u32, j as u32, h0 + delta);
        }
    }
    out
}

/// Anisotropic multi-sector Kuwahara (Kyprianidis-inspired).
pub(super) fn kuwahara(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut h = input.clone();
    let iters = p.iterations.max(1).min(3);
    let sectors = if p.radius >= 3 { 8 } else { 4 };
    for _ in 0..iters {
        h = kuwahara_anisotropic(&h, p.radius.max(1), sectors);
    }
    h
}

fn terrace_controls(p: &EffectFilterParams) -> super::geology::TerraceControls {
    super::geology::TerraceControls {
        height_m: p.terrace_height,
        levels: p.amount.round().clamp(2.0, 32.0) as u32,
        offset: p.terrace_offset,
        top_smoothness: p.top_smoothness,
        riser_sharpness: p.riser_sharpness,
        frequency: p.frequency,
        seed: p.seed,
        slope_min: p.slope_min,
        slope_max: p.slope_max,
    }
}

pub(super) fn terrace_simple(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut c = terrace_controls(p);
    c.riser_sharpness = if p.riser_sharpness > 0.0 {
        p.riser_sharpness
    } else {
        0.75
    };
    super::geology::terrace_simple(input, &c)
}

pub(super) fn terrace_irregular(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut c = terrace_controls(p);
    c.seed = p.seed ^ 0x71CEu64;
    if p.riser_sharpness <= 0.0 {
        c.riser_sharpness = 0.55;
    }
    super::geology::terrace_irregular(input, &c)
}

pub(super) fn terrace_steep(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut c = terrace_controls(p);
    c.levels = p.amount.round().clamp(2.0, 24.0) as u32;
    if p.riser_sharpness <= 0.0 {
        c.riser_sharpness = 0.95;
    }
    // Preferential slope gate when artist left defaults wide.
    if p.slope_min <= 0.0 && p.slope_max >= 89.0 {
        c.slope_min = 10.0;
        c.slope_max = 60.0;
    }
    super::geology::terrace_steep(input, &c)
}

/// Phase 9 geological strata: band displace + soft-bed exposure on steep faces.
pub(super) fn strata_filter(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    use super::filter_kernels::slope_deg as slope_at;
    use super::geology::{
        expose_strata_height, strata_band_displace, strata_fields_with, StrataFieldParams,
    };
    use crate::layer::BedGeometry;

    let freq = p.effective_frequency();
    let banded = strata_band_displace(input, freq, p.amount, p.seed);
    let geom = BedGeometry::Folded {
        amplitude_m: p.amount.max(0.0) * 0.4,
        wavelength_m: (1.0 / freq).clamp(8.0, 400.0),
        seed: p.seed,
    };
    let sp = StrataFieldParams {
        frequency: freq,
        seed: p.seed,
        hardness_contrast: p.rock_hardness.clamp(0.15, 0.95),
        geometry: geom,
    };
    let (hardness, _) = strata_fields_with(&banded, &sp);
    // Cliff mask from slope gate.
    let mut cliff = crate::mask::MaskField::zeros(input.metrics);
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let s = slope_at(&banded, i, j);
            cliff.set(i, j, slope_gate(s, p));
        }
    }
    let exposed = expose_strata_height(&banded, &hardness, p.amount * 0.45, &cliff);
    // Optional light coastal/shore-adjacent erosion hook when beach_width is authored.
    if p.beach_width > 1.0 && p.iterations > 1 {
        let muted = box_blur_height(&exposed, 1);
        let t = 0.15f32;
        let mut out = exposed.clone();
        for j in 0..input.metrics.height {
            for i in 0..input.metrics.width {
                let h0 = exposed.get(i, j);
                out.set(i, j, h0 * (1.0 - t) + muted.get(i, j) * t);
            }
        }
        out
    } else {
        exposed
    }
}

/// Radial signed-distance crater with raised rim, ejecta distortion, optional soft erosion.
pub(super) fn crater_filter(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let cx = input.metrics.world_size_x * 0.5;
    let cz = input.metrics.world_size_z * 0.5;
    let rad_m = (p.crater_radius.clamp(0.02, 0.5)
        * input.metrics.world_size_x.min(input.metrics.world_size_z))
    .max(input.metrics.dx() * 2.0);
    let depth = p.amount.max(0.0);
    let rim_h = depth * 0.18;
    let freq = p.effective_frequency();
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let x = input.metrics.world_x(i);
            let z = input.metrics.world_z(j);
            let (wx, wz) = domain_warp_xz(x, z, p);
            // Ejecta / rim noise distortion of the radial field.
            let n = noise::sample_noise(
                FractalNoiseType::Perlin,
                wx * freq * 2.5,
                wz * freq * 2.5,
                p.seed ^ 0xC2A7,
            );
            let dist = ((x - cx).hypot(z - cz) + n * rad_m * 0.08).max(0.0);
            let d = dist / rad_m;
            let mut delta = 0.0f32;
            if d < 1.0 {
                // Smooth bowl (1 - d^2)^2 style.
                let t = 1.0 - d * d;
                delta = -depth * t * t;
            } else if d < 1.25 {
                // Raised rim.
                let u = (d - 1.0) / 0.25;
                let rim = (1.0 - u) * (1.0 - u) * 4.0 * u; // peak near rim
                delta = rim_h * rim;
            } else if d < 1.7 {
                // Thin ejecta blanket.
                let u = (d - 1.25) / 0.45;
                delta = rim_h * 0.25 * (1.0 - u) * (0.5 + 0.5 * n);
            }
            out.set(i, j, input.get(i, j) + delta);
        }
    }
    // Optional mild erosion afterward (iterations > 1).
    if p.iterations > 1 {
        let soft = box_blur_height(&out, 1);
        let t = 0.2f32;
        for j in 0..input.metrics.height {
            for i in 0..input.metrics.width {
                let h0 = out.get(i, j);
                out.set(i, j, h0 * (1.0 - t) + soft.get(i, j) * t);
            }
        }
    }
    out
}

/// Add constant height, or set absolute height when `sea_level` ≥ 0.5.
pub(super) fn add_set(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let set_mode = p.sea_level >= 0.5;
    let offset = p.amount;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let h0 = input.get(i, j);
            out.set(i, j, if set_mode { offset } else { h0 + offset });
        }
    }
    out
}

pub(super) fn design_voronoi(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let freq = p.effective_frequency();
    let amt = p.amount;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let x = input.metrics.world_x(i);
            let z = input.metrics.world_z(j);
            let (wx, wz) = domain_warp_xz(x, z, p);
            let w = noise::worley2(
                wx * freq,
                wz * freq,
                p.seed,
                crate::noise::WorleyMetric::Euclidean,
            );
            let feature = match p.voronoi_feature {
                WorleyFeature::F1 => 1.0 - w.f1.clamp(0.0, 1.0),
                WorleyFeature::F2 => 1.0 - w.f2.clamp(0.0, 1.2) / 1.2,
                WorleyFeature::F2MinusF1 => (w.f2 - w.f1).clamp(0.0, 1.0),
            };
            out.set(i, j, input.get(i, j) + feature * amt);
        }
    }
    out
}

pub(super) fn noise_billow(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let np = noise_params(p);
    let amt = p.amount;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let (wx, wz) = domain_warp_xz(input.metrics.world_x(i), input.metrics.world_z(j), p);
            let n = billow_mf(wx, wz, &np);
            out.set(i, j, input.get(i, j) + n * amt);
        }
    }
    out
}

pub(super) fn noise_gabor(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    // Sparse convolution Gabor noise (Lagae et al.) — dedicated kernel, not Perlin rename.
    let mut out = input.clone();
    let amt = p.amount;
    let freq = p.effective_frequency();
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let (wx, wz) = domain_warp_xz(input.metrics.world_x(i), input.metrics.world_z(j), p);
            let n = sparse_gabor_fbm(
                wx,
                wz,
                freq,
                p.seed,
                p.octaves.max(1),
                p.lacunarity,
                p.persistence,
                p.rotation_deg,
                p.anisotropy,
            );
            out.set(i, j, input.get(i, j) + n * amt);
        }
    }
    out
}

pub(super) fn noise_perlin(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let np = noise_params(p);
    let amt = p.amount;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let (wx, wz) = domain_warp_xz(input.metrics.world_x(i), input.metrics.world_z(j), p);
            let n = noise::fbm(FractalNoiseType::Perlin, wx, wz, &np);
            out.set(i, j, input.get(i, j) + n * amt);
        }
    }
    out
}

pub(super) fn noise_phasor(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    // Procedural phasor noise (Tricard et al. SIGGRAPH 2019).
    let mut out = input.clone();
    let amt = p.amount;
    let freq = p.effective_frequency();
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let (wx, wz) = domain_warp_xz(input.metrics.world_x(i), input.metrics.world_z(j), p);
            let n = phasor_noise_fbm(
                wx,
                wz,
                freq,
                p.seed,
                p.octaves.max(1),
                p.lacunarity,
                p.persistence,
                p.rotation_deg,
                p.anisotropy,
            );
            out.set(i, j, input.get(i, j) + n * amt);
        }
    }
    out
}

pub(super) fn noise_ridged_filter(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let np = noise_params(p);
    let amt = p.amount;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let (wx, wz) = domain_warp_xz(input.metrics.world_x(i), input.metrics.world_z(j), p);
            let n = noise::ridged_mf(FractalNoiseType::Perlin, wx, wz, &np);
            out.set(i, j, input.get(i, j) + (n * 2.0 - 1.0) * amt);
        }
    }
    out
}

pub(super) fn noise_simplex(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let np = noise_params(p);
    let amt = p.amount;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let (wx, wz) = domain_warp_xz(input.metrics.world_x(i), input.metrics.world_z(j), p);
            let n = noise::fbm(FractalNoiseType::OpenSimplex, wx, wz, &np);
            out.set(i, j, input.get(i, j) + n * amt);
        }
    }
    out
}

pub(super) fn noise_value(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let np = noise_params(p);
    let amt = p.amount;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let (wx, wz) = domain_warp_xz(input.metrics.world_x(i), input.metrics.world_z(j), p);
            let n = noise::fbm(FractalNoiseType::Value, wx, wz, &np);
            out.set(i, j, input.get(i, j) + n * amt);
        }
    }
    out
}

pub(super) fn noise_voronoi(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    // Worley cellular noise (Worley 1996) with selectable F1 / F2 / F2−F1.
    let mut out = input.clone();
    let amt = p.amount;
    let wp = WorleyParams {
        base: noise_params(p),
        feature: p.voronoi_feature,
        distance_metric: crate::noise::WorleyMetric::Euclidean,
    };
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let (wx, wz) = domain_warp_xz(input.metrics.world_x(i), input.metrics.world_z(j), p);
            let n = noise::sample_worley(wx, wz, &wp);
            let centered = (n * 2.0 - 1.0).clamp(-1.0, 1.0);
            out.set(i, j, input.get(i, j) + centered * amt);
        }
    }
    out
}

pub(super) fn noise_wave(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let freq = p.effective_frequency();
    let amt = p.amount;
    let aniso = p.anisotropy.max(0.05);
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let (wx, wz) = domain_warp_xz(input.metrics.world_x(i), input.metrics.world_z(j), p);
            let n = ((wx * freq / aniso).sin() * (wz * freq * 0.7 * aniso.sqrt()).cos())
                .clamp(-1.0, 1.0);
            out.set(i, j, input.get(i, j) + n * amt);
        }
    }
    out
}

pub(super) fn noise_white(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut out = input.clone();
    let amt = p.amount;
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let n = white_noise(i, j, p.seed);
            out.set(i, j, input.get(i, j) + n * amt);
        }
    }
    out
}
