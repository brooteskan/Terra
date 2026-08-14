//! Shared arid / rock landform primitives for EffectFilters.
//!
//! Artist-facing filters (Canyon, Cliffs, Chipped, Rocky*) compose these
//! research-backed operators rather than shipping independent mega-kernels.
//!
//! Building blocks:
//! - cliff detection (multi-scale relief + slope + profile curvature)
//! - geological strata (warped elevation → hardness / erodibility)
//! - fracturing (multi-scale anisotropic breakup field)
//! - ridge enhancement (ridge-aware sharpening)
//! - channel incision (hydrology → discharge → cross-sections)
//! - differential erosion (hard ridges persist, soft rock strips)
//! - talus deposition (thermal apron under steep faces)
//!
//! Citations: Tarboton 1997 (D∞), Horton–Strahler, stream-power incision,
//! Zevenbergen–Thorne curvatures, Musgrave thermal / angle of repose,
//! Cordonnier / Génevaux drainage-driven landforms.

use crate::geomorph::{
    accumulate_drainage_area, build_flow_graph, extract_streams, handle_depressions,
    profile_curvature, ridge_valley_likelihood, slope_magnitude, DepressionMode, FlowModel,
    Precipitation, StreamExtractParams,
};
use crate::heightfield::Heightfield;
use crate::layer::{EffectFilterParams, StreamPowerParams};
use crate::mask::MaskField;
use crate::noise;
use crate::noise::{FractalNoiseType, WorleyMetric};

use super::filter_kernels::{
    dilate, flow_accumulation_dinf, gradient, normalize_field, slope_deg, sparse_gabor,
    thermal_talus,
};

// ── Parameter mapping from EffectFilterParams ───────────────────────────────

/// Semantic controls for canyon / arid filters mapped onto shared EffectFilterParams.
#[derive(Debug, Clone, Copy)]
pub struct CanyonControls {
    pub incision: f32,
    pub canyon_width: f32,
    pub wall_steepness: f32,
    pub tributary_influence: f32,
    pub valley_floor_width: f32,
    pub rock_hardness: f32,
    pub age: u32,
    pub talus: f32,
    pub seed: u64,
}

impl CanyonControls {
    pub fn from_params(p: &EffectFilterParams) -> Self {
        Self {
            incision: p.amount.max(0.0),
            canyon_width: p.radius.max(1) as f32 * (1.0 + p.beach_width.max(0.0) * 0.02),
            wall_steepness: p.wall_steepness.clamp(0.0, 1.0),
            tributary_influence: p.flow_threshold.clamp(0.0, 1.0),
            valley_floor_width: p.valley_floor.clamp(0.0, 1.0),
            rock_hardness: p.rock_hardness.clamp(0.0, 1.0),
            age: p.iterations.max(1),
            talus: p.talus_mix.clamp(0.0, 1.0),
            seed: p.seed,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CliffControls {
    pub detection_radius: u32,
    pub relief_threshold: f32,
    pub cliff_strength: f32,
    pub transition: f32,
    pub talus: f32,
    pub rock_hardness: f32,
    pub slope_min: f32,
    pub slope_max: f32,
    pub seed: u64,
}

impl CliffControls {
    pub fn from_params(p: &EffectFilterParams) -> Self {
        Self {
            detection_radius: p.radius.max(1),
            relief_threshold: p.flow_threshold.clamp(0.02, 1.0),
            cliff_strength: p.amount.max(0.0),
            transition: (1.0 - p.wall_steepness.clamp(0.0, 1.0)).max(0.05),
            talus: p.talus_mix.clamp(0.0, 1.0),
            rock_hardness: p.rock_hardness.clamp(0.0, 1.0),
            slope_min: p.slope_min,
            slope_max: p.slope_max,
            seed: p.seed,
        }
    }
}

// ── 1. Cliff detection ──────────────────────────────────────────────────────

/// Multi-scale cliff candidate mask ∈ \[0,1\].
///
/// Combines local elevation range, slope, and profile curvature so cliffs are
/// detected from relief structure rather than a single height-delta threshold.
pub fn detect_cliffs(
    hf: &Heightfield,
    radius_texels: u32,
    relief_threshold: f32,
    slope_min_deg: f32,
) -> MaskField {
    let m = hf.metrics;
    let r = radius_texels.max(1) as i32;
    let w = m.width as i32;
    let h = m.height as i32;
    let thr = relief_threshold.max(1e-3);
    let mut out = MaskField::zeros(m);

    // Profile curvature at primary scale (world metres ≈ radius * dx).
    let radius_m = r as f32 * m.dx().max(1e-3);
    let profile = profile_curvature(hf, radius_m);
    let slope = slope_magnitude(hf, radius_m);

    let mut max_score = 1e-6f32;
    let mut scores = vec![0.0f32; (m.width * m.height) as usize];

    for j in 0..h {
        for i in 0..w {
            let z0 = hf.get(i as u32, j as u32);
            let mut z_min = z0;
            let mut z_max = z0;
            for dj in -r..=r {
                for di in -r..=r {
                    if di * di + dj * dj > r * r {
                        continue;
                    }
                    let x = (i + di).clamp(0, w - 1) as u32;
                    let y = (j + dj).clamp(0, h - 1) as u32;
                    let z = hf.get(x, y);
                    z_min = z_min.min(z);
                    z_max = z_max.max(z);
                }
            }
            let relief = (z_max - z_min).max(0.0);
            let relief_n = (relief / (thr * m.dx().max(1.0) * r as f32 * 4.0)).clamp(0.0, 1.0);
            // slope_magnitude is already degrees/90 in [0,1].
            let s_deg = slope.get(i as u32, j as u32) * 90.0;
            let slope_n =
                ((s_deg - slope_min_deg) / (90.0 - slope_min_deg).max(1.0)).clamp(0.0, 1.0);
            // Profile curvature is mapped to [0,1] with 0.5 = flat.
            let curv_n = ((profile.get(i as u32, j as u32) - 0.5).abs() * 2.0).clamp(0.0, 1.0);
            let score = relief_n * 0.45 + slope_n * 0.40 + curv_n * 0.15;
            let idx = (j * w + i) as usize;
            scores[idx] = score;
            max_score = max_score.max(score);
        }
    }

    for j in 0..m.height {
        for i in 0..m.width {
            let s = scores[(j * m.width + i) as usize] / max_score;
            // Soft threshold so transitions stay controllable.
            let gated = ((s - 0.25) / 0.55).clamp(0.0, 1.0);
            out.set(i, j, gated * gated * (3.0 - 2.0 * gated));
        }
    }
    out
}

// ── 2. Geological strata ────────────────────────────────────────────────────

/// Synthetic stratigraphy: hardness from warped elevation banding.
///
/// Delegates to [`crate::generators::geology`] so Materials erosion and rock
/// filters share one lithology model.
pub fn strata_fields(
    hf: &Heightfield,
    frequency: f32,
    seed: u64,
    hardness_contrast: f32,
) -> (MaskField, MaskField) {
    super::geology::strata_fields(hf, frequency, seed, hardness_contrast)
}

/// Expose strata by preferential erosion of soft beds on steep / cliff faces.
pub fn expose_strata(
    hf: &Heightfield,
    hardness: &MaskField,
    amount: f32,
    cliff_mask: &MaskField,
) -> Heightfield {
    super::geology::expose_strata_height(hf, hardness, amount, cliff_mask)
}

// ── 3. Fracturing ───────────────────────────────────────────────────────────

/// Multi-scale fracture / chip field ∈ approximately \[-1, 1\].
pub fn fracture_sample(x: f32, z: f32, frequency: f32, seed: u64, scales: u32) -> f32 {
    let freq = frequency.max(1e-5);
    let mut sum = 0.0f32;
    let mut amp = 1.0f32;
    let mut wsum = 0.0f32;
    let n = scales.max(1).min(4);
    for s in 0..n {
        let f = freq * (1.7f32).powi(s as i32);
        let g = sparse_gabor(x, z, f, seed ^ ((s as u64 + 1) * 0x9E37));
        // Cellular / Worley cracks for anisotropic breakup.
        let w = noise::worley2(
            x * f,
            z * f,
            seed.wrapping_add(s as u64 * 17),
            WorleyMetric::Euclidean,
        );
        let cell = ((w.f2 - w.f1) * 2.0 - 0.5).clamp(-1.0, 1.0);
        sum += (g * 0.55 + cell * 0.45) * amp;
        wsum += amp;
        amp *= 0.55;
    }
    if wsum > 1e-5 {
        sum / wsum
    } else {
        0.0
    }
}

/// Build a full-domain fracture field (cached per call).
pub fn fracture_field(hf: &Heightfield, frequency: f32, seed: u64, scales: u32) -> Vec<f32> {
    let m = hf.metrics;
    let n = (m.width * m.height) as usize;
    let mut out = vec![0.0f32; n];
    for j in 0..m.height {
        for i in 0..m.width {
            let x = m.world_x(i);
            let z = m.world_z(j);
            out[(j * m.width + i) as usize] = fracture_sample(x, z, frequency, seed, scales);
        }
    }
    out
}

// ── 4. Ridge enhancement ────────────────────────────────────────────────────

/// Ridge-aware anisotropic sharpening on convex divides.
pub fn enhance_ridges(
    hf: &Heightfield,
    amount: f32,
    slope_min: f32,
    slope_max: f32,
    frequency: f32,
    seed: u64,
    wide: bool,
) -> Heightfield {
    let m = hf.metrics;
    let (ridge, _valley) = ridge_valley_likelihood(hf, if wide { 48.0 } else { 12.0 });
    let freq = frequency.max(1e-5) * if wide { 0.35 } else { 1.0 };
    let amt = amount * if wide { 0.65 } else { 1.0 };
    let mut out = hf.clone();

    let np = crate::noise::NoiseParams {
        seed,
        frequency: freq,
        amplitude: 1.0,
        octaves: if wide { 3 } else { 4 },
        lacunarity: 2.0,
        persistence: 0.5,
        ..crate::noise::NoiseParams::default()
    };

    for j in 0..m.height {
        for i in 0..m.width {
            let slope = slope_deg(hf, i, j);
            let lo = slope_min.min(slope_max);
            let hi = slope_min.max(slope_max);
            let g = if hi <= lo + 1e-3 {
                1.0
            } else {
                ((slope - lo) / (hi - lo)).clamp(0.0, 1.0)
            };
            if g < 1e-4 {
                continue;
            }
            let r = ridge.get(i, j);
            let (gx, gz, mag) = gradient(hf, i, j);
            let x = m.world_x(i);
            let z = m.world_z(j);
            let (ax, az) = if mag < 1e-6 {
                (x, z)
            } else {
                (x * gx - z * gz, x * gz + z * gx)
            };
            let n = noise::ridged_mf(FractalNoiseType::Perlin, ax, az, &np);
            let face = g * (0.35 + r * 0.65);
            let detail = (n * 2.0 - 1.0) * amt * face;
            out.set(i, j, hf.get(i, j) + detail);
        }
    }
    out
}

// ── 5. Channel incision (canyon core) ────────────────────────────────────────

/// Hydrology-driven canyon incision with distance-based cross-sections.
///
/// Pipeline: depressions → D8 routing → accumulation → Strahler channels →
/// discharge-scaled incision → wall profile → lithology-limited wall retreat.
pub fn incise_channels(hf: &Heightfield, c: &CanyonControls) -> Heightfield {
    let m = hf.metrics;
    let w = m.width as usize;
    let h = m.height as usize;

    // 1–3. Hydrology + high-order channels + accumulation.
    let dep = handle_depressions(hf, DepressionMode::Fill);
    let graph = build_flow_graph(&dep.height, FlowModel::D8);
    let area = accumulate_drainage_area(&graph, &Precipitation::uniform(1.0));
    let area_norm = normalize_field(&area);

    // Tributary influence: lower threshold → more tributaries participate.
    // Scale with terrain size so small preview grids still form channels.
    let cell_count = (w * h) as f32;
    let base_thr = (cell_count.sqrt() * 0.35).max(2.0);
    let acc_thr = (base_thr * (1.15 - c.tributary_influence * 0.85)).max(2.0);
    let streams = extract_streams(
        &graph,
        &area,
        m,
        &StreamExtractParams {
            accumulation_threshold: acc_thr,
            width_scale: c.canyon_width.max(1.0) * 0.35,
        },
    );

    let has_channels = streams.channel_mask.data().iter().any(|&v| v > 0.5);

    // Fallback: lightweight D∞ accumulation when Strahler extract is empty
    // (very flat / tiny grids). Still channel-driven — not Voronoi/noise.
    let dinf_norm = if has_channels {
        None
    } else {
        Some(normalize_field(&flow_accumulation_dinf(hf, 8)))
    };

    let dist = &streams.distance_to_channel;
    let order = &streams.order_normalised;
    let hierarchy = &streams.hierarchy;

    // Lithology field (low-frequency) modulates wall retreat / incision.
    let (hardness, _) = strata_fields(hf, 0.008, c.seed ^ 0xCA70, c.rock_hardness);

    let mut out = hf.clone();
    let width_cells = c.canyon_width.max(1.0);
    let floor_w = width_cells * (0.15 + c.valley_floor_width * 0.55);
    let wall_power = 1.1 + c.wall_steepness * 2.2; // higher → steeper walls
    let incision = c.incision.max(0.0);
    // `distance_to_channel` is normalised by grid diagonal — convert back to cells.
    let diag = ((w * w + h * h) as f32).sqrt().max(1.0);
    let flow_gate = (0.08 + (1.0 - c.tributary_influence) * 0.35).clamp(0.05, 0.55);

    for j in 0..h {
        for i in 0..w {
            let iu = i as u32;
            let ju = j as u32;
            let a = if let Some(ref dn) = dinf_norm {
                dn[j * w + i]
            } else {
                area_norm[j * w + i]
            };
            let o = if has_channels {
                order.get(iu, ju)
            } else {
                // Synthetic order from normalised accumulation.
                ((a - flow_gate) / (1.0 - flow_gate).max(1e-3)).clamp(0.0, 1.0)
            };
            let hier = if has_channels {
                hierarchy.get(iu, ju)
            } else {
                o
            };

            let d = if has_channels {
                dist.get(iu, ju) * diag
            } else {
                // Approximate distance from channel-ness: high flow → d≈0.
                (1.0 - o) * width_cells * 1.25
            };

            // Only near channels of sufficient order / discharge.
            let channel_w = width_cells * (0.35 + o * 0.9 + a * 0.4);
            if a < flow_gate * 0.5 && o < 0.05 {
                continue;
            }
            if d > channel_w * 1.35 {
                continue;
            }

            // 4. Incision depth ∝ discharge^m * hierarchy (high-order trunks deeper).
            let discharge = a.powf(0.55) * (0.35 + hier * 0.65);
            let soft = 1.0 - hardness.get(iu, ju) * c.rock_hardness;
            let depth = incision * discharge * (0.55 + soft * 0.45) * (0.6 + o * 0.4);

            // 5–6. Cross-section: flat floor near channel, rising walls.
            let t = if d <= floor_w {
                1.0
            } else {
                let u = ((d - floor_w) / (channel_w - floor_w).max(1e-3)).clamp(0.0, 1.0);
                (1.0 - u).powf(wall_power)
            };

            // 7. Wall retreat: softer rock widens the upper canyon slightly.
            let retreat = soft * (1.0 - c.rock_hardness) * 0.35;
            let profile = (t + retreat * (1.0 - t) * 0.5).clamp(0.0, 1.0);

            let h0 = hf.get(iu, ju);
            out.set(iu, ju, h0 - depth * profile);
        }
    }

    // Age: light valley smoothing + optional SPE polish.
    let age_iters = c.age.min(6);
    if age_iters > 1 {
        // Smooth valley floors (concavity) lightly with age.
        let flow = normalize_field(&flow_accumulation_dinf(&out, 4));
        let mut aged = out.clone();
        for j in 0..h as u32 {
            for i in 0..w as u32 {
                let f = flow[j as usize * w + i as usize];
                if f < 0.08 {
                    continue;
                }
                let mut sum = out.get(i, j);
                let mut n = 1.0f32;
                for (di, dj) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
                        continue;
                    }
                    sum += out.get(ni as u32, nj as u32);
                    n += 1.0;
                }
                let avg = sum / n;
                let t = ((f - 0.08) / 0.4).clamp(0.0, 1.0) * (age_iters as f32 * 0.08);
                aged.set(i, j, out.get(i, j) * (1.0 - t) + avg * t);
            }
        }
        out = aged;
    }

    // Optional stream-power evolution when age is high.
    if c.age >= 4 && incision > 0.5 {
        let spe = StreamPowerParams {
            iterations: (c.age / 2).clamp(1, 4),
            k: 0.0008 * (1.0 - c.rock_hardness * 0.6),
            m: 0.5,
            n: 1.0,
            uplift_rate: 0.0,
            base_level: 0.0,
            dt: 0.35,
            use_dinfinity: false,
            refill_each_iter: false,
            drainage_reuse_stride: 2,
            hardness: c.rock_hardness.max(0.15),
            dendritic_seed: 0.0,
            ..StreamPowerParams::default()
        };
        let k_mask = MaskField::filled(m, spe.hardness);
        let result = crate::hydro::stream_power_erode(&out, &spe, &k_mask);
        // Blend SPE lightly so canyon cross-sections remain the hero.
        let blend = ((c.age as f32 - 3.0) / 6.0).clamp(0.0, 0.45);
        for j in 0..h as u32 {
            for i in 0..w as u32 {
                let a = out.get(i, j);
                let b = result.height.get(i, j);
                out.set(i, j, a * (1.0 - blend) + b * blend);
            }
        }
    }

    out
}

// ── 6. Differential erosion ─────────────────────────────────────────────────

/// Preferentially strip soft material while preserving hard ridges / outcrops.
pub fn differential_erode(
    hf: &Heightfield,
    hardness: &MaskField,
    amount: f32,
    iterations: u32,
    protect_ridges: bool,
) -> Heightfield {
    let m = hf.metrics;
    let (ridge, _) = if protect_ridges {
        ridge_valley_likelihood(hf, 16.0)
    } else {
        (MaskField::zeros(m), MaskField::zeros(m))
    };
    let mut out = hf.clone();
    let iters = iterations.max(1).min(8);
    let amt = amount / iters as f32;

    for _ in 0..iters {
        let mut next = out.clone();
        for j in 0..m.height {
            for i in 0..m.width {
                let soft = 1.0 - hardness.get(i, j);
                let protect = if protect_ridges { ridge.get(i, j) } else { 0.0 };
                let strip = soft * (1.0 - protect * 0.85) * amt;
                if strip < 1e-5 {
                    continue;
                }
                // Local mean of softer neighbors pulls material down (exposes hard).
                let mut sum = 0.0f32;
                let mut n = 0.0f32;
                for dj in -1i32..=1 {
                    for di in -1i32..=1 {
                        let ni = i as i32 + di;
                        let nj = j as i32 + dj;
                        if ni < 0 || nj < 0 || ni >= m.width as i32 || nj >= m.height as i32 {
                            continue;
                        }
                        sum += out.get(ni as u32, nj as u32);
                        n += 1.0;
                    }
                }
                let mean = sum / n.max(1.0);
                let h0 = out.get(i, j);
                let target = h0.min(mean);
                next.set(i, j, h0 + (target - h0) * strip.clamp(0.0, 1.0));
            }
        }
        out = next;
    }
    out
}

/// Build a hardness field from rock hardness + low-frequency lithology noise.
pub fn lithology_hardness(
    hf: &Heightfield,
    rock_hardness: f32,
    frequency: f32,
    seed: u64,
) -> MaskField {
    let m = hf.metrics;
    let base = rock_hardness.clamp(0.05, 0.98);
    let freq = frequency.max(1e-5);
    let mut out = MaskField::zeros(m);
    for j in 0..m.height {
        for i in 0..m.width {
            let x = m.world_x(i);
            let z = m.world_z(j);
            let n = noise::sample_noise(FractalNoiseType::Perlin, x * freq, z * freq, seed);
            // Hard cores with softer surrounds.
            let local = base + (n - 0.5) * (1.0 - base) * 0.55;
            out.set(i, j, local.clamp(0.05, 0.98));
        }
    }
    out
}

// ── 7. Talus deposition ─────────────────────────────────────────────────────

/// Deposit talus / scree below steep faces; hardness optional.
pub fn deposit_talus(
    hf: &Heightfield,
    amount: f32,
    repose_deg: f32,
    iterations: u32,
    hardness: Option<&MaskField>,
) -> Heightfield {
    if amount < 1e-4 {
        return hf.clone();
    }
    let iters = iterations.max(1).min(8);
    let repose = repose_deg.clamp(18.0, 55.0);
    if let Some(k) = hardness {
        crate::analyze::talus_apron(hf, repose, amount.max(0.5), 0.85, iters, Some(k)).height
    } else {
        thermal_talus(hf, repose, amount.max(0.5), iters)
    }
}

// ── Composed artist filters ─────────────────────────────────────────────────

/// Nonlinear slope sharpening on cliff candidates with preserved transitions + talus.
pub fn apply_cliffs(hf: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let c = CliffControls::from_params(p);
    let cliffs = detect_cliffs(hf, c.detection_radius, c.relief_threshold, c.slope_min);
    let hardness = lithology_hardness(hf, c.rock_hardness, p.frequency.max(0.01), c.seed);

    let r = c.detection_radius.max(1) as i32;
    let w = hf.metrics.width as i32;
    let h = hf.metrics.height as i32;
    let amt = c.cliff_strength;
    let mut out = hf.clone();

    for j in 0..h {
        for i in 0..w {
            let mask = cliffs.get(i as u32, j as u32);
            if mask < 1e-4 {
                continue;
            }
            let slope = slope_deg(hf, i as u32, j as u32);
            let lo = c.slope_min.min(c.slope_max);
            let hi = c.slope_min.max(c.slope_max);
            let slope_w = if hi <= lo + 1e-3 {
                1.0
            } else {
                ((slope - lo) / (hi - lo)).clamp(0.0, 1.0)
            };
            let mask = mask * slope_w;
            if mask < 1e-4 {
                continue;
            }
            let hard = hardness.get(i as u32, j as u32);
            let mut local_max = hf.get(i as u32, j as u32);
            let mut local_min = local_max;
            for dj in -r..=r {
                for di in -r..=r {
                    let x = (i + di).clamp(0, w - 1) as u32;
                    let y = (j + dj).clamp(0, h - 1) as u32;
                    let z = hf.get(x, y);
                    local_max = local_max.max(z);
                    local_min = local_min.min(z);
                }
            }
            let h0 = hf.get(i as u32, j as u32);
            // Nonlinear sharpen toward local max on the face; preserve top/bottom via mask.
            let up = (local_max - h0).min(amt) * mask * (0.35 + hard * 0.65);
            let down = (h0 - local_min).min(amt * 0.35) * mask * (1.0 - hard * 0.5) * 0.4;
            let sharpened = h0 + up - down;
            // Transition smoothness blends back toward original at mask edges.
            let t = mask.powf(0.5 + c.transition);
            out.set(i as u32, j as u32, h0 * (1.0 - t) + sharpened * t);
        }
    }

    if c.talus > 1e-3 {
        out = deposit_talus(
            &out,
            c.talus * amt * 0.45,
            32.0 + c.rock_hardness * 12.0,
            p.iterations.max(1).min(6),
            Some(&hardness),
        );
    }
    out
}

/// Cliff/ridge detection + anisotropic fracture breakup (chips removed from faces).
pub fn apply_chipped(hf: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let cliffs = detect_cliffs(
        hf,
        p.radius.max(1) + 1,
        p.flow_threshold.max(0.08),
        p.slope_min,
    );
    let (ridge, _) = ridge_valley_likelihood(hf, 12.0);
    let hardness = lithology_hardness(hf, p.rock_hardness, p.frequency.max(0.02), p.seed);
    let frac = fracture_field(hf, p.frequency.max(0.04), p.seed ^ 0xC11F, 3);
    let w = hf.metrics.width as usize;
    let amt = p.amount;
    let mut out = hf.clone();

    for j in 0..hf.metrics.height {
        for i in 0..hf.metrics.width {
            let face = cliffs.get(i, j).max(ridge.get(i, j) * 0.65);
            let slope = slope_deg(hf, i, j);
            let lo = p.slope_min.min(p.slope_max);
            let hi = p.slope_min.max(p.slope_max);
            let g = if hi <= lo + 1e-3 {
                1.0
            } else {
                ((slope - lo) / (hi - lo)).clamp(0.0, 1.0)
            };
            let drive = face * g;
            if drive < 1e-4 {
                continue;
            }
            let soft = 1.0 - hardness.get(i, j);
            let n = frac[(j as usize) * w + i as usize];
            // Only remove (negative chips) from rock faces — no random bumps.
            let chip = if n > 0.12 {
                -(n - 0.12) * amt * (0.55 + soft * 0.9)
            } else {
                0.0
            };
            out.set(i, j, hf.get(i, j) + chip * drive);
        }
    }
    out
}

/// Canyon: hydrology incision + wall talus.
pub fn apply_canyon(hf: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let c = CanyonControls::from_params(p);
    let mut out = incise_channels(hf, &c);
    if c.talus > 1e-3 {
        // Talus only on steep walls — protect the freshly carved floor from refill.
        let k = lithology_hardness(&out, c.rock_hardness, 0.02, c.seed);
        let walls = detect_cliffs(&out, 2, 0.12, 28.0);
        let mut wall_hard = k.clone();
        for j in 0..hf.metrics.height {
            for i in 0..hf.metrics.width {
                // High hardness on valley floors → thermal moves less into channels.
                let floor = 1.0 - walls.get(i, j);
                let h = k.get(i, j).max(floor * 0.92);
                wall_hard.set(i, j, h);
            }
        }
        out = deposit_talus(
            &out,
            c.talus * c.incision * 0.18,
            36.0 + c.rock_hardness * 10.0,
            c.age.min(4),
            Some(&wall_hard),
        );
    }
    out
}

/// Rocky Sharp — ridge-aware anisotropic sharpening + light controlled erosion.
pub fn apply_rocky_sharp(hf: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let enhanced = enhance_ridges(
        hf,
        p.amount,
        p.slope_min,
        p.slope_max,
        p.frequency,
        p.seed,
        false,
    );
    let k = lithology_hardness(&enhanced, p.rock_hardness.max(0.45), p.frequency, p.seed);
    differential_erode(&enhanced, &k, p.amount * 0.12, 1, true)
}

/// Rocky Wide — lower-frequency ridge/rock enhancement + broad lithology.
pub fn apply_rocky_wide(hf: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let enhanced = enhance_ridges(
        hf,
        p.amount,
        p.slope_min,
        p.slope_max,
        p.frequency.max(0.004),
        p.seed,
        true,
    );
    let k = lithology_hardness(
        &enhanced,
        p.rock_hardness.max(0.4),
        p.frequency * 0.35,
        p.seed,
    );
    differential_erode(&enhanced, &k, p.amount * 0.18, 2, true)
}

/// Rocky Layers — warped strata exposure on cliff faces.
pub fn apply_rocky_layers(hf: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let cliffs = detect_cliffs(hf, p.radius.max(1) + 1, 0.12, p.slope_min.max(12.0));
    let (hardness, _) = strata_fields(hf, p.frequency.max(0.01), p.seed, p.rock_hardness);
    let exposed = expose_strata(hf, &hardness, p.amount * 0.55, &cliffs);
    // Slight ledge reinforcement on hard beds.
    let dilated = dilate(&exposed, 1);
    let mut out = exposed.clone();
    for j in 0..hf.metrics.height {
        for i in 0..hf.metrics.width {
            let face = cliffs.get(i, j);
            if face < 0.05 {
                continue;
            }
            let hard = hardness.get(i, j);
            if hard < 0.5 {
                continue;
            }
            let h0 = exposed.get(i, j);
            let up = (dilated.get(i, j) - h0).min(p.amount * 0.2);
            out.set(i, j, h0 + up * hard * face);
        }
    }
    out
}

/// Rocky Cliffs — cliff detect + multi-scale fracture + talus + fine erosion.
pub fn apply_rocky_cliffs(hf: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut sharp = apply_rocky_sharp(hf, p);
    let cliffs = detect_cliffs(
        &sharp,
        p.radius.max(1),
        p.flow_threshold.max(0.1),
        p.slope_min,
    );
    let frac = fracture_field(&sharp, p.frequency.max(0.03), p.seed ^ 0xA0CC_u64, 3);
    let hardness = lithology_hardness(&sharp, p.rock_hardness.max(0.5), p.frequency, p.seed);
    let w = hf.metrics.width as usize;
    let amt = p.amount * 0.35;

    for j in 0..hf.metrics.height {
        for i in 0..hf.metrics.width {
            let face = cliffs.get(i, j);
            if face < 1e-4 {
                continue;
            }
            let n = frac[j as usize * w + i as usize];
            let soft = 1.0 - hardness.get(i, j);
            let chip = if n > 0.1 {
                -(n - 0.1) * amt * soft
            } else {
                0.0
            };
            // Reinforce hard face toward local max.
            let r = p.radius.max(1) as i32;
            let mut local_max = sharp.get(i, j);
            for dj in -r..=r {
                for di in -r..=r {
                    let x = (i as i32 + di).clamp(0, hf.metrics.width as i32 - 1) as u32;
                    let y = (j as i32 + dj).clamp(0, hf.metrics.height as i32 - 1) as u32;
                    local_max = local_max.max(sharp.get(x, y));
                }
            }
            let h0 = sharp.get(i, j);
            let raised = h0 + (local_max - h0).min(amt) * face * hardness.get(i, j);
            sharp.set(i, j, raised + chip * face);
        }
    }

    if p.talus_mix > 1e-3 {
        sharp = deposit_talus(
            &sharp,
            p.talus_mix * p.amount * 0.3,
            36.0,
            p.iterations.max(1).min(5),
            Some(&hardness),
        );
    }
    sharp
}

/// Rocky Hard — low erodibility substrate + differential stripping of soft surrounds.
pub fn apply_rocky_hard(hf: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut p2 = p.clone();
    p2.rock_hardness = p.rock_hardness.max(0.72);
    p2.frequency = p.frequency.max(0.04) * 1.25;
    p2.amount = p.amount * 1.15;
    p2.slope_min = p.slope_min.max(35.0);
    let cliffs = apply_rocky_cliffs(hf, &p2);
    let k = lithology_hardness(
        &cliffs,
        p2.rock_hardness,
        p2.frequency * 0.5,
        p.seed ^ 0x85AD_u64,
    );
    // Stronger differential: strip soft surrounds, keep hard outcrops.
    differential_erode(
        &cliffs,
        &k,
        p.amount * 0.35,
        p.iterations.max(2).min(5),
        true,
    )
}

/// Rocky Plateaus — preserve flats, scarp at margins, incision, talus, optional strata.
pub fn apply_rocky_plateaus(hf: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let m = hf.metrics;
    let step = p.amount.max(2.0);
    let (ridge, valley) = ridge_valley_likelihood(hf, 64.0);
    let mut plateaus = hf.clone();

    // Broad plateau flatten on low-slope / low-ridge interiors.
    for j in 0..m.height {
        for i in 0..m.width {
            let slope = slope_deg(hf, i, j);
            let flat = (1.0 - (slope / 18.0).clamp(0.0, 1.0)) * (1.0 - ridge.get(i, j));
            if flat < 0.05 {
                continue;
            }
            let h0 = hf.get(i, j);
            let terraced = (h0 / step).floor() * step + step * 0.5;
            plateaus.set(i, j, h0 * (1.0 - flat * 0.75) + terraced * flat * 0.75);
        }
    }

    // Scarp / cliff formation at plateau margins.
    let mut scarp_p = p.clone();
    scarp_p.radius = p.radius.max(2);
    scarp_p.amount = p.amount * 0.45;
    scarp_p.slope_min = 12.0;
    scarp_p.talus_mix = 0.0;
    let with_scarps = apply_cliffs(&plateaus, &scarp_p);

    // Light drainage incision in valleys only.
    let mut canyon_p = p.clone();
    canyon_p.amount = p.amount * 0.25;
    canyon_p.radius = p.radius.max(2);
    canyon_p.flow_threshold = p.flow_threshold.max(0.2);
    canyon_p.talus_mix = 0.0;
    canyon_p.iterations = 1;
    canyon_p.valley_floor = p.valley_floor.max(0.35);
    let mut incised = apply_canyon(&with_scarps, &canyon_p);

    // Protect broad flats from incision over-cut.
    for j in 0..m.height {
        for i in 0..m.width {
            let slope = slope_deg(&plateaus, i, j);
            let flat = (1.0 - (slope / 14.0).clamp(0.0, 1.0)) * (1.0 - valley.get(i, j) * 0.5);
            if flat > 0.4 {
                let a = plateaus.get(i, j);
                let b = incised.get(i, j);
                incised.set(i, j, a * flat + b * (1.0 - flat));
            }
        }
    }

    // Optional strata on scarps.
    if p.frequency > 0.004 {
        let cliffs = detect_cliffs(&incised, 2, 0.1, 15.0);
        let (hardness, _) = strata_fields(&incised, p.frequency, p.seed, p.rock_hardness);
        incised = expose_strata(&incised, &hardness, p.amount * 0.2, &cliffs);
    }

    if p.talus_mix > 1e-3 {
        let k = lithology_hardness(&incised, p.rock_hardness.max(0.5), 0.015, p.seed);
        incised = deposit_talus(
            &incised,
            p.talus_mix * p.amount * 0.28,
            33.0,
            p.iterations.max(1).min(5),
            Some(&k),
        );
    }
    incised
}

/// Generic Rocky — soft drainage + rocky sharp mix.
pub fn apply_rocky(
    hf: &Heightfield,
    p: &EffectFilterParams,
    soft_flows_fn: impl Fn(&Heightfield, &EffectFilterParams) -> Heightfield,
) -> Heightfield {
    let flows = soft_flows_fn(hf, p);
    let rock = apply_rocky_sharp(&flows, p);
    let mut out = hf.clone();
    for j in 0..hf.metrics.height {
        for i in 0..hf.metrics.width {
            out.set(i, j, flows.get(i, j) * 0.45 + rock.get(i, j) * 0.55);
        }
    }
    out
}

/// Cliff reinforce (legacy Wave A) rebuilt on cliff detection.
pub fn apply_cliff_reinforce(hf: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let mut p2 = p.clone();
    p2.talus_mix = 0.0;
    p2.wall_steepness = p.wall_steepness.max(0.55);
    apply_cliffs(hf, &p2)
}
