//! World Creator–style multilevel simulation helpers.
//!
//! Expensive sims (thermal / hydraulic) run primarily at coarse resolutions, then
//! results are upsampled into the target field with optional light polish iterations.
//! Phase I adds Schott-adapted multi-scale amplify on the same `SimLevel` schedule.

use crate::heightfield::{FloatArena, Heightfield, HeightfieldMetrics};
use crate::hydro::{self, StreamPowerParams};
use crate::mask::MaskField;
use std::cell::RefCell;

use super::{HydraulicErosionParams, MultiScaleAmplifyParams, ThermalErosionParams};

thread_local! {
    static DOWNSAMPLE_ARENA: RefCell<FloatArena> = RefCell::new(FloatArena::new());
}

/// One coarse-to-fine simulation level.
#[derive(Debug, Clone, Copy)]
pub struct SimLevel {
    pub resolution: u32,
    /// Fraction of the authoring iteration count to run at this level.
    pub iter_scale: f32,
    /// Strength / rainfall scale at this level (WC: weaker on fine levels).
    pub effect_scale: f32,
    /// Smallest physical feature wavelength this level is responsible for (metres).
    /// Used by the universal terrain pyramid to skip ops that cannot be represented.
    pub physical_wavelength_m: f32,
}

impl SimLevel {
    pub fn new(resolution: u32, iter_scale: f32, effect_scale: f32) -> Self {
        // Wavelength roughly: world_extent / resolution — caller may override.
        let physical_wavelength_m = (8192.0 / resolution.max(1) as f32).max(1.0);
        Self {
            resolution,
            iter_scale,
            effect_scale,
            physical_wavelength_m,
        }
    }

    pub fn with_wavelength(mut self, metres: f32) -> Self {
        self.physical_wavelength_m = metres.max(0.1);
        self
    }
}

/// Default interactive level schedule (coarse → fine).
pub fn default_sim_levels(target_res: u32) -> Vec<SimLevel> {
    let mut levels = Vec::new();
    // Always include a cheap coarse pass.
    if target_res >= 128 {
        levels.push(SimLevel::new(128, 0.55, 1.0));
    }
    if target_res >= 256 {
        levels.push(SimLevel::new(256.min(target_res), 0.35, 0.75));
    }
    if target_res > 256 {
        levels.push(SimLevel::new(target_res, 0.15, 0.4));
    } else if levels.is_empty() {
        levels.push(SimLevel::new(target_res.max(32), 1.0, 1.0));
    }
    // Deduplicate if target equals a prior level.
    levels.dedup_by(|a, b| a.resolution == b.resolution);
    levels
}

/// Draft quality: coarse + mid level so thermal/hydraulic already look eroded.
pub fn draft_sim_levels(target_res: u32) -> Vec<SimLevel> {
    let coarse = target_res.min(128).max(64);
    let mid = target_res.min(192).max(coarse);
    if mid <= coarse {
        vec![SimLevel::new(coarse, 0.65, 1.0)]
    } else {
        vec![
            SimLevel::new(coarse, 0.55, 1.0),
            SimLevel::new(mid, 0.4, 0.85),
        ]
    }
}

/// Apply artist-facing WC level-step controls onto a quality base schedule.
///
/// - `level_count == 0` keeps the base length; else truncates to that many levels.
/// - `start_level` drops coarse levels from the front.
/// - `level_step_strength` scales fine-level `effect_scale` / `iter_scale` (1.0 = unchanged).
pub fn author_sim_levels(
    base: Vec<SimLevel>,
    level_count: u32,
    start_level: u32,
    strength: f32,
) -> Vec<SimLevel> {
    if base.is_empty() {
        return base;
    }
    let mut levels = base;
    if level_count > 0 && (levels.len() as u32) > level_count {
        levels.truncate(level_count as usize);
    }
    let start = (start_level as usize).min(levels.len().saturating_sub(1));
    if start > 0 {
        levels = levels[start..].to_vec();
    }
    let strength = strength.clamp(0.05, 3.0);
    let n = levels.len();
    for (i, level) in levels.iter_mut().enumerate() {
        let t = if n <= 1 {
            0.5
        } else {
            i as f32 / (n - 1) as f32
        };
        // Strength > 1 boosts fine levels; < 1 weakens them relative to coarse.
        let fine_boost = 1.0 + (strength - 1.0) * t;
        level.effect_scale = (level.effect_scale * fine_boost).clamp(0.05, 2.5);
        level.iter_scale = (level.iter_scale * (0.85 + 0.15 * strength)).clamp(0.05, 1.5);
    }
    levels
}

/// Wavelength-aware amplify schedule (Schott-adapted).
///
/// Coarse levels carry large-wavelength thermal; fine levels carry more SPE /
/// deposition detail with lower thermal effect so drainage structure survives.
pub fn amplify_sim_levels(target_res: u32, level_count: u32) -> Vec<SimLevel> {
    let mut levels = Vec::new();
    let res = target_res.max(32);
    // Candidate wavelengths: quarter → half → full (and 1/8 when large).
    let candidates: Vec<u32> = if res >= 1024 {
        vec![res / 8, res / 4, res / 2, res]
    } else if res >= 256 {
        vec![res / 4, res / 2, res]
    } else if res >= 128 {
        vec![res / 2, res]
    } else {
        vec![res]
    };
    let mut seen = Vec::new();
    for (i, &r) in candidates.iter().enumerate() {
        let r = r.max(32).min(res);
        if seen.contains(&r) {
            continue;
        }
        seen.push(r);
        // Thermal heavy at coarse; SPE/detail heavier toward fine (encoded via effect).
        let t = if candidates.len() <= 1 {
            0.5
        } else {
            i as f32 / (candidates.len() - 1) as f32
        };
        levels.push(
            SimLevel::new(r, 0.55 - 0.25 * t, 0.45 + 0.55 * t)
                .with_wavelength((8192.0 / r as f32).max(1.0)),
        );
    }
    if level_count > 0 && (level_count as usize) < levels.len() {
        // Keep coarsest + finest when clamping.
        let n = level_count as usize;
        if n == 1 {
            levels = vec![*levels.last().unwrap()];
        } else {
            let mut kept = vec![levels[0]];
            let mid = &levels[1..levels.len() - 1];
            let take_mid = n - 2;
            if take_mid > 0 && !mid.is_empty() {
                let step = (mid.len() as f32 / take_mid as f32).max(1.0);
                let mut acc = 0.0;
                for _ in 0..take_mid {
                    let idx = (acc as usize).min(mid.len() - 1);
                    kept.push(mid[idx]);
                    acc += step;
                }
            }
            kept.push(*levels.last().unwrap());
            kept.dedup_by(|a, b| a.resolution == b.resolution);
            levels = kept;
        }
    }
    levels
}

/// Outputs of a multi-scale amplify pass.
pub struct AmplifyResult {
    pub height: Heightfield,
    pub erosion: MaskField,
    pub deposition: MaskField,
}

/// Schott-adapted multi-scale amplify on Terra's coarse→fine `SimLevel` path.
///
/// Hardness and optional ridge-lock retarget toward the pre-level surface so
/// hard ridges resist amplify carve while soft areas receive more detail.
pub fn multi_scale_amplify(
    input: &Heightfield,
    p: &MultiScaleAmplifyParams,
    hardness: &MaskField,
    ridge_lock: Option<&MaskField>,
    levels: &[SimLevel],
) -> AmplifyResult {
    let target = input.metrics;
    let mut current = input.clone();
    let mut erosion_acc = MaskField::zeros(target);
    let mut deposit_acc = MaskField::zeros(target);
    let lock_strength = p.lock_strength.clamp(0.0, 1.0);
    let detail_boost = p.detail_boost.max(0.0);

    for (li, level) in levels.iter().enumerate() {
        let before = current.clone();
        let low = downsample_height(&current, level.resolution);
        let k_low = downsample_mask_field(hardness, level.resolution);
        let lock_low = ridge_lock.map(|m| downsample_mask_field(m, level.resolution));

        // Wavelength split: coarse = thermal-dominant; fine = SPE + deposition.
        let fine_t = if levels.len() <= 1 {
            1.0
        } else {
            li as f32 / (levels.len() - 1) as f32
        };
        let thermal_w = (1.0 - 0.65 * fine_t) * level.iter_scale;
        let spe_w = (0.25 + 0.75 * fine_t) * level.effect_scale;
        let dep_w = fine_t * level.effect_scale;

        let thermal_p = ThermalErosionParams {
            iterations: ((p.thermal_iters as f32 * thermal_w).round() as u32).max(1),
            strength: (p.thermal_strength * thermal_w).clamp(0.0, 1.0),
            talus_angle_deg: p.talus_angle_deg,
            hardness: 0.0,
            hardness_source: crate::mask::MaskSource::None,
            ..Default::default()
        };
        let (mut h, e_mask, d_mask) =
            super::erosion::thermal_erode_with_hardness(&low, &thermal_p, &k_low);

        if p.spe_strength > 1e-6 && p.spe_iters > 0 {
            let spe_iters =
                ((p.spe_iters as f32 * spe_w).round() as u32).max(if spe_w > 0.15 { 1 } else { 0 });
            if spe_iters > 0 {
                let spe_p = StreamPowerParams {
                    iterations: spe_iters,
                    k: 0.05 * p.spe_strength * spe_w,
                    m: 0.5,
                    n: 1.0,
                    uplift_rate: 0.0,
                    base_level: 0.0,
                    dt: 0.85,
                    use_dinfinity: false,
                    refill_each_iter: false,
                    drainage_reuse_stride: 1,
                    hardness: 0.0,
                    hardness_source: crate::mask::MaskSource::None,
                    dendritic_seed: 0.0,
                    stream_threshold: 30.0,
                    ..Default::default()
                };
                let spe = hydro::stream_power_erode(&h, &spe_p, &k_low);
                h = spe.height;
            }
        }

        if p.deposition_strength > 1e-6 && dep_w > 0.2 {
            let hyd_p = HydraulicErosionParams {
                iterations: ((8.0 * dep_w).round() as u32).max(2),
                rainfall: 0.02 * dep_w,
                evaporation: 0.015,
                capacity: 0.12,
                erosion: 0.12 * dep_w,
                deposition: (0.55 * p.deposition_strength * dep_w).clamp(0.0, 1.0),
                timestep: 0.2,
                hardness: 0.0,
                hardness_source: crate::mask::MaskSource::None,
                fan_boost: 0.8 * p.deposition_strength,
                floodplain_bias: 0.5 * p.deposition_strength,
                bank_slip: 0.0,
                sediment_softness: 0.0,
                ..Default::default()
            };
            let r = super::erosion::hydraulic_erode_with_hardness(&h, &hyd_p, &k_low);
            h = r.height;
        }

        // Upsample processed level and hardness-aware retarget blend toward `before`.
        let processed = upsample_to_metrics(&h, target);
        let k_full = upsample_mask(&k_low, target);
        let lock_full = lock_low
            .as_ref()
            .map(|m| upsample_mask(m, target))
            .unwrap_or_else(|| MaskField::zeros(target));
        let e_up = upsample_mask(&e_mask, target);
        let d_up = upsample_mask(&d_mask, target);

        let w = target.width as usize;
        let hgt = target.height as usize;
        let mut blended = before.clone();
        for j in 0..hgt {
            for i in 0..w {
                let ii = i as u32;
                let jj = j as u32;
                let base = before.get(ii, jj);
                let amp = processed.get(ii, jj);
                let k = k_full.get(ii, jj).clamp(0.0, 1.0);
                let lock = lock_full.get(ii, jj).clamp(0.0, 1.0);
                // Soft + unlocked → accept amplify delta (optionally boosted).
                let soft = (1.0 - k).max(0.0);
                let unlock = (1.0 - lock * lock_strength).max(0.0);
                let accept = (soft * unlock * detail_boost).clamp(0.0, 1.5);
                let accept = (accept / 1.5).clamp(0.0, 1.0);
                // Hard / locked cells retarget toward pre-level height.
                let preserve = (k.max(lock) * lock_strength).clamp(0.0, 1.0);
                let t = (accept * (1.0 - preserve)).clamp(0.0, 1.0);
                blended.set(ii, jj, base * (1.0 - t) + amp * t);
                let er = erosion_acc.get(ii, jj) + e_up.get(ii, jj) * t;
                let dep = deposit_acc.get(ii, jj) + d_up.get(ii, jj) * t;
                erosion_acc.set(ii, jj, er);
                deposit_acc.set(ii, jj, dep);
            }
        }
        current = blended;
    }

    AmplifyResult {
        height: current,
        erosion: erosion_acc,
        deposition: deposit_acc,
    }
}

pub fn downsample_height(src: &Heightfield, target_res: u32) -> Heightfield {
    let tw = target_res.max(1);
    let th = target_res.max(1);
    let metrics = HeightfieldMetrics {
        width: tw,
        height: th,
        world_size_x: src.metrics.world_size_x,
        world_size_z: src.metrics.world_size_z,
        tile_size: src.metrics.tile_size.min(tw),
        halo: src.metrics.halo,
    };
    if src.metrics.width == tw && src.metrics.height == th {
        return src.clone();
    }
    let dense = src.to_dense();
    let sw = src.metrics.width as usize;
    let sh = src.metrics.height as usize;
    let mut out =
        DOWNSAMPLE_ARENA.with(|arena| arena.borrow_mut().acquire(tw as usize * th as usize));
    if tw < src.metrics.width || th < src.metrics.height {
        for j in 0..th as usize {
            for i in 0..tw as usize {
                out[j * tw as usize + i] =
                    area_sample(&dense, sw, sh, tw as usize, th as usize, i, j);
            }
        }
        let result = Heightfield::from_dense(metrics, &out);
        DOWNSAMPLE_ARENA.with(|arena| arena.borrow_mut().release(out));
        return result;
    }
    for j in 0..th as usize {
        for i in 0..tw as usize {
            let u = (i as f32 + 0.5) / tw as f32;
            let v = (j as f32 + 0.5) / th as f32;
            let x = (u * sw as f32 - 0.5).clamp(0.0, (sw - 1) as f32);
            let z = (v * sh as f32 - 0.5).clamp(0.0, (sh - 1) as f32);
            let x0 = x.floor() as usize;
            let z0 = z.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let z1 = (z0 + 1).min(sh - 1);
            let fx = x - x0 as f32;
            let fz = z - z0 as f32;
            let h00 = dense[z0 * sw + x0];
            let h10 = dense[z0 * sw + x1];
            let h01 = dense[z1 * sw + x0];
            let h11 = dense[z1 * sw + x1];
            out[j * tw as usize + i] = h00 * (1.0 - fx) * (1.0 - fz)
                + h10 * fx * (1.0 - fz)
                + h01 * (1.0 - fx) * fz
                + h11 * fx * fz;
        }
    }
    let result = Heightfield::from_dense(metrics, &out);
    DOWNSAMPLE_ARENA.with(|arena| arena.borrow_mut().release(out));
    result
}

fn area_sample(
    dense: &[f32],
    sw: usize,
    sh: usize,
    tw: usize,
    th: usize,
    i: usize,
    j: usize,
) -> f32 {
    let x0 = i as f32 * sw as f32 / tw as f32;
    let x1 = (i + 1) as f32 * sw as f32 / tw as f32;
    let y0 = j as f32 * sh as f32 / th as f32;
    let y1 = (j + 1) as f32 * sh as f32 / th as f32;
    let sx0 = x0.floor() as usize;
    let sx1 = x1.ceil().min(sw as f32) as usize;
    let sy0 = y0.floor() as usize;
    let sy1 = y1.ceil().min(sh as f32) as usize;
    let mut sum = 0.0;
    let mut weight = 0.0;
    for sy in sy0..sy1 {
        let wy = (y1.min((sy + 1) as f32) - y0.max(sy as f32)).max(0.0);
        for sx in sx0..sx1 {
            let wx = (x1.min((sx + 1) as f32) - x0.max(sx as f32)).max(0.0);
            let w = wx * wy;
            sum += dense[sy * sw + sx] * w;
            weight += w;
        }
    }
    if weight <= f32::EPSILON {
        0.0
    } else {
        sum / weight
    }
}
fn scale_thermal(p: &ThermalErosionParams, level: SimLevel) -> ThermalErosionParams {
    ThermalErosionParams {
        iterations: ((p.iterations as f32 * level.iter_scale).round() as u32).max(1),
        strength: (p.strength * level.effect_scale).clamp(0.0, 1.0),
        talus_angle_deg: p.talus_angle_deg,
        hardness: p.hardness,
        hardness_source: p.hardness_source.clone(),
        material_amount: p.material_amount,
        weathering_rate: p.weathering_rate * level.effect_scale,
        transport_distance: p.transport_distance,
        layered_materials: p.layered_materials,
        level_count: p.level_count,
        start_level: p.start_level,
        level_step_strength: p.level_step_strength,
        level_step_curve: p.level_step_curve.clone(),
    }
}

fn scale_hydraulic(p: &HydraulicErosionParams, level: SimLevel) -> HydraulicErosionParams {
    HydraulicErosionParams {
        iterations: ((p.iterations as f32 * level.iter_scale).round() as u32).max(1),
        rainfall: p.rainfall * level.effect_scale,
        erosion: p.erosion * level.effect_scale,
        deposition: p.deposition * level.effect_scale,
        // Bank-slip / sediment hooks run once at full effect on the finest level only.
        bank_slip: if level.resolution >= 64 {
            p.bank_slip * level.effect_scale
        } else {
            0.0
        },
        ..p.clone()
    }
}

/// Run thermal erosion across multilevel schedule, returning final height + masks at target res.
pub fn thermal_erode_leveled(
    input: &Heightfield,
    p: &ThermalErosionParams,
    levels: &[SimLevel],
) -> (Heightfield, MaskField, MaskField) {
    use super::erosion::thermal_erode;

    let target = input.metrics;
    let mut current = input.clone();
    let mut last_e = MaskField::zeros(target);
    let mut last_d = MaskField::zeros(target);

    for level in levels {
        let low = downsample_height(&current, level.resolution);
        let params = scale_thermal(p, *level);
        let (h, e, d) = thermal_erode(&low, &params);
        current = apply_height_delta(&current, &low, &h, target);
        last_e = upsample_mask(&e, target);
        last_d = upsample_mask(&d, target);
    }
    (current, last_e, last_d)
}

/// Multilevel thermal with a spatial hardness map (downsampled per level).
pub fn thermal_erode_leveled_with_hardness(
    input: &Heightfield,
    p: &ThermalErosionParams,
    hardness: &MaskField,
    levels: &[SimLevel],
) -> (Heightfield, MaskField, MaskField) {
    use super::erosion::thermal_erode_with_hardness;

    let target = input.metrics;
    let mut current = input.clone();
    let mut last_e = MaskField::zeros(target);
    let mut last_d = MaskField::zeros(target);

    for level in levels {
        let low = downsample_height(&current, level.resolution);
        let k_low = downsample_mask_field(hardness, level.resolution);
        let params = scale_thermal(p, *level);
        let (h, e, d) = thermal_erode_with_hardness(&low, &params, &k_low);
        current = apply_height_delta(&current, &low, &h, target);
        last_e = upsample_mask(&e, target);
        last_d = upsample_mask(&d, target);
    }
    (current, last_e, last_d)
}

/// Run hydraulic erosion across multilevel schedule.
pub fn hydraulic_erode_leveled(
    input: &Heightfield,
    p: &HydraulicErosionParams,
    levels: &[SimLevel],
) -> super::erosion::HydraulicResult {
    use super::erosion::{hydraulic_erode, HydraulicResult};

    let target = input.metrics;
    let mut current = input.clone();
    let mut last = HydraulicResult {
        height: current.clone(),
        wetness: MaskField::zeros(target),
        sediment: MaskField::zeros(target),
        erosion: MaskField::zeros(target),
        deposition: MaskField::zeros(target),
        water_raw: MaskField::zeros(target),
        sediment_raw: MaskField::zeros(target),
        erosion_raw: MaskField::zeros(target),
        deposition_raw: MaskField::zeros(target),
    };

    for level in levels {
        let low = downsample_height(&current, level.resolution);
        let params = scale_hydraulic(p, *level);
        let r = hydraulic_erode(&low, &params);
        current = apply_height_delta(&current, &low, &r.height, target);
        last = HydraulicResult {
            height: current.clone(),
            wetness: upsample_mask(&r.wetness, target),
            sediment: upsample_mask(&r.sediment, target),
            erosion: upsample_mask(&r.erosion, target),
            deposition: upsample_mask(&r.deposition, target),
            water_raw: upsample_mask(&r.water_raw, target),
            sediment_raw: upsample_mask(&r.sediment_raw, target),
            erosion_raw: upsample_mask(&r.erosion_raw, target),
            deposition_raw: upsample_mask(&r.deposition_raw, target),
        };
    }
    let hardness = MaskField::filled(target, p.hardness.clamp(0.0, 1.0));
    super::erosion::apply_particle_erosion(last, p, &hardness)
}

/// Multilevel hydraulic with a spatial hardness map (downsampled per level).
pub fn hydraulic_erode_leveled_with_hardness(
    input: &Heightfield,
    p: &HydraulicErosionParams,
    hardness: &MaskField,
    levels: &[SimLevel],
) -> super::erosion::HydraulicResult {
    hydraulic_erode_leveled_with_fields(input, p, hardness, None, None, levels)
}

/// Coarse-to-fine hydraulic solve that carries water and suspended sediment
/// into each finer level and samples spatial rainfall/protection per level.
pub fn hydraulic_erode_leveled_with_fields(
    input: &Heightfield,
    p: &HydraulicErosionParams,
    hardness: &MaskField,
    rainfall: Option<&MaskField>,
    protection: Option<&MaskField>,
    levels: &[SimLevel],
) -> super::erosion::HydraulicResult {
    use super::erosion::{hydraulic_erode_with_fields, HydraulicResult};

    let target = input.metrics;
    let mut current = input.clone();
    let mut state_water: Option<MaskField> = None;
    let mut state_sediment: Option<MaskField> = None;
    let mut last = HydraulicResult {
        height: current.clone(),
        wetness: MaskField::zeros(target),
        sediment: MaskField::zeros(target),
        erosion: MaskField::zeros(target),
        deposition: MaskField::zeros(target),
        water_raw: MaskField::zeros(target),
        sediment_raw: MaskField::zeros(target),
        erosion_raw: MaskField::zeros(target),
        deposition_raw: MaskField::zeros(target),
    };

    for level in levels {
        let low = downsample_height(&current, level.resolution);
        let k_low = downsample_mask_field(hardness, level.resolution);
        let rain_low = rainfall.map(|m| downsample_mask_field(m, level.resolution));
        let protect_low = protection.map(|m| downsample_mask_field(m, level.resolution));
        let water_low = state_water
            .as_ref()
            .map(|m| downsample_mask_field(m, level.resolution));
        let sediment_low = state_sediment
            .as_ref()
            .map(|m| downsample_mask_field(m, level.resolution));
        let params = scale_hydraulic(p, *level);
        let r = hydraulic_erode_with_fields(
            &low,
            &params,
            &k_low,
            rain_low.as_ref(),
            protect_low.as_ref(),
            water_low.as_ref(),
            sediment_low.as_ref(),
        );
        current = apply_height_delta(&current, &low, &r.height, target);
        state_water = Some(r.water_raw.clone());
        state_sediment = Some(r.sediment_raw.clone());
        last = HydraulicResult {
            height: current.clone(),
            wetness: upsample_mask(&r.wetness, target),
            sediment: upsample_mask(&r.sediment, target),
            erosion: upsample_mask(&r.erosion, target),
            deposition: upsample_mask(&r.deposition, target),
            water_raw: upsample_mask(&r.water_raw, target),
            sediment_raw: upsample_mask(&r.sediment_raw, target),
            erosion_raw: upsample_mask(&r.erosion_raw, target),
            deposition_raw: upsample_mask(&r.deposition_raw, target),
        };
    }
    // The droplet detail pass does not yet carry a protection field; skipping
    // it is preferable to cutting protected ridges after the main solver
    // correctly preserved them.
    if protection.is_some() {
        last
    } else {
        super::erosion::apply_particle_erosion(last, p, hardness)
    }
}

fn downsample_mask_field(src: &MaskField, target_res: u32) -> MaskField {
    let hf = Heightfield::from_dense(src.metrics, src.data());
    let low = downsample_height(&hf, target_res);
    let dense = low.to_dense();
    let mut out = MaskField::zeros(low.metrics);
    out.data_mut().copy_from_slice(&dense);
    out
}

fn upsample_to_metrics(src: &Heightfield, target: HeightfieldMetrics) -> Heightfield {
    if src.metrics.width == target.width && src.metrics.height == target.height {
        let mut h = src.clone();
        h.metrics = target;
        return h;
    }
    downsample_height(src, target.width)
}

/// Lift only the simulated low-frequency displacement back to the working
/// resolution. Replacing the base with an upsampled coarse solve destroys every
/// feature above the coarse Nyquist frequency; adding the displacement keeps
/// those authored ridges and surface details while still letting broad erosion
/// reshape the terrain.
fn apply_height_delta(
    base: &Heightfield,
    before_low: &Heightfield,
    after_low: &Heightfield,
    target: HeightfieldMetrics,
) -> Heightfield {
    let before = before_low.to_dense();
    let after = after_low.to_dense();
    let delta: Vec<f32> = after
        .iter()
        .zip(before.iter())
        .map(|(a, b)| a - b)
        .collect();
    let delta_low = Heightfield::from_dense(before_low.metrics, &delta);
    let delta_up = upsample_to_metrics(&delta_low, target);
    let base_values = base.to_dense();
    let displacement = delta_up.to_dense();
    let composed: Vec<f32> = base_values
        .iter()
        .zip(displacement.iter())
        .map(|(height, delta)| height + delta)
        .collect();
    Heightfield::from_dense(target, &composed)
}
fn upsample_mask(src: &MaskField, target: HeightfieldMetrics) -> MaskField {
    let hf = Heightfield::from_dense(src.metrics, src.data());
    let up = downsample_height(&hf, target.width);
    let dense = up.to_dense();
    let mut out = MaskField::zeros(target);
    out.data_mut().copy_from_slice(&dense);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

    #[test]
    fn downsample_upsample_preserves_flat() {
        let m = HeightfieldMetrics {
            width: 64,
            height: 64,
            world_size_x: 1000.0,
            world_size_z: 1000.0,
            tile_size: 64,
            halo: 2,
        };
        let src = Heightfield::from_dense(m, &vec![42.0f32; 64 * 64]);
        let low = downsample_height(&src, 16);
        assert_eq!(low.metrics.width, 16);
        let up = downsample_height(&low, 64);
        let d = up.to_dense();
        assert!((d[0] - 42.0).abs() < 0.01);
    }

    #[test]
    fn default_levels_include_coarse() {
        let levels = default_sim_levels(512);
        assert!(levels[0].resolution <= 128);
        assert_eq!(levels.last().unwrap().resolution, 512);
    }

    #[test]
    fn amplify_levels_end_at_target() {
        let levels = amplify_sim_levels(512, 0);
        assert_eq!(levels.last().unwrap().resolution, 512);
        assert!(levels.len() >= 2);
        let clamped = amplify_sim_levels(512, 2);
        assert!(clamped.len() <= 2);
        assert_eq!(clamped.last().unwrap().resolution, 512);
    }

    #[test]
    fn author_sim_levels_truncates_skips_and_scales() {
        let base = default_sim_levels(512);
        assert!(base.len() >= 2);
        let truncated = author_sim_levels(base.clone(), 1, 0, 1.0);
        assert_eq!(truncated.len(), 1);
        let skipped = author_sim_levels(base.clone(), 0, 1, 1.0);
        assert_eq!(skipped.len(), base.len() - 1);
        let boosted = author_sim_levels(base.clone(), 0, 0, 2.0);
        let last_base = base.last().unwrap().effect_scale;
        let last_boosted = boosted.last().unwrap().effect_scale;
        assert!(last_boosted > last_base);
    }

    #[test]
    fn area_downsample_rejects_checkerboard_aliasing() {
        let m = HeightfieldMetrics::new(64, 64, 640.0, 640.0);
        let mut values = Vec::with_capacity(64 * 64);
        for j in 0..64 {
            for i in 0..64 {
                values.push(if (i + j) % 2 == 0 { 0.0 } else { 1.0 });
            }
        }
        let src = Heightfield::from_dense(m, &values);
        let low = downsample_height(&src, 8);
        for value in low.to_dense() {
            assert!((value - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn coarse_displacement_preserves_fine_surface_contrast() {
        let m = HeightfieldMetrics::new(64, 64, 640.0, 640.0);
        let mut values = Vec::with_capacity(64 * 64);
        for j in 0..64 {
            for i in 0..64 {
                values.push(if (i + j) % 2 == 0 { 102.0 } else { 98.0 });
            }
        }
        let base = Heightfield::from_dense(m, &values);
        let before_low = downsample_height(&base, 8);
        let after_values: Vec<f32> = before_low
            .to_dense()
            .into_iter()
            .map(|height| height + 5.0)
            .collect();
        let after_low = Heightfield::from_dense(before_low.metrics, &after_values);
        let composed = apply_height_delta(&base, &before_low, &after_low, m);
        let result_contrast = composed.get(0, 0) - composed.get(1, 0);
        let replaced = upsample_to_metrics(&after_low, m);
        assert!((result_contrast - 4.0).abs() < 1e-4);
        assert!((replaced.get(0, 0) - replaced.get(1, 0)).abs() < 1e-4);
    }
}
