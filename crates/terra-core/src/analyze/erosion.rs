use crate::fields::{erodibility_at_strata_depth, stability_at_strata_depth};
use crate::generators::geology::strata_depth_m;
use crate::heightfield::Heightfield;
use crate::mask::MaskField;
use crate::material_schema::{BedGeometry, Stratum};

use super::{HydraulicErosionParams, ThermalErosionParams};

/// Thermal erosion via talus-angle redistribution (CPU reference).
/// Returns (height, erosion_mask, deposition_mask).
///
/// Hardness `K ∈ [0,1]` modulates move amount as `(1-K)` (Musgrave-style talus with
/// material resistance). When `hardness` is `None`, uses constant `p.hardness`.
pub fn thermal_erode(
    input: &Heightfield,
    p: &ThermalErosionParams,
) -> (Heightfield, MaskField, MaskField) {
    let k_field = MaskField::filled(input.metrics, p.hardness.clamp(0.0, 1.0));
    thermal_erode_with_hardness(input, p, &k_field)
}

/// Thermal erosion with an explicit hardness map.
///
/// When `layered_materials` is set (default), uses the mass-wasting solver that
/// keeps bedrock and loose debris separate (Yang weathering + Musgrave transport).
/// Otherwise falls back to classical height-only redistribution.
pub fn thermal_erode_with_hardness(
    input: &Heightfield,
    p: &ThermalErosionParams,
    hardness: &MaskField,
) -> (Heightfield, MaskField, MaskField) {
    if p.layered_materials {
        return super::mass_wasting::thermal_erode_mass(input, p, hardness);
    }
    thermal_erode_height_only(input, p, hardness)
}

/// Classical Musgrave height-only talus redistribute (no separate loose layer).
fn thermal_erode_height_only(
    input: &Heightfield,
    p: &ThermalErosionParams,
    hardness: &MaskField,
) -> (Heightfield, MaskField, MaskField) {
    let mut h = input.to_dense();
    let w = input.metrics.width as usize;
    let hh = input.metrics.height as usize;
    let dx = input.metrics.dx();
    let dz = input.metrics.dz();
    let talus_slope = p.talus_angle_deg.to_radians().tan();
    let strength = p.strength.clamp(0.0, 1.0);

    let mut erosion = vec![0.0f32; w * hh];
    let mut deposit = vec![0.0f32; w * hh];

    let diagonal = dx.hypot(dz);
    let neighbors = [
        (-1i32, 0i32, dx),
        (1, 0, dx),
        (0, -1, dz),
        (0, 1, dz),
        (-1, -1, diagonal),
        (1, -1, diagonal),
        (-1, 1, diagonal),
        (1, 1, diagonal),
    ];

    for _ in 0..p.iterations {
        let src = h.clone();
        for j in 0..hh as i32 {
            for i in 0..w as i32 {
                let idx = j as usize * w + i as usize;
                let h0 = src[idx];
                let k = hardness.get(i as u32, j as u32).clamp(0.0, 1.0);
                let soft = 1.0 - k;
                if soft <= 1e-6 {
                    continue;
                }
                let mut deltas = Vec::with_capacity(8);
                let mut sum = 0.0f32;
                for &(di, dj, distance) in &neighbors {
                    let ni = i + di;
                    let nj = j + dj;
                    if ni < 0 || nj < 0 || ni >= w as i32 || nj >= hh as i32 {
                        continue;
                    }
                    let nidx = nj as usize * w + ni as usize;
                    let diff = h0 - src[nidx] - talus_slope * distance;
                    if diff > 0.0 {
                        deltas.push((nidx, diff));
                        sum += diff;
                    }
                }
                if sum <= 0.0 {
                    continue;
                }
                let move_amt = sum * strength * 0.125 * soft;
                h[idx] -= move_amt;
                erosion[idx] += move_amt;
                for (nidx, diff) in deltas {
                    let share = move_amt * (diff / sum);
                    h[nidx] += share;
                    deposit[nidx] += share;
                }
            }
        }
    }

    let height = Heightfield::from_dense(input.metrics, &h);
    let (e_mask, d_mask) = normalize_pair(&erosion, &deposit, input.metrics);
    (height, e_mask, d_mask)
}

fn strata_limited_move(
    strata: &[Stratum],
    depth: f32,
    erosion_potential: f32,
    default_hardness: f32,
) -> f32 {
    let mut remaining_potential = erosion_potential.max(0.0);
    let mut moved = 0.0;
    let mut cursor = depth.max(0.0);
    let mut top = 0.0;

    for stratum in strata {
        let thickness = stratum.thickness.max(0.0);
        let bottom = top + thickness;
        if cursor <= bottom || thickness >= 1.0e5 {
            // Hydraulic path: lithology erodibility; thermal callers scale potential first.
            let softness = stratum.effective_erodibility();
            if softness <= 1e-6 {
                return moved;
            }
            let available = if thickness >= 1.0e5 {
                f32::INFINITY
            } else {
                (bottom - cursor).max(0.0)
            };
            let candidate = remaining_potential * softness;
            if candidate <= available {
                return moved + candidate;
            }
            moved += available;
            remaining_potential -= available / softness;
            cursor = bottom;
        }
        top = bottom;
    }
    moved + remaining_potential * (1.0 - default_hardness).clamp(0.0, 1.0)
}

/// Thermal erosion with depth-aware strata hardness (soft-over-hard stripping).
///
/// `reference` holds Materials-time heights (meters). Depth
/// \(d = \max(0, h_{\mathrm{ref}} - h + \mathrm{warp})\) selects the active stratum
/// each step. Material stability from lithology scales talus mobility.
pub fn thermal_erode_with_strata(
    input: &Heightfield,
    p: &ThermalErosionParams,
    reference: &MaskField,
    strata: &[Stratum],
    default_hardness: f32,
) -> (Heightfield, MaskField, MaskField) {
    thermal_erode_with_strata_ex(
        input,
        p,
        reference,
        strata,
        default_hardness,
        &BedGeometry::Horizontal,
    )
}

/// Thermal strata path with explicit bed geometry.
pub fn thermal_erode_with_strata_ex(
    input: &Heightfield,
    p: &ThermalErosionParams,
    reference: &MaskField,
    strata: &[Stratum],
    default_hardness: f32,
    geom: &BedGeometry,
) -> (Heightfield, MaskField, MaskField) {
    let mut h = input.to_dense();
    let w = input.metrics.width as usize;
    let hh = input.metrics.height as usize;
    let dx = input.metrics.dx();
    let dz = input.metrics.dz();
    let talus_slope = p.talus_angle_deg.to_radians().tan();
    let strength = p.strength.clamp(0.0, 1.0);
    let metrics = input.metrics;

    let mut erosion = vec![0.0f32; w * hh];
    let mut deposit = vec![0.0f32; w * hh];

    let diagonal = dx.hypot(dz);
    let neighbors = [
        (-1i32, 0i32, dx),
        (1, 0, dx),
        (0, -1, dz),
        (0, 1, dz),
        (-1, -1, diagonal),
        (1, -1, diagonal),
        (-1, 1, diagonal),
        (1, 1, diagonal),
    ];

    for _ in 0..p.iterations {
        let src = h.clone();
        for j in 0..hh as i32 {
            for i in 0..w as i32 {
                let idx = j as usize * w + i as usize;
                let h0 = src[idx];
                let x = metrics.world_x(i as u32);
                let z = metrics.world_z(j as u32);
                let depth = strata_depth_m(reference.get(i as u32, j as u32), h0, x, z, geom);
                let stability = stability_at_strata_depth(strata, depth, default_hardness);
                let mut deltas = Vec::with_capacity(8);
                let mut sum = 0.0f32;
                for &(di, dj, distance) in &neighbors {
                    let ni = i + di;
                    let nj = j + dj;
                    if ni < 0 || nj < 0 || ni >= w as i32 || nj >= hh as i32 {
                        continue;
                    }
                    let nidx = nj as usize * w + ni as usize;
                    let diff = h0 - src[nidx] - talus_slope * distance;
                    if diff > 0.0 {
                        deltas.push((nidx, diff));
                        sum += diff;
                    }
                }
                if sum <= 0.0 {
                    continue;
                }
                // Stable lithology resists talus motion; soft beds shed freely.
                let mobility = (1.0 - stability * 0.85).clamp(0.05, 1.0);
                let erosion_potential = sum * strength * 0.125 * mobility;
                let move_amt =
                    strata_limited_move(strata, depth, erosion_potential, default_hardness);
                if move_amt <= 1e-6 {
                    continue;
                }
                h[idx] -= move_amt;
                erosion[idx] += move_amt;
                for (nidx, diff) in deltas {
                    let share = move_amt * (diff / sum);
                    h[nidx] += share;
                    deposit[nidx] += share;
                }
            }
        }
    }

    let height = Heightfield::from_dense(input.metrics, &h);
    let (e_mask, d_mask) = normalize_pair(&erosion, &deposit, input.metrics);
    (height, e_mask, d_mask)
}

pub struct HydraulicResult {
    pub height: Heightfield,
    pub wetness: MaskField,
    pub sediment: MaskField,
    pub erosion: MaskField,
    pub deposition: MaskField,
    /// Final non-normalized water depth (for debug / invariants).
    pub water_raw: MaskField,
    /// Final non-normalized suspended sediment load (for debug / invariants).
    pub sediment_raw: MaskField,
    /// Cumulative eroded height (meters, non-normalized) — mass accounting.
    pub erosion_raw: MaskField,
    /// Cumulative deposited height (meters, non-normalized) — mass accounting / fans.
    pub deposition_raw: MaskField,
}

/// Soft sediment material ID written into aux materials when `sediment_softness > 0`.
pub const SEDIMENT_MATERIAL_ID: u32 = 3;

/// Simplified shallow-water hydraulic erosion (Mei-lite / Beneš steps, CPU oracle).
///
/// Each iteration: rain → compute 4-neighbor outflow → erode/deposit → transport
/// water+sediment to neighbors → evaporate. Hardness modulates erosion as `(1-K)`.
///
/// Phase G landforms: slope-break fan boost, floodplain bias, stagnant-pool settle,
/// optional post-pass bank slippage (thermal), and sediment→softness hooks via
/// [`HydraulicResult::deposition_raw`].
pub fn hydraulic_erode(input: &Heightfield, p: &HydraulicErosionParams) -> HydraulicResult {
    let k_field = MaskField::filled(input.metrics, p.hardness.clamp(0.0, 1.0));
    hydraulic_erode_with_hardness(input, p, &k_field)
}

pub fn hydraulic_erode_with_hardness(
    input: &Heightfield,
    p: &HydraulicErosionParams,
    hardness: &MaskField,
) -> HydraulicResult {
    hydraulic_erode_inner(
        input,
        p,
        HardnessMode::Map(hardness),
        None,
        None,
        None,
        None,
    )
}

/// Hydraulic erosion with depth-aware strata hardness (soft cap strips first).
pub fn hydraulic_erode_with_strata(
    input: &Heightfield,
    p: &HydraulicErosionParams,
    reference: &MaskField,
    strata: &[Stratum],
    default_hardness: f32,
) -> HydraulicResult {
    hydraulic_erode_with_strata_ex(
        input,
        p,
        reference,
        strata,
        default_hardness,
        &BedGeometry::Horizontal,
    )
}

/// Hydraulic strata path with explicit bed geometry (tilted / folded beds).
pub fn hydraulic_erode_with_strata_ex(
    input: &Heightfield,
    p: &HydraulicErosionParams,
    reference: &MaskField,
    strata: &[Stratum],
    default_hardness: f32,
    geom: &BedGeometry,
) -> HydraulicResult {
    hydraulic_erode_inner(
        input,
        p,
        HardnessMode::Strata {
            reference,
            strata,
            default_hardness,
            geom: *geom,
            metrics: input.metrics,
        },
        None,
        None,
        None,
        None,
    )
}

/// Hydraulic solve with spatial forcing and optional persistent matter state.
/// Rainfall is a per-cell multiplier, while protection directly reduces
/// erodibility inside the solver rather than blending its result afterward.
pub fn hydraulic_erode_with_fields(
    input: &Heightfield,
    p: &HydraulicErosionParams,
    hardness: &MaskField,
    rainfall: Option<&MaskField>,
    protection: Option<&MaskField>,
    initial_water: Option<&MaskField>,
    initial_sediment: Option<&MaskField>,
) -> HydraulicResult {
    hydraulic_erode_inner(
        input,
        p,
        HardnessMode::Map(hardness),
        rainfall,
        protection,
        initial_water,
        initial_sediment,
    )
}

enum HardnessMode<'a> {
    Map(&'a MaskField),
    Strata {
        reference: &'a MaskField,
        strata: &'a [Stratum],
        default_hardness: f32,
        geom: BedGeometry,
        metrics: crate::heightfield::HeightfieldMetrics,
    },
}

impl HardnessMode<'_> {
    /// Effective hardness \(K\). Strata path uses `1 - erodibility` so hydraulic
    /// soft factor consumes lithology erodibility rather than only `1 - K`.
    fn sample(&self, i: u32, j: u32, height: f32) -> f32 {
        match self {
            HardnessMode::Map(k) => k.get(i, j).clamp(0.0, 1.0),
            HardnessMode::Strata {
                reference,
                strata,
                default_hardness,
                geom,
                metrics,
            } => {
                let x = metrics.world_x(i);
                let z = metrics.world_z(j);
                let depth = strata_depth_m(reference.get(i, j), height, x, z, geom);
                let e = erodibility_at_strata_depth(strata, depth, *default_hardness);
                (1.0 - e).clamp(0.0, 1.0)
            }
        }
    }
}

fn hydraulic_erode_inner(
    input: &Heightfield,
    p: &HydraulicErosionParams,
    hardness: HardnessMode<'_>,
    rainfall: Option<&MaskField>,
    protection: Option<&MaskField>,
    initial_water: Option<&MaskField>,
    initial_sediment: Option<&MaskField>,
) -> HydraulicResult {
    let w = input.metrics.width as usize;
    let hh = input.metrics.height as usize;
    let dx = input.metrics.dx().max(1e-6);
    let mut height = input.to_dense();
    let mut water = initial_water
        .map(|field| field.data().to_vec())
        .unwrap_or_else(|| vec![0.0f32; w * hh]);
    let mut sediment = initial_sediment
        .map(|field| field.data().to_vec())
        .unwrap_or_else(|| vec![0.0f32; w * hh]);
    let mut erosion = vec![0.0f32; w * hh];
    let mut deposition = vec![0.0f32; w * hh];
    let mut wetness = vec![0.0f32; w * hh];

    // Outflow fluxes: L, R, D, U per cell (Mei pipe-model lite, atomic-free via snapshots).
    let neighbors = [(-1i32, 0i32), (1, 0), (0, 1), (0, -1)];

    for _ in 0..p.iterations {
        for (idx, v) in water.iter_mut().enumerate() {
            let rain = rainfall.map(|m| m.data()[idx]).unwrap_or(1.0).max(0.0);
            *v += p.rainfall * rain;
        }

        let h_snap = height.clone();
        let w_snap = water.clone();
        let s_snap = sediment.clone();

        let mut flux = vec![[0.0f32; 4]; w * hh];
        for j in 0..hh as i32 {
            for i in 0..w as i32 {
                let idx = j as usize * w + i as usize;
                let total = h_snap[idx] + w_snap[idx];
                let mut raw = [0.0f32; 4];
                let mut sum = 0.0f32;
                for (dir, &(di, dj)) in neighbors.iter().enumerate() {
                    let ni = i + di;
                    let nj = j + dj;
                    if ni < 0 || nj < 0 || ni >= w as i32 || nj >= hh as i32 {
                        continue;
                    }
                    let nidx = nj as usize * w + ni as usize;
                    let delta = total - (h_snap[nidx] + w_snap[nidx]);
                    if delta > 0.0 {
                        raw[dir] = delta;
                        sum += delta;
                    }
                }
                if sum <= 0.0 || w_snap[idx] <= 0.0 {
                    continue;
                }
                let scale = (sum * p.timestep).min(w_snap[idx]) / sum;
                for d in 0..4 {
                    flux[idx][d] = raw[d] * scale;
                }
            }
        }

        let mut water_next = w_snap.clone();
        let mut sed_next = s_snap.clone();
        for j in 0..hh as i32 {
            for i in 0..w as i32 {
                let idx = j as usize * w + i as usize;
                let out_sum: f32 = flux[idx].iter().sum();
                let mut inflow = 0.0f32;
                let mut sed_in = 0.0f32;
                if i > 0 {
                    let nidx = j as usize * w + (i - 1) as usize;
                    let f = flux[nidx][1];
                    inflow += f;
                    if w_snap[nidx] > 1e-8 {
                        sed_in += s_snap[nidx] * (f / w_snap[nidx]);
                    }
                }
                if i + 1 < w as i32 {
                    let nidx = j as usize * w + (i + 1) as usize;
                    let f = flux[nidx][0];
                    inflow += f;
                    if w_snap[nidx] > 1e-8 {
                        sed_in += s_snap[nidx] * (f / w_snap[nidx]);
                    }
                }
                if j + 1 < hh as i32 {
                    let nidx = (j + 1) as usize * w + i as usize;
                    let f = flux[nidx][3];
                    inflow += f;
                    if w_snap[nidx] > 1e-8 {
                        sed_in += s_snap[nidx] * (f / w_snap[nidx]);
                    }
                }
                if j > 0 {
                    let nidx = (j - 1) as usize * w + i as usize;
                    let f = flux[nidx][2];
                    inflow += f;
                    if w_snap[nidx] > 1e-8 {
                        sed_in += s_snap[nidx] * (f / w_snap[nidx]);
                    }
                }

                let sed_leave = if w_snap[idx] > 1e-8 {
                    s_snap[idx] * (out_sum / w_snap[idx]).min(1.0)
                } else {
                    0.0
                };
                water_next[idx] = (w_snap[idx] - out_sum + inflow).max(0.0);
                sed_next[idx] = (s_snap[idx] - sed_leave + sed_in).max(0.0);

                let flow = out_sum;
                let terrain_slope = terrain_slope_at(&h_snap, i, j, w, hh, dx);
                let upslope = max_neighbor_slope(&h_snap, i, j, w, hh, dx);
                let flat_t = (1.0 - (terrain_slope / (terrain_slope + 0.08)).clamp(0.0, 1.0))
                    .clamp(0.0, 1.0);
                // Fan only on flats below a steeper neighbor (steep→flat exit), not mid-slope.
                let break_t = if flat_t > 0.4 && upslope > terrain_slope + 0.05 {
                    ((upslope - terrain_slope) / (upslope + 1e-4)).clamp(0.0, 1.0) * flat_t
                } else {
                    0.0
                };
                let flow_proxy = flow + water_next[idx] * 0.35;
                let corridor = (flow_proxy / (flow_proxy + 0.04)).clamp(0.0, 1.0);
                let fan = 1.0 + p.fan_boost.max(0.0) * break_t;
                let flood = 1.0 + p.floodplain_bias.max(0.0) * flat_t * corridor;
                let dep_rate = (p.deposition * fan * flood).clamp(0.0, 2.5);

                // Capacity: flux slope + terrain slope; stagnant water keeps a tiny floor
                // so suspended load settles on flats (Beneš-style low-velocity deposit).
                // Slope-break / floodplain lower effective capacity → deposit sooner (fans).
                let slope_term = (flow * 0.25).max(terrain_slope * 0.5);
                let transport = flow.max(water_next[idx] * 0.08);
                let cap_scale = (1.0
                    + p.fan_boost.max(0.0) * break_t
                    + p.floodplain_bias.max(0.0) * flat_t * corridor * 0.5)
                    .max(1.0);
                let cap = p.capacity * slope_term * transport / cap_scale;

                let k = hardness.sample(i as u32, j as u32, height[idx]);
                let protected = protection
                    .map(|m| m.data()[idx])
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0)
                    * p.protection_strength.clamp(0.0, 1.0);
                let soft = (1.0 - k) * (1.0 - protected);
                let s = sed_next[idx];

                if transport > 1e-8 || s > 1e-8 {
                    if s < cap && flow > 1e-8 {
                        let erode_amt =
                            ((cap - s) * p.erosion * soft).min(height[idx].max(0.0) * 0.1);
                        height[idx] -= erode_amt;
                        sed_next[idx] += erode_amt;
                        erosion[idx] += erode_amt;
                    } else if s > cap {
                        let dep = ((s - cap) * dep_rate).min(s);
                        height[idx] += dep;
                        sed_next[idx] = (sed_next[idx] - dep).max(0.0);
                        deposition[idx] += dep;
                    } else if flow < 1e-6 && s > 1e-8 && flat_t > 0.35 {
                        // Stagnant pool settle — raises valley floors / fan toes.
                        let settle = s * p.deposition * (0.2 + 0.8 * flat_t) * (0.5 + 0.5 * flood);
                        let dep = settle.min(s);
                        height[idx] += dep;
                        sed_next[idx] = (sed_next[idx] - dep).max(0.0);
                        deposition[idx] += dep;
                    }
                }
            }
        }
        water = water_next;
        sediment = sed_next;

        for idx in 0..water.len() {
            water[idx] = (water[idx] * (1.0 - p.evaporation)).max(0.0);
            sediment[idx] = sediment[idx].max(0.0);
            wetness[idx] += water[idx];
        }
    }

    // Hydraulic mass totals (bank-slip is a separate mass-conserving redistribute).
    let erosion_raw = erosion.clone();
    let deposition_raw = deposition.clone();

    // Post-hydraulic bank slippage: soft talus redistributes undercut steep banks.
    if p.bank_slip > 0.0 {
        let slip = p.bank_slip.clamp(0.0, 1.0);
        let mut k_slip = MaskField::filled(input.metrics, 0.15);
        let soft_amt = p.sediment_softness.max(slip);
        if soft_amt > 0.0 {
            let dep_max = deposition.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
            for j in 0..hh {
                for i in 0..w {
                    let d = deposition[j * w + i] / dep_max;
                    // Deposited sediment is soft; uneroded rock keeps mild resistance.
                    let k = (0.55 * (1.0 - soft_amt * d)).clamp(0.0, 0.85);
                    k_slip.set(i as u32, j as u32, k);
                }
            }
        }
        let thermal_p = ThermalErosionParams {
            talus_angle_deg: (32.0 - 8.0 * slip).clamp(18.0, 35.0),
            iterations: ((slip * 14.0).round() as u32).clamp(2, 20),
            strength: (0.35 + 0.55 * slip).clamp(0.0, 1.0),
            hardness: 0.0,
            hardness_source: crate::mask::MaskSource::None,
            ..Default::default()
        };
        let hf = Heightfield::from_dense(input.metrics, &height);
        let (slipped, e_mask, d_mask) = thermal_erode_with_hardness(&hf, &thermal_p, &k_slip);
        height = slipped.to_dense();
        // Fold thermal bank motion into display masks only (not mass raws).
        let e_scale = e_mask
            .data()
            .iter()
            .cloned()
            .fold(0.0f32, f32::max)
            .max(1e-6);
        let d_scale = d_mask
            .data()
            .iter()
            .cloned()
            .fold(0.0f32, f32::max)
            .max(1e-6);
        for idx in 0..erosion.len() {
            erosion[idx] += e_mask.data()[idx] * e_scale * 0.15 * slip;
            deposition[idx] += d_mask.data()[idx] * d_scale * 0.15 * slip;
        }
    }

    let metrics = input.metrics;
    HydraulicResult {
        height: Heightfield::from_dense(metrics, &height),
        wetness: normalize_mask(&wetness, metrics),
        sediment: normalize_mask(&sediment, metrics),
        erosion: normalize_mask(&erosion, metrics),
        deposition: normalize_mask(&deposition, metrics),
        water_raw: MaskField::from_raw(metrics, &water),
        sediment_raw: MaskField::from_raw(metrics, &sediment),
        erosion_raw: MaskField::from_raw(metrics, &erosion_raw),
        deposition_raw: MaskField::from_raw(metrics, &deposition_raw),
    }
}

/// Soften hardness where sediment deposited; returns updated hardness map.
pub fn apply_sediment_softness(
    hardness: &MaskField,
    deposition_raw: &MaskField,
    sediment_softness: f32,
) -> MaskField {
    let soft = sediment_softness.clamp(0.0, 1.0);
    if soft <= 1e-6 {
        return hardness.clone();
    }
    let dep = deposition_raw.data();
    let max_d = dep.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    let mut out = hardness.clone();
    let w = hardness.metrics.width;
    for (idx, &d) in dep.iter().enumerate() {
        let x = (idx as u32) % w;
        let y = (idx as u32) / w;
        let t = (d / max_d).clamp(0.0, 1.0);
        let k0 = out.get(x, y);
        out.set(x, y, (k0 * (1.0 - soft * t)).clamp(0.0, 1.0));
    }
    out
}

/// Overlay soft sediment material IDs onto a materials ID field (values in \[0,1\] = id/16).
pub fn apply_sediment_material_ids(
    materials: &MaskField,
    deposition_raw: &MaskField,
    sediment_softness: f32,
) -> MaskField {
    let soft = sediment_softness.clamp(0.0, 1.0);
    if soft <= 1e-6 {
        return materials.clone();
    }
    let dep = deposition_raw.data();
    let max_d = dep.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    let sed_v = (SEDIMENT_MATERIAL_ID as f32) / 16.0;
    let mut out = materials.clone();
    let w = materials.metrics.width;
    for (idx, &d) in dep.iter().enumerate() {
        let t = (d / max_d).clamp(0.0, 1.0);
        if t * soft > 0.2 {
            let x = (idx as u32) % w;
            let y = (idx as u32) / w;
            out.set(x, y, sed_v);
        }
    }
    out
}

fn terrain_slope_at(h: &[f32], i: i32, j: i32, w: usize, hh: usize, dx: f32) -> f32 {
    let idx = j as usize * w + i as usize;
    let h0 = h[idx];
    let mut max_drop = 0.0f32;
    for &(di, dj) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
        let ni = i + di;
        let nj = j + dj;
        if ni < 0 || nj < 0 || ni >= w as i32 || nj >= hh as i32 {
            continue;
        }
        let drop = h0 - h[nj as usize * w + ni as usize];
        if drop > max_drop {
            max_drop = drop;
        }
    }
    (max_drop / dx).max(0.0)
}

fn max_neighbor_slope(h: &[f32], i: i32, j: i32, w: usize, hh: usize, dx: f32) -> f32 {
    let mut m = 0.0f32;
    for &(di, dj) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
        let ni = i + di;
        let nj = j + dj;
        if ni < 0 || nj < 0 || ni >= w as i32 || nj >= hh as i32 {
            continue;
        }
        m = m.max(terrain_slope_at(h, ni, nj, w, hh, dx));
    }
    m
}

/// Add deterministic inertial droplets after the broad shallow-water solve.
/// This pass resolves sub-basin rills and local sediment fans without replacing
/// the conservative water solver used for the large drainage structure.
pub fn apply_particle_erosion(
    mut result: HydraulicResult,
    p: &HydraulicErosionParams,
    hardness: &MaskField,
) -> HydraulicResult {
    let metrics = result.height.metrics;
    let w = metrics.width as usize;
    let h = metrics.height as usize;
    let density = p.particle_density.max(0.0);
    if density <= 0.0 || w < 4 || h < 4 {
        return result;
    }

    let droplet_count = ((w * h) as f32 * density).round() as usize;
    let droplet_count = droplet_count.clamp(1, 500_000);
    let lifetime = p.particle_lifetime.clamp(4, 128);
    let inertia = p.particle_inertia.clamp(0.0, 0.98);
    let radius = p.particle_radius.clamp(1, 8) as i32;
    let mut heights = result.height.to_dense();
    let mut eroded = vec![0.0f32; w * h];
    let mut deposited = vec![0.0f32; w * h];

    for droplet in 0..droplet_count {
        let seed = (droplet as u32)
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add(metrics.width.wrapping_mul(0x85EB_CA6B))
            .wrapping_add(metrics.height.wrapping_mul(0xC2B2_AE35));
        let mut x = 1.0 + particle_hash(seed) * (w as f32 - 3.0);
        let mut y = 1.0 + particle_hash(seed ^ 0xA511_E9B3) * (h as f32 - 3.0);
        let mut dir_x = 0.0f32;
        let mut dir_y = 0.0f32;
        let mut speed = 1.0f32;
        let mut water = 1.0f32;
        let mut sediment = 0.0f32;

        for _ in 0..lifetime {
            let (old_height, grad_x, grad_y) = sample_surface(&heights, w, h, x, y);
            dir_x = dir_x * inertia - grad_x * (1.0 - inertia);
            dir_y = dir_y * inertia - grad_y * (1.0 - inertia);
            let length = dir_x.hypot(dir_y);
            if length <= 1e-6 {
                break;
            }
            dir_x /= length;
            dir_y /= length;
            let next_x = x + dir_x;
            let next_y = y + dir_y;
            if next_x < 1.0 || next_y < 1.0 || next_x >= w as f32 - 2.0 || next_y >= h as f32 - 2.0
            {
                break;
            }

            let (new_height, _, _) = sample_surface(&heights, w, h, next_x, next_y);
            let delta_height = new_height - old_height;
            let capacity =
                ((-delta_height).max(0.01) * speed * water * p.capacity.max(0.01) * 4.0).max(0.001);

            if delta_height > 0.0 || sediment > capacity {
                let amount = if delta_height > 0.0 {
                    sediment.min(delta_height)
                } else {
                    (sediment - capacity) * p.deposition.clamp(0.0, 1.0)
                };
                let placed = add_bilinear(&mut heights, &mut deposited, w, h, x, y, amount);
                sediment -= placed;
            } else {
                let wanted = ((capacity - sediment) * p.erosion.clamp(0.0, 1.0) * 0.2)
                    .min((-delta_height).max(0.05));
                let removed = erode_particle_brush(
                    &mut heights,
                    &mut eroded,
                    hardness,
                    w,
                    h,
                    x,
                    y,
                    radius,
                    wanted,
                );
                sediment += removed;
            }

            speed = (speed * speed + (-delta_height) * 4.0).max(0.01).sqrt();
            water *= 1.0 - (p.evaporation * 0.5).clamp(0.001, 0.2);
            x = next_x;
            y = next_y;
            if water < 0.01 {
                break;
            }
        }

        if sediment > 0.0 {
            add_bilinear(&mut heights, &mut deposited, w, h, x, y, sediment);
        }
    }

    result.height = Heightfield::from_dense(metrics, &heights);
    for (dst, add) in result.erosion_raw.data_mut().iter_mut().zip(eroded.iter()) {
        *dst += *add;
    }
    for (dst, add) in result
        .deposition_raw
        .data_mut()
        .iter_mut()
        .zip(deposited.iter())
    {
        *dst += *add;
    }
    result.erosion = normalize_mask(result.erosion_raw.data(), metrics);
    result.deposition = normalize_mask(result.deposition_raw.data(), metrics);
    result
}

fn particle_hash(mut value: u32) -> f32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^= value >> 16;
    value as f32 / u32::MAX as f32
}

fn sample_surface(heights: &[f32], w: usize, h: usize, x: f32, y: f32) -> (f32, f32, f32) {
    let x0 = x.floor().clamp(0.0, (w - 2) as f32) as usize;
    let y0 = y.floor().clamp(0.0, (h - 2) as f32) as usize;
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let h00 = heights[y0 * w + x0];
    let h10 = heights[y0 * w + x0 + 1];
    let h01 = heights[(y0 + 1) * w + x0];
    let h11 = heights[(y0 + 1) * w + x0 + 1];
    let top = h00 + (h10 - h00) * tx;
    let bottom = h01 + (h11 - h01) * tx;
    let height = top + (bottom - top) * ty;
    let grad_x = (h10 - h00) * (1.0 - ty) + (h11 - h01) * ty;
    let grad_y = (h01 - h00) * (1.0 - tx) + (h11 - h10) * tx;
    (height, grad_x, grad_y)
}

fn add_bilinear(
    heights: &mut [f32],
    deposition: &mut [f32],
    w: usize,
    h: usize,
    x: f32,
    y: f32,
    amount: f32,
) -> f32 {
    if amount <= 0.0 {
        return 0.0;
    }
    let x0 = x.floor().clamp(0.0, (w - 2) as f32) as usize;
    let y0 = y.floor().clamp(0.0, (h - 2) as f32) as usize;
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let samples = [
        (x0, y0, (1.0 - tx) * (1.0 - ty)),
        (x0 + 1, y0, tx * (1.0 - ty)),
        (x0, y0 + 1, (1.0 - tx) * ty),
        (x0 + 1, y0 + 1, tx * ty),
    ];
    for (sx, sy, weight) in samples {
        let idx = sy * w + sx;
        let placed = amount * weight;
        heights[idx] += placed;
        deposition[idx] += placed;
    }
    amount
}

fn erode_particle_brush(
    heights: &mut [f32],
    erosion: &mut [f32],
    hardness: &MaskField,
    w: usize,
    h: usize,
    x: f32,
    y: f32,
    radius: i32,
    amount: f32,
) -> f32 {
    if amount <= 0.0 {
        return 0.0;
    }
    let cx = x.floor() as i32;
    let cy = y.floor() as i32;
    let mut total_weight = 0.0f32;
    for oy in -radius..=radius {
        for ox in -radius..=radius {
            let sx = cx + ox;
            let sy = cy + oy;
            if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
                continue;
            }
            let distance = (ox as f32).hypot(oy as f32);
            let radial = (1.0 - distance / (radius as f32 + 0.5)).max(0.0);
            let softness = 1.0 - hardness.get(sx as u32, sy as u32).clamp(0.0, 1.0);
            total_weight += radial * softness;
        }
    }
    if total_weight <= f32::EPSILON {
        return 0.0;
    }

    let mut removed = 0.0;
    for oy in -radius..=radius {
        for ox in -radius..=radius {
            let sx = cx + ox;
            let sy = cy + oy;
            if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
                continue;
            }
            let distance = (ox as f32).hypot(oy as f32);
            let radial = (1.0 - distance / (radius as f32 + 0.5)).max(0.0);
            let softness = 1.0 - hardness.get(sx as u32, sy as u32).clamp(0.0, 1.0);
            let cell_amount = amount * radial * softness / total_weight;
            let idx = sy as usize * w + sx as usize;
            heights[idx] -= cell_amount;
            erosion[idx] += cell_amount;
            removed += cell_amount;
        }
    }
    removed
}

fn normalize_mask(data: &[f32], metrics: crate::heightfield::HeightfieldMetrics) -> MaskField {
    let max_v = data.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    let mut m = MaskField::zeros(metrics);
    for (i, &v) in data.iter().enumerate() {
        let x = (i as u32) % metrics.width;
        let y = (i as u32) / metrics.width;
        m.set(x, y, (v / max_v).clamp(0.0, 1.0));
    }
    m
}

fn normalize_pair(
    a: &[f32],
    b: &[f32],
    metrics: crate::heightfield::HeightfieldMetrics,
) -> (MaskField, MaskField) {
    (normalize_mask(a, metrics), normalize_mask(b, metrics))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

    #[test]
    fn thermal_relaxes_steep_spike() {
        let m = HeightfieldMetrics::new(16, 16, 16.0, 16.0);
        let mut hf = Heightfield::filled(m, 10.0);
        hf.set(8, 8, 50.0);
        let p = ThermalErosionParams {
            talus_angle_deg: 25.0,
            iterations: 30,
            strength: 0.8,
            ..ThermalErosionParams::default()
        };
        let (out, _, _) = thermal_erode(&hf, &p);
        assert!(out.get(8, 8) < 50.0);
        assert!(out.get(8, 8) > 10.0);
    }

    #[test]
    fn thermal_flat_unchanged() {
        let m = HeightfieldMetrics::new(8, 8, 8.0, 8.0);
        let hf = Heightfield::filled(m, 5.0);
        let p = ThermalErosionParams::default();
        let (out, _, _) = thermal_erode(&hf, &p);
        for j in 0..8 {
            for i in 0..8 {
                assert!((out.get(i, j) - 5.0).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn thermal_hard_resists_more_than_soft() {
        let m = HeightfieldMetrics::new(16, 16, 16.0, 16.0);
        let mut hf = Heightfield::filled(m, 10.0);
        hf.set(8, 8, 50.0);
        let soft = ThermalErosionParams {
            talus_angle_deg: 25.0,
            iterations: 40,
            strength: 0.8,
            hardness: 0.0,
            ..ThermalErosionParams::default()
        };
        let hard = ThermalErosionParams {
            hardness: 1.0,
            ..soft.clone()
        };
        let (out_soft, _, _) = thermal_erode(&hf, &soft);
        let (out_hard, _, _) = thermal_erode(&hf, &hard);
        let soft_drop = 50.0 - out_soft.get(8, 8);
        let hard_drop = 50.0 - out_hard.get(8, 8);
        assert!(
            hard_drop < 1e-3,
            "K=1 should not move: hard drop {hard_drop}"
        );
        assert!(
            soft_drop > 5.0,
            "soft terrain should erode: soft drop {soft_drop}"
        );
    }

    #[test]
    fn hydraulic_finite() {
        let m = HeightfieldMetrics::new(24, 24, 48.0, 48.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..24 {
            for i in 0..24 {
                hf.set(i, j, (i + j) as f32 * 0.5);
            }
        }
        let r = hydraulic_erode(
            &hf,
            &HydraulicErosionParams {
                iterations: 10,
                ..HydraulicErosionParams::default()
            },
        );
        let d = r.height.to_dense();
        assert!(d.iter().all(|v| v.is_finite()));
        assert!(r.water_raw.data().iter().all(|&v| v >= -1e-5));
        assert!(r.sediment_raw.data().iter().all(|&v| v >= -1e-5));
    }

    #[test]
    fn hydraulic_transports_water_downslope() {
        let m = HeightfieldMetrics::new(8, 8, 8.0, 8.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..8 {
            for i in 0..8 {
                hf.set(i, j, (7 - i) as f32 * 2.0);
            }
        }
        let r = hydraulic_erode(
            &hf,
            &HydraulicErosionParams {
                iterations: 20,
                rainfall: 0.05,
                evaporation: 0.0,
                timestep: 0.5,
                erosion: 0.0,
                deposition: 0.0,
                ..HydraulicErosionParams::default()
            },
        );
        // Low side (i=7) should accumulate more water than high side (i=0).
        let mut high = 0.0f32;
        let mut low = 0.0f32;
        for j in 0..8 {
            high += r.water_raw.get(0, j);
            low += r.water_raw.get(7, j);
        }
        assert!(
            low > high,
            "expected downslope water gather: low={low} high={high}"
        );
    }

    #[test]
    fn hydraulic_mass_closed_without_bank_slip() {
        let m = HeightfieldMetrics::new(24, 24, 48.0, 48.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..24 {
            for i in 0..24 {
                hf.set(i, j, (24 - i) as f32 * 1.2 + (j as f32 * 0.1).sin());
            }
        }
        let r = hydraulic_erode(
            &hf,
            &HydraulicErosionParams {
                iterations: 25,
                rainfall: 0.03,
                evaporation: 0.01,
                bank_slip: 0.0,
                fan_boost: 0.5,
                floodplain_bias: 0.5,
                ..HydraulicErosionParams::default()
            },
        );
        let dh: f32 = r
            .height
            .to_dense()
            .iter()
            .zip(hf.to_dense().iter())
            .map(|(a, b)| a - b)
            .sum();
        let sed: f32 = r.sediment_raw.data().iter().sum();
        assert!(
            (dh + sed).abs() < 0.05 + 0.02 * sed.abs().max(1.0),
            "dh+sed residual {}",
            dh + sed
        );
    }

    #[test]
    fn particle_detail_is_deterministic_and_mass_conserving() {
        let m = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..32 {
            for i in 0..32 {
                let cross_slope = ((j as f32 * 0.7).sin() * 0.25).max(-0.2);
                hf.set(i, j, 80.0 - i as f32 * 1.4 + cross_slope);
            }
        }
        let p = HydraulicErosionParams {
            iterations: 1,
            rainfall: 0.0,
            erosion: 0.55,
            deposition: 0.45,
            particle_density: 0.2,
            particle_lifetime: 24,
            particle_radius: 2,
            ..HydraulicErosionParams::default()
        };
        let mut broad_p = p.clone();
        broad_p.particle_density = 0.0;
        let hardness = MaskField::filled(m, 0.0);
        let a = apply_particle_erosion(hydraulic_erode(&hf, &broad_p), &p, &hardness);
        let b = apply_particle_erosion(hydraulic_erode(&hf, &broad_p), &p, &hardness);
        assert_eq!(a.height.to_dense(), b.height.to_dense());
        let input_sum: f32 = hf.to_dense().iter().sum();
        let output_sum: f32 = a.height.to_dense().iter().sum();
        assert!((output_sum - input_sum).abs() < 0.05);
        let max_change = a
            .height
            .to_dense()
            .iter()
            .zip(hf.to_dense().iter())
            .map(|(after, before)| (after - before).abs())
            .fold(0.0f32, f32::max);
        assert!(max_change > 0.001);
    }

    #[test]
    fn particle_detail_respects_fully_hard_terrain() {
        let m = HeightfieldMetrics::new(24, 24, 240.0, 240.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..24 {
            for i in 0..24 {
                hf.set(i, j, 40.0 - i as f32);
            }
        }
        let p = HydraulicErosionParams {
            iterations: 1,
            rainfall: 0.0,
            particle_density: 0.25,
            ..HydraulicErosionParams::default()
        };
        let mut broad_p = p.clone();
        broad_p.particle_density = 0.0;
        let broad = hydraulic_erode(&hf, &broad_p);
        let expected = broad.height.to_dense();
        let detailed = apply_particle_erosion(broad, &p, &MaskField::filled(m, 1.0));
        assert_eq!(detailed.height.to_dense(), expected);
    }
}
