//! Mass-wasting: layered thermal/talus (Musgrave + Yang 2024) and debris flow (Jain et al. 2024).
//!
//! Bedrock and loose debris/sediment are tracked separately so Landscape Evolution →
//! Hydraulic → Debris Flow → Thermal stacks can exchange real sediment fields.

use crate::geomorph::{
    accumulate_drainage_area, build_flow_graph, priority_flood_fill, FlowModel, Precipitation,
};
use crate::heightfield::{Heightfield, HeightfieldMetrics};
use crate::layer::{DebrisFlowParams, ThermalErosionParams};
use crate::mask::MaskField;

/// Layered terrain state shared by thermal and debris-flow solvers.
#[derive(Debug, Clone)]
pub struct MassWastingState {
    pub metrics: HeightfieldMetrics,
    /// Immovable basement datum (meters, `<= 0`). The material layers above are
    /// non-negative thicknesses, so sub-zero (underwater) terrain would collapse
    /// to 0 without somewhere to live; `base` holds it. Surface is
    /// `base + bedrock + debris + sediment`, which can be negative.
    pub base: Vec<f32>,
    /// Hard substrate height (meters).
    pub bedrock: Vec<f32>,
    /// Loose debris / talus thickness (meters).
    pub debris: Vec<f32>,
    /// Fine sediment thickness (meters) — fluvial / mud / fill soft.
    pub sediment: Vec<f32>,
}

impl MassWastingState {
    pub fn from_height(input: &Heightfield, initial_debris: f32, initial_sediment: f32) -> Self {
        let metrics = input.metrics;
        let n = (metrics.width * metrics.height) as usize;
        let height = input.to_dense();
        let sed = initial_sediment.max(0.0);
        let deb = initial_debris.max(0.0);
        let mut base = Vec::with_capacity(n);
        let mut bedrock = Vec::with_capacity(n);
        let mut debris = vec![deb; n];
        let mut sediment = vec![sed; n];
        for (i, &h) in height.iter().enumerate() {
            // Sub-zero (underwater) terrain lives in the basement datum so the
            // material layers above stay non-negative and bathymetry survives.
            let b = h.min(0.0);
            base.push(b);
            bedrock.push((h - b - deb - sed).max(0.0));
            // Absorb overshoot into debris/sediment if stacked layers exceed height.
            let surface = base[i] + bedrock[i] + debris[i] + sediment[i];
            if surface > h + 1e-5 {
                let excess = surface - h;
                let take = excess.min(sediment[i]);
                sediment[i] -= take;
                let rem = excess - take;
                if rem > 0.0 {
                    let take_d = rem.min(debris[i]);
                    debris[i] -= take_d;
                }
            }
        }
        Self {
            metrics,
            base,
            bedrock,
            debris,
            sediment,
        }
    }

    pub fn from_layers(
        metrics: HeightfieldMetrics,
        bedrock: Vec<f32>,
        debris: Vec<f32>,
        sediment: Vec<f32>,
    ) -> Self {
        // Explicit layers already carry their own reference; no sub-zero datum.
        let base = vec![0.0; bedrock.len()];
        Self {
            metrics,
            base,
            bedrock,
            debris,
            sediment,
        }
    }

    pub fn surface_at(&self, idx: usize) -> f32 {
        self.base[idx] + self.bedrock[idx] + self.debris[idx] + self.sediment[idx]
    }

    pub fn sync_surface(&self) -> Vec<f32> {
        (0..self.bedrock.len())
            .map(|i| self.surface_at(i))
            .collect()
    }

    pub fn heightfield(&self) -> Heightfield {
        Heightfield::from_dense(self.metrics, &self.sync_surface())
    }

    /// Prefer prior bedrock / debris / sediment aux when present and sized correctly.
    /// The residual datum is reconstructed from the authoritative input surface so
    /// supplied material inventories survive unchanged, including below sea level.
    pub fn with_optional_layers(
        input: &Heightfield,
        bedrock: Option<&MaskField>,
        debris: Option<&MaskField>,
        sediment: Option<&MaskField>,
        default_debris: f32,
        default_sediment: f32,
    ) -> Self {
        let mut state = Self::from_height(input, default_debris, default_sediment);
        let n = state.bedrock.len();
        let same_metrics = |field: &&MaskField| {
            field.metrics.width == input.metrics.width
                && field.metrics.height == input.metrics.height
                && field.data().len() == n
        };
        let bedrock = bedrock.filter(same_metrics);
        let debris = debris.filter(same_metrics);
        let sediment = sediment.filter(same_metrics);
        let has_bedrock = bedrock.is_some();
        if let Some(b) = bedrock {
            state.bedrock = b.data().to_vec();
        }
        if let Some(d) = debris {
            state.debris = d.data().to_vec();
        }
        if let Some(s) = sediment {
            state.sediment = s.data().to_vec();
        }
        // If only height was authoritative, keep from_height's default-layer policy.
        if bedrock.is_none() && debris.is_none() && sediment.is_none() {
            return Self::from_height(input, default_debris, default_sediment);
        }
        // Preserve supplied bedrock exactly. If no bedrock inventory was supplied,
        // derive it from the authoritative surface after applying the soft layers.
        // Any residual datum needed below sea level lives in `base`.
        let h = input.to_dense();
        for i in 0..n {
            if !has_bedrock {
                state.base[i] = h[i].min(0.0);
                state.bedrock[i] =
                    (h[i] - state.base[i] - state.debris[i] - state.sediment[i]).max(0.0);
            }
            state.base[i] = h[i] - state.bedrock[i] - state.debris[i] - state.sediment[i];
        }
        state
    }
}

/// Outputs from layered thermal / talus.
#[derive(Debug, Clone)]
pub struct ThermalResult {
    pub height: Heightfield,
    pub bedrock: MaskField,
    pub loose_debris: MaskField,
    pub sediment: MaskField,
    pub erosion: MaskField,
    pub deposition: MaskField,
    pub talus_stability: MaskField,
    pub instability: MaskField,
    pub erosion_raw: MaskField,
    pub deposition_raw: MaskField,
}

/// Conservation diagnostics for a debris-flow solve.
///
/// Sums are height-in-cell units. Multiplying by the uniform cell area gives volume.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DebrisFlowMassLedger {
    pub initial_surface_sum: f64,
    pub eroded_sum: f64,
    pub mobilized_sum: f64,
    pub deposited_sum: f64,
    pub in_flight_sum: f64,
    pub exported_sum: f64,
    pub final_surface_sum: f64,
    /// `final + in_flight + exported - initial`.
    pub residual: f64,
}

/// Outputs from Jain et al. 2024 debris-flow erosion.
#[derive(Debug, Clone)]
pub struct DebrisFlowResult {
    pub height: Heightfield,
    pub bedrock: MaskField,
    pub debris: MaskField,
    pub sediment: MaskField,
    pub erosion: MaskField,
    pub deposition: MaskField,
    pub debris_erosion: MaskField,
    pub debris_deposition: MaskField,
    pub slide_path: MaskField,
    pub instability: MaskField,
    pub flow_accumulation: MaskField,
    pub erosion_raw: MaskField,
    pub deposition_raw: MaskField,
    pub mass_ledger: DebrisFlowMassLedger,
}

fn neighbors_8(dx: f32, dz: f32) -> [(i32, i32, f32); 8] {
    let diag = dx.hypot(dz);
    [
        (-1, 0, dx),
        (1, 0, dx),
        (0, -1, dz),
        (0, 1, dz),
        (-1, -1, diag),
        (1, -1, diag),
        (-1, 1, diag),
        (1, 1, diag),
    ]
}

fn normalize_mask(data: &[f32], metrics: HeightfieldMetrics) -> MaskField {
    let max_v = data.iter().copied().fold(0.0f32, f32::max);
    if max_v <= 1e-12 {
        return MaskField::zeros(metrics);
    }
    let scaled: Vec<f32> = data.iter().map(|v| (v / max_v).clamp(0.0, 1.0)).collect();
    MaskField::from_raw(metrics, &scaled)
}

fn raw_mask(data: &[f32], metrics: HeightfieldMetrics) -> MaskField {
    MaskField::from_raw(metrics, data)
}

fn local_slope_deg(hf: &Heightfield, i: u32, j: u32) -> f32 {
    let dx = hf.metrics.dx().max(1e-5);
    let dz = hf.metrics.dz().max(1e-5);
    let z = hf.get(i, j);
    let zx = if i + 1 < hf.metrics.width {
        hf.get(i + 1, j)
    } else {
        z
    };
    let zy = if j + 1 < hf.metrics.height {
        hf.get(i, j + 1)
    } else {
        z
    };
    let gx = (zx - z) / dx;
    let gz = (zy - z) / dz;
    (gx * gx + gz * gz).sqrt().atan().to_degrees()
}

fn bilateral_settle(input: &Heightfield, radius: u32, sigma_range: f32) -> Heightfield {
    let r = radius.max(1) as i32;
    let w = input.metrics.width as i32;
    let h = input.metrics.height as i32;
    let mut out = input.clone();
    let inv_r2 = 1.0 / ((r * r) as f32).max(1.0);
    let inv_s2 = 1.0 / (sigma_range * sigma_range).max(1e-4);
    for j in 0..h {
        for i in 0..w {
            let c = input.get(i as u32, j as u32);
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for dj in -r..=r {
                for di in -r..=r {
                    let x = (i + di).clamp(0, w - 1) as u32;
                    let y = (j + dj).clamp(0, h - 1) as u32;
                    let v = input.get(x, y);
                    let spat = ((di * di + dj * dj) as f32) * inv_r2;
                    let rang = (v - c) * (v - c) * inv_s2;
                    let wt = (-spat - rang).exp();
                    num += v * wt;
                    den += wt;
                }
            }
            out.set(i as u32, j as u32, num / den.max(1e-6));
        }
    }
    out
}

/// Layered thermal erosion (classical talus + Yang 2024 weathering / rock discharge).
///
/// 1. Weather bedrock → loose debris where slope exceeds the talus angle (`weathering_rate`).
/// 2. Transport loose downhill until slopes are stable, limited by `transport_distance`
///    and `material_amount` — never Laplacian blur.
/// 3. Hard bedrock resists (hardness \(K\)); deposited debris forms talus aprons.
pub fn thermal_erode_layered(
    input: &Heightfield,
    p: &ThermalErosionParams,
    hardness: &MaskField,
    initial: Option<&MassWastingState>,
) -> ThermalResult {
    let metrics = input.metrics;
    let w = metrics.width as usize;
    let hh = metrics.height as usize;
    let n = w * hh;
    let dx = metrics.dx();
    let dz = metrics.dz();
    let neighbors = neighbors_8(dx, dz);
    let talus_slope = p.talus_angle_deg.to_radians().tan();
    let strength = p.strength.clamp(0.0, 1.0);
    let weathering = p.weathering_rate.max(0.0);
    let material_cap = p.material_amount.max(0.0);
    let transport_hops = p.transport_distance.max(1.0).round() as u32;

    let mut state = if let Some(s) = initial {
        s.clone()
    } else {
        MassWastingState::from_height(input, 0.0, 0.0)
    };

    let mut erosion = vec![0.0f32; n];
    let mut deposit = vec![0.0f32; n];
    let mut instability = vec![0.0f32; n];

    for _ in 0..p.iterations.max(1) {
        // --- Weathering: bedrock → debris (Yang K_th excess slope) ---
        let surface = state.sync_surface();
        for j in 0..hh as i32 {
            for i in 0..w as i32 {
                let idx = j as usize * w + i as usize;
                let k = hardness.get(i as u32, j as u32).clamp(0.0, 1.0);
                let soft = 1.0 - k;
                if soft <= 1e-6 || weathering <= 1e-8 {
                    continue;
                }
                let h0 = surface[idx];
                let mut max_excess = 0.0f32;
                for &(di, dj, dist) in &neighbors {
                    let ni = i + di;
                    let nj = j + dj;
                    if ni < 0 || nj < 0 || ni >= w as i32 || nj >= hh as i32 {
                        continue;
                    }
                    let nidx = nj as usize * w + ni as usize;
                    let excess = h0 - surface[nidx] - talus_slope * dist;
                    if excess > max_excess {
                        max_excess = excess;
                    }
                }
                if max_excess <= 1e-6 {
                    continue;
                }
                instability[idx] = instability[idx].max(max_excess);
                // Detach amount: Yang-style K_th * excess, capped by material + bedrock.
                let detach = (weathering * soft * max_excess * strength * 0.25)
                    .min(material_cap)
                    .min(state.bedrock[idx].max(0.0));
                if detach <= 1e-8 {
                    continue;
                }
                state.bedrock[idx] -= detach;
                state.debris[idx] += detach;
                erosion[idx] += detach;
            }
        }

        // --- Transport: move loose debris downhill (Musgrave redistribute on debris only) ---
        for _hop in 0..transport_hops.max(1) {
            let surface = state.sync_surface();
            let debris_src = state.debris.clone();
            let mut delta_debris = vec![0.0f32; n];
            for j in 0..hh as i32 {
                for i in 0..w as i32 {
                    let idx = j as usize * w + i as usize;
                    let available = debris_src[idx];
                    if available <= 1e-8 {
                        continue;
                    }
                    let h0 = surface[idx];
                    let mut deltas = Vec::with_capacity(8);
                    let mut sum = 0.0f32;
                    for &(di, dj, dist) in &neighbors {
                        let ni = i + di;
                        let nj = j + dj;
                        if ni < 0 || nj < 0 || ni >= w as i32 || nj >= hh as i32 {
                            continue;
                        }
                        let nidx = nj as usize * w + ni as usize;
                        let diff = h0 - surface[nidx] - talus_slope * dist;
                        if diff > 0.0 {
                            deltas.push((nidx, diff));
                            sum += diff;
                        }
                    }
                    if sum <= 0.0 {
                        continue;
                    }
                    let move_amt = (sum * strength * 0.125)
                        .min(available)
                        .min(material_cap.max(available));
                    if move_amt <= 1e-8 {
                        continue;
                    }
                    delta_debris[idx] -= move_amt;
                    erosion[idx] += move_amt * 0.15; // scar contribution from remobilised debris
                    for (nidx, diff) in deltas {
                        let share = move_amt * (diff / sum);
                        delta_debris[nidx] += share;
                        deposit[nidx] += share;
                    }
                }
            }
            for i in 0..n {
                state.debris[i] = (debris_src[i] + delta_debris[i]).max(0.0);
            }
        }

        // --- Classical fallback: if still over-steep with no debris, peel thin bedrock ---
        // Preserves Musgrave behaviour on soft rock when weathering alone is insufficient.
        {
            let surface = state.sync_surface();
            let mut delta_b = vec![0.0f32; n];
            let mut delta_d = vec![0.0f32; n];
            for j in 0..hh as i32 {
                for i in 0..w as i32 {
                    let idx = j as usize * w + i as usize;
                    if state.debris[idx] > 1e-5 {
                        continue;
                    }
                    let k = hardness.get(i as u32, j as u32).clamp(0.0, 1.0);
                    let soft = 1.0 - k;
                    if soft <= 1e-6 {
                        continue;
                    }
                    let h0 = surface[idx];
                    let mut deltas = Vec::with_capacity(8);
                    let mut sum = 0.0f32;
                    for &(di, dj, dist) in &neighbors {
                        let ni = i + di;
                        let nj = j + dj;
                        if ni < 0 || nj < 0 || ni >= w as i32 || nj >= hh as i32 {
                            continue;
                        }
                        let nidx = nj as usize * w + ni as usize;
                        let diff = h0 - surface[nidx] - talus_slope * dist;
                        if diff > 0.0 {
                            deltas.push((nidx, diff));
                            sum += diff;
                        }
                    }
                    if sum <= 0.0 {
                        continue;
                    }
                    let move_amt = (sum * strength * 0.125 * soft)
                        .min(state.bedrock[idx])
                        .min(material_cap.max(sum));
                    if move_amt <= 1e-8 {
                        continue;
                    }
                    delta_b[idx] -= move_amt;
                    erosion[idx] += move_amt;
                    for (nidx, diff) in deltas {
                        let share = move_amt * (diff / sum);
                        delta_d[nidx] += share;
                        deposit[nidx] += share;
                    }
                }
            }
            for i in 0..n {
                state.bedrock[i] = (state.bedrock[i] + delta_b[i]).max(0.0);
                state.debris[i] = (state.debris[i] + delta_d[i]).max(0.0);
            }
        }
    }

    let surface = state.sync_surface();
    let mut stability = vec![1.0f32; n];
    for j in 0..hh as i32 {
        for i in 0..w as i32 {
            let idx = j as usize * w + i as usize;
            let h0 = surface[idx];
            let mut max_excess = 0.0f32;
            for &(di, dj, dist) in &neighbors {
                let ni = i + di;
                let nj = j + dj;
                if ni < 0 || nj < 0 || ni >= w as i32 || nj >= hh as i32 {
                    continue;
                }
                let nidx = nj as usize * w + ni as usize;
                let excess = h0 - surface[nidx] - talus_slope * dist;
                if excess > max_excess {
                    max_excess = excess;
                }
            }
            // 1 = fully stable; 0 = strongly over-steep.
            stability[idx] =
                (1.0 - (max_excess / (dx.max(dz) * 4.0)).clamp(0.0, 1.0)).clamp(0.0, 1.0);
            if max_excess > instability[idx] {
                instability[idx] = max_excess;
            }
        }
    }

    let (e_mask, d_mask) = (
        normalize_mask(&erosion, metrics),
        normalize_mask(&deposit, metrics),
    );
    ThermalResult {
        height: Heightfield::from_dense(metrics, &surface),
        bedrock: raw_mask(&state.bedrock, metrics),
        loose_debris: raw_mask(&state.debris, metrics),
        sediment: raw_mask(&state.sediment, metrics),
        erosion: e_mask,
        deposition: d_mask,
        talus_stability: raw_mask(&stability, metrics),
        instability: normalize_mask(&instability, metrics),
        erosion_raw: raw_mask(&erosion, metrics),
        deposition_raw: raw_mask(&deposit, metrics),
    }
}

/// Convenience wrapper matching the classical thermal API.
pub fn thermal_erode_mass(
    input: &Heightfield,
    p: &ThermalErosionParams,
    hardness: &MaskField,
) -> (Heightfield, MaskField, MaskField) {
    let r = thermal_erode_layered(input, p, hardness, None);
    (r.height, r.erosion, r.deposition)
}

/// Artist talus apron: collect debris beneath cliffs, preserve hard bedrock above.
pub fn talus_apron(
    input: &Heightfield,
    talus_angle_deg: f32,
    material_amount: f32,
    weathering_rate: f32,
    iterations: u32,
    hardness: Option<&MaskField>,
) -> ThermalResult {
    let p = ThermalErosionParams {
        talus_angle_deg,
        iterations: iterations.max(1),
        strength: 0.65,
        hardness: 0.15,
        material_amount,
        weathering_rate,
        transport_distance: 3.0,
        ..ThermalErosionParams::default()
    };
    let k = hardness
        .cloned()
        .unwrap_or_else(|| MaskField::filled(input.metrics, p.hardness));
    thermal_erode_layered(input, &p, &k, None)
}

fn slope_magnitude(
    surface: &[f32],
    w: usize,
    h: usize,
    i: usize,
    j: usize,
    dx: f32,
    dz: f32,
) -> f32 {
    let idx = j * w + i;
    let h0 = surface[idx];
    let hx = if i + 1 < w {
        surface[idx + 1]
    } else if i > 0 {
        surface[idx - 1]
    } else {
        h0
    };
    let hz = if j + 1 < h {
        surface[idx + w]
    } else if j > 0 {
        surface[idx - w]
    } else {
        h0
    };
    let gx = (hx - h0) / dx.max(1e-5);
    let gz = (hz - h0) / dz.max(1e-5);
    (gx * gx + gz * gz).sqrt()
}

/// Rotation-invariant upwind-ish slope (Jain Eq. 18 approximation).
fn jain_slope(surface: &[f32], w: usize, h: usize, i: usize, j: usize, dx: f32, dz: f32) -> f32 {
    let idx = j * w + i;
    let h0 = surface[idx];
    let mut sx = 0.0f32;
    let mut sz = 0.0f32;
    if i + 1 < w {
        sx += (h0 - surface[idx + 1]).max(0.0) / dx;
    }
    if i > 0 {
        sx += (h0 - surface[idx - 1]).max(0.0) / dx;
    }
    if j + 1 < h {
        sz += (h0 - surface[idx + w]).max(0.0) / dz;
    }
    if j > 0 {
        sz += (h0 - surface[idx - w]).max(0.0) / dz;
    }
    (sx * sx + sz * sz).sqrt()
}

/// Pick a stochastic single-flow receiver weighted by drop (Jain §5).
fn pick_receiver(
    surface: &[f32],
    w: usize,
    h: usize,
    i: usize,
    j: usize,
    seed: u64,
) -> Option<usize> {
    let idx = j * w + i;
    let h0 = surface[idx];
    let dirs = [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)];
    let mut candidates = [(0usize, 0.0f32); 4];
    let mut n = 0usize;
    let mut total = 0.0f32;
    for &(di, dj) in &dirs {
        let ni = i as i32 + di;
        let nj = j as i32 + dj;
        if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
            continue;
        }
        let nidx = nj as usize * w + ni as usize;
        let drop = h0 - surface[nidx];
        if drop > 1e-6 {
            candidates[n] = (nidx, drop);
            total += drop;
            n += 1;
        }
    }
    if n == 0 || total <= 0.0 {
        return None;
    }
    // Deterministic hash "random" pick.
    let hash = seed
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((idx as u64).wrapping_mul(0xBF58476D1CE4E5B9));
    // Use the top 24 hash bits so the integer is exactly representable as f32;
    // scaling by 2^-24 produces a deterministic value in [0, 1).
    let unit = (hash >> 40) as f32 * (1.0 / (1u32 << 24) as f32);
    let r = unit * total;
    let mut acc = 0.0f32;
    for k in 0..n {
        acc += candidates[k].1;
        if r <= acc {
            return Some(candidates[k].0);
        }
    }
    Some(candidates[n - 1].0)
}

/// Jain et al. 2024 debris-flow erosion / deposition (CPU oracle).
///
/// D = −E_th − E_abr + D_dep with layered bedrock + debris + sediment,
/// coupled fluvial stream-power deposition, and drainage refresh after
/// substantial debris deposition (depression refill).
pub fn debris_flow_erode(
    input: &Heightfield,
    p: &DebrisFlowParams,
    hardness: &MaskField,
    initial: Option<&MassWastingState>,
) -> DebrisFlowResult {
    let metrics = input.metrics;
    let w = metrics.width as usize;
    let hh = metrics.height as usize;
    let n = w * hh;
    let dx = metrics.dx().max(1e-5);
    let dz = metrics.dz().max(1e-5);
    let cell_area = dx * dz;
    let talus = p.talus_angle_deg.to_radians().tan();
    let dt = p.dt.max(1e-4);

    let mut state = if let Some(s) = initial {
        s.clone()
    } else {
        MassWastingState::from_height(
            input,
            p.initial_debris_thickness,
            p.initial_sediment_thickness,
        )
    };
    let initial_surface_sum: f64 = state.sync_surface().into_iter().map(f64::from).sum();

    let mut erosion = vec![0.0f32; n];
    let mut deposit = vec![0.0f32; n];
    let mut debris_erosion = vec![0.0f32; n];
    let mut debris_deposit = vec![0.0f32; n];
    let mut slide_path = vec![0.0f32; n];
    let mut instability = vec![0.0f32; n];
    let mut mobilized_sum = 0.0f64;
    let mut last_acc = MaskField::zeros(metrics);

    let drain_stride = p.drainage_reuse_stride.max(1);

    for iter in 0..p.iterations.max(1) {
        let refresh = iter == 0 || iter % drain_stride == 0;
        let surface_hf = state.heightfield();

        // After deposition obstructs paths, refill depressions so fluvial routing continues.
        if refresh || p.refill_depressions {
            let filled = priority_flood_fill(&surface_hf);
            // Route on filled surface so deposition-obstructed rivers can spill.
            let graph = build_flow_graph(&filled, FlowModel::D8);
            let acc = accumulate_drainage_area(&graph, &Precipitation::uniform(1.0));
            last_acc = MaskField::from_raw(metrics, &acc);
            let surface = state.sync_surface();
            let mut discharge = acc.clone();
            for v in &mut discharge {
                *v *= p.precipitation * cell_area;
            }

            // Build receiver map with Jain random single-flow on actual surface.
            let mut receiver = vec![None; n];
            let seed = p.seed.wrapping_add(iter as u64 * 17);
            for j in 0..hh {
                for i in 0..w {
                    let idx = j * w + i;
                    receiver[idx] = pick_receiver(&surface, w, hh, i, j, seed);
                }
            }

            // Strictly downhill receivers make descending surface order a transport DAG.
            // Each cell consumes deposition from its live load and forwards only the
            // remainder, so cumulative upstream material can be deposited only once.
            let mut q_debris = vec![0.0f32; n];
            let mut q_sed = vec![0.0f32; n];
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| {
                surface[b]
                    .partial_cmp(&surface[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut d_bedrock = vec![0.0f32; n];
            let mut d_debris = vec![0.0f32; n];
            let mut d_sediment = vec![0.0f32; n];
            let height_to_flux = cell_area / dt;
            let flux_to_height = dt / cell_area;

            for &idx in &order {
                let i = idx % w;
                let j = idx / w;
                let slope = jain_slope(&surface, w, hh, i, j, dx, dz);
                let k = hardness.get(i as u32, j as u32).clamp(0.0, 1.0);
                let soft = (1.0 - k).clamp(0.0, 1.0);
                let q = discharge[idx].max(0.0);
                let incoming_debris = q_debris[idx].max(0.0);
                let incoming_sediment = q_sed[idx].max(0.0);

                // Thermal trigger (Jain Eq. 8): E_th = k_th (S - tan theta)+.
                let excess = (slope - talus).max(0.0);
                instability[idx] = instability[idx].max(excess);
                let e_th = p.thermal_k * excess * soft;

                // Abrasion (Jain Eq. 12-13), driven by material arriving upstream.
                let theta = if slope > 1e-6 {
                    let thresh = p.yield_stress
                        * incoming_debris.powf(-p.threshold_exp_q)
                        * slope.powf(-p.threshold_exp_s);
                    (1.0 - thresh).max(0.0)
                } else {
                    0.0
                };
                let e_abr = if incoming_debris > 1e-8 && theta > 0.0 {
                    p.abrasion_k
                        * incoming_debris.powf(p.abrasion_exp_q)
                        * slope.powf(p.abrasion_exp_s)
                        * theta
                        * soft
                } else {
                    0.0
                };

                let d_dep = if theta <= 1e-4 && incoming_debris > 1e-8 {
                    (incoming_debris * flux_to_height).min(p.max_deposit_per_step)
                } else if theta < 0.35 && incoming_debris > 1e-8 {
                    ((1.0 - theta) * incoming_debris * flux_to_height * 0.5)
                        .min(p.max_deposit_per_step)
                } else {
                    0.0
                };

                // Fluvial SPE + deposition (Jain Eq. 1), acting mostly on sediment.
                let e_fluv = if p.fluvial_k > 0.0 && q > 1e-8 {
                    p.fluvial_k * q.powf(p.fluvial_m) * slope.powf(p.fluvial_n) * soft
                } else {
                    0.0
                };
                let d_fluv = if p.fluvial_deposition > 0.0 && incoming_sediment > 1e-8 {
                    (p.fluvial_deposition * incoming_sediment * flux_to_height)
                        * (1.0 - (slope / talus.max(1e-3)).clamp(0.0, 1.0))
                } else {
                    0.0
                };

                // Hillslope Laplacian creep (Jain Eq. 4) - light, optional.
                let e_hill = if p.hillslope_k > 0.0 {
                    let mut lap = 0.0f32;
                    let mut count = 0.0f32;
                    for &(di, dj) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                        let ni = i as i32 + di;
                        let nj = j as i32 + dj;
                        if ni < 0 || nj < 0 || ni >= w as i32 || nj >= hh as i32 {
                            continue;
                        }
                        lap += surface[nj as usize * w + ni as usize] - surface[idx];
                        count += 1.0;
                    }
                    if count > 0.0 {
                        -p.hillslope_k * (lap / count)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let erode_total =
                    ((e_th + e_abr) * dt + e_fluv * dt + e_hill.max(0.0) * dt).max(0.0);
                let mut remaining = erode_total;

                // Erode soft layers first (sediment, debris, then bedrock).
                let take_sed = remaining.min(state.sediment[idx]);
                d_sediment[idx] -= take_sed;
                remaining -= take_sed;
                let take_deb = remaining.min(state.debris[idx]);
                d_debris[idx] -= take_deb;
                remaining -= take_deb;
                let take_bed = remaining.min(state.bedrock[idx]);
                d_bedrock[idx] -= take_bed;

                let eroded = take_sed + take_deb + take_bed;
                erosion[idx] += eroded;
                debris_erosion[idx] += take_bed + take_deb;

                // Process-rate terms already contributed to `erode_total`. Only actual,
                // clamped layer removal enters the transport ledger.
                let mobilized_debris = take_bed + take_deb;
                let mobilized_sediment = take_sed;
                mobilized_sum += f64::from(mobilized_debris + mobilized_sediment);
                let mut debris_load = incoming_debris + mobilized_debris * height_to_flux;
                let mut sediment_load = incoming_sediment + mobilized_sediment * height_to_flux;

                if d_dep > 0.0 {
                    let settled = (debris_load * flux_to_height).min(d_dep);
                    d_debris[idx] += settled;
                    deposit[idx] += settled;
                    debris_deposit[idx] += settled;
                    debris_load = (debris_load - settled * height_to_flux).max(0.0);
                }
                if d_fluv > 0.0 {
                    let settled = (sediment_load * flux_to_height).min(d_fluv);
                    d_sediment[idx] += settled;
                    deposit[idx] += settled;
                    sediment_load = (sediment_load - settled * height_to_flux).max(0.0);
                }

                // Closed boundary: route only the remainder, and settle all terminal load.
                q_debris[idx] = 0.0;
                q_sed[idx] = 0.0;
                if let Some(r) = receiver[idx] {
                    q_debris[r] += debris_load;
                    q_sed[r] += sediment_load;
                    slide_path[r] += debris_load * 0.01;
                    slide_path[idx] += debris_load * 0.02;
                } else {
                    let terminal_debris = debris_load * flux_to_height;
                    let terminal_sediment = sediment_load * flux_to_height;
                    d_debris[idx] += terminal_debris;
                    d_sediment[idx] += terminal_sediment;
                    deposit[idx] += terminal_debris + terminal_sediment;
                    debris_deposit[idx] += terminal_debris;
                }
            }

            debug_assert!(q_debris.iter().all(|v| v.abs() <= 1e-6));
            debug_assert!(q_sed.iter().all(|v| v.abs() <= 1e-6));
            for i in 0..n {
                state.bedrock[i] = (state.bedrock[i] + d_bedrock[i]).max(0.0);
                state.debris[i] = (state.debris[i] + d_debris[i]).max(0.0);
                state.sediment[i] = (state.sediment[i] + d_sediment[i]).max(0.0);
            }
        } else {
            // Lightweight hop: reuse last accumulation without full refill.
            let surface = state.sync_surface();
            let acc = last_acc.data();
            for j in 0..hh {
                for i in 0..w {
                    let idx = j * w + i;
                    let slope = slope_magnitude(&surface, w, hh, i, j, dx, dz);
                    let excess = (slope - talus).max(0.0);
                    if excess <= 1e-6 {
                        continue;
                    }
                    let soft = 1.0 - hardness.get(i as u32, j as u32).clamp(0.0, 1.0);
                    let e = p.thermal_k * excess * soft * dt * 0.5;
                    let take = e.min(state.bedrock[idx]);
                    state.bedrock[idx] -= take;
                    state.debris[idx] += take;
                    erosion[idx] += take;
                    debris_erosion[idx] += take;
                    let a = acc.get(idx).copied().unwrap_or(1.0);
                    slide_path[idx] += a.sqrt() * excess * 0.01;
                }
            }
        }
    }

    let surface = state.sync_surface();
    let final_surface_sum: f64 = surface.iter().copied().map(f64::from).sum();
    let eroded_sum: f64 = erosion.iter().copied().map(f64::from).sum();
    let deposited_sum: f64 = deposit.iter().copied().map(f64::from).sum();
    let mass_ledger = DebrisFlowMassLedger {
        initial_surface_sum,
        eroded_sum,
        mobilized_sum,
        deposited_sum,
        in_flight_sum: 0.0,
        exported_sum: 0.0,
        final_surface_sum,
        residual: final_surface_sum - initial_surface_sum,
    };
    DebrisFlowResult {
        height: Heightfield::from_dense(metrics, &surface),
        bedrock: raw_mask(&state.bedrock, metrics),
        debris: raw_mask(&state.debris, metrics),
        sediment: raw_mask(&state.sediment, metrics),
        erosion: normalize_mask(&erosion, metrics),
        deposition: normalize_mask(&deposit, metrics),
        debris_erosion: normalize_mask(&debris_erosion, metrics),
        debris_deposition: normalize_mask(&debris_deposit, metrics),
        slide_path: normalize_mask(&slide_path, metrics),
        instability: normalize_mask(&instability, metrics),
        flow_accumulation: last_acc,
        erosion_raw: raw_mask(&erosion, metrics),
        deposition_raw: raw_mask(&deposit, metrics),
        mass_ledger,
    }
}

/// Soft fill into concave / low-energy zones using slope, curvature, flow, and sediment budget.
pub fn sediment_fill_soft_mass(
    input: &Heightfield,
    amount: f32,
    slope_max_deg: f32,
    radius: u32,
    sediment_budget: Option<&MaskField>,
    flow: Option<&[f32]>,
) -> (Heightfield, MaskField) {
    let metrics = input.metrics;
    let w = metrics.width as i32;
    let h = metrics.height as i32;
    let r = radius.max(1) as i32;
    let mut out = input.clone();
    let mut used = vec![0.0f32; (metrics.width * metrics.height) as usize];
    let ww = metrics.width as usize;

    for j in 0..h {
        for i in 0..w {
            let slope = local_slope_deg(input, i as u32, j as u32);
            let mut sum = 0.0f32;
            let mut count = 0.0f32;
            for dj in -r..=r {
                for di in -r..=r {
                    let x = (i + di).clamp(0, w - 1) as u32;
                    let y = (j + dj).clamp(0, h - 1) as u32;
                    sum += input.get(x, y);
                    count += 1.0;
                }
            }
            let mean = sum / count.max(1.0);
            let h0 = input.get(i as u32, j as u32);
            let concavity = (mean - h0).max(0.0);
            if concavity <= 1e-6 {
                continue;
            }
            // Slope weights deposition energy; deep sinks still fill even if walls are steep.
            let slope_w = if slope > slope_max_deg {
                (1.0 - ((slope - slope_max_deg) / slope_max_deg.max(1.0)).clamp(0.0, 1.0)) * 0.35
            } else {
                1.0 - (slope / slope_max_deg.max(1.0)).clamp(0.0, 1.0) * 0.5
            };
            let flow_e = flow
                .map(|f| {
                    let v = f[j as usize * ww + i as usize];
                    let t = (v / (v + 8.0)).clamp(0.0, 1.0);
                    (1.0 - (t - 0.35).abs() * 2.0).clamp(0.15, 1.0)
                })
                .unwrap_or(0.75);
            let budget = sediment_budget
                .map(|b| b.get(i as u32, j as u32).max(0.0))
                .unwrap_or(amount);
            let raise = (concavity * 0.65 * slope_w.max(0.15) * flow_e)
                .min(amount)
                .min(budget);
            if raise > 1e-6 {
                out.set(i as u32, j as u32, h0 + raise);
                used[j as usize * ww + i as usize] = raise;
            }
        }
    }
    (out, raw_mask(&used, metrics))
}

/// Cohesive mud settle: low-slope drainage fill + short-wavelength damping, sediment-capped.
pub fn mud_settle_mass(
    input: &Heightfield,
    amount: f32,
    slope_max_deg: f32,
    radius: u32,
    sediment_budget: Option<&MaskField>,
) -> (Heightfield, MaskField) {
    let settled = bilateral_settle(input, radius.max(1), amount.max(1.0));
    let metrics = input.metrics;
    let w = metrics.width as i32;
    let h = metrics.height as i32;
    let ww = metrics.width as usize;
    let mut out = input.clone();
    let mut used = vec![0.0f32; (metrics.width * metrics.height) as usize];

    for j in 0..h {
        for i in 0..w {
            let h0 = input.get(i as u32, j as u32);
            let budget = sediment_budget
                .map(|b| b.get(i as u32, j as u32).max(0.0))
                .unwrap_or(amount);
            if budget <= 1e-8 {
                continue;
            }
            let mut is_sink = true;
            let mut nmax = h0;
            for &(di, dj) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                let ni = (i + di).clamp(0, w - 1) as u32;
                let nj = (j + dj).clamp(0, h - 1) as u32;
                let nh = input.get(ni, nj);
                nmax = nmax.max(nh);
                if nh < h0 - 1e-4 {
                    is_sink = false;
                }
            }
            let hb = settled.get(i as u32, j as u32);
            let slope = local_slope_deg(input, i as u32, j as u32);
            if is_sink {
                let fill = (nmax.min(h0 + amount) - h0).clamp(0.0, budget);
                let target = (h0 + fill).max(hb.min(h0 + budget));
                let raise = (target - h0).clamp(0.0, budget);
                out.set(i as u32, j as u32, h0 + raise);
                used[j as usize * ww + i as usize] = raise;
            } else if slope < slope_max_deg * 0.35 {
                let t = (1.0 - slope / (slope_max_deg * 0.35).max(1.0)).clamp(0.0, 1.0) * 0.45;
                let blended = h0 * (1.0 - t) + hb * t;
                let delta = (blended - h0).clamp(-budget * 0.25, budget);
                out.set(i as u32, j as u32, h0 + delta);
                if delta > 0.0 {
                    used[j as usize * ww + i as usize] = delta;
                }
            }
        }
    }
    (out, raw_mask(&used, metrics))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

    const RECEIVER_SEED_SWEEP: u64 = 16_384;
    const RECEIVER_COUNT_TOLERANCE: usize = RECEIVER_SEED_SWEEP.div_ceil(100) as usize;
    const RECEIVER_INDICES: [usize; 4] = [3, 5, 1, 7];

    #[test]
    fn optional_layers_preserve_inventories_and_reconstruct_the_surface() {
        let m = HeightfieldMetrics::new(2, 2, 2.0, 2.0);
        let height = vec![-5.0, 0.0, 10.0, 2.0];
        let bedrock = vec![0.0, 0.0, 6.0, 1.0];
        let debris = vec![0.5, 0.25, 1.0, 0.25];
        let sediment = vec![1.0, 0.5, 3.0, 0.5];
        let hf = Heightfield::from_dense(m, &height);
        let bedrock_field = MaskField::from_raw(m, &bedrock);
        let debris_field = MaskField::from_raw(m, &debris);
        let sediment_field = MaskField::from_raw(m, &sediment);

        let state = MassWastingState::with_optional_layers(
            &hf,
            Some(&bedrock_field),
            Some(&debris_field),
            Some(&sediment_field),
            0.0,
            0.0,
        );

        assert_eq!(state.bedrock, bedrock);
        assert_eq!(state.debris, debris);
        assert_eq!(state.sediment, sediment);
        for (actual, expected) in state.sync_surface().into_iter().zip(height) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        assert!(
            state.base[0] < -5.0,
            "sub-zero sediment needs a lower datum"
        );
    }

    fn receiver_surface(drops: [f32; 4]) -> Vec<f32> {
        let mut surface = vec![10.0; 9];
        for (idx, drop) in RECEIVER_INDICES.into_iter().zip(drops) {
            surface[idx] -= drop;
        }
        surface
    }

    fn receiver_counts(drops: [f32; 4]) -> [usize; 4] {
        let surface = receiver_surface(drops);
        let mut counts = [0usize; 4];
        for seed in 0..RECEIVER_SEED_SWEEP {
            let receiver = pick_receiver(&surface, 3, 3, 1, 1, seed)
                .expect("the center fixture has downhill candidates");
            let direction = RECEIVER_INDICES
                .iter()
                .position(|&idx| idx == receiver)
                .expect("receiver must be one of the four cardinal neighbors");
            counts[direction] += 1;
        }
        counts
    }

    fn assert_receiver_count_near(actual: usize, expected_share: f64, direction: &str) {
        let expected = expected_share * RECEIVER_SEED_SWEEP as f64;
        let difference = (actual as f64 - expected).abs();
        assert!(
            difference <= RECEIVER_COUNT_TOLERANCE as f64,
            "{direction} count {actual} differs from expected {expected:.1} by more than {}",
            RECEIVER_COUNT_TOLERANCE
        );
    }

    #[test]
    fn pick_receiver_equal_drops_are_balanced() {
        let counts = receiver_counts([1.0; 4]);
        for (direction, count) in ["left", "right", "up", "down"].into_iter().zip(counts) {
            assert_receiver_count_near(count, 0.25, direction);
        }
    }

    #[test]
    fn pick_receiver_follows_unequal_drop_weights() {
        let counts = receiver_counts([1.0, 2.0, 3.0, 4.0]);
        for ((direction, expected_share), count) in [
            ("left", 0.10),
            ("right", 0.20),
            ("up", 0.30),
            ("down", 0.40),
        ]
        .into_iter()
        .zip(counts)
        {
            assert_receiver_count_near(count, expected_share, direction);
        }
    }

    #[test]
    fn pick_receiver_preserves_zero_and_one_candidate_behavior() {
        let flat = receiver_surface([0.0; 4]);
        let one_candidate = receiver_surface([1.0, 0.0, 0.0, 0.0]);
        for seed in 0..RECEIVER_SEED_SWEEP {
            assert_eq!(pick_receiver(&flat, 3, 3, 1, 1, seed), None);
            assert_eq!(
                pick_receiver(&one_candidate, 3, 3, 1, 1, seed),
                Some(RECEIVER_INDICES[0])
            );
        }
    }

    #[test]
    fn pick_receiver_is_repeatable_for_a_fixed_seed_and_surface() {
        let surface = receiver_surface([1.0, 2.0, 3.0, 4.0]);
        let expected = pick_receiver(&surface, 3, 3, 1, 1, 0x64);
        for _ in 0..32 {
            assert_eq!(pick_receiver(&surface, 3, 3, 1, 1, 0x64), expected);
        }
    }

    #[test]
    fn pick_receiver_distribution_rotates_with_the_fixture() {
        let original = receiver_counts([1.0, 2.0, 3.0, 4.0]);
        let rotated = receiver_counts([4.0, 3.0, 1.0, 2.0]);
        // Clockwise rotation maps original left/right/up/down to rotated
        // up/down/right/left respectively.
        let rotated_in_original_order = [rotated[2], rotated[3], rotated[1], rotated[0]];

        for ((direction, original_count), rotated_count) in ["left", "right", "up", "down"]
            .into_iter()
            .zip(original)
            .zip(rotated_in_original_order)
        {
            assert!(
                original_count.abs_diff(rotated_count) <= RECEIVER_COUNT_TOLERANCE,
                "{direction} count did not rotate: original={original_count}, rotated={rotated_count}"
            );
        }
    }

    #[test]
    fn layered_thermal_preserves_mass_approximately() {
        let m = HeightfieldMetrics::new(24, 24, 48.0, 48.0);
        let mut hf = Heightfield::filled(m, 10.0);
        hf.set(12, 12, 40.0);
        let p = ThermalErosionParams {
            talus_angle_deg: 30.0,
            iterations: 20,
            strength: 0.7,
            weathering_rate: 1.0,
            material_amount: 50.0,
            transport_distance: 2.0,
            ..ThermalErosionParams::default()
        };
        let k = MaskField::filled(m, 0.0);
        let r = thermal_erode_layered(&hf, &p, &k, None);
        let before: f32 = hf.to_dense().iter().sum();
        let after: f32 = r.height.to_dense().iter().sum();
        assert!(
            (before - after).abs() < 1.0,
            "mass drift too large: {before} → {after}"
        );
        assert!(r.height.get(12, 12) < 40.0);
        let loose: f32 = r.loose_debris.data().iter().sum();
        assert!(loose > 0.0, "should produce loose debris");
    }

    #[test]
    fn layered_thermal_preserves_sub_zero_bathymetry() {
        // Deep underwater basin with a raised rim. Thermal erosion must not
        // flatten the seabed up to datum 0: the mass-wasting state keeps a
        // basement datum below the non-negative material layers so sub-zero
        // (underwater) terrain survives. Before this fix the -200 m floor
        // collapsed to ~0.
        let m = HeightfieldMetrics::new(24, 24, 48.0, 48.0);
        let mut hf = Heightfield::filled(m, -200.0);
        for j in 0..6 {
            for i in 0..6 {
                hf.set(i, j, 40.0); // rim above water → real slope to weather
            }
        }
        let p = ThermalErosionParams {
            talus_angle_deg: 30.0,
            iterations: 30,
            strength: 0.8,
            weathering_rate: 1.0,
            material_amount: 50.0,
            ..ThermalErosionParams::default()
        };
        let k = MaskField::filled(m, 0.0);
        let r = thermal_erode_layered(&hf, &p, &k, None);
        let min = r
            .height
            .to_dense()
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        assert!(
            min < -150.0,
            "thermal erosion flattened bathymetry: min={min}"
        );
    }

    #[test]
    fn hard_bedrock_resists_weathering() {
        let m = HeightfieldMetrics::new(16, 16, 16.0, 16.0);
        let mut hf = Heightfield::filled(m, 10.0);
        hf.set(8, 8, 50.0);
        let p = ThermalErosionParams {
            talus_angle_deg: 25.0,
            iterations: 30,
            strength: 0.8,
            hardness: 1.0,
            weathering_rate: 1.0,
            ..ThermalErosionParams::default()
        };
        let k = MaskField::filled(m, 1.0);
        let r = thermal_erode_layered(&hf, &p, &k, None);
        assert!((r.height.get(8, 8) - 50.0).abs() < 0.1);
    }

    #[test]
    fn debris_flow_carves_steep_spike() {
        let m = HeightfieldMetrics::new(32, 32, 64.0, 64.0);
        let mut hf = Heightfield::filled(m, 5.0);
        for j in 8..24 {
            for i in 8..24 {
                let d = ((i as f32 - 16.0).hypot(j as f32 - 16.0)).max(0.0);
                hf.set(i, j, 5.0 + (12.0 - d).max(0.0) * 3.0);
            }
        }
        let p = DebrisFlowParams {
            iterations: 12,
            dt: 2.0,
            ..DebrisFlowParams::default()
        };
        let k = MaskField::filled(m, 0.1);
        let r = debris_flow_erode(&hf, &p, &k, None);
        assert!(r.height.get(16, 16) < hf.get(16, 16));
        let deb: f32 = r.debris.data().iter().sum();
        assert!(deb > 0.0 || r.erosion_raw.data().iter().sum::<f32>() > 0.0);
    }

    #[test]
    fn fill_soft_raises_depression_not_ridge() {
        let m = HeightfieldMetrics::new(16, 16, 16.0, 16.0);
        let mut hf = Heightfield::filled(m, 10.0);
        hf.set(8, 8, 4.0); // sink
        hf.set(0, 0, 20.0); // ridge
        let (out, used) = sediment_fill_soft_mass(&hf, 3.0, 45.0, 2, None, None);
        assert!(out.get(8, 8) > 4.0);
        assert!((out.get(0, 0) - 20.0).abs() < 1e-3);
        assert!(used.get(8, 8) > 0.0);
    }
}
