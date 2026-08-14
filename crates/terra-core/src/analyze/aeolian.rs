//! Aeolian sand transport (Desertscapes-class).
//!
//! Shared by the fast procedural **Dunes** filter and the progressive
//! **Sand Simulation** layer. Morphology (transverse / barchan-like / linear /
//! fields) emerges from wind, supply, and transport parameters — not hardcoded meshes.
//!
//! Primary references:
//! - Paris, Guérin, Galin, Peytavie (2019) — Desertscapes Simulation
//! - Taylor & Keyser (2023) — Real-Time Sand Dune Simulation (GPU)
//! - Nilles, Günther, Müller (2024) — Real-Time Desertscapes with CUDA
//!   (deterministic saltation, bilinear shadow/advection, improved avalanching)

use crate::heightfield::{Heightfield, HeightfieldMetrics};
use crate::mask::MaskField;

/// Shared transport controls used by both the procedural filter and the sim.
#[derive(Debug, Clone, Copy)]
pub struct AeolianTransportParams {
    /// Wind direction in degrees (0 = +X, 90 = +Z).
    pub wind_direction_deg: f32,
    /// Relative wind speed / transport energy \[0, 2\].
    pub wind_speed: f32,
    /// Saltation hop length in cells (advection distance per transport step).
    pub transport_length: f32,
    /// Angle of repose for sand avalanching (degrees).
    pub repose_angle_deg: f32,
    /// Slab thickness lifted per saltation event (meters).
    pub slab_size: f32,
    /// Avalanche sweeps per transport step.
    pub avalanche_iters: u32,
    /// Optional bedrock abrasion rate when bouncing on thin sand \[0, 1\].
    pub abrasion: f32,
    /// Reptation strength after deposition \[0, 1\].
    pub reptation: f32,
    /// How strongly terrain warps the wind field \[0, 1\].
    pub wind_warp: f32,
    /// Lateral wind coherence: high → long ridges (linear/transverse); low → more crescent/star-like.
    pub linearity: f32,
}

impl Default for AeolianTransportParams {
    fn default() -> Self {
        Self {
            wind_direction_deg: 0.0,
            wind_speed: 1.0,
            transport_length: 5.0,
            repose_angle_deg: 33.0,
            slab_size: 0.35,
            avalanche_iters: 8,
            abrasion: 0.02,
            reptation: 0.15,
            wind_warp: 0.35,
            linearity: 0.7,
        }
    }
}

/// Layered aeolian state: surface = bedrock + sand.
#[derive(Debug, Clone)]
pub struct AeolianState {
    pub metrics: HeightfieldMetrics,
    pub bedrock: Vec<f32>,
    pub sand: Vec<f32>,
    pub wind_u: Vec<f32>,
    pub wind_v: Vec<f32>,
    pub wind_speed: Vec<f32>,
    pub sheltering: Vec<f32>,
    pub sand_flux: Vec<f32>,
    pub erosion: Vec<f32>,
    pub deposition: Vec<f32>,
}

/// Published multi-channel result.
#[derive(Debug, Clone)]
pub struct AeolianResult {
    pub height: Heightfield,
    pub sand_depth: MaskField,
    pub bedrock: MaskField,
    pub wind_direction: MaskField,
    pub wind_speed: MaskField,
    pub sand_flux: MaskField,
    pub deposition: MaskField,
    pub erosion: MaskField,
    pub sheltering: MaskField,
    pub dune_crest: MaskField,
}

impl AeolianState {
    pub fn from_height_and_sand(
        input: &Heightfield,
        initial_sand: f32,
        sand_field: Option<&[f32]>,
    ) -> Self {
        let metrics = input.metrics;
        let n = (metrics.width * metrics.height) as usize;
        let height = input.to_dense();
        let mut sand = vec![initial_sand.max(0.0); n];
        if let Some(src) = sand_field {
            debug_assert_eq!(src.len(), n);
            for i in 0..n {
                sand[i] = src[i].max(0.0);
            }
        }
        let mut bedrock = Vec::with_capacity(n);
        for i in 0..n {
            bedrock.push((height[i] - sand[i]).max(0.0));
            let surface = bedrock[i] + sand[i];
            if surface > height[i] + 1e-5 {
                sand[i] = (height[i] - bedrock[i]).max(0.0);
            }
        }
        Self {
            metrics,
            bedrock,
            sand,
            wind_u: vec![0.0; n],
            wind_v: vec![0.0; n],
            wind_speed: vec![0.0; n],
            sheltering: vec![0.0; n],
            sand_flux: vec![0.0; n],
            erosion: vec![0.0; n],
            deposition: vec![0.0; n],
        }
    }

    #[inline]
    pub fn surface_at(&self, idx: usize) -> f32 {
        self.bedrock[idx] + self.sand[idx]
    }

    pub fn sync_height(&self) -> Heightfield {
        let mut data = Vec::with_capacity(self.bedrock.len());
        for i in 0..self.bedrock.len() {
            data.push(self.surface_at(i));
        }
        Heightfield::from_dense(self.metrics, &data)
    }

    /// Run `steps` full transport iterations (wind → shadow → saltation → reptation → avalanche).
    pub fn evolve(&mut self, p: &AeolianTransportParams, steps: u32) {
        let steps = steps.max(1);
        for _ in 0..steps {
            self.update_wind(p);
            self.update_sheltering(p);
            self.saltation_step(p);
            if p.reptation > 1e-5 {
                self.reptation_step(p);
            }
            self.avalanche(p, p.avalanche_iters.max(1));
        }
    }

    fn update_wind(&mut self, p: &AeolianTransportParams) {
        let w = self.metrics.width as i32;
        let h = self.metrics.height as i32;
        let dx = self.metrics.dx().max(1e-5);
        // Wind along direction_deg from +X: 0° → (+1, 0), 90° → (0, +1).
        let ang = p.wind_direction_deg.to_radians();
        let (s, c) = ang.sin_cos();
        let bu = c;
        let bv = s;
        let speed = p.wind_speed.max(0.0);
        let warp = p.wind_warp.clamp(0.0, 1.0);
        let lin = p.linearity.clamp(0.0, 1.0);

        for j in 0..h {
            for i in 0..w {
                let idx = (j * w + i) as usize;
                let h0 = self.surface_at(idx);
                // Terrain-gradient wind warp (Paris venturi + perpendicular deflection).
                let hx = sample_surface(self, i + 1, j) - sample_surface(self, i - 1, j);
                let hz = sample_surface(self, i, j + 1) - sample_surface(self, i, j - 1);
                let gx = hx / (2.0 * dx);
                let gz = hz / (2.0 * dx);
                let mut wu = bu - warp * gz * 30.0 * dx;
                let mut wv = bv + warp * gx * 30.0 * dx;
                // High linearity keeps wind aligned; low linearity allows more deflection.
                wu = bu * lin + wu * (1.0 - lin);
                wv = bv * lin + wv * (1.0 - lin);
                let len = (wu * wu + wv * wv).sqrt().max(1e-5);
                let venturi = 1.0 + 0.004 * h0;
                let sp = speed * venturi;
                self.wind_u[idx] = wu / len * sp;
                self.wind_v[idx] = wv / len * sp;
                self.wind_speed[idx] = sp;
            }
        }
    }

    fn update_sheltering(&mut self, p: &AeolianTransportParams) {
        let w = self.metrics.width as i32;
        let h = self.metrics.height as i32;
        let cell = self.metrics.dx().max(1e-5);
        // Look back far enough to catch dune-scale lee shadows (Paris ~10 m, but
        // authoring grids often use larger cells — keep a minimum cell count).
        let max_steps = ((14.0 / cell).ceil() as i32)
            .max(8)
            .min(48)
            .max(p.transport_length.ceil() as i32);
        let t10 = (10.0f32).to_radians().tan();
        let t15 = (15.0f32).to_radians().tan();
        let hop = p.transport_length.max(1.0);

        for j in 0..h {
            for i in 0..w {
                let idx = (j * w + i) as usize;
                let h0 = self.surface_at(idx);
                let wu = self.wind_u[idx];
                let wv = self.wind_v[idx];
                let wlen = (wu * wu + wv * wv).sqrt().max(1e-5);
                let du = wu / wlen;
                let dv = wv / wlen;
                let mut max_tan = 0.0f32;
                // Step upwind looking for sheltering slope (Paris / Nilles).
                for s in 1..=max_steps {
                    let t = s as f32;
                    let x = i as f32 - du * t;
                    let z = j as f32 - dv * t;
                    let hu = sample_surface_bilinear(self, x, z);
                    let ang = (hu - h0) / (t * cell);
                    if ang > max_tan {
                        max_tan = ang;
                    }
                }
                // Longer hops see slightly more shelter sensitivity.
                let scale = 1.0 + (hop - 5.0).clamp(-2.0, 4.0) * 0.02;
                let rs = ((max_tan * scale - t10) / (t15 - t10).max(1e-5)).clamp(0.0, 1.0);
                self.sheltering[idx] = rs;
            }
        }
    }

    fn saltation_step(&mut self, p: &AeolianTransportParams) {
        let w = self.metrics.width as usize;
        let h = self.metrics.height as usize;
        let n = w * h;
        let mut slab = vec![0.0f32; n];
        let eps = p.slab_size.max(1e-4) * p.wind_speed.clamp(0.15, 2.0);
        let hop = p.transport_length.max(0.5);

        // Lift into slab (deterministic expected-value form; Nilles 2024).
        for idx in 0..n {
            let exposed = 1.0 - self.sheltering[idx];
            let wind_f = (self.wind_speed[idx] / p.wind_speed.max(1e-3)).clamp(0.0, 2.0);
            let lift = (eps * exposed * wind_f).min(self.sand[idx]);
            if lift <= 1e-8 {
                continue;
            }
            self.sand[idx] -= lift;
            slab[idx] += lift;
            self.erosion[idx] += lift;
            self.sand_flux[idx] += lift;
        }

        // Advect slab downwind with bilinear splat.
        let mut advected = vec![0.0f32; n];
        for j in 0..h {
            for i in 0..w {
                let idx = j * w + i;
                let amount = slab[idx];
                if amount <= 1e-8 {
                    continue;
                }
                let sp = self.wind_speed[idx].max(1e-5);
                let du = self.wind_u[idx] / sp * hop;
                let dv = self.wind_v[idx] / sp * hop;
                splat_bilinear(&mut advected, w, h, i as f32 + du, j as f32 + dv, amount);
            }
        }

        // Deposit where carrying capacity drops (shadow / slow wind); remainder abrades / stays mobile briefly.
        for idx in 0..n {
            let carried = advected[idx];
            if carried <= 1e-8 {
                continue;
            }
            let shadow = self.sheltering[idx];
            let slow = (1.0 - (self.wind_speed[idx] / p.wind_speed.max(1e-3)).clamp(0.0, 1.5))
                .clamp(0.0, 1.0);
            // Capacity decreases in shadow and on the lee — deposit fraction rises.
            let dep_p = (0.18 + 0.72 * shadow + 0.25 * slow).clamp(0.05, 0.95);
            let deposited = carried * dep_p;
            let bounced = carried - deposited;
            self.sand[idx] += deposited;
            self.deposition[idx] += deposited;

            // Abrasion: bouncing grains convert a little bedrock → sand when cover is thin.
            if bounced > 1e-6 && p.abrasion > 0.0 && self.sand[idx] < p.slab_size * 2.0 {
                let abrade = (bounced * p.abrasion * 0.15).min(self.bedrock[idx].max(0.0));
                self.bedrock[idx] -= abrade;
                self.sand[idx] += abrade;
            }
            // Remainder settles locally (multi-bounce emerges across transport steps).
            self.sand[idx] += bounced;
            self.deposition[idx] += bounced;
        }
    }

    fn reptation_step(&mut self, p: &AeolianTransportParams) {
        let w = self.metrics.width as i32;
        let h = self.metrics.height as i32;
        let cell = self.metrics.dx().max(1e-5);
        let kr = p.reptation.clamp(0.0, 1.0) * 0.08;
        let mut delta = vec![0.0f32; (w * h) as usize];

        for j in 0..h {
            for i in 0..w {
                let idx = (j * w + i) as usize;
                let mut acc = 0.0f32;
                let mut count = 0u32;
                for (di, dj) in [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (1, -1),
                    (-1, 1),
                    (1, 1),
                ] {
                    let ni = i + di;
                    let nj = j + dj;
                    if ni < 0 || nj < 0 || ni >= w || nj >= h {
                        continue;
                    }
                    let nidx = (nj * w + ni) as usize;
                    let dist = ((di * di + dj * dj) as f32).sqrt();
                    let d = (self.surface_at(nidx) - self.surface_at(idx)) / (cell * dist);
                    // Move sand downslope proportional to slope (Nilles reptation fix).
                    let sm =
                        kr * d.abs() * 0.5 * (self.sand_flux[idx] + self.sand_flux[nidx] + 0.05);
                    let transfer = if d >= 0.0 {
                        // Neighbor higher → receive from neighbor limited by neighbor sand.
                        sm.min(self.sand[nidx])
                    } else {
                        -sm.min(self.sand[idx])
                    };
                    acc += transfer;
                    count += 1;
                }
                if count > 0 {
                    delta[idx] = acc / count as f32;
                }
            }
        }
        for idx in 0..delta.len() {
            let d = delta[idx];
            if d > 0.0 {
                self.sand[idx] += d;
            } else {
                let take = (-d).min(self.sand[idx]);
                self.sand[idx] -= take;
            }
        }
    }

    fn avalanche(&mut self, p: &AeolianTransportParams, iters: u32) {
        let w = self.metrics.width as i32;
        let h = self.metrics.height as i32;
        let cell = self.metrics.dx().max(1e-5);
        let tan_r = p.repose_angle_deg.to_radians().tan();
        let kc = 0.5f32;

        for iter in 0..iters {
            let k = if iter + 4 >= iters { 0.5 } else { kc };
            // Snapshot sand for reads; write via deltas to reduce race bias on CPU.
            let sand_snap = self.sand.clone();
            let mut delta = vec![0.0f32; sand_snap.len()];

            for j in 0..h {
                for i in 0..w {
                    let idx = (j * w + i) as usize;
                    if sand_snap[idx] <= 1e-8 {
                        continue;
                    }
                    let mut b_vals = [0.0f32; 8];
                    let mut neighbors = [(0i32, 0i32); 8];
                    let mut b_sum = 0.0f32;
                    let mut b_max = 0.0f32;
                    let mut ncount = 0usize;
                    for (di, dj) in [
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                        (-1, -1),
                        (1, -1),
                        (-1, 1),
                        (1, 1),
                    ] {
                        let ni = i + di;
                        let nj = j + dj;
                        if ni < 0 || nj < 0 || ni >= w || nj >= h {
                            continue;
                        }
                        let nidx = (nj * w + ni) as usize;
                        let dist = ((di * di + dj * dj) as f32).sqrt();
                        // Height drop from i toward neighbor (downslope excess).
                        let drop = (self.bedrock[idx] + sand_snap[idx])
                            - (self.bedrock[nidx] + sand_snap[nidx]);
                        let excess = drop - tan_r * cell * dist;
                        if excess > 0.0 {
                            let b = excess;
                            b_vals[ncount] = b;
                            neighbors[ncount] = (ni, nj);
                            b_sum += b;
                            b_max = b_max.max(b);
                            ncount += 1;
                        }
                    }
                    if ncount == 0 || b_sum <= 1e-8 || b_max <= 1e-8 {
                        continue;
                    }
                    // Nilles: move enough to stabilize the worst neighbor.
                    let ba = (b_max / (1.0 + b_max / b_sum)).min(sand_snap[idx]);
                    if ba <= 1e-8 {
                        continue;
                    }
                    delta[idx] -= k * ba;
                    for n in 0..ncount {
                        let (ni, nj) = neighbors[n];
                        let nidx = (nj * w + ni) as usize;
                        let share = b_vals[n] / b_sum;
                        delta[nidx] += k * ba * share;
                    }
                }
            }

            for idx in 0..self.sand.len() {
                self.sand[idx] = (sand_snap[idx] + delta[idx]).max(0.0);
            }
        }
    }

    pub fn into_result(self, strength: f32, original: &Heightfield) -> AeolianResult {
        let height = self.sync_height();
        let mixed = mix_height(original, &height, strength);
        let metrics = self.metrics;
        let n = self.sand.len();

        let sand_depth = MaskField::from_raw(metrics, &self.sand);
        let bedrock = MaskField::from_raw(metrics, &self.bedrock);

        let mut wind_dir = vec![0.0f32; n];
        for i in 0..n {
            wind_dir[i] = self.wind_v[i].atan2(self.wind_u[i]);
        }
        let wind_direction = MaskField::from_raw(metrics, &wind_dir);
        let wind_speed = normalize_positive(metrics, &self.wind_speed);
        let sand_flux = normalize_positive(metrics, &self.sand_flux);
        let deposition = normalize_positive(metrics, &self.deposition);
        let erosion = normalize_positive(metrics, &self.erosion);
        let sheltering = MaskField::from_raw(metrics, &self.sheltering);

        // Crest proxy: high sand + exposed (low shelter) windward shoulders.
        let mut crest = vec![0.0f32; n];
        let mut cmax = 1e-6f32;
        for i in 0..n {
            let v = self.sand[i] * (1.0 - self.sheltering[i] * 0.65);
            crest[i] = v;
            cmax = cmax.max(v);
        }
        for v in &mut crest {
            *v = (*v / cmax).clamp(0.0, 1.0);
        }
        let dune_crest = MaskField::from_raw(metrics, &crest);

        AeolianResult {
            height: mixed,
            sand_depth,
            bedrock,
            wind_direction,
            wind_speed,
            sand_flux,
            deposition,
            erosion,
            sheltering,
            dune_crest,
        }
    }
}

fn mix_height(original: &Heightfield, evolved: &Heightfield, strength: f32) -> Heightfield {
    let s = strength.clamp(0.0, 1.0);
    if s >= 0.999 {
        return evolved.clone();
    }
    if s <= 1e-5 {
        return original.clone();
    }
    let mut out = original.clone();
    for j in 0..original.metrics.height {
        for i in 0..original.metrics.width {
            let a = original.get(i, j);
            let b = evolved.get(i, j);
            out.set(i, j, a * (1.0 - s) + b * s);
        }
    }
    out
}

fn normalize_positive(metrics: HeightfieldMetrics, data: &[f32]) -> MaskField {
    let mut max_v = 1e-6f32;
    for &v in data {
        max_v = max_v.max(v);
    }
    let mut out = Vec::with_capacity(data.len());
    for &v in data {
        out.push((v / max_v).clamp(0.0, 1.0));
    }
    MaskField::from_raw(metrics, &out)
}

#[inline]
fn sample_surface(state: &AeolianState, i: i32, j: i32) -> f32 {
    let w = state.metrics.width as i32;
    let h = state.metrics.height as i32;
    let ii = i.clamp(0, w - 1) as u32;
    let jj = j.clamp(0, h - 1) as u32;
    let idx = (jj * state.metrics.width + ii) as usize;
    state.surface_at(idx)
}

fn sample_surface_bilinear(state: &AeolianState, x: f32, z: f32) -> f32 {
    let w = state.metrics.width as i32;
    let h = state.metrics.height as i32;
    let x = x.clamp(0.0, (w - 1) as f32);
    let z = z.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let x1 = (x0 + 1).min(w - 1);
    let z1 = (z0 + 1).min(h - 1);
    let tx = x - x0 as f32;
    let tz = z - z0 as f32;
    let h00 = sample_surface(state, x0, z0);
    let h10 = sample_surface(state, x1, z0);
    let h01 = sample_surface(state, x0, z1);
    let h11 = sample_surface(state, x1, z1);
    let a = h00 * (1.0 - tx) + h10 * tx;
    let b = h01 * (1.0 - tx) + h11 * tx;
    a * (1.0 - tz) + b * tz
}

fn splat_bilinear(buf: &mut [f32], w: usize, h: usize, x: f32, z: f32, amount: f32) {
    if amount <= 0.0 {
        return;
    }
    let x = x.clamp(0.0, (w.saturating_sub(1)) as f32);
    let z = z.clamp(0.0, (h.saturating_sub(1)) as f32);
    let x0 = x.floor() as usize;
    let z0 = z.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let z1 = (z0 + 1).min(h - 1);
    let tx = x - x0 as f32;
    let tz = z - z0 as f32;
    let w00 = (1.0 - tx) * (1.0 - tz);
    let w10 = tx * (1.0 - tz);
    let w01 = (1.0 - tx) * tz;
    let w11 = tx * tz;
    buf[z0 * w + x0] += amount * w00;
    buf[z0 * w + x1] += amount * w10;
    buf[z1 * w + x0] += amount * w01;
    buf[z1 * w + x1] += amount * w11;
}

/// Quality-aware transport step budget so Draft stays interactive; Full may run thousands.
pub fn transport_steps_for_quality(authored: u32, quality_cap: u32) -> u32 {
    authored.max(1).min(quality_cap.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mass_roughly_conserved_without_abrasion() {
        let m = HeightfieldMetrics::new(48, 48, 480.0, 480.0);
        let bedrock_h = Heightfield::filled(m, 10.0);
        let mut state = AeolianState::from_height_and_sand(&bedrock_h, 2.0, None);
        let before: f32 = state.sand.iter().sum();
        let p = AeolianTransportParams {
            abrasion: 0.0,
            wind_speed: 1.0,
            transport_length: 4.0,
            avalanche_iters: 6,
            ..AeolianTransportParams::default()
        };
        state.evolve(&p, 12);
        let after: f32 = state.sand.iter().sum();
        let rel = (after - before).abs() / before.max(1e-3);
        assert!(
            rel < 0.08,
            "sand mass drifted too far: before={before} after={after} rel={rel}"
        );
    }

    #[test]
    fn sheltering_marks_lee_of_ridge() {
        let m = HeightfieldMetrics::new(64, 32, 640.0, 320.0);
        let mut hf = Heightfield::filled(m, 5.0);
        for j in 0..32 {
            for i in 20..28 {
                hf.set(i, j, 25.0);
            }
        }
        let mut state = AeolianState::from_height_and_sand(&hf, 0.5, None);
        let p = AeolianTransportParams {
            wind_direction_deg: 0.0,
            wind_warp: 0.0,
            ..AeolianTransportParams::default()
        };
        state.update_wind(&p);
        state.update_sheltering(&p);
        let mid = 16usize;
        let wind_sp = state.wind_speed[mid * 64 + 32];
        let surf_ridge = state.surface_at(mid * 64 + 24);
        let surf_lee = state.surface_at(mid * 64 + 32);
        // Immediately downwind of the ridge should be more sheltered than upwind.
        let up = state.sheltering[mid * 64 + 10];
        let lee = state.sheltering[mid * 64 + 29];
        assert!(
            wind_sp > 0.1 && surf_ridge > surf_lee + 5.0,
            "precondition failed: wind={wind_sp} ridge={surf_ridge} lee_h={surf_lee}"
        );
        assert!(
            lee > up + 0.05,
            "lee sheltering should exceed upwind: lee={lee} up={up} wind={wind_sp}"
        );
    }

    #[test]
    fn avalanche_respects_repose() {
        let m = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
        let hf = Heightfield::filled(m, 0.0);
        let mut sand = vec![0.0f32; 32 * 32];
        sand[16 * 32 + 16] = 40.0;
        let mut state = AeolianState::from_height_and_sand(&hf, 0.0, Some(&sand));
        let p = AeolianTransportParams {
            repose_angle_deg: 33.0,
            avalanche_iters: 40,
            ..AeolianTransportParams::default()
        };
        state.avalanche(&p, 40);
        let cell = m.dx();
        let tan_r = 33.0f32.to_radians().tan();
        let mut max_excess = 0.0f32;
        for j in 1..31 {
            for i in 1..31 {
                let idx = j * 32 + i;
                let h0 = state.surface_at(idx);
                for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let nidx = ((j as i32 + dj) * 32 + (i as i32 + di)) as usize;
                    let drop = h0 - state.surface_at(nidx);
                    let excess = drop - tan_r * cell;
                    max_excess = max_excess.max(excess);
                }
            }
        }
        assert!(
            max_excess < cell * 0.85,
            "avalanche left steep excess {max_excess}"
        );
    }
}
