//! Cordonnier-style iterative stream-power landscape evolution (Accurate mode).
//!
//! Integrates
//!
//! ```text
//! ∂h/∂t = U − K Q^m |∇h|^n  (+ optional deposition)
//! ```
//!
//! with Priority-Flood drainage, coupled uplift each step, and drainage
//! topology reuse until height change invalidates the cache.

use crate::heightfield::Heightfield;
use crate::mask::MaskField;

use super::cache::DrainageCache;
use super::params::LandscapeEvolutionParams;
use super::BoundaryMasks;

const D8_OFFSETS: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

pub fn evolve_iterative(
    initial: &Heightfield,
    uplift: &MaskField,
    hardness: &MaskField,
    boundaries: &BoundaryMasks,
    p: &LandscapeEvolutionParams,
    steps: u32,
) -> (Heightfield, DrainageCache, Vec<f32>, Vec<f32>) {
    let m = initial.metrics;
    let w = m.width as usize;
    let h = m.height as usize;
    let n = w * h;
    let k = p.effective_k();
    let m_exp = p.effective_m();
    let n_exp = p.effective_n();
    let dt = p.dt.max(1.0);
    let rain = p.rainfall.max(0.05);
    let area_scale = (0.35 + 1.3 * p.drainage_scale.clamp(0.0, 1.0)) * rain;
    let cell_area = (m.dx() * m.dz()).max(1e-6);
    let dx = m.dx().max(1e-3);
    let dzm = m.dz().max(1e-3);
    let depos = p.deposition_coeff.max(0.0);
    let base = p.base_level;
    let invalidation = p.drainage_invalidation_m.max(0.1);

    let routing_outlets = &boundaries.routing_outlets;
    let elevation_locks = &boundaries.elevation_locks;
    let mut z = initial.clone();
    let mut cache = DrainageCache::build(&z, true, routing_outlets);
    let mut incision = vec![0.0f32; n];
    let mut erosion_rate = vec![0.0f32; n];
    let mut sediment = vec![0.0f32; n];

    let steps = steps.max(1);
    for step in 0..steps {
        if step == 0 || cache.needs_rebuild(&z, invalidation) {
            cache = DrainageCache::build(&z, true, routing_outlets);
        }

        // Discharge Q ≈ rain × drainage area.
        let mut q = vec![0.0f32; n];
        for idx in 0..n {
            q[idx] = (cache.accumulation[idx] * cell_area * area_scale).max(cell_area);
        }

        let mut max_delta = 0.0f32;
        let prev = z.to_dense();

        for j in 0..h {
            for i in 0..w {
                let idx = j * w + i;
                if elevation_locks.get(i as u32, j as u32) > 0.5 {
                    continue;
                }
                let soft = (1.0 - hardness.get(i as u32, j as u32).clamp(0.0, 1.0)).max(0.0);
                let slope = slope_to_receiver(&z, i, j, &cache).max(1e-6);
                // Shared stream-power law (world-metric slope, discharge Q); the cap
                // is relief-relative for stability. See crate::hydro::spe_increment.
                let (e, e_step) = crate::hydro::spe_increment(
                    q[idx],
                    slope,
                    k,
                    m_exp,
                    n_exp,
                    soft,
                    dt,
                    slope * ((dx + dzm) * 0.5) * 4.0,
                    80.0,
                );
                let u = uplift.get(i as u32, j as u32) * dt;

                let mut dh = u - e_step;
                // Optional soft deposition in low-slope, high-load cells.
                if depos > 1e-6 && slope < 0.08 {
                    let load = sediment[idx] + e_step * 0.35;
                    let deposit = load * depos * (1.0 - slope / 0.08).clamp(0.0, 1.0);
                    dh += deposit;
                    sediment[idx] = (load - deposit).max(0.0);
                } else {
                    sediment[idx] = (sediment[idx] + e_step * 0.1) * 0.92;
                }

                let cur = z.get(i as u32, j as u32);
                let next = (cur + dh).max(base);
                let carved = (cur - next).max(0.0);
                incision[idx] += carved;
                erosion_rate[idx] = e;
                max_delta = max_delta.max((next - cur).abs());
                z.set(i as u32, j as u32, next);
            }
        }

        // Lightweight discontinuity smooth along non-receiver neighbours (Cordonnier tree).
        if step % 3 == 2 {
            smooth_tree_discontinuities(&mut z, &cache, elevation_locks, 0.2);
            restore_locked(&mut z, initial, elevation_locks);
        }

        cache.max_abs_delta = max_delta;
        // Update fingerprint without full rebuild when delta is small.
        if max_delta <= invalidation {
            cache.refresh_fingerprint(&z);
        }

        let _ = prev; // kept for potential future residual diagnostics
    }

    restore_locked(&mut z, initial, elevation_locks);

    // Publish raw accumulation for aux (normalised later by operator).
    (z, cache, incision, erosion_rate)
}

fn slope_to_receiver(z: &Heightfield, i: usize, j: usize, cache: &DrainageCache) -> f32 {
    let w = z.metrics.width as usize;
    let idx = j * w + i;
    let h0 = z.get(i as u32, j as u32);
    let dx = z.metrics.dx().max(1e-3);
    let dz = z.metrics.dz().max(1e-3);
    if cache.receiver[idx] != usize::MAX {
        let r = cache.receiver[idx];
        let (ri, rj) = (r % w, r / w);
        let di = ri as i32 - i as i32;
        let dj = rj as i32 - j as i32;
        let dist = if di != 0 && dj != 0 {
            ((dx * dx + dz * dz) as f32).sqrt()
        } else if di != 0 {
            dx
        } else {
            dz
        };
        return ((h0 - z.get(ri as u32, rj as u32)) / dist.max(1e-6)).max(0.0);
    }
    // Fallback: steepest D8.
    let mut best = 0.0f32;
    for &(di, dj) in &D8_OFFSETS {
        let ni = i as i32 + di;
        let nj = j as i32 + dj;
        if ni < 0 || nj < 0 || ni >= w as i32 || nj >= z.metrics.height as i32 {
            continue;
        }
        let dist = if di != 0 && dj != 0 {
            ((dx * dx + dz * dz) as f32).sqrt()
        } else if di != 0 {
            dx
        } else {
            dz
        };
        best = best.max((h0 - z.get(ni as u32, nj as u32)) / dist.max(1e-6));
    }
    best.max(0.0)
}

fn smooth_tree_discontinuities(
    z: &mut Heightfield,
    cache: &DrainageCache,
    elevation_locks: &MaskField,
    strength: f32,
) {
    let w = z.metrics.width as usize;
    let h = z.metrics.height as usize;
    let dense = z.to_dense();
    let s = strength.clamp(0.0, 1.0);
    for j in 0..h {
        for i in 0..w {
            let idx = j * w + i;
            if elevation_locks.get(i as u32, j as u32) > 0.5 {
                continue;
            }
            if cache.receiver[idx] == usize::MAX {
                continue;
            }
            let r = cache.receiver[idx];
            let drop_r = dense[idx] - dense[r];
            if drop_r <= 0.0 {
                continue;
            }
            // Non-connected 4-neighbours.
            for &(di, dj) in &[(1i32, 0), (-1, 0), (0, 1), (0, -1)] {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
                    continue;
                }
                if elevation_locks.get(ni as u32, nj as u32) > 0.5 {
                    continue;
                }
                let nidx = nj as usize * w + ni as usize;
                if cache.receiver[idx] == nidx || cache.receiver[nidx] == idx {
                    continue;
                }
                let drop_n = dense[idx] - dense[nidx];
                // Tzathas discontinuity: z(c)-z(cn) > z(c)-z(cr)
                if drop_n > drop_r {
                    let excess = (drop_n - drop_r) * 0.5 * s;
                    let cur = z.get(i as u32, j as u32);
                    let other = z.get(ni as u32, nj as u32);
                    z.set(i as u32, j as u32, cur - excess * 0.5);
                    z.set(ni as u32, nj as u32, other + excess * 0.5);
                }
            }
        }
    }
}

fn restore_locked(z: &mut Heightfield, original: &Heightfield, elevation_locks: &MaskField) {
    let m = z.metrics;
    for j in 0..m.height {
        for i in 0..m.width {
            if elevation_locks.get(i, j) > 0.5 {
                z.set(i, j, original.get(i, j));
            }
        }
    }
}
