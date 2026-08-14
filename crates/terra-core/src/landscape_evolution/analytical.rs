//! Tzathas et al. 2024 — analytical / fast stream-power landscape evolution.
//!
//! Evaluates the stream-power law along drainage trees using the method of
//! characteristics (`n = 1`):
//!
//! ```text
//! z(x,t) = z0(D_{x,t}) + ∫_{D}^{x} U(s) / a(s) ds
//! where a(s) = K A(s)^m
//! ```
//!
//! Elevations and the river network are coupled via a fixed-point loop
//! (optionally multigrid-accelerated for interactive rates).

use crate::heightfield::Heightfield;
use crate::mask::MaskField;

use super::cache::DrainageCache;
use super::params::LandscapeEvolutionParams;
use super::BoundaryMasks;

/// Characteristic motion below this fraction of a cell cannot be resolved
/// reliably by the f32 path integrals and is treated as the no-advection limit.
const MIN_CHARACTERISTIC_CELL_FRACTION: f32 = 1e-4;

#[derive(Clone, Copy)]
struct FiniteHorizonBounds {
    min: f32,
    max: f32,
}

/// One analytical evaluation + optional fixed-point / multigrid passes.
pub fn evolve_analytical(
    initial: &Heightfield,
    tectonic_base: &Heightfield,
    uplift: &MaskField,
    hardness: &MaskField,
    boundaries: &BoundaryMasks,
    p: &LandscapeEvolutionParams,
    passes: u32,
) -> (Heightfield, DrainageCache, Vec<f32>) {
    let m = initial.metrics;
    let w = m.width as usize;
    let h = m.height as usize;
    let n = w * h;
    let t = p.evolution_time();
    let k = p.effective_k();
    let m_exp = p.effective_m();
    let rain = p.rainfall.max(0.05);
    let dx = m.dx().max(1e-3);
    let dz = m.dz().max(1e-3);
    let cell_len = ((dx + dz) * 0.5).max(1e-3);
    let cell_area = (dx * dz).max(1e-6);
    // Drainage-scale modulates effective area (Hack-like organisation).
    let area_scale = (0.35 + 1.3 * p.drainage_scale.clamp(0.0, 1.0)) * rain;
    let bounds = finite_horizon_bounds(initial, tectonic_base, uplift, t);

    let mut z = initial.clone();
    let routing_outlets = &boundaries.routing_outlets;
    let elevation_locks = &boundaries.elevation_locks;
    let mut cache = DrainageCache::build(&z, true, routing_outlets);
    let mut incision = vec![0.0f32; n];
    let passes = passes.max(1);

    // Multigrid-inspired: start from a coarse downsample when resolution allows.
    let levels = if w >= 64 && h >= 64 && passes >= 4 {
        3u32
    } else if w >= 32 && h >= 32 && passes >= 3 {
        2
    } else {
        1
    };

    for level in (0..levels).rev() {
        let stride = 1usize << level;
        let level_passes = if level == 0 {
            passes
        } else {
            (passes / 2).max(1)
        };
        for pass in 0..level_passes {
            if pass == 0 || pass % 2 == 0 || level > 0 {
                cache = DrainageCache::build(&z, true, routing_outlets);
            }
            let predicted = analytical_on_tree(
                &z,
                tectonic_base,
                uplift,
                hardness,
                elevation_locks,
                &cache,
                t,
                k,
                m_exp,
                area_scale,
                cell_len,
                cell_area,
                stride,
            );
            // EMA toward predicted elevations (stabilises small-t oscillations).
            let alpha = if pass == 0 { 1.0 } else { 0.55 };
            let mut rejected = 0usize;
            for j in 0..h {
                for i in 0..w {
                    if (i % stride != 0) || (j % stride != 0) {
                        continue;
                    }
                    let idx = j * w + i;
                    if elevation_locks.get(i as u32, j as u32) > 0.5 {
                        continue;
                    }
                    let old = z.get(i as u32, j as u32);
                    let neu = guarded_candidate(predicted[idx], old, bounds, &mut rejected);
                    let blended = old + (neu - old) * alpha;
                    incision[idx] += (old - blended).max(0.0);
                    z.set(i as u32, j as u32, blended);
                }
            }
            if stride > 1 {
                bilinear_fill_stride(&mut z, stride);
                restore_locked(&mut z, initial, elevation_locks);
            }
            if rejected > 0 {
                log::warn!(
                    "Landscape Evolution rejected {rejected} analytical candidates outside the finite-time uplift envelope"
                );
            }
        }
    }

    // Final full-res pass for consistency.
    cache = DrainageCache::build(&z, true, routing_outlets);
    let predicted = analytical_on_tree(
        &z,
        tectonic_base,
        uplift,
        hardness,
        elevation_locks,
        &cache,
        t,
        k,
        m_exp,
        area_scale,
        cell_len,
        cell_area,
        1,
    );
    let mut rejected = 0usize;
    for j in 0..h {
        for i in 0..w {
            let idx = j * w + i;
            if elevation_locks.get(i as u32, j as u32) > 0.5 {
                z.set(i as u32, j as u32, initial.get(i as u32, j as u32));
                continue;
            }
            let old = z.get(i as u32, j as u32);
            let neu = guarded_candidate(predicted[idx], old, bounds, &mut rejected);
            incision[idx] += (old - neu).max(0.0) * 0.25;
            z.set(i as u32, j as u32, neu);
        }
    }
    if rejected > 0 {
        log::warn!(
            "Landscape Evolution rejected {rejected} final analytical candidates outside the finite-time uplift envelope"
        );
    }
    restore_locked(&mut z, initial, elevation_locks);
    cache.refresh_fingerprint(&z);

    (z, cache, incision)
}

fn analytical_on_tree(
    z: &Heightfield,
    z0: &Heightfield,
    uplift: &MaskField,
    hardness: &MaskField,
    elevation_locks: &MaskField,
    cache: &DrainageCache,
    t: f32,
    k: f32,
    m_exp: f32,
    area_scale: f32,
    cell_len: f32,
    cell_area: f32,
    stride: usize,
) -> Vec<f32> {
    let w = z.metrics.width as usize;
    let h = z.metrics.height as usize;
    let n = w * h;
    let mut out = z.to_dense();
    if t <= 1e-6 {
        return out;
    }

    // a[i] = K * A^m * softness, with Tzathas slope correction hint.
    let mut a = vec![0.0f32; n];
    let mut advects = vec![false; n];
    for idx in 0..n {
        let (i, j) = (idx % w, idx / w);
        let soft = 1.0 - hardness.get(i as u32, j as u32).clamp(0.0, 1.0);
        let area = (cache.accumulation[idx] * cell_area * area_scale).max(cell_area);
        let mut ai = k * area.powf(m_exp) * soft;
        // Slope correction: a ← a * δx / |z - zr| * ||∇z||  (stabilises axis bias)
        if let Some(r) = cache
            .receiver
            .get(idx)
            .copied()
            .filter(|&r| r != usize::MAX)
        {
            let (ri, rj) = (r % w, r / w);
            let dz_rec = (z.get(i as u32, j as u32) - z.get(ri as u32, rj as u32))
                .abs()
                .max(1e-4);
            let grad = local_grad_mag(z, i, j).max(1e-4);
            let corr = (cell_len * grad / dz_rec).clamp(0.25, 4.0);
            ai *= corr;
        }
        let characteristic_distance = ai * t;
        let resolvable = cell_len * MIN_CHARACTERISTIC_CELL_FRACTION;
        if ai.is_finite() && ai > 0.0 && characteristic_distance >= resolvable {
            a[idx] = ai;
            advects[idx] = true;
        }
    }

    // Non-advecting cells are local characteristic roots. Active upstream
    // cells may evolve toward them, but no path integral crosses through a
    // zero-speed cell.
    let mut transport_receiver = cache.receiver.clone();
    for idx in 0..n {
        if !advects[idx] {
            transport_receiver[idx] = usize::MAX;
        }
    }

    // Path integrals along each root→leaf chain using recursive Tzathas updates.
    // Process in downstream→upstream order so receivers are known.
    let mut travel = vec![0.0f32; n]; // T(0, x) cumulative time from outlet
    let mut s_uplift = vec![0.0f32; n]; // S integral U/a from outlet

    for &idx in &cache.topo_down_to_up {
        let (i, j) = (idx % w, idx / w);
        if stride > 1 && (i % stride != 0 || j % stride != 0) {
            continue;
        }
        if elevation_locks.get(i as u32, j as u32) > 0.5 || transport_receiver[idx] == usize::MAX {
            travel[idx] = 0.0;
            s_uplift[idx] = 0.0;
            out[idx] = z0.get(i as u32, j as u32);
            continue;
        }
        let r = transport_receiver[idx];
        let dist = receiver_distance(i, j, r, w, cell_len);
        travel[idx] = travel[r] + dist / a[idx];
        s_uplift[idx] = s_uplift[r] + dist * uplift.get(i as u32, j as u32) / a[idx];
    }

    for &idx in &cache.topo_down_to_up {
        let (i, j) = (idx % w, idx / w);
        if stride > 1 && (i % stride != 0 || j % stride != 0) {
            continue;
        }
        if elevation_locks.get(i as u32, j as u32) > 0.5 {
            out[idx] = z0.get(i as u32, j as u32);
            continue;
        }
        if !advects[idx] {
            out[idx] = z0.get(i as u32, j as u32);
            continue;
        }

        // Find characteristic origin D: walk downstream until remaining time budget is spent.
        let (z_d, s_from_d) = characteristic_elevation(
            idx,
            t,
            &transport_receiver,
            &travel,
            &s_uplift,
            &a,
            z0,
            uplift,
            w,
            cell_len,
        );
        out[idx] = z_d + s_from_d;

        // Enforce downhill consistency toward receiver (removes basin-boundary cliffs).
        if transport_receiver[idx] != usize::MAX {
            let r = transport_receiver[idx];
            let min_drop = (uplift.get(i as u32, j as u32) / a[idx]) * cell_len * 0.15;
            let floor = out[r] + min_drop.max(1e-4);
            if out[idx] < floor {
                out[idx] = floor;
            }
        }
    }

    out
}

fn characteristic_elevation(
    idx: usize,
    t: f32,
    receiver: &[usize],
    travel: &[f32],
    s_uplift: &[f32],
    a: &[f32],
    z0: &Heightfield,
    uplift: &MaskField,
    w: usize,
    cell_len: f32,
) -> (f32, f32) {
    // If travel time from outlet already ≤ t, steady-state: D = 0.
    if travel[idx] <= t {
        // Walk to outlet for z0(0).
        let mut c = idx;
        while receiver[c] != usize::MAX {
            c = receiver[c];
        }
        let (oi, oj) = (c % w, c / w);
        return (z0.get(oi as u32, oj as u32), s_uplift[idx]);
    }

    // Need D such that T(D, idx) = t  ⇒  travel[idx] - travel[D] = t
    // Walk downstream collecting path, find segment where cumulative from idx hits t.
    let target_travel = travel[idx] - t;
    let mut path = Vec::new();
    let mut c = idx;
    path.push(c);
    while receiver[c] != usize::MAX {
        c = receiver[c];
        path.push(c);
        if travel[c] <= target_travel {
            break;
        }
    }
    // path[0] = idx, path[last] ≈ at or past D
    let last = *path.last().unwrap_or(&idx);
    let (li, lj) = (last % w, last / w);
    let z_d = z0.get(li as u32, lj as u32);

    // S(D, idx) ≈ s_uplift[idx] - s_uplift[D]
    let s = (s_uplift[idx] - s_uplift[last]).max(0.0);

    // Sub-cell refinement: if we overshot, blend along the last edge.
    if path.len() >= 2 {
        let upstream = path[path.len() - 2];
        let t_up = travel[upstream];
        let t_dn = travel[last];
        if t_up > target_travel && t_dn <= target_travel && (t_up - t_dn) > 1e-9 {
            let f = ((target_travel - t_dn) / (t_up - t_dn)).clamp(0.0, 1.0);
            let (ui, uj) = (upstream % w, upstream / w);
            let z_blend =
                z0.get(li as u32, lj as u32) * (1.0 - f) + z0.get(ui as u32, uj as u32) * f;
            let s_edge = cell_len * uplift.get(ui as u32, uj as u32) / a[upstream];
            let s_blend = (s_uplift[idx] - s_uplift[upstream]) + s_edge * (1.0 - f);
            return (z_blend, s_blend.max(0.0));
        }
    }

    (z_d, s)
}

fn finite_horizon_bounds(
    initial: &Heightfield,
    tectonic_base: &Heightfield,
    uplift: &MaskField,
    t: f32,
) -> FiniteHorizonBounds {
    let (initial_min, initial_max) = initial.min_max();
    let (tectonic_min, tectonic_max) = tectonic_base.min_max();
    let max_uplift = uplift
        .data()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0f32, f32::max)
        .max(0.0);
    let uplift_budget = (max_uplift * t.max(0.0)).max(0.0);
    let raw_min = initial_min.min(tectonic_min);
    let raw_max = initial_max.max(tectonic_max) + uplift_budget;
    let tolerance = (raw_max - raw_min).abs().max(1.0) * 1e-4;
    FiniteHorizonBounds {
        min: raw_min - tolerance,
        max: raw_max + tolerance,
    }
}

fn guarded_candidate(
    candidate: f32,
    fallback: f32,
    bounds: FiniteHorizonBounds,
    rejected: &mut usize,
) -> f32 {
    if candidate.is_finite() && candidate >= bounds.min && candidate <= bounds.max {
        candidate
    } else {
        *rejected += 1;
        if fallback.is_finite() {
            fallback
        } else {
            bounds.min
        }
    }
}

fn receiver_distance(i: usize, j: usize, r: usize, w: usize, cell_len: f32) -> f32 {
    let (ri, rj) = (r % w, r / w);
    let di = ri as i32 - i as i32;
    let dj = rj as i32 - j as i32;
    let diag = di != 0 && dj != 0;
    if diag {
        cell_len * std::f32::consts::SQRT_2
    } else {
        cell_len
    }
}

fn local_grad_mag(z: &Heightfield, i: usize, j: usize) -> f32 {
    let dx = z.metrics.dx().max(1e-3);
    let dz = z.metrics.dz().max(1e-3);
    let gx = (z.get_clamped(i as i32 + 1, j as i32) - z.get_clamped(i as i32 - 1, j as i32))
        / (2.0 * dx);
    let gz = (z.get_clamped(i as i32, j as i32 + 1) - z.get_clamped(i as i32, j as i32 - 1))
        / (2.0 * dz);
    gx.hypot(gz)
}

fn bilinear_fill_stride(z: &mut Heightfield, stride: usize) {
    if stride <= 1 {
        return;
    }
    let w = z.metrics.width as usize;
    let h = z.metrics.height as usize;
    let dense = z.to_dense();
    for j in 0..h {
        for i in 0..w {
            if i % stride == 0 && j % stride == 0 {
                continue;
            }
            let i0 = (i / stride) * stride;
            let j0 = (j / stride) * stride;
            let i1 = (i0 + stride).min(w.saturating_sub(1));
            let j1 = (j0 + stride).min(h.saturating_sub(1));
            let fx = if i1 > i0 {
                (i - i0) as f32 / (i1 - i0) as f32
            } else {
                0.0
            };
            let fy = if j1 > j0 {
                (j - j0) as f32 / (j1 - j0) as f32
            } else {
                0.0
            };
            let v00 = dense[j0 * w + i0];
            let v10 = dense[j0 * w + i1];
            let v01 = dense[j1 * w + i0];
            let v11 = dense[j1 * w + i1];
            let v0 = v00 + (v10 - v00) * fx;
            let v1 = v01 + (v11 - v01) * fx;
            z.set(i as u32, j as u32, v0 + (v1 - v0) * fy);
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
