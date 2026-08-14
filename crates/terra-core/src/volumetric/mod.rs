//! Phase J — optional local volumetric geology (dual-height overhang + local SDF).
//!
//! Heightfields remain the primary authoring and erosion domain. These helpers produce a
//! **dual-height** representation: a carved floor DEM plus a ceiling aux map, limited to a
//! UV region / falloff. A lightweight triangle proxy visualizes the overhang / cave roof.

use crate::heightfield::Heightfield;
use crate::layer::{LocalSdfParams, OverhangStampParams};
use crate::mask::MaskField;
use crate::noise::{canonical_seed32, value_noise2};

/// Result of an opt-in volumetric stamp: carved floor + ceiling dual-height + region mask.
#[derive(Debug, Clone)]
pub struct DualHeightResult {
    /// Heightfield with cavity / undercut floor applied (DEM domain for downstream erosion).
    pub height: Heightfield,
    /// Ceiling height in meters. Equals floor outside the volumetric region.
    pub ceiling: MaskField,
    /// \[0,1\] strength of the volumetric effect (region limiter).
    pub mask: MaskField,
}

/// CPU triangle proxy for viewport / optional mesh export (not a full MC extract).
#[derive(Debug, Clone, Default)]
pub struct OverhangMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

impl OverhangMesh {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Cliff undercut / shelf stamp: lower the DEM floor inside a UV disk while keeping a ceiling.
///
/// The entrance side (controlled by [`OverhangStampParams::entrance_angle_deg`]) opens so the
/// cavity reads as an overhang from the side rather than a closed pit.
pub fn apply_overhang_stamp(hf: &Heightfield, p: &OverhangStampParams) -> DualHeightResult {
    let m = hf.metrics;
    let mut out = hf.clone();
    let mut ceiling_data = vec![0.0f32; (m.width * m.height) as usize];
    let mut mask = MaskField::zeros(m);
    // Seed ceiling with current surface (meters — bypass MaskField \[0,1\] clamp).
    for j in 0..m.height {
        for i in 0..m.width {
            ceiling_data[(j * m.width + i) as usize] = hf.get(i, j);
        }
    }

    let radius = p.radius_uv.max(1e-4);
    let depth = p.depth.max(0.0);
    let lip = p.lip_height.max(0.0);
    let falloff = p.falloff.clamp(0.05, 1.0);
    let noise_amp = p.noise_amplitude.clamp(0.0, 1.0);
    let seed_phase = canonical_seed32(p.seed) as f32;
    let ent = p.entrance_angle_deg.to_radians();
    let ent_dir = (ent.cos(), ent.sin());

    for j in 0..m.height {
        for i in 0..m.width {
            let u = (i as f32 + 0.5) / m.width as f32;
            let v = (j as f32 + 0.5) / m.height as f32;
            let du = (u - p.u) / radius;
            let dv = (v - p.v) / radius;
            let r2 = du * du + dv * dv;
            if r2 >= 1.0 {
                continue;
            }
            let r = r2.sqrt();
            // Smooth radial weight; `falloff` widens the soft rim.
            let soft = ((1.0 - r) / falloff).clamp(0.0, 1.0);
            let soft = soft * soft * (3.0 - 2.0 * soft);
            // Entrance opening fades only near the rim toward the entrance azimuth.
            // Stamp center always keeps a full cavity so dual-height is well-defined.
            let entrance_open = if r > 0.15 {
                let len = r.max(1e-6);
                let ndu = du / len;
                let ndv = dv / len;
                let toward_entrance = ndu * ent_dir.0 + ndv * ent_dir.1;
                let rim = ((r - 0.15) / 0.85).clamp(0.0, 1.0);
                ((toward_entrance + 0.1) * 1.35).clamp(0.0, 1.0) * rim
            } else {
                0.0
            };
            let cavity = soft * (1.0 - entrance_open * 0.92);
            if cavity < 1e-4 {
                continue;
            }

            let h0 = hf.get(i, j);
            let n = if noise_amp > 1e-5 {
                value_noise2(u * 48.0 + seed_phase * 0.01, v * 48.0, p.seed) * 0.5 + 0.5
            } else {
                0.5
            };
            let depth_local = depth * cavity * (1.0 - noise_amp * 0.45 + noise_amp * n * 0.9);
            let floor_h = h0 - depth_local;
            // Lip brow on the back (anti-entrance) rim.
            let toward_entrance = if r > 1e-5 {
                let len = r.max(1e-6);
                (du / len) * ent_dir.0 + (dv / len) * ent_dir.1
            } else {
                -1.0
            };
            let brow = if toward_entrance < -0.2 {
                lip * soft * (-toward_entrance).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let ceil_h = h0 + brow;

            out.set(i, j, floor_h);
            ceiling_data[(j * m.width + i) as usize] = ceil_h.max(floor_h + 0.05);
            mask.set(i, j, cavity.clamp(0.0, 1.0));
        }
    }

    DualHeightResult {
        height: out,
        ceiling: MaskField::from_raw(m, &ceiling_data),
        mask,
    }
}

/// Local analytic SDF cave pocket (ellipsoid chamber + entrance tunnel).
///
/// Guidance: GPU Gems 3 density / MC ideas — here we only evaluate a **local low-res analytic
/// SDF** and project void intervals onto dual-height (no full-world marching cubes).
pub fn apply_local_sdf(hf: &Heightfield, p: &LocalSdfParams) -> DualHeightResult {
    let m = hf.metrics;
    let mut out = hf.clone();
    let mut ceiling_data = vec![0.0f32; (m.width * m.height) as usize];
    let mut mask = MaskField::zeros(m);
    for j in 0..m.height {
        for i in 0..m.width {
            ceiling_data[(j * m.width + i) as usize] = hf.get(i, j);
        }
    }

    let wx = m.world_size_x.max(1.0);
    let wz = m.world_size_z.max(1.0);
    let cx = p.u.clamp(0.0, 1.0) * wx;
    let cz = p.v.clamp(0.0, 1.0) * wz;
    // Sample surface height at center for vertical placement.
    let (ci, cj) = m.sample_index(cx, cz);
    let surface = hf.get(ci, cj);
    let depth = p.depth.max(0.5);
    let chamber_y = surface - depth * 0.55;
    let rx = (p.size_x.max(1.0)) * 0.5;
    let ry = (p.size_y.max(0.5)) * 0.5;
    let rz = (p.size_z.max(1.0)) * 0.5;
    let entrance = p.entrance_radius.max(0.5);
    let ent = p.entrance_angle_deg.to_radians();
    let ent_dir = (ent.cos(), ent.sin());
    // Bounding UV box (with margin for tunnel).
    let margin = (rx.max(rz) + entrance * 2.0 + p.size_x * 0.25) / wx.min(wz);
    let u0 = (p.u - margin).clamp(0.0, 1.0);
    let u1 = (p.u + margin).clamp(0.0, 1.0);
    let v0 = (p.v - margin).clamp(0.0, 1.0);
    let v1 = (p.v + margin).clamp(0.0, 1.0);
    let i0 = (u0 * m.width as f32).floor() as u32;
    let i1 = ((u1 * m.width as f32).ceil() as u32).min(m.width);
    let j0 = (v0 * m.height as f32).floor() as u32;
    let j1 = ((v1 * m.height as f32).ceil() as u32).min(m.height);

    let y_samples = p.vertical_samples.max(4).min(64);
    let noise_amp = p.noise_amplitude.clamp(0.0, 1.0);

    for j in j0..j1 {
        for i in i0..i1 {
            let x = m.world_x(i);
            let z = m.world_z(j);
            let h0 = hf.get(i, j);
            let y_top = h0 + p.lip_height.max(0.0);
            let y_bot = h0 - depth - ry;

            let mut void_lo = f32::INFINITY;
            let mut void_hi = f32::NEG_INFINITY;
            for s in 0..y_samples {
                let t = s as f32 / (y_samples - 1) as f32;
                let y = y_bot + (y_top - y_bot) * t;
                let d = cave_sdf(
                    x, y, z, cx, chamber_y, cz, rx, ry, rz, entrance, ent_dir, depth, surface,
                    p.seed, noise_amp,
                );
                if d < 0.0 {
                    void_lo = void_lo.min(y);
                    void_hi = void_hi.max(y);
                }
            }

            if void_lo.is_finite() && void_hi > void_lo + 0.05 && void_lo < h0 {
                let floor_h = void_lo.clamp(y_bot, h0);
                let ceil_h = void_hi.min(h0 + p.lip_height.max(0.0)).max(floor_h + 0.25);
                let strength = ((h0 - floor_h) / depth.max(1e-3)).clamp(0.0, 1.0);
                out.set(i, j, floor_h);
                ceiling_data[(j * m.width + i) as usize] = ceil_h;
                mask.set(i, j, strength);
            }
        }
    }

    DualHeightResult {
        height: out,
        ceiling: MaskField::from_raw(m, &ceiling_data),
        mask,
    }
}

/// Analytic cave SDF: ellipsoid chamber union soft entrance tunnel toward `ent_dir`.
///
/// Negative = void. Noise warps the shell slightly for organic walls (deterministic seed).
fn cave_sdf(
    x: f32,
    y: f32,
    z: f32,
    cx: f32,
    cy: f32,
    cz: f32,
    rx: f32,
    ry: f32,
    rz: f32,
    entrance_r: f32,
    ent_dir: (f32, f32),
    depth: f32,
    surface: f32,
    seed: u64,
    noise_amp: f32,
) -> f32 {
    // Ellipsoid chamber.
    let qx = (x - cx) / rx.max(1e-3);
    let qy = (y - cy) / ry.max(1e-3);
    let qz = (z - cz) / rz.max(1e-3);
    let d_chamber = (qx * qx + qy * qy + qz * qz).sqrt() - 1.0;

    // Entrance tunnel: horizontal capsule from chamber toward entrance azimuth.
    let tunnel_len = rx.max(rz) + entrance_r * 1.5;
    let ex = cx + ent_dir.0 * tunnel_len;
    let ez = cz + ent_dir.1 * tunnel_len;
    let ey = surface - depth * 0.35;
    let d_tunnel = capsule_xz(x, y, z, cx, cy, cz, ex, ey, ez, entrance_r);

    let mut d = soft_min(d_chamber, d_tunnel, 1.5);

    if noise_amp > 1e-5 {
        let n = value_noise2(
            x * 0.08 + canonical_seed32(seed) as f32 * 0.001,
            z * 0.08 + y * 0.05,
            seed.wrapping_add(17),
        );
        d += n * noise_amp * entrance_r * 0.35;
    }
    d
}

fn capsule_xz(
    x: f32,
    y: f32,
    z: f32,
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    r: f32,
) -> f32 {
    let pax = x - ax;
    let pay = y - ay;
    let paz = z - az;
    let bax = bx - ax;
    let bay = by - ay;
    let baz = bz - az;
    let baba = bax * bax + bay * bay + baz * baz;
    let paba = pax * bax + pay * bay + paz * baz;
    let h = if baba > 1e-8 {
        (paba / baba).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let dx = pax - bax * h;
    let dy = pay - bay * h;
    let dz = paz - baz * h;
    (dx * dx + dy * dy + dz * dz).sqrt() - r
}

fn soft_min(a: f32, b: f32, k: f32) -> f32 {
    let k = k.max(1e-4);
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    a * h + b * (1.0 - h) - k * h * (1.0 - h)
}

/// Build a lightweight ceiling + wall mesh from dual-height fields.
///
/// Only cells with `mask >= threshold` and `ceiling > floor + eps` emit geometry. This is a
/// dual-surface proxy, not marching cubes (deferred).
pub fn build_overhang_mesh(
    floor: &Heightfield,
    ceiling: &MaskField,
    mask: &MaskField,
    threshold: f32,
) -> OverhangMesh {
    let m = floor.metrics;
    let thr = threshold.clamp(0.0, 1.0);
    let eps = 0.08;
    let roof_color = [0.55, 0.48, 0.40, 0.92];
    let wall_color = [0.42, 0.38, 0.34, 0.95];

    let mut mesh = OverhangMesh::default();
    if m.width < 2 || m.height < 2 {
        return mesh;
    }

    let active = |i: u32, j: u32| -> bool {
        mask.get(i, j) >= thr && ceiling.get(i, j) > floor.get(i, j) + eps
    };

    // Roof underside quads (normals face down).
    for j in 0..m.height - 1 {
        for i in 0..m.width - 1 {
            if !(active(i, j) && active(i + 1, j) && active(i, j + 1) && active(i + 1, j + 1)) {
                continue;
            }
            let p00 = [m.world_x(i), ceiling.get(i, j), m.world_z(j)];
            let p10 = [m.world_x(i + 1), ceiling.get(i + 1, j), m.world_z(j)];
            let p01 = [m.world_x(i), ceiling.get(i, j + 1), m.world_z(j + 1)];
            let p11 = [
                m.world_x(i + 1),
                ceiling.get(i + 1, j + 1),
                m.world_z(j + 1),
            ];
            push_quad(&mut mesh, p00, p01, p11, p10, [0.0, -1.0, 0.0], roof_color);
        }
    }

    // Vertical walls where overhang meets open air (mask edge).
    let neighbors = [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)];
    for j in 0..m.height {
        for i in 0..m.width {
            if !active(i, j) {
                continue;
            }
            let ceil_y = ceiling.get(i, j);
            let floor_y = floor.get(i, j);
            let x = m.world_x(i);
            let z = m.world_z(j);
            for (di, dj) in neighbors {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= m.width as i32 || nj >= m.height as i32 {
                    continue;
                }
                let ni = ni as u32;
                let nj = nj as u32;
                if active(ni, nj) {
                    continue;
                }
                // Wall faces outward toward inactive neighbor.
                let nx = di as f32;
                let nz = dj as f32;
                let len = (nx * nx + nz * nz).sqrt().max(1e-3);
                let normal = [nx / len, 0.0, nz / len];
                let hx = m.dx() * 0.5 * di as f32;
                let hz = m.dz() * 0.5 * dj as f32;
                // Quad in the plane between cell centers, offset halfway to neighbor.
                let ox = x + hx;
                let oz = z + hz;
                let along_x = if di == 0 { m.dx() * 0.5 } else { 0.0 };
                let along_z = if dj == 0 { m.dz() * 0.5 } else { 0.0 };
                let c0 = [ox - along_x, floor_y, oz - along_z];
                let c1 = [ox + along_x, floor_y, oz + along_z];
                let c2 = [ox + along_x, ceil_y, oz + along_z];
                let c3 = [ox - along_x, ceil_y, oz - along_z];
                push_quad(&mut mesh, c0, c1, c2, c3, normal, wall_color);
            }
        }
    }

    mesh
}

fn push_quad(
    mesh: &mut OverhangMesh,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    d: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
) {
    let base = mesh.positions.len() as u32;
    for p in [a, b, c, d] {
        mesh.positions.push(p);
        mesh.normals.push(normal);
        mesh.colors.push(color);
    }
    // Two triangles: a-b-c, a-c-d
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;
    use crate::layer::{LocalSdfParams, OverhangStampParams};

    fn cliff_plateau(res: u32) -> Heightfield {
        let m = HeightfieldMetrics::new(res, res, res as f32 * 2.0, res as f32 * 2.0);
        let mut hf = Heightfield::filled(m, 10.0);
        for j in 0..res {
            for i in 0..res {
                // High plateau on +X half — cliff near mid.
                if i > res / 2 {
                    hf.set(i, j, 40.0);
                }
            }
        }
        hf
    }

    #[test]
    fn overhang_deterministic_and_localized() {
        let hf = cliff_plateau(48);
        let p = OverhangStampParams {
            u: 0.55,
            v: 0.5,
            radius_uv: 0.12,
            depth: 12.0,
            lip_height: 1.5,
            entrance_angle_deg: 180.0,
            falloff: 0.35,
            seed: 42,
            noise_amplitude: 0.2,
        };
        let a = apply_overhang_stamp(&hf, &p);
        let b = apply_overhang_stamp(&hf, &p);
        assert_eq!(a.height.to_dense(), b.height.to_dense());
        assert_eq!(a.ceiling.data(), b.ceiling.data());
        assert_eq!(a.mask.data(), b.mask.data());

        // Far from stamp: height unchanged.
        assert!((a.height.get(2, 2) - hf.get(2, 2)).abs() < 1e-5);
        assert!(a.mask.get(2, 2) < 1e-5);

        // Inside stamp on plateau: floor lowered, ceiling above floor.
        let (ci, cj) = hf
            .metrics
            .sample_index(p.u * hf.metrics.world_size_x, p.v * hf.metrics.world_size_z);
        assert!(a.mask.get(ci, cj) > 0.05);
        assert!(a.height.get(ci, cj) < hf.get(ci, cj) - 1.0);
        assert!(a.ceiling.get(ci, cj) > a.height.get(ci, cj) + 0.5);
    }

    #[test]
    fn overhang_high_seed_bits_change_the_carved_field() {
        let hf = cliff_plateau(48);
        let low = OverhangStampParams {
            u: 0.55,
            v: 0.5,
            radius_uv: 0.12,
            depth: 12.0,
            lip_height: 1.5,
            entrance_angle_deg: 180.0,
            falloff: 0.35,
            seed: 7,
            noise_amplitude: 0.4,
        };
        let high = OverhangStampParams {
            seed: low.seed + (1u64 << 32),
            ..low.clone()
        };

        assert_ne!(
            apply_overhang_stamp(&hf, &low).height.to_dense(),
            apply_overhang_stamp(&hf, &high).height.to_dense()
        );
    }

    #[test]
    fn local_sdf_region_limited() {
        let hf = cliff_plateau(40);
        let p = LocalSdfParams {
            u: 0.62,
            v: 0.5,
            size_x: 18.0,
            size_y: 10.0,
            size_z: 14.0,
            depth: 14.0,
            entrance_radius: 4.0,
            entrance_angle_deg: 180.0,
            lip_height: 0.5,
            seed: 7,
            noise_amplitude: 0.15,
            vertical_samples: 24,
        };
        let r = apply_local_sdf(&hf, &p);
        assert!((r.height.get(1, 1) - hf.get(1, 1)).abs() < 1e-5);
        assert!(r.mask.get(1, 1) < 1e-5);
        // Some cell near center should carve.
        let carved = r.mask.data().iter().any(|&v| v > 0.1);
        assert!(carved, "expected a carved cave pocket");
        let mesh = build_overhang_mesh(&r.height, &r.ceiling, &r.mask, 0.15);
        assert!(!mesh.is_empty());
    }
}
