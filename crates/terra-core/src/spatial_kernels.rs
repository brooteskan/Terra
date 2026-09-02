//! Reusable CPU spatial kernels shared by analysis, field caches, and mask execution.

use crate::heightfield::Heightfield;
use crate::mask_field::MaskField;

/// Slope in degrees from central differences, normalized to `[0, 1]` by `/ 90`.
pub fn slope_degrees(hf: &Heightfield) -> MaskField {
    let m = hf.metrics;
    let mut out = MaskField::zeros(m);
    let dx = m.dx();
    let dz = m.dz();
    for j in 0..m.height {
        for i in 0..m.width {
            let (h_l, h_r, h_d, h_u) = neighbors(hf, i, j);
            let gx = (h_r - h_l) / (2.0 * dx);
            let gz = (h_u - h_d) / (2.0 * dz);
            let slope = (gx * gx + gz * gz).sqrt().atan().to_degrees();
            out.set(i, j, (slope / 90.0).clamp(0.0, 1.0));
        }
    }
    out
}

pub fn aspect_mask(hf: &Heightfield, center_deg: f32, width_deg: f32) -> MaskField {
    let m = hf.metrics;
    let mut out = MaskField::zeros(m);
    let dx = m.dx();
    let dz = m.dz();
    for j in 0..m.height {
        for i in 0..m.width {
            let (h_l, h_r, h_d, h_u) = neighbors(hf, i, j);
            let gx = (h_r - h_l) / (2.0 * dx);
            let gz = (h_u - h_d) / (2.0 * dz);
            let aspect = gz.atan2(-gx).to_degrees().rem_euclid(360.0);
            let diff = angle_diff(aspect, center_deg).abs();
            let t = (1.0 - diff / width_deg.max(1e-3)).clamp(0.0, 1.0);
            out.set(i, j, t);
        }
    }
    out
}

/// Mean curvature approximation normalized to `[0, 1]` around 0.5.
pub fn curvature(hf: &Heightfield) -> MaskField {
    let m = hf.metrics;
    let mut out = MaskField::zeros(m);
    let dx = m.dx();
    for j in 0..m.height {
        for i in 0..m.width {
            let h = hf.get(i, j);
            let (h_l, h_r, h_d, h_u) = neighbors(hf, i, j);
            let lap = (h_l + h_r + h_d + h_u - 4.0 * h) / (dx * dx);
            out.set(i, j, (0.5 + lap * 5.0).clamp(0.0, 1.0));
        }
    }
    out
}

pub fn convexity(hf: &Heightfield) -> MaskField {
    let mut c = curvature(hf);
    for v in c.data_mut() {
        *v = (*v - 0.5).max(0.0) * 2.0;
    }
    c
}

pub fn concavity(hf: &Heightfield) -> MaskField {
    let mut c = curvature(hf);
    for v in c.data_mut() {
        *v = (0.5 - *v).max(0.0) * 2.0;
    }
    c
}

/// Cheap heightfield AO: darken when neighbors are higher within radius.
pub fn ambient_occlusion(hf: &Heightfield, radius: u32, strength: f32) -> MaskField {
    let m = hf.metrics;
    let mut out = MaskField::ones(m);
    let r = radius.max(1) as i32;
    for j in 0..m.height {
        for i in 0..m.width {
            let h0 = hf.get(i, j);
            let mut occ: f32 = 0.0;
            let mut count: f32 = 0.0;
            for dj in -r..=r {
                for di in -r..=r {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    let ii = i as i32 + di;
                    let jj = j as i32 + dj;
                    if ii < 0 || jj < 0 || ii >= m.width as i32 || jj >= m.height as i32 {
                        continue;
                    }
                    let h1 = hf.get(ii as u32, jj as u32);
                    let dist = ((di * di + dj * dj) as f32).sqrt();
                    occ += ((h1 - h0) / dist.max(1.0)).max(0.0);
                    count += 1.0;
                }
            }
            let ao = 1.0 - (occ / count.max(1.0) * strength).clamp(0.0, 1.0);
            out.set(i, j, ao);
        }
    }
    out
}

/// Jump-flood style distance field from seed mask (`> 0.5` is a seed).
pub fn jump_flood_distance(seeds: &MaskField) -> MaskField {
    let m = seeds.metrics;
    let w = m.width as usize;
    let h = m.height as usize;
    let mut nearest: Vec<Option<(i32, i32)>> = vec![None; w * h];
    for j in 0..m.height {
        for i in 0..m.width {
            if seeds.get(i, j) > 0.5 {
                nearest[(j * m.width + i) as usize] = Some((i as i32, j as i32));
            }
        }
    }

    let mut step = (w.max(h) as u32).next_power_of_two() as i32 / 2;
    while step >= 1 {
        let prev = nearest.clone();
        for j in 0..h as i32 {
            for i in 0..w as i32 {
                let mut best = prev[j as usize * w + i as usize];
                let mut best_d = dist2(i, j, best);
                for dj in [-step, 0, step] {
                    for di in [-step, 0, step] {
                        let ni = i + di;
                        let nj = j + dj;
                        if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
                            continue;
                        }
                        let cand = prev[nj as usize * w + ni as usize];
                        let d = dist2(i, j, cand);
                        if d < best_d {
                            best_d = d;
                            best = cand;
                        }
                    }
                }
                nearest[j as usize * w + i as usize] = best;
            }
        }
        step /= 2;
    }

    let mut out = MaskField::zeros(m);
    let max_d = ((w * w + h * h) as f32).sqrt();
    for j in 0..m.height {
        for i in 0..m.width {
            let d = match nearest[(j * m.width + i) as usize] {
                Some((x, y)) => {
                    let dx = i as f32 - x as f32;
                    let dy = j as f32 - y as f32;
                    (dx * dx + dy * dy).sqrt()
                }
                None => max_d,
            };
            out.set(i, j, (d / max_d).clamp(0.0, 1.0));
        }
    }
    out
}

/// Point-in-polygon test in an arbitrary 2D coordinate system.
pub fn point_in_polygon(x: f32, y: f32, pts: &[[f32; 2]]) -> bool {
    if pts.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = pts.len() - 1;
    for i in 0..pts.len() {
        let xi = pts[i][0];
        let yi = pts[i][1];
        let xj = pts[j][0];
        let yj = pts[j][1];
        if ((yi > y) != (yj > y)) && x < (xj - xi) * (y - yi) / (yj - yi).max(1e-12) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Distance between a point and a line segment in 2D.
pub fn dist_point_segment(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let abx = bx - ax;
    let aby = by - ay;
    let apx = px - ax;
    let apy = py - ay;
    let denom = abx * abx + aby * aby;
    let t = if denom > 1e-12 {
        ((apx * abx + apy * aby) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let dx = px - (ax + t * abx);
    let dy = py - (ay + t * aby);
    (dx * dx + dy * dy).sqrt()
}

fn dist2(i: i32, j: i32, p: Option<(i32, i32)>) -> f32 {
    match p {
        Some((x, y)) => {
            let dx = (i - x) as f32;
            let dy = (j - y) as f32;
            dx * dx + dy * dy
        }
        None => f32::INFINITY,
    }
}

fn neighbors(hf: &Heightfield, i: u32, j: u32) -> (f32, f32, f32, f32) {
    let w = hf.metrics.width;
    let h = hf.metrics.height;
    let il = i.saturating_sub(1);
    let ir = (i + 1).min(w - 1);
    let jd = j.saturating_sub(1);
    let ju = (j + 1).min(h - 1);
    (hf.get(il, j), hf.get(ir, j), hf.get(i, jd), hf.get(i, ju))
}

fn angle_diff(a: f32, b: f32) -> f32 {
    let mut d = (a - b) % 360.0;
    if d > 180.0 {
        d -= 360.0;
    }
    if d < -180.0 {
        d += 360.0;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

    #[test]
    fn plane_slope_known() {
        let m = HeightfieldMetrics::new(32, 32, 32.0, 32.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..32 {
            for i in 0..32 {
                hf.set(i, j, i as f32);
            }
        }
        let v = slope_degrees(&hf).get(16, 16);
        assert!((v - 0.5).abs() < 0.05, "slope mask {v}");
    }
}
