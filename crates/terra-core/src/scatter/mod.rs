//! Poisson-disk / blue-noise sampling for vegetation.

mod params;

pub use params::VegetationParams;

use crate::heightfield::Heightfield;
use crate::spatial_kernels::slope_degrees;

/// Bridson-style Poisson disk in world XZ, filtered by slope/biome proxies.
pub fn poisson_disk(hf: &Heightfield, p: &VegetationParams) -> Vec<(f32, f32)> {
    let min_dist = p.min_distance.max(0.5);
    let cell = min_dist / std::f32::consts::SQRT_2;
    let gw = (hf.metrics.world_size_x / cell).ceil() as usize;
    let gh = (hf.metrics.world_size_z / cell).ceil() as usize;
    let mut grid = vec![None::<(f32, f32)>; gw * gh];
    let mut points = Vec::new();
    let mut active = Vec::new();
    let slope = slope_degrees(hf);

    let mut rng = Rng::new(p.seed);
    // Seed points proportional to density
    let attempts = ((hf.metrics.world_size_x * hf.metrics.world_size_z) / (min_dist * min_dist)
        * p.density)
        .ceil() as u32;

    for _ in 0..attempts.max(1) {
        let x = rng.f32() * hf.metrics.world_size_x;
        let z = rng.f32() * hf.metrics.world_size_z;
        if !accepted(hf, &slope, p, x, z) {
            continue;
        }
        if try_add(x, z, min_dist, cell, gw, gh, &mut grid, &mut points) {
            active.push(points.len() - 1);
        }
    }

    while let Some(idx) = active.pop() {
        let (px, pz) = points[idx];
        for _ in 0..30 {
            let ang = rng.f32() * std::f32::consts::TAU;
            let rad = min_dist * (1.0 + rng.f32());
            let x = px + ang.cos() * rad;
            let z = pz + ang.sin() * rad;
            if x < 0.0 || z < 0.0 || x >= hf.metrics.world_size_x || z >= hf.metrics.world_size_z {
                continue;
            }
            if !accepted(hf, &slope, p, x, z) {
                continue;
            }
            if try_add(x, z, min_dist, cell, gw, gh, &mut grid, &mut points) {
                active.push(points.len() - 1);
            }
        }
    }

    points
}

fn accepted(
    hf: &Heightfield,
    slope: &crate::mask::MaskField,
    p: &VegetationParams,
    x: f32,
    z: f32,
) -> bool {
    let (i, j) = hf.metrics.sample_index(x, z);
    let s = slope.get(i, j) * 90.0;
    s >= p.min_slope_deg && s <= p.max_slope_deg
}

fn try_add(
    x: f32,
    z: f32,
    min_dist: f32,
    cell: f32,
    gw: usize,
    gh: usize,
    grid: &mut [Option<(f32, f32)>],
    points: &mut Vec<(f32, f32)>,
) -> bool {
    let gx = (x / cell) as isize;
    let gz = (z / cell) as isize;
    for dj in -2..=2 {
        for di in -2..=2 {
            let cx = gx + di;
            let cz = gz + dj;
            if cx < 0 || cz < 0 || cx >= gw as isize || cz >= gh as isize {
                continue;
            }
            if let Some((px, pz)) = grid[cz as usize * gw + cx as usize] {
                let dx = px - x;
                let dz = pz - z;
                if dx * dx + dz * dz < min_dist * min_dist {
                    return false;
                }
            }
        }
    }
    if gx >= 0 && gz >= 0 && (gx as usize) < gw && (gz as usize) < gh {
        grid[gz as usize * gw + gx as usize] = Some((x, z));
    }
    points.push((x, z));
    true
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

    #[test]
    fn poisson_min_distance() {
        let m = HeightfieldMetrics::new(64, 64, 128.0, 128.0);
        let hf = Heightfield::filled(m, 10.0);
        let p = VegetationParams {
            seed: 7,
            density: 0.4,
            min_distance: 5.0,
            min_slope_deg: 0.0,
            max_slope_deg: 90.0,
            biome_id: None,
            root_cohesion: 0.0,
            coverage: Default::default(),
            ..VegetationParams::default()
        };
        let pts = poisson_disk(&hf, &p);
        assert!(!pts.is_empty());
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let dx = pts[i].0 - pts[j].0;
                let dz = pts[i].1 - pts[j].1;
                assert!(dx * dx + dz * dz >= 5.0 * 5.0 - 1e-3);
            }
        }
    }

    #[test]
    fn deterministic_seed() {
        let m = HeightfieldMetrics::new(32, 32, 64.0, 64.0);
        let hf = Heightfield::filled(m, 5.0);
        let p = VegetationParams {
            seed: 99,
            density: 0.3,
            min_distance: 4.0,
            ..VegetationParams::default()
        };
        let a = poisson_disk(&hf, &p);
        let b = poisson_disk(&hf, &p);
        assert_eq!(a, b);
    }
}
