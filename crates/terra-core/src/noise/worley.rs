use super::{canonical_seed32, hash2, WorleyMetric};

pub struct WorleyResult {
    pub f1: f32,
    pub f2: f32,
}

/// Cellular / Worley noise: two nearest feature distances.
pub fn worley2(x: f32, z: f32, seed: u64, metric: WorleyMetric) -> WorleyResult {
    let seed = canonical_seed32(seed);
    let xi = x.floor() as i32;
    let zi = z.floor() as i32;
    let mut f1 = f32::INFINITY;
    let mut f2 = f32::INFINITY;

    for dz in -1..=1 {
        for dx in -1..=1 {
            let cx = xi + dx;
            let cz = zi + dz;
            let h = hash2(cx, cz, seed);
            let fx = cx as f32 + (h as f32 / u32::MAX as f32);
            let hz = hash2(cx, cz, seed ^ 0xA5A5A5A5);
            let fz = cz as f32 + (hz as f32 / u32::MAX as f32);
            let d = dist(metric, x - fx, z - fz);
            if d < f1 {
                f2 = f1;
                f1 = d;
            } else if d < f2 {
                f2 = d;
            }
        }
    }

    WorleyResult { f1, f2 }
}

fn dist(metric: WorleyMetric, dx: f32, dz: f32) -> f32 {
    match metric {
        WorleyMetric::Euclidean => (dx * dx + dz * dz).sqrt(),
        WorleyMetric::Manhattan => dx.abs() + dz.abs(),
        WorleyMetric::Chebyshev => dx.abs().max(dz.abs()),
    }
}
