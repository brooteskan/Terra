use super::{canonical_seed32, hash2};

/// OpenSimplex-style 2D noise (simplified skew lattice). Approx \[-1, 1\].
pub fn open_simplex2(x: f32, z: f32, seed: u64) -> f32 {
    const F2: f32 = 0.366_025_4; // (sqrt(3)-1)/2
    const G2: f32 = 0.211_324_87; // (3-sqrt(3))/6

    let seed = canonical_seed32(seed);
    let s = (x + z) * F2;
    let i = (x + s).floor();
    let j = (z + s).floor();
    let t = (i + j) * G2;
    let x0 = x - (i - t);
    let z0 = z - (j - t);

    let (i1, j1) = if x0 > z0 { (1.0, 0.0) } else { (0.0, 1.0) };
    let x1 = x0 - i1 + G2;
    let z1 = z0 - j1 + G2;
    let x2 = x0 - 1.0 + 2.0 * G2;
    let z2 = z0 - 1.0 + 2.0 * G2;

    let ii = i as i32;
    let jj = j as i32;

    let n0 = contrib(hash2(ii, jj, seed), x0, z0);
    let n1 = contrib(hash2(ii + i1 as i32, jj + j1 as i32, seed), x1, z1);
    let n2 = contrib(hash2(ii + 1, jj + 1, seed), x2, z2);

    (n0 + n1 + n2) * 40.0
}

fn contrib(h: u32, x: f32, z: f32) -> f32 {
    let t = 0.5 - x * x - z * z;
    if t < 0.0 {
        0.0
    } else {
        let t2 = t * t;
        t2 * t2 * grad(h, x, z)
    }
}

fn grad(h: u32, x: f32, z: f32) -> f32 {
    match h & 7 {
        0 => x + z,
        1 => -x + z,
        2 => x - z,
        3 => -x - z,
        4 => x,
        5 => -x,
        6 => z,
        _ => -z,
    }
}
