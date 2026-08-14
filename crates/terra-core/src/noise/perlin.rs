use super::{canonical_seed32, fade, hash2, lerp};

/// Classic gradient (Perlin-style) noise in roughly \[-1, 1\].
pub fn perlin2(x: f32, z: f32, seed: u64) -> f32 {
    let seed = canonical_seed32(seed);
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let fx = x - x0 as f32;
    let fz = z - z0 as f32;
    let u = fade(fx);
    let v = fade(fz);

    let g00 = grad(hash2(x0, z0, seed), fx, fz);
    let g10 = grad(hash2(x0 + 1, z0, seed), fx - 1.0, fz);
    let g01 = grad(hash2(x0, z0 + 1, seed), fx, fz - 1.0);
    let g11 = grad(hash2(x0 + 1, z0 + 1, seed), fx - 1.0, fz - 1.0);

    let nx0 = lerp(g00, g10, u);
    let nx1 = lerp(g01, g11, u);
    // Scale to approx [-1,1]
    lerp(nx0, nx1, v) * std::f32::consts::SQRT_2
}

fn grad(h: u32, x: f32, z: f32) -> f32 {
    match h & 3 {
        0 => x + z,
        1 => -x + z,
        2 => x - z,
        _ => -x - z,
    }
}
