use super::{canonical_seed32, fade, hash2, lerp};

/// Value noise in roughly \[-1, 1\].
pub fn value_noise2(x: f32, z: f32, seed: u64) -> f32 {
    let seed = canonical_seed32(seed);
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let fx = x - x0 as f32;
    let fz = z - z0 as f32;
    let u = fade(fx);
    let v = fade(fz);

    let n00 = grad_value(hash2(x0, z0, seed));
    let n10 = grad_value(hash2(x0 + 1, z0, seed));
    let n01 = grad_value(hash2(x0, z0 + 1, seed));
    let n11 = grad_value(hash2(x0 + 1, z0 + 1, seed));

    let nx0 = lerp(n00, n10, u);
    let nx1 = lerp(n01, n11, u);
    lerp(nx0, nx1, v)
}

fn grad_value(h: u32) -> f32 {
    // Map to [-1, 1]
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}
