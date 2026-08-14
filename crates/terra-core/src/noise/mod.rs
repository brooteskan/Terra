//! Deterministic CPU noise primitives.

mod params;
mod perlin;
mod simplex;
mod value;
mod worley;

pub use params::{FractalNoiseType, NoiseParams, WorleyFeature, WorleyMetric, WorleyParams};
pub use perlin::perlin2;
pub use simplex::open_simplex2;
pub use value::value_noise2;
pub use worley::{worley2, WorleyResult};

fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    x
}

/// Canonicalize a serialized 64-bit procedural seed for the existing 32-bit hashes.
///
/// Seeds that already fit in 32 bits retain their historical value exactly. For larger
/// seeds, the upper half is mixed before it enters the current hash path so values separated
/// by `2^32` no longer alias systematically.
pub(crate) fn canonical_seed32(seed: u64) -> u32 {
    let lower = seed as u32;
    let upper = (seed >> 32) as u32;
    lower ^ hash_u32(upper)
}

pub(crate) fn hash2(ix: i32, iz: i32, seed: u32) -> u32 {
    let mut h = seed;
    h ^= ix as u32;
    h = hash_u32(h);
    h ^= iz as u32;
    h = hash_u32(h);
    h
}

pub(crate) fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

pub(crate) fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn remap(v: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    let t = ((v - in_min) / (in_max - in_min).max(1e-6)).clamp(0.0, 1.0);
    out_min + t * (out_max - out_min)
}

pub fn sample_noise(kind: FractalNoiseType, x: f32, z: f32, seed: u64) -> f32 {
    match kind {
        FractalNoiseType::Value => value_noise2(x, z, seed),
        FractalNoiseType::Perlin => perlin2(x, z, seed),
        FractalNoiseType::OpenSimplex => open_simplex2(x, z, seed),
    }
}

pub fn fbm(kind: FractalNoiseType, x: f32, z: f32, params: &NoiseParams) -> f32 {
    let mut amp = 1.0;
    let mut freq = params.frequency;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for o in 0..params.octaves.max(1) {
        let n = sample_noise(
            kind,
            (x + params.offset_x) * freq,
            (z + params.offset_z) * freq,
            params.seed.wrapping_add(o as u64 * 1013),
        );
        sum += n * amp;
        norm += amp;
        amp *= params.persistence;
        freq *= params.lacunarity;
    }
    let v = if norm > 0.0 { sum / norm } else { 0.0 };
    remap(v, params.remap_min, params.remap_max, -1.0, 1.0) * params.amplitude
}

/// Normalized hybrid ridged multifractal in `[0, amplitude]`.
///
/// The octave envelope is normalized so adding detail cannot silently exceed
/// the artist's amplitude and get flattened by downstream clamps.
pub fn ridged_mf(kind: FractalNoiseType, x: f32, z: f32, params: &NoiseParams) -> f32 {
    let mut amp = 1.0;
    let mut freq = params.frequency;
    let mut sum = 0.0;
    let mut norm = 0.0;
    let mut weight = 1.0;
    for o in 0..params.octaves.max(1) {
        let n = sample_noise(
            kind,
            (x + params.offset_x) * freq,
            (z + params.offset_z) * freq,
            params.seed.wrapping_add(o as u64 * 9173),
        );
        let ridge = (1.0 - n.abs()).clamp(0.0, 1.0);
        let signal = ridge * ridge * weight;
        weight = (signal * 2.0).clamp(0.0, 1.0);
        sum += signal * amp;
        norm += amp;
        amp *= params.persistence;
        freq *= params.lacunarity;
    }
    let normalized = if norm > 0.0 { sum / norm } else { 0.0 };
    normalized.clamp(0.0, 1.0) * params.amplitude
}

pub fn domain_warp_fbm(
    x: f32,
    z: f32,
    params: &NoiseParams,
    warp_strength: f32,
    warp_frequency: f32,
) -> f32 {
    let wx = perlin2(
        (x + params.offset_x) * warp_frequency,
        (z + params.offset_z) * warp_frequency,
        params.seed,
    ) * warp_strength;
    let wz = perlin2(
        (x + params.offset_x) * warp_frequency + 19.1,
        (z + params.offset_z) * warp_frequency + 7.3,
        params.seed.wrapping_add(1),
    ) * warp_strength;
    fbm(FractalNoiseType::Perlin, x + wx, z + wz, params)
}

pub fn sample_worley(x: f32, z: f32, params: &WorleyParams) -> f32 {
    let p = &params.base;
    let r = worley2(
        (x + p.offset_x) * p.frequency,
        (z + p.offset_z) * p.frequency,
        p.seed,
        params.distance_metric,
    );
    let v = match params.feature {
        WorleyFeature::F1 => r.f1,
        WorleyFeature::F2 => r.f2,
        WorleyFeature::F2MinusF1 => r.f2 - r.f1,
    };
    // Worley distances are typically ~0..1.5 in cell space; remap roughly.
    let n = remap(v, 0.0, 1.2, p.remap_min, p.remap_max);
    n * p.amplitude
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_POINTS: [(f32, f32); 4] = [
        (0.125, 0.375),
        (1.25, 3.5),
        (-7.75, 11.125),
        (93.625, -41.875),
    ];

    fn sample_bits(sample: impl Fn(f32, f32, u64) -> f32, seed: u64) -> Vec<u32> {
        SAMPLE_POINTS
            .iter()
            .map(|&(x, z)| sample(x, z, seed).to_bits())
            .collect()
    }

    #[test]
    fn canonical_seed_preserves_representative_low_seed_bits_and_boundaries() {
        for seed in [0, 1, 7, 42, 0x00FF_FFFF, u32::MAX] {
            assert_eq!(canonical_seed32(u64::from(seed)), seed);
        }
        assert_ne!(canonical_seed32(7), canonical_seed32(7 + (1u64 << 32)));
    }

    #[test]
    fn high_seed_bits_change_all_cpu_noise_primitives() {
        let low = 7;
        let high = low + (1u64 << 32);

        for (name, sample) in [
            ("value", value_noise2 as fn(f32, f32, u64) -> f32),
            ("perlin", perlin2 as fn(f32, f32, u64) -> f32),
            ("open_simplex", open_simplex2 as fn(f32, f32, u64) -> f32),
        ] {
            assert_ne!(
                sample_bits(sample, low),
                sample_bits(sample, high),
                "{name} still aliases seeds separated by 2^32"
            );
        }

        let worley_bits = |seed| {
            SAMPLE_POINTS
                .iter()
                .map(|&(x, z)| worley2(x, z, seed, WorleyMetric::Euclidean).f1.to_bits())
                .collect::<Vec<_>>()
        };
        assert_ne!(
            worley_bits(low),
            worley_bits(high),
            "Worley F1 still aliases seeds separated by 2^32"
        );
    }

    #[test]
    fn deterministic_value_noise() {
        let a = value_noise2(1.25, 3.5, 42);
        let b = value_noise2(1.25, 3.5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn fbm_finite() {
        let p = NoiseParams {
            octaves: 4,
            frequency: 0.01,
            ..NoiseParams::default()
        };
        let v = fbm(FractalNoiseType::Perlin, 10.0, 20.0, &p);
        assert!(v.is_finite());
    }

    #[test]
    fn ridged_multifractal_respects_amplitude_across_octaves() {
        for octaves in [1, 3, 6, 9] {
            let p = NoiseParams {
                octaves,
                frequency: 0.013,
                amplitude: 240.0,
                ..NoiseParams::default()
            };
            for z in 0..48 {
                for x in 0..48 {
                    let v = ridged_mf(FractalNoiseType::Perlin, x as f32, z as f32, &p);
                    assert!(v >= 0.0, "ridged field must stay non-negative");
                    assert!(
                        v <= p.amplitude + 1e-4,
                        "octaves={octaves} exceeded amplitude: {v}"
                    );
                }
            }
        }
    }

    #[test]
    fn domain_warp_finite() {
        let p = NoiseParams::default();
        let v = domain_warp_fbm(5.0, 8.0, &p, 50.0, 0.02);
        assert!(v.is_finite());
    }
}
