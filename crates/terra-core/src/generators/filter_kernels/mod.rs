//! Shared research-backed kernels for biome EffectFilters.
//!
//! Citations (public literature; not World Creator proprietary kernels):
//! - Tarboton 1997 — D∞ flow direction / multi-flow accumulation
//! - Lagae et al. — sparse Gabor convolution noise
//! - Tricard et al. SIGGRAPH 2019 — Phasor Noise
//! - Musgrave — ridged / billow multifractals; thermal weathering (angle of repose)
//! - Kuwahara 1976 / Kyprianidis — edge-preserving Kuwahara filtering
//! - Tomasi & Manduchi — bilateral filtering
//! - Werner / Nishimori-class aeolian dune & saltation models

use crate::heightfield::Heightfield;
use crate::noise::{self, canonical_seed32, hash2};

const SQRT2: f32 = std::f32::consts::SQRT_2;
const PI: f32 = std::f32::consts::PI;
const TAU: f32 = std::f32::consts::TAU;

/// Local slope in degrees (forward differences).
pub fn slope_deg(hf: &Heightfield, i: u32, j: u32) -> f32 {
    let dx = hf.metrics.dx().max(1e-5);
    let dz = hf.metrics.dz().max(1e-5);
    let w = hf.metrics.width;
    let h = hf.metrics.height;
    let z = hf.get(i, j);
    let zx = if i + 1 < w { hf.get(i + 1, j) } else { z };
    let zy = if j + 1 < h { hf.get(i, j + 1) } else { z };
    let gx = (zx - z) / dx;
    let gz = (zy - z) / dz;
    (gx * gx + gz * gz).sqrt().atan().to_degrees()
}

/// Gradient unit vector (world-scaled); returns (gx, gz, magnitude).
pub fn gradient(hf: &Heightfield, i: u32, j: u32) -> (f32, f32, f32) {
    let dx = hf.metrics.dx().max(1e-5);
    let dz = hf.metrics.dz().max(1e-5);
    let w = hf.metrics.width;
    let h = hf.metrics.height;
    let z = hf.get(i, j);
    let zx = if i + 1 < w { hf.get(i + 1, j) } else { z };
    let zy = if j + 1 < h { hf.get(i, j + 1) } else { z };
    let gx = (zx - z) / dx;
    let gz = (zy - z) / dz;
    let mag = (gx * gx + gz * gz).sqrt();
    if mag < 1e-8 {
        (0.0, 0.0, 0.0)
    } else {
        (gx / mag, gz / mag, mag)
    }
}

/// D∞ / multi-flow accumulation (Tarboton-inspired).
///
/// Each cell distributes unit rainfall proportionally to downhill neighbors by
/// drop magnitude (multi-flow), iterated for `passes` sweeps.
pub fn flow_accumulation_dinf(hf: &Heightfield, passes: u32) -> Vec<f32> {
    let w = hf.metrics.width as usize;
    let h = hf.metrics.height as usize;
    let n = w * h;
    let mut acc = vec![1.0f32; n];
    let dirs = [
        (-1i32, 0i32),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ];
    for _ in 0..passes.max(1) {
        let mut next = vec![1.0f32; n];
        for j in 0..h as i32 {
            for i in 0..w as i32 {
                let z = hf.get(i as u32, j as u32);
                let mut drops = [0.0f32; 8];
                let mut total = 0.0f32;
                for (k, &(di, dj)) in dirs.iter().enumerate() {
                    let ni = i + di;
                    let nj = j + dj;
                    if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
                        continue;
                    }
                    let nz = hf.get(ni as u32, nj as u32);
                    let drop = (z - nz).max(0.0);
                    // Diagonal steps are longer; scale for fair multi-flow weights.
                    let len = if di != 0 && dj != 0 { SQRT2 } else { 1.0 };
                    let wgt = drop / len;
                    drops[k] = wgt;
                    total += wgt;
                }
                let src = acc[(j as usize) * w + i as usize];
                if total > 1e-8 {
                    for (k, &(di, dj)) in dirs.iter().enumerate() {
                        if drops[k] <= 0.0 {
                            continue;
                        }
                        let ni = i + di;
                        let nj = j + dj;
                        next[(nj as usize) * w + ni as usize] += src * (drops[k] / total);
                    }
                }
            }
        }
        acc = next;
    }
    acc
}

pub fn normalize_field(values: &[f32]) -> Vec<f32> {
    let max_f = values.iter().copied().fold(1e-6f32, f32::max);
    values.iter().map(|v| (v / max_f).clamp(0.0, 1.0)).collect()
}

/// Box blur helper used by several filters.
pub fn box_blur(input: &Heightfield, radius: u32) -> Heightfield {
    let r = radius.max(1) as i32;
    let w = input.metrics.width as i32;
    let h = input.metrics.height as i32;
    let mut out = input.clone();
    for j in 0..h {
        for i in 0..w {
            let mut sum = 0.0f32;
            let mut count = 0u32;
            for dj in -r..=r {
                for di in -r..=r {
                    let x = (i + di).clamp(0, w - 1) as u32;
                    let y = (j + dj).clamp(0, h - 1) as u32;
                    sum += input.get(x, y);
                    count += 1;
                }
            }
            out.set(i as u32, j as u32, sum / count.max(1) as f32);
        }
    }
    out
}

/// Bilateral (edge-aware) filter — Tomasi & Manduchi style on heightfields.
pub fn bilateral(
    input: &Heightfield,
    radius: u32,
    sigma_space: f32,
    sigma_range: f32,
) -> Heightfield {
    let r = radius.max(1) as i32;
    let w = input.metrics.width as i32;
    let h = input.metrics.height as i32;
    let ss = sigma_space.max(0.25);
    let sr = sigma_range.max(1e-3);
    let mut out = input.clone();
    for j in 0..h {
        for i in 0..w {
            let c = input.get(i as u32, j as u32);
            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for dj in -r..=r {
                for di in -r..=r {
                    let x = (i + di).clamp(0, w - 1) as u32;
                    let y = (j + dj).clamp(0, h - 1) as u32;
                    let v = input.get(x, y);
                    let dist2 = (di * di + dj * dj) as f32;
                    let ws = (-dist2 / (2.0 * ss * ss)).exp();
                    let wr = (-((v - c) * (v - c)) / (2.0 * sr * sr)).exp();
                    let wt = ws * wr;
                    sum += v * wt;
                    wsum += wt;
                }
            }
            out.set(i as u32, j as u32, sum / wsum.max(1e-5));
        }
    }
    out
}

/// Directional Gaussian blur along unit direction `(dir_x, dir_z)` in texel steps.
pub fn anisotropic_blur(input: &Heightfield, radius: u32, dir_x: f32, dir_z: f32) -> Heightfield {
    let r = radius.max(1) as i32;
    let w = input.metrics.width as i32;
    let h = input.metrics.height as i32;
    let len = (dir_x * dir_x + dir_z * dir_z).sqrt().max(1e-5);
    let sx = dir_x / len;
    let sz = dir_z / len;
    let mut out = input.clone();
    let sigma = (r as f32 * 0.5).max(0.5);
    for j in 0..h {
        for i in 0..w {
            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for k in -r..=r {
                let fi = (i as f32 + sx * k as f32).round() as i32;
                let fj = (j as f32 + sz * k as f32).round() as i32;
                let xi = fi.clamp(0, w - 1) as u32;
                let yj = fj.clamp(0, h - 1) as u32;
                let wt = (-(k * k) as f32 / (2.0 * sigma * sigma)).exp();
                sum += input.get(xi, yj) * wt;
                wsum += wt;
            }
            out.set(i as u32, j as u32, sum / wsum.max(1e-5));
        }
    }
    out
}

/// Greyscale morphological dilation (local max).
pub fn dilate(input: &Heightfield, radius: u32) -> Heightfield {
    morph_extremum(input, radius, true)
}

/// Greyscale morphological erosion (local min).
pub fn erode(input: &Heightfield, radius: u32) -> Heightfield {
    morph_extremum(input, radius, false)
}

fn morph_extremum(input: &Heightfield, radius: u32, maximize: bool) -> Heightfield {
    let r = radius.max(1) as i32;
    let w = input.metrics.width as i32;
    let h = input.metrics.height as i32;
    let mut out = input.clone();
    for j in 0..h {
        for i in 0..w {
            let mut ext = input.get(i as u32, j as u32);
            for dj in -r..=r {
                for di in -r..=r {
                    let x = (i + di).clamp(0, w - 1) as u32;
                    let y = (j + dj).clamp(0, h - 1) as u32;
                    let v = input.get(x, y);
                    ext = if maximize { ext.max(v) } else { ext.min(v) };
                }
            }
            out.set(i as u32, j as u32, ext);
        }
    }
    out
}

/// Angle-of-repose thermal weathering (Musgrave-style talus).
///
/// Thin wrapper around the layered mass-wasting apron for filter kernels.
pub fn thermal_talus(
    input: &Heightfield,
    repose_deg: f32,
    amount: f32,
    iterations: u32,
) -> Heightfield {
    crate::analyze::talus_apron(input, repose_deg, amount.max(0.5), 0.85, iterations, None).height
}

/// Multi-sector anisotropic Kuwahara (Kyprianidis-inspired sector variance pick).
pub fn kuwahara_anisotropic(input: &Heightfield, radius: u32, sectors: u32) -> Heightfield {
    let r = radius.max(1) as i32;
    let sectors = sectors.clamp(4, 8) as i32;
    let w = input.metrics.width as i32;
    let h = input.metrics.height as i32;
    let mut out = input.clone();
    for j in 0..h {
        for i in 0..w {
            let mut best_mean = input.get(i as u32, j as u32);
            let mut best_var = f32::INFINITY;
            for s in 0..sectors {
                let a0 = (s as f32) * TAU / sectors as f32;
                let a1 = ((s + 1) as f32) * TAU / sectors as f32;
                let mut sum = 0.0f32;
                let mut sum2 = 0.0f32;
                let mut count = 0u32;
                for dj in -r..=r {
                    for di in -r..=r {
                        if di == 0 && dj == 0 {
                            continue;
                        }
                        let ang = (dj as f32).atan2(di as f32);
                        let mut a = ang;
                        if a < 0.0 {
                            a += TAU;
                        }
                        // Wrap-aware sector membership.
                        let in_sector = if a0 <= a1 {
                            a >= a0 && a < a1
                        } else {
                            a >= a0 || a < a1
                        };
                        if !in_sector {
                            continue;
                        }
                        let dist = ((di * di + dj * dj) as f32).sqrt();
                        if dist > r as f32 + 1e-3 {
                            continue;
                        }
                        let x = (i + di).clamp(0, w - 1) as u32;
                        let y = (j + dj).clamp(0, h - 1) as u32;
                        let v = input.get(x, y);
                        sum += v;
                        sum2 += v * v;
                        count += 1;
                    }
                }
                // Include center in every sector for stability.
                let c = input.get(i as u32, j as u32);
                sum += c;
                sum2 += c * c;
                count += 1;
                if count == 0 {
                    continue;
                }
                let n = count as f32;
                let mean = sum / n;
                let var = (sum2 / n - mean * mean).max(0.0);
                if var < best_var {
                    best_var = var;
                    best_mean = mean;
                }
            }
            out.set(i as u32, j as u32, best_mean);
        }
    }
    out
}

fn rand01(ix: i32, iz: i32, seed: u32) -> f32 {
    let h = hash2(ix, iz, seed);
    (h as f32) * (1.0 / 4294967295.0)
}

fn rand_angle(ix: i32, iz: i32, seed: u32) -> f32 {
    rand01(ix, iz, seed) * TAU
}

/// Sparse Gabor noise (Lagae et al. style): weighted cosine kernels at Poisson-ish lattice points.
pub fn sparse_gabor(x: f32, z: f32, frequency: f32, seed: u64) -> f32 {
    sparse_gabor_oriented(x, z, frequency, seed, 0.0, 1.0)
}

/// Oriented / anisotropic sparse Gabor (Lagae sparse convolution).
pub fn sparse_gabor_oriented(
    x: f32,
    z: f32,
    frequency: f32,
    seed: u64,
    rotation_deg: f32,
    anisotropy: f32,
) -> f32 {
    let ang = rotation_deg.to_radians();
    let (s, c) = ang.sin_cos();
    let xr = x * c + z * s;
    let zr = -x * s + z * c;
    let aniso = anisotropy.max(0.05);
    let xx = xr / aniso;
    let zz = zr * aniso.sqrt().max(0.2);
    sparse_gabor_raw(xx, zz, frequency, seed)
}

fn sparse_gabor_raw(x: f32, z: f32, frequency: f32, seed: u64) -> f32 {
    let freq = frequency.max(1e-5);
    let cell = 1.0 / freq;
    let px = x / cell;
    let pz = z / cell;
    let ix = px.floor() as i32;
    let iz = pz.floor() as i32;
    let seed_u = canonical_seed32(seed);
    let mut sum = 0.0f32;
    let mut wsum = 0.0f32;
    // 3×3 neighborhood of impulse cells.
    for dj in -1..=1 {
        for di in -1..=1 {
            let cx = ix + di;
            let cz = iz + dj;
            // One impulse per cell with jittered position (sparse approximation).
            let jx = rand01(cx, cz, seed_u ^ 0xA11Cu32);
            let jz = rand01(cx, cz, seed_u ^ 0xB22Du32);
            let ox = (cx as f32 + jx) * cell;
            let oz = (cz as f32 + jz) * cell;
            let dx = x - ox;
            let dz = z - oz;
            let dist2 = dx * dx + dz * dz;
            let sigma = cell * 0.45;
            let env = (-dist2 / (2.0 * sigma * sigma)).exp();
            if env < 1e-5 {
                continue;
            }
            let theta = rand_angle(cx, cz, seed_u ^ 0xC33Eu32);
            let (st, ct) = theta.sin_cos();
            // Anisotropic frequency along orientation.
            let fmag = freq * (0.85 + 0.3 * rand01(cx, cz, seed_u ^ 0xD44Fu32));
            let phase = rand01(cx, cz, seed_u ^ 0xE55Au32) * TAU;
            let carrier = ((dx * ct + dz * st) * fmag * TAU + phase).cos();
            let amp = rand01(cx, cz, seed_u ^ 0xF66Bu32) * 2.0 - 1.0;
            sum += env * carrier * amp;
            wsum += env;
        }
    }
    if wsum < 1e-5 {
        0.0
    } else {
        (sum / wsum).clamp(-1.0, 1.0)
    }
}

/// Multi-octave sparse Gabor (distinct from Perlin/Value FBM).
pub fn sparse_gabor_fbm(
    x: f32,
    z: f32,
    frequency: f32,
    seed: u64,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
    rotation_deg: f32,
    anisotropy: f32,
) -> f32 {
    let mut amp = 1.0f32;
    let mut freq = frequency.max(1e-5);
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    let lac = lacunarity.max(1.01);
    let gain = persistence.clamp(0.05, 0.95);
    for o in 0..octaves.max(1) {
        let n = sparse_gabor_oriented(
            x,
            z,
            freq,
            seed.wrapping_add(o as u64 * 7919),
            rotation_deg,
            anisotropy,
        );
        sum += n * amp;
        norm += amp;
        amp *= gain;
        freq *= lac;
    }
    if norm > 0.0 {
        (sum / norm).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Phasor noise (Tricard et al.): complex phasor field magnitude/phase → contrasty bands.
pub fn phasor_noise(x: f32, z: f32, frequency: f32, seed: u64) -> f32 {
    phasor_noise_oriented(x, z, frequency, seed, 0.0, 1.0)
}

/// Oriented phasor noise with anisotropy (Tricard et al. SIGGRAPH 2019).
pub fn phasor_noise_oriented(
    x: f32,
    z: f32,
    frequency: f32,
    seed: u64,
    rotation_deg: f32,
    anisotropy: f32,
) -> f32 {
    let ang = rotation_deg.to_radians();
    let (s, c) = ang.sin_cos();
    let xr = x * c + z * s;
    let zr = -x * s + z * c;
    let aniso = anisotropy.max(0.05);
    let xx = xr / aniso;
    let zz = zr * aniso.sqrt().max(0.2);
    phasor_noise_raw(xx, zz, frequency, seed)
}

fn phasor_noise_raw(x: f32, z: f32, frequency: f32, seed: u64) -> f32 {
    let freq = frequency.max(1e-5);
    // Dual-frequency oriented phasors summed in complex plane.
    let mut re = 0.0f32;
    let mut im = 0.0f32;
    for k in 0..4 {
        let sk = seed.wrapping_add(k as u64 * 9173);
        let theta = noise::sample_noise(
            crate::noise::FractalNoiseType::OpenSimplex,
            x * freq * 0.15 + k as f32 * 3.1,
            z * freq * 0.15 - k as f32 * 1.7,
            sk,
        ) * PI;
        let (st, ct) = theta.sin_cos();
        let phase = (x * ct + z * st) * freq * TAU
            + noise::sample_noise(
                crate::noise::FractalNoiseType::OpenSimplex,
                x * freq * 0.07,
                z * freq * 0.07,
                sk ^ 0x51AF,
            ) * PI;
        let amp = 0.55
            + 0.45
                * noise::sample_noise(
                    crate::noise::FractalNoiseType::Value,
                    x * freq * 0.2,
                    z * freq * 0.2,
                    sk ^ 0xC0DE,
                )
                .abs();
        re += amp * phase.cos();
        im += amp * phase.sin();
    }
    let mag = (re * re + im * im).sqrt();
    // Contrast-shaped response in [-1, 1].
    ((mag * 1.15).tanh() * 2.0 - 1.0).clamp(-1.0, 1.0)
}

/// Multi-octave phasor noise (distinct from Perlin/Simplex FBM).
pub fn phasor_noise_fbm(
    x: f32,
    z: f32,
    frequency: f32,
    seed: u64,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
    rotation_deg: f32,
    anisotropy: f32,
) -> f32 {
    let mut amp = 1.0f32;
    let mut freq = frequency.max(1e-5);
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    let lac = lacunarity.max(1.01);
    let gain = persistence.clamp(0.05, 0.95);
    for o in 0..octaves.max(1) {
        let n = phasor_noise_oriented(
            x,
            z,
            freq,
            seed.wrapping_add(o as u64 * 4243),
            rotation_deg,
            anisotropy,
        );
        sum += n * amp;
        norm += amp;
        amp *= gain;
        freq *= lac;
    }
    if norm > 0.0 {
        (sum / norm).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Directional band-limited phasor / wave field for dune seeding.
///
/// `direction_rad` aligns crests perpendicular to the wind. High `linearity`
/// keeps a single orientation; lower values mix secondary lobes (star-like).
pub fn directional_phasor(
    x: f32,
    z: f32,
    frequency: f32,
    direction_rad: f32,
    linearity: f32,
    seed: u64,
) -> f32 {
    let freq = frequency.max(1e-5);
    let lin = linearity.clamp(0.0, 1.0);
    let (s0, c0) = direction_rad.sin_cos();
    // Crests run across-wind; phase advances along wind.
    let along = x * c0 + z * s0;
    let across = -x * s0 + z * c0;

    let mut re = 0.0f32;
    let mut im = 0.0f32;
    let lobes = if lin > 0.75 {
        2
    } else if lin > 0.4 {
        3
    } else {
        4
    };
    for k in 0..lobes {
        let sk = seed.wrapping_add(k as u64 * 9173);
        let spread = (1.0 - lin) * 0.55;
        let yaw = noise::sample_noise(
            crate::noise::FractalNoiseType::OpenSimplex,
            across * freq * 0.12 + k as f32 * 2.7,
            along * freq * 0.05,
            sk,
        ) * PI
            * spread;
        let (st, ct) = (direction_rad + yaw).sin_cos();
        let warp = noise::sample_noise(
            crate::noise::FractalNoiseType::OpenSimplex,
            x * freq * 0.08,
            z * freq * 0.08,
            sk ^ 0x51AF,
        ) * PI
            * (0.35 + 0.65 * (1.0 - lin));
        let phase = (x * ct + z * st) * freq * TAU + warp;
        let amp = 0.6
            + 0.4
                * noise::sample_noise(
                    crate::noise::FractalNoiseType::Value,
                    across * freq * 0.18,
                    along * freq * 0.11,
                    sk ^ 0xC0DE,
                )
                .abs();
        // Prefer the primary wind-aligned lobe when linearity is high.
        let w = if k == 0 { 1.0 } else { (1.0 - lin).powf(0.65) };
        re += amp * w * phase.cos();
        im += amp * w * phase.sin();
    }
    let mag = (re * re + im * im).sqrt();
    ((mag * 1.1).tanh() * 2.0 - 1.0).clamp(-1.0, 1.0)
}

/// Musgrave-style billow: absolute-value FBM centered to [-1, 1].
pub fn billow_mf(x: f32, z: f32, params: &crate::noise::NoiseParams) -> f32 {
    let mut amp = 1.0f32;
    let mut freq = params.frequency;
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    for o in 0..params.octaves.max(1) {
        let n = noise::sample_noise(
            crate::noise::FractalNoiseType::Perlin,
            (x + params.offset_x) * freq,
            (z + params.offset_z) * freq,
            params.seed.wrapping_add(o as u64 * 1301),
        );
        // |n| in [0,1] roughly → billow cushions.
        let billow = n.abs();
        sum += billow * amp;
        norm += amp;
        amp *= params.persistence;
        freq *= params.lacunarity;
    }
    let v = if norm > 0.0 { sum / norm } else { 0.0 };
    (v * 2.0 - 1.0) * params.amplitude
}

/// Deterministic white noise in [-1, 1] from grid indices.
pub fn white_noise(i: u32, j: u32, seed: u64) -> f32 {
    let h = hash2(i as i32, j as i32, canonical_seed32(seed));
    (h as f32 / 4294967295.0) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

    #[test]
    fn gabor_and_phasor_deterministic_and_finite() {
        let a = sparse_gabor(12.5, 8.25, 0.04, 42);
        let b = sparse_gabor(12.5, 8.25, 0.04, 42);
        assert_eq!(a, b);
        assert!(a.is_finite());
        let p = phasor_noise(3.0, 7.0, 0.05, 7);
        assert!(p.is_finite());
        assert_eq!(p, phasor_noise(3.0, 7.0, 0.05, 7));
        let d = directional_phasor(3.0, 7.0, 0.05, 0.3, 0.8, 7);
        assert!(d.is_finite());
        assert_eq!(d, directional_phasor(3.0, 7.0, 0.05, 0.3, 0.8, 7));
    }

    #[test]
    fn high_seed_bits_change_hash_backed_filter_kernels() {
        let low = 7;
        let high = low + (1u64 << 32);
        let points = [(1.25, 3.5), (12.5, 8.25), (-3.75, 19.125)];
        let gabor = |seed| {
            points
                .iter()
                .map(|&(x, z)| sparse_gabor(x, z, 0.04, seed).to_bits())
                .collect::<Vec<_>>()
        };
        let white = |seed| {
            [(1, 3), (12, 8), (37, 19)]
                .iter()
                .map(|&(i, j)| white_noise(i, j, seed).to_bits())
                .collect::<Vec<_>>()
        };

        assert_ne!(gabor(low), gabor(high));
        assert_ne!(white(low), white(high));
    }

    #[test]
    fn dinf_flow_raises_downstream() {
        let m = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..16 {
            for i in 0..16 {
                // Ramp downhill toward +X.
                hf.set(i, j, 100.0 - i as f32 * 4.0);
            }
        }
        let flow = normalize_field(&flow_accumulation_dinf(&hf, 4));
        let up = flow[8 * 16 + 2];
        let down = flow[8 * 16 + 14];
        assert!(
            down > up,
            "downstream cells should accumulate more flow: up={up} down={down}"
        );
    }
}
