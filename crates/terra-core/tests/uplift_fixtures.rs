//! Phase B uplift fixtures: deterministic structure + fewer sinks than pure fBm.

use terra_core::generators::{fbm_field, uplift};
use terra_core::heightfield::HeightfieldMetrics;
use terra_core::hydro::fill_depressions;
use terra_core::layer::{FbmParams, FractalNoiseType, NoiseParams, UpliftParams};
use terra_core::CancelToken;

fn depression_fill_volume(hf: &terra_core::heightfield::Heightfield) -> f32 {
    let filled = fill_depressions(hf);
    let a = hf.to_dense();
    let b = filled.to_dense();
    a.iter().zip(b.iter()).map(|(h, f)| (f - h).max(0.0)).sum()
}

#[test]
fn uplift_is_deterministic() {
    let m = HeightfieldMetrics::new(64, 64, 2000.0, 2000.0);
    let p = UpliftParams {
        seed: 12345,
        ..UpliftParams::default()
    };
    let a = uplift(m, &CancelToken::never(), &p).unwrap();
    let b = uplift(m, &CancelToken::never(), &p).unwrap();
    assert_eq!(a.to_dense(), b.to_dense());
}

#[test]
fn uplift_seed_changes_output() {
    let m = HeightfieldMetrics::new(48, 48, 1500.0, 1500.0);
    let a = uplift(
        m,
        &CancelToken::never(),
        &UpliftParams {
            seed: 1,
            ..UpliftParams::default()
        },
    )
    .unwrap();
    let b = uplift(
        m,
        &CancelToken::never(),
        &UpliftParams {
            seed: 2,
            ..UpliftParams::default()
        },
    )
    .unwrap();
    assert_ne!(a.to_dense(), b.to_dense());
}

#[test]
fn uplift_peak_near_corridor_center() {
    let m = HeightfieldMetrics::new(64, 64, 2000.0, 2000.0);
    let p = UpliftParams {
        seed: 7,
        range_angle: 0.0, // corridor along X → ridge near mid Z
        corridor_width: 0.35,
        warp_strength: 0.0,
        detail_amplitude: 0.0,
        altitude_fade: 1.0,
        ..UpliftParams::default()
    };
    let hf = uplift(m, &CancelToken::never(), &p).unwrap();
    let mut peak_j = 0u32;
    let mut peak = f32::NEG_INFINITY;
    let i = 32;
    for j in 0..64 {
        let h = hf.get(i, j);
        if h > peak {
            peak = h;
            peak_j = j;
        }
    }
    // Primary ridge should sit near the map mid-line for angle=0.
    assert!(
        (peak_j as i32 - 32).abs() <= 10,
        "peak_j={peak_j} should be near mid corridor"
    );
    assert!(
        peak > p.amplitude * 0.15,
        "expected meaningful uplift height"
    );
}

#[test]
fn uplift_fills_fewer_sinks_than_fbm() {
    let m = HeightfieldMetrics::new(96, 96, 3000.0, 3000.0);
    let amp = 300.0;
    let up = uplift(
        m,
        &CancelToken::never(),
        &UpliftParams {
            seed: 42,
            amplitude: amp,
            detail_amplitude: 40.0,
            altitude_fade: 0.85,
            warp_strength: 0.25,
            ..UpliftParams::default()
        },
    )
    .unwrap();
    let noise = fbm_field(
        m,
        &CancelToken::never(),
        &FbmParams {
            base: NoiseParams {
                seed: 42,
                frequency: 0.0015,
                amplitude: amp,
                octaves: 6,
                ..NoiseParams::default()
            },
            noise: FractalNoiseType::Perlin,
        },
    )
    .unwrap();
    // Shift fBm to be non-negative like uplift (fBm is signed).
    let mut noise_pos = noise;
    let min_h = noise_pos
        .to_dense()
        .into_iter()
        .fold(f32::INFINITY, f32::min);
    noise_pos.map_mut(|h| h - min_h);

    let vol_up = depression_fill_volume(&up);
    let vol_noise = depression_fill_volume(&noise_pos);
    assert!(
        vol_up < vol_noise * 0.85,
        "uplift fill volume {vol_up} should be clearly below fBm {vol_noise}"
    );
}
