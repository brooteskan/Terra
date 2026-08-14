//! Phase E hydrology fixtures: SPE + drainage coherence vs noise+carve.

use terra_core::generators::{fbm_field, uplift};
use terra_core::geomorph::{accumulate_drainage_area, build_flow_graph, FlowModel, Precipitation};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::hydro::{carve_rivers, fill_depressions, stream_power_erode};
use terra_core::layer::{
    FbmParams, FractalNoiseType, NoiseParams, RiverCarveParams, StreamPowerParams, UpliftParams,
};
use terra_core::mask::MaskField;

fn depression_fill_volume(hf: &Heightfield) -> f32 {
    let filled = fill_depressions(hf);
    let a = hf.to_dense();
    let b = filled.to_dense();
    a.iter().zip(b.iter()).map(|(h, f)| (f - h).max(0.0)).sum()
}

/// Mean normalized accumulation on the lowest quartile of heights (valley cells).
fn valley_accumulation_score(hf: &Heightfield, acc: &[f32]) -> f32 {
    let dense = hf.to_dense();
    let mut heights: Vec<f32> = dense.clone();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = heights[heights.len() / 4];
    let max_acc = acc.iter().copied().fold(1.0f32, f32::max);
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for (h, &a) in dense.iter().zip(acc.iter()) {
        if *h <= q {
            sum += a / max_acc;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f32
    }
}

#[test]
fn spe_deterministic_fixture() {
    let m = HeightfieldMetrics::new(48, 48, 1500.0, 1500.0);
    let up = uplift(
        m,
        &UpliftParams {
            seed: 99,
            detail_amplitude: 20.0,
            altitude_fade: 0.8,
            ..UpliftParams::default()
        },
    );
    let p = StreamPowerParams {
        iterations: 8,
        k: 0.06,
        dendritic_seed: 0.3,
        ..StreamPowerParams::default()
    };
    let hard = MaskField::filled(m, 0.0);
    let a = stream_power_erode(&up, &p, &hard);
    let b = stream_power_erode(&up, &p, &hard);
    assert_eq!(a.height.to_dense(), b.height.to_dense());
    assert_eq!(a.flow_accumulation.data(), b.flow_accumulation.data());
}

#[test]
fn spe_soft_rock_incises_more() {
    let m = HeightfieldMetrics::new(40, 40, 800.0, 800.0);
    let mut hf = Heightfield::zeros(m);
    for j in 0..40 {
        for i in 0..40 {
            hf.set(i, j, 50.0 - i as f32 * 0.9 + (j as f32 * 0.05).sin());
        }
    }
    let mut k = MaskField::filled(m, 0.0);
    for j in 0..40 {
        for i in 0..20 {
            k.set(i, j, 0.9);
        }
    }
    let p = StreamPowerParams {
        iterations: 14,
        k: 0.1,
        refill_each_iter: true,
        ..StreamPowerParams::default()
    };
    let out = stream_power_erode(&hf, &p, &k);
    let soft_drop: f32 = (20..40)
        .map(|i| hf.get(i, 20) - out.height.get(i, 20))
        .sum();
    let hard_drop: f32 = (0..20).map(|i| hf.get(i, 20) - out.height.get(i, 20)).sum();
    assert!(
        soft_drop > hard_drop * 1.2,
        "soft_drop={soft_drop} hard_drop={hard_drop}"
    );
}

#[test]
fn uplift_spe_aligns_valleys_better_than_noise_carve() {
    let m = HeightfieldMetrics::new(64, 64, 2000.0, 2000.0);
    let up = uplift(
        m,
        &UpliftParams {
            seed: 42,
            amplitude: 300.0,
            detail_amplitude: 25.0,
            altitude_fade: 0.85,
            warp_strength: 0.3,
            ..UpliftParams::default()
        },
    );
    let spe = stream_power_erode(
        &up,
        &StreamPowerParams {
            iterations: 16,
            k: 0.08,
            dendritic_seed: 0.45,
            refill_each_iter: true,
            ..StreamPowerParams::default()
        },
        &MaskField::filled(m, 0.0),
    );

    let mut noise = fbm_field(
        m,
        &FbmParams {
            base: NoiseParams {
                seed: 42,
                frequency: 0.0015,
                amplitude: 300.0,
                octaves: 6,
                ..NoiseParams::default()
            },
            noise: FractalNoiseType::Perlin,
        },
    );
    let min_h = noise.to_dense().into_iter().fold(f32::INFINITY, f32::min);
    noise.map_mut(|h| h - min_h + 20.0);
    let (carved, _, _, _) = carve_rivers(
        &noise,
        &RiverCarveParams {
            accumulation_threshold: 40.0,
            depth: 40.0,
            width: 5.0,
            ..RiverCarveParams::default()
        },
        None,
    );

    let filled_spe = fill_depressions(&spe.height);
    let graph_spe = build_flow_graph(&filled_spe, FlowModel::D8);
    let acc_spe = accumulate_drainage_area(&graph_spe, &Precipitation::uniform(1.0));
    let score_spe = valley_accumulation_score(&spe.height, &acc_spe);

    let filled_carve = fill_depressions(&carved);
    let graph_carve = build_flow_graph(&filled_carve, FlowModel::D8);
    let acc_carve = accumulate_drainage_area(&graph_carve, &Precipitation::uniform(1.0));
    let score_carve = valley_accumulation_score(&carved, &acc_carve);

    assert!(
        score_spe > score_carve * 1.05,
        "uplift+SPE valley acc {score_spe} should beat noise+carve {score_carve}"
    );

    let pits_spe = depression_fill_volume(&spe.height);
    let pits_carve = depression_fill_volume(&carved);
    assert!(
        pits_spe < pits_carve * 1.15 + 50.0,
        "SPE pits {pits_spe} should not greatly exceed carve {pits_carve}"
    );
}

#[test]
fn spe_respects_base_level() {
    let m = HeightfieldMetrics::new(24, 24, 240.0, 240.0);
    let mut hf = Heightfield::filled(m, 30.0);
    for j in 0..24 {
        for i in 0..24 {
            hf.set(i, j, 30.0 - j as f32);
        }
    }
    let p = StreamPowerParams {
        iterations: 20,
        k: 0.5,
        base_level: 10.0,
        dt: 2.0,
        ..StreamPowerParams::default()
    };
    let out = stream_power_erode(&hf, &p, &MaskField::filled(m, 0.0));
    let min_h = out
        .height
        .to_dense()
        .into_iter()
        .fold(f32::INFINITY, f32::min);
    assert!(min_h >= 10.0 - 1e-3, "min={min_h}");
}

/// Draft drainage-reuse SPE stays within ε of export-stride SPE on a smooth ramp.
/// Documented as "preview approximate" — not bit-identical.
#[test]
fn spe_draft_drainage_reuse_within_eps() {
    let m = HeightfieldMetrics::new(32, 32, 640.0, 640.0);
    let mut hf = Heightfield::zeros(m);
    for j in 0..32 {
        for i in 0..32 {
            hf.set(i, j, 40.0 - i as f32 * 0.7 + (j as f32 * 0.08).sin());
        }
    }
    let hard = MaskField::filled(m, 0.0);
    let export = StreamPowerParams {
        iterations: 8,
        k: 0.08,
        drainage_reuse_stride: 1,
        refill_each_iter: false,
        ..StreamPowerParams::default()
    };
    let draft = StreamPowerParams {
        drainage_reuse_stride: 2,
        ..export.clone()
    };
    let a = stream_power_erode(&hf, &export, &hard);
    let b = stream_power_erode(&hf, &draft, &hard);
    let da = a.height.to_dense();
    let db = b.height.to_dense();
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f32;
    for (x, y) in da.iter().zip(db.iter()) {
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sum_sq += d * d;
    }
    let rms = (sum_sq / da.len() as f32).sqrt();
    // Preview approximate: documented ε for drainage-reuse Draft vs export stride.
    // Export path (stride=1) remains the deterministic oracle (see spe_deterministic_fixture).
    assert!(
        max_abs < 4.0 && rms < 0.6,
        "draft vs export max_abs={max_abs} rms={rms} (preview approximate ε)"
    );
}
