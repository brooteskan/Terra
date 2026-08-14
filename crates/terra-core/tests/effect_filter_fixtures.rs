//! Phase B priority filter pack fixtures (World Creator "Filters" walkthrough parity):
//! Smooth, Distortion, Spike Removal, Shore, Strata, Crater, Denoise.

use terra_core::generators::effect_filter;
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{CoastalParams, EffectFilterKind, EffectFilterParams};

fn noisy_field(res: u32) -> Heightfield {
    let m = HeightfieldMetrics::new(res, res, res as f32, res as f32);
    let mut hf = Heightfield::zeros(m);
    for j in 0..res {
        for i in 0..res {
            // Deterministic pseudo-noise: high-frequency checker + low-frequency swell.
            let checker = if (i + j) % 2 == 0 { 6.0 } else { -6.0 };
            let swell = ((i as f32 * 0.15).sin() + (j as f32 * 0.11).cos()) * 4.0;
            hf.set(i, j, 50.0 + checker + swell);
        }
    }
    hf
}

fn variance(hf: &Heightfield) -> f32 {
    let dense = hf.to_dense();
    let n = dense.len() as f32;
    let mean = dense.iter().sum::<f32>() / n;
    dense.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n
}

#[test]
fn smooth_reduces_variance() {
    let hf = noisy_field(48);
    let before = variance(&hf);
    let p = EffectFilterParams {
        kind: EffectFilterKind::Smooth,
        strength: 1.0,
        radius: 2,
        iterations: 2,
        ..EffectFilterParams::default()
    };
    let out = effect_filter(&hf, &p);
    let after = variance(&out);
    assert!(
        after < before * 0.6,
        "smooth should meaningfully reduce local variance: before={before} after={after}"
    );
}

#[test]
fn crater_lowers_center() {
    let res = 64;
    let m = HeightfieldMetrics::new(res, res, res as f32 * 4.0, res as f32 * 4.0);
    let hf = Heightfield::filled(m, 100.0);
    let p = EffectFilterParams {
        kind: EffectFilterKind::Crater,
        strength: 1.0,
        amount: 40.0,
        crater_radius: 0.25,
        ..EffectFilterParams::default()
    };
    let out = effect_filter(&hf, &p);
    let center = out.get(res / 2, res / 2);
    let corner = out.get(1, 1);
    assert!(
        center < 100.0 - 20.0,
        "crater should carve a bowl at/near center: center={center}"
    );
    assert!(
        corner > center,
        "corner (outside crater radius) should stay higher than the carved center: corner={corner} center={center}"
    );
}

#[test]
fn spike_removal_clamps_outlier() {
    let res = 32;
    let m = HeightfieldMetrics::new(res, res, res as f32, res as f32);
    let mut hf = Heightfield::filled(m, 20.0);
    // Inject a single sharp spike in an otherwise flat field.
    hf.set(res / 2, res / 2, 500.0);
    let p = EffectFilterParams {
        kind: EffectFilterKind::SpikeRemoval,
        strength: 1.0,
        radius: 2,
        amount: 5.0,
        ..EffectFilterParams::default()
    };
    let out = effect_filter(&hf, &p);
    let spike = out.get(res / 2, res / 2);
    assert!(
        spike < 100.0,
        "spike removal should clamp the outlier toward the local median, got {spike}"
    );
}

#[test]
fn shore_matches_coastal_flattening_behavior() {
    let res = 32;
    let m = HeightfieldMetrics::new(res, res, res as f32, res as f32);
    let mut hf = Heightfield::zeros(m);
    for j in 0..res {
        for i in 0..res {
            // Ramp from below sea level to well above it.
            let t = i as f32 / res as f32;
            hf.set(i, j, -30.0 + t * 80.0);
        }
    }
    let p = EffectFilterParams {
        kind: EffectFilterKind::Shore,
        strength: 1.0,
        sea_level: 0.0,
        beach_width: 10.0,
        ..EffectFilterParams::default()
    };
    let out = effect_filter(&hf, &p);
    // Underwater columns should be pulled toward sea level (flattened), same intent as `coastal`.
    let underwater_before = hf.get(1, 0);
    let underwater_after = out.get(1, 0);
    assert!(underwater_before < 0.0, "sanity: column starts underwater");
    assert!(
        underwater_after > underwater_before,
        "shore filter should raise underwater terrain toward sea level: before={underwater_before} after={underwater_after}"
    );
}

#[test]
fn strength_zero_is_near_identity() {
    let hf = noisy_field(24);
    let p = EffectFilterParams {
        kind: EffectFilterKind::Smooth,
        strength: 0.0,
        ..EffectFilterParams::default()
    };
    let out = effect_filter(&hf, &p);
    for j in 0..hf.metrics.height {
        for i in 0..hf.metrics.width {
            assert!(
                (hf.get(i, j) - out.get(i, j)).abs() < 1e-4,
                "strength=0 should leave heights unchanged at ({i},{j})"
            );
        }
    }
}

#[test]
fn invert_flips_delta_sign() {
    let res = 32;
    let m = HeightfieldMetrics::new(res, res, res as f32, res as f32);
    let mut hf = Heightfield::filled(m, 50.0);
    // A single raised bump so smoothing has a clear direction to pull toward.
    hf.set(res / 2, res / 2, 90.0);

    let base = EffectFilterParams {
        kind: EffectFilterKind::Smooth,
        strength: 1.0,
        radius: 3,
        iterations: 1,
        invert: false,
        ..EffectFilterParams::default()
    };
    let inverted = EffectFilterParams {
        invert: true,
        ..base.clone()
    };

    let smoothed = effect_filter(&hf, &base).get(res / 2, res / 2);
    let sharpened = effect_filter(&hf, &inverted).get(res / 2, res / 2);
    let original = hf.get(res / 2, res / 2);

    assert!(
        smoothed < original,
        "non-inverted smooth should pull the bump down toward neighbours: smoothed={smoothed} original={original}"
    );
    assert!(
        sharpened > original,
        "inverted smooth should flip the delta and push the bump up: sharpened={sharpened} original={original}"
    );
}

#[test]
fn shore_equivalent_to_coastal_defaults_when_flat() {
    // Sanity: `Shore` should behave consistently with the underlying `coastal` generator
    // it adapts, using the same sea level / beach width knobs.
    let res = 16;
    let m = HeightfieldMetrics::new(res, res, res as f32, res as f32);
    let hf = Heightfield::filled(m, -50.0);
    let coastal_p = CoastalParams {
        sea_level: 0.0,
        beach_width: 10.0,
        flatten_below: true,
        shelf_depth: 0.0,
    };
    let shore_p = EffectFilterParams {
        kind: EffectFilterKind::Shore,
        strength: 1.0,
        sea_level: 0.0,
        beach_width: 10.0,
        amount: 0.0,
        ..EffectFilterParams::default()
    };
    let coastal_out = terra_core::generators::coastal(&hf, &coastal_p);
    let shore_out = effect_filter(&hf, &shore_p);
    let a = coastal_out.get(0, 0);
    let b = shore_out.get(0, 0);
    assert!(
        (a - b).abs() < 1.0,
        "Shore filter with amount=0 shelf should closely track plain coastal flatten: coastal={a} shore={b}"
    );
}

fn max_abs_delta(a: &Heightfield, b: &Heightfield) -> f32 {
    let mut m = 0.0f32;
    for j in 0..a.metrics.height {
        for i in 0..a.metrics.width {
            m = m.max((a.get(i, j) - b.get(i, j)).abs());
        }
    }
    m
}

fn mountain_field(res: u32) -> Heightfield {
    let m = HeightfieldMetrics::new(res, res, res as f32 * 10.0, res as f32 * 10.0);
    let mut hf = Heightfield::zeros(m);
    let cx = res as f32 * 0.5;
    let cy = res as f32 * 0.5;
    for j in 0..res {
        for i in 0..res {
            let dx = i as f32 - cx;
            let dy = j as f32 - cy;
            let r = (dx * dx + dy * dy).sqrt() / (res as f32 * 0.45);
            let peak = (1.0 - r).max(0.0).powi(2) * 120.0;
            let ridge = ((i as f32 * 0.2).sin() * (j as f32 * 0.17).cos()) * 12.0;
            hf.set(i, j, peak + ridge + 20.0);
        }
    }
    hf
}

#[test]
fn morphology_filters_change_height() {
    let hf = mountain_field(64);
    // Cover every EffectFilterKind — including Wave D/E parity additions.
    for &kind in EffectFilterKind::ALL {
        let p = EffectFilterParams {
            kind,
            strength: 1.0,
            amount: 8.0,
            flow_threshold: 0.05,
            radius: 2,
            frequency: 0.04,
            crater_radius: 0.2,
            slope_min: 5.0,
            slope_max: 90.0,
            ..EffectFilterParams::default()
        };
        let out = effect_filter(&hf, &p);
        let delta = max_abs_delta(&hf, &out);
        // ZeroEdge / BorderBlend act on the rim; still measurable on 64².
        assert!(
            delta > 0.05,
            "{:?} should change height (max delta={delta})",
            kind
        );
    }
}

#[test]
fn effect_filter_kind_cycle_covers_all() {
    let mut k = EffectFilterKind::Smooth;
    let mut seen = std::collections::HashSet::new();
    for _ in 0..EffectFilterKind::ALL.len() {
        assert!(seen.insert(k), "cycle repeated {:?} early", k);
        k = k.cycle();
    }
    assert_eq!(k, EffectFilterKind::Smooth);
    assert_eq!(seen.len(), EffectFilterKind::ALL.len());
}

#[test]
fn effect_filter_kind_serde_snake_and_legacy_alias() {
    let snake: EffectFilterKind = serde_json::from_str("\"rocky_plateaus\"").expect("snake_case");
    assert_eq!(snake, EffectFilterKind::RockyPlateaus);
    let legacy: EffectFilterKind =
        serde_json::from_str("\"RockySharp\"").expect("legacy PascalCase alias");
    assert_eq!(legacy, EffectFilterKind::RockySharp);
}

#[test]
fn soft_flows_lowers_channels_on_ramp() {
    let res = 48;
    let m = HeightfieldMetrics::new(res, res, res as f32 * 8.0, res as f32 * 8.0);
    let mut hf = Heightfield::zeros(m);
    for j in 0..res {
        for i in 0..res {
            // Dome so hydraulic drainage incises draining slopes.
            let dx = i as f32 - res as f32 * 0.5;
            let dz = j as f32 - res as f32 * 0.5;
            hf.set(i, j, 120.0 - (dx * dx + dz * dz).sqrt() * 0.55);
        }
    }
    let p = EffectFilterParams {
        kind: EffectFilterKind::SoftFlows,
        strength: 1.0,
        amount: 12.0,
        flow_threshold: 0.05,
        radius: 2,
        ..EffectFilterParams::default()
    };
    let out = effect_filter(&hf, &p);
    let max_dh = hf
        .to_dense()
        .iter()
        .zip(out.to_dense().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_dh > 0.15,
        "soft flows should reshape drainage (max_dh={max_dh})"
    );
    assert!(
        out.get(2, 2) <= hf.get(2, 2) + 0.5,
        "soft flows must not fake-raise far-field divides"
    );
}

#[test]
fn kuwahara_preserves_step_edge_better_than_box_blur_intent() {
    let res = 32;
    let m = HeightfieldMetrics::new(res, res, res as f32, res as f32);
    let mut hf = Heightfield::zeros(m);
    for j in 0..res {
        for i in 0..res {
            let base = if i < res / 2 { 10.0 } else { 40.0 };
            let noise = if (i + j) % 2 == 0 { 1.5 } else { -1.5 };
            hf.set(i, j, base + noise);
        }
    }
    let p = EffectFilterParams {
        kind: EffectFilterKind::Kuwahara,
        strength: 1.0,
        radius: 2,
        iterations: 1,
        ..EffectFilterParams::default()
    };
    let out = effect_filter(&hf, &p);
    let left = out.get(res / 4, res / 2);
    let right = out.get(3 * res / 4, res / 2);
    assert!(
        (right - left).abs() > 20.0,
        "kuwahara should keep the step edge: left={left} right={right}"
    );
}

#[test]
fn noise_gabor_and_phasor_are_deterministic() {
    let hf = mountain_field(24);
    for kind in [EffectFilterKind::NoiseGabor, EffectFilterKind::NoisePhasor] {
        let p = EffectFilterParams {
            kind,
            strength: 1.0,
            amount: 5.0,
            frequency: 0.05,
            seed: 99,
            ..EffectFilterParams::default()
        };
        let a = effect_filter(&hf, &p);
        let b = effect_filter(&hf, &p);
        assert!(
            max_abs_delta(&a, &b) < 1e-5,
            "{kind:?} must be deterministic"
        );
        assert!(
            max_abs_delta(&hf, &a) > 0.1,
            "{kind:?} should modulate height"
        );
    }
}

#[test]
fn talus_moves_mass_on_steep_face() {
    let res = 32;
    let m = HeightfieldMetrics::new(res, res, 320.0, 320.0);
    let mut hf = Heightfield::zeros(m);
    for j in 0..res {
        for i in 0..res {
            // Vertical cliff at mid-X.
            let h = if i < res / 2 { 100.0 } else { 10.0 };
            hf.set(i, j, h);
        }
    }
    let p = EffectFilterParams {
        kind: EffectFilterKind::TalusFill,
        strength: 1.0,
        amount: 8.0,
        slope_min: 30.0,
        iterations: 3,
        ..EffectFilterParams::default()
    };
    let out = effect_filter(&hf, &p);
    // Low side near the cliff should gain some material.
    let low_near = out.get(res / 2 + 1, res / 2);
    assert!(
        low_near > hf.get(res / 2 + 1, res / 2) + 0.05,
        "talus should deposit below the cliff: got {low_near}"
    );
}

// ── Phase 7 arid / rock landforms (geomorph-composed) ───────────────────────

fn valley_field(res: u32) -> Heightfield {
    let m = HeightfieldMetrics::new(res, res, res as f32 * 8.0, res as f32 * 8.0);
    let mut hf = Heightfield::zeros(m);
    let mid = res as f32 * 0.5;
    for j in 0..res {
        for i in 0..res {
            // Broad valley along X midline with gentle tributary slopes.
            let dx = (i as f32 - mid).abs() / mid;
            let branch = ((j as f32 * 0.15).sin() * 8.0).abs() * (1.0 - dx);
            let h = 40.0 + dx * dx * 55.0 + branch;
            hf.set(i, j, h);
        }
    }
    hf
}

#[test]
fn canyon_incises_drainage_not_noise() {
    let hf = valley_field(48);
    let p = EffectFilterParams {
        kind: EffectFilterKind::Canyon,
        strength: 1.0,
        amount: 14.0,
        radius: 4,
        flow_threshold: 0.55,
        rock_hardness: 0.35,
        wall_steepness: 0.75,
        valley_floor: 0.25,
        talus_mix: 0.0,
        iterations: 2,
        ..EffectFilterParams::default()
    };
    let out = effect_filter(&hf, &p);
    let mid = res_mid(&hf);
    // Deepest incision follows the drainage trunk (valley centre column), not
    // necessarily the geometric mid-cell — outlets sit near the N/S edges.
    let mut channel_drop = 0.0f32;
    for j in 0..hf.metrics.height {
        channel_drop = channel_drop.max(hf.get(mid, j) - out.get(mid, j));
    }
    let ridge_drop = hf.get(2, mid) - out.get(2, mid);
    assert!(
        channel_drop > 1.5,
        "canyon should deepen the drainage trunk: drop={channel_drop}"
    );
    assert!(
        channel_drop > ridge_drop + 0.5,
        "incision should concentrate on channels (channel={channel_drop} ridge={ridge_drop})"
    );
}

#[test]
fn cliffs_sharpen_steep_relief() {
    let res = 40;
    let m = HeightfieldMetrics::new(res, res, 400.0, 400.0);
    let mut hf = Heightfield::zeros(m);
    for j in 0..res {
        for i in 0..res {
            let h = if i < res / 2 { 90.0 } else { 15.0 };
            hf.set(i, j, h);
        }
    }
    let p = EffectFilterParams {
        kind: EffectFilterKind::Cliffs,
        strength: 1.0,
        amount: 8.0,
        radius: 3,
        slope_min: 20.0,
        flow_threshold: 0.1,
        rock_hardness: 0.6,
        wall_steepness: 0.7,
        talus_mix: 0.35,
        iterations: 2,
        ..EffectFilterParams::default()
    };
    let out = effect_filter(&hf, &p);
    assert!(
        max_abs_delta(&hf, &out) > 0.5,
        "cliffs should modify the escarpment"
    );
}

#[test]
fn chipped_only_removes_from_faces() {
    let res = 36;
    let m = HeightfieldMetrics::new(res, res, 360.0, 360.0);
    let mut hf = Heightfield::zeros(m);
    for j in 0..res {
        for i in 0..res {
            let h = if i < res / 2 { 80.0 } else { 20.0 };
            hf.set(i, j, h);
        }
    }
    let p = EffectFilterParams {
        kind: EffectFilterKind::Chipped,
        strength: 1.0,
        amount: 6.0,
        frequency: 0.1,
        slope_min: 25.0,
        rock_hardness: 0.4,
        radius: 2,
        flow_threshold: 0.1,
        ..EffectFilterParams::default()
    };
    let out = effect_filter(&hf, &p);
    // Flat plateaus far from the face should stay nearly unchanged.
    let plateau_delta = (out.get(2, res / 2) - hf.get(2, res / 2)).abs();
    assert!(
        plateau_delta < 0.35,
        "chipped must not sprinkle bumps on flats: delta={plateau_delta}"
    );
    // Near the face some material should be removed (net lower or equal).
    let mut face_removed = false;
    for j in 0..res {
        for i in (res / 2).saturating_sub(2)..(res / 2 + 2).min(res) {
            if out.get(i, j) < hf.get(i, j) - 0.05 {
                face_removed = true;
            }
        }
    }
    assert!(
        face_removed,
        "chipped should erode chunks from the rock face"
    );
}

#[test]
fn rocky_layers_exposes_strata_on_cliffs() {
    let res = 40;
    let m = HeightfieldMetrics::new(res, res, 400.0, 400.0);
    let mut hf = Heightfield::zeros(m);
    for j in 0..res {
        for i in 0..res {
            let h = if i < res / 2 { 100.0 } else { 20.0 };
            hf.set(i, j, h);
        }
    }
    let p = EffectFilterParams::rocky_layers();
    let mut p = p;
    p.strength = 1.0;
    p.amount = 8.0;
    let out = effect_filter(&hf, &p);
    assert!(
        max_abs_delta(&hf, &out) > 0.2,
        "rocky layers should recess soft beds on the face"
    );
}

#[test]
fn rocky_hard_preserves_high_points_more_than_soft() {
    let hf = mountain_field(40);
    let hard = EffectFilterParams {
        kind: EffectFilterKind::RockyHard,
        strength: 1.0,
        amount: 8.0,
        rock_hardness: 0.9,
        frequency: 0.05,
        slope_min: 30.0,
        iterations: 2,
        talus_mix: 0.2,
        ..EffectFilterParams::default()
    };
    let out_hard = effect_filter(&hf, &hard);
    // Peak should remain a local high after differential stripping.
    let peak = out_hard.get(20, 20);
    let flank = out_hard.get(8, 20);
    assert!(
        max_abs_delta(&hf, &out_hard) > 0.5,
        "rocky hard should reshape the DEM"
    );
    assert!(
        peak > flank,
        "hard-rock peak should stay above stripped flanks: peak={peak} flank={flank}"
    );
}

#[test]
fn arid_params_serde_defaults() {
    let json = r#"{
        "kind": "canyon",
        "strength": 0.6,
        "invert": false,
        "radius": 4,
        "iterations": 2,
        "seed": 1,
        "frequency": 0.02,
        "amount": 10.0,
        "sea_level": 0.0,
        "beach_width": 8.0,
        "crater_radius": 0.18
    }"#;
    let p: EffectFilterParams =
        serde_json::from_str(json).expect("legacy params without arid fields");
    assert!((p.rock_hardness - 0.55).abs() < 1e-5);
    assert!((p.wall_steepness - 0.65).abs() < 1e-5);
    assert!((p.valley_floor - 0.25).abs() < 1e-5);
    assert!((p.talus_mix - 0.35).abs() < 1e-5);
    // Phase 10 noise defaults.
    assert_eq!(p.octaves, 5);
    assert!((p.lacunarity - 2.0).abs() < 1e-5);
    assert!((p.persistence - 0.5).abs() < 1e-5);
    assert!((p.anisotropy - 1.0).abs() < 1e-5);
}

#[test]
fn phase10_noise_families_use_distinct_kernels_and_scale_m() {
    let hf = mountain_field(48);
    let base = EffectFilterParams {
        strength: 1.0,
        amount: 10.0,
        scale_m: 40.0,
        octaves: 3,
        seed: 11,
        ..EffectFilterParams::default()
    };
    let perlin = effect_filter(
        &hf,
        &EffectFilterParams {
            kind: EffectFilterKind::NoisePerlin,
            ..base.clone()
        },
    );
    let gabor = effect_filter(
        &hf,
        &EffectFilterParams {
            kind: EffectFilterKind::NoiseGabor,
            ..base.clone()
        },
    );
    let phasor = effect_filter(
        &hf,
        &EffectFilterParams {
            kind: EffectFilterKind::NoisePhasor,
            ..base.clone()
        },
    );
    let billow = effect_filter(
        &hf,
        &EffectFilterParams {
            kind: EffectFilterKind::NoiseBillow,
            ..base
        },
    );
    // Distinct families must not collapse to identical fields.
    assert!(max_abs_delta(&perlin, &gabor) > 0.5);
    assert!(max_abs_delta(&perlin, &phasor) > 0.5);
    assert!(max_abs_delta(&gabor, &phasor) > 0.5);
    assert!(max_abs_delta(&perlin, &billow) > 0.2);
}

#[test]
fn phase10_voronoi_f1_f2_difference_modes_differ() {
    let hf = mountain_field(32);
    let mut f1 = EffectFilterParams::design_voronoi();
    f1.strength = 1.0;
    f1.voronoi_feature = terra_core::layer::WorleyFeature::F1;
    let mut f2 = f1.clone();
    f2.voronoi_feature = terra_core::layer::WorleyFeature::F2;
    let mut diff = f1.clone();
    diff.voronoi_feature = terra_core::layer::WorleyFeature::F2MinusF1;
    let a = effect_filter(&hf, &f1);
    let b = effect_filter(&hf, &f2);
    let c = effect_filter(&hf, &diff);
    assert!(max_abs_delta(&a, &b) > 0.2);
    assert!(max_abs_delta(&a, &c) > 0.2);
    assert!(max_abs_delta(&b, &c) > 0.2);
}

#[test]
fn phase10_border_blend_and_zero_edge_are_world_space() {
    let hf = mountain_field(48);
    let mut border = EffectFilterParams::border_blend();
    border.strength = 1.0;
    border.amount = 40.0; // metres
    let mut zero = EffectFilterParams::zero_edge();
    zero.strength = 1.0;
    zero.amount = 40.0;
    let b = effect_filter(&hf, &border);
    let z = effect_filter(&hf, &zero);
    // Corners should move more than the center.
    let mid = res_mid(&hf);
    let center_db = (hf.get(mid, mid) - b.get(mid, mid)).abs();
    let corner_db = (hf.get(0, 0) - b.get(0, 0)).abs();
    assert!(corner_db > center_db);
    let center_dz = (hf.get(mid, mid) - z.get(mid, mid)).abs();
    let corner_dz = (hf.get(0, 0) - z.get(0, 0)).abs();
    assert!(corner_dz > center_dz);
}

#[test]
fn phase10_wind_carve_aeolian_path_changes_height() {
    let hf = mountain_field(40);
    let mut p = EffectFilterParams::wind_carve();
    p.strength = 1.0;
    p.iterations = 4; // aeolian path
    p.amount = 6.0;
    let out = effect_filter(&hf, &p);
    assert!(max_abs_delta(&hf, &out) > 0.05);
}

#[test]
fn phase10_directional_blur_differs_from_angle_blur() {
    let hf = mountain_field(40);
    let mut angle = EffectFilterParams::angle_blur();
    angle.strength = 1.0;
    let mut dir = EffectFilterParams::directional_blur();
    dir.strength = 1.0;
    dir.rotation_deg = 0.0;
    let a = effect_filter(&hf, &angle);
    let d = effect_filter(&hf, &dir);
    assert!(max_abs_delta(&hf, &a) > 0.05);
    assert!(max_abs_delta(&hf, &d) > 0.05);
    assert!(max_abs_delta(&a, &d) > 0.05);
}

#[test]
fn phase10_crater_has_rim_raise() {
    let m = HeightfieldMetrics::new(64, 64, 640.0, 640.0);
    let hf = Heightfield::filled(m, 100.0);
    let mut p = EffectFilterParams::crater();
    p.strength = 1.0;
    p.amount = 40.0;
    p.crater_radius = 0.22;
    let out = effect_filter(&hf, &p);
    let mid = 32u32;
    let center_dh = out.get(mid, mid) - hf.get(mid, mid);
    // Sample near geometric rim (~0.22 * 640 ≈ 140m → ~14 cells).
    let rim_i = mid + 14;
    let rim_dh = out.get(rim_i, mid) - hf.get(rim_i, mid);
    assert!(
        center_dh < -5.0,
        "crater center should excavate: dh={center_dh}"
    );
    assert!(rim_dh > 0.5, "crater rim should raise: dh={rim_dh}");
    assert!(center_dh < rim_dh);
}

fn res_mid(hf: &Heightfield) -> u32 {
    hf.metrics.width / 2
}
