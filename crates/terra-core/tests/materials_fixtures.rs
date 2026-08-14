//! Phase F materials / strata fixtures: Materials→K bake + soft-over-hard stripping.

use std::collections::HashMap;
use terra_core::analyze::{thermal_erode, thermal_erode_with_strata};
use terra_core::fields::{
    bake_hardness_from_materials, bake_hardness_from_materials_ex, hardness_at_strata_depth,
};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::hydro::stream_power_erode_with_strata;
use terra_core::layer::{
    MaterialsParams, Stratum, StreamPowerParams, ThermalErosionParams, OPEN_HEIGHT_MAX,
    OPEN_HEIGHT_MIN,
};
use terra_core::mask::MaskField;
use terra_core::surface::{bake_materials, material_weights};

fn plateau(res: u32, height: f32) -> Heightfield {
    let m = HeightfieldMetrics::new(res, res, res as f32 * 2.0, res as f32 * 2.0);
    let mut hf = Heightfield::filled(m, height * 0.35);
    let c = res as f32 * 0.5;
    let r = res as f32 * 0.28;
    for j in 0..res {
        for i in 0..res {
            let dx = i as f32 - c;
            let dz = j as f32 - c;
            if dx * dx + dz * dz < r * r {
                hf.set(i, j, height);
            }
        }
    }
    hf
}

#[test]
fn materials_bake_deterministic() {
    let hf = plateau(32, 40.0);
    let mats = MaterialsParams::default();
    let a = bake_materials(&hf, &mats, None, &HashMap::new(), &[], None);
    let b = bake_materials(&hf, &mats, None, &HashMap::new(), &[], None);
    assert_eq!(a.0.data(), b.0.data());
    assert_eq!(a.1.data(), b.1.data());
}

#[test]
fn finite_open_material_bounds_match_infinite_reference_at_supported_extremes() {
    let finite = MaterialsParams::default();
    let mut infinite = finite.clone();
    for rule in &mut infinite.rules {
        if rule.min_height == OPEN_HEIGHT_MIN {
            rule.min_height = f32::NEG_INFINITY;
        }
        if rule.max_height == OPEN_HEIGHT_MAX {
            rule.max_height = f32::INFINITY;
        }
    }

    for height in [-200_000.0, 0.0, 200_000.0] {
        let metrics = HeightfieldMetrics::new(4, 4, 4.0, 4.0);
        let hf = Heightfield::filled(metrics, height);
        let finite_ids = material_weights(&hf, &finite, None, &HashMap::new(), &[]);
        let infinite_ids = material_weights(&hf, &infinite, None, &HashMap::new(), &[]);
        assert_eq!(finite_ids.data(), infinite_ids.data(), "height={height}");
    }
}

#[test]
fn materials_rules_bake_hardness_rock_harder_than_grass() {
    let hf = plateau(28, 50.0);
    let mats = MaterialsParams::default();
    let weights = material_weights(&hf, &mats, None, &HashMap::new(), &[]);
    let hardness = bake_hardness_from_materials(&weights, &mats.rules, mats.default_hardness);
    assert!(hardness.data().iter().all(|&v| (0.0..=1.0).contains(&v)));
    let slope = terra_core::analyze::slope_degrees(&hf);
    let mut steep_k = 0.0f32;
    let mut flat_k = 0.0f32;
    let mut n_steep = 0u32;
    let mut n_flat = 0u32;
    for j in 0..28 {
        for i in 0..28 {
            let s = slope.get(i, j) * 90.0;
            let k = hardness.get(i, j);
            if s > 35.0 {
                steep_k += k;
                n_steep += 1;
            } else if s < 20.0 {
                flat_k += k;
                n_flat += 1;
            }
        }
    }
    assert!(n_steep > 0 && n_flat > 0);
    assert!(
        steep_k / n_steep as f32 > flat_k / n_flat as f32,
        "rock slopes should bake harder than grass flats"
    );
}

#[test]
fn strata_hardness_at_depth_soft_then_hard() {
    let strata = MaterialsParams::soft_over_hard(5.0).strata;
    assert!((hardness_at_strata_depth(&strata, 0.0, 0.5) - 0.08).abs() < 1e-5);
    assert!((hardness_at_strata_depth(&strata, 2.0, 0.5) - 0.08).abs() < 1e-5);
    assert!((hardness_at_strata_depth(&strata, 5.01, 0.5) - 0.92).abs() < 1e-5);
}

#[test]
fn soft_over_hard_thermal_strips_soft_then_resists() {
    let m = HeightfieldMetrics::new(32, 32, 64.0, 64.0);
    let mut hf = Heightfield::filled(m, 10.0);
    // Steep cone so talus redistribution has work to do.
    for j in 0..32 {
        for i in 0..32 {
            let dx = i as f32 - 16.0;
            let dz = j as f32 - 16.0;
            let r = (dx * dx + dz * dz).sqrt();
            let h = (28.0 - r * 1.6).max(10.0);
            hf.set(i, j, h);
        }
    }
    let mats = MaterialsParams::soft_over_hard(5.0);
    let reference = MaskField::from_raw(hf.metrics, &hf.to_dense());
    let p = ThermalErosionParams {
        talus_angle_deg: 20.0,
        iterations: 60,
        strength: 0.9,
        hardness: 0.0,
        ..ThermalErosionParams::default()
    };
    let (out, _, _) =
        thermal_erode_with_strata(&hf, &p, &reference, &mats.strata, mats.default_hardness);

    let soft_p = ThermalErosionParams {
        hardness: 0.08,
        ..p.clone()
    };
    let (out_soft, _, _) = thermal_erode(&hf, &soft_p);

    let drop_strata = hf.get(16, 16) - out.get(16, 16);
    let drop_soft = hf.get(16, 16) - out_soft.get(16, 16);

    assert!(
        drop_strata > 1.0,
        "soft cap should strip on steep cone: drop={drop_strata}"
    );
    assert!(
        drop_strata < drop_soft * 0.92 + 0.35,
        "hard base should resist vs all-soft: strata={drop_strata} soft={drop_soft}"
    );

    let depth = (reference.get(16, 16) - out.get(16, 16)).max(0.0);
    let k = hardness_at_strata_depth(&mats.strata, depth, 0.5);
    if depth > 5.0 {
        assert!(k > 0.7, "exposed base should be hard: k={k} depth={depth}");
    }
}

#[test]
fn soft_over_hard_spe_respects_baked_k() {
    let res = 36;
    let hf = plateau(res, 45.0);
    let mats = MaterialsParams::soft_over_hard(6.0);
    let reference = MaskField::from_raw(hf.metrics, &hf.to_dense());
    let p = StreamPowerParams {
        iterations: 10,
        k: 0.12,
        dendritic_seed: 0.2,
        refill_each_iter: true,
        ..StreamPowerParams::default()
    };
    let soft =
        stream_power_erode_with_strata(&hf, &p, &reference, &[Stratum::soft_cap(1.0e6)], 0.5);
    let layered = stream_power_erode_with_strata(&hf, &p, &reference, &mats.strata, 0.5);
    let cx = res / 2;
    let drop_soft = hf.get(cx, cx) - soft.height.get(cx, cx);
    let drop_layered = hf.get(cx, cx) - layered.height.get(cx, cx);
    assert!(
        drop_layered < drop_soft * 0.9 + 0.25,
        "hard base should limit SPE vs infinite soft: layered={drop_layered} soft={drop_soft}"
    );
}

#[test]
fn materials_ex_bake_matches_strata_surface() {
    let hf = plateau(16, 20.0);
    let mats = MaterialsParams::soft_over_hard(8.0);
    let weights = material_weights(&hf, &mats, None, &HashMap::new(), &[]);
    let k =
        bake_hardness_from_materials_ex(&weights, &mats.rules, &mats.strata, mats.default_hardness);
    let expect = hardness_at_strata_depth(&mats.strata, 0.0, 0.5);
    assert!(k.data().iter().all(|&v| (v - expect).abs() < 1e-4));
}
