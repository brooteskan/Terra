//! Geological terrace + strata fixtures (Phase 9 family).

use terra_core::fields::{
    erodibility_at_strata_depth, hardness_at_strata_depth, stability_at_strata_depth,
};
use terra_core::generators;
use terra_core::generators::geology::{
    strata_depth_m, terrace_irregular, terrace_simple, terrace_steep, TerraceControls,
};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{
    BedGeometry, EffectFilterParams, MaterialsParams, Stratum, StratumMaterial,
};

fn ramp(res: u32) -> Heightfield {
    let m = HeightfieldMetrics::new(res, res, res as f32 * 4.0, res as f32 * 4.0);
    let mut hf = Heightfield::zeros(m);
    for j in 0..res {
        for i in 0..res {
            let t = i as f32 / (res - 1) as f32;
            hf.set(i, j, t * 80.0);
        }
    }
    hf
}

#[test]
fn simple_terrace_quantizes_without_post_noise() {
    let hf = ramp(48);
    let c = TerraceControls {
        height_m: 10.0,
        levels: 8,
        offset: 0.0,
        top_smoothness: 0.0,
        riser_sharpness: 0.9,
        ..TerraceControls::default()
    };
    let out = terrace_simple(&hf, &c);
    // Flat treads: many cells share identical heights.
    let mut unique = std::collections::BTreeSet::new();
    for j in 0..48 {
        for i in 0..48 {
            let h = (out.get(i, j) * 100.0).round() as i32;
            unique.insert(h);
        }
    }
    assert!(
        unique.len() < 20,
        "simple terrace should collapse to few bands, got {}",
        unique.len()
    );
}

#[test]
fn irregular_terrace_is_deterministic_and_differs_from_simple() {
    let hf = ramp(40);
    let mut c = TerraceControls {
        height_m: 8.0,
        riser_sharpness: 0.6,
        frequency: 0.05,
        seed: 99,
        top_smoothness: 0.1,
        ..TerraceControls::default()
    };
    let a = terrace_irregular(&hf, &c);
    let b = terrace_irregular(&hf, &c);
    assert_eq!(a.to_dense(), b.to_dense());
    c.seed = 100;
    let simple = terrace_simple(&hf, &c);
    let irreg = terrace_irregular(&hf, &c);
    let mut diff = 0.0f32;
    for j in 0..40 {
        for i in 0..40 {
            diff += (simple.get(i, j) - irreg.get(i, j)).abs();
        }
    }
    assert!(diff > 1.0, "irregular should spatially perturb banding");
}

#[test]
fn steep_terrace_prefers_slopes() {
    let m = HeightfieldMetrics::new(32, 32, 128.0, 128.0);
    let mut hf = Heightfield::filled(m, 20.0);
    // Steep ramp on the right half.
    for j in 0..32 {
        for i in 16..32 {
            hf.set(i, j, 20.0 + (i as f32 - 16.0) * 4.0);
        }
    }
    let c = TerraceControls {
        height_m: 6.0,
        riser_sharpness: 0.95,
        top_smoothness: 0.0,
        slope_min: 15.0,
        slope_max: 50.0,
        ..TerraceControls::default()
    };
    let out = terrace_steep(&hf, &c);
    let flat_delta: f32 = (0..32).map(|j| (out.get(4, j) - hf.get(4, j)).abs()).sum();
    let steep_delta: f32 = (0..32)
        .map(|j| (out.get(28, j) - hf.get(28, j)).abs())
        .sum();
    assert!(
        steep_delta > flat_delta * 1.5,
        "steep terrace should change slopes more than flats ({steep_delta} vs {flat_delta})"
    );
}

#[test]
fn effect_filter_terrace_presets_dispatch() {
    let hf = ramp(24);
    for p in [
        EffectFilterParams::terrace_simple(),
        EffectFilterParams::terrace_irregular(),
        EffectFilterParams::terrace_steep(),
    ] {
        let out = generators::effect_filter(&hf, &p);
        assert_eq!(out.metrics, hf.metrics);
    }
}

#[test]
fn stratum_erodibility_and_stability() {
    let soft = Stratum::soft_cap(5.0);
    let hard = Stratum::hard_base();
    assert!(soft.effective_erodibility() > hard.effective_erodibility());
    assert!(hard.material_stability() > soft.material_stability());
    let strata = MaterialsParams::soft_over_hard(5.0).strata;
    assert!(
        (erodibility_at_strata_depth(&strata, 0.0, 0.5) - soft.effective_erodibility()).abs()
            < 1e-5
    );
    assert!(stability_at_strata_depth(&strata, 6.0, 0.5) > 0.7);
    assert!((hardness_at_strata_depth(&strata, 6.0, 0.5) - 0.92).abs() < 1e-5);
}

#[test]
fn bed_geometry_warps_strata_depth() {
    let geom = BedGeometry::Tilted {
        dip_deg: 30.0,
        azimuth_deg: 0.0,
    };
    let d0 = strata_depth_m(100.0, 90.0, 0.0, 0.0, &geom);
    let d1 = strata_depth_m(100.0, 90.0, 50.0, 0.0, &geom);
    assert!((d0 - 10.0).abs() < 1e-4);
    assert!(d1 > d0, "down-dip should increase effective depth");
}

#[test]
fn strata_filter_uses_geology() {
    let hf = ramp(32);
    let p = EffectFilterParams::strata();
    let out = generators::effect_filter(&hf, &p);
    let mut diff = 0.0f32;
    for j in 0..32 {
        for i in 0..32 {
            diff += (out.get(i, j) - hf.get(i, j)).abs();
        }
    }
    assert!(diff > 0.5);
}

#[test]
fn sedimentary_vs_igneous_stability() {
    let sed = Stratum {
        name: "Shale".into(),
        id: 2,
        hardness: 0.5,
        thickness: 10.0,
        erodibility: 0.5,
        material_type: StratumMaterial::Sedimentary,
    };
    let ign = Stratum {
        name: "Granite".into(),
        id: 1,
        hardness: 0.5,
        thickness: 10.0,
        erodibility: 0.5,
        material_type: StratumMaterial::Igneous,
    };
    assert!(ign.material_stability() > sed.material_stability());
}
