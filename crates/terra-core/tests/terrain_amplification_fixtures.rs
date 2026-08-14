//! Drainage-conditioned multi-scale amplification fixtures.

use terra_core::analyze::{amplify_terrain, AmplificationBands, TerrainAmplificationParams};
use terra_core::authoring::{geomorphic_detail_with_hardness, GeomorphicDetailParams};
use terra_core::fields::keys;
use terra_core::geomorph::{noisy_mountain, single_valley};
use terra_core::heightfield::HeightfieldMetrics;
use terra_core::mask::MaskField;

#[test]
fn geomorphic_detail_publishes_amplification_outputs() {
    let m = HeightfieldMetrics::new(48, 48, 1600.0, 1600.0);
    let hf = noisy_mountain(m);
    let p = GeomorphicDetailParams::default();
    let out = geomorphic_detail_with_hardness(&hf, &p, None, None, None);
    assert!(out.fields.contains_key(keys::DETAIL_MASK));
    assert!(out.fields.contains_key(keys::FINE_FLOW));
    assert!(out.fields.contains_key(keys::MICRO_CHANNEL));
    assert!(out.fields.contains_key(keys::RIDGE_BREAKUP));
    assert!(out.fields.contains_key(keys::FINE_EROSION));
    assert_ne!(out.height.to_dense(), hf.to_dense());
}

#[test]
fn bands_nest_micro_under_meso_under_macro() {
    let bands = AmplificationBands::from_world(HeightfieldMetrics::new(128, 128, 4000.0, 4000.0));
    assert!(bands.micro_m.1 <= bands.meso_m.0 * 1.05);
    assert!(bands.meso_m.1 <= bands.macro_m.0 * 1.05);
}

#[test]
fn no_detail_without_geomorphic_gate() {
    // Flat plane: slope gate should suppress structured amplify.
    let m = HeightfieldMetrics::new(32, 32, 1000.0, 1000.0);
    let hf = terra_core::heightfield::Heightfield::zeros(m);
    let p = TerrainAmplificationParams {
        slope_gate: 0.2,
        meso_amplitude_m: 10.0,
        micro_amplitude_m: 2.0,
        ..Default::default()
    };
    let out = amplify_terrain(&hf, &p, None, None, None);
    let mut max_d = 0.0f32;
    for j in 0..m.height {
        for i in 0..m.width {
            max_d = max_d.max((out.height.get(i, j) - hf.get(i, j)).abs());
        }
    }
    assert!(
        max_d < 0.05,
        "flat terrain should not grow soup detail, max_d={max_d}"
    );
}

#[test]
fn hardness_changes_conditioned_detail() {
    let m = HeightfieldMetrics::new(48, 48, 1600.0, 1600.0);
    let hf = single_valley(m);
    let soft = MaskField::filled(m, 0.0);
    let hard = MaskField::filled(m, 1.0);
    let p = TerrainAmplificationParams {
        meso_amplitude_m: 10.0,
        micro_amplitude_m: 2.5,
        rock_roughness: 1.0,
        gully_strength: 1.0,
        ridge_breakup: 1.0,
        ..Default::default()
    };
    let a = amplify_terrain(&hf, &p, None, Some(&soft), None);
    let b = amplify_terrain(&hf, &p, None, Some(&hard), None);
    assert_ne!(
        a.height.to_dense(),
        b.height.to_dense(),
        "lithology/hardness should condition rock roughness detail"
    );
}
