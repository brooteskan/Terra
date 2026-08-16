//! #95 regression: the reef "Shelf Flatten" biome layer must not bleed the
//! shallow shelf into adjacent deep bathymetry across the shelf->basin depth
//! discontinuity.
//!
//! This guards the *shipped* Shelf Flatten config at the filter level,
//! independent of the Tropical Island preset. Terrain tuning (seed,
//! erosion, `base_level`) can move the reef biome off the discontinuity and
//! silently stop exercising this path — which is exactly how #18 stopped
//! failing without being fixed — so the guard lives here, on a synthetic
//! field, not in the preset-evaluation test.

use terra_core::generators::effect_filter;
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::EffectFilterParams;
use terra_core::{BiomeLibrary, LayerKind};

const RES: u32 = 64;
/// Shelf occupies columns `0..=SHELF_LAST_COL`; the basin is everything to the
/// right, so the discontinuity sits on the 31|32 boundary.
const SHELF_LAST_COL: u32 = 31;
const SHELF_DEPTH: f32 = -7.0;
const BASIN_DEPTH: f32 = -218.0;
const RIPPLE: f32 = 2.0;

/// Synthetic bathymetry: a rippled shallow shelf abutting a flat deep basin
/// across a sharp (~211 m) discontinuity.
fn shelf_basin_field() -> Heightfield {
    let m = HeightfieldMetrics::new(RES, RES, RES as f32 * 10.0, RES as f32 * 10.0);
    let mut hf = Heightfield::zeros(m);
    for j in 0..RES {
        for i in 0..RES {
            let v = if i <= SHELF_LAST_COL {
                // Deterministic +/-2 m checkerboard so the smoother has real
                // high-frequency detail to attenuate on the shelf itself.
                let ripple = if (i + j) % 2 == 0 { RIPPLE } else { -RIPPLE };
                SHELF_DEPTH + ripple
            } else {
                BASIN_DEPTH
            };
            hf.set(i, j, v);
        }
    }
    hf
}

/// Pull the shipped "Shelf Flatten" params straight out of the Tropical Island
/// palette so this test tracks whatever config actually ships. Fails loudly if
/// the layer is renamed or is no longer an EffectFilter.
fn shipped_shelf_flatten() -> EffectFilterParams {
    let lib = BiomeLibrary::tropical_island_palette();
    for def in &lib.definitions {
        for (name, kind) in &def.terrain_layers {
            if name == "Shelf Flatten" {
                match kind {
                    LayerKind::EffectFilter(p) => return p.clone(),
                    _ => panic!("'Shelf Flatten' is no longer an EffectFilter layer"),
                }
            }
        }
    }
    panic!("Tropical Island palette no longer has a 'Shelf Flatten' terrain layer");
}

fn variance(vals: &[f32]) -> f32 {
    let n = vals.len() as f32;
    let mean = vals.iter().sum::<f32>() / n;
    vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n
}

#[test]
fn shelf_flatten_preserves_adjacent_deep_bathymetry() {
    let field = shelf_basin_field();
    let params = shipped_shelf_flatten();
    let out = effect_filter(&field, &params);

    // (1) Deep preservation — the headline. Every basin cell must stay within
    // 5 m of the true basin depth. The box-blur `Smooth` lifts the first basin
    // column by ~42 m and the second by ~21 m (fails); an edge-aware kernel
    // keeps the whole basin within ~1 m (passes).
    let mut worst = 0.0f32;
    let mut worst_at = (0u32, 0u32);
    for j in 0..RES {
        for i in (SHELF_LAST_COL + 1)..RES {
            let delta = (out.get(i, j) - BASIN_DEPTH).abs();
            if delta > worst {
                worst = delta;
                worst_at = (i, j);
            }
        }
    }
    assert!(
        worst <= 5.0,
        "shelf bled into the basin: max deviation {worst:.1} m at (col,row)={worst_at:?} \
         (shipped Shelf Flatten kind={:?}, strength={})",
        params.kind,
        params.strength,
    );

    // (2) Anti-no-op — an identity "fix" must not pass (1). The smoother must
    // actually attenuate the shelf ripple in the interior (columns 8..=24,
    // whose radius-2 windows never reach the basin).
    let interior: Vec<(u32, u32)> = (8..=56)
        .flat_map(|j| (8..=24).map(move |i| (i, j)))
        .collect();
    let before: Vec<f32> = interior.iter().map(|&(i, j)| field.get(i, j)).collect();
    let after: Vec<f32> = interior.iter().map(|&(i, j)| out.get(i, j)).collect();
    let var_before = variance(&before);
    let var_after = variance(&after);
    assert!(
        var_after <= var_before * 0.70,
        "Shelf Flatten no longer smooths the shelf interior: variance {var_before:.2} -> {var_after:.2}"
    );

    // (3) Symptom sentinel tied to #18: the deep basin must survive. A smoother
    // cannot push below the field minimum, so this only trips if a future
    // change raises/clamps the whole basin — the #18 collapse class.
    let (min_h, _max_h) = out.min_max();
    assert!(
        min_h <= -210.0,
        "deep bathymetry collapsed: basin min {min_h:.1} m (expected ~{BASIN_DEPTH})"
    );
}

#[test]
fn raw_smooth_is_not_depth_preserving_by_design() {
    // Documents (and pins) that the general-purpose `Smooth` box blur bleeds
    // across a depth discontinuity — so this property is asserted, never
    // silently rediscovered. If `Smooth` is ever made depth-aware, update this
    // test deliberately rather than letting the reef path rely on it by
    // accident.
    let field = shelf_basin_field();
    let params = EffectFilterParams {
        strength: 1.0,
        ..EffectFilterParams::smooth()
    };
    let out = effect_filter(&field, &params);
    let lifted = out.get(SHELF_LAST_COL + 1, RES / 2) - BASIN_DEPTH;
    assert!(
        lifted > 10.0,
        "expected raw Smooth to bleed the shelf into the basin (>10 m lift), got {lifted:.1} m"
    );
}
