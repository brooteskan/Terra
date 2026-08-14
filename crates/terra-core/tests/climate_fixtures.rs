//! Phase H climate / biome fixtures: orographic rain shadow, lapse cooling,
//! deterministic classification.

use terra_core::climate::{apply_root_cohesion, bake_climate};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{BiomesParams, ClimateParams, OPEN_HEIGHT_MAX, OPEN_HEIGHT_MIN};
use terra_core::mask::MaskField;
use terra_core::surface::bake_biomes_climate;

/// North–south ridge: steep west face, gentle east face (wind from west = 90°).
fn ridge_rain_shadow(res: u32) -> Heightfield {
    let m = HeightfieldMetrics::new(res, res, res as f32 * 4.0, res as f32 * 4.0);
    let mut hf = Heightfield::filled(m, 8.0);
    let crest = res as f32 * 0.45;
    for j in 0..res {
        for i in 0..res {
            let x = i as f32;
            let z = (j as f32 - res as f32 * 0.5) * 0.01;
            let h = if x < crest {
                // Windward (west): steep climb into the wind.
                8.0 + (x / crest) * 120.0 + z.sin() * 1.5
            } else {
                // Leeward (east): gentler descent.
                128.0 - (x - crest) / (res as f32 - crest).max(1.0) * 90.0 + z.sin() * 1.0
            };
            hf.set(i, j, h.max(5.0));
        }
    }
    // Ocean strip on the far west for moisture source.
    for j in 0..res {
        for i in 0..(res / 10).max(2) {
            hf.set(i, j, 4.0);
        }
    }
    hf
}

#[test]
fn climate_bake_deterministic() {
    let hf = ridge_rain_shadow(32);
    let p = BiomesParams::default();
    let a = bake_climate(&hf, &p, None, None);
    let b = bake_climate(&hf, &p, None, None);
    assert_eq!(a.temperature.data(), b.temperature.data());
    assert_eq!(a.rainfall.data(), b.rainfall.data());
    assert_eq!(a.biomes.data(), b.biomes.data());
}

#[test]
fn high_elevation_colder() {
    let m = HeightfieldMetrics::new(24, 24, 96.0, 96.0);
    let mut hf = Heightfield::filled(m, 20.0);
    for j in 0..24 {
        for i in 0..24 {
            // Peak in the center.
            let dx = i as f32 - 12.0;
            let dz = j as f32 - 12.0;
            let r = (dx * dx + dz * dz).sqrt();
            hf.set(i, j, (180.0 - r * 8.0).max(15.0));
        }
    }
    let p = BiomesParams {
        climate: ClimateParams {
            lapse_rate: 0.002,
            temp_gradient: 0.0,
            latitude: 0.0,
            orographic_strength: 0.0,
            rain_shadow_strength: 0.0,
            ..ClimateParams::default()
        },
        ..BiomesParams::default()
    };
    let maps = bake_climate(&hf, &p, None, None);
    let low = maps.temperature.get(0, 0);
    let high = maps.temperature.get(12, 12);
    assert!(
        high < low - 0.05,
        "summit should be colder than lowlands: high={high} low={low}"
    );
}

#[test]
fn windward_wetter_than_leeward() {
    let hf = ridge_rain_shadow(40);
    let p = BiomesParams {
        climate: ClimateParams {
            wind_dir_deg: 90.0, // from west toward east
            orographic_strength: 1.5,
            rain_shadow_strength: 0.9,
            base_precip: 0.5,
            temp_gradient: 0.0,
            ..ClimateParams::default()
        },
        ..BiomesParams::default()
    };
    let maps = bake_climate(&hf, &p, None, None);
    // Sample mid-slope west (windward) vs mid-slope east (leeward).
    let west_i = 14u32;
    let east_i = 28u32;
    let j = 20u32;
    let mut west_sum = 0.0f32;
    let mut east_sum = 0.0f32;
    let mut n = 0u32;
    for dj in 0..8 {
        west_sum += maps.rainfall.get(west_i, j + dj);
        east_sum += maps.rainfall.get(east_i, j + dj);
        n += 1;
    }
    let west_avg = west_sum / n as f32;
    let east_avg = east_sum / n as f32;
    assert!(
        west_avg > east_avg + 0.02,
        "windward should be wetter: west={west_avg} east={east_avg}"
    );
}

#[test]
fn biome_classification_uses_climate_not_height_alone() {
    let m = HeightfieldMetrics::new(20, 20, 80.0, 80.0);
    // Flat mid-elevation plateau — height alone would be one band; climate
    // splits dry vs wet via precip.
    let hf = Heightfield::filled(m, 60.0);
    let dry = BiomesParams {
        climate: ClimateParams {
            base_precip: 0.05,
            base_humidity: 0.05,
            orographic_strength: 0.0,
            rain_shadow_strength: 0.0,
            water_influence: 1.0,
            sea_level: -100.0, // no ocean moisture
            ..ClimateParams::default()
        },
        ..BiomesParams::default()
    };
    let wet = BiomesParams {
        climate: ClimateParams {
            base_precip: 0.95,
            base_humidity: 0.9,
            orographic_strength: 0.0,
            rain_shadow_strength: 0.0,
            sea_level: -100.0,
            ..ClimateParams::default()
        },
        ..BiomesParams::default()
    };
    let dry_maps = bake_biomes_climate(&hf, &dry, None, None);
    let wet_maps = bake_biomes_climate(&hf, &wet, None, None);
    let dry_id = (dry_maps.biomes.get(10, 10) * 16.0).round() as u32;
    let wet_id = (wet_maps.biomes.get(10, 10) * 16.0).round() as u32;
    assert_ne!(
        dry_id, wet_id,
        "same height should yield different biomes for dry vs wet climate"
    );
    // Desert = 1 in default LUT.
    assert_eq!(dry_id, 1, "dry plateau should classify as desert");
}

#[test]
fn root_cohesion_boosts_hardness() {
    let m = HeightfieldMetrics::new(8, 8, 16.0, 16.0);
    let hardness = MaskField::filled(m, 0.4);
    let mut veg = MaskField::zeros(m);
    veg.set(2, 2, 1.0);
    let out = apply_root_cohesion(&hardness, &veg, 0.2);
    assert!((out.get(2, 2) - 0.52).abs() < 1e-4);
    assert!((out.get(0, 0) - 0.4).abs() < 1e-5);
}

#[test]
fn legacy_height_bands_still_work() {
    let m = HeightfieldMetrics::new(16, 16, 64.0, 64.0);
    let mut hf = Heightfield::filled(m, 10.0);
    hf.set(8, 8, 250.0);
    let p = BiomesParams::height_bands();
    assert!(!p.use_climate);
    let maps = bake_biomes_climate(&hf, &p, None, None);
    let alpine = (maps.biomes.get(8, 8) * 16.0).round() as u32;
    let coast = (maps.biomes.get(0, 0) * 16.0).round() as u32;
    assert_eq!(alpine, 1);
    assert_eq!(coast, 3);
}

#[test]
fn finite_open_biome_bounds_match_infinite_reference_at_supported_extremes() {
    let finite = BiomesParams::height_bands();
    let mut infinite = finite.clone();
    for band in &mut infinite.bands {
        if band.min_height == OPEN_HEIGHT_MIN {
            band.min_height = f32::NEG_INFINITY;
        }
        if band.max_height == OPEN_HEIGHT_MAX {
            band.max_height = f32::INFINITY;
        }
    }

    for height in [-200_000.0, 0.0, 200_000.0] {
        let metrics = HeightfieldMetrics::new(4, 4, 4.0, 4.0);
        let hf = Heightfield::filled(metrics, height);
        let finite_maps = bake_biomes_climate(&hf, &finite, None, None);
        let infinite_maps = bake_biomes_climate(&hf, &infinite, None, None);
        assert_eq!(
            finite_maps.biomes.data(),
            infinite_maps.biomes.data(),
            "height={height}"
        );
    }
}
