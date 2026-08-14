//! Simplified climate fields and biome classification (Phase H).
//!
//! Physically motivated artist controls — not a GCM. CPU is the export oracle;
//! GPU preview may skip height-affecting work, while climate AuxMaps are baked
//! on CPU and uploaded as R32Float for overlays.

use crate::analyze::jump_flood_distance;
use crate::heightfield::Heightfield;
use crate::layer::{BiomeBand, BiomesParams, OPEN_HEIGHT_MAX, OPEN_HEIGHT_MIN};
use crate::mask::MaskField;

/// Packed climate / biome output maps (all values typically in \[0,1\]).
#[derive(Debug, Clone)]
pub struct ClimateMaps {
    pub temperature: MaskField,
    pub rainfall: MaskField,
    pub humidity: MaskField,
    pub aridity: MaskField,
    pub snow: MaskField,
    pub soil_moisture: MaskField,
    pub wind_exposure: MaskField,
    pub biomes: MaskField,
}

/// Compute climate fields and classify biomes from environmental drivers.
pub fn bake_climate(
    hf: &Heightfield,
    p: &BiomesParams,
    wetness: Option<&MaskField>,
    sediment: Option<&MaskField>,
) -> ClimateMaps {
    let c = &p.climate;
    let m = hf.metrics;
    let dx = m.dx().max(1e-6);
    let dz = m.dz().max(1e-6);

    let wind_rad = c.wind_dir_deg.to_radians();
    let wx = wind_rad.sin();
    let wz = wind_rad.cos();

    // Moisture source: cells at/below sea level as "ocean".
    let mut water_seeds = MaskField::zeros(m);
    for j in 0..m.height {
        for i in 0..m.width {
            if hf.get(i, j) <= c.sea_level {
                water_seeds.set(i, j, 1.0);
            }
        }
    }
    let water_dist = jump_flood_distance(&water_seeds);
    // jump_flood_distance is normalized \[0,1\]; convert to approximate meters.
    let world_diag = ((m.width as f32 * dx).hypot(m.height as f32 * dz)).max(1.0);
    let influence = c.water_influence.max(1.0);

    let mut temperature = MaskField::zeros(m);
    let mut rainfall = MaskField::zeros(m);
    let mut humidity = MaskField::zeros(m);
    let mut aridity = MaskField::zeros(m);
    let mut snow = MaskField::zeros(m);
    let mut soil_moisture = MaskField::zeros(m);
    let mut wind_exposure = MaskField::zeros(m);

    for j in 0..m.height {
        for i in 0..m.width {
            let h = hf.get(i, j);
            let (h_l, h_r, h_d, h_u) = neighbors(hf, i, j);
            let gx = (h_r - h_l) / (2.0 * dx);
            let gz = (h_u - h_d) / (2.0 * dz);
            let slope = (gx * gx + gz * gz).sqrt();
            let slope_n = (slope.atan().to_degrees() / 90.0).clamp(0.0, 1.0);

            // Aspect: 0 = east-facing gradient convention via atan2.
            let aspect = gz.atan2(-gx).to_degrees().rem_euclid(360.0);

            // --- Temperature ---
            // Latitude / user gradient along Z (north = +Z): cooler toward poles.
            let v = if m.height > 1 {
                j as f32 / (m.height - 1) as f32
            } else {
                0.5
            };
            let lat_term = (v - 0.5) * 2.0; // [-1, 1]
            let base_t = (c.sea_level_temp
                - c.latitude.abs() * 0.35
                - lat_term * c.temp_gradient
                - c.lapse_rate * h.max(0.0))
            .clamp(0.0, 1.0);

            // Solar / aspect: south-facing (180°) warmer in northern hemisphere.
            let sun_az = 180.0;
            let aspect_diff = angle_diff(aspect, sun_az).abs();
            let aspect_warm = (1.0 - aspect_diff / 180.0).clamp(0.0, 1.0);
            let solar = (1.0 - slope_n * 0.35) * (0.65 + 0.35 * aspect_warm);
            let temp = (base_t * 0.85 + solar * 0.15 * base_t.max(0.15)).clamp(0.0, 1.0);

            // --- Moisture & orographic precip ---
            let d_meters = water_dist.get(i, j) * world_diag;
            let coastal = (-d_meters / influence).exp().clamp(0.0, 1.0);
            let wet = wetness.map(|w| w.get(i, j)).unwrap_or(0.0);
            let sed = sediment.map(|s| s.get(i, j)).unwrap_or(0.0);

            // Windward: uphill into the wind (positive directional derivative).
            let oro = gx * wx + gz * wz;
            let windward = oro.max(0.0);
            let leeward = (-oro).max(0.0);
            // Normalize roughly by typical slopes (~0.2–1.0).
            let windward_n = (windward * 2.0).clamp(0.0, 1.0);
            let leeward_n = (leeward * 2.0).clamp(0.0, 1.0);

            let moisture = (c.base_humidity * 0.35
                + coastal * 0.45
                + wet * c.moisture_from_wetness
                + sed * 0.1)
                .clamp(0.0, 1.0);

            let oro_factor = 1.0 + c.orographic_strength * windward_n;
            let shadow_factor = (1.0 - c.rain_shadow_strength * leeward_n).max(0.05);
            let precip = (c.base_precip * moisture * oro_factor * shadow_factor).clamp(0.0, 1.0);

            let humid = (moisture * 0.55 + precip * 0.45).clamp(0.0, 1.0);
            let arid = (1.0 - humid).clamp(0.0, 1.0);

            // Snow: cold + high enough (or below snow-temp threshold).
            let snow_line_h = c.snow_line_height;
            let cold = (c.snow_temp - temp).max(0.0) / c.snow_temp.max(1e-3);
            let elev_snow = if h >= snow_line_h {
                ((h - snow_line_h) / (snow_line_h.max(1.0) * 0.5 + 1.0)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let sn = (cold.max(elev_snow) * (0.4 + 0.6 * precip)).clamp(0.0, 1.0);

            // Soil moisture: precip + wetness + soft sediment, drained by slope.
            let drain = slope_n * 0.55;
            let soil = (precip * 0.45 + wet * 0.4 + sed * 0.15 - drain).clamp(0.0, 1.0);

            // Wind exposure: ridges facing wind / steep windward.
            let exposure = (windward_n * (0.4 + 0.6 * slope_n)).clamp(0.0, 1.0);

            temperature.set(i, j, temp);
            rainfall.set(i, j, precip);
            humidity.set(i, j, humid);
            aridity.set(i, j, arid);
            snow.set(i, j, sn);
            soil_moisture.set(i, j, soil);
            wind_exposure.set(i, j, exposure);
        }
    }

    let biomes = classify_biomes(
        hf,
        p,
        &temperature,
        &rainfall,
        &snow,
        &soil_moisture,
        wetness,
    );

    ClimateMaps {
        temperature,
        rainfall,
        humidity,
        aridity,
        snow,
        soil_moisture,
        wind_exposure,
        biomes,
    }
}

/// Classify biome IDs from climate fields (and optional legacy height/wetness bands).
pub fn classify_biomes(
    hf: &Heightfield,
    p: &BiomesParams,
    temperature: &MaskField,
    rainfall: &MaskField,
    snow: &MaskField,
    soil_moisture: &MaskField,
    wetness: Option<&MaskField>,
) -> MaskField {
    let m = hf.metrics;
    let mut out = MaskField::zeros(m);
    for j in 0..m.height {
        for i in 0..m.width {
            let h = hf.get(i, j);
            let w = wetness.map(|m| m.get(i, j)).unwrap_or(0.5);
            let t = temperature.get(i, j);
            let r = rainfall.get(i, j);
            let sn = snow.get(i, j);
            let soil = soil_moisture.get(i, j);
            let mut id = 0.0;
            for band in &p.bands {
                if band_matches(band, h, w, t, r, sn, soil, p.use_climate) {
                    id = band.id as f32 / 16.0;
                }
            }
            out.set(i, j, id);
        }
    }
    out
}

fn band_matches(
    band: &BiomeBand,
    h: f32,
    w: f32,
    t: f32,
    r: f32,
    sn: f32,
    soil: f32,
    use_climate: bool,
) -> bool {
    let height_ok = h >= band.min_height && h <= band.max_height;
    let wet_ok = w >= band.min_wetness && w <= band.max_wetness;
    if !use_climate {
        return height_ok && wet_ok;
    }
    let temp_ok = t >= band.min_temp && t <= band.max_temp;
    let precip_ok = r >= band.min_precip && r <= band.max_precip;
    let snow_ok = sn >= band.min_snow && sn <= band.max_snow;
    let soil_ok = soil >= band.min_soil_moisture && soil <= band.max_soil_moisture;
    // Climate path: temp/precip/snow/soil drive classification; height/wetness
    // remain as optional filters when bands set finite ranges.
    let height_filter = if band.min_height.is_finite() || band.max_height.is_finite() {
        height_ok
    } else {
        true
    };
    let wet_filter = band.min_wetness > 0.0 || band.max_wetness < 1.0;
    let wet_ok2 = if wet_filter { wet_ok } else { true };
    temp_ok && precip_ok && snow_ok && soil_ok && height_filter && wet_ok2
}

/// Optional Cordonnier-style one-way feedback: vegetation density slightly
/// increases hardness (root cohesion). `boost` ∈ \[0,1\] is the max ΔK.
pub fn apply_root_cohesion(hardness: &MaskField, vegetation: &MaskField, boost: f32) -> MaskField {
    let b = boost.clamp(0.0, 1.0);
    if b <= 1e-6 {
        return hardness.clone();
    }
    let mut out = hardness.clone();
    let w = hardness.metrics.width;
    for (idx, &veg) in vegetation.data().iter().enumerate() {
        let x = (idx as u32) % w;
        let y = (idx as u32) / w;
        let k0 = out.get(x, y);
        let v = veg.clamp(0.0, 1.0);
        out.set(x, y, (k0 + (1.0 - k0) * b * v).clamp(0.0, 1.0));
    }
    out
}

fn neighbors(hf: &Heightfield, i: u32, j: u32) -> (f32, f32, f32, f32) {
    let w = hf.metrics.width;
    let h = hf.metrics.height;
    let h_l = hf.get(i.saturating_sub(1), j);
    let h_r = hf.get((i + 1).min(w - 1), j);
    let h_d = hf.get(i, j.saturating_sub(1));
    let h_u = hf.get(i, (j + 1).min(h - 1));
    (h_l, h_r, h_d, h_u)
}

fn angle_diff(a: f32, b: f32) -> f32 {
    let mut d = (a - b).rem_euclid(360.0);
    if d > 180.0 {
        d -= 360.0;
    }
    d
}

/// Default climate-aware biome LUT used by [`BiomesParams::default`].
pub fn default_climate_bands() -> Vec<BiomeBand> {
    vec![
        BiomeBand {
            name: "Desert".into(),
            id: 1,
            min_height: OPEN_HEIGHT_MIN,
            max_height: OPEN_HEIGHT_MAX,
            min_wetness: 0.0,
            max_wetness: 1.0,
            min_temp: 0.45,
            max_temp: 1.0,
            min_precip: 0.0,
            max_precip: 0.28,
            min_snow: 0.0,
            max_snow: 0.15,
            min_soil_moisture: 0.0,
            max_soil_moisture: 0.35,
        },
        BiomeBand {
            name: "Grassland".into(),
            id: 2,
            min_height: OPEN_HEIGHT_MIN,
            max_height: OPEN_HEIGHT_MAX,
            min_wetness: 0.0,
            max_wetness: 1.0,
            min_temp: 0.35,
            max_temp: 0.85,
            min_precip: 0.22,
            max_precip: 0.55,
            min_snow: 0.0,
            max_snow: 0.25,
            min_soil_moisture: 0.0,
            max_soil_moisture: 1.0,
        },
        BiomeBand {
            name: "Temperate Forest".into(),
            id: 3,
            min_height: OPEN_HEIGHT_MIN,
            max_height: OPEN_HEIGHT_MAX,
            min_wetness: 0.0,
            max_wetness: 1.0,
            min_temp: 0.28,
            max_temp: 0.75,
            min_precip: 0.45,
            max_precip: 1.0,
            min_snow: 0.0,
            max_snow: 0.35,
            min_soil_moisture: 0.15,
            max_soil_moisture: 1.0,
        },
        BiomeBand {
            name: "Wetland".into(),
            id: 4,
            min_height: OPEN_HEIGHT_MIN,
            max_height: OPEN_HEIGHT_MAX,
            min_wetness: 0.0,
            max_wetness: 1.0,
            min_temp: 0.2,
            max_temp: 0.9,
            min_precip: 0.35,
            max_precip: 1.0,
            min_snow: 0.0,
            max_snow: 0.2,
            min_soil_moisture: 0.65,
            max_soil_moisture: 1.0,
        },
        BiomeBand {
            name: "Boreal".into(),
            id: 5,
            min_height: OPEN_HEIGHT_MIN,
            max_height: OPEN_HEIGHT_MAX,
            min_wetness: 0.0,
            max_wetness: 1.0,
            min_temp: 0.12,
            max_temp: 0.4,
            min_precip: 0.25,
            max_precip: 1.0,
            min_snow: 0.0,
            max_snow: 0.7,
            min_soil_moisture: 0.0,
            max_soil_moisture: 1.0,
        },
        BiomeBand {
            name: "Alpine".into(),
            id: 6,
            min_height: OPEN_HEIGHT_MIN,
            max_height: OPEN_HEIGHT_MAX,
            min_wetness: 0.0,
            max_wetness: 1.0,
            min_temp: 0.0,
            max_temp: 0.28,
            min_precip: 0.0,
            max_precip: 1.0,
            min_snow: 0.35,
            max_snow: 1.0,
            min_soil_moisture: 0.0,
            max_soil_moisture: 1.0,
        },
        BiomeBand {
            name: "Coast".into(),
            id: 7,
            min_height: OPEN_HEIGHT_MIN,
            max_height: 25.0,
            min_wetness: 0.0,
            max_wetness: 1.0,
            min_temp: 0.25,
            max_temp: 1.0,
            min_precip: 0.2,
            max_precip: 1.0,
            min_snow: 0.0,
            max_snow: 0.2,
            min_soil_moisture: 0.0,
            max_soil_moisture: 1.0,
        },
    ]
}
