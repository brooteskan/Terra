//! Terrain generator helpers used by processors.

mod arid_landforms;
mod filter_kernels;
pub mod geology;
mod morphology;

use crate::analyze;
use crate::eval::EvalError;
use crate::heightfield::{HeightTile, Heightfield, HeightfieldMetrics, TileId};
use crate::layer::*;
use crate::mask::{dist_point_segment, point_in_polygon, MaskField};
use crate::noise::{self, domain_warp_fbm, fbm, ridged_mf, sample_worley};
use rayon::prelude::*;

pub fn flat(metrics: HeightfieldMetrics, height: f32) -> Heightfield {
    Heightfield::filled(metrics, height)
}
#[inline]
fn smooth01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Sample a sculpt paint buffer into the evaluation grid.
pub fn sculpt_base(metrics: HeightfieldMetrics, p: &SculptParams) -> Heightfield {
    let mut height = Heightfield::zeros(metrics);
    for tz in 0..metrics.tiles_z() {
        for tx in 0..metrics.tiles_x() {
            let id = TileId { tx, tz };
            if let Some(dst) = height.tile_mut(id) {
                *dst = sculpt_base_tile(metrics, p, id);
            }
        }
    }
    height.refresh_halos();
    height
}

/// Sample a single sculpt tile. Brush invalidation can therefore replace only touched pages.
pub fn sculpt_base_tile(metrics: HeightfieldMetrics, p: &SculptParams, id: TileId) -> HeightTile {
    let mut tile = HeightTile::new(id, &metrics);
    let (origin_x, origin_z) = tile.interior_origin(&metrics);
    for lz in 0..tile.interior_height {
        for lx in 0..tile.interior_width {
            let i = origin_x + lx;
            let j = origin_z + lz;
            let u = (i as f32 + 0.5) / metrics.width as f32;
            let v = (j as f32 + 0.5) / metrics.height as f32;
            tile.set_interior(lx, lz, p.sample_bilinear(u, v));
        }
    }
    tile
}

pub fn ramp(metrics: HeightfieldMetrics, p: &RampParams) -> Heightfield {
    let mut hf = Heightfield::zeros(metrics);
    let dir_x = p.direction.cos();
    let dir_z = p.direction.sin();
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            let x = metrics.world_x(i) / metrics.world_size_x;
            let z = metrics.world_z(j) / metrics.world_size_z;
            let t = (x * dir_x + z * dir_z).clamp(0.0, 1.0);
            hf.set(i, j, p.height_min + (p.height_max - p.height_min) * t);
        }
    }
    hf
}

pub fn noise_field(
    metrics: HeightfieldMetrics,
    p: &NoiseParams,
    kind: FractalNoiseType,
) -> Heightfield {
    fill_world(metrics, |x, z| {
        if p.octaves <= 1 {
            noise::sample_noise(
                kind,
                (x + p.offset_x) * p.frequency,
                (z + p.offset_z) * p.frequency,
                p.seed,
            ) * p.amplitude
        } else {
            fbm(kind, x, z, p)
        }
    })
}

pub fn worley_field(metrics: HeightfieldMetrics, p: &WorleyParams) -> Heightfield {
    fill_world(metrics, |x, z| sample_worley(x, z, p))
}

pub fn fbm_field(metrics: HeightfieldMetrics, p: &FbmParams) -> Heightfield {
    fill_world(metrics, |x, z| fbm(p.noise, x, z, &p.base))
}

pub fn ridged_field(metrics: HeightfieldMetrics, p: &FbmParams) -> Heightfield {
    fill_world(metrics, |x, z| ridged_mf(p.noise, x, z, &p.base))
}

pub fn domain_warp_field(metrics: HeightfieldMetrics, p: &DomainWarpParams) -> Heightfield {
    fill_world(metrics, |x, z| {
        domain_warp_fbm(x, z, &p.base, p.warp_strength, p.warp_frequency)
    })
}

pub fn terrace(input: &Heightfield, p: &TerraceParams) -> Heightfield {
    let c = geology::TerraceControls {
        height_m: 0.0,
        levels: p.levels.max(2),
        offset: 0.0,
        top_smoothness: (1.0 - p.sharpness.clamp(0.0, 1.0)) * 0.35,
        riser_sharpness: p.sharpness.clamp(0.05, 1.0),
        frequency: 0.04,
        seed: 1,
        slope_min: 0.0,
        slope_max: 90.0,
    };
    geology::terrace_simple(input, &c)
}

pub fn plateau(input: &Heightfield, p: &PlateauParams) -> Heightfield {
    let mut out = input.clone();
    let soft = p.soft.max(1e-3);
    out.map_mut(|h| {
        if h < p.low {
            let t = ((h - (p.low - soft)) / soft).clamp(0.0, 1.0);
            (p.low - soft) + t * soft
        } else if h > p.high {
            let t = ((h - p.high) / soft).clamp(0.0, 1.0);
            p.high + t * soft * 0.25
        } else {
            // Flatten toward mid plateau.
            let mid = (p.low + p.high) * 0.5;
            h * 0.25 + mid * 0.75
        }
    });
    out
}

/// Hard-cap mesa / butte with steep walls and soft talus skirt.
pub fn mesa(metrics: HeightfieldMetrics, p: &MesaParams) -> Heightfield {
    let short_axis = metrics.world_size_x.min(metrics.world_size_z).max(1.0);
    let radius_m = p.radius.clamp(0.02, 1.0) * short_axis * 0.5;
    let cx = p.center_u.clamp(0.0, 1.0) * metrics.world_size_x;
    let cz = p.center_v.clamp(0.0, 1.0) * metrics.world_size_z;
    let height = p.height.max(0.0);
    let steep = p.edge_steepness.max(1.0);
    let skirt = (p.soft.clamp(0.0, 1.0) * radius_m).max(1e-3);
    let outer = radius_m + skirt;
    let cap_noise = p.cap_noise.max(0.0);
    // Flat cap occupies most of the interior; remaining annulus is the cliff.
    let cap_r = radius_m * 0.82;
    fill_world(metrics, |x, z| {
        let dx = x - cx;
        let dz = z - cz;
        let dist = (dx * dx + dz * dz).sqrt();
        let theta = dz.atan2(dx);
        let footprint_freq = 2.8 / radius_m.max(1.0);
        let edge_noise =
            noise::perlin2(x * footprint_freq, z * footprint_freq, p.seed ^ 0xE063_51D3);
        let lobes = (theta * 4.0 + edge_noise).sin() * 0.045
            + (theta * 7.0 - edge_noise * 0.7).sin() * 0.025;
        let edge_scale = (1.0 + edge_noise * 0.075 + lobes).max(0.72);
        let warped_dist = dist / edge_scale;
        if warped_dist >= outer {
            return 0.0;
        }

        let cap_detail = if cap_noise > 0.0 {
            let broad = noise::perlin2(x * 0.0031, z * 0.0031, p.seed);
            let fine = noise::perlin2(x * 0.0097 + 13.0, z * 0.0097 - 7.0, p.seed ^ 0xA711_5CA1);
            (broad * 0.72 + fine * 0.28) * cap_noise
        } else {
            0.0
        };
        if warped_dist <= cap_r {
            return (height + cap_detail * 0.85).max(0.0);
        }
        if warped_dist <= radius_m {
            let t = ((warped_dist - cap_r) / (radius_m - cap_r).max(1e-3)).clamp(0.0, 1.0);
            let wall = (1.0 - t).powf(steep);
            let strata =
                0.95 + 0.05 * (wall * 11.0 * std::f32::consts::TAU + edge_noise * 1.8).sin();
            return (height * wall * strata + cap_detail * wall * 0.25).max(0.0);
        }

        // Noisy talus apron follows the warped cliff footprint.
        let t = ((warped_dist - radius_m) / skirt).clamp(0.0, 1.0);
        let talus_noise = 0.82
            + 0.18
                * noise::perlin2(
                    x * footprint_freq * 5.0,
                    z * footprint_freq * 5.0,
                    p.seed ^ 0x7A1A_5001,
                );
        let talus = (1.0 - smooth01(t)).powf(1.22);
        (height * 0.13 * talus * talus_noise).max(0.0)
    })
}

/// Height and semantic coastal fields produced by [`island`].
pub struct IslandResult {
    pub height: Heightfield,
    pub land: MaskField,
    /// Signed distance to the nearest shoreline in metres (negative on land).
    pub shore_distance: MaskField,
    /// Positive water depth below sea level in metres.
    pub bathymetry: MaskField,
    pub shelf: MaskField,
    pub beach: MaskField,
    pub reef: MaskField,
    pub mountain: MaskField,
}

/// Coast-aware procedural island based on a noise-deformed anisotropic SDF.
///
/// The SDF provides a guaranteed closed coastline and submerged outer border.
/// Distance-profile zones then build bathymetry, shelf/reef, beach and uplands.
/// Terrain noise is gated by the interior distance so it cannot leak land onto
/// the rectangular heightfield boundary.
pub fn island(metrics: HeightfieldMetrics, p: &IslandParams) -> IslandResult {
    let w = metrics.width as usize;
    let h = metrics.height as usize;
    let count = w * h;
    let mut heights = vec![0.0f32; count];
    let mut signed_raw = vec![0.0f32; count];
    let mut land = vec![0.0f32; count];
    let mut shore_distance = vec![0.0f32; count];
    let mut bathymetry = vec![0.0f32; count];
    let mut shelf = vec![0.0f32; count];
    let mut beach = vec![0.0f32; count];
    let mut reef = vec![0.0f32; count];
    let mut mountain = vec![0.0f32; count];

    let short_axis = metrics.world_size_x.min(metrics.world_size_z).max(1.0);
    let base_radius = p.radius.clamp(0.08, 0.94) * short_axis * 0.5;
    let aspect = p.aspect.clamp(0.35, 2.85);
    let aspect_root = aspect.sqrt();
    let radius_x = base_radius * aspect_root;
    let radius_z = base_radius / aspect_root;
    let cx = p.center_u.clamp(0.05, 0.95) * metrics.world_size_x;
    let cz = p.center_v.clamp(0.05, 0.95) * metrics.world_size_z;
    let angle = p.rotation_deg.to_radians();
    let ca = angle.cos();
    let sa = angle.sin();
    let coast_freq = p.coastline_frequency.max(1e-6);
    let coast_warp = p.coastline_warp.clamp(0.0, 0.42);
    let shelf_width = p.shelf_width.max(metrics.dx().max(metrics.dz()) * 2.0);
    let beach_width = p.beach_width.max(metrics.dx().max(metrics.dz()));
    let reef_width = p.reef_width.max(metrics.dx().max(metrics.dz()));
    let reef_center_m = (shelf_width * 0.42).min((shelf_width - reef_width * 0.35).max(0.0));
    let reef_sigma = (reef_width * 0.42).max(metrics.dx().max(metrics.dz()));
    let ocean_floor = p.ocean_floor.min(p.sea_level - 1.0);

    let island_ridge_params = NoiseParams {
        seed: p.seed ^ 0x51AD_E771,
        frequency: p.ridge_frequency.max(1e-6),
        amplitude: 1.0,
        octaves: 5,
        lacunarity: 2.05,
        persistence: 0.5,
        offset_x: 0.0,
        offset_z: 0.0,
        remap_min: -1.0,
        remap_max: 1.0,
    };
    let island_detail_params = NoiseParams {
        seed: p.seed ^ 0xD37A_11ED,
        frequency: p.ridge_frequency.max(1e-6) * 3.4,
        amplitude: 1.0,
        octaves: 3,
        lacunarity: 2.13,
        persistence: 0.48,
        offset_x: 7.0,
        offset_z: -13.0,
        remap_min: -1.0,
        remap_max: 1.0,
    };
    let seed_phase = noise::canonical_seed32(p.seed) as f32;

    heights
        .par_iter_mut()
        .zip(signed_raw.par_iter_mut())
        .enumerate()
        .for_each(|(idx, (height_out, signed_out))| {
            let i = (idx % w) as u32;
            let j = (idx / w) as u32;
            let x = metrics.world_x(i);
            let z = metrics.world_z(j);
            let dx = x - cx;
            let dz = z - cz;
            let xr = dx * ca + dz * sa;
            let zr = -dx * sa + dz * ca;

            let coastline_noise = noise::perlin2(x * coast_freq, z * coast_freq, p.seed) * 0.68
                + noise::perlin2(
                    x * coast_freq * 2.17 + 31.0,
                    z * coast_freq * 2.17 - 19.0,
                    p.seed ^ 0xA17C_9E5D,
                ) * 0.22;
            let theta = (zr / radius_z).atan2(xr / radius_x);
            let lobes = (theta * 3.0 + seed_phase * 0.013).sin() * 0.22
                + (theta * 5.0 - seed_phase * 0.007).sin() * 0.12;
            let edge_scale = (1.0 + coast_warp * (coastline_noise + lobes)).max(0.58);
            let mut rho = ((xr / radius_x).powi(2) + (zr / radius_z).powi(2)).sqrt() / edge_scale;

            if matches!(p.archetype, IslandArchetype::Archipelago) {
                // Union with two smaller, offset lobes. The shared low-frequency
                // coastline noise keeps the cluster geologically related.
                let lobe_r = 0.58;
                for (ox, oz, scale) in [(-0.42, 0.16, 0.62), (0.38, -0.2, 0.54)] {
                    let lx = xr / radius_x - ox;
                    let lz = zr / radius_z - oz;
                    let lr =
                        ((lx / (scale * lobe_r)).powi(2) + (lz / (scale / lobe_r)).powi(2)).sqrt();
                    rho = rho.min(lr / edge_scale.max(0.7));
                }
            }

            let mut signed = (rho - 1.0) * base_radius;
            if matches!(p.archetype, IslandArchetype::Atoll) {
                // The maximum of the outer and inner SDFs creates a closed ring.
                let inner = (p.lagoon_radius.clamp(0.2, 0.78) - rho) * base_radius;
                signed = signed.max(inner);
            }
            *signed_out = signed;

            let value = if signed >= 0.0 {
                let outward = signed;
                let shelf_t = (outward / shelf_width).clamp(0.0, 1.0);
                let reef_band = (-((outward - reef_center_m) / reef_sigma).powi(2)).exp();
                let base_depth = p.shelf_depth.max(1.0) * shelf_t.powf(1.35);
                let reef_depth = p.reef_depth.max(0.5);
                let shelf_height = p.sea_level - base_depth;
                let reef_height = p.sea_level - reef_depth - shelf_t * reef_depth * 0.35;
                if outward <= shelf_width {
                    shelf_height.max(reef_height * reef_band + shelf_height * (1.0 - reef_band))
                } else {
                    let abyss_t = ((outward - shelf_width) / (shelf_width * 2.8)).clamp(0.0, 1.0);
                    let smooth = abyss_t * abyss_t * (3.0 - 2.0 * abyss_t);
                    (p.sea_level - p.shelf_depth.max(1.0)) * (1.0 - smooth) + ocean_floor * smooth
                }
            } else {
                let inward = -signed;
                if matches!(p.archetype, IslandArchetype::Atoll) {
                    let ring_t = (inward / beach_width.max(1.0)).clamp(0.0, 1.0);
                    p.sea_level + p.beach_height.max(0.5) * ring_t.sqrt()
                } else if inward <= beach_width {
                    let t = (inward / beach_width).clamp(0.0, 1.0);
                    let smooth = t * t * (3.0 - 2.0 * t);
                    p.sea_level + p.beach_height.max(0.5) * smooth
                } else {
                    let interior = ((inward - beach_width)
                        / (base_radius * 0.82 - beach_width).max(1.0))
                    .clamp(0.0, 1.0);
                    let ridge_noise =
                        ridged_mf(FractalNoiseType::Perlin, x, z, &island_ridge_params);
                    let spine_warp = noise::perlin2(
                        x * coast_freq * 0.55 + 7.0,
                        z * coast_freq * 0.55 - 11.0,
                        p.seed ^ 0xC04F_FEE1,
                    ) * radius_z
                        * 0.18;
                    let spine = (-(zr + spine_warp).powi(2)
                        / (2.0 * (radius_z * 0.24).powi(2).max(1.0)))
                    .exp();
                    let ridge_strength = p.ridge_strength.clamp(0.0, 0.9);
                    let macro_mass = 0.5
                        + 0.5
                            * noise::perlin2(
                                x * coast_freq * 0.38 + 3.0,
                                z * coast_freq * 0.38 - 4.0,
                                p.seed ^ 0xBA51_71C5,
                            );
                    let ridge = ((1.0 - ridge_strength)
                        + ridge_strength * (ridge_noise * 0.38 + spine * 0.47 + macro_mass * 0.15))
                        .clamp(0.18, 1.18);

                    // Radial drainage seeds cut the volcanic mass before the
                    // downstream physical erosion pass develops the full network.
                    let drainage_warp = noise::perlin2(
                        x * coast_freq * 0.72,
                        z * coast_freq * 0.72,
                        p.seed ^ 0xD2A1_6E55,
                    ) * 3.2;
                    let gully = (1.0 - (theta * 9.0 + drainage_warp).sin().abs()).powf(6.0)
                        * smooth01(interior);
                    let mountain_height = p.mountain_height.max(0.0);
                    let uplift = interior.powf(p.mountain_power.max(0.35));
                    let detail = fbm(FractalNoiseType::Perlin, x, z, &island_detail_params)
                        * mountain_height
                        * 0.055
                        * smooth01(interior);
                    let base = p.sea_level + p.beach_height.max(0.5);
                    (base + mountain_height * uplift * ridge
                        - mountain_height * 0.105 * gully * uplift
                        + detail)
                        .max(base + mountain_height * interior * 0.012)
                }
            };
            *height_out = value;
        });

    for idx in 0..count {
        let value = heights[idx];
        land[idx] = if value > p.sea_level { 1.0 } else { 0.0 };
        bathymetry[idx] = (p.sea_level - value).max(0.0);
        let d = signed_raw[idx];
        shore_distance[idx] = d;
        let outside = d.max(0.0);
        let inside = (-d).max(0.0);
        shelf[idx] = (1.0 - outside / shelf_width).clamp(0.0, 1.0) * (1.0 - land[idx]);
        beach[idx] = (1.0 - inside / beach_width).clamp(0.0, 1.0) * land[idx];
        reef[idx] = (-((outside - reef_center_m) / reef_sigma).powi(2)).exp() * (1.0 - land[idx]);
        mountain[idx] = if p.mountain_height > 1e-3 {
            ((heights[idx] - p.sea_level - p.beach_height) / p.mountain_height).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }

    IslandResult {
        height: Heightfield::from_dense(metrics, &heights),
        land: MaskField::from_raw(metrics, &land),
        shore_distance: MaskField::from_raw(metrics, &shore_distance),
        bathymetry: MaskField::from_raw(metrics, &bathymetry),
        shelf: MaskField::from_raw(metrics, &shelf),
        beach: MaskField::from_raw(metrics, &beach),
        reef: MaskField::from_raw(metrics, &reef),
        mountain: MaskField::from_raw(metrics, &mountain),
    }
}

pub fn mountains(metrics: HeightfieldMetrics, p: &MountainParams) -> Heightfield {
    let world_scale = metrics.world_size_x.min(metrics.world_size_z).max(1.0);
    let amplitude = p.base.amplitude.max(0.0);

    let mut macro_params = p.base.clone();
    macro_params.frequency = p.base.frequency.max(1e-6) * 0.52;
    macro_params.octaves = p.base.octaves.clamp(2, 4);
    macro_params.amplitude = 1.0;

    let mut detail_params = p.base.clone();
    detail_params.seed ^= 0xC0DE;
    detail_params.frequency = p.base.frequency.max(1e-6) * 4.5;
    detail_params.octaves = 4;
    detail_params.amplitude = 1.0;
    detail_params.lacunarity = 2.15;
    detail_params.persistence = 0.52;

    fill_world(metrics, |x, z| {
        let nx = x / metrics.world_size_x - 0.5;
        let nz = z / metrics.world_size_z - 0.5;
        let ca = p.range_angle.cos();
        let sa = p.range_angle.sin();
        let axis = nx * ca + nz * sa;
        let cross = nx * -sa + nz * ca;

        // Low-frequency warping breaks the perfectly straight range corridor.
        let warp_freq = p.base.frequency.max(1e-6) * 0.31;
        let warp_cross =
            noise::perlin2(x * warp_freq, z * warp_freq, p.base.seed ^ 0x7A11_CE55) * 0.085;
        let warp_axis = noise::perlin2(
            x * warp_freq + 17.3,
            z * warp_freq - 9.1,
            p.base.seed ^ 0x51D3_90AF,
        ) * 0.035;
        let warped_axis = axis + warp_axis;
        let warped_cross = cross + warp_cross;
        let corridor_t = 1.0 - warped_cross.abs() / p.range_width.max(1e-3);
        let range_mask = smooth01(corridor_t).powf(1.18);
        let ridge_x = warped_axis * world_scale;
        let ridge_z = warped_cross * world_scale;

        // Macro uplift and ridge detail are separate frequency bands.
        let macro_ridge = ridged_mf(FractalNoiseType::Perlin, ridge_x, ridge_z, &macro_params);
        let ridge =
            ridged_mf(FractalNoiseType::Perlin, ridge_x, ridge_z, &p.base) / amplitude.max(1e-6);
        let macro_bias = 0.5
            + 0.5
                * noise::perlin2(
                    x * warp_freq * 0.63 + 3.7,
                    z * warp_freq * 0.63 - 5.2,
                    p.base.seed ^ 0xA831_4F29,
                );
        let structure = (ridge.clamp(0.0, 1.0) * 0.72 + macro_ridge.clamp(0.0, 1.0) * 0.28)
            .powf(p.ridge_sharpness.max(0.1));
        let foothills = macro_bias * range_mask.powf(1.8) * 0.13;
        let primary = amplitude * range_mask * (foothills + structure * 0.87);
        let elev_t = (primary / amplitude.max(1e-6)).clamp(0.0, 1.0);
        // Fine ridged gouges along the range axis — seeds SPE couloirs / striations.
        let fine = ridged_mf(FractalNoiseType::Perlin, ridge_x, ridge_z, &detail_params);
        // Signed detail raises ribs and cuts narrow couloirs instead of only
        // piling more positive noise onto every crest.
        let detail = (fine - 0.42) * 2.0 * p.crest_detail.max(0.0) * elev_t.powf(0.68) * range_mask;
        (primary + detail).max(0.0)
    })
}

/// Radial cone + optional crater bowl (World Creator Landscape Volcano intent).
pub fn volcano(metrics: HeightfieldMetrics, p: &VolcanoParams) -> Heightfield {
    let short_axis = metrics.world_size_x.min(metrics.world_size_z).max(1.0);
    let radius_m = p.radius.clamp(0.01, 1.0) * short_axis * 0.5;
    let cx = p.center_u.clamp(0.0, 1.0) * metrics.world_size_x;
    let cz = p.center_v.clamp(0.0, 1.0) * metrics.world_size_z;
    let height = p.height.max(0.0);
    let flank = p.flank_power.max(0.2);
    let crater_r = (p.crater_radius.clamp(0.0, 0.95) * radius_m).max(0.0);
    let crater_d = p.crater_depth.max(0.0);
    let rough = p.roughness.max(0.0);
    fill_world(metrics, |x, z| {
        let dx = x - cx;
        let dz = z - cz;
        let dist = (dx * dx + dz * dz).sqrt();
        let theta = dz.atan2(dx);

        // A volcano is never a mathematically perfect cone: warp the footprint
        // at massif scale, then add radial drainage only on the flanks.
        let macro_freq = 3.2 / radius_m.max(1.0);
        let macro_noise = noise::perlin2(x * macro_freq, z * macro_freq, p.seed ^ 0xA17C_31D5);
        let lobe = (theta * 5.0 + macro_noise * 1.7).sin();
        let footprint = (1.0 + macro_noise * 0.09 + lobe * 0.035).max(0.72);
        let radial = dist / (radius_m * footprint);
        if radial >= 1.0 {
            return 0.0;
        }

        let t = smooth01(1.0 - radial);
        let mut h = height * t.powf(flank);
        if rough > 0.0 {
            let fine = noise::perlin2(
                x * macro_freq * 4.7 + 11.0,
                z * macro_freq * 4.7 - 7.0,
                p.seed ^ 0xC011_A953,
            );
            let groove_phase = theta * 11.0 + macro_noise * 4.2 + fine * 0.9;
            let groove = (1.0 - groove_phase.sin().abs()).powf(5.0);
            let flank_gate = smooth01(radial / 0.22) * smooth01((1.0 - radial) / 0.18);
            h += fine * rough * t * 0.72;
            h -= groove * rough * 1.35 * flank_gate;
        }

        if crater_r > 1e-3 {
            let crater_norm = crater_r / radius_m;
            let rim_sigma = (crater_norm * 0.18).max(0.012);
            let rim = (-((radial - crater_norm) / rim_sigma).powi(2)).exp();
            h += rim * crater_d * 0.22;
            if radial < crater_norm {
                let bowl = smooth01(1.0 - radial / crater_norm);
                let rim_profile = height * smooth01(1.0 - crater_norm).powf(flank);
                let crater_floor = (rim_profile - crater_d).max(0.0);
                let bowl_blend = bowl.powf(1.35);
                h = h * (1.0 - bowl_blend) + crater_floor * bowl_blend;
            }
        }
        h.max(0.0)
    })
}

/// Coherent ridge-corridor uplift with optional altitude-faded detail.
///
/// Primary structure is a warped corridor mask × low-frequency ridge profile.
/// Detail fBm is scaled by local uplift so valleys stay open for drainage
/// (Musgrave altitude-crossover *concept*, not a full multifractal implementation).
pub fn uplift(metrics: HeightfieldMetrics, p: &UpliftParams) -> Heightfield {
    let amp = p.amplitude.max(0.0);
    let corr_w = p.corridor_width.max(1e-3);
    let fade = p.altitude_fade.clamp(0.0, 1.0);
    let seed = p.seed;
    fill_world(metrics, |x, z| {
        // Low-frequency warp so corridors meander without breaking coherence.
        let wx = noise::perlin2(x * p.frequency * 0.35, z * p.frequency * 0.35, seed)
            * p.warp_strength
            * metrics.world_size_x
            * 0.08;
        let wz = noise::perlin2(
            x * p.frequency * 0.35 + 19.0,
            z * p.frequency * 0.35 + 7.0,
            seed ^ 0xA5A5,
        ) * p.warp_strength
            * metrics.world_size_z
            * 0.08;
        let xw = x + wx;
        let zw = z + wz;

        let nx = xw / metrics.world_size_x - 0.5;
        let nz = zw / metrics.world_size_z - 0.5;
        let ca = p.range_angle.cos();
        let sa = p.range_angle.sin();
        let axis = nx * ca + nz * sa;
        let cross = nx * (-sa) + nz * ca;
        let corridor = (1.0 - (cross.abs() / corr_w).clamp(0.0, 1.0)).powf(1.35);

        // Primary ridge: few low-frequency ridged octaves along the corridor axis.
        let ridge_params = NoiseParams {
            seed: p.seed,
            octaves: 3,
            frequency: p.frequency,
            amplitude: 1.0,
            lacunarity: 2.1,
            persistence: 0.45,
            offset_x: 0.0,
            offset_z: 0.0,
            remap_min: -1.0,
            remap_max: 1.0,
        };
        let ridge_x = axis * metrics.world_size_x;
        let ridge_z = cross * metrics.world_size_z;
        let ridge =
            (ridged_mf(FractalNoiseType::Perlin, ridge_x, ridge_z, &ridge_params)).clamp(0.0, 1.0);
        let primary = corridor * ridge.powf(p.ridge_power.max(0.1)) * amp;

        // Detail: fBm attenuated in low-uplift cells so structure drains coherently.
        let detail_params = NoiseParams {
            seed: p.seed ^ 0xC0FFEE,
            octaves: p.detail_octaves.max(1).min(8),
            frequency: p.detail_frequency.max(1e-6),
            amplitude: p.detail_amplitude.max(0.0),
            lacunarity: 2.0,
            persistence: 0.5,
            offset_x: 11.0,
            offset_z: -3.0,
            remap_min: -1.0,
            remap_max: 1.0,
        };
        let detail = fbm(FractalNoiseType::Perlin, x, z, &detail_params);
        let elev_t = if amp > 1e-6 {
            (primary / amp).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let detail_scale = (1.0 - fade) + fade * elev_t;
        primary + detail * detail_scale
    })
}

/// Procedural dunes: directional phasor/wave seed + limited aeolian relaxation.
///
/// Fast interactive approximation. Prefer [`sand_simulation`] for progressive
/// physical evolution. Morphology emerges from wind / supply / linearity —
/// transverse, barchan-like, linear, and fields are not hardcoded meshes.
pub fn dunes(metrics: HeightfieldMetrics, p: &DuneParams) -> Heightfield {
    dunes_with_aux(metrics, p).0
}

/// Dunes generator plus shared sand/wind aux channels.
pub fn dunes_with_aux(
    metrics: HeightfieldMetrics,
    p: &DuneParams,
) -> (Heightfield, analyze::AeolianResult) {
    let scale = p.effective_scale();
    let height = p.effective_height();
    let crest = p.effective_crest_sharpness();
    let trough = p.trough_depth.max(0.0);
    let floor = (p.basin_floor.clamp(0.0, 1.0) * height).max(0.0);

    // Soft bedrock basin so interdunes read as sand-starved lows.
    let bedrock_hf = fill_world(metrics, |x, z| {
        let n = noise::perlin2(
            x * scale * 0.22,
            z * scale * 0.19,
            p.base.seed ^ 0xB0A7_D00D,
        );
        floor + (0.5 + 0.5 * n) * height * 0.08 - trough * (0.35 + 0.25 * n).max(0.0)
    });

    let sand = seed_dune_sand(
        metrics,
        p.direction_deg,
        scale,
        height,
        p.sand_supply,
        crest,
        p.linearity,
        p.base.seed,
    );

    let mut state = analyze::AeolianState::from_height_and_sand(&bedrock_hf, 0.0, Some(&sand));
    state.bedrock = bedrock_hf.to_dense();

    let transport = analyze::AeolianTransportParams {
        wind_direction_deg: p.direction_deg,
        wind_speed: p.wind_strength.max(0.05),
        transport_length: p.transport_length.max(0.5),
        repose_angle_deg: p.avalanche_angle.clamp(15.0, 45.0),
        slab_size: (height * 0.012).clamp(0.08, 1.2),
        avalanche_iters: 4 + (p.iterations / 3).min(12),
        abrasion: 0.01,
        reptation: 0.12,
        wind_warp: 0.25,
        linearity: p.linearity.clamp(0.0, 1.0),
    };
    state.evolve(&transport, p.iterations.max(1).min(48));
    let mut result = state.into_result(1.0, &bedrock_hf);
    // Soft floor: lift only the deepest cells with a little noise so we never
    // create a broad exactly-flat interdune slab.
    let floor_soft = floor * 0.5;
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            let h = result.height.get(i, j);
            if h < floor_soft {
                let n = noise::perlin2(
                    metrics.world_x(i) * scale * 3.1,
                    metrics.world_z(j) * scale * 2.7,
                    p.base.seed ^ 0xF100_B001,
                );
                let lift = floor_soft + n.abs() * height * 0.012;
                result.height.set(i, j, h.max(lift * 0.35 + h * 0.65));
            }
        }
    }
    (result.height.clone(), result)
}

/// Directional band-limited / phasor dune seed (sand thickness in meters).
pub fn seed_dune_sand(
    metrics: HeightfieldMetrics,
    direction_deg: f32,
    dune_scale: f32,
    dune_height: f32,
    sand_supply: f32,
    crest_sharpness: f32,
    linearity: f32,
    seed: u64,
) -> Vec<f32> {
    use filter_kernels::directional_phasor;

    let n = (metrics.width * metrics.height) as usize;
    let mut sand = vec![0.0f32; n];
    let ang = direction_deg.to_radians();
    let (s, c) = ang.sin_cos();
    let freq = dune_scale.max(1e-5);
    let supply = sand_supply.clamp(0.0, 2.0);
    let height = dune_height.max(0.0);
    let sharp = crest_sharpness.clamp(0.0, 1.0);
    let lin = linearity.clamp(0.0, 1.0);
    let base_cover = height * supply * 0.22;

    for j in 0..metrics.height {
        for i in 0..metrics.width {
            let x = metrics.world_x(i);
            let z = metrics.world_z(j);
            let along = x * c + z * s;
            let across = -x * s + z * c;

            let ph = directional_phasor(x, z, freq, ang, lin, seed);
            let lobe = (0.5 + 0.5 * ph).clamp(0.0, 1.0);
            let shaped = lobe.powf(0.55 + sharp * 1.1);
            let ripple =
                noise::perlin2(along * freq * 5.5, across * freq * 1.1, seed ^ 0x51F7_E001)
                    * 0.06
                    * shaped;
            let secondary = if lin < 0.85 {
                let ang2 = ang + std::f32::consts::FRAC_PI_2 * (1.0 - lin);
                let ph2 = directional_phasor(x, z, freq * 0.92, ang2, lin, seed ^ 0xA11C);
                (0.5 + 0.5 * ph2).clamp(0.0, 1.0).powf(0.7) * (1.0 - lin) * 0.55
            } else {
                0.0
            };
            let profile = (shaped * (0.65 + 0.35 * lin) + secondary + ripple.abs()).clamp(0.0, 1.0);
            let idx = (j * metrics.width + i) as usize;
            sand[idx] = base_cover + height * supply * profile * (0.55 + 0.45 * supply.min(1.0));
        }
    }
    sand
}

pub fn canyons(metrics: HeightfieldMetrics, p: &CanyonParams) -> Heightfield {
    let base_width = p.width.max(1e-3);
    let longitudinal_freq = 1.0 / (base_width * 11.0).max(1.0);
    let seed_phase = noise::canonical_seed32(p.seed) as f32;
    fill_world(metrics, |x, z| {
        let broad = noise::perlin2(z * longitudinal_freq, seed_phase * 0.017, p.seed);
        let detail = noise::perlin2(
            z * longitudinal_freq * 3.1,
            seed_phase * 0.043,
            p.seed ^ 0x9E37,
        );
        let meander = broad * 0.72 + detail * 0.28;
        let center =
            metrics.world_size_x * 0.5 + meander * p.meander.clamp(0.0, 2.0) * base_width * 2.35;

        let width_noise = noise::perlin2(
            z * longitudinal_freq * 0.83 + 13.0,
            seed_phase * 0.007,
            p.seed ^ 0x7F4A_71C2,
        );
        let local_width = base_width * (1.0 + width_noise * 0.28).max(0.45);
        let wall_noise = noise::perlin2(
            x / local_width * 0.63,
            z / local_width * 0.47,
            p.seed ^ 0xC411_0A55,
        );
        let q = ((x - center).abs() / local_width + wall_noise * 0.055).max(0.0);
        if q >= 1.0 {
            return 0.0;
        }

        // Broad weathered walls plus a narrow inner incision create a changing
        // canyon hierarchy instead of one constant triangular cross-section.
        let wall = smooth01(1.0 - q).powf(0.72);
        let inner = smooth01(1.0 - q / 0.19);
        let depth_noise = noise::perlin2(
            z * longitudinal_freq * 0.57 - 5.0,
            8.0,
            p.seed ^ 0xD337_1A11,
        );
        let local_depth = p.depth.max(0.0) * (0.88 + depth_noise * 0.16).max(0.5);
        let strata = 0.96 + 0.04 * (q * 9.0 * std::f32::consts::TAU + wall_noise).sin();
        -local_depth * (wall * 0.78 + inner * 0.22) * strata
    })
}

pub fn voronoi_regions(metrics: HeightfieldMetrics, p: &VoronoiParams) -> Heightfield {
    fill_world(metrics, |x, z| {
        let w = sample_worley(
            x,
            z,
            &WorleyParams {
                base: p.base.clone(),
                distance_metric: WorleyMetric::Euclidean,
                feature: WorleyFeature::F1,
            },
        );
        let cell = noise::value_noise2(
            (x + p.base.offset_x) * p.base.frequency,
            (z + p.base.offset_z) * p.base.frequency,
            p.base.seed ^ 0xBEEF,
        );
        w * 0.25 + cell * p.height_per_cell * p.cell_jitter
    })
}

pub fn import_heightmap(
    metrics: HeightfieldMetrics,
    p: &ImportHeightmapParams,
) -> Result<Heightfield, EvalError> {
    if p.path.is_empty() {
        return Ok(Heightfield::zeros(metrics));
    }
    let img = image::open(&p.path).map_err(|e| EvalError::Io(e.to_string()))?;
    let g = img.to_luma16();
    let (iw, ih) = g.dimensions();
    let mut hf = Heightfield::zeros(metrics);
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            let u = i as f32 / metrics.width as f32;
            let v = j as f32 / metrics.height as f32;
            let x = ((u * iw as f32) as u32).min(iw - 1);
            let y = ((v * ih as f32) as u32).min(ih - 1);
            let pix = g.get_pixel(x, y).0[0] as f32 / 65535.0;
            hf.set(i, j, pix * p.height_scale + p.height_offset);
        }
    }
    Ok(hf)
}

/// 3D stamp: OBJ mesh→height, image heightmap, or procedural rock when empty/unreadable.
pub fn stamp_3d(metrics: HeightfieldMetrics, p: &Stamp3dParams) -> Result<Heightfield, EvalError> {
    if p.path.is_empty() {
        return Ok(procedural_rock_stamp(metrics, p));
    }
    let lower = p.path.to_ascii_lowercase();
    if lower.ends_with(".obj") {
        return match stamp_3d_from_obj(metrics, p) {
            Ok(hf) => Ok(hf),
            Err(_) => Ok(procedural_rock_stamp(metrics, p)),
        };
    }
    let ih = ImportHeightmapParams {
        path: p.path.clone(),
        height_scale: p.height_scale,
        height_offset: p.height_offset,
    };
    match import_heightmap(metrics, &ih) {
        Ok(hf) => Ok(hf),
        Err(_) => Ok(procedural_rock_stamp(metrics, p)),
    }
}

fn procedural_rock_stamp(metrics: HeightfieldMetrics, p: &Stamp3dParams) -> Heightfield {
    let mut hf = Heightfield::zeros(metrics);
    let w = metrics.width;
    let h = metrics.height;
    for j in 0..h {
        for i in 0..w {
            let u = (i as f32 + 0.5) / w as f32 - 0.5;
            let v = (j as f32 + 0.5) / h as f32 - 0.5;
            let rx = 0.28;
            let rz = 0.24;
            let eu = u / rx;
            let ev = v / rz;
            let r2 = eu * eu + ev * ev;
            if r2 > 1.0 {
                continue;
            }
            let ellipsoid = (1.0 - r2).max(0.0).sqrt();
            let bumps = noise::value_noise2(u * 14.0 + 3.1, v * 14.0 - 1.7, 11) * 0.45
                + noise::value_noise2(u * 28.0, v * 28.0, 19) * 0.2;
            let shape = (ellipsoid + bumps * (1.0 - r2)).max(0.0);
            hf.set(i, j, shape * p.height_scale + p.height_offset);
        }
    }
    hf
}

fn stamp_3d_from_obj(
    metrics: HeightfieldMetrics,
    p: &Stamp3dParams,
) -> Result<Heightfield, EvalError> {
    let text = std::fs::read_to_string(&p.path).map_err(|e| EvalError::Io(e.to_string()))?;
    let mut verts: Vec<[f32; 3]> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix('v').and_then(|s| {
            if s.starts_with(' ') || s.starts_with('\t') {
                Some(s.trim_start())
            } else {
                None
            }
        }) else {
            continue;
        };
        // Skip vt / vn
        if rest.starts_with('t') || rest.starts_with('n') {
            continue;
        }
        let mut parts = rest.split_whitespace();
        let (Some(xs), Some(ys), Some(zs)) = (parts.next(), parts.next(), parts.next()) else {
            continue;
        };
        let Ok(x) = xs.parse::<f32>() else { continue };
        let Ok(y) = ys.parse::<f32>() else { continue };
        let Ok(z) = zs.parse::<f32>() else { continue };
        verts.push([x, y, z]);
    }
    if verts.is_empty() {
        return Err(EvalError::Io("OBJ has no vertices".into()));
    }
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    for v in &verts {
        min_x = min_x.min(v[0]);
        max_x = max_x.max(v[0]);
        min_z = min_z.min(v[2]);
        max_z = max_z.max(v[2]);
        min_y = min_y.min(v[1]);
    }
    let span_x = (max_x - min_x).max(1e-6);
    let span_z = (max_z - min_z).max(1e-6);
    let mut hf = Heightfield::zeros(metrics);
    let w = metrics.width as usize;
    let h = metrics.height as usize;
    let mut filled = vec![false; w * h];
    for v in &verts {
        let u = ((v[0] - min_x) / span_x).clamp(0.0, 1.0);
        let vv = ((v[2] - min_z) / span_z).clamp(0.0, 1.0);
        let i = ((u * (metrics.width as f32 - 1.0)).round() as u32).min(metrics.width - 1);
        let j = ((vv * (metrics.height as f32 - 1.0)).round() as u32).min(metrics.height - 1);
        let height = (v[1] - min_y) * p.height_scale + p.height_offset;
        let idx = j as usize * w + i as usize;
        if !filled[idx] || height > hf.get(i, j) {
            hf.set(i, j, height);
            filled[idx] = true;
        }
    }
    Ok(hf)
}

/// Evaluate a WC-style procedural shape layer (generator picker).
pub fn procedural_shape(
    metrics: HeightfieldMetrics,
    p: &crate::layer::ProceduralShapeParams,
) -> Heightfield {
    use crate::layer::{FractalNoiseType, ProceduralGenerator};
    match p.generator {
        ProceduralGenerator::Mountain => mountains(metrics, &p.mountain),
        ProceduralGenerator::Hills => fbm_field(metrics, &p.hills),
        ProceduralGenerator::Plateau => {
            let base = fbm_field(metrics, &p.hills);
            plateau(&base, &p.plateau)
        }
        ProceduralGenerator::Mesa => mesa(metrics, &p.mesa),
        ProceduralGenerator::Volcano => volcano(metrics, &p.volcano),
        ProceduralGenerator::Dunes => dunes(metrics, &p.dunes),
        ProceduralGenerator::Canyon => canyons(metrics, &p.canyon),
        ProceduralGenerator::Crater => {
            let base = Heightfield::filled(metrics, 80.0);
            effect_filter(&base, &p.crater)
        }
        ProceduralGenerator::Noise => noise_field(metrics, &p.noise, FractalNoiseType::Perlin),
    }
}

/// Closed polygon raise / carve (normalized UV vertices).
pub fn polygon_height(input: &Heightfield, p: &crate::layer::PolygonHeightParams) -> Heightfield {
    let mut out = input.clone();
    if p.points.len() < 3 {
        return out;
    }
    let short = input
        .metrics
        .world_size_x
        .min(input.metrics.world_size_z)
        .max(1.0);
    let falloff_m = (p.falloff.clamp(0.0, 0.5) * short).max(1e-3);
    let (min_u, min_v, max_u, max_v) = p.points.iter().fold(
        (
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ),
        |(min_u, min_v, max_u, max_v), point| {
            (
                min_u.min(point[0]),
                min_v.min(point[1]),
                max_u.max(point[0]),
                max_v.max(point[1]),
            )
        },
    );
    let (i_start, i_end, j_start, j_end) = world_raster_bounds(
        input.metrics,
        min_u * input.metrics.world_size_x,
        min_v * input.metrics.world_size_z,
        max_u * input.metrics.world_size_x,
        max_v * input.metrics.world_size_z,
        falloff_m,
    );

    if i_start < i_end && j_start < j_end {
        let tile_size = input.metrics.tile_size;
        let tx_start = i_start / tile_size;
        let tx_end = (i_end - 1) / tile_size;
        let tz_start = j_start / tile_size;
        let tz_end = (j_end - 1) / tile_size;
        for tz in tz_start..=tz_end {
            for tx in tx_start..=tx_end {
                let id = TileId { tx, tz };
                if let (Some(dst), Some(tile)) =
                    (out.tile_mut(id), polygon_height_tile(input, p, id))
                {
                    *dst = tile;
                }
            }
        }
    }
    out.refresh_halos();
    out
}

/// Evaluate one polygon tile without allocating or scanning the remaining terrain.
/// Halo exchange is intentionally a separate scheduler step.
pub fn polygon_height_tile(
    input: &Heightfield,
    p: &crate::layer::PolygonHeightParams,
    id: TileId,
) -> Option<HeightTile> {
    let mut out = input.tile(id)?.clone();
    if p.points.len() < 3 {
        return Some(out);
    }
    let short = input
        .metrics
        .world_size_x
        .min(input.metrics.world_size_z)
        .max(1.0);
    let falloff_m = (p.falloff.clamp(0.0, 0.5) * short).max(1.0e-3);
    let (origin_x, origin_z) = out.interior_origin(&input.metrics);
    for lz in 0..out.interior_height {
        for lx in 0..out.interior_width {
            let i = origin_x + lx;
            let j = origin_z + lz;
            let u = (i as f32 + 0.5) / input.metrics.width as f32;
            let v = (j as f32 + 0.5) / input.metrics.height as f32;
            let inside = point_in_polygon(u, v, &p.points);
            let dist = polygon_edge_distance_world(
                u * input.metrics.world_size_x,
                v * input.metrics.world_size_z,
                &p.points,
                input.metrics.world_size_x,
                input.metrics.world_size_z,
            );
            let edge_t = (dist / falloff_m).clamp(0.0, 1.0);
            let weight = if inside {
                1.0
            } else {
                let t = 1.0 - edge_t;
                t * t * (3.0 - 2.0 * t)
            };
            if weight <= 1.0e-4 {
                continue;
            }
            let h0 = input.get(i, j);
            let target = match p.mode {
                crate::layer::PolygonHeightMode::RaiseBy => {
                    h0 + if p.carve { -p.height.abs() } else { p.height }
                }
                crate::layer::PolygonHeightMode::SetElevation => {
                    if p.carve {
                        h0 - p.height.abs()
                    } else {
                        p.height
                    }
                }
            };
            out.set_interior(lx, lz, h0 * (1.0 - weight) + target * weight);
        }
    }
    Some(out)
}

fn world_raster_bounds(
    metrics: HeightfieldMetrics,
    min_x: f32,
    min_z: f32,
    max_x: f32,
    max_z: f32,
    padding: f32,
) -> (u32, u32, u32, u32) {
    let axis = |min_value: f32, max_value: f32, world_size: f32, cells: u32| {
        let scale = cells as f32 / world_size.max(1.0e-6);
        let start = ((min_value - padding) * scale - 1.0)
            .floor()
            .clamp(0.0, cells as f32) as u32;
        let end = ((max_value + padding) * scale + 1.0)
            .ceil()
            .clamp(0.0, cells as f32) as u32;
        (start, end)
    };
    let (i_start, i_end) = axis(min_x, max_x, metrics.world_size_x, metrics.width);
    let (j_start, j_end) = axis(min_z, max_z, metrics.world_size_z, metrics.height);
    (i_start, i_end, j_start, j_end)
}

fn polygon_edge_distance_world(
    x: f32,
    z: f32,
    pts: &[[f32; 2]],
    world_x: f32,
    world_z: f32,
) -> f32 {
    let mut best = f32::INFINITY;
    let n = pts.len();
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        best = best.min(dist_point_segment(
            x,
            z,
            a[0] * world_x,
            a[1] * world_z,
            b[0] * world_x,
            b[1] * world_z,
        ));
    }
    best
}

pub fn blur(input: &Heightfield, p: &BlurParams) -> Heightfield {
    let mut out = input.clone();
    let r = p.radius as i32;
    for _ in 0..p.iterations.max(1) {
        let src = out.to_dense();
        let w = out.metrics.width;
        let h = out.metrics.height;
        for j in 0..h {
            for i in 0..w {
                let mut sum: f32 = 0.0;
                let mut c: f32 = 0.0;
                for dj in -r..=r {
                    for di in -r..=r {
                        let ii = i as i32 + di;
                        let jj = j as i32 + dj;
                        if ii >= 0 && jj >= 0 && ii < w as i32 && jj < h as i32 {
                            sum += src[(jj as u32 * w + ii as u32) as usize];
                            c += 1.0;
                        }
                    }
                }
                out.set(i, j, sum / c.max(1.0));
            }
        }
    }
    out
}

pub fn coastal(input: &Heightfield, p: &CoastalParams) -> Heightfield {
    let mut out = input.clone();
    out.map_mut(|h| {
        if p.flatten_below && h < p.sea_level {
            if p.shelf_depth <= 0.0 {
                p.sea_level
            } else {
                let depth = p.sea_level - h;
                let slope_scale = p.beach_width.max(1e-3);
                p.sea_level - p.shelf_depth * (1.0 - (-depth / slope_scale).exp())
            }
        } else if h < p.sea_level + p.beach_width {
            let t = ((h - p.sea_level) / p.beach_width.max(1e-3)).clamp(0.0, 1.0);
            // Smoothstep keeps the beach tangent gentle at sea level and inland.
            let soft = t * t * (3.0 - 2.0 * t);
            p.sea_level + soft * p.beach_width
        } else {
            h
        }
    });
    out
}

fn fill_world<F: Fn(f32, f32) -> f32 + Sync>(metrics: HeightfieldMetrics, f: F) -> Heightfield {
    let w = metrics.width;
    let h = metrics.height;
    let mut data = vec![0.0f32; (w * h) as usize];
    data.par_iter_mut().enumerate().for_each(|(idx, v)| {
        let i = (idx as u32) % w;
        let j = (idx as u32) / w;
        let x = metrics.world_x(i);
        let z = metrics.world_z(j);
        *v = f(x, z);
    });
    Heightfield::from_dense(metrics, &data)
}

fn mix_strength(
    input: &Heightfield,
    filtered: &Heightfield,
    strength: f32,
    invert: bool,
) -> Heightfield {
    let s = strength.clamp(0.0, 1.0);
    let mut out = input.clone();
    let w = input.metrics.width;
    let h = input.metrics.height;
    for j in 0..h {
        for i in 0..w {
            let a = input.get(i, j);
            let mut b = filtered.get(i, j);
            if invert {
                b = a - (b - a);
            }
            out.set(i, j, a * (1.0 - s) + b * s);
        }
    }
    out
}

pub(super) fn box_blur_height(input: &Heightfield, radius: u32) -> Heightfield {
    filter_kernels::box_blur(input, radius)
}

pub fn effect_filter(input: &Heightfield, p: &EffectFilterParams) -> Heightfield {
    let filtered = match p.kind {
        EffectFilterKind::Smooth => {
            // Low-pass (box) blur: attenuates high-frequency detail and isolated
            // spikes alike. Edge-preserving smoothing lives in `Denoise`
            // (bilateral) — which by design keeps sharp features, including a
            // lone spike, intact and so is the wrong tool for reducing spikes.
            let mut h = input.clone();
            for _ in 0..p.iterations.max(1).min(4) {
                h = filter_kernels::box_blur(&h, p.radius.max(1));
            }
            h
        }
        EffectFilterKind::Distortion => {
            let mut out = input.clone();
            let freq = p.frequency.max(1e-5);
            let amt = p.amount;
            let dx = input.metrics.dx().max(1e-5);
            let dz = input.metrics.dz().max(1e-5);
            for j in 0..input.metrics.height {
                for i in 0..input.metrics.width {
                    let x = input.metrics.world_x(i);
                    let z = input.metrics.world_z(j);
                    let ox =
                        noise::sample_noise(FractalNoiseType::Perlin, x * freq, z * freq, p.seed)
                            * amt;
                    let oz = noise::sample_noise(
                        FractalNoiseType::Perlin,
                        x * freq + 17.0,
                        z * freq + 9.0,
                        p.seed ^ 0xA5A5,
                    ) * amt;
                    let u = ((x + ox) / input.metrics.world_size_x).clamp(0.0, 1.0);
                    let v = ((z + oz) / input.metrics.world_size_z).clamp(0.0, 1.0);
                    let ii = (u * (input.metrics.width - 1) as f32) as u32;
                    let jj = (v * (input.metrics.height - 1) as f32) as u32;
                    let _ = (dx, dz);
                    out.set(i, j, input.get(ii, jj));
                }
            }
            out
        }
        EffectFilterKind::SpikeRemoval => {
            let mut out = input.clone();
            let r = p.radius.max(1) as i32;
            let w = input.metrics.width as i32;
            let h = input.metrics.height as i32;
            for j in 0..h {
                for i in 0..w {
                    let mut vals = Vec::new();
                    for dj in -r..=r {
                        for di in -r..=r {
                            let x = (i + di).clamp(0, w - 1) as u32;
                            let y = (j + dj).clamp(0, h - 1) as u32;
                            vals.push(input.get(x, y));
                        }
                    }
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let median = vals[vals.len() / 2];
                    let center = input.get(i as u32, j as u32);
                    let mad = (center - median).abs();
                    let thresh = p.amount.max(0.5);
                    if mad > thresh {
                        out.set(i as u32, j as u32, median);
                    }
                }
            }
            out
        }
        EffectFilterKind::Shore => {
            let coastal_p = CoastalParams {
                sea_level: p.sea_level,
                beach_width: p.beach_width,
                flatten_below: true,
                shelf_depth: p.amount.max(0.0),
            };
            coastal(input, &coastal_p)
        }
        EffectFilterKind::Strata => {
            // Phase 9 geology: folded/warped beds + soft-bed exposure (Materials-compatible kernels).
            morphology::strata_filter(input, p)
        }
        EffectFilterKind::Crater => morphology::crater_filter(input, p),
        EffectFilterKind::Denoise => {
            // Bilateral denoise preserves cliff edges while removing high-frequency noise.
            let mut h = input.clone();
            let sigma_s = (p.radius.max(1) as f32) * 0.55;
            let sigma_r = p.amount.max(0.5);
            for _ in 0..p.iterations.max(1).min(4) {
                h = filter_kernels::bilateral(&h, p.radius.max(1), sigma_s, sigma_r);
            }
            h
        }
        EffectFilterKind::RockySharp => morphology::rocky_sharp(input, p),
        EffectFilterKind::RockyWide => morphology::rocky_wide(input, p),
        EffectFilterKind::RockyLayers => morphology::rocky_layers(input, p),
        EffectFilterKind::CliffReinforce => morphology::cliff_reinforce(input, p),
        EffectFilterKind::SoftFlows
        | EffectFilterKind::ThinFlows
        | EffectFilterKind::RidgedFlows
        | EffectFilterKind::WideFlows
        | EffectFilterKind::SedimentFlows
        | EffectFilterKind::HydraulicSediment => {
            // Shared HydraulicErosionCore — physical transport biases, not D∞ carve fakes.
            if let Some(core) = crate::analyze::HydraulicErosionCore::from_effect_filter(p) {
                let filtered = core.apply_height(input, None);
                let t = p.strength.clamp(0.0, 1.0);
                if (t - 1.0).abs() < 1e-5 {
                    filtered
                } else {
                    let mut out = input.clone();
                    for j in 0..input.metrics.height {
                        for i in 0..input.metrics.width {
                            let h0 = input.get(i, j);
                            let h1 = filtered.get(i, j);
                            out.set(i, j, h0 * (1.0 - t) + h1 * t);
                        }
                    }
                    out
                }
            } else {
                match p.kind {
                    EffectFilterKind::SoftFlows => morphology::soft_flows(input, p),
                    EffectFilterKind::ThinFlows => morphology::thin_flows(input, p),
                    EffectFilterKind::RidgedFlows => morphology::ridged_flows(input, p),
                    EffectFilterKind::WideFlows => morphology::wide_flows(input, p),
                    EffectFilterKind::SedimentFlows => morphology::sediment_flows(input, p),
                    EffectFilterKind::HydraulicSediment => morphology::hydraulic_sediment(input, p),
                    _ => input.clone(),
                }
            }
        }
        EffectFilterKind::TalusFill => morphology::talus_fill(input, p),
        EffectFilterKind::SedimentFillSoft => morphology::sediment_fill_soft(input, p),
        EffectFilterKind::MudSettle => morphology::mud_settle(input, p),
        EffectFilterKind::RockyPlateaus => morphology::rocky_plateaus(input, p),
        EffectFilterKind::RockyCliffs => morphology::rocky_cliffs(input, p),
        EffectFilterKind::RockyHard => morphology::rocky_hard(input, p),
        EffectFilterKind::Canyon => morphology::canyon(input, p),
        EffectFilterKind::Chipped => morphology::chipped(input, p),
        EffectFilterKind::Cliffs => morphology::cliffs(input, p),
        EffectFilterKind::Rocky => morphology::rocky(input, p),
        EffectFilterKind::AngleBreak => morphology::angle_break(input, p),
        EffectFilterKind::WindCarve => morphology::wind_carve(input, p),
        EffectFilterKind::Inflate => morphology::inflate(input, p),
        EffectFilterKind::Deflate => morphology::deflate(input, p),
        EffectFilterKind::Balloon => morphology::balloon(input, p),
        EffectFilterKind::Blocks => morphology::blocks(input, p),
        EffectFilterKind::Ridged => morphology::ridged_detail(input, p),
        EffectFilterKind::Rugged => morphology::rugged(input, p),
        EffectFilterKind::SmoothRidges => morphology::smooth_ridges(input, p),
        EffectFilterKind::AngleBlur => morphology::angle_blur(input, p),
        EffectFilterKind::DirectionalBlur => morphology::directional_blur(input, p),
        EffectFilterKind::Squeeze => morphology::squeeze(input, p),
        EffectFilterKind::Swirl => morphology::swirl(input, p),
        EffectFilterKind::WashedOff => morphology::washed_off(input, p),
        EffectFilterKind::Hexagons => morphology::hexagons(input, p),
        EffectFilterKind::ScatterDetail => morphology::scatter_detail(input, p),
        EffectFilterKind::FlattenFilter => morphology::flatten_filter(input, p),
        EffectFilterKind::ZeroEdge => morphology::zero_edge(input, p),
        EffectFilterKind::BorderBlend => morphology::border_blend(input, p),
        EffectFilterKind::Curve => morphology::curve_remap(input, p),
        EffectFilterKind::Cutoff => morphology::cutoff(input, p),
        EffectFilterKind::Kuwahara => morphology::kuwahara(input, p),
        EffectFilterKind::TerraceSimple => morphology::terrace_simple(input, p),
        EffectFilterKind::TerraceIrregular => morphology::terrace_irregular(input, p),
        EffectFilterKind::TerraceSteep => morphology::terrace_steep(input, p),
        EffectFilterKind::AddSet => morphology::add_set(input, p),
        EffectFilterKind::DesignVoronoi => morphology::design_voronoi(input, p),
        EffectFilterKind::NoiseBillow => morphology::noise_billow(input, p),
        EffectFilterKind::NoiseGabor => morphology::noise_gabor(input, p),
        EffectFilterKind::NoisePerlin => morphology::noise_perlin(input, p),
        EffectFilterKind::NoisePhasor => morphology::noise_phasor(input, p),
        EffectFilterKind::NoiseRidged => morphology::noise_ridged_filter(input, p),
        EffectFilterKind::NoiseSimplex => morphology::noise_simplex(input, p),
        EffectFilterKind::NoiseValue => morphology::noise_value(input, p),
        EffectFilterKind::NoiseVoronoi => morphology::noise_voronoi(input, p),
        EffectFilterKind::NoiseWave => morphology::noise_wave(input, p),
        EffectFilterKind::NoiseWhite => morphology::noise_white(input, p),
    };
    mix_strength(input, &filtered, p.strength, p.invert)
}

pub fn path_stamp(input: &Heightfield, p: &PathParams) -> (Heightfield, MaskField) {
    if p.nodes.len() < 2 {
        return (input.clone(), MaskField::zeros(input.metrics));
    }
    let mut out = input.clone();
    let mut wetness = MaskField::zeros(input.metrics);
    let world_x = input.metrics.world_size_x;
    let world_z = input.metrics.world_size_z;
    let samples = path_samples(p, world_x, world_z);
    let min_x = samples
        .iter()
        .map(|sample| sample.x)
        .fold(f32::INFINITY, f32::min);
    let max_x = samples
        .iter()
        .map(|sample| sample.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_z = samples
        .iter()
        .map(|sample| sample.z)
        .fold(f32::INFINITY, f32::min);
    let max_z = samples
        .iter()
        .map(|sample| sample.z)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_width_scale = samples
        .iter()
        .map(|sample| sample.width)
        .fold(0.1f32, f32::max);
    let padding = p.width.max(0.1) * max_width_scale.max(0.1) + p.falloff.max(0.1);
    let (i_start, i_end, j_start, j_end) =
        world_raster_bounds(input.metrics, min_x, min_z, max_x, max_z, padding);
    if i_start < i_end && j_start < j_end {
        let tile_size = input.metrics.tile_size;
        let tx_start = i_start / tile_size;
        let tx_end = (i_end - 1) / tile_size;
        let tz_start = j_start / tile_size;
        let tz_end = (j_end - 1) / tile_size;
        for tz in tz_start..=tz_end {
            for tx in tx_start..=tx_end {
                let id = TileId { tx, tz };
                let Some(tile_output) = path_stamp_tile(input, p, id) else {
                    continue;
                };
                let PathStampTile {
                    height,
                    wetness: tile_wetness,
                } = tile_output;
                let (origin_x, origin_z) = height.interior_origin(&input.metrics);
                for lz in 0..height.interior_height {
                    for lx in 0..height.interior_width {
                        let value = tile_wetness[(lz * height.interior_width + lx) as usize];
                        if value > 0.0 {
                            wetness.set(origin_x + lx, origin_z + lz, value);
                        }
                    }
                }
                if let Some(dst) = out.tile_mut(id) {
                    *dst = height;
                }
            }
        }
    }
    out.refresh_halos();
    (out, wetness)
}
#[derive(Debug, Clone)]
pub struct PathStampTile {
    pub height: HeightTile,
    /// Row-major wetness for the tile interior.
    pub wetness: Vec<f32>,
}

/// Evaluate one path tile without touching unrelated tiles.
/// Wetness publication and halo exchange are separate work products in the tiled pipeline.
pub fn path_stamp_tile(input: &Heightfield, p: &PathParams, id: TileId) -> Option<PathStampTile> {
    let mut out = input.tile(id)?.clone();
    let mut wetness = vec![0.0; (out.interior_width * out.interior_height) as usize];
    if p.nodes.len() < 2 {
        return Some(PathStampTile {
            height: out,
            wetness,
        });
    }
    let world_x = input.metrics.world_size_x;
    let world_z = input.metrics.world_size_z;
    let samples = path_samples(p, world_x, world_z);
    let segments = samples.windows(2);
    let (origin_x, origin_z) = out.interior_origin(&input.metrics);
    for lz in 0..out.interior_height {
        for lx in 0..out.interior_width {
            let i = origin_x + lx;
            let j = origin_z + lz;
            let x = (i as f32 + 0.5) / input.metrics.width as f32 * world_x;
            let z = (j as f32 + 0.5) / input.metrics.height as f32 * world_z;
            let mut best = f32::INFINITY;
            let mut width_scale = 1.0f32;
            let mut path_height = 0.0f32;
            for segment in segments.clone() {
                let a = segment[0];
                let b = segment[1];
                let (distance, t) = dist_to_segment(x, z, a.x, a.z, b.x, b.z);
                if distance < best {
                    best = distance;
                    width_scale = a.width + (b.width - a.width) * t;
                    path_height = a.height + (b.height - a.height) * t;
                }
            }
            let half = (p.width * width_scale).max(0.1);
            let falloff = p.falloff.max(0.1);
            if best > half + falloff {
                continue;
            }
            let edge = if best <= half {
                1.0
            } else {
                let t = 1.0 - ((best - half) / falloff).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            }
            .powf(p.profile.clamp(0.25, 4.0));
            let noise = if p.noise_strength > 1.0e-5 {
                noise::sample_noise(
                    FractalNoiseType::Value,
                    x * p.noise_scale,
                    z * p.noise_scale,
                    p.seed,
                ) * p.noise_strength
            } else {
                0.0
            };
            let signed_height = p.height_offset + path_height + noise;
            let delta = if p.carve {
                -signed_height.abs() * edge
            } else {
                signed_height * edge
            };
            out.set_interior(lx, lz, input.get(i, j) + delta);
            if p.carve && delta < 0.0 {
                let wet = (edge * (-delta / signed_height.abs().max(1.0)).clamp(0.0, 1.5))
                    .clamp(0.0, 1.0);
                wetness[(lz * out.interior_width + lx) as usize] = wet;
            }
        }
    }
    Some(PathStampTile {
        height: out,
        wetness,
    })
}

#[derive(Clone, Copy)]
struct PathSample {
    x: f32,
    z: f32,
    height: f32,
    width: f32,
}

fn path_samples(p: &PathParams, world_x: f32, world_z: f32) -> Vec<PathSample> {
    let node = |index: usize| {
        let n = &p.nodes[index];
        PathSample {
            x: n.u.clamp(0.0, 1.0) * world_x,
            z: n.v.clamp(0.0, 1.0) * world_z,
            height: n.height,
            width: n.width.max(0.05),
        }
    };
    if !p.spline || p.nodes.len() < 3 {
        let mut samples: Vec<_> = (0..p.nodes.len()).map(node).collect();
        if p.closed {
            samples.push(samples[0]);
        }
        return samples;
    }

    let n = p.nodes.len();
    let segment_count = if p.closed { n } else { n - 1 };
    let mut out = Vec::new();
    for i in 0..segment_count {
        let i0 = if i == 0 {
            if p.closed {
                n - 1
            } else {
                0
            }
        } else {
            i - 1
        };
        let i1 = i;
        let i2 = (i + 1) % n;
        let i3 = if i + 2 < n {
            i + 2
        } else if p.closed {
            (i + 2) % n
        } else {
            n - 1
        };
        let a = node(i0);
        let b = node(i1);
        let c = node(i2);
        let d = node(i3);
        let length = (c.x - b.x).hypot(c.z - b.z);
        let steps = ((length / p.width.max(1.0)).ceil() as usize).clamp(4, 32);
        for step in 0..steps {
            let t = step as f32 / steps as f32;
            out.push(PathSample {
                x: catmull_rom(a.x, b.x, c.x, d.x, t).clamp(0.0, world_x),
                z: catmull_rom(a.z, b.z, c.z, d.z, t).clamp(0.0, world_z),
                height: catmull_rom(a.height, b.height, c.height, d.height, t),
                width: catmull_rom(a.width, b.width, c.width, d.width, t).max(0.05),
            });
        }
    }
    if p.closed {
        out.push(out[0]);
    } else {
        out.push(node(n - 1));
    }
    out
}

fn catmull_rom(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * b)
        + (-a + c) * t
        + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
        + (-a + 3.0 * b - 3.0 * c + d) * t3)
}

fn dist_to_segment(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> (f32, f32) {
    let abx = bx - ax;
    let aby = by - ay;
    let apx = px - ax;
    let apy = py - ay;
    let ab2 = abx * abx + aby * aby;
    let t = if ab2 < 1e-12 {
        0.0
    } else {
        ((apx * abx + apy * aby) / ab2).clamp(0.0, 1.0)
    };
    let cx = ax + abx * t;
    let cy = ay + aby * t;
    ((px - cx).hypot(py - cy), t)
}

pub fn river_network(input: &Heightfield, p: &RiverNetworkParams) -> (Heightfield, MaskField) {
    let mut out = input.clone();
    let mut wetness = MaskField::zeros(input.metrics);
    let w = input.metrics.width;
    let h = input.metrics.height;
    if p.springs.is_empty() {
        return (out, wetness);
    }
    let cell = input.metrics.dx().max(1e-5);
    // Ensure corridors span at least ~1.5 cells on large worlds.
    let base_width = p.valley_width.max(cell * 1.5);
    let max_len = p.max_length.max(8);
    let nw = w as usize;
    let nh = h as usize;

    // Trace high springs first so lower tributaries can join existing channels.
    let mut springs: Vec<&RiverNode> = p.springs.iter().collect();
    springs.sort_by(|a, b| {
        let ha = sample_uv_height(input, a.u, a.v);
        let hb = sample_uv_height(input, b.u, b.v);
        hb.partial_cmp(&ha).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Occupancy: 0 = empty; else spring ordinal (1-based) that claimed the cell.
    let mut channel_id = vec![0u16; nw * nh];
    let mut next_id = 1u16;

    for spring in springs {
        let mut i = (spring.u.clamp(0.0, 1.0) * (w - 1) as f32) as u32;
        let mut j = (spring.v.clamp(0.0, 1.0) * (h - 1) as f32) as u32;
        let mut path: Vec<(u32, u32)> = Vec::new();
        let mut joined = false;
        let my_id = next_id;
        next_id = next_id.saturating_add(1);

        for _ in 0..max_len {
            let idx = j as usize * nw + i as usize;
            path.push((i, j));
            if channel_id[idx] != 0 && channel_id[idx] != my_id {
                joined = true;
                break;
            }
            if !p.auto_generate {
                break;
            }
            // Steepest descent on the *input* DEM so traces stay stable while carving.
            let mut best = (i, j);
            let mut best_h = input.get(i, j);
            for dj in -1i32..=1 {
                for di in -1i32..=1 {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
                        continue;
                    }
                    let nh = input.get(ni as u32, nj as u32);
                    if nh < best_h {
                        best_h = nh;
                        best = (ni as u32, nj as u32);
                    }
                }
            }
            if best == (i, j) {
                break;
            }
            i = best.0;
            j = best.1;
        }

        // Downstream segments and joins get slightly wider / deeper channels.
        let join_boost = if joined { 1.35 } else { 1.0 };
        let path_len = path.len().max(1) as f32;
        for (step, &(ci, cj)) in path.iter().enumerate() {
            let t = step as f32 / path_len;
            let order = 1.0 + t * 0.75;
            let width = base_width * spring.width.max(0.2) * order * join_boost;
            let depth = p.carve_depth * spring.flow.max(0.1) * order.sqrt() * join_boost;
            carve_disk_wet(&mut out, &mut wetness, ci, cj, width, depth);
            let idx = cj as usize * nw + ci as usize;
            if channel_id[idx] == 0 {
                channel_id[idx] = my_id;
            }
        }
    }
    (out, wetness)
}

fn sample_uv_height(hf: &Heightfield, u: f32, v: f32) -> f32 {
    let i = (u.clamp(0.0, 1.0) * (hf.metrics.width - 1) as f32) as u32;
    let j = (v.clamp(0.0, 1.0) * (hf.metrics.height - 1) as f32) as u32;
    hf.get(i, j)
}

fn carve_disk_wet(
    hf: &mut Heightfield,
    wetness: &mut MaskField,
    ci: u32,
    cj: u32,
    radius_m: f32,
    depth: f32,
) {
    let cell = hf.metrics.dx().max(1e-5);
    let r_cells = (radius_m / cell).ceil() as i32;
    let w = hf.metrics.width as i32;
    let h = hf.metrics.height as i32;
    for dj in -r_cells..=r_cells {
        for di in -r_cells..=r_cells {
            let ni = ci as i32 + di;
            let nj = cj as i32 + dj;
            if ni < 0 || nj < 0 || ni >= w || nj >= h {
                continue;
            }
            let dist = ((di * di + dj * dj) as f32).sqrt() * cell;
            if dist > radius_m {
                continue;
            }
            let t = 1.0 - (dist / radius_m.max(1e-5)).clamp(0.0, 1.0);
            let d = depth * t * t;
            let z = hf.get(ni as u32, nj as u32);
            hf.set(ni as u32, nj as u32, z - d);
            let wet = (t * (depth / 20.0).clamp(0.35, 1.0)).clamp(0.0, 1.0);
            let prev = wetness.get(ni as u32, nj as u32);
            if wet > prev {
                wetness.set(ni as u32, nj as u32, wet);
            }
        }
    }
}

pub fn sand_simulation(input: &Heightfield, p: &SandSimParams) -> (Heightfield, MaskField) {
    let result = sand_simulation_full(input, p, p.iterations.max(1));
    // Normalised material mask for downstream shading.
    let mut depth = MaskField::zeros(input.metrics);
    let raw = result.sand_depth.data();
    let mut max_s = 1e-6f32;
    for &v in raw {
        max_s = max_s.max(v);
    }
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let v = result.sand_depth.get(i, j);
            depth.set(i, j, (v / max_s).clamp(0.0, 1.0));
        }
    }
    (result.height, depth)
}

/// Full aeolian sand simulation with multi-channel outputs.
///
/// Runs Desertscapes-class saltation, sheltering, deposition, and avalanching.
/// `steps` is the transport iteration count (caller applies quality caps).
pub fn sand_simulation_full(
    input: &Heightfield,
    p: &SandSimParams,
    steps: u32,
) -> analyze::AeolianResult {
    let cover = p.spawn_amount.max(0.0);
    let seed_amt = p.seed_amount.max(0.0);
    let sand = if seed_amt > 1e-4 {
        let mut seeded = seed_dune_sand(
            input.metrics,
            p.wind_direction_deg,
            p.seed_scale.max(1e-5),
            cover.max(1.0) * seed_amt,
            seed_amt.clamp(0.1, 2.0),
            0.65,
            p.linearity,
            0x5A17_D00Du64,
        );
        for v in &mut seeded {
            *v = (*v + cover * (1.0 - seed_amt.clamp(0.0, 1.0))).max(0.0);
        }
        seeded
    } else {
        vec![cover; (input.metrics.width * input.metrics.height) as usize]
    };

    let mut state = analyze::AeolianState::from_height_and_sand(input, 0.0, Some(&sand));
    // Treat input height as bedrock + existing cover: peel authored sand off the DEM.
    let dense = input.to_dense();
    for i in 0..dense.len() {
        let s = state.sand[i].min(dense[i].max(0.0));
        state.sand[i] = s;
        state.bedrock[i] = (dense[i] - s).max(0.0);
    }

    let transport = analyze::AeolianTransportParams {
        wind_direction_deg: p.wind_direction_deg,
        wind_speed: p.wind_speed.max(0.05),
        transport_length: p.transport_length.max(0.5),
        repose_angle_deg: p.slope_angle_deg.clamp(12.0, 45.0),
        slab_size: (cover * 0.08).clamp(0.05, 1.5),
        avalanche_iters: p.avalanche_iters.max(1).min(64),
        abrasion: p.abrasion.clamp(0.0, 1.0),
        reptation: p.reptation.clamp(0.0, 1.0),
        wind_warp: 0.4,
        linearity: p.linearity.clamp(0.0, 1.0),
    };
    state.evolve(&transport, steps.max(1));
    state.into_result(p.strength.clamp(0.0, 1.0), input)
}

pub fn fluid_simulation(input: &Heightfield, p: &FluidSimParams) -> (Heightfield, MaskField) {
    let mut water = Heightfield::filled(input.metrics, p.spawn_amount);
    let terrain = input.clone();
    let w = input.metrics.width as i32;
    let h = input.metrics.height as i32;
    let visc = p.viscosity.clamp(0.05, 0.95);
    for _ in 0..p.iterations.max(1) {
        let snap_w = water.clone();
        let snap_t = terrain.clone();
        for j in 0..h {
            for i in 0..w {
                let surface = snap_t.get(i as u32, j as u32) + snap_w.get(i as u32, j as u32);
                let mut outflow = 0.0f32;
                let mut targets = Vec::new();
                for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let ni = i + di;
                    let nj = j + dj;
                    if ni < 0 || nj < 0 || ni >= w || nj >= h {
                        continue;
                    }
                    let ns = snap_t.get(ni as u32, nj as u32) + snap_w.get(ni as u32, nj as u32);
                    if ns < surface {
                        let d = (surface - ns) * (1.0 - visc) * 0.2;
                        outflow += d;
                        targets.push((ni, nj, d));
                    }
                }
                let available = snap_w.get(i as u32, j as u32);
                let scale = if outflow > available && outflow > 1e-6 {
                    available / outflow
                } else {
                    1.0
                };
                let mut moved = 0.0f32;
                for (ni, nj, d) in targets {
                    let amt = d * scale;
                    moved += amt;
                    let dst = water.get(ni as u32, nj as u32);
                    water.set(ni as u32, nj as u32, dst + amt);
                }
                water.set(i as u32, j as u32, (available - moved).max(0.0));
            }
        }
    }
    // Fill depressions: remaining water becomes lake surface + wetness aux.
    let mut out = input.clone();
    let mut mask = MaskField::zeros(input.metrics);
    let spawn = p.spawn_amount.max(1e-3);
    for j in 0..input.metrics.height {
        for i in 0..input.metrics.width {
            let wv = water.get(i, j);
            if wv > 1e-4 {
                // Full water column (scaled by strength) so lakes read on large DEMs.
                out.set(i, j, input.get(i, j) + wv * p.strength);
                mask.set(i, j, (wv / spawn).clamp(0.0, 1.0));
            }
        }
    }
    (out, mask)
}
#[cfg(test)]
mod tests {
    use super::*;

    fn field_bits(height: &Heightfield) -> Vec<u32> {
        height
            .to_dense()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    #[test]
    fn high_seed_bits_change_generators_and_seeded_effect_filters() {
        let low = 7;
        let high = low + (1u64 << 32);
        let metrics = HeightfieldMetrics::new(48, 48, 768.0, 768.0);

        let mut island_low = IslandParams::default();
        island_low.seed = low;
        let mut island_high = island_low.clone();
        island_high.seed = high;
        assert_ne!(
            field_bits(&island(metrics, &island_low).height),
            field_bits(&island(metrics, &island_high).height)
        );

        let mut canyon_low = CanyonParams::default();
        canyon_low.seed = low;
        let mut canyon_high = canyon_low.clone();
        canyon_high.seed = high;
        assert_ne!(
            field_bits(&canyons(metrics, &canyon_low)),
            field_bits(&canyons(metrics, &canyon_high))
        );

        let input = Heightfield::filled(metrics, 25.0);
        for kind in [
            EffectFilterKind::NoiseGabor,
            EffectFilterKind::NoiseWhite,
            EffectFilterKind::ScatterDetail,
        ] {
            let low_params = EffectFilterParams {
                kind,
                seed: low,
                ..EffectFilterParams::default()
            };
            let high_params = EffectFilterParams {
                seed: high,
                ..low_params.clone()
            };
            assert_ne!(
                field_bits(&effect_filter(&input, &low_params)),
                field_bits(&effect_filter(&input, &high_params)),
                "{kind:?} still aliases seeds separated by 2^32"
            );
        }
    }

    #[test]
    fn island_has_submerged_border_single_landmass_scale_and_semantic_fields() {
        let metrics = HeightfieldMetrics::new(128, 128, 4096.0, 4096.0);
        let params = IslandParams::default();
        let result = island(metrics, &params);
        for i in 0..metrics.width {
            assert!(result.height.get(i, 0) < params.sea_level);
            assert!(result.height.get(i, metrics.height - 1) < params.sea_level);
        }
        for j in 0..metrics.height {
            assert!(result.height.get(0, j) < params.sea_level);
            assert!(result.height.get(metrics.width - 1, j) < params.sea_level);
        }
        let dense = result.height.to_dense();
        let land_cells = dense.iter().filter(|&&v| v > params.sea_level).count();
        let land_fraction = land_cells as f32 / dense.len() as f32;
        assert!(
            (0.12..0.68).contains(&land_fraction),
            "land fraction {land_fraction}"
        );
        let max = dense.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = dense.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(max - min > 400.0, "island relief {}", max - min);
        for j in 0..metrics.height {
            for i in 0..metrics.width {
                let is_land = result.land.get(i, j) > 0.5;
                let signed = result.shore_distance.get(i, j);
                assert_eq!(is_land, signed < 0.0);
                if is_land {
                    assert_eq!(result.bathymetry.get(i, j), 0.0);
                }
            }
        }
    }

    #[test]
    fn atoll_keeps_lagoon_submerged_and_ring_above_water() {
        let metrics = HeightfieldMetrics::new(128, 128, 4096.0, 4096.0);
        let params = IslandParams::atoll();
        let result = island(metrics, &params);
        let c = metrics.width / 2;
        assert!(result.height.get(c, c) < params.sea_level);
        let mut ring_found = false;
        for i in c..metrics.width {
            if result.height.get(i, c) > params.sea_level {
                ring_found = true;
                break;
            }
        }
        assert!(ring_found, "atoll ring must rise above sea level");
    }
    #[test]
    fn mesa_has_flat_cap_and_zero_outside() {
        let metrics = HeightfieldMetrics::new(64, 64, 1000.0, 1000.0);
        let p = MesaParams {
            center_u: 0.5,
            center_v: 0.5,
            radius: 0.2,
            height: 200.0,
            cap_noise: 0.0,
            edge_steepness: 3.0,
            soft: 0.1,
            seed: 1,
        };
        let hf = mesa(metrics, &p);
        let center = hf.get(32, 32);
        assert!(
            (center - 200.0).abs() < 1.0,
            "cap should be near height, got {center}"
        );
        let corner = hf.get(0, 0);
        assert!(
            corner < 1.0,
            "outside mesa should be near zero, got {corner}"
        );
    }

    #[test]
    fn dunes_linearity_changes_lateral_coherence() {
        let metrics = HeightfieldMetrics::new(96, 96, 1536.0, 1536.0);
        let mut linear = DuneParams::default();
        linear.linearity = 0.95;
        linear.sand_supply = 1.0;
        linear.iterations = 8;
        linear.direction_deg = 0.0;
        let mut star = linear.clone();
        star.linearity = 0.1;
        let a = dunes(metrics, &linear).to_dense();
        let b = dunes(metrics, &star).to_dense();
        // Cross-wind variance along a column should differ when linearity changes.
        let col_var = |data: &[f32]| {
            let mut sum = 0.0f32;
            let mut sum2 = 0.0f32;
            let n = 96usize;
            for j in 0..n {
                let v = data[j * n + 48];
                sum += v;
                sum2 += v * v;
            }
            let mean = sum / n as f32;
            (sum2 / n as f32 - mean * mean).max(0.0)
        };
        let va = col_var(&a);
        let vb = col_var(&b);
        assert!(
            (va - vb).abs() > 1e-3,
            "linearity should alter crest organisation: linear_var={va} star_var={vb}"
        );
    }

    #[test]
    fn sand_simulation_publishes_transport_fields() {
        let metrics = HeightfieldMetrics::new(64, 64, 640.0, 640.0);
        let input = Heightfield::filled(metrics, 12.0);
        let mut p = SandSimParams::default();
        p.iterations = 24;
        p.avalanche_iters = 4;
        p.spawn_amount = 2.0;
        p.seed_amount = 0.5;
        let result = sand_simulation_full(&input, &p, 24);
        assert!(result.sand_depth.data().iter().any(|&v| v > 0.0));
        assert!(result.sheltering.data().iter().any(|&v| v >= 0.0));
        assert!(result.sand_flux.data().iter().copied().sum::<f32>() > 0.0);
    }
    #[test]
    fn mountains_keep_continuous_crests_without_saturated_plateaus() {
        let metrics = HeightfieldMetrics::new(128, 128, 4096.0, 4096.0);
        let p = MountainParams::default();
        let hf = mountains(metrics, &p);
        let dense = hf.to_dense();
        let (min_h, max_h) = hf.min_max();
        assert!(
            max_h - min_h > 140.0,
            "mountain relief was {min_h}..{max_h}"
        );
        let near_peak = dense
            .iter()
            .filter(|&&h| h >= max_h - (max_h - min_h) * 0.015)
            .count();
        let peak_fraction = near_peak as f32 / dense.len() as f32;
        assert!(
            peak_fraction < 0.06,
            "{peak_fraction:.3} of cells saturated near the peak"
        );
    }

    #[test]
    fn volcano_has_a_crater_below_an_irregular_rim() {
        let metrics = HeightfieldMetrics::new(129, 129, 4096.0, 4096.0);
        let p = VolcanoParams::default();
        let hf = volcano(metrics, &p);
        let center = hf.get(64, 64);
        let max_h = hf.to_dense().into_iter().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_h > center + p.crater_depth * 0.35,
            "crater center {center} should sit below rim {max_h}"
        );
    }

    #[test]
    fn dunes_and_canyons_do_not_repeat_one_flat_cross_section() {
        let dune_metrics = HeightfieldMetrics::new(128, 128, 2048.0, 2048.0);
        let dune_hf = dunes(dune_metrics, &DuneParams::default());
        let dense = dune_hf.to_dense();
        let min_h = dense.iter().copied().fold(f32::INFINITY, f32::min);
        let floor_cells = dense.iter().filter(|&&h| (h - min_h).abs() < 1e-4).count();
        assert!(
            floor_cells as f32 / (dense.len() as f32) < 0.03,
            "dunes contain a broad exactly-flat floor"
        );

        let canyon_metrics = HeightfieldMetrics::new(160, 128, 3200.0, 3200.0);
        let p = CanyonParams::default();
        let canyon = canyons(canyon_metrics, &p);
        let mut widths = Vec::new();
        let mut centers = Vec::new();
        for j in 0..canyon_metrics.height {
            let carved: Vec<u32> = (0..canyon_metrics.width)
                .filter(|&i| canyon.get(i, j) < -p.depth * 0.12)
                .collect();
            if let (Some(first), Some(last)) = (carved.first(), carved.last()) {
                widths.push((*last - *first) as f32);
                centers.push((*last + *first) as f32 * 0.5);
            }
        }
        let span = |values: &[f32]| {
            values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - values.iter().copied().fold(f32::INFINITY, f32::min)
        };
        assert!(span(&widths) > 2.0, "canyon width should vary downstream");
        assert!(span(&centers) > 2.0, "canyon centerline should meander");
    }

    #[test]
    fn polygon_relative_height_adds_without_doubling_existing_terrain() {
        let metrics = HeightfieldMetrics::new(64, 32, 2000.0, 500.0);
        let input = Heightfield::filled(metrics, 100.0);
        let p = crate::layer::PolygonHeightParams {
            points: vec![[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
            height: 40.0,
            falloff: 0.0,
            carve: false,
            mode: crate::layer::PolygonHeightMode::RaiseBy,
        };
        let out = polygon_height(&input, &p);
        assert!((out.get(32, 16) - 140.0).abs() < 1e-4);
        assert!((out.get(2, 2) - 100.0).abs() < 1e-4);
        assert_eq!(
            crate::layer::LayerKind::PolygonHeight(p).default_blend(),
            crate::layer::BlendMode::Normal
        );
    }

    #[test]
    fn path_node_heights_create_an_elevation_profile() {
        let metrics = HeightfieldMetrics::new(101, 101, 1000.0, 1000.0);
        let input = Heightfield::filled(metrics, 100.0);
        let p = PathParams {
            nodes: vec![
                PathNode {
                    u: 0.1,
                    v: 0.5,
                    height: 0.0,
                    width: 1.0,
                },
                PathNode {
                    u: 0.9,
                    v: 0.5,
                    height: 80.0,
                    width: 1.0,
                },
            ],
            width: 10.0,
            falloff: 10.0,
            height_offset: 10.0,
            carve: false,
            spline: false,
            ..PathParams::default()
        };
        let (out, _) = path_stamp(&input, &p);
        let start = out.get(10, 50);
        let end = out.get(90, 50);
        assert!(start > 108.0, "start profile should raise terrain: {start}");
        assert!(
            end > start + 60.0,
            "end profile should rise along path: {start} -> {end}"
        );
        assert!((out.get(50, 5) - 100.0).abs() < 1e-4);
    }

    #[test]
    fn path_width_is_measured_in_world_space_on_rectangular_terrain() {
        let metrics = HeightfieldMetrics::new(101, 101, 4000.0, 1000.0);
        let input = Heightfield::filled(metrics, 0.0);
        let p = PathParams {
            nodes: vec![
                PathNode {
                    u: 0.1,
                    v: 0.5,
                    height: 0.0,
                    width: 1.0,
                },
                PathNode {
                    u: 0.9,
                    v: 0.5,
                    height: 0.0,
                    width: 1.0,
                },
            ],
            width: 10.0,
            falloff: 10.0,
            height_offset: 20.0,
            carve: false,
            spline: false,
            ..PathParams::default()
        };
        let (out, _) = path_stamp(&input, &p);
        // 10 m from the centreline: inside the 10 m core + 10 m shoulder.
        assert!(
            out.get(50, 51) > 0.1,
            "rectangular worlds must not use max-axis UV distance"
        );
        assert!(
            (out.get(50, 55)).abs() < 1e-4,
            "50 m offset should be outside"
        );
    }

    #[test]
    fn path_profile_can_cross_from_cut_to_fill_without_carve_mode() {
        let metrics = HeightfieldMetrics::new(101, 101, 1000.0, 1000.0);
        let input = Heightfield::filled(metrics, 100.0);
        let p = PathParams {
            nodes: vec![
                PathNode {
                    u: 0.1,
                    v: 0.5,
                    height: -30.0,
                    width: 1.0,
                },
                PathNode {
                    u: 0.9,
                    v: 0.5,
                    height: 30.0,
                    width: 1.0,
                },
            ],
            width: 12.0,
            falloff: 8.0,
            height_offset: 0.0,
            carve: false,
            spline: false,
            ..PathParams::default()
        };
        let (out, wetness) = path_stamp(&input, &p);
        let cut = out.get(10, 50);
        let fill = out.get(90, 50);
        assert!(
            cut < 75.0,
            "negative path profile should cut terrain: {cut}"
        );
        assert!(
            fill > 125.0,
            "positive path profile should raise terrain: {fill}"
        );
        assert_eq!(
            wetness.get(10, 50),
            0.0,
            "ordinary signed paths are not rivers"
        );
    }
}
