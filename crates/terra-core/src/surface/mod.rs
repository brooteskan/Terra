//! Materials, biomes, and vegetation density maps.

use crate::fields::bake_hardness_from_materials_ex;
use crate::geology::{hardness_at_strata_depth, material_id_at_strata_depth};
use crate::heightfield::Heightfield;
use crate::mask::{MaskField, MaskSource};
use crate::material_schema::{BiomesParams, MaterialsParams, Stratum};
use crate::scatter::{self, VegetationParams};
use crate::spatial_kernels::slope_degrees;
use std::collections::HashMap;

/// Classify surface material IDs and bake hardness.
///
/// Priority per cell:
/// 1. Rule mask (`MaskSource`) when weight &gt; 0.5
/// 2. Slope/height rules (last matching rule wins)
/// 3. Top stratum ID when a strata stack is present
///
/// Hardness is consistent with the same ID → rule/stratum lookup.
pub fn bake_materials(
    hf: &Heightfield,
    p: &MaterialsParams,
    wetness: Option<&MaskField>,
    aux: &HashMap<String, MaskField>,
    mask_assets: &[crate::mask::MaskAsset],
    baked_masks: Option<&HashMap<crate::mask::MaskId, MaskField>>,
) -> (MaskField, MaskField) {
    let mut weights = material_weights(hf, p, wetness, aux, mask_assets);
    if !p.coverage.is_empty() {
        if let Some(masks) = baked_masks {
            let cov = crate::mask::bake_distribution(&p.coverage, masks, hf.metrics);
            for j in 0..hf.metrics.height {
                for i in 0..hf.metrics.width {
                    let c = cov.get(i, j);
                    if c < 0.05 {
                        weights.set(i, j, 0.0);
                    }
                }
            }
        }
    }
    let hardness =
        bake_hardness_from_materials_ex(&weights, &p.rules, &p.strata, p.default_hardness);
    (weights, hardness)
}

pub fn material_weights(
    hf: &Heightfield,
    p: &MaterialsParams,
    _wetness: Option<&MaskField>,
    aux: &HashMap<String, MaskField>,
    mask_assets: &[crate::mask::MaskAsset],
) -> MaskField {
    let slope = slope_degrees(hf);
    let surface_id = if !p.strata.is_empty() {
        material_id_at_strata_depth(&p.strata, 0.0)
    } else {
        0.0
    };
    let mut out = MaskField::filled(hf.metrics, surface_id);

    // Pre-bake optional rule masks once.
    let mut rule_masks: Vec<Option<MaskField>> = Vec::with_capacity(p.rules.len());
    for rule in &p.rules {
        rule_masks.push(bake_rule_mask(&rule.mask, hf, aux, mask_assets));
    }

    for j in 0..hf.metrics.height {
        for i in 0..hf.metrics.width {
            let h = hf.get(i, j);
            let s = slope.get(i, j) * 90.0;
            let mut best_id = surface_id;

            for (ri, rule) in p.rules.iter().enumerate() {
                let mask_hit = rule_masks[ri]
                    .as_ref()
                    .map(|m| m.get(i, j) > 0.5)
                    .unwrap_or(false);
                let band_hit = s >= rule.min_slope_deg
                    && s <= rule.max_slope_deg
                    && h >= rule.min_height
                    && h <= rule.max_height;
                // Mask force, or slope/height band when no mask / mask not excluding.
                if mask_hit || band_hit {
                    best_id = rule.id as f32 / 16.0;
                }
            }

            out.set(i, j, best_id.clamp(0.0, 1.0));
        }
    }
    out
}

fn bake_rule_mask(
    source: &MaskSource,
    hf: &Heightfield,
    aux: &HashMap<String, MaskField>,
    mask_assets: &[crate::mask::MaskAsset],
) -> Option<MaskField> {
    match source {
        MaskSource::None => None,
        other => {
            let baked = crate::mask::bake_mask_assets(
                &[crate::mask::MaskAsset {
                    id: crate::mask::MaskId::new(),
                    name: "mat_rule".into(),
                    source: other.clone(),
                    ops: Vec::new(),
                    paint: None,
                    display_color: crate::mask::default_mask_display_color(),
                }],
                hf,
                hf.metrics,
                aux,
            );
            // Painted sources need the real asset with paint buffer.
            if let MaskSource::Painted { mask_id } = other {
                let from_assets = crate::mask::bake_mask_assets(mask_assets, hf, hf.metrics, aux);
                return from_assets.get(mask_id).cloned();
            }
            baked.into_values().next()
        }
    }
}

/// Depth-aware surface material IDs after erosion has peeled strata.
pub fn material_weights_at_depth(
    reference: &MaskField,
    current: &Heightfield,
    strata: &[Stratum],
) -> MaskField {
    let mut out = MaskField::zeros(current.metrics);
    for j in 0..current.metrics.height {
        for i in 0..current.metrics.width {
            let depth = (reference.get(i, j) - current.get(i, j)).max(0.0);
            out.set(i, j, material_id_at_strata_depth(strata, depth));
        }
    }
    out
}

/// Convenience: surface hardness from strata at depth 0 (no heightfield needed).
pub fn surface_strata_hardness(p: &MaterialsParams) -> f32 {
    if p.strata.is_empty() {
        p.default_hardness.clamp(0.0, 1.0)
    } else {
        hardness_at_strata_depth(&p.strata, 0.0, p.default_hardness)
    }
}

pub fn biome_mask(hf: &Heightfield, p: &BiomesParams, wetness: Option<&MaskField>) -> MaskField {
    if p.use_climate {
        crate::climate::bake_climate(hf, p, wetness, None).biomes
    } else {
        let mut out = MaskField::zeros(hf.metrics);
        for j in 0..hf.metrics.height {
            for i in 0..hf.metrics.width {
                let h = hf.get(i, j);
                let w = wetness.map(|m| m.get(i, j)).unwrap_or(0.5);
                let mut id = 0.0;
                for band in &p.bands {
                    if h >= band.min_height
                        && h <= band.max_height
                        && w >= band.min_wetness
                        && w <= band.max_wetness
                    {
                        id = band.id as f32 / 16.0;
                    }
                }
                out.set(i, j, id);
            }
        }
        out
    }
}

/// Full climate + biome bake (writes all Phase H aux fields via the returned maps).
pub fn bake_biomes_climate(
    hf: &Heightfield,
    p: &BiomesParams,
    wetness: Option<&MaskField>,
    sediment: Option<&MaskField>,
) -> crate::climate::ClimateMaps {
    if p.use_climate {
        crate::climate::bake_climate(hf, p, wetness, sediment)
    } else {
        let biomes = biome_mask(hf, p, wetness);
        let zeros = MaskField::zeros(hf.metrics);
        crate::climate::ClimateMaps {
            temperature: zeros.clone(),
            rainfall: zeros.clone(),
            humidity: zeros.clone(),
            aridity: MaskField::filled(hf.metrics, 0.5),
            snow: zeros.clone(),
            soil_moisture: wetness.cloned().unwrap_or(zeros.clone()),
            wind_exposure: zeros,
            biomes,
        }
    }
}

pub fn vegetation_density(
    hf: &Heightfield,
    p: &VegetationParams,
    biomes: Option<&MaskField>,
    baked_masks: Option<&HashMap<crate::mask::MaskId, MaskField>>,
    land: Option<&MaskField>,
) -> MaskField {
    let slope = slope_degrees(hf);
    let coverage = if !p.coverage.is_empty() {
        baked_masks.map(|m| crate::mask::bake_distribution(&p.coverage, m, hf.metrics))
    } else {
        None
    };
    let mut suitability = MaskField::zeros(hf.metrics);
    for j in 0..hf.metrics.height {
        for i in 0..hf.metrics.width {
            let s = slope.get(i, j) * 90.0;
            if s < p.min_slope_deg || s > p.max_slope_deg {
                continue;
            }
            if let (Some(b), Some(want)) = (biomes, p.biome_id) {
                let id = (b.get(i, j) * 16.0).round() as u32;
                if id != want {
                    continue;
                }
            }
            let cov = coverage.as_ref().map(|c| c.get(i, j)).unwrap_or(1.0)
                * land.map(|m| m.get(i, j)).unwrap_or(1.0);
            // Density drives scatter rate only; suitability stays in \[0,1\] so the
            // viewport aux map remains readable (old path capped at `density`).
            suitability.set(i, j, cov.clamp(0.0, 1.0));
        }
    }
    // Encode scatter count proxy via poisson acceptance
    let points = scatter::poisson_disk(hf, p);
    let mut dens = MaskField::zeros(hf.metrics);
    for &(x, z) in &points {
        let (i, j) = hf.metrics.sample_index(x, z);
        dens.set(i, j, 1.0);
    }
    // Soften — larger blur so Draft (256) still shows a forest canopy tint.
    let blur_r = if hf.metrics.width <= 512 { 2 } else { 1 };
    crate::mask::apply_mask_ops(&mut dens, &[crate::mask::MaskOp::Blur { radius: blur_r }]);
    for j in 0..hf.metrics.height {
        for i in 0..hf.metrics.width {
            dens.set(
                i,
                j,
                (dens.get(i, j) * suitability.get(i, j)).clamp(0.0, 1.0),
            );
        }
    }
    dens
}
