//! Operational field baking with compatibility re-exports for field data.

pub use crate::field_data::{
    fields_invalidated_by, height_dependents, keys, shared_physical_fields, AuxMaps, FieldId,
    IterativeFieldGuard,
};
pub use crate::geology::{
    erodibility_at_strata_depth, hardness_at_strata_depth, material_id_at_strata_depth,
    stability_at_strata_depth,
};

use crate::heightfield::Heightfield;
use crate::mask_field::MaskField;

/// Resolve a hardness field from constant + optional mask source / baked aux.
pub fn resolve_hardness(
    metrics: crate::heightfield::HeightfieldMetrics,
    constant: f32,
    painted: Option<&MaskField>,
    aux: &AuxMaps,
) -> MaskField {
    if let Some(p) = painted {
        return p.clone();
    }
    if let Some(h) = aux.hardness.as_ref() {
        return h.clone();
    }
    MaskField::filled(metrics, constant.clamp(0.0, 1.0))
}

/// Bake hardness from reference vs current heights using a strata stack.
pub fn bake_hardness_from_strata(
    reference: &MaskField,
    current: &Heightfield,
    strata: &[crate::material_schema::Stratum],
    default_hardness: f32,
) -> MaskField {
    bake_hardness_from_strata_ex(
        reference,
        current,
        strata,
        default_hardness,
        &crate::material_schema::BedGeometry::Horizontal,
    )
}

/// Bake hardness with bed geometry warp (tilted / folded / warped beds).
pub fn bake_hardness_from_strata_ex(
    reference: &MaskField,
    current: &Heightfield,
    strata: &[crate::material_schema::Stratum],
    default_hardness: f32,
    geom: &crate::material_schema::BedGeometry,
) -> MaskField {
    let mut out = MaskField::filled(current.metrics, default_hardness.clamp(0.0, 1.0));
    for j in 0..current.metrics.height {
        for i in 0..current.metrics.width {
            let x = current.metrics.world_x(i);
            let z = current.metrics.world_z(j);
            let depth =
                crate::geology::strata_depth_m(reference.get(i, j), current.get(i, j), x, z, geom);
            out.set(
                i,
                j,
                hardness_at_strata_depth(strata, depth, default_hardness),
            );
        }
    }
    out
}

/// Bake a hardness map from material ID weights + per-rule hardness values.
///
/// Each cell looks up the rule whose quantized id matches the materials mask,
/// else the top stratum hardness (if any), else `default_hardness`.
pub fn bake_hardness_from_materials(
    materials: &MaskField,
    rules: &[crate::material_schema::MaterialRule],
    default_hardness: f32,
) -> MaskField {
    bake_hardness_from_materials_ex(materials, rules, &[], default_hardness)
}

/// Materials → \(K\) bake with optional strata fallback for unmatched IDs.
pub fn bake_hardness_from_materials_ex(
    materials: &MaskField,
    rules: &[crate::material_schema::MaterialRule],
    strata: &[crate::material_schema::Stratum],
    default_hardness: f32,
) -> MaskField {
    let surface_k = if !strata.is_empty() {
        hardness_at_strata_depth(strata, 0.0, default_hardness)
    } else {
        default_hardness.clamp(0.0, 1.0)
    };
    let mut out = MaskField::filled(materials.metrics, surface_k);
    for j in 0..materials.metrics.height {
        for i in 0..materials.metrics.width {
            let id = (materials.get(i, j) * 16.0).round() as u32;
            let mut k = surface_k;
            let mut matched = false;
            for rule in rules {
                if rule.id == id {
                    k = rule.hardness;
                    matched = true;
                    break;
                }
            }
            if !matched {
                for s in strata {
                    if s.id == id {
                        k = s.hardness;
                        matched = true;
                        break;
                    }
                }
            }
            if !matched && id == 0 {
                k = surface_k;
            }
            out.set(i, j, k.clamp(0.0, 1.0));
        }
    }
    out
}

/// Whether aux carries a strata stack for depth-aware erosion.
pub fn has_depth_aware_strata(aux: &AuxMaps) -> bool {
    aux.strata.as_ref().map(|s| !s.is_empty()).unwrap_or(false) && aux.strata_reference.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;
    use std::collections::HashMap;

    #[test]
    fn string_adapter_roundtrip() {
        let m = HeightfieldMetrics::new(4, 4, 4.0, 4.0);
        let mut aux = AuxMaps::new();
        aux.insert(keys::WETNESS, MaskField::filled(m, 0.5));
        aux.insert("custom", MaskField::filled(m, 0.25));
        let map = aux.to_hashmap();
        assert!((map["wetness"].get(0, 0) - 0.5).abs() < 1e-6);
        assert!((map["custom"].get(0, 0) - 0.25).abs() < 1e-6);
        let back = AuxMaps::from_hashmap(&map);
        assert!(back.wetness.is_some());
        assert!(back.extras.contains_key("custom"));
    }

    #[test]
    fn sediment_aliases_ingest_but_only_the_canonical_key_is_emitted() {
        let m = HeightfieldMetrics::new(2, 2, 2.0, 2.0);
        for alias in [keys::SEDIMENT_DEPTH, keys::LOOSE_SEDIMENT] {
            let mut map = HashMap::new();
            map.insert(alias.to_string(), MaskField::from_raw(m, &[1.25; 4]));
            let aux = AuxMaps::from_hashmap(&map);
            assert_eq!(aux.sediment_thickness.as_ref().unwrap().get(0, 0), 1.25);

            let emitted = aux.to_hashmap();
            assert_eq!(emitted[keys::SEDIMENT_THICKNESS].get(0, 0), 1.25);
            assert!(!emitted.contains_key(keys::SEDIMENT_DEPTH));
            assert!(!emitted.contains_key(keys::LOOSE_SEDIMENT));
        }
    }

    #[test]
    fn canonical_sediment_wins_mixed_legacy_ingest_and_debris_stays_distinct() {
        let m = HeightfieldMetrics::new(2, 2, 2.0, 2.0);
        let mut map = HashMap::new();
        map.insert(
            keys::LOOSE_SEDIMENT.into(),
            MaskField::from_raw(m, &[1.0; 4]),
        );
        map.insert(
            keys::SEDIMENT_DEPTH.into(),
            MaskField::from_raw(m, &[2.0; 4]),
        );
        map.insert(
            keys::SEDIMENT_THICKNESS.into(),
            MaskField::from_raw(m, &[3.0; 4]),
        );
        map.insert(keys::DEBRIS_DEPTH.into(), MaskField::from_raw(m, &[4.0; 4]));

        let aux = AuxMaps::from_hashmap(&map);
        assert_eq!(aux.sediment_thickness.as_ref().unwrap().get(0, 0), 3.0);
        assert_eq!(aux.get(keys::DEBRIS_DEPTH).unwrap().get(0, 0), 4.0);

        let emitted = aux.to_hashmap();
        assert_eq!(emitted[keys::SEDIMENT_THICKNESS].get(0, 0), 3.0);
        assert_eq!(emitted[keys::DEBRIS_DEPTH].get(0, 0), 4.0);
        assert!(!emitted.contains_key(keys::SEDIMENT_DEPTH));
        assert!(!emitted.contains_key(keys::LOOSE_SEDIMENT));
    }

    #[test]
    fn derived_slope_curvature() {
        let m = HeightfieldMetrics::new(8, 8, 8.0, 8.0);
        let mut hf = Heightfield::zeros(m);
        hf.set(4, 4, 10.0);
        let mut aux = AuxMaps::new();
        aux.ensure_derived(&hf);
        assert!(aux.slope.is_some());
        assert!(aux.curvature.is_some());
        // Slope is highest on the flanks of the spike, not necessarily at the peak.
        assert!(aux.slope.as_ref().unwrap().get(3, 4) > 0.0);
        assert!(aux.slope.as_ref().unwrap().get(5, 4) > 0.0);
    }

    #[test]
    fn strata_survives_hashmap_roundtrip_when_preserved() {
        use crate::layer::Stratum;
        let m = HeightfieldMetrics::new(4, 4, 4.0, 4.0);
        let mut aux = AuxMaps::new();
        aux.insert(keys::HARDNESS, MaskField::filled(m, 0.4));
        aux.strata = Some(vec![Stratum::soft_cap(10.0), Stratum::hard_base()]);
        aux.strata_reference = Some(MaskField::filled(m, 50.0));
        let map = aux.to_hashmap();
        let lost = AuxMaps::from_hashmap(&map);
        assert!(lost.strata.is_none(), "HashMap alone cannot carry strata");
        let kept = AuxMaps::from_hashmap_preserving_strata(&map, aux.strata.clone());
        assert!(has_depth_aware_strata(&kept));
        assert_eq!(kept.strata.as_ref().unwrap().len(), 2);
    }
}
