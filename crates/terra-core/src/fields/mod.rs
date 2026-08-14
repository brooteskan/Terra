//! Typed auxiliary maps for erosion / analysis, with string-key adapters.
//!
//! Processors should prefer the typed fields on [`AuxMaps`]. The HashMap adapters
//! keep mask baking, disk cache, and export paths working during migration.

mod id;
mod invalidation;

pub use id::FieldId;
pub use invalidation::{
    fields_invalidated_by, height_dependents, shared_physical_fields, IterativeFieldGuard,
};

use crate::analyze::{curvature, slope_degrees};
use crate::heightfield::Heightfield;
use crate::mask::MaskField;
use std::collections::HashMap;

/// Well-known aux map keys (string adapter / disk cache names).
pub mod keys {
    pub const WETNESS: &str = "wetness";
    pub const SEDIMENT: &str = "sediment";
    pub const EROSION: &str = "erosion";
    pub const DEPOSITION: &str = "deposition";
    pub const HARDNESS: &str = "hardness";
    pub const FLOW_DIRECTION: &str = "flow_direction";
    pub const FLOW_ACCUMULATION: &str = "flow_accumulation";
    pub const STREAM_ORDER: &str = "stream_order";
    pub const SPE_INCISION: &str = "spe_incision";
    pub const MATERIALS: &str = "materials";
    pub const BIOMES: &str = "biomes";
    pub const VEGETATION: &str = "vegetation";
    pub const SLOPE: &str = "slope";
    pub const CURVATURE: &str = "curvature";
    pub const LAND_MASK: &str = "land_mask";
    pub const SHORE_DISTANCE: &str = "shore_distance";
    pub const BATHYMETRY: &str = "bathymetry";
    pub const SHELF: &str = "shelf";
    pub const BEACH: &str = "beach";
    pub const REEF: &str = "reef";
    pub const MOUNTAIN_MASK: &str = "mountain_mask";
    /// Reference heights at Materials bake time (meters) for depth-aware strata \(K\).
    pub const STRATA_REFERENCE: &str = "strata_reference";
    // Phase H climate
    // Semantic authoring / coupled landscape workflow.
    pub const CONSTRAINT_TARGET: &str = "constraint_target";
    pub const CONSTRAINT_WEIGHT: &str = "constraint_weight";
    pub const CONSTRAINT_ERROR: &str = "constraint_error";
    pub const SCULPT_PROTECTION: &str = "sculpt_protection";
    pub const UPLIFT_RATE: &str = "uplift_rate";
    /// Smooth tectonic / base structure retained separately from the eroded surface.
    pub const TECTONIC_BASE: &str = "tectonic_base";
    /// Water discharge / drainage contribution (Q).
    pub const WATER_DISCHARGE: &str = "water_discharge";
    /// Legacy fine-sediment spelling. Accepted on ingest; emit [`SEDIMENT_THICKNESS`].
    pub const SEDIMENT_DEPTH: &str = "sediment_depth";
    pub const EDIT_REGION: &str = "edit_region";
    pub const REPAIR_REGION: &str = "repair_region";
    pub const DETAIL_MASK: &str = "geomorphic_detail";
    /// Fine-scale flow organisation from multi-scale amplification \[0,1\].
    pub const FINE_FLOW: &str = "fine_flow";
    /// Nested micro-channel / gully map from amplification \[0,1\].
    pub const MICRO_CHANNEL: &str = "micro_channel";
    /// Ridge-conditioned breakup intensity \[0,1\].
    pub const RIDGE_BREAKUP: &str = "ridge_breakup";
    /// Fine erosion / incision intensity from amplification \[0,1\].
    pub const FINE_EROSION: &str = "fine_erosion";
    pub const ROOT_COHESION: &str = "root_cohesion";

    pub const TEMPERATURE: &str = "temperature";
    pub const RAINFALL: &str = "rainfall";
    pub const HUMIDITY: &str = "humidity";
    pub const ARIDITY: &str = "aridity";
    pub const SNOW: &str = "snow";
    pub const SOIL_MOISTURE: &str = "soil_moisture";
    pub const WIND_EXPOSURE: &str = "wind_exposure";
    /// Prevailing / local wind direction (radians, atan2).
    pub const WIND_DIRECTION: &str = "wind_direction";
    /// Relative wind speed \[0,1\] (normalised).
    pub const WIND_SPEED: &str = "wind_speed";
    /// Aeolian sand flux \[0,1\] (normalised).
    pub const SAND_FLUX: &str = "sand_flux";
    /// Wind-shadow / sheltering \[0,1\].
    pub const SHELTERING: &str = "sheltering";
    // Phase J dual-height volumetric
    /// Ceiling height in meters for overhang / cave roof (equals floor outside region).
    pub const OVERHANG_CEILING: &str = "overhang_ceiling";
    /// \[0,1\] mask of where local volumetric applies.
    pub const OVERHANG_MASK: &str = "overhang_mask";
    // Matter simulation outputs (Water / Snow / Sand / Debris).
    pub const WATER_DEPTH: &str = "water_depth";
    pub const SNOW_DEPTH: &str = "snow_depth";
    pub const SAND_DEPTH: &str = "sand_depth";
    pub const DEBRIS_DEPTH: &str = "debris_depth";
    pub const FLOODPLAIN: &str = "floodplain";
    pub const DUNE_CREST: &str = "dune_crest";
    pub const SLIDE_PATH: &str = "slide_path";
    pub const INSTABILITY: &str = "instability";
    pub const TALUS_STABILITY: &str = "talus_stability";
    pub const MELTWATER: &str = "meltwater";
    pub const DRIFT: &str = "drift";
    pub const RIVER_CHANNEL: &str = "river_channel";
    pub const SAND_MATERIAL_MASK: &str = "sand_material_mask";
    pub const SNOW_MATERIAL_MASK: &str = "snow_material_mask";
    pub const SCATTER_CANDIDATES: &str = "scatter_candidates";
    /// Hydraulic-family channel mask (normalized flux threshold).
    pub const CHANNEL_MASK: &str = "channel_mask";
    /// Bedrock height under loose sediment (layered hydraulic / mass wasting).
    pub const BEDROCK_HEIGHT: &str = "bedrock_height";
    /// Legacy fine-sediment spelling. Accepted on ingest; emit [`SEDIMENT_THICKNESS`].
    pub const LOOSE_SEDIMENT: &str = "loose_sediment";
    /// Canonical loose sediment / alluvium thickness key.
    pub const SEDIMENT_THICKNESS: &str = "sediment_thickness";
    /// Persistent soil / regolith depth (metres).
    pub const SOIL_DEPTH: &str = "soil_depth";
    /// Discrete lithology unit ID (normalised \[0,1\] encoding).
    pub const LITHOLOGY: &str = "lithology";
    /// Water velocity magnitude proxy from hydraulic flux.
    pub const WATER_VELOCITY: &str = "water_velocity";

    /// Canonicalise legacy aux spellings without changing unrelated free-form keys.
    pub fn canonical(key: &str) -> &str {
        match key {
            SEDIMENT_DEPTH | LOOSE_SEDIMENT => SEDIMENT_THICKNESS,
            other => other,
        }
    }

    pub fn is_sediment_thickness_key(key: &str) -> bool {
        matches!(key, SEDIMENT_THICKNESS | SEDIMENT_DEPTH | LOOSE_SEDIMENT)
    }
}

/// Typed auxiliary fields produced by sims and analysis.
#[derive(Debug, Clone, Default)]
pub struct AuxMaps {
    pub wetness: Option<MaskField>,
    pub sediment: Option<MaskField>,
    pub erosion: Option<MaskField>,
    pub deposition: Option<MaskField>,
    pub hardness: Option<MaskField>,
    /// Bedrock elevation (m) under loose cover — shared geological state.
    pub bedrock_height: Option<MaskField>,
    /// Loose sediment / alluvium thickness (m).
    pub sediment_thickness: Option<MaskField>,
    /// Soil / regolith depth (m).
    pub soil_depth: Option<MaskField>,
    /// Discrete lithology unit IDs (normalised encoding).
    pub lithology: Option<MaskField>,
    pub flow_direction: Option<MaskField>,
    pub flow_accumulation: Option<MaskField>,
    /// Normalized Strahler-like stream order from SPE / river routing.
    pub stream_order: Option<MaskField>,
    /// Cumulative stream-power incision delta (meters, then normalized in overlays).
    pub spe_incision: Option<MaskField>,
    pub materials: Option<MaskField>,
    pub biomes: Option<MaskField>,
    pub vegetation: Option<MaskField>,
    /// Derived slope in \[0,1\] (degrees / 90). Lazily filled by [`Self::ensure_derived`].
    pub slope: Option<MaskField>,
    /// Derived mean curvature mapped to \[0,1\]. Lazily filled by [`Self::ensure_derived`].
    pub curvature: Option<MaskField>,
    /// Heightfield snapshot (as MaskField meters) when Materials baked strata.
    pub strata_reference: Option<MaskField>,
    /// Vertical material stack from the last Materials layer (surface → bedrock).
    pub strata: Option<Vec<crate::layer::Stratum>>,
    /// Bed attitude from the last Materials layer (tilted / folded / warped).
    pub bed_geometry: crate::layer::BedGeometry,
    // Phase H climate fields (normalized \[0,1\] unless noted).
    pub temperature: Option<MaskField>,
    pub rainfall: Option<MaskField>,
    pub humidity: Option<MaskField>,
    pub aridity: Option<MaskField>,
    pub snow: Option<MaskField>,
    pub soil_moisture: Option<MaskField>,
    pub wind_exposure: Option<MaskField>,
    /// Phase J: dual-height ceiling (meters) for overhang / local SDF roofs.
    pub overhang_ceiling: Option<MaskField>,
    /// Phase J: \[0,1\] region where volumetric dual-height applies.
    pub overhang_mask: Option<MaskField>,
    /// Escape hatch for unnamed keys during migration.
    pub extras: HashMap<String, MaskField>,
}

impl AuxMaps {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Insert by well-known or free-form string key.
    pub fn insert(&mut self, key: impl Into<String>, field: MaskField) {
        let key = key.into();
        let canonical = keys::canonical(&key);
        match canonical {
            keys::WETNESS => self.wetness = Some(field),
            keys::SEDIMENT => self.sediment = Some(field),
            keys::EROSION => self.erosion = Some(field),
            keys::DEPOSITION => self.deposition = Some(field),
            keys::HARDNESS => self.hardness = Some(field),
            keys::BEDROCK_HEIGHT => self.bedrock_height = Some(field),
            keys::SEDIMENT_THICKNESS => self.sediment_thickness = Some(field),
            keys::SOIL_DEPTH => self.soil_depth = Some(field),
            keys::LITHOLOGY => self.lithology = Some(field),
            keys::FLOW_DIRECTION => self.flow_direction = Some(field),
            keys::FLOW_ACCUMULATION => self.flow_accumulation = Some(field),
            keys::STREAM_ORDER => self.stream_order = Some(field),
            keys::SPE_INCISION => self.spe_incision = Some(field),
            keys::MATERIALS => self.materials = Some(field),
            keys::BIOMES => self.biomes = Some(field),
            keys::VEGETATION => self.vegetation = Some(field),
            keys::SLOPE => self.slope = Some(field),
            keys::CURVATURE => self.curvature = Some(field),
            keys::STRATA_REFERENCE => self.strata_reference = Some(field),
            keys::TEMPERATURE => self.temperature = Some(field),
            keys::RAINFALL => self.rainfall = Some(field),
            keys::HUMIDITY => self.humidity = Some(field),
            keys::ARIDITY => self.aridity = Some(field),
            keys::SNOW => self.snow = Some(field),
            keys::SOIL_MOISTURE => self.soil_moisture = Some(field),
            keys::WIND_EXPOSURE => self.wind_exposure = Some(field),
            keys::OVERHANG_CEILING => self.overhang_ceiling = Some(field),
            keys::OVERHANG_MASK => self.overhang_mask = Some(field),
            _ => {
                self.extras.insert(canonical.to_string(), field);
            }
        }
    }

    pub fn get(&self, key: &str) -> Option<&MaskField> {
        match keys::canonical(key) {
            keys::WETNESS => self.wetness.as_ref(),
            keys::SEDIMENT => self.sediment.as_ref(),
            keys::EROSION => self.erosion.as_ref(),
            keys::DEPOSITION => self.deposition.as_ref(),
            keys::HARDNESS => self.hardness.as_ref(),
            keys::BEDROCK_HEIGHT => self.bedrock_height.as_ref(),
            keys::SEDIMENT_THICKNESS => self.sediment_thickness.as_ref(),
            keys::SOIL_DEPTH => self.soil_depth.as_ref(),
            keys::LITHOLOGY => self.lithology.as_ref(),
            keys::FLOW_DIRECTION => self.flow_direction.as_ref(),
            keys::FLOW_ACCUMULATION => self.flow_accumulation.as_ref(),
            keys::STREAM_ORDER => self.stream_order.as_ref(),
            keys::SPE_INCISION => self.spe_incision.as_ref(),
            keys::MATERIALS => self.materials.as_ref(),
            keys::BIOMES => self.biomes.as_ref(),
            keys::VEGETATION => self.vegetation.as_ref(),
            keys::SLOPE => self.slope.as_ref(),
            keys::CURVATURE => self.curvature.as_ref(),
            keys::STRATA_REFERENCE => self.strata_reference.as_ref(),
            keys::TEMPERATURE => self.temperature.as_ref(),
            keys::RAINFALL => self.rainfall.as_ref(),
            keys::HUMIDITY => self.humidity.as_ref(),
            keys::ARIDITY => self.aridity.as_ref(),
            keys::SNOW => self.snow.as_ref(),
            keys::SOIL_MOISTURE => self.soil_moisture.as_ref(),
            keys::WIND_EXPOSURE => self.wind_exposure.as_ref(),
            keys::OVERHANG_CEILING => self.overhang_ceiling.as_ref(),
            keys::OVERHANG_MASK => self.overhang_mask.as_ref(),
            other => self.extras.get(other),
        }
    }

    pub fn extend(&mut self, other: &AuxMaps) {
        for (k, v) in other.to_hashmap() {
            self.insert(k, v);
        }
        if other.strata.is_some() {
            self.strata = other.strata.clone();
        }
    }

    pub fn extend_hashmap(&mut self, map: &HashMap<String, MaskField>) {
        for (k, v) in map {
            if !keys::is_sediment_thickness_key(k) {
                self.insert(k.clone(), v.clone());
            }
        }
        // HashMap iteration order is intentionally irrelevant. Prefer the canonical
        // representation, then the newer simulation alias, then the oldest alias.
        if let Some(field) = map
            .get(keys::SEDIMENT_THICKNESS)
            .or_else(|| map.get(keys::SEDIMENT_DEPTH))
            .or_else(|| map.get(keys::LOOSE_SEDIMENT))
        {
            self.sediment_thickness = Some(field.clone());
        }
    }

    /// Restore typed maps from a HashMap while preserving an existing strata stack
    /// (strata cannot round-trip through MaskField storage).
    pub fn from_hashmap_preserving_strata(
        map: &HashMap<String, MaskField>,
        strata: Option<Vec<crate::layer::Stratum>>,
    ) -> Self {
        let mut aux = Self::from_hashmap(map);
        if strata.is_some() {
            aux.strata = strata;
        }
        aux
    }

    /// Flatten typed fields into the legacy string HashMap.
    pub fn to_hashmap(&self) -> HashMap<String, MaskField> {
        let mut map = HashMap::new();
        let push = |map: &mut HashMap<String, MaskField>, key: &str, field: &Option<MaskField>| {
            if let Some(f) = field {
                map.insert(key.to_string(), f.clone());
            }
        };
        push(&mut map, keys::WETNESS, &self.wetness);
        push(&mut map, keys::SEDIMENT, &self.sediment);
        push(&mut map, keys::EROSION, &self.erosion);
        push(&mut map, keys::DEPOSITION, &self.deposition);
        push(&mut map, keys::HARDNESS, &self.hardness);
        push(&mut map, keys::BEDROCK_HEIGHT, &self.bedrock_height);
        push(&mut map, keys::SEDIMENT_THICKNESS, &self.sediment_thickness);
        push(&mut map, keys::SOIL_DEPTH, &self.soil_depth);
        push(&mut map, keys::LITHOLOGY, &self.lithology);
        push(&mut map, keys::FLOW_DIRECTION, &self.flow_direction);
        push(&mut map, keys::FLOW_ACCUMULATION, &self.flow_accumulation);
        push(&mut map, keys::STREAM_ORDER, &self.stream_order);
        push(&mut map, keys::SPE_INCISION, &self.spe_incision);
        push(&mut map, keys::MATERIALS, &self.materials);
        push(&mut map, keys::BIOMES, &self.biomes);
        push(&mut map, keys::VEGETATION, &self.vegetation);
        push(&mut map, keys::SLOPE, &self.slope);
        push(&mut map, keys::CURVATURE, &self.curvature);
        push(&mut map, keys::STRATA_REFERENCE, &self.strata_reference);
        push(&mut map, keys::TEMPERATURE, &self.temperature);
        push(&mut map, keys::RAINFALL, &self.rainfall);
        push(&mut map, keys::HUMIDITY, &self.humidity);
        push(&mut map, keys::ARIDITY, &self.aridity);
        push(&mut map, keys::SNOW, &self.snow);
        push(&mut map, keys::SOIL_MOISTURE, &self.soil_moisture);
        push(&mut map, keys::WIND_EXPOSURE, &self.wind_exposure);
        push(&mut map, keys::OVERHANG_CEILING, &self.overhang_ceiling);
        push(&mut map, keys::OVERHANG_MASK, &self.overhang_mask);
        for (k, v) in &self.extras {
            map.insert(k.clone(), v.clone());
        }
        map
    }

    pub fn from_hashmap(map: &HashMap<String, MaskField>) -> Self {
        let mut aux = Self::new();
        aux.extend_hashmap(map);
        aux
    }

    /// Ensure slope / curvature caches exist for `hf`.
    ///
    /// This is the supported helper for processors that hold `AuxMaps` and need
    /// slope or curvature derived from the current heightfield.
    pub fn ensure_derived(&mut self, hf: &Heightfield) {
        if self.slope.is_none() {
            self.slope = Some(slope_degrees(hf));
        }
        if self.curvature.is_none() {
            self.curvature = Some(curvature(hf));
        }
    }

    /// Force recompute of slope / curvature (e.g. after height changed in-place).
    pub fn refresh_derived(&mut self, hf: &Heightfield) {
        self.slope = Some(slope_degrees(hf));
        self.curvature = Some(curvature(hf));
    }

    /// Hardness in \[0,1\]; missing map falls back to a constant field.
    pub fn hardness_or_constant(
        &self,
        metrics: crate::heightfield::HeightfieldMetrics,
        k: f32,
    ) -> MaskField {
        self.hardness
            .clone()
            .unwrap_or_else(|| MaskField::filled(metrics, k.clamp(0.0, 1.0)))
    }
}

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

/// Hardness at eroded depth \(d\) (meters below the Materials reference surface).
///
/// Walks [`crate::layer::Stratum`] from the surface downward. Beyond the stack,
/// returns `default_hardness`.
pub fn hardness_at_strata_depth(
    strata: &[crate::layer::Stratum],
    depth: f32,
    default_hardness: f32,
) -> f32 {
    if strata.is_empty() {
        return default_hardness.clamp(0.0, 1.0);
    }
    let mut remaining = depth.max(0.0);
    for s in strata {
        let t = s.thickness.max(0.0);
        if remaining <= t || t >= 1.0e5 {
            return s.hardness.clamp(0.0, 1.0);
        }
        remaining -= t;
    }
    strata
        .last()
        .map(|s| s.hardness.clamp(0.0, 1.0))
        .unwrap_or_else(|| default_hardness.clamp(0.0, 1.0))
}

/// Hydraulic erodibility at depth (defaults to `1 - K` for legacy strata).
pub fn erodibility_at_strata_depth(
    strata: &[crate::layer::Stratum],
    depth: f32,
    default_hardness: f32,
) -> f32 {
    if strata.is_empty() {
        return (1.0 - default_hardness).clamp(0.0, 1.0);
    }
    let mut remaining = depth.max(0.0);
    for s in strata {
        let t = s.thickness.max(0.0);
        if remaining <= t || t >= 1.0e5 {
            return s.effective_erodibility();
        }
        remaining -= t;
    }
    strata
        .last()
        .map(|s| s.effective_erodibility())
        .unwrap_or_else(|| (1.0 - default_hardness).clamp(0.0, 1.0))
}

/// Thermal material stability at depth ∈ \[0,1\] (1 = holds steep faces).
pub fn stability_at_strata_depth(
    strata: &[crate::layer::Stratum],
    depth: f32,
    default_hardness: f32,
) -> f32 {
    if strata.is_empty() {
        return default_hardness.clamp(0.0, 1.0);
    }
    let mut remaining = depth.max(0.0);
    for s in strata {
        let t = s.thickness.max(0.0);
        if remaining <= t || t >= 1.0e5 {
            return s.material_stability();
        }
        remaining -= t;
    }
    strata
        .last()
        .map(|s| s.material_stability())
        .unwrap_or_else(|| default_hardness.clamp(0.0, 1.0))
}

/// Surface material ID (encoded /16) at depth \(d\).
pub fn material_id_at_strata_depth(strata: &[crate::layer::Stratum], depth: f32) -> f32 {
    if strata.is_empty() {
        return 0.0;
    }
    let mut remaining = depth.max(0.0);
    for s in strata {
        let t = s.thickness.max(0.0);
        if remaining <= t || t >= 1.0e5 {
            return (s.id as f32 / 16.0).clamp(0.0, 1.0);
        }
        remaining -= t;
    }
    strata
        .last()
        .map(|s| (s.id as f32 / 16.0).clamp(0.0, 1.0))
        .unwrap_or(0.0)
}

/// Bake hardness from reference vs current heights using a strata stack.
pub fn bake_hardness_from_strata(
    reference: &MaskField,
    current: &Heightfield,
    strata: &[crate::layer::Stratum],
    default_hardness: f32,
) -> MaskField {
    bake_hardness_from_strata_ex(
        reference,
        current,
        strata,
        default_hardness,
        &crate::layer::BedGeometry::Horizontal,
    )
}

/// Bake hardness with bed geometry warp (tilted / folded / warped beds).
pub fn bake_hardness_from_strata_ex(
    reference: &MaskField,
    current: &Heightfield,
    strata: &[crate::layer::Stratum],
    default_hardness: f32,
    geom: &crate::layer::BedGeometry,
) -> MaskField {
    let mut out = MaskField::filled(current.metrics, default_hardness.clamp(0.0, 1.0));
    for j in 0..current.metrics.height {
        for i in 0..current.metrics.width {
            let x = current.metrics.world_x(i);
            let z = current.metrics.world_z(j);
            let depth = crate::generators::geology::strata_depth_m(
                reference.get(i, j),
                current.get(i, j),
                x,
                z,
                geom,
            );
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
    rules: &[crate::layer::MaterialRule],
    default_hardness: f32,
) -> MaskField {
    bake_hardness_from_materials_ex(materials, rules, &[], default_hardness)
}

/// Materials → \(K\) bake with optional strata fallback for unmatched IDs.
pub fn bake_hardness_from_materials_ex(
    materials: &MaskField,
    rules: &[crate::layer::MaterialRule],
    strata: &[crate::layer::Stratum],
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
