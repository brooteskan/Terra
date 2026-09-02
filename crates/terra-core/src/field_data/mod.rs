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

use crate::heightfield::Heightfield;
use crate::mask_field::MaskField;
use crate::spatial_kernels::{curvature, slope_degrees};
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
    pub strata: Option<Vec<crate::geology::Stratum>>,
    /// Bed attitude from the last Materials layer (tilted / folded / warped).
    pub bed_geometry: crate::geology::BedGeometry,
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
        strata: Option<Vec<crate::geology::Stratum>>,
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
