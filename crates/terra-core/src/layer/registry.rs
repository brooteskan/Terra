//! Layer type registry / factory — UI discovers types via metadata.
//!
//! [`LayerKind`] is the single source of truth for core type identity (stable id,
//! display name, and workflow stage). This registry derives those fields from its
//! factories and adds discoverable metadata such as icons and capabilities. UI
//! catalogs (`terra-app` tool rail / add-layer menu) should derive entries from
//! [`LayerTypeRegistry`] rather than maintaining parallel hand-synced lists.

use super::metadata::{
    AccentCategory, ChannelRequirements, LayerCapabilities, LayerTypeMeta, MaskCompatibility,
};
use super::operation::FieldContract;
use super::workflow::WorkflowStage;
use super::{
    BiomesParams, BlurParams, CanyonParams, CoastalParams, DomainWarpParams, DuneParams,
    EcosystemFeedbackParams, EffectFilterParams, FbmParams, FlatParams, FluidSimParams,
    GeomorphicDetailParams, GradientReconstructParams, HydraulicErosionParams,
    HydrologyRepairParams, ImportHeightmapParams, IslandParams, LandscapeEvolutionParams, Layer,
    LayerKind, LocalSdfParams, MaterialsParams, MesaParams, MountainParams,
    MultiScaleAmplifyParams, NoiseParams, OverhangStampParams, PathParams, PlateauParams,
    PolygonHeightParams, ProceduralShapeParams, RampParams, RiverCarveParams, RiverNetworkParams,
    SandSimParams, SculptParams, SculptStrokeParams, Stamp2dParams, Stamp3dParams,
    StreamPowerParams, TerraceParams, TerrainConstraintParams, ThermalErosionParams, UpliftParams,
    VegetationParams, VolcanoParams, VoronoiParams, WorleyParams,
};
use std::collections::HashMap;

/// Authoring registry: metadata + default factories for layer kinds.
#[derive(Debug, Clone)]
pub struct LayerTypeRegistry {
    entries: Vec<LayerTypeMeta>,
    /// Parallel default-kind factories (same order as entries).
    factories: Vec<fn() -> LayerKind>,
    by_id: HashMap<&'static str, usize>,
}

impl Default for LayerTypeRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

impl LayerTypeRegistry {
    pub fn builtin() -> Self {
        let mut reg = Self {
            entries: Vec::new(),
            factories: Vec::new(),
            by_id: HashMap::new(),
        };
        reg.register_all();
        reg
    }

    fn push(&mut self, meta: LayerTypeMeta, factory: fn() -> LayerKind) {
        let idx = self.entries.len();
        self.by_id.insert(meta.type_id, idx);
        self.entries.push(meta);
        self.factories.push(factory);
    }

    pub fn get(&self, type_id: &str) -> Option<&LayerTypeMeta> {
        self.by_id.get(type_id).map(|&i| &self.entries[i])
    }

    pub fn all(&self) -> &[LayerTypeMeta] {
        &self.entries
    }

    pub fn creatable(&self) -> impl Iterator<Item = &LayerTypeMeta> {
        self.entries
            .iter()
            .filter(|m| m.capabilities.user_creatable)
    }

    pub fn by_stage(&self, stage: WorkflowStage) -> impl Iterator<Item = &LayerTypeMeta> {
        self.entries
            .iter()
            .filter(move |m| m.workflow_stage == stage)
    }

    /// Create a new layer with default params and the type's display name.
    pub fn create(&self, type_id: &str) -> Option<Layer> {
        let idx = *self.by_id.get(type_id)?;
        let meta = &self.entries[idx];
        let kind = (self.factories[idx])();
        Some(Layer::new(meta.display_name, kind))
    }

    /// Create with an explicit display name.
    pub fn create_named(&self, type_id: &str, name: impl Into<String>) -> Option<Layer> {
        let idx = *self.by_id.get(type_id)?;
        let kind = (self.factories[idx])();
        Some(Layer::new(name, kind))
    }

    /// Metadata for an existing kind (lookup by `LayerKind::type_id`).
    pub fn meta_for_kind(&self, kind: &LayerKind) -> Option<&LayerTypeMeta> {
        self.get(kind.type_id())
    }

    /// Icon key for a type id (for UI catalogs deriving from this registry).
    pub fn icon_key(&self, type_id: &str) -> Option<&'static str> {
        self.get(type_id).map(|m| m.icon_key)
    }

    /// Display name for a type id.
    pub fn display_name(&self, type_id: &str) -> Option<&'static str> {
        self.get(type_id).map(|m| m.display_name)
    }

    fn register_all(&mut self) {
        // Helpers shrink boilerplate.
        macro_rules! entry {
            (
                $icon:literal, $desc:literal, $factory:expr, $creatable:expr,
                $sections:expr, $next:expr
            ) => {{
                let factory: fn() -> LayerKind = $factory;
                let kind = factory();
                let contract = FieldContract::from_kind(&kind);
                let op = kind.category();
                let stage = kind.workflow_stage();
                self.push(
                    LayerTypeMeta {
                        type_id: kind.type_id(),
                        display_name: kind.type_display_name(),
                        workflow_stage: stage,
                        icon_key: $icon,
                        accent: AccentCategory::from_workflow(stage),
                        description: $desc,
                        operation_category: op,
                        channels: ChannelRequirements::from_contract(&contract),
                        mask_compatibility: MaskCompatibility::SupportsMasks,
                        capabilities: LayerCapabilities {
                            supports_thumbnails: true,
                            // Compatibility-only kinds may be hidden from create
                            // surfaces while remaining fully editable in projects
                            // that already contain them.
                            supports_duplication: !kind.is_sculpt_base(),
                            can_reorder: !kind.is_sculpt_base(),
                            can_contain_children: false,
                            user_creatable: $creatable,
                        },
                        inspector_sections: $sections,
                        suggested_next: $next,
                    },
                    factory,
                );
            }};
        }

        entry!(
            "pencil",
            "Paintable foundation height buffer.",
            || LayerKind::SculptBase(SculptParams::default()),
            false,
            &["shape"],
            &[] as &[&str]
        );
        entry!(
            "image",
            "Import an external heightmap.",
            || LayerKind::ImportHeightmap(ImportHeightmapParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "pencil",
            "Resolution-independent editable sculpt strokes.",
            || LayerKind::SculptStrokes(SculptStrokeParams::default()),
            true,
            &["shape"],
            &["terrain_constraints"]
        );
        entry!(
            "activity",
            "Author elevations, ridges, valleys, rivers, coasts and protected shapes.",
            || LayerKind::TerrainConstraints(TerrainConstraintParams::default()),
            true,
            &["shape"],
            &["gradient_reconstruct"]
        );
        entry!(
            "activity",
            "Screened-Poisson reconstruction for seamless authored edits.",
            || LayerKind::GradientReconstruct(GradientReconstructParams::default()),
            true,
            &["shape"],
            &["landscape_evolution"]
        );

        entry!(
            "mountain",
            "Constant height contribution.",
            || LayerKind::Flat(FlatParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "mountain",
            "Linear height ramp.",
            || LayerKind::Ramp(RampParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "mountain",
            "Value noise height contribution.",
            || LayerKind::NoiseValue(NoiseParams::default()),
            true,
            &["noise"],
            &[]
        );
        entry!(
            "mountain",
            "Perlin noise height contribution.",
            || LayerKind::NoisePerlin(NoiseParams::default()),
            true,
            &["noise"],
            &[]
        );
        entry!(
            "mountain",
            "OpenSimplex noise height contribution.",
            || LayerKind::NoiseOpenSimplex(NoiseParams::default()),
            true,
            &["noise"],
            &[]
        );
        entry!(
            "mountain",
            "Cellular / Worley noise.",
            || LayerKind::NoiseWorley(WorleyParams::default()),
            true,
            &["noise"],
            &[]
        );
        entry!(
            "mountain",
            "Fractal Brownian motion noise.",
            || LayerKind::Fbm(FbmParams::default()),
            true,
            &["noise"],
            &["thermal_erosion"]
        );
        entry!(
            "mountain",
            "Ridged multifractal detail.",
            || LayerKind::Ridged(FbmParams::default()),
            true,
            &["noise"],
            &[]
        );
        entry!(
            "mountain",
            "Warped noise domain.",
            || LayerKind::DomainWarp(DomainWarpParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "mountain",
            "Ridged mountain range.",
            || LayerKind::Mountains(MountainParams::default()),
            true,
            &["shape"],
            &["hydraulic_erosion", "thermal_erosion"]
        );
        entry!(
            "mountain",
            "Volcanic cone with optional crater.",
            || LayerKind::Volcano(VolcanoParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "mountain",
            "Hard-cap mesa / butte.",
            || LayerKind::Mesa(MesaParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "waves",
            "Closed island landmass with coastal zones and bathymetry.",
            || LayerKind::Island(IslandParams::default()),
            true,
            &["shape"],
            &["coastal", "hydraulic_erosion"]
        );
        entry!(
            "mountain",
            "Large-scale ridge / corridor uplift.",
            || LayerKind::Uplift(UpliftParams::default()),
            false,
            &["shape"],
            &["landscape_evolution"]
        );
        entry!(
            "mountain",
            "Wind dune fields.",
            || LayerKind::Dunes(DuneParams::default()),
            true,
            &["shape"],
            &["sand_simulation"]
        );
        entry!(
            "mountain",
            "Canyon network.",
            || LayerKind::Canyons(CanyonParams::default()),
            true,
            &["shape"],
            &["river_carve"]
        );
        entry!(
            "mountain",
            "Tableland plateau.",
            || LayerKind::Plateau(PlateauParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "mountain",
            "Terraced height quantisation.",
            || LayerKind::Terrace(TerraceParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "mountain",
            "Cellular region heights.",
            || LayerKind::VoronoiRegions(VoronoiParams::default()),
            true,
            &["noise"],
            &[]
        );
        entry!(
            "blend",
            "Gaussian-like height blur.",
            || LayerKind::Blur(BlurParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "sparkles",
            "WC-style effect filters (smooth, distortion, shore, strata, …).",
            || LayerKind::EffectFilter(EffectFilterParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "activity",
            "Drawn spline path (road / ridge / valley).",
            || LayerKind::Path(PathParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "mountain",
            "Procedural landscape shape with a swappable generator.",
            || LayerKind::ProceduralShape(ProceduralShapeParams::default()),
            true,
            &["shape"],
            &["hydraulic_erosion", "thermal_erosion"]
        );
        entry!(
            "image",
            "2D heightmap stamp.",
            || LayerKind::Stamp2d(Stamp2dParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "box",
            "3D mesh stamp — samples an OBJ or height image into the heightfield.",
            || LayerKind::Stamp3d(Stamp3dParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "activity",
            "Closed polygon raise / carve.",
            || LayerKind::PolygonHeight(PolygonHeightParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "box",
            "Cliff undercut / shelf via dual-height.",
            || LayerKind::OverhangStamp(OverhangStampParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "box",
            "Local analytic SDF cave pocket.",
            || LayerKind::LocalSdf(LocalSdfParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "mountain",
            "Drainage-conditioned multi-scale amplification (meso ridges / micro gullies).",
            || LayerKind::GeomorphicDetail(GeomorphicDetailParams::default()),
            true,
            &["shape", "amplify"],
            &[]
        );

        // Simulations
        entry!(
            "activity",
            "Talus / thermal weathering with layered debris.",
            || LayerKind::ThermalErosion(ThermalErosionParams::default()),
            true,
            &["erosion"],
            &["debris_flow"]
        );
        entry!(
            "activity",
            "Jain et al. 2024 debris-flow erosion, scars and deposit cones.",
            || LayerKind::DebrisFlow(crate::layer::DebrisFlowParams::default()),
            true,
            &["erosion"],
            &["thermal_erosion"]
        );
        entry!(
            "droplets",
            "Particle / outflow hydraulic erosion.",
            || LayerKind::HydraulicErosion(HydraulicErosionParams::default()),
            true,
            &["erosion"],
            &["debris_flow", "thermal_erosion"]
        );
        entry!(
            "waves",
            "Stream-power incision (fluvial SPE).",
            || LayerKind::StreamPowerErosion(StreamPowerParams::default()),
            true,
            &["erosion"],
            &["hydraulic_erosion"]
        );
        entry!(
            "activity",
            "Wavelength-aware multi-scale amplify.",
            || LayerKind::MultiScaleAmplify(MultiScaleAmplifyParams::default()),
            true,
            &["erosion"],
            &[]
        );
        entry!(
            "waves",
            "Carve river channels into height.",
            || LayerKind::RiverCarve(RiverCarveParams::default()),
            true,
            &["erosion"],
            &[]
        );
        entry!(
            "waves",
            "Interactive river network (springs / auto-flow).",
            || LayerKind::RiverNetwork(RiverNetworkParams::default()),
            true,
            &["erosion"],
            &["river_carve"]
        );
        entry!(
            "waves",
            "Coastal erosion and shelf shaping.",
            || LayerKind::Coastal(CoastalParams::default()),
            true,
            &["erosion"],
            &[]
        );
        entry!(
            "activity",
            "Coupled uplift, SPE incision, hillslope diffusion and sediment.",
            || LayerKind::LandscapeEvolution(LandscapeEvolutionParams::default()),
            true,
            &["erosion"],
            &["hydrology_repair"]
        );
        entry!(
            "droplets",
            "Re-establish connected drainage after local edits.",
            || LayerKind::HydrologyRepair(HydrologyRepairParams::default()),
            true,
            &["erosion"],
            &[]
        );
        entry!(
            "sparkles",
            "Bounded vegetation/root/soil feedback into terrain evolution.",
            || LayerKind::EcosystemFeedback(EcosystemFeedbackParams::default()),
            true,
            &["erosion"],
            &[]
        );

        // Surface / scatter / objects
        entry!(
            "layers",
            "Climate / biome classification filter (not a biome container).",
            || LayerKind::Biomes(BiomesParams::default()),
            true,
            &["surface"],
            &["materials", "vegetation"]
        );
        entry!(
            "blend",
            "Surface material / hardness rules.",
            || LayerKind::Materials(MaterialsParams::default()),
            true,
            &["surface"],
            &["vegetation"]
        );
        entry!(
            "sparkles",
            "Vegetation density / scatter weights.",
            || LayerKind::Vegetation(VegetationParams::default()),
            true,
            &["surface"],
            &[]
        );
        entry!(
            "mountain",
            "Granular sand local simulation.",
            || LayerKind::SandSimulation(SandSimParams::default()),
            true,
            &["shape"],
            &[]
        );
        entry!(
            "droplets",
            "Fluid / lake local simulation.",
            || LayerKind::FluidSimulation(FluidSimParams::default()),
            true,
            &["shape"],
            &[]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_covers_unique_ids() {
        let reg = LayerTypeRegistry::builtin();
        let mut seen = std::collections::HashSet::new();
        for m in reg.all() {
            assert!(seen.insert(m.type_id), "duplicate type_id {}", m.type_id);
        }
        assert_eq!(reg.entries.len(), reg.factories.len());
    }

    #[test]
    fn create_mountain() {
        let reg = LayerTypeRegistry::builtin();
        let layer = reg.create("mountain").expect("mountain");
        assert!(matches!(layer.kind, LayerKind::Mountains(_)));
        assert_eq!(layer.kind.type_id(), "mountain");
        assert_eq!(layer.common.name, "Mountains");
    }

    #[test]
    fn sculpt_base_not_creatable() {
        let reg = LayerTypeRegistry::builtin();
        let meta = reg.get("sculpt_base").unwrap();
        assert!(!meta.capabilities.user_creatable);
        assert!(reg.create("sculpt_base").is_some()); // factory still works
    }

    #[test]
    fn compatibility_uplift_is_editable_but_not_directly_creatable() {
        let reg = LayerTypeRegistry::builtin();
        let meta = reg.get("uplift").expect("uplift metadata");
        assert!(!meta.capabilities.user_creatable);
        assert!(meta.capabilities.supports_duplication);
        assert!(meta.capabilities.can_reorder);
        assert_eq!(meta.suggested_next, &["landscape_evolution"]);
        assert!(reg.create("uplift").is_some());
    }

    #[test]
    fn stage_filter() {
        let reg = LayerTypeRegistry::builtin();
        let sims: Vec<_> = reg.by_stage(WorkflowStage::Simulation).collect();
        assert!(sims.iter().any(|m| m.type_id == "hydraulic_erosion"));
    }

    #[test]
    fn builtin_metadata_matches_every_factory_kind() {
        let reg = LayerTypeRegistry::builtin();

        for (meta, factory) in reg.entries.iter().zip(&reg.factories) {
            let kind = factory();
            assert_eq!(meta.type_id, kind.type_id(), "type id for {}", meta.type_id);
            assert_eq!(
                meta.display_name,
                kind.type_display_name(),
                "display name for {}",
                meta.type_id
            );
            assert_eq!(
                meta.workflow_stage,
                kind.workflow_stage(),
                "workflow stage for {}",
                meta.type_id
            );
            assert!(std::ptr::eq(
                meta,
                reg.meta_for_kind(&kind)
                    .expect("factory kind is registered")
            ));

            let created = reg.create(meta.type_id).expect("registered type creates");
            assert_eq!(created.kind.type_id(), meta.type_id);
        }

        assert_eq!(
            reg.get("stamp_2d").unwrap().workflow_stage,
            WorkflowStage::Foundation
        );
        assert_eq!(
            reg.get("stamp_3d").unwrap().workflow_stage,
            WorkflowStage::Foundation
        );
        assert_eq!(
            reg.get("coastal").unwrap().workflow_stage,
            WorkflowStage::Simulation
        );
    }
}
