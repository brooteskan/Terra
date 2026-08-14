//! All layer parameter kinds. Extended across phases; serde-stable via enum.
//!
//! Param structs are split by family under this module; [`LayerKind`] stays here
//! so serde variant names/tags remain stable.

mod erosion;
mod filters;
mod hydro;
mod materials;
mod noise;
mod sculpt;
mod shapes;
mod vegetation;

pub use erosion::*;
pub use filters::*;
pub use hydro::*;
pub use materials::*;
pub use noise::*;
pub use sculpt::*;
pub use shapes::*;
pub use vegetation::*;

use super::BlendMode;
pub use crate::authoring::{
    EcosystemFeedbackParams, GeomorphicDetailParams, GradientReconstructParams,
    HydrologyRepairParams, SculptPoint, SculptStroke, SculptStrokeKind, SculptStrokeParams,
    TerrainConstraint, TerrainConstraintKind, TerrainConstraintParams,
};
pub use crate::landscape_evolution::{
    BoundaryMode, EvolutionSolverMode, LandscapeEvolutionParams, UpliftMode,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerKind {
    // Artist foundation — painted height buffer (always bottom in default docs)
    SculptBase(SculptParams),
    /// Resolution-independent, editable sculpt gestures.
    SculptStrokes(SculptStrokeParams),
    /// Explicit elevation/ridge/valley/river/coast/protection constraints.
    TerrainConstraints(TerrainConstraintParams),
    /// Screened-Poisson reconstruction for seamless authored edits.
    GradientReconstruct(GradientReconstructParams),
    /// Coupled uplift, stream-power incision, hillslope diffusion and sediment.
    LandscapeEvolution(LandscapeEvolutionParams),
    /// Re-establish connected drainage after local edits.
    HydrologyRepair(HydrologyRepairParams),
    /// Add drainage- and slope-aligned geomorphic structure.
    GeomorphicDetail(GeomorphicDetailParams),
    /// Bounded vegetation/root/soil feedback into terrain evolution.
    EcosystemFeedback(EcosystemFeedbackParams),

    // Phase 1
    Flat(FlatParams),
    Ramp(RampParams),
    NoiseValue(NoiseParams),
    // Phase 2
    NoisePerlin(NoiseParams),
    NoiseOpenSimplex(NoiseParams),
    NoiseWorley(WorleyParams),
    Fbm(FbmParams),
    Ridged(FbmParams),
    DomainWarp(DomainWarpParams),
    Terrace(TerraceParams),
    Plateau(PlateauParams),
    /// Hard-cap tableland with steep walls and soft talus (mesa / butte).
    Mesa(MesaParams),
    /// Coast-aware oceanic landmass with bathymetry, shelf, beach, reef and uplands.
    Island(IslandParams),
    Mountains(MountainParams),
    /// Radial volcanic cone with optional crater (WC Landscape Volcano intent).
    Volcano(VolcanoParams),
    /// Large-scale ridge/corridor uplift (Schott-inspired authoring; not full SPE).
    Uplift(UpliftParams),
    Dunes(DuneParams),
    Canyons(CanyonParams),
    VoronoiRegions(VoronoiParams),
    ImportHeightmap(ImportHeightmapParams),
    // Phase 5–7 simulations / filters
    ThermalErosion(ThermalErosionParams),
    HydraulicErosion(HydraulicErosionParams),
    /// Debris-flow erosion (Jain, Beneš & Cordonnier 2024) for steep / low-drainage relief.
    DebrisFlow(DebrisFlowParams),
    /// Stream-power incision (fluvial SPE) driven by Priority-Flood + D8/D∞ drainage.
    StreamPowerErosion(StreamPowerParams),
    /// Wavelength-aware multi-scale amplify (Schott 2024 adapted onto `SimLevel`).
    MultiScaleAmplify(MultiScaleAmplifyParams),
    RiverCarve(RiverCarveParams),
    Blur(BlurParams),
    Coastal(CoastalParams),
    /// WC-style effect filters (Smooth / Distortion / Spike Removal / Shore / Strata / Crater / Denoise).
    EffectFilter(EffectFilterParams),
    // Phase 8
    Materials(MaterialsParams),
    Biomes(BiomesParams),
    Vegetation(VegetationParams),
    // Phase J — optional local volumetric (dual-height / local SDF); heightfield remains default
    /// Cliff undercut / shelf via dual-height floor+ceiling (opt-in).
    OverhangStamp(OverhangStampParams),
    /// Local analytic SDF cave pocket projected to dual-height (opt-in).
    LocalSdf(LocalSdfParams),
    /// Drawn spline path (road / ridge / valley) — Wave 1 Phase C.
    Path(PathParams),
    /// Interactive river network (springs / auto-flow) — Wave 1 Phase C.
    RiverNetwork(RiverNetworkParams),
    /// Granular sand Local Sim — Wave 1 Phase D.
    SandSimulation(SandSimParams),
    /// Fluid / lake Local Sim — Wave 1 Phase D.
    FluidSimulation(FluidSimParams),

    // —— WC-style Shape Layer types (Terrain tools) ————————————————
    /// Procedural landscape shape with a swappable generator type.
    ProceduralShape(ProceduralShapeParams),
    /// 2D heightmap stamp.
    Stamp2d(Stamp2dParams),
    /// 3D mesh stamp (OBJ / height image → height contribution).
    Stamp3d(Stamp3dParams),
    /// Closed polygon raise / carve.
    PolygonHeight(PolygonHeightParams),
}

impl LayerKind {
    pub fn is_sculpt_base(&self) -> bool {
        matches!(self, LayerKind::SculptBase(_))
    }

    /// Artist-workflow evaluation stage for dependency ordering.
    pub fn eval_stage(&self) -> crate::landscape_blueprint::EvalStage {
        use crate::landscape_blueprint::EvalStage;
        match self {
            LayerKind::SculptBase(_)
            | LayerKind::Flat(_)
            | LayerKind::ImportHeightmap(_)
            | LayerKind::SculptStrokes(_)
            | LayerKind::TerrainConstraints(_)
            | LayerKind::NoiseValue(_)
            | LayerKind::NoisePerlin(_)
            | LayerKind::NoiseOpenSimplex(_)
            | LayerKind::NoiseWorley(_)
            | LayerKind::Fbm(_)
            | LayerKind::Ridged(_)
            | LayerKind::DomainWarp(_)
            | LayerKind::Mountains(_)
            | LayerKind::Mesa(_)
            | LayerKind::Volcano(_)
            | LayerKind::Uplift(_)
            | LayerKind::Dunes(_)
            | LayerKind::Canyons(_)
            | LayerKind::VoronoiRegions(_)
            | LayerKind::Island(_)
            | LayerKind::Plateau(_)
            | LayerKind::Ramp(_)
            | LayerKind::GradientReconstruct(_)
            | LayerKind::ProceduralShape(_)
            | LayerKind::Stamp2d(_)
            | LayerKind::Stamp3d(_) => EvalStage::Blueprint,

            LayerKind::EffectFilter(_)
            | LayerKind::Terrace(_)
            | LayerKind::Blur(_)
            | LayerKind::Path(_)
            | LayerKind::PolygonHeight(_) => EvalStage::PreBiomeFields,

            LayerKind::Biomes(_) => EvalStage::Climate,

            LayerKind::LandscapeEvolution(_)
            | LayerKind::HydrologyRepair(_)
            | LayerKind::StreamPowerErosion(_)
            | LayerKind::HydraulicErosion(_)
            | LayerKind::ThermalErosion(_)
            | LayerKind::DebrisFlow(_)
            | LayerKind::MultiScaleAmplify(_)
            | LayerKind::RiverCarve(_)
            | LayerKind::RiverNetwork(_)
            | LayerKind::Coastal(_)
            | LayerKind::SandSimulation(_)
            | LayerKind::FluidSimulation(_) => EvalStage::SharedHydro,

            LayerKind::Materials(_) => EvalStage::Materials,

            LayerKind::Vegetation(_) => EvalStage::Vegetation,

            LayerKind::GeomorphicDetail(_)
            | LayerKind::EcosystemFeedback(_)
            | LayerKind::OverhangStamp(_)
            | LayerKind::LocalSdf(_) => EvalStage::FineDetail,
        }
    }

    /// Apply a parameter binding's remapped sample onto a named numeric path.
    pub fn apply_param_binding(&mut self, binding: &crate::layer::ParamBinding, sample: f32) {
        let path = binding.target.0.as_str();
        if path == "opacity" {
            return;
        }
        let get_base = |this: &Self| -> Option<f32> {
            match (path, this) {
                (
                    "amplitude",
                    LayerKind::NoiseValue(p)
                    | LayerKind::NoisePerlin(p)
                    | LayerKind::NoiseOpenSimplex(p),
                ) => Some(p.amplitude),
                ("amplitude", LayerKind::NoiseWorley(p)) => Some(p.base.amplitude),
                ("amplitude", LayerKind::Fbm(p) | LayerKind::Ridged(p)) => Some(p.base.amplitude),
                ("amplitude", LayerKind::Mountains(p)) => Some(p.base.amplitude),
                ("amplitude", LayerKind::Uplift(p)) => Some(p.amplitude),
                ("strength", LayerKind::EffectFilter(p)) => Some(p.strength),
                ("strength", LayerKind::ThermalErosion(p)) => Some(p.strength),
                ("iterations", LayerKind::LandscapeEvolution(p)) => Some(p.iterations as f32),
                ("iterations", LayerKind::HydraulicErosion(p)) => Some(p.iterations as f32),
                ("iterations", LayerKind::StreamPowerErosion(p)) => Some(p.iterations as f32),
                ("density", LayerKind::Vegetation(p)) => Some(p.density),
                ("k", LayerKind::StreamPowerErosion(p)) => Some(p.k),
                _ => None,
            }
        };
        let Some(base) = get_base(self) else {
            return;
        };
        let out = binding.apply_scalar(base, sample);
        match (path, self) {
            (
                "amplitude",
                LayerKind::NoiseValue(p)
                | LayerKind::NoisePerlin(p)
                | LayerKind::NoiseOpenSimplex(p),
            ) => p.amplitude = out,
            ("amplitude", LayerKind::NoiseWorley(p)) => p.base.amplitude = out,
            ("amplitude", LayerKind::Fbm(p) | LayerKind::Ridged(p)) => p.base.amplitude = out,
            ("amplitude", LayerKind::Mountains(p)) => p.base.amplitude = out,
            ("amplitude", LayerKind::Uplift(p)) => p.amplitude = out,
            ("strength", LayerKind::EffectFilter(p)) => p.strength = out,
            ("strength", LayerKind::ThermalErosion(p)) => p.strength = out,
            ("iterations", LayerKind::LandscapeEvolution(p)) => {
                p.iterations = out.round().clamp(1.0, 256.0) as u32;
            }
            ("iterations", LayerKind::HydraulicErosion(p)) => {
                p.iterations = out.round().clamp(1.0, 512.0) as u32;
            }
            ("iterations", LayerKind::StreamPowerErosion(p)) => {
                p.iterations = out.round().clamp(1.0, 256.0) as u32;
            }
            ("density", LayerKind::Vegetation(p)) => p.density = out,
            ("k", LayerKind::StreamPowerErosion(p)) => p.k = out,
            _ => {}
        }
    }

    /// World Creator–style default blend for new layers of this kind.
    ///
    /// Generators contribute absolute heights and should **Add**.
    /// Bases/imports and filters/sims that rewrite the stack use **Normal**.
    pub fn default_blend(&self) -> BlendMode {
        match self {
            LayerKind::SculptBase(_)
            | LayerKind::Flat(_)
            | LayerKind::SculptStrokes(_)
            | LayerKind::TerrainConstraints(_)
            | LayerKind::GradientReconstruct(_)
            | LayerKind::LandscapeEvolution(_)
            | LayerKind::HydrologyRepair(_)
            | LayerKind::GeomorphicDetail(_)
            | LayerKind::EcosystemFeedback(_)
            | LayerKind::Ramp(_)
            | LayerKind::ImportHeightmap(_)
            | LayerKind::Terrace(_)
            | LayerKind::Plateau(_)
            | LayerKind::Island(_)
            | LayerKind::ThermalErosion(_)
            | LayerKind::HydraulicErosion(_)
            | LayerKind::DebrisFlow(_)
            | LayerKind::StreamPowerErosion(_)
            | LayerKind::MultiScaleAmplify(_)
            | LayerKind::RiverCarve(_)
            | LayerKind::Blur(_)
            | LayerKind::Coastal(_)
            | LayerKind::EffectFilter(_)
            | LayerKind::Path(_)
            | LayerKind::PolygonHeight(_)
            | LayerKind::RiverNetwork(_)
            | LayerKind::SandSimulation(_)
            | LayerKind::FluidSimulation(_)
            | LayerKind::Materials(_)
            | LayerKind::Biomes(_)
            | LayerKind::Vegetation(_)
            | LayerKind::OverhangStamp(_)
            | LayerKind::LocalSdf(_)
            | LayerKind::Stamp3d(_) => BlendMode::Normal,

            LayerKind::NoiseValue(_)
            | LayerKind::NoisePerlin(_)
            | LayerKind::NoiseOpenSimplex(_)
            | LayerKind::NoiseWorley(_)
            | LayerKind::Fbm(_)
            | LayerKind::Ridged(_)
            | LayerKind::DomainWarp(_)
            | LayerKind::Mountains(_)
            | LayerKind::Mesa(_)
            | LayerKind::Volcano(_)
            | LayerKind::Uplift(_)
            | LayerKind::Dunes(_)
            | LayerKind::Canyons(_)
            | LayerKind::VoronoiRegions(_)
            | LayerKind::ProceduralShape(_)
            | LayerKind::Stamp2d(_) => BlendMode::Add,
        }
    }
}
