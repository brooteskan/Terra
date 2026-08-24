//! Operation categories and field contracts for terrain layers.

use super::LayerKind;
use crate::field_data::FieldId;
use crate::invalidation::{AuxReach, DirtyClass, Reach};
use serde::{Deserialize, Serialize};

use super::ScaleBand;

/// Internal height-op classification (create vs transform vs sim / surface).
///
/// **Not** the World Creator artist taxonomy. Artist-facing folders are
/// Shape Layers / Biome Filters / Simulation — see [`StackCategory`](super::group_mode::StackCategory)
/// and [`biome_destination_section`](super::stack::biome_destination_section). Do not surface
/// "Generator" / "Modifier" as Shape workflow labels in the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationCategory {
    /// Creates height contribution (internal; often a Shape Layer or Filter by kind).
    Generator,
    /// Transforms existing height (internal; Path/Polygon stay Shape; Terrace/filters → Biome Filters).
    Modifier,
    Simulation,
    Analysis,
    Surface,
    Group,
    ImportedData,
    Organisation,
}

impl OperationCategory {
    /// Internal / debug label — prefer WC folder names in artist UI.
    pub fn label(self) -> &'static str {
        match self {
            OperationCategory::Generator => "Height source",
            OperationCategory::Modifier => "Height transform",
            OperationCategory::Simulation => "Simulation",
            OperationCategory::Analysis => "Analysis",
            OperationCategory::Surface => "Surface",
            OperationCategory::Group => "Group",
            OperationCategory::ImportedData => "Imported",
            OperationCategory::Organisation => "Organisation",
        }
    }

    /// Internal / debug badge — not used for WC Shape Layers UI.
    pub fn short_badge(self) -> &'static str {
        match self {
            OperationCategory::Generator => "HSRC",
            OperationCategory::Modifier => "HXFM",
            OperationCategory::Simulation => "SIM",
            OperationCategory::Analysis => "ANL",
            OperationCategory::Surface => "SRF",
            OperationCategory::Group => "GRP",
            OperationCategory::ImportedData => "IMP",
            OperationCategory::Organisation => "ORG",
        }
    }
}

impl LayerKind {
    pub fn category(&self) -> OperationCategory {
        match self {
            LayerKind::SculptBase(_)
            | LayerKind::SculptStrokes(_)
            | LayerKind::TerrainConstraints(_)
            | LayerKind::Flat(_)
            | LayerKind::Ramp(_)
            | LayerKind::NoiseValue(_)
            | LayerKind::NoisePerlin(_)
            | LayerKind::NoiseOpenSimplex(_)
            | LayerKind::NoiseWorley(_)
            | LayerKind::Fbm(_)
            | LayerKind::Ridged(_)
            | LayerKind::DomainWarp(_)
            | LayerKind::Mesa(_)
            | LayerKind::Island(_)
            | LayerKind::Mountains(_)
            | LayerKind::Volcano(_)
            | LayerKind::Uplift(_)
            | LayerKind::Dunes(_)
            | LayerKind::Canyons(_)
            | LayerKind::VoronoiRegions(_)
            | LayerKind::ProceduralShape(_)
            | LayerKind::Stamp2d(_)
            | LayerKind::Stamp3d(_) => OperationCategory::Generator,

            LayerKind::ImportHeightmap(_) => OperationCategory::ImportedData,

            LayerKind::Terrace(_)
            | LayerKind::GradientReconstruct(_)
            | LayerKind::GeomorphicDetail(_)
            | LayerKind::Plateau(_)
            | LayerKind::Blur(_)
            | LayerKind::Coastal(_)
            | LayerKind::EffectFilter(_)
            | LayerKind::Path(_)
            | LayerKind::PolygonHeight(_)
            | LayerKind::OverhangStamp(_)
            | LayerKind::LocalSdf(_) => OperationCategory::Modifier,

            LayerKind::ThermalErosion(_)
            | LayerKind::LandscapeEvolution(_)
            | LayerKind::HydrologyRepair(_)
            | LayerKind::EcosystemFeedback(_)
            | LayerKind::HydraulicErosion(_)
            | LayerKind::DebrisFlow(_)
            | LayerKind::StreamPowerErosion(_)
            | LayerKind::MultiScaleAmplify(_)
            | LayerKind::RiverCarve(_)
            | LayerKind::RiverNetwork(_)
            | LayerKind::SandSimulation(_)
            | LayerKind::FluidSimulation(_) => OperationCategory::Simulation,

            LayerKind::Materials(_) | LayerKind::Biomes(_) | LayerKind::Vegetation(_) => {
                OperationCategory::Surface
            }
        }
    }

    /// Fields this operation requires to be present (beyond height for modifiers/sims).
    pub fn required_fields(&self) -> Vec<FieldId> {
        match self {
            LayerKind::ThermalErosion(_)
            | LayerKind::HydraulicErosion(_)
            | LayerKind::DebrisFlow(_)
            | LayerKind::StreamPowerErosion(_)
            | LayerKind::SculptStrokes(_)
            | LayerKind::TerrainConstraints(_)
            | LayerKind::GradientReconstruct(_)
            | LayerKind::LandscapeEvolution(_)
            | LayerKind::HydrologyRepair(_)
            | LayerKind::GeomorphicDetail(_)
            | LayerKind::EcosystemFeedback(_)
            | LayerKind::MultiScaleAmplify(_)
            | LayerKind::RiverCarve(_)
            | LayerKind::RiverNetwork(_)
            | LayerKind::SandSimulation(_)
            | LayerKind::FluidSimulation(_)
            | LayerKind::Terrace(_)
            | LayerKind::Plateau(_)
            | LayerKind::Blur(_)
            | LayerKind::Coastal(_)
            | LayerKind::EffectFilter(_)
            | LayerKind::Path(_)
            | LayerKind::PolygonHeight(_) => vec![FieldId::Height],
            LayerKind::Biomes(_) => vec![FieldId::Height],
            LayerKind::Vegetation(_) => vec![FieldId::Height],
            LayerKind::Materials(_) => vec![FieldId::Height],
            _ => Vec::new(),
        }
    }

    pub fn optional_fields(&self) -> Vec<FieldId> {
        match self {
            LayerKind::SandSimulation(_) => vec![
                FieldId::WindExposure,
                FieldId::Vegetation,
                FieldId::SoilMoisture,
            ],
            LayerKind::HydraulicErosion(_)
            | LayerKind::ThermalErosion(_)
            | LayerKind::DebrisFlow(_) => {
                vec![
                    FieldId::Hardness,
                    FieldId::Rainfall,
                    FieldId::SoilDepth,
                    FieldId::Vegetation,
                    FieldId::BedrockHeight,
                    FieldId::SedimentThickness,
                    FieldId::DebrisDepth,
                ]
            }
            LayerKind::StreamPowerErosion(_) => {
                vec![FieldId::Hardness, FieldId::FlowAccumulation]
            }
            LayerKind::Biomes(_) => vec![
                FieldId::Temperature,
                FieldId::Rainfall,
                FieldId::Wetness,
                FieldId::Slope,
            ],
            LayerKind::Vegetation(_) => {
                vec![
                    FieldId::Biomes,
                    FieldId::Wetness,
                    FieldId::Slope,
                    FieldId::Snow,
                ]
            }
            LayerKind::Materials(_) => {
                vec![
                    FieldId::Slope,
                    FieldId::Curvature,
                    FieldId::Wetness,
                    FieldId::Erosion,
                ]
            }
            _ => Vec::new(),
        }
    }

    pub fn produced_fields(&self) -> Vec<FieldId> {
        match self {
            LayerKind::SculptStrokes(_) => vec![
                FieldId::Height,
                FieldId::Hardness,
                FieldId::Named(crate::field_data::keys::SCULPT_PROTECTION.into()),
                FieldId::Named(crate::field_data::keys::UPLIFT_RATE.into()),
                FieldId::SedimentThickness,
                FieldId::Named(crate::field_data::keys::EDIT_REGION.into()),
            ],
            LayerKind::TerrainConstraints(_) => vec![
                FieldId::Height,
                FieldId::Named(crate::field_data::keys::CONSTRAINT_TARGET.into()),
                FieldId::Named(crate::field_data::keys::CONSTRAINT_WEIGHT.into()),
                FieldId::Named(crate::field_data::keys::SCULPT_PROTECTION.into()),
                FieldId::Named(crate::field_data::keys::UPLIFT_RATE.into()),
                FieldId::Named(crate::field_data::keys::EDIT_REGION.into()),
            ],
            LayerKind::GradientReconstruct(_) => vec![
                FieldId::Height,
                FieldId::Named(crate::field_data::keys::CONSTRAINT_ERROR.into()),
            ],
            LayerKind::LandscapeEvolution(_) => vec![
                FieldId::Height,
                FieldId::FlowDirection,
                FieldId::FlowAccumulation,
                FieldId::StreamOrder,
                FieldId::SpeIncision,
                FieldId::Erosion,
                FieldId::Deposition,
                FieldId::WaterDischarge,
                FieldId::SedimentThickness,
                FieldId::Named(crate::field_data::keys::UPLIFT_RATE.into()),
                FieldId::Named(crate::field_data::keys::TECTONIC_BASE.into()),
            ],
            LayerKind::HydrologyRepair(_) => vec![
                FieldId::Height,
                FieldId::FlowDirection,
                FieldId::FlowAccumulation,
                FieldId::StreamOrder,
                FieldId::SpeIncision,
                FieldId::Named(crate::field_data::keys::REPAIR_REGION.into()),
            ],
            LayerKind::GeomorphicDetail(_) => vec![
                FieldId::Height,
                FieldId::Named(crate::field_data::keys::DETAIL_MASK.into()),
                FieldId::FineFlow,
                FieldId::MicroChannel,
                FieldId::RidgeBreakup,
                FieldId::FineErosion,
            ],
            LayerKind::EcosystemFeedback(_) => vec![
                FieldId::Height,
                FieldId::Hardness,
                FieldId::Deposition,
                FieldId::Named(crate::field_data::keys::ROOT_COHESION.into()),
            ],
            LayerKind::Island(_) => vec![
                FieldId::Height,
                FieldId::Named(crate::field_data::keys::LAND_MASK.into()),
                FieldId::Named(crate::field_data::keys::SHORE_DISTANCE.into()),
                FieldId::Named(crate::field_data::keys::BATHYMETRY.into()),
                FieldId::Named(crate::field_data::keys::SHELF.into()),
                FieldId::Named(crate::field_data::keys::BEACH.into()),
                FieldId::Named(crate::field_data::keys::REEF.into()),
                FieldId::Named(crate::field_data::keys::MOUNTAIN_MASK.into()),
            ],
            LayerKind::HydraulicErosion(_) => vec![
                FieldId::Height,
                FieldId::Hardness,
                FieldId::Wetness,
                FieldId::WaterDepth,
                FieldId::Sediment,
                FieldId::Erosion,
                FieldId::Deposition,
                FieldId::Materials,
                FieldId::Water,
                FieldId::WaterVelocity,
                FieldId::FlowAccumulation,
                FieldId::ChannelMask,
                FieldId::BedrockHeight,
                FieldId::SedimentThickness,
                FieldId::Rainfall,
            ],
            LayerKind::SandSimulation(_) | LayerKind::Dunes(_) => vec![
                FieldId::Height,
                FieldId::SandDepth,
                FieldId::BedrockHeight,
                FieldId::WindDirection,
                FieldId::WindSpeed,
                FieldId::SandFlux,
                FieldId::Erosion,
                FieldId::Deposition,
                FieldId::Sheltering,
                FieldId::DuneCrest,
                FieldId::SandMaterialMask,
                FieldId::FlowDirection,
            ],
            LayerKind::ThermalErosion(_) => {
                vec![
                    FieldId::Height,
                    FieldId::Erosion,
                    FieldId::Deposition,
                    FieldId::Hardness,
                    FieldId::Materials,
                    FieldId::BedrockHeight,
                    FieldId::DebrisDepth,
                    FieldId::SedimentThickness,
                    FieldId::TalusStability,
                    FieldId::Instability,
                ]
            }
            LayerKind::DebrisFlow(_) => {
                vec![
                    FieldId::Height,
                    FieldId::Hardness,
                    FieldId::Erosion,
                    FieldId::Deposition,
                    FieldId::BedrockHeight,
                    FieldId::DebrisDepth,
                    FieldId::SedimentThickness,
                    FieldId::SlidePath,
                    FieldId::Instability,
                    FieldId::FlowAccumulation,
                ]
            }
            LayerKind::StreamPowerErosion(_) => vec![
                FieldId::Height,
                FieldId::Hardness,
                FieldId::Materials,
                FieldId::FlowDirection,
                FieldId::FlowAccumulation,
                FieldId::StreamOrder,
                FieldId::SpeIncision,
                FieldId::Erosion,
            ],
            LayerKind::RiverCarve(_) => vec![
                FieldId::Height,
                FieldId::FlowDirection,
                FieldId::FlowAccumulation,
                FieldId::StreamOrder,
                FieldId::SpeIncision,
                FieldId::Wetness,
            ],
            LayerKind::MultiScaleAmplify(_) => vec![
                FieldId::Height,
                FieldId::Hardness,
                FieldId::Erosion,
                FieldId::Deposition,
            ],
            LayerKind::Materials(_) => {
                vec![
                    FieldId::Materials,
                    FieldId::Hardness,
                    FieldId::StrataReference,
                ]
            }
            LayerKind::Biomes(_) => vec![
                FieldId::Biomes,
                FieldId::Temperature,
                FieldId::Rainfall,
                FieldId::Humidity,
                FieldId::Aridity,
                FieldId::Snow,
                FieldId::SoilMoisture,
                FieldId::WindExposure,
            ],
            // Root cohesion writes hardness only when enabled and an incoming
            // hardness field exists. Contracts are static, so declare the
            // conditional write conservatively.
            LayerKind::Vegetation(_) => vec![FieldId::Vegetation, FieldId::Hardness],
            LayerKind::OverhangStamp(_) | LayerKind::LocalSdf(_) => {
                vec![
                    FieldId::Height,
                    FieldId::OverhangCeiling,
                    FieldId::OverhangMask,
                ]
            }
            LayerKind::FluidSimulation(_) => {
                vec![FieldId::Height, FieldId::Wetness, FieldId::WaterDepth]
            }
            LayerKind::RiverNetwork(_) | LayerKind::Path(_) => {
                vec![FieldId::Height, FieldId::Wetness]
            }
            _ if matches!(self.category(), OperationCategory::Generator) => {
                vec![FieldId::Height]
            }
            _ => vec![FieldId::Height],
        }
    }

    pub fn modified_fields(&self) -> Vec<FieldId> {
        self.produced_fields()
            .into_iter()
            .filter(|f| {
                *f == FieldId::Height
                    || (matches!(self, LayerKind::Vegetation(_)) && *f == FieldId::Hardness)
                    || matches!(
                        self.category(),
                        OperationCategory::Simulation | OperationCategory::Modifier
                    )
            })
            .collect()
    }

    /// Coarse intrinsic spatial-dependency of this operator's *height* kernel.
    ///
    /// Exhaustive by construction (no wildcard): a new `LayerKind` must declare a
    /// bucket here, so a globally-coupled operator can never silently inherit
    /// `Local` and be treated as tile-localizable (the #100 correctness hazard).
    /// This is the height kernel only; auxiliary-field coupling is
    /// [`Self::aux_reach`] and the resolved per-config answer is
    /// [`crate::layer_reach::effective_reach`].
    pub fn spatial_dependency(&self) -> DirtyClass {
        match self {
            // --- Input-independent generators: output ignores the composed input
            // below; a dirty region only survives through the layer's blend, which
            // is per-texel. Localizable with no halo.
            LayerKind::Flat(_)
            | LayerKind::Ramp(_)
            | LayerKind::NoiseValue(_)
            | LayerKind::NoisePerlin(_)
            | LayerKind::NoiseOpenSimplex(_)
            | LayerKind::NoiseWorley(_)
            | LayerKind::Fbm(_)
            | LayerKind::Ridged(_)
            | LayerKind::DomainWarp(_)
            | LayerKind::Mesa(_)
            | LayerKind::Island(_)
            | LayerKind::Mountains(_)
            | LayerKind::Volcano(_)
            | LayerKind::Uplift(_)
            | LayerKind::Canyons(_)
            | LayerKind::VoronoiRegions(_)
            | LayerKind::ImportHeightmap(_)
            | LayerKind::ProceduralShape(_)
            | LayerKind::Stamp2d(_)
            | LayerKind::Stamp3d(_)
            // --- Authoring / bounded shape ops: per-texel or bounded-stencil.
            | LayerKind::SculptBase(_)
            | LayerKind::SculptStrokes(_)
            | LayerKind::TerrainConstraints(_)
            | LayerKind::Plateau(_)
            | LayerKind::Coastal(_)
            | LayerKind::PolygonHeight(_)
            | LayerKind::Path(_)
            | LayerKind::OverhangStamp(_)
            | LayerKind::LocalSdf(_)
            | LayerKind::Materials(_)
            | LayerKind::Blur(_) => DirtyClass::Local,

            // `EffectFilter` fronts ~60 sub-kinds; defer to the sub-kind so a
            // Smooth is not lumped with a flow-routing filter.
            LayerKind::EffectFilter(p) => p.kind.spatial_dependency(),

            // Bounded multi-iteration diffusion (root/soil feedback). The height
            // kernel is genuinely bounded; its aux is global, which
            // `effective_reach` folds in separately.
            LayerKind::EcosystemFeedback(_) => DirtyClass::Expanding,

            // --- Whole-field / basin-coupled. Reclassified out of the old
            // wildcard-`Local` and `Expanding` buckets, which were unsound:
            //   Terrace           quantizes against the global field range
            //   Dunes/Sand        aeolian transport with upwind sheltering rays
            //   Thermal/Hydraulic level-step pyramid resample + global normalize
            //   DebrisFlow        sorts all cells by elevation
            //   Fluid             couples whole depressions
            //   GradientRecon     screened-Poisson (elliptic, global support)
            //   LandscapeEvo      priority-flood + D8/D-inf drainage graphs
            //   HydrologyRepair   full-field SPE before its dilated-region blend
            //   GeomorphicDetail  flow-accumulation with global normalize
            //   Biomes            jump-flood coastal distance from every seed
            //   Vegetation        sequential Poisson-disk scatter over the domain
            LayerKind::Terrace(_)
            | LayerKind::Dunes(_)
            | LayerKind::ThermalErosion(_)
            | LayerKind::HydraulicErosion(_)
            | LayerKind::DebrisFlow(_)
            | LayerKind::SandSimulation(_)
            | LayerKind::FluidSimulation(_)
            | LayerKind::GradientReconstruct(_)
            | LayerKind::LandscapeEvolution(_)
            | LayerKind::HydrologyRepair(_)
            | LayerKind::GeomorphicDetail(_)
            | LayerKind::Biomes(_)
            | LayerKind::Vegetation(_)
            | LayerKind::StreamPowerErosion(_)
            | LayerKind::MultiScaleAmplify(_)
            | LayerKind::RiverCarve(_)
            | LayerKind::RiverNetwork(_) => DirtyClass::BasinDependent,
        }
    }

    /// Intrinsic per-side sample halo of the height kernel, as a [`Reach`].
    ///
    /// Resolves the coarse [`Self::spatial_dependency`] bucket into an actual
    /// halo where one is honestly known: `Blur` and the bounded `EffectFilter`
    /// kernels carry `radius * iterations`; `SculptStrokes` reaches one sample past
    /// its footprint for the reconcile 3x3, and a second sample when a
    /// base-neighborhood stroke (Smooth / Pinch / Coastline) feeds that reconcile.
    /// Every other non-`Local` kind returns [`Reach::Full`] here — the localizable
    /// cases are exactly the explicit arms, so nothing globally-coupled leaks
    /// through as a finite halo.
    pub fn intrinsic_reach(&self) -> Reach {
        match self {
            LayerKind::EffectFilter(p) => match p.kind.spatial_dependency() {
                DirtyClass::Local => Reach::LOCAL,
                DirtyClass::Expanding => Reach::Localized {
                    halo_samples: p.kernel_halo(),
                },
                DirtyClass::BasinDependent => Reach::Full,
            },
            LayerKind::Blur(p) => Reach::Localized {
                halo_samples: p.radius.saturating_mul(p.iterations.max(1)),
            },
            LayerKind::SculptStrokes(p) if p.strokes.iter().all(|stroke| !stroke.enabled) => {
                Reach::LOCAL
            }
            LayerKind::SculptStrokes(p) => Reach::Localized {
                // Floor of 1 for the reconcile 3x3 (also covers a base-neighborhood
                // stroke's own stamp read). Reach is 2 only when such a stroke feeds a
                // non-zero reconcile: the base 3x3 shifts the stamped field one texel,
                // then reconcile re-reads that at 3x3. Keeps the CPU oracle in
                // agreement with the GPU plan halo for Smooth (#114). Flatten stays
                // tile-scoped too — its footprint fixpoint (#110) keeps a self-edit
                // recompute bit-exact without escalating the layer's reach.
                halo_samples: 1 + u32::from(
                    p.reconcile > 0.0
                        && p.strokes.iter().any(|s| {
                            s.enabled
                                && matches!(
                                    s.kind,
                                    crate::authoring::SculptStrokeKind::Smooth
                                        | crate::authoring::SculptStrokeKind::Pinch
                                        | crate::authoring::SculptStrokeKind::Coastline
                                )
                        }),
                ),
            },
            other => match other.spatial_dependency() {
                DirtyClass::Local => Reach::LOCAL,
                DirtyClass::Expanding | DirtyClass::BasinDependent => Reach::Full,
            },
        }
    }

    /// Whether this configured kernel has a proven sparse Infinite-world
    /// evaluation contract. This is deliberately narrower than `Local` dirty
    /// reach: authored rasters and kernels still using bounded UV coordinates
    /// must not be admitted merely because their stencil is local.
    pub fn infinite_capability(&self) -> crate::invalidation::InfiniteOperationCapability {
        use crate::invalidation::{InfiniteOperationCapability as Capability, SpatialRejectReason};

        match self {
            // A freshly-created Shape history layer carries bounded-project stroke
            // semantics, but until it contains an enabled stroke its kernel and
            // published auxiliary fields are coordinate-independent no-ops. Infinite
            // project templates include this empty layer so the sculpt tool is ready.
            LayerKind::SculptStrokes(params)
                if params.strokes.iter().all(|stroke| !stroke.enabled) =>
            {
                Capability::Direct
            }
            LayerKind::Flat(_)
            | LayerKind::NoiseValue(_)
            | LayerKind::NoisePerlin(_)
            | LayerKind::NoiseOpenSimplex(_)
            | LayerKind::NoiseWorley(_)
            | LayerKind::Fbm(_)
            | LayerKind::Ridged(_)
            | LayerKind::Blur(_) => Capability::Direct,

            LayerKind::Terrace(_) => {
                Capability::Unsupported(SpatialRejectReason::FullFieldNormalization)
            }

            LayerKind::ThermalErosion(_)
            | LayerKind::HydraulicErosion(_)
            | LayerKind::DebrisFlow(_)
            | LayerKind::SandSimulation(_)
            | LayerKind::FluidSimulation(_)
            | LayerKind::GradientReconstruct(_)
            | LayerKind::LandscapeEvolution(_)
            | LayerKind::HydrologyRepair(_)
            | LayerKind::GeomorphicDetail(_)
            | LayerKind::Biomes(_)
            | LayerKind::Vegetation(_)
            | LayerKind::StreamPowerErosion(_)
            | LayerKind::MultiScaleAmplify(_)
            | LayerKind::RiverCarve(_)
            | LayerKind::RiverNetwork(_)
            | LayerKind::Dunes(_) => Capability::Unsupported(SpatialRejectReason::BasinDependent),

            LayerKind::SculptBase(_)
            | LayerKind::SculptStrokes(_)
            | LayerKind::TerrainConstraints(_)
            | LayerKind::ImportHeightmap(_)
            | LayerKind::ProceduralShape(_)
            | LayerKind::Stamp2d(_)
            | LayerKind::Stamp3d(_)
            | LayerKind::PolygonHeight(_)
            | LayerKind::Path(_)
            | LayerKind::OverhangStamp(_)
            | LayerKind::LocalSdf(_) => {
                Capability::Unsupported(SpatialRejectReason::BoundedAuthoredData)
            }

            LayerKind::Ramp(_)
            | LayerKind::DomainWarp(_)
            | LayerKind::Mesa(_)
            | LayerKind::Island(_)
            | LayerKind::Mountains(_)
            | LayerKind::Volcano(_)
            | LayerKind::Uplift(_)
            | LayerKind::Canyons(_)
            | LayerKind::VoronoiRegions(_)
            | LayerKind::Plateau(_)
            | LayerKind::Coastal(_)
            | LayerKind::Materials(_)
            | LayerKind::EffectFilter(_)
            | LayerKind::EcosystemFeedback(_) => {
                Capability::Unsupported(SpatialRejectReason::MissingDomainCoordinateContract)
            }
        }
    }

    /// How this layer's published auxiliary fields behave under a localized edit.
    ///
    /// Exhaustive; see [`AuxReach`]. `PerTexel` aux (max-merged stamp footprints,
    /// height snapshots) can be patched over the dirty texels, so it does not by
    /// itself force a whole-field recompute; `Global` aux does. `Island` and
    /// `Materials` are the load-bearing cases: their *height* kernel is `Local`
    /// yet they emit globally-derived aux, so only this axis catches them.
    pub fn aux_reach(&self) -> AuxReach {
        match self {
            LayerKind::SculptStrokes(_)
            | LayerKind::TerrainConstraints(_)
            | LayerKind::Path(_)
            | LayerKind::OverhangStamp(_)
            | LayerKind::LocalSdf(_) => AuxReach::PerTexel,

            LayerKind::Island(_)
            | LayerKind::Dunes(_)
            | LayerKind::SandSimulation(_)
            | LayerKind::Materials(_)
            | LayerKind::Biomes(_)
            | LayerKind::Vegetation(_)
            | LayerKind::LandscapeEvolution(_)
            | LayerKind::HydrologyRepair(_)
            | LayerKind::GeomorphicDetail(_)
            | LayerKind::GradientReconstruct(_)
            | LayerKind::EcosystemFeedback(_)
            | LayerKind::ThermalErosion(_)
            | LayerKind::HydraulicErosion(_)
            | LayerKind::DebrisFlow(_)
            | LayerKind::StreamPowerErosion(_)
            | LayerKind::RiverCarve(_)
            | LayerKind::RiverNetwork(_)
            | LayerKind::MultiScaleAmplify(_)
            | LayerKind::FluidSimulation(_) => AuxReach::Global,

            LayerKind::Flat(_)
            | LayerKind::Ramp(_)
            | LayerKind::NoiseValue(_)
            | LayerKind::NoisePerlin(_)
            | LayerKind::NoiseOpenSimplex(_)
            | LayerKind::NoiseWorley(_)
            | LayerKind::Fbm(_)
            | LayerKind::Ridged(_)
            | LayerKind::DomainWarp(_)
            | LayerKind::Terrace(_)
            | LayerKind::Plateau(_)
            | LayerKind::Mesa(_)
            | LayerKind::Mountains(_)
            | LayerKind::Volcano(_)
            | LayerKind::Uplift(_)
            | LayerKind::Canyons(_)
            | LayerKind::VoronoiRegions(_)
            | LayerKind::ImportHeightmap(_)
            | LayerKind::Blur(_)
            | LayerKind::Coastal(_)
            | LayerKind::EffectFilter(_)
            | LayerKind::ProceduralShape(_)
            | LayerKind::Stamp2d(_)
            | LayerKind::Stamp3d(_)
            | LayerKind::PolygonHeight(_)
            | LayerKind::SculptBase(_) => AuxReach::HeightOnly,
        }
    }

    /// Whether this operator's evaluation reads one of the auxiliary fields that a
    /// [`LayerKind::SculptStrokes`] layer publishes: `SCULPT_PROTECTION`,
    /// `UPLIFT_RATE`, `HARDNESS`, `SEDIMENT_THICKNESS`, or `EDIT_REGION`.
    ///
    /// The GPU sculpt-stroke preview kernel (#113) is a *height* preview only — it
    /// does not reproduce these aux maps. So a `SculptStrokes` layer may only run on
    /// the GPU preview path when no enabled downstream layer consumes its aux;
    /// otherwise that downstream layer's preview would read stale/absent aux and
    /// silently diverge from the authoritative CPU eval. `compile_gpu_graph` uses
    /// this to keep such stacks on the CPU resume, mirroring the reason `Coastal` /
    /// `Materials` stay off the GPU today.
    ///
    /// Exhaustive by construction (no wildcard): a new `LayerKind` that reads sculpt
    /// aux must opt in here, or its preview could diverge unnoticed. The truth is the
    /// CPU processor arms: an arm reading `aux_maps.hardness`
    /// (directly or via `bake_layer_hardness`/`resolve_hardness`), `UPLIFT_RATE`,
    /// `SCULPT_PROTECTION`, `SEDIMENT_THICKNESS`, or `EDIT_REGION` is a consumer.
    pub fn consumes_sculpt_aux(&self) -> bool {
        match self {
            // Read sculpt aux directly in their processor arm.
            LayerKind::LandscapeEvolution(_)  // uplift_rate, sculpt_protection, hardness
            | LayerKind::HydrologyRepair(_)   // edit_region, hardness, sculpt_protection
            | LayerKind::GeomorphicDetail(_)  // hardness, sculpt_protection
            | LayerKind::EcosystemFeedback(_) // hardness, sediment_thickness
            // Resolve hardness through `bake_layer_hardness` -> `resolve_hardness`,
            // which returns `aux_maps.hardness` when the source does not override it;
            // several also read `sediment_thickness` for layered materials.
            | LayerKind::ThermalErosion(_)
            | LayerKind::HydraulicErosion(_)
            | LayerKind::DebrisFlow(_)
            | LayerKind::StreamPowerErosion(_)
            | LayerKind::MultiScaleAmplify(_)
            // Guide mask may select `MaskSource::Hardness` / a named sculpt aux.
            | LayerKind::RiverCarve(_)
            // Root-cohesion boost reads `aux_maps.hardness`.
            | LayerKind::Vegetation(_) => true,

            // Height-only / generator / authoring / filter kinds that never read a
            // sculpt aux key. `Materials` and `Biomes` read `wetness` / `sediment`
            // (not the sculpt `sediment_thickness`), so they are not consumers here.
            LayerKind::Flat(_)
            | LayerKind::Ramp(_)
            | LayerKind::NoiseValue(_)
            | LayerKind::NoisePerlin(_)
            | LayerKind::NoiseOpenSimplex(_)
            | LayerKind::NoiseWorley(_)
            | LayerKind::Fbm(_)
            | LayerKind::Ridged(_)
            | LayerKind::DomainWarp(_)
            | LayerKind::Terrace(_)
            | LayerKind::Plateau(_)
            | LayerKind::Mesa(_)
            | LayerKind::Island(_)
            | LayerKind::Mountains(_)
            | LayerKind::Volcano(_)
            | LayerKind::Uplift(_)
            | LayerKind::Dunes(_)
            | LayerKind::Canyons(_)
            | LayerKind::VoronoiRegions(_)
            | LayerKind::ImportHeightmap(_)
            | LayerKind::ProceduralShape(_)
            | LayerKind::Stamp2d(_)
            | LayerKind::Stamp3d(_)
            | LayerKind::PolygonHeight(_)
            | LayerKind::SculptBase(_)
            | LayerKind::SculptStrokes(_)
            | LayerKind::TerrainConstraints(_)
            | LayerKind::GradientReconstruct(_)
            | LayerKind::Path(_)
            | LayerKind::RiverNetwork(_)
            | LayerKind::SandSimulation(_)
            | LayerKind::FluidSimulation(_)
            | LayerKind::Blur(_)
            | LayerKind::Coastal(_)
            | LayerKind::EffectFilter(_)
            | LayerKind::Materials(_)
            | LayerKind::Biomes(_)
            | LayerKind::OverhangStamp(_)
            | LayerKind::LocalSdf(_) => false,
        }
    }

    /// Phase 11 Rule 3 — scale ownership for this operator family.
    ///
    /// Micro / MultiScale operators must not replace the macro silhouette.
    pub fn scale_band(&self) -> ScaleBand {
        match self {
            // --- MACRO: landmass / tectonic silhouette ---
            LayerKind::Flat(_)
            | LayerKind::Ramp(_)
            | LayerKind::SculptBase(_)
            | LayerKind::ImportHeightmap(_)
            | LayerKind::Island(_)
            | LayerKind::Mountains(_)
            | LayerKind::Mesa(_)
            | LayerKind::Volcano(_)
            | LayerKind::Uplift(_)
            | LayerKind::VoronoiRegions(_)
            | LayerKind::ProceduralShape(_)
            | LayerKind::LandscapeEvolution(_) => ScaleBand::Macro,

            // --- MULTI-SCALE: cascade while locking longer wavelengths ---
            LayerKind::MultiScaleAmplify(_)
            | LayerKind::GeomorphicDetail(_)
            | LayerKind::HydrologyRepair(_) => ScaleBand::MultiScale,

            // --- MICRO: fine surface / decorative detail ---
            LayerKind::Blur(_)
            | LayerKind::EffectFilter(_)
            | LayerKind::OverhangStamp(_)
            | LayerKind::LocalSdf(_)
            | LayerKind::SandSimulation(_)
            | LayerKind::Dunes(_) => ScaleBand::Micro,

            // --- MESO: ridges, valleys, drainage, primary erosion ---
            LayerKind::NoiseValue(_)
            | LayerKind::NoisePerlin(_)
            | LayerKind::NoiseOpenSimplex(_)
            | LayerKind::NoiseWorley(_)
            | LayerKind::Fbm(_)
            | LayerKind::Ridged(_)
            | LayerKind::DomainWarp(_)
            | LayerKind::Terrace(_)
            | LayerKind::Plateau(_)
            | LayerKind::Canyons(_)
            | LayerKind::Path(_)
            | LayerKind::PolygonHeight(_)
            | LayerKind::Stamp2d(_)
            | LayerKind::Stamp3d(_)
            | LayerKind::SculptStrokes(_)
            | LayerKind::TerrainConstraints(_)
            | LayerKind::GradientReconstruct(_)
            | LayerKind::ThermalErosion(_)
            | LayerKind::HydraulicErosion(_)
            | LayerKind::DebrisFlow(_)
            | LayerKind::StreamPowerErosion(_)
            | LayerKind::RiverCarve(_)
            | LayerKind::RiverNetwork(_)
            | LayerKind::Coastal(_)
            | LayerKind::FluidSimulation(_)
            | LayerKind::EcosystemFeedback(_)
            | LayerKind::Materials(_)
            | LayerKind::Biomes(_)
            | LayerKind::Vegetation(_) => ScaleBand::Meso,
        }
    }
}

/// Declared fields for an operation (used by evaluator validation / UI).
#[derive(Debug, Clone)]
pub struct FieldContract {
    pub required: Vec<FieldId>,
    pub optional: Vec<FieldId>,
    pub produced: Vec<FieldId>,
    pub modified: Vec<FieldId>,
    pub spatial: DirtyClass,
}

impl Default for FieldContract {
    fn default() -> Self {
        Self {
            required: Vec::new(),
            optional: Vec::new(),
            produced: Vec::new(),
            modified: Vec::new(),
            spatial: DirtyClass::Local,
        }
    }
}

impl FieldContract {
    pub fn from_kind(kind: &LayerKind) -> Self {
        Self {
            required: kind.required_fields(),
            optional: kind.optional_fields(),
            produced: kind.produced_fields(),
            modified: kind.modified_fields(),
            spatial: kind.spatial_dependency(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{
        EffectFilterParams, HydraulicErosionParams, MultiScaleAmplifyParams, ScaleBand,
        UpliftParams,
    };

    #[test]
    fn hydraulic_is_simulation() {
        let k = LayerKind::HydraulicErosion(HydraulicErosionParams::default());
        assert_eq!(k.category(), OperationCategory::Simulation);
        assert!(k.produced_fields().contains(&FieldId::Wetness));
    }

    #[test]
    fn erosion_family_declares_distinct_debris_and_sediment_fields() {
        for kind in [
            LayerKind::ThermalErosion(Default::default()),
            LayerKind::DebrisFlow(Default::default()),
        ] {
            assert!(kind.optional_fields().contains(&FieldId::DebrisDepth));
            assert!(kind.optional_fields().contains(&FieldId::SedimentThickness));
            assert!(kind.produced_fields().contains(&FieldId::DebrisDepth));
            assert!(kind.produced_fields().contains(&FieldId::SedimentThickness));
        }
    }

    #[test]
    fn scale_band_separates_macro_from_micro() {
        assert_eq!(
            LayerKind::Uplift(UpliftParams::default()).scale_band(),
            ScaleBand::Macro
        );
        assert_eq!(
            LayerKind::HydraulicErosion(HydraulicErosionParams::default()).scale_band(),
            ScaleBand::Meso
        );
        assert_eq!(
            LayerKind::EffectFilter(EffectFilterParams::default()).scale_band(),
            ScaleBand::Micro
        );
        assert_eq!(
            LayerKind::MultiScaleAmplify(MultiScaleAmplifyParams::default()).scale_band(),
            ScaleBand::MultiScale
        );
        assert!(ScaleBand::Micro.respects_macro_silhouette());
    }

    #[test]
    fn multi_scale_amplify_declares_the_aux_fields_its_processor_publishes() {
        let fields =
            LayerKind::MultiScaleAmplify(MultiScaleAmplifyParams::default()).produced_fields();
        for field in [
            FieldId::Height,
            FieldId::Hardness,
            FieldId::Erosion,
            FieldId::Deposition,
        ] {
            assert!(fields.contains(&field), "missing {field:?}");
        }
    }

    #[test]
    fn conditional_and_split_evaluator_outputs_are_declared() {
        let cases = [
            (
                LayerKind::HydraulicErosion(Default::default()),
                vec![FieldId::WaterDepth],
            ),
            (
                LayerKind::ThermalErosion(Default::default()),
                vec![FieldId::Materials],
            ),
            (
                LayerKind::DebrisFlow(Default::default()),
                vec![FieldId::Hardness],
            ),
            (
                LayerKind::StreamPowerErosion(Default::default()),
                vec![FieldId::Hardness, FieldId::Materials, FieldId::Erosion],
            ),
            (
                LayerKind::RiverCarve(Default::default()),
                vec![FieldId::Wetness],
            ),
            (
                LayerKind::Vegetation(Default::default()),
                vec![FieldId::Vegetation, FieldId::Hardness],
            ),
            (
                LayerKind::FluidSimulation(Default::default()),
                vec![FieldId::Wetness, FieldId::WaterDepth],
            ),
            (
                LayerKind::RiverNetwork(Default::default()),
                vec![FieldId::Wetness],
            ),
            (LayerKind::Path(Default::default()), vec![FieldId::Wetness]),
        ];

        for (kind, expected) in cases {
            let produced = kind.produced_fields();
            for field in expected {
                assert!(
                    produced.contains(&field),
                    "{} must declare {field:?}",
                    kind.type_id()
                );
            }
            if matches!(&kind, LayerKind::Vegetation(_)) {
                assert!(kind.modified_fields().contains(&FieldId::Hardness));
            }
        }
    }

    #[test]
    fn wildcard_globals_are_no_longer_local() {
        use crate::invalidation::DirtyClass;
        // Kinds that used to fall through the deleted `_ => Local` arm and be
        // silently treated as tile-localizable.
        for kind in [
            LayerKind::Terrace(Default::default()),
            LayerKind::GradientReconstruct(Default::default()),
            LayerKind::LandscapeEvolution(Default::default()),
            LayerKind::HydrologyRepair(Default::default()),
            LayerKind::GeomorphicDetail(Default::default()),
            LayerKind::Biomes(Default::default()),
            LayerKind::Vegetation(Default::default()),
            LayerKind::Dunes(Default::default()),
        ] {
            assert_eq!(
                kind.spatial_dependency(),
                DirtyClass::BasinDependent,
                "{kind:?} must be basin-coupled, not Local"
            );
        }
        // Input-independent generators stay localizable.
        assert_eq!(
            LayerKind::VoronoiRegions(Default::default()).spatial_dependency(),
            DirtyClass::Local
        );
    }

    #[test]
    fn intrinsic_reach_resolves_bounded_halos() {
        use crate::filter_params::EffectFilterKind;
        use crate::invalidation::Reach;
        use crate::layer::BlurParams;

        // Blur: radius x iterations.
        assert_eq!(
            LayerKind::Blur(BlurParams {
                radius: 3,
                iterations: 2
            })
            .intrinsic_reach(),
            Reach::Localized { halo_samples: 6 }
        );
        // Generator: per-texel.
        assert_eq!(
            LayerKind::VoronoiRegions(Default::default()).intrinsic_reach(),
            Reach::LOCAL
        );
        // An empty SculptStrokes history is a coordinate-independent no-op.
        assert_eq!(
            LayerKind::SculptStrokes(Default::default()).intrinsic_reach(),
            Reach::LOCAL
        );
        // Grows to two when a base-neighborhood stroke (Smooth, Pinch, or Coastline)
        // feeds a non-zero reconcile: the base 3x3 shifts the stamped field, then
        // reconcile re-reads it. Without reconcile such a stroke stays at the one-sample
        // floor (its own base read).
        use crate::authoring::{SculptStroke, SculptStrokeKind, SculptStrokeParams};
        let base_neighborhood_strokes = |kind, reconcile| {
            LayerKind::SculptStrokes(SculptStrokeParams {
                strokes: vec![SculptStroke {
                    kind,
                    ..SculptStroke::default()
                }],
                reconcile,
            })
        };
        for kind in [
            SculptStrokeKind::Smooth,
            SculptStrokeKind::Pinch,
            SculptStrokeKind::Coastline,
        ] {
            assert_eq!(
                base_neighborhood_strokes(kind, 0.15).intrinsic_reach(),
                Reach::Localized { halo_samples: 2 },
                "{kind:?}"
            );
            assert_eq!(
                base_neighborhood_strokes(kind, 0.0).intrinsic_reach(),
                Reach::Localized { halo_samples: 1 },
                "{kind:?}"
            );
        }
        // Flatten stays tile-scoped (its #110 footprint fixpoint keeps a self-edit
        // recompute bit-exact), so it does not escalate the layer's reach: a lone
        // Flatten sits at the one-sample reconcile floor like any per-sample stroke.
        assert_eq!(
            base_neighborhood_strokes(SculptStrokeKind::Flatten, 0.15).intrinsic_reach(),
            Reach::Localized { halo_samples: 1 },
        );
        // Basin-coupled kind: whole field.
        assert_eq!(
            LayerKind::ThermalErosion(Default::default()).intrinsic_reach(),
            Reach::Full
        );
        // EffectFilter delegates to the sub-kind: bounded kernel vs global.
        let smooth = EffectFilterParams {
            kind: EffectFilterKind::Smooth,
            radius: 4,
            iterations: 1,
            ..EffectFilterParams::default()
        };
        assert_eq!(
            LayerKind::EffectFilter(smooth).intrinsic_reach(),
            Reach::Localized { halo_samples: 4 }
        );
        let strata = EffectFilterParams {
            kind: EffectFilterKind::Strata,
            ..EffectFilterParams::default()
        };
        assert_eq!(
            LayerKind::EffectFilter(strata).intrinsic_reach(),
            Reach::Full
        );
    }

    #[test]
    fn empty_sculpt_history_is_an_infinite_safe_no_op() {
        use crate::invalidation::{InfiniteOperationCapability, SpatialRejectReason};

        assert_eq!(
            LayerKind::SculptStrokes(Default::default()).infinite_capability(),
            InfiniteOperationCapability::Direct
        );
        let mut params = crate::authoring::SculptStrokeParams::default();
        params
            .strokes
            .push(crate::authoring::SculptStroke::default());
        assert_eq!(
            LayerKind::SculptStrokes(params).infinite_capability(),
            InfiniteOperationCapability::Unsupported(SpatialRejectReason::BoundedAuthoredData)
        );
    }

    #[test]
    fn sculpt_aux_consumers_are_declared_and_pure_kinds_are_not() {
        // Downstream kinds that read a sculpt-published aux (uplift / protection /
        // hardness / sediment_thickness / edit_region) — directly or through
        // `bake_layer_hardness` -> `resolve_hardness`. A GPU sculpt-stroke preview
        // above any of these must resume on the CPU (#113).
        for kind in [
            LayerKind::LandscapeEvolution(Default::default()),
            LayerKind::HydrologyRepair(Default::default()),
            LayerKind::GeomorphicDetail(Default::default()),
            LayerKind::EcosystemFeedback(Default::default()),
            LayerKind::ThermalErosion(Default::default()),
            LayerKind::HydraulicErosion(Default::default()),
            LayerKind::DebrisFlow(Default::default()),
            LayerKind::StreamPowerErosion(Default::default()),
            LayerKind::MultiScaleAmplify(Default::default()),
            LayerKind::RiverCarve(Default::default()),
            LayerKind::Vegetation(Default::default()),
        ] {
            assert!(
                kind.consumes_sculpt_aux(),
                "{kind:?} reads a sculpt aux and must be declared a consumer"
            );
        }
        // Generators, filters, and the authoring kinds themselves never read a
        // sculpt aux, so a sculpt preview does not diverge across them. `Biomes`
        // reads `sediment`/`wetness`, not the sculpt `sediment_thickness`.
        for kind in [
            LayerKind::Flat(Default::default()),
            LayerKind::NoiseValue(Default::default()),
            LayerKind::Blur(Default::default()),
            LayerKind::Terrace(Default::default()),
            LayerKind::EffectFilter(Default::default()),
            LayerKind::SculptBase(Default::default()),
            LayerKind::SculptStrokes(Default::default()),
            LayerKind::TerrainConstraints(Default::default()),
            LayerKind::Materials(Default::default()),
            LayerKind::Biomes(Default::default()),
        ] {
            assert!(
                !kind.consumes_sculpt_aux(),
                "{kind:?} does not read a sculpt aux and must not gate the preview"
            );
        }
    }

    #[test]
    fn aux_reach_catches_local_height_but_global_aux() {
        use crate::invalidation::AuxReach;
        // Island's height kernel is Local, but it emits jump-flood bathymetry —
        // only aux_reach catches that.
        assert_eq!(
            LayerKind::Island(Default::default()).spatial_dependency(),
            DirtyClass::Local
        );
        assert_eq!(
            LayerKind::Island(Default::default()).aux_reach(),
            AuxReach::Global
        );
        // Stamp aux is per-texel; pure generators are height-only.
        assert_eq!(
            LayerKind::SculptStrokes(Default::default()).aux_reach(),
            AuxReach::PerTexel
        );
        assert_eq!(
            LayerKind::Flat(Default::default()).aux_reach(),
            AuxReach::HeightOnly
        );
    }
}
