//! Source and evaluation-resolution semantics for artist-visible layers.
//!
//! This is domain metadata rather than presentation text. Keeping the match on
//! [`LayerKind`] exhaustive prevents the Inspector from becoming a second,
//! hand-maintained catalog of layer behavior.

use super::LayerKind;

/// Dimensions of a stored or imported raster grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridDimensions {
    pub width: u32,
    pub height: u32,
}

impl GridDimensions {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub const fn square(resolution: u32) -> Self {
        Self::new(resolution, resolution)
    }
}

/// Where a layer's authored source detail comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerResolutionSource<'a> {
    /// A dense grid stored directly in the project.
    FixedRaster(GridDimensions),
    /// A raster asset whose dimensions must be read from the source file.
    ImportedRaster { path: &'a str },
    /// Semantic or procedural authoring data with no fixed sample grid.
    ResolutionIndependent,
    /// An operator whose source grid is the active evaluation input.
    EvaluationInput,
    /// Geometry projected onto the active evaluation grid.
    MeshGeometry { path: &'a str },
}

/// How a layer produces its output at the active evaluation resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationResolutionBehavior {
    Resampled,
    Rasterized,
    Generated,
    Processed,
    Simulated,
}

/// Resolution contract for one layer kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerResolutionSemantics<'a> {
    pub source: LayerResolutionSource<'a>,
    pub behavior: EvaluationResolutionBehavior,
}

impl LayerKind {
    /// Describe source detail and evaluation behavior without inspecting output buffers.
    pub fn resolution_semantics(&self) -> LayerResolutionSemantics<'_> {
        use EvaluationResolutionBehavior as Behavior;
        use LayerKind::*;
        use LayerResolutionSource as Source;

        match self {
            SculptBase(params) => LayerResolutionSemantics {
                source: Source::FixedRaster(GridDimensions::square(params.resolution)),
                behavior: Behavior::Resampled,
            },
            ImportHeightmap(params) => LayerResolutionSemantics {
                source: Source::ImportedRaster { path: &params.path },
                behavior: Behavior::Resampled,
            },
            Stamp2d(params) => LayerResolutionSemantics {
                source: Source::ImportedRaster {
                    path: &params.heightmap.path,
                },
                behavior: Behavior::Resampled,
            },
            Stamp3d(params) if params.path.is_empty() => LayerResolutionSemantics {
                source: Source::ResolutionIndependent,
                behavior: Behavior::Generated,
            },
            Stamp3d(params) if is_obj_path(&params.path) => LayerResolutionSemantics {
                source: Source::MeshGeometry { path: &params.path },
                behavior: Behavior::Rasterized,
            },
            Stamp3d(params) => LayerResolutionSemantics {
                source: Source::ImportedRaster { path: &params.path },
                behavior: Behavior::Resampled,
            },

            SculptStrokes(_)
            | TerrainConstraints(_)
            | OverhangStamp(_)
            | LocalSdf(_)
            | Path(_)
            | PolygonHeight(_) => LayerResolutionSemantics {
                source: Source::ResolutionIndependent,
                behavior: Behavior::Rasterized,
            },

            Flat(_) | Ramp(_) | NoiseValue(_) | NoisePerlin(_) | NoiseOpenSimplex(_)
            | NoiseWorley(_) | Fbm(_) | Ridged(_) | DomainWarp(_) | Mesa(_) | Island(_)
            | Mountains(_) | Volcano(_) | Uplift(_) | Dunes(_) | Canyons(_) | VoronoiRegions(_)
            | ProceduralShape(_) => LayerResolutionSemantics {
                source: Source::ResolutionIndependent,
                behavior: Behavior::Generated,
            },

            GradientReconstruct(_)
            | GeomorphicDetail(_)
            | Terrace(_)
            | Plateau(_)
            | Blur(_)
            | Coastal(_)
            | EffectFilter(_)
            | Materials(_)
            | Biomes(_)
            | Vegetation(_) => LayerResolutionSemantics {
                source: Source::EvaluationInput,
                behavior: Behavior::Processed,
            },

            LandscapeEvolution(_)
            | HydrologyRepair(_)
            | EcosystemFeedback(_)
            | ThermalErosion(_)
            | HydraulicErosion(_)
            | DebrisFlow(_)
            | StreamPowerErosion(_)
            | MultiScaleAmplify(_)
            | RiverCarve(_)
            | RiverNetwork(_)
            | SandSimulation(_)
            | FluidSimulation(_) => LayerResolutionSemantics {
                source: Source::EvaluationInput,
                behavior: Behavior::Simulated,
            },
        }
    }
}

fn is_obj_path(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("obj"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{
        EffectFilterParams, ImportHeightmapParams, LayerKind, SculptParams, SculptStrokeParams,
        Stamp2dParams, Stamp3dParams, ThermalErosionParams,
    };

    #[test]
    fn fixed_sculpt_reports_its_stored_grid() {
        let kind = LayerKind::SculptBase(SculptParams::filled(384, 0.0));

        assert_eq!(
            kind.resolution_semantics(),
            LayerResolutionSemantics {
                source: LayerResolutionSource::FixedRaster(GridDimensions::square(384)),
                behavior: EvaluationResolutionBehavior::Resampled,
            }
        );
    }

    #[test]
    fn semantic_procedural_filter_and_simulation_remain_distinct() {
        let strokes = LayerKind::SculptStrokes(SculptStrokeParams::default());
        assert_eq!(
            strokes.resolution_semantics().source,
            LayerResolutionSource::ResolutionIndependent
        );
        assert_eq!(
            strokes.resolution_semantics().behavior,
            EvaluationResolutionBehavior::Rasterized
        );

        let filter = LayerKind::EffectFilter(EffectFilterParams::default());
        assert_eq!(
            filter.resolution_semantics(),
            LayerResolutionSemantics {
                source: LayerResolutionSource::EvaluationInput,
                behavior: EvaluationResolutionBehavior::Processed,
            }
        );

        let simulation = LayerKind::ThermalErosion(ThermalErosionParams::default());
        assert_eq!(
            simulation.resolution_semantics().behavior,
            EvaluationResolutionBehavior::Simulated
        );
    }

    #[test]
    fn imported_and_stamp_sources_borrow_the_authored_path() {
        let imported = LayerKind::ImportHeightmap(ImportHeightmapParams {
            path: "terrain.png".into(),
            ..ImportHeightmapParams::default()
        });
        assert_eq!(
            imported.resolution_semantics().source,
            LayerResolutionSource::ImportedRaster {
                path: "terrain.png"
            }
        );

        let stamp = LayerKind::Stamp2d(Stamp2dParams {
            heightmap: ImportHeightmapParams {
                path: "stamp.tif".into(),
                ..ImportHeightmapParams::default()
            },
        });
        assert_eq!(
            stamp.resolution_semantics().source,
            LayerResolutionSource::ImportedRaster { path: "stamp.tif" }
        );
    }

    #[test]
    fn stamp_3d_distinguishes_procedural_mesh_and_image_sources() {
        let procedural = LayerKind::Stamp3d(Stamp3dParams::default());
        assert_eq!(
            procedural.resolution_semantics().source,
            LayerResolutionSource::ResolutionIndependent
        );

        let mesh = LayerKind::Stamp3d(Stamp3dParams {
            path: "cliff.OBJ".into(),
            ..Stamp3dParams::default()
        });
        assert_eq!(
            mesh.resolution_semantics().source,
            LayerResolutionSource::MeshGeometry { path: "cliff.OBJ" }
        );

        let image = LayerKind::Stamp3d(Stamp3dParams {
            path: "rock.png".into(),
            ..Stamp3dParams::default()
        });
        assert_eq!(
            image.resolution_semantics().source,
            LayerResolutionSource::ImportedRaster { path: "rock.png" }
        );
    }
}
