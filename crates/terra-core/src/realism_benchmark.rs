//! Internal realism benchmark worlds (Phase 11 Rules 5–6).
//!
//! Benchmarks assemble editable documents from real algorithms (not black-box
//! generators) and expose expected morphometric ranges for clay-render review.

use crate::analyze::TerrainStatistics;
use crate::document::TerrainDocument;
use crate::eval::{EvalContext, PreviewQuality, StackEvaluator};
use crate::heightfield::Heightfield;
use crate::landscape_style::LandscapeStyle;
use crate::world_archetype::{
    alpine_world, badlands_world, coastal_world, desert_world, dune_field_world,
    old_mountains_world, young_mountains_world,
};

/// Named internal benchmark project.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RealismBenchmark {
    Alpine,
    DesertMesa,
    Badlands,
    OldMountains,
    YoungMountains,
    DuneField,
    Coastal,
}

impl RealismBenchmark {
    pub fn all() -> &'static [RealismBenchmark] {
        &[
            Self::Alpine,
            Self::DesertMesa,
            Self::Badlands,
            Self::OldMountains,
            Self::YoungMountains,
            Self::DuneField,
            Self::Coastal,
        ]
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::Alpine => "alpine",
            Self::DesertMesa => "desert_mesa",
            Self::Badlands => "badlands",
            Self::OldMountains => "old_mountains",
            Self::YoungMountains => "young_mountains",
            Self::DuneField => "dune_field",
            Self::Coastal => "coastal",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Alpine => "ALPINE",
            Self::DesertMesa => "DESERT MESA",
            Self::Badlands => "BADLANDS",
            Self::OldMountains => "OLD MOUNTAINS",
            Self::YoungMountains => "YOUNG MOUNTAINS",
            Self::DuneField => "DUNE FIELD",
            Self::Coastal => "COASTAL",
        }
    }

    pub fn style(self) -> LandscapeStyle {
        match self {
            Self::Alpine => LandscapeStyle::Alpine,
            Self::DesertMesa => LandscapeStyle::Mesa,
            Self::Badlands => LandscapeStyle::Badlands,
            Self::OldMountains => LandscapeStyle::OldMountains,
            Self::YoungMountains => LandscapeStyle::YoungMountains,
            Self::DuneField => LandscapeStyle::DuneField,
            Self::Coastal => LandscapeStyle::Coastal,
        }
    }

    /// Build the editable document (preview resolution for CI).
    pub fn document(self, world_size_m: f32, preview_res: u32) -> TerrainDocument {
        match self {
            Self::Alpine => alpine_world(world_size_m, preview_res),
            Self::DesertMesa => desert_world(world_size_m, preview_res),
            Self::Badlands => badlands_world(world_size_m, preview_res),
            Self::OldMountains => old_mountains_world(world_size_m, preview_res),
            Self::YoungMountains => young_mountains_world(world_size_m, preview_res),
            Self::DuneField => dune_field_world(world_size_m, preview_res),
            Self::Coastal => coastal_world(world_size_m, preview_res),
        }
    }

    /// Soft expected ranges for clay-geometry review (not hard DEM oracles).
    pub fn expected_ranges(self) -> BenchmarkExpectations {
        match self {
            Self::Alpine => BenchmarkExpectations {
                min_relief_m: 80.0,
                max_mean_slope_deg: 45.0,
                min_drainage_density: 0.02,
                notes: "uplift → drainage → incision → debris → talus",
            },
            Self::DesertMesa => BenchmarkExpectations {
                min_relief_m: 40.0,
                max_mean_slope_deg: 40.0,
                min_drainage_density: 0.0,
                notes: "layered geology, plateau, canyon, differential erosion",
            },
            Self::Badlands => BenchmarkExpectations {
                min_relief_m: 30.0,
                max_mean_slope_deg: 50.0,
                min_drainage_density: 0.05,
                notes: "soft strata + dense drainage",
            },
            Self::OldMountains => BenchmarkExpectations {
                min_relief_m: 40.0,
                max_mean_slope_deg: 28.0,
                min_drainage_density: 0.03,
                notes: "broad valleys, reduced relief",
            },
            Self::YoungMountains => BenchmarkExpectations {
                min_relief_m: 100.0,
                max_mean_slope_deg: 55.0,
                min_drainage_density: 0.01,
                notes: "steep relief, immature valleys",
            },
            Self::DuneField => BenchmarkExpectations {
                min_relief_m: 5.0,
                max_mean_slope_deg: 35.0,
                min_drainage_density: 0.0,
                notes: "wind-driven sand transport",
            },
            Self::Coastal => BenchmarkExpectations {
                min_relief_m: 20.0,
                max_mean_slope_deg: 35.0,
                min_drainage_density: 0.02,
                notes: "shore profile + drainage to sea",
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BenchmarkExpectations {
    pub min_relief_m: f32,
    pub max_mean_slope_deg: f32,
    pub min_drainage_density: f32,
    pub notes: &'static str,
}

/// Evaluate a document to a heightfield and collect morphometrics.
pub fn measure_document(doc: &TerrainDocument) -> Result<(Heightfield, TerrainStatistics), String> {
    let mut ctx = EvalContext::new(doc.metrics);
    ctx.quality = PreviewQuality::Draft;
    let mut eval = StackEvaluator::new();
    let hf = eval
        .rebuild_all(&doc.stack, &mut ctx)
        .map_err(|e| format!("benchmark eval failed: {e}"))?;
    let stats = TerrainStatistics::compute(&hf);
    Ok((hf, stats))
}

/// Soft structural checks for a benchmark (CI-friendly, low resolution).
pub fn validate_benchmark_structure(
    bench: RealismBenchmark,
    preview_res: u32,
) -> Result<(), String> {
    let doc = bench.document(8_000.0, preview_res);
    let layers = doc.stack.flatten_layers();
    if layers.is_empty() {
        return Err(format!("{}: empty stack", bench.id()));
    }
    let has_macro = layers.iter().any(|l| {
        matches!(
            l.kind.scale_band(),
            crate::layer::ScaleBand::Macro | crate::layer::ScaleBand::MultiScale
        )
    });
    if !has_macro {
        return Err(format!("{}: missing macro / multi-scale stage", bench.id()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_benchmarks_have_structure() {
        for b in RealismBenchmark::all() {
            validate_benchmark_structure(*b, 64).unwrap_or_else(|e| panic!("{e}"));
            let _ = b.expected_ranges();
            let _ = b.style().params();
        }
    }
}
