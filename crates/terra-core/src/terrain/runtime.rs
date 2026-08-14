use super::{EditorRefinementState, PyramidConfig, RefinementController, TerrainPyramid};

/// App-facing owner of final-output tile residency, revisioning, and refinement state.
#[derive(Debug)]
pub struct TerrainRuntime {
    pub pyramid: TerrainPyramid,
    pub refinement: RefinementController,
    output_revision: u64,
}

impl TerrainRuntime {
    pub fn new(config: PyramidConfig) -> Self {
        Self {
            pyramid: TerrainPyramid::new(config),
            refinement: RefinementController::default(),
            output_revision: 0,
        }
    }

    pub fn output_revision(&self) -> u64 {
        self.output_revision
    }

    /// Begin a new whole-field output generation and retire prior residency metadata.
    pub fn advance_output_revision(&mut self) -> u64 {
        self.output_revision = self.output_revision.wrapping_add(1);
        self.pyramid.clear_residency();
        self.output_revision
    }

    pub fn reconfigure(&mut self, config: PyramidConfig) {
        self.pyramid = TerrainPyramid::new(config);
        self.output_revision = self.output_revision.wrapping_add(1);
    }

    pub fn update_refinement(
        &mut self,
        now_ms: u64,
        interaction_active: bool,
    ) -> EditorRefinementState {
        self.refinement.update(now_ms, interaction_active)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

    #[test]
    fn output_revision_retires_previous_residency() {
        let mut runtime = TerrainRuntime::new(PyramidConfig::new(512, 1024.0, 1024.0));
        assert_eq!(runtime.output_revision(), 0);
        assert_eq!(runtime.advance_output_revision(), 1);
        assert_eq!(runtime.output_revision(), 1);
    }

    #[test]
    fn reconfigure_rebuilds_levels_and_advances_revision() {
        let mut runtime = TerrainRuntime::new(PyramidConfig::new(512, 1024.0, 1024.0));
        runtime.reconfigure(PyramidConfig::new(1000, 8000.0, 4000.0));
        assert_eq!(runtime.output_revision(), 1);
        assert_eq!(runtime.pyramid.levels.last().unwrap().resolution, 1000);
        assert_eq!(
            runtime.pyramid.levels.last().unwrap().metrics,
            HeightfieldMetrics {
                width: 1000,
                height: 1000,
                world_size_x: 8000.0,
                world_size_z: 4000.0,
                tile_size: runtime.pyramid.config.tile_size.min(1000),
                halo: runtime.pyramid.config.halo,
            }
        );
    }
}
