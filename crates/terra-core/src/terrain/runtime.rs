use super::{EditorRefinementState, PyramidConfig, RefinementController, TerrainPyramid};
use terra_world::{InfiniteTopology, InfiniteTopologyConfig, WorldError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TerrainRuntimeConfig {
    Bounded(PyramidConfig),
    Infinite(InfiniteTopologyConfig),
}

#[derive(Debug)]
pub enum TerrainRuntimeTopology {
    Bounded(TerrainPyramid),
    Infinite(InfiniteTopology),
}

/// App-facing owner of the terrain resolution ladder, output revision, and refinement state.
///
/// GPU tile residency belongs to `GpuTileAtlas`; this runtime deliberately keeps
/// no resident-page records or CPU copy of the atlas page table.
#[derive(Debug)]
pub struct TerrainRuntime {
    topology: TerrainRuntimeTopology,
    pub refinement: RefinementController,
    output_revision: u64,
}

impl TerrainRuntime {
    pub fn new(config: PyramidConfig) -> Self {
        Self {
            topology: TerrainRuntimeTopology::Bounded(TerrainPyramid::new(config)),
            refinement: RefinementController::default(),
            output_revision: 0,
        }
    }

    pub fn try_new(config: TerrainRuntimeConfig) -> Result<Self, WorldError> {
        let topology = match config {
            TerrainRuntimeConfig::Bounded(config) => {
                TerrainRuntimeTopology::Bounded(TerrainPyramid::try_new(config)?)
            }
            TerrainRuntimeConfig::Infinite(config) => {
                TerrainRuntimeTopology::Infinite(InfiniteTopology::try_new(config)?)
            }
        };
        Ok(Self {
            topology,
            refinement: RefinementController::default(),
            output_revision: 0,
        })
    }

    pub const fn topology(&self) -> &TerrainRuntimeTopology {
        &self.topology
    }

    pub const fn bounded_pyramid(&self) -> Option<&TerrainPyramid> {
        match &self.topology {
            TerrainRuntimeTopology::Bounded(pyramid) => Some(pyramid),
            TerrainRuntimeTopology::Infinite(_) => None,
        }
    }

    pub fn bounded_pyramid_mut(&mut self) -> Option<&mut TerrainPyramid> {
        match &mut self.topology {
            TerrainRuntimeTopology::Bounded(pyramid) => Some(pyramid),
            TerrainRuntimeTopology::Infinite(_) => None,
        }
    }

    pub const fn infinite_topology(&self) -> Option<&InfiniteTopology> {
        match &self.topology {
            TerrainRuntimeTopology::Infinite(topology) => Some(topology),
            TerrainRuntimeTopology::Bounded(_) => None,
        }
    }

    pub fn output_revision(&self) -> u64 {
        self.output_revision
    }

    /// Begin a new whole-field output generation. Streamed residency is retired
    /// GPU-side at this same boundary by the app's `advance_output_revision`
    /// (`retire_streamed_residency`): the atlas page table, `TileResidencyCache`,
    /// and the renderer's streaming flag. The pyramid holds no residency to clear.
    pub fn advance_output_revision(&mut self) -> u64 {
        self.output_revision = self.output_revision.wrapping_add(1);
        self.output_revision
    }

    pub fn reconfigure(&mut self, config: PyramidConfig) {
        self.topology = TerrainRuntimeTopology::Bounded(TerrainPyramid::new(config));
        self.output_revision = self.output_revision.wrapping_add(1);
    }

    pub fn try_reconfigure(&mut self, config: TerrainRuntimeConfig) -> Result<(), WorldError> {
        self.topology = match config {
            TerrainRuntimeConfig::Bounded(config) => {
                TerrainRuntimeTopology::Bounded(TerrainPyramid::try_new(config)?)
            }
            TerrainRuntimeConfig::Infinite(config) => {
                TerrainRuntimeTopology::Infinite(InfiniteTopology::try_new(config)?)
            }
        };
        self.output_revision = self.output_revision.wrapping_add(1);
        Ok(())
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
        let pyramid = runtime.bounded_pyramid().unwrap();
        assert_eq!(pyramid.levels().last().unwrap().resolution, 1000);
        assert_eq!(pyramid.config.world_size_x, 8000.0);
        assert_eq!(pyramid.config.world_size_z, 4000.0);
    }

    #[test]
    fn runtime_can_switch_to_sparse_infinite_topology() {
        let mut runtime = TerrainRuntime::new(PyramidConfig::new(512, 1024.0, 1024.0));
        let config = InfiniteTopologyConfig {
            origin: terra_world::WorldPosition::try_new(-10.0, 25.0).unwrap(),
            tile_size: 256,
            finest_spacing_m: 0.5,
            max_lod: terra_world::Lod::try_new(10).unwrap(),
        };

        runtime
            .try_reconfigure(TerrainRuntimeConfig::Infinite(config))
            .unwrap();

        assert!(runtime.bounded_pyramid().is_none());
        assert_eq!(runtime.infinite_topology().unwrap().config(), config);
        assert_eq!(runtime.output_revision(), 1);
    }
}
