use crate::heightfield::{DEFAULT_HALO, DEFAULT_TILE_SIZE};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PyramidConfig {
    pub target_resolution: u32,
    pub world_size_x: f32,
    pub world_size_z: f32,
    pub tile_size: u32,
    pub halo: u32,
}

impl PyramidConfig {
    pub fn new(target_resolution: u32, world_size_x: f32, world_size_z: f32) -> Self {
        Self {
            target_resolution: target_resolution.max(2),
            world_size_x,
            world_size_z,
            tile_size: DEFAULT_TILE_SIZE,
            halo: DEFAULT_HALO,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainLevel {
    pub index: u8,
    pub resolution: u32,
}

/// The final-output resolution ladder: coarse → target, each level a
/// power-of-two upsample of the previous one, with the finest level pinned to
/// `config.target_resolution`.
///
/// This is a plain level ladder, not a residency store. The tile-upload path
/// reads `levels` to stamp streamed pages with the level that matches the final
/// output resolution (`terra-app`'s `queue_final_tile_uploads` /
/// `sync_tile_stream_to_renderer`). Residency itself is GPU-authoritative: the
/// `GpuTileAtlas` page table is what the shader samples and `TileResidencyCache`
/// is its single CPU mirror. Nothing on the CPU keeps a second residency record.
#[derive(Debug, Clone)]
pub struct TerrainPyramid {
    pub config: PyramidConfig,
    pub levels: Vec<TerrainLevel>,
}

impl TerrainPyramid {
    pub fn new(config: PyramidConfig) -> Self {
        let mut resolutions = Vec::new();
        let mut resolution = 2u32;
        while resolution < config.target_resolution {
            resolutions.push(resolution);
            resolution = resolution.saturating_mul(2);
            if resolution == u32::MAX {
                break;
            }
        }
        if resolutions.last().copied() != Some(config.target_resolution) {
            resolutions.push(config.target_resolution);
        }
        let levels = resolutions
            .into_iter()
            .enumerate()
            .map(|(index, resolution)| TerrainLevel {
                index: index as u8,
                resolution,
            })
            .collect();
        Self { config, levels }
    }

    pub fn max_level(&self) -> u8 {
        self.levels.len().saturating_sub(1) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_builds_complete_upsample_chain() {
        let pyramid = TerrainPyramid::new(PyramidConfig::new(1000, 4096.0, 4096.0));
        assert_eq!(pyramid.levels.first().unwrap().resolution, 2);
        assert_eq!(pyramid.levels.last().unwrap().resolution, 1000);
        assert!(pyramid.levels.windows(2).all(|pair| {
            pair[1].resolution == pair[0].resolution * 2 || pair[1].resolution == 1000
        }));
    }
}
