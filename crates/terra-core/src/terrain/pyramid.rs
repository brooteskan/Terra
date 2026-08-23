use crate::heightfield::{HeightfieldMetrics, TileId, DEFAULT_HALO, DEFAULT_TILE_SIZE};
use terra_world::{BoundedTopology, BoundedTopologyConfig, TileAddress};

use super::TerrainTileKey;

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

pub type TerrainLevel = terra_world::BoundedLevel;

/// Interior sample extent of one tile in a pyramid level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerrainTileExtent {
    pub origin_x: u32,
    pub origin_z: u32,
    pub width: u32,
    pub height: u32,
}

/// Inclusive tile-coordinate range covering a normalized terrain footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerrainTileRange {
    pub min: TileId,
    pub max: TileId,
}

impl TerrainTileRange {
    pub fn iter(self) -> impl Iterator<Item = TileId> {
        (self.min.tz..=self.max.tz)
            .flat_map(move |tz| (self.min.tx..=self.max.tx).map(move |tx| TileId { tx, tz }))
    }
}

/// The final-output resolution ladder: coarse → target, with every parent
/// dimension equal to the exact ceil-half of its child.
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
    topology: BoundedTopology,
}

impl TerrainPyramid {
    pub fn new(config: PyramidConfig) -> Self {
        let topology = BoundedTopology::try_new(BoundedTopologyConfig {
            target_resolution: config.target_resolution.max(2),
            world_size_x: f64::from(config.world_size_x),
            world_size_z: f64::from(config.world_size_z),
            tile_size: config.tile_size,
            halo: config.halo,
        })
        .expect("pyramid configuration must contain finite positive world extents and tile size");
        Self { config, topology }
    }

    pub fn topology(&self) -> &BoundedTopology {
        &self.topology
    }

    pub fn levels(&self) -> &[TerrainLevel] {
        self.topology.levels()
    }

    pub fn max_level(&self) -> u8 {
        self.topology.max_level()
    }

    pub fn level(&self, index: u8) -> Option<&TerrainLevel> {
        self.topology.level(index)
    }

    pub fn level_metrics(&self, index: u8) -> Option<HeightfieldMetrics> {
        let level = self.level(index)?;
        Some(HeightfieldMetrics {
            width: level.resolution,
            height: level.resolution,
            world_size_x: self.config.world_size_x,
            world_size_z: self.config.world_size_z,
            tile_size: self.config.tile_size.min(level.resolution).max(1),
            halo: self.config.halo,
        })
    }

    pub fn tile_extent(&self, level: u8, tile: TileId) -> Option<TerrainTileExtent> {
        let address = self.address(level, tile)?;
        let extent = self.topology.tile_extent(address).ok()?.samples;
        Some(TerrainTileExtent {
            origin_x: u32::try_from(extent.origin.x).ok()?,
            origin_z: u32::try_from(extent.origin.z).ok()?,
            width: extent.width,
            height: extent.height,
        })
    }

    pub fn address(&self, level: u8, tile: TileId) -> Option<TileAddress> {
        self.topology.address(level, tile.tx, tile.tz).ok()
    }

    pub fn level_and_tile(&self, address: TileAddress) -> Option<(u8, TileId)> {
        Some((
            self.topology.level_index(address)?,
            TileId {
                tx: u32::try_from(address.coord.x).ok()?,
                tz: u32::try_from(address.coord.z).ok()?,
            },
        ))
    }

    /// Stable dense index used by immutable per-content metadata buffers.
    pub fn tile_metadata_index(&self, level: u8, tile: TileId) -> Option<u32> {
        self.topology.metadata_index(self.address(level, tile)?)
    }

    pub fn address_metadata_index(&self, address: TileAddress) -> Option<u32> {
        self.topology.metadata_index(address)
    }

    pub fn metadata_len(&self) -> u32 {
        self.topology.metadata_len()
    }

    /// Every final-height tile in stable package order: levels coarse to fine,
    /// then tile rows and columns. The iterator is complete by construction and
    /// its position agrees with [`Self::tile_metadata_index`].
    pub fn height_tiles(&self) -> impl Iterator<Item = TerrainTileKey> + '_ {
        self.topology.addresses().map(TerrainTileKey::height)
    }

    /// Stable final-height traversal for one level.
    pub fn height_tiles_at_level(
        &self,
        level: u8,
    ) -> Option<impl Iterator<Item = TerrainTileKey> + '_> {
        Some(
            self.topology
                .addresses_at_level(level)?
                .map(TerrainTileKey::height),
        )
    }

    /// Parent tiles whose normalized sample footprint intersects `tile`.
    ///
    /// A non-power-of-two edge can overlap more than one parent page, so this
    /// deliberately returns a range rather than inventing a one-to-one parent.
    pub fn covering_parent_tiles(&self, level: u8, tile: TileId) -> Option<TerrainTileRange> {
        if level == 0 {
            return None;
        }
        self.covering_tiles_between(level, tile, level - 1)
    }

    /// Child tiles whose normalized sample footprint intersects `tile`.
    pub fn covering_child_tiles(&self, level: u8, tile: TileId) -> Option<TerrainTileRange> {
        if level >= self.max_level() {
            return None;
        }
        self.covering_tiles_between(level, tile, level + 1)
    }

    fn covering_tiles_between(
        &self,
        source_level: u8,
        source_tile: TileId,
        target_level: u8,
    ) -> Option<TerrainTileRange> {
        let source = self.address(source_level, source_tile)?;
        let range = if target_level < source_level {
            self.topology.covering_parent_tiles(source)?
        } else {
            self.topology.covering_child_tiles(source)?
        };
        Some(TerrainTileRange {
            min: TileId {
                tx: u32::try_from(range.min.coord.x).ok()?,
                tz: u32::try_from(range.min.coord.z).ok()?,
            },
            max: TileId {
                tx: u32::try_from(range.max.coord.x).ok()?,
                tz: u32::try_from(range.max.coord.z).ok()?,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_builds_complete_upsample_chain() {
        let pyramid = TerrainPyramid::new(PyramidConfig::new(1000, 4096.0, 4096.0));
        assert_eq!(pyramid.levels().first().unwrap().resolution, 2);
        assert_eq!(pyramid.levels().last().unwrap().resolution, 1000);
        assert_eq!(
            pyramid
                .levels()
                .iter()
                .map(|level| level.resolution)
                .collect::<Vec<_>>(),
            vec![2, 4, 8, 16, 32, 63, 125, 250, 500, 1000]
        );
        assert!(pyramid
            .levels()
            .windows(2)
            .all(|pair| pair[0].resolution == pair[1].resolution.div_ceil(2)));
    }

    #[test]
    fn partial_edge_tiles_have_exact_extents_and_dense_indices() {
        let mut config = PyramidConfig::new(1000, 4096.0, 2048.0);
        config.tile_size = 256;
        let pyramid = TerrainPyramid::new(config);
        let finest = pyramid.max_level();
        assert_eq!(
            pyramid.tile_extent(finest, TileId { tx: 3, tz: 3 }),
            Some(TerrainTileExtent {
                origin_x: 768,
                origin_z: 768,
                width: 232,
                height: 232,
            })
        );
        let mut indices = Vec::new();
        for level in pyramid.levels() {
            let metrics = pyramid.level_metrics(level.index).unwrap();
            for tz in 0..metrics.tiles_z() {
                for tx in 0..metrics.tiles_x() {
                    indices.push(
                        pyramid
                            .tile_metadata_index(level.index, TileId { tx, tz })
                            .unwrap(),
                    );
                }
            }
        }
        assert_eq!(indices, (0..pyramid.metadata_len()).collect::<Vec<_>>());
        let keys = pyramid.height_tiles().collect::<Vec<_>>();
        assert_eq!(keys.len(), pyramid.metadata_len() as usize);
        for (index, key) in keys.iter().enumerate() {
            assert_eq!(
                pyramid.address_metadata_index(key.address),
                Some(index as u32)
            );
        }
    }

    #[test]
    fn non_power_of_two_parent_and_child_ranges_cover_every_tile() {
        let mut config = PyramidConfig::new(513, 4096.0, 2048.0);
        config.tile_size = 128;
        let pyramid = TerrainPyramid::new(config);
        for level in 1..=pyramid.max_level() {
            let child = pyramid.level_metrics(level).unwrap();
            let parent = pyramid.level_metrics(level - 1).unwrap();
            for tz in 0..child.tiles_z() {
                for tx in 0..child.tiles_x() {
                    let range = pyramid
                        .covering_parent_tiles(level, TileId { tx, tz })
                        .unwrap();
                    assert!(range.max.tx < parent.tiles_x());
                    assert!(range.max.tz < parent.tiles_z());
                    assert!(range.iter().next().is_some());
                }
            }
        }

        for level in 0..pyramid.max_level() {
            let parent = pyramid.level_metrics(level).unwrap();
            let child = pyramid.level_metrics(level + 1).unwrap();
            for tz in 0..parent.tiles_z() {
                for tx in 0..parent.tiles_x() {
                    let range = pyramid
                        .covering_child_tiles(level, TileId { tx, tz })
                        .unwrap();
                    assert!(range.max.tx < child.tiles_x());
                    assert!(range.max.tz < child.tiles_z());
                }
            }
        }
    }

    #[test]
    fn power_of_two_parent_and_child_ranges_follow_normalized_footprints() {
        let mut config = PyramidConfig::new(512, 4096.0, 4096.0);
        config.tile_size = 64;
        let pyramid = TerrainPyramid::new(config);
        assert_eq!(
            pyramid
                .levels()
                .iter()
                .map(|level| level.resolution)
                .collect::<Vec<_>>(),
            vec![2, 4, 8, 16, 32, 64, 128, 256, 512]
        );
        assert_eq!(
            pyramid.covering_child_tiles(6, TileId { tx: 1, tz: 1 }),
            Some(TerrainTileRange {
                min: TileId { tx: 2, tz: 2 },
                max: TileId { tx: 3, tz: 3 },
            })
        );
        assert_eq!(
            pyramid.covering_parent_tiles(7, TileId { tx: 2, tz: 2 }),
            Some(TerrainTileRange {
                min: TileId { tx: 1, tz: 1 },
                max: TileId { tx: 1, tz: 1 },
            })
        );
    }
}
