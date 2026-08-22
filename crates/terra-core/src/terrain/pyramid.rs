use crate::heightfield::{HeightfieldMetrics, TileId, DEFAULT_HALO, DEFAULT_TILE_SIZE};

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
    pub levels: Vec<TerrainLevel>,
}

impl TerrainPyramid {
    pub fn new(config: PyramidConfig) -> Self {
        // Build from the requested output down so every parent dimension is the
        // exact ceil-half of its child. This preserves a complete hierarchy for
        // non-power-of-two outputs instead of ending with an irregular 512→1000
        // transition.
        let mut resolutions = vec![config.target_resolution.max(2)];
        while resolutions.last().copied().unwrap_or(2) > 2 {
            let child = resolutions.last().copied().unwrap_or(2);
            resolutions.push(child.div_ceil(2).max(2));
        }
        resolutions.reverse();
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

    pub fn level(&self, index: u8) -> Option<&TerrainLevel> {
        self.levels.get(index as usize)
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
        let metrics = self.level_metrics(level)?;
        if tile.tx >= metrics.tiles_x() || tile.tz >= metrics.tiles_z() {
            return None;
        }
        let origin_x = tile.tx * metrics.tile_size;
        let origin_z = tile.tz * metrics.tile_size;
        Some(TerrainTileExtent {
            origin_x,
            origin_z,
            width: (metrics.width - origin_x).min(metrics.tile_size),
            height: (metrics.height - origin_z).min(metrics.tile_size),
        })
    }

    /// Stable dense index used by immutable per-content metadata buffers.
    pub fn tile_metadata_index(&self, level: u8, tile: TileId) -> Option<u32> {
        self.tile_extent(level, tile)?;
        let before = self
            .levels
            .iter()
            .take(level as usize)
            .map(|entry| {
                let tile_size = self.config.tile_size.min(entry.resolution).max(1);
                entry.resolution.div_ceil(tile_size).pow(2)
            })
            .sum::<u32>();
        let metrics = self.level_metrics(level)?;
        Some(before + tile.tz * metrics.tiles_x() + tile.tx)
    }

    pub fn metadata_len(&self) -> u32 {
        self.levels
            .iter()
            .map(|level| {
                let tile_size = self.config.tile_size.min(level.resolution).max(1);
                level.resolution.div_ceil(tile_size).pow(2)
            })
            .sum()
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
        let source = self.level_metrics(source_level)?;
        let target = self.level_metrics(target_level)?;
        let extent = self.tile_extent(source_level, source_tile)?;
        let x0 = scaled_floor(extent.origin_x, target.width, source.width);
        let z0 = scaled_floor(extent.origin_z, target.height, source.height);
        let x1 = scaled_ceil(
            extent.origin_x.saturating_add(extent.width),
            target.width,
            source.width,
        )
        .saturating_sub(1)
        .min(target.width - 1);
        let z1 = scaled_ceil(
            extent.origin_z.saturating_add(extent.height),
            target.height,
            source.height,
        )
        .saturating_sub(1)
        .min(target.height - 1);
        Some(TerrainTileRange {
            min: TileId {
                tx: x0 / target.tile_size,
                tz: z0 / target.tile_size,
            },
            max: TileId {
                tx: x1 / target.tile_size,
                tz: z1 / target.tile_size,
            },
        })
    }
}

fn scaled_floor(value: u32, target: u32, source: u32) -> u32 {
    ((u64::from(value) * u64::from(target)) / u64::from(source)) as u32
}

fn scaled_ceil(value: u32, target: u32, source: u32) -> u32 {
    let numerator = u64::from(value) * u64::from(target);
    numerator.div_ceil(u64::from(source)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_builds_complete_upsample_chain() {
        let pyramid = TerrainPyramid::new(PyramidConfig::new(1000, 4096.0, 4096.0));
        assert_eq!(pyramid.levels.first().unwrap().resolution, 2);
        assert_eq!(pyramid.levels.last().unwrap().resolution, 1000);
        assert_eq!(
            pyramid
                .levels
                .iter()
                .map(|level| level.resolution)
                .collect::<Vec<_>>(),
            vec![2, 4, 8, 16, 32, 63, 125, 250, 500, 1000]
        );
        assert!(pyramid
            .levels
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
        for level in &pyramid.levels {
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
                .levels
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
