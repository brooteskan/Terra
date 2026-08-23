use crate::heightfield::TileId;
use terra_world::SpatialDomain;

use super::{TerrainContentStamp, TerrainPyramid, TerrainTileExtent, TerrainTileKey};

/// A clamped, level-local sample rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerrainSampleExtent {
    pub origin_x: u32,
    pub origin_z: u32,
    pub width: u32,
    pub height: u32,
}

/// World-space sampling contract for one pyramid level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainWorldTransform {
    pub level_width: u32,
    pub level_height: u32,
    pub world_size_x: f32,
    pub world_size_z: f32,
}

impl TerrainWorldTransform {
    pub fn sample_center(self, x: u32, z: u32) -> (f32, f32) {
        (
            (x as f32 + 0.5) / self.level_width.max(1) as f32 * self.world_size_x,
            (z as f32 + 0.5) / self.level_height.max(1) as f32 * self.world_size_z,
        )
    }
}

/// Backend-neutral domain used to evaluate one requested terrain tile.
///
/// `publication_halo` is retained in the atlas page. `operation_halo` is the
/// additional guard required by the compiled plan and is discarded at publish.
#[derive(Debug, Clone, PartialEq)]
pub struct TerrainEvaluationDomain {
    pub key: TerrainTileKey,
    /// Authoritative backend-independent spatial contract. The flattened
    /// bounded fields below remain as migration adapters for current GPU code.
    pub spatial: SpatialDomain,
    pub interior: TerrainTileExtent,
    pub evaluation: TerrainSampleExtent,
    pub publication_halo: u32,
    pub operation_halo: u32,
    pub publish_offset_x: u32,
    pub publish_offset_z: u32,
    pub world: TerrainWorldTransform,
    pub content: TerrainContentStamp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainDomainError {
    InvalidLevel(u8),
    InvalidTile { level: u8, tile: TileId },
    HaloOverflow,
}

impl TerrainEvaluationDomain {
    pub fn for_tile(
        pyramid: &TerrainPyramid,
        key: TerrainTileKey,
        publication_halo: u32,
        operation_halo: u32,
        content: TerrainContentStamp,
    ) -> Result<Self, TerrainDomainError> {
        let (level, tile) =
            pyramid
                .level_and_tile(key.address)
                .ok_or(TerrainDomainError::InvalidTile {
                    level: key.address.lod.get(),
                    tile: TileId {
                        tx: u32::try_from(key.address.coord.x).unwrap_or(u32::MAX),
                        tz: u32::try_from(key.address.coord.z).unwrap_or(u32::MAX),
                    },
                })?;
        let metrics = pyramid
            .level_metrics(level)
            .ok_or(TerrainDomainError::InvalidLevel(level))?;
        let interior = pyramid
            .tile_extent(level, tile)
            .ok_or(TerrainDomainError::InvalidTile { level, tile })?;
        let spatial = pyramid
            .topology()
            .spatial_domain(key.address, publication_halo, operation_halo)
            .map_err(|_| TerrainDomainError::HaloOverflow)?;
        let evaluation_origin_x = u32::try_from(spatial.evaluation.origin.x)
            .map_err(|_| TerrainDomainError::HaloOverflow)?;
        let evaluation_origin_z = u32::try_from(spatial.evaluation.origin.z)
            .map_err(|_| TerrainDomainError::HaloOverflow)?;
        Ok(Self {
            key,
            spatial,
            interior,
            evaluation: TerrainSampleExtent {
                origin_x: evaluation_origin_x,
                origin_z: evaluation_origin_z,
                width: spatial.evaluation.width,
                height: spatial.evaluation.height,
            },
            publication_halo,
            operation_halo,
            publish_offset_x: spatial.publish_offset_x,
            publish_offset_z: spatial.publish_offset_z,
            world: TerrainWorldTransform {
                level_width: metrics.width,
                level_height: metrics.height,
                world_size_x: metrics.world_size_x,
                world_size_z: metrics.world_size_z,
            },
            content,
        })
    }

    pub fn local_metrics(&self) -> crate::heightfield::HeightfieldMetrics {
        let dx = self.world.world_size_x / self.world.level_width.max(1) as f32;
        let dz = self.world.world_size_z / self.world.level_height.max(1) as f32;
        crate::heightfield::HeightfieldMetrics {
            width: self.evaluation.width,
            height: self.evaluation.height,
            world_size_x: dx * self.evaluation.width as f32,
            world_size_z: dz * self.evaluation.height as f32,
            tile_size: self.evaluation.width.max(self.evaluation.height).max(1),
            halo: 0,
        }
    }

    pub fn published_width(&self) -> u32 {
        self.interior.width + self.publication_halo * 2
    }

    pub fn published_height(&self) -> u32 {
        self.interior.height + self.publication_halo * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PyramidConfig;

    fn stamp() -> TerrainContentStamp {
        TerrainContentStamp {
            document_revision: 1,
            plan_revision: 2,
            output_revision: 3,
            content_revision: 4,
        }
    }

    #[test]
    fn partial_edge_domain_clips_guard_but_preserves_world_lattice() {
        let mut config = PyramidConfig::new(1000, 4000.0, 2000.0);
        config.tile_size = 256;
        let pyramid = TerrainPyramid::new(config);
        let level = pyramid.max_level();
        let domain = TerrainEvaluationDomain::for_tile(
            &pyramid,
            TerrainTileKey::height(pyramid.address(level, TileId { tx: 3, tz: 3 }).unwrap()),
            2,
            7,
            stamp(),
        )
        .unwrap();
        assert_eq!(domain.interior.width, 232);
        assert_eq!(domain.evaluation.origin_x, 759);
        assert_eq!(domain.evaluation.width, 241);
        assert_eq!(domain.publish_offset_x, 7);
        assert_eq!(domain.world.sample_center(999, 999), (3998.0, 1999.0));
    }
}
