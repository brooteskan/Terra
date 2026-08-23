use crate::heightfield::TileId;
use terra_world::{InfiniteTopology, SampleCoord, SpatialDomain};

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

/// Topology-specific execution context for a tile evaluation.
///
/// Bounded domains retain the historical complete-level transform. Infinite
/// domains instead carry the project seed; their authoritative signed sample
/// origin and fixed-origin world transform live in [`SpatialDomain`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TerrainEvaluationSpace {
    Bounded,
    Infinite { project_seed: u64 },
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
    pub space: TerrainEvaluationSpace,
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
            space: TerrainEvaluationSpace::Bounded,
            content,
        })
    }

    /// Build one fixed-size sparse Infinite-world tile domain.
    ///
    /// The signed absolute sample lattice remains in `spatial`. The flattened
    /// `interior` and `evaluation` fields are deliberately local adapters for
    /// reusable bounded-era backend allocations and must not be interpreted as
    /// complete-world coordinates for an Infinite domain.
    pub fn for_infinite_tile(
        topology: &InfiniteTopology,
        key: TerrainTileKey,
        project_seed: u64,
        publication_halo: u32,
        operation_halo: u32,
        content: TerrainContentStamp,
    ) -> Result<Self, TerrainDomainError> {
        let spatial = topology
            .spatial_domain(key.address, publication_halo, operation_halo)
            .map_err(|_| TerrainDomainError::HaloOverflow)?;
        let spacing = spatial.interior.transform.spacing();
        let total_halo = publication_halo
            .checked_add(operation_halo)
            .ok_or(TerrainDomainError::HaloOverflow)?;
        let local_span_x = f64::from(spatial.evaluation.width) * spacing.x_m();
        let local_span_z = f64::from(spatial.evaluation.height) * spacing.z_m();
        if !local_span_x.is_finite()
            || !local_span_z.is_finite()
            || local_span_x > f64::from(f32::MAX)
            || local_span_z > f64::from(f32::MAX)
        {
            return Err(TerrainDomainError::HaloOverflow);
        }
        Ok(Self {
            key,
            spatial,
            interior: TerrainTileExtent {
                origin_x: total_halo,
                origin_z: total_halo,
                width: spatial.interior.samples.width,
                height: spatial.interior.samples.height,
            },
            evaluation: TerrainSampleExtent {
                origin_x: 0,
                origin_z: 0,
                width: spatial.evaluation.width,
                height: spatial.evaluation.height,
            },
            publication_halo,
            operation_halo,
            publish_offset_x: spatial.publish_offset_x,
            publish_offset_z: spatial.publish_offset_z,
            world: TerrainWorldTransform {
                level_width: spatial.evaluation.width,
                level_height: spatial.evaluation.height,
                world_size_x: local_span_x as f32,
                world_size_z: local_span_z as f32,
            },
            space: TerrainEvaluationSpace::Infinite { project_seed },
            content,
        })
    }

    pub const fn is_infinite(&self) -> bool {
        matches!(self.space, TerrainEvaluationSpace::Infinite { .. })
    }

    pub const fn project_seed(&self) -> Option<u64> {
        match self.space {
            TerrainEvaluationSpace::Bounded => None,
            TerrainEvaluationSpace::Infinite { project_seed } => Some(project_seed),
        }
    }

    pub const fn evaluation_sample_origin(&self) -> SampleCoord {
        self.spatial.evaluation.origin
    }

    pub fn interior_local_offset(&self) -> Result<(u32, u32), TerrainDomainError> {
        let x = self
            .spatial
            .interior
            .samples
            .origin
            .x
            .checked_sub(self.spatial.evaluation.origin.x)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or(TerrainDomainError::HaloOverflow)?;
        let z = self
            .spatial
            .interior
            .samples
            .origin
            .z
            .checked_sub(self.spatial.evaluation.origin.z)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or(TerrainDomainError::HaloOverflow)?;
        Ok((x, z))
    }

    pub fn local_metrics(&self) -> crate::heightfield::HeightfieldMetrics {
        let (dx, dz) = match self.space {
            TerrainEvaluationSpace::Bounded => (
                self.world.world_size_x / self.world.level_width.max(1) as f32,
                self.world.world_size_z / self.world.level_height.max(1) as f32,
            ),
            TerrainEvaluationSpace::Infinite { .. } => {
                let spacing = self.spatial.interior.transform.spacing();
                (spacing.x_m() as f32, spacing.z_m() as f32)
            }
        };
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

    #[test]
    fn infinite_domain_keeps_signed_lattice_and_unclamped_halos() {
        let topology = InfiniteTopology::try_new(terra_world::InfiniteTopologyConfig {
            origin: terra_world::WorldPosition::try_new(125.0, -75.0).unwrap(),
            tile_size: 256,
            finest_spacing_m: 0.5,
            max_lod: terra_world::Lod::try_new(8).unwrap(),
        })
        .unwrap();
        let key = TerrainTileKey::height(terra_world::TileAddress::new(
            terra_world::Lod::FINEST,
            terra_world::TileCoord { x: -1, z: 2 },
        ));
        let domain = TerrainEvaluationDomain::for_infinite_tile(
            &topology,
            key,
            0x1_0000_0007,
            2,
            7,
            stamp(),
        )
        .unwrap();

        assert!(domain.is_infinite());
        assert_eq!(domain.project_seed(), Some(0x1_0000_0007));
        assert_eq!(domain.spatial.interior.samples.origin.x, -256);
        assert_eq!(domain.spatial.interior.samples.origin.z, 512);
        assert_eq!(domain.evaluation_sample_origin().x, -265);
        assert_eq!(domain.evaluation_sample_origin().z, 503);
        assert_eq!(domain.spatial.evaluation.width, 274);
        assert_eq!(domain.spatial.evaluation.height, 274);
        assert_eq!(domain.interior_local_offset().unwrap(), (9, 9));
        assert_eq!(domain.published_width(), 260);
        assert_eq!(domain.local_metrics().world_size_x, 137.0);
    }
}
