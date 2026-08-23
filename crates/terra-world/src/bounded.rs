use crate::{
    Lod, SampleCoord, SampleExtent, SampleSpacing, SampleWorldTransform, SpatialDomain,
    TileAddress, TileAddressRange, TileCoord, TileExtent, WorldError, WorldPosition, WorldRect,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundedTopologyConfig {
    pub target_resolution: u32,
    pub world_size_x: f64,
    pub world_size_z: f64,
    pub tile_size: u32,
    pub halo: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundedLevel {
    /// Legacy bounded index: zero is coarsest and indices increase toward fine.
    pub index: u8,
    /// Shared LOD: zero is finest and values increase toward coarse.
    pub lod: Lod,
    pub resolution: u32,
}

/// Finite ceil-halving topology used by Terra's existing bounded pyramid.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedTopology {
    config: BoundedTopologyConfig,
    levels: Vec<BoundedLevel>,
}

impl BoundedTopology {
    pub fn try_new(config: BoundedTopologyConfig) -> Result<Self, WorldError> {
        if config.target_resolution < 2 {
            return Err(WorldError::InvalidResolution(config.target_resolution));
        }
        if config.tile_size == 0 {
            return Err(WorldError::InvalidTileSize(config.tile_size));
        }
        if !config.world_size_x.is_finite()
            || !config.world_size_z.is_finite()
            || config.world_size_x <= 0.0
            || config.world_size_z <= 0.0
        {
            return Err(WorldError::InvalidWorldExtent);
        }
        let mut resolutions = vec![config.target_resolution];
        while resolutions.last().copied().unwrap_or(2) > 2 {
            let child = resolutions.last().copied().unwrap_or(2);
            resolutions.push(child.div_ceil(2).max(2));
        }
        resolutions.reverse();
        let max_level = resolutions.len().saturating_sub(1);
        let levels = resolutions
            .into_iter()
            .enumerate()
            .map(|(index, resolution)| {
                let lod_value = u8::try_from(max_level.saturating_sub(index))
                    .map_err(|_| WorldError::ArithmeticOverflow)?;
                Ok(BoundedLevel {
                    index: u8::try_from(index).map_err(|_| WorldError::ArithmeticOverflow)?,
                    lod: Lod::try_new(lod_value)?,
                    resolution,
                })
            })
            .collect::<Result<Vec<_>, WorldError>>()?;
        let topology = Self { config, levels };
        topology
            .checked_metadata_len()
            .ok_or(WorldError::ArithmeticOverflow)?;
        Ok(topology)
    }

    pub const fn config(&self) -> BoundedTopologyConfig {
        self.config
    }

    pub fn levels(&self) -> &[BoundedLevel] {
        &self.levels
    }

    pub fn max_level(&self) -> u8 {
        self.levels.len().saturating_sub(1) as u8
    }

    pub fn level(&self, index: u8) -> Option<&BoundedLevel> {
        self.levels.get(usize::from(index))
    }

    pub fn level_for_lod(&self, lod: Lod) -> Option<&BoundedLevel> {
        self.levels
            .get(usize::from(self.max_level().checked_sub(lod.get())?))
    }

    pub fn address(&self, level: u8, x: u32, z: u32) -> Result<TileAddress, WorldError> {
        let entry =
            self.level(level)
                .ok_or(WorldError::AddressOutsideTopology(TileAddress::new(
                    Lod::FINEST,
                    TileCoord::ZERO,
                )))?;
        let address = TileAddress::new(
            entry.lod,
            TileCoord {
                x: i64::from(x),
                z: i64::from(z),
            },
        );
        self.tile_extent(address)?;
        Ok(address)
    }

    pub fn level_index(&self, address: TileAddress) -> Option<u8> {
        let level = self.level_for_lod(address.lod)?;
        self.tile_extent(address).ok()?;
        Some(level.index)
    }

    pub fn level_resolution(&self, address: TileAddress) -> Option<u32> {
        Some(self.level_for_lod(address.lod)?.resolution)
    }

    pub fn tiles_x(&self, level: u8) -> Option<u32> {
        let resolution = self.level(level)?.resolution;
        Some(resolution.div_ceil(self.config.tile_size.min(resolution).max(1)))
    }

    pub fn tiles_z(&self, level: u8) -> Option<u32> {
        self.tiles_x(level)
    }

    pub fn spacing(&self, address: TileAddress) -> Result<SampleSpacing, WorldError> {
        let resolution = self
            .level_for_lod(address.lod)
            .ok_or(WorldError::AddressOutsideTopology(address))?
            .resolution;
        SampleSpacing::try_new(
            self.config.world_size_x / f64::from(resolution),
            self.config.world_size_z / f64::from(resolution),
        )
    }

    pub fn tile_extent(&self, address: TileAddress) -> Result<TileExtent, WorldError> {
        let level = self
            .level_for_lod(address.lod)
            .ok_or(WorldError::AddressOutsideTopology(address))?;
        let tile_size = self.config.tile_size.min(level.resolution).max(1);
        let x = u32::try_from(address.coord.x)
            .map_err(|_| WorldError::AddressOutsideTopology(address))?;
        let z = u32::try_from(address.coord.z)
            .map_err(|_| WorldError::AddressOutsideTopology(address))?;
        let tiles = level.resolution.div_ceil(tile_size);
        if x >= tiles || z >= tiles {
            return Err(WorldError::AddressOutsideTopology(address));
        }
        let origin_x = x
            .checked_mul(tile_size)
            .ok_or(WorldError::ArithmeticOverflow)?;
        let origin_z = z
            .checked_mul(tile_size)
            .ok_or(WorldError::ArithmeticOverflow)?;
        let width = (level.resolution - origin_x).min(tile_size);
        let height = (level.resolution - origin_z).min(tile_size);
        let spacing = self.spacing(address)?;
        let transform = SampleWorldTransform::new(WorldPosition::ORIGIN, spacing);
        let min = WorldPosition::try_new(
            f64::from(origin_x) * spacing.x_m(),
            f64::from(origin_z) * spacing.z_m(),
        )?;
        let max = WorldPosition::try_new(
            f64::from(origin_x + width) * spacing.x_m(),
            f64::from(origin_z + height) * spacing.z_m(),
        )?;
        Ok(TileExtent {
            address,
            samples: SampleExtent {
                origin: SampleCoord {
                    x: i64::from(origin_x),
                    z: i64::from(origin_z),
                },
                width,
                height,
            },
            world: WorldRect::try_new(min, max)?,
            transform,
        })
    }

    pub fn spatial_domain(
        &self,
        address: TileAddress,
        publication_halo: u32,
        operation_halo: u32,
    ) -> Result<SpatialDomain, WorldError> {
        let interior = self.tile_extent(address)?;
        let level = self
            .level_for_lod(address.lod)
            .ok_or(WorldError::AddressOutsideTopology(address))?;
        let total = publication_halo
            .checked_add(operation_halo)
            .ok_or(WorldError::HaloOverflow)?;
        let expanded = interior.samples.checked_expand(total)?;
        let origin_x = expanded.origin.x.max(0);
        let origin_z = expanded.origin.z.max(0);
        let end_x = expanded.checked_end_x()?.min(i64::from(level.resolution));
        let end_z = expanded.checked_end_z()?.min(i64::from(level.resolution));
        let evaluation = SampleExtent {
            origin: SampleCoord {
                x: origin_x,
                z: origin_z,
            },
            width: u32::try_from(end_x - origin_x).map_err(|_| WorldError::ArithmeticOverflow)?,
            height: u32::try_from(end_z - origin_z).map_err(|_| WorldError::ArithmeticOverflow)?,
        };
        let publish_origin_x = interior
            .samples
            .origin
            .x
            .saturating_sub(i64::from(publication_halo))
            .max(0);
        let publish_origin_z = interior
            .samples
            .origin
            .z
            .saturating_sub(i64::from(publication_halo))
            .max(0);
        Ok(SpatialDomain {
            interior,
            evaluation,
            publication_halo,
            operation_halo,
            publish_offset_x: u32::try_from(publish_origin_x - origin_x)
                .map_err(|_| WorldError::ArithmeticOverflow)?,
            publish_offset_z: u32::try_from(publish_origin_z - origin_z)
                .map_err(|_| WorldError::ArithmeticOverflow)?,
        })
    }

    pub fn metadata_index(&self, address: TileAddress) -> Option<u32> {
        let level = self.level_for_lod(address.lod)?;
        self.tile_extent(address).ok()?;
        let before =
            self.levels
                .iter()
                .take(usize::from(level.index))
                .try_fold(0u32, |sum, entry| {
                    let tile_size = self.config.tile_size.min(entry.resolution).max(1);
                    sum.checked_add(entry.resolution.div_ceil(tile_size).pow(2))
                })?;
        let tiles_x = self.tiles_x(level.index)?;
        let x = u32::try_from(address.coord.x).ok()?;
        let z = u32::try_from(address.coord.z).ok()?;
        before.checked_add(z.checked_mul(tiles_x)?)?.checked_add(x)
    }

    pub fn metadata_len(&self) -> u32 {
        self.checked_metadata_len()
            .expect("validated bounded topology metadata length")
    }

    fn checked_metadata_len(&self) -> Option<u32> {
        self.levels.iter().try_fold(0u32, |sum, level| {
            let tile_size = self.config.tile_size.min(level.resolution).max(1);
            let tiles = level.resolution.div_ceil(tile_size);
            sum.checked_add(tiles.checked_mul(tiles)?)
        })
    }

    pub fn addresses(&self) -> impl Iterator<Item = TileAddress> + '_ {
        self.levels.iter().flat_map(|level| {
            let tiles = level
                .resolution
                .div_ceil(self.config.tile_size.min(level.resolution).max(1));
            (0..tiles).flat_map(move |z| {
                (0..tiles).map(move |x| {
                    TileAddress::new(
                        level.lod,
                        TileCoord {
                            x: i64::from(x),
                            z: i64::from(z),
                        },
                    )
                })
            })
        })
    }

    pub fn addresses_at_level(&self, level: u8) -> Option<impl Iterator<Item = TileAddress> + '_> {
        let entry = *self.level(level)?;
        let tiles = entry
            .resolution
            .div_ceil(self.config.tile_size.min(entry.resolution).max(1));
        Some((0..tiles).flat_map(move |z| {
            (0..tiles).map(move |x| {
                TileAddress::new(
                    entry.lod,
                    TileCoord {
                        x: i64::from(x),
                        z: i64::from(z),
                    },
                )
            })
        }))
    }

    pub fn covering_parent_tiles(&self, address: TileAddress) -> Option<TileAddressRange> {
        let source_level = self.level_index(address)?;
        if source_level == 0 {
            return None;
        }
        self.covering_tiles_between(address, source_level - 1)
    }

    pub fn covering_child_tiles(&self, address: TileAddress) -> Option<TileAddressRange> {
        let source_level = self.level_index(address)?;
        if source_level >= self.max_level() {
            return None;
        }
        self.covering_tiles_between(address, source_level + 1)
    }

    fn covering_tiles_between(
        &self,
        source_address: TileAddress,
        target_level: u8,
    ) -> Option<TileAddressRange> {
        let source_level = self.level_for_lod(source_address.lod)?;
        let source_resolution = source_level.resolution;
        let target = self.level(target_level)?;
        let extent = self.tile_extent(source_address).ok()?.samples;
        let origin_x = u32::try_from(extent.origin.x).ok()?;
        let origin_z = u32::try_from(extent.origin.z).ok()?;
        let x0 = scaled_floor(origin_x, target.resolution, source_resolution);
        let z0 = scaled_floor(origin_z, target.resolution, source_resolution);
        let x1 = scaled_ceil(
            origin_x.saturating_add(extent.width),
            target.resolution,
            source_resolution,
        )
        .saturating_sub(1)
        .min(target.resolution - 1);
        let z1 = scaled_ceil(
            origin_z.saturating_add(extent.height),
            target.resolution,
            source_resolution,
        )
        .saturating_sub(1)
        .min(target.resolution - 1);
        let target_tile_size = self.config.tile_size.min(target.resolution).max(1);
        TileAddressRange::try_new(
            TileAddress::new(
                target.lod,
                TileCoord {
                    x: i64::from(x0 / target_tile_size),
                    z: i64::from(z0 / target_tile_size),
                },
            ),
            TileAddress::new(
                target.lod,
                TileCoord {
                    x: i64::from(x1 / target_tile_size),
                    z: i64::from(z1 / target_tile_size),
                },
            ),
        )
        .ok()
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

    fn bounded(target_resolution: u32, tile_size: u32) -> BoundedTopology {
        BoundedTopology::try_new(BoundedTopologyConfig {
            target_resolution,
            world_size_x: 4096.0,
            world_size_z: 2048.0,
            tile_size,
            halo: 2,
        })
        .unwrap()
    }

    #[test]
    fn complete_non_power_of_two_ladder_is_preserved() {
        let topology = bounded(1000, 256);
        assert_eq!(
            topology
                .levels()
                .iter()
                .map(|level| level.resolution)
                .collect::<Vec<_>>(),
            vec![2, 4, 8, 16, 32, 63, 125, 250, 500, 1000]
        );
        assert_eq!(topology.level(0).unwrap().lod.get(), topology.max_level());
        assert_eq!(
            topology.level(topology.max_level()).unwrap().lod,
            Lod::FINEST
        );
    }

    #[test]
    fn partial_edge_and_dense_order_are_preserved() {
        let topology = bounded(1000, 256);
        let address = topology.address(topology.max_level(), 3, 3).unwrap();
        let extent = topology.tile_extent(address).unwrap().samples;
        assert_eq!(
            (
                extent.origin.x,
                extent.origin.z,
                extent.width,
                extent.height
            ),
            (768, 768, 232, 232)
        );
        let indices = topology
            .addresses()
            .map(|address| topology.metadata_index(address).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(indices, (0..topology.metadata_len()).collect::<Vec<_>>());
    }

    #[test]
    fn bounded_halo_clips_at_world_edges() {
        let topology = bounded(1000, 256);
        let address = topology.address(topology.max_level(), 3, 3).unwrap();
        let domain = topology.spatial_domain(address, 2, 7).unwrap();
        assert_eq!(domain.evaluation.origin, SampleCoord { x: 759, z: 759 });
        assert_eq!(
            (domain.evaluation.width, domain.evaluation.height),
            (241, 241)
        );
        assert_eq!((domain.publish_offset_x, domain.publish_offset_z), (7, 7));
    }
}
