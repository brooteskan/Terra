use crate::{
    Lod, SampleCoord, SampleExtent, SampleSpacing, SampleWorldTransform, SpatialDomain,
    TileAddress, TileAddressRange, TileCoord, TileExtent, WorldError, WorldPosition, WorldRect,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InfiniteTopologyConfig {
    pub origin: WorldPosition,
    pub tile_size: u32,
    pub finest_spacing_m: f64,
    pub max_lod: Lod,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InfiniteTopology {
    config: InfiniteTopologyConfig,
}

impl InfiniteTopology {
    pub fn try_new(config: InfiniteTopologyConfig) -> Result<Self, WorldError> {
        if config.tile_size == 0 {
            return Err(WorldError::InvalidTileSize(config.tile_size));
        }
        SampleSpacing::try_new(config.finest_spacing_m, config.finest_spacing_m)?;
        WorldPosition::try_new(config.origin.x_m, config.origin.z_m)?;
        Ok(Self { config })
    }

    pub const fn config(&self) -> InfiniteTopologyConfig {
        self.config
    }

    pub fn spacing(&self, lod: Lod) -> Result<SampleSpacing, WorldError> {
        self.validate_lod(lod)?;
        let scale = 2.0f64.powi(i32::from(lod.get()));
        SampleSpacing::try_new(
            self.config.finest_spacing_m * scale,
            self.config.finest_spacing_m * scale,
        )
    }

    pub fn parent(&self, address: TileAddress) -> Result<TileAddress, WorldError> {
        self.validate_address(address)?;
        if address.lod == self.config.max_lod {
            return Err(WorldError::CoarsestLodHasNoParent);
        }
        Ok(TileAddress::new(
            address.lod.coarser()?,
            address.coord.euclidean_parent(),
        ))
    }

    pub fn children(&self, address: TileAddress) -> Result<[TileAddress; 4], WorldError> {
        self.validate_address(address)?;
        let lod = address.lod.finer()?;
        Ok(address
            .coord
            .checked_children()?
            .map(|coord| TileAddress::new(lod, coord)))
    }

    pub fn tile_extent(&self, address: TileAddress) -> Result<TileExtent, WorldError> {
        self.validate_address(address)?;
        let tile_size = i64::from(self.config.tile_size);
        let origin = SampleCoord {
            x: address
                .coord
                .x
                .checked_mul(tile_size)
                .ok_or(WorldError::ArithmeticOverflow)?,
            z: address
                .coord
                .z
                .checked_mul(tile_size)
                .ok_or(WorldError::ArithmeticOverflow)?,
        };
        let spacing = self.spacing(address.lod)?;
        let transform = SampleWorldTransform::new(self.config.origin, spacing);
        let min = WorldPosition::try_new(
            self.config.origin.x_m + origin.x as f64 * spacing.x_m,
            self.config.origin.z_m + origin.z as f64 * spacing.z_m,
        )?;
        let max = WorldPosition::try_new(
            min.x_m + f64::from(self.config.tile_size) * spacing.x_m,
            min.z_m + f64::from(self.config.tile_size) * spacing.z_m,
        )?;
        Ok(TileExtent {
            address,
            samples: SampleExtent {
                origin,
                width: self.config.tile_size,
                height: self.config.tile_size,
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
        let total = publication_halo
            .checked_add(operation_halo)
            .ok_or(WorldError::HaloOverflow)?;
        Ok(SpatialDomain {
            evaluation: interior.samples.checked_expand(total)?,
            interior,
            publication_halo,
            operation_halo,
            publish_offset_x: operation_halo,
            publish_offset_z: operation_halo,
        })
    }

    pub fn address_at_world(
        &self,
        world: WorldPosition,
        lod: Lod,
    ) -> Result<TileAddress, WorldError> {
        let spacing = self.spacing(lod)?;
        let tile_span_x = spacing.x_m * f64::from(self.config.tile_size);
        let tile_span_z = spacing.z_m * f64::from(self.config.tile_size);
        let x = ((world.x_m - self.config.origin.x_m) / tile_span_x).floor();
        let z = ((world.z_m - self.config.origin.z_m) / tile_span_z).floor();
        Ok(TileAddress::new(
            lod,
            TileCoord {
                x: f64_to_i64(x)?,
                z: f64_to_i64(z)?,
            },
        ))
    }

    pub fn addresses_intersecting(
        &self,
        rect: WorldRect,
        lod: Lod,
    ) -> Result<TileAddressRange, WorldError> {
        let min = self.address_at_world(rect.min, lod)?;
        // Rectangles are half-open. Move one representable value inward so an
        // exact maximum tile boundary does not include the following tile.
        let max_x = next_down(rect.max.x_m);
        let max_z = next_down(rect.max.z_m);
        let max = self.address_at_world(WorldPosition::try_new(max_x, max_z)?, lod)?;
        TileAddressRange::try_new(min, max)
    }

    fn validate_lod(&self, lod: Lod) -> Result<(), WorldError> {
        if lod > self.config.max_lod {
            return Err(WorldError::AddressOutsideTopology(TileAddress::new(
                lod,
                TileCoord::ZERO,
            )));
        }
        Ok(())
    }

    fn validate_address(&self, address: TileAddress) -> Result<(), WorldError> {
        self.validate_lod(address.lod)
    }
}

fn f64_to_i64(value: f64) -> Result<i64, WorldError> {
    if !value.is_finite()
        || !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&value)
    {
        return Err(WorldError::ArithmeticOverflow);
    }
    Ok(value as i64)
}

fn next_down(value: f64) -> f64 {
    if value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}
