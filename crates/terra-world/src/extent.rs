use crate::{SampleWorldTransform, TileAddress, WorldError, WorldRect};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleCoord {
    pub x: i64,
    pub z: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleExtent {
    pub origin: SampleCoord,
    pub width: u32,
    pub height: u32,
}

impl SampleExtent {
    pub fn checked_expand(self, halo: u32) -> Result<Self, WorldError> {
        let halo_i64 = i64::from(halo);
        let origin = SampleCoord {
            x: self
                .origin
                .x
                .checked_sub(halo_i64)
                .ok_or(WorldError::HaloOverflow)?,
            z: self
                .origin
                .z
                .checked_sub(halo_i64)
                .ok_or(WorldError::HaloOverflow)?,
        };
        let twice = halo.checked_mul(2).ok_or(WorldError::HaloOverflow)?;
        Ok(Self {
            origin,
            width: self
                .width
                .checked_add(twice)
                .ok_or(WorldError::HaloOverflow)?,
            height: self
                .height
                .checked_add(twice)
                .ok_or(WorldError::HaloOverflow)?,
        })
    }

    pub fn checked_end_x(self) -> Result<i64, WorldError> {
        self.origin
            .x
            .checked_add(i64::from(self.width))
            .ok_or(WorldError::ArithmeticOverflow)
    }

    pub fn checked_end_z(self) -> Result<i64, WorldError> {
        self.origin
            .z
            .checked_add(i64::from(self.height))
            .ok_or(WorldError::ArithmeticOverflow)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileExtent {
    pub address: TileAddress,
    pub samples: SampleExtent,
    pub world: WorldRect,
    pub transform: SampleWorldTransform,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialDomain {
    pub interior: TileExtent,
    pub evaluation: SampleExtent,
    pub publication_halo: u32,
    pub operation_halo: u32,
    pub publish_offset_x: u32,
    pub publish_offset_z: u32,
}
