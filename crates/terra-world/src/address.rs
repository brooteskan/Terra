use crate::WorldError;
use serde::{Deserialize, Serialize};

/// Maximum supported LOD. Keeping this below the signed integer width makes
/// dyadic coordinate operations and scale construction explicitly checkable.
pub const MAX_LOD: u8 = 62;

/// Validated terrain level-of-detail. LOD 0 is the finest level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct Lod(u8);

impl Lod {
    pub const FINEST: Self = Self(0);

    pub fn try_new(value: u8) -> Result<Self, WorldError> {
        if value > MAX_LOD {
            return Err(WorldError::InvalidLod(value));
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u8 {
        self.0
    }

    pub fn coarser(self) -> Result<Self, WorldError> {
        Self::try_new(
            self.0
                .checked_add(1)
                .ok_or(WorldError::ArithmeticOverflow)?,
        )
    }

    pub fn finer(self) -> Result<Self, WorldError> {
        self.0
            .checked_sub(1)
            .map(Self)
            .ok_or(WorldError::FinestLodHasNoChild)
    }
}

impl TryFrom<u8> for Lod {
    type Error = WorldError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Self::try_new(value)
    }
}

impl From<Lod> for u8 {
    fn from(value: Lod) -> Self {
        value.0
    }
}

/// Signed, wide tile coordinate in X/Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TileCoord {
    pub x: i64,
    pub z: i64,
}

impl TileCoord {
    pub const ZERO: Self = Self { x: 0, z: 0 };

    pub fn euclidean_parent(self) -> Self {
        Self {
            x: self.x.div_euclid(2),
            z: self.z.div_euclid(2),
        }
    }

    pub fn checked_children(self) -> Result<[Self; 4], WorldError> {
        let x = self
            .x
            .checked_mul(2)
            .ok_or(WorldError::ArithmeticOverflow)?;
        let z = self
            .z
            .checked_mul(2)
            .ok_or(WorldError::ArithmeticOverflow)?;
        let x1 = x.checked_add(1).ok_or(WorldError::ArithmeticOverflow)?;
        let z1 = z.checked_add(1).ok_or(WorldError::ArithmeticOverflow)?;
        Ok([
            Self { x, z },
            Self { x: x1, z },
            Self { x, z: z1 },
            Self { x: x1, z: z1 },
        ])
    }
}

/// Topology-independent spatial identity of one terrain tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TileAddress {
    pub lod: Lod,
    pub coord: TileCoord,
}

impl TileAddress {
    pub const fn new(lod: Lod, coord: TileCoord) -> Self {
        Self { lod, coord }
    }
}

/// Inclusive rectangular range of addresses at one LOD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileAddressRange {
    pub min: TileAddress,
    pub max: TileAddress,
}

impl TileAddressRange {
    pub fn try_new(min: TileAddress, max: TileAddress) -> Result<Self, WorldError> {
        if min.lod != max.lod || min.coord.x > max.coord.x || min.coord.z > max.coord.z {
            return Err(WorldError::AddressOutsideTopology(min));
        }
        Ok(Self { min, max })
    }

    pub fn iter(self) -> TileAddressRangeIter {
        TileAddressRangeIter {
            range: self,
            next: Some(self.min.coord),
        }
    }
}

pub struct TileAddressRangeIter {
    range: TileAddressRange,
    next: Option<TileCoord>,
}

impl Iterator for TileAddressRangeIter {
    type Item = TileAddress;

    fn next(&mut self) -> Option<Self::Item> {
        let coord = self.next?;
        self.next = if coord.x < self.range.max.coord.x {
            Some(TileCoord {
                x: coord.x + 1,
                z: coord.z,
            })
        } else if coord.z < self.range.max.coord.z {
            Some(TileCoord {
                x: self.range.min.coord.x,
                z: coord.z + 1,
            })
        } else {
            None
        };
        Some(TileAddress::new(self.range.min.lod, coord))
    }
}
