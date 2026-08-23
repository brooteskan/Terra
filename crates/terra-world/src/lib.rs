//! Shared spatial contracts for bounded and sparse terrain worlds.
//!
//! `terra-world` owns fixed-origin CPU coordinates, signed tile addresses, LOD
//! conventions, sample/world transforms, and topology math. It deliberately
//! knows nothing about terrain fields, layers, evaluation, residency, GPU
//! publication, rendering, IO, or application state. Its only external
//! dependency is `serde`, used so a spatial address can be embedded in a
//! persisted content key without making serialization a topology concern.
//!
//! # Conventions and invariants
//!
//! - Authoritative CPU world positions are `f64` metres from one fixed origin.
//! - Tile and global-sample coordinates are signed `i64` values.
//! - LOD 0 is finest; increasing LOD values are coarser.
//! - Rectangles are half-open on their maximum edge.
//! - A sample represents the centre of its cell.
//! - Negative coordinate conversion uses floor/Euclidean behavior, never
//!   truncation toward zero.
//! - Potentially overflowing coordinate, LOD, and halo operations are fallible.
//!
//! `InfiniteTopology` intentionally has no total dimensions, root tile, dense
//! metadata index, or complete-world iterator. Callers may only address a tile
//! directly or request the finite tile range intersecting an explicit rectangle.

mod address;
mod bounded;
mod coordinate;
mod extent;
mod infinite;
mod transform;

pub use address::{Lod, TileAddress, TileAddressRange, TileCoord, MAX_LOD};
pub use bounded::{BoundedLevel, BoundedTopology, BoundedTopologyConfig};
pub use coordinate::{WorldPosition, WorldRect};
pub use extent::{SampleCoord, SampleExtent, SpatialDomain, TileExtent};
pub use infinite::{InfiniteTopology, InfiniteTopologyConfig};
pub use transform::{SampleSpacing, SampleWorldTransform};

use std::fmt;

/// Why a spatial value or conversion violates the shared world contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldError {
    InvalidLod(u8),
    FinestLodHasNoChild,
    CoarsestLodHasNoParent,
    InvalidTileSize(u32),
    InvalidResolution(u32),
    InvalidWorldCoordinate,
    InvalidWorldRect,
    InvalidSpacing,
    InvalidWorldExtent,
    AddressOutsideTopology(TileAddress),
    ArithmeticOverflow,
    HaloOverflow,
}

impl fmt::Display for WorldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLod(lod) => write!(f, "LOD {lod} exceeds the supported maximum {MAX_LOD}"),
            Self::FinestLodHasNoChild => write!(f, "LOD 0 has no finer child"),
            Self::CoarsestLodHasNoParent => write!(f, "the configured coarsest LOD has no parent"),
            Self::InvalidTileSize(size) => write!(f, "tile size must be positive, got {size}"),
            Self::InvalidResolution(value) => {
                write!(f, "resolution must be at least 2, got {value}")
            }
            Self::InvalidWorldCoordinate => write!(f, "world coordinate must be finite"),
            Self::InvalidWorldRect => write!(f, "world rectangle must be finite and non-empty"),
            Self::InvalidSpacing => write!(f, "sample spacing must be finite and positive"),
            Self::InvalidWorldExtent => write!(f, "world extent must be finite and positive"),
            Self::AddressOutsideTopology(address) => {
                write!(f, "address {address:?} is outside the topology")
            }
            Self::ArithmeticOverflow => write!(f, "spatial arithmetic overflow"),
            Self::HaloOverflow => write!(f, "halo arithmetic overflow"),
        }
    }
}

impl std::error::Error for WorldError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_ancestry_is_euclidean_across_zero() {
        assert_eq!(
            TileCoord { x: -1, z: -2 }.euclidean_parent(),
            TileCoord { x: -1, z: -1 }
        );
        assert_eq!(
            TileCoord { x: -1, z: -1 }.checked_children().unwrap(),
            [
                TileCoord { x: -2, z: -2 },
                TileCoord { x: -1, z: -2 },
                TileCoord { x: -2, z: -1 },
                TileCoord { x: -1, z: -1 },
            ]
        );
    }

    #[test]
    fn infinite_world_conversion_uses_floor_for_negative_positions() {
        let topology = InfiniteTopology::try_new(InfiniteTopologyConfig {
            origin: WorldPosition::ORIGIN,
            tile_size: 256,
            finest_spacing_m: 1.0,
            max_lod: Lod::try_new(10).unwrap(),
        })
        .unwrap();
        assert_eq!(
            topology
                .address_at_world(WorldPosition::try_new(-0.001, 0.0).unwrap(), Lod::FINEST)
                .unwrap()
                .coord
                .x,
            -1
        );
        assert_eq!(
            topology
                .address_at_world(WorldPosition::try_new(-256.0, 0.0).unwrap(), Lod::FINEST)
                .unwrap()
                .coord
                .x,
            -1
        );
        assert_eq!(
            topology
                .address_at_world(WorldPosition::try_new(256.0, 0.0).unwrap(), Lod::FINEST)
                .unwrap()
                .coord
                .x,
            1
        );
    }

    #[test]
    fn infinite_halos_expand_without_world_clamping() {
        let topology = InfiniteTopology::try_new(InfiniteTopologyConfig {
            origin: WorldPosition::ORIGIN,
            tile_size: 256,
            finest_spacing_m: 2.0,
            max_lod: Lod::try_new(4).unwrap(),
        })
        .unwrap();
        let address = TileAddress::new(Lod::FINEST, TileCoord::ZERO);
        let domain = topology.spatial_domain(address, 2, 7).unwrap();
        assert_eq!(domain.evaluation.origin, SampleCoord { x: -9, z: -9 });
        assert_eq!(
            (domain.evaluation.width, domain.evaluation.height),
            (274, 274)
        );
        assert_eq!((domain.publish_offset_x, domain.publish_offset_z), (7, 7));
    }

    #[test]
    fn infinite_lod_spacing_bounds_and_ancestry_cross_zero() {
        let topology = InfiniteTopology::try_new(InfiniteTopologyConfig {
            origin: WorldPosition::try_new(100.0, -200.0).unwrap(),
            tile_size: 4,
            finest_spacing_m: 0.5,
            max_lod: Lod::try_new(4).unwrap(),
        })
        .unwrap();
        assert_eq!(topology.spacing(Lod::try_new(3).unwrap()).unwrap().x_m, 4.0);
        let child = TileAddress::new(Lod::try_new(2).unwrap(), TileCoord { x: -1, z: 0 });
        let parent = topology.parent(child).unwrap();
        assert_eq!(
            parent,
            TileAddress::new(Lod::try_new(3).unwrap(), TileCoord { x: -1, z: 0 })
        );
        assert!(topology.children(parent).unwrap().contains(&child));

        let left = topology
            .tile_extent(TileAddress::new(Lod::FINEST, TileCoord { x: -1, z: 0 }))
            .unwrap();
        let right = topology
            .tile_extent(TileAddress::new(Lod::FINEST, TileCoord { x: 0, z: 0 }))
            .unwrap();
        assert_eq!(left.world.max.x_m, right.world.min.x_m);
        assert_eq!(right.world.min.x_m, 100.0);
    }

    #[test]
    fn explicit_finite_range_crosses_negative_and_positive_tiles() {
        let topology = InfiniteTopology::try_new(InfiniteTopologyConfig {
            origin: WorldPosition::ORIGIN,
            tile_size: 4,
            finest_spacing_m: 1.0,
            max_lod: Lod::try_new(2).unwrap(),
        })
        .unwrap();
        let rect = WorldRect::try_new(
            WorldPosition::try_new(-1.0, -1.0).unwrap(),
            WorldPosition::try_new(5.0, 1.0).unwrap(),
        )
        .unwrap();
        let range = topology.addresses_intersecting(rect, Lod::FINEST).unwrap();
        assert_eq!(range.min.coord, TileCoord { x: -1, z: -1 });
        assert_eq!(range.max.coord, TileCoord { x: 1, z: 0 });
        assert_eq!(range.iter().count(), 6);
    }

    #[test]
    fn validated_address_round_trips_through_serde() {
        let address = TileAddress::new(Lod::try_new(7).unwrap(), TileCoord { x: -42, z: 19 });
        let json = serde_json::to_string(&address).unwrap();
        assert_eq!(serde_json::from_str::<TileAddress>(&json).unwrap(), address);
    }

    #[test]
    fn thousands_of_kilometres_retain_sub_metre_precision() {
        let transform = SampleWorldTransform::new(
            WorldPosition::try_new(5_000_000.0, -5_000_000.0).unwrap(),
            SampleSpacing::try_new(0.25, 0.25).unwrap(),
        );
        let a = transform.sample_center(SampleCoord { x: 0, z: 0 }).unwrap();
        let b = transform.sample_center(SampleCoord { x: 1, z: 0 }).unwrap();
        assert_eq!(b.x_m - a.x_m, 0.25);
    }

    #[test]
    fn checked_errors_cover_lod_halo_and_child_overflow() {
        assert_eq!(
            Lod::try_new(MAX_LOD + 1),
            Err(WorldError::InvalidLod(MAX_LOD + 1))
        );
        assert_eq!(
            TileCoord { x: i64::MAX, z: 0 }.checked_children(),
            Err(WorldError::ArithmeticOverflow)
        );
        let extent = SampleExtent {
            origin: SampleCoord { x: 0, z: 0 },
            width: u32::MAX,
            height: 1,
        };
        assert_eq!(extent.checked_expand(1), Err(WorldError::HaloOverflow));
    }
}
