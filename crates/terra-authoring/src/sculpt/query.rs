use super::{SculptStoreError, WorldSculptStore, WorldSculptStroke};
use std::fmt;
use terra_world::{
    InfiniteTopology, Lod, SpatialIndex, SpatialIndexError, TileAddress, WorldBounds,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SculptQueryError {
    Store(SculptStoreError),
    Index(SpatialIndexError<usize>),
}

impl fmt::Display for SculptQueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => error.fmt(f),
            Self::Index(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for SculptQueryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            Self::Index(error) => Some(error),
        }
    }
}

impl From<SculptStoreError> for SculptQueryError {
    fn from(value: SculptStoreError) -> Self {
        Self::Store(value)
    }
}

impl From<SpatialIndexError<usize>> for SculptQueryError {
    fn from(value: SpatialIndexError<usize>) -> Self {
        Self::Index(value)
    }
}

/// Immutable sparse query adapter over one validated world sculpt store.
///
/// Record ordinals are private index keys, so results retain authored history
/// order even when UUID ordering differs. Borrowing the store immutably prevents
/// the persisted records and derived index from drifting apart.
#[derive(Debug)]
pub struct WorldSculptQuery<'a> {
    store: &'a WorldSculptStore,
    index: SpatialIndex<usize>,
}

impl<'a> WorldSculptQuery<'a> {
    pub fn try_new(
        store: &'a WorldSculptStore,
        topology: InfiniteTopology,
        bucket_lod: Lod,
    ) -> Result<Self, SculptQueryError> {
        let records = store.indexed_ordinals()?;
        let index = SpatialIndex::try_from_records(topology, bucket_lod, records)?;
        Ok(Self { store, index })
    }

    pub fn query_bounds(
        &self,
        bounds: WorldBounds,
    ) -> Result<Vec<&'a WorldSculptStroke>, SculptQueryError> {
        self.index
            .query_bounds(bounds)
            .map(|ordinals| self.resolve(ordinals))
            .map_err(Into::into)
    }

    pub fn query_tile(
        &self,
        address: TileAddress,
    ) -> Result<Vec<&'a WorldSculptStroke>, SculptQueryError> {
        self.index
            .query_tile(address)
            .map(|ordinals| self.resolve(ordinals))
            .map_err(Into::into)
    }

    pub fn query_tile_with_halo(
        &self,
        address: TileAddress,
        halo_samples: u32,
    ) -> Result<Vec<&'a WorldSculptStroke>, SculptQueryError> {
        self.index
            .query_tile_with_halo(address, halo_samples)
            .map(|ordinals| self.resolve(ordinals))
            .map_err(Into::into)
    }

    pub fn indexed_len(&self) -> usize {
        self.index.len()
    }

    pub fn occupied_cell_count(&self) -> usize {
        self.index.occupied_cell_count()
    }

    fn resolve(&self, ordinals: Vec<usize>) -> Vec<&'a WorldSculptStroke> {
        ordinals
            .into_iter()
            .filter_map(|index| self.store.records().get(index))
            .collect()
    }
}
