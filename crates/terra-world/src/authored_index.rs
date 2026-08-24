use crate::{InfiniteTopology, Lod, TileAddress, TileCoord, WorldBounds, WorldError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use uuid::Uuid;

/// Stable identity shared by a serialized authored record and its runtime index entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AuthoredFeatureId(pub Uuid);

impl AuthoredFeatureId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for AuthoredFeatureId {
    fn default() -> Self {
        Self::new()
    }
}

/// Why an authored-feature index mutation or query failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureIndexError {
    World(WorldError),
    DuplicateFeature(AuthoredFeatureId),
    UnknownFeature(AuthoredFeatureId),
    CapacityOverflow,
}

impl fmt::Display for FeatureIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::World(error) => error.fmt(f),
            Self::DuplicateFeature(id) => write!(f, "authored feature {id:?} is already indexed"),
            Self::UnknownFeature(id) => write!(f, "authored feature {id:?} is not indexed"),
            Self::CapacityOverflow => write!(f, "authored-feature cell membership is too large"),
        }
    }
}

impl std::error::Error for FeatureIndexError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::World(error) => Some(error),
            Self::DuplicateFeature(_) | Self::UnknownFeature(_) | Self::CapacityOverflow => None,
        }
    }
}

impl From<WorldError> for FeatureIndexError {
    fn from(value: WorldError) -> Self {
        Self::World(value)
    }
}

#[derive(Debug)]
struct IndexedFeature {
    bounds: WorldBounds,
    cells: Vec<TileCoord>,
}

/// Runtime-only sparse lookup from finite authored bounds to Infinite-world tiles.
///
/// Only occupied cells are stored. Queries range over those occupied keys, then
/// filter candidates against their exact closed bounds. The index deliberately
/// has no serialization or complete-world enumeration API; callers rebuild it
/// from source feature records.
#[derive(Debug)]
pub struct AuthoredFeatureIndex {
    topology: InfiniteTopology,
    bucket_lod: Lod,
    entries: BTreeMap<AuthoredFeatureId, IndexedFeature>,
    cells: BTreeMap<i64, BTreeMap<i64, BTreeSet<AuthoredFeatureId>>>,
}

impl AuthoredFeatureIndex {
    pub fn try_new(topology: InfiniteTopology, bucket_lod: Lod) -> Result<Self, FeatureIndexError> {
        topology.spacing(bucket_lod)?;
        Ok(Self {
            topology,
            bucket_lod,
            entries: BTreeMap::new(),
            cells: BTreeMap::new(),
        })
    }

    pub fn try_from_records(
        topology: InfiniteTopology,
        bucket_lod: Lod,
        records: impl IntoIterator<Item = (AuthoredFeatureId, WorldBounds)>,
    ) -> Result<Self, FeatureIndexError> {
        let mut index = Self::try_new(topology, bucket_lod)?;
        for (id, bounds) in records {
            index.insert(id, bounds)?;
        }
        Ok(index)
    }

    pub fn insert(
        &mut self,
        id: AuthoredFeatureId,
        bounds: WorldBounds,
    ) -> Result<(), FeatureIndexError> {
        if self.entries.contains_key(&id) {
            return Err(FeatureIndexError::DuplicateFeature(id));
        }
        let cells = self.cells_for_bounds(bounds)?;
        self.add_memberships(id, &cells);
        self.entries.insert(id, IndexedFeature { bounds, cells });
        Ok(())
    }

    pub fn update(
        &mut self,
        id: AuthoredFeatureId,
        bounds: WorldBounds,
    ) -> Result<(), FeatureIndexError> {
        if !self.entries.contains_key(&id) {
            return Err(FeatureIndexError::UnknownFeature(id));
        }
        let cells = self.cells_for_bounds(bounds)?;
        let previous = self
            .entries
            .remove(&id)
            .ok_or(FeatureIndexError::UnknownFeature(id))?;
        self.remove_memberships(id, &previous.cells);
        self.add_memberships(id, &cells);
        self.entries.insert(id, IndexedFeature { bounds, cells });
        Ok(())
    }

    pub fn remove(&mut self, id: AuthoredFeatureId) -> Result<WorldBounds, FeatureIndexError> {
        let previous = self
            .entries
            .remove(&id)
            .ok_or(FeatureIndexError::UnknownFeature(id))?;
        self.remove_memberships(id, &previous.cells);
        Ok(previous.bounds)
    }

    /// Atomically replace runtime state with an index rebuilt from source records.
    pub fn rebuild(
        &mut self,
        records: impl IntoIterator<Item = (AuthoredFeatureId, WorldBounds)>,
    ) -> Result<(), FeatureIndexError> {
        let replacement = Self::try_from_records(self.topology, self.bucket_lod, records)?;
        *self = replacement;
        Ok(())
    }

    pub fn query_bounds(
        &self,
        bounds: WorldBounds,
    ) -> Result<Vec<AuthoredFeatureId>, FeatureIndexError> {
        let range = self.topology.addresses_touching(bounds, self.bucket_lod)?;
        let mut candidates = BTreeSet::new();
        for (_, z_cells) in self.cells.range(range.min.coord.x..=range.max.coord.x) {
            for (_, ids) in z_cells.range(range.min.coord.z..=range.max.coord.z) {
                candidates.extend(ids.iter().copied());
            }
        }
        Ok(candidates
            .into_iter()
            .filter(|id| {
                self.entries
                    .get(id)
                    .is_some_and(|entry| entry.bounds.intersects(bounds))
            })
            .collect())
    }

    pub fn query_tile(
        &self,
        address: TileAddress,
    ) -> Result<Vec<AuthoredFeatureId>, FeatureIndexError> {
        self.query_tile_with_halo(address, 0)
    }

    pub fn query_tile_with_halo(
        &self,
        address: TileAddress,
        halo_samples: u32,
    ) -> Result<Vec<AuthoredFeatureId>, FeatureIndexError> {
        let extent = self.topology.tile_extent(address)?;
        let spacing = self.topology.spacing(address.lod)?;
        let bounds = WorldBounds::try_new(extent.world.min(), extent.world.max())?;
        let pad_x = spacing.x_m() * f64::from(halo_samples);
        let pad_z = spacing.z_m() * f64::from(halo_samples);
        self.query_bounds(bounds.checked_expand(pad_x, pad_z)?)
    }

    pub fn bounds(&self, id: AuthoredFeatureId) -> Option<WorldBounds> {
        self.entries.get(&id).map(|entry| entry.bounds)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn occupied_cell_count(&self) -> usize {
        self.cells.values().map(BTreeMap::len).sum()
    }

    fn cells_for_bounds(&self, bounds: WorldBounds) -> Result<Vec<TileCoord>, FeatureIndexError> {
        let range = self.topology.addresses_touching(bounds, self.bucket_lod)?;
        let count = range.checked_len()?;
        let mut cells = Vec::new();
        cells
            .try_reserve_exact(count)
            .map_err(|_| FeatureIndexError::CapacityOverflow)?;
        cells.extend(range.iter().map(|address| address.coord));
        Ok(cells)
    }

    fn add_memberships(&mut self, id: AuthoredFeatureId, cells: &[TileCoord]) {
        for cell in cells {
            self.cells
                .entry(cell.x)
                .or_default()
                .entry(cell.z)
                .or_default()
                .insert(id);
        }
    }

    fn remove_memberships(&mut self, id: AuthoredFeatureId, cells: &[TileCoord]) {
        for cell in cells {
            let remove_x = if let Some(z_cells) = self.cells.get_mut(&cell.x) {
                let remove_z = if let Some(ids) = z_cells.get_mut(&cell.z) {
                    ids.remove(&id);
                    ids.is_empty()
                } else {
                    false
                };
                if remove_z {
                    z_cells.remove(&cell.z);
                }
                z_cells.is_empty()
            } else {
                false
            };
            if remove_x {
                self.cells.remove(&cell.x);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InfiniteTopologyConfig, TileAddress, TileCoord, WorldPosition};

    fn point(x: f64, z: f64) -> WorldPosition {
        WorldPosition::try_new(x, z).unwrap()
    }

    fn bounds(min_x: f64, min_z: f64, max_x: f64, max_z: f64) -> WorldBounds {
        WorldBounds::try_new(point(min_x, min_z), point(max_x, max_z)).unwrap()
    }

    fn topology() -> InfiniteTopology {
        InfiniteTopology::try_new(InfiniteTopologyConfig {
            origin: WorldPosition::ORIGIN,
            tile_size: 4,
            finest_spacing_m: 1.0,
            max_lod: Lod::try_new(3).unwrap(),
        })
        .unwrap()
    }

    fn id(value: u128) -> AuthoredFeatureId {
        AuthoredFeatureId(Uuid::from_u128(value))
    }

    #[test]
    fn queries_filter_bucket_candidates_and_deduplicate_multicell_features() {
        let mut index = AuthoredFeatureIndex::try_new(topology(), Lod::FINEST).unwrap();
        index.insert(id(1), bounds(-1.0, -1.0, 5.0, 1.0)).unwrap();
        index.insert(id(2), bounds(3.0, 3.0, 3.5, 3.5)).unwrap();
        index
            .insert(id(3), bounds(100.0, 100.0, 101.0, 101.0))
            .unwrap();

        assert_eq!(
            index.query_bounds(bounds(-2.0, -0.5, 4.5, 0.5)).unwrap(),
            vec![id(1)]
        );
        assert_eq!(index.len(), 3);
        assert!(index.occupied_cell_count() < 20);
    }

    #[test]
    fn seam_features_are_returned_for_both_adjacent_tile_domains() {
        let mut index = AuthoredFeatureIndex::try_new(topology(), Lod::FINEST).unwrap();
        index
            .insert(id(1), WorldBounds::from_point(point(0.0, 1.0)))
            .unwrap();
        let left = TileAddress::new(Lod::FINEST, TileCoord { x: -1, z: 0 });
        let right = TileAddress::new(Lod::FINEST, TileCoord { x: 0, z: 0 });
        assert_eq!(index.query_tile(left).unwrap(), vec![id(1)]);
        assert_eq!(index.query_tile(right).unwrap(), vec![id(1)]);
    }

    #[test]
    fn tile_halo_finds_only_features_in_the_expanded_domain() {
        let mut index = AuthoredFeatureIndex::try_new(topology(), Lod::FINEST).unwrap();
        index
            .insert(id(1), WorldBounds::from_point(point(4.5, 2.0)))
            .unwrap();
        index
            .insert(id(2), WorldBounds::from_point(point(5.5, 2.0)))
            .unwrap();
        let tile = TileAddress::new(Lod::FINEST, TileCoord::ZERO);
        assert!(index.query_tile(tile).unwrap().is_empty());
        assert_eq!(index.query_tile_with_halo(tile, 1).unwrap(), vec![id(1)]);
    }

    #[test]
    fn requested_lod_controls_tile_extent_and_halo_spacing() {
        let mut index = AuthoredFeatureIndex::try_new(topology(), Lod::FINEST).unwrap();
        index
            .insert(id(1), WorldBounds::from_point(point(10.0, 2.0)))
            .unwrap();
        let coarse_tile = TileAddress::new(Lod::try_new(1).unwrap(), TileCoord::ZERO);
        assert!(index.query_tile(coarse_tile).unwrap().is_empty());
        assert_eq!(
            index.query_tile_with_halo(coarse_tile, 1).unwrap(),
            vec![id(1)]
        );
    }

    #[test]
    fn update_remove_and_failed_rebuild_leave_no_stale_entries() {
        let mut index = AuthoredFeatureIndex::try_new(topology(), Lod::FINEST).unwrap();
        index.insert(id(1), bounds(-3.0, 1.0, -2.0, 2.0)).unwrap();
        index.update(id(1), bounds(20.0, 1.0, 21.0, 2.0)).unwrap();
        assert!(index
            .query_bounds(bounds(-4.0, 0.0, 0.0, 4.0))
            .unwrap()
            .is_empty());
        assert_eq!(
            index.query_bounds(bounds(19.0, 0.0, 22.0, 4.0)).unwrap(),
            vec![id(1)]
        );

        let before = index.occupied_cell_count();
        assert_eq!(
            index.rebuild([
                (id(2), bounds(0.0, 0.0, 1.0, 1.0)),
                (id(2), bounds(2.0, 2.0, 3.0, 3.0)),
            ]),
            Err(FeatureIndexError::DuplicateFeature(id(2)))
        );
        assert_eq!(index.occupied_cell_count(), before);
        assert_eq!(index.bounds(id(1)), Some(bounds(20.0, 1.0, 21.0, 2.0)));

        index.remove(id(1)).unwrap();
        assert!(index.is_empty());
        assert_eq!(index.occupied_cell_count(), 0);
    }

    #[test]
    fn sparse_storage_does_not_depend_on_distance_between_features() {
        let mut index =
            AuthoredFeatureIndex::try_new(topology(), Lod::try_new(3).unwrap()).unwrap();
        index
            .insert(id(1), bounds(-1_000_001.0, 0.5, -1_000_000.5, 1.0))
            .unwrap();
        index
            .insert(id(2), bounds(1_000_000.5, 0.5, 1_000_001.0, 1.0))
            .unwrap();
        assert_eq!(index.len(), 2);
        assert_eq!(index.occupied_cell_count(), 2);
        assert!(index
            .query_bounds(bounds(-1.0, -1.0, 1.0, 1.0))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn mutation_errors_preserve_existing_state() {
        let mut index = AuthoredFeatureIndex::try_new(topology(), Lod::FINEST).unwrap();
        index.insert(id(1), bounds(1.0, 1.0, 2.0, 2.0)).unwrap();
        assert_eq!(
            index.insert(id(1), bounds(5.0, 5.0, 6.0, 6.0)),
            Err(FeatureIndexError::DuplicateFeature(id(1)))
        );
        assert_eq!(
            index.update(id(2), bounds(5.0, 5.0, 6.0, 6.0)),
            Err(FeatureIndexError::UnknownFeature(id(2)))
        );
        assert_eq!(
            index.remove(id(2)),
            Err(FeatureIndexError::UnknownFeature(id(2)))
        );
        let address_overflow =
            WorldBounds::from_point(point(-9_223_372_036_854_775_808.0 * 4.0, 1.0));
        assert_eq!(
            index.update(id(1), address_overflow),
            Err(FeatureIndexError::World(WorldError::ArithmeticOverflow))
        );
        assert_eq!(
            index.query_bounds(bounds(0.0, 0.0, 3.0, 3.0)).unwrap(),
            vec![id(1)]
        );
    }

    #[test]
    fn identity_round_trips_and_successful_rebuild_replaces_runtime_state() {
        let feature_id = id(0x1234);
        let json = serde_json::to_string(&feature_id).unwrap();
        assert_eq!(
            serde_json::from_str::<AuthoredFeatureId>(&json).unwrap(),
            feature_id
        );

        let mut index = AuthoredFeatureIndex::try_new(topology(), Lod::FINEST).unwrap();
        index.insert(id(1), bounds(0.5, 0.5, 1.0, 1.0)).unwrap();
        index
            .rebuild([
                (id(2), bounds(-10.0, -2.0, -9.0, -1.0)),
                (id(3), bounds(20.0, 2.0, 21.0, 3.0)),
            ])
            .unwrap();
        assert_eq!(index.len(), 2);
        assert_eq!(index.bounds(id(1)), None);
        assert_eq!(
            index.query_bounds(bounds(19.0, 1.0, 22.0, 4.0)).unwrap(),
            vec![id(3)]
        );
    }
}
