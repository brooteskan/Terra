//! Backend-neutral contracts for persisted authored-terrain data.
//!
//! `terra-authoring` sits between [`terra_world`] and `terra-core`: it may use
//! world coordinates, bounds, topology, tile addressing, and generic spatial
//! indexing, but it must not depend on documents, layers, evaluation backends,
//! rendering, IO, or application state. Authored identities, persisted feature
//! stores, and their mutation and query contracts will move here incrementally.
//!
//! The live sculpt representation intentionally remains in `terra-core` during
//! the first extraction session. This crate initially establishes and tests the
//! dependency seam without changing runtime behavior.

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};
    use terra_world::{
        InfiniteTopology, InfiniteTopologyConfig, Lod, SpatialIndex, WorldBounds, WorldPosition,
    };
    use uuid::Uuid;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    #[serde(transparent)]
    struct PrototypeFeatureId(Uuid);

    #[test]
    fn authored_identity_composes_with_the_generic_world_index() {
        let topology = InfiniteTopology::try_new(InfiniteTopologyConfig {
            origin: WorldPosition::ORIGIN,
            tile_size: 4,
            finest_spacing_m: 1.0,
            max_lod: Lod::try_new(2).unwrap(),
        })
        .unwrap();
        let id = PrototypeFeatureId(Uuid::from_u128(0x221));
        let bounds = WorldBounds::from_point(WorldPosition::try_new(-0.25, 0.5).unwrap());
        let mut index = SpatialIndex::try_new(topology, Lod::FINEST).unwrap();

        index.insert(id, bounds).unwrap();

        assert_eq!(index.query_bounds(bounds).unwrap(), vec![id]);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(
            serde_json::from_str::<PrototypeFeatureId>(&json).unwrap(),
            id
        );
    }
}
