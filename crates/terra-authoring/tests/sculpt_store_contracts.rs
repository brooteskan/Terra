use terra_authoring::sculpt::{
    BoundedSculptStore, SculptHistory, SculptPoint, SculptStoreError, SculptStroke,
    SculptStrokeKind, WorldSculptPoint, WorldSculptQuery, WorldSculptStore, WorldSculptStroke,
};
use terra_authoring::AuthoredFeatureId;
use terra_world::{
    InfiniteTopology, InfiniteTopologyConfig, Lod, TileAddress, TileCoord, WorldBounds,
    WorldPosition,
};
use uuid::Uuid;

fn id(value: u128) -> AuthoredFeatureId {
    AuthoredFeatureId(Uuid::from_u128(value))
}

fn point(x_m: f64, z_m: f64) -> WorldPosition {
    WorldPosition::try_new(x_m, z_m).unwrap()
}

fn bounds(min_x: f64, min_z: f64, max_x: f64, max_z: f64) -> WorldBounds {
    WorldBounds::try_new(point(min_x, min_z), point(max_x, max_z)).unwrap()
}

fn world_record(
    feature_id: AuthoredFeatureId,
    kind: SculptStrokeKind,
    x_m: f64,
    z_m: f64,
    radius_m: f32,
) -> WorldSculptStroke {
    WorldSculptStroke {
        id: feature_id,
        kind,
        points: vec![WorldSculptPoint {
            position: point(x_m, z_m),
            pressure: 1.0,
        }],
        radius_m,
        strength: 4.0,
        target_height: 0.0,
        falloff: 1.5,
        enabled: true,
    }
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

#[test]
fn bounded_store_exposes_ordered_narrow_mutations() {
    let first = SculptStroke {
        kind: SculptStrokeKind::Raise,
        points: vec![SculptPoint {
            u: 0.1,
            v: 0.2,
            pressure: 0.5,
        }],
        radius_m: 20.0,
        ..Default::default()
    };
    let continuation = SculptStroke {
        points: vec![SculptPoint {
            u: 0.3,
            v: 0.4,
            pressure: 1.0,
        }],
        ..first.clone()
    };
    let mut store = BoundedSculptStore::default();

    assert_eq!(store.append_or_extend(first, false), 0);
    assert_eq!(store.append_or_extend(continuation, true), 0);
    assert_eq!(store.len(), 1);
    assert_eq!(store.get(0).unwrap().points.len(), 2);
    assert_eq!(store.set_enabled(0, false), Some(true));

    let replacement = SculptStroke {
        kind: SculptStrokeKind::Smooth,
        ..Default::default()
    };
    assert_eq!(
        store.replace(0, replacement.clone()).unwrap().kind,
        SculptStrokeKind::Raise
    );
    assert_eq!(store.get(0), Some(&replacement));
    assert_eq!(store.delete(0), Some(replacement));
    assert!(store.is_empty());
}

#[test]
fn world_mutations_report_consistent_old_and_new_active_bounds() {
    let feature_id = id(10);
    let mut store = WorldSculptStore::default();
    let created = store
        .append(world_record(
            feature_id,
            SculptStrokeKind::Flatten,
            -10.0,
            5.0,
            4.0,
        ))
        .unwrap();
    assert_eq!(created.previous_bounds, None);
    assert_eq!(
        created.replacement_bounds,
        Some(bounds(-14.0, 1.0, -6.0, 9.0))
    );

    let continued = world_record(id(999), SculptStrokeKind::Flatten, 10.0, -5.0, 4.0);
    let extended = store.append_or_extend(continued, true).unwrap();
    assert_eq!(extended.id, feature_id);
    assert_eq!(extended.previous_bounds, created.replacement_bounds);
    assert_eq!(
        extended.replacement_bounds,
        Some(bounds(-14.0, -9.0, 14.0, 9.0))
    );

    let mut replacement = store.get(feature_id).unwrap().clone();
    replacement.id = id(404);
    replacement.radius_m = 8.0;
    replacement.strength = 12.0;
    let replaced = store.replace(feature_id, replacement).unwrap().unwrap();
    assert_eq!(replaced.previous_bounds, extended.replacement_bounds);
    assert_eq!(
        replaced.replacement_bounds,
        Some(bounds(-18.0, -13.0, 18.0, 13.0))
    );
    assert_eq!(store.get(feature_id).unwrap().strength, 12.0);

    let moved = store.translate(feature_id, 25.0, -40.0).unwrap().unwrap();
    assert_eq!(moved.previous_bounds, replaced.replacement_bounds);
    assert_eq!(
        moved.replacement_bounds,
        Some(bounds(7.0, -53.0, 43.0, -27.0))
    );

    let disabled = store.set_enabled(feature_id, false).unwrap().unwrap();
    assert_eq!(disabled.previous_bounds, moved.replacement_bounds);
    assert_eq!(disabled.replacement_bounds, None);
    assert!(store.index_records().unwrap().is_empty());

    let enabled = store.set_enabled(feature_id, true).unwrap().unwrap();
    assert_eq!(enabled.previous_bounds, None);
    assert_eq!(enabled.replacement_bounds, moved.replacement_bounds);

    let deleted = store.delete(feature_id).unwrap().unwrap();
    assert_eq!(deleted.index, 0);
    assert_eq!(deleted.record.id, feature_id);
    assert_eq!(deleted.change.previous_bounds, moved.replacement_bounds);
    assert_eq!(deleted.change.replacement_bounds, None);
    assert!(store.is_empty());
}

#[test]
fn failed_world_mutations_are_atomic() {
    let feature_id = id(20);
    let mut store = WorldSculptStore::try_from_records(
        vec![world_record(
            feature_id,
            SculptStrokeKind::Raise,
            f64::MAX,
            0.0,
            1.0,
        )],
        0.2,
    )
    .unwrap();

    let before = store.clone();
    let mut invalid = store.get(feature_id).unwrap().clone();
    invalid.radius_m = 0.0;
    assert!(store.replace(feature_id, invalid).is_err());
    assert_eq!(store, before);

    assert!(store.translate(feature_id, f64::MAX, 0.0).is_err());
    assert_eq!(store, before);

    let duplicate = store.get(feature_id).unwrap().clone();
    assert_eq!(
        store.append(duplicate),
        Err(SculptStoreError::DuplicateFeatureId(feature_id))
    );
    assert_eq!(store, before);
}

#[test]
fn construction_rejects_duplicate_ids_and_invalid_disabled_records() {
    let duplicate = world_record(id(30), SculptStrokeKind::Raise, 0.0, 0.0, 2.0);
    assert_eq!(
        WorldSculptStore::try_from_records(vec![duplicate.clone(), duplicate], 0.2),
        Err(SculptStoreError::DuplicateFeatureId(id(30)))
    );

    let mut invalid_disabled = world_record(id(31), SculptStrokeKind::Raise, 0.0, 0.0, 0.0);
    invalid_disabled.enabled = false;
    assert!(WorldSculptStore::try_from_records(vec![invalid_disabled], 0.2).is_err());
}

#[test]
fn duplicate_and_merge_reseed_world_ids_without_reordering_records() {
    let destination = WorldSculptStore::try_from_records(
        vec![world_record(id(40), SculptStrokeKind::Raise, 0.0, 0.0, 1.0)],
        0.2,
    )
    .unwrap();
    let source = WorldSculptStore::try_from_records(
        vec![
            world_record(id(41), SculptStrokeKind::Valley, 10.0, 0.0, 1.0),
            world_record(id(42), SculptStrokeKind::Smooth, 20.0, 0.0, 1.0),
        ],
        0.3,
    )
    .unwrap();

    let reseeded = source.reseeded_clone();
    assert_eq!(
        reseeded
            .iter()
            .map(|record| record.kind)
            .collect::<Vec<_>>(),
        vec![SculptStrokeKind::Valley, SculptStrokeKind::Smooth]
    );
    assert!(reseeded
        .iter()
        .zip(source.iter())
        .all(|(copy, original)| copy.id != original.id));

    let mut merged = destination;
    let changes = merged.merge(&source).unwrap();
    assert_eq!(changes.len(), 2);
    assert_eq!(
        merged.iter().map(|record| record.kind).collect::<Vec<_>>(),
        vec![
            SculptStrokeKind::Raise,
            SculptStrokeKind::Valley,
            SculptStrokeKind::Smooth
        ]
    );
    assert!(merged
        .iter()
        .skip(1)
        .zip(source.iter())
        .all(|(copy, original)| {
            copy.id != original.id && copy.kind == original.kind && copy.points == original.points
        }));
}

#[test]
fn cross_coordinate_merge_is_rejected_without_mutation() {
    let mut bounded = SculptHistory::empty_bounded();
    bounded
        .as_bounded_mut()
        .unwrap()
        .append(SculptStroke::default());
    let before = bounded.clone();

    assert_eq!(
        bounded.merge(&SculptHistory::empty_world()),
        Err(SculptStoreError::CoordinateSpaceMismatch)
    );
    assert_eq!(bounded, before);
}

#[test]
fn spatial_queries_are_sparse_active_only_and_preserve_history_order() {
    let high_id = id(u128::MAX - 1);
    let low_id = id(1);
    let mut disabled = world_record(id(2), SculptStrokeKind::Lower, 0.0, 0.0, 0.5);
    disabled.enabled = false;
    let empty = WorldSculptStroke {
        id: id(3),
        points: Vec::new(),
        ..WorldSculptStroke::default()
    };
    let store = WorldSculptStore::try_from_records(
        vec![
            world_record(high_id, SculptStrokeKind::Raise, 0.0, 0.0, 0.5),
            world_record(low_id, SculptStrokeKind::Smooth, 0.25, 0.0, 0.5),
            disabled,
            empty,
            world_record(id(4), SculptStrokeKind::Valley, 1_000_000.0, 0.0, 0.5),
        ],
        0.2,
    )
    .unwrap();
    let query = WorldSculptQuery::try_new(&store, topology(), Lod::FINEST).unwrap();

    assert_eq!(query.indexed_len(), 3);
    let found = query.query_bounds(bounds(-1.0, -1.0, 1.0, 1.0)).unwrap();
    assert_eq!(
        found.iter().map(|record| record.id).collect::<Vec<_>>(),
        vec![high_id, low_id]
    );
    assert!(query.occupied_cell_count() < 12);
}

#[test]
fn tile_and_halo_queries_return_world_records() {
    let near = id(50);
    let far = id(51);
    let store = WorldSculptStore::try_from_records(
        vec![
            world_record(near, SculptStrokeKind::Raise, 4.5, 2.0, 0.1),
            world_record(far, SculptStrokeKind::Raise, 5.5, 2.0, 0.1),
        ],
        0.2,
    )
    .unwrap();
    let query = WorldSculptQuery::try_new(&store, topology(), Lod::FINEST).unwrap();
    let tile = TileAddress::new(Lod::FINEST, TileCoord::ZERO);

    assert!(query.query_tile(tile).unwrap().is_empty());
    assert_eq!(
        query
            .query_tile_with_halo(tile, 1)
            .unwrap()
            .iter()
            .map(|record| record.id)
            .collect::<Vec<_>>(),
        vec![near]
    );
}
