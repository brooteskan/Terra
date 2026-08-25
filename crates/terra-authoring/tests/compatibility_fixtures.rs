//! Typed compatibility tests for the wire payloads frozen before extraction.

use terra_authoring::sculpt::{SculptHistory, SculptStrokeKind};

const LEGACY_BOUNDED: &str = include_str!("fixtures/legacy_bounded_sculpt_params.json");
const WORLD_METRES_V1: &str = include_str!("fixtures/world_metres_v1_sculpt_params.json");

#[test]
fn legacy_bounded_fixture_round_trips_without_coordinate_reinterpretation() {
    let history: SculptHistory = serde_json::from_str(LEGACY_BOUNDED).unwrap();
    let store = history.as_bounded().expect("bounded history");

    assert_eq!(store.len(), 2);
    assert_eq!(store.reconcile(), 0.2);
    assert_eq!(store.get(0).unwrap().kind, SculptStrokeKind::Terrace);
    assert_eq!(store.get(0).unwrap().points[0].u, 0.125);
    assert_eq!(store.get(0).unwrap().points[0].v, 0.875);
    assert_eq!(store.get(0).unwrap().points[0].pressure, 0.5);

    let saved = serde_json::to_string(&history).unwrap();
    let saved_value: serde_json::Value = serde_json::from_str(&saved).unwrap();
    assert!(saved_value.get("world_metres_v1").is_none());
    assert_eq!(
        serde_json::from_str::<SculptHistory>(&saved).unwrap(),
        history
    );
}

#[test]
fn world_fixture_round_trips_ids_and_f64_coordinates_without_runtime_state() {
    let history: SculptHistory = serde_json::from_str(WORLD_METRES_V1).unwrap();
    let store = history.as_world().expect("world history");
    let records = store.iter().collect::<Vec<_>>();

    assert_eq!(store.len(), 2);
    assert_eq!(store.reconcile(), 0.3);
    assert_eq!(
        records[0].id.0.to_string(),
        "11111111-2222-4333-8444-555555555555"
    );
    assert_eq!(records[0].points[0].position.x_m(), -0.25);
    assert_eq!(records[0].points[1].position.x_m(), 0.25);
    assert_eq!(records[1].points[0].position.x_m(), 4_500_000.125);
    assert_eq!(records[1].points[0].position.z_m(), -7_250_000.375);

    let saved = serde_json::to_string(&history).unwrap();
    let saved_value: serde_json::Value = serde_json::from_str(&saved).unwrap();
    assert_eq!(saved_value["strokes"], serde_json::json!([]));
    assert_eq!(saved_value["world_metres_v1"].as_array().unwrap().len(), 2);
    assert!(!saved.contains("bounds"));
    assert!(!saved.contains("index"));
    assert!(!saved.contains("occupied_cell"));
    assert_eq!(
        serde_json::from_str::<SculptHistory>(&saved).unwrap(),
        history
    );
}

#[test]
fn explicit_empty_world_field_selects_world_but_legacy_empty_remains_bounded() {
    let world: SculptHistory =
        serde_json::from_str(r#"{"strokes":[],"world_metres_v1":[],"reconcile":0.2}"#).unwrap();
    assert!(world.as_world().is_some());
    assert!(serde_json::to_string(&world)
        .unwrap()
        .contains(r#""world_metres_v1":[]"#));

    let legacy: SculptHistory = serde_json::from_str(r#"{"strokes":[],"reconcile":0.2}"#).unwrap();
    assert!(legacy.as_bounded().is_some());
    assert!(legacy.is_empty());
}

#[test]
fn mixed_non_empty_coordinate_spaces_are_rejected() {
    let bounded: serde_json::Value = serde_json::from_str(LEGACY_BOUNDED).unwrap();
    let world: serde_json::Value = serde_json::from_str(WORLD_METRES_V1).unwrap();
    let mixed = serde_json::json!({
        "strokes": bounded["strokes"],
        "world_metres_v1": world["world_metres_v1"],
        "reconcile": 0.2
    });

    assert!(serde_json::from_value::<SculptHistory>(mixed).is_err());
}
