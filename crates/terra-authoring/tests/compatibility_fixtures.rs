//! Structural guardrails for the wire payloads frozen before the authored model moves here.

const LEGACY_BOUNDED: &str = include_str!("fixtures/legacy_bounded_sculpt_params.json");
const WORLD_METRES_V1: &str = include_str!("fixtures/world_metres_v1_sculpt_params.json");

#[test]
fn legacy_bounded_fixture_keeps_the_untagged_uv_contract() {
    let value: serde_json::Value = serde_json::from_str(LEGACY_BOUNDED).unwrap();
    let strokes = value["strokes"].as_array().expect("bounded strokes array");

    assert_eq!(strokes.len(), 2);
    assert!(value.get("world_metres_v1").is_none());
    assert_eq!(strokes[0]["points"][0]["u"], 0.125);
    assert_eq!(strokes[0]["points"][0]["v"], 0.875);
    assert_eq!(strokes[0]["points"][0]["pressure"], 0.5);
}

#[test]
fn world_fixture_keeps_versioned_ids_and_f64_coordinates_without_runtime_state() {
    let value: serde_json::Value = serde_json::from_str(WORLD_METRES_V1).unwrap();
    let strokes = value["world_metres_v1"]
        .as_array()
        .expect("world-metre strokes array");

    assert!(value["strokes"].as_array().unwrap().is_empty());
    assert_eq!(strokes.len(), 2);
    assert_eq!(strokes[0]["id"], "11111111-2222-4333-8444-555555555555");
    assert_eq!(strokes[0]["points"][0]["position"]["x_m"], -0.25);
    assert_eq!(strokes[0]["points"][1]["position"]["x_m"], 0.25);
    assert_eq!(strokes[1]["points"][0]["position"]["x_m"], 4_500_000.125);
    assert_eq!(strokes[1]["points"][0]["position"]["z_m"], -7_250_000.375);
    assert!(!WORLD_METRES_V1.contains("authored_feature_index"));
    assert!(!WORLD_METRES_V1.contains("occupied_cell"));
    assert!(!WORLD_METRES_V1.contains("bounds"));
}
