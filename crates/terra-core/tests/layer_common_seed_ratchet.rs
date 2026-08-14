//! C2-D1 ratchet: `LayerCommon` must not expose an inert seed override.

use terra_core::document::TerrainDocument;
use terra_core::eval::{EvalContext, StackEvaluator};
use terra_core::heightfield::HeightfieldMetrics;
use terra_core::layer::{Layer, LayerKind, LayerStack, NoiseParams};

#[test]
fn layer_common_serializes_without_seed() {
    let layer = noise_layer(1);
    let value = serde_json::to_value(layer).expect("serialize layer");
    let common = value
        .get("common")
        .and_then(serde_json::Value::as_object)
        .expect("serialized LayerCommon object");

    assert!(
        !common.contains_key("seed"),
        "LayerCommon must not serialize a seed; per-kind parameters are authoritative"
    );
}

#[test]
fn project_with_legacy_common_seed_loads_and_normalizes() {
    let layer = noise_layer(1);
    let layer_id = layer.id();
    let mut legacy = serde_json::to_value(TerrainDocument::from_flat_layers(vec![layer]))
        .expect("serialize compatibility fixture");
    let serialized_id = serde_json::to_value(layer_id).expect("serialize layer id");
    assert!(
        insert_common_seed(&mut legacy, &serialized_id, 2),
        "compatibility fixture must contain the target layer"
    );

    let loaded = TerrainDocument::from_json(
        &serde_json::to_string(&legacy).expect("serialize compatibility fixture"),
    )
    .expect("load project containing legacy LayerCommon.seed");
    let loaded_layer = loaded
        .stack
        .find(layer_id)
        .expect("legacy fixture layer survives normalization");
    let LayerKind::NoiseValue(params) = &loaded_layer.kind else {
        panic!("legacy fixture layer kind changed during normalization");
    };
    assert_eq!(
        params.seed, 1,
        "legacy LayerCommon.seed must not replace the per-kind seed"
    );

    let normalized: serde_json::Value = serde_json::from_str(
        &loaded
            .to_json()
            .expect("serialize normalized current document"),
    )
    .expect("parse normalized document");
    assert!(
        !contains_common_seed(&normalized),
        "saving a legacy project must remove LayerCommon.seed"
    );
}

#[test]
fn per_kind_seed_remains_output_authority() {
    let seed_one = evaluate_noise(1);
    let seed_one_again = evaluate_noise(1);
    let seed_two = evaluate_noise(2);

    assert_eq!(
        seed_one, seed_one_again,
        "the same per-kind seed must remain bit-deterministic"
    );
    assert!(
        seed_one
            .iter()
            .zip(&seed_two)
            .any(|(left, right)| left != right),
        "changing the per-kind seed must change the evaluated field"
    );
}

#[test]
fn layer_common_source_declares_no_seed_field() {
    let source = include_str!("../src/layer/mod.rs");
    let (_, after_name) = source
        .split_once("pub struct LayerCommon")
        .expect("LayerCommon declaration");
    let (declaration, _) = after_name
        .split_once("impl LayerCommon")
        .expect("LayerCommon declaration body");
    let declares_seed = declaration
        .lines()
        .map(str::trim_start)
        .any(|line| line.starts_with("pub seed:"));

    assert!(
        !declares_seed,
        "LayerCommon must not restore an unconsumed seed field"
    );
}

fn noise_layer(seed: u64) -> Layer {
    Layer::new(
        "Noise",
        LayerKind::NoiseValue(NoiseParams {
            seed,
            frequency: 0.05,
            amplitude: 10.0,
            octaves: 1,
            ..NoiseParams::default()
        }),
    )
}

fn evaluate_noise(seed: u64) -> Vec<u32> {
    let metrics = HeightfieldMetrics::new(16, 16, 64.0, 64.0);
    let mut stack = LayerStack::new();
    stack.push(noise_layer(seed));
    let mut evaluator = StackEvaluator::new();
    let mut context = EvalContext::new(metrics);
    evaluator
        .rebuild_all(&stack, &mut context)
        .expect("evaluate noise layer")
        .to_dense()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn insert_common_seed(value: &mut serde_json::Value, id: &serde_json::Value, seed: u64) -> bool {
    match value {
        serde_json::Value::Object(object) => {
            if let Some(common) = object
                .get_mut("common")
                .and_then(serde_json::Value::as_object_mut)
            {
                if common.get("id") == Some(id) {
                    common.insert("seed".into(), serde_json::json!(seed));
                    return true;
                }
            }
            object
                .values_mut()
                .any(|child| insert_common_seed(child, id, seed))
        }
        serde_json::Value::Array(values) => values
            .iter_mut()
            .any(|child| insert_common_seed(child, id, seed)),
        _ => false,
    }
}

fn contains_common_seed(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(object) => {
            object
                .get("common")
                .and_then(serde_json::Value::as_object)
                .is_some_and(|common| common.contains_key("seed"))
                || object.values().any(contains_common_seed)
        }
        serde_json::Value::Array(values) => values.iter().any(contains_common_seed),
        _ => false,
    }
}
