//! Persisted document-version compatibility contract.

use terra_core::document::{TerrainDocument, DOCUMENT_VERSION};
use terra_core::layer::{
    BiomeSection, BiomesParams, Layer, LayerId, LayerKind, MaterialsParams, StackNode,
    OPEN_HEIGHT_MAX, OPEN_HEIGHT_MIN,
};
use terra_core::mask::{MaskCombine, MaskId, MaskSource};
use terra_core::world_rules::{WorldRuleEffectKind, WorldRulePhase, WorldRuleScope};
use uuid::Uuid;

// Emitted by TerrainDocument::to_json at audited commit 7b336c4.
const V1_FIXTURE: &str = include_str!("fixtures/document_v1_7b336c4.json");
// Emitted by TerrainDocument::to_json at the authoring rewrite commit 8587974.
const V1_REWRITE_FIXTURE: &str = include_str!("fixtures/document_v1_8587974.json");
// Emitted by TerrainDocument::to_json at historical commit b450e72.
const V2_FIXTURE: &str = include_str!("fixtures/document_v2_b450e72.json");
// Full-feature document emitted by TerrainDocument::to_json at e4ea27f.
const CURRENT_FULL_FIXTURE: &str = include_str!("fixtures/document_v2_e4ea27f_full.json");
// Material/biome bounds emitted as null by serde_json at audited commit 7b336c4.
const NULL_HEIGHT_BOUNDS_FIXTURE: &str =
    include_str!("fixtures/document_v1_null_height_bounds_7b336c4.json");

fn layer_id(value: &str) -> LayerId {
    LayerId(Uuid::parse_str(value).expect("valid fixture layer id"))
}

fn mask_id(value: &str) -> MaskId {
    MaskId(Uuid::parse_str(value).expect("valid fixture mask id"))
}

fn assert_fixture_semantics(doc: &TerrainDocument, expected_name: &str) {
    let base_id = layer_id("11111111-1111-4111-8111-111111111111");
    let hills_id = layer_id("22222222-2222-4222-8222-222222222222");
    let height_mask_id = mask_id("33333333-3333-4333-8333-333333333333");

    assert_eq!(doc.version, DOCUMENT_VERSION);
    assert_eq!(doc.name, expected_name);
    assert_eq!(doc.selected, Some(hills_id));
    assert_eq!(doc.presets_used, ["Fixture Preset"]);

    let base = doc.stack.find(base_id).expect("legacy base survives");
    let LayerKind::SculptBase(params) = &base.kind else {
        panic!("legacy base kind changed");
    };
    assert_eq!(params.resolution, 2);
    assert_eq!(params.samples, [12.5; 4]);
    assert_eq!(params.fill_height, 12.5);

    let hills = doc.stack.find(hills_id).expect("legacy hills survive");
    assert_eq!(hills.common.opacity, 0.625);
    let LayerKind::NoiseValue(params) = &hills.kind else {
        panic!("legacy hills kind changed");
    };
    assert_eq!(params.seed, 4242);
    assert_eq!(params.frequency, 0.0075);
    assert_eq!(params.amplitude, 137.25);
    assert_eq!(params.octaves, 3);
    assert_eq!(params.lacunarity, 2.25);
    assert_eq!(params.persistence, 0.45);
    assert_eq!(params.offset_x, 11.0);
    assert_eq!(params.offset_z, -7.5);
    assert_eq!(params.remap_min, -0.75);
    assert_eq!(params.remap_max, 0.9);

    let entry = hills
        .common
        .masks
        .entries
        .first()
        .expect("legacy mask reference survives");
    assert_eq!(hills.common.masks.entries.len(), 1);
    assert_eq!(entry.mask.id, height_mask_id);
    assert_eq!(entry.mask.strength, 0.7);
    assert!(entry.mask.invert);
    assert_eq!(entry.combine, MaskCombine::Multiply);

    assert_eq!(doc.masks.len(), 1);
    let mask = doc
        .masks
        .iter()
        .find(|mask| mask.id == height_mask_id)
        .expect("legacy mask asset survives");
    assert_eq!(mask.name, "Legacy Height Mask");
    assert!(matches!(
        mask.source,
        MaskSource::Height {
            min: 25.0,
            max: 275.0
        }
    ));
}

fn assert_fixture_round_trip(raw: &str, source_version: u32, expected_name: &str) {
    let raw_value: serde_json::Value = serde_json::from_str(raw).expect("valid fixture JSON");
    assert_eq!(raw_value["version"], source_version);

    let loaded = TerrainDocument::from_json(raw).expect("fixture must load");
    assert_fixture_semantics(&loaded, expected_name);
    let normalized_layer_ids = loaded.stack.layer_ids();

    let saved = loaded.to_json().expect("normalized fixture must save");
    let saved_value: serde_json::Value = serde_json::from_str(&saved).expect("valid saved JSON");
    assert_eq!(saved_value["version"], DOCUMENT_VERSION);
    assert!(DOCUMENT_VERSION >= source_version);
    assert_required_height_bounds_are_numbers(&saved_value);

    let reloaded = TerrainDocument::from_json(&saved).expect("saved fixture must reload");
    assert_fixture_semantics(&reloaded, expected_name);
    assert_eq!(reloaded.stack.layer_ids(), normalized_layer_ids);
}

fn assert_required_height_bounds_are_numbers(value: &serde_json::Value) {
    fn visit(value: &serde_json::Value, path: &str) {
        match value {
            serde_json::Value::Object(object) => {
                for (key, child) in object {
                    let child_path = format!("{path}.{key}");
                    if matches!(key.as_str(), "min_height" | "max_height") {
                        assert!(
                            child.is_number(),
                            "required numeric height bound at {child_path} was {child}"
                        );
                    }
                    visit(child, &child_path);
                }
            }
            serde_json::Value::Array(values) => {
                for (index, child) in values.iter().enumerate() {
                    visit(child, &format!("{path}[{index}]"));
                }
            }
            _ => {}
        }
    }

    visit(value, "document");
}

fn assert_current_fixture_semantics(doc: &TerrainDocument) {
    let biome_id = layer_id("44444444-4444-4444-8444-444444444444");
    let materials_id = layer_id("55555555-5555-4555-8555-555555555555");
    let biomes_id = layer_id("66666666-6666-4666-8666-666666666666");
    let height_mask_id = mask_id("33333333-3333-4333-8333-333333333333");
    let noise_mask_id = mask_id("77777777-7777-4777-8777-777777777777");

    assert_eq!(doc.version, DOCUMENT_VERSION);
    assert_eq!(doc.name, "Current full feature compatibility fixture");
    assert_eq!(doc.selected, Some(materials_id));
    assert_eq!(doc.active_biome, Some(biome_id));
    assert_eq!(doc.presets_used, ["Issue 71 Full Feature"]);

    let biome = doc
        .stack
        .find_group(biome_id)
        .expect("fixture biome survives");
    let materials_section = biome
        .find_section(BiomeSection::Materials)
        .expect("fixture materials section survives");
    let authored_children: Vec<_> = materials_section
        .children
        .iter()
        .filter_map(|node| match node {
            StackNode::Layer(layer) if matches!(layer.id(), id if id == materials_id || id == biomes_id) => {
                Some(layer.id())
            }
            _ => None,
        })
        .collect();
    assert_eq!(authored_children, [materials_id, biomes_id]);

    let materials = doc
        .stack
        .find(materials_id)
        .expect("materials layer survives");
    let LayerKind::Materials(material_params) = &materials.kind else {
        panic!("fixture materials kind changed");
    };
    assert!(!material_params.rules.is_empty());
    assert_eq!(material_params.rules[0].min_height, OPEN_HEIGHT_MIN);
    assert_eq!(material_params.rules[0].max_height, OPEN_HEIGHT_MAX);
    assert_eq!(materials.common.masks.entries.len(), 2);
    assert_eq!(materials.common.masks.entries[0].mask.id, height_mask_id);
    assert_eq!(materials.common.masks.entries[0].mask.strength, 0.8);
    assert_eq!(
        materials.common.masks.entries[0].combine,
        MaskCombine::Multiply
    );
    assert_eq!(materials.common.masks.entries[1].mask.id, noise_mask_id);
    assert_eq!(materials.common.masks.entries[1].mask.strength, 0.35);
    assert!(materials.common.masks.entries[1].mask.invert);
    assert_eq!(materials.common.masks.entries[1].combine, MaskCombine::Add);

    let biomes = doc.stack.find(biomes_id).expect("biomes layer survives");
    let LayerKind::Biomes(biome_params) = &biomes.kind else {
        panic!("fixture biomes kind changed");
    };
    assert!(!biome_params.bands.is_empty());
    assert_eq!(biome_params.bands[0].max_height, OPEN_HEIGHT_MAX);

    assert_eq!(doc.masks.len(), 2);
    assert!(doc.masks.iter().any(|asset| {
        asset.id == height_mask_id
            && matches!(
                asset.source,
                MaskSource::Height {
                    min: 40.0,
                    max: 420.0
                }
            )
    }));
    assert!(doc.masks.iter().any(|asset| {
        asset.id == noise_mask_id
            && matches!(
                asset.source,
                MaskSource::Noise {
                    seed: 8181,
                    frequency: 0.0125
                }
            )
    }));

    let rule = doc.world_rules.rules.first().expect("world rule survives");
    assert_eq!(rule.name, "Fixture Snow Rule");
    assert_eq!(rule.priority, 7);
    assert_eq!(rule.phase_override, Some(WorldRulePhase::Materials));
    assert!(matches!(
        rule.scope,
        WorldRuleScope::PaintedRestriction {
            paint_mask: Some(id)
        } if id == height_mask_id
    ));
    assert!(matches!(
        rule.effects.as_slice(),
        [effect] if effect.kind == WorldRuleEffectKind::Material && effect.strength == 0.65
    ));

    assert_eq!(doc.viewport_lighting.sun_azimuth_deg, 123.0);
    assert_eq!(doc.viewport_lighting.sun_elevation_deg, 37.0);
    assert_eq!(doc.viewport_lighting.sky_color, [0.12, 0.18, 0.31]);
    assert_eq!(doc.viewport_lighting.preset, "Issue 71 Fixture");
}

#[test]
fn original_version_2_writer_fixture_loads_and_round_trips() {
    assert_fixture_round_trip(V2_FIXTURE, 2, "Version 2 compatibility fixture");
}

#[test]
fn regressed_version_1_writer_fixture_loads_and_round_trips() {
    assert_fixture_round_trip(V1_FIXTURE, 1, "Version 1 compatibility fixture");
}

#[test]
fn rewrite_8587974_fixture_loads_normalizes_and_round_trips() {
    assert_fixture_round_trip(V1_REWRITE_FIXTURE, 1, "Version 1 rewrite fixture");
}

#[test]
fn current_full_feature_fixture_preserves_authored_semantics() {
    let raw_value: serde_json::Value =
        serde_json::from_str(CURRENT_FULL_FIXTURE).expect("valid current fixture");
    assert_eq!(raw_value["version"], 2);
    assert_required_height_bounds_are_numbers(&raw_value);

    let loaded = TerrainDocument::from_json(CURRENT_FULL_FIXTURE).expect("current fixture loads");
    assert_current_fixture_semantics(&loaded);
    let authored_order = loaded.stack.layer_ids();

    let saved = loaded.to_json().expect("current fixture saves");
    let saved_value: serde_json::Value = serde_json::from_str(&saved).expect("valid saved JSON");
    assert_eq!(saved_value["version"], DOCUMENT_VERSION);
    assert_required_height_bounds_are_numbers(&saved_value);

    let reloaded = TerrainDocument::from_json(&saved).expect("current fixture reloads");
    assert_current_fixture_semantics(&reloaded);
    assert_eq!(reloaded.stack.layer_ids(), authored_order);
}

#[test]
fn historical_fixtures_default_new_additive_fields() {
    for raw in [V2_FIXTURE, V1_REWRITE_FIXTURE] {
        let loaded = TerrainDocument::from_json(raw).expect("historical fixture loads");
        assert_eq!(
            loaded.viewport_lighting,
            terra_core::document::ViewportLighting::default()
        );
    }
}

#[test]
fn default_materials_and_biomes_remain_writer_safe() {
    let mut doc = TerrainDocument::new_default();
    doc.add_layer(Layer::new(
        "Compatibility Materials",
        LayerKind::Materials(MaterialsParams::default()),
    ));
    doc.add_layer(Layer::new(
        "Compatibility Biomes",
        LayerKind::Biomes(BiomesParams::default()),
    ));

    let saved = doc.to_json().expect("default surface layers save");
    let value: serde_json::Value = serde_json::from_str(&saved).expect("valid saved JSON");
    assert_required_height_bounds_are_numbers(&value);
    let reloaded = TerrainDocument::from_json(&saved).expect("default surface layers reload");
    reloaded
        .to_json()
        .expect("default surface layers save after reload");
}

#[test]
fn legacy_null_material_and_biome_bounds_normalize_on_load() {
    let loaded = TerrainDocument::from_json(NULL_HEIGHT_BOUNDS_FIXTURE)
        .expect("legacy null height bounds must load");
    let mut saw_materials = false;
    let mut saw_biomes = false;

    for layer in loaded.stack.flatten_layers() {
        match &layer.kind {
            LayerKind::Materials(params) if layer.common.name == "Legacy Materials" => {
                saw_materials = true;
                assert_eq!(params.rules[0].min_height, OPEN_HEIGHT_MIN);
                assert_eq!(params.rules[0].max_height, OPEN_HEIGHT_MAX);
            }
            LayerKind::Biomes(params) if layer.common.name == "Legacy Biomes" => {
                saw_biomes = true;
                assert_eq!(params.bands[0].min_height, OPEN_HEIGHT_MIN);
                assert_eq!(params.bands[0].max_height, OPEN_HEIGHT_MAX);
            }
            _ => {}
        }
    }
    assert!(saw_materials && saw_biomes);

    let saved = loaded.to_json().expect("normalized document must save");
    assert!(!saved.contains("\"min_height\":null"));
    assert!(!saved.contains("\"max_height\":null"));
    let value: serde_json::Value = serde_json::from_str(&saved).expect("valid normalized JSON");
    assert_eq!(value["version"], DOCUMENT_VERSION);
    TerrainDocument::from_json(&saved).expect("normalized document must reload");
}

#[test]
fn writer_never_emits_a_version_lower_than_2() {
    let mut doc = TerrainDocument::from_json(V1_FIXTURE).expect("version 1 fixture must load");
    doc.version = 1;
    let saved = doc.to_json().expect("document must save");
    let value: serde_json::Value = serde_json::from_str(&saved).expect("valid saved JSON");
    assert_eq!(value["version"], DOCUMENT_VERSION);
    assert!(DOCUMENT_VERSION >= 2);
}

#[test]
fn genuinely_future_version_is_rejected_clearly() {
    let mut value: serde_json::Value = serde_json::from_str(V2_FIXTURE).expect("valid fixture");
    value["version"] = serde_json::json!(DOCUMENT_VERSION + 1);
    let error = TerrainDocument::from_json(&serde_json::to_string(&value).unwrap())
        .expect_err("future version must be rejected");
    let message = error.to_string();
    assert!(message.contains(&format!(
        "unsupported document version {}",
        DOCUMENT_VERSION + 1
    )));
    assert!(message.contains(&format!("latest supported {DOCUMENT_VERSION}")));
}
