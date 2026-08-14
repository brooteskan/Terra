use super::*;

#[test]
fn viewport_lighting_round_trips_through_json() {
    let mut doc = TerrainDocument::new_default();
    doc.viewport_lighting = ViewportLighting {
        sun_azimuth_deg: 123.5,
        sun_elevation_deg: 22.0,
        sun_intensity: 1.7,
        exposure: 0.8,
        sky_color: [0.1, 0.2, 0.3],
        ambient_strength: 0.4,
        shadow_strength: 0.9,
        fog_strength: 0.25,
        preset: String::new(),
    };
    let json = doc.to_json().expect("serialize");
    let back = TerrainDocument::from_json(&json).expect("deserialize");
    assert_eq!(back.viewport_lighting, doc.viewport_lighting);
}

#[test]
fn legacy_document_without_lighting_loads_with_default() {
    // Project files saved before viewport_lighting existed must still load,
    // taking the default rather than failing to parse (serde(default)).
    let doc = TerrainDocument::new_default();
    let mut value: serde_json::Value =
        serde_json::from_str(&doc.to_json().expect("serialize")).expect("to value");
    value
        .as_object_mut()
        .expect("json object")
        .remove("viewport_lighting");
    let stripped = serde_json::to_string(&value).expect("reserialize");
    let back = TerrainDocument::from_json(&stripped).expect("load legacy");
    assert_eq!(back.viewport_lighting, ViewportLighting::default());
}

#[test]
fn flat_doc_normalizes_into_wc_tree() {
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Base",
        LayerKind::SculptBase(SculptParams::filled(64, 10.0)),
    ));
    stack.push(Layer::new(
        "Hills",
        LayerKind::NoiseValue(NoiseParams::default()),
    ));
    stack.push(Layer::new(
        "Hydraulic",
        LayerKind::HydraulicErosion(Default::default()),
    ));
    let mut doc = TerrainDocument {
        version: DOCUMENT_VERSION,
        name: "Flat".into(),
        metrics: HeightfieldMetrics::preview_default(),
        preview_resolution: 64,
        export_resolution: 64,
        stack,
        masks: Vec::new(),
        selected: None,
        presets_used: Vec::new(),
        active_biome: None,
        level_steps: Default::default(),
        biome_layers: Vec::new(),
        hole_layers: Vec::new(),
        selected_biome_layer: None,
        blueprint: Default::default(),
        shapes: Default::default(),
        biome_library: Default::default(),
        sparse_paint: Default::default(),
        world_rules: Default::default(),
        simulation_scenarios: Default::default(),
        viewport_lighting: Default::default(),
    };
    doc.normalize_wc_tree();
    assert!(doc.stack.find_category(StackCategory::Shape).is_some());
    let shape = doc.stack.find_category(StackCategory::Shape).unwrap();
    assert!(shape
        .children
        .iter()
        .any(|n| matches!(n, StackNode::Layer(l) if l.common.name == "Hills")));
    let biome = doc.stack.first_biome().unwrap();
    assert!(biome.is_biome());
    let filters = biome.find_section(BiomeSection::Filters).unwrap();
    assert!(filters
        .children
        .iter()
        .any(|n| matches!(n, StackNode::Layer(l) if l.common.name == "Hydraulic")));
    // No loose non-foundation layers at root.
    assert!(doc.stack.nodes.iter().all(|n| match n {
        StackNode::Layer(l) => l.kind.is_sculpt_base(),
        StackNode::Group(g) => g.category.is_some() || g.is_biome(),
    }));
}

#[test]
fn from_flat_layers_builds_wc_tree() {
    let layers = vec![
        Layer::new(
            "Base",
            LayerKind::SculptBase(SculptParams::filled(64, 10.0)),
        ),
        Layer::new("Mounds", LayerKind::NoiseValue(NoiseParams::default())),
        Layer::new("Thermal", LayerKind::ThermalErosion(Default::default())),
    ];
    let doc = TerrainDocument::from_flat_layers(layers);
    assert!(doc
        .stack
        .find_category(StackCategory::Shape)
        .unwrap()
        .children
        .iter()
        .any(|n| matches!(n, StackNode::Layer(l) if l.common.name == "Mounds")));
    let filters = doc
        .stack
        .first_biome()
        .unwrap()
        .find_section(BiomeSection::Filters)
        .unwrap();
    assert!(filters
        .children
        .iter()
        .any(|n| matches!(n, StackNode::Layer(l) if l.common.name == "Thermal")));
}

#[test]
fn default_has_wc_categories_and_biome() {
    let doc = TerrainDocument::new_default();
    assert!(doc.stack.find_category(StackCategory::Shape).is_some());
    assert!(doc.stack.find_category(StackCategory::Simulation).is_some());
    assert!(doc.stack.find_category(StackCategory::Mask).is_some());
    assert!(doc.stack.find_category(StackCategory::Surface).is_some());
    let shape = doc.stack.find_category(StackCategory::Shape).unwrap();
    assert_eq!(shape.children.len(), 1);
    let biome = doc.stack.first_biome().unwrap();
    assert!(biome.is_biome());
    assert!(biome.find_section(BiomeSection::Filters).is_some());
    assert_eq!(doc.version, DOCUMENT_VERSION);
    assert!(!doc.stack.layer_ids().is_empty());
}

#[test]
fn default_roundtrip() {
    let doc = TerrainDocument::new_default();
    let before_ids = doc.stack.layer_ids();
    let json = doc.to_json().unwrap();
    let back = TerrainDocument::from_json(&json).unwrap();
    assert_eq!(back.version, DOCUMENT_VERSION);
    assert!(back.stack.first_biome().is_some());
    assert_eq!(back.stack.layer_ids(), before_ids);
}

fn assert_surface_kind_roundtrips(kind: LayerKind) {
    let mut doc = TerrainDocument::new_default();
    doc.add_layer(Layer::new("Issue 70 surface layer", kind));

    let json = doc.to_json().expect("surface document must serialize");
    assert!(!json.contains("\"min_height\":null"));
    assert!(!json.contains("\"max_height\":null"));

    let loaded = TerrainDocument::from_json(&json).expect("surface document must reload");
    let normalized = loaded
        .to_json()
        .expect("reloaded surface document must serialize");
    assert!(!normalized.contains("\"min_height\":null"));
    assert!(!normalized.contains("\"max_height\":null"));
    TerrainDocument::from_json(&normalized).expect("normalized surface document must reload");
}

#[test]
fn default_materials_save_load_save_roundtrip() {
    assert_surface_kind_roundtrips(LayerKind::Materials(
        crate::layer::MaterialsParams::default(),
    ));
}

#[test]
fn default_biomes_save_load_save_roundtrip() {
    assert_surface_kind_roundtrips(LayerKind::Biomes(crate::layer::BiomesParams::default()));
}

#[test]
fn alpine_demo_roundtrip() {
    let doc = TerrainDocument::alpine_demo();
    let json = doc.to_json().unwrap();
    let back = TerrainDocument::from_json(&json).unwrap();
    assert!(back.validate_dependencies().is_ok());
    let biome = back.stack.first_biome().unwrap();
    assert!(biome.is_biome());
    assert!(biome.find_section(BiomeSection::Filters).is_some());
    let filters = biome.find_section(BiomeSection::Filters).unwrap();
    assert!(!filters.children.is_empty());
}

#[test]
fn future_version_json_is_rejected() {
    let mut doc = TerrainDocument::new_default();
    doc.version = DOCUMENT_VERSION + 1;
    let mut value = serde_json::to_value(&doc).unwrap();
    value["version"] = serde_json::json!(DOCUMENT_VERSION + 1);
    let err = TerrainDocument::from_json(&serde_json::to_string(&value).unwrap()).unwrap_err();
    let message = err.to_string();
    assert!(message.contains(&format!(
        "unsupported document version {}",
        DOCUMENT_VERSION + 1
    )));
    assert!(message.contains(&format!("latest supported {DOCUMENT_VERSION}")));
}

#[test]
fn selected_placement_layer_honors_selection() {
    let mut doc = TerrainDocument::new_default();
    let mut second = crate::biome_paint::BiomeLayer::new("Secondary");
    let second_id = second.id;
    let biome = doc.active_biome.unwrap();
    second.stamp(biome, 0.25, 0.25, 0.1, 1.0, false, 32);
    doc.biome_layers.push(second);
    doc.selected_biome_layer = Some(second_id);
    let layer = doc.selected_placement_layer().unwrap();
    assert_eq!(layer.id, second_id);
    assert!(layer.weight_at(biome, 0.25, 0.25) > 0.5);
}
