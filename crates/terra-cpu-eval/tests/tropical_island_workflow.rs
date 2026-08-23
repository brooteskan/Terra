//! Paintable Tropical Island workflow regression.

use std::collections::HashMap;
use terra_core::biome_definition::PlacementCombineMode;
use terra_core::mask::bake_mask_assets;
use terra_core::quality::PreviewQuality;
use terra_core::sparse_paint::SparsePaintChannelKey;
use terra_core::world_archetype::tropical_island_world;
use terra_core::TerrainDocument;
use terra_cpu_eval::{EvalContext, StackEvaluator};

#[test]
fn tropical_island_workflow_structure() {
    let mut doc = tropical_island_world(10_000.0, 256);
    assert_eq!(doc.version, terra_core::document::DOCUMENT_VERSION);
    assert_eq!(doc.shapes.shapes.len(), 3);
    assert_eq!(doc.biome_library.definitions.len(), 5);
    assert!(doc.selected_biome_layer.is_some());
    assert!(doc.active_biome.is_some());

    // Selecting a biome definition makes it paintable.
    let forest = doc
        .biome_library
        .definitions
        .iter()
        .find(|d| d.name == "Tropical Forest")
        .expect("forest");
    let biome_id = forest.group_id.expect("instantiated");
    doc.active_biome = Some(biome_id);
    doc.ensure_placement_layer();

    let placement = doc.ensure_placement_layer();
    let key = SparsePaintChannelKey {
        placement_id: placement,
        biome_id,
    };
    doc.sparse_paint
        .stamp_circle(key, 5000.0, 5000.0, 200.0, 1.0, false);
    assert!(doc.sparse_paint.sample(key, 5000.0, 5000.0) > 0.5);

    // Paint also into dense channel for legacy mask bridge.
    if let Some(layer) = doc.selected_placement_layer_mut() {
        layer.stamp(biome_id, 0.5, 0.5, 0.08, 1.0, false, 128);
        assert!(layer.weight_at(biome_id, 0.5, 0.5) > 0.5);
        layer.normalize_all();
    }

    // Shape edit after paint: translate spine and recompile.
    if let Some(spine) = doc
        .shapes
        .shapes
        .iter_mut()
        .find(|s| s.name.contains("Spine"))
    {
        spine.translate_uv(0.05, 0.0);
    }
    doc.compile_shapes_into_stack();
    assert!(doc.shapes.managed_constraints_layer.is_some());

    // Placement combine modes from definitions.
    let beach = doc
        .biome_library
        .definitions
        .iter()
        .find(|d| d.name == "Beach")
        .unwrap();
    assert!(matches!(
        beach.placement.combine,
        PlacementCombineMode::PaintOverridesRules
    ));
    assert!(
        !beach.terrain_layers.is_empty(),
        "beach must have terrain response"
    );
    assert!(
        beach.placement.rules.is_some(),
        "beach must have DistNode rules"
    );

    // Roundtrip preserves World Design fields.
    let json = doc.to_json().unwrap();
    let back = TerrainDocument::from_json(&json).unwrap();
    assert_eq!(back.shapes.shapes.len(), 3);
    assert_eq!(back.biome_library.definitions.len(), 5);
    assert!(!back.sparse_paint.channel_list.is_empty());
}

#[test]
fn tropical_island_evaluates_with_biome_content() {
    let mut doc = tropical_island_world(4096.0, 96);
    // Keep Draft cheap: trim heavy evolution if present.
    for layer in doc.stack.flatten_layers_mut() {
        if let terra_core::LayerKind::LandscapeEvolution(p) = &mut layer.kind {
            p.iterations = p.iterations.min(4);
        }
    }

    let metrics = doc.bounded_settings().unwrap().metrics;
    let mut ctx = EvalContext::new(metrics);
    ctx.quality = PreviewQuality::Draft;
    ctx.mask_assets = doc.masks.clone();
    let seed = terra_core::Heightfield::zeros(metrics);
    ctx.masks = bake_mask_assets(&doc.masks, &seed, metrics, &HashMap::new());
    let mut eval = StackEvaluator::new();
    let height = eval
        .rebuild_all(&doc.stack, &mut ctx)
        .expect("tropical island Draft eval");
    let (min_h, max_h) = height.min_max();
    assert!(
        max_h > min_h + 1.0,
        "expected real relief, got [{min_h}, {max_h}]"
    );

    // Biome groups must carry DistNode rules from definitions.
    let mut with_rules = 0;
    for def in &doc.biome_library.definitions {
        let gid = def.group_id.expect("group linked");
        let group = doc.stack.find_group(gid).expect("group in stack");
        if !group.masks.nodes.is_empty() {
            with_rules += 1;
        }
        assert!(
            !group.children.is_empty() || !def.terrain_layers.is_empty(),
            "{} group should have content",
            def.name
        );
    }
    assert!(with_rules >= 4, "expected DistNode rules on biomes");
}

#[test]
fn selected_placement_not_first_only() {
    let mut doc = tropical_island_world(8192.0, 128);
    let biome = doc.active_biome.unwrap();
    let mut secondary = terra_core::BiomeLayer::new("Secondary Placement");
    let sid = secondary.id;
    secondary.stamp(biome, 0.2, 0.2, 0.1, 1.0, false, 64);
    doc.biome_layers.push(secondary);
    doc.selected_biome_layer = Some(sid);
    let layer = doc.selected_placement_layer().unwrap();
    assert_eq!(layer.id, sid);
}
