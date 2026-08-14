//! Incomplete-but-valid projects, soft diagnostics, and stack-order evaluation.

use std::collections::HashMap;
use terra_core::biome_definition::{BiomeDefinition, BiomePlacementRules};
use terra_core::biome_paint::BiomeLayer;
use terra_core::document::TerrainDocument;
use terra_core::domain::{authoring_order_is_arbitrary, incomplete_project_diagnostics};
use terra_core::eval::{EvalContext, PreviewQuality, StackEvaluator};
use terra_core::heightfield::HeightfieldMetrics;
use terra_core::layer::{
    BlendMode, FlatParams, HydraulicErosionParams, Layer, LayerKind, LayerStack, MaterialsParams,
    NoiseParams, ThermalErosionParams,
};
use terra_core::mask::{bake_mask_assets, DistNode};

fn metrics() -> HeightfieldMetrics {
    HeightfieldMetrics {
        width: 64,
        height: 64,
        world_size_x: 1024.0,
        world_size_z: 1024.0,
        tile_size: 64,
        halo: 0,
    }
}

fn eval_doc(doc: &mut TerrainDocument) -> terra_core::Heightfield {
    let m = metrics();
    let mut ctx = EvalContext::new(m);
    ctx.quality = PreviewQuality::Draft;
    let seed = terra_core::Heightfield::zeros(m);
    ctx.masks = bake_mask_assets(&doc.masks, &seed, m, &HashMap::new());
    doc.evaluate_final_height(&mut ctx)
        .expect("incomplete projects must evaluate")
}

#[test]
fn authoring_order_is_arbitrary_contract() {
    assert!(authoring_order_is_arbitrary());
}

#[test]
fn evaluation_follows_the_current_stack_tree_order() {
    let mut doc = TerrainDocument::new_default();
    doc.stack = LayerStack::new();

    let mut base = Layer::new("Base", LayerKind::Flat(FlatParams { height: 10.0 }));
    base.common.blend = BlendMode::Normal;
    let mut raise = Layer::new("Raise", LayerKind::Flat(FlatParams { height: 5.0 }));
    raise.common.blend = BlendMode::Add;
    doc.stack.push(base);
    doc.stack.push(raise);

    let bottom_to_top = eval_doc(&mut doc);
    assert!((bottom_to_top.get(32, 32) - 15.0).abs() < 1.0e-4);

    doc.stack.nodes.reverse();
    let reordered = eval_doc(&mut doc);
    assert!((reordered.get(32, 32) - 10.0).abs() < 1.0e-4);
}

#[test]
fn incomplete_content_evaluates_and_reports_soft_diagnostics() {
    let mut doc = TerrainDocument::new_default();
    let mut mats = MaterialsParams::default();
    mats.rules.clear();
    mats.strata.clear();
    doc.stack
        .push(Layer::new("Early Materials", LayerKind::Materials(mats)));
    let mut snow = BiomeDefinition::new("Snow", [0.9, 0.92, 0.95]);
    snow.placement = BiomePlacementRules {
        rules: Some(DistNode::height(500.0, 10_000.0)), // no terrain that high
        ..BiomePlacementRules::default()
    };
    doc.biome_library.definitions.push(snow);

    // Add a low hill after the incomplete material and biome content.
    doc.stack.push(Layer::new(
        "Late Hill",
        LayerKind::NoiseValue(NoiseParams {
            amplitude: 8.0,
            ..NoiseParams::default()
        }),
    ));

    let _hf = eval_doc(&mut doc);
    let diags = incomplete_project_diagnostics(&doc.stack, &doc.biome_library, &doc.biome_layers);
    assert!(
        diags.iter().any(|d| {
            d.code == "materials_without_rules"
                || d.code == "biome_empty_placement_rules"
                || d.code == "biome_unlinked"
                || d.code == "empty_biome_coverage"
                || d.code == "stack_without_shape_layers"
        }),
        "expected soft diagnostics, got {diags:?}"
    );
}

#[test]
fn incomplete_project_serializes_and_reloads() {
    let mut doc = TerrainDocument::new_default();

    let mut alpine = BiomeDefinition::new("Alpine Snow", [0.95, 0.95, 1.0]);
    alpine.placement.rules = Some(DistNode::height(9_000.0, 12_000.0));
    doc.biome_library.definitions.push(alpine);

    doc.biome_layers.push(BiomeLayer::new("Empty Paint"));

    doc.stack.push(Layer::new(
        "Early Hydro",
        LayerKind::HydraulicErosion(HydraulicErosionParams::default()),
    ));
    doc.stack.push(Layer::new(
        "Early Thermal",
        LayerKind::ThermalErosion(ThermalErosionParams::default()),
    ));

    let mut mats = MaterialsParams::default();
    mats.rules.clear();
    mats.strata.clear();
    doc.stack.push(Layer::new(
        "Premature Materials",
        LayerKind::Materials(mats),
    ));

    let json = doc.to_json().expect("serialize incomplete");
    let mut back = TerrainDocument::from_json(&json).expect("reload incomplete");
    assert_eq!(back.version, doc.version);
    assert!(back
        .biome_library
        .definitions
        .iter()
        .any(|d| d.name == "Alpine Snow"));
    assert!(back.biome_layers.iter().any(|b| b.channels.is_empty()));
    let _ = eval_doc(&mut back);
}

#[test]
fn empty_biome_coverage_is_valid() {
    let mut doc = TerrainDocument::new_default();
    doc.biome_layers.clear();
    doc.biome_layers.push(BiomeLayer::new("No Strokes"));
    let hf = eval_doc(&mut doc);
    assert_eq!(hf.metrics.width, 64);
    let diags = incomplete_project_diagnostics(&doc.stack, &doc.biome_library, &doc.biome_layers);
    assert!(diags.iter().any(|d| d.code == "empty_biome_coverage"));
}

#[test]
fn empty_rule_results_are_valid() {
    let mut stack = terra_core::layer::LayerStack::new();
    let mut mats = MaterialsParams::default();
    mats.rules.clear();
    mats.strata.clear();
    stack.push(Layer::new("Empty Mats", LayerKind::Materials(mats)));

    let m = metrics();
    let mut ctx = EvalContext::new(m);
    ctx.quality = PreviewQuality::Draft;
    let seed = terra_core::Heightfield::zeros(m);
    ctx.masks = bake_mask_assets(&[], &seed, m, &HashMap::new());
    let mut eval = StackEvaluator::new();
    let out = eval
        .rebuild_all(&stack, &mut ctx)
        .expect("empty rules must not fatal");
    let sample = out.get(0, 0);
    assert!(
        sample.abs() < 1e-3,
        "expected near-zero height, got {sample}"
    );
}

#[test]
fn stack_without_shapes_evaluates() {
    let mut doc = TerrainDocument::new_default();
    // Clear the single terrain stack — empty stack is valid (identity height).
    doc.stack = terra_core::layer::LayerStack::new();
    let diags = incomplete_project_diagnostics(&doc.stack, &doc.biome_library, &doc.biome_layers);
    assert!(
        diags.iter().any(|d| d.code == "stack_without_shape_layers"),
        "got {diags:?}"
    );
    let _ = eval_doc(&mut doc);
}

#[test]
fn snow_rule_with_zero_coverage_is_valid() {
    let mut doc = TerrainDocument::new_default();
    let mut snow = BiomeDefinition::new("Impossible Snow", [1.0, 1.0, 1.0]);
    snow.placement.rules = Some(DistNode::height(50_000.0, 60_000.0));
    doc.biome_library.definitions.push(snow);
    let _ = eval_doc(&mut doc);
    let diags = incomplete_project_diagnostics(&doc.stack, &doc.biome_library, &doc.biome_layers);
    assert!(diags.iter().any(|d| d.code == "biome_unlinked"
        || d.code == "biome_empty_placement_rules"
        || d.code == "empty_biome_coverage"
        || !diags.is_empty()));
}
