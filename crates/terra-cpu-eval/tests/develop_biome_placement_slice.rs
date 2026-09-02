//! Develop workspace — Biome-scoped operation placement.

use std::collections::HashMap;
use terra_core::command::{apply, CommandHistory, EditorCommand};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{
    BiomeSection, Layer, LayerGroup, LayerKind, LayerStack, MaterialsParams, MountainParams,
    StackNode, VegetationParams,
};
use terra_core::mask::{bake_mask_assets, DistNode, DistNodeKind};
use terra_core::operation_placement::{
    create_develop_operation, ApplyWhere, DevelopCategory, OperationPlacement,
};
use terra_core::quality::PreviewQuality;
use terra_cpu_eval::{EvalContext, StackEvaluator};

fn metrics() -> HeightfieldMetrics {
    HeightfieldMetrics::new(32, 32, 1024.0, 1024.0)
}

fn alpine_biome() -> LayerGroup {
    let mut biome = LayerGroup::biome("Alpine Mountains");
    biome.ensure_biome_sections();
    // Biome Effective Placement: soft coverage (not a raw Mask ID in UI).
    biome
        .masks
        .push_node(DistNode::new(DistNodeKind::Fill { value: 0.85 }));
    biome
}

#[test]
fn new_operations_default_to_entire_biome() {
    let layer = create_develop_operation(DevelopCategory::Terrain, "Mountain Detail");
    assert!(layer.common.operation_placement.is_entire_biome());
    assert_eq!(
        layer.common.operation_placement.apply_where,
        ApplyWhere::EntireBiome
    );
    assert!(layer.common.masks.is_empty());
    assert_eq!(
        layer.common.develop_category,
        Some(DevelopCategory::Terrain)
    );
}

#[test]
fn automatic_biome_inheritance_summary_no_mask_ids() {
    let mut stack = LayerStack::new();
    let mut biome = alpine_biome();
    let op = create_develop_operation(DevelopCategory::Terrain, "Ridge Enhancement");
    let op_id = op.id();
    biome
        .find_section_mut(BiomeSection::Filters)
        .unwrap()
        .children
        .push(StackNode::Layer(op));
    let biome_id = biome.id;
    stack.push_group(biome);

    assert_eq!(
        stack.enclosing_biome(op_id).map(|g| g.name.as_str()),
        Some("Alpine Mountains")
    );
    let layer = stack.find(op_id).unwrap();
    let summary = layer
        .common
        .operation_placement
        .summary_paragraph("Alpine Mountains");
    assert!(summary.contains("Applies within Alpine Mountains"));
    assert!(summary.contains("Entire Biome"));
    assert!(!summary.to_lowercase().contains("maskid"));
    assert!(!summary.contains(&format!("{}", biome_id.0)));
}

#[test]
fn local_placement_compiles_height_and_slope() {
    let mut layer = create_develop_operation(DevelopCategory::Materials, "Snow Patches");
    layer.common.operation_placement.apply_where = ApplyWhere::HeightRange;
    layer.common.operation_placement.height_min = 1200.0;
    layer.common.operation_placement.height_max = 5000.0;
    layer.sync_operation_placement_masks();
    assert!(!layer.common.masks.is_empty());

    let text = layer
        .common
        .operation_placement
        .summary_paragraph("Alpine Mountains");
    assert!(text.contains("Applies within Alpine Mountains"));
    assert!(text.contains("1200"));
    assert!(!text.contains("MaskId"));

    // Second condition style (slope) for readable AND summaries.
    let mut slope = OperationPlacement::default();
    slope.apply_where = ApplyWhere::SlopeRange;
    slope.slope_min = 0.0;
    slope.slope_max = 50.0;
    slope.sync_definition_from_apply_where();
    let slope_text = slope.summary_paragraph("Alpine Mountains");
    assert!(slope_text.contains("50"));
}

#[test]
fn operation_moving_between_biomes_keeps_local_placement() {
    let mut stack = LayerStack::new();
    let mut alpine = alpine_biome();
    let mut desert = LayerGroup::biome("Desert Dunes");
    desert.ensure_biome_sections();

    let mut op = create_develop_operation(DevelopCategory::Terrain, "Thermal Erosion");
    op.common.operation_placement.apply_where = ApplyWhere::SlopeRange;
    op.common.operation_placement.slope_max = 40.0;
    op.sync_operation_placement_masks();
    let op_id = op.id();
    let local_hash = op.common.operation_placement.definition.content_hash;

    alpine
        .find_section_mut(BiomeSection::Filters)
        .unwrap()
        .children
        .push(StackNode::Layer(op));
    let alpine_id = alpine.id;
    let desert_id = desert.id;
    stack.push_group(alpine);
    stack.push_group(desert);

    assert_eq!(stack.enclosing_biome(op_id).map(|g| g.id), Some(alpine_id));
    assert!(stack.move_layer_to_biome_section(op_id, desert_id, BiomeSection::Filters));
    assert_eq!(
        stack.enclosing_biome(op_id).map(|g| g.name.as_str()),
        Some("Desert Dunes")
    );
    let moved = stack.find(op_id).unwrap();
    assert_eq!(
        moved.common.operation_placement.apply_where,
        ApplyWhere::SlopeRange
    );
    assert_eq!(
        moved.common.operation_placement.definition.content_hash,
        local_hash
    );
    // Inheritance is enclosure-based — no manual Biome Mask reassignment.
    let summary = moved
        .common
        .operation_placement
        .summary_paragraph("Desert Dunes");
    assert!(summary.contains("Desert Dunes"));
}

#[test]
fn empty_operation_coverage_still_allowed() {
    // Conditions that match nowhere must not prevent the operation existing.
    let mut layer = create_develop_operation(DevelopCategory::Vegetation, "Pine Forest");
    layer.common.operation_placement.apply_where = ApplyWhere::HeightRange;
    layer.common.operation_placement.height_min = 9_000.0;
    layer.common.operation_placement.height_max = 9_100.0;
    layer.sync_operation_placement_masks();
    assert!(!layer.common.masks.is_empty());

    let mut stack = LayerStack::new();
    let mut biome = alpine_biome();
    let id = layer.id();
    biome
        .find_section_mut(BiomeSection::Objects)
        .unwrap()
        .children
        .push(StackNode::Layer(layer));
    stack.push_group(biome);

    let m = metrics();
    let mut ctx = EvalContext::new(m);
    ctx.quality = PreviewQuality::Draft;
    let seed = Heightfield::zeros(m);
    ctx.masks = bake_mask_assets(&[], &seed, m, &HashMap::new());
    let mut eval = StackEvaluator::new();
    let _ = eval
        .rebuild_all(&stack, &mut ctx)
        .expect("eval with empty-ish coverage");
    assert!(stack.find(id).is_some());
}

#[test]
fn placement_change_invalidates_operation() {
    let mut stack = LayerStack::new();
    let mut biome = alpine_biome();
    let layer = create_develop_operation(DevelopCategory::Terrain, "Cliff Enhancement");
    let id = layer.id();
    biome
        .find_section_mut(BiomeSection::Filters)
        .unwrap()
        .children
        .push(StackNode::Layer(layer));
    stack.push_group(biome);

    let m = metrics();
    let mut ctx = EvalContext::new(m);
    ctx.quality = PreviewQuality::Draft;
    let seed = Heightfield::zeros(m);
    ctx.masks = bake_mask_assets(&[], &seed, m, &HashMap::new());
    let mut eval = StackEvaluator::new();
    let _ = eval.rebuild_all(&stack, &mut ctx).expect("eval");
    // Mark clean then dirty via stage path.
    if let Some(e) = eval.cache.get_mut(id) {
        e.dirty = false;
    }
    eval.mark_dirty_from_stage(&stack, id);
    assert!(eval.cache.is_dirty(id));
}

#[test]
fn serialization_preserves_operation_placement() {
    let mut placement = OperationPlacement::entire_biome();
    placement.apply_where = ApplyWhere::HeightRange;
    placement.height_min = 1200.0;
    placement.sync_definition_from_apply_where();
    let json = serde_json::to_string(&placement).expect("serialize");
    assert!(json.contains("height_min") || json.contains("1200"));
    let back: OperationPlacement = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.apply_where, ApplyWhere::HeightRange);
    assert!((back.height_min - 1200.0).abs() < 1e-3);
    assert!(!back.compile_local_distribution().is_empty());
}

#[test]
fn undo_redo_operation_placement() {
    let mut stack = LayerStack::new();
    let layer = create_develop_operation(DevelopCategory::Terrain, "Mountain Detail");
    let id = layer.id();
    stack.nodes.push(StackNode::Layer(layer));

    let previous = stack.find(id).unwrap().common.operation_placement.clone();
    let previous_masks = stack.find(id).unwrap().common.masks.clone();
    let mut placement = previous.clone();
    placement.apply_where = ApplyWhere::SlopeRange;
    placement.slope_max = 50.0;
    placement.sync_definition_from_apply_where();

    let cmd = EditorCommand::SetOperationPlacement {
        id,
        placement: placement.clone(),
        previous: previous.clone(),
        previous_masks: previous_masks.clone(),
    };
    let mut history = CommandHistory::new(32);
    apply(&cmd, &mut stack);
    history.push_executed(cmd);
    assert_eq!(
        stack
            .find(id)
            .unwrap()
            .common
            .operation_placement
            .apply_where,
        ApplyWhere::SlopeRange
    );

    history.undo(&mut stack);
    assert_eq!(
        stack
            .find(id)
            .unwrap()
            .common
            .operation_placement
            .apply_where,
        ApplyWhere::EntireBiome
    );

    history.redo(&mut stack);
    assert_eq!(
        stack
            .find(id)
            .unwrap()
            .common
            .operation_placement
            .apply_where,
        ApplyWhere::SlopeRange
    );
}

#[test]
fn material_changes_do_not_dirty_height_layers() {
    let mut stack = LayerStack::new();
    let mountains = Layer::new("Mountains", LayerKind::Mountains(MountainParams::default()));
    let materials = Layer::new("Rock", LayerKind::Materials(MaterialsParams::default()));
    let mid = mountains.id();
    let mat_id = materials.id();
    stack.nodes.push(StackNode::Layer(mountains));
    stack.nodes.push(StackNode::Layer(materials));

    let m = metrics();
    let mut ctx = EvalContext::new(m);
    ctx.quality = PreviewQuality::Draft;
    let seed = Heightfield::zeros(m);
    ctx.masks = bake_mask_assets(&[], &seed, m, &HashMap::new());
    let mut eval = StackEvaluator::new();
    let _ = eval.rebuild_all(&stack, &mut ctx).expect("eval");
    if let Some(e) = eval.cache.get_mut(mid) {
        e.dirty = false;
    }
    if let Some(e) = eval.cache.get_mut(mat_id) {
        e.dirty = false;
    }

    eval.mark_dirty_from_stage(&stack, mat_id);
    assert!(!eval.cache.is_dirty(mid), "height layer must stay clean");
    assert!(eval.cache.is_dirty(mat_id));
}

#[test]
fn scatter_changes_do_not_rebuild_materials() {
    let mut stack = LayerStack::new();
    let materials = Layer::new(
        "Alpine Grass",
        LayerKind::Materials(MaterialsParams::default()),
    );
    let veg = Layer::new(
        "Pine Forest",
        LayerKind::Vegetation(VegetationParams::default()),
    );
    let mat_id = materials.id();
    let veg_id = veg.id();
    stack.nodes.push(StackNode::Layer(materials));
    stack.nodes.push(StackNode::Layer(veg));

    let m = metrics();
    let mut ctx = EvalContext::new(m);
    ctx.quality = PreviewQuality::Draft;
    let seed = Heightfield::zeros(m);
    ctx.masks = bake_mask_assets(&[], &seed, m, &HashMap::new());
    let mut eval = StackEvaluator::new();
    let _ = eval.rebuild_all(&stack, &mut ctx).expect("eval");
    if let Some(e) = eval.cache.get_mut(mat_id) {
        e.dirty = false;
    }
    if let Some(e) = eval.cache.get_mut(veg_id) {
        e.dirty = false;
    }

    eval.mark_dirty_from_stage(&stack, veg_id);
    assert!(!eval.cache.is_dirty(mat_id), "materials must stay clean");
    assert!(eval.cache.is_dirty(veg_id));
}

#[test]
fn advanced_mask_preserves_custom_stack_access() {
    let mut layer = create_develop_operation(DevelopCategory::Terrain, "Custom Cliffs");
    layer
        .common
        .masks
        .push_node(DistNode::new(DistNodeKind::Fill { value: 0.5 }));
    layer.common.operation_placement.apply_where = ApplyWhere::AdvancedMask;
    layer
        .common
        .operation_placement
        .definition
        .mark_custom(layer.common.masks.clone());
    assert!(matches!(
        layer.common.operation_placement.apply_where,
        ApplyWhere::AdvancedMask
    ));
    assert!(layer
        .common
        .operation_placement
        .definition
        .custom_stack
        .is_some());
    let lines = layer
        .common
        .operation_placement
        .summary_lines("Alpine Mountains");
    assert!(lines.iter().any(|l| l.contains("Advanced Mask")));
}
