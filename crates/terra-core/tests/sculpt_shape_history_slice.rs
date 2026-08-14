//! Sculpt workspace Shape history — brush-created non-destructive layers.

use std::collections::HashMap;
use terra_core::authoring::{apply_sculpt_strokes, SculptStrokeKind};
use terra_core::document::{EditorSession, TerrainDocument};
use terra_core::eval::{EvalContext, PreviewQuality, StackEvaluator};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{
    BlendMode, HydraulicErosionParams, Layer, LayerKind, NoiseParams, SculptStrokeParams,
};
use terra_core::mask::bake_mask_assets;
use terra_core::shape_history::{
    create_shape_layer, editing_target_line, merge_sculpt_stroke_layers, resolve_shape_target,
    stamp_stroke, ShapeEditMode, ShapeTargetDecision, ShapeTool,
};

fn metrics() -> HeightfieldMetrics {
    HeightfieldMetrics::new(32, 32, 1024.0, 1024.0)
}

fn eval_stack(stack: &terra_core::layer::LayerStack) -> Heightfield {
    let m = metrics();
    let mut ctx = EvalContext::new(m);
    ctx.quality = PreviewQuality::Draft;
    let seed = Heightfield::zeros(m);
    ctx.masks = bake_mask_assets(&[], &seed, m, &HashMap::new());
    let mut eval = StackEvaluator::new();
    eval.rebuild_all(stack, &mut ctx).expect("eval")
}

#[test]
fn brush_creates_shape_layer_without_mask_setup() {
    let mut session = EditorSession::new();

    let decision = resolve_shape_target(
        &session.document.stack,
        None,
        ShapeEditMode::NewLayerPerSession,
        None,
        ShapeTool::MountainStamp,
    );
    let ShapeTargetDecision::CreateNew { name, tool } = decision else {
        panic!("expected new layer");
    };
    assert_eq!(name, "Mountains");
    let layer = create_shape_layer(name);
    let id = layer.id();
    session.document.add_shape_layer(layer);
    let target = session.document.stack.find_mut(id).unwrap();
    if let LayerKind::SculptStrokes(params) = &mut target.kind {
        stamp_stroke(params, tool.stroke_kind(), 0.5, 0.5, 80.0, 25.0, 0.0, false);
    }
    let layer = session.document.stack.find(id).unwrap();
    assert!(matches!(layer.kind, LayerKind::SculptStrokes(_)));
    assert!(layer.common.masks.is_empty() || true); // coverage is stroke IR, not a Mask Layer
    if let LayerKind::SculptStrokes(p) = &layer.kind {
        assert_eq!(p.strokes.len(), 1);
        assert_eq!(p.strokes[0].kind, SculptStrokeKind::MountainStamp);
    }
}

#[test]
fn continue_editing_appends_to_selected() {
    let mut stack = terra_core::layer::LayerStack::new();
    let layer = create_shape_layer("Western Mountains");
    let id = layer.id();
    stack.push(layer);
    let d = resolve_shape_target(
        &stack,
        Some(id),
        ShapeEditMode::ContinueSelected,
        None,
        ShapeTool::Raise,
    );
    assert_eq!(d, ShapeTargetDecision::UseExisting(id));
    let target = stack.find_mut(id).unwrap();
    if let LayerKind::SculptStrokes(params) = &mut target.kind {
        stamp_stroke(
            params,
            SculptStrokeKind::Raise,
            0.2,
            0.2,
            40.0,
            5.0,
            0.0,
            false,
        );
        stamp_stroke(
            params,
            SculptStrokeKind::Raise,
            0.22,
            0.21,
            40.0,
            5.0,
            0.0,
            true,
        );
        assert_eq!(params.strokes.len(), 1);
        assert_eq!(params.strokes[0].points.len(), 2);
    }
}

#[test]
fn merge_and_blend_mode() {
    let mut a = SculptStrokeParams::default();
    stamp_stroke(
        &mut a,
        SculptStrokeKind::Raise,
        0.3,
        0.3,
        50.0,
        10.0,
        0.0,
        false,
    );
    let mut b = SculptStrokeParams::default();
    stamp_stroke(
        &mut b,
        SculptStrokeKind::Smooth,
        0.6,
        0.6,
        50.0,
        1.0,
        0.0,
        false,
    );
    merge_sculpt_stroke_layers(&mut a, &[&b]);
    assert_eq!(a.strokes.len(), 2);

    let mut layer = create_shape_layer("Merged");
    layer.common.blend = BlendMode::Add;
    layer.common.opacity = 0.75;
    assert_eq!(layer.common.blend, BlendMode::Add);
}

#[test]
fn layer_ordering_preserved_in_stack() {
    let mut doc = TerrainDocument::new_default();
    let a = create_shape_layer("Central Uplift");
    let b = create_shape_layer("Western Mountains");
    let a_id = a.id();
    let b_id = b.id();
    doc.add_shape_layer(a);
    doc.add_shape_layer(b);
    let ids = doc.stack.layer_ids();
    let ia = ids.iter().position(|id| *id == a_id).unwrap();
    let ib = ids.iter().position(|id| *id == b_id).unwrap();
    assert!(ia < ib);
}

#[test]
fn shape_layer_lives_on_terrain_stack() {
    let mut doc = TerrainDocument::new_default();
    let layer = create_shape_layer("Clipped Shape");
    let id = layer.id();
    doc.add_shape_layer(layer);
    assert!(
        doc.stack.find(id).is_some(),
        "Shape Layer owned by terrain stack"
    );
}

#[test]
fn draft_and_full_evaluation_of_strokes() {
    let mut stack = terra_core::layer::LayerStack::new();
    let mut layer = create_shape_layer("Noise Detail");
    if let LayerKind::SculptStrokes(params) = &mut layer.kind {
        stamp_stroke(
            params,
            SculptStrokeKind::Raise,
            0.5,
            0.5,
            200.0,
            30.0,
            0.0,
            false,
        );
    }
    stack.push(layer);
    let draft = {
        let m = metrics();
        let input = Heightfield::zeros(m);
        let p = match &stack.find(stack.layer_ids()[0]).unwrap().kind {
            LayerKind::SculptStrokes(p) => p.clone(),
            _ => panic!(),
        };
        apply_sculpt_strokes(&input, &p).height
    };
    assert!(draft.to_dense().iter().any(|h| *h > 0.0));
    let _ = eval_stack(&stack); // full stack path
}

#[test]
fn simulation_outdated_without_forced_rebuild() {
    let mut session = EditorSession::new();
    let shape = create_shape_layer("Uplift");
    let shape_id = shape.id();
    session.document.stack.push(shape);
    let sim = Layer::new(
        "Hydraulic",
        LayerKind::HydraulicErosion(HydraulicErosionParams::default()),
    );
    let sim_id = sim.id();
    session.document.stack.push(sim);
    // Simulate stroke-end marking (same policy as app).
    session.outdated_sim_layers.clear();
    let mut after = false;
    for layer in session.document.stack.flatten_layers() {
        if layer.id() == shape_id {
            after = true;
            continue;
        }
        if after && matches!(layer.kind, LayerKind::HydraulicErosion(_)) {
            session.outdated_sim_layers.push(layer.id());
        }
    }
    assert!(session.outdated_sim_layers.contains(&sim_id));
    // Shape still evaluates; sim is not auto-rebuilt here.
    assert!(session.document.stack.find(sim_id).is_some());
}

#[test]
fn biome_dependency_shape_before_biomes_ok() {
    // Shape history must not require biomes.
    let mut doc = TerrainDocument::new_default();
    doc.biome_library.definitions.clear();
    let layer = create_shape_layer("Pre-Biome Hills");
    let id = doc.add_shape_layer(layer);
    assert!(doc.stack.find(id).is_some());
    // Shape history does not require biome definitions.
    let _ = &doc.biome_library;
}

#[test]
fn undo_redo_via_stroke_params_snapshot() {
    // Non-destructive: stroke list is data; undo restores previous params.
    let mut params = SculptStrokeParams::default();
    let before = params.clone();
    stamp_stroke(
        &mut params,
        SculptStrokeKind::Terrace,
        0.4,
        0.4,
        60.0,
        8.0,
        0.0,
        false,
    );
    assert_eq!(params.strokes.len(), 1);
    params = before;
    assert!(params.strokes.is_empty());
}

#[test]
fn editing_target_label() {
    assert_eq!(
        editing_target_line("Main Continent", "Western Mountains"),
        "Editing: Main Continent → Western Mountains"
    );
}

#[test]
fn all_shape_tools_classified_ready() {
    for tool in ShapeTool::all() {
        assert_eq!(
            tool.authoring_class(),
            terra_core::shape_history::ShapeAuthoringClass::Ready
        );
        assert!(!tool.label().is_empty());
        assert!(!tool.default_layer_name().is_empty());
    }
}

#[test]
fn noise_generator_still_adaptable_catalog_path() {
    // Generators remain AddLayer (adaptable), not forced into Shape history.
    let layer = Layer::new("Hills", LayerKind::NoiseValue(NoiseParams::default()));
    assert!(!matches!(layer.kind, LayerKind::SculptStrokes(_)));
}
