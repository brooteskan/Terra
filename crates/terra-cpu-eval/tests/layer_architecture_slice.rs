//! Vertical slice: ordered stack + isolated alpine group + masking + cache.

use std::collections::HashMap;
use terra_core::document::TerrainDocument;
use terra_core::heightfield::HeightfieldMetrics;
use terra_core::layer::{
    GroupEvalMode, GroupInputMode, HydraulicErosionParams, Layer, LayerGroup, LayerKind,
    MountainParams, StackCategory, StackNode, ThermalErosionParams,
};
use terra_core::mask::{
    bake_mask_assets, Distribution, MaskAsset, MaskCombine, MaskField, MaskId, MaskRef, MaskSource,
};
use terra_core::quality::PreviewQuality;
use terra_cpu_eval::{EvalContext, StackEvaluator};

fn eval_doc(doc: &TerrainDocument, metrics: HeightfieldMetrics) -> (terra_core::Heightfield, u64) {
    let mut ctx = EvalContext::new(metrics);
    ctx.quality = PreviewQuality::Draft;
    ctx.mask_assets = doc.masks.clone();
    let seed = terra_core::Heightfield::zeros(metrics);
    let baked = bake_mask_assets(&doc.masks, &seed, metrics, &HashMap::new());
    ctx.masks = baked;
    let mut eval = StackEvaluator::new();
    let height = eval.rebuild_all(&doc.stack, &mut ctx).expect("eval");
    (height, eval.cache.generation)
}

#[test]
fn default_stack_uses_wc_categories() {
    let doc = TerrainDocument::new_default();
    assert!(doc.stack.find_category(StackCategory::Shape).is_some());
    assert!(doc.stack.find_category(StackCategory::Simulation).is_some());
    assert!(doc.stack.find_category(StackCategory::Mask).is_some());
    assert!(doc.stack.find_category(StackCategory::Surface).is_some());
}

#[test]
fn alpine_vertical_slice_evaluates() {
    let mut doc = TerrainDocument::new_default();
    let metrics = HeightfieldMetrics::new(64, 64, 256.0, 256.0);
    doc.bounded_settings_mut().unwrap().metrics = metrics;
    if let Some(shape) = doc.stack.find_category_mut(StackCategory::Shape) {
        shape.children.clear();
    }

    let mid = MaskId::new();
    let mask = MaskAsset {
        id: mid,
        name: "Alpine Area".into(),
        source: MaskSource::Constant(1.0),
        ops: Vec::new(),
        paint: None,
        display_color: terra_core::mask::default_mask_display_color(),
    };
    doc.masks.push(mask);

    let mut alpine = LayerGroup::isolated("Alpine Mountains");
    alpine.masks.push(MaskRef::new(mid));
    alpine.children.push(StackNode::Layer(Layer::new(
        "Mountain Range",
        LayerKind::Mountains(MountainParams {
            base: terra_core::layer::NoiseParams {
                amplitude: 40.0,
                frequency: 0.02,
                octaves: 3,
                ..Default::default()
            },
            ..Default::default()
        }),
    )));
    let mut hyd = HydraulicErosionParams::default();
    hyd.iterations = 2;
    alpine.children.push(StackNode::Layer(Layer::new(
        "Hydraulic Erosion",
        LayerKind::HydraulicErosion(hyd),
    )));
    let mut therm = ThermalErosionParams::default();
    therm.iterations = 2;
    alpine.children.push(StackNode::Layer(Layer::new(
        "Thermal Erosion",
        LayerKind::ThermalErosion(therm),
    )));
    doc.stack
        .find_category_mut(StackCategory::Shape)
        .unwrap()
        .children
        .push(StackNode::Group(alpine));

    let (height, _) = eval_doc(&doc, metrics);
    let (min, max) = height.min_max();
    assert!(max > min, "terrain should vary");
    assert!(doc.validate_dependencies().is_ok());
    assert_eq!(
        doc.stack
            .find_category(StackCategory::Shape)
            .unwrap()
            .children
            .len(),
        1
    );
    let _ = GroupEvalMode::IsolatedComposite;
}

#[test]
fn isolated_group_mask_limits_result() {
    let metrics = HeightfieldMetrics::new(32, 32, 128.0, 128.0);
    let mut doc = TerrainDocument::new_default();
    doc.bounded_settings_mut().unwrap().metrics = metrics;
    doc.stack.nodes.clear();
    doc.stack.push(Layer::new(
        "Base",
        LayerKind::Flat(terra_core::layer::FlatParams { height: 10.0 }),
    ));
    doc.stack.ensure_category_folders();

    let mid = MaskId::new();
    doc.masks.push(MaskAsset {
        id: mid,
        name: "Empty".into(),
        source: MaskSource::Constant(0.0),
        ops: Vec::new(),
        paint: None,
        display_color: terra_core::mask::default_mask_display_color(),
    });

    let mut alpine = LayerGroup::isolated("Alpine");
    alpine.masks.push(MaskRef::new(mid));
    alpine.children.push(StackNode::Layer(Layer::new(
        "Mountains",
        LayerKind::Mountains(MountainParams {
            base: terra_core::layer::NoiseParams {
                amplitude: 80.0,
                frequency: 0.05,
                octaves: 2,
                ..Default::default()
            },
            ..Default::default()
        }),
    )));
    doc.stack
        .find_category_mut(StackCategory::Shape)
        .unwrap()
        .children
        .push(StackNode::Group(alpine));

    let (height, _) = eval_doc(&doc, metrics);
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            assert!(
                (height.get(i, j) - 10.0).abs() < 0.5,
                "masked-out group must not change base"
            );
        }
    }
}

#[test]
fn pass_through_vs_isolated_input_modes() {
    let metrics = HeightfieldMetrics::new(16, 16, 64.0, 64.0);
    let mut doc = TerrainDocument::new_default();
    doc.bounded_settings_mut().unwrap().metrics = metrics;
    doc.stack.nodes.clear();
    doc.stack.push(Layer::new(
        "Base",
        LayerKind::Flat(terra_core::layer::FlatParams { height: 5.0 }),
    ));
    let mut g = LayerGroup::isolated("Feature");
    g.input_mode = GroupInputMode::EmptyHeight;
    g.opacity = 1.0;
    g.children.push(StackNode::Layer(Layer::new(
        "Flat Feature",
        LayerKind::Flat(terra_core::layer::FlatParams { height: 20.0 }),
    )));
    doc.stack.push_group(g);
    let (height, _) = eval_doc(&doc, metrics);
    assert!((height.get(8, 8) - 20.0).abs() < 0.1);
}

#[test]
fn mask_combine_subtract_and_replace() {
    let metrics = HeightfieldMetrics::new(4, 4, 4.0, 4.0);
    let id_a = MaskId::new();
    let id_b = MaskId::new();
    let mut masks = HashMap::new();
    masks.insert(id_a, MaskField::filled(metrics, 0.8));
    masks.insert(id_b, MaskField::filled(metrics, 0.3));
    let dist = Distribution {
        entries: vec![
            terra_core::mask::DistributionEntry {
                mask: MaskRef::new(id_a),
                combine: MaskCombine::Replace,
            },
            terra_core::mask::DistributionEntry {
                mask: MaskRef::new(id_b),
                combine: MaskCombine::Subtract,
            },
        ],
        ..Default::default()
    };
    let baked = terra_core::mask::bake_distribution(&dist, &masks, metrics);
    assert!((baked.get(0, 0) - 0.5).abs() < 1e-5);
}

#[test]
fn editing_later_layer_keeps_prefix_cache() {
    let metrics = HeightfieldMetrics::new(32, 32, 128.0, 128.0);
    let mut stack = terra_core::layer::LayerStack::new();
    let base = Layer::new(
        "Base",
        LayerKind::Flat(terra_core::layer::FlatParams { height: 5.0 }),
    );
    let base_id = base.id();
    stack.push(base);
    let mtn = Layer::new(
        "M",
        LayerKind::Mountains(MountainParams {
            base: terra_core::layer::NoiseParams {
                amplitude: 20.0,
                frequency: 0.04,
                octaves: 2,
                ..Default::default()
            },
            ..Default::default()
        }),
    );
    let mtn_id = mtn.id();
    stack.push(mtn);
    let mut hyd = HydraulicErosionParams::default();
    hyd.iterations = 1;
    let hyd_layer = Layer::new("H", LayerKind::HydraulicErosion(hyd));
    let hyd_id = hyd_layer.id();
    stack.push(hyd_layer);

    let mut ctx = EvalContext::new(metrics);
    ctx.quality = PreviewQuality::Draft;
    let mut eval = StackEvaluator::new();
    let _ = eval.rebuild_all(&stack, &mut ctx).unwrap();

    eval.mark_dirty_from(&stack, hyd_id);
    assert!(!eval.cache.is_dirty(base_id));
    assert!(!eval.cache.is_dirty(mtn_id));
    assert!(eval.cache.is_dirty(hyd_id));
    let _ = eval.rebuild_incremental(&stack, &mut ctx).unwrap();
}

#[test]
fn field_contract_and_categories() {
    use terra_core::fields::FieldId;
    use terra_core::layer::OperationCategory;
    let k = LayerKind::HydraulicErosion(HydraulicErosionParams::default());
    assert_eq!(k.category(), OperationCategory::Simulation);
    assert_eq!(
        StackCategory::from_operation(k.category()),
        StackCategory::Simulation
    );
    assert!(k.produced_fields().contains(&FieldId::Wetness));
    assert!(k.required_fields().contains(&FieldId::Height));
}

#[test]
fn add_routes_into_shape_and_simulation() {
    let mut stack = terra_core::layer::LayerStack::new();
    stack.push(Layer::new(
        "Base",
        LayerKind::Flat(terra_core::layer::FlatParams { height: 1.0 }),
    ));
    stack.push_into_category(Layer::new(
        "Mountains",
        LayerKind::Mountains(MountainParams::default()),
    ));
    stack.push_into_category(Layer::new(
        "Hydraulic",
        LayerKind::HydraulicErosion(HydraulicErosionParams::default()),
    ));
    assert_eq!(
        stack
            .find_category(StackCategory::Shape)
            .unwrap()
            .children
            .len(),
        1
    );
    assert_eq!(
        stack
            .find_category(StackCategory::Simulation)
            .unwrap()
            .children
            .len(),
        1
    );
}

#[test]
fn registry_creates_and_stage_metadata() {
    use terra_core::layer::{LayerTypeRegistry, WorkflowStage};
    let reg = LayerTypeRegistry::builtin();
    let mountain = reg.create("mountain").expect("mountain");
    assert!(matches!(mountain.kind, LayerKind::Mountains(_)));
    assert_eq!(mountain.kind.workflow_stage(), WorkflowStage::Generation);
    let hyd = reg.create("hydraulic_erosion").unwrap();
    assert_eq!(hyd.kind.workflow_stage(), WorkflowStage::Simulation);
    let meta = reg.meta_for_kind(&hyd.kind).unwrap();
    assert!(!meta.suggested_next.is_empty() || meta.type_id == "hydraulic_erosion");
    assert!(meta.capabilities.user_creatable);
}

#[test]
fn stack_order_and_dirty_suffix() {
    use terra_core::layer::LayerStack;
    use terra_cpu_eval::StackEvaluator;
    let mut stack = LayerStack::new();
    let a = Layer::new(
        "A",
        LayerKind::Flat(terra_core::layer::FlatParams { height: 1.0 }),
    );
    let b = Layer::new(
        "B",
        LayerKind::Flat(terra_core::layer::FlatParams { height: 2.0 }),
    );
    let id_a = a.id();
    let id_b = b.id();
    stack.push(a);
    stack.push(b);
    let ids = stack.layer_ids();
    assert_eq!(ids, vec![id_a, id_b]);

    let mut eval = StackEvaluator::new();
    eval.mark_dirty_from(&stack, id_a);
    assert!(eval.cache.is_dirty(id_a));
    assert!(eval.cache.is_dirty(id_b));
}

#[test]
fn project_round_trip_preserves_hierarchy() {
    let doc = TerrainDocument::new_default();
    let json = doc.to_json().expect("serialize");
    let back = TerrainDocument::from_json(&json).expect("deserialize");
    assert_eq!(back.version, doc.version);
    assert!(back.stack.find_category(StackCategory::Shape).is_some());
    assert!(back
        .stack
        .find_category(StackCategory::Simulation)
        .is_some());
    assert_eq!(back.stack.layer_ids().len(), doc.stack.layer_ids().len());
    // Selection is domain-persisted for project restore.
    assert_eq!(back.selected, doc.selected);
}

#[test]
fn mask_dependency_edges_recorded() {
    use terra_core::deps::{DepKind, NodeRef};
    let mut doc = TerrainDocument::new_default();
    let mid = MaskId::new();
    doc.masks.push(MaskAsset {
        id: mid,
        name: "Paint".into(),
        source: MaskSource::Constant(1.0),
        ops: Vec::new(),
        paint: None,
        display_color: terra_core::mask::default_mask_display_color(),
    });
    let layer = Layer::new(
        "Masked Hills",
        LayerKind::NoiseValue(terra_core::layer::NoiseParams::default()),
    );
    let lid = layer.id();
    let mut layer = layer;
    layer.common.masks.push(MaskRef::new(mid));
    doc.stack
        .find_category_mut(StackCategory::Shape)
        .unwrap()
        .children
        .push(StackNode::Layer(layer));

    let g = doc.dependency_graph();
    assert!(g.edges.iter().any(|e| {
        e.kind == DepKind::MaskRef && e.from == NodeRef::Mask(mid) && e.to == NodeRef::Layer(lid)
    }));
    assert!(g.detect_cycle().is_ok());
}

#[test]
fn sort_by_eval_stage_orders_generation_before_simulation() {
    use terra_core::layer::LayerStack;
    let mut stack = LayerStack::new();
    stack.ensure_category_folders();
    // Use Shape folder for a consecutive leaf run with intentional reverse order.
    let shape = stack.find_category_mut(StackCategory::Shape).unwrap();
    shape.children.clear();
    shape.children.push(StackNode::Layer(Layer::new(
        "Hyd",
        LayerKind::HydraulicErosion(HydraulicErosionParams {
            iterations: 1,
            ..Default::default()
        }),
    )));
    shape.children.push(StackNode::Layer(Layer::new(
        "Mtn",
        LayerKind::Mountains(MountainParams::default()),
    )));
    stack.sort_by_eval_stage();
    let shape = stack.find_category(StackCategory::Shape).unwrap();
    match (&shape.children[0], &shape.children[1]) {
        (StackNode::Layer(a), StackNode::Layer(b)) => {
            assert!(a.kind.eval_stage() <= b.kind.eval_stage());
            assert!(matches!(a.kind, LayerKind::Mountains(_)));
            assert!(matches!(b.kind, LayerKind::HydraulicErosion(_)));
        }
        _ => panic!("expected two layers"),
    }
}
