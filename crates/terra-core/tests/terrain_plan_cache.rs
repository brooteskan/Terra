use terra_core::authoring::SculptStroke;
use terra_core::command::EditorCommand;
use terra_core::deps::NodeRef;
use terra_core::field_data::FieldId;
use terra_core::ids::LayerId;
use terra_core::layer::{FlatParams, Layer, LayerKind, LayerStack};
use terra_core::mask::{MaskId, MaskRef};
use terra_core::terrain_plan::{
    PlanDirtyScope, PlanNodeSelection, TerrainEditClass, TerrainPlanCache, TerrainPlanDiagnostic,
};
use terra_core::tiling::UvRect;

fn flat(id: u128, height: f32) -> Layer {
    let mut layer = Layer::new("Flat", LayerKind::Flat(FlatParams { height }));
    layer.common.id = LayerId::from_u128(id);
    layer
}

fn region_edit(owner: LayerId) -> TerrainEditClass {
    TerrainEditClass::Content {
        owner: NodeRef::Layer(owner),
        fields: vec![FieldId::Height],
        scope: PlanDirtyScope::Region(UvRect::from_center_radius(0.5, 0.5, 0.02)),
    }
}

#[test]
fn content_and_parameter_edits_reuse_the_structural_plan() {
    let id = LayerId::from_u128(1);
    let mut strokes = Layer::new("Strokes", LayerKind::SculptStrokes(Default::default()));
    strokes.common.id = id;
    let mut stack = LayerStack::new();
    stack.push(strokes);
    let mut cache = TerrainPlanCache::new();

    cache
        .update(&stack, &[], &[region_edit(id)])
        .expect("initial plan");
    if let LayerKind::SculptStrokes(params) = &mut stack.find_mut(id).unwrap().kind {
        params.strokes.push(SculptStroke::default());
    }
    let second = cache
        .update(&stack, &[], &[region_edit(id)])
        .expect("content patch");
    let parameter = TerrainEditClass::Parameters {
        owner: NodeRef::Layer(id),
    };
    cache
        .update(&stack, &[], &[parameter])
        .expect("parameter patch");

    let stats = cache.stats().snapshot();
    assert_eq!(stats.plan_compiles, 1);
    assert_eq!(stats.successful_compiles, 1);
    assert_eq!(stats.plan_cache_hits, 2);
    assert!(!second.patched_operations.is_empty());
}

#[test]
fn sculpt_base_content_edits_never_advance_structure() {
    let id = LayerId::from_u128(7);
    let mut base = Layer::new("Base", LayerKind::SculptBase(Default::default()));
    base.common.id = id;
    let mut stack = LayerStack::new();
    stack.push(base);
    let mut cache = TerrainPlanCache::new();
    cache.acquire(&stack, &[]).expect("initial plan");
    let revision = cache.structure_revision();

    cache
        .update(&stack, &[], &[region_edit(id)])
        .expect("base content patch");
    assert_eq!(cache.structure_revision(), revision);
    assert_eq!(cache.stats().snapshot().plan_compiles, 1);
    assert_eq!(cache.stats().snapshot().plan_cache_hits, 1);
}

#[test]
fn a_structural_batch_compiles_once_before_execution() {
    let mut stack = LayerStack::new();
    stack.push(flat(1, 1.0));
    let mut cache = TerrainPlanCache::new();
    cache.acquire(&stack, &[]).expect("initial plan");

    stack.push(flat(2, 2.0));
    cache
        .update(
            &stack,
            &[],
            &[TerrainEditClass::Structure, TerrainEditClass::Structure],
        )
        .expect("one candidate for the batch");
    assert_eq!(cache.structure_revision().get(), 1);
    assert_eq!(cache.stats().snapshot().plan_compiles, 2);
    assert_eq!(cache.current_plan().unwrap().operations().len(), 7);

    stack.reorder(0, 1);
    cache
        .update(&stack, &[], &[TerrainEditClass::Structure])
        .expect("reorder candidate");
    stack.remove(LayerId::from_u128(2));
    cache
        .update(&stack, &[], &[TerrainEditClass::Structure])
        .expect("remove candidate");
    assert_eq!(cache.stats().snapshot().plan_compiles, 4);
}

#[test]
fn failed_candidate_preserves_last_good_but_rejects_stale_execution() {
    let mut stack = LayerStack::new();
    stack.push(flat(1, 1.0));
    let mut cache = TerrainPlanCache::new();
    let original_signature = cache
        .acquire(&stack, &[])
        .expect("initial plan")
        .structure_signature();

    stack
        .find_mut(LayerId::from_u128(1))
        .unwrap()
        .common
        .masks
        .push(MaskRef::new(MaskId::new()));
    cache.note_edit(&TerrainEditClass::Structure);
    let diagnostics = cache.acquire(&stack, &[]).expect_err("invalid candidate");
    assert!(diagnostics
        .iter()
        .any(|diagnostic| matches!(diagnostic, TerrainPlanDiagnostic::MissingMask { .. })));
    assert!(cache.current_plan().is_err());
    let mut backend_mutations = 0;
    if cache.current_plan().is_ok() {
        backend_mutations += 1;
    }
    assert_eq!(backend_mutations, 0);
    assert_eq!(
        cache.last_good_plan().unwrap().structure_signature(),
        original_signature
    );
}

#[test]
fn editor_commands_define_the_structural_revision_boundary() {
    let id = LayerId::from_u128(11);
    let mut stack = LayerStack::new();
    stack.push(flat(11, 1.0));

    let numeric_parameter = EditorCommand::SetKind {
        id,
        kind: LayerKind::Flat(FlatParams { height: 2.0 }),
        previous: LayerKind::Flat(FlatParams { height: 1.0 }),
    };
    assert!(matches!(
        numeric_parameter.terrain_edit_class(&stack),
        TerrainEditClass::Parameters { owner: NodeRef::Layer(owner) } if owner == id
    ));

    let operation_shape = EditorCommand::SetKind {
        id,
        kind: LayerKind::SculptStrokes(Default::default()),
        previous: LayerKind::Flat(FlatParams { height: 1.0 }),
    };
    assert_eq!(
        operation_shape.terrain_edit_class(&stack),
        TerrainEditClass::Structure
    );

    let solo = EditorCommand::SetSolo {
        id,
        solo: true,
        previous: false,
    };
    assert_eq!(solo.terrain_edit_class(&stack), TerrainEditClass::Structure);

    let rename = EditorCommand::Rename {
        id,
        name: "Renamed".into(),
        previous: "Flat".into(),
    };
    assert_eq!(
        rename.terrain_edit_class(&stack),
        TerrainEditClass::ViewOnly
    );
}

#[test]
fn solo_toggles_compile_one_transactional_replacement_each() {
    let solo_id = LayerId::from_u128(22);
    let mut stack = LayerStack::new();
    stack.push(flat(21, 10.0));
    stack.push(flat(22, 20.0));
    let mut cache = TerrainPlanCache::new();
    let original_signature = cache
        .acquire(&stack, &[])
        .expect("initial plan")
        .structure_signature();

    stack.find_mut(solo_id).unwrap().common.solo = true;
    cache
        .update(&stack, &[], &[TerrainEditClass::Structure])
        .expect("solo replacement");
    let solo_plan = cache.current_plan().expect("current solo plan");
    assert_ne!(solo_plan.structure_signature(), original_signature);
    assert_eq!(
        solo_plan
            .provenance()
            .selection_for(NodeRef::Layer(solo_id)),
        Some(PlanNodeSelection::IncludedBySolo)
    );
    assert_eq!(cache.stats().snapshot().plan_compiles, 2);

    stack.find_mut(solo_id).unwrap().common.solo = false;
    cache
        .update(&stack, &[], &[TerrainEditClass::Structure])
        .expect("unsolo replacement");
    assert_eq!(
        cache
            .current_plan()
            .unwrap()
            .provenance()
            .selection_for(NodeRef::Layer(solo_id)),
        Some(PlanNodeSelection::Unfiltered)
    );
    assert_eq!(cache.stats().snapshot().plan_compiles, 3);
    assert_eq!(cache.stats().snapshot().successful_compiles, 3);
}

#[test]
fn invalid_solo_candidate_retains_the_last_good_plan() {
    let solo_id = LayerId::from_u128(32);
    let mut stack = LayerStack::new();
    stack.push(flat(31, 10.0));
    stack.push(flat(32, 20.0));
    let mut cache = TerrainPlanCache::new();
    let original_signature = cache
        .acquire(&stack, &[])
        .expect("initial plan")
        .structure_signature();

    let authored = stack.find_mut(solo_id).unwrap();
    authored.common.solo = true;
    authored.common.masks.push(MaskRef::new(MaskId::new()));
    let diagnostics = cache
        .update(&stack, &[], &[TerrainEditClass::Structure])
        .expect_err("invalid solo replacement");
    assert!(diagnostics
        .iter()
        .any(|diagnostic| matches!(diagnostic, TerrainPlanDiagnostic::MissingMask { .. })));
    assert!(cache.current_plan().is_err());
    assert_eq!(
        cache.last_good_plan().unwrap().structure_signature(),
        original_signature
    );
    assert_eq!(cache.stats().snapshot().plan_compiles, 2);
    assert_eq!(cache.stats().snapshot().successful_compiles, 1);
}
