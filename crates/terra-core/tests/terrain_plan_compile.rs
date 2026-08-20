use terra_core::deps::NodeRef;
use terra_core::field_data::FieldId;
use terra_core::ids::LayerId;
use terra_core::layer::{
    BiomesParams, FlatParams, GroupInputMode, Layer, LayerGroup, LayerKind, LayerStack,
    NamedOutputDecl, StackNode, VolcanoParams,
};
use terra_core::mask::{DistNode, MaskAsset, MaskId, MaskRef, MaskSource};
use terra_core::terrain_plan::{
    compile_terrain_plan, GroupCompositeMode, PlanStructureRevision, SeedSource, TerrainOpKind,
    TerrainPlanDiagnostic, TerrainPlanStamp,
};

fn stamp() -> TerrainPlanStamp {
    TerrainPlanStamp::new(PlanStructureRevision::new(7))
}

fn flat(id: u128, height: f32) -> Layer {
    let mut layer = Layer::new("Flat", LayerKind::Flat(FlatParams { height }));
    layer.common.id = LayerId::from_u128(id);
    layer
}

fn volcano(id: u128) -> Layer {
    let mut layer = Layer::new("Volcano", LayerKind::Volcano(VolcanoParams::default()));
    layer.common.id = LayerId::from_u128(id);
    layer
}

fn shape(plan: &terra_core::terrain_plan::CompiledTerrainPlan) -> Vec<String> {
    plan.operations()
        .iter()
        .map(|operation| match &operation.kind {
            TerrainOpKind::Seed { source, .. } => format!("seed:{source:?}"),
            TerrainOpKind::EvaluateMask { .. } => "mask".into(),
            TerrainOpKind::RunLayerKernel { type_id, .. } => format!("kernel:{type_id}"),
            TerrainOpKind::CompositeLayer { .. } => "layer-composite".into(),
            TerrainOpKind::CompositeGroup { mode, .. } => format!("group:{mode:?}"),
            TerrainOpKind::PublishOutput { .. } => "publish".into(),
        })
        .collect()
}

#[test]
fn flat_stack_lowers_in_explicit_bottom_to_top_order() {
    let mut stack = LayerStack::new();
    stack.push(flat(1, 12.0));
    stack.push(volcano(2));

    let plan = compile_terrain_plan(&stack, &[], stamp()).expect("flat plan");
    assert_eq!(
        shape(&plan),
        [
            "seed:Zero",
            "kernel:flat",
            "mask",
            "layer-composite",
            "kernel:volcano",
            "mask",
            "layer-composite",
        ]
    );
}

#[test]
fn pass_through_folder_allocates_no_private_group_work() {
    let child = volcano(11);
    let mut group = LayerGroup::new("Folder");
    group.id = LayerId::from_u128(10);
    group.children.push(StackNode::Layer(child));
    let mut stack = LayerStack::new();
    stack.push_group(group);

    let plan = compile_terrain_plan(&stack, &[], stamp()).expect("pass-through plan");
    assert_eq!(
        shape(&plan),
        ["seed:Zero", "kernel:volcano", "mask", "layer-composite"]
    );
    assert!(plan
        .operations()
        .iter()
        .all(|operation| !matches!(operation.kind, TerrainOpKind::CompositeGroup { .. })));
}

#[test]
fn pass_through_group_and_named_output_have_complete_provenance() {
    let mut child = flat(82, 5.0);
    let declaration = NamedOutputDecl::new("Height", FieldId::Height);
    let output = declaration.id;
    child.common.outputs.push(declaration);
    let mut group = LayerGroup::new("Folder");
    group.id = LayerId::from_u128(81);
    group.children.push(StackNode::Layer(child));
    let mut stack = LayerStack::new();
    stack.push_group(group);

    let plan = compile_terrain_plan(&stack, &[], stamp()).expect("pass-through plan");
    let group_owner = NodeRef::Group(LayerId::from_u128(81));
    assert!(!plan.provenance().spans_for(group_owner).is_empty());
    assert!(!plan.provenance().fields_for(group_owner).is_empty());
    let output = plan.provenance().output(output).expect("published output");
    assert_eq!(output.owner, Some(NodeRef::Layer(LayerId::from_u128(82))));
    assert!(matches!(
        plan.operation(output.publisher).unwrap().kind,
        TerrainOpKind::PublishOutput { .. }
    ));
}

#[test]
fn output_mask_dependency_cycles_are_structured_diagnostics() {
    let mask = MaskId::new();
    let mut layer = flat(90, 1.0);
    let declaration = NamedOutputDecl::new("Height", FieldId::Height);
    let output = declaration.id;
    layer.common.outputs.push(declaration);
    layer.common.masks.push(MaskRef::new(mask));
    let mut stack = LayerStack::new();
    stack.push(layer);
    let asset = MaskAsset::new(
        mask,
        "Feedback",
        MaskSource::LayerOutput { output_id: output },
    );

    let diagnostics = compile_terrain_plan(&stack, &[asset], stamp()).expect_err("cycle");
    assert!(diagnostics.iter().any(|diagnostic| matches!(
        diagnostic,
        TerrainPlanDiagnostic::DependencyCycle { nodes }
            if nodes.contains(&NodeRef::Output(output))
                && nodes.contains(&NodeRef::Mask(mask))
    )));
}

#[test]
fn isolated_copy_input_and_empty_height_keep_private_seeds() {
    let mut copy = LayerGroup::isolated("Copy");
    copy.id = LayerId::from_u128(20);
    copy.children.push(StackNode::Layer(volcano(21)));
    let mut empty = LayerGroup::isolated("Empty");
    empty.id = LayerId::from_u128(30);
    empty.input_mode = GroupInputMode::EmptyHeight;
    empty.children.push(StackNode::Layer(flat(31, 3.0)));
    let mut stack = LayerStack::new();
    stack.push(flat(1, 5.0));
    stack.push_group(copy);
    stack.push_group(empty);

    let plan = compile_terrain_plan(&stack, &[], stamp()).expect("isolated plan");
    let group_seeds: Vec<_> = plan
        .operations()
        .iter()
        .filter_map(|operation| match operation.kind {
            TerrainOpKind::Seed { source, .. } if operation.origin.authored().is_some() => {
                Some(source)
            }
            _ => None,
        })
        .collect();
    assert!(matches!(group_seeds[0], SeedSource::Copy(_)));
    assert_eq!(group_seeds[1], SeedSource::Zero);
    assert_eq!(
        plan.operations()
            .iter()
            .filter(|operation| matches!(operation.kind, TerrainOpKind::CompositeGroup { .. }))
            .count(),
        2
    );
}

#[test]
fn nested_pass_through_and_isolated_groups_preserve_boundaries() {
    let mut folder = LayerGroup::new("Nested folder");
    folder.id = LayerId::from_u128(42);
    folder.children.push(StackNode::Layer(volcano(43)));
    let mut isolated = LayerGroup::isolated("Nested isolated");
    isolated.id = LayerId::from_u128(41);
    isolated.children.push(StackNode::Group(folder));
    let mut outer = LayerGroup::new("Outer folder");
    outer.id = LayerId::from_u128(40);
    outer.children.push(StackNode::Group(isolated));
    let mut stack = LayerStack::new();
    stack.push_group(outer);

    let plan = compile_terrain_plan(&stack, &[], stamp()).expect("nested plan");
    assert_eq!(
        plan.operations()
            .iter()
            .filter(|operation| matches!(operation.kind, TerrainOpKind::CompositeGroup { .. }))
            .count(),
        1
    );
}

#[test]
fn disabled_subtree_emits_nothing() {
    let mut disabled = LayerGroup::isolated("Disabled");
    disabled.id = LayerId::from_u128(50);
    disabled.enabled = false;
    disabled.children.push(StackNode::Layer(volcano(51)));
    let mut stack = LayerStack::new();
    stack.push_group(disabled);

    let plan = compile_terrain_plan(&stack, &[], stamp()).expect("disabled plan");
    assert_eq!(shape(&plan), ["seed:Zero"]);
    assert_eq!(plan.fields().len(), 1);
}

#[test]
fn biome_copy_input_selects_height_delta_composition() {
    let mut biome = LayerGroup::biome("Biome");
    biome.id = LayerId::from_u128(60);
    biome.opacity = 0.65;
    biome.filter_blending = 0.4;
    biome.masks.push_node(DistNode::fill(0.5));
    biome.children.push(StackNode::Layer(volcano(61)));
    let mut stack = LayerStack::new();
    stack.push(flat(1, 10.0));
    stack.push_group(biome);

    let plan = compile_terrain_plan(&stack, &[], stamp()).expect("biome plan");
    assert!(plan.operations().iter().any(|operation| matches!(
        operation.kind,
        TerrainOpKind::CompositeGroup {
            mode: GroupCompositeMode::BiomeHeightDelta,
            ..
        }
    )));
}

#[test]
fn isolated_auxiliary_outputs_cross_the_group_only_through_explicit_merges() {
    let mut surface = Layer::new("Biomes", LayerKind::Biomes(BiomesParams::default()));
    surface.common.id = LayerId::from_u128(65);
    let mut group = LayerGroup::isolated("Aux group");
    group.id = LayerId::from_u128(64);
    group.children.push(StackNode::Layer(surface));
    let mut stack = LayerStack::new();
    stack.push_group(group);

    let plan = compile_terrain_plan(&stack, &[], stamp()).expect("aux plan");
    let aux = plan
        .operations()
        .iter()
        .find_map(|operation| match &operation.kind {
            TerrainOpKind::CompositeGroup { aux, .. } => Some(aux),
            _ => None,
        })
        .expect("isolated group composite");
    assert!(!aux.is_empty());
    assert!(aux.iter().all(|field| field.parent.is_none()));
    assert!(aux
        .iter()
        .all(|field| plan.provenance().producer_of(field.output).is_some()));
}

#[test]
fn recompiling_unchanged_stack_has_identical_structure() {
    let mut stack = LayerStack::new();
    stack.push(flat(1, 10.0));
    stack.push(volcano(2));

    let first = compile_terrain_plan(&stack, &[], stamp()).expect("first plan");
    let second = compile_terrain_plan(&stack, &[], stamp()).expect("second plan");
    assert_eq!(first.structure_signature(), second.structure_signature());
    assert_eq!(first.fields(), second.fields());
    assert_eq!(first.operations(), second.operations());
    assert_eq!(first.final_height(), second.final_height());
}

#[test]
fn parameter_changes_preserve_signature_but_reordering_changes_it() {
    let mut stack = LayerStack::new();
    stack.push(flat(1, 10.0));
    stack.push(volcano(2));
    let original = compile_terrain_plan(&stack, &[], stamp()).expect("original plan");

    let LayerKind::Flat(params) = &mut stack.find_mut(LayerId::from_u128(1)).unwrap().kind else {
        unreachable!();
    };
    params.height = 250.0;
    let parameter_edit = compile_terrain_plan(&stack, &[], stamp()).expect("parameter plan");
    assert_eq!(
        original.structure_signature(),
        parameter_edit.structure_signature()
    );

    stack.nodes.swap(0, 1);
    let reordered = compile_terrain_plan(&stack, &[], stamp()).expect("reordered plan");
    assert_ne!(
        original.structure_signature(),
        reordered.structure_signature()
    );
}

#[test]
fn deferred_semantics_and_broken_nested_mask_refs_are_diagnostic() {
    let missing = MaskId::new();
    let mut solo = flat(70, 1.0);
    solo.common.solo = true;
    solo.common
        .masks
        .push_node(DistNode::mask_ref(MaskRef::new(missing)));
    let solo_id = solo.id();
    let mut selected = LayerGroup::isolated("Selected");
    selected.id = LayerId::from_u128(71);
    selected.input_mode = GroupInputMode::SelectedField(
        terra_core::layer::SelectedGroupInput::Field(terra_core::field_data::FieldId::Height),
    );
    let mut stack = LayerStack::new();
    stack.push(solo);
    stack.push_group(selected);

    let diagnostics = compile_terrain_plan(&stack, &[], stamp()).expect_err("must diagnose");
    assert!(diagnostics.iter().any(|diagnostic| matches!(
        diagnostic,
        TerrainPlanDiagnostic::UnsupportedSolo { layer } if *layer == solo_id
    )));
    assert!(diagnostics.iter().any(|diagnostic| matches!(
        diagnostic,
        TerrainPlanDiagnostic::MissingMask { mask, .. } if *mask == missing
    )));
    assert!(diagnostics.iter().any(|diagnostic| matches!(
        diagnostic,
        TerrainPlanDiagnostic::UnsupportedSelectedField { group }
            if *group == LayerId::from_u128(71)
    )));
}
