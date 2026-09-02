use terra_core::deps::NodeRef;
use terra_core::field_data::FieldId;
use terra_core::ids::LayerId;
use terra_core::layer::{
    BindingSource, BiomesParams, FlatParams, GroupInputMode, Layer, LayerGroup, LayerKind,
    LayerStack, NamedOutputDecl, ParamBinding, StackNode, VolcanoParams,
};
use terra_core::mask::{DistNode, MaskAsset, MaskId, MaskRef, MaskSource};
use terra_core::terrain_plan::{
    compile_terrain_plan, GroupCompositeMode, PlanNodeSelection, PlanStructureRevision, SeedSource,
    TerrainOpKind, TerrainPlanDiagnostic, TerrainPlanStamp,
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
            TerrainOpKind::CompositeAuxField { .. } => "aux-composite".into(),
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
            TerrainOpKind::CompositeAuxField { composite, .. } => Some(composite),
            _ => None,
        })
        .expect("isolated group composite");
    assert!(aux.parent.is_none());
    assert!(plan.provenance().producer_of(aux.output).is_some());
    assert!(!plan.analysis().field_is_live(aux.output));
}

#[test]
fn output_parameter_binding_is_an_explicit_kernel_input_and_dependency() {
    let mut producer = flat(140, 0.5);
    let declaration = NamedOutputDecl::new("Control", FieldId::Height);
    let output = declaration.id;
    producer.common.outputs.push(declaration);
    let mut consumer = volcano(141);
    consumer.common.param_bindings.push(ParamBinding::new(
        "amplitude",
        BindingSource::LayerOutput(output),
    ));
    let consumer_id = consumer.id();
    let mut stack = LayerStack::new();
    stack.push(producer);
    stack.push(consumer);

    let plan = compile_terrain_plan(&stack, &[], stamp()).expect("binding plan");
    let published = plan
        .provenance()
        .field_for_output(output)
        .expect("published slot");
    let (operation_id, inputs) = plan
        .operations()
        .iter()
        .enumerate()
        .find_map(|(index, operation)| match &operation.kind {
            TerrainOpKind::RunLayerKernel {
                layer,
                input_fields,
                ..
            } if *layer == consumer_id => Some((
                terra_core::terrain_plan::PlanOpId::from_index(index),
                input_fields,
            )),
            _ => None,
        })
        .expect("consumer kernel");
    assert!(inputs.contains(&published));
    assert!(plan.provenance().dependencies().iter().any(|dependency| {
        dependency.consumer == operation_id
            && dependency.source == NodeRef::Output(output)
            && dependency.kind == terra_core::deps::DepKind::ParamBinding
    }));
}

#[test]
fn disabled_publication_has_a_structured_diagnostic() {
    let mut producer = flat(150, 1.0);
    producer.common.enabled = false;
    let declaration = NamedOutputDecl::new("Disabled", FieldId::Height);
    let output = declaration.id;
    producer.common.outputs.push(declaration);
    let mut selected = LayerGroup::isolated("Consumer");
    selected.id = LayerId::from_u128(151);
    selected.input_mode =
        GroupInputMode::SelectedField(terra_core::layer::SelectedGroupInput::Output(output));
    let mut stack = LayerStack::new();
    stack.push(producer);
    stack.push_group(selected);

    let diagnostics = compile_terrain_plan(&stack, &[], stamp()).expect_err("disabled output");
    assert!(diagnostics.iter().any(|diagnostic| matches!(
        diagnostic,
        TerrainPlanDiagnostic::DisabledOutput { owner, output: candidate }
            if *owner == NodeRef::Group(LayerId::from_u128(151)) && *candidate == output
    )));
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
fn root_solo_filters_siblings_and_records_selection_provenance() {
    let base = flat(70, 100.0);
    let base_id = base.id();
    let mut solo = flat(71, 20.0);
    solo.common.solo = true;
    let solo_id = solo.id();
    let sibling = flat(72, 50.0);
    let sibling_id = sibling.id();
    let mut stack = LayerStack::new();
    stack.push(base);
    stack.push(solo);
    stack.push(sibling);

    let plan = compile_terrain_plan(&stack, &[], stamp()).expect("solo plan");
    assert_eq!(
        shape(&plan),
        ["seed:Zero", "kernel:flat", "mask", "layer-composite"]
    );
    assert_eq!(
        plan.provenance().selection_for(NodeRef::Layer(solo_id)),
        Some(PlanNodeSelection::IncludedBySolo)
    );
    for excluded in [base_id, sibling_id] {
        let owner = NodeRef::Layer(excluded);
        assert_eq!(
            plan.provenance().selection_for(owner),
            Some(PlanNodeSelection::ExcludedBySolo)
        );
        assert!(plan.provenance().operations_for(owner).is_empty());
        assert!(plan.provenance().fields_for(owner).is_empty());
    }
}

#[test]
fn solo_paths_preserve_pass_through_and_isolated_ancestors() {
    let mut selected = flat(83, 25.0);
    selected.common.solo = true;
    let selected_id = selected.id();
    let excluded_id = LayerId::from_u128(84);
    let mut folder = LayerGroup::new("Folder");
    folder.id = LayerId::from_u128(82);
    folder.children.push(StackNode::Layer(selected));
    folder.children.push(StackNode::Layer(flat(84, 50.0)));
    let mut isolated = LayerGroup::isolated("Isolated");
    isolated.id = LayerId::from_u128(81);
    isolated.children.push(StackNode::Group(folder));
    let mut stack = LayerStack::new();
    stack.push(flat(80, 10.0));
    stack.push_group(isolated);

    let plan = compile_terrain_plan(&stack, &[], stamp()).expect("nested solo plan");
    assert_eq!(
        plan.operations()
            .iter()
            .filter(|operation| matches!(operation.kind, TerrainOpKind::CompositeGroup { .. }))
            .count(),
        1
    );
    for owner in [
        NodeRef::Group(LayerId::from_u128(81)),
        NodeRef::Group(LayerId::from_u128(82)),
        NodeRef::Layer(selected_id),
    ] {
        assert_eq!(
            plan.provenance().selection_for(owner),
            Some(PlanNodeSelection::IncludedBySolo)
        );
        assert!(!plan.provenance().spans_for(owner).is_empty());
    }
    assert_eq!(
        plan.provenance().selection_for(NodeRef::Layer(excluded_id)),
        Some(PlanNodeSelection::ExcludedBySolo)
    );
}

#[test]
fn multiple_solo_branches_compile_in_authored_order() {
    let mut first_solo = flat(92, 2.0);
    first_solo.common.solo = true;
    let mut first = LayerGroup::new("First");
    first.id = LayerId::from_u128(91);
    first.children.push(StackNode::Layer(first_solo));
    first.children.push(StackNode::Layer(flat(93, 3.0)));

    let mut second_solo = flat(95, 5.0);
    second_solo.common.solo = true;
    let mut second = LayerGroup::new("Second");
    second.id = LayerId::from_u128(94);
    second.children.push(StackNode::Layer(flat(96, 6.0)));
    second.children.push(StackNode::Layer(second_solo));

    let mut stack = LayerStack::new();
    stack.push_group(first);
    stack.push(flat(90, 1.0));
    stack.push_group(second);
    let first = compile_terrain_plan(&stack, &[], stamp()).expect("multiple solo plan");
    let second = compile_terrain_plan(&stack, &[], stamp()).expect("deterministic solo plan");
    let kernels: Vec<_> = first
        .operations()
        .iter()
        .filter_map(|operation| match operation.kind {
            TerrainOpKind::RunLayerKernel { layer, .. } => Some(layer),
            _ => None,
        })
        .collect();
    assert_eq!(kernels, [LayerId::from_u128(92), LayerId::from_u128(95)]);
    assert_eq!(first.structure_signature(), second.structure_signature());
    assert_eq!(first.operations(), second.operations());
}

#[test]
fn disabled_solo_nodes_select_but_emit_no_work() {
    let mut disabled_solo = flat(101, 10.0);
    disabled_solo.common.enabled = false;
    disabled_solo.common.solo = true;
    let disabled_id = disabled_solo.id();
    let mut stack = LayerStack::new();
    stack.push(flat(100, 1.0));
    stack.push(disabled_solo);
    let plan = compile_terrain_plan(&stack, &[], stamp()).expect("disabled solo plan");
    assert_eq!(shape(&plan), ["seed:Zero"]);
    assert_eq!(
        plan.provenance().selection_for(NodeRef::Layer(disabled_id)),
        Some(PlanNodeSelection::IncludedBySolo)
    );

    let mut nested_solo = flat(104, 4.0);
    nested_solo.common.solo = true;
    let mut disabled_group = LayerGroup::isolated("Disabled solo group");
    disabled_group.id = LayerId::from_u128(103);
    disabled_group.enabled = false;
    disabled_group.children.push(StackNode::Layer(nested_solo));
    let mut nested_stack = LayerStack::new();
    nested_stack.push(flat(102, 2.0));
    nested_stack.push_group(disabled_group);
    let nested = compile_terrain_plan(&nested_stack, &[], stamp()).expect("disabled group plan");
    assert_eq!(shape(&nested), ["seed:Zero"]);
    assert_eq!(
        nested
            .provenance()
            .selection_for(NodeRef::Group(LayerId::from_u128(103))),
        Some(PlanNodeSelection::IncludedBySolo)
    );
}

#[test]
fn solo_toggle_changes_selection_signature_even_when_operations_do_not() {
    let mut stack = LayerStack::new();
    stack.push(flat(110, 1.0));
    let unfiltered = compile_terrain_plan(&stack, &[], stamp()).expect("unfiltered plan");
    stack.find_mut(LayerId::from_u128(110)).unwrap().common.solo = true;
    let solo = compile_terrain_plan(&stack, &[], stamp()).expect("solo plan");
    assert_eq!(unfiltered.operations(), solo.operations());
    assert_ne!(unfiltered.structure_signature(), solo.structure_signature());
}

#[test]
fn selected_field_lowers_and_broken_nested_mask_refs_are_diagnostic() {
    let missing = MaskId::new();
    let mut layer = flat(120, 1.0);
    layer
        .common
        .masks
        .push_node(DistNode::mask_ref(MaskRef::new(missing)));
    let mut selected = LayerGroup::isolated("Selected");
    selected.id = LayerId::from_u128(121);
    selected.input_mode = GroupInputMode::SelectedField(
        terra_core::layer::SelectedGroupInput::Field(terra_core::field_data::FieldId::Height),
    );
    let mut stack = LayerStack::new();
    stack.push(layer);
    stack.push_group(selected);

    let diagnostics = compile_terrain_plan(&stack, &[], stamp()).expect_err("must diagnose");
    assert!(diagnostics.iter().any(|diagnostic| matches!(
        diagnostic,
        TerrainPlanDiagnostic::MissingMask { mask, .. } if *mask == missing
    )));
    assert_eq!(diagnostics.len(), 1);
}

#[test]
fn selected_named_output_lowers_to_explicit_seed_dependency() {
    let mut producer = flat(130, 3.0);
    let output = NamedOutputDecl::new("Published height", FieldId::Height);
    let output_id = output.id;
    producer.common.outputs.push(output);

    let mut selected = LayerGroup::isolated("Selected");
    selected.id = LayerId::from_u128(131);
    selected.input_mode =
        GroupInputMode::SelectedField(terra_core::layer::SelectedGroupInput::Output(output_id));
    selected.children.push(StackNode::Layer(flat(132, 1.0)));

    let mut stack = LayerStack::new();
    stack.push(producer);
    stack.push_group(selected);
    let plan = compile_terrain_plan(&stack, &[], stamp()).expect("selected output plan");

    let selected_seed = plan
        .operations()
        .iter()
        .enumerate()
        .find_map(|(index, operation)| match operation.kind {
            TerrainOpKind::Seed {
                source: SeedSource::Selected(source),
                ..
            } => Some((
                terra_core::terrain_plan::PlanOpId::from_index(index),
                source,
            )),
            _ => None,
        });
    let (seed_op, source) = selected_seed.expect("explicit selected seed");
    assert!(plan.provenance().output(output_id).is_some());
    assert!(plan
        .provenance()
        .dependencies()
        .iter()
        .any(|dependency| dependency.consumer == seed_op
            && dependency.source == NodeRef::Output(output_id)
            && dependency.kind == terra_core::deps::DepKind::GroupInput));
    assert!(plan.operations().iter().any(|operation| matches!(
        operation.kind,
        TerrainOpKind::PublishOutput { output, source: published }
            if output == output_id && published == source
    )));
}
