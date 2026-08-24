use terra_core::deps::NodeRef;
use terra_core::field_data::FieldId;
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::ids::LayerId;
use terra_core::layer::{
    BlendMode, FlatParams, GroupInputMode, GroupKind, Layer, LayerGroup, LayerKind, LayerStack,
    NamedOutputDecl, SculptStrokeParams, SelectedGroupInput, StackNode, ThermalErosionParams,
};
use terra_core::mask::{bake_mask_assets, MaskAsset, MaskId, MaskRef, MaskSource};
use terra_core::terrain_plan::{
    compile_terrain_plan, CompiledTerrainPlan, GroupCompositeMode, PlanOpId, PlanOrigin,
    PlanStructureRevision, TerrainOpKind, TerrainPlanStamp,
};
use terra_core::test_fixtures::{untitled6_document, Untitled6Variant};
use terra_cpu_eval::{EvalContext, StackEvaluator};
use terra_gpu::compiled_plan::{
    GpuFieldResidency, GpuGroupCompositeParams, GpuPlanOperations, GpuPlanResourceCache,
    GpuPlanResourceKey, GpuPlanResourceLayout,
};

fn flat(id: u128, height: f32) -> Layer {
    let mut layer = Layer::new("Flat", LayerKind::Flat(FlatParams { height }));
    layer.common.id = LayerId::from_u128(id);
    layer
}

fn compile(stack: &LayerStack) -> CompiledTerrainPlan {
    compile_with_assets(stack, &[])
}

fn compile_with_assets(stack: &LayerStack, assets: &[MaskAsset]) -> CompiledTerrainPlan {
    compile_terrain_plan(
        stack,
        assets,
        TerrainPlanStamp::new(PlanStructureRevision::new(7)),
    )
    .expect("fixture plan compiles")
}

#[test]
fn group_mask_is_evaluated_from_the_plan_input_field() {
    let gpu = terra_test_gpu::headless_required();
    let mask = MaskAsset::new(MaskId::new(), "constant", MaskSource::Constant(0.6));
    let group_id = LayerId::from_u128(42);
    let mut group = LayerGroup::isolated("Masked group");
    group.id = group_id;
    let mut reference = MaskRef::new(mask.id);
    reference.strength = 0.5;
    group.masks.push(reference);
    group.children.push(StackNode::Layer(flat(43, 20.0)));
    let mut stack = LayerStack::new();
    stack.push(flat(41, 10.0));
    stack.push_group(group);
    let plan = compile_with_assets(&stack, std::slice::from_ref(&mask));
    let (input_height, output_mask) = plan
        .operations()
        .iter()
        .find_map(|operation| match operation {
            terra_core::terrain_plan::TerrainOp {
                origin: PlanOrigin::Authored(NodeRef::Group(id)),
                kind:
                    TerrainOpKind::EvaluateMask {
                        input_height,
                        output_mask,
                        ..
                    },
                ..
            } if *id == group_id => Some((*input_height, *output_mask)),
            _ => None,
        })
        .expect("group mask operation");
    let mut cache = GpuPlanResourceCache::default();
    let resources = cache
        .realize(&gpu.device, &plan, GpuPlanResourceKey::new(64, 4, 1))
        .expect("resources");
    let operations = GpuPlanOperations::new(&gpu.device);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compiled-plan-group-mask"),
        });
    operations
        .fill_field(&gpu.device, &mut encoder, resources, input_height, 123.0)
        .unwrap();
    let authored = &stack.find_group(group_id).unwrap().masks;
    operations
        .evaluate_distribution(
            &gpu.device,
            &mut encoder,
            resources,
            input_height,
            output_mask,
            authored,
            std::slice::from_ref(&mask),
            10.0,
            10.0,
        )
        .unwrap();
    gpu.queue.submit(Some(encoder.finish()));
    let values = read_field(gpu, resources, output_mask);
    assert!(values.iter().all(|value| (*value - 0.3).abs() <= 1.0e-5));
}

#[test]
fn selected_output_seed_copies_the_resolved_gpu_field() {
    let gpu = terra_test_gpu::headless_required();
    let mut producer = flat(44, 2.0);
    let declaration = NamedOutputDecl::new("Seed", FieldId::Height);
    let published = declaration.id;
    producer.common.outputs.push(declaration);
    let mut group = LayerGroup::isolated("Selected");
    group.id = LayerId::from_u128(45);
    group.input_mode = GroupInputMode::SelectedField(SelectedGroupInput::Output(published));
    group.children.push(StackNode::Layer(flat(46, 1.0)));
    let mut stack = LayerStack::new();
    stack.push(producer);
    stack.push_group(group);
    let plan = compile(&stack);
    let (source, output) = plan
        .operations()
        .iter()
        .find_map(|operation| match operation.kind {
            TerrainOpKind::Seed {
                source: terra_core::terrain_plan::SeedSource::Selected(source),
                output,
            } => Some((source, output)),
            _ => None,
        })
        .expect("selected seed");
    let mut cache = GpuPlanResourceCache::default();
    let resources = cache
        .realize(&gpu.device, &plan, GpuPlanResourceKey::new(64, 4, 1))
        .expect("resources");
    let operations = GpuPlanOperations::new(&gpu.device);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compiled-plan-selected-seed"),
        });
    operations
        .fill_field(&gpu.device, &mut encoder, resources, source, 0.375)
        .unwrap();
    operations
        .seed_field(
            &gpu.device,
            &mut encoder,
            resources,
            terra_core::terrain_plan::SeedSource::Selected(source),
            output,
        )
        .unwrap();
    gpu.queue.submit(Some(encoder.finish()));
    assert!(read_field(gpu, resources, output)
        .iter()
        .all(|value| (*value - 0.375).abs() <= 1.0e-6));
}

#[test]
fn selected_output_group_matches_the_cpu_oracle_on_gpu() {
    let gpu = terra_test_gpu::headless_required();
    let metrics = HeightfieldMetrics::new(64, 4, 640.0, 40.0);
    let mut producer = flat(144, 12.0);
    let declaration = NamedOutputDecl::new("Seed", FieldId::Height);
    let published = declaration.id;
    producer.common.outputs.push(declaration);
    let mut group = LayerGroup::isolated("Selected");
    group.id = LayerId::from_u128(145);
    group.input_mode = GroupInputMode::SelectedField(SelectedGroupInput::Output(published));
    let mut child = flat(146, 3.0);
    child.common.blend = BlendMode::Add;
    group.children.push(StackNode::Layer(child));
    let mut stack = LayerStack::new();
    stack.push(producer);
    stack.push(flat(147, 80.0));
    stack.push_group(group);

    let plan = compile(&stack);
    let gpu_height = execute_flat_plan(gpu, &stack, &plan, &[], metrics);
    let mut context = EvalContext::new(metrics);
    let cpu_height = StackEvaluator::new()
        .evaluate_nodes(&stack.nodes, &mut context, &Heightfield::zeros(metrics))
        .expect("CPU oracle");
    let max_error = gpu_height
        .to_dense()
        .iter()
        .zip(cpu_height.to_dense())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_error <= 1.0e-5,
        "selected GPU/CPU max error: {max_error}"
    );
}

#[test]
fn output_backed_mask_reads_the_published_gpu_field() {
    let gpu = terra_test_gpu::headless_required();
    let mut producer = flat(47, 0.0);
    let declaration = NamedOutputDecl::new("Control", FieldId::Height);
    let published = declaration.id;
    producer.common.outputs.push(declaration);
    let asset = MaskAsset::new(
        MaskId::new(),
        "output mask",
        MaskSource::LayerOutput {
            output_id: published,
        },
    );
    let mut consumer = flat(48, 1.0);
    consumer.common.masks.push(MaskRef::new(asset.id));
    let consumer_id = consumer.id();
    let mut stack = LayerStack::new();
    stack.push(producer);
    stack.push(consumer);
    let plan = compile_with_assets(&stack, std::slice::from_ref(&asset));
    let source = plan
        .provenance()
        .field_for_output(published)
        .expect("output slot");
    let (input_height, output_mask) = plan
        .operations()
        .iter()
        .find_map(|operation| match &operation.kind {
            TerrainOpKind::EvaluateMask {
                input_height,
                output_mask,
                ..
            } if operation.origin == PlanOrigin::Authored(NodeRef::Layer(consumer_id)) => {
                Some((*input_height, *output_mask))
            }
            _ => None,
        })
        .expect("consumer mask");
    let mut cache = GpuPlanResourceCache::default();
    let resources = cache
        .realize(&gpu.device, &plan, GpuPlanResourceKey::new(64, 4, 1))
        .expect("resources");
    let operations = GpuPlanOperations::new(&gpu.device);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compiled-plan-output-mask"),
        });
    operations
        .fill_field(&gpu.device, &mut encoder, resources, source, 0.625)
        .unwrap();
    let resolved = std::collections::HashMap::from([(published, source)]);
    operations
        .evaluate_distribution_resolved_region(
            &gpu.device,
            &mut encoder,
            resources,
            input_height,
            output_mask,
            &stack.find(consumer_id).unwrap().common.masks,
            std::slice::from_ref(&asset),
            &resolved,
            1.0,
            1.0,
            None,
        )
        .unwrap();
    gpu.queue.submit(Some(encoder.finish()));
    assert!(read_field(gpu, resources, output_mask)
        .iter()
        .all(|value| (*value - 0.625).abs() <= 1.0e-6));
}

#[test]
fn representative_nested_biome_plan_matches_the_cpu_oracle() {
    let gpu = terra_test_gpu::headless_required();
    let metrics = HeightfieldMetrics::new(64, 4, 640.0, 40.0);
    let mask = MaskAsset::new(MaskId::new(), "biome", MaskSource::Constant(0.6));

    let biome_id = LayerId::from_u128(52);
    let mut biome = LayerGroup::biome("Biome");
    biome.id = biome_id;
    biome.opacity = 0.8;
    biome.filter_blending = 0.5;
    biome.masks.push(MaskRef::new(mask.id));
    biome.children.push(StackNode::Layer(flat(53, 30.0)));
    let mut folder = LayerGroup::new("Folder");
    folder.id = LayerId::from_u128(51);
    folder.children.push(StackNode::Group(biome));

    let empty_id = LayerId::from_u128(54);
    let mut empty = LayerGroup::isolated("Empty");
    empty.id = empty_id;
    empty.input_mode = GroupInputMode::EmptyHeight;
    empty.opacity = 0.25;
    empty.children.push(StackNode::Layer(flat(55, 8.0)));

    let mut stack = LayerStack::new();
    stack.push(flat(50, 10.0));
    stack.push_group(folder);
    stack.push_group(empty);
    let assets = vec![mask];
    let plan = compile_with_assets(&stack, &assets);
    let actual = execute_flat_plan(gpu, &stack, &plan, &assets, metrics);

    let mut context = EvalContext::new(metrics);
    context.mask_assets = assets.clone();
    context.masks = bake_mask_assets(
        &assets,
        &Heightfield::zeros(metrics),
        metrics,
        &std::collections::HashMap::new(),
    );
    let expected = StackEvaluator::new()
        .rebuild_all(&stack, &mut context)
        .expect("CPU oracle");
    terra_gpu::parity::assert_field_parity(
        "compiled-plan.group-tree",
        &actual,
        &expected,
        terra_gpu::parity::EXACT_HEIGHT,
    );
}

fn execute_flat_plan(
    gpu: &terra_test_gpu::TestGpu,
    stack: &LayerStack,
    plan: &CompiledTerrainPlan,
    assets: &[MaskAsset],
    metrics: HeightfieldMetrics,
) -> Heightfield {
    let mut cache = GpuPlanResourceCache::default();
    let resources = cache
        .realize(
            &gpu.device,
            plan,
            GpuPlanResourceKey::new(metrics.width, metrics.height, 1),
        )
        .expect("resources");
    let operations = GpuPlanOperations::new(&gpu.device);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compiled-plan-conformance-executor"),
        });

    for operation in plan.operations() {
        match &operation.kind {
            TerrainOpKind::Seed { source, output } => operations
                .seed_field(&gpu.device, &mut encoder, resources, *source, *output)
                .unwrap(),
            TerrainOpKind::RunLayerKernel {
                layer,
                output_candidate,
                output_fields,
                ..
            } => {
                assert!(output_fields.is_empty(), "flat fixture has no aux outputs");
                let authored = stack.find(*layer).expect("authored layer");
                let LayerKind::Flat(params) = &authored.kind else {
                    panic!("conformance executor only admits Flat kernels");
                };
                operations
                    .fill_field(
                        &gpu.device,
                        &mut encoder,
                        resources,
                        *output_candidate,
                        params.height,
                    )
                    .unwrap();
            }
            TerrainOpKind::EvaluateMask {
                input_height,
                input_fields,
                output_mask,
            } => {
                assert!(input_fields.is_empty(), "aux mask inputs are #147");
                let distribution = match operation.origin {
                    PlanOrigin::Authored(NodeRef::Layer(id)) => {
                        &stack.find(id).expect("mask layer").common.masks
                    }
                    PlanOrigin::Authored(NodeRef::Group(id)) => {
                        &stack.find_group(id).expect("mask group").masks
                    }
                    PlanOrigin::Root => panic!("root mask operation"),
                    PlanOrigin::Authored(NodeRef::Mask(_) | NodeRef::Output(_)) => {
                        panic!("invalid mask owner")
                    }
                };
                operations
                    .evaluate_distribution(
                        &gpu.device,
                        &mut encoder,
                        resources,
                        *input_height,
                        *output_mask,
                        distribution,
                        assets,
                        metrics.dx(),
                        metrics.dz(),
                    )
                    .unwrap();
            }
            TerrainOpKind::CompositeLayer {
                layer,
                base,
                candidate,
                mask,
                output,
            } => {
                let authored = stack.find(*layer).expect("composite layer");
                operations
                    .composite_group(
                        &gpu.device,
                        &mut encoder,
                        resources,
                        *base,
                        *base,
                        *candidate,
                        *mask,
                        *output,
                        GpuGroupCompositeParams {
                            blend: authored.common.blend,
                            opacity: authored.common.opacity,
                            mode: GroupCompositeMode::Standard,
                        },
                    )
                    .unwrap();
            }
            TerrainOpKind::CompositeGroup {
                group,
                parent,
                private_seed,
                child_output,
                mask,
                output,
                mode,
            } => {
                let authored = stack.find_group(*group).expect("composite group");
                let opacity = if authored.group_kind == GroupKind::Biome {
                    authored.opacity * authored.filter_blending
                } else {
                    authored.opacity
                };
                operations
                    .composite_group(
                        &gpu.device,
                        &mut encoder,
                        resources,
                        *parent,
                        *private_seed,
                        *child_output,
                        *mask,
                        *output,
                        GpuGroupCompositeParams {
                            blend: authored.blend,
                            opacity,
                            mode: *mode,
                        },
                    )
                    .unwrap();
            }
            TerrainOpKind::CompositeAuxField { .. } => {
                panic!("height fixture has no aux outputs")
            }
            TerrainOpKind::PublishOutput { .. } => {}
        }
    }
    gpu.queue.submit(Some(encoder.finish()));
    Heightfield::from_dense(metrics, &read_field(gpu, resources, plan.final_height()))
}

fn sibling_groups() -> (LayerStack, LayerId, LayerId) {
    let first_id = LayerId::from_u128(2);
    let second_id = LayerId::from_u128(4);
    let mut first = LayerGroup::isolated("First");
    first.id = first_id;
    first.children.push(StackNode::Layer(flat(3, 20.0)));
    let mut second = LayerGroup::isolated("Second");
    second.id = second_id;
    second.children.push(StackNode::Layer(flat(5, 30.0)));
    let mut stack = LayerStack::new();
    stack.push(flat(1, 10.0));
    stack.push_group(first);
    stack.push_group(second);
    (stack, first_id, second_id)
}

fn group_fields(
    plan: &CompiledTerrainPlan,
    group: LayerId,
) -> (
    terra_core::terrain_plan::FieldSlot,
    terra_core::terrain_plan::FieldSlot,
) {
    plan.operations()
        .iter()
        .find_map(|operation| match operation.kind {
            TerrainOpKind::CompositeGroup {
                group: candidate,
                private_seed,
                child_output,
                ..
            } if candidate == group => Some((private_seed, child_output)),
            _ => None,
        })
        .expect("group composite")
}

#[test]
fn disjoint_sibling_private_fields_reuse_transient_storage() {
    let (stack, first, second) = sibling_groups();
    let plan = compile(&stack);
    let layout = GpuPlanResourceLayout::build(&plan).expect("resource layout");
    let (first_seed, _) = group_fields(&plan, first);
    let (second_seed, _) = group_fields(&plan, second);
    let first_binding = layout.binding(first_seed).expect("first seed binding");
    let second_binding = layout.binding(second_seed).expect("second seed binding");
    assert_eq!(first_binding.residency, GpuFieldResidency::Transient);
    assert_eq!(second_binding.residency, GpuFieldResidency::Transient);
    assert_eq!(
        first_binding.physical, second_binding.physical,
        "disjoint sibling private seeds should reuse compatible backing storage"
    );
    assert!(layout.transient_allocation_count() < layout.transient_field_count());
}

#[test]
fn nested_private_fields_with_overlapping_lifetimes_do_not_alias() {
    let outer_id = LayerId::from_u128(11);
    let inner_id = LayerId::from_u128(12);
    let mut inner = LayerGroup::isolated("Inner");
    inner.id = inner_id;
    inner.children.push(StackNode::Layer(flat(13, 30.0)));
    let mut outer = LayerGroup::isolated("Outer");
    outer.id = outer_id;
    outer.children.push(StackNode::Group(inner));
    let mut stack = LayerStack::new();
    stack.push(flat(10, 10.0));
    stack.push_group(outer);
    let plan = compile(&stack);
    let layout = GpuPlanResourceLayout::build(&plan).expect("resource layout");
    let (outer_seed, _) = group_fields(&plan, outer_id);
    let (inner_seed, _) = group_fields(&plan, inner_id);
    assert_ne!(
        layout.binding(outer_seed).unwrap().physical,
        layout.binding(inner_seed).unwrap().physical,
        "nested private seeds are simultaneously live"
    );
}

#[test]
fn every_live_operation_has_distinct_outputs_and_no_read_write_alias() {
    let (stack, _, _) = sibling_groups();
    let plan = compile(&stack);
    let layout = GpuPlanResourceLayout::build(&plan).expect("resource layout");
    for operation_index in 0..plan.operations().len() {
        let operation = PlanOpId::from_index(operation_index);
        if !plan.analysis().operation_is_live(operation) {
            continue;
        }
        let mut outputs = Vec::new();
        for field in plan.analysis().outputs(operation) {
            if !plan.analysis().field_is_live(*field) {
                continue;
            }
            let id = layout.binding(*field).expect("live binding").physical;
            assert!(
                !outputs.contains(&id),
                "operation {operation_index} has aliased outputs at field {field:?}"
            );
            outputs.push(id);
        }
        for field in plan.analysis().inputs(operation) {
            if !plan.analysis().field_is_live(*field) {
                continue;
            }
            let id = layout.binding(*field).expect("live binding").physical;
            assert!(
                !outputs.contains(&id),
                "operation {operation_index} aliases input {field:?} with an output"
            );
        }
    }
}

#[test]
fn production_untitled6_empty_biomes_share_read_only_seed_resources() {
    let (document, ids) = untitled6_document(64, Untitled6Variant::ProductionTopology);
    let plan = compile(&document.stack);
    let layout = GpuPlanResourceLayout::build(&plan).expect("production resource layout");

    assert_eq!(ids.empty_biomes.len(), 4);
    for group_id in ids.empty_biomes {
        let (private_seed, child_output, output) = plan
            .operations()
            .iter()
            .find_map(|operation| match operation.kind {
                TerrainOpKind::CompositeGroup {
                    group,
                    private_seed,
                    child_output,
                    output,
                    ..
                } if group == group_id => Some((private_seed, child_output, output)),
                _ => None,
            })
            .expect("empty biome composite");
        assert_eq!(
            private_seed, child_output,
            "empty biome must keep its intentional duplicate read-only input"
        );
        assert_ne!(
            layout.binding(private_seed).unwrap().physical,
            layout.binding(output).unwrap().physical,
            "empty biome composite must still keep its write target distinct"
        );
    }
}

#[test]
fn materialization_walk_rebuilds_transient_inputs_but_stops_at_persistent_parent() {
    let (stack, first, _) = sibling_groups();
    let plan = compile(&stack);
    let layout = GpuPlanResourceLayout::build(&plan).expect("resource layout");
    let (composite_index, parent, private_seed) = plan
        .operations()
        .iter()
        .enumerate()
        .find_map(|(index, operation)| match operation.kind {
            TerrainOpKind::CompositeGroup {
                group,
                parent,
                private_seed,
                ..
            } if group == first => Some((index, parent, private_seed)),
            _ => None,
        })
        .expect("first composite");
    let requested = PlanOpId::from_index(composite_index);
    let materialized = layout.materialization_operations(&plan, &[requested]);
    let seed_producer = plan.provenance().producer_of(private_seed).unwrap();
    let parent_producer = plan.provenance().producer_of(parent).unwrap();
    assert!(materialized.contains(&requested));
    assert!(materialized.contains(&seed_producer));
    assert!(
        !materialized.contains(&parent_producer),
        "persistent parent checkpoint must stop backend reconstruction"
    );
}

fn single_group_plan(input_mode: GroupInputMode) -> (LayerStack, LayerId, CompiledTerrainPlan) {
    let group_id = LayerId::from_u128(22);
    let mut group = LayerGroup::isolated("Group");
    group.id = group_id;
    group.input_mode = input_mode;
    group.children.push(StackNode::Layer(flat(23, 18.0)));
    let mut stack = LayerStack::new();
    stack.push(flat(21, 10.0));
    stack.push_group(group);
    let plan = compile(&stack);
    (stack, group_id, plan)
}

#[test]
fn realization_cache_reuses_warm_resources_and_preserves_last_good_on_failure() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let (_, _, plan) = single_group_plan(GroupInputMode::CopyInput);
    let mut cache = GpuPlanResourceCache::default();
    let key = GpuPlanResourceKey::new(64, 8, 1);
    cache
        .realize(&gpu.device, &plan, key)
        .expect("cold realization");
    cache
        .realize(&gpu.device, &plan, key)
        .expect("warm realization");
    assert_eq!(cache.stats().realizations, 1);
    assert_eq!(cache.stats().warm_hits, 1);

    let error = match cache.realize(&gpu.device, &plan, GpuPlanResourceKey::new(0, 8, 1)) {
        Ok(_) => panic!("empty extent must fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("non-empty"));
    assert_eq!(cache.current().expect("last good").key(), key);
    assert_eq!(cache.stats().failed_realizations, 1);

    let resized = GpuPlanResourceKey::new(32, 16, 1);
    cache
        .realize(&gpu.device, &plan, resized)
        .expect("resolution realization");
    assert_eq!(cache.current().unwrap().key(), resized);
    assert_eq!(cache.stats().realizations, 2);
}

#[test]
fn failed_staged_execution_cannot_overwrite_active_last_good_textures() {
    let gpu = terra_test_gpu::headless_required();
    let (_, _, plan) = single_group_plan(GroupInputMode::CopyInput);
    let key = GpuPlanResourceKey::new(64, 2, 1);
    let mut cache = GpuPlanResourceCache::default();
    cache
        .realize(&gpu.device, &plan, key)
        .expect("active resources");
    let operations = GpuPlanOperations::new(&gpu.device);
    {
        let active = cache.current().unwrap();
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compiled-plan-last-good-seed"),
            });
        operations
            .fill_field(&gpu.device, &mut encoder, active, plan.final_height(), 7.0)
            .unwrap();
        gpu.queue.submit(Some(encoder.finish()));
    }

    let candidate = cache
        .stage_candidate(&gpu.device, &plan, key)
        .expect("candidate resources");
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compiled-plan-failed-candidate"),
        });
    assert!(operations
        .fill_field(
            &gpu.device,
            &mut encoder,
            &candidate,
            plan.final_height(),
            f32::NAN,
        )
        .is_err());
    drop(encoder);
    drop(candidate);

    let values = read_field(
        gpu,
        cache.current().expect("active last good"),
        plan.final_height(),
    );
    assert!(values.iter().all(|value| (*value - 7.0).abs() <= 1.0e-6));
    assert_eq!(cache.stats().staged_candidates, 1);
    assert_eq!(cache.stats().committed_candidates, 0);
}

#[test]
fn copy_zero_standard_and_height_delta_primitives_execute_on_gpu() {
    let gpu = terra_test_gpu::headless_required();
    let (_, group_id, plan) = single_group_plan(GroupInputMode::CopyInput);
    let mut cache = GpuPlanResourceCache::default();
    let resources = cache
        .realize(&gpu.device, &plan, GpuPlanResourceKey::new(64, 4, 1))
        .expect("resources");
    let operations = GpuPlanOperations::new(&gpu.device);
    let (parent, private_seed, child_output, mask, output) = plan
        .operations()
        .iter()
        .find_map(|operation| match operation.kind {
            TerrainOpKind::CompositeGroup {
                group,
                parent,
                private_seed,
                child_output,
                mask,
                output,
                ..
            } if group == group_id => Some((parent, private_seed, child_output, mask, output)),
            _ => None,
        })
        .expect("group fields");

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compiled-plan-primitives-test"),
        });
    operations
        .fill_field(&gpu.device, &mut encoder, resources, parent, 10.0)
        .unwrap();
    operations
        .copy_field(&gpu.device, &mut encoder, resources, parent, private_seed)
        .unwrap();
    operations
        .fill_field(&gpu.device, &mut encoder, resources, child_output, 18.0)
        .unwrap();
    operations
        .fill_field(&gpu.device, &mut encoder, resources, mask, 0.5)
        .unwrap();
    operations
        .composite_group(
            &gpu.device,
            &mut encoder,
            resources,
            parent,
            private_seed,
            child_output,
            mask,
            output,
            GpuGroupCompositeParams {
                blend: BlendMode::Normal,
                opacity: 0.4,
                mode: GroupCompositeMode::BiomeHeightDelta,
            },
        )
        .unwrap();
    gpu.queue.submit(Some(encoder.finish()));
    let values = read_field(gpu, resources, output);
    assert!(values.iter().all(|value| (*value - 11.6).abs() <= 1.0e-5));

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compiled-plan-standard-test"),
        });
    operations
        .composite_group(
            &gpu.device,
            &mut encoder,
            resources,
            parent,
            private_seed,
            child_output,
            mask,
            output,
            GpuGroupCompositeParams {
                blend: BlendMode::Add,
                opacity: 0.5,
                mode: GroupCompositeMode::Standard,
            },
        )
        .unwrap();
    gpu.queue.submit(Some(encoder.finish()));
    let values = read_field(gpu, resources, output);
    assert!(values.iter().all(|value| (*value - 14.5).abs() <= 1.0e-5));
}

#[test]
fn group_aux_composite_merges_only_declared_fields() {
    let gpu = terra_test_gpu::headless_required();
    let group_id = LayerId::from_u128(32);
    let mut outside = Layer::new(
        "Outside strokes",
        LayerKind::SculptStrokes(SculptStrokeParams::default()),
    );
    outside.common.id = LayerId::from_u128(31);
    let mut inside = Layer::new(
        "Inside strokes",
        LayerKind::SculptStrokes(SculptStrokeParams::default()),
    );
    inside.common.id = LayerId::from_u128(33);
    let mut group = LayerGroup::isolated("Aux group");
    group.id = group_id;
    group.children.push(StackNode::Layer(inside));
    let mut stack = LayerStack::new();
    stack.push(outside);
    stack.push_group(group);
    stack.push(Layer::new(
        "Aux consumer",
        LayerKind::ThermalErosion(ThermalErosionParams::default()),
    ));
    let plan = compile(&stack);
    let (parent, child, mask, output) = plan
        .operations()
        .iter()
        .find_map(|operation| match &operation.kind {
            TerrainOpKind::CompositeAuxField {
                owner,
                mask,
                composite,
            } if *owner == terra_core::deps::NodeRef::Group(group_id)
                && plan.analysis().field_is_live(composite.output) =>
            {
                Some((composite.parent, composite.child, *mask, composite.output))
            }
            _ => None,
        })
        .expect("group aux merge");
    let parent = parent.expect("outside layer produces the same aux field");
    let mut cache = GpuPlanResourceCache::default();
    let resources = cache
        .realize(&gpu.device, &plan, GpuPlanResourceKey::new(64, 4, 1))
        .expect("resources");
    let operations = GpuPlanOperations::new(&gpu.device);
    let run = |parent_value, child_value, mask_value, channel_class| {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compiled-plan-aux-test"),
            });
        operations
            .fill_field(&gpu.device, &mut encoder, resources, parent, parent_value)
            .unwrap();
        operations
            .fill_field(&gpu.device, &mut encoder, resources, child, child_value)
            .unwrap();
        operations
            .fill_field(&gpu.device, &mut encoder, resources, mask, mask_value)
            .unwrap();
        operations
            .composite_aux(
                &gpu.device,
                &mut encoder,
                resources,
                Some(parent),
                child,
                mask,
                output,
                1.0,
                channel_class,
            )
            .unwrap();
        gpu.queue.submit(Some(encoder.finish()));
        read_field(gpu, resources, output)[0]
    };

    use terra_core::field_data::ChannelClass;
    assert!((run(100.0, 250.0, 1.0, ChannelClass::Metric) - 250.0).abs() <= 1.0e-5);
    assert!((run(100.0, 250.0, 0.4, ChannelClass::Metric) - 160.0).abs() <= 1.0e-5);
    assert_eq!(run(2.0, 3.0, 0.4, ChannelClass::Categorical), 2.0);
    assert_eq!(run(2.0, 3.0, 0.6, ChannelClass::Categorical), 3.0);
    assert!((run(0.2, 0.8, 0.4, ChannelClass::Weight) - 0.44).abs() <= 1.0e-5);
    assert_eq!(run(0.8, 2.0, 1.0, ChannelClass::Weight), 1.0);
}

fn read_field(
    gpu: &terra_test_gpu::TestGpu,
    resources: &terra_gpu::compiled_plan::GpuPlanResources,
    field: terra_core::terrain_plan::FieldSlot,
) -> Vec<f32> {
    let key = resources.key();
    assert_eq!(key.width * 4 % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
    let byte_size = u64::from(key.width) * u64::from(key.height) * 4;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("compiled-plan-field-readback-source"),
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compiled-plan-field-readback"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: resources.texture(field).expect("field texture"),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(key.width * 4),
                rows_per_image: Some(key.height),
            },
        },
        wgpu::Extent3d {
            width: key.width,
            height: key.height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(Some(encoder.finish()));
    terra_gpu::readback_f32(
        &gpu.device,
        &gpu.queue,
        &buffer,
        (key.width * key.height) as usize,
    )
    .expect("field readback")
}

#[test]
fn empty_height_group_seed_is_a_zero_fill() {
    let gpu = terra_test_gpu::headless_required();
    let (_, group_id, plan) = single_group_plan(GroupInputMode::EmptyHeight);
    let (private_seed, _) = group_fields(&plan, group_id);
    let mut cache = GpuPlanResourceCache::default();
    let resources = cache
        .realize(&gpu.device, &plan, GpuPlanResourceKey::new(64, 2, 1))
        .expect("resources");
    let operations = GpuPlanOperations::new(&gpu.device);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compiled-plan-zero-seed"),
        });
    operations
        .fill_field(&gpu.device, &mut encoder, resources, private_seed, 0.0)
        .unwrap();
    gpu.queue.submit(Some(encoder.finish()));
    assert!(read_field(gpu, resources, private_seed)
        .iter()
        .all(|value| *value == 0.0));
}
