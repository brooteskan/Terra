use std::collections::HashMap;

use terra_core::deps::NodeRef;
use terra_core::eval::{EvalContext, ProcessorRegistry, StackEvaluator};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::ids::LayerId;
use terra_core::layer::{
    blend_heights, FlatParams, GroupInputMode, GroupKind, Layer, LayerGroup, LayerKind, LayerStack,
    StackNode, VolcanoParams,
};
use terra_core::mask::{bake_distribution_with_context, DistBakeContext, DistNode, MaskField};
use terra_core::terrain_plan::{
    compile_terrain_plan, CompiledTerrainPlan, FieldSlot, GroupCompositeMode, PlanOrigin,
    PlanStructureRevision, SeedSource, TerrainOpKind, TerrainPlanStamp,
};

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

fn compile(stack: &LayerStack) -> CompiledTerrainPlan {
    compile_terrain_plan(
        stack,
        &[],
        TerrainPlanStamp::new(PlanStructureRevision::new(1)),
    )
    .expect("representative stack compiles")
}

fn plan_interpret(
    stack: &LayerStack,
    plan: &CompiledTerrainPlan,
    metrics: HeightfieldMetrics,
) -> Heightfield {
    let registry = ProcessorRegistry::builtin();
    let mut ctx = EvalContext::new(metrics);
    let mut heights: HashMap<FieldSlot, Heightfield> = HashMap::new();
    let mut masks: HashMap<FieldSlot, MaskField> = HashMap::new();

    for operation in plan.operations() {
        match &operation.kind {
            TerrainOpKind::Seed { source, output } => {
                let height = match source {
                    SeedSource::Zero => Heightfield::zeros(metrics),
                    SeedSource::Copy(source) => heights[source].clone(),
                    SeedSource::Selected(_) => panic!("selected fields are not in #141 plans"),
                };
                heights.insert(*output, height);
            }
            TerrainOpKind::RunLayerKernel {
                layer,
                input_height,
                input_fields,
                output_candidate,
                output_fields,
                ..
            } => {
                assert!(
                    input_fields.is_empty(),
                    "height fixture unexpectedly reads aux"
                );
                assert!(
                    output_fields.is_empty(),
                    "height fixture unexpectedly writes aux"
                );
                let authored = stack.find(*layer).expect("plan layer exists");
                let candidate = registry
                    .evaluate(&mut ctx, &heights[input_height], authored)
                    .expect("kernel evaluation");
                heights.insert(*output_candidate, candidate);
            }
            TerrainOpKind::EvaluateMask {
                input_height,
                input_fields,
                output_mask,
            } => {
                assert!(
                    input_fields.is_empty(),
                    "height fixture unexpectedly masks with aux"
                );
                let distribution = match operation.origin {
                    PlanOrigin::Authored(NodeRef::Layer(id)) => {
                        &stack.find(id).expect("mask layer exists").common.masks
                    }
                    PlanOrigin::Authored(NodeRef::Group(id)) => {
                        &stack.find_group(id).expect("mask group exists").masks
                    }
                    _ => panic!("mask operation has an invalid owner"),
                };
                let input = &heights[input_height];
                let bake_context = DistBakeContext {
                    height: Some(input),
                    slope_deg: None,
                    curvature: None,
                    flow: None,
                    masks: &ctx.masks,
                    aux: Some(&ctx.aux),
                };
                let mask = bake_distribution_with_context(distribution, metrics, &bake_context);
                masks.insert(*output_mask, mask);
            }
            TerrainOpKind::CompositeLayer {
                layer,
                base,
                candidate,
                mask,
                output,
            } => {
                let authored = stack.find(*layer).expect("composite layer exists");
                let base = &heights[base];
                let candidate = &heights[candidate];
                let mask = &masks[mask];
                let mut composed = base.clone();
                for j in 0..metrics.height {
                    for i in 0..metrics.width {
                        composed.set(
                            i,
                            j,
                            blend_heights(
                                authored.common.blend,
                                base.get(i, j),
                                candidate.get(i, j),
                                authored.common.opacity,
                                mask.get(i, j),
                            ),
                        );
                    }
                }
                composed.refresh_halos();
                heights.insert(*output, composed);
            }
            TerrainOpKind::CompositeGroup {
                group,
                parent,
                private_seed,
                child_output,
                mask,
                output,
                mode,
                aux,
            } => {
                assert!(aux.is_empty(), "height fixture unexpectedly merges aux");
                let authored = stack.find_group(*group).expect("composite group exists");
                let parent = &heights[parent];
                let private_seed = &heights[private_seed];
                let child = &heights[child_output];
                let mask = &masks[mask];
                let opacity = if matches!(authored.group_kind, GroupKind::Biome) {
                    authored.opacity * authored.filter_blending
                } else {
                    authored.opacity
                };
                let mut composed = parent.clone();
                for j in 0..metrics.height {
                    for i in 0..metrics.width {
                        let value = match mode {
                            GroupCompositeMode::Standard => blend_heights(
                                authored.blend,
                                parent.get(i, j),
                                child.get(i, j),
                                opacity,
                                mask.get(i, j),
                            ),
                            GroupCompositeMode::BiomeHeightDelta => {
                                let weight = (mask.get(i, j) * opacity).clamp(0.0, 1.0);
                                parent.get(i, j)
                                    + weight * (child.get(i, j) - private_seed.get(i, j))
                            }
                        };
                        composed.set(i, j, value);
                    }
                }
                composed.refresh_halos();
                heights.insert(*output, composed);
            }
            TerrainOpKind::PublishOutput { .. } => {}
        }
    }

    heights[&plan.final_height()].clone()
}

fn assert_cpu_parity(stack: LayerStack) {
    let metrics = HeightfieldMetrics::new(24, 24, 240.0, 240.0);
    let plan = compile(&stack);
    let planned = plan_interpret(&stack, &plan, metrics);
    let mut context = EvalContext::new(metrics);
    let oracle = StackEvaluator::new()
        .rebuild_all(&stack, &mut context)
        .expect("CPU oracle");
    let max_error = planned
        .to_dense()
        .iter()
        .zip(oracle.to_dense())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_error <= 1.0e-5, "plan/CPU max error: {max_error}");
}

#[test]
fn flat_and_pass_through_plan_matches_cpu_oracle() {
    let mut folder = LayerGroup::new("Folder");
    folder.id = LayerId::from_u128(2);
    folder.children.push(StackNode::Layer(volcano(3)));
    let mut stack = LayerStack::new();
    stack.push(flat(1, 20.0));
    stack.push_group(folder);
    assert_cpu_parity(stack);
}

#[test]
fn isolated_copy_input_and_empty_height_plans_match_cpu_oracle() {
    let mut copy = LayerGroup::isolated("Copy");
    copy.id = LayerId::from_u128(11);
    copy.opacity = 0.7;
    copy.children.push(StackNode::Layer(volcano(12)));
    let mut empty = LayerGroup::isolated("Empty");
    empty.id = LayerId::from_u128(13);
    empty.input_mode = GroupInputMode::EmptyHeight;
    empty.opacity = 0.35;
    empty.children.push(StackNode::Layer(flat(14, 80.0)));
    let mut stack = LayerStack::new();
    stack.push(flat(10, 15.0));
    stack.push_group(copy);
    stack.push_group(empty);
    assert_cpu_parity(stack);
}

#[test]
fn nested_and_masked_biome_plan_matches_cpu_oracle() {
    let mut inner_folder = LayerGroup::new("Inner folder");
    inner_folder.id = LayerId::from_u128(22);
    inner_folder.children.push(StackNode::Layer(volcano(23)));
    let mut biome = LayerGroup::biome("Biome");
    biome.id = LayerId::from_u128(21);
    biome.opacity = 0.8;
    biome.filter_blending = 0.45;
    biome.masks.push_node(DistNode::fill(0.6));
    biome.children.push(StackNode::Group(inner_folder));
    let mut outer_folder = LayerGroup::new("Outer folder");
    outer_folder.id = LayerId::from_u128(20);
    outer_folder.children.push(StackNode::Group(biome));
    let mut stack = LayerStack::new();
    stack.push(flat(19, 30.0));
    stack.push_group(outer_folder);
    assert_cpu_parity(stack);
}
