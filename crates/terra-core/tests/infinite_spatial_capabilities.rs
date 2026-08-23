use terra_core::invalidation::{InfiniteOperationCapability, SpatialRejectReason};
use terra_core::layer::{
    BlurParams, FbmParams, FlatParams, Layer, LayerGroup, LayerKind, LayerStack, LayerTypeRegistry,
    NoiseParams, StackNode, WorleyParams,
};
use terra_core::mask::{DistNode, DistNodeKind, Distribution};
use terra_core::terrain_plan::{
    compile_terrain_plan, resolve_infinite_plan_domain, PlanStructureRevision,
    TerrainPlanDomainRejectReason, TerrainPlanStamp,
};

fn compile(stack: &LayerStack) -> terra_core::terrain_plan::CompiledTerrainPlan {
    compile_terrain_plan(
        stack,
        &[],
        TerrainPlanStamp::new(PlanStructureRevision::new(184)),
    )
    .unwrap()
}

fn available_halo(stack: &LayerStack) -> u32 {
    let plan = compile(stack);
    resolve_infinite_plan_domain(stack, &[], &plan, plan.final_height())
        .unwrap()
        .operation_halo
}

#[test]
fn slice_one_coordinate_generators_are_zero_halo() {
    let kinds = [
        LayerKind::Flat(FlatParams::default()),
        LayerKind::NoiseValue(NoiseParams::default()),
        LayerKind::NoisePerlin(NoiseParams::default()),
        LayerKind::NoiseOpenSimplex(NoiseParams::default()),
        LayerKind::NoiseWorley(WorleyParams::default()),
        LayerKind::Fbm(FbmParams::default()),
        LayerKind::Ridged(FbmParams::default()),
    ];
    for kind in kinds {
        let mut stack = LayerStack::new();
        stack.push(Layer::new("supported", kind));
        assert_eq!(available_halo(&stack), 0);
    }
}

#[test]
fn bounded_blur_reports_and_accumulates_configured_halo() {
    let mut stack = LayerStack::new();
    stack.push(Layer::new("flat", LayerKind::Flat(FlatParams::default())));
    stack.push(Layer::new(
        "blur-a",
        LayerKind::Blur(BlurParams {
            radius: 3,
            iterations: 2,
        }),
    ));
    stack.push(Layer::new(
        "blur-b",
        LayerKind::Blur(BlurParams {
            radius: 5,
            iterations: 1,
        }),
    ));
    assert_eq!(available_halo(&stack), 11);
}

#[test]
fn converging_dependency_branches_keep_the_widest_halo() {
    let mut stack = LayerStack::new();
    stack.push(Layer::new("flat", LayerKind::Flat(FlatParams::default())));
    let mut group = LayerGroup::isolated("branch");
    group.masks = Distribution::from_nodes(vec![DistNode::new(DistNodeKind::Slope {
        min_deg: 0.0,
        max_deg: 90.0,
    })]);
    group.children.push(StackNode::Layer(Layer::new(
        "blur",
        LayerKind::Blur(BlurParams {
            radius: 5,
            iterations: 1,
        }),
    )));
    stack.push_group(group);

    assert_eq!(available_halo(&stack), 5);
}

#[test]
fn local_downstream_blend_cannot_hide_basin_dependency() {
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "basin",
        LayerKind::RiverCarve(Default::default()),
    ));
    stack.push(Layer::new("local", LayerKind::Flat(FlatParams::default())));
    let plan = compile(&stack);
    let rejection =
        resolve_infinite_plan_domain(&stack, &[], &plan, plan.final_height()).unwrap_err();
    assert_eq!(
        rejection.reason,
        TerrainPlanDomainRejectReason::Infinite(SpatialRejectReason::BasinDependent)
    );
}

#[test]
fn every_registered_layer_is_explicitly_classified() {
    for metadata in LayerTypeRegistry::builtin().all() {
        let layer = LayerTypeRegistry::builtin()
            .create(metadata.type_id)
            .expect("registered layer factory");
        assert_ne!(
            layer.kind.infinite_capability(),
            InfiniteOperationCapability::Unsupported(SpatialRejectReason::UnclassifiedOperation),
            "{} lacks an Infinite capability",
            metadata.type_id
        );
    }
}
