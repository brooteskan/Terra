use super::*;
use std::collections::HashMap;
use terra_core::heightfield::HeightfieldMetrics;
use terra_core::layer::{
    BindingSource, BlendMode, BlurParams, CoastalParams, DomainWarpParams, EffectFilterParams,
    FbmParams, FlatParams, FractalNoiseType, GroupInputMode, ImportHeightmapParams, IslandParams,
    Layer, LayerGroup, LayerKind, LayerStack, MaterialsParams, MultiScaleAmplifyParams,
    NamedOutputDecl, NoiseParams, ParamBinding, RiverCarveParams, SculptParams, StackNode,
    Stamp2dParams, StreamPowerParams, ThermalErosionParams, VoronoiParams,
};
use terra_core::layer::{BrushDab, BrushEditable, SculptStrokeKind};
use terra_core::mask::{
    bake_mask_assets, DistributionEntry, MaskAsset, MaskCombine, MaskId, MaskOp, MaskRef,
    MaskSource,
};
use terra_core::quality::PreviewQuality;
use terra_core::terrain_plan::{PlanInvalidation, TerrainPlanCache};
use terra_core::test_fixtures::{untitled6_document, Untitled6Variant};
use terra_core::tiling::UvRect;
use terra_cpu_eval::{EvalContext, StackEvaluator};

fn cpu_oracle(stack: &LayerStack, metrics: HeightfieldMetrics) -> Heightfield {
    let mut evaluator = StackEvaluator::new();
    let mut ctx = EvalContext::new(metrics);
    evaluator
        .rebuild_all(stack, &mut ctx)
        .expect("CPU stack oracle")
}

fn cpu_mask_oracle(
    stack: &LayerStack,
    metrics: HeightfieldMetrics,
    assets: &[MaskAsset],
) -> Heightfield {
    let mut evaluator = StackEvaluator::new();
    let mut ctx = EvalContext::new(metrics);
    ctx.masks = bake_mask_assets(
        assets,
        &Heightfield::zeros(metrics),
        metrics,
        &HashMap::new(),
    );
    ctx.mask_assets = assets.to_vec();
    evaluator
        .rebuild_all(stack, &mut ctx)
        .expect("CPU mask oracle")
}

fn masked_probe_stack(source: MaskSource) -> (LayerStack, MaskAsset, LayerId) {
    let resolution = 32;
    let samples = (0..resolution)
        .flat_map(|j| (0..resolution).map(move |i| i as f32 * 0.5 + j as f32 * 0.25))
        .collect();
    let sculpt = SculptParams {
        width: resolution,
        height: resolution,
        samples,
        fill_height: 0.0,
    };
    let asset = MaskAsset::new(MaskId::new(), "probe mask", source);
    let mut probe = Layer::new("unit probe", LayerKind::Flat(FlatParams { height: 1.0 }));
    probe.common.blend = BlendMode::Add;
    let mut binding = MaskRef::new(asset.id);
    binding.strength = 0.65;
    binding.invert = true;
    probe.common.masks.push(binding);

    let mut stack = LayerStack::new();
    let base = Layer::new("authored base", LayerKind::SculptBase(sculpt));
    let base_id = base.id();
    stack.push(base);
    stack.push(probe);
    (stack, asset, base_id)
}

#[test]
fn cpu_resume_prefix_requires_a_complete_height_only_checkpoint() {
    let flat = Layer::new("Flat", LayerKind::Flat(FlatParams { height: 10.0 }));
    let flat_layers = [&flat];
    assert!(cpu_resume_prefix_is_height_only(&flat_layers, 1));

    let island = Layer::new("Island", LayerKind::Island(IslandParams::default()));
    let island_layers = [&island];
    assert!(!cpu_resume_prefix_is_height_only(&island_layers, 1));

    let mut published = Layer::new("Published", LayerKind::Flat(FlatParams::default()));
    published
        .common
        .outputs
        .push(NamedOutputDecl::new("height checkpoint", FieldId::Height));
    let published_layers = [&published];
    assert!(!cpu_resume_prefix_is_height_only(&published_layers, 1));

    let mut disabled_island = island;
    disabled_island.common.enabled = false;
    let disabled_layers = [&disabled_island];
    assert!(cpu_resume_prefix_is_height_only(&disabled_layers, 1));

    let stream_power = Layer::new(
        "Stream Power",
        LayerKind::StreamPowerErosion(StreamPowerParams::default()),
    );
    let stream_power_layers = [&stream_power];
    assert!(!cpu_resume_prefix_is_height_only(&stream_power_layers, 1));
}

#[test]
fn compatibility_bridge_parameter_stays_connected_to_compiled_execution() {
    let runtime = include_str!("runtime.rs");
    assert!(runtime.contains("bridge_prefix: Option<&Heightfield>"));
    assert!(!runtime.contains("_bridge_prefix"));
    assert!(runtime.contains("Ok(BridgePrefix {"));
    assert!(runtime.contains("evaluate_compiled_with_bridge("));
}

#[test]
fn compatibility_bridge_seeds_the_first_dirty_layer_input() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let mut stack = LayerStack::new();
    let mut cpu_baked = Layer::new(
        "CPU-baked prefix",
        LayerKind::Flat(FlatParams { height: 10.0 }),
    );
    cpu_baked
        .common
        .param_bindings
        .push(ParamBinding::new("height", BindingSource::Constant(1.0)));
    stack.push(cpu_baked);
    let mut suffix = Layer::new("GPU suffix", LayerKind::Flat(FlatParams { height: 2.0 }));
    suffix.common.blend = BlendMode::Add;
    let suffix_id = suffix.id();
    stack.push(suffix);

    let bridge = Heightfield::filled(metrics, 10.0);
    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_dirty(suffix_id);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            Some(&bridge),
        )
        .expect("height-only bridge should execute the GPU suffix");

    assert!(result.fully_gpu);
    assert_eq!(result.resume_cpu_from, None);
    let actual = result.cpu.expect("bridged suffix readback");
    assert!((actual.get(8, 8) - 12.0).abs() < 0.01);
}

/// A precise compiled-plan fallback presents the truthful prefix entering the
/// unsupported operation, with or without a synchronous CPU checkpoint.
#[test]
fn cpu_resume_readback_stops_before_unsupported_suffix() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Base",
        LayerKind::Flat(FlatParams { height: 10.0 }),
    ));
    let mut unsupported = Layer::new(
        "CPU-bound opacity binding",
        LayerKind::Flat(FlatParams { height: 10.0 }),
    );
    unsupported
        .common
        .param_bindings
        .push(ParamBinding::new("opacity", BindingSource::Constant(0.5)));
    stack.push(unsupported);
    let mut downstream = Layer::new(
        "Downstream add",
        LayerKind::Flat(FlatParams { height: 2.0 }),
    );
    downstream.common.blend = BlendMode::Add;
    stack.push(downstream);

    let expected = cpu_oracle(&stack, metrics);

    let mut preview_engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let preview = preview_engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("speculative preview");
    assert_eq!(preview.resume_cpu_from, Some(1));
    assert!(preview.cpu.is_none());
    let speculative = preview_engine
        .readback_current(&gpu.device, &gpu.queue)
        .expect("speculative preview readback for test");
    assert!((speculative.get(8, 8) - 10.0).abs() < 0.01);

    let mut resume_engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let result = resume_engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("CPU checkpoint evaluation");
    assert!(!result.fully_gpu);
    assert_eq!(result.resume_cpu_from, Some(1));
    let checkpoint = result.cpu.expect("height entering unsupported layer");
    assert!((checkpoint.get(8, 8) - 10.0).abs() < 0.01);

    let mut evaluator = StackEvaluator::new();
    let mut ctx = EvalContext::new(metrics);
    let completed = evaluator
        .evaluate_suffix(&stack, &mut ctx, 1, checkpoint)
        .expect("CPU suffix");
    let max_error = completed
        .to_dense()
        .iter()
        .zip(expected.to_dense())
        .map(|(actual, oracle)| (actual - oracle).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error < 0.01, "hybrid vs CPU max error {max_error}");
}

#[test]
fn aux_producing_gpu_prefix_forces_full_cpu_restart() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let land_mask = MaskAsset::new(
        MaskId::new(),
        "Island land",
        MaskSource::Named(terra_core::fields::keys::LAND_MASK.into()),
    );
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Island",
        LayerKind::Island(IslandParams::default()),
    ));
    let mut upper = Layer::new("Land-only add", LayerKind::Flat(FlatParams { height: 5.0 }));
    upper.common.blend = BlendMode::Add;
    upper.common.masks.push(MaskRef::new(land_mask.id));
    stack.push(upper);
    let assets = vec![land_mask];

    let expected = cpu_mask_oracle(&stack, metrics, &assets);
    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &assets,
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("conservative CPU restart");
    assert_eq!(engine.last_graph.cpu_from, Some(1));
    assert_eq!(result.resume_cpu_from, Some(0));
    let checkpoint = result.cpu.expect("layer-zero seed");
    assert!(checkpoint.to_dense().iter().all(|height| *height == 0.0));

    let mut evaluator = StackEvaluator::new();
    let mut ctx = EvalContext::new(metrics);
    ctx.mask_assets = assets.clone();
    ctx.masks = bake_mask_assets(&assets, &checkpoint, metrics, &HashMap::new());
    let completed = evaluator
        .evaluate_suffix(&stack, &mut ctx, 0, checkpoint)
        .expect("full CPU restart");
    assert!(ctx
        .aux_maps
        .get(terra_core::fields::keys::LAND_MASK)
        .is_some());
    assert_eq!(completed.to_dense(), expected.to_dense());
}

/// #144: scoped groups execute from the compiled tree plan instead of falling back.
#[test]
fn scoped_group_executes_compiled_tree_plan() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Base",
        LayerKind::Flat(FlatParams { height: 10.0 }),
    ));
    let mut group = LayerGroup::isolated("Scoped");
    group.input_mode = GroupInputMode::EmptyHeight;
    group.opacity = 0.5;
    group.children.push(StackNode::Layer(Layer::new(
        "Feature",
        LayerKind::Flat(FlatParams { height: 20.0 }),
    )));
    stack.push_group(group);

    let expected = cpu_oracle(&stack, metrics);
    assert!((expected.get(8, 8) - 15.0).abs() < 1.0e-4);

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("compiled tree plan should execute on the GPU");
    assert!(result.fully_gpu);
    assert_eq!(result.cpu_fallback, None);
    let actual = result.cpu.expect("GPU readback");
    assert!((actual.get(8, 8) - expected.get(8, 8)).abs() < 0.01);
}

#[test]
fn stale_compiled_plan_cannot_publish_resources_or_engine_state() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Base",
        LayerKind::Flat(FlatParams { height: 10.0 }),
    ));
    let revision = PlanStructureRevision::new(7);
    let plan = compile_terrain_plan(&stack, &[], TerrainPlanStamp::new(revision))
        .expect("valid flat plan");
    let invalidation = propagate_plan_edits(&plan, &[TerrainEditClass::Structure]);
    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);

    let result = engine.evaluate_compiled_with_intent(
        &gpu.device,
        &gpu.queue,
        &stack,
        &[],
        &plan,
        revision.next(),
        &invalidation,
        metrics,
        PreviewQuality::Draft,
        false,
        GpuEvaluationIntent::Complete,
    );

    assert!(matches!(
        result,
        Err(GpuError::StalePlan {
            plan_revision: 7,
            expected_revision: 8
        })
    ));
    assert!(engine.plan_resources.current().is_none());
    assert_eq!(engine.active_plan_revision, None);
    assert_eq!(engine.last_quality, None);
}

/// #146: solo filtering is compiled as tree selection and executes without fallback.
#[test]
fn solo_stack_executes_compiled_tree_plan() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Base",
        LayerKind::Flat(FlatParams { height: 100.0 }),
    ));
    let mut solo = Layer::new("Solo", LayerKind::Flat(FlatParams { height: 20.0 }));
    solo.common.blend = BlendMode::Add;
    solo.common.solo = true;
    stack.push(solo);
    let mut sibling = Layer::new("Sibling", LayerKind::Flat(FlatParams { height: 50.0 }));
    sibling.common.blend = BlendMode::Add;
    sibling
        .common
        .param_bindings
        .push(ParamBinding::new("height", BindingSource::Constant(0.5)));
    stack.push(sibling);

    let expected = cpu_oracle(&stack, metrics);
    assert!((expected.get(8, 8) - 20.0).abs() < 1.0e-4);

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("compiled solo plan should execute on the GPU");
    assert!(result.fully_gpu);
    assert_eq!(result.cpu_fallback, None);
    assert_eq!(result.resume_cpu_from, None);
    let actual = result.cpu.expect("GPU readback");
    assert!((actual.get(8, 8) - expected.get(8, 8)).abs() < 0.01);
}

#[test]
fn pass_through_group_remains_fully_gpu() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Base",
        LayerKind::Flat(FlatParams { height: 10.0 }),
    ));
    let mut folder = LayerGroup::new("Folder");
    let mut child = Layer::new("Child", LayerKind::Flat(FlatParams { height: 5.0 }));
    child.common.blend = BlendMode::Add;
    folder.children.push(StackNode::Layer(child));
    stack.push_group(folder);

    let expected = cpu_oracle(&stack, metrics);
    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("pass-through folders are flattenable");
    assert!(result.fully_gpu);
    assert_eq!(result.resume_cpu_from, None);
    let actual = result.cpu.expect("GPU readback");
    assert!((actual.get(8, 8) - expected.get(8, 8)).abs() < 0.01);
}

/// Revert check for #48: Coastal must request CPU instead of completing as identity.
#[test]
fn coastal_marks_gpu_preview_incomplete() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Base",
        LayerKind::Flat(FlatParams { height: 20.0 }),
    ));
    stack.push(Layer::new(
        "Coastal",
        LayerKind::Coastal(CoastalParams::default()),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("unsupported layer should select fallback");
    assert!(!result.fully_gpu);
    assert_eq!(result.resume_cpu_from, Some(1));
    assert_eq!(engine.last_graph.cpu_from, Some(1));
}

/// Revert check for #48: OpenSimplex fractals must not error or become Perlin.
#[test]
fn open_simplex_fractals_mark_gpu_preview_incomplete() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    for kind in [
        LayerKind::Fbm(FbmParams {
            noise: FractalNoiseType::OpenSimplex,
            ..FbmParams::default()
        }),
        LayerKind::Ridged(FbmParams {
            noise: FractalNoiseType::OpenSimplex,
            ..FbmParams::default()
        }),
    ] {
        let mut stack = LayerStack::new();
        stack.push(Layer::new("OpenSimplex", kind));
        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        let result = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Draft,
                false,
                None,
            )
            .expect("unsupported noise should select fallback");
        assert!(!result.fully_gpu);
        assert_eq!(result.resume_cpu_from, Some(0));
    }
}

/// A cached height does not make Materials' missing aux outputs GPU-complete.
#[test]
fn cached_materials_height_keeps_cpu_boundary() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Base",
        LayerKind::Flat(FlatParams { height: 20.0 }),
    ));
    let materials = Layer::new(
        "Materials",
        LayerKind::Materials(MaterialsParams::default()),
    );
    let materials_id = materials.id();
    stack.push(materials);
    let mut upper = Layer::new("Upper", LayerKind::Flat(FlatParams { height: 2.0 }));
    upper.common.blend = BlendMode::Add;
    stack.push(upper);

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let cached = Heightfield::filled(metrics, 20.0);
    engine.ingest_height(&gpu.device, &gpu.queue, materials_id, &cached, (20.0, 20.0));
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("cached unsupported height can be presented speculatively");
    assert!(!result.fully_gpu);
    assert_eq!(result.resume_cpu_from, Some(1));
    assert_eq!(engine.last_graph.cpu_from, Some(1));
}

#[test]
fn materials_aux_affects_cpu_suffix_fixture() {
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let hardness_id = MaskId::new();
    let hardness_asset = MaskAsset::new(hardness_id, "Hardness", MaskSource::Hardness);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Base",
        LayerKind::Flat(FlatParams { height: 20.0 }),
    ));
    stack.push(Layer::new(
        "Materials",
        LayerKind::Materials(MaterialsParams::default()),
    ));
    let mut upper = Layer::new("Upper", LayerKind::Flat(FlatParams { height: 10.0 }));
    upper.common.blend = BlendMode::Add;
    upper.common.masks.push(MaskRef::new(hardness_id));
    stack.push(upper);

    let graph = compile_gpu_graph(&stack, std::slice::from_ref(&hardness_asset));
    assert_eq!(graph.cpu_from, Some(1));

    let mut evaluator = StackEvaluator::new();
    let mut ctx = EvalContext::new(metrics);
    ctx.mask_assets = vec![hardness_asset];
    let result = evaluator
        .rebuild_all(&stack, &mut ctx)
        .expect("CPU fallback oracle");
    assert!(ctx.aux_maps.hardness.is_some());
    assert!(
        (result.get(8, 8) - 22.0).abs() < 0.1,
        "downstream hardness mask must observe Materials aux"
    );
}

#[test]
fn constant_height_and_slope_masks_match_cpu_oracle() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    // Equal texture dimensions but unequal world spacing catches dx/dz substitution.
    let metrics = HeightfieldMetrics::new(32, 32, 320.0, 80.0);
    for source in [
        MaskSource::Constant(0.35),
        MaskSource::Height {
            min: 4.0,
            max: 16.0,
        },
        MaskSource::Slope {
            min_deg: 2.0,
            max_deg: 18.0,
        },
    ] {
        let (stack, asset, base_id) = masked_probe_stack(source);
        let expected = cpu_mask_oracle(&stack, metrics, std::slice::from_ref(&asset));
        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        engine.mark_dirty(base_id);
        let result = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                std::slice::from_ref(&asset),
                metrics,
                PreviewQuality::Draft,
                true,
                None,
            )
            .expect("supported GPU mask");
        assert!(result.fully_gpu);
        assert_eq!(result.resume_cpu_from, None);
        assert_eq!(engine.last_eval_stats().mask_scratch_texture_allocations, 4);
        let actual = result.cpu.expect("GPU mask readback");
        let max_error = actual
            .to_dense()
            .iter()
            .zip(expected.to_dense())
            .map(|(gpu, cpu)| (gpu - cpu).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= 1.0e-3,
            "GPU mask exceeded documented tolerance: {max_error}"
        );

        engine.mark_dirty(base_id);
        let warm = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                std::slice::from_ref(&asset),
                metrics,
                PreviewQuality::Draft,
                false,
                None,
            )
            .expect("warm supported GPU mask");
        assert!(warm.fully_gpu);
        let warm_stats = engine.last_eval_stats();
        assert_eq!(warm_stats.mask_scratch_texture_allocations, 0);
        assert_eq!(warm_stats.mask_scratch_reuses, 1);
    }
}

/// #138: a GPU-bakeable mask participates in an in-place filter's preserved
/// outer composite.
#[test]
fn masked_blur_uses_gpu_outer_composite() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let samples = (0..16 * 16)
        .map(|index| if index % 2 == 0 { 0.0 } else { 100.0 })
        .collect();
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "varying base",
        LayerKind::SculptBase(SculptParams {
            width: 16,
            height: 16,
            samples,
            fill_height: 0.0,
        }),
    ));
    let asset = MaskAsset::new(MaskId::new(), "half", MaskSource::Constant(0.5));
    let mut blur = Layer::new("masked blur", LayerKind::Blur(BlurParams::default()));
    blur.common.masks.push(MaskRef::new(asset.id));
    stack.push(blur);

    let expected = cpu_mask_oracle(&stack, metrics, std::slice::from_ref(&asset));
    assert!(
        expected
            .to_dense()
            .iter()
            .any(|height| *height > 1.0 && *height < 99.0),
        "fixture must exercise partial masked filtering"
    );
    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            std::slice::from_ref(&asset),
            metrics,
            PreviewQuality::Full,
            true,
            None,
        )
        .expect("masked blur GPU evaluation");
    assert!(result.fully_gpu);
    let actual = result.cpu.expect("GPU readback");
    let max_error = actual
        .to_dense()
        .iter()
        .zip(expected.to_dense())
        .map(|(gpu, cpu)| (gpu - cpu).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error < 2.2, "masked blur max error {max_error}");
}

/// Revert check for #50: a simulation result must be composited with
/// LayerCommon opacity rather than mutating the entering field directly.
#[test]
fn partial_opacity_simulation_uses_gpu_outer_composite() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 16.0, 16.0);
    let mut samples = vec![0.0; 16 * 16];
    samples[8 * 16 + 8] = 100.0;
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "peaked base",
        LayerKind::SculptBase(SculptParams {
            width: 16,
            height: 16,
            samples,
            fill_height: 0.0,
        }),
    ));
    let mut simulation = Layer::new(
        "partial thermal",
        LayerKind::ThermalErosion(ThermalErosionParams {
            iterations: 2,
            layered_materials: false,
            weathering_rate: 0.0,
            ..ThermalErosionParams::default()
        }),
    );
    simulation.common.opacity = 0.5;
    let simulation_id = simulation.id();
    stack.push(simulation);

    let expected = cpu_oracle(&stack, metrics);
    assert!(
        expected.get(8, 8) < 99.0,
        "fixture must exercise partial outer compositing"
    );
    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Full,
            true,
            None,
        )
        .expect("partial thermal GPU evaluation");
    assert!(result.fully_gpu);
    let actual = result.cpu.expect("GPU readback");
    let mut full_stack = stack.clone();
    full_stack
        .find_mut(simulation_id)
        .expect("thermal layer")
        .common
        .opacity = 1.0;
    let mut full_engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let full = full_engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &full_stack,
            &[],
            metrics,
            PreviewQuality::Full,
            true,
            None,
        )
        .expect("full-opacity thermal GPU evaluation")
        .cpu
        .expect("GPU readback");
    let max_error = actual
        .to_dense()
        .iter()
        .zip(full.to_dense())
        .enumerate()
        .map(|(index, (partial, filtered))| {
            let base = if index == 8 * 16 + 8 { 100.0 } else { 0.0 };
            (partial - (base + (filtered - base) * 0.5)).abs()
        })
        .fold(0.0f32, f32::max);
    // Thermal redistribution is a bounded preview approximation; independent
    // dispatches need headroom for its ordering variance. A missing outer
    // composite on this fixture misses by roughly fifty metres.
    assert!(
        max_error < 5.0,
        "partial thermal composite error {max_error}"
    );
}

/// Configuration coverage for #138: every persisted height blend equation is
/// evaluated by its matching WGSL formula rather than selecting CPU fallback.
#[test]
fn extended_generator_blends_match_cpu_oracle() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    for mode in [
        BlendMode::HeightBlend,
        BlendMode::SmoothMaximum,
        BlendMode::SmoothMinimum,
        BlendMode::SmoothUnion,
        BlendMode::SmoothSubtraction,
    ] {
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "base",
            LayerKind::Flat(FlatParams { height: 10.0 }),
        ));
        let mut contribution =
            Layer::new("contribution", LayerKind::Flat(FlatParams { height: 20.0 }));
        contribution.common.blend = mode;
        stack.push(contribution);

        let expected = cpu_oracle(&stack, metrics);
        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        let result = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Full,
                true,
                None,
            )
            .expect("extended blend GPU evaluation");
        assert!(result.fully_gpu, "{mode:?}");
        let actual = result.cpu.expect("GPU readback");
        let max_error = actual
            .to_dense()
            .iter()
            .zip(expected.to_dense())
            .map(|(gpu, cpu)| (gpu - cpu).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 1.0e-4, "{mode:?} max error {max_error}");
    }
}

#[test]
fn ordered_and_operated_masks_match_cpu_oracle_on_gpu() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);

    let first = MaskAsset::new(MaskId::new(), "first", MaskSource::Constant(0.8));
    let second = MaskAsset::new(MaskId::new(), "second", MaskSource::Constant(0.25));
    let mut ordered = Layer::new("ordered", LayerKind::Flat(FlatParams { height: 1.0 }));
    ordered.common.masks.push(MaskRef::new(first.id));
    ordered.common.masks.entries.push(DistributionEntry {
        mask: MaskRef::new(second.id),
        combine: MaskCombine::Subtract,
    });

    let mut operated = MaskAsset::new(MaskId::new(), "operated", MaskSource::Constant(0.2));
    operated.ops.push(MaskOp::Invert);
    let operated_layer = {
        let mut layer = Layer::new("operated", LayerKind::Flat(FlatParams { height: 1.0 }));
        layer.common.masks.push(MaskRef::new(operated.id));
        layer
    };

    for (layer, assets, expected) in [
        (ordered, vec![first, second], 0.55),
        (operated_layer, vec![operated], 0.8),
    ] {
        let layer_id = layer.id();
        let mut stack = LayerStack::new();
        stack.push(layer);
        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        engine.mark_dirty(layer_id);
        let result = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &assets,
                metrics,
                PreviewQuality::Full,
                true,
                None,
            )
            .expect("GPU mask program");
        assert!(result.fully_gpu);
        let actual = result.cpu.expect("GPU readback");
        assert!(actual
            .to_dense()
            .iter()
            .all(|v| (v - expected).abs() < 1.0e-5));
    }
}

#[test]
fn changed_mask_operations_remain_gpu_resident_after_invalidation() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let mut asset = MaskAsset::new(MaskId::new(), "mask", MaskSource::Constant(0.5));
    let mut layer = Layer::new("masked", LayerKind::Flat(FlatParams { height: 10.0 }));
    layer.common.masks.push(MaskRef::new(asset.id));
    let layer_id = layer.id();
    let mut stack = LayerStack::new();
    stack.push(layer);

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_dirty(layer_id);
    let first = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            std::slice::from_ref(&asset),
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("initial supported mask");
    assert!(first.fully_gpu);

    asset.ops.push(MaskOp::Invert);
    engine.mark_dirty(layer_id);
    let second = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            std::slice::from_ref(&asset),
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("operated mask should stay on GPU");
    assert!(second.fully_gpu);
    assert_eq!(second.resume_cpu_from, None);
}

/// Regression for uniform isolation via pool slots: Draft must composite layers
/// even when blend and cache share one submit (each pass gets its own uniform buffer).
#[test]
fn draft_eval_composites_sculpt_and_noise() {
    let Some(gpu) = terra_test_gpu::headless() else {
        // Headless CI without a GPU adapter.
        return;
    };
    let metrics = HeightfieldMetrics::new(64, 64, 64.0, 64.0);
    let mut stack = LayerStack::new();
    let base = Layer::new(
        "Base",
        LayerKind::SculptBase(SculptParams::filled(64, 20.0)),
    );
    let base_id = base.id();
    stack.push(base);
    stack.push(Layer::new(
        "Hills",
        LayerKind::NoiseValue(NoiseParams {
            seed: 1,
            frequency: 0.05,
            amplitude: 10.0,
            octaves: 1,
            lacunarity: 2.0,
            persistence: 0.5,
            ..NoiseParams::default()
        }),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_dirty(base_id);
    let before = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("draft eval");
    let hf0 = before.cpu.expect("cpu readback");
    let center0 = hf0.get(32, 32);
    // Without per-pass submit, blends see opacity 0 and the field stays ~0.
    assert!(
        center0 > 15.0,
        "expected sculpt base (~20) through Draft blend, got {center0}"
    );

    // Live raise: stamp then re-eval Draft without waiting for CPU refine.
    {
        let mut layers = stack.flatten_layers_mut();
        if let LayerKind::SculptBase(params) = &mut layers[0].kind {
            params.stamp_circle(0.5, 0.5, 0.15, 25.0, 0);
        }
    }
    engine.mark_dirty_from(&stack, base_id);
    let after = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("draft eval after stamp");
    let hf1 = after.cpu.expect("cpu readback");
    let center1 = hf1.get(32, 32);
    assert!(
        center1 > center0 + 5.0,
        "live Raise should lift Draft heights while held; before={center0} after={center1}"
    );
}

#[test]
fn draft_eval_applies_effect_filter() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(64, 64, 64.0, 64.0);
    let mut stack = LayerStack::new();
    let base = Layer::new(
        "Base",
        LayerKind::SculptBase(SculptParams::filled(64, 20.0)),
    );
    let base_id = base.id();
    stack.push(base);
    let filter = Layer::new(
        "Inflate",
        LayerKind::EffectFilter(terra_core::layer::EffectFilterParams::inflate()),
    );
    let filter_id = filter.id();
    stack.push(filter);

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_dirty(base_id);
    let before = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("draft with filter");
    assert!(before.did_eval);
    assert!(before.fully_gpu);
    let h0 = before.cpu.expect("cpu").get(32, 32);

    // Disable filter and compare — inflate should have raised the surface.
    if let Some(layer) = stack.find_mut(filter_id) {
        layer.common.enabled = false;
    }
    engine.mark_dirty_from(&stack, base_id);
    let after = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("draft without filter");
    let h1 = after.cpu.expect("cpu").get(32, 32);
    assert!(
        h0 > h1 + 0.5,
        "Inflate EffectFilter should raise Draft heights; with={h0} without={h1}"
    );
}

#[test]
fn draft_eval_flat_survives_cache_copy_uniform() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(32, 32, 32.0, 32.0);
    let mut stack = LayerStack::new();
    let layer = Layer::new("Flat", LayerKind::Flat(FlatParams { height: 50.0 }));
    let id = layer.id();
    stack.push(layer);

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_dirty(id);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("flat draft");
    let hf = result.cpu.expect("cpu");
    let mid = hf.get(16, 16);
    assert!(
        (mid - 50.0).abs() < 0.01,
        "Flat blend must not be clobbered by cache CopyU; got {mid}"
    );
}

fn assert_working_texture_dimensions(engine: &GpuTerrainEngine, width: u32, height: u32) {
    for texture in [
        &engine.ping,
        &engine.pong,
        &engine.layer_tex,
        &engine.mask_ones,
        &engine.unit_mask,
        &engine.stamp_mask,
        &engine.hardness,
        &engine.water_a,
        &engine.water_b,
        &engine.delta,
        &engine.sed_a,
        &engine.sed_b,
        &engine.rainfall,
        &engine.loose_sediment,
        &engine.amplify_a,
        &engine.amplify_b,
    ] {
        assert_eq!((texture.width, texture.height), (width, height));
        assert_eq!(
            (texture.texture.width(), texture.texture.height()),
            (width, height)
        );
    }
    assert_eq!(
        (
            engine.outflow._texture.width(),
            engine.outflow._texture.height()
        ),
        (width, height)
    );
}

/// Revert check for #35: reset must release project-sized evaluator textures,
/// and the existing evaluation size check must restore the next document size.
#[test]
fn project_reset_shrinks_working_set_and_evaluate_restores_size() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let mut engine = GpuTerrainEngine::new(&gpu.device, 64);
    let cached_layer = Layer::new("Cached", LayerKind::Flat(FlatParams { height: 1.0 }));
    let cached_id = cached_layer.id();
    engine.layer_cache.insert(
        cached_id,
        HeightTex::new(&gpu.device, "reset-test-cache", 64, 64),
    );
    engine.mark_dirty(cached_id);

    engine.reset_project_state(&gpu.device, &gpu.queue);

    assert_working_texture_dimensions(
        &engine,
        PROJECT_RESET_TEXTURE_EXTENT,
        PROJECT_RESET_TEXTURE_EXTENT,
    );
    assert_eq!(
        (engine.metrics.width, engine.metrics.height),
        (PROJECT_RESET_TEXTURE_EXTENT, PROJECT_RESET_TEXTURE_EXTENT)
    );
    assert!(engine.layer_cache.is_empty());
    assert!(engine.dirty.is_empty());
    assert!(engine.dirty_tiles().is_empty());
    assert_eq!(engine.current, 0);

    let next_metrics = HeightfieldMetrics::new(32, 48, 320.0, 480.0);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &LayerStack::new(),
            &[],
            next_metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("empty evaluation after reset");

    assert_eq!((result.width, result.height), (32, 48));
    assert_working_texture_dimensions(&engine, 32, 48);
}

/// A spatially varying sculpt buffer so smoothing filters have real gradients
/// to act on (a flat fill would make Smooth an identity and hide halo errors).
fn varied_sculpt(res: u32) -> SculptParams {
    let mut sculpt = SculptParams::filled(res, 20.0);
    for y in 0..res {
        for x in 0..res {
            let fx = x as f32;
            let fy = y as f32;
            sculpt.samples[(y * res + x) as usize] =
                20.0 + 12.0 * (fx * 0.35).sin() + 10.0 * (fy * 0.27).cos();
        }
    }
    sculpt
}

/// B1-D6 revert guard: the executor must dispatch exactly the kernels the
/// compiler recorded, in flat order — not a re-derived plan, and not a
/// decorative list the walk ignores.
#[test]
fn engine_executes_kernels_from_the_compiled_plan() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(48, 48, 96.0, 96.0);
    let mut stack = LayerStack::new();
    let base = Layer::new("base", LayerKind::Flat(FlatParams { height: 8.0 }));
    let base_id = base.id();
    stack.push(base);
    stack.push(Layer::new(
        "noise",
        LayerKind::NoiseValue(NoiseParams::default()),
    ));
    stack.push(Layer::new(
        "smooth",
        LayerKind::EffectFilter(EffectFilterParams::smooth()),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_dirty(base_id);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("fully-GPU evaluation");

    let planned: Vec<GpuKernel> = engine
        .last_graph
        .plans
        .iter()
        .flatten()
        .map(|plan| plan.kernel)
        .collect();
    assert_eq!(
        planned,
        vec![GpuKernel::Fill, GpuKernel::Noise, GpuKernel::EffectFilter]
    );
    assert_eq!(
        engine.executed_kernels, planned,
        "executor must consume the compiled plan, not re-plan or ignore it"
    );
}

fn raise_strokes(u: f32, v: f32, strength: f32) -> SculptStrokeParams {
    SculptStrokeParams {
        strokes: vec![terra_core::layer::SculptStroke {
            kind: SculptStrokeKind::Raise,
            points: vec![terra_core::layer::SculptPoint {
                u,
                v,
                pressure: 1.0,
            }],
            radius_m: 60.0,
            strength,
            target_height: 0.0,
            falloff: 1.5,
            enabled: true,
        }],
        reconcile: 0.15,
    }
}

fn flatten_strokes(u: f32, v: f32) -> SculptStrokeParams {
    SculptStrokeParams {
        strokes: vec![terra_core::layer::SculptStroke {
            kind: SculptStrokeKind::Flatten,
            points: vec![terra_core::layer::SculptPoint {
                u,
                v,
                pressure: 1.0,
            }],
            radius_m: 60.0,
            strength: 1.0,
            target_height: 0.0,
            falloff: 1.5,
            enabled: true,
        }],
        reconcile: 0.0,
    }
}

/// #107: the historical #98 Voronoi + Flatten stack now has executable
/// kernels for both layers, so a Full app-style evaluation needs no CPU resume.
#[test]
fn voronoi_flatten_stack_is_fully_gpu_at_full_quality() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(48, 40, 240.0, 160.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "voronoi",
        LayerKind::VoronoiRegions(VoronoiParams::default()),
    ));
    stack.push(Layer::new(
        "flatten",
        LayerKind::SculptStrokes(flatten_strokes(0.5, 0.5)),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Full,
            false,
            None,
        )
        .expect("Full Voronoi + Flatten GPU evaluation");

    assert!(result.fully_gpu);
    assert_eq!(result.freshness, GpuPreviewFreshness::Current);
    assert_eq!(result.resume_cpu_from, None);
    assert_eq!(result.cpu_fallback, None);
    assert_eq!(
        engine.executed_kernels,
        vec![GpuKernel::Noise, GpuKernel::SculptStrokes]
    );
}

#[test]
fn sculpt_strokes_kernel_is_executed_from_the_plan() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
    let mut stack = LayerStack::new();
    let base = Layer::new("base", LayerKind::Flat(FlatParams { height: 5.0 }));
    let base_id = base.id();
    stack.push(base);
    stack.push(Layer::new(
        "strokes",
        LayerKind::SculptStrokes(raise_strokes(0.5, 0.5, 10.0)),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_dirty(base_id);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("fully-GPU stroke evaluation");

    let planned: Vec<GpuKernel> = engine
        .last_graph
        .plans
        .iter()
        .flatten()
        .map(|plan| plan.kernel)
        .collect();
    assert_eq!(planned, vec![GpuKernel::Fill, GpuKernel::SculptStrokes]);
    assert_eq!(engine.executed_kernels, planned);
}

/// #126 regression: procedural shapes publish a reusable contribution. An
/// upstream sculpt edit must recompute the input-dependent stroke layer but
/// blend the cached Volcano contribution without dispatching Shape again.
#[test]
fn warm_cache_base_edit_reuses_input_independent_volcano_contribution() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
    let mut stack = LayerStack::new();
    let base = Layer::new("base", LayerKind::SculptBase(SculptParams::filled(32, 5.0)));
    let base_id = base.id();
    stack.push(base);
    stack.push(Layer::new(
        "strokes",
        LayerKind::SculptStrokes(raise_strokes(0.5, 0.5, 10.0)),
    ));
    stack.push(Layer::new(
        "volcano",
        LayerKind::Volcano(terra_core::layer::VolcanoParams::default()),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm shape contribution cache");
    assert!(engine.plan_resources.current().is_some());

    let Some(layer) = stack.find_mut(base_id) else {
        panic!("base layer disappeared");
    };
    let LayerKind::SculptBase(params) = &mut layer.kind else {
        panic!("base changed kind");
    };
    params.samples[(16 * 32 + 16) as usize] += 3.0;
    engine.set_dirty_rect(Some((16, 16, 1, 1)));
    engine.mark_dirty(base_id);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("incremental base edit");

    assert_eq!(
        engine.executed_kernels,
        vec![GpuKernel::Sculpt, GpuKernel::SculptStrokes]
    );
    assert!(
        !engine.executed_kernels.contains(&GpuKernel::Shape),
        "cached Volcano contribution must avoid Shape dispatch"
    );
    let stats = engine.last_eval_stats();
    assert!(stats.used_layer_zero_region);
    assert_eq!(stats.reused_contributions, 1);
    assert!(stats.sculpt_resampled_texels < u64::from(metrics.width * metrics.height));
}

/// #107: VoronoiRegions is input-independent, so an upstream bounded edit
/// re-blends its warm contribution without re-running the 3x3 Worley search.
#[test]
fn warm_cache_base_edit_reuses_voronoi_contribution() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 48u32;
    let metrics = HeightfieldMetrics::new(res, res, 240.0, 240.0);
    let rect = (20u32, 20u32, 8u32, 8u32);
    let mut stack = LayerStack::new();
    let base = Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(res, 5.0)),
    );
    let base_id = base.id();
    stack.push(base);
    let mut voronoi = Layer::new(
        "voronoi",
        LayerKind::VoronoiRegions(VoronoiParams::default()),
    );
    voronoi.common.blend = BlendMode::Add;
    stack.push(voronoi);

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm Voronoi contribution cache");

    let LayerKind::SculptBase(params) = &mut stack.find_mut(base_id).expect("base layer").kind
    else {
        panic!("base changed kind");
    };
    for y in rect.1..rect.1 + rect.3 {
        for x in rect.0..rect.0 + rect.2 {
            params.samples[(y * res + x) as usize] += 7.0;
        }
    }
    engine.set_dirty_rect(Some(rect));
    engine.mark_dirty(base_id);
    let incremental = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("bounded edit with cached Voronoi contribution")
        .cpu
        .expect("incremental GPU readback");

    let mut oracle_engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    oracle_engine.mark_all_dirty(&stack);
    let oracle = oracle_engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("fresh Voronoi GPU oracle")
        .cpu
        .expect("oracle GPU readback");

    let error = terra_gpu::parity::max_abs_diff(&incremental.to_dense(), &oracle.to_dense());
    assert!(
        error <= 1.0e-3,
        "cached Voronoi contribution drifted by {error}"
    );
    assert_eq!(engine.executed_kernels, vec![GpuKernel::Sculpt]);
    assert_eq!(engine.last_eval_stats().reused_contributions, 1);
}

/// #136: a warm first-layer SculptBase edit remains bounded through upload,
/// SculptStrokes, cached Volcano re-blend, composite-cache maintenance, and
/// presentation while matching a fresh full-field GPU oracle everywhere.
#[test]
fn warm_layer_zero_sculpt_edit_is_region_complete() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 96u32;
    let mut metrics = HeightfieldMetrics::new(res, res, 960.0, 960.0);
    metrics.tile_size = 16;
    metrics.halo = 2;
    let rect = (40u32, 40u32, 16u32, 16u32);

    let mut stack = LayerStack::new();
    let base = Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(24, 12.0)),
    );
    let base_id = base.id();
    stack.push(base);
    stack.push(Layer::new(
        "strokes",
        LayerKind::SculptStrokes(raise_strokes(0.5, 0.5, 8.0)),
    ));
    stack.push(Layer::new(
        "volcano",
        LayerKind::Volcano(terra_core::layer::VolcanoParams::default()),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, res);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm representative stack");

    let LayerKind::SculptBase(params) = &mut stack.find_mut(base_id).expect("base layer").kind
    else {
        panic!("base changed kind");
    };
    params.stamp_circle(0.5, 0.5, 0.06, 4.0, 0);

    engine.set_dirty_rect(Some(rect));
    engine.mark_dirty(base_id);
    let incremental = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("bounded warm edit")
        .cpu
        .expect("incremental readback");

    let mut oracle_engine = GpuTerrainEngine::new(&gpu.device, res);
    let oracle = oracle_engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("fresh full-field oracle")
        .cpu
        .expect("oracle readback");
    let error = terra_gpu::parity::max_abs_diff(&incremental.to_dense(), &oracle.to_dense());
    let worst = incremental
        .to_dense()
        .iter()
        .zip(oracle.to_dense())
        .enumerate()
        .max_by(|(_, (a0, b0)), (_, (a1, b1))| (*a0 - *b0).abs().total_cmp(&(*a1 - *b1).abs()))
        .map(|(index, (a, b))| (index % res as usize, index / res as usize, *a, b));
    assert!(
        error <= 1.0e-3,
        "bounded layer-zero edit drifted by {error} at {worst:?}"
    );

    let stats = engine.last_eval_stats();
    assert!(stats.used_layer_zero_region);
    assert_eq!(stats.reused_contributions, 1);
    assert_eq!(
        engine.executed_kernels,
        vec![GpuKernel::Sculpt, GpuKernel::SculptStrokes]
    );
    assert!(
        engine.dirty_tiles().len() < (metrics.tiles_x() * metrics.tiles_z()) as usize,
        "a warm compiled-plan stroke suffix must preserve bounded presentation"
    );
    assert!(
        stats.upload_bytes < u64::from(metrics.width * metrics.height * 4),
        "compiled-plan Base upload must scale with the expanded edit region"
    );
}

#[test]
fn layer_zero_region_requires_warm_stable_unmasked_caches() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 48u32;
    let mut metrics = HeightfieldMetrics::new(res, res, 480.0, 480.0);
    metrics.tile_size = 8;
    let rect = (20, 20, 4, 4);
    let build = || {
        let mut stack = LayerStack::new();
        let base = Layer::new(
            "base",
            LayerKind::SculptBase(SculptParams::filled(res, 10.0)),
        );
        let id = base.id();
        stack.push(base);
        stack.push(Layer::new(
            "strokes",
            LayerKind::SculptStrokes(raise_strokes(0.5, 0.5, 5.0)),
        ));
        stack.push(Layer::new(
            "volcano",
            LayerKind::Volcano(terra_core::layer::VolcanoParams::default()),
        ));
        (stack, id)
    };

    let (cold_stack, cold_id) = build();
    let mut cold = GpuTerrainEngine::new(&gpu.device, res);
    cold.set_dirty_rect(Some(rect));
    cold.mark_dirty(cold_id);
    cold.evaluate(
        &gpu.device,
        &gpu.queue,
        &cold_stack,
        &[],
        metrics,
        PreviewQuality::Draft,
        false,
        None,
    )
    .expect("cold fallback");
    assert!(!cold.last_eval_stats().used_layer_zero_region);
    assert_eq!(
        cold.dirty_tiles().len(),
        (metrics.tiles_x() * metrics.tiles_z()) as usize
    );

    let (quality_stack, quality_id) = build();
    let mut quality = GpuTerrainEngine::new(&gpu.device, res);
    quality.mark_all_dirty(&quality_stack);
    quality
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &quality_stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm quality caches");
    quality.set_dirty_rect(Some(rect));
    quality.mark_dirty(quality_id);
    quality
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &quality_stack,
            &[],
            metrics,
            PreviewQuality::Medium,
            false,
            None,
        )
        .expect("quality-change fallback");
    assert!(!quality.last_eval_stats().used_layer_zero_region);

    let mask = MaskAsset::new(MaskId::new(), "constant", MaskSource::Constant(0.5));
    let (mut masked_stack, masked_id) = build();
    masked_stack
        .find_mut(masked_id)
        .expect("masked base")
        .common
        .masks
        .push(MaskRef::new(mask.id));
    let mut masked = GpuTerrainEngine::new(&gpu.device, res);
    masked.mark_all_dirty(&masked_stack);
    masked
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &masked_stack,
            std::slice::from_ref(&mask),
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm masked stack");
    masked.set_dirty_rect(Some(rect));
    masked.mark_dirty(masked_id);
    masked
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &masked_stack,
            std::slice::from_ref(&mask),
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("masked fallback");
    assert!(masked.last_eval_stats().used_layer_zero_region);
}

/// #133 regression: decoded source textures are shared by asset identity,
/// while each layer keeps an output-sized contribution cache. A bounded base
/// edit therefore reblends both raster contributions without decoding,
/// uploading, or dispatching the sampling kernel again.
#[test]
fn warm_cache_base_edit_reuses_heightmap_contributions_and_source_texture() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("terra-gpu-cache-{unique}.png"));
    let fixture = image::ImageBuffer::from_fn(5, 7, |x, y| {
        image::Luma([((x * 8000 + y * 6000) % 65536) as u16])
    });
    fixture.save(&path).expect("write source fixture");

    let metrics = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
    let mut stack = LayerStack::new();
    let base = Layer::new("base", LayerKind::SculptBase(SculptParams::filled(32, 5.0)));
    let base_id = base.id();
    stack.push(base);
    let params = ImportHeightmapParams {
        path: path.to_string_lossy().into_owned(),
        height_scale: 20.0,
        height_offset: 2.0,
    };
    stack.push(Layer::new(
        "import",
        LayerKind::ImportHeightmap(params.clone()),
    ));
    stack.push(Layer::new(
        "stamp",
        LayerKind::Stamp2d(Stamp2dParams { heightmap: params }),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm raster contributions");
    assert_eq!(engine.source_upload_count, 1, "same asset uploads once");

    let Some(layer) = stack.find_mut(base_id) else {
        panic!("base layer disappeared");
    };
    let LayerKind::SculptBase(params) = &mut layer.kind else {
        panic!("base changed kind");
    };
    params.samples[16 * 32 + 16] += 3.0;
    engine.set_dirty_rect(Some((16, 16, 1, 1)));
    engine.mark_dirty(base_id);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("incremental base edit");

    assert_eq!(engine.executed_kernels, vec![GpuKernel::Sculpt]);
    assert_eq!(engine.source_upload_count, 1);
    let _ = std::fs::remove_file(path);
}

/// #132 regression: the picker wrapper has the same input-independent cache
/// semantics as its delegated generator. A base edit reblends the cached
/// contribution without redispatching the procedural kernel.
#[test]
fn warm_cache_base_edit_reuses_procedural_shape_contribution() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(24, 24, 240.0, 240.0);
    let mut stack = LayerStack::new();
    let base = Layer::new("base", LayerKind::SculptBase(SculptParams::filled(24, 5.0)));
    let base_id = base.id();
    stack.push(base);
    stack.push(Layer::new(
        "procedural volcano",
        LayerKind::ProceduralShape(ProceduralShapeParams::with_generator(
            ProceduralGenerator::Volcano,
        )),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm procedural contribution cache");
    assert!(engine.plan_resources.current().is_some());

    let Some(layer) = stack.find_mut(base_id) else {
        panic!("base layer disappeared");
    };
    let LayerKind::SculptBase(params) = &mut layer.kind else {
        panic!("base changed kind");
    };
    params.samples[12 * 24 + 12] += 3.0;
    engine.set_dirty_rect(Some((12, 12, 1, 1)));
    engine.mark_dirty(base_id);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("incremental base edit");

    assert_eq!(engine.executed_kernels, vec![GpuKernel::Sculpt]);
    assert!(
        !engine
            .executed_kernels
            .contains(&GpuKernel::ProceduralShape),
        "cached picker contribution must avoid ProceduralShape dispatch"
    );
}

/// #127 interaction policy: a local edit may update its cheap prefix while a
/// full-field river pass waits for refinement. The deferred suffix must stay
/// dirty so the next non-interactive evaluation cannot reuse a stale top cache.
#[test]
fn local_edit_defers_full_field_river_suffix_until_refinement() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
    let mut stack = LayerStack::new();
    let base = Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(32, 12.0)),
    );
    let base_id = base.id();
    stack.push(base);
    let river = Layer::new(
        "rivers",
        LayerKind::RiverCarve(RiverCarveParams {
            accumulation_threshold: 2.0,
            width: 1.0,
            bank_smooth: 0.0,
            use_dinfinity: false,
            ..RiverCarveParams::default()
        }),
    );
    let river_id = river.id();
    stack.push(river);
    stack.push(Layer::new("blur", LayerKind::Blur(BlurParams::default())));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm full stack");

    let Some(layer) = stack.find_mut(base_id) else {
        panic!("base layer disappeared");
    };
    let LayerKind::SculptBase(params) = &mut layer.kind else {
        panic!("base changed kind");
    };
    params.samples[16 * 32 + 16] += 3.0;
    engine.set_dirty_rect(Some((16, 16, 1, 1)));
    engine.mark_dirty(base_id);
    let interactive = engine
        .evaluate_with_intent(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
            GpuEvaluationIntent::InteractiveLocal,
        )
        .expect("interactive prefix");
    assert_eq!(
        interactive.freshness,
        GpuPreviewFreshness::Deferred {
            from_index: 1,
            from_layer: river_id,
            deferred_layers: 2,
        }
    );
    assert!(!interactive.fully_gpu);
    assert_eq!(interactive.resume_cpu_from, None);
    assert_eq!(engine.executed_kernels, vec![GpuKernel::Sculpt]);
    let stats = engine.last_eval_stats();
    assert!(stats.used_layer_zero_region);
    assert!(
        stats.upload_bytes < u64::from(metrics.width * metrics.height * 4),
        "interactive prefix must upload only the bounded edit"
    );

    let refined = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("refined full-field suffix");
    assert_eq!(refined.freshness, GpuPreviewFreshness::Current);
    assert!(refined.fully_gpu);
    assert_eq!(
        engine.executed_kernels,
        vec![GpuKernel::RiverCarve, GpuKernel::Blur]
    );

    engine.set_dirty_rect(Some((16, 16, 1, 1)));
    engine.mark_dirty(river_id);
    let boundary_edit = engine
        .evaluate_with_intent(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
            GpuEvaluationIntent::InteractiveLocal,
        )
        .expect("edit at full-field boundary");
    assert!(!boundary_edit.did_eval, "there is no exact local prefix");
    assert_eq!(
        boundary_edit.freshness,
        GpuPreviewFreshness::Deferred {
            from_index: 1,
            from_layer: river_id,
            deferred_layers: 2,
        }
    );
    assert!(engine.executed_kernels.is_empty());
}

/// The first enabled FullField pass is a hard boundary: later FullField and
/// local layers all belong to one suffix and execute once during completion.
#[test]
fn local_edit_defers_entire_suffix_from_first_full_field_boundary() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let mut stack = LayerStack::new();
    let base = Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(16, 12.0)),
    );
    let base_id = base.id();
    stack.push(base);
    let river = Layer::new(
        "rivers",
        LayerKind::RiverCarve(RiverCarveParams {
            accumulation_threshold: 2.0,
            width: 1.0,
            bank_smooth: 0.0,
            use_dinfinity: false,
            ..RiverCarveParams::default()
        }),
    );
    let river_id = river.id();
    stack.push(river);
    stack.push(Layer::new(
        "stream power",
        LayerKind::StreamPowerErosion(StreamPowerParams {
            iterations: 1,
            k: 0.002,
            base_level: 0.0,
            ..StreamPowerParams::default()
        }),
    ));
    stack.push(Layer::new("blur", LayerKind::Blur(BlurParams::default())));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm multi-boundary stack");

    let LayerKind::SculptBase(params) = &mut stack.find_mut(base_id).expect("base").kind else {
        panic!("base changed kind");
    };
    params.samples[8 * 16 + 8] += 3.0;
    engine.set_dirty_rect(Some((8, 8, 1, 1)));
    engine.mark_dirty(base_id);
    let interactive = engine
        .evaluate_with_intent(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
            GpuEvaluationIntent::InteractiveLocal,
        )
        .expect("interactive prefix");
    assert_eq!(
        interactive.freshness,
        GpuPreviewFreshness::Deferred {
            from_index: 1,
            from_layer: river_id,
            deferred_layers: 3,
        }
    );
    assert_eq!(engine.executed_kernels, vec![GpuKernel::Sculpt]);

    let complete = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("complete entire suffix");
    assert_eq!(complete.freshness, GpuPreviewFreshness::Current);
    assert!(!complete.fully_gpu);
    assert!(engine.executed_kernels.is_empty());
    let fallback = complete
        .cpu_fallback
        .expect("live river auxiliary dependency must identify its plan boundary");
    assert_eq!(fallback.reason.code, GpuFallbackCode::AuxiliaryDependency);
    assert_eq!(fallback.layer_id, river_id);
    assert!(fallback.operation.is_some());
    assert_eq!(fallback.owner, Some(NodeRef::Layer(river_id)));
}

/// #129 interaction policy: StreamPower shares the generic FullField deferral
/// path, so no obsolete accumulation/incision sequence launches per pointer dab.
#[test]
fn local_edit_defers_full_field_stream_power_suffix_until_refinement() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let mut stack = LayerStack::new();
    let base = Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(16, 12.0)),
    );
    let base_id = base.id();
    stack.push(base);
    stack.push(Layer::new(
        "stream power",
        LayerKind::StreamPowerErosion(StreamPowerParams {
            iterations: 1,
            k: 0.002,
            base_level: 0.0,
            ..StreamPowerParams::default()
        }),
    ));
    stack.push(Layer::new("blur", LayerKind::Blur(BlurParams::default())));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm stream-power stack");

    let Some(layer) = stack.find_mut(base_id) else {
        panic!("base layer disappeared");
    };
    let LayerKind::SculptBase(params) = &mut layer.kind else {
        panic!("base changed kind");
    };
    params.samples[8 * 16 + 8] += 3.0;
    engine.set_dirty_rect(Some((8, 8, 1, 1)));
    engine.mark_dirty(base_id);
    let interactive = engine
        .evaluate_with_intent(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
            GpuEvaluationIntent::InteractiveLocal,
        )
        .expect("interactive stream-power prefix");
    assert!(interactive.freshness.is_deferred());
    assert!(!interactive.fully_gpu);
    assert_eq!(interactive.resume_cpu_from, None);
    assert_eq!(engine.executed_kernels, vec![GpuKernel::Sculpt]);

    let refined = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("refined stream-power suffix");
    assert_eq!(refined.freshness, GpuPreviewFreshness::Current);
    assert!(refined.fully_gpu);
    assert_eq!(
        engine.executed_kernels,
        vec![GpuKernel::StreamPower, GpuKernel::Blur]
    );
}

/// #131 interaction policy: the multi-level solver must use the same generic
/// FullField deferral as the other drainage-coupled GPU previews.
#[test]
fn local_edit_defers_multi_scale_amplify_suffix_until_refinement() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let mut stack = LayerStack::new();
    let base = Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(16, 12.0)),
    );
    let base_id = base.id();
    stack.push(base);
    stack.push(Layer::new(
        "multi scale",
        LayerKind::MultiScaleAmplify(MultiScaleAmplifyParams {
            thermal_iters: 1,
            spe_strength: 0.0,
            spe_iters: 0,
            deposition_strength: 0.0,
            hardness_source: MaskSource::Constant(0.2),
            ridge_lock: MaskSource::Constant(0.15),
            ..MultiScaleAmplifyParams::default()
        }),
    ));
    stack.push(Layer::new("blur", LayerKind::Blur(BlurParams::default())));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm multi-scale stack");

    let Some(layer) = stack.find_mut(base_id) else {
        panic!("base layer disappeared");
    };
    let LayerKind::SculptBase(params) = &mut layer.kind else {
        panic!("base changed kind");
    };
    params.samples[8 * 16 + 8] += 3.0;
    engine.set_dirty_rect(Some((8, 8, 1, 1)));
    engine.mark_dirty(base_id);
    let interactive = engine
        .evaluate_with_intent(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
            GpuEvaluationIntent::InteractiveLocal,
        )
        .expect("interactive multi-scale prefix");
    assert!(interactive.freshness.is_deferred());
    assert!(!interactive.fully_gpu);
    assert_eq!(engine.executed_kernels, vec![GpuKernel::Sculpt]);

    let refined = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("refined multi-scale suffix");
    assert_eq!(refined.freshness, GpuPreviewFreshness::Current);
    assert!(refined.fully_gpu);
    assert_eq!(
        engine.executed_kernels,
        vec![GpuKernel::MultiScaleAmplify, GpuKernel::Blur]
    );
}

/// Two `SculptStrokes` layers in one evaluate walk share the stamp/edited/layer
/// scratch textures. The second must read the first layer's composited result
/// (not a clobbered scratch), so the whole field must still match the CPU, which
/// applies the layers in the same order.
#[test]
fn stacked_sculpt_stroke_layers_apply_in_order_and_match_cpu() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(40, 40, 400.0, 400.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(40, 12.0)),
    ));
    stack.push(Layer::new(
        "s1",
        LayerKind::SculptStrokes(raise_strokes(0.45, 0.5, 10.0)),
    ));
    stack.push(Layer::new(
        "s2",
        LayerKind::SculptStrokes(raise_strokes(0.55, 0.5, 6.0)),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    let gpu_h = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("stacked stroke evaluation")
        .cpu
        .expect("gpu readback");

    let cpu = cpu_oracle(&stack, metrics);
    terra_gpu::parity::assert_field_parity(
        "authoring.sculpt-strokes-stacked",
        &gpu_h,
        &cpu,
        terra_gpu::parity::SCULPT_STROKES_PREVIEW,
    );
}

/// A SculptStrokes layer sits above the base, so a stroke dab is an incremental
/// eval that seeds `approx_range` from cache rather than resetting it. Additive
/// strokes must not widen that carried range, or it drifts on every drag step and
/// the renderer's slab base (`min_h - f(max_h - min_h)`) visibly sinks. Repeated
/// identical dabs must leave the presentation range fixed.
#[test]
fn incremental_sculpt_stroke_dabs_do_not_drift_the_presentation_range() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let metrics = HeightfieldMetrics::new(48, 48, 480.0, 480.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(48, 30.0)),
    ));
    let strokes_layer = Layer::new(
        "strokes",
        LayerKind::SculptStrokes(raise_strokes(0.5, 0.5, 12.0)),
    );
    let strokes_id = strokes_layer.id();
    stack.push(strokes_layer);

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("full stroke evaluation");
    let range_after_full = engine.approx_range;

    for _ in 0..6 {
        engine.set_dirty_rect(Some((16, 16, 16, 16)));
        engine.mark_dirty(strokes_id);
        engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Draft,
                false,
                None,
            )
            .expect("incremental stroke dab");
    }
    assert_eq!(
        engine.approx_range, range_after_full,
        "incremental stroke dabs drifted the presentation range"
    );
}

/// #145: a continuing drag patches only the changed stroke header and the
/// appended point. It reuses both the compiled-plan realization and spare
/// buffer capacity, while the settled regional result remains identical to
/// a fresh full evaluation.
#[test]
fn warm_stroke_append_uploads_only_the_runtime_tail() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 48u32;
    let metrics = HeightfieldMetrics::new(res, res, 480.0, 480.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "base",
        LayerKind::SculptBase(SculptParams::filled(res, 20.0)),
    ));
    let params = raise_strokes(0.46, 0.5, 8.0);
    let strokes = Layer::new("strokes", LayerKind::SculptStrokes(params));
    let strokes_id = strokes.id();
    stack.push(strokes);

    let mut engine = GpuTerrainEngine::new(&gpu.device, res);
    engine.mark_all_dirty(&stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("prime stroke runtime with spare geometric capacity");

    let LayerKind::SculptStrokes(params) =
        &mut stack.find_mut(strokes_id).expect("stroke layer").kind
    else {
        panic!("stroke layer changed kind");
    };
    params.strokes[0]
        .points
        .push(terra_core::layer::SculptPoint {
            u: 0.52,
            v: 0.5,
            pressure: 1.0,
        });
    engine.set_dirty_rect(Some((18, 18, 16, 12)));
    engine.mark_dirty(strokes_id);
    let incremental = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("append one warm stroke point")
        .cpu
        .expect("incremental readback");

    let stats = engine.last_eval_stats();
    assert_eq!(stats.stroke_payload_rebuilds, 0);
    assert_eq!(
        stats.stroke_header_upload_bytes,
        std::mem::size_of::<StrokeHeaderGpu>() as u64
    );
    assert_eq!(stats.stroke_point_upload_bytes, 16);
    assert_eq!(stats.warm_plan_resource_reuses, 1);
    assert!(
        stats.blend_workgroups < u64::from(res.div_ceil(8) * res.div_ceil(8)),
        "warm stroke work must remain below a full-field dispatch"
    );

    let mut oracle_engine = GpuTerrainEngine::new(&gpu.device, res);
    let oracle = oracle_engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("fresh stroke oracle")
        .cpu
        .expect("oracle readback");
    let error = terra_gpu::parity::max_abs_diff(&incremental.to_dense(), &oracle.to_dense());
    assert!(error <= 1.0e-3, "warm stroke append drifted by {error}");
}

/// #125: domain displacement changes only which procedural-noise coordinate is
/// generated. It never samples a displaced texel from the entering height, so a
/// bounded upstream edit still passes through its Add blend without a
/// warp-strength-sized dirty halo.
#[test]
fn dirty_rect_domain_warp_matches_full_field_evaluation() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 64u32;
    let metrics = HeightfieldMetrics::new(res, res, 192.0, 128.0);
    let rect = (24u32, 20u32, 12u32, 10u32);
    let mut stack = LayerStack::new();
    let base = Layer::new("base", LayerKind::Flat(FlatParams { height: 4.0 }));
    let base_id = base.id();
    stack.push(base);
    let sculpt = Layer::new(
        "sculpt",
        LayerKind::SculptBase(SculptParams::filled(res, 20.0)),
    );
    let sculpt_id = sculpt.id();
    stack.push(sculpt);
    stack.push(Layer::new(
        "warp",
        LayerKind::DomainWarp(DomainWarpParams {
            warp_strength: 35.0,
            warp_frequency: 0.018,
            ..DomainWarpParams::default()
        }),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_dirty(base_id);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm domain-warp stack");

    if let LayerKind::SculptBase(params) = &mut stack.flatten_layers_mut()[1].kind {
        for y in rect.1..rect.1 + rect.3 {
            for x in rect.0..rect.0 + rect.2 {
                params.samples[(y * res + x) as usize] = 65.0;
            }
        }
    }
    engine.set_dirty_rect(Some(rect));
    engine.mark_dirty(sculpt_id);
    let incremental = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("incremental domain-warp evaluation")
        .cpu
        .expect("incremental readback");

    let mut oracle_engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let oracle = oracle_engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("full domain-warp evaluation")
        .cpu
        .expect("oracle readback");
    let error = terra_gpu::parity::max_abs_diff(&incremental.to_dense(), &oracle.to_dense());
    let worst = incremental
        .to_dense()
        .iter()
        .zip(oracle.to_dense())
        .enumerate()
        .max_by(|(_, (a0, b0)), (_, (a1, b1))| (*a0 - *b0).abs().total_cmp(&(*a1 - *b1).abs()))
        .map(|(index, (a, b))| (index % res as usize, index / res as usize, *a, b));
    assert!(
        error <= 1.0e-3,
        "incremental DomainWarp drifted by {error} at {worst:?}"
    );
}

/// B1-D6 / C1-C2 — #90's explicit rect-edge-artifact answer. A flat field with
/// a tall bump inside the edit rect: a wide Smooth (radius 16, 2 iters) spreads
/// that bump ~32 texels. A probe in the spread ring — outside the retired
/// 8-texel halo but inside the true reach — must match the full-field oracle.
/// The plan halo covers it; the hardcoded 8-texel halo left the ring unfiltered
/// (the visible artifact). Away from the bump the field is flat, so filtered ==
/// unfiltered there and the probe isolates exactly the halo coverage.
#[test]
fn dirty_rect_effect_filter_recomputes_full_kernel_reach() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 96u32;
    let metrics = HeightfieldMetrics::new(res, res, 192.0, 192.0);
    let rect = (40u32, 40u32, 16u32, 16u32);

    // Flat base at index 0 gives the sculpt a cached prefix (first_dirty > 0 =>
    // incremental rect path).
    let build = || {
        let mut sculpt = SculptParams::filled(res, 20.0);
        for y in rect.1..rect.1 + rect.3 {
            for x in rect.0..rect.0 + rect.2 {
                sculpt.samples[(y * res + x) as usize] = 420.0;
            }
        }
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "base",
            LayerKind::Flat(FlatParams { height: 4.0 }),
        ));
        let sculpt_layer = Layer::new("sculpt", LayerKind::SculptBase(sculpt));
        let sculpt_id = sculpt_layer.id();
        stack.push(sculpt_layer);
        stack.push(Layer::new(
            "smooth",
            LayerKind::EffectFilter(EffectFilterParams {
                radius: 16,
                iterations: 2,
                strength: 1.0,
                ..EffectFilterParams::smooth()
            }),
        ));
        (stack, sculpt_id)
    };

    let (stack, sculpt_id) = build();
    let base_id = stack.flatten_layers()[0].id();

    // Warm the caches with a full-field pass.
    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_dirty(base_id);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("warm full evaluation");

    // Incremental rect eval: dirty the sculpt, bound the edit to `rect`.
    engine.set_dirty_rect(Some(rect));
    engine.mark_dirty(sculpt_id);
    let incremental = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("incremental rect eval")
        .cpu
        .expect("incremental readback");

    // Full-field oracle from a fresh engine.
    let mut oracle_engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let oracle = oracle_engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("oracle eval")
        .cpu
        .expect("oracle readback");

    // Probe the spread ring: outside rect+8 (the retired halo would leave it
    // unfiltered) but inside rect+32 (the true reach), where the bump has spread.
    let (px, py) = (70u32, 48u32);
    let spread = oracle.get(px, py) - 20.0;
    assert!(
        spread > 2.0,
        "fixture bump did not spread into the probe ring (spread {spread})"
    );
    let err = (incremental.get(px, py) - oracle.get(px, py)).abs();
    assert!(
        err < 0.5,
        "incremental filter left the spread ring stale vs the full oracle by {err}; \
             the recompute halo under-covers the kernel reach (rect-edge artifact)"
    );
}

/// The mask bake must cover the whole field, not just the edit rect: it feeds a
/// full-field blend and a full-field layer cache, so a region-only bake would
/// leave mask = 1.0 (and the wrong blended height) outside the rect. This guards
/// the mask-bake fix that landed with B1-D6 (#90).
#[test]
fn dirty_rect_masked_generator_bakes_full_field() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 64u32;
    let metrics = HeightfieldMetrics::new(res, res, 128.0, 128.0);
    let mask = MaskAsset::new(MaskId::new(), "constant", MaskSource::Constant(0.25));
    let assets = vec![mask.clone()];

    let build = || {
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "base",
            LayerKind::Flat(FlatParams { height: 4.0 }),
        ));
        let sculpt_layer = Layer::new("sculpt", LayerKind::SculptBase(varied_sculpt(res)));
        let sculpt_id = sculpt_layer.id();
        stack.push(sculpt_layer);
        let mut masked = Layer::new("masked add", LayerKind::Flat(FlatParams { height: 50.0 }));
        masked.common.blend = BlendMode::Add;
        masked.common.masks.push(MaskRef::new(mask.id));
        let masked_id = masked.id();
        stack.push(masked);
        (stack, sculpt_id, masked_id)
    };

    // Warm the engine with a full-field pass.
    let (stack, _sculpt_id, masked_id) = build();
    let base_id = stack.flatten_layers()[0].id();
    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    engine.mark_dirty(base_id);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &assets,
            metrics,
            PreviewQuality::Draft,
            false,
            None,
        )
        .expect("masked full evaluation");

    // Incremental: re-bake the masked layer under a small dirty rect. No filter is
    // involved, so a correct full-field bake makes the whole readback match the
    // oracle; a region-only bake corrupts everything outside the rect.
    let rect = (24u32, 24u32, 12u32, 12u32);
    engine.set_dirty_rect(Some(rect));
    engine.mark_dirty(masked_id);
    let incremental = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &assets,
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("incremental masked evaluation")
        .cpu
        .expect("incremental readback");

    let mut oracle_engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    let oracle = oracle_engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &assets,
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("oracle masked evaluation")
        .cpu
        .expect("oracle readback");

    let mut max_err = 0.0f32;
    for y in 0..res {
        for x in 0..res {
            max_err = max_err.max((incremental.get(x, y) - oracle.get(x, y)).abs());
        }
    }
    assert!(
        max_err < 0.05,
        "masked incremental diverged from the full oracle by {max_err}; the mask bake \
             did not cover the full field outside the dirty rect"
    );
}

/// #148: warm Raise and Pinch gestures on both authored sculpt payloads
/// remain on the bounded compiled tree plan.
#[test]
fn untitled6_tree_warm_brushes_stay_bounded() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 96u32;
    for (target_strokes, brush) in [
        (false, SculptStrokeKind::Raise),
        (false, SculptStrokeKind::Pinch),
        (true, SculptStrokeKind::Raise),
        (true, SculptStrokeKind::Pinch),
    ] {
        let (mut document, ids) = untitled6_document(res, Untitled6Variant::ProductionTopology);
        document.metrics.tile_size = 16;
        document.metrics.halo = 2;
        let target = if target_strokes {
            ids.semantic_sculpt
        } else {
            ids.base
        };
        let mut cache = TerrainPlanCache::new();
        let cold = cache
            .update(
                &document.stack,
                &document.masks,
                &[TerrainEditClass::Structure],
            )
            .expect("compile fixture");
        let mut engine = GpuTerrainEngine::new(&gpu.device, res);
        engine
            .evaluate_compiled_with_intent(
                &gpu.device,
                &gpu.queue,
                &document.stack,
                &document.masks,
                cache.current_plan().unwrap(),
                cache.structure_revision(),
                &cold,
                document.metrics,
                PreviewQuality::Draft,
                false,
                GpuEvaluationIntent::Complete,
            )
            .expect("warm resources");
        let plan_before = cache.stats().snapshot();

        let layer = document.stack.find_mut(target).expect("gesture target");
        for (index, u) in [0.47, 0.50, 0.53].into_iter().enumerate() {
            layer.apply_brush(
                brush,
                BrushDab {
                    u,
                    v: 0.5,
                    radius_uv: 0.035,
                    radius_m: 90.0,
                    strength: 4.0,
                    target_height: 24.0,
                    falloff: 0.55,
                    continuing: index != 0,
                },
            );
        }
        let invalidation = cache
            .update(
                &document.stack,
                &document.masks,
                &[TerrainEditClass::Content {
                    owner: NodeRef::Layer(target),
                    fields: vec![FieldId::Height],
                    scope: PlanDirtyScope::Region(UvRect::from_center_radius(0.5, 0.5, 0.07)),
                }],
            )
            .expect("patch gesture");
        let result = engine
            .evaluate_compiled_with_intent(
                &gpu.device,
                &gpu.queue,
                &document.stack,
                &document.masks,
                cache.current_plan().unwrap(),
                cache.structure_revision(),
                &invalidation,
                document.metrics,
                PreviewQuality::Draft,
                false,
                GpuEvaluationIntent::InteractiveLocal,
            )
            .expect("interactive gesture");
        assert!(result.fully_gpu, "{target_strokes}/{brush:?}");
        assert_eq!(result.freshness, GpuPreviewFreshness::Current);
        assert!(result.cpu_fallback.is_none());

        let plan_after = cache.stats().snapshot();
        assert_eq!(plan_after.plan_compiles, plan_before.plan_compiles);
        assert_eq!(
            plan_after.authored_tree_walks,
            plan_before.authored_tree_walks
        );
        assert_eq!(plan_after.dependency_builds, plan_before.dependency_builds);
        let stats = engine.last_eval_stats();
        assert!(stats.operations_dispatched > 0);
        assert!(
            stats.operations_reused > 0,
            "Volcano contribution should be reused"
        );
        assert_eq!(stats.operations_deferred, 0);
        assert_eq!(stats.readback_bytes, 0);
        assert!(stats.upload_bytes < u64::from(res * res * 4), "{stats:?}");
        assert!(engine.last_plan_operation_trace().iter().any(|operation| {
            operation.disposition == GpuPlanOperationDisposition::Dispatched
                && operation.output_region.2 < res
                && operation.output_region.3 < res
        }));

        for quality in [PreviewQuality::Medium, PreviewQuality::Full] {
            let result = engine
                .evaluate_compiled_with_intent(
                    &gpu.device,
                    &gpu.queue,
                    &document.stack,
                    &document.masks,
                    cache.current_plan().unwrap(),
                    cache.structure_revision(),
                    &PlanInvalidation::default(),
                    document.metrics,
                    quality,
                    false,
                    GpuEvaluationIntent::Complete,
                )
                .unwrap_or_else(|error| panic!("{quality:?} refinement failed: {error}"));
            assert!(result.fully_gpu, "{target_strokes}/{brush:?}/{quality:?}");
            assert_eq!(result.freshness, GpuPreviewFreshness::Current);
            assert!(result.cpu_fallback.is_none(), "{quality:?}");
            let stats = engine.last_eval_stats();
            assert_eq!(stats.operations_deferred, 0, "{quality:?}");
            assert_eq!(stats.readback_bytes, 0, "{quality:?}");
        }

        let settled = engine
            .readback_current(&gpu.device, &gpu.queue)
            .expect("settled preview");
        let oracle = cpu_oracle(&document.stack, document.metrics);
        terra_gpu::parity::assert_field_parity(
            &format!("Untitled6 target_strokes={target_strokes} brush={brush:?}"),
            &settled,
            &oracle,
            terra_gpu::parity::UNTITLED6_INTERACTION,
        );
    }
}

/// #154: optional refinement is split into fenced units, keeps its plan
/// resources private until publication, and produces the same field as the
/// existing complete evaluator.
#[test]
fn resumable_refinement_is_depth_one_transactional_and_matches_complete_eval() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let res = 64u32;
    let (document, _) = untitled6_document(res, Untitled6Variant::ProductionTopology);
    let mut cache = TerrainPlanCache::new();
    let invalidation = cache
        .update(
            &document.stack,
            &document.masks,
            &[TerrainEditClass::Structure],
        )
        .expect("compile fixture");
    let plan = cache.current_plan().expect("compiled plan");
    let revision = cache.structure_revision();

    let mut complete = GpuTerrainEngine::new(&gpu.device, res);
    complete
        .evaluate_compiled_with_intent(
            &gpu.device,
            &gpu.queue,
            &document.stack,
            &document.masks,
            plan,
            revision,
            &invalidation,
            document.metrics,
            PreviewQuality::Medium,
            false,
            GpuEvaluationIntent::Complete,
        )
        .expect("complete Medium evaluation");
    let expected = complete
        .readback_current(&gpu.device, &gpu.queue)
        .expect("complete readback");

    let mut resumable = GpuTerrainEngine::new(&gpu.device, res);
    let committed_before = resumable.plan_resources.stats().committed_candidates;
    let mut job = resumable
        .begin_compiled_refinement(
            &gpu.device,
            &document.stack,
            &document.masks,
            plan,
            revision,
            &invalidation,
            document.metrics,
            PreviewQuality::Medium,
        )
        .expect("begin resumable Medium evaluation");
    assert!(resumable.plan_resources.current().is_none());
    assert!(resumable.last_graph.plans.is_empty());

    let mut submitted = 0u32;
    let mut max_depth = 0u8;
    for _ in 0..10_000 {
        let step = resumable
            .advance_compiled_refinement(&gpu.device, &gpu.queue, &mut job)
            .expect("advance resumable evaluation");
        let progress = job.progress();
        max_depth = max_depth.max(progress.submissions_in_flight);
        if matches!(step, GpuRefinementStep::Submitted { .. }) {
            submitted += 1;
            let _ = gpu.device.poll(wgpu::Maintain::Wait);
        }
        if matches!(step, GpuRefinementStep::AwaitingGpu) {
            let _ = gpu.device.poll(wgpu::Maintain::Wait);
        }
        if matches!(step, GpuRefinementStep::ReadyToPublish) {
            break;
        }
    }
    assert!(
        job.final_copy_complete,
        "refinement did not reach its final fence"
    );
    assert!(submitted > 1, "the plan should be split across submissions");
    assert_eq!(max_depth, 1, "refinement queue depth must remain bounded");
    assert!(resumable.plan_resources.current().is_none());
    assert_eq!(
        resumable.plan_resources.stats().committed_candidates,
        committed_before,
        "candidate resources must not become canonical before publication"
    );

    let result = resumable
        .publish_compiled_refinement(job)
        .expect("publish fenced candidate");
    assert!(result.fully_gpu);
    assert_eq!(result.freshness, GpuPreviewFreshness::Current);
    assert!(resumable.plan_resources.current().is_some());
    assert!(!resumable.last_graph.plans.is_empty());
    assert_eq!(
        resumable.plan_resources.stats().committed_candidates,
        committed_before + 1
    );
    let actual = resumable
        .readback_current(&gpu.device, &gpu.queue)
        .expect("resumable readback");
    terra_gpu::parity::assert_field_parity(
        "resumable Medium vs complete Medium",
        &actual,
        &expected,
        terra_gpu::parity::UNTITLED6_INTERACTION,
    );

    // A stroke arriving after submission abandons the cursor but cannot
    // cancel that submission. Its fence must constrain the next generation.
    let mut stale = resumable
        .begin_compiled_refinement(
            &gpu.device,
            &document.stack,
            &document.masks,
            plan,
            revision,
            &invalidation,
            document.metrics,
            PreviewQuality::Full,
        )
        .expect("begin stale Full generation");
    loop {
        if matches!(
            resumable
                .advance_compiled_refinement(&gpu.device, &gpu.queue, &mut stale)
                .expect("advance stale generation"),
            GpuRefinementStep::Submitted { .. }
        ) {
            break;
        }
    }
    resumable.abandon_compiled_refinement(stale);
    let fresh = resumable
        .begin_compiled_refinement(
            &gpu.device,
            &document.stack,
            &document.masks,
            plan,
            revision,
            &invalidation,
            document.metrics,
            PreviewQuality::Medium,
        )
        .expect("begin replacement generation");
    assert_eq!(
        resumable.refinement_submissions_in_flight(&fresh),
        1,
        "the abandoned submission fence must remain in the global depth bound"
    );
    resumable.abandon_compiled_refinement(fresh);
    let _ = gpu.device.poll(wgpu::Maintain::Wait);
}

/// Manual release probe for the acceptance resolutions. Adapter timing is
/// reported, not asserted, because CI hardware is intentionally variable.
#[test]
#[ignore = "run in release mode to record #150/#152 adapter timings"]
fn untitled6_release_timing_probe_2048_4096() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    for res in [2048u32, 4096] {
        let (mut document, ids) = untitled6_document(res, Untitled6Variant::ProductionTopology);
        let mut cache = TerrainPlanCache::new();
        let cold_invalidation = cache
            .update(
                &document.stack,
                &document.masks,
                &[TerrainEditClass::Structure],
            )
            .unwrap();
        let mut engine = GpuTerrainEngine::new(&gpu.device, res);
        let cold_started = std::time::Instant::now();
        let cold_result = engine
            .evaluate_compiled_with_intent(
                &gpu.device,
                &gpu.queue,
                &document.stack,
                &document.masks,
                cache.current_plan().unwrap(),
                cache.structure_revision(),
                &cold_invalidation,
                document.metrics,
                PreviewQuality::Draft,
                false,
                GpuEvaluationIntent::Complete,
            )
            .unwrap();
        let _ = gpu.device.poll(wgpu::Maintain::Wait);
        let cold_ms = cold_started.elapsed().as_secs_f64() * 1000.0;
        let cold_stats = engine.last_eval_stats();
        let cold_plan_stats = cache.stats().snapshot();
        assert!(cold_result.fully_gpu);
        assert!(cold_result.cpu_fallback.is_none());
        assert_eq!(cold_stats.readback_bytes, 0);

        document.stack.find_mut(ids.base).unwrap().apply_brush(
            SculptStrokeKind::Raise,
            BrushDab {
                u: 0.5,
                v: 0.5,
                radius_uv: 0.01,
                radius_m: 40.0,
                strength: 3.0,
                target_height: 0.0,
                falloff: 0.5,
                continuing: false,
            },
        );
        let invalidation = cache
            .update(
                &document.stack,
                &document.masks,
                &[TerrainEditClass::Content {
                    owner: NodeRef::Layer(ids.base),
                    fields: vec![FieldId::Height],
                    scope: PlanDirtyScope::Region(UvRect::from_center_radius(0.5, 0.5, 0.01)),
                }],
            )
            .unwrap();
        let warm_started = std::time::Instant::now();
        let warm_result = engine
            .evaluate_compiled_with_intent(
                &gpu.device,
                &gpu.queue,
                &document.stack,
                &document.masks,
                cache.current_plan().unwrap(),
                cache.structure_revision(),
                &invalidation,
                document.metrics,
                PreviewQuality::Draft,
                false,
                GpuEvaluationIntent::InteractiveLocal,
            )
            .unwrap();
        let _ = gpu.device.poll(wgpu::Maintain::Wait);
        let warm_ms = warm_started.elapsed().as_secs_f64() * 1000.0;
        let warm_stats = engine.last_eval_stats();
        let warm_plan_stats = cache.stats().snapshot();
        assert!(warm_result.fully_gpu);
        assert!(warm_result.cpu_fallback.is_none());
        assert_eq!(warm_stats.readback_bytes, 0);

        let mut transition_full_ms = Vec::with_capacity(10);
        for sample in 0..10 {
            let draft = engine
                .evaluate_compiled_with_intent(
                    &gpu.device,
                    &gpu.queue,
                    &document.stack,
                    &document.masks,
                    cache.current_plan().unwrap(),
                    cache.structure_revision(),
                    &PlanInvalidation::default(),
                    document.metrics,
                    PreviewQuality::Draft,
                    false,
                    GpuEvaluationIntent::Complete,
                )
                .unwrap();
            let _ = gpu.device.poll(wgpu::Maintain::Wait);
            assert!(draft.fully_gpu);
            document.stack.find_mut(ids.base).unwrap().apply_brush(
                SculptStrokeKind::Raise,
                BrushDab {
                    u: 0.40 + sample as f32 * 0.002,
                    v: 0.5,
                    radius_uv: 0.01,
                    radius_m: 40.0,
                    strength: 0.25,
                    target_height: 0.0,
                    falloff: 0.5,
                    continuing: false,
                },
            );
            let transition_invalidation = cache
                .update(
                    &document.stack,
                    &document.masks,
                    &[TerrainEditClass::Content {
                        owner: NodeRef::Layer(ids.base),
                        fields: vec![FieldId::Height],
                        scope: PlanDirtyScope::Region(UvRect::from_center_radius(
                            0.40 + sample as f32 * 0.002,
                            0.5,
                            0.01,
                        )),
                    }],
                )
                .unwrap();
            let started = std::time::Instant::now();
            let full = engine
                .evaluate_compiled_with_intent(
                    &gpu.device,
                    &gpu.queue,
                    &document.stack,
                    &document.masks,
                    cache.current_plan().unwrap(),
                    cache.structure_revision(),
                    &transition_invalidation,
                    document.metrics,
                    PreviewQuality::Full,
                    false,
                    GpuEvaluationIntent::Complete,
                )
                .unwrap();
            let _ = gpu.device.poll(wgpu::Maintain::Wait);
            assert!(full.fully_gpu);
            transition_full_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        transition_full_ms.sort_by(f64::total_cmp);
        let transition_p50 = transition_full_ms[(transition_full_ms.len() * 50).div_ceil(100) - 1];
        let transition_p95 = transition_full_ms[(transition_full_ms.len() * 95).div_ceil(100) - 1];
        let transition_max = *transition_full_ms.last().unwrap();

        let mut full_ms = Vec::with_capacity(20);
        for sample in 0..20 {
            document.stack.find_mut(ids.base).unwrap().apply_brush(
                SculptStrokeKind::Raise,
                BrushDab {
                    u: 0.45 + sample as f32 * 0.002,
                    v: 0.5,
                    radius_uv: 0.01,
                    radius_m: 40.0,
                    strength: 0.25,
                    target_height: 0.0,
                    falloff: 0.5,
                    continuing: false,
                },
            );
            let full_invalidation = cache
                .update(
                    &document.stack,
                    &document.masks,
                    &[TerrainEditClass::Content {
                        owner: NodeRef::Layer(ids.base),
                        fields: vec![FieldId::Height],
                        scope: PlanDirtyScope::Region(UvRect::from_center_radius(
                            0.45 + sample as f32 * 0.002,
                            0.5,
                            0.01,
                        )),
                    }],
                )
                .unwrap();
            let started = std::time::Instant::now();
            let full = engine
                .evaluate_compiled_with_intent(
                    &gpu.device,
                    &gpu.queue,
                    &document.stack,
                    &document.masks,
                    cache.current_plan().unwrap(),
                    cache.structure_revision(),
                    &full_invalidation,
                    document.metrics,
                    PreviewQuality::Full,
                    false,
                    GpuEvaluationIntent::Complete,
                )
                .unwrap();
            let _ = gpu.device.poll(wgpu::Maintain::Wait);
            assert!(full.fully_gpu);
            assert_eq!(engine.last_eval_stats().readback_bytes, 0);
            full_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        full_ms.sort_by(f64::total_cmp);
        let p50 = full_ms[(full_ms.len() * 50).div_ceil(100) - 1];
        let p95 = full_ms[(full_ms.len() * 95).div_ceil(100) - 1];
        let max = *full_ms.last().unwrap();
        println!(
                "issue150_152 resolution={res} adapter={:?} cold_ms={cold_ms:.3} warm_ms={warm_ms:.3} transition_n={} transition_full_p50_ms={transition_p50:.3} transition_full_p95_ms={transition_p95:.3} transition_full_max_ms={transition_max:.3} resident_n={} resident_full_p50_ms={p50:.3} resident_full_p95_ms={p95:.3} resident_full_max_ms={max:.3} cold_stats={cold_stats:?} warm_stats={warm_stats:?} full_stats={:?} cold_plan_stats={cold_plan_stats:?} warm_plan_stats={warm_plan_stats:?}",
                gpu.adapter_info,
                transition_full_ms.len(),
                full_ms.len(),
                engine.last_eval_stats(),
            );
    }
}
