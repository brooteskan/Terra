use super::*;

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

    // Simulate a persisted pre-contract layer with no generated named outputs.
    // The canonical kind contract must still prevent an aux-losing bridge.
    let mut raise_path = Layer::new(
        "Raise Path",
        LayerKind::Path(terra_core::layer::PathParams {
            carve: false,
            ..Default::default()
        }),
    );
    raise_path.common.outputs.clear();
    let raise_path_layers = [&raise_path];
    assert!(!cpu_resume_prefix_is_height_only(&raise_path_layers, 1));
}

#[test]
fn compatibility_bridge_parameter_stays_connected_to_compiled_execution() {
    let runtime = include_str!("../runtime.rs");
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
