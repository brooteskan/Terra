use super::*;

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
            riser_width_m: 0.0,
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
