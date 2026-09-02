use super::*;

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
